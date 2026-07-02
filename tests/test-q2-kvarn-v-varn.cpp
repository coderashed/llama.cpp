// Item 07b recipe test (KVARN_FAITHFUL/07): the exact op sequence cpy_v_regions
// (write) and get_v (read) will use for faithful V, proven in isolation before
// touching the live graph -- the same discipline that saved the K build twice
// (self-study 2026-07-01_faithful-kvarn_build_and_varn.md, lesson 2).
//
// V's block container is the EXISTING per-token q2_kvarn (design:
// kvarn_faithful_v_design.md, "V is ALREADY per-token... NO axis transpose is
// needed" -- for the BLOCK FORMAT). But VarN's op wants a token-fastest tile
// while v_cur/v_body are channel-fastest, so unlike K (whose channel-major block
// format happens to match VarN's natural output shape, needing only ONE
// transpose before VarN), V needs a transpose BEFORE VarN and one AFTER:
//
//   write: v_cur [C,G] (channel-fastest, natural)
//          -> transpose+cont -> [G,C] (token-fastest, what VarN wants)
//          -> ggml_kvarn_varn -> T_norm [G,C] (token-fastest, same layout as input)
//          -> transpose+cont -> [C,G] (channel-fastest, what a per-token q2_kvarn
//             block wants: 128 contiguous channel values per token)
//          -> cpy F32 -> Q2_KVARN into v_body
//
//   read:  v_body [C,G] Q2_KVARN -> cast F32 -> [C,G] (channel-fastest already,
//          NO permute needed here, unlike K's dequant-then-permute -- the extra
//          transpose V pays at write, K pays at read)
//          -> broadcast-multiply by S_r (per-channel) and S_c (per-token)
//             (mirrors get_k's Dp4/scb broadcast recipe exactly)

#include "ggml.h"
#include "ggml-cpu.h"
#include "ggml-quants.h"
#include "ggml-backend.h"
#include "../src/llama-kvarn.h"

#undef NDEBUG
#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <vector>
#include <random>

static const int C  = 128; // channels (head_dim, n_head=1 whole-tile per self-study #7)
static const int G  = 128; // group size (tokens per group)
static const int NG = 2;   // groups
static const int P  = G*NG;

int main(void) {
    ggml_cpu_init();
    printf("test-q2-kvarn-v-varn:\n");

    // Standard V layout [C, P]: channel-major (dim0=channel, contiguous), matching
    // v_cur's natural shape after merging head_dim+n_head into one channel axis.
    std::mt19937 rng(0xFA17FA11U);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    std::vector<float> Vstd(C * P);
    for (int p = 0; p < P; p++) {
        const float tok_scale = (p % 16 == 0) ? 8.0f : 1.0f; // magnitude-imbalanced tokens
        for (int c = 0; c < C; c++) {
            Vstd[c + p*C] = nd(rng) * tok_scale;
        }
    }

    struct ggml_init_params params = {
        /*.mem_size   =*/ ggml_tensor_overhead()*128 + ggml_graph_overhead()*2,
        /*.mem_buffer =*/ NULL, /*.no_alloc =*/ true,
    };
    struct ggml_context * ctx = ggml_init(params);

    // v_cur-like source: [C, P] F32, channel-fastest (mirrors the merged
    // [n_embd_gqa, n_tokens] view cpy_k_regions builds from k_cur).
    struct ggml_tensor * Vsrc = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, C, P);

    // v_body: per-token q2_kvarn blocks, channel-fastest, grouped. [C, G, NG].
    struct ggml_tensor * v_body = ggml_new_tensor_3d(ctx, GGML_TYPE_Q2_KVARN, C, G, NG);
    // Persistent VarN scales, mirroring k_sr/k_sc shapes (n_head=1 here).
    struct ggml_tensor * v_sr = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, C, 1, NG); // per-channel
    struct ggml_tensor * v_sc = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, G, 1, NG); // per-token

    std::vector<ggml_tensor *> write_roots;
    for (int g = 0; g < NG; g++) {
        // [C, G] channel-fastest slice of this group.
        struct ggml_tensor * tile_ch = ggml_view_2d(ctx, Vsrc, C, G, Vsrc->nb[1], (size_t)g*G*Vsrc->nb[1]);
        // -> [G, C] token-fastest (what ggml_kvarn_varn wants).
        struct ggml_tensor * tile_tok = ggml_cont(ctx, ggml_transpose(ctx, tile_ch));
        struct ggml_tensor * tile3d = ggml_reshape_3d(ctx, tile_tok, G, C, 1); // [n_tok=G, head_dim=C, n_head=1]

        struct ggml_tensor * op = ggml_kvarn_varn(ctx, tile3d); // packed [T_norm(G*C) ++ S_r(C) ++ S_c(G)]

        struct ggml_tensor * Tn_tok = ggml_view_2d(ctx, op, G, C, (size_t)G*sizeof(float), 0); // [G,C] token-fastest
        struct ggml_tensor * Tn_ch  = ggml_cont(ctx, ggml_transpose(ctx, Tn_tok));              // [C,G] channel-fastest

        struct ggml_tensor * sr = ggml_view_1d(ctx, op, C, (size_t)(G*C)*sizeof(float));       // [C] per-channel
        struct ggml_tensor * sc = ggml_view_1d(ctx, op, G, (size_t)(G*C + C)*sizeof(float));   // [G] per-token

        struct ggml_tensor * body_dst = ggml_view_2d(ctx, v_body, C, G, v_body->nb[1], (size_t)g*v_body->nb[2]);
        struct ggml_tensor * sr_dst   = ggml_view_2d(ctx, v_sr, C, 1, v_sr->nb[1], (size_t)g*v_sr->nb[2]);
        struct ggml_tensor * sc_dst   = ggml_view_2d(ctx, v_sc, G, 1, v_sc->nb[1], (size_t)g*v_sc->nb[2]);

        write_roots.push_back(ggml_cpy(ctx, Tn_ch, body_dst)); // F32 -> Q2_KVARN
        write_roots.push_back(ggml_cpy(ctx, sr, sr_dst));
        write_roots.push_back(ggml_cpy(ctx, sc, sc_dst));
    }

    struct ggml_cgraph * gf_write = ggml_new_graph(ctx);
    for (auto * r : write_roots) ggml_build_forward_expand(gf_write, r);

    ggml_backend_t cpu = ggml_backend_cpu_init();
    assert(cpu != NULL);
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, cpu);
    assert(buf != NULL);
    ggml_backend_tensor_set(Vsrc, Vstd.data(), 0, ggml_nbytes(Vsrc));
    assert(ggml_backend_graph_compute(cpu, gf_write) == GGML_STATUS_SUCCESS);

    // ---- read: dequant v_body -> F32 (item 06's new cpy), broadcast S_r/S_c ----
    struct ggml_context * ctx2 = ggml_init(params);
    struct ggml_tensor * vb_in  = ggml_new_tensor_3d(ctx2, GGML_TYPE_Q2_KVARN, C, G, NG);
    struct ggml_tensor * sr_in  = ggml_new_tensor_3d(ctx2, GGML_TYPE_F32, C, 1, NG);
    struct ggml_tensor * sc_in  = ggml_new_tensor_3d(ctx2, GGML_TYPE_F32, G, 1, NG);

    struct ggml_tensor * Tn = ggml_cast(ctx2, vb_in, GGML_TYPE_F32); // [C,G,NG], channel-fastest, no permute needed
    struct ggml_tensor * Tn4 = ggml_reshape_4d(ctx2, Tn, C, 1, G, NG);
    struct ggml_tensor * sr4 = ggml_reshape_4d(ctx2, sr_in, C, 1, 1, NG);
    Tn4 = ggml_mul(ctx2, Tn4, sr4); // per-channel broadcast over G

    // per-token broadcast: sc [G,1,NG] -> permute so G lands on Tn4's dim2.
    struct ggml_tensor * scb = ggml_cont(ctx2, ggml_permute(ctx2, ggml_reshape_4d(ctx2, sc_in, G, 1, 1, NG), 2, 1, 0, 3)); // [1,1,G,NG]
    Tn4 = ggml_mul(ctx2, Tn4, scb);

    struct ggml_tensor * recon = ggml_reshape_2d(ctx2, ggml_cont(ctx2, Tn4), C, P);

    struct ggml_cgraph * gf_read = ggml_new_graph(ctx2);
    ggml_build_forward_expand(gf_read, recon);

    ggml_backend_buffer_t buf2 = ggml_backend_alloc_ctx_tensors(ctx2, cpu);
    assert(buf2 != NULL);

    std::vector<block_q2_kvarn> vb_data(C/QK2_KVARN * G * NG);
    ggml_backend_tensor_get(v_body, vb_data.data(), 0, ggml_nbytes(v_body));
    ggml_backend_tensor_set(vb_in, vb_data.data(), 0, ggml_nbytes(vb_in));

    std::vector<float> sr_data(C*NG), sc_data(G*NG);
    ggml_backend_tensor_get(v_sr, sr_data.data(), 0, ggml_nbytes(v_sr));
    ggml_backend_tensor_get(v_sc, sc_data.data(), 0, ggml_nbytes(v_sc));
    ggml_backend_tensor_set(sr_in, sr_data.data(), 0, ggml_nbytes(sr_in));
    ggml_backend_tensor_set(sc_in, sc_data.data(), 0, ggml_nbytes(sc_in));

    assert(ggml_backend_graph_compute(cpu, gf_read) == GGML_STATUS_SUCCESS);

    std::vector<float> got(C*P);
    ggml_backend_tensor_get(recon, got.data(), 0, sizeof(float)*C*P);

    ggml_backend_buffer_free(buf);
    ggml_backend_buffer_free(buf2);
    ggml_backend_free(cpu);
    ggml_free(ctx);
    ggml_free(ctx2);

    // No-VarN baseline (straight per-token quant on Vstd) for a relative check:
    // the recipe should not be worse than the trivial path on this same data.
    double mse_varn = 0.0, mse_novarn = 0.0;
    {
        std::vector<block_q2_kvarn> blocks(P);
        quantize_row_q2_kvarn_ref(Vstd.data(), blocks.data(), C*P);
        std::vector<float> nv(C*P);
        dequantize_row_q2_kvarn(blocks.data(), nv.data(), C*P);
        for (int i = 0; i < C*P; i++) { double d = nv[i]-Vstd[i]; mse_novarn += d*d; }
        mse_novarn /= (C*P);
    }
    for (int i = 0; i < C*P; i++) { double d = got[i]-Vstd[i]; mse_varn += d*d; }
    mse_varn /= (C*P);

    printf("  recipe round-trip MSE (VarN):    %.6e\n", mse_varn);
    printf("  baseline round-trip MSE (no-VarN): %.6e\n", mse_novarn);
    printf("  ratio: %.4f\n", mse_varn / (mse_novarn + 1e-30));

    // Falsifiable: the exact write+read recipe must reconstruct Vstd at all
    // (rules out an axis/broadcast bug), and improve on the magnitude-imbalanced
    // tokens vs the no-VarN baseline (mirrors test-q2-kvarn-k-varn's premise gate).
    for (int i = 0; i < C*P; i++) {
        assert(std::isfinite(got[i]));
    }
    assert(mse_varn < mse_novarn);

    printf("\nPASSED\n");
    return 0;
}
