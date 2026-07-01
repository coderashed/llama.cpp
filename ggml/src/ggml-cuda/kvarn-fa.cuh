#include "common.cuh"

// Bespoke per-channel-K attention increments (KVARN_FAITHFUL, fused FA read path).

// Graph op forward (GGML_OP_KVARN_FA). See kvarn-fa.cu for the src/dst layout contract.
void ggml_cuda_op_kvarn_fa(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

// KQ scores for one group / single head: scores[q,t] = Sc[t] * sum_c Q[c,q] *
// (code_{c,t} + z[c]) * (s[c] * Sr[c]). Kblocks = C channel-major block_q2_kvarn_k;
// Q [D,n_q] column-major; scores [n_q,G]. Host buffers (test scaffolding wrapper).
extern "C" void kvarn_kq_scores_cuda(const void * Kblocks, const float * Q,
        const float * Sr, const float * Sc, float * scores, int D, int n_q, int G);

// Full single-group / single-head attention: O[c,q] = sum_t softmax_t(scale*KQ[q,t] +
// mask[t,q]) * V[c,t]. V_f16 is token-major [D,G] half; mask [G,n_q] F32; O [D,n_q].
extern "C" void kvarn_fa_group_cuda(const void * Kblocks, const float * Q,
        const float * Sr, const float * Sc, const void * V_f16, const float * mask,
        float * O, int D, int n_q, int G, float scale);

// Generalized attention over n_kv keys (multiple groups) with GQA multi-head. Layouts
// documented in kvarn-fa.cu. This is the compute the graph op will run.
extern "C" void kvarn_fa_cuda(const void * Kblocks, const float * Q,
        const float * Sr, const float * Sc, const void * V_f16, const float * mask,
        float * O, int head_dim, int n_head, int n_head_kv, int n_tok, int n_kv, float scale);

// Tiled fused FA v2, increment 1 (test scaffolding): coalesced channel-major K tile
// load+dequant. K = [head_dim] channel-major blocks for one (head, group); Sr [head_dim];
// Sc [G]; out = [head_dim * T_tile] token-major, out[d*T_tile+tt] = (code+z_d)*s_d*Sr_d*Sc_t.
extern "C" void kvarn_ktile_load_cuda(const void * K, const float * Sr, const float * Sc,
        float * out, int head_dim, int G, int t0, int T_tile);
