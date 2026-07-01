// Bespoke per-channel-K attention (KVARN_FAITHFUL, fused FA read path).
//
// block_q2_kvarn_k is CHANNEL-MAJOR: the block for (channel c, group g) in a k_body of
// shape [G, C, n_groups] is at linear block index c + g*C, and key token t_local's code
// is qs[t_local/4] >> ((t_local&3)*2) & 3. llama.cpp's FA vec machinery is token-major
// and cannot express this gather, so per-channel K needs a bespoke kernel. This file is
// built incrementally, tested per layer:
//   [this increment] KQ scores: for each (query q, key token t)
//       score[q,t] = S_c[t] * sum_c Q[c,q] * (code_{c,t} + z[c]) * (s[c] * S_r[c])
//   pinned host-side by test-q2-kvarn-k-fa; validated on-device by test-q2-kvarn-k-fa-gpu.
// Later increments add online-softmax + V weighted-sum and graph wiring.

#include "common.cuh"
#include "kvarn-fa.cuh"

// One block per query column; blockDim.x threads cooperate over key tokens.
// K is one group: C blocks of block_q2_kvarn_k covering G key tokens (channel-major).
static __global__ void kvarn_kq_scores_kernel(
        const block_q2_kvarn_k * __restrict__ K, const float * __restrict__ Q,
        const float * __restrict__ Sr, const float * __restrict__ Sc,
        float * __restrict__ scores, int D, int n_q, int G) {
    const int q = blockIdx.x;
    const float * Qq = Q + (size_t) q * D; // query column [D]

    for (int t = threadIdx.x; t < G; t += blockDim.x) {
        float acc = 0.0f;
        const int byte_idx = t >> 2;
        const int shift    = (t & 3) * 2;
        for (int c = 0; c < D; c++) {
            const block_q2_kvarn_k & b = K[c];
            const float s = __half2float(b.s);
            const float z = __half2float(b.z);
            const int code = (b.qs[byte_idx] >> shift) & 0x03;
            acc += Qq[c] * ((float)code + z) * (s * Sr[c]);
        }
        scores[(size_t) q * G + t] = Sc[t] * acc;
    }
}

// Host wrapper: one group, single head. Kblocks is the raw channel-major block bytes
// (C * sizeof(block_q2_kvarn_k)); Q is [D, n_q] column-major; scores is [n_q, G].
extern "C" void kvarn_kq_scores_cuda(const void * Kblocks, const float * Q,
        const float * Sr, const float * Sc, float * scores, int D, int n_q, int G) {
    const size_t kbytes = (size_t) D * sizeof(block_q2_kvarn_k);
    block_q2_kvarn_k * dK = nullptr;
    float *dQ = nullptr, *dSr = nullptr, *dSc = nullptr, *dSco = nullptr;
    CUDA_CHECK(cudaMalloc(&dK,  kbytes));
    CUDA_CHECK(cudaMalloc(&dQ,  (size_t) D * n_q * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dSr, (size_t) D * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dSc, (size_t) G * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dSco, (size_t) n_q * G * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(dK, Kblocks, kbytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dQ, Q, (size_t) D * n_q * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dSr, Sr, (size_t) D * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dSc, Sc, (size_t) G * sizeof(float), cudaMemcpyHostToDevice));

    kvarn_kq_scores_kernel<<<n_q, 128>>>(dK, dQ, dSr, dSc, dSco, D, n_q, G);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(scores, dSco, (size_t) n_q * G * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(dK));  CUDA_CHECK(cudaFree(dQ));  CUDA_CHECK(cudaFree(dSr));
    CUDA_CHECK(cudaFree(dSc)); CUDA_CHECK(cudaFree(dSco));
}

// Increment 3: full single-group, single-head attention.
//   O[c,q] = sum_t softmax_t(scale*KQ[q,t] + mask[t,q]) * V[c,t]
// with KQ the per-channel K-dot above. One block per query; 128 threads. V is F16,
// token-major [D, G] (V[c + t*D]); mask is [G, n_q] F32 (0 keep, -inf drop). Numerically
// stable (subtract row max). This isolates the softmax + V accumulation; q4_0 V and
// multi-group merging are handled at wiring time.
static __global__ void kvarn_fa_group_kernel(
        const block_q2_kvarn_k * __restrict__ K, const float * __restrict__ Q,
        const float * __restrict__ Sr, const float * __restrict__ Sc,
        const half * __restrict__ V, const float * __restrict__ mask,
        float * __restrict__ O, int D, int n_q, int G, float scale) {
    const int q  = blockIdx.x;
    const int tx = threadIdx.x;          // 0..127
    const float * Qq = Q + (size_t) q * D;

    __shared__ float sc[128];            // scores / probabilities for this query
    __shared__ float red[128];           // reduction scratch

    // Scores (one thread per key token; G == 128).
    float score = -INFINITY;
    if (tx < G) {
        const int byte_idx = tx >> 2;
        const int shift    = (tx & 3) * 2;
        float acc = 0.0f;
        for (int c = 0; c < D; c++) {
            const block_q2_kvarn_k & b = K[c];
            const int code = (b.qs[byte_idx] >> shift) & 0x03;
            acc += Qq[c] * ((float)code + __half2float(b.z)) * (__half2float(b.s) * Sr[c]);
        }
        score = scale * Sc[tx] * acc + mask[(size_t) q * G + tx];
    }
    sc[tx] = score;
    __syncthreads();

    // Max-reduce.
    red[tx] = score;
    __syncthreads();
    for (int stride = 64; stride > 0; stride >>= 1) {
        if (tx < stride) red[tx] = fmaxf(red[tx], red[tx + stride]);
        __syncthreads();
    }
    const float m = red[0];
    __syncthreads();

    // exp and sum-reduce.
    const float e = (tx < G) ? expf(sc[tx] - m) : 0.0f;
    sc[tx] = e;
    red[tx] = e;
    __syncthreads();
    for (int stride = 64; stride > 0; stride >>= 1) {
        if (tx < stride) red[tx] += red[tx + stride];
        __syncthreads();
    }
    const float denom = red[0];
    const float inv = denom > 0.0f ? 1.0f / denom : 0.0f;
    __syncthreads();

    // O[c] = sum_t p[t] * V[c,t]. One thread per output channel (D == 128).
    if (tx < D) {
        float o = 0.0f;
        for (int t = 0; t < G; t++) {
            o += sc[t] * __half2float(V[(size_t) t * D + tx]);
        }
        O[(size_t) q * D + tx] = o * inv;
    }
}

// Increment 4: generalized attention over n_kv keys (multiple groups) with GQA
// multi-head. This is the real op's compute. Logical (contiguous) layouts:
//   Q  [head_dim, n_head, n_tok]    Q[d + head_dim*(h + n_head*qt)]
//   K  channel-major blocks: block(kv head hk, dim d, group g) at index
//        (hk*head_dim + d) + g*C, C = head_dim*n_head_kv; code = qs[t/4], t = T%G.
//   Sr [head_dim, n_head_kv, ng]    Sr[d + head_dim*(hk + n_head_kv*g)]
//   Sc [G,        n_head_kv, ng]    Sc[t + G*(hk + n_head_kv*g)]
//   V  [head_dim, n_head_kv, n_kv]  half, V[d + head_dim*(hk + n_head_kv*T)]
//   mask [n_kv, n_tok]              mask[T + n_kv*qt]  (0 keep, -inf drop)
//   O  [head_dim, n_head, n_tok]    O[d + head_dim*(h + n_head*qt)]
// GQA: hk = h / (n_head / n_head_kv). One block per (qt, h); 128 threads; scores live
// in dynamic shared memory (n_kv floats). G is fixed at QG2_KVARN (128).
static __global__ void kvarn_fa_kernel(
        const block_q2_kvarn_k * __restrict__ K, const float * __restrict__ Q,
        const float * __restrict__ Sr, const float * __restrict__ Sc,
        const half * __restrict__ V, const float * __restrict__ mask,
        float * __restrict__ O, int head_dim, int n_head, int n_head_kv,
        int n_tok, int n_kv, float scale, int mask_stride) {
    extern __shared__ float sh[];        // [n_kv] scores/probabilities
    const int qt = blockIdx.x;
    const int h  = blockIdx.y;
    const int hk = h / (n_head / n_head_kv);
    const int tx = threadIdx.x;
    const int G  = QG2_KVARN;
    const int C  = head_dim * n_head_kv;

    const float * Qq = Q + (size_t) head_dim * (h + (size_t) n_head * qt);

    // Scores for all keys.
    for (int T = tx; T < n_kv; T += blockDim.x) {
        const int g = T / G, t = T % G;
        const int byte_idx = t >> 2, shift = (t & 3) * 2;
        const block_q2_kvarn_k * Kg = K + (size_t) g * C + (size_t) hk * head_dim;
        const float * Srg = Sr + (size_t) head_dim * (hk + (size_t) n_head_kv * g);
        float acc = 0.0f;
        for (int d = 0; d < head_dim; d++) {
            const block_q2_kvarn_k & b = Kg[d];
            const int code = (b.qs[byte_idx] >> shift) & 0x03;
            acc += Qq[d] * ((float)code + __half2float(b.z)) * (__half2float(b.s) * Srg[d]);
        }
        const float sc_t = Sc[t + (size_t) G * (hk + (size_t) n_head_kv * g)];
        sh[T] = scale * sc_t * acc + mask[(size_t) T + (size_t) mask_stride * qt];
    }
    __syncthreads();

    // Parallel stable softmax over n_kv: block-reduce max, then exp + block-reduce sum.
    // blockDim.x is a power of two (128); threads past n_kv contribute the identity.
    __shared__ float red[128];
    float pm = -INFINITY;
    for (int T = tx; T < n_kv; T += blockDim.x) pm = fmaxf(pm, sh[T]);
    red[tx] = pm; __syncthreads();
    for (int s = blockDim.x >> 1; s > 0; s >>= 1) { if (tx < s) red[tx] = fmaxf(red[tx], red[tx+s]); __syncthreads(); }
    const float m = red[0];
    __syncthreads();

    float ps = 0.0f;
    for (int T = tx; T < n_kv; T += blockDim.x) { float e = expf(sh[T] - m); sh[T] = e; ps += e; }
    red[tx] = ps; __syncthreads();
    for (int s = blockDim.x >> 1; s > 0; s >>= 1) { if (tx < s) red[tx] += red[tx+s]; __syncthreads(); }
    const float denom = red[0];
    __syncthreads();
    const float inv = denom > 0.0f ? 1.0f / denom : 0.0f;

    // O[d] = sum_T p[T] * V(d,hk,T). Stride channels so head_dim > blockDim (e.g. 256) is covered.
    for (int d = tx; d < head_dim; d += blockDim.x) {
        float o = 0.0f;
        for (int T = 0; T < n_kv; T++) {
            o += sh[T] * __half2float(V[(size_t) head_dim * (hk + (size_t) n_head_kv * T) + d]);
        }
        O[(size_t) head_dim * (h + (size_t) n_head * qt) + d] = o * inv;
    }
}

extern "C" void kvarn_fa_cuda(const void * Kblocks, const float * Q,
        const float * Sr, const float * Sc, const void * V_f16, const float * mask,
        float * O, int head_dim, int n_head, int n_head_kv, int n_tok, int n_kv, float scale) {
    const int G = QG2_KVARN;
    const int n_groups = n_kv / G;
    const int C = head_dim * n_head_kv;
    const size_t kbytes = (size_t) C * n_groups * sizeof(block_q2_kvarn_k);
    block_q2_kvarn_k * dK = nullptr;
    float *dQ=nullptr,*dSr=nullptr,*dSc=nullptr,*dmask=nullptr,*dO=nullptr; half * dV=nullptr;
    CUDA_CHECK(cudaMalloc(&dK, kbytes));
    CUDA_CHECK(cudaMalloc(&dQ, (size_t) head_dim*n_head*n_tok*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dSr, (size_t) head_dim*n_head_kv*n_groups*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dSc, (size_t) G*n_head_kv*n_groups*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dV, (size_t) head_dim*n_head_kv*n_kv*sizeof(half)));
    CUDA_CHECK(cudaMalloc(&dmask, (size_t) n_kv*n_tok*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dO, (size_t) head_dim*n_head*n_tok*sizeof(float)));
    CUDA_CHECK(cudaMemcpy(dK, Kblocks, kbytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dQ, Q, (size_t) head_dim*n_head*n_tok*sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dSr, Sr, (size_t) head_dim*n_head_kv*n_groups*sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dSc, Sc, (size_t) G*n_head_kv*n_groups*sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dV, V_f16, (size_t) head_dim*n_head_kv*n_kv*sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dmask, mask, (size_t) n_kv*n_tok*sizeof(float), cudaMemcpyHostToDevice));

    dim3 grid(n_tok, n_head);
    kvarn_fa_kernel<<<grid, 128, n_kv*sizeof(float)>>>(dK, dQ, dSr, dSc, dV, dmask, dO,
            head_dim, n_head, n_head_kv, n_tok, n_kv, scale, n_kv);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(O, dO, (size_t) head_dim*n_head*n_tok*sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(dK)); CUDA_CHECK(cudaFree(dQ)); CUDA_CHECK(cudaFree(dSr));
    CUDA_CHECK(cudaFree(dSc)); CUDA_CHECK(cudaFree(dV)); CUDA_CHECK(cudaFree(dmask)); CUDA_CHECK(cudaFree(dO));
}

// Graph op forward (GGML_OP_KVARN_FA). srcs: q [head_dim,n_head,n_tok] F32,
// k_body [G,C,n_groups] Q2_KVARN_K, S_r [head_dim,n_head_kv,1,ng], S_c [G,n_head_kv,1,ng],
// V F16 [head_dim,n_head_kv,n_kv], mask F32 [n_kv_pad,n_tok_pad]. scale in op_params[0].
// dst F32 [head_dim,n_head,n_tok]. All device tensors; no host round-trip.
void ggml_cuda_op_kvarn_fa(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * q  = dst->src[0];
    const ggml_tensor * kb = dst->src[1];
    const ggml_tensor * sr = dst->src[2];
    const ggml_tensor * sc = dst->src[3];
    const ggml_tensor * v  = dst->src[4];
    const ggml_tensor * mask = dst->src[5];
    GGML_ASSERT(q->type == GGML_TYPE_F32 && sr->type == GGML_TYPE_F32);
    GGML_ASSERT(sc->type == GGML_TYPE_F32 && mask->type == GGML_TYPE_F32);
    GGML_ASSERT(v->type == GGML_TYPE_F16 && dst->type == GGML_TYPE_F32);
    GGML_ASSERT(ggml_is_contiguous(q) && ggml_is_contiguous(kb));
    GGML_ASSERT(ggml_is_contiguous(sr) && ggml_is_contiguous(sc) && ggml_is_contiguous(v));

    const int head_dim  = (int) q->ne[0];
    const int n_head    = (int) q->ne[1];
    const int n_tok     = (int) q->ne[2];
    const int n_head_kv = (int) sr->ne[1];
    const int n_kv      = (int) v->ne[2];
    const int mask_stride = (int) (mask->nb[1] / sizeof(float));
    GGML_ASSERT(head_dim <= 256 && (n_kv % QG2_KVARN) == 0 && (n_head % n_head_kv) == 0);

    float scale;
    memcpy(&scale, dst->op_params, sizeof(float));

    dim3 grid(n_tok, n_head);
    kvarn_fa_kernel<<<grid, 128, (size_t) n_kv*sizeof(float), ctx.stream()>>>(
            (const block_q2_kvarn_k *) kb->data, (const float *) q->data,
            (const float *) sr->data, (const float *) sc->data, (const half *) v->data,
            (const float *) mask->data, (float *) dst->data,
            head_dim, n_head, n_head_kv, n_tok, n_kv, scale, mask_stride);
    CUDA_CHECK(cudaGetLastError());
}

extern "C" void kvarn_fa_group_cuda(const void * Kblocks, const float * Q,
        const float * Sr, const float * Sc, const void * V_f16, const float * mask,
        float * O, int D, int n_q, int G, float scale) {
    const size_t kbytes = (size_t) D * sizeof(block_q2_kvarn_k);
    block_q2_kvarn_k * dK = nullptr;
    float *dQ = nullptr, *dSr = nullptr, *dSc = nullptr, *dmask = nullptr, *dO = nullptr;
    half * dV = nullptr;
    CUDA_CHECK(cudaMalloc(&dK, kbytes));
    CUDA_CHECK(cudaMalloc(&dQ, (size_t) D * n_q * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dSr, (size_t) D * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dSc, (size_t) G * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dV, (size_t) D * G * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&dmask, (size_t) n_q * G * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dO, (size_t) D * n_q * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(dK, Kblocks, kbytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dQ, Q, (size_t) D * n_q * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dSr, Sr, (size_t) D * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dSc, Sc, (size_t) G * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dV, V_f16, (size_t) D * G * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dmask, mask, (size_t) n_q * G * sizeof(float), cudaMemcpyHostToDevice));

    kvarn_fa_group_kernel<<<n_q, 128>>>(dK, dQ, dSr, dSc, dV, dmask, dO, D, n_q, G, scale);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(O, dO, (size_t) D * n_q * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(dK));  CUDA_CHECK(cudaFree(dQ)); CUDA_CHECK(cudaFree(dSr));
    CUDA_CHECK(cudaFree(dSc)); CUDA_CHECK(cudaFree(dV)); CUDA_CHECK(cudaFree(dmask));
    CUDA_CHECK(cudaFree(dO));
}

// ============================================================================
// Tiled fused FA v2 (design: .ai/design/kvarn_tiled_fused_fa_design.md).
// Increment 1: coalesced channel-major K tile load + dequant.
// ============================================================================

// Load+dequant a [head_dim x T_tile] K tile for one (head hk, group g) into `kf`,
// token-major: kf[d*T_tile + tt] = dequant of channel d, token t0+tt, VarN scales
// folded in as (code + z_d) * s_d * Sr_d * Sc_t. Kg[d] is channel d's block, so
// consecutive threads read consecutive blocks -> COALESCED; each thread reads its
// block once and extracts its T_tile token codes. This is the load primitive the
// tiled kernel is built on.
static __device__ __forceinline__ void kvarn_load_ktile(
        const block_q2_kvarn_k * __restrict__ Kg,   // [head_dim] blocks for (hk, g)
        const float * __restrict__ Srg,             // [head_dim] Sr for (hk, g)
        const float * __restrict__ Scg,             // [G] Sc for (hk, g), per token in group
        float * __restrict__ kf,                    // out tile [head_dim * T_tile], token-major
        int head_dim, int t0, int T_tile) {
    for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
        const block_q2_kvarn_k b = Kg[d];
        const float ssr = __half2float(b.s) * Srg[d];   // s[d] * Sr[d]
        const float zd  = __half2float(b.z);
        for (int tt = 0; tt < T_tile; ++tt) {
            const int t    = t0 + tt;
            const int code = (b.qs[t >> 2] >> ((t & 3) * 2)) & 0x03;
            kf[(size_t) d * T_tile + tt] = ((float) code + zd) * ssr * Scg[t];
        }
    }
}

// Test scaffolding: run kvarn_load_ktile for a single (head, group), write the tile
// to global memory for host readback. One block.
static __global__ void kvarn_ktile_load_test_kernel(
        const block_q2_kvarn_k * __restrict__ Kg, const float * __restrict__ Srg,
        const float * __restrict__ Scg, float * __restrict__ out,
        int head_dim, int t0, int T_tile) {
    kvarn_load_ktile(Kg, Srg, Scg, out, head_dim, t0, T_tile);
}

// Host wrapper (test only): K = [head_dim] blocks, Sr = [head_dim], Sc = [G];
// out = [head_dim * T_tile] token-major dequant of tokens [t0, t0+T_tile).
extern "C" void kvarn_ktile_load_cuda(const void * K, const float * Sr, const float * Sc,
        float * out, int head_dim, int G, int t0, int T_tile) {
    const size_t kbytes = (size_t) head_dim * sizeof(block_q2_kvarn_k);
    block_q2_kvarn_k * dK = nullptr;
    float * dSr = nullptr, * dSc = nullptr, * dOut = nullptr;
    CUDA_CHECK(cudaMalloc(&dK,   kbytes));
    CUDA_CHECK(cudaMalloc(&dSr,  (size_t) head_dim * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dSc,  (size_t) G * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dOut, (size_t) head_dim * T_tile * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(dK,  K,  kbytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dSr, Sr, (size_t) head_dim * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dSc, Sc, (size_t) G * sizeof(float), cudaMemcpyHostToDevice));
    const int nthreads = head_dim < 256 ? head_dim : 256;
    kvarn_ktile_load_test_kernel<<<1, nthreads>>>(dK, dSr, dSc, dOut, head_dim, t0, T_tile);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(out, dOut, (size_t) head_dim * T_tile * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(dK)); CUDA_CHECK(cudaFree(dSr)); CUDA_CHECK(cudaFree(dSc)); CUDA_CHECK(cudaFree(dOut));
}
