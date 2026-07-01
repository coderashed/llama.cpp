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
