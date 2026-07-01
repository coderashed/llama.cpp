// GPU VarN kernel (KVARN_FAITHFUL/04, GPU-ization step). Faithful port of the CPU
// kvarn_variance_normalize (Algorithm 1, SINQ log-domain std-scaling) to a device
// kernel so VarN no longer forces a CPU custom-op round-trip. One block per tile,
// single thread per block (parallel across tiles, sequential within) -- simple and
// bit-close to the CPU reference. The 128x128 working tile lives in global scratch;
// only the log-scale vectors are per-thread local. best_T is not stored: by the
// invariant orig = T*exp(L_c)*exp(L_r), the best tile is recomputed as
// in / exp(best_L_c) / exp(best_L_r).
//
// NOTE: not yet wired into the graph (ggml custom ops are CPU-only; graph use needs
// a new GGML op or a backend hook). This is the reusable kernel + host wrapper +
// bit-close test, mirroring the Phase A per-channel quantizer pattern.

#include "common.cuh"

#define VARN_MAXDIM 128
#define VARN_VAR_EPS 1e-12
#define VARN_K       12
#define VARN_CMIN    (-5.0f)
#define VARN_CMAX    ( 5.0f)

static __device__ __forceinline__ float varn_clampf(float x, float lo, float hi) {
    return x < lo ? lo : (x > hi ? hi : x);
}

// One block per tile; thread 0 runs the whole VarN for its [R x C] tile (row-major).
static __global__ void kvarn_varn_tile_kernel(
        const float * __restrict__ Tin, float * __restrict__ Tout,
        float * __restrict__ Sr, float * __restrict__ Sc, int R, int C) {
    if (threadIdx.x != 0) {
        return;
    }
    const int tile = blockIdx.x;
    const float * in   = Tin  + (size_t) tile * R * C;
    float *       work = Tout + (size_t) tile * R * C;
    float *       sr   = Sr   + (size_t) tile * R;
    float *       sc   = Sc   + (size_t) tile * C;

    float L_c[VARN_MAXDIM], L_r[VARN_MAXDIM], bLc[VARN_MAXDIM], bLr[VARN_MAXDIM];
    for (int j = 0; j < C; j++) { L_c[j] = 0.0f; bLc[j] = 0.0f; }
    for (int i = 0; i < R; i++) { L_r[i] = 0.0f; bLr[i] = 0.0f; }
    for (int k = 0; k < R*C; k++) work[k] = in[k];

    double best_imb = 1e300; // FLT_MAX-equivalent; CPU snapshots only when improved

    for (int iter = 0; iter < VARN_K; iter++) {
        // --- normalize columns ---
        double tile_mean = 0.0;
        for (int k = 0; k < R*C; k++) tile_mean += (double) work[k];
        tile_mean /= (double)(R*C);
        for (int j = 0; j < C; j++) {
            double mean = 0.0;
            for (int i = 0; i < R; i++) mean += (double) work[i*C + j];
            mean /= (double) R;
            double m2 = 0.0;
            for (int i = 0; i < R; i++) { double d = (double) work[i*C + j] - mean; m2 += d*d; }
            double v = m2 / (double) R;
            if (v <= VARN_VAR_EPS) { double dev = mean - tile_mean; v = dev*dev; if (v <= VARN_VAR_EPS) continue; }
            double delta = 0.5 * log(v);
            float  Lnew  = varn_clampf((float)((double) L_c[j] + delta), VARN_CMIN, VARN_CMAX);
            float  f     = expf(Lnew - L_c[j]);
            for (int i = 0; i < R; i++) work[i*C + j] /= f;
            L_c[j] = Lnew;
        }
        // --- normalize rows ---
        tile_mean = 0.0;
        for (int k = 0; k < R*C; k++) tile_mean += (double) work[k];
        tile_mean /= (double)(R*C);
        for (int i = 0; i < R; i++) {
            double mean = 0.0;
            for (int j = 0; j < C; j++) mean += (double) work[i*C + j];
            mean /= (double) C;
            double m2 = 0.0;
            for (int j = 0; j < C; j++) { double d = (double) work[i*C + j] - mean; m2 += d*d; }
            double v = m2 / (double) C;
            if (v <= VARN_VAR_EPS) { double dev = mean - tile_mean; v = dev*dev; if (v <= VARN_VAR_EPS) continue; }
            double delta = 0.5 * log(v);
            float  Lnew  = varn_clampf((float)((double) L_r[i] + delta), VARN_CMIN, VARN_CMAX);
            float  f     = expf(Lnew - L_r[i]);
            for (int j = 0; j < C; j++) work[i*C + j] /= f;
            L_r[i] = Lnew;
        }
        // --- imbalance metric (product form, matches CPU) ---
        float minc = 3.4e38f, maxc = -3.4e38f, minr = 3.4e38f, maxr = -3.4e38f;
        for (int j = 0; j < C; j++) {
            double mean = 0.0, m2 = 0.0;
            for (int i = 0; i < R; i++) mean += work[i*C + j];
            mean /= (double) R;
            for (int i = 0; i < R; i++) { double d = work[i*C + j] - mean; m2 += d*d; }
            float vv = (float)(m2 / (double) R);
            if (vv < minc) minc = vv; if (vv > maxc) maxc = vv;
        }
        for (int i = 0; i < R; i++) {
            double mean = 0.0, m2 = 0.0;
            for (int j = 0; j < C; j++) mean += work[i*C + j];
            mean /= (double) C;
            for (int j = 0; j < C; j++) { double d = work[i*C + j] - mean; m2 += d*d; }
            float vv = (float)(m2 / (double) C);
            if (vv < minr) minr = vv; if (vv > maxr) maxr = vv;
        }
        const float eps = 1e-8f;
        float imb = (maxc / fmaxf(minc, eps)) * (maxr / fmaxf(minr, eps));
        if ((double) imb < best_imb) {
            best_imb = imb;
            for (int j = 0; j < C; j++) bLc[j] = L_c[j];
            for (int i = 0; i < R; i++) bLr[i] = L_r[i];
        }
    }

    for (int j = 0; j < C; j++) sc[j] = expf(bLc[j]);
    for (int i = 0; i < R; i++) sr[i] = expf(bLr[i]);
    for (int i = 0; i < R; i++)
        for (int j = 0; j < C; j++)
            work[i*C + j] = in[i*C + j] / sr[i] / sc[j];
}

// Host wrapper: run VarN on n_tiles [R x C] tiles. Tnorm/Sr/Sc are host buffers.
extern "C" void kvarn_varn_tile_cuda(const float * tiles, float * Tnorm,
        float * Sr, float * Sc, int n_tiles, int R, int C) {
    GGML_ASSERT(R <= VARN_MAXDIM && C <= VARN_MAXDIM);
    const size_t nt = (size_t) n_tiles * R * C;
    float *dTin = nullptr, *dTout = nullptr, *dSr = nullptr, *dSc = nullptr;
    CUDA_CHECK(cudaMalloc(&dTin,  nt * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dTout, nt * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dSr, (size_t) n_tiles * R * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dSc, (size_t) n_tiles * C * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(dTin, tiles, nt * sizeof(float), cudaMemcpyHostToDevice));
    kvarn_varn_tile_kernel<<<n_tiles, 1>>>(dTin, dTout, dSr, dSc, R, C);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(Tnorm, dTout, nt * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(Sr, dSr, (size_t) n_tiles * R * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(Sc, dSc, (size_t) n_tiles * C * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(dTin)); CUDA_CHECK(cudaFree(dTout));
    CUDA_CHECK(cudaFree(dSr));  CUDA_CHECK(cudaFree(dSc));
}
