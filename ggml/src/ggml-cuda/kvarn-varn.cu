// GPU VarN kernel (KVARN_FAITHFUL/04, GPU-ization step). Faithful port of the CPU
// kvarn_variance_normalize (Algorithm 1, SINQ log-domain std-scaling). One block per
// tile, VARN_NTHREADS threads per block cooperating on the [R x C] tile (row-major).
// Within a VarN step the columns are mutually independent (each touches only its own
// column) and likewise the rows, so thread j owns column j / thread i owns row i and
// keeps the same sequential double-precision mean/variance the CPU uses -- bit-close by
// construction. Only the tile_mean (used solely in the degenerate v<=eps branch) and
// the imbalance min/max are cross-thread reductions. The 128x128 working tile lives in
// global scratch; log-scale vectors live in shared memory. best_T is not stored: by the
// invariant orig = T*exp(L_c)*exp(L_r), the best tile is recomputed as
// in / exp(best_L_c) / exp(best_L_r).

#include "common.cuh"
#include "kvarn-varn.cuh"

#define VARN_MAXDIM  256
#define VARN_NTHREADS 128
#define VARN_VAR_EPS 1e-12
#define VARN_K       12
#define VARN_CMIN    (-5.0f)
#define VARN_CMAX    ( 5.0f)

static __device__ __forceinline__ float varn_clampf(float x, float lo, float hi) {
    return x < lo ? lo : (x > hi ? hi : x);
}

// Block-wide tile mean over R*C floats (double accumulation). All threads participate.
static __device__ __forceinline__ double varn_block_tile_mean(
        const float * __restrict__ work, int RC, int t, int nth, double * red) {
    double part = 0.0;
    for (int k = t; k < RC; k += nth) part += (double) work[k];
    red[t] = part;
    __syncthreads();
    for (int s = nth >> 1; s > 0; s >>= 1) { if (t < s) red[t] += red[t + s]; __syncthreads(); }
    double m = red[0] / (double) RC;
    __syncthreads();
    return m;
}

// One block per tile; VARN_NTHREADS threads run VarN for the [R x C] tile (row-major).
static __global__ void kvarn_varn_tile_kernel(
        const float * __restrict__ Tin, float * __restrict__ Tout,
        float * __restrict__ Sr, float * __restrict__ Sc, int R, int C) {
    const int tile = blockIdx.x;
    const int t    = threadIdx.x;
    const int nth  = blockDim.x;
    const int RC   = R * C;
    const float * in   = Tin  + (size_t) tile * RC;
    float *       work = Tout + (size_t) tile * RC;
    float *       sr   = Sr   + (size_t) tile * R;
    float *       sc   = Sc   + (size_t) tile * C;

    __shared__ float  L_c[VARN_MAXDIM], L_r[VARN_MAXDIM], bLc[VARN_MAXDIM], bLr[VARN_MAXDIM];
    __shared__ double red[VARN_NTHREADS];
    __shared__ float  redf[VARN_NTHREADS];
    __shared__ double s_best_imb;
    __shared__ int    s_improved;

    for (int j = t; j < C;  j += nth) { L_c[j] = 0.0f; bLc[j] = 0.0f; }
    for (int i = t; i < R;  i += nth) { L_r[i] = 0.0f; bLr[i] = 0.0f; }
    for (int k = t; k < RC; k += nth) work[k] = in[k];
    if (t == 0) s_best_imb = 1e300; // CPU snapshots only when improved
    __syncthreads();

    for (int iter = 0; iter < VARN_K; iter++) {
        // --- normalize columns (each thread owns a strided set of columns) ---
        double tile_mean = varn_block_tile_mean(work, RC, t, nth, red);
        for (int j = t; j < C; j += nth) {
            double mean = 0.0;
            for (int i = 0; i < R; i++) mean += (double) work[i*C + j];
            mean /= (double) R;
            double m2 = 0.0;
            for (int i = 0; i < R; i++) { double d = (double) work[i*C + j] - mean; m2 += d*d; }
            double v = m2 / (double) R;
            bool skip = false;
            if (v <= VARN_VAR_EPS) { double dev = mean - tile_mean; v = dev*dev; if (v <= VARN_VAR_EPS) skip = true; }
            if (!skip) {
                double delta = 0.5 * log(v);
                float  Lnew  = varn_clampf((float)((double) L_c[j] + delta), VARN_CMIN, VARN_CMAX);
                float  f     = expf(Lnew - L_c[j]);
                for (int i = 0; i < R; i++) work[i*C + j] /= f;
                L_c[j] = Lnew;
            }
        }
        __syncthreads();

        // --- normalize rows (each thread owns a strided set of rows) ---
        tile_mean = varn_block_tile_mean(work, RC, t, nth, red);
        for (int i = t; i < R; i += nth) {
            double mean = 0.0;
            for (int j = 0; j < C; j++) mean += (double) work[i*C + j];
            mean /= (double) C;
            double m2 = 0.0;
            for (int j = 0; j < C; j++) { double d = (double) work[i*C + j] - mean; m2 += d*d; }
            double v = m2 / (double) C;
            bool skip = false;
            if (v <= VARN_VAR_EPS) { double dev = mean - tile_mean; v = dev*dev; if (v <= VARN_VAR_EPS) skip = true; }
            if (!skip) {
                double delta = 0.5 * log(v);
                float  Lnew  = varn_clampf((float)((double) L_r[i] + delta), VARN_CMIN, VARN_CMAX);
                float  f     = expf(Lnew - L_r[i]);
                for (int j = 0; j < C; j++) work[i*C + j] /= f;
                L_r[i] = Lnew;
            }
        }
        __syncthreads();

        // --- imbalance metric (product form, matches CPU): min/max of col & row var ---
        float my_min = 3.4e38f, my_max = -3.4e38f;
        for (int j = t; j < C; j += nth) {
            double mean = 0.0, m2 = 0.0;
            for (int i = 0; i < R; i++) mean += work[i*C + j];
            mean /= (double) R;
            for (int i = 0; i < R; i++) { double d = work[i*C + j] - mean; m2 += d*d; }
            float vv = (float)(m2 / (double) R);
            my_min = fminf(my_min, vv); my_max = fmaxf(my_max, vv);
        }
        redf[t] = my_min; __syncthreads();
        for (int s = nth >> 1; s > 0; s >>= 1) { if (t < s) redf[t] = fminf(redf[t], redf[t+s]); __syncthreads(); }
        float minc = redf[0]; __syncthreads();
        redf[t] = my_max; __syncthreads();
        for (int s = nth >> 1; s > 0; s >>= 1) { if (t < s) redf[t] = fmaxf(redf[t], redf[t+s]); __syncthreads(); }
        float maxc = redf[0]; __syncthreads();

        my_min = 3.4e38f; my_max = -3.4e38f;
        for (int i = t; i < R; i += nth) {
            double mean = 0.0, m2 = 0.0;
            for (int j = 0; j < C; j++) mean += work[i*C + j];
            mean /= (double) C;
            for (int j = 0; j < C; j++) { double d = work[i*C + j] - mean; m2 += d*d; }
            float vv = (float)(m2 / (double) C);
            my_min = fminf(my_min, vv); my_max = fmaxf(my_max, vv);
        }
        redf[t] = my_min; __syncthreads();
        for (int s = nth >> 1; s > 0; s >>= 1) { if (t < s) redf[t] = fminf(redf[t], redf[t+s]); __syncthreads(); }
        float minr = redf[0]; __syncthreads();
        redf[t] = my_max; __syncthreads();
        for (int s = nth >> 1; s > 0; s >>= 1) { if (t < s) redf[t] = fmaxf(redf[t], redf[t+s]); __syncthreads(); }
        float maxr = redf[0]; __syncthreads();

        if (t == 0) {
            const float eps = 1e-8f;
            float imb = (maxc / fmaxf(minc, eps)) * (maxr / fmaxf(minr, eps));
            s_improved = ((double) imb < s_best_imb) ? 1 : 0;
            if (s_improved) s_best_imb = imb;
        }
        __syncthreads();
        if (s_improved) {
            for (int j = t; j < C; j += nth) bLc[j] = L_c[j];
            for (int i = t; i < R; i += nth) bLr[i] = L_r[i];
        }
        __syncthreads();
    }

    for (int j = t; j < C; j += nth) sc[j] = expf(bLc[j]);
    for (int i = t; i < R; i += nth) sr[i] = expf(bLr[i]);
    __syncthreads();
    for (int k = t; k < RC; k += nth) { int i = k / C, j = k % C; work[k] = in[k] / sr[i] / sc[j]; }
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
    kvarn_varn_tile_kernel<<<n_tiles, VARN_NTHREADS>>>(dTin, dTout, dSr, dSc, R, C);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(Tnorm, dTout, nt * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(Sr, dSr, (size_t) n_tiles * R * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(Sc, dSc, (size_t) n_tiles * C * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(dTin)); CUDA_CHECK(cudaFree(dTout));
    CUDA_CHECK(cudaFree(dSr));  CUDA_CHECK(cudaFree(dSc));
}

// Graph op forward (GGML_OP_KVARN_VARN). No host round-trip: src and the packed dst
// live on the device. One block per head; the kernel writes T_norm/S_r/S_c straight
// into the packed dst by pointer offset (Sr = dst+nt, Sc = dst+nt+n_ch), which lines
// up with the head-major CPU custom-op layout because per head the kernel strides are
// tile*R*C (T_norm), tile*R (S_r) and tile*C (S_c).
void ggml_cuda_op_kvarn_varn(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src = dst->src[0];
    GGML_ASSERT(src->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);
    GGML_ASSERT(ggml_is_contiguous(src));

    const int n_tok    = (int) src->ne[0];
    const int head_dim = (int) src->ne[1];
    const int n_head   = (int) src->ne[2];
    GGML_ASSERT(head_dim <= VARN_MAXDIM && n_tok <= VARN_MAXDIM);

    const size_t nt   = (size_t) head_dim * n_head * n_tok; // T_norm length (n_ch*n_tok)
    const size_t n_ch = (size_t) head_dim * n_head;

    const float * src_d = (const float *) src->data;
    float *       dst_d = (float *)       dst->data;
    float *       Sr    = dst_d + nt;
    float *       Sc    = dst_d + nt + n_ch;

    kvarn_varn_tile_kernel<<<n_head, VARN_NTHREADS, 0, ctx.stream()>>>(
            src_d, dst_d, Sr, Sc, head_dim, n_tok);
    CUDA_CHECK(cudaGetLastError());
}
