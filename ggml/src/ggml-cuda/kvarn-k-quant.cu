// Phase A per-channel K quantizer kernel + host wrapper (KVARN_FAITHFUL/03).
// Exposes quantize_k_q2_kvarn_perchannel_tile_cuda for tests and future
// use by the cache write path (Phase B).

#include "common.cuh"
#include "cpy-utils.cuh"

// One thread per channel; all channels in a single CUDA block.
// tile is channel-major [n_ch x n_tok]; out has n_ch blocks.
static __global__ void quantize_k_q2_kvarn_perchannel_kernel(
        const float * __restrict__ tile,
        block_q2_kvarn_k * __restrict__ out,
        int n_tok) {
    const int ch = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    quantize_f32_q2_kvarn_k_block(tile + ch * n_tok, n_tok, &out[ch]);
}

// C-linkage host wrapper: allocates GPU buffers, runs the kernel, copies back.
// Bit-identical to quantize_row_q2_kvarn_k_ref (mirror invariant).
extern "C" void quantize_k_q2_kvarn_perchannel_tile_cuda(
        const float * tile, block_q2_kvarn_k * out, int n_ch, int n_tok) {
    float * d_tile = nullptr;
    block_q2_kvarn_k * d_out = nullptr;
    CUDA_CHECK(cudaMalloc(&d_tile, (size_t)n_ch * n_tok * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_out,  (size_t)n_ch * sizeof(block_q2_kvarn_k)));
    CUDA_CHECK(cudaMemcpy(d_tile, tile, (size_t)n_ch * n_tok * sizeof(float),
                          cudaMemcpyHostToDevice));
    quantize_k_q2_kvarn_perchannel_kernel<<<1, n_ch>>>(d_tile, d_out, n_tok);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(out, d_out, (size_t)n_ch * sizeof(block_q2_kvarn_k),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_tile));
    CUDA_CHECK(cudaFree(d_out));
}
