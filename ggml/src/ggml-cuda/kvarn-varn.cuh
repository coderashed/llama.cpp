#include "common.cuh"

// KVARN VarN (item 04) graph op forward. dst->src[0] is F32 [n_tok, head_dim, n_head]
// (contiguous); dst is the packed 1D F32 output [T_norm ++ S_r ++ S_c] on the same
// device. Launches the faithful VarN tile kernel on ctx's stream (one block per head).
void ggml_cuda_op_kvarn_varn(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
