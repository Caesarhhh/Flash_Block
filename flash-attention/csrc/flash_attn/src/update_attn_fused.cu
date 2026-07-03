#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <stdint.h>
#include "namespace_config.h"

// =============================================================
//  exp helpers
// =============================================================
template<typename T>
__device__ __forceinline__ float to_float(T x) { return static_cast<float>(x); }

template<>
__device__ __forceinline__ float to_float(half x) { return __half2float(x); }

template<>
__device__ __forceinline__ float to_float(__nv_bfloat16 x) { return __bfloat162float(x); }

template<typename T>
__device__ __forceinline__ T from_float(float x) { return static_cast<T>(x); }

template<>
__device__ __forceinline__ half from_float(float x) { return __float2half(x); }

template<>
__device__ __forceinline__ __nv_bfloat16 from_float(float x) { return __float2bfloat16(x); }

// =============================================================
//  fused kernel
// =============================================================
namespace FLASH_NAMESPACE {

template<typename T>
__global__ void update_attn_fused_kernel(
    const float* __restrict__ softmax_lse,     // [B,Q,H,1], current output lse
    const float* __restrict__ self_lse,        // [B,Q,H,1], cached history lse
    const T* __restrict__ out_hist,            // [B,Q,H,D]
    const float* __restrict__ softmax_lse_hist,// [B,Q,H,1], history lse
    const T* __restrict__ attn_past,           // [B,Q,H,D]
    const T* __restrict__ o,                   // [B,Q,H,D]
    const bool* __restrict__ need_mask,        // [B,H,Q,1]
    const bool* __restrict__ store_mask,       // [B,H,Q,1]
    const bool* __restrict__ dirty_mask,       // [B,H,Q,1]
    T* __restrict__ out,
    T* __restrict__ attn_out_tensor,
    float* __restrict__ logsumexp_tensor,
    int B, int H, int Q, int D
){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = B * H * Q * D;
    if (idx >= total) return;

    // ---- Recover (b,q,h,d), matching flash_attn_with_kvcache output layout ----
    int d = idx % D;
    int tmp = idx / D;
    int h = tmp % H;
    tmp /= H;
    int q = tmp % Q;
    int b = tmp / Q;

    // ---- Offsets ----
    int off_BQH1 = ((b * Q + q) * H + h);          // last dim = 1
    int off_BQHD = off_BQH1 * D + d;               // last dim = D
    int off_BHQ = ((b * H + h) * Q + q);           // mask layout

    // broadcast masks
    bool nm = need_mask[off_BHQ];
    bool sm = store_mask == nullptr ? nm : store_mask[off_BHQ];

    // Step1: refresh cached history for samples that need full recomputation.
    T attn = (nm && sm) ? out_hist[off_BQHD] : attn_past[off_BQHD];
    float past_lse = (nm && sm) ? softmax_lse_hist[off_BQH1] : self_lse[off_BQH1];

    attn_out_tensor[off_BQHD]  = attn;
    logsumexp_tensor[off_BQH1] = past_lse;

    // Step2: stable exp merge
    float cur_lse = softmax_lse[off_BQH1];
    float m = fmaxf(past_lse, cur_lse);

    float exp_past = expf(past_lse - m);
    float exp_cur  = expf(cur_lse - m);
    float denom = exp_past + exp_cur;

    // Step4: veri
    float veri = (to_float(attn) * exp_past + to_float(o[off_BQHD]) * exp_cur) / denom;

    // Step5: final
    out[off_BQHD] = nm ? o[off_BQHD] : from_float<T>(veri);
}



// =============================================================
//  host wrapper (callable from C++)
// =============================================================
template<typename T>
void launch_update_attn_fused_kernel(
    const float* softmax_lse,
    const float* self_lse,
    const T* out_hist,
    const float* softmax_lse_hist,
    const T* attn_past,
    const T* o,
    const bool* need_mask,
    const bool* store_mask,
    const bool* dirty_mask,
    T* out,
    T* attn_out_tensor,
    float* logsumexp_tensor,
    int B, int H, int Q, int D
){
    int threads = 256;
    int total = B * H * Q * D;
    int blocks = (total + threads - 1) / threads;

    update_attn_fused_kernel<T><<<blocks, threads>>>(
        softmax_lse, self_lse,
        out_hist, softmax_lse_hist,
        attn_past, o,
        need_mask, store_mask, dirty_mask,
        out, attn_out_tensor, logsumexp_tensor,
        B, H, Q, D
    );
}

template<typename T>
__global__ void update_attn_fused_cur_hist_kernel(
    const float* __restrict__ softmax_lse_cur,
    const float* __restrict__ self_lse,
    const T* __restrict__ out_hist,
    const float* __restrict__ softmax_lse_hist,
    const T* __restrict__ attn_past,
    const T* __restrict__ out_cur,
    const bool* __restrict__ need_mask,
    const bool* __restrict__ store_mask,
    const bool* __restrict__ dirty_mask,
    T* __restrict__ out,
    T* __restrict__ attn_out_tensor,
    float* __restrict__ logsumexp_tensor,
    int B, int H, int Q, int D
){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = B * H * Q * D;
    if (idx >= total) return;

    int d = idx % D;
    int tmp = idx / D;
    int h = tmp % H;
    tmp /= H;
    int q = tmp % Q;
    int b = tmp / Q;

    int off_BQH1 = ((b * Q + q) * H + h);
    int off_BQHD = off_BQH1 * D + d;
    int off_BHQ = ((b * H + h) * Q + q);

    bool nm = need_mask[off_BHQ];
    bool sm = store_mask == nullptr ? nm : store_mask[off_BHQ];

    T hist_out = sm ? out_hist[off_BQHD] : attn_past[off_BQHD];
    float hist_lse = sm ? softmax_lse_hist[off_BQH1] : self_lse[off_BQH1];

    attn_out_tensor[off_BQHD] = hist_out;
    logsumexp_tensor[off_BQH1] = hist_lse;

    if (sm) {
        out[off_BQHD] = out_cur[off_BQHD];
        return;
    }

    if (!sm && nm) {
        out[off_BQHD] = out_cur[off_BQHD];
        return;
    }

    float cur_lse = softmax_lse_cur[off_BQH1];
    float m = fmaxf(hist_lse, cur_lse);
    float exp_hist = expf(hist_lse - m);
    float exp_cur = expf(cur_lse - m);
    float denom = exp_hist + exp_cur;
    float merged = (to_float(hist_out) * exp_hist + to_float(out_cur[off_BQHD]) * exp_cur) / denom;

    out[off_BQHD] = from_float<T>(merged);
}

template<typename T>
void launch_update_attn_fused_cur_hist_kernel(
    const float* softmax_lse_cur,
    const float* self_lse,
    const T* out_hist,
    const float* softmax_lse_hist,
    const T* attn_past,
    const T* out_cur,
    const bool* need_mask,
    const bool* store_mask,
    const bool* dirty_mask,
    T* out,
    T* attn_out_tensor,
    float* logsumexp_tensor,
    int B, int H, int Q, int D
){
    int threads = 256;
    int total = B * H * Q * D;
    int blocks = (total + threads - 1) / threads;

    update_attn_fused_cur_hist_kernel<T><<<blocks, threads>>>(
        softmax_lse_cur, self_lse,
        out_hist, softmax_lse_hist,
        attn_past, out_cur,
        need_mask, store_mask, dirty_mask,
        out, attn_out_tensor, logsumexp_tensor,
        B, H, Q, D
    );
}

template<typename T>
__global__ void update_attn_fused_kernel_fp32_cache(
    const float* __restrict__ softmax_lse,
    const float* __restrict__ self_lse,
    const float* __restrict__ out_hist,
    const float* __restrict__ softmax_lse_hist,
    const float* __restrict__ attn_past,
    const T* __restrict__ o,
    const bool* __restrict__ need_mask,
	    const bool* __restrict__ store_mask,
	    const bool* __restrict__ dirty_mask,
	    T* __restrict__ out,
	    float* __restrict__ attn_out_tensor,
	    float* __restrict__ logsumexp_tensor,
    int B, int H, int Q, int D
){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = B * H * Q * D;
    if (idx >= total) return;

    int d = idx % D;
    int tmp = idx / D;
    int h = tmp % H;
    tmp /= H;
    int q = tmp % Q;
    int b = tmp / Q;

    int off_BQH1 = ((b * Q + q) * H + h);
    int off_BQHD = off_BQH1 * D + d;
    int off_BHQ = ((b * H + h) * Q + q);

    bool nm = need_mask[off_BHQ];
    bool sm = store_mask == nullptr ? nm : store_mask[off_BHQ];

    float attn = (nm && sm) ? out_hist[off_BQHD] : attn_past[off_BQHD];
    float past_lse = (nm && sm) ? softmax_lse_hist[off_BQH1] : self_lse[off_BQH1];

    attn_out_tensor[off_BQHD] = attn;
    logsumexp_tensor[off_BQH1] = past_lse;

    float cur_lse = softmax_lse[off_BQH1];
    float m = fmaxf(past_lse, cur_lse);
    float exp_past = expf(past_lse - m);
    float exp_cur = expf(cur_lse - m);
    float denom = exp_past + exp_cur;
    float veri = (attn * exp_past + to_float(o[off_BQHD]) * exp_cur) / denom;

    out[off_BQHD] = nm ? o[off_BQHD] : from_float<T>(veri);
}

template<typename T>
void launch_update_attn_fused_kernel_fp32_cache(
    const float* softmax_lse,
    const float* self_lse,
    const float* out_hist,
    const float* softmax_lse_hist,
    const float* attn_past,
    const T* o,
    const bool* need_mask,
    const bool* store_mask,
    const bool* dirty_mask,
    T* out,
    float* attn_out_tensor,
    float* logsumexp_tensor,
    int B, int H, int Q, int D
){
    int threads = 256;
    int total = B * H * Q * D;
    int blocks = (total + threads - 1) / threads;

    update_attn_fused_kernel_fp32_cache<T><<<blocks, threads>>>(
        softmax_lse, self_lse,
        out_hist, softmax_lse_hist,
        attn_past, o,
        need_mask, store_mask, dirty_mask,
        out, attn_out_tensor, logsumexp_tensor,
        B, H, Q, D
    );
}

template<typename T>
__global__ void update_attn_fused_cur_hist_kernel_fp32_cache(
    const float* __restrict__ softmax_lse_cur,
    const float* __restrict__ self_lse,
    const float* __restrict__ out_hist,
    const float* __restrict__ softmax_lse_hist,
    const float* __restrict__ attn_past,
    const float* __restrict__ out_cur,
    const T* __restrict__ out_full_or_cur,
    const bool* __restrict__ need_mask,
    const bool* __restrict__ store_mask,
	    const bool* __restrict__ dirty_mask,
	    T* __restrict__ out,
	    float* __restrict__ full_lse_out,
	    float* __restrict__ attn_out_tensor,
	    float* __restrict__ logsumexp_tensor,
    int B, int H, int Q, int D
){
    int row_idx = blockIdx.x;
    int total_rows = B * Q * H;
    if (row_idx >= total_rows) return;

    int h = row_idx % H;
    int tmp = row_idx / H;
    int q = tmp % Q;
    int b = tmp / Q;

    int off_BQH1 = row_idx;
    int off_BHQ = ((b * H + h) * Q + q);

    bool nm = need_mask == nullptr ? true : need_mask[off_BHQ];
    bool sm = store_mask == nullptr ? nm : store_mask[off_BHQ];
    const bool use_new_hist = sm;

    int base = off_BQH1 * D;
    if (!sm && nm) {
        if (threadIdx.x == 0) {
            full_lse_out[off_BQH1] = softmax_lse_cur[off_BQH1];
        }
        for (int d = threadIdx.x; d < D; d += blockDim.x) {
            int off_BQHD = base + d;
            out[off_BQHD] = out_full_or_cur[off_BQHD];
        }
        return;
    }

    __shared__ float s_exp_hist;
    __shared__ float s_exp_cur;
    __shared__ float s_inv_denom;

    if (threadIdx.x == 0) {
        float hist_lse = use_new_hist ? softmax_lse_hist[off_BQH1] : self_lse[off_BQH1];
        float cur_lse = softmax_lse_cur[off_BQH1];
        float m = fmaxf(hist_lse, cur_lse);
        float exp_hist = expf(hist_lse - m);
        float exp_cur = expf(cur_lse - m);
        float denom = exp_hist + exp_cur;
        float inv_denom = 1.f / denom;
        s_exp_hist = exp_hist;
        s_exp_cur = exp_cur;
        s_inv_denom = inv_denom;
        full_lse_out[off_BQH1] = sm
            ? ((denom == 0.f || denom != denom) ? INFINITY : m + logf(denom))
            : softmax_lse_cur[off_BQH1];
        if (sm) {
            logsumexp_tensor[off_BQH1] = softmax_lse_hist[off_BQH1];
        }
    }
    __syncthreads();

    for (int d = threadIdx.x; d < D; d += blockDim.x) {
        int off_BQHD = base + d;
        float hist_out = use_new_hist ? out_hist[off_BQHD] : attn_past[off_BQHD];
        float cur_out = sm ? out_cur[off_BQHD] : to_float(out_full_or_cur[off_BQHD]);
        float merged = (hist_out * s_exp_hist + cur_out * s_exp_cur) * s_inv_denom;
        out[off_BQHD] = from_float<T>(merged);
        if (sm) {
            attn_out_tensor[off_BQHD] = out_hist[off_BQHD];
        }
    }
}

template<typename T>
void launch_update_attn_fused_cur_hist_kernel_fp32_cache(
    const float* softmax_lse_cur,
    const float* self_lse,
    const float* out_hist,
    const float* softmax_lse_hist,
    const float* attn_past,
    const float* out_cur,
    const T* out_full_or_cur,
    const bool* need_mask,
	    const bool* store_mask,
	    const bool* dirty_mask,
	    T* out,
	    float* full_lse_out,
	    float* attn_out_tensor,
	    float* logsumexp_tensor,
    int B, int H, int Q, int D
){
    int threads = 128;
    int blocks = B * H * Q;

    update_attn_fused_cur_hist_kernel_fp32_cache<T><<<blocks, threads>>>(
        softmax_lse_cur, self_lse,
        out_hist, softmax_lse_hist,
        attn_past, out_cur, out_full_or_cur,
        need_mask, store_mask, dirty_mask,
	        out, full_lse_out, attn_out_tensor, logsumexp_tensor,
	        B, H, Q, D
		    );
}

template<typename T>
__global__ void all_update_store_history_fp32_kernel(
    float* __restrict__ softmax_lse_full,       // [B,Q,H,1], cur on entry; full on exit for store rows
    const float* __restrict__ out_hist,          // [B,Q,H,D]
    const float* __restrict__ softmax_lse_hist,  // [B,Q,H,1]
    const float* __restrict__ out_cur,           // [B,Q,H,D]
    const bool* __restrict__ store_mask,         // [B,H,Q,1]
    T* __restrict__ out,                         // [B,Q,H,D]
    float* __restrict__ attn_out_tensor,         // [B,Q,H,D]
    float* __restrict__ logsumexp_tensor,        // [B,Q,H,1]
    int B, int H, int Q, int D
){
    int row_idx = blockIdx.x;
    int total_rows = B * Q * H;
    if (row_idx >= total_rows) return;

    int h = row_idx % H;
    int tmp = row_idx / H;
    int q = tmp % Q;
    int b = tmp / Q;
    int off_BHQ = ((b * H + h) * Q + q);

    if (!store_mask[off_BHQ]) {
        return;
    }

    int base = row_idx * D;

    __shared__ float s_exp_hist;
    __shared__ float s_exp_cur;
    __shared__ float s_inv_denom;

    if (threadIdx.x == 0) {
        float hist_lse = softmax_lse_hist[row_idx];
        float cur_lse = softmax_lse_full[row_idx];
        float m = fmaxf(hist_lse, cur_lse);
        float exp_hist = expf(hist_lse - m);
        float exp_cur = expf(cur_lse - m);
        float denom = exp_hist + exp_cur;
        s_exp_hist = exp_hist;
        s_exp_cur = exp_cur;
        s_inv_denom = 1.f / denom;
        softmax_lse_full[row_idx] = (denom == 0.f || denom != denom) ? INFINITY : m + logf(denom);
        logsumexp_tensor[row_idx] = hist_lse;
    }
    __syncthreads();

    for (int d = threadIdx.x; d < D; d += blockDim.x) {
        int off = base + d;
        float hist_out = out_hist[off];
        float cur_out = out_cur[off];
        float merged = (hist_out * s_exp_hist + cur_out * s_exp_cur) * s_inv_denom;
        out[off] = from_float<T>(merged);
        attn_out_tensor[off] = hist_out;
    }
}

template<typename T>
void launch_all_update_store_history_fp32_kernel(
    float* softmax_lse_full,
    const float* out_hist,
    const float* softmax_lse_hist,
    const float* out_cur,
    const bool* store_mask,
    T* out,
    float* attn_out_tensor,
    float* logsumexp_tensor,
    int B, int H, int Q, int D
){
    int threads = 128;
    int blocks = B * H * Q;

    all_update_store_history_fp32_kernel<T><<<blocks, threads>>>(
        softmax_lse_full,
        out_hist,
        softmax_lse_hist,
        out_cur,
        store_mask,
        out,
        attn_out_tensor,
        logsumexp_tensor,
        B, H, Q, D
    );
}

template <typename T, int MaxBlock>
__global__ void current_block_fp32_kernel(
    const T* __restrict__ q,
    const T* __restrict__ k_cache,
    const T* __restrict__ v_cache,
    const int* __restrict__ cache_seqlens,
    const int* __restrict__ cache_batch_idx,
    const int* __restrict__ block_table,
    const bool* __restrict__ store_mask,
    float* __restrict__ out_cur,
    float* __restrict__ softmax_lse,
    int B,
    int Q,
    int H,
    int Hk,
    int D,
    int block_size,
    int seqlen_knew,
    int page_block_size,
    int max_blocks_per_seq,
    int64_t q_batch_stride,
    int64_t q_row_stride,
    int64_t q_head_stride,
    int64_t k_batch_stride,
    int64_t k_row_stride,
    int64_t k_head_stride,
    int64_t v_batch_stride,
    int64_t v_row_stride,
    int64_t v_head_stride,
    int64_t block_table_batch_stride,
    float softmax_scale,
    bool is_paged,
    bool is_causal
) {
    constexpr int Threads = 128;
    __shared__ float partial[MaxBlock][Threads];
    __shared__ float scores[MaxBlock];
    __shared__ float probs[MaxBlock];

    const int row_idx = blockIdx.x;
    const int tid = threadIdx.x;
    const int total_rows = B * Q * H;
    if (row_idx >= total_rows) return;

    const int h = row_idx % H;
    int tmp = row_idx / H;
    const int q_idx = tmp % Q;
    const int b = tmp / Q;
    const int mask_offset = (b * H + h) * Q + q_idx;
    if (store_mask != nullptr && !store_mask[mask_offset]) return;

    const int kv_h = h / (H / Hk);
    const int cache_b = cache_batch_idx == nullptr ? b : cache_batch_idx[b];
    const int old_len = cache_seqlens == nullptr ? 0 : cache_seqlens[b];
    const int actual_len = old_len + seqlen_knew;
    const int cur_start = actual_len - block_size;
    const int query_abs = actual_len - Q + q_idx;

    float q_val = 0.f;
    if (tid < D) {
        const int64_t q_off = int64_t(b) * q_batch_stride + int64_t(q_idx) * q_row_stride
            + int64_t(h) * q_head_stride + tid;
        q_val = to_float(q[q_off]);
    }

    #pragma unroll
    for (int j = 0; j < MaxBlock; ++j) {
        float val = 0.f;
        if (j < block_size && tid < D) {
            const int key_abs = cur_start + j;
            bool keep = key_abs >= 0;
            if (is_causal) {
                keep = keep && key_abs <= query_abs;
            }
            if (keep) {
                int64_t k_off;
                if (is_paged) {
                    const int table_idx = key_abs / page_block_size;
                    const int table_off = key_abs - table_idx * page_block_size;
                    const int physical_block = block_table[int64_t(b) * block_table_batch_stride + table_idx];
                    k_off = int64_t(physical_block) * k_batch_stride + int64_t(table_off) * k_row_stride
                        + int64_t(kv_h) * k_head_stride + tid;
                } else {
                    k_off = int64_t(cache_b) * k_batch_stride + int64_t(key_abs) * k_row_stride
                        + int64_t(kv_h) * k_head_stride + tid;
                }
                val = q_val * to_float(k_cache[k_off]);
            }
        }
        partial[j][tid] = val;
    }
    __syncthreads();

    if (tid < MaxBlock) {
        float sum = 0.f;
        if (tid < block_size) {
            #pragma unroll
            for (int d = 0; d < Threads; ++d) {
                sum += partial[tid][d];
            }
            const int key_abs = cur_start + tid;
            bool keep = key_abs >= 0;
            if (is_causal) {
                keep = keep && key_abs <= query_abs;
            }
            scores[tid] = keep ? sum * softmax_scale : -INFINITY;
        } else {
            scores[tid] = -INFINITY;
        }
    }
    __syncthreads();

    if (tid == 0) {
        float m = scores[0];
        #pragma unroll
        for (int j = 1; j < MaxBlock; ++j) {
            m = fmaxf(m, scores[j]);
        }
        float denom = 0.f;
        #pragma unroll
        for (int j = 0; j < MaxBlock; ++j) {
            const float p = j < block_size ? expf(scores[j] - m) : 0.f;
            probs[j] = p;
            denom += p;
        }
        const int lse_off = (b * H + h) * Q + q_idx;
        softmax_lse[lse_off] = (denom == 0.f || denom != denom) ? INFINITY : m + logf(denom);
        const float inv = denom == 0.f ? 0.f : 1.f / denom;
        #pragma unroll
        for (int j = 0; j < MaxBlock; ++j) {
            probs[j] *= inv;
        }
    }
    __syncthreads();

    if (tid < D) {
        float acc = 0.f;
        #pragma unroll
        for (int j = 0; j < MaxBlock; ++j) {
            if (j < block_size) {
                const int key_abs = cur_start + j;
                bool keep = key_abs >= 0;
                if (is_causal) {
                    keep = keep && key_abs <= query_abs;
                }
                if (keep) {
                    int64_t v_off;
                    if (is_paged) {
                        const int table_idx = key_abs / page_block_size;
                        const int table_off = key_abs - table_idx * page_block_size;
                        const int physical_block = block_table[int64_t(b) * block_table_batch_stride + table_idx];
                        v_off = int64_t(physical_block) * v_batch_stride + int64_t(table_off) * v_row_stride
                            + int64_t(kv_h) * v_head_stride + tid;
                    } else {
                        v_off = int64_t(cache_b) * v_batch_stride + int64_t(key_abs) * v_row_stride
                            + int64_t(kv_h) * v_head_stride + tid;
                    }
                    acc += probs[j] * to_float(v_cache[v_off]);
                }
            }
        }
        const int out_off = ((b * Q + q_idx) * H + h) * D + tid;
        out_cur[out_off] = acc;
    }
}

template<typename T>
void launch_current_block_fp32_kernel(
    const T* q,
    const T* k_cache,
    const T* v_cache,
    const int* cache_seqlens,
    const int* cache_batch_idx,
    const int* block_table,
    const bool* store_mask,
    float* out_cur,
    float* softmax_lse,
    int B,
    int Q,
    int H,
    int Hk,
    int D,
    int block_size,
    int seqlen_knew,
    int page_block_size,
    int max_blocks_per_seq,
    int64_t q_batch_stride,
    int64_t q_row_stride,
    int64_t q_head_stride,
    int64_t k_batch_stride,
    int64_t k_row_stride,
    int64_t k_head_stride,
    int64_t v_batch_stride,
    int64_t v_row_stride,
    int64_t v_head_stride,
    int64_t block_table_batch_stride,
    float softmax_scale,
    bool is_paged,
    bool is_causal
) {
    int threads = 128;
    int blocks = B * Q * H;
    if (block_size <= 4) {
        current_block_fp32_kernel<T, 4><<<blocks, threads>>>(
            q, k_cache, v_cache, cache_seqlens, cache_batch_idx, block_table, store_mask,
            out_cur, softmax_lse, B, Q, H, Hk, D, block_size, seqlen_knew,
            page_block_size, max_blocks_per_seq, q_batch_stride, q_row_stride, q_head_stride,
            k_batch_stride, k_row_stride, k_head_stride, v_batch_stride, v_row_stride,
            v_head_stride, block_table_batch_stride, softmax_scale, is_paged, is_causal);
    } else {
        current_block_fp32_kernel<T, 8><<<blocks, threads>>>(
            q, k_cache, v_cache, cache_seqlens, cache_batch_idx, block_table, store_mask,
            out_cur, softmax_lse, B, Q, H, Hk, D, block_size, seqlen_knew,
            page_block_size, max_blocks_per_seq, q_batch_stride, q_row_stride, q_head_stride,
            k_batch_stride, k_row_stride, k_head_stride, v_batch_stride, v_row_stride,
            v_head_stride, block_table_batch_stride, softmax_scale, is_paged, is_causal);
    }
}

template<typename T>
__global__ void update_attn_fused_varlen_kernel(
    const float* __restrict__ softmax_lse,       // [H, T]
    const float* __restrict__ self_lse,          // [H, T]
    const T* __restrict__ out_hist,              // [T, H, D]
    const float* __restrict__ softmax_lse_hist,  // [H, T]
    const T* __restrict__ attn_past,             // dense [B, Q, H, D], linear-compatible with [T, H, D]
    const T* __restrict__ o,                     // [T, H, D]
    const bool* __restrict__ head_mask,          // [H], null means all false
    T* __restrict__ out,                         // [T, H, D]
    T* __restrict__ attn_out_tensor,             // dense [B, Q, H, D]
    float* __restrict__ logsumexp_tensor,        // [H, T]
    int Tq, int H, int D
){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = Tq * H * D;
    if (idx >= total) return;

    int d = idx % D;
    int tmp = idx / D;
    int h = tmp % H;
    int t = tmp / H;

    int off_THD = (t * H + h) * D + d;
    int off_HT = h * Tq + t;

    bool nm = head_mask != nullptr && head_mask[h];
    if (nm) {
        attn_out_tensor[off_THD] = out_hist[off_THD];
        if (d == 0) {
            logsumexp_tensor[off_HT] = softmax_lse_hist[off_HT];
        }
        out[off_THD] = o[off_THD];
        return;
    }

    T attn = attn_past[off_THD];
    float past_lse = self_lse[off_HT];
    attn_out_tensor[off_THD] = attn;
    if (d == 0) {
        logsumexp_tensor[off_HT] = past_lse;
    }

    float cur_lse = softmax_lse[off_HT];
    float m = fmaxf(past_lse, cur_lse);
    float exp_past = expf(past_lse - m);
    float exp_cur = expf(cur_lse - m);
    float denom = exp_past + exp_cur;
    float merged = (to_float(attn) * exp_past + to_float(o[off_THD]) * exp_cur) / denom;

    out[off_THD] = from_float<T>(merged);
}

template<typename T>
void launch_update_attn_fused_varlen_kernel(
    const float* softmax_lse,
    const float* self_lse,
    const T* out_hist,
    const float* softmax_lse_hist,
    const T* attn_past,
    const T* o,
    const bool* head_mask,
    T* out,
    T* attn_out_tensor,
    float* logsumexp_tensor,
    int Tq, int H, int D
){
    int threads = 256;
    int total = Tq * H * D;
    int blocks = (total + threads - 1) / threads;

    update_attn_fused_varlen_kernel<T><<<blocks, threads>>>(
        softmax_lse, self_lse,
        out_hist, softmax_lse_hist,
        attn_past, o,
        head_mask,
        out, attn_out_tensor, logsumexp_tensor,
        Tq, H, D
    );
}


template
void launch_update_attn_fused_kernel<float>(
    const float*,        // softmax_lse
    const float*,        // self_lse
    const float*,        // out_hist
    const float*,        // softmax_lse_hist
    const float*,        // attn_past
    const float*,        // o
    const bool*,         // need_mask
    const bool*,         // store_mask
    const bool*,         // dirty_mask
    float*,              // out
    float*,              // attn_out_tensor
    float*,              // logsumexp_tensor
    int, int, int, int   // B, H, Q, D
);

template
void launch_update_attn_fused_kernel_fp32_cache<float>(
    const float*, const float*, const float*, const float*,
    const float*, const float*, const bool*, const bool*, const bool*,
    float*, float*, float*, int, int, int, int
);

template
void launch_update_attn_fused_cur_hist_kernel<float>(
    const float*,
    const float*,
    const float*,
    const float*,
    const float*,
    const float*,
    const bool*,
    const bool*,
    const bool*,
    float*,
    float*,
    float*,
    int, int, int, int
);

template
void launch_update_attn_fused_cur_hist_kernel_fp32_cache<float>(
    const float*, const float*, const float*, const float*,
    const float*, const float*, const float*, const bool*, const bool*, const bool*,
    float*, float*, float*, float*, int, int, int, int
);

template
void launch_all_update_store_history_fp32_kernel<float>(
    float*, const float*, const float*, const float*, const bool*,
    float*, float*, float*, int, int, int, int
);

template
void launch_current_block_fp32_kernel<float>(
    const float*, const float*, const float*, const int*, const int*, const int*,
    const bool*, float*, float*, int, int, int, int, int, int, int, int, int,
    int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t,
    int64_t, int64_t, float, bool, bool
);

template
void launch_update_attn_fused_kernel<half>(
		    const float*,
		    const float*,
    const half*,
    const float*,
    const half*,
		    const half*,
	    const bool*,
	    const bool*,
	    const bool*,
		    half*,
		    half*,
		    float*,
	    int, int, int, int
		);

template
void launch_update_attn_fused_kernel_fp32_cache<half>(
    const float*,
    const float*,
    const float*,
    const float*,
    const float*,
    const half*,
	    const bool*,
	    const bool*,
	    const bool*,
	    half*,
	    float*,
	    float*,
	    int, int, int, int
	);

template
void launch_update_attn_fused_cur_hist_kernel<half>(
    const float*,
    const float*,
    const half*,
    const float*,
    const half*,
    const half*,
    const bool*,
    const bool*,
    const bool*,
    half*,
    half*,
    float*,
    int, int, int, int
);

template
void launch_update_attn_fused_cur_hist_kernel_fp32_cache<half>(
    const float*,
    const float*,
    const float*,
    const float*,
	    const float*,
	    const float*,
	    const half*,
	    const bool*,
	    const bool*,
	    const bool*,
	    half*,
	    float*,
	    float*,
	    float*,
	    int, int, int, int
	);

template
void launch_all_update_store_history_fp32_kernel<half>(
    float*, const float*, const float*, const float*, const bool*,
    half*, float*, float*, int, int, int, int
);

template
void launch_current_block_fp32_kernel<half>(
    const half*, const half*, const half*, const int*, const int*, const int*,
    const bool*, float*, float*, int, int, int, int, int, int, int, int, int,
    int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t,
    int64_t, int64_t, float, bool, bool
);

template
void launch_update_attn_fused_kernel<__nv_bfloat16>(
		    const float*,
		    const float*,
    const __nv_bfloat16*,
    const float*,
    const __nv_bfloat16*,
	    const __nv_bfloat16*,
	    const bool*,
	    const bool*,
	    const bool*,
		    __nv_bfloat16*,
		    __nv_bfloat16*,
		    float*,
	    int, int, int, int
			);

template
void launch_update_attn_fused_kernel_fp32_cache<__nv_bfloat16>(
    const float*,
    const float*,
    const float*,
    const float*,
    const float*,
    const __nv_bfloat16*,
	    const bool*,
	    const bool*,
	    const bool*,
	    __nv_bfloat16*,
	    float*,
	    float*,
	    int, int, int, int
	);

template
void launch_update_attn_fused_cur_hist_kernel<__nv_bfloat16>(
    const float*,
    const float*,
    const __nv_bfloat16*,
    const float*,
    const __nv_bfloat16*,
    const __nv_bfloat16*,
    const bool*,
    const bool*,
    const bool*,
    __nv_bfloat16*,
    __nv_bfloat16*,
    float*,
    int, int, int, int
);

template
void launch_update_attn_fused_cur_hist_kernel_fp32_cache<__nv_bfloat16>(
    const float*,
    const float*,
    const float*,
    const float*,
	    const float*,
	    const float*,
	    const __nv_bfloat16*,
	    const bool*,
	    const bool*,
	    const bool*,
	    __nv_bfloat16*,
	    float*,
	    float*,
	    float*,
	    int, int, int, int
	);

template
void launch_all_update_store_history_fp32_kernel<__nv_bfloat16>(
    float*, const float*, const float*, const float*, const bool*,
    __nv_bfloat16*, float*, float*, int, int, int, int
);

template
void launch_current_block_fp32_kernel<__nv_bfloat16>(
    const __nv_bfloat16*, const __nv_bfloat16*, const __nv_bfloat16*,
    const int*, const int*, const int*, const bool*, float*, float*,
    int, int, int, int, int, int, int, int, int,
    int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t,
    int64_t, int64_t, float, bool, bool
);

template
void launch_update_attn_fused_varlen_kernel<float>(
    const float*, const float*, const float*, const float*,
    const float*, const float*, const bool*,
    float*, float*, float*, int, int, int
);

template
void launch_update_attn_fused_varlen_kernel<half>(
    const float*, const float*, const half*, const float*,
    const half*, const half*, const bool*,
    half*, half*, float*, int, int, int
);

template
void launch_update_attn_fused_varlen_kernel<__nv_bfloat16>(
    const float*, const float*, const __nv_bfloat16*, const float*,
    const __nv_bfloat16*, const __nv_bfloat16*, const bool*,
    __nv_bfloat16*, __nv_bfloat16*, float*, int, int, int
);


}
