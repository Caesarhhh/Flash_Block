/******************************************************************************
 * Copyright (c) 2024, Tri Dao.
 ******************************************************************************/

// Include these 2 headers instead of torch/extension.h since we don't need all of the torch headers.
#include <torch/python.h>
#include <torch/nn/functional.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <ATen/cuda/CUDAGeneratorImpl.h>  // For at::Generator and at::PhiloxCudaState
#include "philox_unpack.cuh"  // For at::cuda::philox::unpack

#include <cutlass/numeric_types.h>

#include "namespace_config.h"
#include "hardware_info.h"
#include "flash.h"
#include "static_switch.h"

#define CHECK_DEVICE(x) TORCH_CHECK(x.is_cuda(), #x " must be on CUDA")
#define CHECK_SHAPE(x, ...) TORCH_CHECK(x.sizes() == torch::IntArrayRef({__VA_ARGS__}), #x " must have shape (" #__VA_ARGS__ ")")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")

namespace FLASH_NAMESPACE {

void set_params_fprop(Flash_fwd_params &params,
                      // sizes
                      const size_t b,
                      const size_t seqlen_q,
                      const size_t seqlen_k,
                      const size_t seqlen_q_rounded,
                      const size_t seqlen_k_rounded,
                      const size_t h,
                      const size_t h_k,
                      const size_t d,
                      const size_t d_rounded,
                      // device pointers
                      const at::Tensor q,
                      const at::Tensor k,
                      const at::Tensor v,
                      at::Tensor out,
                      void *cu_seqlens_q_d,
                      void *cu_seqlens_k_d,
                      void *seqused_k,
                      void *p_d,
                      void *softmax_lse_d,
                      float p_dropout,
                      float softmax_scale,
                      int window_size_left,
                      int window_size_right,
                      const float softcap,
                      bool seqlenq_ngroups_swapped=false,
                      const bool unpadded_lse=false) {

    // Reset the parameters
    params = {};

    params.is_bf16 = q.dtype() == torch::kBFloat16;

    // Set the pointers and strides.
    params.q_ptr = q.data_ptr();
    params.k_ptr = k.data_ptr();
    params.v_ptr = v.data_ptr();
    // All stride are in elements, not bytes.
    params.q_row_stride = q.stride(-3);
    params.k_row_stride = k.stride(-3);
    params.v_row_stride = v.stride(-3);
    params.q_head_stride = q.stride(-2);
    params.k_head_stride = k.stride(-2);
    params.v_head_stride = v.stride(-2);
    params.o_ptr = out.data_ptr();
    params.o_row_stride = out.stride(-3);
    params.o_head_stride = out.stride(-2);

    if (cu_seqlens_q_d == nullptr) {
        params.q_batch_stride = q.stride(0);
        params.k_batch_stride = k.stride(0);
        params.v_batch_stride = v.stride(0);
        params.o_batch_stride = out.stride(0);
        if (seqlenq_ngroups_swapped) {
             params.q_batch_stride *= seqlen_q;
             params.o_batch_stride *= seqlen_q;
        }
    }

    params.cu_seqlens_q = static_cast<int *>(cu_seqlens_q_d);
    params.cu_seqlens_k = static_cast<int *>(cu_seqlens_k_d);
    params.seqused_k = static_cast<int *>(seqused_k);

    // P = softmax(QK^T)
    params.p_ptr = p_d;

    // Softmax sum
    params.softmax_lse_ptr = softmax_lse_d;

    // Set the dimensions.
    params.b = b;
    params.h = h;
    params.h_k = h_k;
    params.h_h_k_ratio = h / h_k;
    params.seqlen_q = seqlen_q;
    params.seqlen_k = seqlen_k;
    params.seqlen_q_rounded = seqlen_q_rounded;
    params.seqlen_k_rounded = seqlen_k_rounded;
    params.d = d;
    params.d_rounded = d_rounded;

    // Set the different scale values.
    #ifdef FLASHATTENTION_DISABLE_SOFTCAP
        TORCH_CHECK(softcap <= 0.0, "This flash attention build does not support softcap.");
    #endif
    if (softcap > 0.0) {
        params.softcap = softmax_scale / softcap;
        params.scale_softmax = softcap;
        params.scale_softmax_log2 = softcap * M_LOG2E;
    } else{
        // Remove potential NaN
        params.softcap = 0.0;
        params.scale_softmax = softmax_scale;
        params.scale_softmax_log2 = softmax_scale * M_LOG2E;
    }

    // Set this to probability of keeping an element to simplify things.
    params.p_dropout = 1.f - p_dropout;
    // Convert p from float to int so we don't have to convert the random uint to float to compare.
    // [Minor] We want to round down since when we do the comparison we use <= instead of <
    // params.p_dropout_in_uint = uint32_t(std::floor(params.p_dropout * 4294967295.0));
    // params.p_dropout_in_uint16_t = uint16_t(std::floor(params.p_dropout * 65535.0));
    params.p_dropout_in_uint8_t = uint8_t(std::floor(params.p_dropout * 255.0));
    params.rp_dropout = 1.f / params.p_dropout;
    params.scale_softmax_rp_dropout = params.rp_dropout * params.scale_softmax;
    TORCH_CHECK(p_dropout < 1.f);
    #ifdef FLASHATTENTION_DISABLE_DROPOUT
        TORCH_CHECK(p_dropout == 0.0f, "This flash attention build does not support dropout.");
    #endif

    // Causal is the special case where window_size_right == 0 and window_size_left < 0.
    // Local is the more general case where window_size_right >= 0 or window_size_left >= 0.
    params.is_causal = window_size_left < 0 && window_size_right == 0;

    if (window_size_left < 0 && window_size_right >= 0) { window_size_left = seqlen_k; }
    if (window_size_left >= 0 && window_size_right < 0) { window_size_right = seqlen_k; }
    params.window_size_left = window_size_left;
    params.window_size_right = window_size_right;

    #ifdef FLASHATTENTION_DISABLE_LOCAL
        TORCH_CHECK(params.is_causal || (window_size_left < 0 && window_size_right < 0),
            "This flash attention build does not support local attention.");
    #endif

    params.is_seqlens_k_cumulative = true;

    #ifdef FLASHATTENTION_DISABLE_UNEVEN_K
        TORCH_CHECK(d == d_rounded, "This flash attention build does not support headdim not being a multiple of 32.");
    #endif

    params.unpadded_lse = unpadded_lse;
    params.seqlenq_ngroups_swapped = seqlenq_ngroups_swapped;
}

void set_params_dgrad(Flash_bwd_params &params,
                      // sizes
                      const size_t b,
                      const size_t seqlen_q,
                      const size_t seqlen_k,
                      const size_t seqlen_q_rounded,
                      const size_t seqlen_k_rounded,
                      const size_t h,
                      const size_t h_k,
                      const size_t d,
                      const size_t d_rounded,
                      // device pointers
                      const at::Tensor q,
                      const at::Tensor k,
                      const at::Tensor v,
                      const at::Tensor out,
                      const at::Tensor dout,
                      at::Tensor dq,
                      at::Tensor dk,
                      at::Tensor dv,
                      void *cu_seqlens_q_d,
                      void *cu_seqlens_k_d,
                      void *dq_accum_d,
                      void *dk_accum_d,
                      void *dv_accum_d,
                      void *softmax_lse_d,
                      void *dsoftmax_sum_d,
                      float p_dropout,
                      float softmax_scale,
                      int window_size_left,
                      int window_size_right,
                      const float softcap,
                      bool deterministic,
                      const bool unpadded_lse) {

    set_params_fprop(params,
                     b, seqlen_q, seqlen_k, seqlen_q_rounded, seqlen_k_rounded, h, h_k, d, d_rounded,
                     q, k, v, out,
                     cu_seqlens_q_d,
                     cu_seqlens_k_d,
                     nullptr,
                     nullptr,
                     softmax_lse_d,
                     p_dropout,
                     softmax_scale,
                     window_size_left,
                     window_size_right,
                     softcap,
                     false, // seqlenq_ngroups_swapped
                     unpadded_lse);

    // Set the pointers and strides.
    params.do_ptr = dout.data_ptr();
    params.do_row_stride = dout.stride(-3);
    params.do_head_stride = dout.stride(-2);
    params.dq_ptr = dq.data_ptr();
    params.dk_ptr = dk.data_ptr();
    params.dv_ptr = dv.data_ptr();
    params.dq_row_stride = dq.stride(-3);
    params.dk_row_stride = dk.stride(-3);
    params.dv_row_stride = dv.stride(-3);
    params.dq_head_stride = dq.stride(-2);
    params.dk_head_stride = dk.stride(-2);
    params.dv_head_stride = dv.stride(-2);

    if (cu_seqlens_q_d == nullptr) {
        params.do_batch_stride = dout.stride(0);
        params.dq_batch_stride = dq.stride(0);
        params.dk_batch_stride = dk.stride(0);
        params.dv_batch_stride = dv.stride(0);
    }

    params.dq_accum_ptr = dq_accum_d;
    params.dk_accum_ptr = dk_accum_d;
    params.dv_accum_ptr = dv_accum_d;

    // Softmax sum
    params.dsoftmax_sum = dsoftmax_sum_d;

    params.deterministic = deterministic;
}

// void run_mha_fwd(Flash_fwd_params &params, cudaStream_t stream, bool force_split_kernel=false) {
//     printf("********************run_mha_fwd******************\n");
//     FP16_SWITCH(!params.is_bf16, [&] {
//         HEADDIM_SWITCH(params.d, [&] {
//             BOOL_SWITCH(params.is_causal, Is_causal, [&] {
//                 if (params.num_splits <= 1 && !force_split_kernel) {  // If we don't set it num_splits == 0
//                     printf("********************run_mha_fwd_******************\n");
//                     run_mha_fwd_<elem_type, kHeadDim, Is_causal>(params, stream);
//                 } else {
//                     printf("********************run_mha_fwd_splitkv_dispatch******************\n");
//                     run_mha_fwd_splitkv_dispatch<elem_type, kHeadDim, Is_causal>(params, stream);
//                 }
//             });
//         });
//     });
// }

void run_mha_fwd(Flash_fwd_params &params, cudaStream_t stream, bool force_split_kernel = false) {
    using elem_type = cutlass::bfloat16_t;
    constexpr int kHeadDim = 128;
    constexpr bool Is_causal = false;  // 固定为非因果注意力

    if (params.num_splits <= 1 && !force_split_kernel) {  // If we don't set it num_splits == 0
                    run_mha_fwd_<elem_type, kHeadDim, Is_causal>(params, stream);
                } else {
                    run_mha_fwd_splitkv_dispatch<elem_type, kHeadDim, Is_causal>(params, stream);
                }
}



// Find the number of splits that maximizes the occupancy. For example, if we have
// batch * n_heads = 48 and we have 108 SMs, having 2 splits (efficiency = 0.89) is
// better than having 3 splits (efficiency = 0.67). However, we also don't want too many
// splits as that would incur more HBM reads/writes.
// So we find the best efficiency, then find the smallest number of splits that gets 85%
// of the best efficiency.
inline int num_splits_heuristic(int batch_nheads_mblocks, int num_SMs, int num_n_blocks, int max_splits) {
    // If we have enough to almost fill the SMs, then just use 1 split
    if (batch_nheads_mblocks >= 0.8f * num_SMs) { return 1; }
    max_splits = std::min({max_splits, num_SMs, num_n_blocks});
    float max_efficiency = 0.f;
    std::vector<float> efficiency;
    efficiency.reserve(max_splits);
    auto ceildiv = [](int a, int b) { return (a + b - 1) / b; };
    // Some splits are not eligible. For example, if we have 64 blocks and choose 11 splits,
    // we'll have 6 * 10 + 4 blocks. If we choose 12 splits, we'll have 6 * 11 + (-2) blocks
    // (i.e. it's 11 splits anyway).
    // So we check if the number of blocks per split is the same as the previous num_splits.
    auto is_split_eligible = [&ceildiv, &num_n_blocks](int num_splits) {
        return num_splits == 1 || ceildiv(num_n_blocks, num_splits) != ceildiv(num_n_blocks, num_splits - 1);
    };
    for (int num_splits = 1; num_splits <= max_splits; num_splits++) {
        if (!is_split_eligible(num_splits)) {
            efficiency.push_back(0.f);
        } else {
            float n_waves = float(batch_nheads_mblocks * num_splits) / num_SMs;
            float eff = n_waves / ceil(n_waves);
            // printf("num_splits = %d, eff = %f\n", num_splits, eff);
            if (eff > max_efficiency) { max_efficiency = eff; }
            efficiency.push_back(eff);
        }
    }
    for (int num_splits = 1; num_splits <= max_splits; num_splits++) {
        if (!is_split_eligible(num_splits)) { continue; }
        if (efficiency[num_splits - 1] >= 0.85 * max_efficiency) {
            // printf("num_splits chosen = %d\n", num_splits);
            return num_splits;
        }
    }
    return 1;
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> set_params_splitkv(Flash_fwd_params &params, const int batch_size,
    const int num_heads, const int head_size, const int max_seqlen_k, const int max_seqlen_q,
    const int head_size_rounded, const float p_dropout,
    const int num_splits, const int num_sm, struct c10::TensorOptions opts) {

    // This needs to match with run_mha_fwd_splitkv_dispatch
    const int block_n = head_size <= 64 ? 256 : (head_size <= 128 ? 128 : 64);
    const int num_n_blocks = (max_seqlen_k + block_n - 1) / block_n;
    // Technically kBlockM = 64 only for the splitKV kernels, not the standard kernel.
    // In any case we don't expect seqlen_q to be larger than 64 for inference.
    const int num_m_blocks = (max_seqlen_q + 64 - 1) / 64;
    params.num_splits = num_splits;
    at::Tensor softmax_lse_accum;
    at::Tensor softmax_lse_hist_accum;
    at::Tensor out_accum;
    at::Tensor out_accum_hist;

    if (p_dropout == 0.0f) {  // SplitKV is not implemented for dropout
        if (num_splits < 1) {
            // We multiply number of SMs by 2 to hard-code the fact that we're using 128 threads per block.
            params.num_splits = num_splits_heuristic(batch_size * num_heads * num_m_blocks, num_sm * 2, num_n_blocks, 128);
        }
        if (params.num_splits > 1) {
            softmax_lse_accum = torch::empty({params.num_splits, batch_size, num_heads, max_seqlen_q}, opts.dtype(at::kFloat));
            softmax_lse_hist_accum = torch::empty({params.num_splits, batch_size, num_heads, max_seqlen_q}, opts.dtype(at::kFloat));
            out_accum = torch::empty({params.num_splits, batch_size, num_heads, max_seqlen_q, head_size_rounded}, opts.dtype(at::kFloat));
            out_accum_hist = torch::empty({params.num_splits, batch_size, num_heads, max_seqlen_q, head_size_rounded}, opts.dtype(at::kFloat));
            params.softmax_lseaccum_ptr = softmax_lse_accum.data_ptr();
            params.softmax_lseaccum_hist_ptr = softmax_lse_hist_accum.data_ptr();
            params.oaccum_ptr = out_accum.data_ptr();
            params.oaccum_hist_ptr = out_accum_hist.data_ptr();
        }
        TORCH_CHECK(params.num_splits <= 128, "num_splits > 128 not supported");
    }

    return std::make_tuple(softmax_lse_accum, out_accum, softmax_lse_hist_accum);
}

void set_params_alibi(Flash_fwd_params &params, std::optional<at::Tensor> &alibi_slopes_, int batch_size, int num_heads){
#ifdef FLASHATTENTION_DISABLE_ALIBI
    TORCH_CHECK(!alibi_slopes_.has_value(), "This flash attention build does not support alibi.");
    params.alibi_slopes_ptr = nullptr;
#else
    if (alibi_slopes_.has_value()) {
        auto alibi_slopes = alibi_slopes_.value();
        TORCH_CHECK(alibi_slopes.dtype() == torch::kFloat32, "ALiBi slopes must have dtype fp32");
        CHECK_DEVICE(alibi_slopes);
        TORCH_CHECK(alibi_slopes.stride(-1) == 1, "ALiBi slopes tensor must have contiguous last dimension");
        TORCH_CHECK(alibi_slopes.sizes() == torch::IntArrayRef({num_heads}) || alibi_slopes.sizes() == torch::IntArrayRef({batch_size, num_heads}));
        params.alibi_slopes_ptr = alibi_slopes.data_ptr();
        params.alibi_slopes_batch_stride = alibi_slopes.dim() == 2 ? alibi_slopes.stride(0) : 0;
    } else {
        params.alibi_slopes_ptr = nullptr;
    }
#endif
}

std::vector<at::Tensor>
mha_fwd(at::Tensor &q,         // batch_size x seqlen_q x num_heads x round_multiple(head_size, 8)
        const at::Tensor &k,         // batch_size x seqlen_k x num_heads_k x round_multiple(head_size, 8)
        const at::Tensor &v,         // batch_size x seqlen_k x num_heads_k x round_multiple(head_size, 8)
        std::optional<at::Tensor> &out_,             // batch_size x seqlen_q x num_heads x round_multiple(head_size, 8)
        std::optional<at::Tensor> &alibi_slopes_, // num_heads or batch_size x num_heads
        const float p_dropout,
        const float softmax_scale,
        bool is_causal,
        int window_size_left,
        int window_size_right,
        const float softcap,
        const bool return_softmax,
        std::optional<at::Generator> gen_) {

    // Otherwise the kernel will be launched from cuda:0 device
    at::cuda::CUDAGuard device_guard{q.device()};

    auto [cc_major, cc_minor] = get_compute_capability(get_current_device());
    bool is_sm8x_min = cc_major >= 8;
    TORCH_CHECK(is_sm8x_min, "FlashAttention only supports Ampere GPUs or newer.");

    auto q_dtype = q.dtype();
    TORCH_CHECK(q_dtype == torch::kFloat16 || q_dtype == torch::kBFloat16,
                "FlashAttention only support fp16 and bf16 data type");
    TORCH_CHECK(k.dtype() == q_dtype, "query and key must have the same dtype");
    TORCH_CHECK(v.dtype() == q_dtype, "query and value must have the same dtype");

    CHECK_DEVICE(q); CHECK_DEVICE(k); CHECK_DEVICE(v);

    TORCH_CHECK(q.stride(-1) == 1, "Input tensor must have contiguous last dimension");
    TORCH_CHECK(k.stride(-1) == 1, "Input tensor must have contiguous last dimension");
    TORCH_CHECK(v.stride(-1) == 1, "Input tensor must have contiguous last dimension");

    const auto sizes = q.sizes();

    const int batch_size = sizes[0];
    int seqlen_q = sizes[1];
    int num_heads = sizes[2];
    const int head_size = sizes[3];
    const int seqlen_k = k.size(1);
    const int num_heads_k = k.size(2);
    TORCH_CHECK(batch_size > 0, "batch size must be positive");
    TORCH_CHECK(head_size <= 256, "FlashAttention forward only supports head dimension at most 256");
    TORCH_CHECK(head_size % 8 == 0, "query, key, value, and out_ must have a head_size that is a multiple of 8");
    TORCH_CHECK(num_heads % num_heads_k == 0, "Number of heads in key/value must divide number of heads in query");

    if (softcap > 0.f) { TORCH_CHECK(p_dropout == 0.f, "Softcapping does not support dropout for now"); }

    if (window_size_left >= seqlen_k) { window_size_left = -1; }
    if (window_size_right >= seqlen_k) { window_size_right = -1; }

    // causal=true is the same as causal=false in this case
    if (seqlen_q == 1 && !alibi_slopes_.has_value()) { is_causal = false; }
    if (is_causal) { window_size_right = 0; }

    // Faster to transpose q from (b, 1, (nheads_kv ngroups), d) to (b, ngroups, nheads_kv, d) in this case
    // H/t Daniel Haziza
    const int seqlenq_ngroups_swapped = seqlen_q == 1 && num_heads > num_heads_k && window_size_left < 0 && window_size_right < 0 && p_dropout == 0.f && head_size % 8 == 0 && !alibi_slopes_.has_value();
    const int ngroups = num_heads / num_heads_k;
    if (seqlenq_ngroups_swapped) {
        q = q.reshape({batch_size, num_heads_k, ngroups, head_size}).transpose(1, 2);
        seqlen_q = ngroups;
        num_heads = num_heads_k;
    }

    CHECK_SHAPE(q, batch_size, seqlen_q, num_heads, head_size);
    CHECK_SHAPE(k, batch_size, seqlen_k, num_heads_k, head_size);
    CHECK_SHAPE(v, batch_size, seqlen_k, num_heads_k, head_size);

    at::Tensor out;
    if (out_.has_value()) {
        out = out_.value();
        TORCH_CHECK(out.dtype() == q_dtype, "Output must have the same dtype as inputs");
        CHECK_DEVICE(out);
        TORCH_CHECK(out.stride(-1) == 1, "Output tensor must have contiguous last dimension");
        CHECK_SHAPE(out, batch_size, sizes[1], sizes[2], head_size);
        if (seqlenq_ngroups_swapped) {
            out = out.reshape({batch_size, num_heads_k, ngroups, head_size}).transpose(1, 2);
        }
    } else {
        out = torch::empty_like(q);
    }

    auto round_multiple = [](int x, int m) { return (x + m - 1) / m * m; };
    const int head_size_rounded = round_multiple(head_size, head_size <= 128 ? 32 : 64);
    const int seqlen_q_rounded = round_multiple(seqlen_q, 128);
    const int seqlen_k_rounded = round_multiple(seqlen_k, 128);

    auto opts = q.options();

    auto softmax_lse = torch::empty({batch_size, num_heads, seqlen_q}, opts.dtype(at::kFloat));
    at::Tensor p;
    // Only return softmax if there's dropout to reduce compilation time
    if (return_softmax) {
        TORCH_CHECK(p_dropout > 0.0f, "return_softmax is only supported when p_dropout > 0.0");
        p = torch::empty({ batch_size, num_heads, seqlen_q_rounded, seqlen_k_rounded }, opts);
    }
    else {
        p = torch::empty({ 0 }, opts);
    }

    Flash_fwd_params params;
    set_params_fprop(params,
                     batch_size,
                     seqlen_q, seqlen_k,
                     seqlen_q_rounded, seqlen_k_rounded,
                     num_heads, num_heads_k,
                     head_size, head_size_rounded,
                     q, k, v, out,
                     /*cu_seqlens_q_d=*/nullptr,
                     /*cu_seqlens_k_d=*/nullptr,
                     /*seqused_k=*/nullptr,
                     return_softmax ? p.data_ptr() : nullptr,
                     softmax_lse.data_ptr(),
                     p_dropout,
                     softmax_scale,
                     window_size_left,
                     window_size_right,
                     softcap
                     );

    // Keep references to these tensors to extend their lifetime
    at::Tensor softmax_lse_accum, out_accum, softmax_lse_hist_accum;
    std::tie(softmax_lse_accum, out_accum, softmax_lse_hist_accum) = set_params_splitkv(
        params, batch_size, num_heads, head_size, seqlen_k, seqlen_q,
        head_size_rounded, p_dropout, /*num_splits*/ 0, get_num_sm(get_current_device()), opts);

    // number of times random will be generated per thread, to offset philox counter in thc random
    // state
    // We use a custom RNG that increases the offset by batch_size * nheads * 32.
    int64_t counter_offset = params.b * params.h * 32;
    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
    auto rng_state = torch::empty({2}, options.dtype(torch::kInt64));
    // Forward kernel will populate memory with the seed and offset.
    params.rng_state = reinterpret_cast<uint64_t*>(rng_state.data_ptr());

    if (p_dropout > 0.0)  {
        auto gen = at::get_generator_or_default<at::CUDAGeneratorImpl>(
            gen_, at::cuda::detail::getDefaultCUDAGenerator());
        // See Note [Acquire lock when using random generators]
        std::lock_guard<std::mutex> lock(gen->mutex_);
        params.philox_args = gen->philox_cuda_state(counter_offset);
    }

    set_params_alibi(params, alibi_slopes_, batch_size, num_heads);

    if (seqlen_k > 0) {
        auto stream = at::cuda::getCurrentCUDAStream().stream();
        run_mha_fwd(params, stream);
    } else {
        // If seqlen_k == 0, then we have an empty tensor. We need to set the output to 0.
        out.zero_();
        softmax_lse.fill_(std::numeric_limits<float>::infinity());
    }

    if (seqlenq_ngroups_swapped) {
        out = out.transpose(1, 2).reshape({batch_size, 1, num_heads_k * seqlen_q, head_size});
        q = q.transpose(1, 2).reshape({batch_size, 1, num_heads_k * seqlen_q, head_size});
        softmax_lse = softmax_lse.reshape({batch_size, num_heads_k * seqlen_q, 1});
    }
    return {out, softmax_lse, p, rng_state};
}

at::Tensor update_attn_fused(
    at::Tensor softmax_lse,
    at::Tensor logsumexp_tensor,
    at::Tensor out_hist,
    at::Tensor softmax_lse_hist,
    at::Tensor attn_out_tensor,
    at::Tensor o,
    at::Tensor need_mask,
    at::Tensor store_mask,
    at::Tensor dirty_mask
);

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
);

std::vector<at::Tensor> update_attn_fused_varlen(
    at::Tensor softmax_lse,
    at::Tensor logsumexp_tensor,
    at::Tensor out_hist,
    at::Tensor softmax_lse_hist,
    at::Tensor attn_out_tensor,
    at::Tensor o,
    std::optional<at::Tensor> head_mask
) {
    CHECK_DEVICE(softmax_lse);
    CHECK_DEVICE(logsumexp_tensor);
    CHECK_DEVICE(out_hist);
    CHECK_DEVICE(softmax_lse_hist);
    CHECK_DEVICE(attn_out_tensor);
    CHECK_DEVICE(o);
    if (head_mask.has_value()) { CHECK_DEVICE(head_mask.value()); }
    CHECK_CONTIGUOUS(softmax_lse);
    CHECK_CONTIGUOUS(logsumexp_tensor);
    CHECK_CONTIGUOUS(out_hist);
    CHECK_CONTIGUOUS(softmax_lse_hist);
    CHECK_CONTIGUOUS(attn_out_tensor);
    CHECK_CONTIGUOUS(o);
    if (head_mask.has_value()) { CHECK_CONTIGUOUS(head_mask.value()); }
    TORCH_CHECK(logsumexp_tensor.scalar_type() == at::kFloat, "logsumexp must be float32");
    TORCH_CHECK(softmax_lse.scalar_type() == at::kFloat, "softmax_lse must be float32");
    TORCH_CHECK(softmax_lse_hist.scalar_type() == at::kFloat, "softmax_lse_hist must be float32");
    TORCH_CHECK(out_hist.scalar_type() == o.scalar_type(), "out_hist must match output dtype");
    TORCH_CHECK(attn_out_tensor.scalar_type() == o.scalar_type(), "attn_output_past must match output dtype");
    if (head_mask.has_value()) {
        TORCH_CHECK(head_mask.value().scalar_type() == at::kBool, "head_mask must be bool");
    }

    const int Tq = o.size(0);
    const int H = o.size(1);
    const int D = o.size(2);
    CHECK_SHAPE(out_hist, Tq, H, D);
    CHECK_SHAPE(softmax_lse, H, Tq);
    CHECK_SHAPE(softmax_lse_hist, H, Tq);
    CHECK_SHAPE(logsumexp_tensor, H, Tq);
    if (head_mask.has_value()) { CHECK_SHAPE(head_mask.value(), H); }
    TORCH_CHECK(attn_out_tensor.numel() == o.numel(), "attn_output_past must have the same numel as output");

    at::Tensor out = torch::empty_like(o);
    at::Tensor attn_out_updated = torch::empty_like(attn_out_tensor);
    at::Tensor logsumexp_updated = torch::empty_like(logsumexp_tensor);
    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half,
        at::ScalarType::BFloat16,
        o.scalar_type(),
        "update_attn_fused_varlen",
        [&](){
            using kernel_t =
                std::conditional_t<std::is_same_v<scalar_t, at::Half>,
                    half,
                std::conditional_t<std::is_same_v<scalar_t, at::BFloat16>,
                    __nv_bfloat16,
                    float>>;
            launch_update_attn_fused_varlen_kernel<kernel_t>(
                softmax_lse.data_ptr<float>(),
                logsumexp_tensor.data_ptr<float>(),
                reinterpret_cast<const kernel_t*>(out_hist.data_ptr<scalar_t>()),
                softmax_lse_hist.data_ptr<float>(),
                reinterpret_cast<const kernel_t*>(attn_out_tensor.data_ptr<scalar_t>()),
                reinterpret_cast<const kernel_t*>(o.data_ptr<scalar_t>()),
                head_mask.has_value() ? head_mask.value().data_ptr<bool>() : nullptr,
                reinterpret_cast<kernel_t*>(out.data_ptr<scalar_t>()),
                reinterpret_cast<kernel_t*>(attn_out_updated.data_ptr<scalar_t>()),
                logsumexp_updated.data_ptr<float>(),
                Tq, H, D
            );
        }
    );
    return {out, attn_out_updated, logsumexp_updated};
}

std::vector<at::Tensor>
mha_varlen_fwd(at::Tensor &q,  // total_q x num_heads x head_size, total_q := \sum_{i=0}^{b} s_i
               const at::Tensor &k,  // total_k x num_heads_k x head_size, total_k := \sum_{i=0}^{b} s_i or num_blocks x page_block_size x num_heads_k x head_size if there's a block_table.
               const at::Tensor &v,  // total_k x num_heads_k x head_size, total_k := \sum_{i=0}^{b} s_i or num_blocks x page_block_size x num_heads_k x head_size if there's a block_table.
               std::optional<at::Tensor> &out_, // total_q x num_heads x head_size, total_q := \sum_{i=0}^{b} s_i
               const at::Tensor &cu_seqlens_q,  // b+1
               const at::Tensor &cu_seqlens_k,  // b+1
               std::optional<at::Tensor> &seqused_k, // b. If given, only this many elements of each batch element's keys are used.
               std::optional<const at::Tensor> &leftpad_k_, // batch_size
               std::optional<at::Tensor> &block_table_, // batch_size x max_num_blocks_per_seq
               std::optional<at::Tensor> &alibi_slopes_, // num_heads or b x num_heads
               int max_seqlen_q,
               const int max_seqlen_k,
               const float p_dropout,
               const float softmax_scale,
               const bool zero_tensors,
               bool is_causal,
               int window_size_left,
               int window_size_right,
               const float softcap,
               const bool return_softmax,
               std::optional<at::Generator> gen_,
               std::optional<at::Tensor> head_mask,
               std::optional<at::Tensor> attn_output_past,
               std::optional<at::Tensor> logsumexp,
               std::optional<at::Tensor> need_update_mask,
               std::optional<at::Tensor> dirty_mask,
               const int block_size=0) {

    // Otherwise the kernel will be launched from cuda:0 device
    at::cuda::CUDAGuard device_guard{q.device()};

    auto [cc_major, cc_minor] = get_compute_capability(get_current_device());
    bool is_sm8x_min = cc_major >= 8;
    TORCH_CHECK(is_sm8x_min, "FlashAttention only supports Ampere GPUs or newer.");

    auto q_dtype = q.dtype();
    TORCH_CHECK(q_dtype == torch::kFloat16 || q_dtype == torch::kBFloat16,
                "FlashAttention only support fp16 and bf16 data type");
    TORCH_CHECK(k.dtype() == q_dtype, "query and key must have the same dtype");
    TORCH_CHECK(v.dtype() == q_dtype, "query and value must have the same dtype");
    TORCH_CHECK(cu_seqlens_q.dtype() == torch::kInt32, "cu_seqlens_q must have dtype int32");
    TORCH_CHECK(cu_seqlens_k.dtype() == torch::kInt32, "cu_seqlens_k must have dtype int32");

    CHECK_DEVICE(q); CHECK_DEVICE(k); CHECK_DEVICE(v);
    CHECK_DEVICE(cu_seqlens_q);
    CHECK_DEVICE(cu_seqlens_k);

    at::Tensor block_table;
    const bool paged_KV = block_table_.has_value();
    if (paged_KV) {
        block_table = block_table_.value();
        CHECK_DEVICE(block_table);
        TORCH_CHECK(block_table.dtype() == torch::kInt32, "block_table must have dtype torch.int32");
        TORCH_CHECK(block_table.stride(-1) == 1, "block_table must have contiguous last dimension");
    }

    TORCH_CHECK(q.stride(-1) == 1, "Input tensor must have contiguous last dimension");
    TORCH_CHECK(k.stride(-1) == 1, "Input tensor must have contiguous last dimension");
    TORCH_CHECK(v.stride(-1) == 1, "Input tensor must have contiguous last dimension");
    CHECK_CONTIGUOUS(cu_seqlens_q);
    CHECK_CONTIGUOUS(cu_seqlens_k);

    const auto sizes = q.sizes();

    const int batch_size = cu_seqlens_q.numel() - 1;
    int num_heads = sizes[1];
    const int head_size = sizes[2];
    const int num_heads_k = paged_KV ? k.size(2) : k.size(1);

    if (softcap > 0.f) { TORCH_CHECK(p_dropout == 0.f, "Softcapping does not support dropout for now"); }

    const int max_num_blocks_per_seq = !paged_KV ? 0 : block_table.size(1);
    const int num_blocks = !paged_KV ? 0 : k.size(0);
    const int page_block_size = !paged_KV ? 1 : k.size(1);
    TORCH_CHECK(!paged_KV || page_block_size % 256 == 0, "Paged KV cache block size must be divisible by 256");

    if (max_seqlen_q == 1 && !alibi_slopes_.has_value()) { is_causal = false; }  // causal=true is the same as causal=false in this case
    if (is_causal) { window_size_right = 0; }

    void *cu_seqlens_q_d = cu_seqlens_q.data_ptr();

    // Faster to transpose q from (b, 1, (nheads_kv ngroups), d) to (b, ngroups, nheads_kv, d) in this case
    // H/t Daniel Haziza
    const int seqlenq_ngroups_swapped = max_seqlen_q == 1 && num_heads > num_heads_k && window_size_left < 0 && window_size_right < 0 && p_dropout == 0.f && head_size % 8 == 0 && !alibi_slopes_.has_value();
    const int ngroups = num_heads / num_heads_k;
    if (seqlenq_ngroups_swapped) {
        q = q.reshape({batch_size, num_heads_k, ngroups, head_size}).transpose(1, 2).reshape({batch_size * ngroups, num_heads_k, head_size});
        max_seqlen_q = ngroups;
        num_heads = num_heads_k;
        cu_seqlens_q_d = nullptr;
    }

    const int total_q = q.sizes()[0];

    TORCH_CHECK(batch_size > 0, "batch size must be positive");
    TORCH_CHECK(head_size <= 256, "FlashAttention forward only supports head dimension at most 256");
    TORCH_CHECK(head_size % 8 == 0, "query, key, value, and out_ must have a head_size that is a multiple of 8");
    TORCH_CHECK(num_heads % num_heads_k == 0, "Number of heads in key/value must divide number of heads in query");

    if (window_size_left >= max_seqlen_k) { window_size_left = -1; }
    if (window_size_right >= max_seqlen_k) { window_size_right = -1; }

    CHECK_SHAPE(q, total_q, num_heads, head_size);
    if (!paged_KV) {
        const int total_k = k.size(0);
        CHECK_SHAPE(k, total_k, num_heads_k, head_size);
        CHECK_SHAPE(v, total_k, num_heads_k, head_size);
    } else {
        CHECK_SHAPE(k, num_blocks, page_block_size, num_heads_k, head_size);
        CHECK_SHAPE(v, num_blocks, page_block_size, num_heads_k, head_size);
        CHECK_SHAPE(block_table, batch_size, max_num_blocks_per_seq);
    }

    CHECK_SHAPE(cu_seqlens_q, batch_size + 1);
    CHECK_SHAPE(cu_seqlens_k, batch_size + 1);
    if (seqused_k.has_value()){
        auto seqused_k_ = seqused_k.value();
        TORCH_CHECK(seqused_k_.dtype() == torch::kInt32, "seqused_k must have dtype int32");
        TORCH_CHECK(seqused_k_.is_cuda(), "seqused_k must be on CUDA device");
        TORCH_CHECK(seqused_k_.is_contiguous(), "seqused_k must be contiguous");
        CHECK_SHAPE(seqused_k_, batch_size);
    }

    at::Tensor out;
    if (out_.has_value()) {
        out = out_.value();
        TORCH_CHECK(out.dtype() == q_dtype, "Output must have the same dtype as inputs");
        CHECK_DEVICE(out);
        TORCH_CHECK(out.stride(-1) == 1, "Output tensor must have contiguous last dimension");
        CHECK_SHAPE(out, sizes[0], sizes[1], head_size);
        if (seqlenq_ngroups_swapped) {
            out = out.reshape({batch_size, num_heads_k, ngroups, head_size}).transpose(1, 2).reshape({batch_size * ngroups, num_heads_k, head_size});
        }
    } else {
        out = torch::empty_like(q);
    }

    auto round_multiple = [](int x, int m) { return (x + m - 1) / m * m; };
    const int head_size_rounded = round_multiple(head_size, head_size <= 128 ? 32 : 64);
    const int seqlen_q_rounded = round_multiple(max_seqlen_q, 128);
    const int seqlen_k_rounded = round_multiple(max_seqlen_k, 128);

    auto opts = q.options();
    auto softmax_lse = torch::empty({num_heads, total_q}, opts.dtype(at::kFloat));
    auto softmax_lse_hist = torch::empty({num_heads, total_q}, opts.dtype(at::kFloat));
    at::Tensor p;
    at::Tensor p_hist;
    // Only return softmax if there's dropout to reduce compilation time
    if (return_softmax) {
        TORCH_CHECK(p_dropout > 0.0f, "return_softmax is only supported when p_dropout > 0.0");
        p = torch::empty({ batch_size, num_heads, seqlen_q_rounded, seqlen_k_rounded }, opts);
        p_hist = torch::empty({ batch_size, num_heads, seqlen_q_rounded, seqlen_k_rounded }, opts);
    }
    else {
        p = torch::empty({ 0 }, opts);
        p_hist = torch::empty({ 0 }, opts);
    }

    if (zero_tensors) {
        out.zero_();
        softmax_lse.fill_(-std::numeric_limits<float>::infinity());
        softmax_lse_hist.fill_(-std::numeric_limits<float>::infinity());
        if (return_softmax) {p.zero_();p_hist.zero_();}
    }

    at::Tensor out_hist;
    at::Tensor out_cur_fp32;
    out_hist = torch::empty_like(q);

    Flash_fwd_params params;
    set_params_fprop(params,
                     batch_size,
                     max_seqlen_q, max_seqlen_k,
                     seqlen_q_rounded, seqlen_k_rounded,
                     num_heads, num_heads_k,
                     head_size, head_size_rounded,
                     q, k, v, out,
                     cu_seqlens_q_d,
                     cu_seqlens_k.data_ptr(),
                     seqused_k.has_value() ? seqused_k.value().data_ptr() : nullptr,
                     return_softmax ? p.data_ptr() : nullptr,
                     softmax_lse.data_ptr(),
                     p_dropout,
                     softmax_scale,
                     window_size_left,
                     window_size_right,
                     softcap,
                     seqlenq_ngroups_swapped,
                     /*unpadded_lse*/true);
    
    params.block_size = block_size;
    params.need_store_history_mask = nullptr;
    params.lse_hist_ptr=softmax_lse_hist.data_ptr();
    params.o_hist_ptr=out_hist.data_ptr();
    params.total_q = total_q;
    params.p_hist_ptr = p_hist.data_ptr();
    params.use_head_mask = false;
    params.update_past = false;
    params.merge_full_in_kernel = false;
    if (head_mask.has_value()){
        params.use_head_mask = true;
        params.head_mask = head_mask.value().data_ptr();
    }
    if (attn_output_past.has_value()) {
        TORCH_CHECK(logsumexp.has_value(), "logsumexp must be passed for fused varlen reuse merge");
        TORCH_CHECK(!seqlenq_ngroups_swapped, "fused varlen reuse merge does not support seqlenq_ngroups_swapped");
        TORCH_CHECK(total_q == batch_size * max_seqlen_q, "fused varlen reuse merge expects dense equal-length batch");
        CHECK_DEVICE(attn_output_past.value());
        CHECK_DEVICE(logsumexp.value());
        if (head_mask.has_value()) { CHECK_DEVICE(head_mask.value()); }
        TORCH_CHECK(attn_output_past.value().dtype() == q_dtype, "attn_output_past must have the same dtype as query");
        TORCH_CHECK(logsumexp.value().dtype() == torch::kFloat32, "logsumexp must be fp32 for fused varlen reuse merge");
        if (head_mask.has_value()) {
            TORCH_CHECK(head_mask.value().dtype() == torch::kBool, "head_mask must be bool");
        }
        TORCH_CHECK(attn_output_past.value().stride(-1) == 1, "attn_output_past must have contiguous last dimension");
        CHECK_CONTIGUOUS(logsumexp.value());
        if (head_mask.has_value()) { CHECK_CONTIGUOUS(head_mask.value()); }
        CHECK_SHAPE(attn_output_past.value(), batch_size, max_seqlen_q, num_heads, head_size);
        CHECK_SHAPE(logsumexp.value(), num_heads, total_q);
        if (head_mask.has_value()) { CHECK_SHAPE(head_mask.value(), num_heads); }
    }

    if (paged_KV) {
        params.block_table = block_table.data_ptr<int>();
        params.block_table_batch_stride = block_table.stride(0);
        params.k_batch_stride = k.stride(0);
        params.v_batch_stride = v.stride(0);
    }
    params.page_block_size = page_block_size;
    // Keep references to these tensors to extend their lifetime
    at::Tensor softmax_lse_accum,softmax_lse_hist_accum, out_accum,out_accum_hist;
    if (seqlenq_ngroups_swapped) {
        // Only apply split-k for decoding
        std::tie(softmax_lse_accum, out_accum, softmax_lse_hist_accum) =
            set_params_splitkv(params, batch_size, num_heads, head_size,
                               max_seqlen_k, max_seqlen_q, head_size_rounded,
                               p_dropout, /*num_splits*/ 0, get_num_sm(get_current_device()), opts);
    }

    if (leftpad_k_.has_value()) {
        auto leftpad_k = leftpad_k_.value();
        TORCH_CHECK(!paged_KV, "We don't support Paged KV and leftpad_k running at the same time yet");
        TORCH_CHECK(leftpad_k.dtype() == torch::kInt32, "leftpad_k must have dtype int32");
        CHECK_DEVICE(leftpad_k);
        CHECK_CONTIGUOUS(leftpad_k);
        CHECK_SHAPE(leftpad_k, batch_size);
        params.leftpad_k = static_cast<int *>(leftpad_k.data_ptr());
    }

    // number of times random will be generated per thread, to offset philox counter in thc random
    // state
    // We use a custom RNG that increases the offset by batch_size * nheads * 32.
    int64_t counter_offset = params.b * params.h * 32;
    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
    auto rng_state = torch::empty({2}, options.dtype(torch::kInt64));
    // Forward kernel will populate memory with the seed and offset.
    params.rng_state = reinterpret_cast<uint64_t*>(rng_state.data_ptr());

    if (p_dropout > 0.0)  {
        auto gen = at::get_generator_or_default<at::CUDAGeneratorImpl>(
            gen_, at::cuda::detail::getDefaultCUDAGenerator());
        // See Note [Acquire lock when using random generators]
        std::lock_guard<std::mutex> lock(gen->mutex_);
        params.philox_args = gen->philox_cuda_state(counter_offset);
    }

    set_params_alibi(params, alibi_slopes_, batch_size, num_heads);

    if (max_seqlen_k > 0) {
        auto stream = at::cuda::getCurrentCUDAStream().stream();
        run_mha_fwd(params, stream, paged_KV);
    } else {
        // If seqlen_k == 0, then we have an empty tensor. We need to set the output to 0.
        out.zero_();
        softmax_lse.fill_(std::numeric_limits<float>::infinity());
        if (block_size>0){
            out_hist.zero_();
            softmax_lse_hist.fill_(std::numeric_limits<float>::infinity());
        }
    }

    if (seqlenq_ngroups_swapped) {
        int64_t size_before[] = {batch_size, max_seqlen_q, num_heads_k, head_size};
        int64_t size_after[] = {batch_size, num_heads_k * max_seqlen_q, head_size};
        out = out.reshape(size_before).transpose(1, 2).reshape(size_after);
        q = q.reshape(size_before).transpose(1, 2).reshape(size_after);
        softmax_lse = softmax_lse.reshape({num_heads * max_seqlen_q, batch_size});
        if (block_size>0){
            softmax_lse_hist = softmax_lse_hist.reshape({num_heads * max_seqlen_q, batch_size});
            out_hist = out_hist.reshape(size_before).transpose(1, 2).reshape(size_after);
        }
    }
    if(block_size>0){
        if (attn_output_past.has_value()) {
            auto fused_outputs = update_attn_fused_varlen(
                softmax_lse,
                logsumexp.value(),
                out_hist,
                softmax_lse_hist,
                attn_output_past.value(),
                out,
                head_mask
            );
            return {
                fused_outputs[0],
                softmax_lse,
                out_hist,
                softmax_lse_hist,
                p,
                rng_state,
                fused_outputs[1],
                fused_outputs[2]
            };
        }
        return {out, softmax_lse, out_hist, softmax_lse_hist,p, rng_state};
    }

    return {out, softmax_lse, p, rng_state};
}

// void run_mha_bwd(Flash_bwd_params &params, cudaStream_t stream) {
//     FP16_SWITCH(!params.is_bf16, [&] {
//         HEADDIM_SWITCH(params.d, [&] {
//             BOOL_SWITCH(params.is_causal, Is_causal, [&] {
//                 run_mha_bwd_<elem_type, kHeadDim, Is_causal>(params, stream);
//             });
//         });
//     });
// }

void run_mha_bwd(Flash_bwd_params &params, cudaStream_t stream) {
    using elem_type = cutlass::bfloat16_t;
    constexpr int kHeadDim = 128;
    constexpr bool Is_causal = false;  // 固定为非因果注意力
    run_mha_bwd_<elem_type, kHeadDim, Is_causal>(params, stream);
}



std::vector<at::Tensor>
mha_bwd(const at::Tensor &dout,  // batch_size x seqlen_q x num_heads, x multiple_of(head_size_og, 8)
        const at::Tensor &q,   // batch_size x seqlen_q x num_heads x head_size
        const at::Tensor &k,   // batch_size x seqlen_k x num_heads_k x head_size
        const at::Tensor &v,   // batch_size x seqlen_k x num_heads_k x head_size
        const at::Tensor &out,   // batch_size x seqlen_q x num_heads x head_size
        const at::Tensor &softmax_lse,     // b x h x seqlen_q
        std::optional<at::Tensor> &dq_,   // batch_size x seqlen_q x num_heads x head_size
        std::optional<at::Tensor> &dk_,   // batch_size x seqlen_k x num_heads_k x head_size
        std::optional<at::Tensor> &dv_,   // batch_size x seqlen_k x num_heads_k x head_size
        std::optional<at::Tensor> &alibi_slopes_, // num_heads or batch_size x num_heads
        const float p_dropout,         // probability to drop
        const float softmax_scale,
        const bool is_causal,
        int window_size_left,
        int window_size_right,
        const float softcap,
        const bool deterministic,
        std::optional<at::Generator> gen_,
        std::optional<at::Tensor> &rng_state) {

    #ifdef FLASHATTENTION_DISABLE_BACKWARD
        TORCH_CHECK(false, "This flash attention build does not support backward.");
    #endif
    if (is_causal) { window_size_right = 0; }

    // Otherwise the kernel will be launched from cuda:0 device
    at::cuda::CUDAGuard device_guard{q.device()};

    auto [cc_major, cc_minor] = get_compute_capability(get_current_device());
    bool is_sm8x_min = cc_major >= 8;
    TORCH_CHECK(is_sm8x_min, "FlashAttention only supports Ampere GPUs or newer.");

    bool is_dropout = p_dropout > 0.0;
    auto stream = at::cuda::getCurrentCUDAStream().stream();

    auto q_dtype = q.dtype();
    TORCH_CHECK(q_dtype == torch::kFloat16 || q_dtype == torch::kBFloat16,
                "FlashAttention only support fp16 and bf16 data type");
    TORCH_CHECK(k.dtype() == q_dtype, "query and key must have the same dtype");
    TORCH_CHECK(v.dtype() == q_dtype, "query and value must have the same dtype");
    TORCH_CHECK(out.dtype() == q_dtype, "query and out must have the same dtype");
    TORCH_CHECK(dout.dtype() == q_dtype, "query and dout must have the same dtype");

    CHECK_DEVICE(q); CHECK_DEVICE(k); CHECK_DEVICE(v);
    CHECK_DEVICE(out); CHECK_DEVICE(dout); CHECK_DEVICE(softmax_lse);

    TORCH_CHECK(q.stride(-1) == 1, "Input tensor must have contiguous last dimension");
    TORCH_CHECK(k.stride(-1) == 1, "Input tensor must have contiguous last dimension");
    TORCH_CHECK(v.stride(-1) == 1, "Input tensor must have contiguous last dimension");
    TORCH_CHECK(out.stride(-1) == 1, "out tensor must have contiguous last dimension");
    TORCH_CHECK(dout.stride(-1) == 1, "dout tensor must have contiguous last dimension");

    const auto sizes = q.sizes();

    const int batch_size = sizes[0];
    const int seqlen_q = sizes[1];
    const int num_heads = sizes[2];
    const int head_size = sizes[3];
    const int seqlen_k = k.size(1);
    const int num_heads_k = k.size(2);
    TORCH_CHECK(batch_size > 0, "batch size must be positive");
    TORCH_CHECK(head_size % 8 == 0, "head_size should be a multiple of 8");
    TORCH_CHECK(head_size <= 256, "FlashAttention backward only supports head dimension at most 256");
    TORCH_CHECK(num_heads % num_heads_k == 0, "Number of heads in key/value must divide number of heads in query");

    auto round_multiple = [](int x, int m) { return (x + m - 1) / m * m; };
    const int head_size_rounded = round_multiple(head_size, head_size <= 128 ? 32 : 64);
    const int seqlen_q_rounded = round_multiple(seqlen_q, 128);
    const int seqlen_k_rounded = round_multiple(seqlen_k, 128);

    if (softcap > 0.f) { TORCH_CHECK(p_dropout == 0.f, "Softcapping does not support dropout for now"); }

    if (window_size_left >= seqlen_k) { window_size_left = -1; }
    if (window_size_right >= seqlen_k) { window_size_right = -1; }

    CHECK_SHAPE(q, batch_size, seqlen_q, num_heads, head_size);
    CHECK_SHAPE(k, batch_size, seqlen_k, num_heads_k, head_size);
    CHECK_SHAPE(v, batch_size, seqlen_k, num_heads_k, head_size);
    CHECK_SHAPE(out, batch_size, seqlen_q, num_heads, head_size);
    CHECK_SHAPE(dout, batch_size, seqlen_q, num_heads, head_size);

    at::Tensor dq, dk, dv;
    if (dq_.has_value()) {
        dq = dq_.value();
        TORCH_CHECK(dq.dtype() == q_dtype, "dq must have the same dtype as q");
        CHECK_DEVICE(dq);
        TORCH_CHECK(dq.stride(-1) == 1, "dq must have contiguous last dimension");
        CHECK_SHAPE(dq, batch_size, seqlen_q, num_heads, head_size);
    } else {
        dq = torch::empty_like(q);
    }
    if (dk_.has_value()) {
        dk = dk_.value();
        TORCH_CHECK(dk.dtype() == q_dtype, "dk must have the same dtype as q");
        CHECK_DEVICE(dk);
        TORCH_CHECK(dk.stride(-1) == 1, "dk must have contiguous last dimension");
        CHECK_SHAPE(dk, batch_size, seqlen_k, num_heads_k, head_size);
    } else {
        dk = torch::empty_like(k);
    }
    if (dv_.has_value()) {
        dv = dv_.value();
        TORCH_CHECK(dv.dtype() == q_dtype, "dv must have the same dtype as q");
        CHECK_DEVICE(dv);
        TORCH_CHECK(dv.stride(-1) == 1, "dv must have contiguous last dimension");
        CHECK_SHAPE(dv, batch_size, seqlen_k, num_heads_k, head_size);
    } else {
        dv = torch::empty_like(v);
    }

    // bool loop = seqlen_k > blocksize_c;
    // TODO: change later, for now set to true for simplicity
    bool loop = true;

    auto opts = q.options();
    auto softmax_d = torch::empty({batch_size, num_heads, seqlen_q_rounded}, opts.dtype(at::kFloat));
    at::Tensor dq_accum;
    at::Tensor dk_accum, dv_accum;
    if (loop) {
        if (!deterministic) {
            dq_accum = torch::empty({batch_size, seqlen_q_rounded, num_heads, head_size_rounded}, opts.dtype(at::kFloat));
        } else {
            const int nsplits = (get_num_sm(get_current_device()) + batch_size * num_heads - 1) / (batch_size * num_heads);
            dq_accum = torch::zeros({nsplits, batch_size, seqlen_q_rounded, num_heads, head_size_rounded}, opts.dtype(at::kFloat));
        }
        // dk_accum = torch::empty({batch_size, num_heads_k, seqlen_k_rounded, head_size_rounded}, opts.dtype(at::kFloat));
        // dv_accum = torch::empty({batch_size, num_heads_k, seqlen_k_rounded, head_size_rounded}, opts.dtype(at::kFloat));
    }

    at::Tensor dk_expanded, dv_expanded;
    if (num_heads_k != num_heads) {  // MQA / GQA
        dk_expanded = torch::empty({batch_size, seqlen_k, num_heads, head_size}, opts);
        dv_expanded = torch::empty({batch_size, seqlen_k, num_heads, head_size}, opts);
    } else {
        dk_expanded = dk;
        dv_expanded = dv;
    }

    Flash_bwd_params params;

    set_params_dgrad(params,
                     batch_size,
                     seqlen_q, seqlen_k,
                     seqlen_q_rounded, seqlen_k_rounded,
                     num_heads, num_heads_k,
                     head_size, head_size_rounded,
                     q, k, v, out,
                     dout, dq, dk_expanded, dv_expanded,
                     nullptr,
                     nullptr,
                     loop ? dq_accum.data_ptr() : nullptr,
                     // loop ? dk_accum.data_ptr() : nullptr,
                     // loop ? dv_accum.data_ptr() : nullptr,
                     nullptr,
                     nullptr,
                     softmax_lse.data_ptr(),
                     softmax_d.data_ptr(),
                     p_dropout,
                     softmax_scale,
                     window_size_left,
                     window_size_right,
                     softcap,
                     deterministic,
                     /*unpadded_lse*/false);
    params.dq_accum_split_stride = !deterministic ? 0 : dq_accum.stride(0);

    auto launch = &run_mha_bwd;

    auto gen = at::get_generator_or_default<at::CUDAGeneratorImpl>(
        gen_, at::cuda::detail::getDefaultCUDAGenerator());

    // We use a custom RNG that increases the offset by batch_size * nheads * 32.
    int64_t counter_offset = params.b * params.h * 32;

    if ( rng_state.has_value() ) {
        params.rng_state = reinterpret_cast<uint64_t*>(rng_state.value().data_ptr());
    } else if( is_dropout ) {
        // See Note [Acquire lock when using random generators]
        std::lock_guard<std::mutex> lock(gen->mutex_);
        params.philox_args = gen->philox_cuda_state(counter_offset);
        auto seeds = at::cuda::philox::unpack(params.philox_args);
        params.rng_state[0] = std::get<0>(seeds);
        params.rng_state[1] = std::get<1>(seeds);
    }

    set_params_alibi(params, alibi_slopes_, batch_size, num_heads);

    if (seqlen_q > 0) {
        launch(params, stream);
    } else {
        // If seqlen_q == 0, then we have an empty tensor. We need to set the output to 0.
        dk_expanded.zero_();
        dv_expanded.zero_();
        softmax_d.zero_();
    }

    // For MQA/GQA we need to sum dK and dV across the groups
    if (num_heads_k != num_heads) {
        at::sum_out(dk, at::reshape(dk_expanded, {batch_size, seqlen_k, num_heads_k, num_heads / num_heads_k, head_size}), {3});
        at::sum_out(dv, at::reshape(dv_expanded, {batch_size, seqlen_k, num_heads_k, num_heads / num_heads_k, head_size}), {3});
    }

    return { dq, dk, dv, softmax_d };
}

std::vector<at::Tensor>
mha_varlen_bwd(const at::Tensor &dout,  // total_q x num_heads, x head_size
               const at::Tensor &q,   // total_q x num_heads x head_size, total_q := \sum_{i=0}^{b} s_i
               const at::Tensor &k,   // total_k x num_heads_k x head_size, total_k := \sum_{i=0}^{b} s_i
               const at::Tensor &v,   // total_k x num_heads_k x head_size, total_k := \sum_{i=0}^{b} s_i
               const at::Tensor &out,   // total_q x num_heads x head_size
               const at::Tensor &softmax_lse,    // h x total_q, softmax logsumexp
               std::optional<at::Tensor> &dq_,   // total_q x num_heads x head_size, total_q := \sum_{i=0}^{b} s_i
               std::optional<at::Tensor> &dk_,   // total_k x num_heads_k x head_size, total_k := \sum_{i=0}^{b} s_i
               std::optional<at::Tensor> &dv_,   // total_k x num_heads_k x head_size, total_k := \sum_{i=0}^{b} s_i
               const at::Tensor &cu_seqlens_q,  // b+1
               const at::Tensor &cu_seqlens_k,  // b+1
               std::optional<at::Tensor> &alibi_slopes_, // num_heads or b x num_heads
               const int max_seqlen_q,
               const int max_seqlen_k,          // max sequence length to choose the kernel
               const float p_dropout,         // probability to drop
               const float softmax_scale,
               const bool zero_tensors,
               const bool is_causal,
               int window_size_left,
               int window_size_right,
               const float softcap,
               const bool deterministic,
               std::optional<at::Generator> gen_,
               std::optional<at::Tensor> &rng_state) {

    #ifdef FLASHATTENTION_DISABLE_BACKWARD
        TORCH_CHECK(false, "This flash attention build does not support backward.");
    #endif
    if (is_causal) { window_size_right = 0; }

    // Otherwise the kernel will be launched from cuda:0 device
    at::cuda::CUDAGuard device_guard{q.device()};

    auto [cc_major, cc_minor] = get_compute_capability(get_current_device());
    bool is_sm8x_min = cc_major >= 8;
    TORCH_CHECK(is_sm8x_min, "FlashAttention only supports Ampere GPUs or newer.");

    bool is_dropout = p_dropout > 0.0;
    auto stream = at::cuda::getCurrentCUDAStream().stream();

    auto q_dtype = q.dtype();
    TORCH_CHECK(q_dtype == torch::kFloat16 || q_dtype == torch::kBFloat16,
                "FlashAttention only support fp16 and bf16 data type");
    TORCH_CHECK(k.dtype() == q_dtype, "query and key must have the same dtype");
    TORCH_CHECK(v.dtype() == q_dtype, "query and value must have the same dtype");
    TORCH_CHECK(out.dtype() == q_dtype, "query and out must have the same dtype");
    TORCH_CHECK(dout.dtype() == q_dtype, "query and dout must have the same dtype");
    TORCH_CHECK(cu_seqlens_q.dtype() == torch::kInt32, "cu_seqlens_q must have dtype int32");
    TORCH_CHECK(cu_seqlens_k.dtype() == torch::kInt32, "cu_seqlens_k must have dtype int32");

    CHECK_DEVICE(q); CHECK_DEVICE(k); CHECK_DEVICE(v);
    CHECK_DEVICE(out); CHECK_DEVICE(dout); CHECK_DEVICE(softmax_lse);
    CHECK_DEVICE(cu_seqlens_q); CHECK_DEVICE(cu_seqlens_k);

    TORCH_CHECK(q.stride(-1) == 1, "Input tensor must have contiguous last dimension");
    TORCH_CHECK(k.stride(-1) == 1, "Input tensor must have contiguous last dimension");
    TORCH_CHECK(v.stride(-1) == 1, "Input tensor must have contiguous last dimension");
    TORCH_CHECK(out.stride(-1) == 1, "out tensor must have contiguous last dimension");
    TORCH_CHECK(dout.stride(-1) == 1, "dout tensor must have contiguous last dimension");
    CHECK_CONTIGUOUS(cu_seqlens_q);
    CHECK_CONTIGUOUS(cu_seqlens_k);

    const auto sizes = q.sizes();

    const int total_q = sizes[0];
    const int batch_size = cu_seqlens_q.numel() - 1;
    const int num_heads = sizes[1];
    const int head_size = sizes[2];
    const int total_k = k.size(0);
    const int num_heads_k = k.size(1);
    TORCH_CHECK(batch_size > 0, "batch size must be positive");
    TORCH_CHECK(head_size % 8 == 0, "head_size should be a multiple of 8");
    TORCH_CHECK(head_size <= 256, "FlashAttention backward only supports head dimension at most 256");
    TORCH_CHECK(num_heads % num_heads_k == 0, "Number of heads in key/value must divide number of heads in query");
    if (softcap > 0.f) { TORCH_CHECK(p_dropout == 0.f, "Softcapping does not support dropout for now"); }

    auto round_multiple = [](int x, int m) { return (x + m - 1) / m * m; };
    const int head_size_rounded = round_multiple(head_size, head_size <= 128 ? 32 : 64);
    const int seqlen_q_rounded = round_multiple(max_seqlen_q, 128);
    const int seqlen_k_rounded = round_multiple(max_seqlen_k, 128);

    if (window_size_left >= max_seqlen_k) { window_size_left = -1; }
    if (window_size_right >= max_seqlen_k) { window_size_right = -1; }

    CHECK_SHAPE(q, total_q, num_heads, head_size);
    CHECK_SHAPE(k, total_k, num_heads_k, head_size);
    CHECK_SHAPE(v, total_k, num_heads_k, head_size);
    CHECK_SHAPE(out, total_q, num_heads, head_size);
    CHECK_SHAPE(dout, total_q, num_heads, head_size);
    CHECK_SHAPE(cu_seqlens_q, batch_size + 1);
    CHECK_SHAPE(cu_seqlens_k, batch_size + 1);

    at::Tensor dq, dk, dv;
    if (dq_.has_value()) {
        dq = dq_.value();
        TORCH_CHECK(dq.dtype() == q_dtype, "dq must have the same dtype as q");
        CHECK_DEVICE(dq);
        TORCH_CHECK(dq.stride(-1) == 1, "dq must have contiguous last dimension");
        CHECK_SHAPE(dq, total_q, num_heads, head_size);
    } else {
        dq = torch::empty_like(q);
    }
    if (dk_.has_value()) {
        dk = dk_.value();
        TORCH_CHECK(dk.dtype() == q_dtype, "dk must have the same dtype as q");
        CHECK_DEVICE(dk);
        TORCH_CHECK(dk.stride(-1) == 1, "dk must have contiguous last dimension");
        CHECK_SHAPE(dk, total_k, num_heads_k, head_size);
    } else {
        dk = torch::empty_like(k);
    }
    if (dv_.has_value()) {
        dv = dv_.value();
        TORCH_CHECK(dv.dtype() == q_dtype, "dv must have the same dtype as q");
        CHECK_DEVICE(dv);
        TORCH_CHECK(dv.stride(-1) == 1, "dv must have contiguous last dimension");
        CHECK_SHAPE(dv, total_k, num_heads_k, head_size);
    } else {
        dv = torch::empty_like(v);
    }

    // bool loop = max_seqlen_k > blocksize_c;
    // TODO: change later, for now set to true for simplicity
    bool loop = true;

    auto opts = q.options();
    auto softmax_d = torch::empty({num_heads, total_q + 128 * batch_size}, opts.dtype(at::kFloat));
    at::Tensor dq_accum;
    if (loop) {
        // We don't want to allocate dq_accum of size (batch, seqlen_q_rounded, num_heads, head_size_rounded)
        // because that would be too large if there is a very long sequence and the rest of the sequences are short.
        // Instead, we allocate dq_accum of size (total_q + 128 * batch, num_heads, head_size_rounded).
        // Note that 128 is the max block size on the seqlen_q dimension.
        // For dQ, the i-th sequence is stored in indices from cu_seqlens[i] + 128 * i to
        // cu_seqlens[i + 1] * 128 * i - 1. This ensures that the i-th sequence and (i + 1)-th sequence will
        // be at least 128 apart. It's ok for us to do atomicAdds up to 128 rows beyond what we're normally
        // allowed to do. So we won't have to do any bound checking, and performance should stay the same.
        // Same holds for softmax_d, since LSE is stored in unpadded format.
        if (!deterministic) {
            dq_accum = torch::empty({total_q + 128 * batch_size, num_heads, head_size_rounded}, opts.dtype(at::kFloat));
        } else {
            const int nsplits = (get_num_sm(get_current_device()) + batch_size * num_heads - 1) / (batch_size * num_heads);
            dq_accum = torch::zeros({nsplits, total_q + 128 * batch_size, num_heads, head_size_rounded}, opts.dtype(at::kFloat));
        }
    }

    at::Tensor dk_expanded, dv_expanded;
    if (num_heads_k != num_heads) {  // MQA / GQA
        dk_expanded = torch::empty({total_k, num_heads, head_size}, opts);
        dv_expanded = torch::empty({total_k, num_heads, head_size}, opts);
    } else {
        dk_expanded = dk;
        dv_expanded = dv;
    }

    if( zero_tensors ) {
        dq.zero_();
        dk_expanded.zero_();
        dv_expanded.zero_();
        softmax_d.zero_();
    }

    Flash_bwd_params params;

    set_params_dgrad(params,
                     batch_size,
                     max_seqlen_q, max_seqlen_k,
                     seqlen_q_rounded, seqlen_k_rounded,
                     num_heads, num_heads_k,
                     head_size, head_size_rounded,
                     q, k, v, out,
                     dout, dq, dk_expanded, dv_expanded,
                     cu_seqlens_q.data_ptr(),
                     cu_seqlens_k.data_ptr(),
                     loop ? dq_accum.data_ptr() : nullptr,
                     nullptr,
                     nullptr,
                     softmax_lse.data_ptr(),
                     softmax_d.data_ptr(),
                     p_dropout,
                     softmax_scale,
                     window_size_left,
                     window_size_right,
                     softcap,
                     deterministic,
                     /*unpadded_lse*/true);
    params.dq_accum_split_stride = !deterministic ? 0 : dq_accum.stride(0);
    params.total_q = total_q;

    auto launch = &run_mha_bwd;

    auto gen = at::get_generator_or_default<at::CUDAGeneratorImpl>(
        gen_, at::cuda::detail::getDefaultCUDAGenerator());

    // We use a custom RNG that increases the offset by batch_size * nheads * 32.
    int64_t counter_offset = params.b * params.h * 32;

    if ( rng_state.has_value() ) {
        params.rng_state = reinterpret_cast<uint64_t*>(rng_state.value().data_ptr());
    } else if( is_dropout ) {
        // See Note [Acquire lock when using random generators]
        std::lock_guard<std::mutex> lock(gen->mutex_);
        params.philox_args = gen->philox_cuda_state(counter_offset);
        auto seeds = at::cuda::philox::unpack(params.philox_args);
        params.rng_state[0] = std::get<0>(seeds);
        params.rng_state[1] = std::get<1>(seeds);
    }

    set_params_alibi(params, alibi_slopes_, batch_size, num_heads);

    if (max_seqlen_q > 0) {
        launch(params, stream);
    } else {
        // If seqlen_q == 0, then we have an empty tensor. We need to set the output to 0.
        dk_expanded.zero_();
        dv_expanded.zero_();
        softmax_d.zero_();
    }

    // For MQA/GQA we need to sum dK and dV across the groups
    if (num_heads_k != num_heads) {
        at::sum_out(dk, at::reshape(dk_expanded, {total_k, num_heads_k, num_heads / num_heads_k, head_size}), {2});
        at::sum_out(dv, at::reshape(dv_expanded, {total_k, num_heads_k, num_heads / num_heads_k, head_size}), {2});
    }

    return { dq, dk, dv, softmax_d };
}


// need_update_mask: bool tensor, shape [N] or [N,1] or broadcastable to attn dims
at::Tensor update_attn(
    const at::Tensor& logsumexp,
    const at::Tensor& self_logsumexp,
    const at::Tensor& need_update_mask,  // bool
    at::Tensor& attn_output_past,        // [N,H]
    const at::Tensor& dirty_mask,        // [N,H] bool
    const at::Tensor& o                  // [N,H]
) {

    auto m = torch::maximum(self_logsumexp, logsumexp);

    auto exp_past = (self_logsumexp - m).exp();
    auto exp_cur  = (logsumexp - m).exp();
    auto denom    = exp_past + exp_cur;

    attn_output_past.masked_fill_(dirty_mask, 0);

    auto veri_attn_output =
        (attn_output_past * exp_past +
         o * exp_cur) / denom;

    auto updated_o =
        torch::where(need_update_mask,   // condition
                     o,                   // true: keep original o
                     veri_attn_output);    // false: use computed value

    return updated_o;
}

// ============================================================================
//  update_attn_fused - C++ Binding for fused CUDA kernel
// ============================================================================

// Forward declaration (host wrapper in .cu)
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
);

template<typename T>
void launch_update_attn_fused_cur_hist_kernel(
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
);

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
);

template<typename T>
void launch_update_attn_fused_cur_hist_kernel_fp32_cache(
    const float* softmax_lse,
    const float* self_lse,
    const float* out_hist,
	    const float* softmax_lse_hist,
	    const float* attn_past,
	    const float* o,
	    const T* out_full_or_cur,
	    const bool* need_mask,
	    const bool* store_mask,
	    const bool* dirty_mask,
	    T* out,
	    float* full_lse_out,
	    float* attn_out_tensor,
	    float* logsumexp_tensor,
	    int B, int H, int Q, int D
);

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
);

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
);


// One layer of wrapper for AT_DISPATCH
// One layer of wrapper for AT_DISPATCH
template <typename scalar_t>
void launch_fused(
    const at::Tensor& softmax_lse,
    const at::Tensor& self_lse,
    const at::Tensor& out_hist,
    const at::Tensor& softmax_lse_hist,
    const at::Tensor& attn_past,
    const at::Tensor& o,
    const at::Tensor& need_mask,
	    const at::Tensor& store_mask,
	    const at::Tensor& dirty_mask,
	    at::Tensor& out,
	    at::Tensor& attn_out_tensor,
	    at::Tensor& logsumexp_tensor
){
    // map PyTorch dtype → CUDA kernel dtype
    using kernel_t =
        std::conditional_t<std::is_same_v<scalar_t, at::Half>,
            half,
        std::conditional_t<std::is_same_v<scalar_t, at::BFloat16>,
            __nv_bfloat16,
            float>>;

    int B = out.size(0);
    int Q = out.size(1);
    int H = out.size(2);
    int D = out.size(3);

    launch_update_attn_fused_kernel<kernel_t>(
        softmax_lse.data_ptr<float>(),
        self_lse.data_ptr<float>(),
        reinterpret_cast<const kernel_t*>(out_hist.data_ptr<scalar_t>()),
        softmax_lse_hist.data_ptr<float>(),
        reinterpret_cast<const kernel_t*>(attn_out_tensor.data_ptr<scalar_t>()),
        reinterpret_cast<const kernel_t*>(o.data_ptr<scalar_t>()),
        need_mask.data_ptr<bool>(),
        store_mask.data_ptr<bool>(),
        dirty_mask.data_ptr<bool>(),
        reinterpret_cast<kernel_t*>(out.data_ptr<scalar_t>()),
        reinterpret_cast<kernel_t*>(attn_out_tensor.data_ptr<scalar_t>()),
        logsumexp_tensor.data_ptr<float>(),
        B, H, Q, D
    );

}

template <typename scalar_t>
void launch_fused_cur_hist(
    const at::Tensor& softmax_lse,
    const at::Tensor& self_lse,
    const at::Tensor& out_hist,
    const at::Tensor& softmax_lse_hist,
    const at::Tensor& attn_past,
    const at::Tensor& o,
    const at::Tensor& need_mask,
	    const at::Tensor& store_mask,
	    const at::Tensor& dirty_mask,
	    at::Tensor& out,
	    at::Tensor& attn_out_tensor,
	    at::Tensor& logsumexp_tensor
){
    using kernel_t =
        std::conditional_t<std::is_same_v<scalar_t, at::Half>,
            half,
        std::conditional_t<std::is_same_v<scalar_t, at::BFloat16>,
            __nv_bfloat16,
            float>>;

    int B = out.size(0);
    int Q = out.size(1);
    int H = out.size(2);
    int D = out.size(3);

    launch_update_attn_fused_cur_hist_kernel<kernel_t>(
        softmax_lse.data_ptr<float>(),
        self_lse.data_ptr<float>(),
        reinterpret_cast<const kernel_t*>(out_hist.data_ptr<scalar_t>()),
        softmax_lse_hist.data_ptr<float>(),
        reinterpret_cast<const kernel_t*>(attn_out_tensor.data_ptr<scalar_t>()),
        reinterpret_cast<const kernel_t*>(o.data_ptr<scalar_t>()),
        need_mask.data_ptr<bool>(),
        store_mask.data_ptr<bool>(),
        dirty_mask.data_ptr<bool>(),
        reinterpret_cast<kernel_t*>(out.data_ptr<scalar_t>()),
        reinterpret_cast<kernel_t*>(attn_out_tensor.data_ptr<scalar_t>()),
        logsumexp_tensor.data_ptr<float>(),
        B, H, Q, D
    );
}

template <typename scalar_t>
void launch_fused_fp32_cache(
    const at::Tensor& softmax_lse,
    const at::Tensor& self_lse,
    const at::Tensor& out_hist,
    const at::Tensor& softmax_lse_hist,
    const at::Tensor& attn_past,
    const at::Tensor& o,
    const at::Tensor& need_mask,
    const at::Tensor& store_mask,
	    const at::Tensor& dirty_mask,
	    at::Tensor& out,
	    at::Tensor& attn_out_tensor,
	    at::Tensor& logsumexp_tensor
){
    using kernel_t =
        std::conditional_t<std::is_same_v<scalar_t, at::Half>,
            half,
        std::conditional_t<std::is_same_v<scalar_t, at::BFloat16>,
            __nv_bfloat16,
            float>>;

    int B = out.size(0);
    int Q = out.size(1);
    int H = out.size(2);
    int D = out.size(3);

    launch_update_attn_fused_kernel_fp32_cache<kernel_t>(
        softmax_lse.data_ptr<float>(),
        self_lse.data_ptr<float>(),
        out_hist.data_ptr<float>(),
        softmax_lse_hist.data_ptr<float>(),
        attn_past.data_ptr<float>(),
        reinterpret_cast<const kernel_t*>(o.data_ptr<scalar_t>()),
        need_mask.data_ptr<bool>(),
	        store_mask.data_ptr<bool>(),
	        dirty_mask.data_ptr<bool>(),
	        reinterpret_cast<kernel_t*>(out.data_ptr<scalar_t>()),
	        attn_out_tensor.data_ptr<float>(),
	        logsumexp_tensor.data_ptr<float>(),
        B, H, Q, D
    );
}

template <typename scalar_t>
void launch_fused_cur_hist_fp32_cache(
    const at::Tensor& softmax_lse,
    const at::Tensor& self_lse,
    const at::Tensor& out_hist,
    const at::Tensor& softmax_lse_hist,
    const at::Tensor& attn_past,
    const at::Tensor& out_cur,
    const at::Tensor& out_template,
    const at::Tensor& need_mask,
    const at::Tensor& store_mask,
	    const at::Tensor& dirty_mask,
	    at::Tensor& out,
	    at::Tensor& full_lse_out,
	    at::Tensor& attn_out_tensor,
	    at::Tensor& logsumexp_tensor
){
    using kernel_t =
        std::conditional_t<std::is_same_v<scalar_t, at::Half>,
            half,
        std::conditional_t<std::is_same_v<scalar_t, at::BFloat16>,
            __nv_bfloat16,
            float>>;

    int B = out_template.size(0);
    int Q = out_template.size(1);
    int H = out_template.size(2);
    int D = out_template.size(3);

    launch_update_attn_fused_cur_hist_kernel_fp32_cache<kernel_t>(
        softmax_lse.data_ptr<float>(),
        self_lse.data_ptr<float>(),
        out_hist.data_ptr<float>(),
	        softmax_lse_hist.data_ptr<float>(),
	        attn_past.data_ptr<float>(),
	        out_cur.data_ptr<float>(),
	        reinterpret_cast<const kernel_t*>(out_template.data_ptr<scalar_t>()),
	        need_mask.data_ptr<bool>(),
        store_mask.data_ptr<bool>(),
	        dirty_mask.data_ptr<bool>(),
	        reinterpret_cast<kernel_t*>(out.data_ptr<scalar_t>()),
	        full_lse_out.data_ptr<float>(),
	        attn_out_tensor.data_ptr<float>(),
	        logsumexp_tensor.data_ptr<float>(),
        B, H, Q, D
		    );
}

template <typename scalar_t>
void launch_fused_cur_hist_fp32_cache_all_update(
    at::Tensor& softmax_lse,
    const at::Tensor& self_lse,
    const at::Tensor& out_hist,
    const at::Tensor& softmax_lse_hist,
    at::Tensor& attn_past,
    const at::Tensor& out_cur,
    at::Tensor& out,
    const at::Tensor& store_mask,
    at::Tensor& logsumexp_tensor
){
    using kernel_t =
        std::conditional_t<std::is_same_v<scalar_t, at::Half>,
            half,
        std::conditional_t<std::is_same_v<scalar_t, at::BFloat16>,
            __nv_bfloat16,
            float>>;

    int B = out.size(0);
    int Q = out.size(1);
    int H = out.size(2);
    int D = out.size(3);

    launch_update_attn_fused_cur_hist_kernel_fp32_cache<kernel_t>(
        softmax_lse.data_ptr<float>(),
        self_lse.data_ptr<float>(),
        out_hist.data_ptr<float>(),
        softmax_lse_hist.data_ptr<float>(),
        attn_past.data_ptr<float>(),
        out_cur.data_ptr<float>(),
        reinterpret_cast<const kernel_t*>(out.data_ptr<scalar_t>()),
        nullptr,
        store_mask.data_ptr<bool>(),
        nullptr,
        reinterpret_cast<kernel_t*>(out.data_ptr<scalar_t>()),
        softmax_lse.data_ptr<float>(),
        attn_past.data_ptr<float>(),
        logsumexp_tensor.data_ptr<float>(),
        B, H, Q, D
    );
}

template <typename scalar_t>
void launch_all_update_store_history_fp32(
    at::Tensor& softmax_lse,
    const at::Tensor& out_hist,
    const at::Tensor& softmax_lse_hist,
    const at::Tensor& out_cur,
    const at::Tensor& store_mask,
    at::Tensor& out,
    at::Tensor& attn_out_tensor,
    at::Tensor& logsumexp_tensor
){
    using kernel_t =
        std::conditional_t<std::is_same_v<scalar_t, at::Half>,
            half,
        std::conditional_t<std::is_same_v<scalar_t, at::BFloat16>,
            __nv_bfloat16,
            float>>;

    int B = out.size(0);
    int Q = out.size(1);
    int H = out.size(2);
    int D = out.size(3);

    launch_all_update_store_history_fp32_kernel<kernel_t>(
        softmax_lse.data_ptr<float>(),
        out_hist.data_ptr<float>(),
        softmax_lse_hist.data_ptr<float>(),
        out_cur.data_ptr<float>(),
        store_mask.data_ptr<bool>(),
        reinterpret_cast<kernel_t*>(out.data_ptr<scalar_t>()),
        attn_out_tensor.data_ptr<float>(),
        logsumexp_tensor.data_ptr<float>(),
        B, H, Q, D
    );
}



// ===========================
// main API
// ===========================
at::Tensor update_attn_fused(
    at::Tensor softmax_lse,
    at::Tensor logsumexp_tensor,
    at::Tensor out_hist,
    at::Tensor softmax_lse_hist,
    at::Tensor attn_out_tensor,
    at::Tensor o,
    at::Tensor need_mask,
    at::Tensor store_mask,
    at::Tensor dirty_mask
){
    CHECK_DEVICE(softmax_lse);
    CHECK_DEVICE(softmax_lse_hist);
    CHECK_DEVICE(o);
    CHECK_CONTIGUOUS(softmax_lse);
    CHECK_CONTIGUOUS(softmax_lse_hist);
    CHECK_CONTIGUOUS(o);
    TORCH_CHECK(logsumexp_tensor.scalar_type() == at::kFloat, "logsumexp must be float32");
    TORCH_CHECK(softmax_lse.scalar_type() == at::kFloat, "softmax_lse must be float32");
    TORCH_CHECK(softmax_lse_hist.scalar_type() == at::kFloat, "softmax_lse_hist must be float32");
    const bool fp32_cache = attn_out_tensor.scalar_type() == at::kFloat && o.scalar_type() != at::kFloat;
    TORCH_CHECK((fp32_cache && out_hist.scalar_type() == at::kFloat) ||
                (!fp32_cache && out_hist.scalar_type() == o.scalar_type()),
                "out_hist must match output dtype, or be float32 when attn_output_past is float32");
    TORCH_CHECK(attn_out_tensor.scalar_type() == o.scalar_type() || attn_out_tensor.scalar_type() == at::kFloat,
                "attn_output_past must match output dtype or be float32");

    at::Tensor out = torch::empty_like(o);

    // dtype: float, half, bfloat16
    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half,
        at::ScalarType::BFloat16,
        o.scalar_type(),
        "update_attn_fused",
        [&](){
            if (fp32_cache) {
                launch_fused_fp32_cache<scalar_t>(
                    softmax_lse,
                    logsumexp_tensor,
                    out_hist,
                    softmax_lse_hist,
                    attn_out_tensor,
                    o,
                    need_mask,
                    store_mask,
                    dirty_mask,
                    out,
                    attn_out_tensor,
                    logsumexp_tensor
                );
            } else {
                launch_fused<scalar_t>(
                    softmax_lse,
                    logsumexp_tensor,
                    out_hist,
                    softmax_lse_hist,
                    attn_out_tensor,
                    o,
                    need_mask,
                    store_mask,
                    dirty_mask,
                    out,
                    attn_out_tensor,
                    logsumexp_tensor
                );
            }
        }
    );

    return out;
}

std::vector<at::Tensor> update_attn_fused_cur_hist_fp32_outcur(
    at::Tensor softmax_lse,
    at::Tensor logsumexp_tensor,
    at::Tensor out_hist,
    at::Tensor softmax_lse_hist,
    at::Tensor attn_out_tensor,
    at::Tensor out_cur,
    at::Tensor out_template,
    at::Tensor need_mask,
    at::Tensor store_mask,
    at::Tensor dirty_mask
){
    CHECK_DEVICE(softmax_lse);
    CHECK_DEVICE(softmax_lse_hist);
    CHECK_DEVICE(out_cur);
    CHECK_DEVICE(out_template);
    CHECK_CONTIGUOUS(softmax_lse);
    CHECK_CONTIGUOUS(softmax_lse_hist);
    CHECK_CONTIGUOUS(out_cur);
    CHECK_CONTIGUOUS(out_template);
    TORCH_CHECK(logsumexp_tensor.scalar_type() == at::kFloat, "logsumexp must be float32");
    TORCH_CHECK(softmax_lse.scalar_type() == at::kFloat, "softmax_lse must be float32");
    TORCH_CHECK(softmax_lse_hist.scalar_type() == at::kFloat, "softmax_lse_hist must be float32");
    TORCH_CHECK(out_hist.scalar_type() == at::kFloat, "out_hist must be float32");
    TORCH_CHECK(attn_out_tensor.scalar_type() == at::kFloat, "attn_output_past must be float32");
    TORCH_CHECK(out_cur.scalar_type() == at::kFloat, "out_cur must be float32");

    at::Tensor out = torch::empty_like(out_template);
    at::Tensor full_lse = torch::empty_like(softmax_lse);

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half,
        at::ScalarType::BFloat16,
        out_template.scalar_type(),
        "update_attn_fused_cur_hist_fp32_outcur",
        [&](){
            launch_fused_cur_hist_fp32_cache<scalar_t>(
                softmax_lse,
                logsumexp_tensor,
                out_hist,
                softmax_lse_hist,
                attn_out_tensor,
                out_cur,
                out_template,
                need_mask,
                store_mask,
	                dirty_mask,
	                out,
	                full_lse,
	                attn_out_tensor,
	                logsumexp_tensor
	            );
        }
    );

    return {out, full_lse};
}


std::vector<at::Tensor>
mha_fwd_kvcache(at::Tensor &q,                 // batch_size x seqlen_q x num_heads x head_size
                const at::Tensor &kcache,            // batch_size_c x seqlen_k x num_heads_k x head_size or num_blocks x page_block_size x num_heads_k x head_size if there's a block_table.
                const at::Tensor &vcache,            // batch_size_c x seqlen_k x num_heads_k x head_size or num_blocks x page_block_size x num_heads_k x head_size if there's a block_table.
                std::optional<const at::Tensor> &k_, // batch_size x seqlen_knew x num_heads_k x head_size
                std::optional<const at::Tensor> &v_, // batch_size x seqlen_knew x num_heads_k x head_size
                std::optional<const at::Tensor> &seqlens_k_, // batch_size
                std::optional<const at::Tensor> &rotary_cos_, // seqlen_ro x (rotary_dim / 2)
                std::optional<const at::Tensor> &rotary_sin_, // seqlen_ro x (rotary_dim / 2)
                std::optional<const at::Tensor> &cache_batch_idx_, // indices to index into the KV cache
                std::optional<const at::Tensor> &leftpad_k_, // batch_size
                std::optional<at::Tensor> &block_table_, // batch_size x max_num_blocks_per_seq
                std::optional<at::Tensor> &alibi_slopes_, // num_heads or batch_size x num_heads
                std::optional<at::Tensor> &out_,             // batch_size x seqlen_q x num_heads x head_size
                const float softmax_scale,
                bool is_causal,
                int window_size_left,
                int window_size_right,
                const float softcap,
                bool is_rotary_interleaved,   // if true, rotary combines indices 0 & 1, else indices 0 & rotary_dim / 2
                int num_splits,
                std::optional<const at::Tensor> &attn_output_past, // indices to index into the KV cache
                std::optional<const at::Tensor> &logsumexp, // indices to index into the KV cache
                std::optional<const at::Tensor> &dirty_mask, // indices to index into the KV cache
                std::optional<const at::Tensor> &need_update_kvcache_mask, // batch_size
                std::optional<const at::Tensor> &need_store_history_mask, // batch_size
                std::optional<const at::Tensor> &delta_seqlens_k_,
                int block_size = 0
                ) {

    // Otherwise the kernel will be launched from cuda:0 device
    at::cuda::CUDAGuard device_guard{q.device()};

    auto [cc_major, cc_minor] = get_compute_capability(get_current_device());
    bool is_sm8x_min = cc_major >= 8;
    TORCH_CHECK(is_sm8x_min, "FlashAttention only supports Ampere GPUs or newer.");

    auto q_dtype = q.dtype();
    TORCH_CHECK(q_dtype == torch::kFloat16 || q_dtype == torch::kBFloat16,
                "FlashAttention only support fp16 and bf16 data type");
    TORCH_CHECK(kcache.dtype() == q_dtype, "query and key must have the same dtype");
    TORCH_CHECK(vcache.dtype() == q_dtype, "query and value must have the same dtype");

    CHECK_DEVICE(q); CHECK_DEVICE(kcache); CHECK_DEVICE(vcache);

    TORCH_CHECK(q.stride(-1) == 1, "Input tensor must have contiguous last dimension");
    TORCH_CHECK(kcache.stride(-1) == 1, "Input tensor must have contiguous last dimension");
    TORCH_CHECK(vcache.stride(-1) == 1, "Input tensor must have contiguous last dimension");

    at::Tensor block_table;
    const bool paged_KV = block_table_.has_value();
    if (paged_KV) {
        TORCH_CHECK(!cache_batch_idx_.has_value(), "Paged KVcache does not support cache_batch_idx");
        block_table = block_table_.value();
        CHECK_DEVICE(block_table);
        TORCH_CHECK(block_table.dtype() == torch::kInt32, "block_table must have dtype torch.int32");
        TORCH_CHECK(block_table.stride(-1) == 1, "block_table must have contiguous last dimension");
    }

    const auto sizes = q.sizes();

    const int batch_size = sizes[0];
    int seqlen_q = sizes[1];
    int num_heads = sizes[2];
    const int head_size_og = sizes[3];

    const int max_num_blocks_per_seq = !paged_KV ? 0 : block_table.size(1);
    const int num_blocks = !paged_KV ? 0 : kcache.size(0);
    const int page_block_size = !paged_KV ? 1 : kcache.size(1);
    TORCH_CHECK(!paged_KV || page_block_size % 256 == 0, "Paged KV cache block size must be divisible by 256");
    const int seqlen_k = !paged_KV ? kcache.size(1) : max_num_blocks_per_seq * page_block_size;
    const int num_heads_k = kcache.size(2);
    const int batch_size_c = !paged_KV ? kcache.size(0) : batch_size;
    TORCH_CHECK(batch_size > 0, "batch size must be positive");
    TORCH_CHECK(head_size_og <= 256, "FlashAttention forward only supports head dimension at most 256");
    TORCH_CHECK(num_heads % num_heads_k == 0, "Number of heads in key/value must divide number of heads in query");

    // causal=true is the same as causal=false in this case
    if (seqlen_q == 1 && !alibi_slopes_.has_value()) { is_causal = false; }
    if (is_causal) { window_size_right = 0; }

    // Faster to transpose q from (b, 1, (nheads_kv ngroups), d) to (b, ngroups, nheads_kv, d) in this case
    // H/t Daniel Haziza
    const int seqlenq_ngroups_swapped = seqlen_q == 1 && num_heads > num_heads_k && window_size_left < 0 && window_size_right < 0 && head_size_og % 8 == 0 && !alibi_slopes_.has_value();
    if (seqlenq_ngroups_swapped) {
        const int ngroups = num_heads / num_heads_k;
        q = q.reshape({batch_size, num_heads_k, ngroups, head_size_og}).transpose(1, 2);
        seqlen_q = ngroups;
        num_heads = num_heads_k;
    }

    if (window_size_left >= seqlen_k) { window_size_left = -1; }
    if (window_size_right >= seqlen_k) { window_size_right = -1; }

    CHECK_SHAPE(q, batch_size, seqlen_q, num_heads, head_size_og);
    if (!paged_KV) {
        CHECK_SHAPE(kcache, batch_size_c, seqlen_k, num_heads_k, head_size_og);
        CHECK_SHAPE(vcache, batch_size_c, seqlen_k, num_heads_k, head_size_og);
    } else {
        CHECK_SHAPE(kcache, num_blocks, page_block_size, num_heads_k, head_size_og);
        CHECK_SHAPE(vcache, num_blocks, page_block_size, num_heads_k, head_size_og);
        CHECK_SHAPE(block_table, batch_size, max_num_blocks_per_seq);
    }

    at::Tensor q_padded, kcache_padded, vcache_padded;
    if (head_size_og % 8 != 0) {
        q_padded = torch::nn::functional::pad(q, torch::nn::functional::PadFuncOptions({0, 8 - head_size_og % 8}));
        kcache_padded = torch::nn::functional::pad(kcache, torch::nn::functional::PadFuncOptions({0, 8 - head_size_og % 8}));
        vcache_padded = torch::nn::functional::pad(vcache, torch::nn::functional::PadFuncOptions({0, 8 - head_size_og % 8}));
    } else {
        q_padded = q;
        kcache_padded = kcache;
        vcache_padded = vcache;
    }

    at::Tensor out;
    if (out_.has_value()) {
        printf("out has value\n");
        out = out_.value();
        TORCH_CHECK(out.dtype() == q_dtype, "Output must have the same dtype as inputs");
        CHECK_DEVICE(out);
        TORCH_CHECK(out.stride(-1) == 1, "Output tensor must have contiguous last dimension");
        CHECK_SHAPE(out, batch_size, seqlen_q, num_heads, head_size_og);
        if (head_size_og % 8 != 0) { out = torch::empty_like(q_padded); }
    } else {
        out = torch::empty_like(q_padded);
    }
    at::Tensor attn_output_past_;
    at::Tensor logsumexp_;
    at::Tensor need_update_mask;
    at::Tensor need_store_mask;
    at::Tensor dirty_mask_;


    auto round_multiple = [](int x, int m) { return (x + m - 1) / m * m; };
    const int head_size = round_multiple(head_size_og, 8);
    const int head_size_rounded = round_multiple(head_size, head_size <= 128 ? 32 : 64);
    const int seqlen_q_rounded = round_multiple(seqlen_q, 128);
    const int seqlen_k_rounded = round_multiple(seqlen_k, 128);

    auto opts = q.options();

    auto softmax_lse = torch::empty({batch_size, num_heads, seqlen_q}, opts.dtype(at::kFloat));

    const bool use_fp32_history_output =
        attn_output_past.has_value() && attn_output_past.value().scalar_type() == at::kFloat;
    at::Tensor out_hist;
    at::Tensor out_cur_fp32;
    auto softmax_lse_hist = torch::empty({batch_size, num_heads, seqlen_q}, opts.dtype(at::kFloat));
    logsumexp_ = torch::empty({batch_size, num_heads, seqlen_q}, opts.dtype(at::kFloat));
    out_hist = use_fp32_history_output
        ? torch::empty(q_padded.sizes(), opts.dtype(at::kFloat))
        : torch::empty_like(q_padded);
    out_cur_fp32 = use_fp32_history_output
        ? torch::empty(q_padded.sizes(), opts.dtype(at::kFloat))
        : torch::empty({0}, opts.dtype(at::kFloat));
    attn_output_past_ = torch::empty_like(q_padded);
    
    Flash_fwd_params params;
    set_params_fprop(params,
                     batch_size, 
                     seqlen_q, seqlen_k,
                     seqlen_q_rounded, seqlen_k_rounded,
                     num_heads, num_heads_k,
                     head_size, head_size_rounded,
                     q_padded, kcache_padded, vcache_padded, out,
                     /*cu_seqlens_q_d=*/nullptr,
                     /*cu_seqlens_k_d=*/nullptr,
                     /*seqused_k=*/nullptr,
                     /*p_ptr=*/nullptr,
                     softmax_lse.data_ptr(),
                     /*p_dropout=*/0.f,
                     softmax_scale,
                     window_size_left,
                     window_size_right,
                     softcap
                     );
    params.block_size=block_size;
    params.need_store_history_mask = nullptr;
    params.need_update_mask = nullptr;
    params.lse_hist_ptr=softmax_lse_hist.data_ptr();
    params.o_hist_ptr=out_hist.data_ptr();
    params.self_o_hist_ptr = use_fp32_history_output ? out_cur_fp32.data_ptr() : nullptr;
    params.o_hist_fp32 = use_fp32_history_output;
    params.update_past=false;
    params.merge_full_in_kernel=false;
        if (attn_output_past.has_value()){
        attn_output_past_=attn_output_past.value();
        logsumexp_=logsumexp.value();
        TORCH_CHECK(logsumexp_.scalar_type() == at::kFloat, "logsumexp cache must be float32");
        CHECK_CONTIGUOUS(logsumexp_);
        if (need_update_kvcache_mask.has_value()) {
            need_update_mask=need_update_kvcache_mask.value();
            params.need_update_mask = need_update_mask.data_ptr();
        }
        need_store_mask=need_store_history_mask.has_value() ? need_store_history_mask.value() : need_update_mask;
        if (dirty_mask.has_value()) {
            dirty_mask_=dirty_mask.value();
        }
        params.need_store_history_mask = need_store_mask.data_ptr();
        params.merge_full_in_kernel = false;
    }

    at::Tensor k, v, k_padded, v_padded;
    if (k_.has_value()) {
        TORCH_CHECK(v_.has_value(), "If key is supplied, value must also be passed in");
        TORCH_CHECK(seqlens_k_.has_value(), "If key is supplied, seqlens_k must also be passed in");
        TORCH_CHECK(seqlen_q <= seqlen_k, "If key is supplied, it must have seqlen <= the seqlen of the KV cache");
        k = k_.value();
        v = v_.value();
        TORCH_CHECK(k.dtype() == q_dtype, "Key must have the same dtype as query");
        TORCH_CHECK(v.dtype() == q_dtype, "Value must have the same dtype as query");
        CHECK_DEVICE(k); CHECK_DEVICE(v);
        TORCH_CHECK(k.stride(-1) == 1, "Key tensor must have contiguous last dimension");
        TORCH_CHECK(v.stride(-1) == 1, "Value tensor must have contiguous last dimension");
        int seqlen_knew = k.size(1);
        CHECK_SHAPE(k, batch_size, seqlen_knew, num_heads_k, head_size_og);
        CHECK_SHAPE(v, batch_size, seqlen_knew, num_heads_k, head_size_og);
        if (head_size_og % 8 != 0) {
            k_padded = torch::nn::functional::pad(k, torch::nn::functional::PadFuncOptions({0, 8 - head_size_og % 8}));
            v_padded = torch::nn::functional::pad(v, torch::nn::functional::PadFuncOptions({0, 8 - head_size_og % 8}));
        } else {
            k_padded = k;
            v_padded = v;
        }
        params.seqlen_knew = seqlen_knew;
        params.knew_ptr = k_padded.data_ptr();
        params.vnew_ptr = v_padded.data_ptr();
        // All stride are in elements, not bytes.
        params.knew_batch_stride = k_padded.stride(0);
        params.vnew_batch_stride = v_padded.stride(0);
        params.knew_row_stride = k_padded.stride(-3);
        params.vnew_row_stride = v_padded.stride(-3);
        params.knew_head_stride = k_padded.stride(-2);
        params.vnew_head_stride = v_padded.stride(-2);
    }

    if (seqlens_k_.has_value()) {
        auto seqlens_k = seqlens_k_.value();
        TORCH_CHECK(seqlens_k.dtype() == torch::kInt32, "seqlens_k must have dtype int32");
        CHECK_DEVICE(seqlens_k);
        CHECK_CONTIGUOUS(seqlens_k);
        CHECK_SHAPE(seqlens_k, batch_size);
        params.cu_seqlens_k = static_cast<int *>(seqlens_k.data_ptr());
    }
    if (delta_seqlens_k_.has_value()) {
        auto delta_seqlens_k = delta_seqlens_k_.value();
        TORCH_CHECK(delta_seqlens_k.dtype() == torch::kInt32, "delta_seqlens_k must have dtype int32");
        CHECK_DEVICE(delta_seqlens_k);
        CHECK_CONTIGUOUS(delta_seqlens_k);
        CHECK_SHAPE(delta_seqlens_k, batch_size);
        params.cu_delta_seqlens_k = static_cast<int *>(delta_seqlens_k.data_ptr());
    }
    params.is_seqlens_k_cumulative = !(seqlens_k_.has_value());
    if (leftpad_k_.has_value()) {
        TORCH_CHECK(!paged_KV, "We don't support Paged KV and leftpad_k running at the same time yet");
        auto leftpad_k = leftpad_k_.value();
        TORCH_CHECK(leftpad_k.dtype() == torch::kInt32, "leftpad_k must have dtype int32");
        CHECK_DEVICE(leftpad_k);
        CHECK_CONTIGUOUS(leftpad_k);
        CHECK_SHAPE(leftpad_k, batch_size);
        params.leftpad_k = static_cast<int *>(leftpad_k.data_ptr());
    }

    if (rotary_cos_.has_value()) {
        TORCH_CHECK(k_.has_value(), "If rotary cos/sin are provided, new key / value to be appended to KV cache must also be provided");
        auto rotary_cos = rotary_cos_.value();
        CHECK_DEVICE(rotary_cos);
        params.rotary_dim = rotary_cos.size(1) * 2;
        TORCH_CHECK(params.rotary_dim <= head_size, "rotary_dim must be <= headdim");
        TORCH_CHECK(params.rotary_dim % 16 == 0, "Only rotary dimensions divisible by 16 are currently supported");
        const int seqlen_ro = rotary_cos.size(0);
        TORCH_CHECK(seqlen_ro >= seqlen_k, "cos/sin seqlen must be at least the seqlen of KV cache");
        CHECK_SHAPE(rotary_cos, seqlen_ro, params.rotary_dim / 2);
        CHECK_CONTIGUOUS(rotary_cos);
        TORCH_CHECK(rotary_cos.scalar_type() == q_dtype, "rotary_cos must have the same dtype as query");

        TORCH_CHECK(rotary_sin_.has_value(), "If rotary cos is provided, rotary sin must also be provided");
        auto rotary_sin = rotary_sin_.value();
        CHECK_DEVICE(rotary_sin);
        CHECK_SHAPE(rotary_sin, seqlen_ro, params.rotary_dim / 2);
        CHECK_CONTIGUOUS(rotary_sin);
        TORCH_CHECK(rotary_sin.scalar_type() == q_dtype, "rotary_cos must have the same dtype as query");
        params.rotary_cos_ptr = rotary_cos.data_ptr();
        params.rotary_sin_ptr = rotary_sin.data_ptr();
        params.is_rotary_interleaved = is_rotary_interleaved;
    } else {
        params.rotary_dim = 0;
    }

    if (cache_batch_idx_.has_value()) {
        auto cache_batch_idx = cache_batch_idx_.value();
        CHECK_DEVICE(cache_batch_idx);
        CHECK_CONTIGUOUS(cache_batch_idx);
        TORCH_CHECK(cache_batch_idx.scalar_type() == torch::kInt32, "cache_batch_idx must have dtype int32");
        params.cache_batch_idx = reinterpret_cast<int *>(cache_batch_idx.data_ptr());
    }

    // Keep references to these tensors to extend their lifetime
    at::Tensor softmax_lse_accum, out_accum, softmax_lse_hist_accum;
    std::tie(softmax_lse_accum, out_accum, softmax_lse_hist_accum) = set_params_splitkv(
        params, batch_size, num_heads, head_size, seqlen_k, seqlen_q,
        head_size_rounded, /*dropout*/ 0.f, num_splits, get_num_sm(get_current_device()), opts);

    if (paged_KV) {
        params.block_table = block_table.data_ptr<int>();
        params.block_table_batch_stride = block_table.stride(0);
    }
    params.page_block_size = page_block_size;


    set_params_alibi(params, alibi_slopes_, batch_size, num_heads);

    auto stream = at::cuda::getCurrentCUDAStream().stream();
    // Only split kernel supports appending to KV cache, or indexing to the cache with cache_batch_idx,
    // or paged KV cache
    run_mha_fwd(params, stream, /*force_split_kernel=*/k_.has_value() || cache_batch_idx_.has_value() || paged_KV);

    if (block_size > 0) {
        TORCH_CHECK(block_size == 4 || block_size == 8, "tiny current kernel only supports block_size 4 or 8");
        TORCH_CHECK(block_size == seqlen_q, "tiny current kernel expects block_size == seqlen_q");
        TORCH_CHECK(use_fp32_history_output, "tiny current kernel mode requires fp32 history cache");
        TORCH_CHECK(attn_output_past.has_value(), "tiny current kernel mode requires attn_output_past");
        TORCH_CHECK(k_.has_value() && seqlens_k_.has_value(), "tiny current kernel mode requires appended k/v and cache_seqlens");
        TORCH_CHECK(head_size <= 128, "tiny current kernel only supports head_dim <= 128");
        TORCH_CHECK(!rotary_cos_.has_value() && !rotary_sin_.has_value(), "tiny current kernel mode does not support flash-internal rotary");
        TORCH_CHECK(!alibi_slopes_.has_value(), "tiny current kernel mode does not support alibi");
        TORCH_CHECK(!seqlenq_ngroups_swapped, "tiny current kernel mode does not support q/ngroups swapped layout");
        TORCH_CHECK(!is_causal && window_size_left < 0 && window_size_right < 0,
                    "tiny current kernel mode only supports full-context non-causal attention");
        AT_DISPATCH_FLOATING_TYPES_AND2(
            at::ScalarType::Half,
            at::ScalarType::BFloat16,
            q_padded.scalar_type(),
            "current_block_fp32_cache",
            [&](){
                using kernel_t =
                    std::conditional_t<std::is_same_v<scalar_t, at::Half>,
                        half,
                    std::conditional_t<std::is_same_v<scalar_t, at::BFloat16>,
                        __nv_bfloat16,
                        float>>;
                launch_current_block_fp32_kernel<kernel_t>(
                    reinterpret_cast<const kernel_t*>(q_padded.data_ptr<scalar_t>()),
                    reinterpret_cast<const kernel_t*>(kcache_padded.data_ptr<scalar_t>()),
                    reinterpret_cast<const kernel_t*>(vcache_padded.data_ptr<scalar_t>()),
                    static_cast<const int*>(params.cu_seqlens_k),
                    static_cast<const int*>(params.cache_batch_idx),
                    paged_KV ? block_table.data_ptr<int>() : nullptr,
                    reinterpret_cast<const bool*>(params.need_store_history_mask),
                    out_cur_fp32.data_ptr<float>(),
                    softmax_lse.data_ptr<float>(),
                    batch_size,
                    seqlen_q,
                    num_heads,
                    num_heads_k,
                    head_size,
                    block_size,
                    params.seqlen_knew,
                    page_block_size,
                    max_num_blocks_per_seq,
                    q_padded.stride(0),
                    q_padded.stride(-3),
                    q_padded.stride(-2),
                    kcache_padded.stride(0),
                    kcache_padded.stride(-3),
                    kcache_padded.stride(-2),
                    vcache_padded.stride(0),
                    vcache_padded.stride(-3),
                    vcache_padded.stride(-2),
                    paged_KV ? block_table.stride(0) : 0,
                    softmax_scale,
                    paged_KV,
                    is_causal
                );
            }
        );
    }

    if (head_size_og % 8 != 0) {
        out = out.index({"...", torch::indexing::Slice(torch::indexing::None, head_size_og)});
        if (out_.has_value()) { out_.value().copy_(out); }
        if (k_.has_value()) {
            // It's expensive to copy the KV cache here for the case where head size not divisible by 8,
            // but we don't expect to get this case in practice. This is just so that the code works for that case.
            kcache.copy_(kcache_padded.index({"...", torch::indexing::Slice(torch::indexing::None, head_size_og)}));
            vcache.copy_(vcache_padded.index({"...", torch::indexing::Slice(torch::indexing::None, head_size_og)}));
        }
    }

    if (seqlenq_ngroups_swapped) {
        out = out.transpose(1, 2).reshape({batch_size, 1, num_heads_k * seqlen_q, head_size_og});
        softmax_lse = softmax_lse.reshape({batch_size, num_heads_k * seqlen_q, 1});
    }
    softmax_lse=softmax_lse.transpose(1, 2).reshape({batch_size, seqlen_q, num_heads, 1}).contiguous();
    if(block_size>0){
        softmax_lse_hist=softmax_lse_hist.transpose(1, 2).reshape({batch_size, seqlen_q, num_heads, 1}).contiguous();
        if (seqlenq_ngroups_swapped) {
            out_hist = out_hist.transpose(1, 2).reshape({batch_size, 1, num_heads_k * seqlen_q, head_size_og});
            softmax_lse_hist = softmax_lse_hist.reshape({batch_size, num_heads_k * seqlen_q, 1}).contiguous();
        }
    }
    softmax_lse = softmax_lse.contiguous();
    // if(block_size>0){
    //     if (attn_output_past.has_value()){
    //         at::Tensor attn_out_tensor =

    //                 attn_output_past.has_value() ? attn_output_past.value() : at::Tensor();
    //         at::Tensor need_update_kvcache_mask_tensor =
    //                 need_update_kvcache_mask.has_value() ? need_update_kvcache_mask.value() : at::Tensor();
    //         at::Tensor dirty_mask_tensor =
    //                 dirty_mask.has_value() ? dirty_mask.value() : at::Tensor();
    //         at::Tensor logsumexp_tensor =
    //                 logsumexp.has_value() ? logsumexp.value() : at::Tensor();

    //         attn_out_tensor = at::where(
    //             need_update_kvcache_mask_tensor.to(at::kBool),
    //             out_hist,
    //             attn_out_tensor
    //         );
    //         logsumexp_tensor = at::where(
    //             need_update_kvcache_mask_tensor.to(at::kBool),
    //             softmax_lse_hist,
    //             logsumexp_tensor
    //         ).to(out.dtype());

    //         out = update_attn(softmax_lse,logsumexp_tensor,need_update_kvcache_mask_tensor,attn_out_tensor,dirty_mask_tensor,out);

    //         return {out, softmax_lse, out_hist, softmax_lse_hist, attn_out_tensor,logsumexp_tensor};
    //     }
    //     return {out, softmax_lse, out_hist, softmax_lse_hist};
    // }
    // else{
    //     return {out, softmax_lse};
    // }
    if (block_size > 0) {
        if (attn_output_past.has_value()) {
            if (!need_update_kvcache_mask.has_value()) {
                if (need_store_history_mask.has_value()) {
                    AT_DISPATCH_FLOATING_TYPES_AND2(
                        at::ScalarType::Half,
                        at::ScalarType::BFloat16,
                        out.scalar_type(),
                        "cur_hist_fp32_cache_all_update",
                        [&](){
                            launch_fused_cur_hist_fp32_cache_all_update<scalar_t>(
                                softmax_lse,
                                logsumexp_,
                                out_hist,
                                softmax_lse_hist,
                                attn_output_past_,
                                out_cur_fp32,
                                out,
                                need_store_mask,
                                logsumexp_
                            );
                        }
                    );
                    return {out, softmax_lse, out_hist, softmax_lse_hist,
                            attn_output_past_, logsumexp_};
                }
                return {out, softmax_lse, out_hist, softmax_lse_hist,
                        out_hist, softmax_lse_hist};
            }
            auto fused_outputs = update_attn_fused_cur_hist_fp32_outcur(
                softmax_lse,
                logsumexp_,
                out_hist,
                softmax_lse_hist,
                attn_output_past_,
                out_cur_fp32,
                out,
                need_update_mask,
                need_store_mask,
                dirty_mask_
            );
            at::Tensor out_fused = fused_outputs[0];
            softmax_lse = fused_outputs[1];

            return {out_fused, softmax_lse, out_hist, softmax_lse_hist,
                    attn_output_past_, logsumexp_};
        }

        return {out, softmax_lse, out_hist, softmax_lse_hist};
    } else {
        return {out, softmax_lse};
    }
    return {out, softmax_lse};
}
} // namespace FLASH_NAMESPACE

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "FlashAttention";
    m.def("fwd", &FLASH_NAMESPACE::mha_fwd, "Forward pass");
    m.def("varlen_fwd", &FLASH_NAMESPACE::mha_varlen_fwd, "Forward pass (variable length)");
    m.def("bwd", &FLASH_NAMESPACE::mha_bwd, "Backward pass");
    m.def("varlen_bwd", &FLASH_NAMESPACE::mha_varlen_bwd, "Backward pass (variable length)");
    m.def("fwd_kvcache", &FLASH_NAMESPACE::mha_fwd_kvcache, "Forward pass, with KV-cache");
    m.def("update_attn_fused", &FLASH_NAMESPACE::update_attn_fused,
          "Fully fused update_attn kernel (BF16/FP16/FP32)");
}
