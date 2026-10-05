/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 * See LICENSE.txt for more license information.
 */

#pragma once

#include "device/device_primitives.cuh"
#include "device/ll/ll_ep_smem.cuh"
#include "device/ll_ep_adapter.cuh"
#include "device/jit/jit_runtime.hpp"
#include "device/jit/jit_source_literals.hpp"
#include "quantization_recipe.hpp"

#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <sstream>
#include <string>

namespace nccl_ep {
namespace ll {
namespace jit {

constexpr const char* kLlDispatchJitEntryName = "nccl_ep_jit_ll_dispatch_kernel";

// Selects which device dispatch kernel implementation gets JIT-compiled,
// chosen automatically per call by ll_dispatch_select_algo() below -- callers
// never pick a kernel directly.
//   kDefault:     the general kernel (every recipe/layout, RDMA/GIN capable).
//                 Also used for CFT; the LSA-only kernel has no CFT transport.
//   k2SidedRmLsa: LSA-only rank-major kernel that stages payload through
//                 RDMA buffers. Selected for LSA-only rank-major calls with
//                 CFT disabled, using NONE or fused DS_FP8E3M4. NONE still writes each token's payload
//                 directly into the peer's registered output window instead
//                 of staging through RDMA whenever the EP group's zero-copy
//                 flag (ncclEpGroupConfig_t::zero_copy) is ON and a window
//                 is present (args.recvDataWindow) -- that's an independent,
//                 runtime toggle inside the kernel, not a separate algo.
//   k2SidedRmLsaTma: TMA-pipelined SEND phase (dispatch_kernel_impl_2sided_rm_lsa_tma,
//                 ll/ll_dispatch_lsa.cuh). The default for LSA-eligible, unquantized (NONE) calls
//                 (NCCL_EP_DISPATCH_LSA=tma, see NcclEpDispatchLsaMode's doc comment in
//                 device/ll_ep_adapter.cuh) -- the TMA kernel takes no recipe template, so it is
//                 never selected for DS_FP8E3M4.
enum class LlDispatchAlgo { kDefault, k2SidedRmLsa, k2SidedRmLsaTma };

// Picks the most performant kernel for the configuration; overridable via NCCL_EP_DISPATCH_LSA
// (lsaMode, see NcclEpDispatchLsaMode's doc comment in device/ll_ep_adapter.cuh).
inline LlDispatchAlgo ll_dispatch_select_algo(
    bool nvlinkOnly, ncclEpDispQuant_t recipe, ncclEpLayout_t layout, bool cftEnabled,
    NcclEpDispatchLsaMode lsaMode) {
    // Explicit override: NCCL_EP_DISPATCH_LSA=universal always forces the general kernel,
    // regardless of eligibility.
    if (lsaMode == NcclEpDispatchLsaMode::kUniversal) {
        return LlDispatchAlgo::kDefault;
    }

    const bool lsaRecipe =
        recipe == NCCL_EP_DISP_QUANT_NONE || recipe == NCCL_EP_DISP_QUANT_DS_FP8E3M4;
    const bool lsaEligible =
        !cftEnabled && nvlinkOnly && lsaRecipe && layout == NCCL_EP_LAYOUT_RANK_MAJOR;
    const bool tmaEligible = recipe == NCCL_EP_DISP_QUANT_NONE;

    // Most preferred: the TMA kernel, when both LSA- and TMA-eligible.
    if (lsaEligible && tmaEligible && lsaMode == NcclEpDispatchLsaMode::kTma) {
        return LlDispatchAlgo::k2SidedRmLsaTma;
    }
    // LSA-eligible but TMA isn't (e.g. DS_FP8E3M4, which the TMA kernel doesn't support): fall
    // back to the SIMT LSA kernel.
    if (lsaEligible) {
        return LlDispatchAlgo::k2SidedRmLsa;
    }

    return LlDispatchAlgo::kDefault;
}

inline std::string ll_dispatch_jit_source(
    const DispatchKernelSpec& kernel_spec,
    int hidden,
    int num_topk,
    ncclEpLayout_t layout,
    bool nvlinkOnly,
    bool topkIdxIsInt64,
    ncclDataType_t tokenDtype,
    LlDispatchAlgo algo,
    bool compactQuant,
    bool stageQuant,
    bool cftEnabled,
    int numTmaStages) {
    const char* layout_literal = ::nccl_ep::jit::layout_literal(layout);
    const char* topk_type = topkIdxIsInt64 ? "int64_t" : "int32_t";
    const char* token_dtype_literal = ::nccl_ep::jit::token_dtype_literal(tokenDtype);

    // The JIT source must reference the same arg struct definition that the
    // host packs. Including device/ll_ep_adapter.cuh keeps the layout in sync;
    // ll_ep.cuh pulls in all kernel templates and helpers (including
    // dispatch_kernel_impl_2sided_rm_lsa and dispatch_kernel_impl_2sided_rm_lsa_tma, via
    // device/ll/ll_dispatch_lsa.cuh).
    std::ostringstream src;
    src << "#include \"device/ll_ep.cuh\"\n"
        << "#include \"device/ll_ep_adapter.cuh\"\n"
        << "\n"
        << "extern \"C\" __launch_bounds__("
        << (compactQuant ? kLlDsFp8CompactWarps * 32 : 1024)
        << ", 1)\n"
        << "__global__ void " << kLlDispatchJitEntryName << "(\n"
        << "    const __grid_constant__ nccl_ep::ll::dispatch_kernel_args_t p) {\n";
    switch (algo) {
    case LlDispatchAlgo::k2SidedRmLsaTma:
        src << "  nccl_ep::ll::dispatch_kernel_impl_2sided_rm_lsa_tma<\n"
            << "      " << hidden << ",\n"
            << "      " << num_topk << ",\n"
            << "      " << numTmaStages << ",\n"
            << "      " << topk_type << ",\n"
            << "      " << token_dtype_literal << ">(p);\n";
        break;
    case LlDispatchAlgo::k2SidedRmLsa:
        // dispatch_kernel_impl_2sided_rm_lsa hardcodes NCCL_EP_LAYOUT_RANK_MAJOR
        // (the only layout it supports) and takes the same (p) grid-constant
        // arg struct -- no layout template argument, no positional arg list,
        // here.
        src << "  nccl_ep::ll::dispatch_kernel_impl_2sided_rm_lsa<\n"
            << "      " << kernel_spec.recipe_source_literal << ",\n"
            << "      " << hidden << ",\n"
            << "      " << num_topk << ",\n"
            << "      " << topk_type << ",\n"
            << "      " << token_dtype_literal << ", " << (stageQuant ? "true" : "false") << ">(p);\n";
        break;
    case LlDispatchAlgo::kDefault:
        src << "  nccl_ep::ll::dispatch_kernel_impl<\n"
            << "      " << kernel_spec.recipe_source_literal << ",\n"
            << "      " << hidden << ",\n"
            << "      " << num_topk << ",\n"
            << "      " << layout_literal << ",\n"
            << "      " << ::nccl_ep::jit::bool_literal(nvlinkOnly) << ",\n"
            << "      " << topk_type << ",\n"
            << "      " << token_dtype_literal << ",\n"
            << "      " << kernel_spec.scale_type_literal << ",\n"
            << "      " << ::nccl_ep::jit::bool_literal(cftEnabled) << ">(p);\n";
        break;
    }
    src << "}\n";
    return src.str();
}

inline ncclResult_t launch_ll_dispatch(
    int hidden,
    ncclEpLayout_t layout,
    bool nvlinkOnly,
    bool topkIdxIsInt64,
    const DispatchKernelSpec& kernel_spec,
    ncclDataType_t tokenDtype,
    ncclEpDispQuant_t recipe,
    int num_topk,
    int numSms,
    int numWarps,
    bool stageQuant,
    int cft_smem_bytes,
    NcclEpDispatchLsaMode lsaMode,
    int maxDynamicSmem,
    const dispatch_kernel_args_t& args,
    cudaStream_t stream) {
    const bool cftEnabled = cft_smem_bytes != 0;
    const LlDispatchAlgo algo = ll_dispatch_select_algo(nvlinkOnly, recipe, layout, cftEnabled, lsaMode);
    const bool twoSidedRmLsa = algo == LlDispatchAlgo::k2SidedRmLsa;
    const bool twoSidedRmLsaTma = algo == LlDispatchAlgo::k2SidedRmLsaTma;
    // Keep the compact thread limit with a one-block launch bound: repeated
    // single- and multi-node tuning did not favor the tighter register budget.
    const bool compactQuant = stageQuant && numWarps <= kLlDsFp8CompactWarps;

    // TMA smem budget: stage-ring + rank-pointer-cache sizing is resolved here (host-side,
    // JIT-variant-selection time) from (hidden, dtype, maxDynamicSmem) only -- see
    // ll/ll_ep_smem.cuh's dispatch_tma_smem namespace doc comment for why this must not depend on
    // this call's token count.
    int numTmaStages = 0;
    int tmaDynamicSmemBytes = 0;
    if (twoSidedRmLsaTma) {
        const int hiddenBytes = static_cast<int>(::nccl_ep::token_row_bytes(tokenDtype, hidden));
        const size_t dispatchHdrSz = get_dispatch_hdr_sz(num_topk, layout);
        // Reserve syncAndSendCounts's static __shared__ state out of the budget before sizing the
        // dynamic stage ring -- see dispatch_tma_smem::kStaticSmemBytes's doc comment; same split
        // call_combine already applies for combine_smem::kLsaStaticSmemBytes.
        const int maxDynamicSmemForTma = maxDynamicSmem - dispatch_tma_smem::kStaticSmemBytes;
        const auto smemConfig = dispatch_tma_smem::choose_config(
            hiddenBytes, static_cast<int>(dispatchHdrSz), maxDynamicSmemForTma, args.numRanks);
        if (!smemConfig.feasible) {
            std::fprintf(
                stderr,
                "[nccl_ep jit] LL dispatch TMA shared memory cannot fit: hidden=%d, dtype=%d, numRanks=%d, "
                "limit=%d bytes (device limit=%d bytes, %d reserved for static __shared__ state).\n",
                hidden, static_cast<int>(tokenDtype), args.numRanks, maxDynamicSmemForTma, maxDynamicSmem,
                dispatch_tma_smem::kStaticSmemBytes);
            return ncclInvalidArgument;
        }
        numTmaStages = smemConfig.num_stages;
        tmaDynamicSmemBytes = smemConfig.dynamic_smem_bytes;
    }

    static const int variant_identity_default = 0;
    static const int variant_identity_2sided_rm_lsa = 0;
    static const int variant_identity_2sided_rm_lsa_tma = 0;

    ::nccl_ep::jit::JitKernelVariant variant;
    switch (algo) {
    case LlDispatchAlgo::k2SidedRmLsaTma:
        variant.kernel_family = "ll_dispatch_2sided_rm_lsa_tma";
        variant.identity = &variant_identity_2sided_rm_lsa_tma;
        break;
    case LlDispatchAlgo::k2SidedRmLsa:
        variant.kernel_family = "ll_dispatch_2sided_rm_lsa";
        variant.identity = &variant_identity_2sided_rm_lsa;
        break;
    case LlDispatchAlgo::kDefault:
    default:
        variant.kernel_family = "ll_dispatch";
        variant.identity = &variant_identity_default;
        break;
    }
    variant.entry_name = kLlDispatchJitEntryName;
    // Derived from the raw parameters so the warm-cache launch path never has
    // to build the variant-name string.
    std::uint64_t key = ::nccl_ep::jit::kRuntimeKeySeed;
    key = ::nccl_ep::jit::runtime_key_mix(key, static_cast<std::uint64_t>(hidden));
    key = ::nccl_ep::jit::runtime_key_mix(key, static_cast<std::uint64_t>(layout));
    key = ::nccl_ep::jit::runtime_key_mix(key, static_cast<std::uint64_t>(num_topk));
    key = ::nccl_ep::jit::runtime_key_mix(key, kernel_spec.recipe_cache_tag);
    key = ::nccl_ep::jit::runtime_key_mix(key, kernel_spec.payload_cache_tag);
    key = ::nccl_ep::jit::runtime_key_mix(key, kernel_spec.scale_cache_tag);
    key = ::nccl_ep::jit::runtime_key_mix(
        key, (nvlinkOnly ? 1u : 0u) | (topkIdxIsInt64 ? 2u : 0u) | (cftEnabled ? 4u : 0u));
    key = ::nccl_ep::jit::runtime_key_mix(key, static_cast<std::uint64_t>(tokenDtype));
    key = ::nccl_ep::jit::runtime_key_mix(key, static_cast<std::uint64_t>(compactQuant));
    key = ::nccl_ep::jit::runtime_key_mix(key, static_cast<std::uint64_t>(stageQuant));
    key = ::nccl_ep::jit::runtime_key_mix(
        key, twoSidedRmLsaTma ? (1u | (static_cast<std::uint64_t>(numTmaStages) << 2)) : 0u);
    variant.runtime_key = key;
    variant.num_blocks = numSms;
    variant.block_dim = numWarps * 32;
    // CFT staging and the TMA variant (mutually exclusive -- TMA is only ever selected when CFT is
    // disabled, see ll_dispatch_select_algo) use dynamic shared memory; store-only dispatch needs
    // none.
    variant.dynamic_smem_bytes = cftEnabled ? cft_smem_bytes : tmaDynamicSmemBytes;
    // Always cooperative: kDefault needs its grid-wide SEND/RECV sync;
    // k2SidedRmLsa/k2SidedRmLsaTma use it for efficient cross-CTA atomic coordination.
    variant.cooperative = true;
    // Pair SMs into clusters of 2 when possible to share distributed SMEM.
    variant.cluster_dim_x = (numSms % 2 == 0) ? 2 : 1;

    std::string error;
    // Warm-cache fast path: launches without materializing variant_name/source.
    // sizeof = 0 means "kernel_param points to a single fixed-size arg struct".
    ::nccl_ep::jit::JitKernelStatus status = ::nccl_ep::jit::launch_jit_kernel_cached(
        variant, const_cast<dispatch_kernel_args_t*>(&args), 0, stream, &error);

    std::string variant_name;
    if (status != ::nccl_ep::jit::JitKernelStatus::kLaunched &&
        status != ::nccl_ep::jit::JitKernelStatus::kLaunchFailed) {
        // Cache miss: build the name + source and take the compile/load path.
        std::ostringstream name;
        name << "ll_dispatch"
             << "_hdim" << hidden << ::nccl_ep::jit::layout_name_tag(layout)
             << "_topk" << num_topk
             << "_recipe" << kernel_spec.recipe_cache_tag
             << "_payload" << kernel_spec.payload_cache_tag
             << "_scale" << kernel_spec.scale_cache_tag
             << (nvlinkOnly ? "_nvlinkonly" : "")
             << (twoSidedRmLsa ? "_2sidedrmlsa" : "")
             << (twoSidedRmLsaTma ? "_2sidedrmlsatma" : "")
             << (twoSidedRmLsaTma ? ("_stages" + std::to_string(numTmaStages)) : "")
             << (cftEnabled ? "_cft" : "")
             << (compactQuant ? "_compact" : "")
             << (stageQuant ? "_staged" : "")
             << (topkIdxIsInt64 ? "_topk64" : "_topk32")
             << ::nccl_ep::jit::token_dtype_name_tag(tokenDtype);
        variant_name = name.str();
        const std::string source = ll_dispatch_jit_source(
            kernel_spec, hidden, num_topk, layout, nvlinkOnly, topkIdxIsInt64, tokenDtype, algo,
            compactQuant, stageQuant, cftEnabled, numTmaStages);
        variant.variant_name = variant_name;
        variant.source = source;
        status = ::nccl_ep::jit::launch_jit_kernel(
            variant, const_cast<dispatch_kernel_args_t*>(&args), stream, &error);
    }

    if (status != ::nccl_ep::jit::JitKernelStatus::kLaunched) {
        std::fprintf(stderr, "[nccl_ep jit] fatal LL dispatch JIT launch failure for %s: %s%s%s\n",
                     variant_name.c_str(), ::nccl_ep::jit::jit_kernel_status_name(status), error.empty() ? "" : ": ",
                     error.empty() ? "" : error.c_str());
        return ncclInternalError;
    }
    return ncclSuccess;
}

} // namespace jit
} // namespace ll
} // namespace nccl_ep
