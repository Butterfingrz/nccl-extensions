/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 * See LICENSE.txt for more license information.
 */

#pragma once

#include "device/ll_ep_adapter.cuh"
#include "device/jit/jit_runtime.hpp"
#include "device/jit/jit_source_literals.hpp"

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

constexpr const char* kLlCombineJitEntryName = "nccl_ep_jit_ll_combine_kernel";

// The inner-loop unroll factor is compile-time fixed; top-k is passed to the
// source generator so each handle receives an exact specialization.
constexpr int kLlCombineMaxUnrolls = 4;

// Selects which device combine kernel implementation gets JIT-compiled,
// chosen automatically per call by ll_combine_select_algo() below -- callers
// never pick a kernel directly.
//   kDefault:     the general kernel (every recipe/layout, RDMA/GIN capable).
//                 Selected whenever the call isn't LSA-only, or its
//                 recipe/layout/LogFMT combination isn't eligible for the
//                 LSA-only kernel below.
//   k2SidedRmLsa: LSA-only rank-major kernel that stages payload through
//                 RDMA buffers on both send and receive. Selected for every
//                 LSA-only, NONE-recipe, rank-major call with CFT disabled.
enum class LlCombineAlgo { kDefault, k2SidedRmLsa };

// Picks the most performant kernel based on the configuration
inline LlCombineAlgo ll_combine_select_algo(
    bool nvlinkOnly, ncclEpCombQuant_t qrecipe, bool useLogFmt, ncclEpLayout_t layout,
    bool cftEnabled = false) {
    // CFT dispatch skips the next-bank wipe. Pair it with the general combine
    // kernel, which clears consumed flags; the LSA kernel leaves them set.
    const bool lsaEligible = !cftEnabled && nvlinkOnly && qrecipe == NCCL_EP_COMB_QUANT_NONE && !useLogFmt &&
        layout == NCCL_EP_LAYOUT_RANK_MAJOR;
    return lsaEligible ? LlCombineAlgo::k2SidedRmLsa : LlCombineAlgo::kDefault;
}

inline std::string ll_combine_jit_source(
    bool useLogFmt,
    const char* recipe_literal,
    int hidden,
    int num_topk,
    ncclEpLayout_t layout,
    bool topkIdxIsInt64,
    ncclDataType_t tokenDtype,
    LlCombineAlgo algo) {
    const char* layout_literal = ::nccl_ep::jit::layout_literal(layout);
    const char* topk_type = topkIdxIsInt64 ? "int64_t" : "int32_t";
    const char* token_dtype_literal = ::nccl_ep::jit::token_dtype_literal(tokenDtype);
    std::ostringstream src;
    src << "#include \"device/ll_ep.cuh\"\n"
        << "#include \"device/ll_ep_adapter.cuh\"\n"
        << "\n"
        << "extern \"C\" __launch_bounds__(1024, 1)\n"
        << "__global__ void " << kLlCombineJitEntryName << "(\n"
        << "    const __grid_constant__ nccl_ep::ll::combine_kernel_args_t p) {\n";
    if (algo == LlCombineAlgo::k2SidedRmLsa) {
        src << "  nccl_ep::ll::combine_kernel_impl_2sided_rm_lsa<\n"
            << "      " << hidden << ",\n"
            << "      " << num_topk << ",\n"
            << "      " << kLlCombineMaxUnrolls << ",\n"
            << "      " << topk_type << ",\n"
            << "      " << token_dtype_literal << ">(\n"
            << "      p.inData, p.srcInfo,\n"
            << "      static_cast<const " << topk_type << "*>(p.inTopkIdx),\n"
            << "      p.rankMask, p.asyncErrorFlag,\n"
            << "      p.outData,\n"
            << "      p.rdmaBuf,\n"
            << "      p.sendOff, p.recvOff, p.recvFlagOff,\n"
            << "      p.combineSync, p.nextRecvCntBufSize,\n"
            << "      p.waitStats, p.epochState, p.payloadSlotStride, p.signalSlotStride,\n"
            << "      p.numCombinedTokens, p.maxTokensPerRank,\n"
            << "      p.numExperts, p.currRank, p.numRanks,\n"
            << "      p.numWarpGroups, p.numWarpsPerGroup,\n"
            << "      p.phases, p.zeroCopy,\n"
            << "      p.devComm, p.windows, p.timeoutCycles);\n";
    } else {
        src << "  nccl_ep::ll::combine_kernel_impl<\n"
            << "      " << ::nccl_ep::jit::bool_literal(useLogFmt) << ",\n"
            << "      " << hidden << ",\n"
            << "      " << num_topk << ",\n"
            << "      " << kLlCombineMaxUnrolls << ",\n"
            << "      " << layout_literal << ",\n"
            << "      " << topk_type << ",\n"
            << "      " << token_dtype_literal << ", " << recipe_literal << ">(\n"
            << "      p.inData, p.inGlobalScales, p.srcInfo, p.layoutRange,\n"
            << "      static_cast<const " << topk_type << "*>(p.inTopkIdx), p.topkWeights,\n"
            << "      p.rankMask, p.asyncErrorFlag,\n"
            << "      p.outData,\n"
            << "      p.rdmaBuf,\n"
            << "      p.sendOff, p.recvOff, p.recvFlagOff,\n"
            << "      p.combineSync, p.nextRecvCntBufSize,\n"
            << "      p.waitStats, p.epochState, p.payloadSlotStride, p.signalSlotStride,\n"
            << "      p.numCombinedTokens, p.hidden, p.maxTokensPerRank,\n"
            << "      p.numExperts, p.currRank, p.numRanks,\n"
            << "      p.numWarpGroups, p.numWarpsPerGroup,\n"
            << "      p.phases, p.zeroCopy,\n"
            << "      p.devComm, p.windows, p.signalsBase, p.timeoutCycles);\n";
    }
    src << "}\n";
    return src.str();
}

inline ncclResult_t launch_ll_combine(
    bool nvlinkOnly,
    bool cftEnabled,
    bool useLogFmt,
    ncclEpCombQuant_t qrecipe,
    unsigned int device_sm,
    int hidden,
    ncclEpLayout_t layout,
    bool topkIdxIsInt64,
    ncclDataType_t tokenDtype,
    int num_topk,
    int numSms,
    int numWarps,
    int dynamic_smem_bytes,
    const combine_kernel_args_t& args,
    cudaStream_t stream) {
    const LlCombineAlgo algo = ll_combine_select_algo(nvlinkOnly, qrecipe, useLogFmt, layout, cftEnabled);
    const bool twoSidedRmLsa = algo == LlCombineAlgo::k2SidedRmLsa;

    const char* recipe_literal = twoSidedRmLsa ? "" : ::nccl_ep::jit::combine_recipe_literal(qrecipe);
    if (!twoSidedRmLsa && recipe_literal == nullptr) {
        std::fprintf(stderr, "ncclEpCombine: unsupported LL combine recipe %d\n", qrecipe);
        return ncclInvalidArgument;
    }
    static const int variant_identity_default = 0;
    static const int variant_identity_2sided_rm_lsa = 0;

    ::nccl_ep::jit::JitKernelVariant variant;
    variant.kernel_family = twoSidedRmLsa ? "ll_combine_2sided_rm_lsa" : "ll_combine";
    variant.entry_name = kLlCombineJitEntryName;
    variant.identity = twoSidedRmLsa ? &variant_identity_2sided_rm_lsa : &variant_identity_default;
    // Derived from the raw parameters so the warm-cache launch path never has
    // to build the variant-name string.
    std::uint64_t key = ::nccl_ep::jit::kRuntimeKeySeed;
    key = ::nccl_ep::jit::runtime_key_mix(key, static_cast<std::uint64_t>(hidden));
    key = ::nccl_ep::jit::runtime_key_mix(key, static_cast<std::uint64_t>(layout));
    key = ::nccl_ep::jit::runtime_key_mix(key, static_cast<std::uint64_t>(num_topk));
    key = ::nccl_ep::jit::runtime_key_mix(key, static_cast<std::uint64_t>(qrecipe));
    key = ::nccl_ep::jit::runtime_key_mix(key, (useLogFmt ? 1u : 0u) | (topkIdxIsInt64 ? 2u : 0u));
    key = ::nccl_ep::jit::runtime_key_mix(key, static_cast<std::uint64_t>(tokenDtype));
    variant.runtime_key = key;
    variant.num_blocks = numSms;
    variant.block_dim = numWarps * 32;
    variant.dynamic_smem_bytes = dynamic_smem_bytes;
    if (qrecipe == NCCL_EP_COMB_QUANT_NVFP4) {
        // Recipe validation has already checked toolkit and device support.
        variant.min_sm = 100;
        variant.target_arch = host_device_fp4_target_arch(device_sm);
    }
    // Always cooperative: kDefault/2sided_rm_lsa need it for their grid-wide
    // SEND/RECV sync.
    variant.cooperative = true;
    // Pair SMs into clusters of 2 when possible to share distributed SMEM.
    variant.cluster_dim_x = (numSms % 2 == 0) ? 2 : 1;

    std::string error;
    // Warm-cache fast path: launches without materializing variant_name/source.
    ::nccl_ep::jit::JitKernelStatus status = ::nccl_ep::jit::launch_jit_kernel_cached(
        variant, const_cast<combine_kernel_args_t*>(&args), 0, stream, &error);

    std::string variant_name;
    if (status != ::nccl_ep::jit::JitKernelStatus::kLaunched &&
        status != ::nccl_ep::jit::JitKernelStatus::kLaunchFailed) {
        // Cache miss: build the name + source and take the compile/load path.
        std::ostringstream name;
        name << "ll_combine"
             << "_hdim" << hidden << ::nccl_ep::jit::layout_name_tag(layout)
             << "_topk" << num_topk
             << (useLogFmt ? "_logfmt" : "")
             << (twoSidedRmLsa ? "_2sidedrmlsa" : "")
             << (topkIdxIsInt64 ? "_topk64" : "_topk32")
             << ::nccl_ep::jit::token_dtype_name_tag(tokenDtype)
             << (twoSidedRmLsa ? "" : ::nccl_ep::jit::combine_recipe_name_tag(qrecipe));
        variant_name = name.str();
        const std::string source = ll_combine_jit_source(
            useLogFmt, recipe_literal, hidden, num_topk, layout, topkIdxIsInt64, tokenDtype, algo);
        variant.variant_name = variant_name;
        variant.source = source;
        status = ::nccl_ep::jit::launch_jit_kernel(
            variant, const_cast<combine_kernel_args_t*>(&args), stream, &error);
    }

    if (status != ::nccl_ep::jit::JitKernelStatus::kLaunched) {
        std::fprintf(stderr, "[nccl_ep jit] LL combine JIT launch failure for %s: %s%s%s\n", variant_name.c_str(),
                     ::nccl_ep::jit::jit_kernel_status_name(status), error.empty() ? "" : ": ",
                     error.empty() ? "" : error.c_str());
        return ncclInternalError;
    }
    return ncclSuccess;
}

} // namespace jit
} // namespace ll
} // namespace nccl_ep
