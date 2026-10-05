/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 * See LICENSE.txt for more license information.
 */

#pragma once

#include <cooperative_groups.h>
#include "../device_primitives.cuh"
#include "nccl_device.h"
#include "../ll_ep_adapter.cuh"
#include "ll_ep_smem.cuh"
#include "ll_common.cuh"
#include "ll_lsa_primitives.cuh"
#include "ll_mask.cuh"

namespace cg = cooperative_groups;

// LSA-only, unquantized, rank-major-only LL combine. Every destination is
// assumed NVLink-reachable (same LSA team) -- there is no RDMA/GIN fallback
// anywhere in this file, unlike the general combine_kernel_impl in ll_ep.cuh,
// which resolves each peer's P2P pointer at runtime and falls back to GIN
// whenever it comes back null. This is a faithful, behavior-preserving
// extraction/specialization of that general kernel, meant as a clean
// prototyping base, following the same pattern already used for dispatch
// (ll/ll_dispatch_lsa.cuh).
//
// Scope cuts relative to combine_kernel_impl:
//   - kRecipe is fixed to NCCL_EP_COMB_QUANT_NONE: no NVFP4 support at all
//     (no CombineRecipeTraits<NVFP4>/CombineRecipeSendOps<NVFP4>, no
//     nvfp4_* packing helpers, no decodeAndAccumulateNvfp4, no
//     __CUDA_ARCH_FAMILY_SPECIFIC__ NVFP4-unsupported-device guard -- that
//     guard exists purely because of NVFP4, so it goes with it).
//   - kUseLogFMT is fixed to false: no LogFMT encode/decode anywhere. The
//     NONE-recipe wire slot size (numBytesPerSlot) intentionally keeps the
//     same reserved LogFMT-metadata tail bytes the general kernel's NONE
//     recipe always carries (that reservation does not depend on kUseLogFMT
//     even in the general kernel), so the wire format stays byte-compatible;
//     this file simply never writes or reads that tail. Shared-memory
//     regions that exist purely to stage LogFMT metadata (per-division
//     logAmax/logAmin/castInfo buffers) are dropped, since nothing populates
//     or consumes them once LogFMT is gone -- that's a smem-only footprint
//     reduction, not a wire-format change.
//   - kLayout is fixed to NCCL_EP_LAYOUT_RANK_MAJOR: no EXPERT_MAJOR support
///    (matches the dispatch LSA extraction's own layout scope cut). The
//     EXPERT_MAJOR-specific per-topk send loop and its weight lookup are
//     dropped entirely; rank-major's "one send per received slot" loop and
//     its caller-applies-the-weight convention (topkWeight is always 1.0f
//     here) are the only path.
//   - Every P2P pointer this file resolves is asserted nonzero
//     (EP_DEVICE_ASSERT) instead of falling back to a GIN put/signal-wait --
//     sendRankFinishFlagLsa/waitForAllRanksFinishLsa/processAndSendTokenLsa
//     have no RDMA branch, and sendTokenViaRdma has no LSA counterpart at all.
//   - `layoutRange` and the runtime `hidden` parameter are dropped from the
//     signature: neither is referenced anywhere in the general kernel's body
//     either, so there is nothing to extract.
//   - TMA/shared-memory pipelining (combine_smem, PatternVisitor, TMA
//     buffer/barrier setup) is kept as-is: it's a data-movement optimization
//     orthogonal to quantization/RDMA/layout, not something this scope cut
//     touches.
//
// `zeroCopy` (whether the send-phase TMA copy source is the local
// send-staging slot vs `inData` directly) is unrelated to any of the above
// and stays a runtime toggle, same as in the general kernel.

namespace nccl_ep {

namespace ll {

// kMaxLsaCombineRanks/kActiveRankMaskWords/kLsaStaticSmemBytes live in
// combine_smem (ll_ep_smem.cuh) so this file and choose_combine_smem_config
// (ll_ep_adapter.cuh) share one definition.

// Clean next receive count buffer. Pure local buffer zeroing, no P2P/GIN
// dependency, no atomic touch -- this kernel's completion signaling is
// syncAndSendFinishLsa's own CTA-election counter (ctaDone), not this
// buffer, so there's nothing to notify here.
__forceinline__ __device__ void cleanNextRecvCntBufAndNotifyLsa(int* nextRecvCntBuf, int nextRecvCntBufSize, int laneId) {
#pragma unroll
    for (int i = laneId; i < nextRecvCntBufSize; i += 32) {
        nextRecvCntBuf[i] = 0;
    }
}

// Rank-scoped finish-flag publish, used by combine_kernel_impl_2sided_rm_lsa's
// single-shot send completion: writes into `dstRank`'s recvFlagBuf at index
// `currRank` (the sender's own identity) -- mirrors sendPeerTokenCountLsa in
// ll_dispatch_lsa.cuh exactly (send keyed by the sender's rank, receive
// polls by source rank), adapted to a plain ready flag instead of an encoded
// count, since combine's receiver already knows how many tokens to expect
// from topk info -- there is nothing to count, only "am I ready" to signal.
__forceinline__ __device__ void sendRankFinishFlagLsa(
    int dstRank,
    int currRank,
    int* recvFlagBufLocal,
    size_t recvFlagOff,
    int* rankMask,
    const ncclWindow_t* windows,
    ncclDevComm* devComm) {
    const uint64_t recvFlagPtr = reinterpret_cast<uint64_t>(recvFlagBufLocal + currRank);
    const size_t recvFlagOffset = recvFlagOff + static_cast<size_t>(currRank) * sizeof(int);
    const auto dstP2pPtr = ncclGetP2pPtr(recvFlagPtr, recvFlagOffset, currRank, dstRank, windows, devComm);
    if (not isRankMasked(rankMask, dstRank)) {
        EP_DEVICE_ASSERT(dstP2pPtr != 0);
        st_relaxed_sys_global(reinterpret_cast<int*>(dstP2pPtr), 1);
    }
}

// Rank-centric recv-side readiness wait, mirroring ll_dispatch_lsa.cuh's
// syncAndRecvCounts closely: up to numRanks threads each independently
// confirm one active peer's finish flag has landed in parallel (relaxed
// poll, one shared timeout deadline across the loop, same as dispatch) --
// peers' arrival times are independent, so checking them on separate
// threads overlaps their tail latencies instead of stacking them the way a
// single serial thread would. Each participating thread acquire-fences its
// own now-confirmed read(s) (and, transitively, that peer's payload writes
// ordered before it by the peer's own release fence); __syncthreads()
// propagates that visibility to the rest of the CTA, including threads that
// checked a different peer or none at all -- same reliance on
// __syncthreads() propagating another thread's fence as the original
// single-thread version had. No grid-wide barrier needed for cross-rank
// readiness -- every CTA confirms independently. Must be called by every
// thread of the CTA.
//
// Each CTA observes remote signals independently. To avoid cross-CTA race
// condition or performing expensive synchronization, each CTA builds a local
// rank mask based on the existing global mask and uses it during data
// processing.
__forceinline__ __device__ void waitForAllRanksFinishLsa(
    int threadId,
    int numRanks,
    int* rankMask,
    uint32_t* activeRankMask,
    const int* recvFlagBuf,
    uint64_t timeoutCycles,
    int currRank,
    int* asyncErrorFlag,
    int64_t* waitStats) {
    EP_DEVICE_ASSERT(numRanks <= combine_smem::kMaxLsaCombineRanks);
#pragma unroll
    for (int w = threadId; w < combine_smem::kActiveRankMaskWords; w += blockDim.x) {
        activeRankMask[w] = 0;
    }
    // Every thread must see the zero-init before any atomicOr below.
    __syncthreads();

    const auto startTime = clock64();
    bool didWork = false;
    for (int peer = threadId; peer < numRanks; peer += blockDim.x) {
        didWork = true;
        if (isRankMasked(rankMask, peer)) continue;
        uint64_t waitCost = 0;
        bool flagReady;
        do {
            flagReady = ld_relaxed_sys_global(reinterpret_cast<const uint32_t*>(recvFlagBuf + peer)) != 0;
            waitCost = clock64() - startTime;
        } while (!flagReady && waitCost <= timeoutCycles);
        if (waitCost > timeoutCycles) {
            printf("Warning: NCCL EP timeout for combine receive, rank %d, src_rank %d\n", currRank, peer);
            if (rankMask == nullptr) trap();
            atomicExch(rankMask + peer, 0);
            if (asyncErrorFlag != nullptr) atomicExch_system(asyncErrorFlag, 1);
        } else {
            // Confirmed ready by this CTA's own poll.
            atomicOr(&activeRankMask[peer / 32], 1u << (peer % 32));
        }
        if (waitStats != nullptr) {
            atomicAdd(reinterpret_cast<unsigned long long*>(waitStats + peer), waitCost);
        }
    }
    if (didWork) {
        // Acquire-fence this thread's now-confirmed reads (and, transitively,
        // the peer's payload writes ordered before it by the peer's own
        // release fence) before any thread in this CTA touches recvBuf.
        memory_fence();
    }
    __syncthreads();
}

// Unquantized (no LogFMT) recv-side decode/accumulate: casts the raw
// BF16/FP16/FP32 wire bytes and accumulates into `accum`, weighted.
template <int kNumRecvUnrolls, ncclDataType_t kTokenDtype>
__forceinline__ __device__ void decodeAndAccumulateLsa(
    uint32_t* ldBuffer,
    float* accum,
    const float& weight,
    int validUnrolls) {
    if constexpr (kTokenDtype == ncclFloat32) {
#pragma unroll
        for (int k = 0; k < kNumRecvUnrolls * 4; ++k) {
            if (k < validUnrolls * 4)
                accum[k] += *reinterpret_cast<float*>(ldBuffer + k) * weight;
        }
    } else if constexpr (kTokenDtype == ncclFloat16) {
#pragma unroll
        for (int k = 0; k < kNumRecvUnrolls * 4; ++k) {
            if (k < validUnrolls * 4) {
                auto fp16Pack = *reinterpret_cast<__half2*>(ldBuffer + k);
                accum[k * 2 + 0] += static_cast<float>(fp16Pack.x) * weight;
                accum[k * 2 + 1] += static_cast<float>(fp16Pack.y) * weight;
            }
        }
    } else {
#pragma unroll
        for (int k = 0; k < kNumRecvUnrolls * 4; ++k) {
            if (k < validUnrolls * 4) {
                auto bf16Pack = *reinterpret_cast<__nv_bfloat162*>(ldBuffer + k);
                accum[k * 2 + 0] += static_cast<float>(bf16Pack.x) * weight;
                accum[k * 2 + 1] += static_cast<float>(bf16Pack.y) * weight;
            }
        }
    }
}

// LSA-only counterpart of processAndSendToken: copies one combined token's
// payload via TMA from `srcDataInt4Ptr` (or the local send-staging slot,
// when `zeroCopy` is set) directly into the peer's already-resolved,
// already-nonzero P2P destination -- unquantized, non-LogFMT BF16-family
// wire format only, so there is no smem quantization/packing step and no
// RDMA fallback (dstP2pPtr is asserted nonzero, never checked against 0).
template <
    int kNumSendUnrolls,
    int kNumStages,
    int kNumPrefetch,
    typename TmaBuffersT,
    typename FullBarriersT,
    typename TmaLoadAndArriveT,
    typename GetNumTmaBytesT>
__forceinline__ __device__ void processAndSendTokenLsa(
    const int4* srcDataInt4Ptr,
    int4* sendBufInt4Ptr,
    uint64_t dstP2pPtr,
    int hiddenBf16Int4,
    int hiddenBf16Int4Pad,
    bool zeroCopy,
    TmaBuffersT tmaBuffers,
    FullBarriersT fullBarriers,
    uint32_t& tmaPhase,
    TmaLoadAndArriveT& tmaLoadAndArrive,
    GetNumTmaBytesT& getNumTmaBytes,
    int laneId) {
    EP_DEVICE_ASSERT(dstP2pPtr != 0);
    const int kNumIters = hiddenBf16Int4Pad / (32 * kNumSendUnrolls);

    const auto copySrcPtr = zeroCopy ? sendBufInt4Ptr : srcDataInt4Ptr;
    const auto copyDstPtr = reinterpret_cast<int4*>(dstP2pPtr);

    // Prefetch
    if (elect_one_sync()) {
        tmaLoadAndArrive(0, copySrcPtr, getNumTmaBytes(0));
    }
    __syncwarp();

#pragma unroll
    for (int i = laneId * kNumSendUnrolls, iterIdx = 0; i < hiddenBf16Int4Pad; i += 32 * kNumSendUnrolls, ++iterIdx) {
        // Load the next iteration
        const int& stageIdx = iterIdx % kNumStages;
        const int& nextStageIdx = (iterIdx + 1) % kNumStages;
        if (iterIdx + 1 < kNumIters and elect_one_sync()) {
            tma_store_wait<kNumStages - kNumPrefetch - 1>();
            const auto offsetInt4 = i + 32 * kNumSendUnrolls;
            tmaLoadAndArrive(nextStageIdx, copySrcPtr + offsetInt4, getNumTmaBytes(offsetInt4));
        }
        __syncwarp();

        // Wait the current TMA arrival
        EP_STATIC_ASSERT(kNumStages < 32, "Too many stages");
        mbarrier_wait<true>(fullBarriers[stageIdx], tmaPhase, stageIdx);

        // BF16-family original values, unquantized, no LogFMT packing.
        if (elect_one_sync()) {
            tma_store_1d(tmaBuffers[stageIdx], copyDstPtr + i, getNumTmaBytes(i));
        }
        __syncwarp();
    }

    // Flush all stores
    tma_store_wait<0>();
    __syncwarp();
}

// Reduction-warp body for one token, used by combine_kernel_impl_2sided_rm_lsa:
// pulls each of this token's up-to-numTopk contributions out of the
// TMA-load warp's ring buffer, decodes+accumulates them weighted, and stores
// the combined result. getDstRank(i) resolves topk slot i's destination rank
// (or a negative sentinel to skip a duplicate-rank/invalid slot), re-derived
// per token from inTopkIdx via warpFirstTopKOccurence. tmaPhase/stageIdx are
// threaded through by reference: this ring buffer state persists across
// tokens, not just within one call.
template <int kNumStages, int kNumElemsPerInt4, int kNumRecvUnrolls, ncclDataType_t kTokenDtype,
    typename FullBarriersT, typename EmptyBarriersT, typename TmaLdBuffersT, typename TmaStBuffersT,
    typename GetDstRankT>
__forceinline__ __device__ void combineReduceTokenLsa(
    int tokenIdx,
    int numTopk,
    int laneId,
    int decodeWarpIdx,
    int kNumBF16PerWarpBytes,
    int64_t hiddenBf16Int4,
    const uint32_t* activeRankMask,
    FullBarriersT fullBarriers,
    EmptyBarriersT emptyBarriers,
    TmaLdBuffersT tmaLdBuffers,
    TmaStBuffersT tmaStBuffers,
    uint32_t& tmaPhase,
    int& stageIdx,
    void* outData,
    GetDstRankT getDstRank) {
    float combinedData[kNumElemsPerInt4 * kNumRecvUnrolls] = {0.0f};
    for (int i = 0; i < numTopk; ++i) {
        const int dstRank = getDstRank(i);
        if (dstRank < 0 or not isRankActiveLocal(activeRankMask, dstRank)) continue;
        // Per-rank weight sum is applied by the caller before ncclEpCombine.
        constexpr float topkWeight = 1.0f;

        mbarrier_wait<true>(fullBarriers[stageIdx], tmaPhase, stageIdx);
        int tmaOffset = kNumBF16PerWarpBytes * decodeWarpIdx;
        const int firstInt4 = (decodeWarpIdx * 32 + laneId) * kNumRecvUnrolls;
        const int validUnrolls = max(0, min(kNumRecvUnrolls,
                                           static_cast<int>(hiddenBf16Int4) - firstInt4));
        decodeAndAccumulateLsa<kNumRecvUnrolls, kTokenDtype>(
            reinterpret_cast<uint32_t*>(
                tmaLdBuffers[stageIdx] + tmaOffset + kNumBF16PerWarpBytes / 32 * laneId),
            combinedData,
            topkWeight,
            validUnrolls);

        // Order generic-proxy reads before releasing the stage to the async TMA producer.
        fence_view_async_shared();
        if (elect_one_sync()) mbarrier_arrive(emptyBarriers[stageIdx]);
        stageIdx = (stageIdx + 1) % kNumStages;
    }
    tma_store_wait<0>();

#pragma unroll
    for (int k = 0; k < kNumRecvUnrolls * 4; ++k) {
        uint32_t packed;
        if constexpr (kTokenDtype == ncclFloat32) {
            packed = *reinterpret_cast<uint32_t*>(&combinedData[k]);
        } else if constexpr (kTokenDtype == ncclFloat16) {
            auto fp16Pack =
                __half2(__float2half(combinedData[k * 2]), __float2half(combinedData[k * 2 + 1]));
            packed = *reinterpret_cast<uint32_t*>(&fp16Pack);
        } else {
            auto bf16Pack = __nv_bfloat162(combinedData[k * 2], combinedData[k * 2 + 1]);
            packed = *reinterpret_cast<uint32_t*>(&bf16Pack);
        }
        tmaStBuffers[decodeWarpIdx][kNumRecvUnrolls * 4 * laneId + k] = packed;
    }
    tma_store_fence();
    if (elect_one_sync()) {
        const int storeOffsetInt4 = decodeWarpIdx * kNumRecvUnrolls * 32;
        const int storeBytes = min(kNumBF16PerWarpBytes,
                                   static_cast<int>((hiddenBf16Int4 - storeOffsetInt4) * sizeof(int4)));
        tma_store_1d(
            tmaStBuffers[decodeWarpIdx],
            static_cast<int4*>(outData) + tokenIdx * hiddenBf16Int4 + storeOffsetInt4,
            storeBytes);
    }
    __syncwarp();
}

// SEND-phase completion for combine_kernel_impl_2sided_rm_lsa, mirroring
// syncAndSendCounts in ll_dispatch_lsa.cuh: a CTA-scope barrier, then
// electing the last-finishing CTA in the grid (one release-scoped atomic per
// CTA) to publish readiness to every peer -- once, instead of every
// responsible-expert channel independently signaling completion via its own
// atomicCleanFlag countdown gate (that per-channel replication exists in the
// general kernel only to spread completion across multiple GIN QPs; plain
// NVLink P2P has no such requirement, so one signal per peer suffices, same
// as dispatch's own syncAndSendCounts). Unlike syncAndSendCounts, there is
// no per-destination count to publish -- combine's receiver already knows
// how many tokens to expect from topk info, so this is a plain readiness
// flag (sendRankFinishFlagLsa), not a value the receiver reads.
__forceinline__ __device__ bool syncAndSendFinishLsa(
    int warpId,
    int numWarps,
    int laneId,
    int* ctaDone,
    int numSms,
    int numRanks,
    int currRank,
    int* recvFlagBuf,
    size_t recvFlagOff,
    int* rankMask,
    const ncclWindow_t* windows,
    ncclDevComm* devComm) {
    // Step 1: CTA-scope barrier (also a full memory fence at CTA scope, per
    // bar.sync semantics) -- every write issued above by any warp of this
    // CTA (this round's token sends) is visible to every thread in this CTA
    // from this point on.
    __syncthreads();

    __shared__ bool shIsLastCta;

    // Steps 2-4: elect the last-finishing CTA in the grid and have it, once,
    // publish readiness to every peer.
    if (warpId == numWarps - 1) {
        // Step 2: one release-scoped atomic per CTA (not per responsible-expert
        // channel). Every CTA in the grid reaches this exactly once (including
        // ones with no responsible experts this round), so gridDim.x (==
        // numSms) is the correct total.
        int prevDone = 0;
        if (laneId == 0) {
            prevDone = atomic_add_release_global(ctaDone, 1);
        }
        prevDone = __shfl_sync(0xffffffff, prevDone, 0);
        const bool isLastCta = (prevDone + 1 == numSms);
        if (laneId == 0) {
            shIsLastCta = isLastCta;
        }

        if (isLastCta) {
            // Step 3: pairing acquire fence for the atomic release on ctaDone
            // above -- required to observe every other CTA's preceding writes
            // (their token sends) before publishing readiness for them.
            memory_fence_gpu();
            __syncwarp();

            // Step 4: publish this rank's writes for this round (now provably
            // visible, via the acquire above) to the whole system before
            // telling peers we're done, then one relaxed store per active
            // peer (see sendRankFinishFlagLsa).
            memory_fence_release_sys();
#pragma unroll 1
            for (int dstRank = laneId; dstRank < numRanks; dstRank += 32) {
                sendRankFinishFlagLsa(dstRank, currRank, recvFlagBuf, recvFlagOff, rankMask, windows, devComm);
            }

            if (laneId == 0) {
                *ctaDone = 0;
            }
        }
    }
    // Propagate shIsLastCta (and, transitively, the electing warp's own
    // fences above) from warpId == numWarps-1 to every other warp of this
    // CTA -- CTA-local only, not a grid-wide barrier.
    __syncthreads();
    return shIsLastCta;
}

// LSA-only, unquantized, rank-major-only LL combine entry point. See the
// scope-cuts doc comment at the top of this file.
template <int kHidden, int kNumTopk, int kNumMaxUnrolls, typename TopkIdxT, ncclDataType_t kTokenDtype>
__device__ __forceinline__ void combine_kernel_impl_2sided_rm_lsa( // INPUT
    const void* inData,
    const int* srcInfo,
    const TopkIdxT* inTopkIdx,
    int* rankMask,
    int* asyncErrorFlag,
    // OUTPUT
    void* outData,
    // INTERMEDIATE
    void* commBuf,
    size_t sendOffBase,
    size_t recvOffBase,
    size_t recvFlagOffBase,
    int* ctaDone,
    int nextRecvCntBufSize,
    int64_t* waitStats,
    LowLatencyEpochState* epochState,
    size_t payloadSlotStride,
    size_t signalSlotStride,
    // CONFIG
    int numCombinedTokens,
    int maxTokensPerRank,
    int numExperts,
    int currRank,
    int numRanks,
    int numWarpGroups,
    int numWarpsPerGroup,
    int phases,
    bool zeroCopy,
    ncclDevComm* devComm,
    const ncclWindow_t* windows,
    uint64_t timeoutCycles) {
    static constexpr ncclEpLayout_t kLayout = NCCL_EP_LAYOUT_RANK_MAJOR;
    constexpr int numTopk = kNumTopk;
    EP_STATIC_ASSERT(numTopk > 0, "numTopk must be positive");
    // Token dtype derivations
    constexpr int kElemBytes = (kTokenDtype == ncclFloat32) ? 4 : 2;

    const auto smId = __shfl_sync(0xffffffff, static_cast<int>(blockIdx.x), 0);
    const auto numSms = __shfl_sync(0xffffffff, static_cast<int>(gridDim.x), 0);
    const auto threadId = static_cast<int>(threadIdx.x);
    const auto numThreads = __shfl_sync(0xffffffff, static_cast<int>(blockDim.x), 0);
    const auto warpId = __shfl_sync(0xffffffff, threadId / 32, 0), laneId = get_lane_id();
    const auto numLocalExperts = numExperts / numRanks;
    const auto numWarps = numWarpGroups * numWarpsPerGroup;
    const auto warpGroupId = warpId / numWarpsPerGroup;
    const auto subWarpId = warpId % numWarpsPerGroup;
    const auto responsibleExpertIdx = smId * numWarpGroups + warpGroupId;

    const unsigned int epoch = selectLowLatencyEpoch(epochState, phases, smId, threadId);
    const size_t bank = static_cast<size_t>(epoch & 1U);
    const size_t bank_next = bank ^ 1U;
    const size_t sendOff = sendOffBase + bank * payloadSlotStride;
    const size_t recvOff = recvOffBase + bank * payloadSlotStride;
    const size_t recvFlagOff = recvFlagOffBase + bank * signalSlotStride;
    char* const commBase = static_cast<char*>(commBuf);
    void* const sendBuf = commBase + sendOff;
    void* const recvBuf = commBase + recvOff;
    int* const recvFlagBuf = reinterpret_cast<int*>(commBase + recvFlagOff);
    int* const nextRecvCntBuf = reinterpret_cast<int*>(commBase + recvFlagOffBase + bank_next * signalSlotStride);

    extern __shared__ __align__(1024) uint8_t smemBuffer[];

    // Data type staffs
    constexpr int kNumElemsPerInt4 = combine_smem::elements_per_int4(kElemBytes);
    constexpr int kWarpSize = combine_smem::kWarpSize;
    constexpr int64_t hiddenBf16Int4 = kHidden / kNumElemsPerInt4;

    // Use different unroll factors for send and recv phases
    constexpr int kNumSendUnrolls = combine_smem::send_unrolls(kHidden, kElemBytes);
    constexpr int kNumRecvUnrolls = combine_smem::kNumRecvUnrolls;
    constexpr int hiddenBf16Int4Pad = align(static_cast<int>(hiddenBf16Int4), kWarpSize * kNumSendUnrolls);
    EP_STATIC_ASSERT(kNumSendUnrolls <= kNumMaxUnrolls and kNumRecvUnrolls <= kNumMaxUnrolls, "Invalid unrolls");
    EP_STATIC_ASSERT(hiddenBf16Int4 % kNumSendUnrolls == 0, "Invalid hidden");
    EP_STATIC_ASSERT(kNumSendUnrolls >= kNumRecvUnrolls, "Invalid unroll factors");

    // Message package: the NONE-recipe wire slot size, including its
    // reserved (here, always-unused) LogFMT metadata tail -- see the
    // top-of-file doc comment for why that tail stays.
    constexpr size_t numBytesPerSlot = align(
        static_cast<size_t>(kHidden * kElemBytes +
                            kHidden / combine_smem::kLogFmtElementsPerMeta * sizeof(nv_bfloat162)),
        sizeof(int4));
    EP_STATIC_ASSERT(numBytesPerSlot % sizeof(int4) == 0, "Invalid vectorization");

    // Set by syncAndSendFinishLsa below; declared (and default-initialized)
    // here so it's still in scope, and safely readable, at the RECV label
    // below even for a RECV-only call that jumps straight past the SEND
    // phase via the goto immediately below.
    bool isLastCta = false;

    // Populated by waitForAllRanksFinishLsa below (RECV phase); see its doc
    // comment. Declared ahead of the SEND/RECV goto, like isLastCta above,
    // purely for scope.
    __shared__ uint32_t activeRankMaskShared[combine_smem::kActiveRankMaskWords];

    // Sending phase
    if ((phases & LOW_LATENCY_SEND_PHASE) == 0) {
        goto LOW_LATENCY_COMBINE_LSA_RECV;
    }

    // Clean up next buffer. ctaDone (syncAndSendFinishLsa's own CTA-arrival
    // counter, below) is untouched here -- nothing to notify.
    if (smId == 0 and warpGroupId == 0 and subWarpId == 0) {
        cleanNextRecvCntBufAndNotifyLsa(nextRecvCntBuf, nextRecvCntBufSize, laneId);
    }

    // Issue tokens sending
    if (responsibleExpertIdx < numExperts) {
        const auto dstRank = responsibleExpertIdx / numLocalExperts;
        // Each pair of rank establish numLocalExperts channels for parallelization
        const auto rankLaneIdx = responsibleExpertIdx % numLocalExperts;

        // Read # of tokens received from this rank and set it's source info
        const auto numRecvTokens = __shfl_sync(0xffffffff, __ldg(srcInfo + dstRank), 0);

        auto slotsPerToken = numTopk + 1; // token_id + topk_idx's
        // We have numRanks entries with per-rank count first, then the rest is (tokenId + numTopk) entries for each token
        const auto localSrcInfo = srcInfo + numRanks + dstRank * maxTokensPerRank * slotsPerToken;

        // TMA stuffs
        constexpr int kNumTMABufferBytes = sizeof(int4) * kWarpSize * kNumSendUnrolls;
        constexpr int kNumStages = combine_smem::kNumStages;
        constexpr int kNumPrefetch = combine_smem::kNumTmaPrefetch;
        EP_STATIC_ASSERT(kNumStages == 3 and kNumPrefetch == 1, "Invalid stages");

        auto smemPtr =
            smemBuffer + warpId * (kNumStages * (kNumTMABufferBytes + combine_smem::kTmaBarrierAndPaddingBytes));
        uint32_t tmaPhase = 0;
        auto tmaBuffers = PatternVisitor([=](const int& i) {
            return reinterpret_cast<int4*>(
                smemPtr + i * (kNumTMABufferBytes + combine_smem::kTmaBarrierAndPaddingBytes));
        });
        auto fullBarriers = PatternVisitor([=](const int& i) {
            return reinterpret_cast<uint64_t*>(
                smemPtr + i * (kNumTMABufferBytes + combine_smem::kTmaBarrierAndPaddingBytes) + kNumTMABufferBytes);
        });
        EP_STATIC_ASSERT(kNumSendUnrolls * kNumStages <= 12, "TMA buffer size exceed limit");

        // Initialize m-barriers
        if (laneId < kNumStages) {
            mbarrier_init(fullBarriers[laneId], 1);
            fence_barrier_init();
        }
        __syncwarp();

        auto tmaLoadAndArrive = [&](const int& stageIdx, const int4* gmemPtr, const int& numBytes) {
            tma_load_1d(tmaBuffers[stageIdx], gmemPtr, fullBarriers[stageIdx], numBytes);
            mbarrier_arrive_and_expect_tx(fullBarriers[stageIdx], numBytes);
        };
        auto getNumTmaBytes = [&](const int& offsetInt4) {
            return min(kNumTMABufferBytes, static_cast<int>((hiddenBf16Int4 - offsetInt4) * sizeof(int4)));
        };

        // Issue sends for each token from the responsible dstRank. Rank-major:
        // one send per received slot. srcInfo[0] = linear slot into inData;
        // srcInfo[1] = j_eff (topk return index). The receive slot is
        // (tokenIdx * numTopk + j_eff) * numBytesPerSlot, placing each expert
        // rank's contribution into a distinct position in the recv buffer.
        if (not isRankMasked<true>(rankMask, dstRank)) {
            for (int i = subWarpId * numLocalExperts + rankLaneIdx; i < numRecvTokens;
                 i += numWarpsPerGroup * numLocalExperts) {
                auto localSrcTokenInfo = localSrcInfo + i * slotsPerToken;
                auto localSrcTopkInfo = localSrcTokenInfo + 1;
                int tokenIdx = __shfl_sync(0xffffffff, __ldg(localSrcTokenInfo), 0);

                int slot = __shfl_sync(0xffffffff, __ldg(localSrcTopkInfo + 0), 0);
                int j_eff = (numTopk > 1) ? __shfl_sync(0xffffffff, __ldg(localSrcTopkInfo + 1), 0) : 0;
                if (j_eff < 0) continue; // no local expert found (shouldn't happen)

                // Byte offsets scale with slot (= srcRank*maxTokensPerRank + i)
                // and tokenIdx*numTopk, which times numBytesPerSlot can exceed
                // INT_MAX -- use size_t to avoid 32-bit truncation.
                size_t sndTokenOffset = static_cast<size_t>(slot) * numBytesPerSlot;
                size_t rcvTokenOffset = static_cast<size_t>(tokenIdx * numTopk + j_eff) * numBytesPerSlot;

                const auto srcDataInt4Ptr = static_cast<const int4*>(inData) + (int64_t)slot * hiddenBf16Int4;
                const auto sendBufUint8 = static_cast<uint8_t*>(sendBuf) + sndTokenOffset;
                const auto sendBufPtr = reinterpret_cast<int4*>(sendBufUint8);
                const auto recvPtr = reinterpret_cast<uint64_t>(recvBuf) + rcvTokenOffset;
                const auto expectedDstOffset = recvOff + rcvTokenOffset;
                const auto dstP2pPtr =
                    ncclGetP2pPtr(recvPtr, expectedDstOffset, currRank, dstRank, windows, devComm);

                processAndSendTokenLsa<kNumSendUnrolls, kNumStages, kNumPrefetch>(
                    srcDataInt4Ptr,
                    sendBufPtr,
                    dstP2pPtr,
                    hiddenBf16Int4,
                    hiddenBf16Int4Pad,
                    zeroCopy,
                    tmaBuffers,
                    fullBarriers,
                    tmaPhase,
                    tmaLoadAndArrive,
                    getNumTmaBytes,
                    laneId);
            }
        }
        // Destroy m-barriers (per-warp, no cross-warp dependency -- each
        // warp owns its own m-barrier region via smemPtr's warpId offset
        // above, so no bar.sync is needed to gate this).
        if (laneId < kNumStages) {
            mbarrier_inval(fullBarriers[laneId]);
            fence_barrier_init();
        }
        __syncwarp();
    
        // Wait for all stores to complete
        tma_store_wait_complete<0>();
    }

    // Every CTA that had a responsible expert (or none at all) has now
    // finished its own sends. Elect the last-finishing CTA and have it
    // broadcast readiness to every peer, replacing the old per-channel
    // atomicCleanFlag countdown gate + per-channel sendFinishFlagLsa above.
    isLastCta = syncAndSendFinishLsa(
        warpId, numWarps, laneId, ctaDone, numSms, numRanks, currRank, recvFlagBuf, recvFlagOff, rankMask,
        windows, devComm);

// Receiving phase
LOW_LATENCY_COMBINE_LSA_RECV:
    if ((phases & LOW_LATENCY_RECV_PHASE) == 0) {
        return;
    }

    // Split invocation (SEND phase didn't run here): syncAndSendFinishLsa
    // never elected anyone, so nominate smId == 0 instead -- same fallback
    // as dispatch_kernel_impl_2sided_rm_lsa.
    if ((phases & LOW_LATENCY_SEND_PHASE) == 0) {
        isLastCta = (smId == 0);
    }

    // Cross-rank readiness: every CTA independently confirms every active
    // peer has broadcast readiness (dispatch's syncAndRecvCounts pattern,
    // reused as-is via waitForAllRanksFinishLsa since neither cares about
    // the flag's value, only that it's non-zero) -- no grid-wide barrier
    // needed for this, unlike the old waitForRecvFlagLsa (which was keyed
    // per (srcRank, localExpert) and relied on a cg::this_grid().sync() to
    // propagate readiness). completeFullLowLatencyEpoch below relies on
    // isLastCta (elected above, or the split-invocation fallback) rather
    // than the "smId == 0" convention, which would have no ordering
    // guarantee against other CTAs' epoch reads without a grid-wide barrier.
    waitForAllRanksFinishLsa(
        threadId, numRanks, rankMask, activeRankMaskShared, recvFlagBuf, timeoutCycles, currRank, asyncErrorFlag,
        waitStats);
    completeFullLowLatencyEpoch(epochState, phases, isLastCta, threadId);

    // Reassign warp groups; FP32 doubles SMEM per group, drop to 1 group to stay within limits
    constexpr int kMaxNumGroups = combine_smem::max_recv_groups(kElemBytes);
    const int numDecodeWarps = hiddenBf16Int4Pad / (kNumRecvUnrolls * kWarpSize);
    const int warpsPerRecvGroup = numDecodeWarps + combine_smem::kNumRecvTmaWarps;
    const int numGroups = min(kMaxNumGroups, (numThreads / kWarpSize) / warpsPerRecvGroup);
    const int decodeWarpIdx = __shfl_sync(0xffffffff, warpId % warpsPerRecvGroup, 0);
    const int groupIdx = __shfl_sync(0xffffffff, warpId / warpsPerRecvGroup, 0);
    EP_DEVICE_ASSERT(numTopk <= kWarpSize);
    EP_DEVICE_ASSERT(numGroups > 0);

    if (groupIdx < numGroups) {
        constexpr int kNumStages = combine_smem::kNumStages;
        constexpr int kNumTMABufferBytes = combine_smem::recv_tma_buffer_bytes(kHidden, kElemBytes, /*nvfp4=*/false);
        constexpr int kNumBF16PerWarpBytes = kWarpSize * kNumRecvUnrolls * kNumElemsPerInt4 * kElemBytes;
        constexpr int kPaddedTokenBytes = hiddenBf16Int4Pad * sizeof(int4);
        // No LogFMT metadata region (logAmax/logAmin/castInfo) -- nothing
        // populates or consumes it once LogFMT is gone.
        constexpr int kNumBytesPerGroup = kNumStages * kNumTMABufferBytes + kPaddedTokenBytes;

        // Reallocate shared memory
        const auto smemGroupBuffer = smemBuffer + kNumBytesPerGroup * groupIdx;
        auto fullBarriers = PatternVisitor(
            [=](const int& i) { return reinterpret_cast<uint64_t*>(smemGroupBuffer + i * kNumTMABufferBytes); });
        auto emptyBarriers = PatternVisitor([=](const int& i) {
            return reinterpret_cast<uint64_t*>(
                smemGroupBuffer + i * kNumTMABufferBytes + combine_smem::kRecvTmaEmptyBarrierOffsetBytes);
        });
        auto tmaLdBuffers = PatternVisitor([=](const int& i) {
            return reinterpret_cast<uint8_t*>(
                smemGroupBuffer + i * kNumTMABufferBytes + combine_smem::kRecvTmaPayloadOffsetBytes);
        });
        auto tmaStBuffers = PatternVisitor([=](const int& i) {
            return reinterpret_cast<uint32_t*>(
                smemGroupBuffer + kNumStages * kNumTMABufferBytes + i * kNumBF16PerWarpBytes);
        });

        uint32_t tmaPhase = 0;
        EP_STATIC_ASSERT(kNumStages < kWarpSize, "Too many stages");
        if (decodeWarpIdx == numDecodeWarps) tmaPhase = (1 << kNumStages) - 1;

        // Initialize m-barriers
        if (decodeWarpIdx == numDecodeWarps and laneId < kNumStages) {
            mbarrier_init(fullBarriers[laneId], 1);
            mbarrier_init(emptyBarriers[laneId], numDecodeWarps);
        }
        syncSmGroup(groupIdx + 1, warpsPerRecvGroup * kWarpSize);

        int stageIdx = 0, topkIdxByLane = 0;
        EP_STATIC_ASSERT(
            kNumTopk <= kWarpSize,
            "numTopk must not exceed warp size: warpFirstTopKOccurence relies on active "
            "lanes (0..numTopk-1) having lower IDs than inactive lanes (numTopk..31)");

        if (decodeWarpIdx == numDecodeWarps) {
            // TMA load warp
            for (int tokenIdx = smId + numSms * groupIdx; tokenIdx < numCombinedTokens;
                 tokenIdx += numSms * numGroups) {
                if (laneId < numTopk) topkIdxByLane = static_cast<int>(__ldg(inTopkIdx + tokenIdx * numTopk + laneId));

                topkIdxByLane = warpFirstTopKOccurence(topkIdxByLane, numLocalExperts, laneId);

                for (int i = 0; i < numTopk; ++i) {
                    int topkIdxReg = __shfl_sync(0xffffffff, topkIdxByLane, i);
                    if (topkIdxReg < 0) continue;
                    if (not isRankActiveLocal(activeRankMaskShared, topkIdxReg / numLocalExperts)) continue;

                    mbarrier_wait<true>(emptyBarriers[stageIdx], tmaPhase, stageIdx);
                    auto recvBufData = static_cast<uint8_t*>(recvBuf) + (tokenIdx * numTopk + i) * numBytesPerSlot;

                    if (elect_one_sync()) {
                        const int numTmaBytes = kHidden * kElemBytes;
                        tma_load_1d(tmaLdBuffers[stageIdx], recvBufData, fullBarriers[stageIdx], numTmaBytes);
                        mbarrier_arrive_and_expect_tx(fullBarriers[stageIdx], numTmaBytes);
                    }
                    __syncwarp();
                    stageIdx = (stageIdx + 1) % kNumStages;
                }
            }
        } else {
            // Reduction warps
            for (int tokenIdx = smId + numSms * groupIdx; tokenIdx < numCombinedTokens;
                 tokenIdx += numSms * numGroups) {
                if (laneId < numTopk) {
                    topkIdxByLane = static_cast<int>(__ldg(inTopkIdx + tokenIdx * numTopk + laneId));
                }
                __syncwarp();

                topkIdxByLane = warpFirstTopKOccurence(topkIdxByLane, numLocalExperts, laneId);

                combineReduceTokenLsa<kNumStages, kNumElemsPerInt4, kNumRecvUnrolls, kTokenDtype>(
                    tokenIdx, numTopk, laneId, decodeWarpIdx, kNumBF16PerWarpBytes, hiddenBf16Int4,
                    activeRankMaskShared, fullBarriers, emptyBarriers, tmaLdBuffers, tmaStBuffers, tmaPhase, stageIdx,
                    outData,
                    [&](int i) {
                        const int topkIdxReg = __shfl_sync(0xffffffff, topkIdxByLane, i);
                        return topkIdxReg < 0 ? -1 : topkIdxReg / numLocalExperts;
                    });
            }
        }
    }
}

} // namespace ll

} // namespace nccl_ep
