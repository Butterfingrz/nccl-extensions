/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 * See LICENSE.txt for more license information.
 */

#pragma once

#include "../device_primitives.cuh"
#include "nccl_device.h"
#include "../ll_ep_adapter.cuh"
#include "ll_common.cuh"
#include "ll_dispatch_recipe.cuh"
#include "ll_lsa_primitives.cuh"
#include "ll_mask.cuh"

// LSA-only LL dispatch for unquantized and fused DS-FP8E3M4 payloads. Every
// destination is assumed NVLink-reachable (same LSA team) -- there is no RDMA/GIN
// fallback anywhere in this file, unlike the general dispatch_kernel_impl in
// ll_ep.cuh. Keeping the supported recipe set deliberately narrow lets this path
// retain its single-signal completion protocol without carrying QUANT_FWD,
// zero-copy scale-window, or RDMA-fallback plumbing.

#define SYNC_DISP_LSA_SEND_COPY 1

namespace nccl_ep {

namespace ll {

__forceinline__ __device__ void cleanNextRecvCntBufLsa(int* nextRecvCntBuf, int nextRecvCntBufSize, int offset, int stride) {
#pragma unroll
    for (int i = offset; i < nextRecvCntBufSize; i += stride) nextRecvCntBuf[i] = 0;
}

// Copy the split-layout routing header and return the peer payload slot.
// Both send paths use the same per-source-rank addressing and header stores.
__forceinline__ __device__ uint8_t* sendTokenHeaderLsa(
    uint64_t dstSrcRankP2pPtr,
    const int4* sendDataInt4,
    int slotIdx,
    size_t numBytesPerMsg,
    size_t dispatch_hdr_sz,
    int maxTokensPerRank,
    int laneId) {
    const size_t hdrSectionBytes = static_cast<size_t>(maxTokensPerRank) * dispatch_hdr_sz;
    const size_t payloadBytes = numBytesPerMsg - dispatch_hdr_sz;
    const int numHdrInt4 = static_cast<int>(dispatch_hdr_sz / sizeof(int4));

    auto* dstSrcRankBase = reinterpret_cast<uint8_t*>(dstSrcRankP2pPtr);
    auto* dstHdrSlot = dstSrcRankBase + slotIdx * dispatch_hdr_sz;
    auto* dstPayloadSlot = dstSrcRankBase + hdrSectionBytes + slotIdx * payloadBytes;

    int4* dstHdrInt4 = reinterpret_cast<int4*>(dstHdrSlot);
    for (int i = laneId; i < numHdrInt4; i += 32) {
        st_na_global(dstHdrInt4 + i, sendDataInt4[i]);
    }
    return dstPayloadSlot;
}

// Intra-LSA dispatch send (NVLink direct path, split layout, unquantized): header
// copied from the local staging slot, payload cast directly from `srcData` into
// the peer's per-srcRank region (or straight into the peer's output window when
// zero-copy is enabled). Takes the already-resolved peer pointer; the caller is
// responsible for guaranteeing NVLink reachability (this file has no fallback).
__forceinline__ __device__ void sendTokenLsa(
    uint64_t dstSrcRankP2pPtr, // already-resolved pointer into peer's per-srcRank region.
    const int4* sendDataInt4, // local staging slot base (header).
    const int4* srcData, // input data for this token.
    int slotIdx, // slot within the per-srcRank region.
    size_t numBytesPerMsg,
    size_t dispatch_hdr_sz,
    size_t hiddenBytes,
    size_t hiddenInt4,
    int maxTokensPerRank,
    int dstRank,
    int currRank,
    // Zero-copy output: direct NVLink writes may target the peer's output window.
    ncclWindow_t recvDataWindow,
    size_t recvDataOffset,
    int laneId) {
    void* payloadDst = sendTokenHeaderLsa(
        dstSrcRankP2pPtr, sendDataInt4, slotIdx, numBytesPerMsg,
        dispatch_hdr_sz, maxTokensPerRank, laneId);
    if (recvDataWindow != ncclWindow_t{}) {
        const size_t recvSlot = static_cast<size_t>(currRank) * maxTokensPerRank + slotIdx;
        payloadDst = ncclGetPeerPointer(recvDataWindow, recvDataOffset + recvSlot * hiddenBytes, dstRank);
    }
    auto* dstDataVec = reinterpret_cast<int4*>(payloadDst);

    // Unquantized copy: plain per-lane vector store, no recipe branching.
    for (int i = laneId; i < static_cast<int>(hiddenInt4); i += 32) {
        dstDataVec[i] = __ldg(srcData + i);
    }
}

// DS-FP8 NVLink send. Reuse the prequantized CTA payload when supplied;
// otherwise quantize BF16 input directly into the peer slot.
__forceinline__ __device__ void sendTokenDsFp8Lsa(
    uint64_t dstSrcRankP2pPtr,
    const int4* sendDataInt4,
    const int4* srcData,
    int slotIdx,
    size_t numBytesPerMsg,
    size_t dispatch_hdr_sz,
    size_t hiddenBytes,
    size_t inputHiddenInt4,
    int scaleBytes,
    int maxTokensPerRank,
    bool roundScale,
    int laneId,
    const int2* quantized,
    const float* quantizedScales) {
    auto* dstPayloadSlot = sendTokenHeaderLsa(
        dstSrcRankP2pPtr, sendDataInt4, slotIdx, numBytesPerMsg,
        dispatch_hdr_sz, maxTokensPerRank, laneId);
    constexpr int kElementsPerRead = sizeof(int4) / sizeof(nv_bfloat16);
    EP_STATIC_ASSERT(kElementsPerRead * 32 / kDsFp8E3M4ElementsPerScale == 2,
                     "Invalid DS_FP8E3M4 vectorization");
    auto* dstDataVec = reinterpret_cast<int2*>(dstPayloadSlot);
    auto* dstScales = reinterpret_cast<float*>(dstPayloadSlot + hiddenBytes);
    EP_DEVICE_ASSERT(scaleBytes ==
                     static_cast<int>(inputHiddenInt4 * kElementsPerRead /
                                      kDsFp8E3M4ElementsPerScale * sizeof(float)));

    if (quantized != nullptr) {
        // Payload slots and shared staging are 16-byte aligned. Copy two
        // quantization vectors at a time without changing the wire layout.
        auto* dstDataInt4 = reinterpret_cast<int4*>(dstPayloadSlot);
        const auto* quantizedInt4 = reinterpret_cast<const int4*>(quantized);
        for (int i = laneId; i < static_cast<int>(hiddenBytes / sizeof(int4)); i += 32) {
            dstDataInt4[i] = quantizedInt4[i];
        }
        for (int i = laneId; i < scaleBytes / static_cast<int>(sizeof(float)); i += 32) {
            dstScales[i] = quantizedScales[i];
        }
    } else {
        castAndWriteToSendBuf<NCCL_EP_DISP_QUANT_DS_FP8E3M4, float>(
            srcData, dstDataVec, dstScales, laneId, 32, laneId,
            static_cast<int>(inputHiddenInt4), roundScale);
    }
}

// Copies a received token and, for DS_FP8E3M4, its generated FP32 scales.
template <ncclEpDispQuant_t kRecipe>
__forceinline__ __device__ void copyRecvTokenDataLsa(
    const uint8_t* recvBufUint8,
    int recvIdx,
    int tokenIdx,
    int4* outDataInt4,
    void* outScales,
    int hiddenInt4,
    int hiddenBytes,
    int scaleBytes,
    int numBytesPerMsg,
    int dispatch_hdr_sz,
    int maxTokensPerRank,
    int laneId) {
    const int payloadBytes = numBytesPerMsg - dispatch_hdr_sz;
    const uint8_t* recvPayloadPtr = recvBufUint8 + maxTokensPerRank * dispatch_hdr_sz + recvIdx * payloadBytes;
    const auto recvDataInt4 = reinterpret_cast<const int4*>(recvPayloadPtr);
    const auto outDataInt4Ptr = outDataInt4 + tokenIdx * hiddenInt4;
    UNROLLED_WARP_COPY(7, laneId, hiddenInt4, outDataInt4Ptr, recvDataInt4, ld_nc_global, st_na_global);

    if constexpr (kRecipe == NCCL_EP_DISP_QUANT_DS_FP8E3M4) {
        const int numScales = scaleBytes / sizeof(float);
        const auto* recvScales = reinterpret_cast<const float*>(recvPayloadPtr + hiddenBytes);
        auto* outScalesTyped = static_cast<float*>(outScales) + tokenIdx * numScales;
        for (int scaleIdx = laneId; scaleIdx < numScales; scaleIdx += 32) {
            outScalesTyped[scaleIdx] = ld_nc_global(recvScales + scaleIdx);
        }
    }
}

// LSA-only peer token-count publish: every destination is NVLink-reachable,
// so this is always a direct P2P store. Single write per round, into a
// single per-srcRank slot -- unlike GIN, which spreads a rank's completion
// signal across numLocalExperts channels/QPs, plain NVLink P2P has no reason
// to replicate the write. A relaxed store: the caller (the elected last CTA,
// see dispatch_kernel_impl_2sided_rm_lsa below) already issued a system-scope
// release fence before calling this for any peer, so this store's own
// ordering doesn't need to do anything further -- it only needs to become
// observable. The stored value doubles as the peer's readiness flag (decoded
// by the receiver as `-value - 1`; 0 means "not arrived yet").
__forceinline__ __device__ void sendPeerTokenCountLsa(
    int numTokensSent,
    int dstRank,
    int currRank,
    int* recvCntBufLocal,
    size_t recvCntOff,
    int* rankMask,
    const ncclWindow_t* windows,
    ncclDevComm* devComm) {
    const uint64_t recvCntPtr = reinterpret_cast<uint64_t>(recvCntBufLocal + currRank);
    const size_t recvCntOffset = recvCntOff + static_cast<size_t>(currRank) * sizeof(int);
    const auto dstP2pPtr = ncclGetP2pPtr(recvCntPtr, recvCntOffset, currRank, dstRank, windows, devComm);
    if (not isRankMasked(rankMask, dstRank)) {
        EP_DEVICE_ASSERT(dstP2pPtr != 0);
        st_relaxed_sys_global(reinterpret_cast<int*>(dstP2pPtr), -numTokensSent - 1);
    }
}

// Steps 1-4 of the LSA dispatch send-side completion protocol, shared by
// every dispatch_kernel_impl_*_rm_lsa variant regardless of send-phase grid
// model: a CTA-scope barrier, then electing the last-finishing CTA in the
// grid (one release-scoped atomic per CTA, not per store) to publish this
// round's per-destination counts to every peer. Every warp of every CTA must
// call this (only the control warp, warpId == numWarps-1, actually does
// anything past the barrier; the others just pass through it).
//
// ctaDone counts CTAs (== SMs, one CTA per SM in this grid model), not ranks
// -- despite living in the same host-allocated workspace region as the
// general kernel's args.rankDone (which genuinely is per-rank there; see
// dispatch_kernel_impl in ll_ep.cuh), this file only ever touches index 0 of
// it, as a scalar "how many CTAs have finished SEND" census counter. Named
// for what it counts in this file specifically, not for the shared field's
// name at the call site.
//
// The elected last CTA also resets rankSentCnt[]/ctaDone[0] to 0 for the next
// epoch, right here, instead of every CTA separately resetting its own
// rankSentCnt[responsibleExpertIdx]/rankDone[responsibleExpertIdx] slot back
// in the RECV phase (this file's earlier design). That RECV-phase reset was
// exactly why the SEND/RECV phase boundary used to need a
// cg::this_grid().sync() (or, briefly, a software wait on this same
// counter): rankSentCnt is still being atomicAdd'd into by whichever CTAs
// haven't finished their SEND-phase token loop yet, so *any* CTA racing
// ahead to reset it -- before every other CTA's atomicAdd's are done -- can
// lose a concurrent increment or hand out a colliding slotIdx. The elected
// last CTA doesn't have that problem: by construction (prevDone + 1 ==
// numSms, i.e. every CTA already did its own atomic_add_release_global into
// ctaDone, paired with this CTA's own acquire fence below) it already knows
// every other CTA's SEND-phase writes -- including their rankSentCnt
// atomicAdd's -- are complete and visible, before it even starts reading
// rankSentCnt to publish it to peers. So resetting immediately afterward,
// still inside this same elected CTA, needs no additional synchronization:
// no other CTA ever touches rankSentCnt/ctaDone again this epoch, and this
// reset -- being part of this kernel invocation -- is guaranteed complete
// before the next epoch's SEND phase (a later kernel launch on the same
// stream) issues its own atomicAdd's into the same memory. No CTA needs to
// wait on anything here at all, unlike a grid-wide barrier (which stalls
// every SM) or a software poll (which still occupies an SM spinning).
// rankSentCnt/ctaDone are otherwise untouched by the RECV phase (nothing
// reads them there), so this fully replaces the RECV-phase reset -- callers
// must not also reset them.
//
// Returns whether this CTA was the elected last one, broadcast to every warp
// via shared memory + a trailing CTA-scope __syncthreads() (cheap and purely
// intra-CTA, unlike the grid-wide barrier this replaced). Callers pass this
// straight to the completeFullLowLatencyEpoch(..., bool isElectedCta, ...)
// overload in ll_common.cuh, which performs the epoch bump on exactly this
// CTA instead of the other overload's "smId == 0" convention (which, without
// a grid-wide barrier, would have no guarantee every other CTA is done).
__forceinline__ __device__ bool syncAndSendCounts(
    int warpId,
    int numWarps,
    int laneId,
    int* ctaDone,
    int numSms,
    int numRanks,
    int* rankSentCnt,
    int currRank,
    int* recvCntBuf,
    size_t recvCntOff,
    int* rankMask,
    const ncclWindow_t* windows,
    ncclDevComm* devComm) {

    // Step 1: CTA-scope barrier (also a full memory fence at CTA scope, per
    // bar.sync semantics) -- every write issued above by any warp of this
    // CTA (header/payload stores to peer memory, sentinel clears) is visible
    // to every thread in this CTA from this point on.
    __syncthreads();

    __shared__ bool shIsLastCta;

    // Steps 2-4: elect the last-finishing CTA in the grid and have it, once,
    // publish this round's per-destination counts to every peer, instead of
    // every CTA independently signaling completion per (srcRank, localExpert)
    // channel -- that per-channel replication exists in the general kernel
    // only to spread completion across multiple GIN QPs; plain NVLink P2P
    // has no such requirement, so one signal per peer suffices.
    if (warpId == numWarps - 1) {

        // Step 2: one release-scoped atomic per CTA (not per store, per
        // token). atom.add.release.gpu ensures every write issued by any
        // warp of this CTA above is ordered before this counter update
        // w.r.t. any other CTA on this GPU. Every CTA in the grid reaches
        // this exactly once (including ones that processed zero tokens when
        // numTokens < numSms), so gridDim.x (== numSms) is the correct total.
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

            // Step 3.1 Pairing Acquire to other CTAs' 
            // atomic_add_release_global(ctaDone, 1) above.
            memory_fence_gpu();

            // Step 3.2 Transitively observe the updates made visible to lane0 by the
            // above acquire GPU-wide fence
            __syncwarp();


            // Step 3.3 Publish all memory accesses observed by this rank 
            // to the whole system before telling peers that the data is ready.
            memory_fence_release_sys();

            // Step 4. Inform peers using relaxed store that the data is ready.
#pragma unroll 1
            for (int dstRank = laneId; dstRank < numRanks; dstRank += 32) {
                sendPeerTokenCountLsa(
                    rankSentCnt[dstRank], dstRank, currRank, recvCntBuf, recvCntOff, rankMask, windows, devComm);
            }

            // Reset control counters for the next epoch
            // No other CTA touches either array again this epoch, so it is safe to reset.
#pragma unroll 1
            for (int dstRank = laneId; dstRank < numRanks; dstRank += 32) {
                rankSentCnt[dstRank] = 0;
            }
            if (laneId == 0) {
                // Only index 0 is ever read (the scalar census counter this
                // function itself maintains via atomic_add_release_global
                // above); the rest of the shared array is never read
                // anywhere in this file, so there is nothing else to reset.
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

// REceive-side completion protocol:
// each CTA's thread independently confirms every active peer's
// token count has landed (relaxed poll on the count-doubles-as-flag
// buffer, with a timeout that masks off an unresponsive peer),
// acquire-fences those now-confirmed reads (and,
// transitively, the peer's payload writes ordered before it by the peer's
// own release fence), then propagates that visibility to the rest of the
// CTA via __syncthreads(). Must be called by every thread of the CTA.
__forceinline__ __device__ void syncAndRecvCounts(
    int threadId,
    int numRanks,
    int* rankMask,
    const int* recvCntBuf,
    uint64_t timeoutCycles,
    int currRank,
    int* asyncErrorFlag) {
    // Step 5: every CTA's thread 0 independently confirms every active
    // peer's token count has landed. Relaxed poll -- ordering is established
    // by the acquire fence in step 6 below, not by this load itself.
    if (threadId == 0) {
        const auto startTime = clock64();
        for (int peer = 0; peer < numRanks; peer++) {
            if (isRankMasked(rankMask, peer)) continue;
            uint64_t waitCost = 0;
            bool flagReady;
            do {
                flagReady = ld_relaxed_sys_global(reinterpret_cast<const uint32_t*>(recvCntBuf + peer)) != 0;
                waitCost = clock64() - startTime;
            } while (!flagReady && waitCost <= timeoutCycles);
            if (waitCost > timeoutCycles) {
                printf("Warning: NCCL EP timeout for dispatch receive, rank %d, src_rank %d\n", currRank, peer);
                if (rankMask == nullptr) trap();
                atomicExch(rankMask + peer, 0);
                if (asyncErrorFlag != nullptr) atomicExch_system(asyncErrorFlag, 1);
            }
        }
        // Step 6: acquire-fence this thread's now-confirmed reads (and,
        // transitively, the peer's payload writes ordered before it by the
        // peer's own release fence) before any thread in this CTA touches
        // recvBuf.
        memory_fence();
    }
    // Step 7: propagate that visibility to the rest of the CTA.
    __syncthreads();
}

// Record source information for every
// token received from `responsibleExpertIdx`'s source rank. Per-(srcRank,
// localExpert) warp-group assignment is unchanged from the general kernel (a
// global load-balanced rework is a separate follow-up); what changed is how
// readiness/count is established -- decoded directly from recvCntBuf (see
// syncAndRecvCounts above), no more per-channel wait or rankArrivedCnt
// rendezvous. Rank-major only: hardcodes DispatchHdr<NCCL_EP_LAYOUT_RANK_MAJOR>,
// matching every dispatch_kernel_impl_*_rm_lsa caller.
template <ncclEpDispQuant_t kRecipe>
__forceinline__ __device__ void recordRecvTokensLsa(
    int responsibleExpertIdx,
    int numExperts,
    int numLocalExperts,
    int laneId,
    int subWarpId,
    int numWarpsPerGroup,
    int numTopk,
    int numRanks,
    int currRank,
    int maxTokensPerRank,
    size_t numBytesPerMsg,
    size_t dispatch_hdr_sz,
    size_t hiddenInt4,
    size_t hiddenBytes,
    int scaleBytes,
    int* rankMask,
    const int* recvCntBuf,
    const void* recvBuf,
    int* outSrcInfo,
    int* outRecvRankCounter,
    int32_t* outRecvTopkIdx,
    float* outRecvTopkWeights,
    void* outDataBuf,
    void* outScalesBuf,
    ncclEpExpertIdKind_t recvTopkIdxKind,
    ncclWindow_t recvDataWindow) {
    if (responsibleExpertIdx >= numExperts) return;

    const auto srcRank = responsibleExpertIdx / numLocalExperts;
    const auto rankLaneIdx = responsibleExpertIdx % numLocalExperts;
    const auto globalExpertStartIdx = currRank * numLocalExperts;

    const int encoded = recvCntBuf[srcRank];
    const int numRecvTokens = isRankMasked(rankMask, srcRank) ? 0 : (-encoded - 1);
    if (laneId == 0 and rankLaneIdx == 0) {
        outSrcInfo[srcRank] = numRecvTokens;
        if (outRecvRankCounter != nullptr) outRecvRankCounter[srcRank] = numRecvTokens;
    }

    const uint8_t* const srcRankRecvBase = reinterpret_cast<const uint8_t*>(recvBuf) +
        static_cast<size_t>(relativeRankSlot(srcRank, currRank, numRanks)) * maxTokensPerRank * numBytesPerMsg;

    // Rank-major: one output slot per received token. outRecvTopkIdx/Weights
    // are written so the caller can route and reduce; zero-copy (when
    // recvDataWindow is bound) skips the token data copy entirely, since
    // the payload was already written directly into the peer's output
    // window during the send phase (sendTokenLsa).
    for (int i = rankLaneIdx * numWarpsPerGroup + subWarpId; i < numRecvTokens;
         i += numWarpsPerGroup * numLocalExperts) {
        const auto recvBufUint8 = srcRankRecvBase;
        const int slot = srcRank * maxTokensPerRank + i;
        const auto recvBufHdr = reinterpret_cast<const DispatchHdr<NCCL_EP_LAYOUT_RANK_MAJOR>*>(
            recvBufUint8 + i * dispatch_hdr_sz);
        const auto rtr = recvBufHdr->rtr;

        const auto recvSrcInfoBase = outSrcInfo + numRanks;
        const auto slotsPerToken = numTopk + 1;
        const auto recvSrcInfo = recvSrcInfoBase + (srcRank * maxTokensPerRank + i) * slotsPerToken;
        const auto recvSrcTopkInfo = recvSrcInfo + 1;
        if (laneId == 0) {
            recvSrcInfo[0] = recvBufHdr->token_id;
            recvSrcTopkInfo[0] = slot;
        }
        if (numTopk > 1) {
            bool matchesCurrRank = (laneId < numTopk) &&
                (getExpertRankIdx((int)rtr[laneId].expert_id, numLocalExperts) == currRank);
            uint32_t currRankMask = __ballot_sync(0xffffffff, matchesCurrRank);
            int j_eff = currRankMask ? (__ffs(currRankMask) - 1) : -1;
            if (laneId == 0) {
                recvSrcTopkInfo[1] = j_eff;
            }
        }

        EP_DEVICE_ASSERT(outRecvTopkIdx != nullptr && outRecvTopkWeights != nullptr);
        EP_DEVICE_ASSERT(recvTopkIdxKind == NCCL_EP_EXPERT_ID_LOCAL || recvTopkIdxKind == NCCL_EP_EXPERT_ID_GLOBAL);
        if (laneId < numTopk) {
            int globalExpertIdx = (int)rtr[laneId].expert_id;
            int localExpertIdx = globalExpertIdx - globalExpertStartIdx;
            bool valid = (localExpertIdx >= 0 && localExpertIdx < numLocalExperts);
            int writeIdx = (recvTopkIdxKind == NCCL_EP_EXPERT_ID_GLOBAL) ? globalExpertIdx : localExpertIdx;
            outRecvTopkIdx[slot * numTopk + laneId] = valid ? (int32_t)writeIdx : (int32_t)-1;
            outRecvTopkWeights[slot * numTopk + laneId] = rtr[laneId].topk_weight;
        }

        auto* outDataInt4 = static_cast<int4*>(outDataBuf);
        const bool zeroCopy = kRecipe == NCCL_EP_DISP_QUANT_NONE && recvDataWindow != ncclWindow_t{};
        if (!zeroCopy) {
            copyRecvTokenDataLsa<kRecipe>(
                recvBufUint8, i, slot, outDataInt4, outScalesBuf, static_cast<int>(hiddenInt4),
                static_cast<int>(hiddenBytes), scaleBytes,
                static_cast<int>(numBytesPerMsg), static_cast<int>(dispatch_hdr_sz), maxTokensPerRank, laneId);
        }
    }
}

// Send-phase per-token body used by dispatch_kernel_impl_2sided_rm_lsa:
// stages the routing header, dedups by destination rank (one send per
// (token, destRank) pair, elected via the lowest topk slot targeting that
// rank), and selects the unquantized or DS-FP8 send helper.
template <ncclEpDispQuant_t kRecipe, int kHidden, typename TopkIdxT, ncclDataType_t kTokenDtype, bool kUseSharedQuant>
__forceinline__ __device__ void dispatchSendTokenLsa(
    const dispatch_kernel_args_t& args,
    int tokenIdx,
    int warpId,
    int laneId,
    int numTopk,
    int numLocalExperts,
    size_t srcRankRegionBytes,
    size_t numBytesPerMsg,
    size_t dispatch_hdr_sz,
    size_t hiddenBytes,
    size_t inputHiddenInt4,
    int scaleBytes,
    void* sendBuf,
    void* recvBuf,
    size_t recvOff,
    int* rankSentCnt,
    int numWritingThreads) {
    const auto srcDataInt4 = static_cast<const int4*>(args.inData) + tokenIdx * inputHiddenInt4;
    // Quantize once per token across the forwarding warps, then reuse the
    // result for every destination. Bound static shared memory; larger rows
    // retain direct per-peer quantization.
    constexpr bool kStageQuant =
        kUseSharedQuant && kRecipe == NCCL_EP_DISP_QUANT_DS_FP8E3M4 && kHidden <= kLlDsFp8SharedHiddenLimit;
    __shared__ __align__(16) int2 quantized[kStageQuant ? kHidden / 8 : 1];
    __shared__ float quantizedScales[kStageQuant ? kHidden / kDsFp8E3M4ElementsPerScale : 1];

    // The local global-memory staging slot holds only the routing header.
    // Payload is read from input or CTA shared memory, never RDMA staging.
    auto* sendBufBase = static_cast<uint8_t*>(sendBuf) + tokenIdx * numBytesPerMsg;

    // Each expert is handled by a different warp in the SM.
    auto dstExpertIdx = warpId < numTopk
        ? static_cast<int>(__ldg(static_cast<const TopkIdxT*>(args.inTopkIdx) + tokenIdx * numTopk + warpId))
        : -1;
    auto dstRank = getExpertRankIdx(dstExpertIdx, numLocalExperts);

    // Write token_id and routing information into the local staging
    // slot. Lane 0 of warps 0..numTopk-1 contribute one rtr entry each.
    auto* sendBufHdr = reinterpret_cast<DispatchHdr<NCCL_EP_LAYOUT_RANK_MAJOR>*>(sendBufBase);
    if (warpId < numTopk and laneId == 0) {
        if (warpId == 0) {
            sendBufHdr->token_id = tokenIdx;
        }
        sendBufHdr->rtr[warpId].expert_id = static_cast<uint16_t>(dstExpertIdx);
        sendBufHdr->rtr[warpId].topk_weight = __ldg(args.inTopkWeights + tokenIdx * numTopk + warpId);
    }

    if constexpr (kStageQuant) {
        castAndWriteToSendBuf<kRecipe, float>(
            srcDataInt4, quantized, quantizedScales, warpId * 32 + laneId,
            numWritingThreads, laneId, static_cast<int>(inputHiddenInt4), args.roundScale);
    }
    // Publish the routing header and any shared payload to the sending warps.
    syncSmGroup(SYNC_DISP_LSA_SEND_COPY, numWritingThreads);

    // Do filtering to avoid duplicate sending of tokens to the same rank.
    int slotIdx = -1;
    if (dstExpertIdx >= 0) {
        // Optimized: rank-level slot allocation (aggregates across experts)
        int minTopkIdx = numTopk;
        for (int i = laneId; i < numTopk; i += 32) {
            const auto otherExpertIdx = sendBufHdr->rtr[i].expert_id;
            const auto otherRank = getExpertRankIdx(otherExpertIdx, numLocalExperts);
            if (otherRank == dstRank) {
                minTopkIdx = min(minTopkIdx, i);
            }
        }
        minTopkIdx = warp_reduce_min(minTopkIdx);

        // If this warp is the first in topK for the dstRank, send the token.
        if (minTopkIdx == warpId) {
            slotIdx = laneId == 0 ? atomicAdd(rankSentCnt + dstRank, 1) : 0;
            slotIdx = __shfl_sync(0xffffffff, slotIdx, 0);

            const size_t srcRankOffset = static_cast<size_t>(relativeRankSlot(args.currRank, dstRank, args.numRanks)) *
                srcRankRegionBytes;
            const auto srcRankLocalPtr = reinterpret_cast<uint64_t>(recvBuf) + srcRankOffset;
            const auto sendBufInt4 = reinterpret_cast<const int4*>(sendBufBase);
            const auto dstSrcRankP2pPtr = ncclGetP2pPtr(
                srcRankLocalPtr, recvOff + srcRankOffset, args.currRank, dstRank, args.windows,
                args.devComm);
            if (!isRankMasked<true>(args.rankMask, dstRank)) {
                EP_DEVICE_ASSERT(dstSrcRankP2pPtr != 0);
                if constexpr (kRecipe == NCCL_EP_DISP_QUANT_NONE) {
                    sendTokenLsa(
                        dstSrcRankP2pPtr, sendBufInt4, srcDataInt4, slotIdx,
                        numBytesPerMsg, dispatch_hdr_sz, hiddenBytes, inputHiddenInt4,
                        args.maxTokensPerRank, dstRank, args.currRank,
                        args.recvDataWindow, args.recvDataOffset, laneId);
                } else {
                    sendTokenDsFp8Lsa(
                        dstSrcRankP2pPtr, sendBufInt4, srcDataInt4, slotIdx,
                        numBytesPerMsg, dispatch_hdr_sz, hiddenBytes, inputHiddenInt4,
                        scaleBytes, args.maxTokensPerRank, args.roundScale, laneId,
                        kStageQuant ? quantized : nullptr,
                        kStageQuant ? quantizedScales : nullptr);
                }
            }
        }
    }
    if constexpr (kStageQuant) {
        // All peer copies must finish before the next token overwrites CTA storage.
        syncSmGroup(SYNC_DISP_LSA_SEND_COPY, numWritingThreads);
    }
}

// SEND-phase warp split used by dispatch_kernel_impl_2sided_rm_lsa: every
// warp but the last forwards top-k tokens via dispatchSendTokenLsa above;
// the last warp resets this round's routing/count bookkeeping for the next
// epoch. Callers still call syncAndSendCounts themselves right after this
// returns -- that's the SEND/RECV synchronization handoff, kept visible at
// the kernel top level rather than folded in here.
template <ncclEpDispQuant_t kRecipe, int kHidden, typename TopkIdxT, ncclDataType_t kTokenDtype, bool kUseSharedQuant>
__forceinline__ __device__ void dispatchSendPhaseLsa(
    const dispatch_kernel_args_t& args,
    int smId,
    int warpId,
    int laneId,
    int numSms,
    int numWarps,
    int numTopk,
    int numLocalExperts,
    size_t numBytesPerMsg,
    size_t dispatch_hdr_sz,
    size_t hiddenBytes,
    size_t inputHiddenInt4,
    int scaleBytes,
    void* sendBuf,
    void* recvBuf,
    size_t recvOff,
    int* nextRecvCntBuf,
    int* rankSentCnt) {
    // There are 2 kinds of warps in this part:
    // 1. The first-kind warps for forwarding top-k tokens
    // 2. The last warp for reading `topk_idx` and count for per-expert information
    if (warpId < numWarps - 1) {
        constexpr int kNumElemsPerRead = sizeof(int4) / size_u8<kTokenDtype>();
        if constexpr (kRecipe == NCCL_EP_DISP_QUANT_DS_FP8E3M4) {
            EP_STATIC_ASSERT(kHidden % (32 * kNumElemsPerRead) == 0, "Invalid hidden");
        }
        if constexpr (kRecipe == NCCL_EP_DISP_QUANT_DS_FP8E3M4) {
            EP_STATIC_ASSERT(kNumElemsPerRead * 32 % kDsFp8E3M4ElementsPerScale == 0,
                             "Invalid DS_FP8E3M4 vectorization");
        }
        const auto numWritingThreads = (numWarps - 1) * 32;
        const size_t srcRankRegionBytes = static_cast<size_t>(args.maxTokensPerRank) * numBytesPerMsg;

        // Split token processing across SMs
        for (int tokenIdx = smId; tokenIdx < args.numTokens; tokenIdx += numSms) {
            dispatchSendTokenLsa<kRecipe, kHidden, TopkIdxT, kTokenDtype, kUseSharedQuant>(
                args, tokenIdx, warpId, laneId, numTopk, numLocalExperts, srcRankRegionBytes, numBytesPerMsg,
                dispatch_hdr_sz, hiddenBytes, inputHiddenInt4, scaleBytes, sendBuf, recvBuf, recvOff, rankSentCnt,
                numWritingThreads);
        }
    } else if (warpId == numWarps - 1) {
        // Rank-major: clear every routing entry while sends are in flight, so
        // every unwritten row is an all--1 sentinel row.
        const size_t totalTopkEntries = static_cast<size_t>(args.numRanks) * args.maxTokensPerRank * numTopk;
        for (size_t i = static_cast<size_t>(smId) * 32 + laneId; i < totalTopkEntries;
             i += static_cast<size_t>(numSms) * 32) {
            args.outRecvTopkIdx[i] = -1;
        }
        EP_DEVICE_ASSERT(numSms > 1);
        if (smId == 0) {
            cleanNextRecvCntBufLsa(nextRecvCntBuf, args.nextRecvCntBufSize, laneId, 32);
        }
    }
}

// LSA-only NONE/DS_FP8E3M4 LL dispatch entry point. Every destination is assumed
// NVLink-reachable; there is no GIN/RDMA runtime code path anywhere in this
// function. Rank-major output only (EXPERT_MAJOR is not supported here -- see
// ll_ep.cuh's general dispatch_kernel_impl for that layout), which is also
// the only layout that supports zero-copy direct-to-peer-window output.
// Faithful (behavior-preserving) simplification of
// nccl_ep::ll::dispatch_kernel_impl in ll_ep.cuh with a supported kRecipe,
// kNvlinkOnly = true, and kLayout = NCCL_EP_LAYOUT_RANK_MAJOR baked in, as a
// base for further (e.g. CTA-per-token) prototyping.
template <ncclEpDispQuant_t kRecipe, int kHidden, int kNumTopk, typename TopkIdxT, ncclDataType_t kTokenDtype, bool kUseSharedQuant>
__device__ __forceinline__ void dispatch_kernel_impl_2sided_rm_lsa(const dispatch_kernel_args_t& args) {
    static constexpr ncclEpLayout_t kLayout = NCCL_EP_LAYOUT_RANK_MAJOR;
    EP_STATIC_ASSERT(
        kRecipe == NCCL_EP_DISP_QUANT_NONE || kRecipe == NCCL_EP_DISP_QUANT_DS_FP8E3M4,
        "Unsupported LSA dispatch recipe");
    EP_STATIC_ASSERT(
        kRecipe != NCCL_EP_DISP_QUANT_DS_FP8E3M4 || kTokenDtype == ncclBfloat16,
        "DS_FP8E3M4 requires BF16 input");
    EP_STATIC_ASSERT(kNumTopk > 0 && kNumTopk <= combine_smem::kWarpSize - kLlDispatchControlWarps,
                     "LL dispatch top-k must leave one control warp");
    constexpr int numTopk = kNumTopk;

    const auto smId = static_cast<int>(blockIdx.x);
    const auto threadId = static_cast<int>(threadIdx.x);
    const auto warpId = threadId / 32, laneId = get_lane_id();
    const auto numSms = static_cast<int>(gridDim.x);
    const auto numWarps = args.numWarpGroups * args.numWarpsPerGroup;
    const auto numLocalExperts = args.numExperts / args.numRanks;
    const auto warpGroupId = warpId / args.numWarpsPerGroup;
    const auto subWarpId = warpId % args.numWarpsPerGroup;
    const auto responsibleExpertIdx = smId * args.numWarpGroups + warpGroupId;

    const unsigned int epoch = selectLowLatencyEpoch(args.epochState, args.phases, smId, threadId);
    const size_t bank = static_cast<size_t>(epoch & 1U);
    const size_t bank_next = bank ^ 1U;
    const size_t sendOff = args.sendOff + bank * args.payloadSlotStride;
    const size_t recvOff = args.recvOff + bank * args.payloadSlotStride;
    const size_t recvCntOff = args.recvCntOff + bank * args.signalSlotStride;
    char* const rdmaBase = static_cast<char*>(args.rdmaBuf);
    void* const sendBuf = rdmaBase + sendOff;
    void* const recvBuf = rdmaBase + recvOff;
    int* const recvCntBuf = reinterpret_cast<int*>(rdmaBase + recvCntOff);
    int* const nextRecvCntBuf =
        reinterpret_cast<int*>(rdmaBase + args.recvCntOff + bank_next * args.signalSlotStride);

    auto rankSentCnt = args.rankSentCnt;

    const size_t inputHiddenBytes = static_cast<size_t>(kHidden) * size_u8<kTokenDtype>();
    const size_t inputHiddenInt4 = inputHiddenBytes / sizeof(int4);
    const size_t hiddenBytes = kRecipe == NCCL_EP_DISP_QUANT_DS_FP8E3M4
        ? static_cast<size_t>(kHidden) * sizeof(uint8_t)
        : inputHiddenBytes;
    const size_t hiddenInt4 = hiddenBytes / sizeof(int4);
    const int numScales = kRecipe == NCCL_EP_DISP_QUANT_DS_FP8E3M4
        ? kHidden / kDsFp8E3M4ElementsPerScale
        : 0;
    const int scaleBytes = numScales * sizeof(float);

    // Message package: header + recipe-specific token data and scales.
    const size_t dispatch_hdr_sz = get_dispatch_hdr_sz<kLayout>(numTopk);
    const size_t numBytesPerMsg = dispatch_hdr_sz + hiddenBytes + scaleBytes;

    EP_DEVICE_ASSERT(numBytesPerMsg % sizeof(int4) == 0);

    // Set by syncAndSendCounts below; declared (and default-initialized)
    // here so it's still in scope, and safely readable, at the RECV label
    // below even for a RECV-only call that jumps straight past the SEND
    // phase via the goto immediately below.
    bool isLastCta = false;

    // Sending phase
    if ((args.phases & LOW_LATENCY_SEND_PHASE) == 0) {
        goto LOW_LATENCY_DISPATCH_LSA_RECV;
    }

    dispatchSendPhaseLsa<kRecipe, kHidden, TopkIdxT, kTokenDtype, kUseSharedQuant>(
        args, smId, warpId, laneId, numSms, numWarps, numTopk, numLocalExperts, numBytesPerMsg, dispatch_hdr_sz,
        hiddenBytes, inputHiddenInt4, scaleBytes, sendBuf, recvBuf, recvOff, nextRecvCntBuf, rankSentCnt);

    isLastCta = syncAndSendCounts(
        warpId, numWarps, laneId, args.rankDone, numSms, args.numRanks, rankSentCnt, args.currRank, recvCntBuf,
        recvCntOff, args.rankMask, args.windows, args.devComm);

LOW_LATENCY_DISPATCH_LSA_RECV:
    if ((args.phases & LOW_LATENCY_RECV_PHASE) == 0) return;

    // Split invocation (SEND phase didn't run here): syncAndSendCounts never
    // elected anyone, so nominate smId == 0 instead.
    if ((args.phases & LOW_LATENCY_SEND_PHASE) == 0) {
        isLastCta = (smId == 0);
    }

    // No cg::this_grid().sync() here, and no per-CTA rankSentCnt/rankDone
    // reset either: syncAndSendCounts's elected last CTA already reset both
    // arrays for the next epoch, immediately after publishing them -- see
    // its doc comment for why that needs no further synchronization here.
    // The completeFullLowLatencyEpoch(..., bool, ...) overload piggybacks
    // the epoch bump on that same elected CTA (isLastCta) instead of the
    // general kernel's "smId == 0" convention, which has no such guarantee
    // without a grid-wide barrier.
    completeFullLowLatencyEpoch(args.epochState, args.phases, isLastCta, threadId);

    syncAndRecvCounts(
        threadId, args.numRanks, args.rankMask, recvCntBuf, args.timeoutCycles, args.currRank, args.asyncErrorFlag);

    recordRecvTokensLsa<kRecipe>(
        responsibleExpertIdx, args.numExperts, numLocalExperts, laneId, subWarpId, args.numWarpsPerGroup, numTopk,
        args.numRanks, args.currRank, args.maxTokensPerRank, numBytesPerMsg, dispatch_hdr_sz, hiddenInt4, hiddenBytes,
        scaleBytes,
        args.rankMask, recvCntBuf, recvBuf, args.outSrcInfo, args.outRecvRankCounter, args.outRecvTopkIdx,
        args.outRecvTopkWeights, args.outDataBuf, args.outScalesBuf, args.recvTopkIdxKind, args.recvDataWindow);
}

// ============================================================================
// TMA-pipelined variant of the SEND phase above (dispatch_kernel_impl_2sided_rm_lsa_tma). The
// default LL dispatch kernel for LSA-eligible, unquantized calls (NCCL_EP_DISPATCH_LSA=tma, see
// jit/ll_dispatch_jit.cuh and NcclEpDispatchLsaMode's doc comment in ll_ep_adapter.cuh).
// Unquantized (NCCL_EP_DISP_QUANT_NONE) and rank-major only -- unlike
// dispatch_kernel_impl_2sided_rm_lsa above, this kernel takes no recipe template; the RECV phase
// below still goes through the shared, recipe-templated recordRecvTokensLsa, instantiated at
// kRecipe = NCCL_EP_DISP_QUANT_NONE.
//
// Design summary:
//   - Warp 0 alone drives an N-stage software pipeline across this CTA's token range: each stage
//     is a whole-token shared-memory buffer (payload + header + one mbarrier), TMA-loaded from
//     args.inData ahead of use and TMA-stored out to each destination rank once ready. N is fixed
//     per JIT variant (dispatch_tma_smem::choose_config, keyed by hidden/dtype/smem budget only,
//     NOT by this call's token count -- see that namespace's doc comment in ll_ep_smem.cuh for
//     why), then clamped at runtime to this CTA's actual token count so a CTA with fewer tokens
//     than stages never issues a needless prefetch.
//   - Every other warp (1..numWarps-1) spends the whole SEND phase on the routing-sentinel
//     (outRecvTopkIdx = -1) init and the next-epoch recv-count-buffer clear, running CONCURRENTLY
//     with warp 0's entire pipeline (not just its trailing stores) -- idle warps doing this work
//     from the start, rather than warp 0 doing it serially after its own loop, maximizes overlap.
//     No extra barrier is needed before syncAndSendCounts: that function already opens with its
//     own __syncthreads(), which naturally waits for both warp 0's tma_store_wait_complete<0>()
//     and every other warp's postamble loop to finish.
//   - Deduplication (which lane -- i.e. which topk slot -- actually owns a given destination rank
//     for this token) is done via warpFirstTopKOccurence (ll_common.cuh), a register/shuffle-only
//     operation -- no cross-warp barrier, no global-memory round trip, unlike the per-warp dedup in
//     dispatchSendTokenLsa above.
//   - Each smem stage's payload/header must stay valid until every store reading it has completed.
//     cp.async.bulk's commit-group counters are PER-THREAD, so to make tma_store_wait<N> a
//     deterministic "this stage is free" check, a single lane (0) issues every winning lane's
//     (header, payload) stores -- shuffled in from the winners -- and commits ONCE per token
//     (not once per destination), letting tma_store_wait<kNumStages - 1> below wait on lane 0 alone.
// ============================================================================

template <int kHidden, int kNumStages, typename TopkIdxT, ncclDataType_t kTokenDtype>
__forceinline__ __device__ void dispatchSendPhaseLsaTma(
    const dispatch_kernel_args_t& args,
    int smId,
    int warpId,
    int laneId,
    int numSms,
    int numWarps,
    int numTopk,
    int numLocalExperts,
    size_t dispatch_hdr_sz,
    size_t hiddenBytes,
    size_t hiddenInt4,
    void* recvBuf,
    size_t recvOff,
    int* nextRecvCntBuf,
    int* rankSentCnt,
    uint8_t* smemBase) {
    // Each token's whole row moves in one TMA transfer, so the only alignment requirement is
    // numBytesPerMsg % sizeof(int4) == 0, already asserted by the caller
    // (dispatch_kernel_impl_2sided_rm_lsa_tma).
    EP_STATIC_ASSERT(kNumStages > 0 && kNumStages < 32, "Invalid stage count (tmaPhase is a 32-bit mask)");

    const size_t hdrSectionBytes = static_cast<size_t>(args.maxTokensPerRank) * dispatch_hdr_sz;
    const size_t srcRankRegionBytes = static_cast<size_t>(args.maxTokensPerRank) * (dispatch_hdr_sz + hiddenBytes);
    const size_t stageBytes = hiddenBytes + dispatch_hdr_sz + dispatch_tma_smem::kTmaBarrierAndPaddingBytes;
    // Rank-pointer cache lives right after the kNumStages-deep stage ring, in the same dynamic
    // smem allocation (see dispatch_tma_smem::choose_config -- host sizing matches this layout).
    uint64_t* const rankP2pCache =
        reinterpret_cast<uint64_t*>(smemBase + static_cast<size_t>(kNumStages) * stageBytes);

    if (warpId == 0) {
        auto stagePayload = [&](int s) { return reinterpret_cast<int4*>(smemBase + static_cast<size_t>(s) * stageBytes); };
        auto stageHeader = [&](int s) {
            return reinterpret_cast<DispatchHdr<NCCL_EP_LAYOUT_RANK_MAJOR>*>(
                smemBase + static_cast<size_t>(s) * stageBytes + hiddenBytes);
        };
        auto stageBarrier = [&](int s) {
            return reinterpret_cast<uint64_t*>(
                smemBase + static_cast<size_t>(s) * stageBytes + hiddenBytes + dispatch_hdr_sz);
        };

        const int numTokensThisCta = nccl_ep::ceil_div(max(args.numTokens - smId, 0), numSms);

        if (numTokensThisCta > 0) {
            // Resolve each destination rank's P2P header/payload base pointer exactly once per
            // CTA, concurrently with the mbarrier inits below (independent smem regions, same
            // warp). ncclGetP2pPtr's result depends only on (currRank, dstRank, recvOff) -- all
            // loop-invariant for this whole SEND phase -- so recomputing it per (token, winning
            // lane) below would be pure redundant work: same team-membership check + window-table
            // lookup, same answer, every time a CTA sends to a rank it's already sent to. Computed
            // regardless of isRankMasked (matches dispatchSendTokenLsa's ordering: the pointer is
            // always resolved first, masking only gates whether the store using it is issued).
            for (int r = laneId; r < args.numRanks; r += 32) {
                const size_t srcRankOffset =
                    static_cast<size_t>(relativeRankSlot(args.currRank, r, args.numRanks)) * srcRankRegionBytes;
                const auto srcRankLocalPtr = reinterpret_cast<uint64_t>(recvBuf) + srcRankOffset;
                rankP2pCache[r] = ncclGetP2pPtr(
                    srcRankLocalPtr, recvOff + srcRankOffset, args.currRank, r, args.windows, args.devComm);
            }
            // Always init all kNumStages barriers, regardless of numTokensThisCta: the ring is a
            // fixed kNumStages-deep circular buffer (stage = iter % kNumStages below), not sized
            // down to the token count. A CTA with fewer tokens than stages simply never reuses a
            // slot (every token gets its own, no wraparound), rather than using a smaller ring.
            if (laneId < kNumStages) {
                mbarrier_init(stageBarrier(laneId), 1);
                fence_barrier_init();
            }
            __syncwarp();

            auto issueLoad = [&](int iter, int stage) {
                const int tokenIdx = smId + iter * numSms;
                const auto* srcPtr = static_cast<const int4*>(args.inData) + static_cast<size_t>(tokenIdx) * hiddenInt4;
                // Single-lane read issue
                if (laneId == 0) {
                    tma_load_1d(stagePayload(stage), srcPtr, stageBarrier(stage), static_cast<int>(hiddenBytes));
                    mbarrier_arrive_and_expect_tx(stageBarrier(stage), static_cast<int>(hiddenBytes));
                }
                __syncwarp();
            };

            // Prefetch only the first token; the loop below issues each subsequent token's load
            // one iteration ahead of when it's needed (see the reload step at the end of the loop
            // body), filling the rest of the kNumStages-deep ring as it goes.
            issueLoad(0, 0);

            uint32_t tmaPhase = 0;
            for (int iter = 0; iter < numTokensThisCta; ++iter) {
                const int stage = iter % kNumStages;
                const int tokenIdx = smId + iter * numSms;

                // a) topk indices into lane registers (lane i <-> topk slot i).
                const int dstExpertIdx = laneId < numTopk
                    ? static_cast<int>(
                          __ldg(static_cast<const TopkIdxT*>(args.inTopkIdx) + static_cast<size_t>(tokenIdx) * numTopk + laneId))
                    : -1;
                const int dstRank = getExpertRankIdx(dstExpertIdx, numLocalExperts);

                // b) assemble the header in this stage's smem, one entry per lane.
                auto* hdr = stageHeader(stage);
                if (laneId == 0) hdr->token_id = tokenIdx;
                if (laneId < numTopk) {
                    hdr->rtr[laneId].expert_id = static_cast<uint16_t>(dstExpertIdx);
                    hdr->rtr[laneId].topk_weight =
                        __ldg(args.inTopkWeights + static_cast<size_t>(tokenIdx) * numTopk + laneId);
                }
                __syncwarp(); // header entries visible to whichever lane(s) store the whole header below

                // Ensure that header updates are visible to the TMA engine
                tma_store_fence();

                // c) deduplicate target ranks
                const bool hasValidSlot = dstExpertIdx >= 0;
                const int firstOccurrence = warpFirstTopKOccurence(dstExpertIdx, numLocalExperts, laneId);
                const bool isWinner = hasValidSlot && firstOccurrence >= 0;

                // d) wait for the current stage's payload load to finish, then issue the store(s).
                mbarrier_wait<true>(stageBarrier(stage), tmaPhase, stage);

                // Every winning (and unmasked) lane resolves its OWN destination addresses in
                // parallel -- this part is plain global-memory arithmetic/atomics, not a TMA
                // operation, so it stays per-lane. Slot allocation happens even for a masked
                // destination (matches dispatchSendTokenLsa's non-TMA ordering: slot numbering must
                // stay consistent regardless of masking); only the actual store is skipped when
                // masked, via doStore below.
                uint8_t* dstHdrSlot = nullptr;
                void* payloadDst = nullptr;
                bool doStore = false;
                if (isWinner) {
                    // Locally allocate a slot in the remote range reserved for this rank
                    const int slotIdx = atomicAdd(rankSentCnt + dstRank, 1);
                    // Resolved once per CTA into rankP2pCache before this loop started (see the
                    // doc comment there) instead of being recomputed here every iteration.
                    const uint64_t dstSrcRankP2pPtr = rankP2pCache[dstRank];
                    doStore = !isRankMasked<true>(args.rankMask, dstRank);
                    if (doStore) {
                        EP_DEVICE_ASSERT(dstSrcRankP2pPtr != 0);
                        auto* dstSrcRankBase = reinterpret_cast<uint8_t*>(dstSrcRankP2pPtr);
                        dstHdrSlot = dstSrcRankBase + static_cast<size_t>(slotIdx) * dispatch_hdr_sz;
                        payloadDst = dstSrcRankBase + hdrSectionBytes + static_cast<size_t>(slotIdx) * hiddenBytes;
                        if (args.recvDataWindow != ncclWindow_t{}) {
                            const size_t recvSlot =
                                static_cast<size_t>(args.currRank) * args.maxTokensPerRank + slotIdx;
                            payloadDst = ncclGetPeerPointer(
                                args.recvDataWindow, args.recvDataOffset + recvSlot * hiddenBytes, dstRank);
                        }
                    }
                }

                // Loop over topK lanes, picking the winning ones using mask
                // Use warp-shuffling to deliver required information from the
                // owning lane to the issuing lane (lane 0)
                const uint32_t winnerStoreMask = __ballot_sync(0xffffffffu, doStore);
                if (winnerStoreMask != 0) {
                    for (int srcLane = 0; srcLane < numTopk; ++srcLane) {
                        if ((winnerStoreMask >> srcLane) & 1u) {
                            const uint64_t hdrSlotW =
                                __shfl_sync(0xffffffffu, reinterpret_cast<uint64_t>(dstHdrSlot), srcLane);
                            const uint64_t payloadW =
                                __shfl_sync(0xffffffffu, reinterpret_cast<uint64_t>(payloadDst), srcLane);
                            if (laneId == 0) {
                                tma_store_1d_no_commit(
                                    hdr, reinterpret_cast<void*>(hdrSlotW), static_cast<int>(dispatch_hdr_sz));
                                tma_store_1d_no_commit(
                                    stagePayload(stage), reinterpret_cast<void*>(payloadW),
                                    static_cast<int>(hiddenBytes));
                            }
                        }
                    }
                }
                // Issue commit-group on per-token basis (irrespective of actual sends)
                // to keep the pipeline synchronization correct
                if (laneId == 0) tma_store_commit_group();

                const int nextIter = iter + 1;
                const bool willReload = nextIter < numTokensThisCta;
                if (willReload) {
                    const int nextStage = nextIter % kNumStages;
                    // nextStage's previous occupant (if any) was nextIter - kNumStages, i.e.
                    // kNumStages iterations before nextIter -- tma_store_wait<kNumStages - 1>
                    // always correctly bounds that, regardless of numTokensThisCta, because the
                    // ring is always exactly kNumStages deep (see the mbarrier_init comment
                    // above): there's no separate "active" ring size to track or get out of sync
                    // with this compile-time wait bound.
                    //
                    // Lane 0 waits for all issued stores to
                    // allocate the stage for the next load
                    if (laneId == 0) {
                        tma_store_wait<kNumStages - 1>();
                    }
                    __syncwarp();
                    // Prefetch-issue another token if not done
                    issueLoad(nextIter, nextStage);
                }
            }

            // Ensure that all stores are visible at NVL-level (not just read from smem)
            // before signalling remote peers that the data is ready
            tma_store_wait_complete<0>();
            __syncwarp();

            if (laneId < kNumStages) {
                mbarrier_inval(stageBarrier(laneId));
                fence_barrier_init();
            }
            __syncwarp();
        }
    } else {
        // Use available warps to clean the workspace for the next operation
        const int postThreadId = (warpId - 1) * 32 + laneId;
        const int numPostThreads = (numWarps - 1) * 32;
        const size_t totalTopkEntries = static_cast<size_t>(args.numRanks) * args.maxTokensPerRank * numTopk;
        const size_t totalThreads = static_cast<size_t>(numSms) * numPostThreads;
        const size_t globalIdx = static_cast<size_t>(smId) * numPostThreads + postThreadId;
        const int4 negOnesInt4 = make_int4(-1, -1, -1, -1);
        // outRecvTopkIdx is a caller-owned output tensor (ncclEpDispatchOutputs_t::topk_idx);
        // ncclEpDispatch only validates its ndim/datatype, not 16-byte alignment, so the int4
        // vector path below is only safe when the caller's pointer happens to be aligned for it.
        if ((reinterpret_cast<uintptr_t>(args.outRecvTopkIdx) & 0xF) == 0) {
            int4* const outRecvTopkIdxVec4 = reinterpret_cast<int4*>(args.outRecvTopkIdx);
            const size_t totalTopkEntriesVec4 = totalTopkEntries / 4;
            for (size_t i = globalIdx; i < totalTopkEntriesVec4; i += totalThreads) {
                outRecvTopkIdxVec4[i] = negOnesInt4;
            }
            // Scalar tail for the (totalTopkEntries % 4) elements the vectorized loop above can't reach.
            for (size_t i = totalTopkEntriesVec4 * 4 + globalIdx; i < totalTopkEntries; i += totalThreads) {
                args.outRecvTopkIdx[i] = -1;
            }
        } else {
            for (size_t i = globalIdx; i < totalTopkEntries; i += totalThreads) {
                args.outRecvTopkIdx[i] = -1;
            }
        }
        EP_DEVICE_ASSERT(numSms > 1);
        if (smId == 0) {
            cleanNextRecvCntBufLsa(nextRecvCntBuf, args.nextRecvCntBufSize, postThreadId, numPostThreads);
        }
    }
}

template <int kHidden, int kNumTopk, int kNumStages, typename TopkIdxT, ncclDataType_t kTokenDtype>
__device__ __forceinline__ void dispatch_kernel_impl_2sided_rm_lsa_tma(const dispatch_kernel_args_t& args) {
    EP_STATIC_ASSERT(kNumTopk > 0 && kNumTopk <= 32, "LL dispatch TMA top-k must fit one warp's lanes");
    extern __shared__ __align__(1024) uint8_t dispatchTmaSmem[];
    static constexpr ncclEpLayout_t kLayout = NCCL_EP_LAYOUT_RANK_MAJOR;
    constexpr int numTopk = kNumTopk;

    const auto smId = static_cast<int>(blockIdx.x);
    const auto threadId = static_cast<int>(threadIdx.x);
    const auto warpId = threadId / 32, laneId = get_lane_id();
    const auto numSms = static_cast<int>(gridDim.x);
    const auto numWarps = args.numWarpGroups * args.numWarpsPerGroup;
    const auto numLocalExperts = args.numExperts / args.numRanks;
    const auto warpGroupId = warpId / args.numWarpsPerGroup;
    const auto subWarpId = warpId % args.numWarpsPerGroup;
    const auto responsibleExpertIdx = smId * args.numWarpGroups + warpGroupId;

    const unsigned int epoch = selectLowLatencyEpoch(args.epochState, args.phases, smId, threadId);
    const size_t bank = static_cast<size_t>(epoch & 1U);
    const size_t bank_next = bank ^ 1U;
    const size_t recvOff = args.recvOff + bank * args.payloadSlotStride;
    const size_t recvCntOff = args.recvCntOff + bank * args.signalSlotStride;
    char* const rdmaBase = static_cast<char*>(args.rdmaBuf);
    void* const recvBuf = rdmaBase + recvOff;
    int* const recvCntBuf = reinterpret_cast<int*>(rdmaBase + recvCntOff);
    int* const nextRecvCntBuf =
        reinterpret_cast<int*>(rdmaBase + args.recvCntOff + bank_next * args.signalSlotStride);

    auto rankSentCnt = args.rankSentCnt;

    const size_t hiddenBytes = static_cast<size_t>(kHidden) * size_u8<kTokenDtype>();
    const size_t hiddenInt4 = hiddenBytes / sizeof(int4);

    // Message package: header + token data (no scale payload, unquantized).
    const size_t dispatch_hdr_sz = get_dispatch_hdr_sz<kLayout>(numTopk);
    const size_t numBytesPerMsg = dispatch_hdr_sz + hiddenBytes;

    EP_DEVICE_ASSERT(numBytesPerMsg % sizeof(int4) == 0);

    // Set by syncAndSendCounts below; declared (and default-initialized) here so it's still in
    // scope, and safely readable, at the RECV label below even for a RECV-only call that jumps
    // straight past the SEND phase via the goto immediately below.
    bool isLastCta = false;

    // Sending phase
    if ((args.phases & LOW_LATENCY_SEND_PHASE) == 0) {
        goto LOW_LATENCY_DISPATCH_LSA_TMA_RECV;
    }

    dispatchSendPhaseLsaTma<kHidden, kNumStages, TopkIdxT, kTokenDtype>(
        args, smId, warpId, laneId, numSms, numWarps, numTopk, numLocalExperts, dispatch_hdr_sz, hiddenBytes,
        hiddenInt4, recvBuf, recvOff, nextRecvCntBuf, rankSentCnt, dispatchTmaSmem);

    isLastCta = syncAndSendCounts(
        warpId, numWarps, laneId, args.rankDone, numSms, args.numRanks, rankSentCnt, args.currRank, recvCntBuf,
        recvCntOff, args.rankMask, args.windows, args.devComm);

LOW_LATENCY_DISPATCH_LSA_TMA_RECV:
    if ((args.phases & LOW_LATENCY_RECV_PHASE) == 0) return;

    // Split invocation (SEND phase didn't run here): syncAndSendCounts never
    // elected anyone, so nominate smId == 0 instead.
    if ((args.phases & LOW_LATENCY_SEND_PHASE) == 0) {
        isLastCta = (smId == 0);
    }

    // RECV-phase tail, unquantized/rank-major only. No rankSentCnt/rankDone reset here: that's
    // already done by syncAndSendCounts's elected last CTA (see its doc comment).
    completeFullLowLatencyEpoch(args.epochState, args.phases, isLastCta, threadId);

    syncAndRecvCounts(
        threadId, args.numRanks, args.rankMask, recvCntBuf, args.timeoutCycles, args.currRank, args.asyncErrorFlag);

    recordRecvTokensLsa<NCCL_EP_DISP_QUANT_NONE>(
        responsibleExpertIdx, args.numExperts, numLocalExperts, laneId, subWarpId, args.numWarpsPerGroup, numTopk,
        args.numRanks, args.currRank, args.maxTokensPerRank, numBytesPerMsg, dispatch_hdr_sz, hiddenInt4, hiddenBytes,
        /*scaleBytes=*/0, args.rankMask, recvCntBuf, recvBuf, args.outSrcInfo, args.outRecvRankCounter,
        args.outRecvTopkIdx, args.outRecvTopkWeights, args.outDataBuf, /*outScalesBuf=*/nullptr,
        args.recvTopkIdxKind, args.recvDataWindow);
}

} // namespace ll

} // namespace nccl_ep
