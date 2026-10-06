/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 * See LICENSE.txt for more license information.
 */

#pragma once

#include <cuda_runtime.h>

#include <cstdint>

namespace nccl_ep::ll::combine_smem {

// These values define the LL combine kernel's dynamic shared-memory layout.
// Keep all host launch accounting and device layout derivations based on them.
constexpr int kWarpSize = 32;
constexpr int kMaxSendUnrolls = 4;
constexpr int kMinSendUnrolls = 2;
constexpr int kNumRecvUnrolls = 2;
constexpr int kNumStages = 3;
constexpr int kNumTmaPrefetch = 1;
constexpr int kNumRecvTmaWarps = 1;
constexpr int kSendTmaBarrierBytes = sizeof(std::uint64_t);
constexpr int kSendTmaPaddingBytes = sizeof(std::uint64_t);
constexpr int kTmaBarrierAndPaddingBytes = kSendTmaBarrierBytes + kSendTmaPaddingBytes;
constexpr int kNumRecvTmaBarriers = 2;
constexpr int kRecvTmaBarrierBytes = kNumRecvTmaBarriers * sizeof(std::uint64_t);
constexpr int kRecvTmaPaddingBytes = sizeof(int4);
constexpr int kRecvTmaEmptyBarrierOffsetBytes = sizeof(std::uint64_t);
constexpr int kRecvTmaPayloadOffsetBytes = kRecvTmaBarrierBytes;
constexpr int kMetaElementBytes = sizeof(std::uint32_t);
constexpr int kLogFmtElementsPerMeta = 128;
constexpr int kLogFmtInputBytesPerChunk = 16;
constexpr int kLogFmtOutputBytesPerChunk = 10;
constexpr int kNumRecvMetaBuffers = 3;
constexpr int kTwoByteElementBytes = sizeof(std::uint16_t);
constexpr int kMaxRecvGroupsForTwoByteElements = 2;
constexpr int kMaxRecvGroupsForOtherElements = 1;
constexpr int kNvfp4ElementsPerPackedByte = 2;
constexpr int kNvfp4ElementsPerScale = 16;
constexpr int kNvfp4GlobalScaleBytes = sizeof(float);

constexpr int elements_per_int4(int elem_bytes) {
    return static_cast<int>(sizeof(int4)) / elem_bytes;
}

constexpr int send_unrolls(int hidden, int elem_bytes) {
    return hidden % (kWarpSize * kMaxSendUnrolls * elements_per_int4(elem_bytes)) == 0 ? kMaxSendUnrolls :
                                                                                           kMinSendUnrolls;
}

constexpr int padded_hidden_int4(int hidden, int elem_bytes) {
    const int unrolls = send_unrolls(hidden, elem_bytes);
    const int hidden_int4 = hidden / elements_per_int4(elem_bytes);
    const int alignment = kWarpSize * unrolls;
    return ((hidden_int4 + alignment - 1) / alignment) * alignment;
}

constexpr int recv_decode_warps(int hidden, int elem_bytes) {
    return padded_hidden_int4(hidden, elem_bytes) / (kNumRecvUnrolls * kWarpSize);
}

constexpr int max_recv_groups(int elem_bytes) {
    return elem_bytes == kTwoByteElementBytes ? kMaxRecvGroupsForTwoByteElements : kMaxRecvGroupsForOtherElements;
}

constexpr int meta_bytes(int hidden) {
    return hidden / kLogFmtElementsPerMeta * kMetaElementBytes;
}

constexpr int align_to_int4(int bytes) {
    return ((bytes + static_cast<int>(sizeof(int4)) - 1) / static_cast<int>(sizeof(int4))) *
           static_cast<int>(sizeof(int4));
}

constexpr int nvfp4_slot_bytes(int hidden) {
    return align_to_int4(
        hidden / kNvfp4ElementsPerPackedByte + hidden / kNvfp4ElementsPerScale + kNvfp4GlobalScaleBytes);
}

constexpr int send_warp_smem_bytes(int hidden, int elem_bytes) {
    const int tma_bytes = kWarpSize * static_cast<int>(sizeof(int4)) * send_unrolls(hidden, elem_bytes) +
                          kTmaBarrierAndPaddingBytes;
    // Each warp owns TMA buffers followed by a 4-byte-granular metadata tail.
    // Round the complete stride so every warp's first TMA buffer remains aligned.
    return align_to_int4(kNumStages * tma_bytes + meta_bytes(hidden));
}

constexpr int send_smem_bytes(int hidden, int elem_bytes, int num_warps) {
    return num_warps * send_warp_smem_bytes(hidden, elem_bytes);
}

constexpr int recv_tma_buffer_bytes(int hidden, int elem_bytes, bool nvfp4) {
    const int padded_token_bytes = padded_hidden_int4(hidden, elem_bytes) * static_cast<int>(sizeof(int4));
    return kRecvTmaBarrierBytes + (nvfp4 ? nvfp4_slot_bytes(hidden) : kRecvTmaPaddingBytes + padded_token_bytes);
}

constexpr int recv_group_smem_bytes(int hidden, int elem_bytes, bool nvfp4) {
    const int tma_bytes = recv_tma_buffer_bytes(hidden, elem_bytes, nvfp4);
    const int padded_token_bytes = padded_hidden_int4(hidden, elem_bytes) * static_cast<int>(sizeof(int4));
    // Every receive group contains TMA barriers and bulk-copy buffers. Keep the
    // group stride int4-aligned so group 1+ has the same alignment as group 0.
    // The metadata tail is only 4-byte granular and can otherwise leave the
    // next group 8 bytes off a TMA-compatible address (for example H=2880).
    return align_to_int4(
        kNumStages * tma_bytes + padded_token_bytes + kNumStages * meta_bytes(hidden) * kNumRecvMetaBuffers);
}

constexpr int dynamic_smem_bytes(int hidden, int elem_bytes, int num_warps, int num_recv_groups, bool nvfp4) {
    const int send_bytes = send_smem_bytes(hidden, elem_bytes, num_warps);
    const int recv_bytes = num_recv_groups * recv_group_smem_bytes(hidden, elem_bytes, nvfp4);
    return send_bytes > recv_bytes ? send_bytes : recv_bytes;
}

// Upper bound on numRanks for combine_kernel_impl_2sided_rm_lsa's CTA-local
// active-rank bitmask (ll/ll_combine_lsa.cuh), which has no other
// compile-time bound. 128 matches the HT path's enforced lsa_team_size cap.
constexpr int kMaxLsaCombineRanks = 128;
constexpr int kActiveRankMaskWords = (kMaxLsaCombineRanks + 31) / 32;

// Static __shared__ footprint of that bitmask -- separate from (and not
// covered by) the dynamic_smem_bytes() budget above; must be reserved out
// of max_dynamic_smem by choose_combine_smem_config (ll_ep_adapter.cuh).
constexpr int kLsaStaticSmemBytes = kActiveRankMaskWords * static_cast<int>(sizeof(std::uint32_t));

} // namespace nccl_ep::ll::combine_smem
