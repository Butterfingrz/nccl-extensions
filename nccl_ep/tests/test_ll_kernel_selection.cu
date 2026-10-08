/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <gtest/gtest.h>
#include "nccl_ep.h"
#include "device/jit/ll_dispatch_jit.cuh"
#include "device/jit/ll_combine_jit.cuh"

using namespace nccl_ep::ll;
using namespace nccl_ep::ll::jit;

TEST(LlKernelSelection, PartialDispatchTilesUseUniversalInBothModes) {
    for (auto dtype : {ncclBfloat16, ncclFloat16, ncclFloat32}) {
        const int tile = dtype == ncclFloat32 ? 128 : 256;
        for (auto mode : {NcclEpDispatchLsaMode::kSimt, NcclEpDispatchLsaMode::kTma}) {
            for (int hidden : {2880, tile - 16, tile + 16}) {
                EXPECT_EQ(ll_dispatch_select_algo(true, NCCL_EP_DISP_QUANT_NONE,
                    NCCL_EP_LAYOUT_RANK_MAJOR, false, mode, hidden, dtype), LlDispatchAlgo::kDefault);
            }
            for (int hidden : {tile, 3072, 7168}) {
                EXPECT_EQ(ll_dispatch_select_algo(true, NCCL_EP_DISP_QUANT_NONE,
                    NCCL_EP_LAYOUT_RANK_MAJOR, false, mode, hidden, dtype),
                    mode == NcclEpDispatchLsaMode::kTma ? LlDispatchAlgo::k2SidedRmLsaTma :
                                                        LlDispatchAlgo::k2SidedRmLsa);
            }
        }
    }
}

TEST(LlKernelSelection, DispatchPreservesOtherFallbacks) {
    EXPECT_EQ(ll_dispatch_select_algo(true, NCCL_EP_DISP_QUANT_DS_FP8E3M4,
        NCCL_EP_LAYOUT_RANK_MAJOR, false, NcclEpDispatchLsaMode::kTma, 7168, ncclBfloat16),
        LlDispatchAlgo::k2SidedRmLsa);
    EXPECT_EQ(ll_dispatch_select_algo(true, NCCL_EP_DISP_QUANT_DS_FP8E3M4,
        NCCL_EP_LAYOUT_RANK_MAJOR, false, NcclEpDispatchLsaMode::kTma, 128, ncclBfloat16),
        LlDispatchAlgo::kDefault);
    for (bool nvlink : {false, true}) {
        for (bool cft : {false, true}) {
            EXPECT_EQ(ll_dispatch_select_algo(nvlink, NCCL_EP_DISP_QUANT_NONE,
                NCCL_EP_LAYOUT_RANK_MAJOR, cft, NcclEpDispatchLsaMode::kUniversal, 7168, ncclBfloat16),
                LlDispatchAlgo::kDefault);
            if (!nvlink || cft) {
                EXPECT_EQ(ll_dispatch_select_algo(nvlink, NCCL_EP_DISP_QUANT_NONE,
                    NCCL_EP_LAYOUT_RANK_MAJOR, cft, NcclEpDispatchLsaMode::kTma, 7168, ncclBfloat16),
                    LlDispatchAlgo::kDefault);
            }
        }
    }
}

TEST(LlKernelSelection, CombineRequiresFullReceiveTiles) {
    for (auto dtype : {ncclBfloat16, ncclFloat16, ncclFloat32}) {
        const int tile = dtype == ncclFloat32 ? 256 : 512;
        for (int hidden : {2880, tile / 2, tile - 16, tile + 16}) {
            EXPECT_EQ(ll_combine_select_algo(true, NCCL_EP_COMB_QUANT_NONE, false,
                NCCL_EP_LAYOUT_RANK_MAJOR, false, hidden, dtype), LlCombineAlgo::kDefault);
        }
        for (int hidden : {tile, 3072, 7168}) {
            EXPECT_EQ(ll_combine_select_algo(true, NCCL_EP_COMB_QUANT_NONE, false,
                NCCL_EP_LAYOUT_RANK_MAJOR, false, hidden, dtype), LlCombineAlgo::k2SidedRmLsa);
        }
    }
    EXPECT_EQ(ll_combine_select_algo(true, NCCL_EP_COMB_QUANT_NONE, true,
        NCCL_EP_LAYOUT_RANK_MAJOR, false, 7168, ncclBfloat16), LlCombineAlgo::kDefault);
    EXPECT_EQ(ll_combine_select_algo(true, NCCL_EP_COMB_QUANT_NONE, false,
        NCCL_EP_LAYOUT_RANK_MAJOR, true, 7168, ncclBfloat16), LlCombineAlgo::kDefault);
    EXPECT_EQ(ll_combine_select_algo(false, NCCL_EP_COMB_QUANT_NONE, false,
        NCCL_EP_LAYOUT_RANK_MAJOR, false, 7168, ncclBfloat16), LlCombineAlgo::kDefault);
}
