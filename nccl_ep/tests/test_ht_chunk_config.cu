/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include <gtest/gtest.h>
#include <climits>
#include "nccl_ep_env.h"

TEST(HtChunkConfig, CombinePreservesDefault) {
    ncclEpEnvConfig cfg;
    for (int sms : {1, 8, 16, 17, 32, 64, 152, 212}) {
        EXPECT_EQ(nccl_ep_resolve_combine_chunk(cfg, sms, 1), (4 * sms + 31) & ~31);
        EXPECT_EQ(nccl_ep_resolve_combine_chunk(cfg, sms, 2), 64);
    }
}

TEST(HtChunkConfig, CombineOverrideAndRounding) {
    ncclEpEnvConfig cfg;
    cfg.tokens_per_chunk.is_set = true;
    for (int teams : {1, 2}) {
        for (unsigned long value : {1ul, 31ul, 32ul, 33ul, 64ul, 96ul, 128ul}) {
            cfg.tokens_per_chunk.value.ul = value;
            EXPECT_EQ(nccl_ep_resolve_combine_chunk(cfg, 16, teams), (value + 31) & ~31ul);
        }
    }
    cfg.tokens_per_chunk.value.ul = 0;
    EXPECT_EQ(nccl_ep_resolve_combine_chunk(cfg, 32, 1), 128);
    EXPECT_EQ(nccl_ep_resolve_combine_chunk(cfg, 32, 2), 64);
}

TEST(HtChunkConfig, DispatchOverrideDoesNotChangeCombine) {
    ncclEpEnvConfig cfg;
    cfg.dispatch_tokens_per_chunk.is_set = true;
    cfg.dispatch_tokens_per_chunk.value.ul = 32;
    const int combine = nccl_ep_resolve_combine_chunk(cfg, 32, 1);
    EXPECT_EQ(combine, 128);
    EXPECT_EQ(nccl_ep_resolve_dispatch_chunk(cfg, combine, 1), 32);
    EXPECT_EQ(nccl_ep_resolve_dispatch_chunk(cfg, combine, 2), 128);
}

TEST(HtChunkConfig, PreservesDefault) {
    ncclEpEnvConfig cfg;
    EXPECT_EQ(nccl_ep_resolve_dispatch_chunk(cfg, 64, 1), 64);
    EXPECT_EQ(nccl_ep_resolve_dispatch_chunk(cfg, 128, 2), 128);
}

TEST(HtChunkConfig, ValidOverride) {
    ncclEpEnvConfig cfg;
    cfg.dispatch_tokens_per_chunk.is_set = true;
    for (unsigned long value : {32ul, 64ul, 128ul}) {
        cfg.dispatch_tokens_per_chunk.value.ul = value;
        EXPECT_EQ(nccl_ep_resolve_dispatch_chunk(cfg, 128, 1), value);
    }
}

TEST(HtChunkConfig, RejectsInvalidAndRdma) {
    ncclEpEnvConfig cfg;
    cfg.dispatch_tokens_per_chunk.is_set = true;
    for (unsigned long value : {0ul, 1ul, 31ul, 48ul, 96ul, 256ul, ULONG_MAX}) {
        cfg.dispatch_tokens_per_chunk.value.ul = value;
        EXPECT_EQ(nccl_ep_resolve_dispatch_chunk(cfg, 128, 1), 128);
    }
    cfg.dispatch_tokens_per_chunk.value.ul = 32;
    EXPECT_EQ(nccl_ep_resolve_dispatch_chunk(cfg, 64, 2), 64);
}
