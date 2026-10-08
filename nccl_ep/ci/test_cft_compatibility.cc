/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <mpi.h>
#include <cuda.h>
#include <nccl_device.h>
#include <nccl_ep.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "nccl_ep_cft.h"

#if CUDA_VERSION < 13040
#error "CFT compatibility CI must be built with CUDA 13.4 or later"
#endif

namespace {
int rank = -1;

void require(bool condition, const char* message) {
    if (condition) return;
    std::fprintf(stderr, "CFT compatibility: rank %d: %s\n", rank, message);
    MPI_Abort(MPI_COMM_WORLD, 1);
    std::exit(1);
}

#define MPI_CHECK(call) require((call) == MPI_SUCCESS, #call)
#define CUDA_CHECK(call) require((call) == cudaSuccess, #call)
#define NCCL_CHECK(call) require((call) == ncclSuccess, #call)
}  // namespace

int main(int argc, char** argv) {
    if (argc != 2 || (std::strcmp(argv[1], "missing-api") != 0 && std::strcmp(argv[1], "supported") != 0)) {
        std::fprintf(stderr, "Usage: %s missing-api|supported\n", argv[0]);
        return 1;
    }
    const bool supported = std::strcmp(argv[1], "supported") == 0;
    if (MPI_Init(&argc, &argv) != MPI_SUCCESS) return 1;
    MPI_CHECK(MPI_Comm_rank(MPI_COMM_WORLD, &rank));
    int ranks = 0;
    MPI_CHECK(MPI_Comm_size(MPI_COMM_WORLD, &ranks));
    require(ranks == 4, "requires exactly four ranks");

    MPI_Comm local_comm;
    MPI_CHECK(MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &local_comm));
    int local_rank = 0, local_ranks = 0;
    MPI_CHECK(MPI_Comm_rank(local_comm, &local_rank));
    MPI_CHECK(MPI_Comm_size(local_comm, &local_ranks));
    MPI_CHECK(MPI_Comm_free(&local_comm));
    require(local_ranks == ranks, "all ranks must run on one node");
    CUDA_CHECK(cudaSetDevice(local_rank));
    char device_bus[32] = {};
    char devices[4][32] = {};
    CUDA_CHECK(cudaDeviceGetPCIBusId(device_bus, sizeof(device_bus), local_rank));
    MPI_CHECK(MPI_Allgather(device_bus, sizeof(device_bus), MPI_CHAR,
                           devices, sizeof(device_bus), MPI_CHAR, MPI_COMM_WORLD));
    for (int i = 0; i < ranks; ++i)
        for (int j = i + 1; j < ranks; ++j)
            require(std::strcmp(devices[i], devices[j]) != 0, "each rank must use a different GPU");

    // Query the loaded production library, including the toolkit it was built with.
    ncclEpCftPlatformFacts facts;
    nccl_ep_cft_platform_query(local_rank, &facts);
    const auto verdict = nccl_ep_cft_platform_judge(facts, ncclEpCftPlatformFloors{});
    std::printf("CFT compatibility: rank=%d mode=%s %s\n", rank, argv[1],
                nccl_ep_cft_platform_summary(facts, verdict).c_str());
    std::fflush(stdout);
    require(facts.built_with_cft && facts.toolkit_version >= 13040,
            "libnccl_ep must be built with CUDA 13.4 or later");
    require(facts.runtime_version >= 13040, "requires the CUDA 13.4 runtime or later");
    require(facts.query_ok, "CUDA device query failed");
    if (supported) {
        require(verdict.failed == 0, "expected a CFT-capable driver, GPU, and JIT compiler");
    } else {
        require(facts.driver_version > 0 && facts.driver_version < 13030,
                "expected a driver older than the CUDA 13.3 logical endpoint API");
        require(!facts.driver_api_available && (verdict.failed & NCCL_EP_CFT_FAIL_DRIVER_API),
                "expected the real driver to lack required logical endpoint entry points");
    }

    ncclUniqueId id;
    if (rank == 0) NCCL_CHECK(ncclGetUniqueId(&id));
    MPI_CHECK(MPI_Bcast(&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD));
    ncclComm_t comm = nullptr;
    NCCL_CHECK(ncclCommInitRank(&comm, ranks, id, rank));
    ncclCommProperties_t properties = NCCL_COMM_PROPERTIES_INITIALIZER;
    NCCL_CHECK(ncclCommQueryProperties(comm, &properties));
    require(properties.nLsaTeams == 1, "requires one NVLink domain");

    ncclEpGroupConfig_t config = NCCL_EP_GROUP_CONFIG_INIT;
    config.algorithm = NCCL_EP_ALGO_LOW_LATENCY;
    config.num_experts = 128;
    config.max_dispatch_tokens_per_rank = 32;
    config.max_token_bytes = 7168 * sizeof(uint16_t);
    config.zero_copy = NCCL_EP_ZERO_COPY_OFF;

    // First prove this configuration works without CFT on the same communicator.
    require(unsetenv("NCCL_EP_LL_CFT") == 0, "could not disable CFT");
    ncclEpGroup_t group = nullptr;
    NCCL_CHECK(ncclEpCreateGroup(&group, comm, &config));
    require(group != nullptr, "store group creation returned a null handle");
    NCCL_CHECK(ncclEpGroupDestroy(group));
    group = nullptr;

    require(setenv("NCCL_EP_LL_CFT", "1", 1) == 0, "could not enable CFT");
    const ncclResult_t result = ncclEpCreateGroup(&group, comm, &config);
    int local[2] = {static_cast<int>(result), group != nullptr};
    int results[4][2] = {};
    MPI_CHECK(MPI_Allgather(local, 2, MPI_INT, results, 2, MPI_INT, MPI_COMM_WORLD));
    const ncclResult_t expected = supported ? ncclSuccess : ncclInvalidUsage;
    for (int i = 0; i < ranks; ++i) {
        if (rank == 0)
            std::printf("CFT group result: rank=%d result=%d has_group=%d\n", i, results[i][0], results[i][1]);
        require(results[i][0] == expected, "CFT group returned an unexpected result on a rank");
        require(results[i][1] == int(supported), "CFT group handle does not match the expected result");
    }
    if (supported) NCCL_CHECK(ncclEpGroupDestroy(group));
    NCCL_CHECK(ncclCommDestroy(comm));
    CUDA_CHECK(cudaDeviceSynchronize());
    if (rank == 0) std::printf("CFT compatibility: %s PASSED on all four ranks\n", argv[1]);
    MPI_CHECK(MPI_Finalize());
    return 0;
}
