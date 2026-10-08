#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
INSTALL="${CFT_INSTALL_DIR:-${ROOT}/cft_install}"
: "${CUDA_HOME:?CUDA 13.4 toolkit required}"
export MPI_HOME="${MPI_HOME:-/usr/local/openmpi}"
export PATH="${CUDA_HOME}/bin:${MPI_HOME}/bin:${PATH}"
export LIBRARY_PATH="${CUDA_HOME}/lib64/stubs:${LIBRARY_PATH:-}"
export LD_LIBRARY_PATH="${INSTALL}/lib:${CUDA_HOME}/lib64:${MPI_HOME}/lib:${LD_LIBRARY_PATH:-}"
GENCODE='-gencode=arch=compute_100,code=sm_100'
NCCL_SOURCE="${NCCL_CI_SOURCE:-${ROOT}/.ci/nccl}"
GTEST_SOURCE="${GTEST_CI_SOURCE:-${ROOT}/.ci/googletest/googletest}"

JOBS="$(nproc)"
echo "build host: $(uname -m), nproc=${JOBS}"
mkdir -p "${INSTALL}"
nvcc --version
make -C "${NCCL_SOURCE}" -j"${JOBS}" CUDA_HOME="${CUDA_HOME}" \
  BUILDDIR="${INSTALL}" NVCC_GENCODE="${GENCODE}" src.build
# Correctness jobs do not need a CUPTI dependency on the older driver.
EP_BUILD=(make -C "${ROOT}/nccl_ep" -j"${JOBS}" CUDA_HOME="${CUDA_HOME}"
  NCCL_HOME="${INSTALL}" NCCL_SUBMODULE_HOME="${NCCL_SOURCE}"
  BUILDDIR="${INSTALL}" MPI=1 MPI_HOME="${MPI_HOME}"
  NVCC_GENCODE="${GENCODE}" CUPTI_CFLAGS= CUPTI_LDFLAGS=)
# Finish shared-library symlinks before -lnccl_ep can select the static archive.
"${EP_BUILD[@]}" lib
"${EP_BUILD[@]}" ep_test ep_bench

"${MPI_HOME}/bin/mpicxx" -std=c++17 \
  -I"${CUDA_HOME}/include" -I"${INSTALL}/include" -I"${ROOT}/nccl_ep" \
  "${ROOT}/nccl_ep/ci/test_cft_compatibility.cc" \
  -L"${INSTALL}/lib" -lnccl_ep -Wl,--no-as-needed -lnccl -Wl,--as-needed \
  -L"${CUDA_HOME}/lib64" -lcudart_static -L"${CUDA_HOME}/lib64/stubs" -lcuda \
  -ldl -lrt -pthread -o "${INSTALL}/test/nccl_ep/test_cft_compatibility"

make -C "${ROOT}/nccl_ep/tests" -j"${JOBS}" NVCC="${CUDA_HOME}/bin/nvcc" \
  NCCL_HOME="${INSTALL}" BUILDDIR="${INSTALL}" GTEST_DIR="${GTEST_SOURCE}" \
  NVCC_GENCODE="${GENCODE}" "${INSTALL}/test/nccl_ep/test_cft_driver_api"
NCCL_EP_BUILDDIR="${INSTALL}" TEST_SUITE=test_cft_driver_api \
  bash "${ROOT}/nccl_ep/tests/run_tests.sh"

{
  git -C "${ROOT}" rev-parse HEAD
  cat "${NCCL_SOURCE}/NCCL_COMMIT"
  nvcc --version
} > "${INSTALL}/provenance.txt"
(
  cd "${INSTALL}"
  sha256sum lib/libnccl.so.2 lib/libnccl_ep.so.0 \
    test/nccl_ep/ep_bench test/nccl_ep/ep_test test/nccl_ep/test_cft_compatibility \
    > SHA256SUMS
)
