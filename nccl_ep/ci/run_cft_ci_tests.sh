#!/usr/bin/env bash
# Run inside one exclusive, single-node allocation with four GPUs.
set -euo pipefail

MODE="${1:-}"
case "${MODE}" in
  missing-api|supported) ;;
  *) echo "Usage: $0 missing-api|supported" >&2; exit 1 ;;
esac
: "${SLURM_JOB_ID:?requires a Slurm allocation}"
: "${CFT_INSTALL_DIR:?must contain the CUDA 13.4 NCCL EP build}"
: "${CFT_CUDA_HOME:?must select the JIT toolkit inside the container}"
: "${CFT_CONTAINER_IMAGE:?must select the CI container}"
: "${CFT_CONTAINER_MOUNTS:?must mount the build and JIT toolkit}"

LOG_DIR="${CFT_CI_LOG_DIR:-${PWD}/cft-ci-results}"
mkdir -p "${LOG_DIR}"
PROBE="${CFT_INSTALL_DIR}/test/nccl_ep/test_cft_compatibility"
BENCH="${CFT_INSTALL_DIR}/test/nccl_ep/ep_bench"
EP_TEST="${CFT_INSTALL_DIR}/test/nccl_ep/ep_test"
STEP_TIMEOUT="${CFT_CI_STEP_TIMEOUT:-600}"
unset LD_PRELOAD

SRUN=(srun --jobid="${SLURM_JOB_ID}" -N1 -n4 --ntasks-per-node=4
      --mpi=pmix --cpu-bind=none --kill-on-bad-exit=1 --export=ALL
      --container-image="${CFT_CONTAINER_IMAGE}"
      --container-mounts="${CFT_CONTAINER_MOUNTS}" --container-workdir="${PWD}")

# Use container libraries and tools, not MPI or CUDA inherited from the runner.
RANK_ENV=(env -u LD_PRELOAD -u NCCL_EP_LL_CFT -u NCCL_EP_LL_CFT_DEBUG_MIN_DRIVER
          CUDA_HOME="${CFT_CUDA_HOME}" MPI_HOME=/usr/local/openmpi
          NCCL_HOME="${CFT_INSTALL_DIR}" NCCL_EP_HOME="${CFT_INSTALL_DIR}"
          NCCL_EP_JIT_NVCC="${CFT_CUDA_HOME}/bin/nvcc"
          NCCL_EP_JIT_SOURCE_DIR="${CFT_INSTALL_DIR}/include/nccl_ep"
          NCCL_EP_JIT_BUILD_INCLUDE_DIR="${CFT_INSTALL_DIR}/include"
          NCCL_EP_JIT_CUDA_INCLUDE_DIR="${CFT_CUDA_HOME}/include"
          NCCL_EP_JIT_CACHE_DIR="${CFT_INSTALL_DIR}/jit-${MODE}-${SLURM_JOB_ID}"
          PATH="${CFT_CUDA_HOME}/bin:/usr/local/openmpi/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
          LD_LIBRARY_PATH="${CFT_INSTALL_DIR}/lib:${CFT_CUDA_HOME}/lib64:${CFT_CUDA_HOME}/lib:/usr/local/openmpi/lib"
          OMPI_MCA_pml=ob1 "OMPI_MCA_btl=self,vader,tcp"
          OMPI_MCA_coll_hcoll_enable=0 OMPI_MCA_opal_cuda_support=0
          NCCL_GIN_TYPE=3
          LD_BIND_NOW=1)

run_step() {
    local name="$1" cft="$2"
    shift 2
    local transport=()
    [[ "${cft}" == 0 ]] || transport=(NCCL_EP_LL_CFT=1 NCCL_EP_DEBUG=1)
    echo "CFT CI: ${MODE} / ${name}"
    timeout --signal=TERM --kill-after=30 "${STEP_TIMEOUT}" \
        "${SRUN[@]}" "${RANK_ENV[@]}" "${transport[@]}" "$@" \
        2>&1 | tee "${LOG_DIR}/${name}.log"
}

# shellcheck disable=SC2016 # Expand these variables inside the container.
run_step provenance 0 bash -e -c '
    if [[ ${SLURM_PROCID} == 0 ]]; then
        cd "$NCCL_EP_HOME"
        sha256sum -c SHA256SUMS
        "$CUDA_HOME/bin/nvcc" --version
        nvidia-smi --query-gpu=index,name,pci.bus_id,uuid,driver_version --format=csv
        for binary in "$@"; do
            dependencies=$(ldd "$binary")
            printf "%s\n%s\n" "$binary" "$dependencies"
            found_ep=0
            found_nccl=0
            while read -r soname arrow library rest; do
                case "$soname" in
                    libnccl_ep.so*) found_ep=1 ;;
                    libnccl.so*) found_nccl=1 ;;
                    *) continue ;;
                esac
                actual=$(realpath -e "$library")
                expected=$(realpath -e "$NCCL_EP_HOME/lib/$soname")
                if [[ "$arrow" != "=>" || "$actual" != "$expected" ]]; then
                    echo "ERROR: $binary loaded $soname outside the tested build" >&2
                    exit 1
                fi
            done <<<"$dependencies"
            if [[ "$found_ep" != 1 || "$found_nccl" != 1 ]]; then
                echo "ERROR: $binary must load the tested shared EP and NCCL libraries" >&2
                exit 1
            fi
        done
    fi
' _ "${PROBE}" "${BENCH}" "${EP_TEST}"
run_step driver_compatibility 0 "${PROBE}" "${MODE}"

for layout in rm em; do
    for tokens in 32 512; do
        run_step "store_${layout}_${tokens}" 0 "${BENCH}" \
            --algorithm low-latency --layout "${layout}" --tokens "${tokens}" \
            --hidden 7168 --experts 128 --top-k 8 --datatype bf16 \
            --warmup 2 --iters 5 --validate
        if [[ "${MODE}" == supported ]]; then
            run_step "cft_${layout}_${tokens}" 1 "${BENCH}" \
                --algorithm low-latency --layout "${layout}" --tokens "${tokens}" \
                --hidden 7168 --experts 128 --top-k 8 --datatype bf16 \
                --warmup 2 --iters 5 --validate
        fi
    done
done
if [[ "${MODE}" == supported ]]; then
    run_step ep_test_ll 1 "${EP_TEST}" -a ll -t 128 -d 7168
fi
