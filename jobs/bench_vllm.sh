#!/usr/bin/env bash

# @file jobs/bench_vllm.sh
# @brief Measure what the GPUs can do, with no agent framework in the way.
# @description
#   Every throughput figure this repository has was taken through
#   agentsociety2, so none of them says what the hardware is capable of and
#   there is no denominator to judge them against. This sweeps concurrency
#   against a plain vLLM server and reports where throughput stops rising.
#
#   Two regimes, measured separately because they are bound by different
#   things. Decode rereads every weight for each token and is memory-bandwidth
#   bound: on an H100 the ratio of compute to bandwidth (about 990 TFLOPS over
#   3.35 TB/s) says roughly 300 concurrent sequences are needed before the
#   arithmetic units matter, which is why a 128-agent run sits at 37 C. Prefill
#   processes a whole prompt at once and is compute-bound immediately. A single
#   averaged number hides which regime a workload is in.
#
#   The GPU sampler runs throughout, so temperature and power can be read
#   against each concurrency level rather than inferred.
#
#   @example
#     scripts/submit.sh jobs/bench_vllm.sh -l node_f=1 -l h_rt=1:00:00
#
#   @example
#     scripts/submit.sh jobs/bench_vllm.sh -l h_rt=1:00:00 -v MODEL=Qwen/Qwen3.6-27B

#$ -cwd
#$ -V
#$ -N as2-bench
#$ -j y
#$ -l h_rt=1:00:00

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-${SGE_O_WORKDIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}}"
# shellcheck source=../scripts/lib/common.sh
source "${REPO_ROOT}/scripts/lib/common.sh"

RUN_DIR="${RUNS_DIR}/bench-${JOB_ID:-local-$$}"
mkdir -p "${RUN_DIR}"

log "host    $(hostname)"
log "job     ${JOB_ID:-<interactive>}"
log "run dir ${RUN_DIR}"
log "model   ${MODEL}"

assert_env_matches_source "${VENV}" "${REPO_ROOT}"

GPU_COUNT="$(detect_gpu_count)"
[[ "${GPU_COUNT}" -gt 0 ]] || die "no GPU visible; submit with a GPU resource type"
DP_SIZE="${DP_SIZE:-${GPU_COUNT}}"
CPU_CORES="$(detect_cpu_cores)"
WORKERS="$(worker_budget "${CPU_CORES}" "${GPU_COUNT}")"
export MAX_JOBS="${MAX_JOBS:-${WORKERS}}"

log "gpus    ${GPU_COUNT} (${PARALLELISM:-tensor} parallel, size ${DP_SIZE})"

export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"

start_gpu_sampler "${RUN_DIR}/gpu.csv"
start_vllm "${DP_SIZE}" "${RUN_DIR}/vllm.log"

LEVELS="${LEVELS:-1,2,4,8,16,32,64,128,256}"

# Decode first, while the KV cache is empty: a long-output sweep run after a
# prefill sweep would inherit its cache pressure and read as slower for a
# reason that has nothing to do with concurrency.
log "sweeping decode (short prompt, long output)"
"${VENV}/bin/python" "${REPO_ROOT}/scripts/bench_vllm.py" \
    --endpoint "http://${VLLM_HOST}:${VLLM_PORT}" \
    --model "${MODEL}" \
    --levels "${LEVELS}" \
    --prompt-tokens 64 \
    --max-tokens 512 \
    --label decode | tee "${RUN_DIR}/decode.txt"

log "sweeping prefill (long prompt, short output)"
"${VENV}/bin/python" "${REPO_ROOT}/scripts/bench_vllm.py" \
    --endpoint "http://${VLLM_HOST}:${VLLM_PORT}" \
    --model "${MODEL}" \
    --levels "${LEVELS}" \
    --prompt-tokens 8192 \
    --max-tokens 16 \
    --label prefill | tee "${RUN_DIR}/prefill.txt"

log "benchmark complete: ${RUN_DIR}"
