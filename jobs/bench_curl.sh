#!/usr/bin/env bash

# @file jobs/bench_curl.sh
# @brief Saturate vLLM with parallel curl and watch the GPU, nothing else.
# @description
#   The previous benchmark measured its own client. It used asyncio's default
#   thread pool, which caps at min(32, cpu_count + 4), so every level above 32
#   was queued in the client and the wall time doubled at 64, quadrupled at 128
#   and so on. That was read as the server saturating at 985 tokens a second,
#   and a conclusion about the hardware was drawn from it. It was wrong.
#
#   This removes the client from the question. curl processes are independent
#   and xargs -P imposes the only limit, which is explicit. --max-num-seqs is
#   raised past every level swept, so the server does not impose one either.
#
#   @example
#     scripts/submit.sh jobs/bench_curl.sh -l h_rt=1:00:00

#$ -cwd
#$ -V
#$ -N as2-curl
#$ -j y
#$ -l h_rt=1:00:00

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-${SGE_O_WORKDIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}}"
# shellcheck source=../scripts/lib/common.sh
source "${REPO_ROOT}/scripts/lib/common.sh"

RUN_DIR="${RUNS_DIR}/curl-${JOB_ID:-local-$$}"
mkdir -p "${RUN_DIR}"

log "host    $(hostname)"
log "run dir ${RUN_DIR}"
log "model   ${MODEL}"

assert_env_matches_source "${VENV}" "${REPO_ROOT}"

GPU_COUNT="$(detect_gpu_count)"
[[ "${GPU_COUNT}" -gt 0 ]] || die "no GPU visible; submit with a GPU resource type"
DP_SIZE="${DP_SIZE:-${GPU_COUNT}}"
CPU_CORES="$(detect_cpu_cores)"
export MAX_JOBS="${MAX_JOBS:-$(worker_budget "${CPU_CORES}" "${GPU_COUNT}")}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"

# Past every level swept below, so any ceiling that appears belongs to the
# hardware rather than to a server-side queue.
export MAX_NUM_SEQS="${MAX_NUM_SEQS:-1024}"

log "gpus    ${GPU_COUNT} (${PARALLELISM:-tensor} parallel), max-num-seqs ${MAX_NUM_SEQS}"

start_gpu_sampler "${RUN_DIR}/gpu.csv"
start_vllm "${DP_SIZE}" "${RUN_DIR}/vllm.log"

ENDPOINT="http://${VLLM_HOST}:${VLLM_PORT}/v1/completions"
OUT_TOKENS="${OUT_TOKENS:-512}"
# A prompt long enough to be worth a request and short enough that the level is
# measuring decode rather than prefill.
PROMPT="$(printf 'token %.0s' $(seq 64))"
printf '{"model":"%s","prompt":"%s","max_tokens":%s,"ignore_eos":true,"temperature":0}' \
    "${MODEL}" "${PROMPT}" "${OUT_TOKENS}" >"${RUN_DIR}/body.json"

printf 'conc  wall_s  tok/s   温度C  電力W\n' | tee "${RUN_DIR}/result.txt"

for conc in ${LEVELS:-1 8 32 64 128 256 512}; do
    started="$(date +%s.%N)"
    # Each curl is its own process, so nothing in this script serialises them;
    # -P is the only limit and it is the variable under test.
    seq "${conc}" | xargs -P "${conc}" -I{} \
        curl -s -o /dev/null -X POST "${ENDPOINT}" \
        -H 'Content-Type: application/json' \
        --data-binary "@${RUN_DIR}/body.json" || true
    finished="$(date +%s.%N)"

    wall="$(awk -v a="${started}" -v b="${finished}" 'BEGIN{printf "%.1f", b-a}')"
    rate="$(awk -v c="${conc}" -v o="${OUT_TOKENS}" -v w="${wall}" \
        'BEGIN{printf "%.0f", (w>0) ? c*o/w : 0}')"
    # Read the samples this level produced, not the run's average: the point is
    # how temperature and power move with concurrency.
    stats="$(awk -F, -v s="${started}" 'NR>1 && $5+0>1000 {n++; t+=$7; p+=$6}
        END{if(n) printf "%.0f %.0f", t/n, p/n; else printf "- -"}' \
        "${RUN_DIR}/gpu.csv")"
    printf '%4s  %6s  %6s  %s\n' "${conc}" "${wall}" "${rate}" "${stats}" \
        | tee -a "${RUN_DIR}/result.txt"
done

log "benchmark complete: ${RUN_DIR}"
