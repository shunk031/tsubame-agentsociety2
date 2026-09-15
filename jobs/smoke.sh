#!/usr/bin/env bash

# @file jobs/smoke.sh
# @brief Start vLLM on the allocated GPUs and check that it answers correctly.
# @description
#   Deliberately stops short of AgentSociety. When a full run fails it is rarely
#   obvious whether vLLM or the simulator is at fault, so this job settles the
#   vLLM half on its own.
#
#   Three things are checked, in order of how quietly they fail:
#     1. the server becomes healthy and lists the model;
#     2. a chat completion returns non-empty content;
#     3. no reasoning trace comes back. Qwen3.5 and newer default to thinking
#        mode, and vLLM issue #35574 reports `enable_thinking: false` being
#        ignored. Silent thinking would not break a run, it would just inflate
#        output tokens and quietly cost throughput, so it is asserted here.
#
#   @example
#     scripts/submit.sh jobs/smoke.sh
#
#   @example
#     scripts/submit.sh jobs/smoke.sh -l node_f=1 \
#       -v MODEL=Qwen/Qwen3.6-35B-A3B-FP8

#$ -cwd
#$ -V
#$ -N as2-smoke
#$ -j y
#$ -l gpu_1=1
# An hour, not the few minutes the checks need. The first run of a Qwen3.5-class
# hybrid model JIT-compiles FlashInfer's GDN prefill kernels, which took eight
# minutes here and logs nothing while it works. The result is cached under
# ~/.cache/flashinfer, so later runs start in about a minute. Drop to
# `-l h_rt=0:15:00` once the cache is warm, or set
# VLLM_EXTRA_ARGS='--gdn-prefill-backend triton' to skip the JIT entirely.
#$ -l h_rt=1:00:00

set -euo pipefail

# Grid Engine runs a spooled copy of this file under /var/spool/age/<node>/, so
# BASH_SOURCE points somewhere with no repository around it. SGE_O_WORKDIR holds
# the directory the job was submitted from, which `#$ -cwd` also makes the
# working directory -- which scripts/submit.sh makes this submission's source
# snapshot, and which it also passes explicitly as REPO_ROOT. The BASH_SOURCE
# fallback is for running this script directly from an interactive iqrsh
# session.
REPO_ROOT="${REPO_ROOT:-${SGE_O_WORKDIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}}"
# shellcheck source=../scripts/lib/common.sh
source "${REPO_ROOT}/scripts/lib/common.sh"

RUN_DIR="${RUNS_DIR}/smoke-${JOB_ID:-local-$$}"
mkdir -p "${RUN_DIR}"

log "host    $(hostname)"
log "job     ${JOB_ID:-<interactive>}"
log_source_snapshot
log "run dir ${RUN_DIR}"
log "model   ${MODEL}"

# A smoke test against the wrong environment is as misleading as a simulation
# against one: it reports that "the environment works" about an environment
# nobody asked for. The check costs a fraction of a second.
assert_env_matches_source "${VENV}" "${REPO_ROOT}"

GPU_COUNT="$(detect_gpu_count)"
[[ "${GPU_COUNT}" -gt 0 ]] || die "no GPU visible; submit with a GPU resource type"
DP_SIZE="${DP_SIZE:-${GPU_COUNT}}"
log "gpus    ${GPU_COUNT} (data parallel size ${DP_SIZE})"

# Weights are staged by scripts/setup/03_download_models.sh. Refusing to reach
# the network keeps a missing cache from silently turning into a long download
# that eats the reservation.
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"

start_vllm "${DP_SIZE}" "${RUN_DIR}/vllm.log"

log "checking the OpenAI-compatible endpoint"
MODEL="${MODEL}" \
ENDPOINT="http://${VLLM_HOST}:${VLLM_PORT}" \
    "${VENV}/bin/python" "${REPO_ROOT}/scripts/check_endpoint.py"

log "smoke test passed"
