#!/usr/bin/env bash

# @file jobs/template_cache_probe.sh
# @brief Ask the embedding server what the codegen template cache actually sees.
# @description
#   Two 128-agent runs recorded zero template-cache hits, with eight in ten
#   misses reported as below_similarity_threshold -- including for instruction
#   strings byte-identical to one already cached. Every other explanation was
#   ruled out from the logs: the lookup is an IndexFlatIP over L2-normalised
#   vectors, so its score is a cosine similarity and not a distance; the
#   embedding server started healthy and was still serving; nothing errored.
#
#   What remains can only be measured where the server runs, which is why this
#   is a job and not a script. It starts the embedding server alone -- no
#   generation model, so it costs a minute rather than the twenty a simulation
#   spends loading weights -- and reports the similarity an identical string
#   receives.
#
#   @example
#     scripts/submit.sh jobs/template_cache_probe.sh -l h_rt=0:30:00

#$ -cwd
#$ -V
#$ -N as2-probe
#$ -j y
#$ -l h_rt=0:30:00

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-${SGE_O_WORKDIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}}"
# shellcheck source=../scripts/lib/common.sh
source "${REPO_ROOT}/scripts/lib/common.sh"

RUN_DIR="${RUNS_DIR}/probe-${JOB_ID:-local-$$}"
mkdir -p "${RUN_DIR}"

log "host    $(hostname)"
log "job     ${JOB_ID:-<interactive>}"
log "run dir ${RUN_DIR}"
log "model   ${EMBEDDING_MODEL}"

assert_env_matches_source "${VENV}" "${REPO_ROOT}"

GPU_COUNT="$(detect_gpu_count)"
[[ "${GPU_COUNT}" -gt 0 ]] || die "no GPU visible; submit with a GPU resource type"

export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"

start_embedding_vllm "${RUN_DIR}/vllm-embedding.log"

log "measuring what the cache lookup would score"
EMBEDDING_ENDPOINT="http://${VLLM_HOST}:${EMBEDDING_PORT}" \
EMBEDDING_MODEL="${EMBEDDING_MODEL}" \
    "${VENV}/bin/python" "${REPO_ROOT}/scripts/check_template_similarity.py"

log "probe complete: ${RUN_DIR}"
