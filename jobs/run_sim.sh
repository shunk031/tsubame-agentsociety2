#!/usr/bin/env bash

# @file jobs/run_sim.sh
# @brief Run an AgentSociety simulation against a vLLM server on the same node.
# @description
#   Starts vLLM on the allocated GPUs, generates the configuration pair the CLI
#   needs, runs the simulation, and then checks that the run actually produced
#   data.
#
#   That last step is not ceremony. agentsociety2 swallows exceptions on several
#   paths — a failing embedding call, for instance, degrades to a cache miss and
#   logs a warning — so a run whose LLM calls all failed can still exit 0. The
#   replay files are the evidence that something happened.
#
#   @example
#     scripts/submit.sh jobs/run_sim.sh
#
#   @example
#     scripts/submit.sh jobs/run_sim.sh -l node_f=1 -l h_rt=3:00:00 \
#       -v MODEL=Qwen/Qwen3.6-35B-A3B-FP8,NUM_AGENTS=16,NUM_ROUNDS=10

#$ -cwd
#$ -V
#$ -N as2-sim
#$ -j y
#$ -l gpu_1=1
#$ -l h_rt=1:00:00

set -euo pipefail

# Grid Engine runs a spooled copy of this file, so BASH_SOURCE points somewhere
# with no repository around it. See the same note in jobs/smoke.sh.
REPO_ROOT="${REPO_ROOT:-${SGE_O_WORKDIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}}"
# shellcheck source=../scripts/lib/common.sh
source "${REPO_ROOT}/scripts/lib/common.sh"

RUN_DIR="${RUNS_DIR}/sim-${JOB_ID:-local-$$}"
mkdir -p "${RUN_DIR}"

log "host    $(hostname)"
log "job     ${JOB_ID:-<interactive>}"
log "run dir ${RUN_DIR}"
log "model   ${MODEL}"
log "env     ${ENV_MODULE}"
log "agents  ${NUM_AGENTS:-<env default>} over ${NUM_ROUNDS} rounds of ${TICK_SECONDS}s"

GPU_COUNT="$(detect_gpu_count)"
[[ "${GPU_COUNT}" -gt 0 ]] || die "no GPU visible; submit with a GPU resource type"
DP_SIZE="${DP_SIZE:-${GPU_COUNT}}"

# Ray reads os.cpu_count() otherwise, which reports the whole physical node
# regardless of how many slots Grid Engine granted, and over-subscribes the job.
CPU_CORES="$(detect_cpu_cores)"
export AGENTSOCIETY_LLM_RAY_MAX_WORKERS="${AGENTSOCIETY_LLM_RAY_MAX_WORKERS:-${CPU_CORES}}"

# ray.init is called without _temp_dir, so Ray falls back to /tmp. Grid Engine
# gives each job a private TMPDIR on node-local storage; pointing Ray at it
# keeps runs from colliding over a shared path.
export RAY_TMPDIR="${RAY_TMPDIR:-${TMPDIR:-/tmp}}"

log "cpus    ${CPU_CORES} (Ray workers ${AGENTSOCIETY_LLM_RAY_MAX_WORKERS})"
log "gpus    ${GPU_COUNT} (data parallel size ${DP_SIZE})"

export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"

start_vllm "${DP_SIZE}" "${RUN_DIR}/vllm.log"

# Second, and after the generation server: it is the one whose memory budget
# matters, and starting it first means a failure there costs nothing else.
if [[ "${ENABLE_EMBEDDING}" != "0" ]]; then
    start_embedding_vllm "${RUN_DIR}/vllm-embedding.log"
    export AGENTSOCIETY_EMBEDDING_API_BASE="http://${VLLM_HOST}:${EMBEDDING_PORT}/v1"
    export AGENTSOCIETY_EMBEDDING_MODEL="${EMBEDDING_MODEL}"
    log "embeddings   ${AGENTSOCIETY_EMBEDDING_MODEL} at ${AGENTSOCIETY_EMBEDDING_API_BASE}"
else
    log "embeddings   disabled; every ask_env will miss the codegen cache"
fi

log "verifying the endpoint before handing it to the simulator"
MODEL="${MODEL}" \
ENDPOINT="http://${VLLM_HOST}:${VLLM_PORT}" \
    "${VENV}/bin/python" "${REPO_ROOT}/scripts/check_endpoint.py"

log "generating the configuration"
"${VENV}/bin/python" "${REPO_ROOT}/scripts/gen_config.py" \
    --out-dir "${RUN_DIR}" \
    --env-module "${ENV_MODULE}" \
    --num-agents "${NUM_AGENTS}" \
    --num-rounds "${NUM_ROUNDS}" \
    --tick "${TICK_SECONDS}" \
    --pool-resources "${POOL_RESOURCES}" \
    --max-extraction "${MAX_EXTRACTION}"

# `python -m`, not the `agentsociety` console script: agentsociety 1.x defines a
# console script by that same name, so the module path is unambiguous.
log "running the simulation"
"${VENV}/bin/python" -m agentsociety2.society.cli \
    --config "${RUN_DIR}/init_config.json" \
    --steps "${RUN_DIR}/steps.yaml" \
    --run-dir "${RUN_DIR}" \
    --log-level "${AGENTSOCIETY_LOG_LEVEL:-INFO}" \
    2>&1 | tee "${RUN_DIR}/sim.log"

log "checking that the run produced data"
RUN_DIR="${RUN_DIR}" MIN_REPLAY_RECORDS="${MIN_REPLAY_RECORDS}" \
MIN_PARTICIPATION="${MIN_PARTICIPATION}" \
    "${VENV}/bin/python" "${REPO_ROOT}/scripts/check_replay.py"

log "simulation complete: ${RUN_DIR}"
