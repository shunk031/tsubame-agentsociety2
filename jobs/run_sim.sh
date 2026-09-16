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
#$ -l h_rt=1:00:00

set -euo pipefail

# Grid Engine runs a spooled copy of this file, so BASH_SOURCE points somewhere
# with no repository around it. See the same note in jobs/smoke.sh.
#
# Submitted through scripts/submit.sh, REPO_ROOT arrives already set to this
# submission's source snapshot and both fallbacks are unused. They still matter:
# SGE_O_WORKDIR is the same snapshot by another route, and BASH_SOURCE is for
# running this file by hand from an iqrsh session.
REPO_ROOT="${REPO_ROOT:-${SGE_O_WORKDIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}}"
# shellcheck source=../scripts/lib/common.sh
source "${REPO_ROOT}/scripts/lib/common.sh"

RUN_DIR="${RUNS_DIR}/sim-${JOB_ID:-local-$$}"
mkdir -p "${RUN_DIR}"

# Keep the codegen template cache inside the run unless asked otherwise, so a
# run's numbers do not depend on which runs preceded it. See the note in
# scripts/lib/common.sh.
if [[ "${AGENT_HOME_MODE}" == "per-run" ]]; then
    export AGENTSOCIETY_HOME_DIR="${AGENTSOCIETY_HOME_DIR:-${RUN_DIR}/agent-home}"
else
    export AGENTSOCIETY_HOME_DIR="${AGENTSOCIETY_HOME_DIR:-${WORK_ROOT}/agent-home}"
fi
mkdir -p "${AGENTSOCIETY_HOME_DIR}"

log "host    $(hostname)"
log "job     ${JOB_ID:-<interactive>}"
log_source_snapshot
log "run dir ${RUN_DIR}"
log "model   ${MODEL}"
log "env     ${ENV_MODULE}"
log "state   ${AGENT_HOME_MODE} (${AGENTSOCIETY_HOME_DIR})"
log "agents  ${NUM_AGENTS:-<env default>} over ${NUM_ROUNDS} rounds of ${TICK_SECONDS}s"

# Before anything expensive. An environment built from other source runs, exits
# 0 and writes a replay that reads like every other replay, so the only place to
# catch it is here -- ahead of the model load, not after the numbers exist.
assert_env_matches_source "${VENV}" "${REPO_ROOT}"

GPU_COUNT="$(detect_gpu_count)"
[[ "${GPU_COUNT}" -gt 0 ]] || die "no GPU visible; submit with a GPU resource type"
DP_SIZE="${DP_SIZE:-${GPU_COUNT}}"

# Ray reads os.cpu_count() otherwise, which reports the whole physical node
# regardless of how many slots Grid Engine granted, and over-subscribes the job.
CPU_CORES="$(detect_cpu_cores)"
# Not the raw core count. Grid Engine reports no NSLOTS for a whole node, so
# detect_cpu_cores falls through to nproc and answers with the machine's 192 --
# more workers than there are agents, which floors the batch to one agent per
# task. See worker_budget for what that cost when it happened.
WORKERS="$(worker_budget "${CPU_CORES}" "${GPU_COUNT}")"
export AGENTSOCIETY_LLM_RAY_MAX_WORKERS="${AGENTSOCIETY_LLM_RAY_MAX_WORKERS:-${WORKERS}}"

# Declaring the worker budget above buys nothing on its own: the tick only
# submits ceil(NUM_AGENTS / batch) tasks, and upstream's default batch of 256
# makes that exactly one for every population run here. Size the batch to the
# budget so the workers actually receive work.
BATCH_SIZE="${BATCH_SIZE:-$(ray_batch_size "${NUM_AGENTS}" "${AGENTSOCIETY_LLM_RAY_MAX_WORKERS}")}"
RAY_TASKS=$(( NUM_AGENTS > 0 ? (NUM_AGENTS + BATCH_SIZE - 1) / BATCH_SIZE : 1 ))

# vLLM JIT-compiles CUDA kernels at startup (FlashInfer's gated-delta-net
# prefill, among others) and ninja sizes itself from nproc unless MAX_JOBS says
# otherwise. nproc reports the whole physical node, so ninja fans out to
# hundreds of nvcc processes, the kernel OOM killer takes cicc with signal 9,
# and the only visible symptom is "Ninja build failed" and a vLLM that never
# binds its port. Nothing caps this job's memory (see the probe below), so the
# build competes for the node's physical RAM with whatever else is running
# there and the victim need not be ours. Three 35B jobs died this way before
# the cause was found; the granted slot count is the bound that keeps the
# build proportional to the slots actually scheduled.
export MAX_JOBS="${MAX_JOBS:-${WORKERS}}"

# ray.init is called without _temp_dir, so Ray falls back to /tmp. Grid Engine
# gives each job a private TMPDIR on node-local storage; pointing Ray at it
# keeps runs from colliding over a shared path.
export RAY_TMPDIR="${RAY_TMPDIR:-${TMPDIR:-/tmp}}"

# What the JIT compile has to fit inside. Measured here rather than assumed,
# because the answer decides who a runaway build hurts: the first reading came
# back as the cgroup-v1 "no limit" sentinel, which means nothing caps this job
# and an over-parallel ninja competes for the node's physical memory with
# whatever else is running on it.
MEM_LIMIT="$(cat /sys/fs/cgroup/memory.max 2>/dev/null \
    || cat /sys/fs/cgroup/memory/memory.limit_in_bytes 2>/dev/null \
    || echo unknown)"
# 2^63 rounded down to a page boundary is how cgroup v1 spells "unlimited".
[[ "${MEM_LIMIT}" == "9223372036854771712" ]] && MEM_LIMIT="unlimited"

log "cpus    ${CPU_CORES} available, ${AGENTSOCIETY_LLM_RAY_MAX_WORKERS} Ray workers, ${MAX_JOBS} JIT jobs"
log "memory  ${MEM_LIMIT} (cgroup limit)"
log "batch   ${BATCH_SIZE} agents per task, ${RAY_TASKS} task(s) per tick"
log "gpus    ${GPU_COUNT} (data parallel size ${DP_SIZE})"

export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"

# Before vLLM, so the trace covers model load and warmup as well as the
# simulation: a run that looks idle is a different problem from one that never
# got started, and the difference is visible only in the first minutes.
start_gpu_sampler "${RUN_DIR}/gpu.csv"

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
    --language "${AGENT_LANGUAGE:-en}" \
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
    --batch-size "${BATCH_SIZE}" \
    --log-level "${AGENTSOCIETY_LOG_LEVEL:-INFO}" \
    2>&1 | tee "${RUN_DIR}/sim.log"

log "checking that the run produced data"
RUN_DIR="${RUN_DIR}" MIN_REPLAY_RECORDS="${MIN_REPLAY_RECORDS}" \
MIN_PARTICIPATION="${MIN_PARTICIPATION}" \
    "${VENV}/bin/python" "${REPO_ROOT}/scripts/check_replay.py"

log "simulation complete: ${RUN_DIR}"
