#!/usr/bin/env bash

# @file jobs/run_cotenant.sh
# @brief Run two independent simulations against one vLLM, offset in time.
# @description
#   One population is a poor customer for a GPU. AgentSociety steps every agent
#   in lockstep, so the work it offers arrives in waves: a step opens with the
#   whole population asking at once, decays as agents finish, and ends with a
#   handful of stragglers holding the barrier while the rest wait. Measured on
#   one H100 with 128 agents, that shape holds a 148-minute run at 81.9% of
#   rated draw on average but only 64% of the time at or above 80% -- the mean
#   clears the target and the floor does not.
#
#   No server-side setting closes that gap, because nothing is queued during the
#   trough: vLLM reports `Waiting: 0` while it idles. The gap is in the arrival
#   pattern, so the fix has to be there too. Two populations stepping out of
#   phase put one's peak over the other's trough.
#
#   The two are independent simulations, not one larger population: the point is
#   that their barriers are unrelated, which a single population of 256 would
#   not give.
#
# @example
#   scripts/submit.sh jobs/run_cotenant.sh -l node_q=1 -l h_rt=8:00:00 \
#     -v NUM_AGENTS=128 -v ENV_MODULE=SocialMediaSpace

#$ -cwd
#$ -V
#$ -N as2-coten
#$ -j y
#$ -l h_rt=1:00:00

set -euo pipefail

# See the same note in jobs/run_sim.sh.
REPO_ROOT="${REPO_ROOT:-${SGE_O_WORKDIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}}"
# shellcheck source=../scripts/lib/common.sh
source "${REPO_ROOT}/scripts/lib/common.sh"

RUN_DIR="${RUNS_DIR}/coten-${JOB_ID:-local-$$}"
mkdir -p "${RUN_DIR}"

# How far the second population starts behind the first. The trough to cover is
# the tail of a step plus the questionnaire that follows it, which ran 6-21
# minutes in the measurements this job exists to answer, so the default puts the
# second population's step opening somewhere inside the first one's tail.
SIM_STAGGER_SECONDS="${SIM_STAGGER_SECONDS:-600}"

log "host    $(hostname)"
log "job     ${JOB_ID:-<interactive>}"
log_source_snapshot
log "run dir ${RUN_DIR}"
log "model   ${MODEL}"
log "env     ${ENV_MODULE}"
log "agents  ${NUM_AGENTS} x2 populations, second starts +${SIM_STAGGER_SECONDS}s"

assert_env_matches_source "${VENV}" "${REPO_ROOT}"

GPU_COUNT="$(detect_gpu_count)"
[[ "${GPU_COUNT}" -gt 0 ]] || die "no GPU visible; submit with a GPU resource type"
DP_SIZE="${DP_SIZE:-${GPU_COUNT}}"
log "gpus    ${GPU_COUNT}"

export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"

start_gpu_sampler "${RUN_DIR}/gpu.csv"
start_vllm "${DP_SIZE}" "${RUN_DIR}/vllm.log"

log "checking the OpenAI-compatible endpoint"
MODEL="${MODEL}" ENDPOINT="http://${VLLM_HOST}:${VLLM_PORT}" \
    "${VENV}/bin/python" "${REPO_ROOT}/scripts/check_endpoint.py"

# Each population gets its own state directory. Sharing one would let the
# codegen template cache of the first seed the second, which would make the two
# arrival patterns depend on each other -- the opposite of what is being tested.
#
# Ray likewise gets a directory each. Both processes call ray.init() with no
# address, so each starts a private local cluster; pointing them at one temp
# root invites two head nodes to contend for the same session files.
#
# @description Launch one population in the background.
# @arg $1 string Label, used for the sub-directory and the log prefix.
start_population() {
    local label="$1" dir="${RUN_DIR}/$1"
    mkdir -p "${dir}/agent-home" "${dir}/ray"

    "${VENV}/bin/python" "${REPO_ROOT}/scripts/gen_config.py" \
        --out-dir "${dir}" \
        --env-module "${ENV_MODULE}" \
        --language "${AGENT_LANGUAGE:-en}" \
        --num-agents "${NUM_AGENTS}" \
        --num-rounds "${NUM_ROUNDS}" \
        --tick "${TICK_SECONDS}" \
        --pool-resources "${POOL_RESOURCES}" \
        --max-extraction "${MAX_EXTRACTION}"

    log "starting population ${label}"
    (
        export AGENTSOCIETY_HOME_DIR="${dir}/agent-home"
        export RAY_TMPDIR="${dir}/ray"
        "${VENV}/bin/python" -m agentsociety2.society.cli \
            --config "${dir}/init_config.json" \
            --steps "${dir}/steps.yaml" \
            --run-dir "${dir}" \
            --batch-size "${BATCH_SIZE}" \
            --log-level "${AGENTSOCIETY_LOG_LEVEL:-INFO}" \
            2>&1 | sed "s/^/[${label}] /" | tee "${dir}/sim.log"
    ) &
}

start_population a
A_PID=$!
log "waiting ${SIM_STAGGER_SECONDS}s before the second population"
sleep "${SIM_STAGGER_SECONDS}"
start_population b
B_PID=$!

# Wait on each explicitly rather than a bare `wait`, so a population that fails
# fails the job instead of being averaged into a partial success.
FAILED=0
wait "${A_PID}" || { log "population a exited non-zero"; FAILED=1; }
wait "${B_PID}" || { log "population b exited non-zero"; FAILED=1; }

for label in a b; do
    log "checking that population ${label} produced data"
    RUN_DIR="${RUN_DIR}/${label}" MIN_REPLAY_RECORDS="${MIN_REPLAY_RECORDS}" \
    MIN_PARTICIPATION="${MIN_PARTICIPATION}" \
        "${VENV}/bin/python" "${REPO_ROOT}/scripts/check_replay.py" || FAILED=1
done

[[ "${FAILED}" -eq 0 ]] || die "at least one population did not complete cleanly"
log "co-tenant run complete: ${RUN_DIR}"
