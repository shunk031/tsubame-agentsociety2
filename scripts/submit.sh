#!/usr/bin/env bash

# @file scripts/submit.sh
# @brief Sync, submit a job, and follow its logs, all from your own machine.
# @description
#   Wraps the three steps that always go together, so a change here reaches the
#   compute node without a forgotten sync in between.
#
#   Resource type and wall time come from the `#$` directives in the job script
#   and can be overridden per invocation, since Grid Engine lets command-line
#   options win over the embedded ones.
#
#   Submits to the reservation when AR_ID is set in config/env.local, and to the
#   prior queue otherwise. Note that reservations here only admit `node_f`, so a
#   single-GPU smoke run has to go to the prior queue.
#
#   The sync makes a snapshot of its own for this submission and the job is
#   pinned to it, so what a queued job will run stops changing the moment this
#   returns. Grid Engine's output file goes to RUNS_DIR rather than the job's
#   working directory, which is now that snapshot and is meant to be prunable.
#
# @arg $1 string Job script, relative to the repository root.
# @arg $@ string Further arguments are passed through to qsub.
#
# @example
#   scripts/submit.sh jobs/smoke.sh
#
# @example
#   scripts/submit.sh jobs/run_sim.sh -l node_f=1 -l h_rt=3:00:00 \
#     -v MODEL=Qwen/Qwen3.6-35B-A3B-FP8,NUM_AGENTS=64
#
# @example
#   NO_WATCH=1 scripts/submit.sh jobs/smoke.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib/common.sh
source "${SCRIPT_DIR}/lib/common.sh"

: "${TSUBAME_USER:?not set - see config/env.example}"
: "${TSUBAME_LOGIN_HOST:?not set - see config/env.example}"

[[ $# -ge 1 ]] || die "usage: scripts/submit.sh <job script> [qsub args...]"

JOB_SCRIPT="$1"
shift

[[ -f "${REPO_ROOT}/${JOB_SCRIPT}" ]] || die "no such job script: ${JOB_SCRIPT}"

snapshot_dir="$("${SCRIPT_DIR}/sync.sh")"
[[ -n "${snapshot_dir}" ]] || die "sync.sh reported no snapshot directory"

qsub_args=(-g "${TSUBAME_GROUP}")
if [[ -n "${AR_ID:-}" ]] && [[ -z "${NO_AR:-}" ]]; then
    qsub_args+=(-ar "${AR_ID}")
    log "submitting into reservation ${AR_ID}"
else
    qsub_args+=(-q prior)
    log "submitting to the prior queue"
fi
# Added here rather than as a "#$ -l" in the job script: Grid Engine merges
# the two and rejects node_f alongside gpu_1, so a hardcoded default would make
# whole-node submission impossible.
gpu_resource="$(default_gpu_resource "$@")"
[[ -n "${gpu_resource}" ]] && qsub_args+=(-l "${gpu_resource}")

# Grid Engine opens `<job name>.o<job id>` in the job's working directory, which
# `#$ -cwd` makes the snapshot. Keeping it there would tie the log's lifetime to
# a directory that exists to be pruned. A trailing slash asks Grid Engine for
# that same default filename inside RUNS_DIR instead, next to the run's own
# vllm.log and sim.log. Passed before "$@" so an explicit -o still wins.
qsub_args+=(-o "${RUNS_DIR}/")
qsub_args+=("$@" "${JOB_SCRIPT}")

# Two independent things point the job at its snapshot, and they agree:
#
#   cd        makes the snapshot SGE_O_WORKDIR, which `#$ -cwd` turns into the
#             job's working directory and which jobs/*.sh fall back to.
#   REPO_ROOT is exported into qsub's environment, and `#$ -V` carries it to the
#             job, which prefers an explicit REPO_ROOT over the fallback.
#
# The explicit variable is not passed with `-v`. Callers already spend `-v` on
# MODEL and friends, and whether Altair Grid Engine merges repeated `-v` options
# or lets the last one win is not something this repository can check without
# submitting a job. `-V` has no such ambiguity and is already relied on.
#
# The arguments are expanded here on purpose: the remote side just runs the
# finished command line.
# shellcheck disable=SC2029
submission="$(
    ssh "${TSUBAME_USER}@${TSUBAME_LOGIN_HOST}" \
        "cd $(printf '%q' "${snapshot_dir}") && REPO_ROOT=$(printf '%q' "${snapshot_dir}") qsub $(printf '%q ' "${qsub_args[@]}")"
)"
echo "${submission}"

# "Your job 8634774 ("as2-smoke") has been submitted"
job_id="$(printf '%s' "${submission}" | awk '{print $3; exit}')"
[[ "${job_id}" =~ ^[0-9]+$ ]] || die "could not parse a job id from: ${submission}"

log "job ${job_id} is pinned to snapshot $(basename "${snapshot_dir}")"
log "its output will be ${RUNS_DIR}/<job name>.o${job_id}"

if [[ -n "${NO_WATCH:-}" ]]; then
    log "submitted job ${job_id}; follow it with scripts/watch.sh ${job_id}"
    exit 0
fi

# The output file only appears once the job starts, which can take a moment in
# a busy queue.
log "waiting for job ${job_id} to start"
sleep 10

exec "${SCRIPT_DIR}/watch.sh" "${job_id}"
