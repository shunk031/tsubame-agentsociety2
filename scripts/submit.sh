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

REMOTE_REPO="${REMOTE_REPO:-tsubame-agentsociety2}"

"${SCRIPT_DIR}/sync.sh"

qsub_args=(-g "${TSUBAME_GROUP}")
if [[ -n "${AR_ID:-}" ]] && [[ -z "${NO_AR:-}" ]]; then
    qsub_args+=(-ar "${AR_ID}")
    log "submitting into reservation ${AR_ID}"
else
    qsub_args+=(-q prior)
    log "submitting to the prior queue"
fi
qsub_args+=("$@" "${JOB_SCRIPT}")

# REMOTE_REPO and the qsub arguments are expanded here on purpose: the remote
# side just runs the finished command line.
# shellcheck disable=SC2029
submission="$(
    ssh "${TSUBAME_USER}@${TSUBAME_LOGIN_HOST}" \
        "cd ~/${REMOTE_REPO} && qsub $(printf '%q ' "${qsub_args[@]}")"
)"
echo "${submission}"

# "Your job 8634774 ("as2-smoke") has been submitted"
job_id="$(printf '%s' "${submission}" | awk '{print $3; exit}')"
[[ "${job_id}" =~ ^[0-9]+$ ]] || die "could not parse a job id from: ${submission}"

if [[ -n "${NO_WATCH:-}" ]]; then
    log "submitted job ${job_id}; follow it with scripts/watch.sh ${job_id}"
    exit 0
fi

# The output file only appears once the job starts, which can take a moment in
# a busy queue.
log "waiting for job ${job_id} to start"
sleep 10

exec "${SCRIPT_DIR}/watch.sh" "${job_id}"
