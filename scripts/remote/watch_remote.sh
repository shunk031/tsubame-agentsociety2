#!/usr/bin/env bash

# @file scripts/remote/watch_remote.sh
# @brief Follow a job's logs. Runs on the login node.
# @description
#   Fed to the login node over stdin by scripts/watch.sh, so it always reflects
#   the local checkout and needs no quoting gymnastics in the caller.
#
#   Two log streams matter and they appear at different times: Grid Engine's
#   job output in the submit directory, and the per-run logs under RUNS_DIR.
#   Only files that already exist are passed to tail, which otherwise complains
#   about every missing path on each poll.
#
# @arg $1 path Repository directory on the login node.
# @arg $2 path Run output directory.
# @arg $3 int Job id. Empty to pick the newest running job.
#
# @example
#   ssh "$host" bash -s -- "$repo" "$runs_dir" "$job_id" \
#     < scripts/remote/watch_remote.sh

set -uo pipefail

remote_repo="$1"
runs_dir="$2"
job_id="${3:-}"

# With no job id, keep following whatever is newest. Resubmitting is the normal
# rhythm of getting a job right, and having to restart the watcher each time is
# friction with no purpose. A job id given explicitly is honoured as given.
follow_newest=0
[[ -z "${job_id}" ]] && follow_newest=1

# @description Print the newest queued or running job id, if any.
# @description
#   qstat is the source of truth: scanning for output files misses a job that
#   has not started, and finds stale files from jobs that ended long ago.
# @stdout Job id, or nothing.
newest_job_id() {
    qstat 2>/dev/null | awk 'NR > 2 && $1 ~ /^[0-9]+$/ { print $1 }' | tail -1
}

if [[ "${follow_newest}" -eq 1 ]]; then
    job_id="$(newest_job_id)"
fi

if [[ -z "${job_id}" ]]; then
    echo "no job id given and nothing of yours is queued or running" >&2
    echo "pass one explicitly: scripts/watch.sh <job id>" >&2
    exit 1
fi

# @description Fill the global `logs` array with the job's existing log files.
# @description
#   A job id appears in at most one output file and one run directory, so the
#   globs are used directly rather than sorted; an unmatched glob stays literal
#   and fails the -f test.
collect_logs() {
    local candidate

    logs=()

    for candidate in "${remote_repo}"/*."o${job_id}"; do
        [[ -f "${candidate}" ]] && logs+=("${candidate}")
    done

    for candidate in "${runs_dir}"/*-"${job_id}"/*.log; do
        [[ -f "${candidate}" ]] && logs+=("${candidate}")
    done
}

# @description Wait for the current job's logs to appear, then follow them.
# @description
#   tail runs in the background so this can keep an eye on qstat. In
#   follow-newest mode a resubmission replaces the job being watched, which is
#   the common case while a job script is still being got right.
# @exitcode 0 The job ended and no newer one replaced it.
follow() {
    local waited=0 tail_pid newer

    while [[ "${#logs[@]}" -eq 0 ]] && [[ "${waited}" -lt 300 ]]; do
        [[ "${waited}" -eq 0 ]] && echo "job ${job_id} is queued; waiting for output" >&2
        sleep 5
        waited=$((waited + 5))
        collect_logs
    done

    if [[ "${#logs[@]}" -eq 0 ]]; then
        echo "no output from job ${job_id} after five minutes; try: qstat -j ${job_id}" >&2
        return 1
    fi

    echo "=== job ${job_id} ==="
    tail -n 30 -F "${logs[@]}" &
    tail_pid=$!

    while true; do
        sleep 5

        if [[ "${follow_newest}" -eq 1 ]]; then
            newer="$(newest_job_id)"
            if [[ -n "${newer}" ]] && [[ "${newer}" != "${job_id}" ]]; then
                kill "${tail_pid}" 2>/dev/null
                wait "${tail_pid}" 2>/dev/null
                echo
                echo "=== switching to newer job ${newer} ==="
                job_id="${newer}"
                logs=()
                return 2
            fi
        fi

        # A finished job stops producing output; without this the watcher would
        # sit on a dead log forever.
        if ! qstat -j "${job_id}" >/dev/null 2>&1; then
            sleep 3
            kill "${tail_pid}" 2>/dev/null
            wait "${tail_pid}" 2>/dev/null
            echo
            echo "=== job ${job_id} finished ==="
            return 0
        fi
    done
}

trap 'kill 0 2>/dev/null; exit 0' INT TERM

echo "watching (Ctrl-C to stop; jobs keep running)"

while true; do
    collect_logs
    follow
    case $? in
        2) continue ;;   # switched to a newer job
        0) break ;;
        *) exit 1 ;;
    esac
done
