#!/usr/bin/env bash

# @file scripts/remote/status_remote.sh
# @brief Report every job as a rate over a known span, not as a bare count.
# @description
#   Runs on the login node, piped in by scripts/status.sh.
#
#   Three columns exist because each one was learned the hard way. Liveness
#   comes from qstat, because a job that exited leaves its log behind and a
#   monitor reading only the log will follow a dead job for hours. Elapsed time
#   turns a count into a rate, without which two runs cannot be compared --
#   numbers taken over different spans were once read as a difference between
#   configurations. Log age separates a working job from a hung one during the
#   long stretches when the scheduler still calls it running.
#
# @arg $1 path Directory holding the runs and the scheduler's output files.

set -uo pipefail

RUNS_DIR="${1:?runs directory required}"
NOW="$(date +%s)"

printf '%-9s %-6s %-9s %-8s %-9s %-7s %s\n' \
    JOB STATE ELAPSED LOGAGE EVENTS RATE/h POWER

# qstat is the authority on what is alive; the run directories are the
# authority on what exists. A job appears here if either knows about it.
alive="$(qstat 2>/dev/null | awk 'NR>2 {print $1"="$5}')"

for dir in "${RUNS_DIR}"/sim-* "${RUNS_DIR}"/bench-* "${RUNS_DIR}"/curl-*; do
    [[ -d "${dir}" ]] || continue
    job="${dir##*-}"
    [[ "${job}" =~ ^[0-9]+$ ]] || continue

    state="$(printf '%s\n' "${alive}" | sed -n "s/^${job}=//p")"
    [[ -n "${state}" ]] || state="done"

    log="${RUNS_DIR}/as2-sim.o${job}"
    [[ -f "${log}" ]] || log="${RUNS_DIR}/as2-bench.o${job}"
    [[ -f "${log}" ]] || log=""

    started="" logage="-"
    if [[ -n "${log}" ]]; then
        # The job script's first timestamped line is when it began; birth time
        # is not portable enough to rely on here.
        started="$(head -5 "${log}" 2>/dev/null \
            | grep -oE '[0-9]{4}-[0-9]{2}-[0-9]{2} [0-9]{2}:[0-9]{2}:[0-9]{2}' | head -1)"
        logage="$(( (NOW - $(stat -c %Y "${log}")) / 60 ))m"
    fi

    elapsed="-"
    if [[ -n "${started}" ]]; then
        elapsed="$(( (NOW - $(date -d "${started}" +%s)) / 60 ))m"
    fi

    # From the replay rather than from a phrase in the log. The first version
    # counted lines matching "created post", which is English, so a run told to
    # write in Japanese reported zero activity for its whole life and was read
    # as broken. What a run produced must not depend on the language it wrote in.
    posts="-"
    if compgen -G "${dir}/replay/social_media_event.*.jsonl" >/dev/null 2>&1; then
        posts="$(cat "${dir}"/replay/social_media_event.*.jsonl 2>/dev/null | wc -l)"
    fi

    # The number that can actually be compared between two runs.
    rate="-"
    if [[ "${posts}" != "-" ]] && [[ "${elapsed}" != "-" ]]; then
        mins="${elapsed%m}"
        (( mins > 0 )) && rate="$(( posts * 60 / mins ))"
    fi

    power="-"
    if [[ -f "${dir}/gpu.csv" ]]; then
        power="$(awk -F, 'NR>1 && $5+0>1000 {n++; p+=$6}
            END{if(n) printf "%.0fW/%.0f%%", p/n, p/n/700*100; else print "-"}' \
            "${dir}/gpu.csv")"
    fi

    printf '%-9s %-6s %-9s %-8s %-9s %-7s %s\n' \
        "${job}" "${state}" "${elapsed}" "${logage}" "${posts}" "${rate}" "${power}"
done
