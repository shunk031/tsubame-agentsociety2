#!/usr/bin/env bash

# @file scripts/remote/snapshots_remote.sh
# @brief List, and optionally delete, source snapshots. Runs on the login node.
# @description
#   Fed to the login node over stdin by scripts/snapshots.sh, so it always
#   reflects the local checkout and needs no quoting gymnastics in the caller.
#
#   Three kinds of snapshot are never deleted:
#
#     - the one a queued or running job is pinned to. Grid Engine is asked
#       directly, through each job's sge_o_workdir, rather than any bookkeeping
#       this repository keeps: a job submitted from another checkout, by another
#       person, or before this script existed still counts.
#     - whatever `current` points at, since the setup scripts run there.
#     - the newest few, so a finished run can still be compared against the code
#       that produced it.
#
#   Deleting is never the default. Snapshots are the only record of what a run
#   actually executed, and that call belongs to a person.
#
# @arg $1 path Snapshot root.
# @arg $2 int How many of the newest snapshots to keep regardless of age.
# @arg $3 int 1 to delete prunable snapshots, 0 to only report them.
#
# @example
#   ssh "$host" bash -s -- "$src_root" 5 0 < scripts/remote/snapshots_remote.sh

set -uo pipefail

src_root="$1"
keep="$2"
prune="$3"

if [[ ! -d "${src_root}" ]]; then
    echo "no snapshots yet: ${src_root} does not exist" >&2
    exit 0
fi

# Without qstat there is no way to tell a snapshot a job is queued against from
# one nothing needs, and guessing wrong deletes the source out from under a run
# that has not started yet. Listing is still useful, so only pruning stops.
if [[ "${prune}" -eq 1 ]] && ! command -v qstat >/dev/null 2>&1; then
    echo "qstat not found; refusing to prune without knowing which snapshots are in use" >&2
    exit 1
fi

current=""
[[ -L "${src_root}/current" ]] && current="$(readlink -f "${src_root}/current")"

# @description Fill the global `pinned` map with snapshot path -> job ids.
# @description
#   qstat with no arguments lists this account's queued and running jobs; a job
#   that has finished drops out of it, which is exactly the point at which its
#   snapshot stops being needed.
collect_pinned() {
    local job_id workdir

    declare -gA pinned=()

    command -v qstat >/dev/null 2>&1 || return 0

    while read -r job_id; do
        workdir="$(qstat -j "${job_id}" 2>/dev/null |
            awk -F': *' '$1 == "sge_o_workdir" { print $2; exit }')"
        [[ -n "${workdir}" ]] || continue
        pinned["${workdir}"]+="${job_id} "
    done < <(qstat 2>/dev/null | awk 'NR > 2 && $1 ~ /^[0-9]+$/ { print $1 }')
}

collect_pinned

# The identifier starts with a UTC timestamp, so a plain sort is chronological
# and the tail of the list is the newest.
mapfile -t snapshots < <(
    find "${src_root}" -mindepth 1 -maxdepth 1 -type d -printf '%f\n' 2>/dev/null | sort
)

if [[ "${#snapshots[@]}" -eq 0 ]]; then
    echo "no snapshots under ${src_root}"
    exit 0
fi

# @description Describe a snapshot from the marker scripts/sync.sh left in it.
# @arg $1 path Snapshot directory.
# @stdout "<branch> <commit>", or "-" when there is no marker.
describe() {
    local dir="$1" branch commit

    [[ -f "${dir}/.snapshot" ]] || {
        printf '%s' "-"
        return 0
    }

    branch="$(awk '$1 == "branch" { print $2 }' "${dir}/.snapshot")"
    commit="$(awk '$1 == "commit" { print $2 }' "${dir}/.snapshot")"
    printf '%s %s' "${branch:--}" "${commit:--}"
}

newest_start=$((${#snapshots[@]} - keep))
[[ "${newest_start}" -lt 0 ]] && newest_start=0

kept=0
removed=0
prunable=0

for index in "${!snapshots[@]}"; do
    name="${snapshots[${index}]}"
    dir="${src_root}/${name}"
    reason=""

    if [[ -n "${pinned[${dir}]:-}" ]]; then
        reason="job ${pinned[${dir}]% }"
    elif [[ "${dir}" == "${current}" ]]; then
        reason="current"
    elif [[ "${index}" -ge "${newest_start}" ]]; then
        reason="recent"
    fi

    if [[ -n "${reason}" ]]; then
        printf 'keep   %s  %-40s  %s\n' "${name}" "$(describe "${dir}")" "${reason}"
        kept=$((kept + 1))
        continue
    fi

    prunable=$((prunable + 1))

    if [[ "${prune}" -ne 1 ]]; then
        printf 'stale  %s  %-40s  %s\n' "${name}" "$(describe "${dir}")" "removable"
        continue
    fi

    # Belt and braces around an rm -rf that is assembled from a directory
    # listing: the name has to be one path component and the target has to sit
    # directly under the root it was listed from.
    if [[ -z "${name}" ]] || [[ "${name}" == */* ]] || [[ "${dir}" != "${src_root}/"* ]]; then
        printf 'ERROR  %s  refusing to remove an unexpected path\n' "${name}" >&2
        continue
    fi

    # Read the marker before the directory it lives in goes away, or the line
    # reporting the removal cannot say what was removed.
    described="$(describe "${dir}")"
    rm -rf -- "${dir}"
    printf 'removed %s  %s\n' "${name}" "${described}"
    removed=$((removed + 1))
done

if [[ "${prune}" -eq 1 ]]; then
    printf '\n%d kept, %d removed\n' "${kept}" "${removed}"
else
    printf '\n%d kept, %d removable\n' "${kept}" "${prunable}"
    [[ "${prunable}" -gt 0 ]] && printf 'delete them with: scripts/snapshots.sh --prune\n'
fi

exit 0
