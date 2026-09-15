#!/usr/bin/env bash

# @file scripts/snapshots.sh
# @brief List the source snapshots on the login node, and prune old ones.
# @description
#   Every submission leaves a copy of the source under `$WORK_ROOT/src`, which
#   is what stops a queued job's code from changing underneath it. They are
#   hardlinked against each other so they cost little, but they do accumulate.
#
#   With no arguments this only reports. `--prune` is what deletes, and it is a
#   separate, deliberate command: a snapshot is the only record of what a run
#   actually executed, so removing one is a decision rather than housekeeping.
#
#   Never removed, whatever the flags say: a snapshot a queued or running job is
#   pinned to, the one `current` points at, and the newest --keep.
#
# @option --prune Delete the snapshots reported as removable.
# @option --keep <n> Keep this many of the newest regardless. Defaults to 5.
#
# @example
#   scripts/snapshots.sh
#
# @example
#   scripts/snapshots.sh --prune --keep 10

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib/common.sh
source "${SCRIPT_DIR}/lib/common.sh"

: "${TSUBAME_USER:?not set - see config/env.example}"
: "${TSUBAME_LOGIN_HOST:?not set - see config/env.example}"

prune=0
keep=5

while [[ $# -gt 0 ]]; do
    case "$1" in
        --prune)
            prune=1
            shift
            ;;
        --keep)
            keep="${2:?--keep needs a number}"
            shift 2
            ;;
        *)
            die "usage: scripts/snapshots.sh [--prune] [--keep <n>]"
            ;;
    esac
done

[[ "${keep}" =~ ^[0-9]+$ ]] || die "--keep takes a number, not '${keep}'"

ssh "${TSUBAME_USER}@${TSUBAME_LOGIN_HOST}" \
    bash -s -- "${SRC_ROOT}" "${keep}" "${prune}" \
    < "${SCRIPT_DIR}/remote/snapshots_remote.sh"
