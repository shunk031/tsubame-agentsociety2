#!/usr/bin/env bash

# @file scripts/sync.sh
# @brief Copy this repository to the login node as a fresh, private snapshot.
# @description
#   Every invocation creates a directory of its own under `$WORK_ROOT/src` and
#   fills it with the current working tree. Nothing ever writes into a snapshot
#   again, so the tree a queued job will run is settled the moment it is
#   submitted -- a later sync, from any branch and any terminal, lands somewhere
#   else entirely.
#
#   The previous snapshot is used as `--link-dest`, so unchanged files are
#   hardlinked rather than re-sent. A submission costs roughly what it changed.
#
#   `$WORK_ROOT/src/current` is repointed at the new snapshot at the end. That
#   symlink is for people -- the setup scripts and one-off ssh commands follow
#   it. Jobs are handed a resolved path instead.
#
#   config/env.local is intentionally not excluded. It is git-ignored, but the
#   scripts on the login node need it.
#
# @stdout The absolute path of the snapshot on the login node.
#
# @example
#   scripts/sync.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib/common.sh
source "${SCRIPT_DIR}/lib/common.sh"

: "${TSUBAME_USER:?not set - see config/env.example}"
: "${TSUBAME_LOGIN_HOST:?not set - see config/env.example}"

remote="${TSUBAME_USER}@${TSUBAME_LOGIN_HOST}"

assert_not_behind_upstream "${REPO_ROOT}"

# Name the commit on every sync, not only when the guard fires. The tree is
# what gets copied, so the commit is the only handle tying a run back to the
# code that produced it.
sync_head="$(git -C "${REPO_ROOT}" rev-parse --short HEAD 2>/dev/null || echo '<not a git repo>')"
sync_branch="$(git -C "${REPO_ROOT}" rev-parse --abbrev-ref HEAD 2>/dev/null || echo '-')"
sync_dirty="$(git -C "${REPO_ROOT}" status --porcelain 2>/dev/null | grep -cv '^?? ' || true)"

snapshot_id="$(new_snapshot_id)"
snapshot_dir="${SRC_ROOT}/${snapshot_id}"

log "syncing ${REPO_ROOT} (${sync_branch} ${sync_head}, ${sync_dirty} uncommitted) to ${TSUBAME_LOGIN_HOST}:${snapshot_dir}"

# `mkdir` without -p on the snapshot itself, deliberately. -p succeeds on a
# directory that already exists, and a name collision is precisely the case
# where two submissions would share a tree again. Let it fail instead.
#
# RUNS_DIR is created here too because qsub needs it to exist before the job
# starts: that is where Grid Engine's own output file is opened.
# shellcheck disable=SC2029
ssh "${remote}" "mkdir -p $(printf '%q' "${SRC_ROOT}") $(printf '%q' "${RUNS_DIR}") && mkdir $(printf '%q' "${snapshot_dir}")"

build_snapshot_rsync_args "${snapshot_dir}"
rsync "${RSYNC_ARGS[@]}" "${REPO_ROOT}/" "${remote}:${snapshot_dir}/"

# Written after the transfer, so rsync neither carries an older snapshot's
# marker in nor overwrites this one. It is what jobs/*.sh echo into their logs
# and what scripts/snapshots.sh shows for each snapshot.
#
# The symlink swap goes through a temporary name and `mv -T` because `ln -sfn`
# unlinks before it symlinks, leaving a window where `current` does not exist.
# Two syncs racing here is fine: one of them wins and both snapshots are intact.
# shellcheck disable=SC2029
printf 'id      %s\nbranch  %s\ncommit  %s\ndirty   %s\nsynced  %s\n' \
    "${snapshot_id}" "${sync_branch}" "${sync_head}" "${sync_dirty}" "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" |
    ssh "${remote}" "
        cat > $(printf '%q' "${snapshot_dir}/.snapshot") &&
        ln -sn $(printf '%q' "${snapshot_dir}") $(printf '%q' "${SRC_ROOT}/.current.${snapshot_id}") &&
        mv -Tf $(printf '%q' "${SRC_ROOT}/.current.${snapshot_id}") $(printf '%q' "${REMOTE_REPO}")
    "

log "synced to snapshot ${snapshot_id}"
printf '%s\n' "${snapshot_dir}"
