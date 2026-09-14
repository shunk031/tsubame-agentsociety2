#!/usr/bin/env bash

# @file scripts/sync.sh
# @brief Copy this repository to the login node.
# @description
#   Mirrors the working tree so `qsub` on the login node runs what is checked
#   out here. `--delete` keeps removed files from lingering, which is why the
#   excludes matter: Grid Engine writes job output (`*.o<id>`) into the submit
#   directory, and a bare `--delete` would erase the logs of a job that is still
#   running.
#
#   config/env.local is intentionally not excluded. It is git-ignored, but the
#   scripts on the login node need it.
#
# @example
#   scripts/sync.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib/common.sh
source "${SCRIPT_DIR}/lib/common.sh"

: "${TSUBAME_USER:?not set - see config/env.example}"
: "${TSUBAME_LOGIN_HOST:?not set - see config/env.example}"

REMOTE_REPO="${REMOTE_REPO:-tsubame-agentsociety2}"

assert_not_behind_upstream "${REPO_ROOT}"

# Name the commit on every sync, not only when the guard fires. The tree is
# what gets copied, so the commit is the only handle tying a run back to the
# code that produced it.
sync_head="$(git -C "${REPO_ROOT}" rev-parse --short HEAD 2>/dev/null || echo '<not a git repo>')"
sync_branch="$(git -C "${REPO_ROOT}" rev-parse --abbrev-ref HEAD 2>/dev/null || echo '-')"
sync_dirty="$(git -C "${REPO_ROOT}" status --porcelain 2>/dev/null | grep -cv '^?? ' || true)"

log "syncing ${REPO_ROOT} (${sync_branch} ${sync_head}, ${sync_dirty} uncommitted) to ${TSUBAME_LOGIN_HOST}:~/${REMOTE_REPO}"

rsync -az --delete \
    --exclude '.git' \
    --exclude '.claude' \
    --exclude '__pycache__' \
    --exclude '*.o[0-9]*' \
    --exclude '*.e[0-9]*' \
    --exclude '*.po[0-9]*' \
    --exclude '*.pe[0-9]*' \
    "${REPO_ROOT}/" \
    "${TSUBAME_USER}@${TSUBAME_LOGIN_HOST}:~/${REMOTE_REPO}/"

log "synced"
