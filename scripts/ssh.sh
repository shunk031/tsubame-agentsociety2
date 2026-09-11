#!/usr/bin/env bash

# @file scripts/ssh.sh
# @brief Run a command on the login node without naming it.
# @description
#   Account and host come from config/env.local, so neither appears in the
#   command you type, in shell history, or in anything pasted into a chat or an
#   issue. With no arguments it opens an interactive session.
#
#   REMOTE_REPO is exported to the remote command, so callers can refer to the
#   repository without spelling out a path.
#
# @arg $@ string Command to run. Omit for an interactive shell.
#
# @example
#   scripts/ssh.sh qstat
#
# @example
#   scripts/ssh.sh 'tail -20 "$REMOTE_REPO"/as2-sim.o*'
#
# @example
#   scripts/ssh.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib/common.sh
source "${SCRIPT_DIR}/lib/common.sh"

: "${TSUBAME_USER:?not set - see config/env.example}"
: "${TSUBAME_LOGIN_HOST:?not set - see config/env.example}"

REMOTE_REPO="${REMOTE_REPO:-tsubame-agentsociety2}"

if [[ $# -eq 0 ]]; then
    exec ssh "${TSUBAME_USER}@${TSUBAME_LOGIN_HOST}"
fi

# REMOTE_REPO, WORK_ROOT and RUNS_DIR are defined for the remote command so it
# can reference them instead of hardcoding paths.
# shellcheck disable=SC2029
exec ssh "${TSUBAME_USER}@${TSUBAME_LOGIN_HOST}" \
    "REMOTE_REPO=\"\$HOME/${REMOTE_REPO}\" WORK_ROOT=$(printf '%q' "${WORK_ROOT}") RUNS_DIR=$(printf '%q' "${RUNS_DIR}") bash -lc $(printf '%q' "$*")"
