#!/usr/bin/env bash

# @file scripts/redact.sh
# @brief Strip site identifiers from stdin before a log leaves the machine.
# @description
#   Logs are worth sharing in an issue, a chat or a report, but they carry the
#   account name, group, host and work path in every other line. This replaces
#   them with the names of the settings they came from, so the shape of the log
#   survives and the identifiers do not.
#
#   Values come from config/env.local, so whatever is configured is what gets
#   removed.
#
# @example
#   scripts/watch.sh 8634860 | scripts/redact.sh
#
# @example
#   scripts/ssh.sh 'tail -50 "$RUNS_DIR"/as2-sim.o8634860' | scripts/redact.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib/common.sh
source "${SCRIPT_DIR}/lib/common.sh"

# Longest first, so a path is replaced whole rather than leaving a tail behind
# once the group inside it has already been substituted.
sed \
    -e "s|${WORK_ROOT}|\$WORK_ROOT|g" \
    -e "s|${HOME}|\$HOME|g" \
    -e "s|${TSUBAME_LOGIN_HOST:-__unset__}|<login-host>|g" \
    -e "s|${TSUBAME_GROUP}|<group>|g" \
    -e "s|${TSUBAME_USER:-__unset__}|<user>|g"
