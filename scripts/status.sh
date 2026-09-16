#!/usr/bin/env bash

# @file scripts/status.sh
# @brief Show every run as a rate over a known span.
# @description
#   Pipes scripts/remote/status_remote.sh to the login node, so the local
#   checkout is what runs and no sync is needed first.
#
#   Use this instead of writing an ssh pipeline by hand. Every such pipeline
#   written during one long session left something out, and the omissions were
#   what misled: a job called "still starting" an hour after submission, two
#   hours of polling one that had already exited, and counts quoted without the
#   span they were taken over.
#
# @example
#   scripts/status.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib/common.sh
source "${SCRIPT_DIR}/lib/common.sh"

: "${TSUBAME_USER:?not set - see config/env.example}"
: "${TSUBAME_LOGIN_HOST:?not set - see config/env.example}"

ssh "${TSUBAME_USER}@${TSUBAME_LOGIN_HOST}" \
    bash -s -- "${RUNS_DIR}" \
    < "${SCRIPT_DIR}/remote/status_remote.sh"
