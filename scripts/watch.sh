#!/usr/bin/env bash

# @file scripts/watch.sh
# @brief Follow a job's logs from your own machine.
# @description
#   Pipes scripts/remote/watch_remote.sh to the login node, so the local
#   checkout is what runs and no sync is needed first.
#
#   Stop with Ctrl-C; the job keeps running.
#
# @arg $1 int Job id. Defaults to the newest job of yours that qstat reports.
#
# @example
#   scripts/watch.sh
#
# @example
#   scripts/watch.sh 8634774

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib/common.sh
source "${SCRIPT_DIR}/lib/common.sh"

: "${TSUBAME_USER:?not set - see config/env.example}"
: "${TSUBAME_LOGIN_HOST:?not set - see config/env.example}"

# One directory is enough now. Grid Engine's own output file used to land in
# the submit directory and the run's logs under RUNS_DIR; scripts/submit.sh
# redirects the former, so both are here.
ssh -t "${TSUBAME_USER}@${TSUBAME_LOGIN_HOST}" \
    bash -s -- "${RUNS_DIR}" "${1:-}" \
    < "${SCRIPT_DIR}/remote/watch_remote.sh"
