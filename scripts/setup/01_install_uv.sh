#!/usr/bin/env bash

# @file scripts/setup/01_install_uv.sh
# @brief Ensure uv is available on the TSUBAME login node.
# @description
#   Run on the login node, which has direct outbound network access and no
#   proxy. Idempotent: exits early when a working uv is already on PATH.
#
#   uv is installed from PyPI rather than through the upstream
#   `curl ... | sh` installer, so no downloaded script is piped into a shell.
#   TSUBAME's system Python is 3.9, which is old for the projects here but
#   perfectly able to install uv, since uv ships as a self-contained binary.
#
#   @example
#     scripts/ssh.sh 'bash "$REMOTE_REPO"/scripts/setup/01_install_uv.sh'

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=../lib/common.sh
source "${SCRIPT_DIR}/../lib/common.sh"

export PATH="${HOME}/.local/bin:${PATH}"

if command -v uv >/dev/null 2>&1; then
    log "uv already installed: $(uv --version)"
    exit 0
fi

log "installing uv from PyPI into ${HOME}/.local"
python3 -m pip install --user --quiet --upgrade uv

command -v uv >/dev/null 2>&1 || die "uv not on PATH after install; add ${HOME}/.local/bin to PATH"

log "installed uv: $(uv --version)"
