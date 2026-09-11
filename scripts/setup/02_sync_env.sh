#!/usr/bin/env bash

# @file scripts/setup/02_sync_env.sh
# @brief Materialise the Python environment from uv.lock.
# @description
#   Installs exactly what the lockfile records, so the environment on the
#   compute node matches the one resolved on the login node. `--locked` makes uv
#   fail rather than silently re-resolve when pyproject.toml and uv.lock
#   disagree.
#
#   The environment goes under WORK_ROOT on the group area rather than beside
#   pyproject.toml, because it is several GB and home is shared with everything
#   else. UV_PROJECT_ENVIRONMENT is what moves it there.
#
#   @example
#     ssh "${TSUBAME_USER}@${TSUBAME_LOGIN_HOST}" \
#       bash ~/tsubame-agentsociety2/scripts/setup/02_sync_env.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=../lib/common.sh
source "${SCRIPT_DIR}/../lib/common.sh"

export PATH="${HOME}/.local/bin:${PATH}"
export UV_PROJECT_ENVIRONMENT="${VENV}"

command -v uv >/dev/null 2>&1 || die "uv not found; run 01_install_uv.sh first"

mkdir -p "${WORK_ROOT}" "${HF_HOME}" "${RUNS_DIR}"

log "syncing ${REPO_ROOT} into ${VENV}"
uv sync --locked --project "${REPO_ROOT}"

log "verifying the installation"
"${VENV}/bin/python" - <<'PY'
import importlib.metadata as metadata
import importlib.util

import torch

for package in ("vllm", "agentsociety2", "protobuf", "transformers"):
    print(f"{package:16s} {metadata.version(package)}")
print(f"{'torch':16s} {torch.__version__}")
print(f"{'torch.cuda':16s} {torch.version.cuda}")

if importlib.util.find_spec("agentsociety") is not None:
    raise SystemExit(
        "legacy agentsociety 1.x is present; it collides with agentsociety2 on "
        "the 'agentsociety' console script"
    )

# Import the module the jobs actually run, so a broken environment fails here
# rather than several minutes into a GPU job.
import agentsociety2.society.cli  # noqa: F401

print("agentsociety2.society.cli import OK")
PY

log "environment ready (the GPU check happens in jobs/smoke.sh)"
