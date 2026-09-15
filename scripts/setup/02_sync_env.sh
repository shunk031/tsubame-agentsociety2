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
#   The build ends by stamping the environment with what it was built from, so
#   that a job can refuse to run against it later. See the provenance section of
#   scripts/lib/common.sh for what goes into that stamp and why.
#
#   @example
#     scripts/ssh.sh 'bash "$REMOTE_REPO"/scripts/setup/02_sync_env.sh'

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=../lib/common.sh
source "${SCRIPT_DIR}/../lib/common.sh"

export PATH="${HOME}/.local/bin:${PATH}"
export UV_PROJECT_ENVIRONMENT="${VENV}"

command -v uv >/dev/null 2>&1 || die "uv not found; run 01_install_uv.sh first"

mkdir -p "${WORK_ROOT}" "${HF_HOME}" "${RUNS_DIR}"

# Drop the previous stamp before touching anything. A build that dies partway --
# killed on the login node, out of disk, connection lost -- leaves an
# environment that still imports and still reports a version, and an inherited
# stamp would let the next job accept it. Removing it first means an
# unfinished build leaves no claim at all, which is what jobs check for.
rm -f "${VENV}/${ENV_PROVENANCE_NAME}"

# Digest the source before installing, and again afterwards. A build once ran
# while the vendored source was being replaced underneath it, and the
# environment that came out held a mix that nothing reported; comparing the two
# digests is what makes that case loud rather than something to find later by
# grepping site-packages.
source_before="$(source_fingerprint "${REPO_ROOT}")" ||
    die "could not digest the source under ${REPO_ROOT}"

log "syncing ${REPO_ROOT} (source ${source_before:0:12}) into ${VENV}"
# --reinstall-package is not belt and braces. agentsociety2 comes from
# vendor/AgentSociety as a path dependency, and uv.lock records only
# `source = { directory = ... }` -- no digest of the directory's contents. So
# `uv sync --locked` treats the package as satisfied however the source
# changed, reuses the wheel it built the first time, and reports success in
# seconds. A patched module then never reaches site-packages, the run looks
# ordinary, and the measurement is of code nobody meant to test.
uv sync --locked --project "${REPO_ROOT}" --reinstall-package agentsociety2

source_after="$(source_fingerprint "${REPO_ROOT}")" ||
    die "could not digest the source under ${REPO_ROOT} after the install"
[[ "${source_before}" == "${source_after}" ]] ||
    die "the source changed while the environment was being built (${source_before:0:12} -> ${source_after:0:12}); ${VENV} may hold a mix of both. Re-run this script once the tree has settled."

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

# Last, and only now: the stamp says the environment is usable, so it is written
# after the import check rather than after the install.
write_env_provenance "${VENV}" "${REPO_ROOT}" ||
    die "the environment was built but could not be stamped; jobs would refuse it"

log "environment ready (the GPU check happens in jobs/smoke.sh)"
