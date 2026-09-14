#!/usr/bin/env bash

# @file scripts/setup/03_download_models.sh
# @brief Pre-fetch model weights into the shared HuggingFace cache.
# @description
#   Run on the login node, which has direct outbound network access. Downloading
#   inside a job would spend GPU time on network transfer, and a node_f job
#   holds four H100s while it waits.
#
#   Downloads MODEL unless overridden. Already-cached files are skipped, so
#   re-running is cheap.
#
#   @example
#     scripts/ssh.sh 'bash "$REMOTE_REPO"/scripts/setup/03_download_models.sh'
#
#   @example
#     scripts/ssh.sh 'MODEL=Qwen/Qwen3.6-35B-A3B-FP8 \
#       bash "$REMOTE_REPO"/scripts/setup/03_download_models.sh'

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=../lib/common.sh
source "${SCRIPT_DIR}/../lib/common.sh"

[[ -x "${VENV}/bin/python" ]] || die "environment missing; run 02_sync_env.sh first"

mkdir -p "${HF_HOME}"

# The embedding model is fetched alongside the generation model because a run
# needs both: without it every ask_env regenerates its code. ENABLE_EMBEDDING=0
# skips it for a job that deliberately goes without.
targets=("${MODEL}")
[[ "${ENABLE_EMBEDDING}" == "0" ]] || targets+=("${EMBEDDING_MODEL}")

for target in "${targets[@]}"; do
log "downloading ${target} into ${HF_HOME}"

MODEL="${target}" "${VENV}/bin/python" - <<'PY'
import os

from huggingface_hub import snapshot_download

model = os.environ["MODEL"]

path = snapshot_download(
    repo_id=model,
    # The tokenizer and config are needed alongside the weights; everything
    # else in a repo (ONNX exports, GGUF conversions, original/ checkpoints)
    # can be many extra GB that vLLM never opens.
    allow_patterns=[
        "*.safetensors",
        "*.safetensors.index.json",
        "*.json",
        "*.txt",
        "*.model",
        "*.jinja",
    ],
    ignore_patterns=["original/*", "*.gguf", "*.onnx", "*.bin"],
    max_workers=4,
)

total = sum(
    os.path.getsize(os.path.join(root, name))
    for root, _, names in os.walk(path)
    for name in names
)
print(f"cached at {path}")
print(f"size      {total / 1e9:.1f} GB")
PY
done

log "download complete"
