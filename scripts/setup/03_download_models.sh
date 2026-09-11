#!/usr/bin/env bash

# @file scripts/setup/05_download_models.sh
# @brief Pre-fetch model weights into the shared HuggingFace cache.
# @description
#   Run on the login node, which has direct outbound network access. Downloading
#   inside a job would burn reserved GPU time on network transfer, and a
#   `node_f` job holds four H100s while it waits.
#
#   Downloads MODEL unless overridden. Already-cached files are skipped, so
#   re-running is cheap.
#
#   @example
#     ssh "${TSUBAME_USER}@${TSUBAME_LOGIN_HOST}" \
#       bash ~/tsubame-agentsociety2/scripts/setup/05_download_models.sh
#
#   @example
#     ssh "${TSUBAME_USER}@${TSUBAME_LOGIN_HOST}" \
#       MODEL=Qwen/Qwen3.6-35B-A3B-FP8 \
#       bash ~/tsubame-agentsociety2/scripts/setup/05_download_models.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=../lib/common.sh
source "${SCRIPT_DIR}/../lib/common.sh"

[[ -x "${VENV}/bin/python" ]] || die "environment missing; run 02_sync_env.sh first"

mkdir -p "${HF_HOME}"

log "downloading ${MODEL} into ${HF_HOME}"

MODEL="${MODEL}" "${VENV}/bin/python" - <<'PY'
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

log "download complete"
