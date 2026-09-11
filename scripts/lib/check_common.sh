#!/usr/bin/env bash

# @file scripts/lib/check_common.sh
# @brief Self-check for the helpers in common.sh, runnable without TSUBAME.
# @description
#   Exercises build_vllm_args across the model families this repository targets
#   and asserts the flags that are easy to get silently wrong: thinking
#   suppression on Qwen3.5+, and expert parallelism only for multi-GPU MoE.
#
#   Run it from the repository root: `bash scripts/lib/check_common.sh`

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

failures=0

# @description Assert that build_vllm_args emits (or omits) a flag for a model.
# @arg $1 string Model identifier passed as MODEL.
# @arg $2 int Data parallel size.
# @arg $3 string Flag expected in the generated arguments.
# @arg $4 string "present" or "absent".
assert_flag() {
    local model="$1" dp="$2" flag="$3" expectation="$4"
    local rendered

    rendered="$(
        MODEL="${model}" VLLM_EXTRA_ARGS="" \
            bash -c "source '${SCRIPT_DIR}/common.sh'; build_vllm_args ${dp}; printf '%s\n' \"\${VLLM_ARGS[*]}\""
    )"

    if [[ "${rendered}" == *"${flag}"* ]]; then
        if [[ "${expectation}" == "present" ]]; then
            printf 'ok   %-34s dp=%s has %s\n' "${model}" "${dp}" "${flag}"
        else
            printf 'FAIL %-34s dp=%s unexpectedly has %s\n' "${model}" "${dp}" "${flag}"
            failures=$((failures + 1))
        fi
    else
        if [[ "${expectation}" == "absent" ]]; then
            printf 'ok   %-34s dp=%s lacks %s\n' "${model}" "${dp}" "${flag}"
        else
            printf 'FAIL %-34s dp=%s is missing %s\n' "${model}" "${dp}" "${flag}"
            failures=$((failures + 1))
        fi
    fi
}

# Qwen3.5+ reasons by default, and the flag has to say so explicitly rather
# than rely on the model's own default.
assert_flag "Qwen/Qwen3.5-4B" 1 '"enable_thinking": true' present
assert_flag "Qwen/Qwen3.6-35B-A3B-FP8" 4 '"enable_thinking": true' present

# Qwen3 (no minor version) predates the thinking default, so no kwarg is sent.
assert_flag "Qwen/Qwen3-30B-A3B-Instruct-2507" 1 "enable_thinking" absent

# Expert parallelism needs both an MoE checkpoint and more than one GPU.
assert_flag "Qwen/Qwen3.6-35B-A3B-FP8" 4 "--enable-expert-parallel" present
assert_flag "Qwen/Qwen3.6-35B-A3B-FP8" 1 "--enable-expert-parallel" absent
assert_flag "Qwen/Qwen3.5-4B" 4 "--enable-expert-parallel" absent

# Data parallel size must reach the server.
assert_flag "Qwen/Qwen3.5-4B" 4 "--data-parallel-size 4" present

# AgentSociety sends tool_choice="auto" on every agent step; vLLM rejects those
# outright unless both of these are set.
assert_flag "Qwen/Qwen3.5-4B" 1 "--enable-auto-tool-choice" present
assert_flag "Qwen/Qwen3.5-4B" 1 "--tool-call-parser qwen3_coder" present
assert_flag "Qwen/Qwen3.6-35B-A3B-FP8" 4 "--enable-auto-tool-choice" present

# ENABLE_THINKING=0 is the throughput escape hatch.
(
    export ENABLE_THINKING=0
    assert_flag "Qwen/Qwen3.5-4B" 1 '"enable_thinking": false' present
) || failures=$((failures + 1))

if [[ "${failures}" -gt 0 ]]; then
    printf '\n%d check(s) failed\n' "${failures}"
    exit 1
fi

printf '\nall checks passed\n'
