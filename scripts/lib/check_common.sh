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

# --- port isolation ---------------------------------------------------------

# @description Assert the port derived for a given job id.
# @arg $1 string JOB_ID to export, or empty for an interactive run.
# @arg $2 int Expected port.
assert_port_for_job() {
    local job_id="$1" expected="$2" rendered label

    label="${job_id:-<interactive>}"
    rendered="$(
        JOB_ID="${job_id}" VLLM_PORT="" \
            bash -c "source '${SCRIPT_DIR}/common.sh'; printf '%s' \"\${VLLM_PORT}\""
    )"

    if [[ "${rendered}" == "${expected}" ]]; then
        printf 'ok   %-34s port %s\n' "JOB_ID=${label}" "${rendered}"
    else
        printf 'FAIL %-34s port %s, expected %s\n' "JOB_ID=${label}" "${rendered}" "${expected}"
        failures=$((failures + 1))
    fi
}

# Two jobs on one node must not derive the same port; the ids that collided in
# practice are a good pair to pin.
assert_port_for_job 8660581 8581
assert_port_for_job 8660582 8582
assert_port_for_job "" 8000

# assert_port_free has to fail loudly on a busy port. It once failed silently:
# a bare `exec` with a redirection applies it to the shell, and the cleanup
# line sent every later message to /dev/null. Assert the message, not just the
# exit status.
(
    python3 - <<'LISTENER' &
import socket, time
s = socket.socket()
s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
s.bind(("127.0.0.1", 8199))
s.listen(1)
time.sleep(10)
LISTENER
    listener=$!
    sleep 1

    output="$(bash -c "source '${SCRIPT_DIR}/common.sh'; assert_port_free 8199" 2>&1)"
    status=$?
    kill "${listener}" 2>/dev/null
    wait "${listener}" 2>/dev/null

    if [[ "${status}" -ne 0 ]] && [[ "${output}" == *"already serving"* ]]; then
        printf 'ok   %-34s busy port refused, with a message\n' "assert_port_free"
        exit 0
    fi
    printf 'FAIL %-34s status=%s output=%s\n' "assert_port_free" "${status}" "${output}"
    exit 1
) || failures=$((failures + 1))

(
    if bash -c "source '${SCRIPT_DIR}/common.sh'; assert_port_free 8198" 2>/dev/null; then
        printf 'ok   %-34s free port accepted\n' "assert_port_free"
        exit 0
    fi
    printf 'FAIL %-34s rejected a free port\n' "assert_port_free"
    exit 1
) || failures=$((failures + 1))

# --- staleness guard --------------------------------------------------------

# assert_not_behind_upstream decides whether the cluster gets the code the git
# history claims. Exercise it against real repositories: a fixture built from
# strings would only test the string handling, and what failed in practice was
# the comparison against a remote-tracking ref.
(
    fixture="$(mktemp -d)"
    trap 'rm -rf "${fixture}"' EXIT
    cd "${fixture}" || exit 1

    git init -q --bare origin.git
    # Without this the bare HEAD names a branch that does not exist, the second
    # clone checks out nothing, and the fixture silently tests the wrong thing.
    git -C origin.git symbolic-ref HEAD refs/heads/main

    git clone -q origin.git work 2>/dev/null
    git -C work config user.email t@example.invalid
    git -C work config user.name t
    git -C work checkout -qb main
    echo one >"work/f"
    git -C work add f
    git -C work commit -qm one
    git -C work push -qu origin main

    local_failures=0

    # Up to date: the guard has nothing to say.
    if bash -c "source '${SCRIPT_DIR}/common.sh'; assert_not_behind_upstream '${fixture}/work'" 2>/dev/null; then
        printf 'ok   %-34s up to date passes\n' "assert_not_behind_upstream"
    else
        printf 'FAIL %-34s blocked an up-to-date tree\n' "assert_not_behind_upstream"
        local_failures=$((local_failures + 1))
    fi

    # Move the upstream ahead of the checkout.
    git clone -q origin.git pusher 2>/dev/null
    git -C pusher config user.email t@example.invalid
    git -C pusher config user.name t
    echo two >>"pusher/f"
    git -C pusher add f
    git -C pusher commit -qm two
    git -C pusher push -q origin main
    git -C work fetch -q origin

    output="$(bash -c "source '${SCRIPT_DIR}/common.sh'; assert_not_behind_upstream '${fixture}/work'; echo REACHED" 2>&1)"
    if [[ "${output}" != *REACHED* ]] && [[ "${output}" == *"commit(s) behind"* ]]; then
        printf 'ok   %-34s behind upstream refused, with a message\n' "assert_not_behind_upstream"
    else
        printf 'FAIL %-34s behind upstream not refused: %s\n' "assert_not_behind_upstream" "${output}"
        local_failures=$((local_failures + 1))
    fi

    # The deliberate escape hatch warns and continues.
    output="$(bash -c "source '${SCRIPT_DIR}/common.sh'; SYNC_ALLOW_STALE=1 assert_not_behind_upstream '${fixture}/work'; echo REACHED" 2>&1)"
    if [[ "${output}" == *REACHED* ]] && [[ "${output}" == *WARNING* ]]; then
        printf 'ok   %-34s SYNC_ALLOW_STALE warns and continues\n' "assert_not_behind_upstream"
    else
        printf 'FAIL %-34s SYNC_ALLOW_STALE did not continue: %s\n' "assert_not_behind_upstream" "${output}"
        local_failures=$((local_failures + 1))
    fi

    # A branch with no upstream is an ordinary thing to sync. Firing here would
    # train the reader to ignore the guard.
    git -C work checkout -qb feature
    if bash -c "source '${SCRIPT_DIR}/common.sh'; assert_not_behind_upstream '${fixture}/work'" 2>/dev/null; then
        printf 'ok   %-34s no upstream stays quiet\n' "assert_not_behind_upstream"
    else
        printf 'FAIL %-34s fired on a branch with no upstream\n' "assert_not_behind_upstream"
        local_failures=$((local_failures + 1))
    fi

    exit "${local_failures}"
) || failures=$((failures + $?))

if [[ "${failures}" -gt 0 ]]; then
    printf '\n%d check(s) failed\n' "${failures}"
    exit 1
fi

printf '\nall checks passed\n'
