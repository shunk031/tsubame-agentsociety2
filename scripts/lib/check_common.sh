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

# --- embedding server -------------------------------------------------------

# The generation and embedding servers share one GPU, so their reservations
# have to add up to less than the card. Nothing at runtime checks that, and
# exceeding it fails at model load with an out-of-memory error that reads like
# a model-size problem rather than a configuration one.
(
    total="$(
        bash -c "source '${SCRIPT_DIR}/common.sh'; awk -v a=\"\${GPU_MEMORY_UTILIZATION}\" -v b=\"\${EMBEDDING_GPU_MEMORY_UTILIZATION}\" 'BEGIN{printf \"%.2f\", a+b}'"
    )"
    if awk -v t="${total}" 'BEGIN{exit !(t < 0.95)}'; then
        printf 'ok   %-34s GPU budget %s leaves headroom\n' "embedding" "${total}"
        exit 0
    fi
    printf 'FAIL %-34s GPU budget %s is too tight\n' "embedding" "${total}"
    exit 1
) || failures=$((failures + 1))

# Both servers on one node need different ports, and the embedding port has to
# be derived per job for the same reason the generation port is.
assert_embedding_port() {
    local job_id="$1" expected="$2" rendered
    rendered="$(
        JOB_ID="${job_id}" EMBEDDING_PORT="" \
            bash -c "source '${SCRIPT_DIR}/common.sh'; printf '%s' \"\${EMBEDDING_PORT}\""
    )"
    if [[ "${rendered}" == "${expected}" ]]; then
        printf 'ok   %-34s JOB_ID=%s port %s\n' "embedding port" "${job_id:-<interactive>}" "${rendered}"
    else
        printf 'FAIL %-34s JOB_ID=%s port %s, expected %s\n' "embedding port" "${job_id:-<interactive>}" "${rendered}" "${expected}"
        failures=$((failures + 1))
    fi
}
assert_embedding_port 8663355 9355
assert_embedding_port "" 9000

# The two ports must not collide with each other for the same job.
(
    read -r gen emb <<<"$(
        JOB_ID=8663355 VLLM_PORT="" EMBEDDING_PORT="" \
            bash -c "source '${SCRIPT_DIR}/common.sh'; printf '%s %s' \"\${VLLM_PORT}\" \"\${EMBEDDING_PORT}\""
    )"
    if [[ "${gen}" != "${emb}" ]]; then
        printf 'ok   %-34s generation %s and embedding %s differ\n' "embedding port" "${gen}" "${emb}"
        exit 0
    fi
    printf 'FAIL %-34s both servers would bind %s\n' "embedding port" "${gen}"
    exit 1
) || failures=$((failures + 1))

# --- run state isolation ----------------------------------------------------

# Upstream's AGENTSOCIETY_HOME_DIR default is relative to the working
# directory, so every job on this cluster shared one codegen cache and a
# measurement depended on which runs came before it. Assert the mode variable
# and that jobs/run_sim.sh actually acts on it, since the value alone changes
# nothing.
assert_home_mode() {
    local mode="$1" expected="$2" rendered
    rendered="$(
        AGENT_HOME_MODE="${mode}" \
            bash -c "source '${SCRIPT_DIR}/common.sh'; printf '%s' \"\${AGENT_HOME_MODE}\""
    )"
    if [[ "${rendered}" == "${expected}" ]]; then
        printf 'ok   %-34s %s\n' "agent home mode" "${rendered}"
    else
        printf 'FAIL %-34s %s, expected %s\n' "agent home mode" "${rendered}" "${expected}"
        failures=$((failures + 1))
    fi
}
assert_home_mode "" per-run
assert_home_mode shared shared

(
    job="${SCRIPT_DIR}/../../jobs/run_sim.sh"
    missing=0
    # The run directory must be the default home, and the export has to happen
    # before the simulation starts rather than only being computed.
    # Literal, not an expansion: this is the text the job script must contain.
    # shellcheck disable=SC2016
    grep -qF 'AGENTSOCIETY_HOME_DIR="${AGENTSOCIETY_HOME_DIR:-${RUN_DIR}/agent-home}"' "${job}" || missing=1
    grep -q 'export AGENTSOCIETY_HOME_DIR' "${job}" || missing=1
    grep -q 'AGENT_HOME_MODE' "${job}" || missing=1
    if [[ "${missing}" -eq 0 ]]; then
        printf 'ok   %-34s run_sim.sh exports a per-run home\n' "agent home mode"
        exit 0
    fi
    printf 'FAIL %-34s run_sim.sh does not export a per-run home\n' "agent home mode"
    exit 1
) || failures=$((failures + 1))

# Upstream submits ceil(N / BATCH_SIZE) step_agent_batch Ray Tasks per tick and
# runs at most LLM_RAY_MAX_WORKERS of them at once, so its default batch of 256
# gives every population this repository has run a single task -- one process
# holding one AIMD semaphore, which is why neither more agents nor more workers
# ever raised the concurrent request count. The rule to restore is upstream's
# own: pick a batch where ceil(N / BATCH_SIZE) >= worker count.
assert_batch_size() {
    local agents="$1" workers="$2" expected="$3" got tasks
    got="$(
        bash -c "source '${SCRIPT_DIR}/common.sh'; ray_batch_size '${agents}' '${workers}'"
    )"
    if [[ "${got}" != "${expected}" ]]; then
        printf 'FAIL %-34s N=%s w=%s -> %s, expected %s\n' \
            "ray batch size" "${agents}" "${workers}" "${got}" "${expected}"
        failures=$((failures + 1))
        return
    fi
    # The batch is only right if it actually fills the workers, so assert the
    # saturation rule itself rather than trusting the arithmetic above.
    tasks=$(( (agents + got - 1) / got ))
    if (( agents > 0 && tasks < workers && agents >= workers )); then
        printf 'FAIL %-34s N=%s w=%s -> %s tasks, workers idle\n' \
            "ray batch size" "${agents}" "${workers}" "${tasks}"
        failures=$((failures + 1))
        return
    fi
    printf 'ok   %-34s N=%s w=%s -> batch %s, %s task(s)\n' \
        "ray batch size" "${agents}" "${workers}" "${got}" "${tasks}"
}
assert_batch_size 128 8 16
assert_batch_size 100 8 12
assert_batch_size 17 8 2
assert_batch_size 16 8 2
# Fewer agents than workers cannot fill them; one agent per task is the best
# available and must not round down to a zero-sized batch.
assert_batch_size 4 8 1
# An unknown population (NUM_AGENTS=0 means "use the config default") has no
# basis for a split, so fall back to upstream's own default rather than guess.
assert_batch_size 0 8 256

(
    job="${SCRIPT_DIR}/../../jobs/run_sim.sh"
    missing=0
    grep -q 'ray_batch_size' "${job}" || missing=1
    grep -q -- '--batch-size' "${job}" || missing=1
    if [[ "${missing}" -eq 0 ]]; then
        printf 'ok   %-34s run_sim.sh passes --batch-size\n' "ray batch size"
        exit 0
    fi
    printf 'FAIL %-34s run_sim.sh never passes --batch-size\n' "ray batch size"
    exit 1
) || failures=$((failures + 1))

# The JIT compilers vLLM invokes at startup size their own parallelism from
# nproc, which reports the whole physical node rather than the slots Grid
# Engine granted -- the same trap detect_cpu_cores already covers for Ray.
# ninja then launches ~200 nvcc processes competing for the node's physical
# memory -- nothing caps the job -- and the kernel OOM killer takes cicc with
# signal 9, which surfaces as an unexplained "Ninja build failed" and a vLLM
# that never comes up.
(
    job="${SCRIPT_DIR}/../../jobs/run_sim.sh"
    missing=0
    grep -q 'export MAX_JOBS' "${job}" || missing=1
    # Bounded by the granted slots, not by nproc: a literal or an nproc call
    # would reintroduce the bug the export exists to prevent.
    grep -q 'MAX_JOBS="${MAX_JOBS:-${CPU_CORES}}"' "${job}" || missing=1
    if [[ "${missing}" -eq 0 ]]; then
        printf 'ok   %-34s bounded by the granted slots\n' "jit build parallelism"
        exit 0
    fi
    printf 'FAIL %-34s run_sim.sh leaves MAX_JOBS unbounded\n' "jit build parallelism"
    exit 1
) || failures=$((failures + 1))

# The FP8 MoE path JIT-compiles a TensorRT-LLM GEMM, writes the cubin under
# tmp/ and renames it into cache/. Four jobs that start together compile the
# same shape into the same shared directory and three of them lose the rename
# to ENOENT, which surfaces as "Assertion failed: !cubin.empty()" and a vLLM
# that never starts. Both halves are asserted: the path is switched off, and
# the cache is per-job so nothing else on that path can collide either.
(
    rendered="$(
        JOB_ID=4242 TMPDIR=/tmp/check-jobtmp \
            bash -c "source '${SCRIPT_DIR}/common.sh'; printf '%s|%s' \"\${VLLM_BLOCKSCALE_FP8_GEMM_FLASHINFER}\" \"\${TRTLLM_DG_CACHE_DIR}\""
    )"
    flag="${rendered%%|*}" cache="${rendered##*|}"
    missing=0
    [[ "${flag}" == "0" ]] || missing=1
    # Per job, not merely somewhere writable: a constant path outside home
    # would still be shared by every job on the cluster. It must also land on
    # the job's own TMPDIR, so the artefacts go to node-local scratch and are
    # reclaimed with the job rather than accumulating on a shared filesystem.
    [[ "${cache}" == *4242* ]] || missing=1
    [[ "${cache}" == /tmp/check-jobtmp/* ]] || missing=1
    if [[ "${missing}" -eq 0 ]]; then
        printf 'ok   %-34s off, cache at %s\n' "trtllm gemm jit" "${cache}"
        exit 0
    fi
    printf 'FAIL %-34s flag=%s cache=%s\n' "trtllm gemm jit" "${flag}" "${cache}"
    exit 1
) || failures=$((failures + 1))

(
    # Exported, not merely assigned: vLLM runs in a child process and a shell
    # variable would never reach it.
    lib="${SCRIPT_DIR}/common.sh"
    missing=0
    grep -q 'export VLLM_BLOCKSCALE_FP8_GEMM_FLASHINFER' "${lib}" || missing=1
    grep -q 'export TRTLLM_DG_CACHE_DIR' "${lib}" || missing=1
    if [[ "${missing}" -eq 0 ]]; then
        printf 'ok   %-34s both exported\n' "trtllm gemm jit"
        exit 0
    fi
    printf 'FAIL %-34s not exported to the vLLM child\n' "trtllm gemm jit"
    exit 1
) || failures=$((failures + 1))

# Every claim made so far about whether the GPU is busy has come from a proxy:
# queue depth in vLLM's log, KV-cache occupancy, generated tokens per second.
# None of them is utilisation, and reasoning about saturation without it has
# already produced one wrong root cause. Sample the device directly.
(
    missing=0
    grep -q 'start_gpu_sampler' "${SCRIPT_DIR}/common.sh" || missing=1
    grep -q 'start_gpu_sampler' "${SCRIPT_DIR}/../../jobs/run_sim.sh" || missing=1
    # It has to run for the whole simulation, so it must start before the CLI
    # rather than alongside the summary at the end.
    cli_line="$(grep -n 'agentsociety2.society.cli' "${SCRIPT_DIR}/../../jobs/run_sim.sh" | head -1 | cut -d: -f1)"
    sampler_line="$(grep -n 'start_gpu_sampler' "${SCRIPT_DIR}/../../jobs/run_sim.sh" | head -1 | cut -d: -f1)"
    [[ -n "${cli_line}" ]] && [[ -n "${sampler_line}" ]] && (( sampler_line < cli_line )) || missing=1
    if [[ "${missing}" -eq 0 ]]; then
        printf 'ok   %-34s sampled for the whole run\n' "gpu telemetry"
        exit 0
    fi
    printf 'FAIL %-34s not sampled, or started too late\n' "gpu telemetry"
    exit 1
) || failures=$((failures + 1))

(
    # A sampler that dies with the job but leaves no file is worse than none:
    # it looks like it ran. Assert the query actually asks for utilisation, not
    # only memory, which is the proxy that has been misleading us.
    missing=0
    grep -q 'utilization.gpu' "${SCRIPT_DIR}/common.sh" || missing=1
    grep -q 'utilization.memory' "${SCRIPT_DIR}/common.sh" || missing=1
    # The job names the file; the library only writes where it is told.
    grep -q 'gpu.csv' "${SCRIPT_DIR}/../../jobs/run_sim.sh" || missing=1
    # Defining a stopper is not stopping. start_vllm and start_embedding_vllm
    # each install a bare `trap ... EXIT` that overwrites any other handler, so
    # the only place the sampler can be torn down is inside those handlers.
    for handler in stop_vllm stop_all_vllm; do
        awk -v fn="^${handler}\\(\\)" '
            $0 ~ fn {inside=1} inside && /stop_gpu_sampler/ {found=1} inside && /^}/ {exit}
            END {exit !found}' "${SCRIPT_DIR}/common.sh" || missing=1
    done
    if [[ "${missing}" -eq 0 ]]; then
        printf 'ok   %-34s records SM and memory utilisation\n' "gpu telemetry"
        exit 0
    fi
    printf 'FAIL %-34s does not record utilisation\n' "gpu telemetry"
    exit 1
) || failures=$((failures + 1))

# Grid Engine combines an embedded "#$ -l" with the one on the command line
# instead of letting the caller win, so a job script that hardcodes gpu_1=1 can
# never be submitted to a whole node: "Job is rejected because multiple
# specifying of node_? and cpu(or gpu)". run_sim.sh's own usage example showed
# -l node_f=1 and had therefore never worked.
(
    job="${SCRIPT_DIR}/../../jobs/run_sim.sh"
    if grep -q '^#\$ -l gpu_1=' "${job}"; then
        printf 'FAIL %-34s run_sim.sh hardcodes a GPU resource\n' "gpu resource"
        exit 1
    fi
    printf 'ok   %-34s chosen at submit time\n' "gpu resource"
) || failures=$((failures + 1))

# @description Assert which GPU resource submit.sh would request.
# @arg $1 string Expected resource, or "" when the caller supplies its own.
# @arg $@ string Arguments the caller passes to submit.sh.
assert_gpu_default() {
    local expected="$1"; shift
    local rendered
    rendered="$(
        bash -c "source '${SCRIPT_DIR}/common.sh'; default_gpu_resource \"\$@\"" _ "$@"
    )"
    if [[ "${rendered}" == "${expected}" ]]; then
        printf 'ok   %-34s %s\n' "gpu resource" "${expected:-caller supplied}"
    else
        printf 'FAIL %-34s got "%s", expected "%s"\n' "gpu resource" "${rendered}" "${expected}"
        failures=$((failures + 1))
    fi
}
# Nothing asked for: one GPU, which is what every run so far has used.
assert_gpu_default "gpu_1=1"
assert_gpu_default "gpu_1=1" -v FOO=bar
# The caller named a resource, so submit.sh must not add a second one.
assert_gpu_default "" -l node_f=1
assert_gpu_default "" -l gpu_h=1
assert_gpu_default "" -l h_rt=6:00:00 -l node_f=1
# An unrelated -l must not be mistaken for a GPU request.
assert_gpu_default "gpu_1=1" -l h_rt=6:00:00

if [[ "${failures}" -gt 0 ]]; then
    printf '\n%d check(s) failed\n' "${failures}"
    exit 1
fi

printf '\nall checks passed\n'
