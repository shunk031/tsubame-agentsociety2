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
    # Bounded by the worker budget, not by a literal and not by nproc. It was
    # the granted slot count until a whole-node job showed that Grid Engine
    # reports no slots there and nproc answers 192 -- which is the very
    # over-subscription this export exists to prevent.
    grep -q 'MAX_JOBS="${MAX_JOBS:-${WORKERS}}"' "${job}" || missing=1
    grep -q 'WORKERS="$(worker_budget' "${job}" || missing=1
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
        awk -v fn="^${handler}[(]" '
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

# uv.lock records a path dependency as `source = { directory = ... }` with no
# digest of what is in that directory, so `uv sync --locked` considers the
# vendored package satisfied no matter how its source changed and reuses the
# wheel it built the first time. The sync then reports success in seconds and
# the environment silently keeps the old code -- twice in one day a measurement
# was taken against a patch the venv did not contain.
(
    setup="${SCRIPT_DIR}/../setup/02_sync_env.sh"
    if grep -q -- '--reinstall-package agentsociety2' "${setup}"; then
        printf 'ok   %-34s vendored package rebuilt every sync\n' "env sync"
        exit 0
    fi
    printf 'FAIL %-34s uv sync will reuse a stale vendored build\n' "env sync"
    exit 1
) || failures=$((failures + 1))

# --- source snapshots -------------------------------------------------------

# A submission's source is copied into a directory of its own, and the job is
# pinned to that copy. What follows checks the three things that has to be true
# of: the copies land on the group area, two submissions never pick the same
# one, and an existing copy does not change when the next submission runs.
#
# None of these print WORK_ROOT or anything derived from it. This output gets
# pasted into issues.

(
    read -r work src runs <<<"$(
        bash -c "source '${SCRIPT_DIR}/common.sh'; printf '%s %s %s' \"\${WORK_ROOT}\" \"\${SRC_ROOT}\" \"\${RUNS_DIR}\""
    )"

    local_failures=0

    # Snapshots grow with every submission. Home is a shared quota; the group
    # area is what WORK_ROOT exists to name.
    if [[ -n "${src}" ]] && [[ "${src}" == "${work}/"* ]]; then
        printf 'ok   %-34s under the group area\n' "snapshot root"
    else
        printf 'FAIL %-34s not under the group area\n' "snapshot root"
        local_failures=$((local_failures + 1))
    fi

    if [[ "${src}" != "${HOME}/"* ]]; then
        printf 'ok   %-34s not on home\n' "snapshot root"
    else
        printf 'FAIL %-34s on home, which is a shared quota\n' "snapshot root"
        local_failures=$((local_failures + 1))
    fi

    # Grid Engine's own output file is directed here at submit time, so it
    # outlives the snapshot the job ran from.
    if [[ -n "${runs}" ]] && [[ "${runs}" != "${src}"* ]]; then
        printf 'ok   %-34s separate from the snapshot root\n' "run output"
    else
        printf 'FAIL %-34s inside the snapshot root; pruning would take the logs\n' "run output"
        local_failures=$((local_failures + 1))
    fi

    exit "${local_failures}"
) || failures=$((failures + $?))

# Two submit.sh processes started in the same second must not pick the same
# directory. That is the whole isolation guarantee, so test it in bulk rather
# than by generating two ids and hoping.
(
    ids="$(
        bash -c "source '${SCRIPT_DIR}/common.sh'; for _ in \$(seq 200); do new_snapshot_id; echo; done"
    )"
    total="$(printf '%s\n' "${ids}" | wc -l)"
    unique="$(printf '%s\n' "${ids}" | sort -u | wc -l)"

    local_failures=0

    if [[ "${total}" -eq 200 ]] && [[ "${unique}" -eq 200 ]]; then
        printf 'ok   %-34s 200 ids, 200 distinct\n' "snapshot id"
    else
        printf 'FAIL %-34s %s ids, only %s distinct\n' "snapshot id" "${total}" "${unique}"
        local_failures=$((local_failures + 1))
    fi

    # scripts/snapshots.sh decides which snapshots are newest by sorting these
    # names, and a person reads the commit out of them. Both depend on the shape.
    first="$(printf '%s\n' "${ids}" | head -1)"
    if [[ "${first}" =~ ^[0-9]{8}T[0-9]{6}Z-([0-9a-f]{7}|nogit)-[0-9a-f]{6}$ ]]; then
        printf 'ok   %-34s <utc>-<commit>-<random>\n' "snapshot id"
    else
        printf 'FAIL %-34s unexpected shape: %s\n' "snapshot id" "${first}"
        local_failures=$((local_failures + 1))
    fi

    exit "${local_failures}"
) || failures=$((failures + $?))

# The rsync that fills a snapshot. --delete against a shared tree is what used
# to erase a running job's output; a fresh directory per submission means there
# is nothing to delete, and reintroducing it would mean the destination is
# shared again.
(
    rendered="$(
        SRC_ROOT=/tmp/check-src REMOTE_REPO=/tmp/check-src/current \
            bash -c "source '${SCRIPT_DIR}/common.sh'; build_snapshot_rsync_args /tmp/check-src/snap; printf '%s\n' \"\${RSYNC_ARGS[*]}\""
    )"

    local_failures=0

    if [[ "${rendered}" != *"--delete"* ]]; then
        printf 'ok   %-34s no --delete\n' "snapshot rsync"
    else
        printf 'FAIL %-34s has --delete\n' "snapshot rsync"
        local_failures=$((local_failures + 1))
    fi

    # Without this every submission re-sends ~250 MB and keeps another copy of
    # it. The feature is affordability, not correctness, but losing it silently
    # would make snapshots per submission untenable.
    if [[ "${rendered}" == *"--link-dest=/tmp/check-src/current"* ]]; then
        printf 'ok   %-34s hardlinks against the previous snapshot\n' "snapshot rsync"
    else
        printf 'FAIL %-34s no --link-dest: %s\n' "snapshot rsync" "${rendered}"
        local_failures=$((local_failures + 1))
    fi

    exit "${local_failures}"
) || failures=$((failures + $?))

# The property itself, with a real rsync into a real directory: a snapshot does
# not change when the next submission runs. This is the failure that started it
# -- a sync from another branch between submission and start, and ten jobs
# running a commit nobody chose.
(
    fixture="$(mktemp -d)"
    trap 'rm -rf "${fixture}"' EXIT

    checkout="${fixture}/checkout"
    mkdir -p "${checkout}/scripts"
    printf 'first\n' >"${checkout}/scripts/gen_config.py"
    printf 'unchanged\n' >"${checkout}/uv.lock"

    export SRC_ROOT="${fixture}/src"
    export REMOTE_REPO="${SRC_ROOT}/current"
    mkdir -p "${SRC_ROOT}"

    # Not source=common.sh: letting shellcheck inline it here pairs the
    # ENABLE_THINKING export a few blocks up with the read inside
    # build_vllm_args and reports a subshell-scope warning about neither.
    # shellcheck source=/dev/null
    source "${SCRIPT_DIR}/common.sh"

    # The same sequence scripts/sync.sh runs, minus ssh: mint a name, create the
    # directory, fill it, repoint current.
    #
    # `mkdir -p` here where sync.sh uses a bare `mkdir`, on purpose. sync.sh
    # refuses a name that already exists and the block below asserts that it
    # still does; letting the fixture reuse a directory instead is what makes a
    # repeated name show up here as two submissions sharing one tree -- the
    # thing these four assertions are about -- rather than as a mkdir error.
    take_snapshot() {
        local id dir
        id="$(new_snapshot_id)"
        dir="${SRC_ROOT}/${id}"
        mkdir -p "${dir}" || return 1
        build_snapshot_rsync_args "${dir}"
        rsync "${RSYNC_ARGS[@]}" "${checkout}/" "${dir}/" || return 1
        ln -sn "${dir}" "${SRC_ROOT}/.current.${id}" || return 1
        mv -Tf "${SRC_ROOT}/.current.${id}" "${REMOTE_REPO}" || return 1
        printf '%s' "${dir}"
    }

    local_failures=0

    # A previous snapshot to link against, which is the steady state: only the
    # very first submission into a fresh WORK_ROOT has none. That case is
    # checked separately below, because rsync writes a warning about it and it
    # would otherwise look like a failure in this block.
    seed="${SRC_ROOT}/20250101T000000Z-0000000-000000"
    mkdir -p "${seed}"
    ln -sn "${seed}" "${REMOTE_REPO}"

    submitted="$(take_snapshot)"

    # rsync compares size and mtime at one-second granularity, so a rewrite in
    # the same second as the previous one can be missed. That is rsync's
    # behaviour, not this design's, but the fixture has to step past it or it
    # would be testing the quick check instead of the isolation.
    sleep 1.1
    printf 'second, from another branch\n' >"${checkout}/scripts/gen_config.py"
    later="$(take_snapshot)"

    if [[ "$(cat "${submitted}/scripts/gen_config.py")" == "first" ]]; then
        printf 'ok   %-34s a later sync leaves it alone\n' "snapshot isolation"
    else
        printf 'FAIL %-34s a later sync rewrote it\n' "snapshot isolation"
        local_failures=$((local_failures + 1))
    fi

    if [[ "$(cat "${later}/scripts/gen_config.py")" == "second, from another branch" ]]; then
        printf 'ok   %-34s the later sync got its own copy\n' "snapshot isolation"
    else
        printf 'FAIL %-34s the later sync did not land\n' "snapshot isolation"
        local_failures=$((local_failures + 1))
    fi

    if [[ "${submitted}" != "${later}" ]]; then
        printf 'ok   %-34s two syncs, two directories\n' "snapshot isolation"
    else
        printf 'FAIL %-34s both syncs wrote the same directory\n' "snapshot isolation"
        local_failures=$((local_failures + 1))
    fi

    # Unchanged files share an inode, which is what keeps this affordable.
    if [[ "$(stat -c %i "${submitted}/uv.lock")" == "$(stat -c %i "${later}/uv.lock")" ]]; then
        printf 'ok   %-34s unchanged files are hardlinked\n' "snapshot isolation"
    else
        printf 'FAIL %-34s unchanged files were copied again\n' "snapshot isolation"
        local_failures=$((local_failures + 1))
    fi

    # The first submission into a fresh WORK_ROOT has nothing to link against.
    # rsync treats that as a warning; were it ever an error, the first sync
    # after the group area is set up would fail and nothing else would say why.
    if rsync -a --link-dest="${fixture}/never-existed" "${checkout}/" "${fixture}/cold/" 2>/dev/null; then
        printf 'ok   %-34s a missing basis is not an error\n' "snapshot isolation"
    else
        printf 'FAIL %-34s a missing basis stopped the transfer\n' "snapshot isolation"
        local_failures=$((local_failures + 1))
    fi

    exit "${local_failures}"
) || failures=$((failures + $?))

# Functions that isolate nothing unless the scripts call them. Assert the call
# sites, the same way the per-run agent home is asserted above.
(
    sync_sh="${SCRIPT_DIR}/../sync.sh"
    missing=0

    grep -q 'new_snapshot_id' "${sync_sh}" || missing=1
    grep -q 'build_snapshot_rsync_args' "${sync_sh}" || missing=1
    # The transfer must go to the snapshot, not to a path shared between syncs.
    # Literal, not an expansion: this is the text the script must contain.
    # shellcheck disable=SC2016
    grep -qF '"${remote}:${snapshot_dir}/"' "${sync_sh}" || missing=1
    # `mkdir -p` would succeed on a directory that already exists, which is
    # exactly the case that must fail.
    # shellcheck disable=SC2016
    grep -qF 'mkdir $(printf '"'"'%q'"'"' "${snapshot_dir}")' "${sync_sh}" || missing=1

    if [[ "${missing}" -eq 0 ]]; then
        printf 'ok   %-34s sync.sh writes a fresh snapshot\n' "snapshot wiring"
        exit 0
    fi
    printf 'FAIL %-34s sync.sh does not write a fresh snapshot\n' "snapshot wiring"
    exit 1
) || failures=$((failures + 1))

(
    submit_sh="${SCRIPT_DIR}/../submit.sh"
    missing=0

    # The job has to be pinned to the snapshot sync.sh just made, by both
    # routes: the working directory and an explicit REPO_ROOT.
    # shellcheck disable=SC2016
    grep -qF 'cd $(printf '"'"'%q'"'"' "${snapshot_dir}")' "${submit_sh}" || missing=1
    # shellcheck disable=SC2016
    grep -qF 'REPO_ROOT=$(printf '"'"'%q'"'"' "${snapshot_dir}")' "${submit_sh}" || missing=1
    # And its output has to outlive that snapshot.
    # shellcheck disable=SC2016
    grep -qF 'qsub_args+=(-o "${RUNS_DIR}/")' "${submit_sh}" || missing=1

    if [[ "${missing}" -eq 0 ]]; then
        printf 'ok   %-34s submit.sh pins the job to it\n' "snapshot wiring"
        exit 0
    fi
    printf 'FAIL %-34s submit.sh does not pin the job to it\n' "snapshot wiring"
    exit 1
) || failures=$((failures + 1))

(
    watch_remote_sh="${SCRIPT_DIR}/../remote/watch_remote.sh"
    # Grid Engine's output moved out of the submit directory. A watcher still
    # looking in the old place finds nothing and says the job produced none.
    # shellcheck disable=SC2016
    if grep -qF 'for candidate in "${runs_dir}"/*."o${job_id}"' "${watch_remote_sh}"; then
        printf 'ok   %-34s watch looks for it under RUNS_DIR\n' "job output"
        exit 0
    fi
    printf 'FAIL %-34s watch does not look for it under RUNS_DIR\n' "job output"
    exit 1
) || failures=$((failures + 1))

# On the login node the tree is normally reached through the `current` symlink,
# which the next sync repoints. Resolving REPO_ROOT physically is what keeps a
# `uv sync` that started in one snapshot from finishing in another -- the third
# way this went wrong in practice.
(
    fixture="$(mktemp -d)"
    trap 'rm -rf "${fixture}"' EXIT

    repo="$(cd "${SCRIPT_DIR}/../.." && pwd -P)"
    ln -s "${repo}" "${fixture}/current"

    rendered="$(
        REPO_ROOT="" \
            bash -c "source '${fixture}/current/scripts/lib/common.sh'; printf '%s' \"\${REPO_ROOT}\""
    )"

    if [[ "${rendered}" == "${repo}" ]]; then
        printf 'ok   %-34s resolves through a symlink\n' "repo root"
        exit 0
    fi
    printf 'FAIL %-34s followed the symlink instead of resolving it\n' "repo root"
    exit 1
) || failures=$((failures + 1))

# --- snapshot pruning -------------------------------------------------------

# Snapshots accumulate, so there is a way to delete them; deleting the wrong one
# takes the source out from under a job that has not started yet. Exercise the
# remote script against a fixture with a stubbed qstat, since the decision it
# makes is entirely about what qstat reports.
(
    fixture="$(mktemp -d)"
    trap 'rm -rf "${fixture}"' EXIT

    src_root="${fixture}/src"
    mkdir -p "${src_root}" "${fixture}/bin"

    names=(
        20260101T000000Z-aaaaaaa-000001
        20260102T000000Z-aaaaaaa-000002
        20260103T000000Z-aaaaaaa-000003
        20260104T000000Z-aaaaaaa-000004
        20260105T000000Z-aaaaaaa-000005
        20260106T000000Z-aaaaaaa-000006
    )
    for name in "${names[@]}"; do
        mkdir -p "${src_root}/${name}"
        printf 'id      %s\nbranch  main\ncommit  aaaaaaa\n' "${name}" >"${src_root}/${name}/.snapshot"
    done

    # The third is queued behind a job; the fourth is what `current` points at.
    pinned="${src_root}/${names[2]}"
    ln -sn "${src_root}/${names[3]}" "${src_root}/current"

    cat >"${fixture}/bin/qstat" <<QSTAT
#!/usr/bin/env bash
if [[ "\${1:-}" == "-j" ]]; then
    [[ "\${2:-}" == "9000001" ]] || exit 1
    echo "job_number:                 9000001"
    echo "sge_o_workdir:              ${pinned}"
    exit 0
fi
cat <<'TABLE'
job-ID  prior   name     user   state submit/start at     queue        slots
---------------------------------------------------------------------------
9000001 0.55500 as2-sim  someone r    09/15/2026 10:00:00 all.q@node    8
TABLE
QSTAT
    chmod +x "${fixture}/bin/qstat"

    remote_sh="${SCRIPT_DIR}/../remote/snapshots_remote.sh"
    local_failures=0

    # Reporting must not delete anything. Deleting a snapshot is a decision, so
    # it takes a flag.
    PATH="${fixture}/bin:${PATH}" bash "${remote_sh}" "${src_root}" 2 0 >/dev/null 2>&1
    remaining="$(find "${src_root}" -mindepth 1 -maxdepth 1 -type d | wc -l)"
    if [[ "${remaining}" -eq 6 ]]; then
        printf 'ok   %-34s reporting deletes nothing\n' "snapshot prune"
    else
        printf 'FAIL %-34s reporting removed %s of 6\n' "snapshot prune" "$((6 - remaining))"
        local_failures=$((local_failures + 1))
    fi

    PATH="${fixture}/bin:${PATH}" bash "${remote_sh}" "${src_root}" 2 1 >/dev/null 2>&1

    # Queued job, `current`, and the newest two survive; the two nothing needs
    # go. Each is checked on its own, because "four directories left" would also
    # be true if it had removed the wrong four.
    check_state() {
        local name="$1" want="$2" label="$3"
        if [[ "${want}" == "present" && -d "${src_root}/${name}" ]] ||
            [[ "${want}" == "gone" && ! -d "${src_root}/${name}" ]]; then
            printf 'ok   %-34s %s\n' "snapshot prune" "${label}"
            return 0
        fi
        printf 'FAIL %-34s %s\n' "snapshot prune" "${label}"
        return 1
    }

    check_state "${names[2]}" present "kept the one a queued job is pinned to" || local_failures=$((local_failures + 1))
    check_state "${names[3]}" present "kept the one current points at" || local_failures=$((local_failures + 1))
    check_state "${names[5]}" present "kept the newest" || local_failures=$((local_failures + 1))
    check_state "${names[0]}" gone "removed the oldest unused" || local_failures=$((local_failures + 1))
    check_state "${names[1]}" gone "removed the second oldest unused" || local_failures=$((local_failures + 1))

    # Without qstat there is no way to know what is queued, and guessing deletes
    # the source of a job that has not started. An empty PATH is enough: the
    # check runs before the script needs any other command.
    bash_path="$(command -v bash)"
    output="$(PATH="" "${bash_path}" "${remote_sh}" "${src_root}" 2 1 2>&1)"
    status=$?
    if [[ "${status}" -ne 0 ]] && [[ "${output}" == *"refusing to prune"* ]]; then
        printf 'ok   %-34s refuses to prune without qstat\n' "snapshot prune"
    else
        printf 'FAIL %-34s pruned without qstat: status=%s %s\n' "snapshot prune" "${status}" "${output}"
        local_failures=$((local_failures + 1))
    fi

    exit "${local_failures}"
) || failures=$((failures + $?))

# --- environment provenance -------------------------------------------------

# @description Build a repository and a matching environment under one directory.
# @description
#   Real directories rather than canned digests: what fails in practice is the
#   walk over the trees -- which files are counted, which are skipped -- and a
#   fixture of strings would only test the string handling.
#
#   The environment holds a copy of the vendored package, which is what uv does
#   for a path dependency (nothing is installed editable, so patching the source
#   after a build does not reach the environment). tests/ exists in the vendored
#   package and not in the environment, because hatchling does not ship it.
# @arg $1 path Directory to build under.
make_provenance_fixture() {
    local base="$1"
    local repo="${base}/repo" venv="${base}/venv" package site

    package="${repo}/vendor/AgentSociety/packages/agentsociety2"
    site="${venv}/lib/python3.12/site-packages/agentsociety2"

    mkdir -p "${package}/agentsociety2/society" "${package}/tests" "${site}/society"

    printf 'version = 1\n' >"${repo}/uv.lock"
    printf '[project]\nname = "fixture"\n' >"${repo}/pyproject.toml"
    printf '[project]\nname = "agentsociety2"\nversion = "2.8.7"\n' >"${package}/pyproject.toml"
    printf '__version__ = "2.8.7"\n' >"${package}/agentsociety2/__init__.py"
    printf 'ROUTER = "react"\n' >"${package}/agentsociety2/society/cli.py"
    printf 'def test_nothing():\n    pass\n' >"${package}/tests/test_nothing.py"

    cp "${package}/agentsociety2/__init__.py" "${site}/__init__.py"
    cp "${package}/agentsociety2/society/cli.py" "${site}/society/cli.py"

    bash -c "source '${SCRIPT_DIR}/common.sh'; write_env_provenance '${venv}' '${repo}'" 2>/dev/null
}

# @description Run assert_env_matches_source over a fixture and judge the result.
# @arg $1 string Label for the output line.
# @arg $2 path Fixture environment.
# @arg $3 path Fixture repository.
# @arg $4 string Expected outcome: "pass", "warn" or "fail".
# @arg $5 string Substring the message must contain. Empty to skip the check.
# @arg $6 string Extra environment assignments, e.g. ENV_ALLOW_MISMATCH=1.
# @exitcode 1 The outcome or the message was not the expected one.
assert_provenance() {
    local label="$1" venv="$2" repo="$3" outcome="$4" needle="$5" env_prefix="${6:-}"
    local output reached=0

    # Intentional word splitting: the caller passes VAR=value assignments.
    # shellcheck disable=SC2086
    output="$(
        env ${env_prefix} bash -c \
            "source '${SCRIPT_DIR}/common.sh'; assert_env_matches_source '${venv}' '${repo}'; echo REACHED" 2>&1
    )"
    [[ "${output}" == *REACHED* ]] && reached=1

    case "${outcome}" in
        pass) [[ "${reached}" -eq 1 ]] && [[ "${output}" != *WARNING* ]] || { printf 'FAIL %-34s did not pass cleanly: %s\n' "${label}" "${output}"; return 1; } ;;
        warn) [[ "${reached}" -eq 1 ]] && [[ "${output}" == *WARNING* ]] || { printf 'FAIL %-34s did not warn and continue: %s\n' "${label}" "${output}"; return 1; } ;;
        fail) [[ "${reached}" -eq 0 ]] || { printf 'FAIL %-34s ran anyway: %s\n' "${label}" "${output}"; return 1; } ;;
        *) printf 'FAIL %-34s unknown expectation %s\n' "${label}" "${outcome}"; return 1 ;;
    esac

    if [[ -n "${needle}" ]] && [[ "${output}" != *"${needle}"* ]]; then
        printf 'FAIL %-34s message lacks "%s": %s\n' "${label}" "${needle}" "${output}"
        return 1
    fi

    printf 'ok   %-34s %s\n' "${label}" "${outcome}"
}

(
    fixture="$(mktemp -d)"
    trap 'rm -rf "${fixture}"' EXIT
    local_failures=0

    make_provenance_fixture "${fixture}"
    repo="${fixture}/repo"
    venv="${fixture}/venv"
    package="${repo}/vendor/AgentSociety/packages/agentsociety2"
    site="${venv}/lib/python3.12/site-packages/agentsociety2"

    # A freshly built environment matches the tree it was built from. Everything
    # below is a departure from this state.
    assert_provenance "provenance" "${venv}" "${repo}" pass "matches" || local_failures=$((local_failures + 1))

    # The version number does not move when the vendored source is patched, so
    # this is the case no version check can see. It is also the one that
    # produced a measurement of a patch that was never installed.
    cp "${package}/agentsociety2/society/cli.py" "${fixture}/cli.py.orig"
    printf 'ROUTER = "patched"\n' >"${package}/agentsociety2/society/cli.py"
    assert_provenance "provenance patched source" "${venv}" "${repo}" fail "differing in vendor" || local_failures=$((local_failures + 1))

    # ... and the escape hatch says so in the log rather than in someone's head.
    assert_provenance "provenance opt-out" "${venv}" "${repo}" warn "ENV_ALLOW_MISMATCH" ENV_ALLOW_MISMATCH=1 || local_failures=$((local_failures + 1))
    cp "${fixture}/cli.py.orig" "${package}/agentsociety2/society/cli.py"

    # Named components, so the message distinguishes a relocked dependency from
    # a patched one.
    printf 'version = 2\n' >"${repo}/uv.lock"
    assert_provenance "provenance relocked" "${venv}" "${repo}" fail "differing in lock" || local_failures=$((local_failures + 1))
    printf 'version = 1\n' >"${repo}/uv.lock"

    printf '[project]\nname = "agentsociety2"\nversion = "2.8.8"\n' >"${package}/pyproject.toml"
    assert_provenance "provenance vendored metadata" "${venv}" "${repo}" fail "differing in vendor_project" || local_failures=$((local_failures + 1))
    printf '[project]\nname = "agentsociety2"\nversion = "2.8.7"\n' >"${package}/pyproject.toml"

    # The environment itself is re-derived rather than taken on trust, so an
    # environment that was edited or half-deleted after its build is caught too.
    printf 'ROUTER = "edited in place"\n' >"${site}/society/cli.py"
    assert_provenance "provenance edited env" "${venv}" "${repo}" fail "no longer holds what its own build produced" || local_failures=$((local_failures + 1))
    cp "${package}/agentsociety2/society/cli.py" "${site}/society/cli.py"

    # A build that was killed partway leaves an environment that still imports.
    # The stamp is removed before a build and written after the import check, so
    # its absence is what that case looks like from here.
    mv "${venv}/.source-provenance" "${fixture}/stamp"
    assert_provenance "provenance unstamped env" "${venv}" "${repo}" fail "records no provenance" || local_failures=$((local_failures + 1))
    mv "${fixture}/stamp" "${venv}/.source-provenance"

    # Byte-compiled files are written by whoever imports the code first -- the
    # vendored tree by a local run, site-packages by the first job. Counting
    # them would make a stamp stop matching itself.
    mkdir -p "${site}/society/__pycache__" "${package}/agentsociety2/__pycache__"
    printf 'not really bytecode\n' >"${site}/society/__pycache__/cli.cpython-312.pyc"
    printf 'not really bytecode\n' >"${package}/agentsociety2/__pycache__/__init__.cpython-314.pyc"
    assert_provenance "provenance bytecode" "${venv}" "${repo}" pass "matches" || local_failures=$((local_failures + 1))

    # docs/, tests/ and examples/ live in the vendored package and never reach
    # the environment. A guard that fires on those gets switched off.
    printf 'def test_nothing():\n    assert True\n' >"${package}/tests/test_nothing.py"
    assert_provenance "provenance vendored tests" "${venv}" "${repo}" pass "matches" || local_failures=$((local_failures + 1))

    # An unpopulated submodule is an empty directory, which would otherwise
    # digest as a valid empty tree and match another empty one.
    mv "${package}/agentsociety2" "${fixture}/pkg"
    mkdir -p "${package}/agentsociety2"
    assert_provenance "provenance empty submodule" "${venv}" "${repo}" fail "no agentsociety2 package under" || local_failures=$((local_failures + 1))
    rmdir "${package}/agentsociety2"
    mv "${fixture}/pkg" "${package}/agentsociety2"

    # The digest has to describe content and not location: the same tree is read
    # from a checkout here, from the login node's copy and from site-packages.
    cp -r "${repo}" "${fixture}/elsewhere"
    here="$(bash -c "source '${SCRIPT_DIR}/common.sh'; source_fingerprint '${repo}'")"
    there="$(bash -c "source '${SCRIPT_DIR}/common.sh'; source_fingerprint '${fixture}/elsewhere'")"
    if [[ -n "${here}" ]] && [[ "${here}" == "${there}" ]]; then
        printf 'ok   %-34s same tree, two paths, one digest\n' "provenance location"
    else
        printf 'FAIL %-34s %s vs %s\n' "provenance location" "${here}" "${there}"
        local_failures=$((local_failures + 1))
    fi

    exit "${local_failures}"
) || failures=$((failures + $?))

# The stamp only means anything if the build maintains it: removed before the
# install so an interrupted build leaves no claim, written after the import
# check so it is never a claim about an environment nobody verified.
(
    setup="${SCRIPT_DIR}/../setup/02_sync_env.sh"
    missing=0
    # Literal text the build script must contain, not an expansion.
    # shellcheck disable=SC2016
    clear='rm -f "${VENV}/${ENV_PROVENANCE_NAME}"'

    # @description Line number of the first match. Empty when there is none --
    #   deliberately not 0, which would compare as "before everything" and let a
    #   missing line pass the ordering test below.
    # @arg $1 string Fixed string to look for.
    line_of() { grep -nF -m1 "$1" "${setup}" | cut -d: -f1; }

    cleared="$(line_of "${clear}")"
    installed="$(line_of 'uv sync --locked')"
    stamped="$(line_of 'write_env_provenance')"

    for line in "${cleared}" "${installed}" "${stamped}"; do
        [[ "${line}" =~ ^[0-9]+$ ]] || missing=1
    done
    # Order is the whole point: clearing after the install, or stamping before
    # the import check, would let an unfinished build keep a usable-looking
    # claim.
    [[ "${missing}" -eq 1 ]] || [[ "${cleared}" -lt "${installed}" ]] || missing=1
    [[ "${missing}" -eq 1 ]] || [[ "${installed}" -lt "${stamped}" ]] || missing=1
    if [[ "${missing}" -eq 0 ]]; then
        printf 'ok   %-34s 02_sync_env.sh clears then writes the stamp\n' "provenance build"
        exit 0
    fi
    printf 'FAIL %-34s 02_sync_env.sh does not maintain the stamp\n' "provenance build"
    exit 1
) || failures=$((failures + 1))

# A guard nothing calls is a guard that does not exist, and both jobs run out of
# the environment this check describes.
for job in run_sim smoke; do
    if grep -q 'assert_env_matches_source' "${SCRIPT_DIR}/../../jobs/${job}.sh"; then
        printf 'ok   %-34s jobs/%s.sh checks its environment\n' "provenance job" "${job}"
    else
        printf 'FAIL %-34s jobs/%s.sh does not check its environment\n' "provenance job" "${job}"
        failures=$((failures + 1))
    fi
done

# agentsociety2 never sets max_tokens, so a reply is bounded only by what is
# left of the context. With reasoning on, an agent can spend the whole budget
# thinking: four runs held the GPU at 93-95% for an hour and recorded no
# environment events at all, while a fifth on the same model and settings
# produced 110. The cap bounds the spend without switching reasoning off, which
# is why it is a separate knob from ENABLE_THINKING.
# @description Assert the generation cap vLLM would be started with.
# @arg $1 string Value of GENERATION_MAX_TOKENS.
# @arg $2 string Expected flag text, or "" when no flag should be emitted.
assert_generation_cap() {
    local value="$1" expected="$2" rendered
    rendered="$(
        GENERATION_MAX_TOKENS="${value}" MODEL=Qwen/Qwen3.6-27B \
            bash -c "source '${SCRIPT_DIR}/common.sh'; build_vllm_args 1; printf '%s' \"\${VLLM_ARGS[*]}\""
    )"
    if [[ -z "${expected}" ]]; then
        if [[ "${rendered}" != *override-generation-config* ]]; then
            printf 'ok   %-34s uncapped by default\n' "generation cap"
        else
            printf 'FAIL %-34s capped when it should not be\n' "generation cap"
            failures=$((failures + 1))
        fi
        return
    fi
    if [[ "${rendered}" == *"${expected}"* ]]; then
        printf 'ok   %-34s %s\n' "generation cap" "${expected}"
    else
        printf 'FAIL %-34s missing %s\n' "generation cap" "${expected}"
        failures=$((failures + 1))
    fi
}
# Unset keeps the present behaviour, so the cap is an experiment rather than a
# silent change to every run that came before it.
assert_generation_cap "" ""
assert_generation_cap 4096 '{"max_new_tokens": 4096}'

# The cap bounds a reply; it has nothing to do with whether the model reasons.
# Nesting it under the reasoning-model branch would make it a silent no-op for
# any other family, which is the failure mode this repository keeps hitting.
(
    rendered="$(
        GENERATION_MAX_TOKENS=4096 MODEL=meta-llama/Llama-3.1-8B-Instruct \
            bash -c "source '${SCRIPT_DIR}/common.sh'; build_vllm_args 1; printf '%s' \"\${VLLM_ARGS[*]}\""
    )"
    if [[ "${rendered}" == *'{"max_new_tokens": 4096}'* ]]; then
        printf 'ok   %-34s applies to any model family\n' "generation cap"
        exit 0
    fi
    printf 'FAIL %-34s skipped for a non-reasoning model\n' "generation cap"
    exit 1
) || failures=$((failures + 1))

# Reasoning must survive the cap: the point is to bound how long an agent
# thinks, not to stop it thinking.
(
    rendered="$(
        GENERATION_MAX_TOKENS=4096 ENABLE_THINKING=1 MODEL=Qwen/Qwen3.6-27B \
            bash -c "source '${SCRIPT_DIR}/common.sh'; build_vllm_args 1; printf '%s' \"\${VLLM_ARGS[*]}\""
    )"
    if [[ "${rendered}" == *'"enable_thinking": true'* ]]; then
        printf 'ok   %-34s reasoning still on\n' "generation cap"
        exit 0
    fi
    printf 'FAIL %-34s the cap switched reasoning off\n' "generation cap"
    exit 1
) || failures=$((failures + 1))

# vLLM waits VLLM_ENGINE_READY_TIMEOUT_S (600 by default) for its engine cores
# to come up. One GPU already takes 1285 s to become healthy with this model;
# four data-parallel processes load the same weights off the same filesystem
# and JIT-compile at the same time, so they never fit in 600. A whole-node run
# died that way after holding four GPUs for four hours, and the only visible
# symptom was a shared-memory broadcast warning -- the real error sat 150 lines
# further up. The wait has to scale with the number of processes, not stay at a
# single-GPU default.
# @description Assert the engine-ready timeout for a given GPU count.
# @arg $1 int Data parallel size.
# @arg $2 string Lower bound the timeout must meet or exceed.
assert_ready_timeout() {
    local dp="$1" minimum="$2" rendered
    # Through build_vllm_args, not by presetting DP_SIZE before the library is
    # sourced: jobs/run_sim.sh only computes DP_SIZE after sourcing, so a value
    # read at source time is always the single-GPU one. An earlier version of
    # this assertion set DP_SIZE first and passed against exactly that bug.
    rendered="$(
        MODEL=Qwen/Qwen3.6-27B bash -c \
            "source '${SCRIPT_DIR}/common.sh'; build_vllm_args '${dp}' >/dev/null; printf '%s' \"\${VLLM_ENGINE_READY_TIMEOUT_S}\""
    )"
    if [[ -n "${rendered}" ]] && (( rendered >= minimum )); then
        printf 'ok   %-34s dp=%s -> %ss\n' "engine ready timeout" "${dp}" "${rendered}"
    else
        printf 'FAIL %-34s dp=%s -> "%s", need >= %s\n' \
            "engine ready timeout" "${dp}" "${rendered}" "${minimum}"
        failures=$((failures + 1))
    fi
}
# One GPU measured 1285 s, so even the single-process case needs more than the
# 600 s default.
assert_ready_timeout 1 1800
assert_ready_timeout 4 3600

(
    # Exported, or the vLLM child never sees it.
    if grep -q 'export VLLM_ENGINE_READY_TIMEOUT_S' "${SCRIPT_DIR}/common.sh"; then
        printf 'ok   %-34s exported to the vLLM child\n' "engine ready timeout"
        exit 0
    fi
    printf 'FAIL %-34s not exported\n' "engine ready timeout"
    exit 1
) || failures=$((failures + 1))

# What sets the useful number of step_agent_batch tasks is how many concurrent
# LLM requests the GPUs can serve, not how many cores the node has. Those two
# came apart the first time a job asked for a whole node: gpu_1 exports
# NSLOTS=8 so detect_cpu_cores answered 8, but node_f exports no NSLOTS at all
# and nproc reports the node's 192. With 192 workers and 128 agents the batch
# floors to one agent per task, so 128 Ray processes start, each with its own
# LLM client and its own AIMD semaphore, and the run never advances -- SM
# utilisation sat at 0% on all four GPUs for forty minutes while the
# single-GPU run beside it held 90%.
(
    job="${SCRIPT_DIR}/../../jobs/run_sim.sh"
    missing=0
    grep -q 'worker_budget' "${job}" || missing=1
    # MAX_JOBS has to follow the same budget. Left on the raw core count it
    # puts 192 nvcc processes on a whole node, which is the OOM this repository
    # already fixed once for gpu_1.
    grep -q 'MAX_JOBS="${MAX_JOBS:-${CPU_CORES}}"' "${job}" && missing=1
    if [[ "${missing}" -eq 0 ]]; then
        printf 'ok   %-34s run_sim.sh uses it for Ray and the JIT\n' "worker budget"
        exit 0
    fi
    printf 'FAIL %-34s run_sim.sh still sizes from raw cores\n' "worker budget"
    exit 1
) || failures=$((failures + 1))

# @description Assert the worker budget for a core count and GPU count.
assert_worker_budget() {
    local cores="$1" gpus="$2" expected="$3" rendered
    rendered="$(
        bash -c "source '${SCRIPT_DIR}/common.sh'; worker_budget '${cores}' '${gpus}'"
    )"
    if [[ "${rendered}" == "${expected}" ]]; then
        printf 'ok   %-34s %s cores, %s gpu -> %s\n' "worker budget" "${cores}" "${gpus}" "${rendered}"
    else
        printf 'FAIL %-34s %s cores, %s gpu -> %s, expected %s\n' \
            "worker budget" "${cores}" "${gpus}" "${rendered}" "${expected}"
        failures=$((failures + 1))
    fi
}
# gpu_1 grants 8 slots per GPU and that configuration reached 90% SM, so the
# cluster's own ratio is the one to keep when the slot count is missing.
assert_worker_budget 8 1 8
assert_worker_budget 192 4 32
# Never above what the node actually has, however many GPUs are attached.
assert_worker_budget 8 4 8
# A GPU-less shell (the checks themselves) must still answer something usable.
assert_worker_budget 8 0 8

# The pathology to prevent: more workers than agents floors the batch to one
# agent per task.
(
    # Print the budget too: without it a missing worker_budget makes the
    # arithmetic fall back to a single task, and this assertion passes for
    # exactly the reason it exists to catch.
    rendered="$(
        bash -c "source '${SCRIPT_DIR}/common.sh'; w=\$(worker_budget 192 4); b=\$(ray_batch_size 128 \"\${w}\"); echo \"\${w} \$(( (128 + b - 1) / b ))\""
    )"
    budget="${rendered%% *}" tasks="${rendered##* }"
    if [[ -n "${budget}" ]] && (( budget > 1 )) && (( tasks > 1 )) && (( tasks <= 64 )); then
        printf 'ok   %-34s 128 agents on a whole node -> %s task(s)\n' "worker budget" "${tasks}"
        exit 0
    fi
    printf 'FAIL %-34s 128 agents -> %s tasks, one process per agent\n' "worker budget" "${tasks}"
    exit 1
) || failures=$((failures + 1))

if [[ "${failures}" -gt 0 ]]; then
    printf '\n%d check(s) failed\n' "${failures}"
    exit 1
fi

printf '\nall checks passed\n'
