#!/usr/bin/env bash

# @file scripts/lib/common.sh
# @brief Shared paths, defaults and helpers for the TSUBAME jobs.
# @description
#   Sourced by every script here. Every value is overridable from the
#   environment, so one job script runs unchanged on a single-GPU smoke node and
#   on a full node_f with four H100s.
#
#   Sourcing this file starts no process and writes nothing.

# --- Site configuration -----------------------------------------------------

COMMON_SH_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "${COMMON_SH_DIR}/../.." && pwd)}"

# Account, group and reservation id identify one particular allocation, so they
# stay out of version control. Copy config/env.example to config/env.local and
# fill it in; that file is git-ignored and rsynced to the login node with the
# rest of the tree.
if [[ -f "${REPO_ROOT}/config/env.local" ]]; then
    # shellcheck source=/dev/null
    source "${REPO_ROOT}/config/env.local"
fi

: "${TSUBAME_GROUP:?not set - copy config/env.example to config/env.local and fill it in}"

# The environment, model weights and run outputs live on the group area. Home is
# a shared quota that the weights alone can exhaust. The path names the group,
# so it is configured rather than derived.
: "${WORK_ROOT:?not set - copy config/env.example to config/env.local and fill it in}"

# One environment serves both vLLM and AgentSociety. uv resolves them together;
# see the note in pyproject.toml about why vLLM is left unpinned.
VENV="${VENV:-${WORK_ROOT}/venv}"
RUNS_DIR="${RUNS_DIR:-${WORK_ROOT}/runs}"

export HF_HOME="${HF_HOME:-${WORK_ROOT}/hf-cache}"

# --- uv ---------------------------------------------------------------------

# The uv cache stays on home. Source builds run their isolation interpreter from
# inside the cache, and doing that under the group area fails with EPERM —
# `stringcase`, a hard dependency of agentsociety2 that ships no wheel, is the
# one that trips over it.
export UV_CACHE_DIR="${UV_CACHE_DIR:-${HOME}/.cache/uv}"

# Home and the group area are separate Lustre mounts, so uv cannot hardlink
# between them.
export UV_LINK_MODE="${UV_LINK_MODE:-copy}"

# The login node reports 96 cores but caps a user at 150 processes
# (`ulimit -u`). Left alone, uv sizes its rayon pool from the core count and
# aborts partway through a large install with "failed to initialize global
# rayon pool ... Resource temporarily unavailable".
export UV_CONCURRENT_INSTALLS="${UV_CONCURRENT_INSTALLS:-8}"
export UV_CONCURRENT_DOWNLOADS="${UV_CONCURRENT_DOWNLOADS:-8}"
export UV_CONCURRENT_BUILDS="${UV_CONCURRENT_BUILDS:-2}"

# --- Model selection --------------------------------------------------------

# The default is deliberately small so pipeline bugs surface in seconds rather
# than after a multi-minute model load. See README.md for production values.
MODEL="${MODEL:-Qwen/Qwen3.5-4B}"
# agentsociety2 never sets max_tokens — the string does not appear anywhere in
# the package — so a reply is bounded only by what is left of the context after
# the prompt. With reasoning switched on, a thinking model can spend that budget
# before writing any content, and the caller sees "LLM returned empty content".
# Raising this buys room for both. It is not free: KV cache scales with it, and
# a large model on one GPU will refuse to start if the cache cannot hold a full
# sequence. Qwen3.5-4B accepts up to 262144.
MAX_MODEL_LEN="${MAX_MODEL_LEN:-65536}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.85}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-256}"

# Let agents reason before acting. Set to 0 to trade deliberation for
# throughput; see build_vllm_args for what it switches.
export ENABLE_THINKING="${ENABLE_THINKING:-1}"

# vLLM ships qwen3_coder and qwen3_xml for this family. The Qwen3.x chat
# templates emit <tool_call><function=...>, which is what qwen3_coder reads.
TOOL_CALL_PARSER="${TOOL_CALL_PARSER:-qwen3_coder}"
VLLM_HOST="${VLLM_HOST:-127.0.0.1}"

# Grid Engine places jobs by slot, so two of them can land on one node, and the
# loopback interface is not namespaced between them. With a fixed port the
# second job fails to bind, wait_for_vllm's probe is answered by the first
# job's server, and the run proceeds against a model it never asked for with
# its own flags silently ignored. Deriving the port from the job id keeps
# co-located jobs off each other; assert_port_free is what catches the rest.
if [[ -n "${JOB_ID:-}" ]]; then
    VLLM_PORT="${VLLM_PORT:-$((8000 + 10#${JOB_ID} % 1000))}"
else
    VLLM_PORT="${VLLM_PORT:-8000}"
fi

# --- TensorRT-LLM kernel cache ----------------------------------------------

# On SM90+ the FP8 MoE path defaults to a TensorRT-LLM block-scale GEMM that is
# JIT-compiled at startup: nvcc writes a cubin under <cache>/tmp/<shape>/ and it
# is renamed into <cache>/cache/<shape>/. The cache directory defaults to
# $HOME/.tensorrt_llm, which is shared by every job this account runs, and the
# rename is not safe against a second job compiling the same shape. Four 35B
# jobs submitted together all died on the identical shape at the same moment,
# the rename failing with ENOENT, which reaches the log only as
# "Assertion failed: !cubin.empty() || isPathValid(path_)".
#
# Two independent guards, because either alone leaves a hole. Switching the
# path off is what actually unblocks FP8 MoE today -- it costs the small-batch
# TRT-LLM optimisation and saves a multi-minute DeepGEMM warmup -- but it does
# not protect any other deep_gemm kernel that may reach the same cache. Giving
# each job its own cache directory does that, and costs nothing while the path
# above stays off, since nothing is compiled into it.
#
# Grid Engine gives every job a private TMPDIR on node-local storage and removes
# it at exit, which is what a per-job compile cache wants; RAY_TMPDIR already
# relies on the same guarantee. RUN_DIR would be wrong here -- the job scripts
# define it after sourcing this file, so it is still unset at this point.
export VLLM_BLOCKSCALE_FP8_GEMM_FLASHINFER="${VLLM_BLOCKSCALE_FP8_GEMM_FLASHINFER:-0}"
export TRTLLM_DG_CACHE_DIR="${TRTLLM_DG_CACHE_DIR:-${TMPDIR:-/tmp}/trtllm-cache-${JOB_ID:-local}}"

# --- Embedding --------------------------------------------------------------

# Every ask_env misses the code-generation template cache without one. The
# router asks for an embedding, the call fails, `_lookup` returns
# `embedding_unavailable`, and the instruction is regenerated from scratch. On
# a 128-agent run that was 1487 misses and 11179 seconds of ask_env — and the
# env actor serialises those, so it is the run's critical path rather than a
# per-agent cost.
#
# Serving one costs a second vLLM process: vLLM runs one model per process.
# Set ENABLE_EMBEDDING=0 to go without it.
ENABLE_EMBEDDING="${ENABLE_EMBEDDING:-1}"

# 0.6B, ~1.2 GB. Its 1024 dimensions already match what agentsociety2 expects,
# so AGENTSOCIETY_EMBEDDING_DIMS needs no override.
EMBEDDING_MODEL="${EMBEDDING_MODEL:-Qwen/Qwen3-Embedding-0.6B}"

# A separate port from the generation server, derived the same way so two jobs
# on one node stay apart. assert_port_free covers what derivation misses.
if [[ -n "${JOB_ID:-}" ]]; then
    EMBEDDING_PORT="${EMBEDDING_PORT:-$((9000 + 10#${JOB_ID} % 1000))}"
else
    EMBEDDING_PORT="${EMBEDDING_PORT:-9000}"
fi

# Both servers share one GPU, so their reservations have to add up to less than
# the card. The generation model keeps the bulk; this is the remainder it
# leaves, not an independent budget.
EMBEDDING_GPU_MEMORY_UTILIZATION="${EMBEDDING_GPU_MEMORY_UTILIZATION:-0.06}"
EMBEDDING_MAX_MODEL_LEN="${EMBEDDING_MAX_MODEL_LEN:-2048}"

# --- AgentSociety -----------------------------------------------------------

# A placeholder, not a secret: vLLM serves an unauthenticated local endpoint and
# ignores the value. It still has to be set, because agentsociety2 validates it
# while importing agentsociety2.config (config/config.py:492) — merely importing
# the CLI module raises ValueError without it.
export AGENTSOCIETY_LLM_API_KEY="${AGENTSOCIETY_LLM_API_KEY:-tsubame-local}"
export AGENTSOCIETY_LLM_API_BASE="${AGENTSOCIETY_LLM_API_BASE:-http://${VLLM_HOST}:${VLLM_PORT}/v1}"
export AGENTSOCIETY_LLM_MODEL="${AGENTSOCIETY_LLM_MODEL:-${MODEL}}"

# Both libraries phone home by default, and the compute nodes have outbound
# network access, so this has to be switched off rather than left to fail.
# agentsociety2's CLI sets neither.
export MEM0_TELEMETRY="${MEM0_TELEMETRY:-False}"
export ANONYMIZED_TELEMETRY="${ANONYMIZED_TELEMETRY:-False}"

# The environment decides the shape of the scenario. CommonsTragedyEnv gives
# agents a depletable shared pool to draw from, so their choices interact:
# what one takes changes what is left for the others.
ENV_MODULE="${ENV_MODULE:-CommonsTragedyEnv}"

# Empty means "whatever the environment expects".
NUM_AGENTS="${NUM_AGENTS:-0}"

# One round is a run step followed by a questionnaire. The questionnaire is
# where the data comes from; a run on its own records almost nothing. There is
# no ask or intervene step — see the docstring on gen_config.build_steps.
NUM_ROUNDS="${NUM_ROUNDS:-4}"
TICK_SECONDS="${TICK_SECONDS:-900}"

POOL_RESOURCES="${POOL_RESOURCES:-100}"
MAX_EXTRACTION="${MAX_EXTRACTION:-10}"

# Where agentsociety2 keeps state that outlives a run. Upstream defaults
# AGENTSOCIETY_HOME_DIR to "./agentsociety_data", which is relative to the
# working directory, and Grid Engine runs jobs from the submit directory --
# so every job on this cluster shared one codegen template cache and kept
# adding to it.
#
# That makes a measurement depend on which runs happened before it. A 16-agent
# run logged 175-237 template cache misses early on and 11-12 for the same
# configuration later, purely because the cache had warmed up in between, and
# a comparison across that gap measures the cache rather than the change.
#
# Pointing this at the run directory gives every run a cold cache, so two runs
# of the same configuration produce the same numbers. Set AGENT_HOME_MODE to
# `shared` to opt back into a cache that persists across runs -- worth it when
# throughput matters more than comparability.
AGENT_HOME_MODE="${AGENT_HOME_MODE:-per-run}"

# Replay records required before a run counts as successful.
MIN_REPLAY_RECORDS="${MIN_REPLAY_RECORDS:-1}"

# Fraction of agents that must act per round. A round resolves as soon as one
# agent submits, so "every round resolved" can still mean one agent playing and
# the rest looking on — a different simulation from the one being described.
MIN_PARTICIPATION="${MIN_PARTICIPATION:-0.5}"

# Seconds to wait for vLLM to report healthy.
VLLM_STARTUP_TIMEOUT="${VLLM_STARTUP_TIMEOUT:-1800}"

# --- Logging ----------------------------------------------------------------

# @description Print a timestamped message to stderr.
# @arg $1 string Message to log.
log() {
    printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$1" >&2
}

# @description Log an error and terminate the script.
# @arg $1 string Message to log.
# @exitcode 1 Always.
die() {
    log "ERROR: $1"
    exit 1
}

# --- Environment detection --------------------------------------------------

# @description Count the GPUs visible to this process.
# @stdout GPU count, or 0 when nvidia-smi is unavailable.
detect_gpu_count() {
    if command -v nvidia-smi >/dev/null 2>&1; then
        nvidia-smi -L 2>/dev/null | grep -c '^GPU' || echo 0
    else
        echo 0
    fi
}

# @description Report the CPU cores this job may actually use.
# @description
#   Grid Engine exposes the granted slot count as NSLOTS. Left alone, Ray falls
#   back to os.cpu_count(), which reports the whole physical node (192 cores on
#   node_f) regardless of what UGE granted, and over-subscribes the job.
# @stdout Usable core count.
detect_cpu_cores() {
    if [[ -n "${NSLOTS:-}" ]] && [[ "${NSLOTS}" -gt 0 ]]; then
        echo "${NSLOTS}"
    else
        nproc
    fi
}

# @description Agents per step_agent_batch Ray Task, sized to fill the workers.
# @description
#   Upstream chunks the agent list every tick and submits ceil(N / BATCH_SIZE)
#   Ray Tasks, of which at most AGENTSOCIETY_LLM_RAY_MAX_WORKERS run at once.
#   Its default batch of 256 is therefore a trap at this scale: every
#   population this repository has run -- 4, 16, 128, 256 agents -- produced a
#   single task, so one process held the only AIMD semaphore and neither more
#   agents nor more workers could raise the concurrent request count. Upstream
#   states the rule in config.py: "choose a size where ceil(N / BATCH_SIZE) >=
#   LLM_RAY_MAX_WORKERS to saturate the workers; otherwise some workers sit
#   idle."
#
#   Dividing down rather than up is what makes that inequality hold. Rounding
#   up gives ceil(17 / ceil(17/8)) = 6 tasks for 8 workers; rounding down gives
#   9, which fills them. A population smaller than the worker count cannot fill
#   them at all, and clamps to one agent per task.
# @arg $1 int Agent count; 0 means the count is left to the config.
# @arg $2 int Concurrent worker count.
# @stdout Batch size.
ray_batch_size() {
    local agents="$1" workers="$2" size
    # No population to split: upstream's own default is the only honest answer.
    (( agents > 0 )) || { echo 256; return; }
    (( workers > 0 )) || workers=1
    size=$(( agents / workers ))
    (( size > 0 )) || size=1
    echo "${size}"
}

# --- Source tree ------------------------------------------------------------

# @description Refuse to ship a checkout that is behind its upstream.
# @description
#   scripts/sync.sh mirrors the working tree, not a commit, so a checkout that
#   has not been pulled puts old code on the cluster with nothing to say so.
#   That is not a theoretical risk: a batch of ten jobs once ran the previous
#   commit this way, silently, because the sync happened before the pull.
#
#   The comparison is against the remote-tracking ref as it stands, with no
#   network access, so it reflects the last fetch. It stays quiet when there is
#   no upstream to compare against — a feature branch that was never pushed is
#   a normal thing to sync, and a guard that fires on those gets ignored.
#
#   Set SYNC_ALLOW_STALE=1 to ship an older tree deliberately.
# @arg $1 path Repository to inspect. Defaults to REPO_ROOT.
# @exitcode 1 The branch is behind its upstream and SYNC_ALLOW_STALE is unset.
assert_not_behind_upstream() {
    local repo="${1:-${REPO_ROOT}}" branch upstream behind ahead head

    git -C "${repo}" rev-parse --git-dir >/dev/null 2>&1 || return 0

    branch="$(git -C "${repo}" rev-parse --abbrev-ref HEAD 2>/dev/null)" || return 0
    [[ "${branch}" != "HEAD" ]] || return 0

    upstream="$(git -C "${repo}" rev-parse --abbrev-ref --symbolic-full-name '@{upstream}' 2>/dev/null)" || return 0
    [[ -n "${upstream}" ]] || return 0

    behind="$(git -C "${repo}" rev-list --count "HEAD..${upstream}" 2>/dev/null)" || return 0
    [[ "${behind}" -gt 0 ]] || return 0

    ahead="$(git -C "${repo}" rev-list --count "${upstream}..HEAD" 2>/dev/null || echo 0)"
    head="$(git -C "${repo}" rev-parse --short HEAD 2>/dev/null)"

    if [[ -n "${SYNC_ALLOW_STALE:-}" ]]; then
        log "WARNING: ${branch} (${head}) is ${behind} commit(s) behind ${upstream}; shipping it anyway because SYNC_ALLOW_STALE is set"
        return 0
    fi

    die "${branch} (${head}) is ${behind} commit(s) behind ${upstream} (${ahead} ahead). sync.sh copies the working tree, so the cluster would run code your git history does not match. Pull first, or set SYNC_ALLOW_STALE=1 to ship this tree on purpose."
}

# --- vLLM -------------------------------------------------------------------

# @description Populate the global VLLM_ARGS array for the selected model.
# @description
#   Reasoning is requested explicitly for the model families that support it,
#   rather than left to the model's own default. Expert parallelism only helps
#   MoE checkpoints spread over more than one GPU.
# @arg $1 int Data parallel size.
build_vllm_args() {
    local dp_size="$1"

    VLLM_ARGS=(
        serve "${MODEL}"
        --host "${VLLM_HOST}"
        --port "${VLLM_PORT}"
        --data-parallel-size "${dp_size}"
        --max-model-len "${MAX_MODEL_LEN}"
        --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}"
        --max-num-seqs "${MAX_NUM_SEQS}"
        --enable-prefix-caching
    )

    # Qwen3.5 and newer think by default, and that default is kept: agents that
    # deliberate before acting are closer to what a social simulation is trying
    # to model. It is not free — reasoning adds hundreds to thousands of output
    # tokens per call, and output tokens are what this workload spends its time
    # on — so ENABLE_THINKING=0 turns it off when throughput matters more.
    # `--reasoning-parser none` does not exist; the parser stays on either way
    # and the chat template kwarg is what switches thinking.
    if [[ "${MODEL}" =~ Qwen3\.[5-9] ]]; then
        VLLM_ARGS+=(--reasoning-parser qwen3)
        if [[ "${ENABLE_THINKING}" == "0" ]]; then
            VLLM_ARGS+=(--default-chat-template-kwargs '{"enable_thinking": false}')
        else
            VLLM_ARGS+=(--default-chat-template-kwargs '{"enable_thinking": true}')
        fi
    fi

    # AgentSociety sends tool_choice="auto" on its agent-step calls. Without
    # these two flags vLLM rejects the request outright:
    #   "auto" tool choice requires --enable-auto-tool-choice and
    #   --tool-call-parser to be set
    # litellm retries, the retries fail the same way, and the run limps on with
    # a thinner replay than it should have. qwen3_coder matches the
    # <tool_call><function=...> shape in the Qwen3.x chat templates; qwen3_xml
    # is the other parser vLLM ships for this family.
    if [[ "${MODEL}" =~ Qwen3 ]]; then
        VLLM_ARGS+=(
            --enable-auto-tool-choice
            --tool-call-parser "${TOOL_CALL_PARSER}"
        )
    fi

    # MoE checkpoints are named after their active parameter count: A3B, A10B.
    if [[ "${MODEL}" =~ -A[0-9]+B ]] && [[ "${dp_size}" -gt 1 ]]; then
        VLLM_ARGS+=(--enable-expert-parallel)
    fi

    if [[ -n "${VLLM_EXTRA_ARGS:-}" ]]; then
        # Intentional word splitting: callers pass a flag string.
        # shellcheck disable=SC2206
        VLLM_ARGS+=(${VLLM_EXTRA_ARGS})
    fi
}

# @description Fail unless nothing is already listening on the vLLM port.
# @description
#   This is the check that makes a port collision loud. Deriving the port from
#   the job id makes a clash unlikely, not impossible, and a stale server from
#   an earlier job can hold the port too. Without this, the health probe is
#   satisfied by whoever is listening: the job reports "vLLM healthy after 0s",
#   runs its whole simulation against another job's model, and exits 0. When
#   the two jobs happen to serve the same model even check_endpoint.py passes,
#   so nothing downstream would notice.
# @arg $1 int Port to test.
# @exitcode 1 Something is already listening.
assert_port_free() {
    local port="$1"

    # The probe runs in a subshell, so the descriptor it opens is never held by
    # this shell and needs no closing here. Do not add a bare `exec` with
    # redirections to tidy up: that applies them to the shell itself, and
    # `exec 2>/dev/null` would silently discard the message below.
    if (exec 3<>"/dev/tcp/${VLLM_HOST}/${port}") 2>/dev/null; then
        die "${VLLM_HOST}:${port} on $(hostname) is already serving; refusing to start rather than talk to another job's vLLM. Override VLLM_PORT to pick another."
    fi
}

# @description Block until the vLLM server reports healthy.
# @description
#   Polls /health instead of sleeping a fixed interval, because model load time
#   varies by an order of magnitude between a cold and a warm HuggingFace cache.
#   Gives up as soon as the server process dies rather than waiting out the
#   whole timeout.
# @arg $1 int PID of the vLLM process to watch.
# @exitcode 0 Server became healthy.
# @exitcode 1 Server died, or stayed unhealthy for VLLM_STARTUP_TIMEOUT seconds.
wait_for_vllm() {
    local vllm_pid="$1"
    local url="http://${VLLM_HOST}:${VLLM_PORT}/health"
    local waited=0

    log "waiting for vLLM at ${url} (timeout ${VLLM_STARTUP_TIMEOUT}s)"
    while [[ "${waited}" -lt "${VLLM_STARTUP_TIMEOUT}" ]]; do
        if ! kill -0 "${vllm_pid}" 2>/dev/null; then
            log "vLLM process ${vllm_pid} exited before becoming healthy"
            return 1
        fi
        if curl -sf -o /dev/null --max-time 5 "${url}"; then
            log "vLLM healthy after ${waited}s"
            return 0
        fi
        sleep 5
        waited=$((waited + 5))
    done

    log "vLLM did not become healthy within ${VLLM_STARTUP_TIMEOUT}s"
    return 1
}

# @description Start a second vLLM serving the embedding model.
# @description
#   vLLM serves one model per process, so the codegen cache needs its own
#   server. It shares the GPU with the generation model, which is why the two
#   memory fractions are set together rather than each taking a default.
#
#   `--runner pooling --convert embed` is required rather than optional:
#   Qwen3-Embedding is built on the Qwen3 causal-LM architecture, so `auto`
#   loads it as a generative model and /v1/embeddings never appears. vLLM
#   0.22.1 has no `--task`; that flag was replaced by this pair.
# @arg $1 path File to write the server log to.
# @exitcode 1 Server failed to start.
start_embedding_vllm() {
    local log_file="$1"

    assert_port_free "${EMBEDDING_PORT}"

    log "starting embedding vLLM (${EMBEDDING_MODEL}) on ${VLLM_HOST}:${EMBEDDING_PORT}, logging to ${log_file}"
    "${VENV}/bin/vllm" serve "${EMBEDDING_MODEL}" \
        --host "${VLLM_HOST}" \
        --port "${EMBEDDING_PORT}" \
        --runner pooling \
        --convert embed \
        --max-model-len "${EMBEDDING_MAX_MODEL_LEN}" \
        --gpu-memory-utilization "${EMBEDDING_GPU_MEMORY_UTILIZATION}" \
        >"${log_file}" 2>&1 &
    EMBEDDING_PID=$!
    trap stop_all_vllm EXIT

    local url="http://${VLLM_HOST}:${EMBEDDING_PORT}/health" waited=0
    while [[ "${waited}" -lt "${VLLM_STARTUP_TIMEOUT}" ]]; do
        if ! kill -0 "${EMBEDDING_PID}" 2>/dev/null; then
            log "embedding vLLM exited before becoming healthy; last 40 lines:"
            tail -40 "${log_file}" >&2 || true
            die "embedding vLLM failed to start"
        fi
        if curl -sf -o /dev/null --max-time 5 "${url}"; then
            log "embedding vLLM healthy after ${waited}s"
            return 0
        fi
        sleep 5
        waited=$((waited + 5))
    done

    log "last 40 lines of ${log_file}:"
    tail -40 "${log_file}" >&2 || true
    die "embedding vLLM did not become healthy within ${VLLM_STARTUP_TIMEOUT}s"
}

# @description Stop both vLLM servers, if they are still up.
# @description
#   Replaces stop_vllm as the EXIT trap once a second server exists. Without it
#   the embedding server survives the job and holds its GPU memory until the
#   wall clock runs out.
# shellcheck disable=SC2317
stop_all_vllm() {
    local pid
    for pid in "${EMBEDDING_PID:-}" "${VLLM_PID:-}"; do
        if [[ -n "${pid}" ]] && kill -0 "${pid}" 2>/dev/null; then
            log "stopping vLLM (pid ${pid})"
            kill "${pid}" 2>/dev/null || true
            wait "${pid}" 2>/dev/null || true
        fi
    done
}

# @description Stop the vLLM server started by start_vllm, if it is still up.
# @description
#   Installed as an EXIT trap. Without it a failure further down the job leaves
#   the server holding its GPUs until the wall clock runs out.
# shellcheck disable=SC2317
stop_vllm() {
    if [[ -n "${VLLM_PID:-}" ]] && kill -0 "${VLLM_PID}" 2>/dev/null; then
        log "stopping vLLM (pid ${VLLM_PID})"
        kill "${VLLM_PID}" 2>/dev/null || true
        wait "${VLLM_PID}" 2>/dev/null || true
    fi
}

# @description Start vLLM in the background and block until it answers.
# @description
#   Sets the global VLLM_PID and installs the EXIT trap that stops it. On
#   failure it prints the tail of the server log, because the job's own output
#   says nothing about why the server gave up.
# @arg $1 int Data parallel size.
# @arg $2 path File to write the server log to.
# @exitcode 1 Server failed to start.
start_vllm() {
    local dp_size="$1" log_file="$2"

    assert_port_free "${VLLM_PORT}"

    build_vllm_args "${dp_size}"
    log "starting vLLM on ${VLLM_HOST}:${VLLM_PORT}, logging to ${log_file}"
    "${VENV}/bin/vllm" "${VLLM_ARGS[@]}" >"${log_file}" 2>&1 &
    VLLM_PID=$!
    trap stop_vllm EXIT

    if ! wait_for_vllm "${VLLM_PID}"; then
        log "last 40 lines of ${log_file}:"
        tail -40 "${log_file}" >&2 || true
        die "vLLM failed to start"
    fi
}
