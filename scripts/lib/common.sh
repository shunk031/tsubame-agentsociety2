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

# `pwd -P`, not `pwd`: on the login node the source tree is usually reached
# through the `current` symlink, which every sync repoints at a newer snapshot.
# Resolving it here pins a long-running command -- `uv sync` is the one that
# matters -- to the tree it started in, instead of letting the symlink move
# underneath it halfway through.
COMMON_SH_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
REPO_ROOT="${REPO_ROOT:-$(cd "${COMMON_SH_DIR}/../.." && pwd -P)}"

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

# --- Source snapshots -------------------------------------------------------

# Every submission copies the working tree into a directory of its own under
# here, and the job is pinned to that copy. The tree a queued job will run is
# therefore decided at submit time and cannot be edited afterwards -- not by
# the next sync, and not by whoever else is submitting at the same moment.
#
# On the group area rather than home: snapshots are hardlinked against each
# other so they cost little, but home is a shared quota and this grows with
# every submission. scripts/snapshots.sh is how they go away again.
SRC_ROOT="${SRC_ROOT:-${WORK_ROOT}/src}"

# The newest snapshot, for the things a human runs by hand: the setup scripts,
# and one-off commands through scripts/ssh.sh. Jobs never go through it. They
# are given a resolved snapshot path at submit time, because this symlink moves
# and the whole point is that their source does not.
REMOTE_REPO="${REMOTE_REPO:-${SRC_ROOT}/current}"

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

# Upper bound on a single reply, applied server-side because agentsociety2
# never sends max_tokens (see MAX_MODEL_LEN above). Unset leaves the reply
# bounded only by the remaining context, which is what lets a reasoning model
# spend the entire budget thinking and return nothing: four 128-agent runs held
# the GPU at 93-95% for an hour and recorded no environment events, while a
# fifth on the same model and settings recorded 110.
#
# Deliberately not a sibling of ENABLE_THINKING. Switching reasoning off buys
# throughput by giving up deliberation; this bounds how long the deliberation
# may run and keeps it.
GENERATION_MAX_TOKENS="${GENERATION_MAX_TOKENS:-}"

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

# --- Scheduler resources ----------------------------------------------------

# @description Report the GPU resource submit.sh should add, if any.
# @description
#   Grid Engine merges an embedded "#$ -l" with the command line rather than
#   letting the command line win, so a job script cannot both default to one
#   GPU and be submittable to a whole node: asking for node_f alongside a
#   hardcoded gpu_1 is rejected outright. The default therefore lives here,
#   where it can stand aside when the caller names its own resource.
# @arg $@ string The arguments submit.sh was given.
# @stdout The resource to add, or nothing when the caller already chose one.
default_gpu_resource() {
    local arg want=0
    for arg in "$@"; do
        if [[ "${want}" -eq 1 ]]; then
            # Only a GPU resource counts. An -l h_rt or -l m_mem_free says
            # nothing about devices and must not suppress the default.
            [[ "${arg}" == node_?=* ]] || [[ "${arg}" == gpu_*=* ]] && return 0
            want=0
            continue
        fi
        [[ "${arg}" == "-l" ]] && want=1
    done
    echo "gpu_1=1"
}

# --- GPU telemetry ----------------------------------------------------------

# @description Sample GPU utilisation into the run directory until the job ends.
# @description
#   vLLM's log offers queue depth, KV-cache occupancy and tokens per second.
#   None of those is utilisation: a run can hold nineteen requests, report seven
#   percent of the KV cache and still leave the SMs mostly idle, and reasoning
#   about saturation from those proxies has already produced one wrong answer
#   here. nvidia-smi is the only thing on this node that measures the device.
#
#   Stopped from inside stop_vllm / stop_all_vllm, which are what the EXIT trap
#   actually runs, so the sampler cannot outlive the run and keep writing into a
#   finished directory. A third bare `trap ... EXIT` would not survive: start_vllm
#   and start_embedding_vllm each overwrite the handler.
# @arg $1 path CSV to write.
# @set GPU_SAMPLER_PID int PID of the background sampler, when one started.
start_gpu_sampler() {
    local out="$1"
    if ! command -v nvidia-smi >/dev/null 2>&1; then
        log "gpu     no nvidia-smi; utilisation will not be recorded"
        return 0
    fi
    nvidia-smi \
        --query-gpu=timestamp,index,utilization.gpu,utilization.memory,memory.used,power.draw \
        --format=csv,nounits -l 5 >"${out}" 2>/dev/null &
    GPU_SAMPLER_PID=$!
    log "gpu     sampling every 5s to ${out}"
}

# @description Stop the sampler started by start_gpu_sampler, if any.
stop_gpu_sampler() {
    [[ -n "${GPU_SAMPLER_PID:-}" ]] || return 0
    kill "${GPU_SAMPLER_PID}" 2>/dev/null || true
    wait "${GPU_SAMPLER_PID}" 2>/dev/null || true
    GPU_SAMPLER_PID=""
}

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

# @description Mint the name for one submission's source snapshot.
# @description
#   Two properties matter, and only one of them is cosmetic.
#
#   The timestamp prefix sorts chronologically under a plain `sort`, which is
#   what scripts/snapshots.sh relies on to decide which snapshots are the newest
#   and what a human reads to see when a tree was shipped. The commit is there
#   so a directory listing on the login node answers "which code is this".
#
#   The random suffix is the part that has to be right. Two submissions started
#   in the same second -- two agents, two terminals -- would otherwise pick the
#   same name and land in the same directory, which is the failure this whole
#   mechanism exists to remove. It comes from /dev/urandom rather than $RANDOM
#   because two shells forked from one parent seed $RANDOM identically.
# @stdout Snapshot identifier, e.g. 20260915T104512Z-800e63d-3f9a1c.
new_snapshot_id() {
    local stamp head rand

    stamp="$(date -u '+%Y%m%dT%H%M%SZ')"
    head="$(git -C "${REPO_ROOT}" rev-parse --short=7 HEAD 2>/dev/null || echo nogit)"
    # od rather than `tr -dc ... | head -c`: head closing the pipe early makes
    # tr die of SIGPIPE, and under `set -o pipefail` that aborts the caller.
    rand="$(od -An -tx1 -N3 /dev/urandom | tr -d ' \n')"

    printf '%s-%s-%s' "${stamp}" "${head}" "${rand}"
}

# @description Populate the global RSYNC_ARGS array for a snapshot transfer.
# @description
#   The destination is a directory that does not exist yet, so there is nothing
#   to delete and `--delete` is gone. That removes the hazard the excludes below
#   used to guard: `--delete` against the one shared tree erased the job output
#   of whatever was still running. They stay only to keep stray local job output
#   out of a snapshot.
#
#   `--link-dest` is what makes a snapshot per submission affordable. The tree
#   is ~250 MB, nearly all of it vendor/AgentSociety, and a full copy per
#   submission would be paid twice: once on the wire and once on the group area.
#   Files identical to the previous snapshot are neither transferred nor copied,
#   just hardlinked.
#
#   The consequence is worth knowing: unchanged files in two snapshots are one
#   inode. rsync never trips over this -- it writes a new file and renames -- but
#   appending to a file inside a snapshot by hand would change it in every
#   snapshot that shares it. Edit the checkout and sync, do not edit a snapshot.
# @arg $1 path Snapshot directory on the login node. Absolute.
build_snapshot_rsync_args() {
    local dest="$1"

    [[ -n "${dest}" ]] || die "build_snapshot_rsync_args needs a destination"

    # Filled here, read by scripts/sync.sh, the same arrangement build_vllm_args
    # has with VLLM_ARGS. shellcheck only sees the assignment.
    # shellcheck disable=SC2034
    RSYNC_ARGS=(
        -az
        # Resolved on the receiving side. A missing one is a warning, not an
        # error, so the first snapshot into an empty SRC_ROOT works.
        --link-dest="${REMOTE_REPO}"
        --exclude '.git'
        --exclude '.claude'
        --exclude '__pycache__'
        --exclude '*.o[0-9]*'
        --exclude '*.e[0-9]*'
        --exclude '*.po[0-9]*'
        --exclude '*.pe[0-9]*'
    )
}

# @description Log which snapshot of the source this process is running from.
# @description
#   A job reads its code from a directory, and until now nothing in its output
#   said which commit that directory held. Ten jobs once ran the previous commit
#   without a word about it. The marker is written by scripts/sync.sh; running a
#   job script by hand from a checkout leaves it absent, which is itself worth
#   saying.
log_source_snapshot() {
    local marker="${REPO_ROOT}/.snapshot" line

    log "source  ${REPO_ROOT}"

    if [[ ! -f "${marker}" ]]; then
        log "source  no snapshot marker; this tree was not shipped by scripts/submit.sh"
        return 0
    fi

    while IFS= read -r line; do
        [[ -n "${line}" ]] && log "source  ${line}"
    done <"${marker}"

    return 0
}

# --- Environment provenance -------------------------------------------------

# agentsociety2 comes from vendor/AgentSociety, and nothing uv records moves
# when that source is patched: uv.lock names the dependency
# `source = { directory = ... }` and stores no digest of what is in the
# directory, and the version stays 2.8.7 across every patch. Two environments
# under WORK_ROOT/venvs can therefore report identical versions and hold
# different code -- which is deliberate, they are how patches are compared, and
# also how a run once reported on a patch that was never installed.
#
# The answer is a pair of digests written into the environment at build time
# and re-checked by every job: what the build read, and what it produced.

# Stamp name, relative to the environment root. It lives inside the environment
# rather than beside it so that deleting the environment takes its claim with
# it, and so an environment copied elsewhere keeps its own provenance.
ENV_PROVENANCE_NAME=".source-provenance"

# @description Digest a directory tree by content, independent of where it sits.
# @description
#   Paths enter the digest relative to the root, so the vendored source and the
#   copy uv installs into site-packages digest the same way, and a tree digests
#   the same on a laptop, on the login node and on a compute node.
#
#   __pycache__ is excluded because it is written by whoever imports the code
#   first: the vendored tree collects it from local runs, site-packages
#   collects it the moment a job starts. Including it would make a recorded
#   digest stop matching itself after the first run.
# @arg $1 path Directory to digest.
# @stdout 64 hex characters.
# @exitcode 1 No such directory.
fingerprint_tree() {
    local root="$1"

    [[ -d "${root}" ]] || return 1

    # LC_ALL=C so the ordering does not depend on the caller's locale; -print0
    # and -0 so a path with a space cannot split one entry into two.
    (
        cd "${root}" || exit 1
        find . -type f \
            ! -path '*/__pycache__/*' \
            ! -name '*.py[co]' \
            ! -path '*/.git/*' \
            ! -path '*.egg-info/*' \
            -print0 |
            LC_ALL=C sort -z |
            xargs -0 -r sha256sum
    ) | sha256sum | cut -d' ' -f1
}

# @description Print the digests that decide what `uv sync` will install.
# @description
#   Named components rather than one opaque number, so a mismatch can say which
#   half moved: a changed lockfile and a patched vendored source call for
#   different responses.
#
#   The vendored package contributes two of them. `vendor` is the importable
#   subtree hatchling puts in the wheel; docs/, tests/ and examples/ sit beside
#   it and never reach the environment, so editing one of those does not fire
#   the guard. `vendor_project` is the vendored pyproject.toml, which decides
#   what the wheel declares and therefore what uv resolves.
#
#   pyproject.toml at the repository root is deliberately not a component.
#   `uv sync --locked` refuses to run when it disagrees with uv.lock, so the
#   lockfile already stands for it, while hashing it too would fire on a
#   comment or a pytest setting that cannot reach the environment.
# @arg $1 path Repository root. Defaults to REPO_ROOT.
# @stdout One `<component> <digest>` line per component.
# @exitcode 1 Something the digest needs is missing.
source_fingerprint_components() {
    local repo="${1:-${REPO_ROOT}}"
    local package="${repo}/vendor/AgentSociety/packages/agentsociety2"
    local lock vendor_project vendor

    [[ -f "${repo}/uv.lock" ]] || {
        log "ERROR: no uv.lock under ${repo}"
        return 1
    }
    # An unpopulated submodule leaves an empty directory, which would otherwise
    # digest as a perfectly valid empty tree.
    [[ -f "${package}/agentsociety2/__init__.py" ]] || {
        log "ERROR: no agentsociety2 package under ${package}; run git submodule update --init --recursive"
        return 1
    }

    # Redirected rather than passed as arguments: sha256sum prints the path it
    # was given, and the digest has to depend on the bytes alone.
    lock="$(sha256sum <"${repo}/uv.lock" | cut -d' ' -f1)"
    vendor_project="$(sha256sum <"${package}/pyproject.toml" | cut -d' ' -f1)"
    vendor="$(fingerprint_tree "${package}/agentsociety2")" || return 1

    printf 'lock %s\nvendor_project %s\nvendor %s\n' \
        "${lock}" "${vendor_project}" "${vendor}"
}

# @description Digest the source an environment would be built from.
# @arg $1 path Repository root. Defaults to REPO_ROOT.
# @stdout 64 hex characters.
# @exitcode 1 The components could not be computed.
source_fingerprint() {
    local components

    components="$(source_fingerprint_components "${1:-${REPO_ROOT}}")" || return 1
    printf '%s\n' "${components}" | sha256sum | cut -d' ' -f1
}

# @description Locate the installed agentsociety2 package inside an environment.
# @description
#   The interpreter version is part of the path and this project pins 3.12, but
#   globbing keeps the helper from lying if that ever moves.
# @arg $1 path Virtualenv. Defaults to VENV.
# @stdout Absolute path to the package directory.
# @exitcode 1 The environment has no agentsociety2 in it.
venv_package_dir() {
    local venv="${1:-${VENV}}" candidate

    for candidate in "${venv}"/lib/python3.*/site-packages/agentsociety2; do
        if [[ -d "${candidate}" ]]; then
            printf '%s' "${candidate}"
            return 0
        fi
    done
    return 1
}

# @description Digest agentsociety2 as it is actually installed.
# @description
#   Only agentsociety2, not the whole of site-packages: every other package
#   comes from a wheel the lockfile pins by hash, and digesting torch and vLLM
#   as well would turn a sub-second check into a walk over tens of thousands of
#   files.
# @arg $1 path Virtualenv. Defaults to VENV.
# @stdout 64 hex characters.
# @exitcode 1 The environment has no agentsociety2 in it.
installed_fingerprint() {
    local package

    package="$(venv_package_dir "${1:-${VENV}}")" || return 1
    fingerprint_tree "${package}"
}

# @description Record what an environment was built from, inside the environment.
# @description
#   Two digests, because they fail apart. `source` is what the build read, and
#   catches a job launched from a tree that is not the one the environment came
#   from. `installed` is what the build produced, and catches an environment
#   that was edited, half-deleted or half-written after the fact.
#
#   Call this only after the installation has been verified. A stamp is a claim
#   that the environment is usable, and writing one for a build that has not
#   been checked would make the claim worthless.
# @arg $1 path Virtualenv. Defaults to VENV.
# @arg $2 path Repository the build read. Defaults to REPO_ROOT.
# @exitcode 1 Either digest could not be computed; no stamp is written.
write_env_provenance() {
    local venv="${1:-${VENV}}" repo="${2:-${REPO_ROOT}}"
    local stamp="${venv}/${ENV_PROVENANCE_NAME}"
    local components source installed version

    components="$(source_fingerprint_components "${repo}")" || return 1
    source="$(printf '%s\n' "${components}" | sha256sum | cut -d' ' -f1)"
    installed="$(installed_fingerprint "${venv}")" || {
        log "ERROR: no agentsociety2 installed in ${venv}"
        return 1
    }
    # Informational: the version is the thing that does not distinguish these
    # environments, which is worth being able to see at a glance.
    version="$("${venv}/bin/python" -c \
        'import importlib.metadata as m; print(m.version("agentsociety2"))' 2>/dev/null || echo unknown)"

    {
        printf 'schema 1\n'
        printf '%s\n' "${components}"
        printf 'source %s\n' "${source}"
        printf 'installed %s\n' "${installed}"
        printf 'version %s\n' "${version}"
        printf 'built_at %s\n' "$(date --iso-8601=seconds)"
    } >"${stamp}"

    log "recorded provenance: source ${source:0:12}, installed ${installed:0:12} (${stamp})"
}

# @description Refuse to run a job against an environment built from other source.
# @description
#   The failure this exists for is quiet by construction. An environment built
#   from the wrong source imports, runs and produces a replay that looks like
#   every other replay; the only way it was ever caught was by grepping the
#   installed package by hand afterwards. So the check is made at job start,
#   before the GPU time is spent, and it fails rather than warns.
#
#   Both recorded digests are re-derived rather than trusted: the source, from
#   the tree this job was launched with, and the installed package, from the
#   files in the environment. A stamp on its own would only prove that a build
#   once happened.
#
#   Running against a deliberately different environment is a normal thing to
#   do -- that is what the environments under WORK_ROOT/venvs are for. Set
#   ENV_ALLOW_MISMATCH=1 to say so; the job then records in its own log that it
#   did, because a run whose environment does not match its source is a run
#   whose log has to say which environment it described.
# @arg $1 path Virtualenv. Defaults to VENV.
# @arg $2 path Repository this job was launched from. Defaults to REPO_ROOT.
# @exitcode 1 They disagree and ENV_ALLOW_MISMATCH is unset.
assert_env_matches_source() {
    local venv="${1:-${VENV}}" repo="${2:-${REPO_ROOT}}"
    local stamp="${venv}/${ENV_PROVENANCE_NAME}"
    local key value components current installed problem=""
    local -A recorded=()
    local -a differing=()

    if [[ -r "${stamp}" ]]; then
        while read -r key value; do
            recorded["${key}"]="${value}"
        done <"${stamp}"
    else
        problem="${venv} records no provenance: either it predates this check, or its build never finished. A build killed partway leaves an environment that still imports and still reports a version, holding whatever it had copied by then."
    fi

    if [[ -z "${problem}" ]] && ! components="$(source_fingerprint_components "${repo}")"; then
        problem="the source under ${repo} could not be digested, so nothing can be said about ${venv}."
    fi

    if [[ -z "${problem}" ]]; then
        current="$(printf '%s\n' "${components}" | sha256sum | cut -d' ' -f1)"
        if [[ "${current}" != "${recorded[source]:-}" ]]; then
            while read -r key value; do
                [[ "${value}" == "${recorded[${key}]:-}" ]] || differing+=("${key}")
            done <<<"${components}"
            problem="${venv} was built from source this job was not launched with: recorded ${recorded[source]:0:12}, current ${current:0:12}, differing in ${differing[*]:-<unrecorded>}."
        fi
    fi

    if [[ -z "${problem}" ]]; then
        if ! installed="$(installed_fingerprint "${venv}")"; then
            problem="${venv} has no agentsociety2 installed, though it claims a build."
        elif [[ "${installed}" != "${recorded[installed]:-}" ]]; then
            problem="${venv} no longer holds what its own build produced: recorded ${recorded[installed]:0:12}, current ${installed:0:12}. Something rewrote the installed files after the build."
        fi
    fi

    if [[ -z "${problem}" ]]; then
        log "env     ${current:0:12} matches ${venv##*/} (built ${recorded[built_at]:-unknown}, agentsociety2 ${recorded[version]:-unknown})"
        return 0
    fi

    if [[ -n "${ENV_ALLOW_MISMATCH:-}" ]]; then
        log "WARNING: ${problem}"
        log "WARNING: continuing because ENV_ALLOW_MISMATCH is set. This run describes ${venv}, not the checkout it was launched from."
        return 0
    fi

    die "${problem} Rebuild it with scripts/setup/02_sync_env.sh, point VENV at the environment this source belongs to, or set ENV_ALLOW_MISMATCH=1 to run against it on purpose."
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

    # Ahead of the reasoning branch, and outside it on purpose: a cap bounds a
    # reply, which has nothing to do with whether the model reasons, and
    # nesting it there would make it a silent no-op for every other family.
    if [[ -n "${GENERATION_MAX_TOKENS}" ]]; then
        VLLM_ARGS+=(--override-generation-config \
            "{\"max_new_tokens\": ${GENERATION_MAX_TOKENS}}")
    fi

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
    # start_vllm and start_embedding_vllm each install a bare `trap ... EXIT`,
    # which overwrites any handler set before it. Tearing the sampler down from
    # inside the vLLM stoppers is therefore the only placement that survives
    # both, and it keeps the sampler from writing into a finished run.
    stop_gpu_sampler
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
    stop_gpu_sampler
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
