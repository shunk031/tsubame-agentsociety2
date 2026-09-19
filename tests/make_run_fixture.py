#!/usr/bin/env python3
"""Write a synthetic run directory that matches what a real TSUBAME run leaves.

The point is to develop and test ``scripts/plot_run.py`` without a job: every
quirk the report has to survive is reproduced deliberately.

Reproduced faithfully from the agentsociety2 2.8.7 source:

- ``replay/{table}.{shard:02x}.jsonl``, sharded by ``zlib.crc32(line) % 256``
  exactly as ``storage/replay_sink.py`` does, so the reader really has to glob.
- JSON columns are serialized as JSON *strings*, not nested objects
  (``replay_sink._normalize``). ``last_round`` therefore needs a second
  ``json.loads``.
- ``last_round`` repeats the previous round's summary on a step where no round
  executed (``contrib/env/commons_tragedy.py`` reads ``round_history[-1]``
  before deciding whether to run a round).
- ``pending_extractions`` / ``submitted_agents`` are cleared *before* the row is
  written, so they are empty on every row. Participation lives only in the keys
  of ``last_round.extractions``.
- Artifact filenames carry the *simulation* timestamp, not wall clock
  (``society/cli.py``).

The scenario is the bad case from shunk031/tsubame-agentsociety2#2: four agents,
four planned rounds, at most two participants per round, and a final round that
never resolved because nobody submitted.
"""

from __future__ import annotations

import argparse
import json
import zlib
from datetime import datetime, timedelta
from pathlib import Path

NUM_SHARDS = 256
START_T = datetime(2026, 1, 1, 9, 0, 0)
TICK = 900
POOL = 100
MAX_EXTRACTION = 10

ROLE = (
    "You are a participant in a shared-resource experiment. Your only activity "
    "is deciding how much to extract from a common pool each round; you have no "
    "job, home or errands to attend to."
)

AGENTS = [
    (1, "Agent-1", 47, "competitive and opportunistic, wants to come out ahead"),
    (2, "Agent-2", 31, "cautious and cooperative, dislikes waste"),
    (3, "Agent-3", 58, "analytical, reasons about long-term consequences"),
    (4, "Agent-4", 24, "impulsive, acts on what feels right in the moment"),
]

# scenario -> round -> {agent_name: requested extraction}. An absent agent did
# not submit at all; that absence is the whole point of the report.
#
# "bad" is the run shunk031/tsubame-agentsociety2#2 describes: at most two
# participants per round and a final round nobody resolved. "good" is what the
# scenario is supposed to look like — everyone acts, the pool actually drains,
# and the last round has to be allocated proportionally because the requests
# exceed what is left.
SCENARIOS: dict[str, dict[int, dict[str, int]]] = {
    "bad": {
        1: {"Agent-1": 5, "Agent-4": 5},
        2: {"Agent-1": 5},
        3: {"Agent-3": 1},
        4: {},
    },
    "good": {
        1: {"Agent-1": 9, "Agent-2": 3, "Agent-3": 5, "Agent-4": 8},
        2: {"Agent-1": 10, "Agent-2": 4, "Agent-3": 5, "Agent-4": 10},
        3: {"Agent-1": 10, "Agent-2": 2, "Agent-3": 4, "Agent-4": 10},
        4: {"Agent-1": 10, "Agent-2": 2, "Agent-3": 3, "Agent-4": 10},
    },
}
SUBMISSIONS: dict[int, dict[str, int]] = SCENARIOS["bad"]

# round -> {agent_name: (answer_text, reason)}. Self-reports drift from reality
# on purpose: Agent-2 never acts yet reports numbers, and round 4 is reported by
# everyone although no round executed.
SELF_REPORTS: dict[int, dict[str, tuple[str, str]]] = {
    1: {
        "Agent-1": ('{"reason": "I want an early lead.", "answer": 5}', None),
        "Agent-2": ('{"reason": "I took a modest share.", "answer": 3}', None),
        "Agent-3": ('{"reason": "I observed first.", "answer": 0}', None),
        "Agent-4": ('{"reason": "Felt right.", "answer": 5}', None),
    },
    2: {
        "Agent-1": ('{"reason": "Same as before.", "answer": 5}', None),
        "Agent-2": ('{"reason": "Still being careful.", "answer": 3}', None),
        "Agent-3": ('{"reason": "Nothing yet.", "answer": 0}', None),
        "Agent-4": ('{"reason": "I skipped this one.", "answer": 0}', None),
    },
    3: {
        "Agent-1": ('{"reason": "The pool is thinning.", "answer": 4}', None),
        "Agent-2": ('{"reason": "A small amount.", "answer": 2}', None),
        "Agent-3": ('{"reason": "One unit, to be safe.", "answer": 1}', None),
        # Prose instead of an integer: parse_success is False downstream.
        "Agent-4": ("I did not take anything this round.", None),
    },
    4: {
        "Agent-1": ('{"reason": "Last chance.", "answer": 6}', None),
        "Agent-2": ('{"reason": "Kept it small.", "answer": 2}', None),
        "Agent-3": ('{"reason": "Held back.", "answer": 1}', None),
        "Agent-4": ('{"reason": "Took a few.", "answer": 3}', None),
    },
}

# round -> list of (agent_id, action, observation) ReAct failures written to
# sim.log. These are the three failure shapes quoted in shunk031/tsubame-agentsociety2#2.
FAILURES: dict[int, list[tuple[int, str, str]]] = {
    1: [
        (2, "think", "Unknown tool: think"),
        (3, "get_current_time", "_tool_get_current_time() got an unexpected keyword argument 'param1'"),
    ],
    2: [
        (2, "think", "Unknown tool: think"),
        (3, "think", "Unknown tool: think"),
        (4, "ask_env", "ask_env mutation is disabled in readonly mode"),
    ],
    3: [
        (1, "ask_env", "ask_env mutation is disabled in readonly mode"),
        (2, "think", "Unknown tool: think"),
        (4, "plan", "Unknown tool: plan"),
    ],
    4: [
        (1, "think", "Unknown tool: think"),
        (2, "think", "Unknown tool: think"),
        (3, "ask_env", "ask_env mutation is disabled in readonly mode"),
        (4, "think", "Unknown tool: think"),
    ],
}


def select_scenario(name: str) -> None:
    """Point the module tables at one scenario.

    The "good" run needs no hand-written self-reports or failures: agents that
    act report what they did, and there is nothing to explain.
    """
    global SUBMISSIONS, SELF_REPORTS, FAILURES
    SUBMISSIONS = SCENARIOS[name]
    if name == "bad":
        return
    SELF_REPORTS = {
        round_number: {
            name_: (
                json.dumps(
                    {"reason": "I stated my amount to the environment.", "answer": amount}
                ),
                None,
            )
            for name_, amount in submissions.items()
        }
        for round_number, submissions in SUBMISSIONS.items()
    }
    FAILURES = {1: [(2, "think", "Unknown tool: think")]}


def normalize(value):
    """Mirror ``replay_sink._normalize``: datetime -> ISO, dict/list -> JSON text."""
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False)
    return value


def append_row(replay_dir: Path, table: str, row: dict) -> None:
    line = (
        json.dumps({k: normalize(v) for k, v in row.items()}, ensure_ascii=False) + "\n"
    ).encode("utf-8")
    shard = zlib.crc32(line) % NUM_SHARDS
    path = replay_dir / f"{table}.{shard:02x}.jsonl"
    with path.open("ab") as handle:
        handle.write(line)


def write_configs(run_dir: Path, num_rounds: int, layout: str) -> None:
    agents = [
        {
            "agent_id": agent_id,
            "agent_type": "PersonAgent",
            "kwargs": {
                "id": agent_id,
                "name": name,
                "age": age,
                "personality": f"{ROLE} Your disposition: {disposition}.",
                "max_react_turns": 8,
            },
        }
        for agent_id, name, age, disposition in AGENTS
    ]
    init_config = {
        "env_modules": [
            {
                "module_type": "CommonsTragedyEnv",
                "kwargs": {
                    "num_agents": len(AGENTS),
                    "initial_pool_resources": POOL,
                    "max_extraction_per_agent": MAX_EXTRACTION,
                },
            }
        ],
        "agents": agents,
    }
    (run_dir / "init_config.json").write_text(
        json.dumps(init_config, indent=2) + "\n", encoding="utf-8"
    )

    steps: list[dict] = []
    if layout == "ask-intervene":
        steps.append(
            {
                "type": "ask",
                "question": (
                    f"You share a common pool of {POOL} resource units with "
                    f"{len(AGENTS) - 1} others."
                ),
            }
        )
    for round_number in range(1, num_rounds + 1):
        if layout == "ask-intervene":
            steps.append(
                {
                    "type": "intervene",
                    "instruction": (
                        f"It is now round {round_number}. First pick a whole number "
                        f"between 1 and {MAX_EXTRACTION}: how many units you want "
                        "from the shared pool. Then use ask_env once, writing that "
                        "number out in the request."
                    ),
                }
            )
        steps.append({"type": "run", "num_steps": 1, "tick": TICK})
        steps.append(
            {
                "type": "questionnaire",
                "questionnaire_id": f"round_{round_number}",
                "title": f"Round {round_number}",
                "description": "",
                "questions": [
                    {
                        "id": "amount",
                        "prompt": (
                            f"In round {round_number} of the shared-resource game, "
                            "how many units did you request from the pool?"
                        ),
                        "response_type": "integer",
                    }
                ],
                "target_agent_ids": [a[0] for a in AGENTS],
            }
        )
    # gen_config.py writes JSON into steps.yaml; JSON is valid YAML.
    (run_dir / "steps.yaml").write_text(
        json.dumps({"start_t": START_T.isoformat(), "steps": steps}, indent=2) + "\n",
        encoding="utf-8",
    )


def write_profiles(replay_dir: Path) -> None:
    for agent_id, name, age, disposition in AGENTS:
        append_row(
            replay_dir,
            "core_agent_profile",
            {
                "id": agent_id,
                "name": name,
                "profile": {
                    "name": name,
                    "age": age,
                    "personality": f"{ROLE} Your disposition: {disposition}.",
                },
                "created_at": START_T,
            },
        )


def allocate(requested: dict[str, int], pool_before: int) -> dict[str, int]:
    """Port of ``CommonsTragedyEnv._calculate_actual_extractions_sync``.

    Worth copying exactly rather than approximating: when the requests exceed
    the pool, the largest fractional parts absorb the remainder, so the pool
    lands on exactly zero. The report uses that to tell a rationed round from a
    round where the agents simply asked for less than was there.
    """
    total = sum(requested.values())
    if not total or pool_before <= 0:
        return {name: 0 for name in requested}
    if total <= pool_before:
        return dict(requested)

    scale = pool_before / total
    actual = {name: int(value * scale) for name, value in requested.items()}
    remainder = pool_before - sum(actual.values())
    fractional = sorted(
        ((requested[name] * scale - actual[name], name) for name in requested),
        key=lambda pair: pair[0],
        reverse=True,
    )
    for _ in range(int(remainder)):
        if not fractional:
            break
        _, name = fractional.pop(0)
        actual[name] += 1
    return actual


def simulate(num_rounds: int) -> list[dict]:
    """Replay the env's own bookkeeping to produce one row per run step."""
    pool = POOL
    round_number = 0
    history: list[dict] = []
    rows: list[dict] = []

    for step_index in range(num_rounds):
        requested = SUBMISSIONS.get(step_index + 1, {})
        last_round = history[-1] if history else None

        if requested:
            round_number += 1
            pool_before = pool
            actual = allocate(requested, pool_before)
            pool = pool_before - sum(actual.values())
            last_round = {
                "round": round_number,
                "pool_before_round": pool_before,
                "extractions": actual,
                "pool_after_round": pool,
                "payoffs": dict(actual),
            }
            history.append(last_round)

        rows.append(
            {
                "step": step_index,
                "t": START_T + timedelta(seconds=TICK * (step_index + 1)),
                "round_number": round_number,
                "current_pool_resources": pool,
                "last_round": last_round,
                # Cleared by the env before the row is written — always empty.
                "pending_extractions": {},
                "submitted_agents": [],
                "initial_pool_resources": POOL,
                "max_extraction_per_agent": MAX_EXTRACTION,
            }
        )
    return rows


def write_questionnaires(artifacts: Path, num_rounds: int, layout: str) -> None:
    for round_number in range(1, num_rounds + 1):
        sim_time = START_T + timedelta(seconds=TICK * round_number)
        # ask + (intervene, run, questionnaire)* vs (run, questionnaire)*
        step_idx = 3 * round_number if layout == "ask-intervene" else 2 * round_number - 1
        responses = []
        for agent_id, name, _age, _disposition in AGENTS:
            raw, _ = SELF_REPORTS[round_number][name]
            payload = None
            try:
                payload = json.loads(raw)
            except json.JSONDecodeError:
                payload = None
            if isinstance(payload, dict) and "answer" in payload:
                answer_text = str(payload["answer"])
                reason = payload.get("reason")
                parsed_value = payload["answer"]
                ok = isinstance(parsed_value, int)
                error = None if ok else "no integer found"
            else:
                answer_text = raw
                reason = None
                parsed_value = raw
                ok = False
                error = "no integer found"
            responses.append(
                {
                    "agent_id": agent_id,
                    "agent_name": name,
                    "answers": [
                        {
                            "question_id": "amount",
                            "raw_text": answer_text,
                            "raw_response": raw,
                            "parsed_value": parsed_value,
                            "reason": reason,
                            "parse_success": ok,
                            "parse_error": error,
                        }
                    ],
                }
            )
        document = {
            "questionnaire_id": f"round_{round_number}",
            "title": f"Round {round_number}",
            "description": "",
            "simulation_time": sim_time.isoformat(),
            "step_count": round_number,
            "target_agent_ids": [a[0] for a in AGENTS],
            "questions": [
                {
                    "id": "amount",
                    "prompt": (
                        f"In round {round_number} of the shared-resource game, how "
                        "many units did you request from the pool?"
                    ),
                    "response_type": "integer",
                    "choices": [],
                }
            ],
            "responses": responses,
            "context_snapshots": [],
        }
        name = f"questionnaire_step_{step_idx}_{sim_time:%Y%m%d_%H%M%S}.json"
        (artifacts / name).write_text(
            json.dumps(document, indent=2, ensure_ascii=False), encoding="utf-8"
        )


def write_intervene_artifacts(artifacts: Path, num_rounds: int) -> None:
    for round_number in range(1, num_rounds + 1):
        sim_time = START_T + timedelta(seconds=TICK * (round_number - 1))
        step_idx = 3 * round_number - 2
        instruction = (
            f"It is now round {round_number}. First pick a whole number between 1 "
            f"and {MAX_EXTRACTION}: how many units you want from the shared pool. "
            "Then use ask_env once, writing that number out in the request, for "
            "example 'submit an extraction of 7 units for me' if you chose 7."
        )
        body = (
            "---\n"
            f"instruction: {instruction}\n"
            "---\n\n"
            f"Delivered the round {round_number} instruction to 4 agents.\n"
        )
        name = f"intervene_step_{step_idx}_{sim_time:%Y%m%d_%H%M%S}.md"
        (artifacts / name).write_text(body, encoding="utf-8")


def write_sim_log(run_dir: Path, num_rounds: int, layout: str) -> None:
    """Emit the log shape agentsociety2 actually writes.

    Format is ``[%(asctime)s] %(levelname)-7s %(message)s`` with
    ``datefmt="%Y-%m-%d %H:%M:%S"`` (agentsociety2/logger/__init__.py). Wall
    clock, not simulation time.
    """
    wall = datetime(2026, 9, 12, 4, 30, 0)
    lines: list[str] = []

    def emit(level: str, message: str, seconds: int = 0) -> None:
        nonlocal wall
        wall += timedelta(seconds=seconds)
        lines.append(f"[{wall:%Y-%m-%d %H:%M:%S}] {level:<7} {message}")

    emit("INFO", "Loading config from init_config.json")
    emit("INFO", "Society initialized with 4 agents", 12)
    if layout == "ask-intervene":
        emit("INFO", "Asking: You share a common pool of 100 resource units with 3 others.", 3)
        emit("INFO", "Answer: acknowledged by 4 agents", 41)

    # Each round costs a different amount of wall time; round 4 is the longest
    # because every agent burned its ReAct budget without producing an action.
    round_cost = {1: 148, 2: 121, 3: 133, 4: 206}
    for round_number in range(1, num_rounds + 1):
        if layout == "ask-intervene":
            emit("INFO", f"Intervening: It is now round {round_number}. First pick a whole number between 1 and 10", 2)
            emit("INFO", "Result: instruction delivered to 4 agents", 18)
        emit("INFO", "Running 1/1 steps with tick=900", 1)
        cost = round_cost.get(round_number, 130)
        per_failure = max(1, cost // (len(FAILURES.get(round_number, [])) + 2))
        for agent_id, action, observation in FAILURES.get(round_number, []):
            emit(
                "WARNING",
                f"Agent {agent_id}: ReAct tool failed: action={action} observation={observation}",
                per_failure,
            )
        if round_number == 4:
            emit("WARNING", "Agent 4: invalid ReAct decision: LLM returned empty content", per_failure)
        emit("INFO", f"Running questionnaire round_{round_number} with 1 questions", per_failure)
        emit("INFO", "Questionnaire result saved to artifacts/", 58)

    emit("INFO", "Simulation finished", 4)
    (run_dir / "sim.log").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out_dir", type=Path, help="Run directory to create")
    parser.add_argument("--num-rounds", type=int, default=4)
    parser.add_argument("--scenario", choices=sorted(SCENARIOS), default="bad")
    parser.add_argument(
        "--layout",
        choices=("run-only", "ask-intervene"),
        default="run-only",
        help=(
            "Step layout. 'run-only' is the (run, questionnaire) sequence this "
            "repository generates, which writes no intervene artifacts and no "
            "'Intervening:' log lines; 'ask-intervene' is the retired ask + "
            "(intervene, run, questionnaire) sequence, kept so the report can "
            "still be checked against runs recorded before "
            "shunk031/tsubame-agentsociety2#6"
        ),
    )
    args = parser.parse_args()

    select_scenario(args.scenario)
    run_dir = args.out_dir
    replay_dir = run_dir / "replay"
    artifacts = run_dir / "artifacts"
    replay_dir.mkdir(parents=True, exist_ok=True)
    artifacts.mkdir(parents=True, exist_ok=True)

    # Shards are append-only, so regenerating into an existing directory would
    # otherwise stack a second copy of every row on top of the first.
    for stale in list(replay_dir.glob("*.jsonl")) + list(artifacts.glob("*")):
        stale.unlink()

    write_configs(run_dir, args.num_rounds, args.layout)
    write_profiles(replay_dir)
    for row in simulate(args.num_rounds):
        append_row(replay_dir, "commons_tragedy_env_state", row)
    write_questionnaires(artifacts, args.num_rounds, args.layout)
    if args.layout == "ask-intervene":
        write_intervene_artifacts(artifacts, args.num_rounds)
    write_sim_log(run_dir, args.num_rounds, args.layout)
    (run_dir / "vllm.log").write_text("INFO: Uvicorn running on http://127.0.0.1:8000\n", encoding="utf-8")

    shards = sorted(p.name for p in replay_dir.glob("*.jsonl"))
    print(f"wrote {run_dir}")
    print(f"  replay shards: {len(shards)} ({', '.join(shards)})")
    print(f"  artifacts:     {len(list(artifacts.iterdir()))} files")


if __name__ == "__main__":
    main()
