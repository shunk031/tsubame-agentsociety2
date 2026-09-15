#!/usr/bin/env python3
"""Confirm that a finished run actually recorded agent activity.

agentsociety2 catches exceptions on several paths, so a run whose LLM calls all
failed can still exit 0. The replay files are the evidence that the simulation
did something, which makes them the honest success criterion.

Counting rows is not enough on its own, and neither is finding a resolved round.
A commons-tragedy run can resolve every round with one agent acting and the rest
looking on, which is a different simulation from the one the scenario describes.
So the last thing checked is how many agents reached the environment each round.

Storage layout, per agentsociety2.storage.replay_sink: every process appends
JSONL to ``<run_dir>/replay/{table}.{shard:02x}.jsonl``, 256 shards guarded by
flock. There is no database, so counting lines is the whole job. One wrinkle:
``_normalize`` runs ``json.dumps`` over any dict or list before the row is
written, so a JSON column arrives as a string that has to be decoded again.

Reads RUN_DIR, MIN_REPLAY_RECORDS and MIN_PARTICIPATION from the environment.
"""

from __future__ import annotations

import collections
import json
import os
import sys
from pathlib import Path


def main() -> int:
    run_dir = Path(os.environ["RUN_DIR"])
    minimum = int(os.environ.get("MIN_REPLAY_RECORDS", "1"))
    min_participation = float(os.environ.get("MIN_PARTICIPATION", "0.5"))

    replay_dir = run_dir / "replay"
    if not replay_dir.is_dir():
        print(f"FAIL no replay directory at {replay_dir}", file=sys.stderr)
        return 1

    # Shards of one table differ only by suffix; report per table, since an
    # empty table is more informative than an empty total.
    per_table: collections.Counter[str] = collections.Counter()

    for shard in sorted(replay_dir.glob("*.jsonl")):
        # "<table>.<shard>.jsonl" -> "<table>"
        table = shard.name.rsplit(".", 2)[0]
        with shard.open(encoding="utf-8") as handle:
            per_table[table] += sum(1 for line in handle if line.strip())

    if not per_table:
        print(f"FAIL {replay_dir} holds no replay shards", file=sys.stderr)
        return 1

    width = max(len(table) for table in per_table)
    for table, count in sorted(per_table.items()):
        print(f"{table:{width}}  {count} records")

    total = sum(per_table.values())
    print(f"{'total':{width}}  {total} records")

    if total < minimum:
        print(
            f"FAIL only {total} replay records, expected at least {minimum}; "
            "the simulation ran but recorded almost nothing, which usually "
            "means the LLM calls failed",
            file=sys.stderr,
        )
        return 1

    return _report_interaction(replay_dir, min_participation)


def _decode(value):
    """Return a JSON column as a Python object.

    replay_sink._normalize json.dumps-es any dict or list before the row is
    written, so these columns come back as strings. Objects are accepted too, in
    case that ever changes.
    """
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return None
    return value


def round_summaries(states: list[dict]) -> list[dict]:
    """Return one summary per resolved round, in order.

    ``last_round`` holds the summary of the most recently resolved round and is
    rewritten unchanged on every step that resolved nothing, so the column has
    to be deduplicated by round number before it can be counted.
    """
    by_round: dict[int, dict] = {}
    for row in states:
        summary = _decode(row.get("last_round"))
        if isinstance(summary, dict) and "round" in summary:
            by_round[int(summary["round"])] = summary
    return [by_round[number] for number in sorted(by_round)]


def participation(summary: dict) -> int:
    """Count the agents that reached the environment in this round.

    Keys of ``extractions``, not non-zero values: the environment splits a short
    pool proportionally and rounds down, so an agent that submitted can still
    come away with nothing. It acted either way, and acting is what is in
    question here. Agents that never submitted are absent from the dict.
    """
    extractions = summary.get("extractions") or {}
    return len(extractions)


def uniform_amounts(summaries: list[dict]) -> bool:
    """True when every recorded extraction, in every round, is the same amount.

    This is the signature of an amount that never reached the environment: the
    instruction illustrated it with a placeholder, agents repeated the
    placeholder, and the environment clamped each request to its fallback of 1.
    Full participation with one repeated number looks healthy and is not.

    A single recorded extraction is not evidence of anything, so it takes at
    least two before the shape counts as a signature.
    """
    amounts = [
        amount
        for summary in summaries
        for amount in (summary.get("extractions") or {}).values()
    ]
    return len(amounts) > 1 and len(set(amounts)) == 1


def unknown_participants(summaries: list[dict], names: set[str]) -> set[str]:
    """Extraction keys that belong to no agent in this run.

    The environment keys its tables on the ``agent_name`` argument it is handed
    and creates whatever key it is given, so a request written as "Agent 3"
    instead of "Agent-3" is accepted, resolved and attributed to nobody. An
    agent did precisely that in the run this check was added for.
    """
    seen = {
        name for summary in summaries for name in (summary.get("extractions") or {})
    }
    return seen - names


def _agent_names(replay_dir: Path) -> set[str]:
    """Names of the agents in the run, from the profiles written at startup."""
    names = set()
    for shard in sorted(replay_dir.glob("core_agent_profile.*.jsonl")):
        with shard.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    names.add(json.loads(line).get("name"))
    return names


def _count_rows(replay_dir: Path, table: str) -> int:
    """Count the rows a table wrote across its shards.

    @arg replay_dir Directory holding the run's ``<table>.<shard>.jsonl`` files.
    @arg table Table name, without the shard suffix.
    """
    total = 0
    for shard in sorted(replay_dir.glob(f"{table}.*.jsonl")):
        with shard.open() as handle:
            total += sum(1 for line in handle if line.strip())
    return total


def _report_interaction(replay_dir: Path, min_participation: float) -> int:
    """Check that agents actually acted on each other, not merely that rows exist.

    A run can fill core_agent_profile, log a tidy sequence of observations and
    still be inert: every agent looked around and nobody did anything. Counting
    rows does not catch that, because the profile rows are written at startup.
    The environment state is where a decision leaves a trace.
    """
    states = []
    for shard in sorted(replay_dir.glob("*_env_state.*.jsonl")) or sorted(
        replay_dir.glob("*env_state*.jsonl")
    ):
        with shard.open(encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    states.append(json.loads(line))

    if not states:
        print("no environment state was recorded", file=sys.stderr)
        return 1

    states.sort(key=lambda row: row.get("step", 0))

    # Field names differ per environment; report whichever are present rather
    # than hardcoding one environment's schema.
    interesting = ("round_number", "current_pool_resources", "total_messages_sent")
    print()
    for row in states:
        shown = {k: row[k] for k in interesting if k in row}
        print(f"step {row.get('step')}: {shown}")

    rounds = max((row.get("round_number", 0) for row in states), default=0)
    messages = max((row.get("total_messages_sent", 0) for row in states), default=0)
    # SocialMediaSpace keeps no running counter in environment state: every
    # post, follow and like is a row in its own event table. Counting only
    # environment state would call a run in which all 128 agents posted an
    # empty simulation.
    events = _count_rows(replay_dir, "social_media_event")

    if rounds == 0 and messages == 0 and events == 0:
        print(
            "\nFAIL the simulation completed but no agent acted: no rounds were "
            "resolved, no messages were sent and no social media events were "
            "recorded. The plumbing works; the scenario did not get the agents "
            "to do anything.",
            file=sys.stderr,
        )
        return 1

    print(
        f"\ninteraction confirmed: {rounds} rounds, {messages} messages, "
        f"{events} social media events"
    )

    return _report_participation(replay_dir, states, min_participation)


def _report_participation(
    replay_dir: Path, states: list[dict], minimum: float
) -> int:
    """Check how many agents acted per round, not merely that rounds resolved.

    A round resolves as soon as one agent submits; the rest are recorded as
    having extracted nothing. Three rounds resolved by one agent each is a
    different simulation from three rounds the whole group took part in, and
    only this number tells them apart.
    """
    summaries = round_summaries(states)
    if not summaries:
        # Environments without rounds — SimpleSocialSpace — record nothing to
        # measure here, and the message count above already covered them.
        return 0

    names = _agent_names(replay_dir)
    agent_count = len(names)
    if agent_count < 1:
        print("no agent profiles were recorded", file=sys.stderr)
        return 1

    print(f"\nparticipation (agents that submitted, out of {agent_count}):")
    for summary in summaries:
        acted = participation(summary)
        extractions = summary.get("extractions") or {}
        print(
            f"  round {summary['round']}: {acted}/{agent_count}  "
            f"{json.dumps(extractions, sort_keys=True)}"
        )

    rate = sum(participation(s) for s in summaries) / (len(summaries) * agent_count)
    print(f"  mean {rate:.0%}")

    strangers = unknown_participants(summaries, names)
    if strangers:
        print(
            f"\nFAIL {sorted(strangers)} took from the pool and none of them is "
            f"an agent in this run ({sorted(names)}). The environment creates "
            "whatever key it is handed, so a misspelt name is resolved and "
            "attributed to nobody.",
            file=sys.stderr,
        )
        return 1

    if uniform_amounts(summaries):
        print(
            "\nWARNING every agent extracted the same amount in every round, "
            "which is what it looks like when the amount never reached the "
            "environment and each request was clamped to the fallback",
            file=sys.stderr,
        )

    if rate < minimum:
        print(
            f"\nFAIL mean participation {rate:.0%} is below the required "
            f"{minimum:.0%}. The rounds resolved, but most agents sat them out, "
            "so the run says little about what happens when everyone draws from "
            "the same pool.",
            file=sys.stderr,
        )
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
