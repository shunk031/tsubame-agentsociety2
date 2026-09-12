#!/usr/bin/env python3
"""Confirm that a finished run actually recorded agent activity.

agentsociety2 catches exceptions on several paths, so a run whose LLM calls all
failed can still exit 0. The replay files are the evidence that the simulation
did something, which makes them the honest success criterion.

Storage layout, per agentsociety2.storage.replay_sink: every process appends
JSONL to ``<run_dir>/replay/{table}.{shard:02x}.jsonl``, 256 shards guarded by
flock. There is no database, so counting lines is the whole job.

Reads RUN_DIR and MIN_REPLAY_RECORDS from the environment.
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

    return _report_interaction(replay_dir)


def _report_interaction(replay_dir: Path) -> int:
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

    if rounds == 0 and messages == 0:
        print(
            "\nFAIL the simulation completed but no agent acted: no rounds were "
            "resolved and no messages were sent. The plumbing works; the "
            "scenario did not get the agents to do anything.",
            file=sys.stderr,
        )
        return 1

    print(f"\ninteraction confirmed: {rounds} rounds, {messages} messages")
    return 0


if __name__ == "__main__":
    sys.exit(main())
