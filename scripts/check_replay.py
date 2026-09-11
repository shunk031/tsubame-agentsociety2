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

    return 0


if __name__ == "__main__":
    sys.exit(main())
