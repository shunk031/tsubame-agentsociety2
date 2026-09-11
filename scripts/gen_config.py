#!/usr/bin/env python3
"""Generate the ``init_config.json`` and ``steps.yaml`` pair a run needs.

AgentSociety 2 has no bulk agent count in its schema the way v1 did: ``agents``
is a list and every participant is spelled out. Anything beyond a toy run is
therefore expected to be generated, which is what this script does.

The schema is enforced by the pydantic models in ``agentsociety2.society.models``
and is easy to get subtly wrong, so the notable constraints are:

- ``env_modules`` and ``agents`` both need at least one entry.
- Every agent's ``kwargs`` must carry ``id``, matching its ``agent_id``.
- All agents must share one ``agent_type``.
- ``max_react_turns`` belongs to the agent config, while ``name``, ``age`` and
  ``personality`` are profile fields; the CLI splits them apart by key.
- The step type for advancing time is ``run``. (The upstream docs table in
  ``docs/cli.rst`` calls it ``step``, which the models reject.)
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

# Kept deliberately plain: these feed an LLM prompt, and florid personas make it
# harder to tell a model problem from a pipeline problem during a smoke run.
FIRST_NAMES = [
    "Alice", "Bob", "Carol", "Dave", "Erin", "Frank", "Grace", "Heidi",
    "Ivan", "Judy", "Karl", "Laura", "Mallory", "Niaj", "Olivia", "Peggy",
    "Quentin", "Rupert", "Sybil", "Trent", "Uma", "Victor", "Wendy", "Xavier",
]

PERSONALITIES = [
    "friendly and curious",
    "skeptical and analytical",
    "cautious and detail-oriented",
    "outgoing and persuasive",
    "quiet and observant",
    "pragmatic and direct",
]


def build_agents(num_agents: int, seed: int) -> list[dict]:
    """Return ``num_agents`` PersonAgent entries with stable pseudo-random traits."""
    rng = random.Random(seed)
    agents = []

    for index in range(num_agents):
        agent_id = index + 1
        base = FIRST_NAMES[index % len(FIRST_NAMES)]
        # Names repeat once the list is exhausted; a suffix keeps them distinct.
        suffix = index // len(FIRST_NAMES)
        name = base if suffix == 0 else f"{base}{suffix + 1}"

        agents.append(
            {
                "agent_id": agent_id,
                "agent_type": "PersonAgent",
                "kwargs": {
                    "id": agent_id,
                    "name": name,
                    "age": rng.randint(20, 65),
                    "personality": rng.choice(PERSONALITIES),
                    "max_react_turns": 4,
                },
            }
        )

    return agents


def build_init_config(agents: list[dict]) -> dict:
    """Wrap agents in a SimpleSocialSpace environment.

    SimpleSocialSpace is the only environment that needs no external data. The
    alternative, MobilitySpace, pulls a map and a routing binary from servers in
    China and cannot run offline without staging both first.
    """
    pairs = [[agent["agent_id"], agent["kwargs"]["name"]] for agent in agents]

    return {
        "env_modules": [
            {
                "module_type": "SimpleSocialSpace",
                "kwargs": {"agent_id_name_pairs": pairs},
            }
        ],
        "agents": agents,
    }


def build_steps(num_steps: int, tick: int, question: str, start_t: str) -> dict:
    """Build a steps document that asks one question then advances time."""
    return {
        "start_t": start_t,
        "steps": [
            {"type": "ask", "question": question},
            {"type": "run", "num_steps": num_steps, "tick": tick},
        ],
    }


def render_steps_yaml(steps: dict) -> str:
    """Serialise the steps document without requiring PyYAML.

    Only two step shapes are emitted, so hand-rolling the YAML avoids adding a
    dependency to whichever interpreter happens to run this script.
    """
    lines = [f'start_t: "{steps["start_t"]}"', "steps:"]

    for step in steps["steps"]:
        if step["type"] == "ask":
            question = step["question"].replace('"', '\\"')
            lines.append(f'  - {{type: ask, question: "{question}"}}')
        elif step["type"] == "run":
            lines.append(
                f'  - {{type: run, num_steps: {step["num_steps"]}, '
                f'tick: {step["tick"]}}}'
            )
        else:
            raise ValueError(f"unsupported step type: {step['type']}")

    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, required=True,
                        help="Directory to write init_config.json and steps.yaml into")
    parser.add_argument("--num-agents", type=int, default=8)
    parser.add_argument("--num-steps", type=int, default=2)
    parser.add_argument("--tick", type=int, default=3600,
                        help="Seconds of simulated time per step")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--start-t", default="2026-01-01T09:00:00")
    parser.add_argument("--question",
                        default="Introduce yourself to the group, then say what "
                                "you hope to get out of this conversation.")
    args = parser.parse_args()

    if args.num_agents < 1:
        parser.error("--num-agents must be at least 1")
    if args.num_steps < 1:
        parser.error("--num-steps must be at least 1")

    args.out_dir.mkdir(parents=True, exist_ok=True)

    agents = build_agents(args.num_agents, args.seed)
    init_config = build_init_config(agents)
    steps = build_steps(args.num_steps, args.tick, args.question, args.start_t)

    config_path = args.out_dir / "init_config.json"
    steps_path = args.out_dir / "steps.yaml"

    config_path.write_text(json.dumps(init_config, indent=2) + "\n")
    steps_path.write_text(render_steps_yaml(steps))

    print(f"wrote {config_path} ({len(agents)} agents)")
    print(f"wrote {steps_path} ({args.num_steps} steps of {args.tick}s)")


if __name__ == "__main__":
    main()
