#!/usr/bin/env python3
"""Generate the ``init_config.json`` and ``steps.yaml`` pair a run needs.

AgentSociety 2 has no bulk agent count in its schema the way v1 did: ``agents``
is a list and every participant is spelled out. Anything beyond a toy run is
therefore expected to be generated, which is what this script does.

The step layout follows the only upstream example that still runs, the
``daily_mobility`` one: alternate a short ``run`` with a ``questionnaire``
rather than running a long stretch and hoping something was recorded. The
questionnaire is the measurement instrument; without one a run produces agent
profiles and little else.

Schema constraints worth knowing, all enforced by the pydantic models in
``agentsociety2.society.models``:

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

PERSONALITIES = [
    "cautious and cooperative, dislikes waste",
    "competitive and opportunistic, wants to come out ahead",
    "analytical, reasons about long-term consequences",
    "impulsive, acts on what feels right in the moment",
    "fair-minded, watches what others take",
    "anxious about scarcity, wants a safety margin",
]

# Each entry describes one environment: the kwargs it is constructed with, how
# many agents it expects, and the briefing the agents are given up front. The
# briefing matters more than it looks — agents that are not told the rules have
# nothing to act on, and the run degenerates into observation with no decisions.
ENVIRONMENTS = {
    "CommonsTragedyEnv": {
        "default_agents": 4,
        "role": (
            "You are a participant in a shared-resource experiment. Your only "
            "activity is deciding how much to extract from a common pool each "
            "round; you have no job, home or errands to attend to."
        ),
        "env_kwargs": lambda n, args: {
            "num_agents": n,
            "initial_pool_resources": args.pool_resources,
            "max_extraction_per_agent": args.max_extraction,
        },
        "briefing": (
            "You share a common pool of {pool} resource units with {n} others. "
            "Each round you may extract between 1 and {max_extract} units, and "
            "each unit you take is worth one point to you. The pool does not "
            "refill. If everyone's requests together exceed what is left, the "
            "remainder is split in proportion to what each of you asked for. "
            "Decide how much to extract this round by calling submit_extraction "
            "with your own agent name, and say briefly why you chose that amount."
        ),
        # Delivered before every run step. The opening briefing alone does not
        # survive: left to themselves the agents invent unrelated daily lives
        # and never touch the pool, so the instruction is repeated each round.
        #
        # It asks for the action in plain words rather than naming the tool.
        # Agents reach the environment through ask_env, which generates the call
        # for them; an instruction to "call submit_extraction" makes the agent
        # try it as a ReAct action and fail with "Unknown tool".
        "instruction": (
            "It is now round {round}. Decide how many units to extract from the "
            "shared pool, between 1 and {max_extract}. Then use ask_env to tell "
            "the environment your decision, phrased as: submit an extraction of "
            "N units for me. Do this before anything else, and do it only once."
        ),
        "question": (
            "In round {round} of the shared-resource game, how many units did you "
            "request from the pool? Answer with the number only."
        ),
        "response_type": "integer",
    },
    "SimpleSocialSpace": {
        "default_agents": 8,
        "role": (
            "You are a participant in a group conversation experiment. Your only "
            "activity is talking with the others."
        ),
        "env_kwargs": lambda n, args: {},
        "briefing": (
            "You are in a group with {n} other people. Introduce yourself and "
            "talk to them."
        ),
        "instruction": (
            "Send a message to at least one other person in the group now."
        ),
        "question": "How many messages have you sent so far? Answer with the number only.",
        "response_type": "integer",
    },
}


def build_agents(num_agents: int, seed: int, role: str) -> list[dict]:
    """Return ``num_agents`` PersonAgent entries with stable pseudo-random traits.

    Names follow ``Agent-{id}``. That is not cosmetic: the environment tools
    take an ``agent_name`` argument and their docstrings specify this format, so
    an agent that calls itself "Alice" writes into a key nothing else reads.

    The role is prepended to each personality because a bare trait leaves the
    agent unanchored: given only "competitive and opportunistic" it will invent
    a day job and spend the simulation attending to that instead of the
    scenario. The profile is the one piece of text that stays in context.
    """
    rng = random.Random(seed)

    return [
        {
            "agent_id": index + 1,
            "agent_type": "PersonAgent",
            "kwargs": {
                "id": index + 1,
                "name": f"Agent-{index + 1}",
                "age": rng.randint(20, 65),
                "personality": f"{role} Your disposition: {rng.choice(PERSONALITIES)}.",
                "max_react_turns": 8,
            },
        }
        for index in range(num_agents)
    ]


def build_init_config(agents: list[dict], env_module: str, env_kwargs: dict) -> dict:
    """Wrap agents in the chosen environment module."""
    if env_module == "SimpleSocialSpace":
        env_kwargs = {
            "agent_id_name_pairs": [
                [agent["agent_id"], agent["kwargs"]["name"]] for agent in agents
            ]
        }

    return {
        "env_modules": [{"module_type": env_module, "kwargs": env_kwargs}],
        "agents": agents,
    }


def build_steps(args, spec: dict, num_agents: int) -> dict:
    """Alternate a short run with a questionnaire, once per round."""
    agent_ids = list(range(1, num_agents + 1))

    briefing = spec["briefing"].format(
        n=num_agents - 1, pool=args.pool_resources, max_extract=args.max_extraction
    )

    steps: list[dict] = [{"type": "ask", "question": briefing}]

    for round_number in range(1, args.num_rounds + 1):
        steps.append(
            {
                "type": "intervene",
                "instruction": spec["instruction"].format(
                    round=round_number, max_extract=args.max_extraction
                ),
            }
        )
        steps.append({"type": "run", "num_steps": 1, "tick": args.tick})
        steps.append(
            {
                "type": "questionnaire",
                "questionnaire_id": f"round_{round_number}",
                "title": f"Round {round_number}",
                "description": "",
                "questions": [
                    {
                        "id": "amount",
                        "prompt": spec["question"].format(round=round_number),
                        "response_type": spec["response_type"],
                    }
                ],
                "target_agent_ids": agent_ids,
            }
        )

    return {"start_t": args.start_t, "steps": steps}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, required=True,
                        help="Directory to write init_config.json and steps.yaml into")
    parser.add_argument("--env-module", default="CommonsTragedyEnv",
                        choices=sorted(ENVIRONMENTS))
    parser.add_argument("--num-agents", type=int, default=0,
                        help="Defaults to whatever the environment expects")
    parser.add_argument("--num-rounds", type=int, default=4,
                        help="Each round is one run step plus one questionnaire")
    parser.add_argument("--tick", type=int, default=900,
                        help="Seconds of simulated time per run step")
    parser.add_argument("--pool-resources", type=int, default=100)
    parser.add_argument("--max-extraction", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--start-t", default="2026-01-01T09:00:00")
    args = parser.parse_args()

    spec = ENVIRONMENTS[args.env_module]
    num_agents = args.num_agents or spec["default_agents"]

    if num_agents < 1:
        parser.error("--num-agents must be at least 1")
    if args.num_rounds < 1:
        parser.error("--num-rounds must be at least 1")

    args.out_dir.mkdir(parents=True, exist_ok=True)

    agents = build_agents(num_agents, args.seed, spec["role"])
    init_config = build_init_config(
        agents, args.env_module, spec["env_kwargs"](num_agents, args)
    )
    steps = build_steps(args, spec, num_agents)

    config_path = args.out_dir / "init_config.json"
    steps_path = args.out_dir / "steps.yaml"

    config_path.write_text(json.dumps(init_config, indent=2) + "\n")
    steps_path.write_text(json.dumps(steps, indent=2) + "\n")

    print(f"wrote {config_path} ({args.env_module}, {num_agents} agents)")
    print(f"wrote {steps_path} ({args.num_rounds} rounds of {args.tick}s)")


if __name__ == "__main__":
    main()
