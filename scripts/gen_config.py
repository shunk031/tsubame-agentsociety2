#!/usr/bin/env python3
"""Generate the ``init_config.json`` and ``steps.yaml`` pair a run needs.

AgentSociety 2 has no bulk agent count in its schema the way v1 did: ``agents``
is a list and every participant is spelled out. Anything beyond a toy run is
therefore expected to be generated, which is what this script does.

The steps alternate a short ``run`` with a ``questionnaire``, following the only
upstream example that still runs, the ``daily_mobility`` one. The questionnaire
is the measurement instrument; without one a run produces agent profiles and
little else.

There are no ``ask`` or ``intervene`` steps, and that is deliberate — see
``build_steps``. Everything the agents need is in their profile, which is why
``role`` below carries the rules and the procedure rather than a description.

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
# many agents it expects, and the standing orders its agents are given.
#
# `role` does the work that a briefing or a per-round instruction would do
# elsewhere. During a `run` step there is no <question>, so nothing external
# reaches the agent — the prompt is built from the profile, the automatic
# observation and memory, and the profile is the only part this script controls.
# It therefore has to carry the rules of the game and the action to take, not a
# description of the situation.
#
# The wording of the action follows the contract in ask_env's own tool schema:
# "write `instruction` as a reusable template with stable wording, and put
# changing runtime values in `variables` instead of embedding them directly in
# the text". The code-generating router is told the same thing from the other
# side — it is instructed to read changing values from ctx['variables'] rather
# than hard-code literals — so the instruction has to name the keys it expects.
#
# Writing the amount into the sentence instead works, but only by accident: it
# is a different instruction every round, so the template cache can never match
# it, and it puts the one value that must survive into the part of the call the
# router has to parse out of prose. Upstream's own CommonsTragedyAgent passes
# agent_name and requested_extraction through ctx['variables']; this is the same
# call, made from a ReAct loop instead of from Python.
#
# The environment tool is deliberately not named. An instruction that says
# "call submit_extraction" makes the agent attempt it as a ReAct action and
# fail with "Unknown tool" — cause 3 in PR #1.
#
# Nor is any forbidden form exhibited. An earlier version of this text listed
# the placeholders not to write, and the model copied one of them out of the
# prohibition into a live request. Say what to send, not what to avoid.
#
# Formatted with `n` (the number of *other* agents), `pool` and `max_extract`.
ENVIRONMENTS = {
    "CommonsTragedyEnv": {
        "default_agents": 4,
        "role": (
            "You are a participant in a shared-resource experiment, and it is "
            "the only thing you do: no job, no home, no errands. You share a "
            "common pool of {pool} resource units with {n} others. Each round "
            "you may take between 1 and {max_extract} units, every unit you "
            "take is worth one point to you, and the pool never refills. If "
            "everyone's requests together exceed what is left, the remainder "
            "is split in proportion to what each of you asked for. So on every "
            "step, before anything else: choose a whole number between 1 and "
            "{max_extract}, then use ask_env exactly once. Send this "
            "instruction word for word, the same every round: \"submit an "
            "extraction from the shared pool using agent_name and "
            "requested_extraction from ctx['variables']\". Your two values go "
            "in the ask_env variables argument, not into that sentence: "
            "agent_name is your own name, copied exactly as it appears in your "
            "profile, and requested_extraction is the whole number you chose."
        ),
        "env_kwargs": lambda n, args: {
            "num_agents": n,
            "initial_pool_resources": args.pool_resources,
            "max_extraction_per_agent": args.max_extraction,
        },
        # Answering 0 has to be allowed. An agent that took nothing otherwise
        # has no answer to give, and tries to extract during the questionnaire
        # instead — which is read-only, so it fails with "ask_env mutation is
        # disabled in readonly mode". Observed for two agents in the last run.
        "question": (
            "In round {round} of the shared-resource game, how many units did "
            "you request from the pool? Answer with the number only, and "
            "answer 0 if you did not request any."
        ),
        "response_type": "integer",
    },
    # The simultaneous-move games above and the psychology experiments in the
    # same contrib package share a property that makes them poor fits for a
    # large population: no tool takes another agent's id, so agents never
    # address one another, and the round resolves only once every agent has
    # submitted. Adding agents changes the arithmetic, not the structure, and
    # the round costs whatever the slowest single agent costs.
    #
    # SocialMediaSpace is the opposite on both counts. Nine of its eleven tools
    # take another agent's id, and nothing waits for a round boundary, so the
    # population is the environment rather than a divisor.
    "SocialMediaSpace": {
        "default_agents": 32,
        "role": (
            "You are one of {n} others on a social network, and it is the only "
            "thing you do: no job, no home, no errands. On every step, before "
            "anything else, use ask_env exactly twice, in this order. First, "
            "send this instruction word for word: \"refresh the feed using "
            "user_id from ctx['variables']\", with user_id set to your own id. "
            "Read what came back. Then send this instruction word for word: "
            "\"create a post using author_id and content from "
            "ctx['variables']\", with author_id set to your own id and content "
            "set to what you want to say. Write content that replies to "
            "something you just read, naming the person you are replying to, "
            "unless the feed came back empty — then write whatever is on your "
            "mind. Your id is the number in your profile, copied exactly as it "
            "appears there."
        ),
        # The environment reads feed_source and polarization_mode from kwargs;
        # the defaults ("global", "none") are the neutral setting, so a first
        # study measures interaction rather than an imposed structure.
        "env_kwargs": lambda n, args: {},
        "question": (
            "How many posts have you written so far? Answer with the number "
            "only, and answer 0 if you have written none."
        ),
        "response_type": "integer",
    },
    "SimpleSocialSpace": {
        "default_agents": 8,
        "role": (
            "You are a participant in a group conversation experiment, and it "
            "is the only thing you do. You are in a group with {n} others. On "
            "every step, before anything else: use ask_env exactly once. Send "
            "this instruction word for word, the same every step: \"send a "
            "message using sender_id, receiver_id and content from "
            "ctx['variables']\". Your three values go in the ask_env variables "
            "argument, not into that sentence: sender_id is your own id, "
            "receiver_id is the id of the person you are writing to, and "
            "content is what you want to say to them."
        ),
        "env_kwargs": lambda n, args: {},
        "question": (
            "How many messages have you sent so far? Answer with the number "
            "only, and answer 0 if you have sent none."
        ),
        "response_type": "integer",
    },
}


def build_agents(num_agents: int, seed: int, role: str) -> list[dict]:
    """Return ``num_agents`` PersonAgent entries with stable pseudo-random traits.

    Names follow ``Agent-{id}``. That is not cosmetic: the environment tools
    take an ``agent_name`` argument and their docstrings specify this format, so
    an agent that calls itself "Alice" — or "Agent 3" — writes into a key
    nothing else reads.

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
                # A ReAct turn spent on a failed call must not exhaust the
                # budget before the agent reaches ask_env.
                "max_react_turns": 12,
            },
        }
        for index in range(num_agents)
    ]


def build_init_config(agents: list[dict], env_module: str, env_kwargs: dict) -> dict:
    """Wrap agents in the chosen environment module."""
    # Both environments key on an explicit agent-to-identity mapping. Without
    # one SocialMediaSpace auto-creates a user for whatever id it is handed, so
    # an id the model invented becomes a silent extra participant; with one, the
    # same id raises instead.
    if env_module in ("SimpleSocialSpace", "SocialMediaSpace"):
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
    """One round is one ``run`` step followed by the questionnaire measuring it.

    There is deliberately no ``ask`` and no ``intervene`` step, though both look
    like the natural way to brief agents and to nudge them each round. Neither
    is delivered to the agents. ``AskStep`` and ``InterveneStep`` carry no
    recipient list — only ``QuestionnaireStep`` has ``target_agent_ids`` — so
    the CLI hands their text to ``AgentSocietyHelper``, a plan-and-execute loop
    running on the same model the agents do, and lets it decide what to do with
    it. Three things follow, all of them seen in the logs of the run that
    prompted this:

    - It rarely chose ``ask_agents``, the only tool that reaches an agent, and
      it has no way of knowing how many agents exist. Nothing arrived.
    - It submitted extractions itself through ``ask_environment``, inventing an
      ``agent_name`` as it went. Rounds 1 and 2 recorded an extraction for
      Agent-1 that Agent-1 never made: a fifth participant, which is worse than
      a missing one.
    - Its own planning prompt taught it to fail. It fabricated a ``think`` tool
      and copied the ``{"param1": "value1"}`` example out of that prompt into a
      call that takes no arguments.

    A ``run`` step has none of these properties: ``AgentSociety.step`` fans every
    agent id out to ``step_agent_batch`` with no filtering, the environment's
    observe-kind tools are called automatically so each agent already sees the
    pool, and the ReAct loop runs with mutation allowed. Participation becomes
    N independent agent loops instead of one planner's guess.

    (An intervene would not have survived to the run step in any case. The
    instruction is delivered through ``agent.ask``, which never calls
    ``memory_runtime.after_step`` — only ``step`` does — so nothing the agent
    decided there is still in context when the round is resolved.)
    """
    agent_ids = list(range(1, num_agents + 1))
    steps: list[dict] = []

    for round_number in range(1, args.num_rounds + 1):
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


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
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
    args = parser.parse_args(argv)

    if (args.num_agents or ENVIRONMENTS[args.env_module]["default_agents"]) < 1:
        parser.error("--num-agents must be at least 1")
    if args.num_rounds < 1:
        parser.error("--num-rounds must be at least 1")

    return args


def build_role(spec: dict, num_agents: int, args) -> str:
    """Render the standing orders written into every agent's personality."""
    return spec["role"].format(
        n=num_agents - 1,
        pool=args.pool_resources,
        max_extract=args.max_extraction,
    )


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    spec = ENVIRONMENTS[args.env_module]
    num_agents = args.num_agents or spec["default_agents"]

    args.out_dir.mkdir(parents=True, exist_ok=True)

    agents = build_agents(num_agents, args.seed, build_role(spec, num_agents, args))
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
