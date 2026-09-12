"""Checks on the configuration ``gen_config.py`` emits.

Almost everything here is an assertion about wording, which is unusual. It is
because wording is the interface: during a ``run`` step an agent is prompted
from its profile, an automatic observation and its memory, and the profile is
the only one of those this repository controls. There is no API to call and no
schema that would catch a regression in it.

The other thing pinned here is the absence of ``ask`` and ``intervene`` steps.
They are not omitted by oversight — see ``build_steps`` — and a well-meaning
edit that adds one back is exactly the regression these tests exist to catch.
"""

from __future__ import annotations

import json

import gen_config
import pytest


def make_args(tmp_path, **overrides):
    """Parse a default argument set, with overrides applied afterwards."""
    args = gen_config.parse_args(["--out-dir", str(tmp_path)])
    for key, value in overrides.items():
        setattr(args, key, value)
    return args


def role_for(env_module="CommonsTragedyEnv", num_agents=4, **overrides):
    args = make_args(overrides.pop("tmp_path", "/tmp"), env_module=env_module, **overrides)
    return gen_config.build_role(gen_config.ENVIRONMENTS[env_module], num_agents, args)


# --- the steps ---------------------------------------------------------------


def test_a_round_is_a_run_then_a_questionnaire(tmp_path):
    args = make_args(tmp_path, num_rounds=3)
    steps = gen_config.build_steps(args, gen_config.ENVIRONMENTS[args.env_module], 4)

    assert [step["type"] for step in steps["steps"]] == [
        "run",
        "questionnaire",
        "run",
        "questionnaire",
        "run",
        "questionnaire",
    ]


def test_no_step_is_routed_through_the_helper(tmp_path):
    """`ask` and `intervene` go to a planner, not to the agents.

    In the run this replaced, the planner submitted extractions itself through
    `ask_environment` under a name it invented, recording an extraction for
    Agent-1 that Agent-1 never made. A phantom participant is worse than a
    missing one, so neither step type may reappear.
    """
    for num_rounds in (1, 4):
        args = make_args(tmp_path, num_rounds=num_rounds)
        steps = gen_config.build_steps(args, gen_config.ENVIRONMENTS[args.env_module], 4)
        types = {step["type"] for step in steps["steps"]}

        assert "ask" not in types
        assert "intervene" not in types


def test_the_questionnaire_reaches_every_agent(tmp_path):
    """Unlike ask and intervene, this step takes an explicit recipient list."""
    args = make_args(tmp_path)
    steps = gen_config.build_steps(args, gen_config.ENVIRONMENTS[args.env_module], 4)

    for step in steps["steps"]:
        if step["type"] == "questionnaire":
            assert step["target_agent_ids"] == [1, 2, 3, 4]


def test_the_questionnaire_accepts_having_done_nothing():
    """An agent with no answer tries to act instead, during a read-only step.

    Two agents did exactly that in the last run and failed with "ask_env
    mutation is disabled in readonly mode".
    """
    question = gen_config.ENVIRONMENTS["CommonsTragedyEnv"]["question"].format(round=1)
    assert "0" in question


# --- the role, which is the whole interface during a run step ---------------


def test_role_carries_the_rules_of_the_game():
    """No briefing step survives, so the rules have to live in the profile."""
    role = role_for(pool_resources=100, max_extraction=10)

    assert "100" in role  # the pool
    assert "3 others" in role  # four agents, three of them other people
    assert "1 and 10" in role  # the per-round bounds
    assert "never refills" in role
    assert "proportion" in role  # what happens when the pool runs short


def test_role_carries_the_action():
    role = role_for()

    assert "ask_env" in role
    assert "every step" in role
    assert any(character.isdigit() for character in role)


def test_role_sends_values_through_variables():
    """ask_env's own schema asks for a stable instruction plus `variables`.

    The router is told from the other side to read changing values from
    ctx['variables'] rather than hard-code literals, so the instruction has to
    name the keys it expects to find there.
    """
    role = role_for()

    assert "ctx['variables']" in role
    assert "agent_name" in role
    assert "requested_extraction" in role
    assert "word for word" in role


@pytest.mark.parametrize("env_module", sorted(gen_config.ENVIRONMENTS))
def test_role_exhibits_no_placeholder_to_copy(env_module):
    """Naming a forbidden form hands the model the token to emit.

    An earlier version listed `{amount}` and `N` as placeholders not to write.
    The next run produced `submit an extraction of N units for Agent-1` — the
    model lifted the example straight out of the prohibition. Say what to send.
    """
    role = role_for(env_module, num_agents=4)

    assert "{" not in role.replace("ctx['variables']", "")
    assert "placeholder" not in role


def test_role_tracks_the_scenario_parameters():
    role = role_for(pool_resources=250, max_extraction=6)

    assert "250" in role
    assert "1 and 6" in role


def test_role_scales_with_the_agent_count():
    assert "15 others" in role_for(num_agents=16)


def test_role_asks_for_the_name_the_environment_keys_on():
    """The environment tools create whatever agent_name key they are handed.

    An agent that writes "Agent 3" instead of "Agent-3" depletes the pool and
    credits nobody. A run produced exactly that, so the name is not to be
    retyped from memory.
    """
    assert "copied exactly as it appears in your profile" in role_for()


def test_role_never_names_an_environment_tool():
    """Naming one makes the agent try it as a ReAct action and give up.

    "Unknown tool: submit_extraction" — cause 3 in PR #1. The environment is
    reached through ask_env in plain words.
    """
    for env_module in gen_config.ENVIRONMENTS:
        role = role_for(env_module, num_agents=4)
        for tool in ("submit_extraction", "get_pool_resources", "get_round_history"):
            assert tool not in role


def test_role_reaches_every_agent_profile():
    role = role_for()
    agents = gen_config.build_agents(4, seed=42, role=role)

    assert len(agents) == 4
    for index, agent in enumerate(agents, 1):
        assert agent["agent_id"] == index
        assert agent["kwargs"]["id"] == index
        assert agent["kwargs"]["name"] == f"Agent-{index}"
        assert agent["kwargs"]["personality"].startswith(role)


# --- the files as a whole ---------------------------------------------------


def test_generated_pair_is_valid_and_consistent(tmp_path):
    gen_config.main(["--out-dir", str(tmp_path), "--num-agents", "4", "--num-rounds", "2"])

    init_config = json.loads((tmp_path / "init_config.json").read_text())
    steps = json.loads((tmp_path / "steps.yaml").read_text())

    assert len(init_config["agents"]) == 4
    assert init_config["env_modules"][0]["module_type"] == "CommonsTragedyEnv"
    assert init_config["env_modules"][0]["kwargs"]["num_agents"] == 4
    assert len(steps["steps"]) == 4


def test_simple_social_space_still_generates(tmp_path):
    """The other environment stays usable through ENV_MODULE."""
    gen_config.main(
        ["--out-dir", str(tmp_path), "--env-module", "SimpleSocialSpace", "--num-rounds", "1"]
    )

    init_config = json.loads((tmp_path / "init_config.json").read_text())
    steps = json.loads((tmp_path / "steps.yaml").read_text())

    assert init_config["env_modules"][0]["module_type"] == "SimpleSocialSpace"
    assert len(init_config["env_modules"][0]["kwargs"]["agent_id_name_pairs"]) == 8
    assert [step["type"] for step in steps["steps"]] == ["run", "questionnaire"]

    # Its standing order has to be operative too, for the same reason.
    personality = init_config["agents"][0]["kwargs"]["personality"]
    assert "ask_env" in personality
    assert "every step" in personality


@pytest.mark.parametrize("bad", [["--num-rounds", "0"], ["--num-agents", "-1"]])
def test_nonsense_counts_are_rejected(tmp_path, bad):
    with pytest.raises(SystemExit):
        gen_config.parse_args(["--out-dir", str(tmp_path), *bad])
