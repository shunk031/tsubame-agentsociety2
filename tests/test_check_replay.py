"""Checks on the run verdict ``check_replay.py`` reaches.

The script is the repository's only honest success criterion — agentsociety2
swallows exceptions on several paths, so a run whose LLM calls all failed still
exits 0. These tests cover the part that decides whether a run counts: how many
agents actually acted each round.

The awkward detail the fixtures encode is that ``last_round`` repeats on steps
where no round resolved, and that agents which did not submit are absent from
``extractions`` rather than present with a zero.
"""

from __future__ import annotations

import json

import check_replay
import pytest


def env_state_row(step, round_number, pool, last_round):
    """One row as the replay sink writes it.

    JSON columns arrive as JSON sub-strings: replay_sink._normalize runs
    json.dumps over every dict or list before the line is written.
    """
    return {
        "step": step,
        "t": "2026-01-01T09:00:00",
        "round_number": round_number,
        "current_pool_resources": pool,
        "last_round": json.dumps(last_round) if last_round is not None else None,
        "pending_extractions": json.dumps({}),
        "submitted_agents": json.dumps([]),
        "initial_pool_resources": 100,
        "max_extraction_per_agent": 10,
    }


def summary(round_number, pool_before, extractions):
    pool_after = pool_before - sum(extractions.values())
    return {
        "round": round_number,
        "pool_before_round": pool_before,
        "extractions": extractions,
        "pool_after_round": pool_after,
        "payoffs": dict(extractions),
    }


def write_run(tmp_path, rows, agent_count=4):
    replay = tmp_path / "replay"
    replay.mkdir(parents=True, exist_ok=True)

    with (replay / "commons_tragedy_env_state.00.jsonl").open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")

    with (replay / "core_agent_profile.00.jsonl").open("w") as handle:
        for agent_id in range(1, agent_count + 1):
            handle.write(
                json.dumps(
                    {
                        "id": agent_id,
                        "name": f"Agent-{agent_id}",
                        "profile": json.dumps({"name": f"Agent-{agent_id}"}),
                        "created_at": "2026-01-01T09:00:00",
                    }
                )
                + "\n"
            )
    return tmp_path


# --- reading the rounds back out --------------------------------------------


def test_round_summaries_drop_the_repeats(tmp_path):
    """last_round is rewritten unchanged on steps where nothing resolved."""
    first = summary(1, 100, {"Agent-1": 5, "Agent-4": 5})
    rows = [
        env_state_row(0, 0, 100, None),
        env_state_row(1, 1, 90, first),
        env_state_row(2, 1, 90, first),
        env_state_row(3, 2, 85, summary(2, 90, {"Agent-1": 5})),
    ]

    summaries = check_replay.round_summaries(
        [json.loads(line) for line in _lines(write_run(tmp_path, rows))]
    )

    assert [item["round"] for item in summaries] == [1, 2]


def test_participation_counts_submitters_not_takers():
    """An agent that submitted but got nothing still participated.

    The environment splits a short pool proportionally and rounds down, so a
    submitter can end a round with zero units. It still acted.
    """
    assert check_replay.participation(summary(1, 100, {"Agent-1": 5, "Agent-4": 5})) == 2
    assert check_replay.participation(summary(2, 4, {"Agent-1": 0, "Agent-2": 4})) == 2
    assert check_replay.participation({"round": 3}) == 0


# --- the verdict ------------------------------------------------------------


def test_full_participation_passes(tmp_path, monkeypatch):
    rows = [
        env_state_row(0, 0, 100, None),
        env_state_row(
            1, 1, 78, summary(1, 100, {"Agent-1": 7, "Agent-2": 3, "Agent-3": 9, "Agent-4": 3})
        ),
        env_state_row(
            2, 2, 60, summary(2, 78, {"Agent-1": 6, "Agent-2": 2, "Agent-3": 8, "Agent-4": 2})
        ),
    ]
    monkeypatch.setenv("RUN_DIR", str(write_run(tmp_path, rows)))
    monkeypatch.setenv("MIN_PARTICIPATION", "1.0")

    assert check_replay.main() == 0


def test_the_run_that_opened_issue_2_fails(tmp_path, monkeypatch):
    """Four agents, one or two acting: exit 0 is not success."""
    rows = [
        env_state_row(0, 0, 100, None),
        env_state_row(1, 1, 90, summary(1, 100, {"Agent-1": 5, "Agent-4": 5})),
        env_state_row(2, 2, 85, summary(2, 90, {"Agent-1": 5})),
        env_state_row(3, 3, 84, summary(3, 85, {"Agent-3": 1})),
    ]
    monkeypatch.setenv("RUN_DIR", str(write_run(tmp_path, rows)))
    monkeypatch.setenv("MIN_PARTICIPATION", "0.5")

    assert check_replay.main() == 1


def test_participation_is_not_checked_when_no_round_carries_one(tmp_path, monkeypatch):
    """SimpleSocialSpace records messages, not rounds; it must still pass."""
    rows = [
        {
            "step": 0,
            "t": "2026-01-01T09:00:00",
            "round_number": 0,
            "total_messages_sent": 6,
        }
    ]
    monkeypatch.setenv("RUN_DIR", str(write_run(tmp_path, rows)))

    assert check_replay.main() == 0


# --- the failure that hides behind a healthy-looking run --------------------


@pytest.mark.parametrize(
    ("extractions", "uniform"),
    [
        ({"Agent-1": 1, "Agent-2": 1, "Agent-3": 1, "Agent-4": 1}, True),
        ({"Agent-1": 7, "Agent-2": 7, "Agent-3": 7, "Agent-4": 7}, True),
        ({"Agent-1": 7, "Agent-2": 3, "Agent-3": 7, "Agent-4": 7}, False),
    ],
    ids=["clamped-to-one", "same-number-everywhere", "varies"],
)
def test_uniform_amounts_detects_the_clamped_request(extractions, uniform):
    """Every agent extracting the identical amount is the placeholder signature.

    PR #1: the instruction illustrated the amount with a placeholder, agents
    repeated it verbatim, the environment could not read it as an integer and
    clamped every request to its fallback of 1. Full participation with one
    repeated number looks healthy and is not.
    """
    summaries = [summary(1, 100, extractions), summary(2, 60, extractions)]
    assert check_replay.uniform_amounts(summaries) is uniform


def test_a_name_no_agent_answers_to_is_rejected(tmp_path, monkeypatch):
    """"Agent 3" is accepted by the environment and read by nothing.

    The tools key on the `agent_name` they are handed and create the key on
    demand, so a hyphen dropped on the way through the code-generating router
    produces a round that resolves, depletes the pool and credits nobody.
    """
    rows = [
        env_state_row(0, 0, 100, None),
        env_state_row(
            1, 1, 84, summary(1, 100, {"Agent-1": 7, "Agent-2": 3, "Agent 3": 3, "Agent-4": 3})
        ),
    ]
    monkeypatch.setenv("RUN_DIR", str(write_run(tmp_path, rows)))
    monkeypatch.setenv("MIN_PARTICIPATION", "0.5")

    assert check_replay.main() == 1


def test_unknown_participants_is_empty_when_every_name_is_real():
    names = {"Agent-1", "Agent-2", "Agent-3", "Agent-4"}
    summaries = [summary(1, 100, {"Agent-1": 5, "Agent-3": 2})]

    assert check_replay.unknown_participants(summaries, names) == set()
    assert check_replay.unknown_participants(
        [summary(1, 100, {"Alice": 5})], names
    ) == {"Alice"}


def test_a_lone_extraction_is_not_a_uniformity_signature():
    """One number cannot be evidence that a number was repeated."""
    assert check_replay.uniform_amounts([summary(1, 100, {"Agent-1": 5})]) is False
    assert check_replay.uniform_amounts([]) is False


def _lines(run_dir):
    path = run_dir / "replay" / "commons_tragedy_env_state.00.jsonl"
    return [line for line in path.read_text().splitlines() if line.strip()]


def test_social_media_events_count_as_interaction(tmp_path):
    """SocialMediaSpace writes no env-state counter, only an event table.

    Its posts, follows and likes land in ``social_media_event`` rather than in
    a column like ``total_messages_sent``, so a check that looks only at
    environment state sees an empty simulation and fails a run in which every
    agent acted.
    """
    replay = tmp_path / "replay"
    replay.mkdir()
    (replay / "social_media_space_env_state.00.jsonl").write_text(
        json.dumps({"step": 0}) + "\n"
    )
    (replay / "social_media_event.00.jsonl").write_text(
        "\n".join(
            json.dumps({"id": i, "step": 1, "sender_id": i, "action": "create_post"})
            for i in range(5)
        )
        + "\n"
    )

    assert check_replay._report_interaction(replay, 0.0) == 0


def test_a_social_media_run_where_nobody_posted_still_fails(tmp_path):
    replay = tmp_path / "replay"
    replay.mkdir()
    (replay / "social_media_space_env_state.00.jsonl").write_text(
        json.dumps({"step": 0}) + "\n"
    )
    (replay / "social_media_event.00.jsonl").write_text("")

    assert check_replay._report_interaction(replay, 0.0) == 1
