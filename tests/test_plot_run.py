"""Checks on the HTML report ``plot_run.py`` writes.

Two properties are worth pinning. The first is that the file is genuinely
self-contained: it is opened on a laptop that may be offline, so a single
external reference makes it a broken page rather than a slow one. The second is
that the numbers in it are read out of the run — a report that hardcodes a
scenario or a figure is worse than no report, because it looks authoritative.

The fixtures come from ``tests/make_run_fixture.py``, which reproduces the
storage quirks the reader has to survive: JSON columns stored as JSON strings,
``last_round`` repeating on steps that resolved nothing, and participation
recorded only as the key set of ``extractions``.
"""

from __future__ import annotations

import importlib.util
import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests" / "make_run_fixture.py"


def load_plot_run():
    """Import the script by path; it is an entry point, not a package.

    The module is registered in ``sys.modules`` before it executes because its
    dataclasses carry string annotations, and ``dataclasses`` resolves those
    through ``sys.modules[cls.__module__]``. Skipping the registration makes
    every dataclass in the file fail to build.
    """
    spec = importlib.util.spec_from_file_location(
        "plot_run", ROOT / "scripts" / "plot_run.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def plot_run():
    return load_plot_run()


def make_run(tmp_path_factory, scenario: str, layout: str) -> Path:
    out = tmp_path_factory.mktemp(f"{scenario}-{layout}") / "run"
    subprocess.run(
        [sys.executable, str(FIXTURE), str(out), "--scenario", scenario,
         "--layout", layout],
        check=True, capture_output=True,
    )
    return out


@pytest.fixture(scope="module")
def good_run(tmp_path_factory):
    return make_run(tmp_path_factory, "good", "run-only")


@pytest.fixture(scope="module")
def bad_run(tmp_path_factory):
    return make_run(tmp_path_factory, "bad", "ask-intervene")


def render(plot_run, *run_dirs: Path) -> str:
    return plot_run.render([plot_run.load_run(d) for d in run_dirs])


# --- the file has to work with nothing else present ------------------------


def test_the_page_reaches_for_nothing(plot_run, good_run):
    html = render(plot_run, good_run)

    for pattern in (r'src\s*=\s*"http', r'href\s*=\s*"http', r"@import", r"<script"):
        assert not re.search(pattern, html), pattern
    assert "<svg" in html  # the charts really are inline


def test_the_page_is_small_enough_to_mail(plot_run, good_run):
    assert len(render(plot_run, good_run).encode()) < 200 * 1024


# --- the numbers come from the run -----------------------------------------


def test_participation_counts_submitters_per_round(plot_run, good_run):
    run = plot_run.load_run(good_run)
    size = len(run.agents)

    assert size == 4
    for number, submitted in run.participation_per_round():
        found = run.round_by_number(number)
        assert found is not None
        assert submitted == len(found.extractions)
        assert submitted <= size


def test_repeated_last_round_rows_are_not_double_counted(plot_run, bad_run):
    """The column repeats on steps that resolved nothing."""
    run = plot_run.load_run(bad_run)
    numbers = [r.number for r in run.rounds]

    assert numbers == sorted(set(numbers))


def test_the_scenario_is_read_not_assumed(plot_run, good_run):
    run = plot_run.load_run(good_run)

    assert run.env_module == "CommonsTragedy"
    assert plot_run.scenario_name(run) == "共有地の悲劇"
    # An environment with no entry keeps its class name rather than borrowing
    # a title from a scenario it never ran.
    other = plot_run.load_run(good_run)
    other.env_module = "SomethingElse"
    assert plot_run.scenario_name(other) == "SomethingElse"


def test_the_rules_track_the_run(plot_run, good_run):
    run = plot_run.load_run(good_run)
    text = " ".join(
        item for line in plot_run.rules_lines(run)
        for item in ([line] if isinstance(line, str) else line)
    )

    assert str(run.initial_pool) in text
    assert str(run.max_extraction) in text


# --- what the headings say --------------------------------------------------


def test_every_heading_carries_its_number(plot_run, good_run):
    html = render(plot_run, good_run)
    headings = re.findall(r"<h2[^>]*>(.*?)</h2>", html, re.S)

    panels = [h for h in headings if re.match(r"^\d+\.", re.sub(r"<[^>]+>", "", h))]
    assert len(panels) == 7
    for heading in panels:
        assert re.search(r"\d", re.sub(r"<[^>]+>", "", heading)), heading


def test_a_run_with_the_helper_says_so(plot_run, bad_run):
    """ask/intervene means a recorded extraction may not be the agent's.

    No post-hoc check can catch a submission made under a name that exists, so
    the report leads with the fact that the configuration allowed it.
    """
    html = render(plot_run, bad_run)

    assert "ask" in html and "intervene" in html
    assert "断定できない" in html


def test_a_comparison_draws_the_runs_together(plot_run, good_run, bad_run):
    html = render(plot_run, good_run, bad_run)

    assert "ラン比較" in html
    assert "結論" in html


# --- the storage layout the reader depends on -------------------------------


def test_json_columns_arrive_as_strings(good_run):
    """If this ever stops being true, the reader's second json.loads breaks."""
    shard = next((good_run / "replay").glob("*_env_state.*.jsonl"))
    row = json.loads(shard.read_text().splitlines()[0])

    assert isinstance(row["last_round"], (str, type(None)))
    assert isinstance(row["pending_extractions"], str)
