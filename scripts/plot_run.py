#!/usr/bin/env python3
"""Render one or more run directories as a single self-contained HTML report.

Standard library only, on purpose. The report has to open on a laptop that has
nothing installed, and it has to be *producible* on a login node without first
activating the vLLM environment, which lives on a shared filesystem and is slow
to import. The data is also tiny — four agents by four rounds — so a charting
library would be more bytes of dependency than bytes of data. Everything is
inline SVG, inline CSS and no JavaScript.

Storage layout this reads, per agentsociety2 2.8.7:

- ``<run>/replay/{table}.{shard:02x}.jsonl`` — 256 shards per table, appended by
  every writer process. JSON-typed columns are stored as JSON *strings*
  (``storage/replay_sink._normalize``), so they need a second ``json.loads``.
- ``<run>/replay/commons_tragedy_env_state.*`` — one row per ``run`` step. The
  table prefix comes from the env class name with ``Env`` stripped
  (``env/base.py:_state_table_prefix_from_class``).
- ``<run>/artifacts/questionnaire_step_{idx}_{simtime}.json`` — one file per
  questionnaire step; ``questionnaire_id`` is the authoritative round key.
- ``<run>/sim.log`` — ``[%(asctime)s] %(levelname)-7s %(message)s``.

Two traps that shape the parsing:

1. ``last_round`` repeats the previous round's summary on any step where no
   round executed, so rounds must be deduplicated by their ``round`` field.
2. ``pending_extractions`` and ``submitted_agents`` are cleared before the row
   is written and are therefore empty on every row. The only record of who
   submitted is the key set of ``last_round.extractions``; an agent that did not
   submit is *absent*, not zero.
"""

from __future__ import annotations

import argparse
import html
import json
import re
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Sequence, Union

# A telegraph outline: a line, or a list of lines supporting the one before it.
Lines = Sequence[Union[str, Sequence[str]]]

# --------------------------------------------------------------------------
# Data model
# --------------------------------------------------------------------------


@dataclass
class Agent:
    id: int
    name: str
    age: int | None
    disposition: str


@dataclass
class Round:
    number: int
    step: int
    t: str
    pool_before: int
    pool_after: int
    extractions: dict[str, int]

    @property
    def rationed(self) -> bool:
        """True when the pool ran out and the env split it proportionally.

        The replay stores what each agent *received*, never what it asked for,
        so a rationed round makes the questionnaire's self-reported request
        legitimately larger than the recorded extraction. Without this the
        report would flag honest agents as misremembering. The env allocates
        exactly ``pool_before`` in that branch, so the pool landing on zero with
        the full amount distributed is the signature.
        """
        return bool(
            self.extractions
            and self.pool_after == 0
            and sum(self.extractions.values()) == self.pool_before
        )


@dataclass
class Answer:
    value: object
    text: str
    reason: str | None
    ok: bool


@dataclass
class LogItem:
    """Anything read out of sim.log, placed by the phase it appeared in.

    ``round`` and ``phase_kind`` are filled in after the walk, once the phases
    have been resolved to round numbers.
    """

    phase_index: int | None = None
    round: int | None = None
    phase_kind: str = ""


@dataclass
class Failure(LogItem):
    """One ReAct-level failure attributed to an agent and a round."""

    ts: datetime | None = None
    agent_id: int = 0
    action: str = ""
    category: str = ""
    message: str = ""


@dataclass
class EnvAsk(LogItem):
    """One ask_env round trip: the opening of the request, and its duration."""

    instruction: str = ""
    seconds: float = 0.0


@dataclass
class Wording(LogItem):
    """What an agent actually asked the environment, in its own words.

    Only reaches the log because no embedding model is served: every ask_env
    misses the codegen template cache, and the miss is logged with the
    instruction text. With embeddings configured this goes quiet, which is worth
    knowing before relying on it.
    """

    text: str = ""


@dataclass
class HelperSubmit(LogItem):
    """An extraction the plan-and-execute helper submitted on its own.

    The helper runs the ask/intervene steps and reaches the environment through
    ``ask_environment``, inventing an ``agent_name`` as it goes. The resulting
    row is indistinguishable from a real agent's in the replay, so the log is
    the only place the impersonation is visible.
    """

    amount: int = 0
    name: str = ""


@dataclass
class Phase:
    """A stretch of sim.log between two step-boundary lines."""

    round: int | None
    kind: str
    start: datetime
    end: datetime | None = None

    @property
    def seconds(self) -> float:
        if self.end is None:
            return 0.0
        return (self.end - self.start).total_seconds()


@dataclass
class LogFacts:
    """Everything read out of sim.log."""

    failures: list[Failure] = field(default_factory=list)
    phases: list[Phase] = field(default_factory=list)
    env_asks: list[EnvAsk] = field(default_factory=list)
    wordings: list[Wording] = field(default_factory=list)
    helper_submits: list[HelperSubmit] = field(default_factory=list)


@dataclass
class Run:
    path: Path
    label: str
    agents: list[Agent] = field(default_factory=list)
    rounds: list[Round] = field(default_factory=list)
    planned_rounds: int = 0
    initial_pool: int = 0
    max_extraction: int = 0
    unresolved_steps: int = 0
    answers: dict[int, dict[str, Answer]] = field(default_factory=dict)
    instructions: dict[int, str] = field(default_factory=dict)
    standing_orders: str = ""
    env_module: str = ""
    log: LogFacts = field(default_factory=LogFacts)
    log_missing: bool = False

    # -- derived -----------------------------------------------------------

    @property
    def failures(self) -> list[Failure]:
        return self.log.failures

    @property
    def phases(self) -> list[Phase]:
        return self.log.phases

    @property
    def agent_names(self) -> list[str]:
        return [a.name for a in self.agents]

    @property
    def unknown_takers(self) -> dict[str, list[int]]:
        """Extraction keys that match no agent in core_agent_profile.

        The environment takes ``agent_name`` as a free string and never
        validates it, so an agent that writes "Agent 3" instead of "Agent-3"
        opens an account belonging to nobody, and the units are gone from the
        pool all the same. Nothing raises, so this check is the only way to see
        it. Note the limit: it cannot catch a submission made under a name that
        does exist, which is what the plan-and-execute helper did.
        """
        known = set(self.agent_names)
        out: dict[str, list[int]] = {}
        for found in self.rounds:
            for name in found.extractions:
                if name not in known:
                    out.setdefault(name, []).append(found.number)
        return out

    @property
    def helper_active(self) -> bool:
        """Whether this run had ask/intervene steps at all.

        Those steps are executed by AgentSocietyHelper, a plan-and-execute loop
        that reaches the environment through ``ask_environment`` and picks the
        ``agent_name`` itself. Where it ran, no recorded extraction can be
        attributed to an agent with certainty, whether or not the regexes below
        catch the helper saying so. That makes this flag, not the match count,
        the honest headline.
        """
        return any(phase.kind in ("ask", "intervene") for phase in self.phases)

    @property
    def suspect_rounds(self) -> dict[int, list[HelperSubmit]]:
        out: dict[int, list[HelperSubmit]] = {}
        for submit in self.log.helper_submits:
            if submit.round is not None:
                out.setdefault(submit.round, []).append(submit)
        return out

    @property
    def round_numbers(self) -> list[int]:
        """Every planned round, resolved or not."""
        return list(range(1, max(self.planned_rounds, len(self.rounds)) + 1))

    def round_by_number(self, number: int) -> Round | None:
        return next((r for r in self.rounds if r.number == number), None)

    @property
    def participation(self) -> float:
        """Submissions divided by opportunities. The headline number."""
        opportunities = len(self.agents) * len(self.round_numbers)
        if not opportunities:
            return 0.0
        return sum(len(r.extractions) for r in self.rounds) / opportunities

    def participation_per_round(self) -> list[tuple[int, int]]:
        out = []
        for number in self.round_numbers:
            found = self.round_by_number(number)
            out.append((number, len(found.extractions) if found else 0))
        return out

    def pool_trajectory(self) -> list[tuple[int, int]]:
        points = [(0, self.initial_pool)]
        for number in self.round_numbers:
            found = self.round_by_number(number)
            points.append((number, found.pool_after if found else points[-1][1]))
        return points

    def cumulative_payoff(self, name: str) -> list[int]:
        total = 0
        out = [0]
        for number in self.round_numbers:
            found = self.round_by_number(number)
            total += (found.extractions.get(name, 0) if found else 0)
            out.append(total)
        return out


# --------------------------------------------------------------------------
# Parsing
# --------------------------------------------------------------------------


def _maybe_json(value):
    """Decode a column the replay sink serialized as a JSON string."""
    if isinstance(value, str):
        stripped = value.strip()
        if stripped[:1] in "{[":
            try:
                return json.loads(stripped)
            except json.JSONDecodeError:
                return value
    return value


def read_table(replay_dir: Path, table: str) -> list[dict]:
    rows: list[dict] = []
    for shard in sorted(replay_dir.glob(f"{table}.*.jsonl")):
        with shard.open(encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
    return rows


def find_env_table(replay_dir: Path) -> str | None:
    """The env table name is derived from the env class, so discover it."""
    names = {p.name.rsplit(".", 2)[0] for p in replay_dir.glob("*_env_state.*.jsonl")}
    return sorted(names)[0] if names else None


def env_module_from_table(table: str) -> str:
    """Recover the environment class name from its replay table name.

    ``env/base.py:_state_table_prefix_from_class`` snake-cases the class and
    drops a trailing ``Env``, so ``CommonsTragedyEnv`` writes
    ``commons_tragedy_env_state``. Going back is lossy about that suffix, but
    the table is always present, whereas ``init_config.json`` is not on a
    directory that was copied off the cluster.
    """
    stem = table[: -len("_env_state")] if table.endswith("_env_state") else table
    return "".join(part.title() for part in stem.split("_") if part)


_DISPOSITION = re.compile(r"Your disposition:\s*(.+?)\.?\s*$", re.S)


def parse_agents(replay_dir: Path) -> list[Agent]:
    """One Agent per id.

    Deduplicated deliberately: the replay shards are append-only, so a rerun or
    a ``--resume`` into the same directory leaves two profile rows per agent and
    an undeduplicated read would report twice the agents and halve every
    participation rate.
    """
    by_id: dict[int, Agent] = {}
    for row in read_table(replay_dir, "core_agent_profile"):
        profile = _maybe_json(row.get("profile")) or {}
        personality = str(profile.get("personality", ""))
        # gen_config.py prefixes a shared role to every personality; only the
        # trailing disposition distinguishes one agent from another.
        match = _DISPOSITION.search(personality)
        agent_id = int(row["id"])
        by_id[agent_id] = Agent(
            id=agent_id,
            name=str(row.get("name") or profile.get("name") or agent_id),
            age=profile.get("age"),
            disposition=(match.group(1).strip() if match else personality.strip()),
        )
    return [by_id[key] for key in sorted(by_id)]


def parse_rounds(replay_dir: Path, table: str) -> tuple[list[Round], int, int, int, int]:
    """Return (rounds, unresolved_steps, initial_pool, max_extraction, final_pool)."""
    rows = read_table(replay_dir, table)
    rows.sort(key=lambda r: r.get("step", 0))

    rounds: list[Round] = []
    seen: set[int] = set()
    unresolved = 0
    initial_pool = 0
    max_extraction = 0
    final_pool = 0

    for row in rows:
        initial_pool = int(row.get("initial_pool_resources", initial_pool) or 0)
        max_extraction = int(row.get("max_extraction_per_agent", max_extraction) or 0)
        final_pool = int(row.get("current_pool_resources", final_pool) or 0)

        last_round = _maybe_json(row.get("last_round"))
        if not isinstance(last_round, dict):
            unresolved += 1
            continue
        number = int(last_round.get("round", 0))
        if number in seen:
            # The env re-reports the previous summary on a step where nobody
            # submitted. Same round twice means this step resolved nothing.
            unresolved += 1
            continue
        seen.add(number)
        rounds.append(
            Round(
                number=number,
                step=int(row.get("step", 0)),
                t=str(row.get("t", "")),
                pool_before=int(last_round.get("pool_before_round", 0)),
                pool_after=int(last_round.get("pool_after_round", 0)),
                extractions={
                    str(k): int(v)
                    for k, v in (last_round.get("extractions") or {}).items()
                },
            )
        )

    rounds.sort(key=lambda r: r.number)
    return rounds, unresolved, initial_pool, max_extraction, final_pool


_ROUND_ID = re.compile(r"round[_-]?(\d+)", re.I)


def parse_questionnaires(artifacts: Path) -> dict[int, dict[str, Answer]]:
    out: dict[int, dict[str, Answer]] = {}
    for path in sorted(artifacts.glob("questionnaire_step_*.json")):
        try:
            document = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        match = _ROUND_ID.search(str(document.get("questionnaire_id", "")))
        if not match:
            continue
        number = int(match.group(1))
        per_agent: dict[str, Answer] = {}
        for response in document.get("responses", []):
            answers = response.get("answers") or []
            if not answers:
                continue
            answer = answers[0]
            per_agent[str(response.get("agent_name"))] = Answer(
                value=answer.get("parsed_value"),
                text=str(answer.get("raw_text", "")),
                reason=answer.get("reason"),
                ok=bool(answer.get("parse_success", True)),
            )
        out[number] = per_agent
    return out


_INSTRUCTION_ROUND = re.compile(r"round\s+(\d+)", re.I)


def parse_instructions(artifacts: Path) -> dict[int, str]:
    """Pull each round's intervene wording out of the artifact front matter.

    Empty under the step layout shunk031/tsubame-agentsociety2#2 moves to, which
    has no intervene steps at all; ``parse_standing_orders`` covers that case.
    """
    out: dict[int, str] = {}
    for path in sorted(artifacts.glob("intervene_step_*.md")):
        try:
            text = path.read_text(encoding="utf-8")
        except OSError:
            continue
        if not text.startswith("---"):
            continue
        _, _, rest = text.partition("---\n")
        front, _, _body = rest.partition("---\n")
        _, _, instruction = front.partition("instruction:")
        instruction = " ".join(instruction.split())
        match = _INSTRUCTION_ROUND.search(instruction)
        if match:
            out[int(match.group(1))] = instruction
    return out


def parse_standing_orders(run_dir: Path) -> str:
    """The rules-and-action text every agent carried in its profile.

    Once the per-round intervene step is gone, this is the only wording the
    experiment controls, and it is the first thing to suspect when participation
    changes between runs. It is the shared prefix of every agent's
    ``personality``, before the per-agent ``Your disposition:`` clause.
    """
    path = run_dir / "init_config.json"
    if not path.is_file():
        return ""
    try:
        agents = json.loads(path.read_text(encoding="utf-8")).get("agents", [])
    except (json.JSONDecodeError, OSError):
        return ""
    texts = [
        str(agent.get("kwargs", {}).get("personality", ""))
        for agent in agents
    ]
    texts = [t for t in texts if t]
    if not texts:
        return ""
    head, _, _tail = texts[0].partition("Your disposition:")
    return " ".join(head.split())


# sim.log is the job's whole stdout, so Ray's worker output is interleaved with
# the driver's. A worker line arrives as
# ``\x1b[36m(step_agent_batch pid=2116)\x1b[0m [ts] WARNING Agent 4: ...``:
# colour codes, then a prefix, then the ordinary record. Two thirds of the ReAct
# failures in a real run come through that path, so a reader anchored at ``^[``
# silently drops most of them — measured on run sim-8645779: 4 of 15 seen.
_ANSI = re.compile(r"\x1b\[[0-9;]*m")
_WORKER = re.compile(r"^\((\w+) pid=\d+(?:, ip=[\d.]+)?\)\s*")
_LOG_LINE = re.compile(r"^\[(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d)\]\s+(\w+)\s+(.*)$")
_TOOL_FAILED = re.compile(
    r"^Agent (\d+): ReAct tool failed: action=(\S+) observation=(.*)$", re.S
)
_BAD_DECISION = re.compile(r"^Agent (\d+): invalid ReAct decision: (.*)$", re.S)

# Untimestamped lines the env router actor prints around every ask_env call.
# They arrive in flushed blocks that are internally ordered but out of band
# relative to the timestamped records, so their position says nothing about
# which round they belong to — measured on sim-8645779, where the round 1 trace
# appears in the file after the round 3 boundary. Durations are still durations.
_ENV_ASK = re.compile(r"^\[EnvActor\] ask instruction='(.*?)' took ([\d.]+)s$")
# Neither does the ordering of the helper's own task blocks: in sim-8645779 the
# submission stamped 09:15:00 (round 2) sits inside the round 3 block. What does
# hold is the simulation timestamp the environment echoes back inside the tool
# result, because simulation time advances by exactly one tick per round.
_HELPER_TASK = re.compile(r"^Task: It is now round (\d+)\b")
_SIM_TIME = re.compile(r"\[\w+, (\d{4}-\d\d-\d\d \d\d:\d\d:\d\d)\]")
# Every ask_env is a codegen cache miss when no embedding model is served, and
# the miss line is the one place the agent's own wording survives into the log.
_CACHE_MISS = re.compile(r"^Template cache miss: reason=(\S+) instruction=(.*)$", re.S)
# The helper narrates its own submissions in free-form LLM prose, so these two
# shapes are what run sim-8645779 happened to produce, not a closed set. They
# find helper submissions; they cannot prove there were no others. The panel
# says so rather than implying the list is exhaustive.
_HELPER_SUBMITS = [
    re.compile(
        r"submitted an extraction request for (?P<amount>\d+) units? "
        r"to (?P<name>[A-Za-z][\w -]*?)[.\"]"
    ),
    re.compile(
        r"(?P<name>Agent[- ]?\d+) successfully submitted an extraction request "
        r"for (?P<amount>\d+) units?"
    ),
]

FAILURE_CATEGORIES = [
    ("unknown_tool", re.compile(r"Unknown tool|unknown action:"), "存在しないツールを呼んだ"),
    ("readonly", re.compile(r"mutation is disabled in readonly mode"), "読み取り専用ステップで行動しようとした"),
    ("bad_args", re.compile(r"unexpected keyword argument|validation error"), "ツールの引数が合わない"),
    ("workspace", re.compile(r"Path escapes agent workspace|File name too long"), "ワークスペース外を触ろうとした"),
    ("skill", re.compile(r"execute_skill_script|Script .*not found"), "スキルスクリプトの実行に失敗"),
    ("empty", re.compile(r"empty content|Empty response"), "LLM の応答が空"),
]


def classify(message: str) -> str:
    for key, pattern, _label in FAILURE_CATEGORIES:
        if pattern.search(message):
            return key
    return "other"


def category_label(key: str) -> str:
    for name, _pattern, label in FAILURE_CATEGORIES:
        if name == key:
            return label
    return "その他"


def parse_schedule(run_dir: Path) -> tuple[datetime | None, int]:
    """Return ``(start_t, tick)`` from steps.yaml, the simulation clock's origin."""
    path = run_dir / "steps.yaml"
    if not path.is_file():
        return None, 0
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None, 0
    ticks = [
        int(step.get("tick", 0))
        for step in document.get("steps", [])
        if step.get("type") == "run"
    ]
    try:
        start = datetime.fromisoformat(str(document.get("start_t")))
    except (TypeError, ValueError):
        start = None
    return start, (ticks[0] if ticks else 0)


def _round_of(line: str, schedule: tuple[datetime | None, int]) -> int | None:
    """Map a simulation timestamp echoed in a log line to its round number."""
    start, tick = schedule
    if start is None or tick <= 0:
        return None
    found = _SIM_TIME.search(line)
    if not found:
        return None
    try:
        stamp = datetime.strptime(found.group(1), "%Y-%m-%d %H:%M:%S")
    except ValueError:
        return None
    offset = (stamp - start).total_seconds()
    if offset < 0:
        return None
    return int(offset // tick) + 1


def _resolve_phase_rounds(phases: list[Phase]) -> None:
    """Give every run phase the round number it belongs to.

    Only a questionnaire names its round outright (``Running questionnaire
    round_3``), and it comes *after* the run step it measures. So the round is
    filled in backwards from each questionnaire, then forwards for any trailing
    run that no questionnaire followed.

    This is what makes the report survive both step layouts: the original one,
    where an ``Intervening: It is now round 3`` line opened each round, and the
    one shunk031/tsubame-agentsociety2#2 moves to, where a round is nothing but
    a ``run`` step followed by its questionnaire.
    """
    next_known: int | None = None
    for phase in reversed(phases):
        if phase.round is not None:
            next_known = phase.round
        elif phase.kind in ("run", "intervene"):
            phase.round = next_known

    last = 0
    for phase in phases:
        if phase.round is not None:
            last = phase.round
        elif phase.kind == "run":
            last += 1
            phase.round = last


def parse_log(path: Path, schedule: tuple[datetime | None, int] = (None, 0)) -> LogFacts:
    """Walk sim.log once, splitting it into phases and collecting what it holds.

    Step boundaries are unambiguous log lines written by ``society/cli.py``:
    ``Running n/m steps with tick=`` opens a run phase, ``Running questionnaire
    <id>`` opens a questionnaire phase and names its round, and ``Intervening:``
    / ``Asking:`` open the helper-driven phases that the current step layout no
    longer produces. Everything found is attached to the phase it occurred in
    and resolved to a round afterwards.

    Besides failures this collects three things that only exist in the log:
    ask_env round-trip times, the agents' own request wording (via the codegen
    cache-miss line), and the plan-and-execute helper's own submissions.
    """
    facts = LogFacts()
    failures = facts.failures
    phases = facts.phases

    def close(ts: datetime) -> None:
        if phases and phases[-1].end is None:
            phases[-1].end = ts

    last_ts: datetime | None = None
    helper_round: int | None = None
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = _WORKER.sub("", _ANSI.sub("", raw.rstrip()))

        # Untimestamped trace lines, attributed by the helper's own task header
        # rather than by file position.
        task = _HELPER_TASK.match(line)
        if task:
            helper_round = int(task.group(1))
            continue
        asked = _ENV_ASK.match(line)
        if asked:
            facts.env_asks.append(
                EnvAsk(
                    instruction=asked.group(1),
                    seconds=float(asked.group(2)),
                )
            )
            continue
        for pattern in _HELPER_SUBMITS:
            helper = pattern.search(line)
            if helper:
                facts.helper_submits.append(
                    HelperSubmit(
                        amount=int(helper.group("amount")),
                        name=helper.group("name").strip(),
                        round=_round_of(line, schedule) or helper_round,
                    )
                )
                break

        match = _LOG_LINE.match(line)
        if not match:
            continue  # continuation line of a multi-line message
        stamp, _level, message = match.groups()
        ts = datetime.strptime(stamp, "%Y-%m-%d %H:%M:%S")
        last_ts = ts

        missed = _CACHE_MISS.match(message)
        if missed:
            facts.wordings.append(
                Wording(
                    phase_index=len(phases) - 1,
                    text=" ".join(missed.group(2).split()),
                )
            )
            continue

        if message.startswith("Intervening:"):
            found = _INSTRUCTION_ROUND.search(message)
            close(ts)
            phases.append(
                Phase(int(found.group(1)) if found else None, "intervene", ts)
            )
            continue
        if message.startswith("Running ") and "steps with tick=" in message:
            close(ts)
            phases.append(Phase(None, "run", ts))
            continue
        if message.startswith("Running questionnaire"):
            found = _ROUND_ID.search(message)
            close(ts)
            phases.append(
                Phase(int(found.group(1)) if found else None, "questionnaire", ts)
            )
            continue
        if message.startswith("Asking:"):
            close(ts)
            phases.append(Phase(None, "ask", ts))
            continue

        failed = _TOOL_FAILED.match(message)
        bad = None if failed else _BAD_DECISION.match(message)
        if failed or bad:
            if failed:
                agent_id, action, detail = failed.groups()
            else:
                agent_id, detail = bad.groups()
                action = "(decision)"
            failures.append(
                Failure(
                    ts=ts,
                    agent_id=int(agent_id),
                    action=action,
                    category=classify(detail),
                    message=detail.strip(),
                    phase_index=len(phases) - 1 if phases else None,
                )
            )

    if last_ts is not None:
        close(last_ts)
    _resolve_phase_rounds(phases)
    for item in [*failures, *facts.wordings]:
        if item.phase_index is not None and 0 <= item.phase_index < len(phases):
            item.round = phases[item.phase_index].round
            item.phase_kind = phases[item.phase_index].kind
    return facts


def count_planned_rounds(run_dir: Path) -> int:
    """Count the rounds steps.yaml asks for, without needing a YAML parser.

    A questionnaire per round is the one constant across step layouts, so its
    ``round_N`` id is the primary signal; the intervene and run counts are
    fallbacks for layouts that measure differently. gen_config.py writes JSON
    into steps.yaml, so json.loads handles the file we produce.
    """
    path = run_dir / "steps.yaml"
    if not path.is_file():
        return 0
    text = path.read_text(encoding="utf-8")
    try:
        steps = json.loads(text).get("steps", [])
    except json.JSONDecodeError:
        return len(re.findall(r"questionnaire_id:\s*[\"']?round[_-]?\d", text)) or len(
            re.findall(r"type:\s*[\"']?intervene", text)
        )

    rounds = {
        int(found.group(1))
        for step in steps
        if step.get("type") == "questionnaire"
        for found in [_ROUND_ID.search(str(step.get("questionnaire_id", "")))]
        if found
    }
    if rounds:
        return max(rounds)
    for kind in ("intervene", "run"):
        count = sum(1 for step in steps if step.get("type") == kind)
        if count:
            return count
    return 0


def load_run(run_dir: Path, label: str | None = None) -> Run:
    replay_dir = run_dir / "replay"
    if not replay_dir.is_dir():
        raise SystemExit(f"no replay directory at {replay_dir}")

    table = find_env_table(replay_dir)
    if table is None:
        raise SystemExit(f"{replay_dir} holds no *_env_state shards")

    rounds, unresolved, pool, max_extraction, _final = parse_rounds(replay_dir, table)
    run = Run(
        path=run_dir,
        label=label or run_dir.name,
        agents=parse_agents(replay_dir),
        rounds=rounds,
        unresolved_steps=unresolved,
        initial_pool=pool,
        max_extraction=max_extraction,
        planned_rounds=count_planned_rounds(run_dir),
        answers=parse_questionnaires(run_dir / "artifacts"),
        instructions=parse_instructions(run_dir / "artifacts"),
        standing_orders=parse_standing_orders(run_dir),
        env_module=env_module_from_table(table),
    )
    log_path = run_dir / "sim.log"
    if log_path.is_file():
        run.log = parse_log(log_path, parse_schedule(run_dir))
    else:
        run.log_missing = True
    return run


# --------------------------------------------------------------------------
# SVG primitives
# --------------------------------------------------------------------------

PLOT_W, PLOT_H = 620, 240
PAD_L, PAD_R, PAD_T, PAD_B = 44, 16, 14, 32


def _scale(value: float, lo: float, hi: float, out_lo: float, out_hi: float) -> float:
    if hi == lo:
        return (out_lo + out_hi) / 2
    return out_lo + (value - lo) * (out_hi - out_lo) / (hi - lo)


def nice_max(value: float) -> float:
    """Round an axis maximum up to something a reader can divide by four."""
    if value <= 0:
        return 1.0
    exponent = 10 ** int(f"{value:e}".split("e")[1])
    for factor in (1, 1.5, 2, 2.5, 3, 4, 5, 6, 8, 10):
        candidate = factor * exponent
        if candidate >= value:
            return candidate
    return 10 * exponent


def _axes(x_max: float, y_max: float, y_label: str, x_label: str,
          x_ticks: bool = True) -> list[str]:
    parts = [
        f'<line class="axis" x1="{PAD_L}" y1="{PAD_T}" x2="{PAD_L}" y2="{PLOT_H - PAD_B}"/>',
        f'<line class="axis" x1="{PAD_L}" y1="{PLOT_H - PAD_B}" '
        f'x2="{PLOT_W - PAD_R}" y2="{PLOT_H - PAD_B}"/>',
    ]
    steps = 4
    for index in range(steps + 1):
        value = y_max * index / steps
        y = _scale(value, 0, y_max, PLOT_H - PAD_B, PAD_T)
        parts.append(f'<line class="grid" x1="{PAD_L}" y1="{y:.1f}" x2="{PLOT_W - PAD_R}" y2="{y:.1f}"/>')
        parts.append(f'<text class="tick" x="{PAD_L - 6}" y="{y + 3.5:.1f}" text-anchor="end">{value:.0f}</text>')
    if x_ticks:
        for index in range(int(x_max) + 1):
            x = _scale(index, 0, x_max, PAD_L, PLOT_W - PAD_R)
            parts.append(
                f'<text class="tick" x="{x:.1f}" y="{PLOT_H - PAD_B + 15}" text-anchor="middle">{index}</text>'
            )
    parts.append(
        f'<text class="axis-label" x="{(PAD_L + PLOT_W - PAD_R) / 2:.0f}" '
        f'y="{PLOT_H - 4}" text-anchor="middle">{html.escape(x_label)}</text>'
    )
    parts.append(
        f'<text class="axis-label" x="{-(PAD_T + PLOT_H - PAD_B) / 2:.0f}" y="11" '
        f'transform="rotate(-90)" text-anchor="middle">{html.escape(y_label)}</text>'
    )
    return parts


def line_chart(
    series: list[tuple[str, list[tuple[float, float]], str]],
    *,
    x_max: float,
    y_max: float,
    x_label: str,
    y_label: str,
    markers: list[tuple[float, str]] | None = None,
    band: list[tuple[float, float]] | None = None,
    band_label: str = "",
) -> str:
    """One or more polylines on shared axes. ``series`` is (label, points, css class)."""
    y_max = nice_max(max(y_max, 1))
    parts = [
        f'<svg class="chart" viewBox="0 0 {PLOT_W} {PLOT_H}" role="img" '
        f'aria-label="{html.escape(y_label)} / {html.escape(x_label)}">'
    ]
    parts += _axes(x_max, y_max, y_label, x_label)

    if band:
        upper = " ".join(
            f"{_scale(x, 0, x_max, PAD_L, PLOT_W - PAD_R):.1f},"
            f"{_scale(y, 0, y_max, PLOT_H - PAD_B, PAD_T):.1f}"
            for x, y in band
        )
        floor = (
            f"{_scale(band[-1][0], 0, x_max, PAD_L, PLOT_W - PAD_R):.1f},{PLOT_H - PAD_B} "
            f"{PAD_L},{PLOT_H - PAD_B}"
        )
        parts.append(f'<polygon class="band" points="{upper} {floor}"/>')
        if band_label:
            parts.append(
                f'<text class="band-label" x="{PAD_L + 8}" '
                f'y="{_scale(band[-1][1], 0, y_max, PLOT_H - PAD_B, PAD_T) - 6:.1f}">'
                f"{html.escape(band_label)}</text>"
            )

    for marker_x, marker_text in markers or []:
        x = _scale(marker_x, 0, x_max, PAD_L, PLOT_W - PAD_R)
        parts.append(f'<line class="marker" x1="{x:.1f}" y1="{PAD_T}" x2="{x:.1f}" y2="{PLOT_H - PAD_B}"/>')
        # Keep the label inside the plot when the marker sits on an edge.
        anchor = "end" if x > PLOT_W - PAD_R - 40 else ("start" if x < PAD_L + 40 else "middle")
        offset = {"end": -4, "start": 4, "middle": 0}[anchor]
        parts.append(
            f'<text class="marker-label" x="{x + offset:.1f}" y="{PAD_T + 10}" '
            f'text-anchor="{anchor}">{html.escape(marker_text)}</text>'
        )

    for label, points, css in series:
        coords = " ".join(
            f"{_scale(x, 0, x_max, PAD_L, PLOT_W - PAD_R):.1f},"
            f"{_scale(y, 0, y_max, PLOT_H - PAD_B, PAD_T):.1f}"
            for x, y in points
        )
        parts.append(f'<polyline class="series {css}" points="{coords}"><title>{html.escape(label)}</title></polyline>')
        for x, y in points:
            cx = _scale(x, 0, x_max, PAD_L, PLOT_W - PAD_R)
            cy = _scale(y, 0, y_max, PLOT_H - PAD_B, PAD_T)
            parts.append(
                f'<circle class="dot {css}" cx="{cx:.1f}" cy="{cy:.1f}" r="3">'
                f"<title>{html.escape(label)}: {y:g}</title></circle>"
            )
    parts.append("</svg>")
    return "".join(parts)


def bar_chart(bars: list[tuple[str, float]], *, y_label: str, unit: str = "s") -> str:
    if not bars:
        return ""
    y_max = nice_max(max(v for _, v in bars) or 1)
    width = PLOT_W - PAD_L - PAD_R
    slot = width / len(bars)
    parts = [
        f'<svg class="chart" viewBox="0 0 {PLOT_W} {PLOT_H}" role="img" '
        f'aria-label="{html.escape(y_label)}">'
    ]
    parts += _axes(0, y_max, y_label, "", x_ticks=False)
    for index, (label, value) in enumerate(bars):
        x = PAD_L + slot * index + slot * 0.15
        bar_w = slot * 0.7
        y = _scale(value, 0, y_max, PLOT_H - PAD_B, PAD_T)
        parts.append(
            f'<rect class="bar" x="{x:.1f}" y="{y:.1f}" width="{bar_w:.1f}" '
            f'height="{PLOT_H - PAD_B - y:.1f}"><title>{html.escape(label)}: '
            f"{value:.0f}{unit}</title></rect>"
        )
        parts.append(
            f'<text class="tick" x="{x + bar_w / 2:.1f}" y="{PLOT_H - PAD_B + 15}" '
            f'text-anchor="middle">{html.escape(label)}</text>'
        )
        parts.append(
            f'<text class="bar-value" x="{x + bar_w / 2:.1f}" y="{y - 4:.1f}" '
            f'text-anchor="middle">{value:.0f}{unit}</text>'
        )
    parts.append("</svg>")
    return "".join(parts)


# --------------------------------------------------------------------------
# Panels
# --------------------------------------------------------------------------


def e(value) -> str:
    return html.escape(str(value))


def verdict_band(run: Run) -> str:
    planned = len(run.round_numbers)
    resolved = len(run.rounds)
    participation = run.participation
    final_pool = run.rounds[-1].pool_after if run.rounds else run.initial_pool
    drawn = run.initial_pool - final_pool

    # A run only tests the scenario if most agents act in most rounds. Below
    # half, what the pool does says more about the harness than about the agents.
    if participation >= 0.75 and resolved == planned:
        verdict, tone = "このランは使える", "ok"
    elif participation >= 0.4:
        verdict, tone = "参加が薄い ➜ 傾向の読み取りは慎重に", "warn"
    else:
        verdict, tone = "参加不足 ➜ シナリオを検証できていない", "bad"

    tiles = [
        ("参加率", f"{participation * 100:.0f}%",
         f"{sum(len(r.extractions) for r in run.rounds)} / {len(run.agents) * planned} 機会"),
        ("成立ラウンド", f"{resolved} / {planned}",
         f"未成立 {planned - resolved}"),
        ("プール残量", f"{final_pool} / {run.initial_pool}",
         f"採取 {drawn} 単位"),
        ("ReAct 失敗", f"{len(run.failures)} 件",
         "sim.log なし" if run.log_missing else f"{len({f.agent_id for f in run.failures})} エージェントに分布"),
    ]
    cells = "".join(
        f'<div class="tile"><div class="tile-label">{e(label)}</div>'
        f'<div class="tile-value">{e(value)}</div>'
        f'<div class="tile-note">{e(note)}</div></div>'
        for label, value, note in tiles
    )
    return (
        f'<div class="verdict {tone}">{e(verdict)}</div>'
        f'<div class="tiles">{cells}</div>'
    )


def comparison_band(runs: list[Run]) -> str:
    """The comparison's opening summary.

    A single run gets a verdict band; without an equivalent here, a comparison
    opened at the top says only which runs were compared, not what came of it.
    The verdict states the spread and nothing more — a difference between three
    runs of one each is not a result, and the closing section says so.
    """
    rates = [r.participation for r in runs]
    span = f"{rates[0] * 100:.0f}% → {rates[-1] * 100:.0f}%"
    if rates[-1] >= 0.75:
        tone = "ok"
    elif rates[-1] >= 0.4:
        tone = "warn"
    else:
        tone = "bad"
    return (
        f'<div class="verdict {tone}">参加率 {e(span)}'
        f"（{len(runs)} 構成・各 1 ラン）</div>"
    )


def so_what(runs: list[Run]) -> str:
    """The closing section: what the numbers above amount to.

    Every line is built from the runs. Facts are stated plainly; anything that
    interprets them is marked 推論 or 示唆, and the limits sit next to the
    numbers they qualify rather than in a footnote nobody reaches.
    """
    lines: list = []

    def series(fmt) -> str:
        return " → ".join(fmt(r) for r in runs)

    if len(runs) > 1:
        lines.append(f"参加率: {series(lambda r: f'{r.participation * 100:.0f}%')}")
        lines.append(
            [
                f"成立ラウンド: {series(lambda r: f'{len(r.rounds)}/{len(r.round_numbers)}')}",
                f"ReAct 失敗: {series(lambda r: str(len(r.failures)))} 件",
                "プール残量: "
                + " / ".join(
                    str(r.rounds[-1].pool_after if r.rounds else r.initial_pool)
                    for r in runs
                ),
            ]
        )

        helper_runs = [r for r in runs if r.helper_active]
        clean_runs = [r for r in runs if not r.helper_active]
        if helper_runs and clean_runs:
            worst = min(runs, key=lambda r: r.participation)
            names = " / ".join(e(r.label) for r in helper_runs)
            lines.append(f"<code>ask</code> / <code>intervene</code> を持つラン: {names}")
            support = [
                f"最も参加率が低いラン: {e(worst.label)}"
                f"（{worst.participation * 100:.0f}%）",
            ]
            if worst in helper_runs:
                support.append("➜ 推論: 参加率の差はステップ構成と対応している")
            lines.append(support)

        # Where the standing orders differ, the step layout is not the only
        # thing that changed, and the report should not let the reader assume
        # it was.
        orders = {r.standing_orders for r in runs if r.standing_orders}
        if len(orders) > 1:
            lines.append("profile の指示文もランごとに違う")
            lines.append(["➜ 差の原因をステップ構成だけに帰せない"])

        lines.append(f"限界: 各構成 1 ラン（n={len(runs)} 構成 × 1）")
        lines.append(
            [
                "ラン間の差は観測であって知見ではない",
                "➜ 示唆: 同じ構成を複数回走らせないと、構成由来か実行ごとのばらつきか分けられない",
            ]
        )
    else:
        run = runs[0]
        submitted = sum(len(r.extractions) for r in run.rounds)
        lines.append(
            f"参加: {submitted}/{len(run.agents) * len(run.round_numbers)} 機会 / "
            f"成立ラウンド: {len(run.rounds)}/{len(run.round_numbers)}"
        )
        if run.answers:
            stats = report_stats(run)
            lines.append(
                f"自己申告と実測の食い違い: {stats['mismatches']}/{stats['cells']} セル"
            )
            detail = []
            if stats["impossible"]:
                detail.append(
                    f"別に、1〜{run.max_extraction} の範囲外の回答が "
                    f"{stats['impossible']} セル"
                )
            detail.append("➜ 推論: questionnaire は測定器としてまだ信頼できない")
            lines.append(detail)
        lines.append("限界: このランは 1 本（n=1）")
        lines.append(["他のランとの差を語るには、同じ構成の反復がいる"])

    # This one holds regardless of layout, and belongs wherever an extraction
    # count is quoted.
    lines.append("限界: 名前の照合は書き損じしか捉えない")
    lines.append(["➜ 実在する名前を騙った提出は素通りする"])
    if any(r.helper_active for r in runs):
        lines.append(
            "限界: <code>ask</code> / <code>intervene</code> のあるランでは、"
            "採取を提出したのが本人だと確定できない"
        )

    # The closing heading states the conclusion like every other heading here;
    # "まとめ" would make the one section that exists to answer the question
    # the only one that does not answer it.
    if len(runs) > 1:
        title = (
            f"結論: 参加率 {runs[0].participation * 100:.0f}% → "
            f"{runs[-1].participation * 100:.0f}%、ただし各構成 1 ラン"
        )
    else:
        run = runs[0]
        submitted = sum(len(r.extractions) for r in run.rounds)
        opportunities = len(run.agents) * len(run.round_numbers)
        title = f"結論: 参加 {submitted}/{opportunities} 機会"
        if run.answers:
            stats = report_stats(run)
            title += f"、自己申告は {stats['mismatches']}/{stats['cells']} セルが食い違い"
    return f"<h2>{e(title)}</h2>{bullets(lines)}"


def participation_matrix(run: Run) -> str:
    """Rows are agents, columns are rounds. The centrepiece of the report.

    Extraction heatmap and participation rate are the same matrix read along two
    axes, so they are one figure: splitting them would force the reader to
    cross-reference two charts to answer one question.
    """
    numbers = run.round_numbers
    suspect = run.suspect_rounds
    header = "".join(f"<th>R{n}</th>" for n in numbers)
    rows = []
    for agent in run.agents:
        cells = []
        for number in numbers:
            found = run.round_by_number(number)
            if found is None:
                cells.append('<td class="cell dead" title="ラウンド自体が不成立">·</td>')
                continue
            if agent.name not in found.extractions:
                cells.append(
                    f'<td class="cell absent" title="{e(agent.name)} / R{number}: 提出なし">—</td>'
                )
                continue
            amount = found.extractions[agent.name]
            t = amount / run.max_extraction if run.max_extraction else 0
            # A round the helper also submitted in cannot be read as the
            # agent's own action, so the cell says so rather than looking clean.
            claimed = [s for s in suspect.get(number, []) if s.name == agent.name]
            mark = " suspect" if claimed else ""
            tip = f"{agent.name} / R{number}: {amount} 単位"
            if claimed:
                tip += "／helper が同ラウンドで同名の提出をログに記録"
            cells.append(
                f'<td class="cell taken{mark}" style="--t:{t:.2f}" '
                f'title="{e(tip)}">{amount}{"*" if claimed else ""}</td>'
            )
        submitted = sum(
            1 for n in numbers
            if (r := run.round_by_number(n)) and agent.name in r.extractions
        )
        total = sum(
            r.extractions.get(agent.name, 0) for r in run.rounds
        )
        rows.append(
            f"<tr><th scope=\"row\"><span class=\"who\">{e(agent.name)}</span>"
            f'<span class="disposition">{e(agent.disposition)}</span></th>'
            + "".join(cells)
            + f'<td class="summary">{submitted}/{len(numbers)}</td>'
            f'<td class="summary">{total}</td></tr>'
        )

    foot_cells = []
    for number in numbers:
        found = run.round_by_number(number)
        count = len(found.extractions) if found else 0
        tone = "zero" if count == 0 else ("thin" if count < len(run.agents) else "full")
        foot_cells.append(f'<td class="count {tone}">{count}/{len(run.agents)}</td>')
    pool_cells = []
    for number in numbers:
        found = run.round_by_number(number)
        if found is None:
            pool_cells.append('<td class="pool">—</td>')
        elif found.rationed:
            pool_cells.append(
                f'<td class="pool" title="要求合計 &gt; プール残量 → 比例配分">'
                f"{found.pool_after} ⚖</td>"
            )
        else:
            pool_cells.append(f'<td class="pool">{found.pool_after}</td>')

    return (
        '<table class="matrix">'
        f"<thead><tr><th>エージェント</th>{header}"
        "<th>提出</th><th>累計</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody>"
        f'<tfoot><tr><th scope="row">提出数</th>{"".join(foot_cells)}<td colspan="2"></td></tr>'
        f'<tr><th scope="row">ラウンド後プール</th>{"".join(pool_cells)}<td colspan="2"></td></tr></tfoot>'
        "</table>"
    )


def run_legend(runs: list[Run]) -> str:
    """Name the lines. Without this a multi-run chart is decoration."""
    if len(runs) < 2:
        return ""
    keys = "".join(
        f'<span class="key c{index % 8}">{e(run.label)}</span>'
        for index, run in enumerate(runs)
    )
    return f'<div class="legend">{keys}</div>'


def pool_panel(runs: list[Run]) -> str:
    x_max = max(len(run.round_numbers) for run in runs)
    y_max = max(run.initial_pool for run in runs)
    series = [
        (run.label, [(float(x), float(y)) for x, y in run.pool_trajectory()], f"c{i % 8}")
        for i, run in enumerate(runs)
    ]
    band = None
    band_label = ""
    if len(runs) == 1:
        run = runs[0]
        # What the pool would do if every agent drew its maximum every round.
        # The gap between this and the actual line is how far the run is from
        # exercising the scenario at all.
        level = float(run.initial_pool)
        drain = len(run.agents) * run.max_extraction
        band = [(0.0, level)]
        for number in run.round_numbers:
            level = max(0.0, level - drain)
            band.append((float(number), level))
        band_label = "全員が上限まで採取した場合"
    markers = []
    if len(runs) == 1:
        for number in runs[0].round_numbers:
            if runs[0].round_by_number(number) is None:
                markers.append((float(number), "未成立"))
    return line_chart(
        series, x_max=x_max, y_max=y_max, x_label="ラウンド", y_label="プール残量",
        markers=markers, band=band, band_label=band_label,
    ) + run_legend(runs)


def participation_panel(runs: list[Run]) -> str:
    x_max = max(len(run.round_numbers) for run in runs)
    series = [
        (
            run.label,
            [(float(n), 100.0 * c / max(1, len(run.agents)))
             for n, c in run.participation_per_round()],
            f"c{i % 8}",
        )
        for i, run in enumerate(runs)
    ]
    return line_chart(
        series, x_max=x_max, y_max=100, x_label="ラウンド", y_label="参加率 (%)"
    ) + run_legend(runs)


def payoff_panel(run: Run) -> str:
    numbers = [0] + run.round_numbers
    y_max = max(
        (max(run.cumulative_payoff(a.name)) for a in run.agents), default=1
    )
    series = [
        (
            agent.name,
            list(zip((float(n) for n in numbers), map(float, run.cumulative_payoff(agent.name)))),
            f"c{index % 8}",
        )
        for index, agent in enumerate(run.agents)
    ]
    legend = "".join(
        f'<span class="key c{index % 8}">{e(agent.name)}</span>'
        for index, agent in enumerate(run.agents)
    )
    chart = line_chart(
        series, x_max=max(numbers), y_max=max(y_max, 1),
        x_label="ラウンド", y_label="累積ペイオフ",
    )
    return f'{chart}<div class="legend">{legend}</div>'


def compare_cell(run: Run, agent: Agent, number: int) -> tuple[str, str, str]:
    """Classify one self-report against the replay. Returns (class, text, tooltip).

    One function so the heading's count and the table's colours cannot drift
    apart: both read this.
    """
    answer = run.answers.get(number, {}).get(agent.name)
    found = run.round_by_number(number)
    actual = found.extractions.get(agent.name) if found else None
    if answer is None:
        return "none", "—", ""
    if not answer.ok:
        return "unparsed", "解析不能", answer.text

    reported = answer.value
    tip = f"申告 {reported} / 実測 {'なし' if actual is None else actual}"
    # A number the rules do not allow is not a misremembered amount, it is an
    # answer that was never about the question. Counting it as a discrepancy
    # would overstate how well the questionnaire works.
    if (
        run.max_extraction
        and isinstance(reported, int)
        and not isinstance(reported, bool)
        and (reported < 0 or reported > run.max_extraction)
    ):
        return (
            "impossible",
            f"{reported}?",
            f"{tip}／1〜{run.max_extraction} の範囲外 ➜ 質問に答えていない",
        )
    if found is None:
        tip += "／誰も提出せず・環境側で不成立"
    if answer.reason:
        tip += f"\n理由: {answer.reason}"

    if actual is None:
        # Reporting an amount for a round it never submitted in is the cell this
        # panel exists for.
        klass = "phantom" if isinstance(reported, int) and reported > 0 else "agree"
        return klass, f"{reported} / —", tip
    if reported == actual:
        return "agree", f"{reported}", tip
    if found.rationed and isinstance(reported, int) and reported > actual:
        # The agent asked for more than the pool could give. That is the
        # environment rationing, not the agent misreporting.
        return "rationed", f"{reported} / {actual}", tip + "／プール枯渇 → 比例配分"
    return "differ", f"{reported} / {actual}", tip


def report_stats(run: Run) -> dict[str, int]:
    """Count the self-report cells by class.

    ``mismatches`` counts every cell where the self-report and the replay
    disagree, including the phantom ones, so the panel's "うち" relation holds.
    """
    counts = {"mismatches": 0, "phantom": 0, "rationed": 0, "impossible": 0, "cells": 0}
    for agent in run.agents:
        for number in run.round_numbers:
            klass, _shown, _tip = compare_cell(run, agent, number)
            if klass == "none":
                continue
            counts["cells"] += 1
            if klass == "phantom":
                counts["phantom"] += 1
                counts["mismatches"] += 1
            elif klass in ("differ", "unparsed"):
                counts["mismatches"] += 1
            elif klass == "rationed":
                counts["rationed"] += 1
            elif klass == "impossible":
                counts["impossible"] += 1
    return counts


def reported_panel(run: Run) -> str:
    """Self-reported extraction against what the environment recorded.

    Deliberately a table rather than the scatter the issue suggests: with a
    handful of points a scatter hides exactly the cell that matters, the agent
    that reports a number for a round in which it never submitted.
    """
    if not run.answers:
        return '<p class="empty">questionnaire の artifact: なし</p>'

    numbers = run.round_numbers
    head = "".join(f"<th>R{n}</th>" for n in numbers)
    rows = []
    stats = report_stats(run)
    mismatches = stats["mismatches"]
    phantom = stats["phantom"]
    rationed = stats["rationed"]
    impossible = stats["impossible"]
    for agent in run.agents:
        cells = []
        for number in numbers:
            klass, shown, tip = compare_cell(run, agent, number)
            title = f' title="{e(tip)}"' if tip else ""
            cells.append(f'<td class="cmp {klass}"{title}>{e(shown)}</td>')
        rows.append(
            f'<tr><th scope="row">{e(agent.name)}</th>{"".join(cells)}</tr>'
        )
    lines: list = [
        f"申告と実測が食い違ったセル: {mismatches} 件",
        [
            f"うち「提出していないのに数値を申告した」セル: {phantom} 件",
            "➜ エージェントが自分の行動を誤って記憶している証拠",
        ],
    ]
    if rationed:
        lines += [
            f"プール枯渇による比例配分で申告（要求量）と実測（配分量）がずれたセル: "
            f"{rationed} 件",
            ["食い違いには数えていない"],
        ]
    if impossible:
        lines += [
            f"1〜{run.max_extraction} の範囲外を答えたセル: {impossible} 件（末尾に ?）",
            [
                "採取量の記憶違いではなく、質問に答えていない回答",
                "➜ 自己申告の精度を測る材料にはならない",
            ],
        ]
    note = bullets(lines, css="note")
    return (
        f'<table class="compare"><thead><tr><th>申告 / 実測</th>{head}</tr></thead>'
        f"<tbody>{''.join(rows)}</tbody></table>{note}"
    )


def provenance_panel(run: Run) -> str:
    """Is a recorded extraction really that agent's?

    Three ways it can fail to be, in decreasing order of how visible they are:
    the helper submitted it under a name it chose; an agent wrote a name the
    profile table does not contain, opening an account belonging to nobody; an
    agent wrote a placeholder instead of a number and the environment silently
    clamped the request to 1. None of them raises.
    """
    blocks: list[str] = []
    unknown = run.unknown_takers
    suspect = run.suspect_rounds

    if run.helper_active:
        rows = "".join(
            f"<tr><td>R{number}</td><td>{e(s.name)}</td><td>{s.amount}</td></tr>"
            for number in sorted(suspect)
            for s in suspect[number]
        )
        table = (
            '<table class="plain"><thead><tr><th>ラウンド</th><th>名乗った名前</th>'
            f"<th>単位</th></tr></thead><tbody>{rows}</tbody></table>"
            if rows
            else ""
        )
        blocks.append(
            bullets(
                [
                    "このランには <code>ask</code> / <code>intervene</code> ステップがある",
                    [
                        "実行するのは AgentSocietyHelper",
                        "helper 自身が <code>ask_environment</code> 経由で環境に到達 → "
                        "<code>agent_name</code> を自分で決めて採取を提出",
                    ],
                    "➜ 記録された採取はどれも、そのエージェント本人のものだと断定できない",
                ],
                css="note alarm",
            )
            + (
                bullets(
                    [
                        "ログから helper の提出だと特定できた行",
                        ["helper の出力は自由文 ➜ これが全部とは限らない"],
                    ],
                    css="note",
                )
                + table
                if rows
                else bullets(
                    [
                        "helper の提出だと名指しで特定できた行: ログになし",
                        "特定できない ≠ なかった",
                    ],
                    css="note",
                )
            )
        )

    if unknown:
        rows = "".join(
            f"<tr><td>{e(name)}</td><td>{', '.join('R%d' % n for n in numbers)}</td></tr>"
            for name, numbers in sorted(unknown.items())
        )
        blocks.append(
            bullets(
                [
                    "profile に存在しない名前で採取が記録されている",
                    [
                        "環境は <code>agent_name</code> を検証しない",
                        "<code>Agent 3</code> のような書き損じ → 誰のものでもない口座が開く",
                        "プールからは確かに減る",
                    ],
                ],
                css="note alarm",
            )
            + '<table class="plain"><thead><tr><th>記録された名前</th><th>ラウンド</th>'
            f"</tr></thead><tbody>{rows}</tbody></table>"
        )
    else:
        blocks.append(
            bullets(
                [
                    "記録された採取の名前: すべて <code>core_agent_profile</code> と一致",
                    "この照合が捉えられるのは書き損じだけ",
                    ["➜ 実在する名前を騙った提出は素通りする"],
                ],
                css="note",
            )
        )

    if run.log.wordings:
        items = "".join(
            f"<li><code>R{w.round if w.round else '-'} / {e(w.phase_kind or '?')}</code> "
            f"{e(w.text[:160])}</li>"
            for w in run.log.wordings[:30]
        )
        blocks.append(
            f"<details><summary>エージェントが実際に環境へ言った言葉 "
            f"({len(run.log.wordings)} 件)</summary>"
            + bullets(
                [
                    "embedding モデルなし → ask_env が毎回 codegen テンプレートの"
                    "キャッシュミス → 言葉がログに残る",
                    "embedding を設定すると、この一覧は消える",
                ],
                css="note",
            )
            + f'<ul class="loglines">{items}</ul></details>'
        )

    return "".join(blocks)


def ask_panel(run: Run) -> str:
    """Distribution of ask_env round-trip times.

    Not charted per round on purpose: these lines carry no timestamp and Ray
    flushes them out of band, so their position in the file does not say which
    round they belong to. The durations are still durations.
    """
    asks = run.log.env_asks
    if not asks:
        return ""
    ordered = sorted(asks, key=lambda a: -a.seconds)
    total = sum(a.seconds for a in asks)
    rows = "".join(
        f"<tr><td>{a.seconds:.1f}s</td><td class=\"said\">{e(a.instruction)}…</td></tr>"
        for a in ordered[:8]
    )
    median = sorted(a.seconds for a in asks)[len(asks) // 2]
    return (
        bullets(
            [
                f"ask_env の往復: {len(asks)} 回 / 合計 {total:.0f} 秒",
                [f"中央値 {median:.1f}s / 最大 {ordered[0].seconds:.1f}s"],
                "➜ エージェント 1 体あたりの待ち時間はここが支配する",
            ],
            css="note",
        )
        + '<table class="plain"><thead><tr><th>所要</th><th>リクエストの冒頭</th>'
        f"</tr></thead><tbody>{rows}</tbody></table>"
    )


def failure_panel(run: Run) -> str:
    if run.log_missing:
        return '<p class="empty">sim.log なし ➜ 失敗の内訳は出せない</p>'
    if not run.failures:
        return '<p class="empty">ReAct レベルの失敗: 記録なし</p>'

    by_agent: dict[int, dict[str, int]] = {}
    by_category: dict[str, int] = {}
    for failure in run.failures:
        by_agent.setdefault(failure.agent_id, {})
        by_agent[failure.agent_id][failure.category] = (
            by_agent[failure.agent_id].get(failure.category, 0) + 1
        )
        by_category[failure.category] = by_category.get(failure.category, 0) + 1

    categories = sorted(by_category, key=lambda k: -by_category[k])
    head = "".join(f"<th>{e(category_label(c))}</th>" for c in categories)
    rows = []
    for agent in run.agents:
        counts = by_agent.get(agent.id, {})
        submitted = sum(1 for r in run.rounds if agent.name in r.extractions)
        cells = "".join(
            f'<td class="{"hit" if counts.get(c) else "miss"}">{counts.get(c, 0) or ""}</td>'
            for c in categories
        )
        rows.append(
            f'<tr><th scope="row">{e(agent.name)}</th>{cells}'
            f'<td class="summary">{submitted}/{len(run.round_numbers)}</td></tr>'
        )

    lines = "".join(
        f"<li><code>R{f.round if f.round else '?'} / Agent-{f.agent_id} / {e(f.action)}</code> "
        f"{e(f.message[:180])}</li>"
        for f in run.failures[:40]
    )
    return (
        f'<table class="failures"><thead><tr><th>エージェント</th>{head}<th>提出</th></tr></thead>'
        f"<tbody>{''.join(rows)}</tbody></table>"
        f"<details><summary>sim.log の該当行 ({len(run.failures)} 件中 "
        f"{min(40, len(run.failures))} 件)</summary><ul class=\"loglines\">{lines}</ul></details>"
    )


def timing_panel(run: Run) -> str:
    if run.log_missing or not run.phases:
        return '<p class="empty">sim.log なし ➜ 所要時間は出せない</p>'
    per_round: dict[int, float] = {}
    for phase in run.phases:
        if phase.round is None:
            continue
        per_round[phase.round] = per_round.get(phase.round, 0.0) + phase.seconds
    if not per_round:
        return '<p class="empty">ラウンド境界を sim.log から読み取れず</p>'
    bars = [(f"R{n}", per_round[n]) for n in sorted(per_round)]
    total = sum(per_round.values())
    return bar_chart(bars, y_label="所要時間 (秒)") + bullets(
        [f"ラウンドに帰属できた時間の合計: {total / 60:.1f} 分"],
        css="note",
    )


def appendix_panel(run: Run) -> str:
    agents = "".join(
        f"<tr><td>{agent.id}</td><td>{e(agent.name)}</td><td>{e(agent.age)}</td>"
        f"<td>{e(agent.disposition)}</td></tr>"
        for agent in run.agents
    )
    instructions = "".join(
        f"<li><strong>R{number}</strong> {e(text)}</li>"
        for number, text in sorted(run.instructions.items())
    )
    # Whichever of the two the run used: the per-round intervene text, or — once
    # the intervene step is gone — the standing orders carried in every profile.
    if instructions:
        wording = f'<ol class="instructions">{instructions}</ol>'
    elif run.standing_orders:
        wording = (
            '<p class="instructions"><strong>全ラウンド共通の指示（profile 内）</strong><br>'
            f"{e(run.standing_orders)}</p>"
        )
    else:
        wording = ""
    return (
        "<details><summary>エージェント一覧と、エージェントに与えられた指示文</summary>"
        '<table class="plain"><thead><tr><th>id</th><th>name</th><th>age</th>'
        f"<th>disposition</th></tr></thead><tbody>{agents}</tbody></table>"
        f"{wording}</details>"
    )


# --------------------------------------------------------------------------
# Document
# --------------------------------------------------------------------------

CSS = """
:root{
  --bg:#fbfbfa; --surface:#ffffff; --ink:#1b1b19; --muted:#6b6b66;
  --line:#e3e2de; --heat:#b3541e; --ok:#2f7d4f; --warn:#a5761b; --bad:#b02a2a;
  --absent:#cfcec9;
  --c0:#3d6ea8; --c1:#b3541e; --c2:#2f7d4f; --c3:#8a4f9e;
  --c4:#a5761b; --c5:#4a8d99; --c6:#b02a56; --c7:#5c6570;
}
@media (prefers-color-scheme: dark){
  :root{
    --bg:#17181a; --surface:#1f2124; --ink:#e8e7e3; --muted:#9b9a94;
    --line:#33353a; --heat:#e08a4a; --ok:#67b98a; --warn:#d6ab5a; --bad:#e06a6a;
    --absent:#494b50;
    --c0:#7fb0e0; --c1:#e08a4a; --c2:#67b98a; --c3:#c093d6;
    --c4:#d6ab5a; --c5:#7fc4cf; --c6:#e0748f; --c7:#a3adb8;
  }
}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);
  font:15px/1.65 -apple-system,BlinkMacSystemFont,"Hiragino Sans","Noto Sans JP",sans-serif;}
main{max-width:920px;margin:0 auto;padding:32px 20px 80px}
h1{font-size:22px;margin:0 0 4px}
h2{font-size:17px;margin:44px 0 6px;padding-top:14px;border-top:1px solid var(--line)}
h3{font-size:14px;margin:26px 0 4px;color:var(--muted);font-weight:600}
p{margin:6px 0 14px}
.lede,.note,.empty{color:var(--muted);font-size:13.5px}

/* Telegraph outlines. Both levels carry a visible marker: the shape of the
   text is a bullet outline, so it has to read as one at a glance rather than
   as prose that happens to be broken into lines. The supporting level keeps a
   thin rule as well, so the single indent step survives a narrow window. */
ul.lede,ul.note,ul.meta{list-style:disc outside;margin:6px 0 16px;padding:0 0 0 17px;
  line-height:1.85}
ul.lede>li,ul.note>li,ul.meta>li{margin:0}
ul.lede>li::marker,ul.note>li::marker,ul.meta>li::marker{color:var(--muted)}
/* A nested list is a child of the li that owns it, so the wrapper li must not
   draw a marker of its own — the marker belongs to the lines inside. */
ul.lede>li.sub,ul.note>li.sub,ul.meta>li.sub{margin:1px 0 5px;list-style:none}
ul.lede li.sub>ul,ul.note li.sub>ul,ul.meta li.sub>ul{list-style:circle outside;margin:0;
  padding:0 0 0 17px;border-left:1px solid var(--line)}
ul.lede li.sub>ul>li,ul.note li.sub>ul>li,ul.meta li.sub>ul>li{opacity:.85}
ul.lede li.sub>ul>li::marker,ul.note li.sub>ul>li::marker,ul.meta li.sub>ul>li::marker{color:var(--muted)}
ul.note{font-size:13px;margin:6px 0 12px}
ul.note.alarm{color:var(--bad);border-left:3px solid var(--bad);padding-left:11px}
ul.note.alarm li.sub>ul{border-left-color:var(--bad);opacity:.9}
ul.meta{font-size:13px;margin:6px 0 24px}
.meta code{font-size:12.5px}

.verdict{font-size:16px;font-weight:700;padding:10px 14px;border-radius:6px;
  border-left:4px solid;background:var(--surface)}
.verdict.ok{border-color:var(--ok);color:var(--ok)}
.verdict.warn{border-color:var(--warn);color:var(--warn)}
.verdict.bad{border-color:var(--bad);color:var(--bad)}
.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));
  gap:10px;margin:12px 0 4px}
.tile{background:var(--surface);border:1px solid var(--line);border-radius:6px;padding:10px 12px}
.tile-label{font-size:12px;color:var(--muted)}
.swatch{display:inline-block;width:12px;height:3px;margin-right:5px;
  vertical-align:middle;background:currentColor}
.tile.c0 .swatch{color:var(--c0)} .tile.c1 .swatch{color:var(--c1)}
.tile.c2 .swatch{color:var(--c2)} .tile.c3 .swatch{color:var(--c3)}
.tile.c4 .swatch{color:var(--c4)} .tile.c5 .swatch{color:var(--c5)}
.tile.c6 .swatch{color:var(--c6)} .tile.c7 .swatch{color:var(--c7)}
.tile-value{font-size:22px;font-weight:700;line-height:1.25;
  font-variant-numeric:tabular-nums}
.tile-note{font-size:11.5px;color:var(--muted)}

table{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}
.scroll{overflow-x:auto}
th,td{border:1px solid var(--line);padding:5px 8px;text-align:center;font-size:13px}
thead th,tfoot th{background:var(--surface);font-weight:600;color:var(--muted)}
.failures thead th:first-child,.failures tbody th,
.failures thead th:last-child{white-space:nowrap}
tbody th[scope=row]{text-align:left;font-weight:600;white-space:nowrap}
.matrix .who{display:block}
.matrix .disposition{display:block;font-weight:400;font-size:11px;color:var(--muted);
  max-width:210px;white-space:normal}
.cell{font-weight:600;min-width:42px}
.cell.taken{background-color:color-mix(in oklab,var(--heat) calc(var(--t)*72%),var(--surface))}
.cell.absent{color:var(--muted);background:repeating-linear-gradient(-45deg,
  transparent 0 4px,var(--absent) 4px 5px)}
.cell.dead{color:var(--muted);background:var(--absent)}
.cell.suspect{outline:2px dashed var(--bad);outline-offset:-3px}
.alarm{color:var(--bad);font-size:13.5px;border-left:3px solid var(--bad);
  padding-left:10px;margin:10px 0}
.said{text-align:left;font-size:12px;color:var(--muted)}
.count.zero{color:var(--bad);font-weight:700}
.count.thin{color:var(--warn);font-weight:600}
.count.full{color:var(--ok);font-weight:600}
.summary{font-weight:600}
.pool{color:var(--muted)}

.cmp.agree{color:var(--ok)}
.cmp.differ{color:var(--warn);font-weight:600}
.cmp.phantom{color:var(--bad);font-weight:700}
.cmp.rationed{color:var(--muted)}
.cmp.impossible{color:var(--bad);font-weight:700;
  background:repeating-linear-gradient(-45deg,transparent 0 5px,var(--absent) 5px 6px)}
.cmp.unparsed,.cmp.none{color:var(--muted);font-size:11.5px}
.failures .hit{font-weight:700;color:var(--bad)}
.failures .miss{color:var(--muted)}

.chart{width:100%;height:auto;display:block;margin:10px 0 4px;overflow:visible}
.axis{stroke:var(--line);stroke-width:1}
.grid{stroke:var(--line);stroke-width:1;stroke-dasharray:2 4}
.tick,.axis-label,.bar-value,.band-label,.marker-label{fill:var(--muted);font-size:10px;
  font-family:inherit}
.bar-value{fill:var(--ink)}
.series{fill:none;stroke-width:2;stroke-linejoin:round;stroke-linecap:round}
.band{fill:var(--absent);opacity:.28}
.marker{stroke:var(--bad);stroke-width:1;stroke-dasharray:3 3}
.marker-label{fill:var(--bad)}
.bar{fill:var(--c0);opacity:.85}
.legend{display:flex;flex-wrap:wrap;gap:12px;font-size:12px;color:var(--muted)}
.key::before{content:"";display:inline-block;width:14px;height:3px;
  margin-right:5px;vertical-align:middle;background:currentColor}
.c0,.key.c0{stroke:var(--c0);fill:var(--c0);color:var(--c0)}
.c1,.key.c1{stroke:var(--c1);fill:var(--c1);color:var(--c1)}
.c2,.key.c2{stroke:var(--c2);fill:var(--c2);color:var(--c2)}
.c3,.key.c3{stroke:var(--c3);fill:var(--c3);color:var(--c3)}
.c4,.key.c4{stroke:var(--c4);fill:var(--c4);color:var(--c4)}
.c5,.key.c5{stroke:var(--c5);fill:var(--c5);color:var(--c5)}
.c6,.key.c6{stroke:var(--c6);fill:var(--c6);color:var(--c6)}
.c7,.key.c7{stroke:var(--c7);fill:var(--c7);color:var(--c7)}
polyline.series{fill:none}

details{margin:10px 0;font-size:13px}
summary{cursor:pointer;color:var(--muted)}
.loglines{font-size:12px;line-height:1.7;padding-left:18px}
.loglines code{background:var(--surface);border:1px solid var(--line);
  border-radius:3px;padding:0 4px;margin-right:6px;white-space:nowrap}
.instructions{font-size:12.5px;color:var(--muted);line-height:1.6}
.plain td{text-align:left}
"""


def payoff_extremes(run: Run) -> tuple[tuple[str, int], tuple[str, int]] | None:
    """Highest- and lowest-earning agent, or None when nobody earned anything."""
    totals = [
        (agent.name, sum(r.extractions.get(agent.name, 0) for r in run.rounds))
        for agent in run.agents
    ]
    if not totals or not any(t for _n, t in totals):
        return None
    ordered = sorted(totals, key=lambda pair: -pair[1])
    return ordered[0], ordered[-1]


def round_seconds(run: Run) -> dict[int, float]:
    per_round: dict[int, float] = {}
    for phase in run.phases:
        if phase.round is not None:
            per_round[phase.round] = per_round.get(phase.round, 0.0) + phase.seconds
    return per_round


def headline(run: Run, panel: int, runs: list[Run] | None = None) -> str:
    """The number that belongs in a panel's heading.

    Every value is computed from the run, and a panel with nothing to report
    says so rather than showing a zero that reads like a measurement.
    """
    if panel == 1:
        submitted = sum(len(r.extractions) for r in run.rounds)
        return f"{submitted}/{len(run.agents) * len(run.round_numbers)} 機会が提出"
    if panel == 2:
        if runs and len(runs) > 1:
            # One arrow means one thing in these headings: movement along the
            # axis the chart below plots. This panel plots rounds, so the runs
            # are separated the way panel 3 separates them, with a slash. The
            # coloured tiles above the chart already fix the run order.
            return " / ".join(f"{r.participation * 100:.0f}%" for r in runs)
        per_round = run.participation_per_round()
        if not per_round:
            return "記録なし"
        size = max(1, len(run.agents))
        return (
            f"{100 * per_round[0][1] / size:.0f}% → {100 * per_round[-1][1] / size:.0f}%"
        )
    if panel == 3:
        def final_pool(one: Run) -> int:
            return one.rounds[-1].pool_after if one.rounds else one.initial_pool

        if runs and len(runs) > 1:
            # Shared start, one ending per run: "100 → 84 / 43 / 20".
            ends = " / ".join(str(final_pool(r)) for r in runs)
            return f"{run.initial_pool} → {ends}"
        return f"{run.initial_pool} → {final_pool(run)}"
    if panel == 4:
        extremes = payoff_extremes(run)
        if extremes is None:
            return "全員 0 点"
        (top_name, top), (low_name, low) = extremes
        if top_name == low_name:
            return f"{top_name} のみ {top} 点"
        return f"最多 {top_name} {top} 点 / 最少 {low_name} {low} 点"
    if panel == 5:
        if not run.answers:
            return "記録なし"
        stats = report_stats(run)
        return f"食い違い {stats['mismatches']}/{stats['cells']} セル"
    if panel == 6:
        if run.log_missing:
            return "sim.log なし"
        return f"ReAct 失敗 {len(run.failures)} 件"
    if panel == 7:
        if run.log_missing:
            return "sim.log なし"
        total = sum(round_seconds(run).values())
        if not total:
            return "測定できず"
        return f"{total / 60:.1f} 分"
    return ""


def bullets(lines: Lines, css: str = "lede") -> str:
    """Render a telegraph-style outline.

    ``lines`` is a list of strings; a nested list holds the lines that support
    the string before it, which is the one indent level the outline needs. Items
    are pre-escaped by the caller where they carry ``<code>``.
    """
    if not lines:
        return ""
    out = [f'<ul class="{css}">']
    for item in lines:
        if isinstance(item, (list, tuple)):
            out.append(f"<li class=\"sub\"><ul>{''.join(f'<li>{x}</li>' for x in item)}</ul></li>")
        else:
            out.append(f"<li>{item}</li>")
    out.append("</ul>")
    return "".join(out)


def section(title: str, lede: Lines, body: str) -> str:
    return f"<h2>{e(title)}</h2>{bullets(lede)}{body}"


# A scenario name a reader can use, plus the rules in plain words. Anything not
# listed falls back to the class name: better an unfamiliar identifier than a
# title asserting a scenario the run did not use.
SCENARIOS: dict[str, str] = {
    "CommonsTragedy": "共有地の悲劇",
    "SimpleSocialSpace": "グループ会話",
}


def scenario_name(run: "Run") -> str:
    return SCENARIOS.get(run.env_module, run.env_module or "不明な環境")


def rules_lines(run: "Run") -> Lines:
    """State the scenario from the run's own numbers.

    Without this the report is a page of percentages about a game it never
    explains: "参加率 81%" and "プール 100 → 20" only mean something once the
    reader knows what a round is and what taking costs. Every value is read
    back out of the run, so a report can never describe a game other than the
    one it is reporting on.
    """
    if run.env_module != "CommonsTragedy":
        return []
    others = max(0, len(run.agents) - 1)
    return [
        f"共有プール {run.initial_pool} 単位を {len(run.agents)} エージェントで分け合う",
        [
            f"1 ラウンドの採取上限: 1 体あたり {run.max_extraction} 単位",
            "採取 1 単位 = 1 点 / プールは補充されない",
            "全員の要求がプール残量を超えた場合は、要求量に比例して配分",
        ],
        f"{run.planned_rounds or len(run.round_numbers)} ラウンド"
        f"・各ラウンドで全員が独立に採取量を決める",
        [
            f"自制すれば全員が長く採れる ⇔ 取り急げば {others} 体の取り分が減る",
            "➜ このレポートが見るのは、エージェントがどちらを選んだか",
        ],
    ]


def meta_lines(runs: list["Run"], generated: str) -> Lines:
    """Identify the runs without repeating what they share.

    A run directory is named after its Grid Engine job, and that is its
    identity; the path it happens to sit at is the reader's least useful fact
    about it, and on a fetched copy it is an absolute temp path. Where the runs
    agree on the setup — the usual case, since a comparison is only meaningful
    when they do — the setup is stated once and the ids hang under it.
    """
    def setup(run: "Run") -> str:
        return (
            f"{len(run.agents)} エージェント / {len(run.round_numbers)} ラウンド"
            f" / プール {run.initial_pool}"
        )

    shared = setup(runs[0]) if len({setup(r) for r in runs}) == 1 else None
    ids = [
        (f"{e(run.label)}: " if run.label != run.path.name else "")
        + f"<code>{e(run.path.name)}</code>"
        + ("" if shared else f"（{setup(run)}）")
        for run in runs
    ]

    if len(runs) == 1 and shared:
        return [f"{ids[0]}（{shared}）", f"生成: {e(generated)}"]
    if shared:
        return [f"{len(runs)} ラン比較: {shared}", ids, f"生成: {e(generated)}"]
    return [*ids, f"生成: {e(generated)}"]


def render(runs: list[Run]) -> str:
    primary = runs[0]
    generated = datetime.now().strftime("%Y-%m-%d %H:%M")
    # The scenario is whatever the run recorded, not whatever this script was
    # written against. A SimpleSocialSpace run titled 共有地の悲劇 would be a
    # report asserting a game that never happened.
    names = sorted({scenario_name(run) for run in runs})
    scenario = names[0] if len(names) == 1 else " / ".join(names)
    title = (
        f"{scenario} — {primary.label}"
        if len(runs) == 1
        else f"{scenario} — {len(runs)} ラン比較"
    )

    parts = [
        "<!doctype html><html lang='ja'><head><meta charset='utf-8'>",
        "<meta name='viewport' content='width=device-width,initial-scale=1'>",
        f"<title>{e(title)}</title><style>{CSS}</style></head><body><main>",
        f"<h1>{e(title)}</h1>",
        bullets(meta_lines(runs, generated), css="meta"),
    ]

    # The rules belong before the numbers: every figure below is a quantity in
    # a game, and a reader who has not been told the game cannot read them.
    if len(names) == 1:
        rules = rules_lines(primary)
        if rules:
            parts.append(f"<h2>{e(scenario)}のルール</h2>{bullets(rules)}")

    if len(runs) == 1:
        parts.append(verdict_band(primary))
    else:
        parts.append(comparison_band(runs))
        parts.append(
            '<div class="tiles">'
            + "".join(
                f'<div class="tile c{index % 8}">'
                f'<div class="tile-label"><span class="swatch"></span>{e(run.label)}</div>'
                f'<div class="tile-value">{run.participation * 100:.0f}%</div>'
                f'<div class="tile-note">参加率・成立 {len(run.rounds)}/'
                f"{len(run.round_numbers)}</div></div>"
                for index, run in enumerate(runs)
            )
            + "</div>"
        )

    if len(runs) == 1:
        parts.append(
            section(
                f"1. 参加マトリクス: {headline(primary, 1)}",
                [
                    "行: エージェント / 列: ラウンド ➜ 誰がどのラウンドで実際に採取したか",
                    ["斜線: 提出なし", "灰色: ラウンド自体が不成立"],
                    "採取量のヒートマップと参加率を 1 枚に統合",
                    [
                        "同じ行列の縦読み ⇔ 横読み",
                        "2 枚に割ると、1 つの問いに 2 図を往復させることになる",
                    ],
                ],
                f'<div class="scroll">{participation_matrix(primary)}</div>'
                + provenance_panel(primary),
            )
        )

    # Numbered headings only make sense when there is one run; a comparison
    # interleaves shared-axis panels with per-run ones and the numbers stop
    # matching the reading order.
    n = (lambda i, title: f"{i}. {title}" if len(runs) == 1 else title)

    parts.append(
        section(
            n(2, f"参加率の推移: {headline(primary, 2, runs)}"),
            [
                "ラウンドごとの提出割合",
                "shunk031/tsubame-agentsociety2#2 が問題にしている量そのもの",
            ],
            participation_panel(runs),
        )
    )

    parts.append(
        section(
            n(3, f"プールの減り方: {headline(primary, 3, runs)}"),
            # The shaded floor is drawn for a single run only; two runs would
            # need two bands. Say what is actually on the chart.
            [
                "実線: 実際の残量 / 網かけ: 全員が毎ラウンド上限まで採取した場合の下限",
                ["➜ 両者の差 = このランがシナリオをどれだけ動かせていないか"],
            ]
            if len(runs) == 1
            else [
                "ラン別の実際の残量",
                "上限まで採取した場合の下限は、ラン 1 件のときだけ網かけで重ねる",
            ],
            pool_panel(runs),
        )
    )

    for run in runs:
        prefix = "" if len(runs) == 1 else f"【{run.label}】"
        if len(runs) > 1:
            parts.append(
                section(
                    f"{prefix}参加マトリクス: {headline(run, 1)}",
                    ["行: エージェント / 列: ラウンド ➜ 誰がどのラウンドで実際に採取したか"],
                    f'<div class="scroll">{participation_matrix(run)}</div>'
                    + provenance_panel(run),
                )
            )
        parts.append(
            section(
                prefix + n(4, f"累積ペイオフ: {headline(run, 4)}"),
                [
                    "多く採ったエージェントが最後に得をしたか",
                    "採取 1 単位 = 1 点",
                ],
                payoff_panel(run),
            )
        )
        parts.append(
            section(
                prefix + n(5, f"申告と実測: {headline(run, 5)}"),
                [
                    "左: questionnaire の自己申告 / 右: replay の実測",
                    "赤: 提出していないのに数値を申告したセル ➜ この表で最初に見る場所",
                    "範囲外の回答は別枠で集計",
                    ["理由: 採取量の記憶違いではなく、質問に答えていない回答"],
                ],
                reported_panel(run),
            )
        )
        parts.append(
            section(
                prefix + n(6, f"動かなかった理由: {headline(run, 6)}"),
                [
                    "sim.log の ReAct 失敗: エージェント別 × 原因別",
                    "参加率が低いとき、モデル能力 / 指示文 / ツール定義 を切り分ける材料",
                ],
                failure_panel(run),
            )
        )
        parts.append(
            section(
                prefix + n(7, f"ラウンド所要時間: {headline(run, 7)}"),
                [
                    "sim.log のステップ境界から測った実時間",
                    "エージェント数を増やすときの見積もり材料",
                ],
                timing_panel(run) + ask_panel(run),
            )
        )
        parts.append(appendix_panel(run))

    parts.append(so_what(runs))
    parts.append("</main></body></html>")
    return "".join(parts)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dirs", nargs="+", type=Path,
                        help="One or more run directories; several are drawn on shared axes")
    parser.add_argument("-o", "--out", type=Path, default=Path("report.html"))
    parser.add_argument("--label", action="append", default=[],
                        help="Label for each run, in order (default: directory name)")
    args = parser.parse_args()

    labels = list(args.label) + [None] * len(args.run_dirs)
    runs = [load_run(d, labels[i]) for i, d in enumerate(args.run_dirs)]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(render(runs), encoding="utf-8")
    size = args.out.stat().st_size
    print(f"wrote {args.out} ({size / 1024:.1f} KiB)")
    for run in runs:
        print(
            f"  {run.label}: {len(run.agents)} agents, "
            f"{len(run.rounds)}/{len(run.round_numbers)} rounds resolved, "
            f"participation {run.participation * 100:.0f}%, "
            f"{len(run.failures)} ReAct failures"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
