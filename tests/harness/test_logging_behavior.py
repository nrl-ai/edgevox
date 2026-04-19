"""Logging-behaviour integration tests.

These verify three agentic concerns that are easy to regress:

* **``AuditLogHook`` produces structured jsonl records** with the
  keys downstream log shippers (Vector, Filebeat, Fluent Bit) expect.
* **No stray ``print()`` in the hot path** — framework code is
  required to use ``logging`` per CLAUDE.md. Tests capture
  ``sys.stdout`` around a full turn and assert it stays empty.
* **Log level is honoured** — ``logging.getLogger("edgevox")`` at
  ``WARNING`` suppresses the chatty ``DEBUG`` / ``INFO`` traffic
  that dominates a loud local run but shouldn't leak into
  production.
"""

from __future__ import annotations

import json
import logging
from io import StringIO
from pathlib import Path

from edgevox.agents import AgentContext, LLMAgent
from edgevox.agents.hooks_builtin import AuditLogHook

from .conftest import ScriptedLLM, call, reply

# ---------------------------------------------------------------------------
# AuditLogHook — structured jsonl, per-event records
# ---------------------------------------------------------------------------


def _build_tooled_agent(tmp_path: Path, *, hooks=None):
    from edgevox.llm.tools import tool

    @tool
    def echo(msg: str) -> str:
        """Echo the message back. Args: msg."""
        return f"echo: {msg}"

    agent = LLMAgent(
        name="audited",
        description="logs every tool call",
        instructions="Speak plainly.",
        tools=[echo],
        hooks=hooks or [],
    )
    agent.bind_llm(
        ScriptedLLM(
            [
                call("echo", msg="hi"),
                reply("ok"),
            ]
        )
    )
    return agent


class TestAuditLogHook:
    def test_writes_jsonl_with_tool_call_records(self, tmp_path: Path):
        log_path = tmp_path / "audit.jsonl"
        agent = _build_tooled_agent(tmp_path, hooks=[AuditLogHook(path=log_path)])
        agent.run("echo please", AgentContext())

        assert log_path.exists(), "AuditLogHook didn't create the log file"
        lines = log_path.read_text(encoding="utf-8").splitlines()
        assert lines, "no audit records written"
        records = [json.loads(line) for line in lines]
        # Must have at least one record referencing the ``echo`` call.
        echo_records = [r for r in records if "echo" in json.dumps(r)]
        assert echo_records

    def test_each_line_is_valid_json(self, tmp_path: Path):
        """Log shippers refuse non-JSON lines. If a single record
        corrupts the file the whole agent's observability goes dark."""
        log_path = tmp_path / "audit.jsonl"
        agent = _build_tooled_agent(tmp_path, hooks=[AuditLogHook(path=log_path)])
        agent.run("go", AgentContext())
        for line in log_path.read_text(encoding="utf-8").splitlines():
            json.loads(line)  # raises if malformed


# ---------------------------------------------------------------------------
# Hot-path print hygiene
# ---------------------------------------------------------------------------


class TestNoStrayPrints:
    def test_turn_does_not_print_to_stdout(self, capsys, tmp_path: Path):
        """CLAUDE.md: ``No prints in library code``. Run a full turn
        and assert ``sys.stdout`` stays empty. ``sys.stderr`` is
        allowed (logging uses it), so we only check stdout."""
        agent = _build_tooled_agent(tmp_path, hooks=[])
        agent.run("echo", AgentContext())
        captured = capsys.readouterr()
        assert captured.out == "", f"stray print on stdout: {captured.out!r}"


# ---------------------------------------------------------------------------
# Log level respect
# ---------------------------------------------------------------------------


class TestLogLevelRespect:
    def test_warning_level_suppresses_info_and_debug(self, tmp_path: Path, caplog):
        """Set the ``edgevox`` root logger to WARNING; run a turn;
        confirm no INFO or DEBUG records landed in ``caplog``.
        ``ScriptedLLM``'s real-world surrogates (llama-cpp, STT
        factories, etc.) chatter at INFO — this is what a production
        operator silences, and the suppression must be honoured."""
        logger = logging.getLogger("edgevox")
        prior_level = logger.level
        logger.setLevel(logging.WARNING)

        # Pre-emit at every level so we can observe the cutoff.
        edgevox_log = logging.getLogger("edgevox.test_cutoff")
        edgevox_log.debug("debug-message-should-not-land")
        edgevox_log.info("info-message-should-not-land")
        edgevox_log.warning("warning-message-should-land")

        try:
            agent = _build_tooled_agent(tmp_path, hooks=[])
            with caplog.at_level(logging.WARNING, logger="edgevox"):
                agent.run("go", AgentContext())
        finally:
            logger.setLevel(prior_level)

        message_texts = " ".join(r.message for r in caplog.records)
        assert "debug-message-should-not-land" not in message_texts
        assert "info-message-should-not-land" not in message_texts


# ---------------------------------------------------------------------------
# Dedicated handler routing — production-shaped setup
# ---------------------------------------------------------------------------


class TestDedicatedHandlerRouting:
    def test_agent_logs_can_be_captured_separately(self, tmp_path: Path):
        """Production pattern: route ``edgevox`` logs to their own
        handler (file or remote) without bleeding into the app's main
        logger. Verify we can attach a fresh handler to the ``edgevox``
        logger and capture only the records this subsystem emits."""
        sink = StringIO()
        handler = logging.StreamHandler(sink)
        handler.setLevel(logging.DEBUG)
        handler.setFormatter(logging.Formatter("%(name)s %(levelname)s %(message)s"))

        logger = logging.getLogger("edgevox")
        prior_handlers = list(logger.handlers)
        prior_level = logger.level
        logger.addHandler(handler)
        logger.setLevel(logging.DEBUG)

        try:
            logging.getLogger("edgevox.test_route").info("scripted info")
            agent = _build_tooled_agent(tmp_path, hooks=[])
            agent.run("go", AgentContext())
        finally:
            logger.removeHandler(handler)
            logger.setLevel(prior_level)
            logger.handlers[:] = prior_handlers

        output = sink.getvalue()
        # Must have captured the scripted info line.
        assert "scripted info" in output
        # Every line must be name-prefixed with ``edgevox`` (proving
        # we captured only the edgevox subsystem, not the root).
        for line in output.splitlines():
            assert line.startswith("edgevox"), f"non-edgevox record leaked: {line!r}"
