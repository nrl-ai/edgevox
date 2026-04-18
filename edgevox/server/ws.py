"""WebSocket endpoint and per-turn pipeline runner.

Wire format is documented in the plan; in short:
  client → server: text JSON control + binary int16 PCM @ 16 kHz frames
  server → client: text JSON events + binary WAV bytes (one per sentence)

Each speech segment runs the same STT → LLM (streaming) → sentence split → TTS
chain as ``StreamingPipeline.process``. The model singletons live on
``ServerCore``; we swap ``llm._history`` to the session's snapshot under the
shared inference lock so multiple sessions can coexist with isolated context.

Inference is blocking (llama-cpp / faster-whisper / kokoro), so each turn runs
inside a worker thread (``asyncio.to_thread``). The worker pushes outbound
events through ``run_coroutine_threadsafe`` so the asyncio event loop stays
responsive — most importantly, the receive loop can keep handling
``{"type": "interrupt"}`` messages while the LLM is generating.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import time
from typing import TYPE_CHECKING

import numpy as np
from fastapi import WebSocket, WebSocketDisconnect

from edgevox.core.config import LANGUAGES
from edgevox.core.pipeline import stream_sentences
from edgevox.server.audio_utils import float32_to_wav_bytes, int16_bytes_to_float32
from edgevox.server.session import SessionState
from edgevox.tts import create_tts

if TYPE_CHECKING:
    from edgevox.server.core import ServerCore

log = logging.getLogger(__name__)


async def _send_json(ws: WebSocket, payload: dict) -> None:
    await ws.send_text(json.dumps(payload))


def _wire_chess_state_forwarder(ws: WebSocket, core: ServerCore):
    """Subscribe a ``chess_state`` forwarder to ``core.deps`` for one connection.

    Called once per :func:`handle_connection` (not per agent turn) so
    subscribers don't accumulate across turns. Primes the client with
    the current snapshot so a freshly-opened window renders the board
    immediately instead of showing a placeholder until the first move.

    For single-user desktop apps (chess_robot et al.), we also reset
    the environment's game state when a new connection arrives *and*
    there are no other active sessions — this makes "refresh to start
    over" the natural behavior without breaking multi-user setups.

    Returns an ``unsubscribe()`` callable, or ``None`` if ``core.deps``
    isn't chess-shaped. Callers use the return value in the connection
    ``finally`` block to release the subscription on disconnect.

    The send is scheduled on the running event loop via
    ``run_coroutine_threadsafe`` because environment listeners fire
    synchronously from whatever thread applied the move — typically
    the agent worker thread, never the asyncio loop.
    """
    # Use ``getattr`` so test doubles that don't carry a ``deps`` attr
    # (e.g. legacy FakeCore in ``tests/test_server_app.py``) no-op
    # gracefully. A real ServerCore always has ``deps``.
    deps = getattr(core, "deps", None)
    if deps is None or not hasattr(deps, "subscribe") or not hasattr(deps, "snapshot"):
        return None

    # Fresh-game on connect. A desktop app is single-user by design, and
    # the user's natural expectation when they open the window (or
    # reload) is a clean board. Always reset on connect rather than
    # gating on ``len(core.sessions)`` — that check races with the
    # previous session's async cleanup and skips the reset when two
    # connects overlap by milliseconds. Multi-user serving is a
    # separate config if we ever need it.
    try:
        if hasattr(deps, "new_game"):
            deps.new_game()
    except Exception:
        log.debug("chess_state: reset-on-connect failed (non-fatal)", exc_info=True)

    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:  # pragma: no cover — always called from the loop
        return None

    def _forward(state) -> None:
        to_json = getattr(state, "to_json", None)
        payload = to_json() if callable(to_json) else dict(state)
        coro = _send_json(ws, {"type": "chess_state", **payload})
        fut = asyncio.run_coroutine_threadsafe(coro, loop)
        # Fire-and-forget: the WebSocket send is best-effort from the
        # env thread, and exceptions are suppressed so an already-closed
        # client doesn't poison later moves.
        with contextlib.suppress(Exception):
            fut.result(timeout=1.0)

    try:
        unsub = deps.subscribe(_forward)
    except Exception:
        log.exception("Failed to subscribe chess_state forwarder")
        return None

    # Prime the connection. We keep the task reference (RUF006) so GC
    # can't yank it before the send completes; the connection's lifetime
    # covers it, so no explicit cleanup is needed.
    try:
        prime_task = asyncio.create_task(_send_json(ws, {"type": "chess_state", **deps.snapshot().to_json()}))
    except Exception:
        log.debug("chess_state priming failed", exc_info=True)
    else:
        prime_task.add_done_callback(lambda _t: None)

    return unsub


async def _send_state(ws: WebSocket, state: str) -> None:
    await _send_json(ws, {"type": "state", "value": state})


async def handle_connection(ws: WebSocket, core: ServerCore) -> None:
    """Top-level coroutine for one /ws connection."""
    await ws.accept()
    session = SessionState(language=core.language, history=core.fresh_history())
    core.sessions[session.id] = session
    log.info("Session %s connected (active=%d)", session.id, len(core.sessions))

    info = core.info()
    # Wire the chess_state forwarder once per connection (not per turn)
    # so subscribers don't stack and the client sees the starting
    # position the moment it connects. ``chess_state_unsub`` stays
    # ``None`` for non-chess agents; the finally block treats it as a
    # no-op in that case.
    chess_state_unsub = _wire_chess_state_forwarder(ws, core)
    try:
        await _send_json(
            ws,
            {
                "type": "ready",
                "session_id": session.id,
                "language": session.language,
                "languages": info["languages"],
                "voice": info["voice"],
                "voices": info["voices"],
                "tts_sample_rate": info["tts_sample_rate"],
                "frame_size": 512,
                "sample_rate": 16_000,
            },
        )
        await _send_state(ws, "listening")

        turn_tasks: set[asyncio.Task] = set()
        while True:
            msg = await ws.receive()
            if msg.get("type") == "websocket.disconnect":
                break

            if (data := msg.get("bytes")) is not None:
                samples = int16_bytes_to_float32(data)
                session.feed_audio(samples)
                if session.level > 0:
                    with contextlib.suppress(Exception):
                        await _send_json(ws, {"type": "level", "value": round(session.level, 3)})
                # VAD skips processing while busy, so segments only appear
                # when the session is idle and ready for a new turn.
                for segment in session.drain_segments():
                    task = asyncio.create_task(_run_turn(ws, core, session, segment))
                    turn_tasks.add(task)
                    task.add_done_callback(turn_tasks.discard)
                continue

            if (text := msg.get("text")) is not None:
                try:
                    payload = json.loads(text)
                except json.JSONDecodeError:
                    await _send_json(ws, {"type": "error", "message": "invalid json"})
                    continue

                # Text input and /say spawn async turns like voice segments
                if payload.get("type") == "text_input":
                    user_text = payload.get("text", "").strip()
                    if user_text and not session.busy:
                        task = asyncio.create_task(_run_text_turn(ws, core, session, user_text))
                        turn_tasks.add(task)
                        task.add_done_callback(turn_tasks.discard)
                    continue
                if payload.get("type") == "say":
                    say_text = payload.get("text", "").strip()
                    if say_text and not session.busy:
                        task = asyncio.create_task(_run_say(ws, core, session, say_text))
                        turn_tasks.add(task)
                        task.add_done_callback(turn_tasks.discard)
                    continue

                await _handle_control(ws, core, session, payload)

    except WebSocketDisconnect:
        log.info("Session %s disconnected", session.id)
    except Exception:
        log.exception("Session %s crashed", session.id)
        with contextlib.suppress(Exception):
            await _send_json(ws, {"type": "error", "message": "internal error"})
    finally:
        if chess_state_unsub is not None:
            with contextlib.suppress(Exception):
                chess_state_unsub()
        core.sessions.pop(session.id, None)
        with contextlib.suppress(Exception):
            await ws.close()


async def _handle_control(ws: WebSocket, core: ServerCore, session: SessionState, payload: dict) -> None:
    kind = payload.get("type")
    if kind == "hello":
        return  # already sent ready on accept
    if kind == "interrupt":
        session.interrupt_event.set()
        session.reset_audio()
        await _send_state(ws, "listening")
        return
    if kind == "reset":
        session.history = core.fresh_history()
        session.reset_audio()
        await _send_json(ws, {"type": "info", "message": "history cleared"})
        return
    if kind == "set_language":
        code = payload.get("language", "").strip().lower()
        if code not in LANGUAGES:
            await _send_json(ws, {"type": "error", "message": f"unknown language: {code}"})
            return
        cfg = LANGUAGES[code]
        session.language = code
        core.voice = cfg.default_voice
        # Switch TTS language if it supports it (Kokoro can switch without reload)
        if hasattr(core.tts, "set_language"):
            core.tts.set_language(cfg.kokoro_lang, cfg.default_voice)
        voices = core.voices_for_language(code)
        await _send_json(ws, {"type": "info", "message": f"language set to {cfg.name}"})
        await _send_json(
            ws,
            {"type": "language_changed", "language": code, "voice": cfg.default_voice, "voices": voices},
        )
        return
    if kind == "set_voice":
        voice = payload.get("voice", "").strip()
        if not voice:
            await _send_json(ws, {"type": "error", "message": "missing voice"})
            return
        available = core.voices_for_language(session.language)
        if voice not in available:
            await _send_json(ws, {"type": "error", "message": f"unknown voice: {voice}"})
            return
        core.voice = voice
        if hasattr(core.tts, "set_language"):
            cfg = LANGUAGES.get(session.language, LANGUAGES["en"])
            core.tts.set_language(cfg.kokoro_lang, voice)
        else:
            # Piper/Supertonic reload is blocking — run off the event loop
            core.tts = await asyncio.to_thread(create_tts, language=session.language, voice=voice)
        await _send_json(ws, {"type": "info", "message": f"voice set to {voice}"})
        await _send_json(ws, {"type": "voice_changed", "voice": voice})
        return
    if kind == "text_input":
        text = payload.get("text", "").strip()
        if not text:
            return
        return  # handled by caller — see handle_connection
    if kind == "say":
        text = payload.get("text", "").strip()
        if not text:
            return
        return  # handled by caller — see handle_connection
    await _send_json(ws, {"type": "error", "message": f"unknown control type: {kind}"})


async def _run_text_turn(ws: WebSocket, core: ServerCore, session: SessionState, text: str) -> None:
    """Run an LLM→TTS turn from typed text (skip STT)."""
    if session.busy:
        return
    session.busy = True
    session.interrupt_event.clear()
    loop = asyncio.get_running_loop()

    def _send_threadsafe(payload: dict) -> None:
        fut = asyncio.run_coroutine_threadsafe(_send_json(ws, payload), loop)
        with contextlib.suppress(Exception):
            fut.result(timeout=5.0)

    def _send_state_ts(state: str) -> None:
        _send_threadsafe({"type": "state", "value": state})

    def _send_bytes_ts(data: bytes) -> None:
        fut = asyncio.run_coroutine_threadsafe(ws.send_bytes(data), loop)
        with contextlib.suppress(Exception):
            fut.result(timeout=5.0)

    try:
        async with core.inference_lock:
            await asyncio.to_thread(
                _run_text_turn_blocking, core, session, text, _send_threadsafe, _send_state_ts, _send_bytes_ts
            )
    except WebSocketDisconnect:
        log.info("Session %s disconnected mid-turn", session.id)
    except Exception:
        log.exception("Text turn failed for session %s", session.id)
        with contextlib.suppress(Exception):
            await _send_json(ws, {"type": "error", "message": "turn failed"})
    finally:
        session.reset_audio()
        session.busy = False
        with contextlib.suppress(Exception):
            await _send_state(ws, "listening")


def _run_text_turn_blocking(
    core: ServerCore,
    session: SessionState,
    text: str,
    send_json,
    send_state,
    send_bytes,
) -> None:
    """LLM→TTS turn from typed text. Runs on a worker thread."""
    send_json({"type": "user_text", "text": text, "latency": 0})
    send_state("thinking")

    if core.agent is not None:
        _run_agent_turn(core, session, text, send_json, send_state, send_bytes, stt_elapsed=0.0, audio_duration=0.0)
        return

    saved_history = core.llm._history
    core.llm._history = session.history
    audio_id = 0
    t_start = time.perf_counter()
    t_first_token: float | None = None
    t_tts_total = 0.0
    full_reply: list[str] = []
    try:
        token_stream = core.llm.chat_stream(text)
        token_iter = _emit_tokens(token_stream, send_json, session)
        for sentence in stream_sentences(token_iter):
            if session.interrupt_event.is_set():
                break
            if t_first_token is None:
                t_first_token = time.perf_counter() - t_start

            full_reply.append(sentence)
            send_state("speaking")

            t_tts_start = time.perf_counter()
            audio_out = core.tts.synthesize(sentence)
            t_tts_total += time.perf_counter() - t_tts_start
            if session.interrupt_event.is_set():
                break

            sr = getattr(core.tts, "sample_rate", 24_000)
            wav_bytes = float32_to_wav_bytes(audio_out, sr)
            audio_id += 1
            send_json(
                {
                    "type": "bot_sentence",
                    "text": sentence,
                    "audio_id": audio_id,
                    "sample_rate": sr,
                    "bytes": len(wav_bytes),
                }
            )
            send_bytes(wav_bytes)
    finally:
        session.history = core.llm._history
        core.llm._history = saved_history

    t_total = time.perf_counter() - t_start
    t_llm = t_total - t_tts_total
    reply = " ".join(full_reply).strip()
    send_json({"type": "bot_text", "text": reply, "latency": round(t_llm, 3)})
    send_json(
        {
            "type": "metrics",
            "stt": 0,
            "llm": round(t_llm, 3),
            "ttft": round(t_first_token or 0.0, 3),
            "tts": round(t_tts_total, 3),
            "total": round(t_total, 3),
            "audio_duration": 0,
        }
    )


def agent_event_to_ws_message(event) -> dict | None:
    """Map an :class:`AgentEvent` to the outbound WebSocket message.

    Returns ``None`` for events the client doesn't care about, so the
    caller can skip the ``send_json``. Pure function — extracted from
    the per-turn worker so it's directly unit-testable and so new
    event kinds land as a single branch with a single test.

    Kinds forwarded today:

    - ``tool_call`` — tool observability (name, args, ok, result, error).
    - ``skill_goal`` / ``skill_cancelled`` / ``handoff`` / ``safety_preempt``
      — control-flow events, forwarded verbatim.
    - ``robot_face`` — Rook face expression signals from
      :class:`RobotFaceHook` (mood, gaze, tempo, persona).
    """
    kind = getattr(event, "kind", None)
    payload = getattr(event, "payload", None)
    if kind == "tool_call":
        r = payload
        return {
            "type": "tool_call",
            "name": getattr(r, "name", ""),
            "arguments": getattr(r, "arguments", {}),
            "ok": getattr(r, "ok", False),
            "result": getattr(r, "result", None),
            "error": getattr(r, "error", None),
        }
    if kind in {"skill_goal", "skill_cancelled", "handoff", "safety_preempt"}:
        return {"type": kind, "agent": getattr(event, "agent_name", ""), "payload": payload}
    if kind == "robot_face":
        # Payload is already a JSON-safe dict (``asdict(RobotFaceEvent)``),
        # so we splat it next to the ``type`` discriminator.
        return {"type": "robot_face", **(payload or {})}
    return None


def _run_agent_turn(
    core: ServerCore,
    session: SessionState,
    text: str,
    send_json,
    send_state,
    send_bytes,
    *,
    stt_elapsed: float,
    audio_duration: float,
) -> None:
    """Run one turn through :class:`LLMAgent` and drive TTS off the final reply.

    Compared to the legacy ``llm.chat_stream`` path this sacrifices
    mid-reply token streaming (``bot_token``) for the full agent
    surface: hooks, tools, handoffs, cancellable skills, typed
    ``ctx.deps``. Sentence-split TTS happens after generation by
    wrapping the reply as a one-token iterator; users notice slightly
    higher first-TTS latency on long replies but gain everything a
    :class:`LLMAgent` does.

    Tool-call events + custom domain events (``chess_state`` when
    ``ctx.deps`` is a :class:`ChessEnvironment`) are forwarded to the
    WebSocket client so rich UI surfaces — move list, eval bar, tool
    trace — light up automatically.
    """
    from edgevox.agents import AgentContext

    agent = core.agent
    assert agent is not None  # callers gate on core.agent

    t_start = time.perf_counter()

    def _on_event(event) -> None:
        message = agent_event_to_ws_message(event)
        if message is not None:
            send_json(message)

    # ``chess_state`` forwarder is wired once per connection in
    # :func:`handle_connection` — see ``_wire_chess_state_forwarder``.
    # Don't re-subscribe here; it would stack one listener per turn.

    ctx = AgentContext(
        session=session.agent_session,
        deps=core.deps,
        on_event=_on_event,
    )

    try:
        result = agent.run(text, ctx)
    except Exception:
        log.exception("Agent run failed for session %s", session.id)
        send_json({"type": "error", "message": "agent turn failed"})
        return

    reply = (result.reply or "").strip()
    t_llm = time.perf_counter() - t_start

    # Sentence-split + TTS. We wrap the reply as a single-token iterator
    # so ``stream_sentences`` works unchanged. Long replies get
    # split into sentences for TTS chunking.
    audio_id = 0
    t_tts_total = 0.0
    if reply:
        send_state("speaking")
        for sentence in stream_sentences(iter([reply])):
            if session.interrupt_event.is_set():
                break
            t_tts_start = time.perf_counter()
            audio_out = core.tts.synthesize(sentence)
            t_tts_total += time.perf_counter() - t_tts_start
            if session.interrupt_event.is_set():
                break
            sr = getattr(core.tts, "sample_rate", 24_000)
            wav_bytes = float32_to_wav_bytes(audio_out, sr)
            audio_id += 1
            send_json(
                {
                    "type": "bot_sentence",
                    "text": sentence,
                    "audio_id": audio_id,
                    "sample_rate": sr,
                    "bytes": len(wav_bytes),
                }
            )
            send_bytes(wav_bytes)

    t_total = time.perf_counter() - t_start
    send_json({"type": "bot_text", "text": reply, "latency": round(t_llm, 3)})
    send_json(
        {
            "type": "metrics",
            "stt": round(stt_elapsed, 3),
            "llm": round(t_llm - t_tts_total, 3),
            "ttft": round(t_llm, 3),
            "tts": round(t_tts_total, 3),
            "total": round(t_total + stt_elapsed, 3),
            "audio_duration": round(audio_duration, 3),
        }
    )


async def _run_say(ws: WebSocket, core: ServerCore, session: SessionState, text: str) -> None:
    """TTS-only preview — synthesize and send audio without LLM."""
    if session.busy:
        return
    session.busy = True
    try:
        async with core.inference_lock:

            def _synth():
                audio_out = core.tts.synthesize(text)
                sr = getattr(core.tts, "sample_rate", 24_000)
                return float32_to_wav_bytes(audio_out, sr), sr

            wav_bytes, sr = await asyncio.to_thread(_synth)
            await _send_json(
                ws,
                {"type": "bot_sentence", "text": text, "audio_id": 1, "sample_rate": sr, "bytes": len(wav_bytes)},
            )
            await ws.send_bytes(wav_bytes)
            await _send_json(ws, {"type": "info", "message": f"TTS preview: {text}"})
    except Exception:
        log.exception("Say failed for session %s", session.id)
    finally:
        session.reset_audio()
        session.busy = False


async def _run_turn(ws: WebSocket, core: ServerCore, session: SessionState, audio: np.ndarray) -> None:
    """Schedule one STT→LLM→TTS turn on a worker thread."""
    if session.busy:
        return
    session.busy = True
    session.interrupt_event.clear()
    loop = asyncio.get_running_loop()

    def _send_threadsafe(payload: dict) -> None:
        fut = asyncio.run_coroutine_threadsafe(_send_json(ws, payload), loop)
        with contextlib.suppress(Exception):
            fut.result(timeout=5.0)

    def _send_state_ts(state: str) -> None:
        _send_threadsafe({"type": "state", "value": state})

    def _send_bytes_ts(data: bytes) -> None:
        fut = asyncio.run_coroutine_threadsafe(ws.send_bytes(data), loop)
        with contextlib.suppress(Exception):
            fut.result(timeout=5.0)

    try:
        async with core.inference_lock:
            await asyncio.to_thread(
                _run_turn_blocking, core, session, audio, _send_threadsafe, _send_state_ts, _send_bytes_ts
            )
    except WebSocketDisconnect:
        log.info("Session %s disconnected mid-turn", session.id)
    except Exception:
        log.exception("Turn failed for session %s", session.id)
        with contextlib.suppress(Exception):
            await _send_json(ws, {"type": "error", "message": "turn failed"})
    finally:
        # Reset VAD and speech buffer so the next utterance starts clean —
        # mirrors the TUI's resume_after_cooldown() which drains + resets.
        session.reset_audio()
        session.busy = False
        with contextlib.suppress(Exception):
            await _send_state(ws, "listening")


def _run_turn_blocking(
    core: ServerCore,
    session: SessionState,
    audio: np.ndarray,
    send_json,
    send_state,
    send_bytes,
) -> None:
    """Runs entirely on a worker thread. Holds no asyncio primitives directly."""
    send_state("transcribing")
    t_stt_start = time.perf_counter()
    text = core.stt.transcribe(audio, language=session.language)
    t_stt = time.perf_counter() - t_stt_start

    if not text or text.isspace():
        return

    send_json({"type": "user_text", "text": text, "latency": round(t_stt, 3)})
    send_state("thinking")

    if core.agent is not None:
        _run_agent_turn(
            core,
            session,
            text,
            send_json,
            send_state,
            send_bytes,
            stt_elapsed=t_stt,
            audio_duration=float(SessionState.segment_duration(audio)),
        )
        return

    # Swap this session's history into the shared LLM under the lock.
    saved_history = core.llm._history
    core.llm._history = session.history
    audio_id = 0
    t_llm_start = time.perf_counter()
    t_first_token: float | None = None
    t_tts_total = 0.0
    full_reply: list[str] = []
    try:
        token_stream = core.llm.chat_stream(text)
        token_iter = _emit_tokens(token_stream, send_json, session)
        for sentence in stream_sentences(token_iter):
            if session.interrupt_event.is_set():
                break
            if t_first_token is None:
                t_first_token = time.perf_counter() - t_llm_start

            full_reply.append(sentence)
            send_state("speaking")

            t_tts_start = time.perf_counter()
            audio_out = core.tts.synthesize(sentence)
            t_tts_total += time.perf_counter() - t_tts_start
            if session.interrupt_event.is_set():
                break

            sr = getattr(core.tts, "sample_rate", 24_000)
            wav_bytes = float32_to_wav_bytes(audio_out, sr)
            audio_id += 1
            send_json(
                {
                    "type": "bot_sentence",
                    "text": sentence,
                    "audio_id": audio_id,
                    "sample_rate": sr,
                    "bytes": len(wav_bytes),
                }
            )
            send_bytes(wav_bytes)
    finally:
        session.history = core.llm._history
        core.llm._history = saved_history

    t_total = time.perf_counter() - t_stt_start
    t_llm = time.perf_counter() - t_llm_start - t_tts_total
    reply = " ".join(full_reply).strip()
    send_json({"type": "bot_text", "text": reply, "latency": round(t_llm, 3)})
    send_json(
        {
            "type": "metrics",
            "stt": round(t_stt, 3),
            "llm": round(t_llm, 3),
            "ttft": round(t_first_token or 0.0, 3),
            "tts": round(t_tts_total, 3),
            "total": round(t_total, 3),
            "audio_duration": round(SessionState.segment_duration(audio), 3),
        }
    )


def _emit_tokens(token_stream, send_json, session: SessionState):
    """Forward LLM tokens to the client and to the sentence splitter.

    Stops yielding (closing the generator) once an interrupt is requested so
    ``stream_sentences`` exits cleanly.
    """
    for tok in token_stream:
        if session.interrupt_event.is_set():
            break
        send_json({"type": "bot_token", "text": tok})
        yield tok
