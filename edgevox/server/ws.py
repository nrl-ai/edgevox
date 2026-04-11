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

from edgevox.core.pipeline import stream_sentences
from edgevox.server.audio_utils import float32_to_wav_bytes, int16_bytes_to_float32
from edgevox.server.session import SessionState

if TYPE_CHECKING:
    from edgevox.server.core import ServerCore

log = logging.getLogger(__name__)


async def _send_json(ws: WebSocket, payload: dict) -> None:
    await ws.send_text(json.dumps(payload))


async def _send_state(ws: WebSocket, state: str) -> None:
    await _send_json(ws, {"type": "state", "value": state})


async def handle_connection(ws: WebSocket, core: ServerCore) -> None:
    """Top-level coroutine for one /ws connection."""
    await ws.accept()
    session = SessionState(language=core.language, history=core.fresh_history())
    core.sessions[session.id] = session
    log.info("Session %s connected (active=%d)", session.id, len(core.sessions))

    info = core.info()
    try:
        await _send_json(
            ws,
            {
                "type": "ready",
                "session_id": session.id,
                "language": session.language,
                "languages": info["languages"],
                "voice": info["voice"],
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
                for segment in session.drain_segments():
                    if session.busy:
                        log.debug("Session %s busy; dropping a queued segment", session.id)
                        continue
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
                await _handle_control(ws, core, session, payload)

    except WebSocketDisconnect:
        log.info("Session %s disconnected", session.id)
    except Exception:
        log.exception("Session %s crashed", session.id)
        with contextlib.suppress(Exception):
            await _send_json(ws, {"type": "error", "message": "internal error"})
    finally:
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
    await _send_json(ws, {"type": "error", "message": f"unknown control type: {kind}"})


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
