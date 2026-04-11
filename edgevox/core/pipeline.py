"""High-performance streaming pipeline: LLM streams → sentence split → TTS → audio.

Key optimization: start speaking the first sentence while the LLM is still
generating the rest. This cuts perceived latency by 30-50%.
"""

from __future__ import annotations

import logging
import re
import threading
import time
from collections.abc import Generator

import numpy as np

from edgevox.audio import play_audio
from edgevox.llm import LLM
from edgevox.stt import STT
from edgevox.tts import SAMPLE_RATE as TTS_SAMPLE_RATE
from edgevox.tts import TTS

log = logging.getLogger(__name__)

ABBREVIATIONS = {
    "mr.",
    "mrs.",
    "ms.",
    "dr.",
    "prof.",
    "sr.",
    "jr.",
    "st.",
    "ave.",
    "vs.",
    "etc.",
    "approx.",
    "dept.",
    "est.",
    "vol.",
    "fig.",
    "inc.",
    "ltd.",
    "e.g.",
    "i.e.",
    "u.s.",
    "u.k.",
}

# Simple sentence-end: .!? followed by whitespace
_SPLIT_RE = re.compile(r"([.!?])\s+")

MAX_CHUNK_CHARS = 200


def _is_sentence_boundary(text_before_punct: str) -> bool:
    """Check if the period is a real sentence boundary, not an abbreviation or decimal."""
    text = text_before_punct.rstrip()
    if not text:
        return True
    # Get the last word (the one before the period)
    last_word = text.split()[-1].lower() if text.split() else ""
    # Check abbreviations: "dr" matches "dr.", "e.g" matches "e.g."
    for abbr in ABBREVIATIONS:
        bare = abbr.rstrip(".")
        if last_word in (bare, abbr):
            return False
    # Decimal number: digit right before the period (e.g. "3.14")
    if text[-1].isdigit():
        return False
    # Ellipsis: multiple dots
    if text.endswith(".."):
        return False
    # Short word (1-2 chars) followed by period is likely abbreviation
    return not (len(last_word) <= 2 and last_word.isalpha())


def _find_sentence_break(buffer: str) -> int | None:
    """Find the position to split the buffer, or None if no boundary found.

    Returns the index right after the sentence (including punctuation),
    so buffer[:pos] is the sentence and buffer[pos:].lstrip() is the rest.
    """
    for match in _SPLIT_RE.finditer(buffer):
        before = buffer[: match.start() + 1]  # include .!?
        punct = match.group(1)
        if punct in "!?" or _is_sentence_boundary(before[:-1]):
            return match.start() + 1  # position after the punctuation
    return None


def stream_sentences(token_stream: Generator[str, None, None]) -> Generator[str, None, None]:
    """Accumulate tokens and yield complete sentences.

    Splits on sentence boundaries (.!?) while avoiding false splits on
    abbreviations (Dr., U.S.) and decimals (3.14). Also breaks very long
    clauses at commas to keep TTS chunks manageable.
    """
    buffer = ""
    for token in token_stream:
        buffer += token

        # Try all potential sentence boundaries in the buffer
        while True:
            pos = _find_sentence_break(buffer)
            if pos is None:
                break
            sentence = buffer[:pos].strip()
            if sentence:
                yield sentence
            buffer = buffer[pos:].lstrip()

        # Break very long clauses at comma/semicolon for responsiveness
        if len(buffer) > MAX_CHUNK_CHARS:
            for sep in ["; ", ", ", ": "]:
                idx = buffer.rfind(sep, MAX_CHUNK_CHARS // 2)
                if idx > 0:
                    chunk = buffer[: idx + len(sep)].strip()
                    if chunk:
                        yield chunk
                    buffer = buffer[idx + len(sep) :]
                    break

    remaining = buffer.strip()
    if remaining:
        yield remaining


class StreamingPipeline:
    """Orchestrates the full voice pipeline with streaming for minimum latency.

    Flow:
        audio → STT → LLM (streaming) → sentence splitter → TTS → speaker
                                         ↑ sentences are spoken as they arrive
    """

    def __init__(
        self,
        stt: STT,
        llm: LLM,
        tts: TTS,
        on_state_change=None,
        on_user_text=None,
        on_bot_text=None,
        on_metrics=None,
    ):
        self.stt = stt
        self.llm = llm
        self.tts = tts
        self._on_state_change = on_state_change or (lambda s: None)
        self._on_user_text = on_user_text or (lambda t, d: None)
        self._on_bot_text = on_bot_text or (lambda t, d: None)
        self._on_metrics = on_metrics or (lambda m: None)
        self._interrupt = threading.Event()

    def interrupt(self):
        """Call to stop current response (e.g., user started speaking again)."""
        self._interrupt.set()

    def process(self, audio: np.ndarray, language: str = "en") -> dict:
        """Process a speech segment through the full pipeline.

        Returns dict with timing metrics.
        """
        self._interrupt.clear()
        from edgevox.audio import TARGET_SAMPLE_RATE as MIC_SR

        audio_duration = len(audio) / MIC_SR

        # === STT ===
        self._on_state_change("transcribing")
        t_stt_start = time.perf_counter()
        text = self.stt.transcribe(audio, language=language)
        t_stt = time.perf_counter() - t_stt_start

        if not text or text.isspace():
            self._on_state_change("listening")
            return {"skipped": True}

        self._on_user_text(text, t_stt)

        # === LLM streaming → TTS sentence-by-sentence ===
        self._on_state_change("thinking")
        t_llm_start = time.perf_counter()
        t_first_token = None
        t_tts_total = 0.0
        full_reply = []

        token_stream = self.llm.chat_stream(text)
        sentence_stream = stream_sentences(token_stream)

        first_sentence = True
        for sentence in sentence_stream:
            if self._interrupt.is_set():
                log.info("Pipeline interrupted by user")
                break

            if first_sentence:
                t_first_token = time.perf_counter() - t_llm_start
                first_sentence = False

            full_reply.append(sentence)

            # TTS this sentence immediately
            self._on_state_change("speaking")
            t_tts_start = time.perf_counter()
            audio_out = self.tts.synthesize(sentence)
            t_tts_total += time.perf_counter() - t_tts_start

            if self._interrupt.is_set():
                break

            # Play
            play_audio(audio_out, sample_rate=TTS_SAMPLE_RATE)

        t_total = time.perf_counter() - t_stt_start
        t_llm = time.perf_counter() - t_llm_start - t_tts_total

        reply = " ".join(full_reply)
        self._on_bot_text(reply, t_llm)

        metrics = {
            "stt": t_stt,
            "llm": t_llm,
            "ttft": t_first_token or 0,
            "tts": t_tts_total,
            "total": t_total,
            "audio_duration": audio_duration,
        }
        self._on_metrics(metrics)
        self._on_state_change("listening")
        return metrics
