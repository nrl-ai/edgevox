"""EdgeVox CLI — main pipeline.

Listen -> Transcribe -> Think -> Speak

Usage:
    python -m edgevox.main
    # or after pip install:
    edgevox-cli
"""

from __future__ import annotations

import argparse
import logging
import signal
import threading
import time

import numpy as np

from edgevox.audio import TARGET_SAMPLE_RATE as MIC_SAMPLE_RATE
from edgevox.audio import AudioRecorder, play_audio
from edgevox.llm import LLM
from edgevox.stt import create_stt
from edgevox.tts import create_tts

log = logging.getLogger(__name__)


class VoiceBot:
    """Wires STT → LLM → TTS into a conversational loop."""

    def __init__(
        self,
        stt_model: str | None = None,
        stt_device: str | None = None,
        llm_model: str | None = None,
        tts_backend: str | None = None,
        voice: str | None = None,
        language: str = "en",
    ):
        print("Loading models... (this may take a minute on first run)")

        t0 = time.perf_counter()
        self._stt = create_stt(language=language, model_size=stt_model, device=stt_device)
        self._llm = LLM(model_path=llm_model, language=language)
        self._tts = create_tts(language=language, voice=voice, backend=tts_backend)
        elapsed = time.perf_counter() - t0
        print(f"All models loaded in {elapsed:.1f}s")

        self._language = language
        self._processing = threading.Lock()
        self._recorder: AudioRecorder | None = None

    def _on_speech(self, audio: np.ndarray):
        """Called by AudioRecorder when a speech segment is detected."""
        if not self._processing.acquire(blocking=False):
            log.debug("Skipping overlapping speech segment (still processing)")
            return

        try:
            duration = len(audio) / MIC_SAMPLE_RATE
            print(f"\n🎤 Heard {duration:.1f}s of speech, processing...")

            # 1. Speech-to-Text
            t0 = time.perf_counter()
            text = self._stt.transcribe(audio, language=self._language)
            stt_time = time.perf_counter() - t0

            if not text or text.isspace():
                print("  (no speech detected)")
                return

            print(f'  📝 You said: "{text}" ({stt_time:.2f}s)')

            # 2. LLM response
            t1 = time.perf_counter()
            reply = self._llm.chat(text)
            llm_time = time.perf_counter() - t1
            print(f'  🤖 Reply: "{reply}" ({llm_time:.2f}s)')

            # 3. Text-to-Speech
            t2 = time.perf_counter()
            audio_out = self._tts.synthesize(reply)
            tts_time = time.perf_counter() - t2

            total = time.perf_counter() - t0
            print(f"  ⏱️  Latency: STT={stt_time:.2f}s LLM={llm_time:.2f}s TTS={tts_time:.2f}s Total={total:.2f}s")

            # 4. Play audio
            play_audio(audio_out, sample_rate=self._tts.sample_rate)

        except Exception:
            log.exception("Error in voice pipeline")
        finally:
            self._processing.release()

    def run(self):
        """Start the voice bot. Blocks until Ctrl+C."""
        print("\n" + "=" * 60)
        print("  VOXPILOT — Local Voice AI")
        print("  Speak naturally — I'll respond when you pause.")
        print("  Press Ctrl+C to quit.")
        print("=" * 60 + "\n")

        # Warm up TTS with a short utterance
        print("Warming up TTS...")
        _ = self._tts.synthesize("Ready.")
        print("Ready! Start speaking.\n")

        self._recorder = AudioRecorder(on_speech=self._on_speech)
        self._recorder.start()

        stop_event = threading.Event()

        def _handle_signal(sig, frame):
            print("\nShutting down...")
            stop_event.set()

        signal.signal(signal.SIGINT, _handle_signal)

        stop_event.wait()
        self._recorder.stop()


class TextBot:
    """Text-only mode for testing without a microphone."""

    def __init__(
        self,
        llm_model: str | None = None,
        tts_backend: str | None = None,
        voice: str | None = None,
        language: str = "en",
    ):
        print("Loading models...")
        t0 = time.perf_counter()
        self._llm = LLM(model_path=llm_model, language=language)
        self._tts = create_tts(language=language, voice=voice, backend=tts_backend)
        elapsed = time.perf_counter() - t0
        print(f"Models loaded in {elapsed:.1f}s\n")

    def run(self):
        print("Text mode — type your message (Ctrl+C to quit):\n")
        while True:
            try:
                text = input("You: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nBye!")
                break
            if not text:
                continue

            t0 = time.perf_counter()
            reply = self._llm.chat(text)
            llm_time = time.perf_counter() - t0
            print(f"Bot: {reply} ({llm_time:.2f}s)")

            t1 = time.perf_counter()
            audio = self._tts.synthesize(reply)
            tts_time = time.perf_counter() - t1
            print(f"  [TTS: {tts_time:.2f}s]")

            play_audio(audio, sample_rate=self._tts.sample_rate)


def main():
    parser = argparse.ArgumentParser(description="EdgeVox CLI — Local Voice AI")
    parser.add_argument(
        "--stt",
        type=str,
        default=None,
        help="STT model: tiny, base, small, medium, large-v3-turbo, or chunkformer (auto-detected)",
    )
    parser.add_argument("--stt-device", type=str, default=None, help="STT device: cuda, cpu (auto-detected)")
    parser.add_argument(
        "--llm",
        type=str,
        default=None,
        help="LLM model: local GGUF path or hf:repo/name:file.gguf (default: Gemma 4 E2B Q4_K_M)",
    )
    parser.add_argument(
        "--tts",
        type=str,
        default=None,
        choices=["kokoro", "piper"],
        help="TTS backend: kokoro or piper (auto from language)",
    )
    parser.add_argument("--voice", type=str, default=None, help="TTS voice name (default: per language)")
    parser.add_argument("--language", type=str, default="en", help="Speech language code (default: en)")
    parser.add_argument("--text-mode", action="store_true", help="Text-only mode (no microphone needed)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable debug logging")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    if args.text_mode:
        bot = TextBot(llm_model=args.llm, tts_backend=args.tts, voice=args.voice, language=args.language)
    else:
        bot = VoiceBot(
            stt_model=args.stt,
            stt_device=args.stt_device,
            llm_model=args.llm,
            tts_backend=args.tts,
            voice=args.voice,
            language=args.language,
        )
    bot.run()


if __name__ == "__main__":
    main()
