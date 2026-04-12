# Voice Pipeline

Deep dive into EdgeVox's streaming voice pipeline.

## Sentence-Level Streaming

The core insight: **don't wait for the full LLM response before speaking**.

```python
token_stream = llm.chat_stream(text)        # yields tokens
sentence_stream = stream_sentences(token_stream)  # yields sentences

for sentence in sentence_stream:
    audio = tts.synthesize(sentence)    # TTS one sentence
    play_audio(audio)                   # play while LLM continues
```

The `stream_sentences()` function buffers tokens and splits on sentence boundaries (`[.!?]\s+`). Each sentence is sent to TTS immediately.

## VAD Configuration

Silero VAD v6 runs with these parameters:

- **Chunk size**: 512 samples (32ms at 16kHz)
- **Threshold**: Configured in `AudioRecorder`
- **Padding**: Speech chunks are padded to capture full utterances

The microphone records at native sample rate and resamples to 16kHz for VAD/STT.

## STT Backends

### Whisper (default)

Auto-selects model size based on hardware:

| VRAM | Model | Device | Compute |
|------|-------|--------|---------|
| >= 8GB | `large-v3-turbo` | CUDA | float16 |
| < 8GB | `small` | CUDA | float16 |
| CPU (32GB+ RAM) | `large-v3-turbo` | CPU | int8 |
| CPU (16GB+ RAM) | `medium` | CPU | int8 |
| CPU (< 16GB) | `small` | CPU | int8 |

### Sherpa-ONNX (Vietnamese)

- Model: Zipformer transducer, 30M params (int8)
- RTF ~0.01 on CPU — extremely fast
- Uses encoder + decoder + joiner architecture
- Apache 2.0 licensed
- Falls back to Whisper on load failure

### ChunkFormer (Vietnamese, legacy)

- Model: `khanhld/chunkformer-ctc-large-vie`
- 110M parameters, 4.18% WER on VIVOS
- Falls back to Whisper on load failure

## TTS Backends

### Kokoro-82M

- 82M parameter model, 9 native languages
- 24kHz output sample rate
- 25 voices across languages (see `/voices`)
- Streaming support via `synthesize_stream()`
- Language switching without model reload

### Piper ONNX

- Lightweight VITS architecture, real-time on CPU
- 22,050 Hz output sample rate
- 20 voices across Vietnamese, German, Russian, Arabic, Indonesian
- Models hosted on `nrl-ai/edgevox-models` with upstream fallbacks

### Supertonic (Korean)

- ONNX model, real-time on CPU
- 44,100 Hz output sample rate
- 10 voice styles (5 female, 5 male)
- MIT code + OpenRAIL-M weights

### PyThaiTTS (Thai)

- Tacotron2 ONNX architecture
- 22,050 Hz output sample rate
- Apache 2.0 licensed

## Audio Playback

`InterruptiblePlayer` runs a **callback-based** PortAudio output stream. The audio thread is fed by a numpy buffer that the main thread fills via `play()`; on every callback the audio thread copies up to `frames` samples from the buffer into the device's `outdata`, padding with silence on underrun.

This design replaces the older "blocking `stream.write()` in a loop" approach, which suffered intermittent ALSA failures (`alsa_snd_pcm_mmap_begin`, `PaAlsaStream_SetUpBuffers`) whenever streaming TTS could not deliver chunks fast enough — ALSA's mmap recovery path is unreliable across PipeWire/PulseAudio bridges.

Properties of the callback design:

- **No underruns.** The audio thread always has something to emit (real samples or silence), so the device never starves.
- **Persistent stream.** The output stream is opened once per device + sample rate and reused across `play()` calls; no per-chunk init latency.
- **Lock-free interrupts.** `interrupt()` only flushes the queued buffer — it never touches the stream itself, eliminating the abort/restart race that caused segfaults under load.
- **No inter-chunk gap.** `play()` returns as soon as the in-process buffer has been consumed by the callback. The next streaming-TTS chunk lands before PortAudio's internal ring drains, so audio plays continuously without per-chunk silence.

## Interrupt Mechanism

```
User speaks → VAD detects → _on_interrupt() called
  → _interrupted Event is set
  → InterruptiblePlayer.interrupt() flushes the playback buffer
  → callback emits silence on the next tick
  → play_audio() returns False
  → sentence loop breaks
  → new speech is captured and processed
```

The interrupt is non-blocking — the pipeline naturally exits at the next checkpoint (sentence boundary or audio chunk). Cutoff lag is bounded by one PortAudio callback period (~10–20 ms with default latency).

## Echo Suppression

During TTS playback, the microphone is suppressed to prevent the bot's audio from being captured as speech:

- **Pause**: Mic enters echo suppression when TTS starts playing.
- **Generation counter**: Stale cooldown timers are ignored if a newer pause has been issued.
- **Force resume**: When the pipeline finishes, the mic is re-enabled after a brief delay (0.3s) instead of waiting through the full cooldown.

When an AEC backend is enabled, the player also captures the reference signal (the audio that actually reached the speaker) via a thread-safe **chunked ring buffer** (`_RefBuffer`). The audio-thread producer pushes numpy chunks in O(1) — no Python-level per-sample iteration — and the recorder loop pops fixed-size frames to feed the echo canceller (NLMS, spectral subtraction, or DTLN).

## LLM Configuration

Gemma 4 E2B IT runs via `llama-cpp-python`:

- Quantization: Q4_K_M (balance of quality and speed)
- Context: Maintains conversation history
- Streaming: Token-by-token generation
- Reset: `/reset` clears context for fresh conversation
