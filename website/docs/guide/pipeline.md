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

## Interrupt Mechanism

```
User speaks → VAD detects → _on_interrupt() called
  → _interrupted Event is set
  → play_audio() returns False (stops playback)
  → sentence loop breaks
  → new speech is captured and processed
```

The interrupt is non-blocking — the pipeline naturally exits at the next checkpoint (sentence boundary or audio chunk).

## Echo Suppression

During TTS playback, the microphone is suppressed to prevent the bot's audio from being captured as speech:

- **Pause**: Mic enters echo suppression when TTS starts playing
- **Generation counter**: Stale cooldown timers are ignored if a newer pause has been issued
- **Force resume**: When the pipeline finishes, the mic is re-enabled after a brief delay (0.3s) instead of waiting through the full cooldown

## LLM Configuration

Gemma 4 E2B IT runs via `llama-cpp-python`:

- Quantization: Q4_K_M (balance of quality and speed)
- Context: Maintains conversation history
- Streaming: Token-by-token generation
- Reset: `/reset` clears context for fresh conversation
