"""Download and cache all required models ahead of time."""

from __future__ import annotations

import argparse
import logging

log = logging.getLogger(__name__)


def download_llm(repo: str = "unsloth/gemma-4-E2B-it-GGUF", filename: str = "gemma-4-E2B-it-Q4_K_M.gguf"):
    from huggingface_hub import hf_hub_download

    print(f"Downloading LLM: {repo}/{filename} ...")
    path = hf_hub_download(repo_id=repo, filename=filename)
    print(f"  Cached at: {path}")
    return path


def download_whisper(model_size: str = "small"):
    from faster_whisper import WhisperModel

    print(f"Downloading Whisper model: {model_size} ...")
    _ = WhisperModel(model_size, device="cpu", compute_type="int8")
    print(f"  Whisper {model_size} ready.")


def download_kokoro():
    import kokoro

    print("Downloading Kokoro TTS model + voice pack ...")
    pipeline = kokoro.KPipeline(lang_code="a")
    # Trigger voice download
    for _ in pipeline("Test.", voice="af_heart"):
        pass
    print("  Kokoro ready.")


def download_sherpa_vi():
    from edgevox.stt.sherpa_stt import _ensure_model

    print("Downloading Sherpa-ONNX Zipformer Vietnamese (30M int8) ...")
    model_dir = _ensure_model()
    print(f"  Cached at: {model_dir}")


def download_silero_vad():
    from silero_vad import load_silero_vad

    print("Downloading Silero VAD ...")
    _ = load_silero_vad(onnx=True)
    print("  Silero VAD ready.")


def main():
    parser = argparse.ArgumentParser(description="Download all models for EdgeVox")
    parser.add_argument("--whisper-model", type=str, default="small", help="Whisper model to download (default: small)")
    parser.add_argument("--llm-quant", type=str, default="Q4_K_M", help="GGUF quantization (default: Q4_K_M)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    print("=" * 50)
    print("  VOXPILOT — Model Setup")
    print("=" * 50 + "\n")

    filename = f"gemma-4-E2B-it-{args.llm_quant}.gguf"

    download_silero_vad()
    print()
    download_whisper(args.whisper_model)
    print()
    download_sherpa_vi()
    print()
    download_llm(filename=filename)
    print()
    download_kokoro()

    print("\n" + "=" * 50)
    print("  All models downloaded! Run with:")
    print("    edgevox")
    print("  Or CLI mode:")
    print("    edgevox-cli")
    print("=" * 50)


if __name__ == "__main__":
    main()
