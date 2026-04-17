"""Download and cache all required models ahead of time."""

from __future__ import annotations

import argparse
import logging

log = logging.getLogger(__name__)


def download_llm(preset: str | None = None, repo: str | None = None, filename: str | None = None):
    """Download an LLM GGUF by preset slug or explicit (repo, filename).

    Preset slugs come from :mod:`edgevox.llm.models`. If neither preset nor
    repo+filename are given, the default preset is used.
    """
    from huggingface_hub import hf_hub_download

    from edgevox.llm.models import DEFAULT_PRESET, resolve_preset

    if repo and filename:
        target_repo, target_file = repo, filename
    else:
        p = resolve_preset(preset or DEFAULT_PRESET)
        target_repo, target_file = p.repo, p.filename

    print(f"Downloading LLM: {target_repo}/{target_file} ...")
    path = hf_hub_download(repo_id=target_repo, filename=target_file)
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
    from edgevox.llm.models import DEFAULT_PRESET, PRESETS

    parser = argparse.ArgumentParser(description="Download all models for EdgeVox")
    parser.add_argument("--whisper-model", type=str, default="small", help="Whisper model to download (default: small)")
    parser.add_argument(
        "--llm-preset",
        type=str,
        default=DEFAULT_PRESET,
        choices=sorted(PRESETS),
        help=f"LLM preset slug to cache (default: {DEFAULT_PRESET})",
    )
    parser.add_argument("--list-llm-presets", action="store_true", help="List available LLM presets and exit")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    if args.list_llm_presets:
        for p in PRESETS.values():
            tag = " [embodied]" if p.embodied else ""
            print(f"  {p.slug:<22} {p.family:<10} ~{p.size_gb:.1f} GB{tag}  {p.description}")
        return

    print("=" * 50)
    print("  EDGEVOX — Model Setup")
    print("=" * 50 + "\n")

    download_silero_vad()
    print()
    download_whisper(args.whisper_model)
    print()
    download_sherpa_vi()
    print()
    download_llm(preset=args.llm_preset)
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
