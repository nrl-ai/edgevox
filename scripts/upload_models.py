#!/usr/bin/env python3
"""Download all Piper TTS voice models and upload to nrl-ai/edgevox-models.

Usage:
    # First, login to HuggingFace:
    huggingface-cli login

    # Then run:
    python scripts/upload_models.py
"""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download, snapshot_download

REPO_ID = "nrl-ai/edgevox-models"

# --- STT models ---
# Format: (model_id, original_repo, files, license, license_note, attribution)
STT_MODELS: list[tuple[str, str, list[str], str, str, str]] = [
    (
        "sherpa-zipformer-vi-30M-int8",
        "csukuangfj2/sherpa-onnx-zipformer-vi-30M-int8-2026-02-09",
        ["encoder.int8.onnx", "decoder.onnx", "joiner.int8.onnx", "tokens.txt"],
        "Apache-2.0",
        "Free to use. No restrictions.",
        "Vietnamese speech data, Sherpa-ONNX project (k2-fsa)",
    ),
]

# --- Kokoro TTS model (~338 MB) ---
KOKORO_MODEL = {
    "id": "kokoro-v1.0",
    "source_url": "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0",
    "files": ["kokoro-v1.0.onnx", "voices-v1.0.bin"],
    "license": "Apache-2.0",
    "license_note": "Apache 2.0. Kokoro-82M by hexgrad. Some training data uses CC BY 3.0 (Koniwa) "
    "and CC BY 4.0 (SIWIS) — attribution to those datasets is required.",
    "attribution": "hexgrad/Kokoro-82M, thewh1teagle/kokoro-onnx",
}

# --- Supertonic TTS model (~255 MB) ---
SUPERTONIC_MODEL = {
    "id": "supertonic-2",
    "source_repo": "Supertone/supertonic-2",
    "onnx_files": [
        "onnx/text_encoder.onnx",
        "onnx/vector_estimator.onnx",
        "onnx/vocoder.onnx",
        "onnx/duration_predictor.onnx",
        "onnx/tts.json",
        "onnx/unicode_indexer.json",
    ],
    "voice_files": [
        "voice_styles/F1.json",
        "voice_styles/F2.json",
        "voice_styles/F3.json",
        "voice_styles/F4.json",
        "voice_styles/F5.json",
        "voice_styles/M1.json",
        "voice_styles/M2.json",
        "voice_styles/M3.json",
        "voice_styles/M4.json",
        "voice_styles/M5.json",
    ],
    "config_files": ["config.json"],
    "license": "MIT (code) + OpenRAIL-M (model weights)",
    "license_note": "Code is MIT. Model weights are OpenRAIL-M — redistribution allowed but use-based "
    "restrictions apply (no illegal use, deepfakes without consent, harassment, etc.). "
    "Derivatives must include the same use restrictions.",
    "attribution": "Supertone Inc.",
}

# --- PyThaiTTS model (~163 MB) ---
PYTHAITTS_MODEL = {
    "id": "pythaitts-lunarlist",
    "source_repo": "pythainlp/thaitts-onnx",
    "files": [
        "tacotron2encoder-th.onnx",
        "tacotron2decoder-th.onnx",
        "tacotron2postnet-th.onnx",
        "vocoder.onnx",
    ],
    "license": "Apache-2.0",
    "license_note": "Apache 2.0. Free to use with attribution.",
    "attribution": "PyThaiNLP / lunarlist, trained on Thai Common Voice data",
}

# --- Piper TTS voices ---
# Format: (voice_id, original_repo, license, license_note, attribution)
PIPER_VOICES: list[tuple[str, str, str, str, str]] = [
    # Vietnamese
    (
        "vi-vais1000",
        "speaches-ai/piper-vi_VN-vais1000-medium",
        "CC-BY-4.0",
        "Free to use with attribution.",
        "VAIS-1000 Vietnamese Speech Synthesis Corpus",
    ),
    (
        "vi-25hours",
        "speaches-ai/piper-vi_VN-25hours_single-low",
        "Unknown",
        "License unclear. Original source: InfoRe Technology. Use at your own risk.",
        "InfoRe Technology 1",
    ),
    (
        "vi-vivos",
        "speaches-ai/piper-vi_VN-vivos-x_low",
        "CC-BY-NC-SA-4.0",
        "Non-commercial use only. ShareAlike required.",
        "VIVOS Corpus / InfoRe Technology",
    ),
    # German — CC0 (Public Domain)
    (
        "de-thorsten-high",
        "speaches-ai/piper-de_DE-thorsten-high",
        "CC0-1.0",
        "Public domain. No restrictions.",
        "Thorsten Voice Dataset by Thorsten Mueller",
    ),
    (
        "de-thorsten",
        "speaches-ai/piper-de_DE-thorsten-medium",
        "CC0-1.0",
        "Public domain. No restrictions.",
        "Thorsten Voice Dataset by Thorsten Mueller",
    ),
    (
        "de-thorsten-low",
        "speaches-ai/piper-de_DE-thorsten-low",
        "CC0-1.0",
        "Public domain. No restrictions.",
        "Thorsten Voice Dataset by Thorsten Mueller",
    ),
    (
        "de-thorsten-emotional",
        "speaches-ai/piper-de_DE-thorsten_emotional-medium",
        "CC0-1.0",
        "Public domain. No restrictions.",
        "Thorsten Voice Dataset (emotional) by Thorsten Mueller",
    ),
    (
        "de-kerstin",
        "speaches-ai/piper-de_DE-kerstin-low",
        "CC0-1.0",
        "Public domain. No restrictions.",
        "Kerstin dataset by Thorsten Mueller project",
    ),
    # German — M-AILABS (license unverifiable, source down)
    (
        "de-ramona",
        "speaches-ai/piper-de_DE-ramona-low",
        "M-AILABS (unverifiable)",
        "Originally from M-AILABS Speech Dataset (caito.de). Based on LibriVox public domain recordings. "
        "Original license page is offline. Believed permissive but cannot confirm.",
        "M-AILABS Speech Dataset / LibriVox",
    ),
    (
        "de-eva",
        "speaches-ai/piper-de_DE-eva_k-x_low",
        "M-AILABS (unverifiable)",
        "Originally from M-AILABS Speech Dataset (caito.de). Based on LibriVox public domain recordings. "
        "Original license page is offline. Believed permissive but cannot confirm.",
        "M-AILABS Speech Dataset / LibriVox",
    ),
    (
        "de-karlsson",
        "speaches-ai/piper-de_DE-karlsson-low",
        "M-AILABS (unverifiable)",
        "Originally from M-AILABS Speech Dataset (caito.de). Based on LibriVox public domain recordings. "
        "Original license page is offline. Believed permissive but cannot confirm.",
        "M-AILABS Speech Dataset / LibriVox",
    ),
    # German — other
    (
        "de-pavoque",
        "speaches-ai/piper-de_DE-pavoque-low",
        "CC-BY-NC-SA-4.0",
        "Non-commercial use only. ShareAlike required.",
        "PAVOQUE Corpus",
    ),
    (
        "de-mls",
        "speaches-ai/piper-de_DE-mls-medium",
        "CC-BY-4.0",
        "Free to use with attribution.",
        "Multilingual LibriSpeech (OpenSLR 94)",
    ),
    # Russian
    (
        "ru-irina",
        "speaches-ai/piper-ru_RU-irina-medium",
        "Unknown",
        "License unclear. Original source: RHVoice project. Use at your own risk.",
        "RHVoice project",
    ),
    (
        "ru-dmitri",
        "speaches-ai/piper-ru_RU-dmitri-medium",
        "CC0-1.0",
        "Public domain. No restrictions.",
        "Dmitri voice dataset",
    ),
    (
        "ru-denis",
        "speaches-ai/piper-ru_RU-denis-medium",
        "CC0-1.0",
        "Public domain. No restrictions.",
        "Denis voice dataset",
    ),
    (
        "ru-ruslan",
        "speaches-ai/piper-ru_RU-ruslan-medium",
        "CC-BY-NC-SA-4.0",
        "Non-commercial use only. ShareAlike required.",
        "Ruslan dataset",
    ),
    # Arabic
    (
        "ar-kareem",
        "speaches-ai/piper-ar_JO-kareem-medium",
        "No license",
        "No license found in original repo. Original source: github.com/AliMokhammad/arabicttstrain. "
        "Use at your own risk.",
        "AliMokhammad/arabicttstrain",
    ),
    (
        "ar-kareem-low",
        "speaches-ai/piper-ar_JO-kareem-low",
        "No license",
        "No license found in original repo. Original source: github.com/AliMokhammad/arabicttstrain. "
        "Use at your own risk.",
        "AliMokhammad/arabicttstrain",
    ),
    # Indonesian
    (
        "id-news",
        "giganticlab/piper-id_ID-news_tts-medium",
        "No license",
        "No license found in original repo. Use at your own risk.",
        "Unknown",
    ),
]


def generate_voice_readme(
    voice_id: str, original_repo: str, license_id: str, license_note: str, attribution: str
) -> str:
    safe = license_id in ("CC0-1.0", "CC-BY-4.0", "Apache-2.0")
    status = "Freely redistributable" if safe else "See license note below"
    return f"""# {voice_id}

Piper ONNX TTS voice model, redistributed as part of [EdgeVox](https://github.com/edgevox-ai/edgevox).

| Field | Value |
|-------|-------|
| Voice ID | `{voice_id}` |
| Original source | [{original_repo}](https://huggingface.co/{original_repo}) |
| License | {license_id} |
| Status | {status} |
| Training data | {attribution} |

## License Note

{license_note}

## Usage

```python
from huggingface_hub import hf_hub_download

model_path = hf_hub_download("{REPO_ID}", "piper/{voice_id}/model.onnx")
config_path = hf_hub_download("{REPO_ID}", "piper/{voice_id}/config.json")
```
"""


def generate_repo_readme() -> str:
    rows = []
    for voice_id, original_repo, license_id, _license_note, _attribution in PIPER_VOICES:
        safe = license_id in ("CC0-1.0", "CC-BY-4.0", "Apache-2.0")
        icon = "Y" if safe else "?"
        rows.append(
            f"| `{voice_id}` | {license_id} | {icon} | [{original_repo}](https://huggingface.co/{original_repo}) |"
        )

    voice_table = "\n".join(rows)

    stt_rows = []
    for model_id, original_repo, _files, license_id, _note, _attr in STT_MODELS:
        stt_rows.append(f"| `{model_id}` | {license_id} | [{original_repo}](https://huggingface.co/{original_repo}) |")
    stt_table = "\n".join(stt_rows)

    return f"""---
license: other
license_name: mixed
license_link: LICENSE.md
language:
  - vi
  - de
  - ru
  - ar
  - id
  - ko
  - th
tags:
  - tts
  - stt
  - piper
  - onnx
  - edgevox
  - edge-ai
---

# EdgeVox Models

Consolidated STT and TTS models for [EdgeVox](https://github.com/edgevox-ai/edgevox) — sub-second local voice AI for robots and edge devices.

## STT Models

| Model | License | Original Source |
|-------|---------|---------|
{stt_table}

## Piper Voices

All Piper voices are ONNX models based on the VITS architecture. They run in real-time on CPU.

| Voice | License | Unrestricted? | Original Source |
|-------|---------|:---:|---------|
{voice_table}

**Legend:** Y = freely redistributable, ? = license unclear or has restrictions (see per-voice README)

## Other TTS Models (also included in this repo)

| Model | Languages | License | Size |
|-------|-----------|---------|------|
| Kokoro-82M | en, en-gb, fr, es, hi, it, pt, ja, zh (25 voices) | Apache 2.0 | ~338 MB |
| Supertonic-2 | ko, en, es, pt, fr (10 voices) | MIT (code) + OpenRAIL-M (weights) | ~255 MB |
| PyThaiTTS | th (1 voice) | Apache 2.0 | ~163 MB |

## Usage

```python
from huggingface_hub import hf_hub_download

# Piper voice
model = hf_hub_download("nrl-ai/edgevox-models", "piper/de-thorsten/model.onnx")
config = hf_hub_download("nrl-ai/edgevox-models", "piper/de-thorsten/config.json")

# Kokoro
kokoro_model = hf_hub_download("nrl-ai/edgevox-models", "kokoro/kokoro-v1.0.onnx")
kokoro_voices = hf_hub_download("nrl-ai/edgevox-models", "kokoro/voices-v1.0.bin")

# Supertonic
sup_encoder = hf_hub_download("nrl-ai/edgevox-models", "supertonic/onnx/text_encoder.onnx")
```

## License

This repository contains models under **mixed licenses**. Each voice subdirectory has its own README
with license details. See individual voice directories for specifics.

Models marked as CC0 or CC-BY-4.0 are freely usable. Models with unknown or NC/SA licenses are
included for convenience with clear notes — check the license before commercial use.
"""


def generate_license_md() -> str:
    stt_sections = []
    for model_id, original_repo, _files, license_id, license_note, _attr in STT_MODELS:
        stt_sections.append(
            f"### stt/{model_id}\n- License: {license_id}\n- Source: {original_repo}\n- {license_note}\n"
        )

    tts_sections = []
    for voice_id, original_repo, license_id, license_note, _attribution in PIPER_VOICES:
        tts_sections.append(
            f"### piper/{voice_id}\n- License: {license_id}\n- Source: {original_repo}\n- {license_note}\n"
        )
    return f"""# License Information

This repository contains STT and TTS models under mixed licenses.
Each model's license is determined by its original training data.

## STT Models

{chr(10).join(stt_sections)}

## Kokoro TTS

### kokoro/
- License: {KOKORO_MODEL["license"]}
- Source: {KOKORO_MODEL["source_url"]}
- {KOKORO_MODEL["license_note"]}

## Supertonic TTS

### supertonic/
- License: {SUPERTONIC_MODEL["license"]}
- Source: {SUPERTONIC_MODEL["source_repo"]}
- {SUPERTONIC_MODEL["license_note"]}

## PyThaiTTS

### pythaitts/
- License: {PYTHAITTS_MODEL["license"]}
- Source: {PYTHAITTS_MODEL["source_repo"]}
- {PYTHAITTS_MODEL["license_note"]}

## Piper Voice Models

{chr(10).join(tts_sections)}

## Disclaimer

Models with \"Unknown\" or \"No license\" status are redistributed for research and
educational convenience. If you are the rights holder and wish to have your model
removed, please open an issue.
"""


def main():
    api = HfApi()

    # Create the repo if it doesn't exist
    try:
        api.create_repo(repo_id=REPO_ID, repo_type="model", exist_ok=True)
        print(f"Repo {REPO_ID} ready.")
    except Exception as e:
        print(f"Note: {e}")

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)

        # Generate repo-level files
        (tmp / "README.md").write_text(generate_repo_readme())
        (tmp / "LICENSE.md").write_text(generate_license_md())

        # --- Download STT models ---
        for model_id, original_repo, files, license_id, license_note, attribution in STT_MODELS:
            model_dir = tmp / "stt" / model_id
            model_dir.mkdir(parents=True, exist_ok=True)

            print(f"Downloading STT {model_id} from {original_repo}...")
            try:
                src_dir = Path(snapshot_download(original_repo, allow_patterns=files))
                for f in files:
                    shutil.copy2(src_dir / f, model_dir / f)
            except Exception as e:
                print(f"  ERROR downloading {model_id}: {e}")
                shutil.rmtree(model_dir)
                continue

            (model_dir / "README.md").write_text(
                f"# {model_id}\n\n"
                f"Sherpa-ONNX STT model, redistributed as part of EdgeVox.\n\n"
                f"| Field | Value |\n|-------|-------|\n"
                f"| Original source | [{original_repo}](https://huggingface.co/{original_repo}) |\n"
                f"| License | {license_id} |\n"
                f"| Training data | {attribution} |\n\n"
                f"## License Note\n\n{license_note}\n"
            )
            print(f"  OK: {model_id}")

        # --- Download Kokoro TTS model ---
        kokoro_dir = tmp / "kokoro"
        kokoro_dir.mkdir(parents=True, exist_ok=True)
        print(f"Downloading Kokoro from {KOKORO_MODEL['source_url']}...")
        try:
            import urllib.request

            for fname in KOKORO_MODEL["files"]:
                url = f"{KOKORO_MODEL['source_url']}/{fname}"
                dest = kokoro_dir / fname
                if not dest.exists():
                    print(f"  Fetching {fname}...")
                    urllib.request.urlretrieve(url, dest)
            (kokoro_dir / "README.md").write_text(
                f"# {KOKORO_MODEL['id']}\n\n"
                f"Kokoro-82M ONNX TTS model (9 languages, 25 voices).\n\n"
                f"| Field | Value |\n|-------|-------|\n"
                f"| License | {KOKORO_MODEL['license']} |\n"
                f"| Attribution | {KOKORO_MODEL['attribution']} |\n\n"
                f"## License Note\n\n{KOKORO_MODEL['license_note']}\n"
            )
            print("  OK: kokoro")
        except Exception as e:
            print(f"  ERROR downloading kokoro: {e}")

        # --- Download Supertonic TTS model ---
        sup = SUPERTONIC_MODEL
        sup_dir = tmp / "supertonic"
        sup_dir.mkdir(parents=True, exist_ok=True)
        all_sup_files = sup["onnx_files"] + sup["voice_files"] + sup["config_files"]
        print(f"Downloading Supertonic from {sup['source_repo']}...")
        try:
            for fname in all_sup_files:
                src = hf_hub_download(sup["source_repo"], fname)
                dest = sup_dir / fname
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dest)
            (sup_dir / "README.md").write_text(
                f"# {sup['id']}\n\n"
                f"Supertonic-2 ONNX TTS model (Korean + 4 other languages, 10 voices).\n\n"
                f"| Field | Value |\n|-------|-------|\n"
                f"| Original source | [{sup['source_repo']}](https://huggingface.co/{sup['source_repo']}) |\n"
                f"| License | {sup['license']} |\n"
                f"| Attribution | {sup['attribution']} |\n\n"
                f"## License Note\n\n{sup['license_note']}\n"
            )
            print("  OK: supertonic")
        except Exception as e:
            print(f"  ERROR downloading supertonic: {e}")

        # --- Download PyThaiTTS model ---
        thai = PYTHAITTS_MODEL
        thai_dir = tmp / "pythaitts"
        thai_dir.mkdir(parents=True, exist_ok=True)
        print(f"Downloading PyThaiTTS from {thai['source_repo']}...")
        try:
            for fname in thai["files"]:
                src = hf_hub_download(thai["source_repo"], fname)
                shutil.copy2(src, thai_dir / fname)
            (thai_dir / "README.md").write_text(
                f"# {thai['id']}\n\n"
                f"PyThaiTTS Tacotron2 ONNX model for Thai.\n\n"
                f"| Field | Value |\n|-------|-------|\n"
                f"| Original source | [{thai['source_repo']}](https://huggingface.co/{thai['source_repo']}) |\n"
                f"| License | {thai['license']} |\n"
                f"| Attribution | {thai['attribution']} |\n\n"
                f"## License Note\n\n{thai['license_note']}\n"
            )
            print("  OK: pythaitts")
        except Exception as e:
            print(f"  ERROR downloading pythaitts: {e}")

        # --- Download Piper TTS voices ---
        for voice_id, original_repo, license_id, license_note, attribution in PIPER_VOICES:
            voice_dir = tmp / "piper" / voice_id
            voice_dir.mkdir(parents=True, exist_ok=True)

            print(f"Downloading TTS {voice_id} from {original_repo}...")
            try:
                model_path = hf_hub_download(repo_id=original_repo, filename="model.onnx")
                config_path = hf_hub_download(repo_id=original_repo, filename="config.json")
                shutil.copy2(model_path, voice_dir / "model.onnx")
                shutil.copy2(config_path, voice_dir / "config.json")
            except Exception as e:
                print(f"  ERROR downloading {voice_id}: {e}")
                shutil.rmtree(voice_dir)
                continue

            readme = generate_voice_readme(voice_id, original_repo, license_id, license_note, attribution)
            (voice_dir / "README.md").write_text(readme)

            print(f"  OK: {voice_id}")

        # Upload everything
        print(f"\nUploading to {REPO_ID}...")
        api.upload_folder(
            repo_id=REPO_ID,
            folder_path=str(tmp),
            repo_type="model",
            commit_message="Add all STT and TTS models for EdgeVox",
        )
        print("Done!")


if __name__ == "__main__":
    main()
