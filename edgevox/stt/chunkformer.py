"""ChunkFormer-CTC-Large-Vie STT backend for Vietnamese.

110M params, 4.18% WER on VIVOS — best accuracy-to-size ratio.
"""

from __future__ import annotations

import logging
import tempfile
import time
from pathlib import Path

import numpy as np
import soundfile as sf

from edgevox.stt import BaseSTT

log = logging.getLogger(__name__)


class ChunkFormerSTT(BaseSTT):
    """ChunkFormer-CTC-Large-Vie for Vietnamese."""

    MODEL_ID = "khanhld/chunkformer-ctc-large-vie"

    def __init__(self, device: str | None = None):
        from chunkformer import ChunkFormerModel

        from edgevox.core.gpu import has_cuda

        if device is None:
            device = "cuda" if has_cuda() else "cpu"
        self._device = device
        self._model_size = "chunkformer-ctc-large-vie"

        log.info(f"Loading ChunkFormer-CTC-Large-Vie on {device}...")
        self._model = ChunkFormerModel.from_pretrained(self.MODEL_ID)
        if device == "cuda":
            self._model = self._model.cuda()
        self._model.eval()
        self._warmed_up = False
        log.info("ChunkFormer loaded (110M params).")

    def transcribe(self, audio: np.ndarray, language: str = "vi") -> str:
        t0 = time.perf_counter()

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            sf.write(tmp.name, audio, 16000)
            tmp_path = tmp.name

        try:
            result = self._model.endless_decode(
                audio_path=tmp_path,
                chunk_size=64,
                left_context_size=128,
                right_context_size=128,
            )
        finally:
            Path(tmp_path).unlink(missing_ok=True)

        if isinstance(result, list):
            text = (
                " ".join(r.get("decode", r.get("text", str(r))) if isinstance(r, dict) else str(r) for r in result)
                .strip()
                .capitalize()
            )
        elif isinstance(result, dict):
            text = result.get("decode", result.get("text", str(result))).strip().capitalize()
        else:
            text = str(result).strip().capitalize()

        elapsed = time.perf_counter() - t0
        if not self._warmed_up:
            self._warmed_up = True
            log.info(f'STT ChunkFormer (warmup): {elapsed:.2f}s -> "{text}"')
        else:
            log.info(f'STT ChunkFormer: {elapsed:.2f}s -> "{text}"')
        return text
