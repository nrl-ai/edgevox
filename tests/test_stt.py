"""Tests for edgevox.stt — STT backends and factory, with mocked model loading."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from edgevox.stt import BaseSTT


class TestBaseSTT:
    def test_transcribe_raises(self):
        with pytest.raises(NotImplementedError):
            BaseSTT().transcribe(np.zeros(1000))


class TestCreateSTTFactory:
    @pytest.fixture(autouse=True)
    def _mock_sherpa_module(self):
        """Pre-populate sys.modules for sherpa_onnx since it needs a native lib."""
        import sys

        mock_module = MagicMock()
        needs_cleanup = "sherpa_onnx" not in sys.modules
        if needs_cleanup:
            sys.modules["sherpa_onnx"] = mock_module
            sys.modules["sherpa_onnx.lib"] = MagicMock()
            sys.modules["sherpa_onnx.lib._sherpa_onnx"] = MagicMock()
        yield mock_module
        if needs_cleanup:
            sys.modules.pop("sherpa_onnx", None)
            sys.modules.pop("sherpa_onnx.lib", None)
            sys.modules.pop("sherpa_onnx.lib._sherpa_onnx", None)
            sys.modules.pop("edgevox.stt.sherpa_stt", None)

    @patch("faster_whisper.WhisperModel")
    @patch("edgevox.core.gpu.get_nvidia_vram_gb", return_value=None)
    @patch("edgevox.core.gpu.get_ram_gb", return_value=16.0)
    def test_english_uses_whisper(self, _ram, _vram, mock_whisper):
        from edgevox.stt import create_stt

        stt = create_stt("en")
        assert type(stt).__name__ == "WhisperSTT"

    @patch("faster_whisper.WhisperModel")
    @patch("edgevox.core.gpu.get_nvidia_vram_gb", return_value=None)
    @patch("edgevox.core.gpu.get_ram_gb", return_value=16.0)
    def test_french_uses_whisper(self, _ram, _vram, mock_whisper):
        from edgevox.stt import create_stt

        stt = create_stt("fr")
        assert type(stt).__name__ == "WhisperSTT"

    @patch("edgevox.stt.sherpa_stt.hf_hub_download", return_value="/tmp/fake_model")
    @patch("edgevox.stt.sherpa_stt.snapshot_download", return_value="/tmp/fake_dir")
    @patch("edgevox.core.gpu.has_cuda", return_value=False)
    def test_vi_uses_sherpa(self, _cuda, _snap, _hf, _mock_sherpa_module):
        import sys

        sys.modules.pop("edgevox.stt.sherpa_stt", None)
        from edgevox.stt import create_stt

        stt = create_stt("vi")
        assert type(stt).__name__ == "SherpaSTT"

    @patch("faster_whisper.WhisperModel")
    @patch("edgevox.core.gpu.get_nvidia_vram_gb", return_value=None)
    @patch("edgevox.core.gpu.get_ram_gb", return_value=16.0)
    def test_vi_sherpa_fallback_to_whisper(self, _ram, _vram, mock_whisper, _mock_sherpa_module):
        import sys

        # Make sherpa_onnx.OfflineRecognizer.from_transducer raise so SherpaSTT init fails
        sherpa_mock = sys.modules["sherpa_onnx"]
        sherpa_mock.OfflineRecognizer.from_transducer.side_effect = RuntimeError("no model")

        sys.modules.pop("edgevox.stt.sherpa_stt", None)
        from edgevox.stt import create_stt

        stt = create_stt("vi")
        assert type(stt).__name__ == "WhisperSTT"

        # Reset side_effect for other tests
        sherpa_mock.OfflineRecognizer.from_transducer.side_effect = None


class TestWhisperSTT:
    @patch("faster_whisper.WhisperModel")
    @patch("edgevox.core.gpu.get_nvidia_vram_gb", return_value=None)
    @patch("edgevox.core.gpu.get_ram_gb", return_value=16.0)
    def test_auto_model_cpu_16gb(self, _ram, _vram, mock_whisper):
        from edgevox.stt.whisper import WhisperSTT

        stt = WhisperSTT()
        assert stt._model_size == "medium"
        assert stt._device == "cpu"

    @patch("faster_whisper.WhisperModel")
    @patch("edgevox.core.gpu.get_nvidia_vram_gb", return_value=8.0)
    @patch("edgevox.core.gpu.get_ram_gb", return_value=32.0)
    def test_auto_model_gpu_8gb(self, _ram, _vram, mock_whisper):
        from edgevox.stt.whisper import WhisperSTT

        stt = WhisperSTT()
        assert stt._model_size == "large-v3-turbo"
        assert stt._device == "cuda"

    @patch("faster_whisper.WhisperModel")
    def test_explicit_model_size(self, mock_whisper):
        from edgevox.stt.whisper import WhisperSTT

        stt = WhisperSTT(model_size="tiny", device="cpu")
        assert stt._model_size == "tiny"
        assert stt._device == "cpu"

    @patch("faster_whisper.WhisperModel")
    def test_transcribe(self, mock_whisper_cls):
        from edgevox.stt.whisper import WhisperSTT

        mock_seg = MagicMock()
        mock_seg.text = " hello world "
        mock_model = MagicMock()
        mock_model.transcribe.return_value = ([mock_seg], MagicMock())
        mock_whisper_cls.return_value = mock_model

        stt = WhisperSTT(model_size="tiny", device="cpu")
        text = stt.transcribe(np.zeros(16000, dtype=np.float32))
        assert text == "hello world"


class TestSherpaSTT:
    @pytest.fixture(autouse=True)
    def _mock_sherpa_module(self):
        """Pre-populate sys.modules for sherpa_onnx since it needs a native lib."""
        import sys

        mock_module = MagicMock()
        needs_cleanup = "sherpa_onnx" not in sys.modules
        if needs_cleanup:
            sys.modules["sherpa_onnx"] = mock_module
            sys.modules["sherpa_onnx.lib"] = MagicMock()
            sys.modules["sherpa_onnx.lib._sherpa_onnx"] = MagicMock()
        yield mock_module
        if needs_cleanup:
            sys.modules.pop("sherpa_onnx", None)
            sys.modules.pop("sherpa_onnx.lib", None)
            sys.modules.pop("sherpa_onnx.lib._sherpa_onnx", None)
            sys.modules.pop("edgevox.stt.sherpa_stt", None)

    @patch("edgevox.stt.sherpa_stt.hf_hub_download", return_value="/tmp/fake")
    @patch("edgevox.core.gpu.has_cuda", return_value=False)
    def test_transcribe(self, _cuda, _hf, _mock_sherpa_module):
        import sys

        mock_recognizer = MagicMock()
        mock_stream = MagicMock()
        mock_stream.result.text = " xin chào "
        mock_recognizer.create_stream.return_value = mock_stream

        # Set up the mock so sherpa_onnx.OfflineRecognizer.from_transducer returns our mock
        sherpa_mock = sys.modules["sherpa_onnx"]
        sherpa_mock.OfflineRecognizer.from_transducer.return_value = mock_recognizer

        sys.modules.pop("edgevox.stt.sherpa_stt", None)
        from edgevox.stt.sherpa_stt import SherpaSTT

        stt = SherpaSTT()
        text = stt.transcribe(np.zeros(16000, dtype=np.float32))
        assert text == "Xin chào"


class TestChunkFormerSTT:
    @pytest.fixture(autouse=True)
    def _mock_chunkformer_module(self):
        """Pre-populate sys.modules for chunkformer since it may not be installed."""
        import sys

        mock_module = MagicMock()
        needs_cleanup = "chunkformer" not in sys.modules
        if needs_cleanup:
            sys.modules["chunkformer"] = mock_module
        yield mock_module
        if needs_cleanup:
            sys.modules.pop("chunkformer", None)
            # Also remove cached edgevox.stt.chunkformer to force reimport
            sys.modules.pop("edgevox.stt.chunkformer", None)

    @patch("edgevox.core.gpu.has_cuda", return_value=False)
    def test_transcribe_list_result(self, _cuda, _mock_chunkformer_module):
        import sys

        mock_cf = sys.modules["chunkformer"]
        mock_model = MagicMock()
        mock_cf.ChunkFormerModel.from_pretrained.return_value = mock_model
        mock_model.eval.return_value = mock_model
        mock_model.endless_decode.return_value = [{"decode": "xin chào"}]

        # Force reimport to pick up our mock
        sys.modules.pop("edgevox.stt.chunkformer", None)
        from edgevox.stt.chunkformer import ChunkFormerSTT

        stt = ChunkFormerSTT()
        text = stt.transcribe(np.zeros(16000, dtype=np.float32))
        assert text == "Xin chào"

    @patch("edgevox.core.gpu.has_cuda", return_value=False)
    def test_transcribe_dict_result(self, _cuda, _mock_chunkformer_module):
        import sys

        mock_cf = sys.modules["chunkformer"]
        mock_model = MagicMock()
        mock_cf.ChunkFormerModel.from_pretrained.return_value = mock_model
        mock_model.eval.return_value = mock_model
        mock_model.endless_decode.return_value = {"text": "hello"}

        sys.modules.pop("edgevox.stt.chunkformer", None)
        from edgevox.stt.chunkformer import ChunkFormerSTT

        stt = ChunkFormerSTT()
        text = stt.transcribe(np.zeros(16000, dtype=np.float32))
        assert text == "Hello"
