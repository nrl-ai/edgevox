"""Tests for edgevox.llm — LLM wrapper with mocked llama-cpp-python."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from edgevox.llm.llamacpp import (
    DEFAULT_HF_FILE,
    DEFAULT_HF_REPO,
    LANGUAGE_HINTS,
    SYSTEM_PROMPT,
    _detect_gpu_layers,
    _resolve_model_path,
    get_system_prompt,
)


class TestGetSystemPrompt:
    def test_english_no_hint(self):
        prompt = get_system_prompt("en")
        assert prompt == SYSTEM_PROMPT

    def test_vietnamese_has_hint(self):
        prompt = get_system_prompt("vi")
        assert prompt.startswith("Respond in Vietnamese")
        assert SYSTEM_PROMPT in prompt

    def test_unknown_language_no_hint(self):
        prompt = get_system_prompt("xx")
        assert prompt == SYSTEM_PROMPT

    def test_all_hints_present(self):
        for lang_code in LANGUAGE_HINTS:
            prompt = get_system_prompt(lang_code)
            assert SYSTEM_PROMPT in prompt
            assert len(prompt) > len(SYSTEM_PROMPT)


class TestDetectGpuLayers:
    @patch("edgevox.core.gpu.has_metal", return_value=False)
    @patch("edgevox.core.gpu.get_nvidia_vram_gb", return_value=8.0)
    def test_nvidia_8gb(self, _vram, _metal):
        assert _detect_gpu_layers() == -1

    @patch("edgevox.core.gpu.has_metal", return_value=False)
    @patch("edgevox.core.gpu.get_nvidia_vram_gb", return_value=4.0)
    def test_nvidia_4gb(self, _vram, _metal):
        assert _detect_gpu_layers() == 20

    @patch("edgevox.core.gpu.has_metal", return_value=True)
    @patch("edgevox.core.gpu.get_nvidia_vram_gb", return_value=None)
    def test_metal(self, _vram, _metal):
        assert _detect_gpu_layers() == -1

    @patch("edgevox.core.gpu.has_metal", return_value=False)
    @patch("edgevox.core.gpu.get_nvidia_vram_gb", return_value=None)
    def test_cpu_only(self, _vram, _metal):
        assert _detect_gpu_layers() == 0


class TestResolveModelPath:
    def test_local_file(self, tmp_path):
        model_file = tmp_path / "model.gguf"
        model_file.touch()
        result = _resolve_model_path(str(model_file))
        assert result == str(model_file)

    @patch("huggingface_hub.hf_hub_download", return_value="/cache/model.gguf")
    def test_hf_prefix(self, mock_hf):
        result = _resolve_model_path("hf:user/repo:model.gguf")
        assert result == "/cache/model.gguf"
        mock_hf.assert_called_once_with(repo_id="user/repo", filename="model.gguf")

    def test_hf_no_filename_raises(self):
        with pytest.raises(ValueError, match="hf:repo/name:filename"):
            _resolve_model_path("hf:user/repo")

    @patch("huggingface_hub.hf_hub_download", return_value="/cache/default.gguf")
    def test_none_downloads_default(self, mock_hf):
        result = _resolve_model_path(None)
        assert result == "/cache/default.gguf"
        mock_hf.assert_called_once_with(repo_id=DEFAULT_HF_REPO, filename=DEFAULT_HF_FILE)

    @patch("huggingface_hub.hf_hub_download", return_value="/cache/qwen.gguf")
    def test_preset_prefix(self, mock_hf):
        from edgevox.llm.models import PRESETS

        result = _resolve_model_path("preset:qwen3-1.7b")
        assert result == "/cache/qwen.gguf"
        expected = PRESETS["qwen3-1.7b"]
        mock_hf.assert_called_once_with(repo_id=expected.repo, filename=expected.filename)

    @patch("huggingface_hub.hf_hub_download", return_value="/cache/llama.gguf")
    def test_bare_preset_slug(self, mock_hf):
        from edgevox.llm.models import PRESETS

        result = _resolve_model_path("llama-3.2-3b")
        assert result == "/cache/llama.gguf"
        expected = PRESETS["llama-3.2-3b"]
        mock_hf.assert_called_once_with(repo_id=expected.repo, filename=expected.filename)

    def test_unknown_preset_raises(self):
        with pytest.raises(KeyError, match="Unknown LLM preset"):
            _resolve_model_path("preset:does-not-exist")

    def test_bogus_path_raises(self):
        with pytest.raises(FileNotFoundError):
            _resolve_model_path("/nonexistent/path/to/model.gguf")


class TestLLM:
    def _make_llm(self, mock_llama_cls):
        """Helper to create an LLM with mocked dependencies."""
        mock_llama = MagicMock()
        mock_llama_cls.return_value = mock_llama

        with patch("edgevox.llm.llamacpp._resolve_model_path", return_value="/tmp/model.gguf"):
            from edgevox.llm.llamacpp import LLM

            llm = LLM(model_path="/tmp/model.gguf")
        return llm, mock_llama

    @patch("llama_cpp.Llama")
    @patch("edgevox.core.gpu.has_metal", return_value=False)
    @patch("edgevox.core.gpu.get_nvidia_vram_gb", return_value=None)
    def test_init(self, _vram, _metal, mock_llama_cls):
        llm, _ = self._make_llm(mock_llama_cls)
        assert llm._language == "en"
        assert len(llm._history) == 1
        assert llm._history[0]["role"] == "system"

    @patch("llama_cpp.Llama")
    @patch("edgevox.core.gpu.has_metal", return_value=False)
    @patch("edgevox.core.gpu.get_nvidia_vram_gb", return_value=None)
    def test_chat(self, _vram, _metal, mock_llama_cls):
        llm, mock_llama = self._make_llm(mock_llama_cls)
        mock_llama.create_chat_completion.return_value = {"choices": [{"message": {"content": "Hello back!"}}]}
        reply = llm.chat("Hello")
        assert reply == "Hello back!"
        assert len(llm._history) == 3  # system + user + assistant

    @patch("llama_cpp.Llama")
    @patch("edgevox.core.gpu.has_metal", return_value=False)
    @patch("edgevox.core.gpu.get_nvidia_vram_gb", return_value=None)
    def test_chat_stream(self, _vram, _metal, mock_llama_cls):
        llm, mock_llama = self._make_llm(mock_llama_cls)
        mock_llama.create_chat_completion.return_value = iter(
            [
                {"choices": [{"delta": {"content": "Hello"}}]},
                {"choices": [{"delta": {"content": " world"}}]},
                {"choices": [{"delta": {}}]},
            ]
        )
        tokens = list(llm.chat_stream("Hi"))
        assert tokens == ["Hello", " world"]

    @patch("llama_cpp.Llama")
    @patch("edgevox.core.gpu.has_metal", return_value=False)
    @patch("edgevox.core.gpu.get_nvidia_vram_gb", return_value=None)
    def test_reset(self, _vram, _metal, mock_llama_cls):
        llm, mock_llama = self._make_llm(mock_llama_cls)
        mock_llama.create_chat_completion.return_value = {"choices": [{"message": {"content": "reply"}}]}
        llm.chat("test")
        assert len(llm._history) == 3
        llm.reset()
        assert len(llm._history) == 1
        assert llm._history[0]["role"] == "system"

    @patch("llama_cpp.Llama")
    @patch("edgevox.core.gpu.has_metal", return_value=False)
    @patch("edgevox.core.gpu.get_nvidia_vram_gb", return_value=None)
    def test_set_language(self, _vram, _metal, mock_llama_cls):
        llm, _ = self._make_llm(mock_llama_cls)
        llm.set_language("fr")
        assert llm._language == "fr"
        assert "French" in llm._history[0]["content"]

    @patch("llama_cpp.Llama")
    @patch("edgevox.core.gpu.has_metal", return_value=False)
    @patch("edgevox.core.gpu.get_nvidia_vram_gb", return_value=None)
    def test_history_truncation(self, _vram, _metal, mock_llama_cls):
        llm, mock_llama = self._make_llm(mock_llama_cls)
        mock_llama.create_chat_completion.return_value = {"choices": [{"message": {"content": "reply"}}]}
        for i in range(15):
            llm.chat(f"msg {i}")
        # 1 system + 15 user + 15 assistant = 31, truncated to 1 + 20 = 21
        assert len(llm._history) == 21
        assert llm._history[0]["role"] == "system"
