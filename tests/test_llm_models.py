"""Tests for edgevox.llm.models — the GGUF preset catalog."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from edgevox.llm.models import (
    DEFAULT_PRESET,
    PRESETS,
    ModelPreset,
    download_preset,
    list_presets,
    resolve_preset,
)


class TestRegistry:
    def test_default_preset_exists(self):
        assert DEFAULT_PRESET in PRESETS

    def test_list_presets_returns_all(self):
        presets = list_presets()
        assert len(presets) == len(PRESETS)
        assert all(isinstance(p, ModelPreset) for p in presets)

    def test_each_preset_has_required_fields(self):
        for slug, preset in PRESETS.items():
            assert preset.slug == slug
            assert preset.repo, f"{slug} missing repo"
            assert preset.filename.endswith(".gguf"), f"{slug} filename not .gguf"
            assert preset.family, f"{slug} missing family"
            assert preset.size_gb > 0, f"{slug} non-positive size"
            assert preset.description, f"{slug} missing description"

    def test_resolve_preset_known(self):
        preset = resolve_preset(DEFAULT_PRESET)
        assert preset.slug == DEFAULT_PRESET

    def test_resolve_preset_unknown_raises(self):
        with pytest.raises(KeyError, match="Unknown LLM preset"):
            resolve_preset("totally-bogus-preset")

    def test_embodied_flag_has_at_least_one(self):
        # RoboBrain is the embodied entry — sanity-check we didn't drop it.
        assert any(p.embodied for p in PRESETS.values())


class TestDownloadPreset:
    @patch("huggingface_hub.hf_hub_download", return_value="/cache/llama.gguf")
    def test_download_preset_dispatches(self, mock_hf):
        path = download_preset("llama-3.2-1b")
        assert path == "/cache/llama.gguf"
        expected = PRESETS["llama-3.2-1b"]
        mock_hf.assert_called_once_with(repo_id=expected.repo, filename=expected.filename)

    def test_download_preset_unknown_raises(self):
        with pytest.raises(KeyError):
            download_preset("does-not-exist")
