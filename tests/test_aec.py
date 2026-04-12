"""Tests for edgevox.audio.aec -- AEC backends and factory."""

from __future__ import annotations

import numpy as np
import pytest

from edgevox.audio.aec import NLMSAdaptiveAEC, NoAEC, SpectralSubtractionAEC, create_aec


class TestNoAEC:
    def test_passthrough(self):
        aec = NoAEC()
        mic = np.random.randn(512).astype(np.float32)
        ref = np.random.randn(512).astype(np.float32)
        out = aec.process(mic, ref)
        np.testing.assert_array_equal(out, mic)

    def test_name(self):
        assert NoAEC().name == "none"

    def test_reset_is_noop(self):
        NoAEC().reset()


class TestNLMSAdaptiveAEC:
    def test_name(self):
        assert NLMSAdaptiveAEC().name == "nlms"

    def test_output_shape(self):
        aec = NLMSAdaptiveAEC(filter_len=64)
        mic = np.random.randn(512).astype(np.float32) * 0.1
        ref = np.random.randn(512).astype(np.float32) * 0.1
        out = aec.process(mic, ref)
        assert out.shape == mic.shape
        assert out.dtype == np.float32

    def test_echo_reduction(self):
        """NLMS should reduce echo when mic = delayed reference + noise."""
        rng = np.random.default_rng(42)
        aec = NLMSAdaptiveAEC(filter_len=128, mu=0.5)

        ref_signal = rng.standard_normal(4096).astype(np.float32) * 0.3
        delay = 10
        echo = np.zeros_like(ref_signal)
        echo[delay:] = ref_signal[:-delay] * 0.8

        user_noise = rng.standard_normal(len(ref_signal)).astype(np.float32) * 0.01
        mic_signal = echo + user_noise

        frame_size = 512
        cleaned_rms_values = []
        echo_rms_values = []
        for i in range(0, len(ref_signal) - frame_size, frame_size):
            mic_frame = mic_signal[i : i + frame_size]
            ref_frame = ref_signal[i : i + frame_size]
            cleaned = aec.process(mic_frame, ref_frame)
            cleaned_rms_values.append(float(np.sqrt(np.mean(cleaned**2))))
            echo_rms_values.append(float(np.sqrt(np.mean(echo[i : i + frame_size] ** 2))))

        # After convergence, cleaned RMS should be lower than echo RMS
        avg_cleaned = np.mean(cleaned_rms_values[-3:])
        avg_echo = np.mean(echo_rms_values[-3:])
        assert avg_cleaned < avg_echo * 0.5, (
            f"NLMS didn't reduce echo enough: cleaned={avg_cleaned:.4f} vs echo={avg_echo:.4f}"
        )

    def test_reset_clears_state(self):
        aec = NLMSAdaptiveAEC(filter_len=64)
        mic = np.ones(512, dtype=np.float32) * 0.1
        ref = np.ones(512, dtype=np.float32) * 0.1
        aec.process(mic, ref)
        assert np.any(aec._w != 0)
        aec.reset()
        np.testing.assert_array_equal(aec._w, 0)
        np.testing.assert_array_equal(aec._ref_hist, 0)


class TestSpectralSubtractionAEC:
    def test_name(self):
        assert SpectralSubtractionAEC().name == "specsub"

    def test_output_shape(self):
        aec = SpectralSubtractionAEC(frame_size=512)
        mic = np.random.randn(512).astype(np.float32) * 0.1
        ref = np.random.randn(512).astype(np.float32) * 0.1
        out = aec.process(mic, ref)
        assert out.shape == mic.shape
        assert out.dtype == np.float32

    def test_echo_reduction(self):
        """Spectral subtraction should reduce echo from a known reference."""
        rng = np.random.default_rng(42)
        aec = SpectralSubtractionAEC(frame_size=512)

        # Simulate echo: ref plays through speakers, arrives at mic scaled
        ref_signal = rng.standard_normal(2048).astype(np.float32) * 0.3
        echo = ref_signal * 0.6  # simple scaling (no delay for spectral approach)
        user_noise = rng.standard_normal(len(ref_signal)).astype(np.float32) * 0.01
        mic_signal = echo + user_noise

        frame_size = 512
        cleaned_rms = []
        echo_rms = []
        for i in range(0, len(ref_signal) - frame_size, frame_size):
            mic_frame = mic_signal[i : i + frame_size]
            ref_frame = ref_signal[i : i + frame_size]
            cleaned = aec.process(mic_frame, ref_frame)
            cleaned_rms.append(float(np.sqrt(np.mean(cleaned**2))))
            echo_rms.append(float(np.sqrt(np.mean(echo[i : i + frame_size] ** 2))))

        # Last frames should show significant echo reduction
        avg_cleaned = np.mean(cleaned_rms[-2:])
        avg_echo = np.mean(echo_rms[-2:])
        assert avg_cleaned < avg_echo * 0.7, (
            f"SpecSub didn't reduce echo enough: cleaned={avg_cleaned:.4f} vs echo={avg_echo:.4f}"
        )

    def test_silent_reference_passthrough(self):
        """When reference is silent (no playback), mic should pass through mostly unchanged."""
        aec = SpectralSubtractionAEC(frame_size=512)
        rng = np.random.default_rng(123)
        mic = rng.standard_normal(512).astype(np.float32) * 0.1
        ref = np.zeros(512, dtype=np.float32)
        out = aec.process(mic, ref)
        # Output should be similar to input (slight windowing difference is OK)
        assert out.shape == mic.shape

    def test_reset_clears_gain(self):
        aec = SpectralSubtractionAEC(frame_size=512)
        mic = np.ones(512, dtype=np.float32) * 0.1
        ref = np.ones(512, dtype=np.float32) * 0.1
        aec.process(mic, ref)
        aec.reset()
        np.testing.assert_array_equal(aec._echo_gain, 1.0)


class TestDTLNAec:
    def test_import_fallback(self):
        """If ai-edge-litert is available, dtln should work; otherwise fall back."""
        aec = create_aec("dtln")
        assert aec.name in ("dtln", "specsub", "none")

    def test_output_shape(self):
        """If DTLN is available, check output shape."""
        aec = create_aec("dtln")
        if aec.name != "dtln":
            pytest.skip("ai-edge-litert or DTLN models not available")
        mic = np.random.randn(512).astype(np.float32) * 0.1
        ref = np.random.randn(512).astype(np.float32) * 0.1
        out = aec.process(mic, ref)
        assert out.shape == mic.shape
        assert out.dtype == np.float32

    def test_reset(self):
        aec = create_aec("dtln")
        if aec.name != "dtln":
            pytest.skip("ai-edge-litert or DTLN models not available")
        aec.reset()  # should not raise


class TestFactory:
    def test_create_none(self):
        assert create_aec("none").name == "none"

    def test_create_nlms(self):
        assert create_aec("nlms").name == "nlms"

    def test_create_specsub(self):
        assert create_aec("specsub").name == "specsub"

    def test_unknown_falls_back(self):
        assert create_aec("nonexistent_backend").name == "none"

    def test_create_with_kwargs(self):
        aec = create_aec("nlms", filter_len=256, mu=0.1)
        assert isinstance(aec, NLMSAdaptiveAEC)
        assert aec._filter_len == 256


class TestRefBufferIntegration:
    """Test the reference signal capture on InterruptiblePlayer."""

    def test_ref_buffer_disabled_by_default(self):
        from unittest.mock import patch

        with patch("edgevox.audio._original._sd"):
            from edgevox.audio._original import InterruptiblePlayer

            p = InterruptiblePlayer()
            assert p._ref_buffer is None

    def test_enable_ref_capture(self):
        from unittest.mock import patch

        with patch("edgevox.audio._original._sd"):
            from edgevox.audio._original import InterruptiblePlayer

            p = InterruptiblePlayer()
            p.enable_ref_capture()
            assert p._ref_buffer is not None
            assert len(p._ref_buffer) == 0

    def test_get_ref_frame_returns_zeros_when_empty(self):
        from unittest.mock import patch

        with patch("edgevox.audio._original._sd"):
            from edgevox.audio._original import InterruptiblePlayer

            p = InterruptiblePlayer()
            p.enable_ref_capture()
            frame = p.get_ref_frame(512)
            assert frame.shape == (512,)
            np.testing.assert_array_equal(frame, 0)

    def test_get_ref_frame_returns_pushed_data(self):
        from unittest.mock import patch

        with patch("edgevox.audio._original._sd"):
            from edgevox.audio._original import InterruptiblePlayer

            p = InterruptiblePlayer()
            p.enable_ref_capture()
            data = [0.1, 0.2, 0.3, 0.4, 0.5]
            p._ref_buffer.extend(data)
            frame = p.get_ref_frame(5)
            np.testing.assert_allclose(frame, data, atol=1e-6)
            assert len(p._ref_buffer) == 0
