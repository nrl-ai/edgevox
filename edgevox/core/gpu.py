"""Lightweight GPU detection without torch.

Uses nvidia-smi (always shipped with NVIDIA drivers) and platform checks
for Apple Metal, so the rest of the codebase never needs to import torch
just for hardware probing.
"""

from __future__ import annotations

import logging
import platform
import shutil
import subprocess

log = logging.getLogger(__name__)


def get_nvidia_vram_gb() -> float | None:
    """Return total VRAM in GB for GPU 0 via nvidia-smi, or None."""
    nvsmi = shutil.which("nvidia-smi")
    if not nvsmi:
        return None
    try:
        out = subprocess.check_output(
            [nvsmi, "--query-gpu=memory.total", "--format=csv,noheader,nounits", "--id=0"],
            timeout=5,
            text=True,
        )
        mb = float(out.strip())
        return mb / 1024
    except Exception as e:
        log.debug(f"nvidia-smi failed: {e}")
        return None


def get_nvidia_gpu_name() -> str | None:
    """Return GPU name string, or None."""
    nvsmi = shutil.which("nvidia-smi")
    if not nvsmi:
        return None
    try:
        out = subprocess.check_output(
            [nvsmi, "--query-gpu=name", "--format=csv,noheader", "--id=0"],
            timeout=5,
            text=True,
        )
        return out.strip() or None
    except Exception:
        return None


def get_nvidia_used_mb() -> float | None:
    """Return used VRAM in MB for GPU 0, or None."""
    nvsmi = shutil.which("nvidia-smi")
    if not nvsmi:
        return None
    try:
        out = subprocess.check_output(
            [nvsmi, "--query-gpu=memory.used", "--format=csv,noheader,nounits", "--id=0"],
            timeout=5,
            text=True,
        )
        return float(out.strip())
    except Exception:
        return None


def has_cuda() -> bool:
    """Check if CUDA GPU is available (nvidia-smi present and working)."""
    return get_nvidia_vram_gb() is not None


def has_metal() -> bool:
    """Check if Apple Metal (MPS) is likely available."""
    return platform.system() == "Darwin" and platform.machine() == "arm64"


def get_ram_gb() -> float:
    """Return total system RAM in GB."""
    try:
        import psutil

        return psutil.virtual_memory().total / (1024**3)
    except ImportError:
        pass
    try:
        with open("/proc/meminfo") as f:
            mem_kb = int(f.readline().split()[1])
        return mem_kb / (1024**2)
    except Exception:
        return 8.0
