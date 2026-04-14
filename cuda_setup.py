"""
KoKoFish — CUDA upgrade helper.

Downloads and installs the CUDA-enabled PyTorch build,
replacing the CPU-only version. Called from the Settings tab
when the user enables GPU acceleration.
"""

import logging
import os
import subprocess
import sys
import threading
from typing import Callable, Optional

CREATE_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0x08000000)

logger = logging.getLogger("KoKoFish.cuda_setup")


def get_venv_pip() -> str:
    """Get the pip executable path in the current venv."""
    venv_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "venv")
    pip_path = os.path.join(venv_dir, "Scripts", "pip.exe")
    if os.path.isfile(pip_path):
        return pip_path
    # Fallback: use current Python's pip
    return f"{sys.executable} -m pip"


def is_cuda_torch_installed() -> bool:
    """Check if the currently installed PyTorch has CUDA support."""
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


def _nvidia_smi_name() -> str:
    """Try nvidia-smi with full common paths, return GPU name or empty string."""
    smi_paths = [
        "nvidia-smi",
        r"C:\Windows\System32\nvidia-smi.exe",
        r"C:\Program Files\NVIDIA Corporation\NVSMI\nvidia-smi.exe",
    ]
    for smi in smi_paths:
        try:
            result = subprocess.run(
                [smi, "--query-gpu=name", "--format=csv,noheader"],
                capture_output=True, text=True, timeout=5,
                creationflags=CREATE_NO_WINDOW
            )
            if result.returncode == 0 and result.stdout.strip():
                return result.stdout.strip().split("\n")[0]
        except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
            continue
    return ""


def _wmi_gpu_name() -> str:
    """Use WMI to detect NVIDIA GPU when nvidia-smi is not on PATH."""
    try:
        result = subprocess.run(
            ["powershell", "-NoProfile", "-Command",
             "Get-WmiObject Win32_VideoController | Where-Object { $_.Name -like '*NVIDIA*' } | Select-Object -ExpandProperty Name -First 1"],
            capture_output=True, text=True, timeout=8,
            creationflags=CREATE_NO_WINDOW
        )
        name = result.stdout.strip()
        if name:
            return name
    except Exception:
        pass
    return ""


def has_nvidia_gpu() -> bool:
    """Check if an NVIDIA GPU is present (even without CUDA torch)."""
    name = _nvidia_smi_name() or _wmi_gpu_name()
    if name:
        logger.info("NVIDIA GPU detected: %s", name)
        return True
    return False


def get_nvidia_gpu_name() -> str:
    """Get the name of the NVIDIA GPU."""
    return _nvidia_smi_name() or _wmi_gpu_name() or "Unknown GPU"



def install_cuda_pytorch(
    on_progress: Optional[Callable[[str], None]] = None,
    on_complete: Optional[Callable[[bool, str], None]] = None,
):
    """
    Download and install CUDA-enabled PyTorch, replacing the CPU build in-place.
    Does NOT uninstall CPU torch first — pip --upgrade handles the swap atomically.
    If the download fails, the existing CPU torch is untouched.
    """

    def _worker():
        pip_exe = get_venv_pip()

        try:
            if on_progress:
                on_progress("Downloading CUDA PyTorch (~2.5 GB)...\nThis may take several minutes.")
            logger.info("Installing CUDA PyTorch (cu124)...")

            result = subprocess.run(
                [
                    pip_exe, "install",
                    "torch==2.6.0+cu124",
                    "torchaudio==2.6.0+cu124",
                    "--index-url", "https://download.pytorch.org/whl/cu124",
                    "--upgrade",
                ],
                capture_output=True, text=True, timeout=1800,
                creationflags=CREATE_NO_WINDOW
            )

            if result.returncode != 0:
                error_msg = result.stderr[-500:] if result.stderr else "Unknown error"
                logger.error("CUDA PyTorch install failed: %s", error_msg)
                if on_complete:
                    on_complete(False, f"Installation failed:\n{error_msg}")
                return

            # Verify
            if on_progress:
                on_progress("Verifying CUDA installation...")

            verify = subprocess.run(
                [sys.executable, "-c",
                 "import torch; print(f'PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}')"],
                capture_output=True, text=True, timeout=30,
                creationflags=CREATE_NO_WINDOW
            )

            if "CUDA: True" in verify.stdout:
                logger.info("CUDA PyTorch installed and verified!")
                if on_complete:
                    on_complete(True, "CUDA acceleration is now enabled!\nRestart KoKoFish for full effect.")
            else:
                logger.warning("CUDA PyTorch installed but CUDA not available: %s", verify.stdout)
                if on_complete:
                    on_complete(
                        True,
                        "PyTorch installed, but CUDA may not be available.\n"
                        "Make sure you have the latest NVIDIA drivers.\n"
                        f"Output: {verify.stdout.strip()}"
                    )

        except subprocess.TimeoutExpired:
            logger.error("CUDA install timed out")
            if on_complete:
                on_complete(False, "Download timed out. Check your internet connection and try again.")
        except Exception as exc:
            logger.error("CUDA install error: %s", exc, exc_info=True)
            if on_complete:
                on_complete(False, f"Error: {exc}")

    thread = threading.Thread(target=_worker, daemon=True, name="CUDA-Install")
    thread.start()
    return thread


def revert_to_cpu_pytorch(
    on_progress: Optional[Callable[[str], None]] = None,
    on_complete: Optional[Callable[[bool, str], None]] = None,
):
    """
    Revert from CUDA PyTorch back to CPU-only version.
    Uses --upgrade to swap in-place — does NOT uninstall CUDA torch first.
    Frees ~2GB of disk space once pip completes the swap.
    """

    def _worker():
        pip_exe = get_venv_pip()

        try:
            if on_progress:
                on_progress("Switching back to CPU PyTorch...")

            result = subprocess.run(
                [
                    pip_exe, "install",
                    "torch==2.6.0+cpu",
                    "torchaudio==2.6.0+cpu",
                    "--index-url", "https://download.pytorch.org/whl/cpu",
                    "--upgrade",
                ],
                capture_output=True, text=True, timeout=600,
                creationflags=CREATE_NO_WINDOW
            )

            if result.returncode == 0:
                if on_complete:
                    on_complete(True, "Reverted to CPU mode. Restart KoKoFish for full effect.")
            else:
                if on_complete:
                    on_complete(False, f"Error: {result.stderr[-300:]}")

        except Exception as exc:
            if on_complete:
                on_complete(False, f"Error: {exc}")

    thread = threading.Thread(target=_worker, daemon=True, name="CPU-Revert")
    thread.start()
    return thread
