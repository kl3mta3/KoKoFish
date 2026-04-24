"""
KoKoFish — Application settings and configuration.

Handles persistent settings (JSON), CUDA detection, GPU architecture
probing, and Fish-Speech path validation.
"""

import json
import logging
import os
import sys

from lang import t

logger = logging.getLogger("KoKoFish.settings")

# Engine IDs persisted to settings.json.
#   "kokoro"     — Kokoro ONNX (default, always installed)
#   "voxcpm_05b" — VoxCPM 0.5B (16 kHz, low VRAM)
#   "voxcpm_2b"  — VoxCPM 2B (48 kHz, top quality, 30 langs)
#   "omnivoice"  — k2-fsa/OmniVoice (24 kHz, 600+ langs)
VALID_ENGINES = ("kokoro", "voxcpm_05b", "voxcpm_2b", "omnivoice")

# Default settings values
DEFAULTS = {
    "engine": "kokoro",
    "kokoro_voice": "af_bella",  # Active Kokoro preset voice ID
    "hf_token": "",         # HuggingFace access token (optional, for gated models)
    "use_cuda": False,
    "memory_saver": False,
    "silent_mode": False,
    "whisper_model_size": "base",
    "default_voice": "",
    "speed": 1.0,
    "volume": 80,
    "cadence": 50,
    "window_geometry": "1280x800",
    # CPU thread limit (0 = use all cores)
    "cpu_threads": 0,
    # Active LLM model key (display name from LLM_MODELS in tag_suggester.py)
    "llm_model": "",
}


def _get_app_dir() -> str:
    """Return the directory where the main app lives."""
    if getattr(sys, "frozen", False):
        return os.path.dirname(sys.executable)
    return os.path.dirname(os.path.abspath(__file__))


def _settings_path() -> str:
    return os.path.join(_get_app_dir(), "settings.json")


class Settings:
    """
    Persistent application settings backed by a JSON file.

    Usage:
        settings = Settings.load()
        settings.use_cuda = True
        settings.save()
    """

    def __init__(self, data: dict = None):
        data = data or {}
        for key, default in DEFAULTS.items():
            setattr(self, key, data.get(key, default))

    @classmethod
    def load(cls) -> "Settings":
        """Load settings from disk, falling back to defaults."""
        path = _settings_path()
        if os.path.isfile(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                logger.info("Settings loaded from %s", path)
                instance = cls(data)
                # If any keys from DEFAULTS are missing in the file (e.g. newly
                # added settings), write them back so the file stays complete.
                missing = [k for k in DEFAULTS if k not in data]
                if missing:
                    logger.info("Adding new settings keys to file: %s", missing)
                    instance.save()
                return instance
            except Exception as exc:
                logger.warning("Failed to load settings: %s — using defaults", exc)
        return cls()

    def save(self):
        """Persist current settings to disk."""
        path = _settings_path()
        data = {key: getattr(self, key) for key in DEFAULTS}
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
            logger.info("Settings saved to %s", path)
        except Exception as exc:
            logger.error("Failed to save settings: %s", exc)

    def to_dict(self) -> dict:
        return {key: getattr(self, key) for key in DEFAULTS}


# ---------------------------------------------------------------------------
# CUDA / GPU detection
# ---------------------------------------------------------------------------

def detect_cuda() -> bool:
    """Check if an NVIDIA CUDA GPU is available."""
    try:
        import torch
        available = torch.cuda.is_available()
        if available:
            name = torch.cuda.get_device_name(0)
            logger.info("CUDA GPU detected: %s", name)
        else:
            logger.info("No CUDA GPU detected.")
        return available
    except Exception as exc:
        logger.warning("CUDA detection failed: %s", exc)
        return False


def detect_gpu_arch() -> str:
    """
    Detect GPU compute capability to choose precision.

    Returns:
        "ampere+"  — SM 8.0+ (RTX 30xx, 40xx, A100…) → use bfloat16
        "older"    — SM < 8.0 (RTX 20xx, 10xx…)      → use float16
        "cpu"      — no CUDA GPU                      → use float32
    """
    try:
        import torch
        if not torch.cuda.is_available():
            return "cpu"
        cap = torch.cuda.get_device_capability(0)
        major = cap[0]
        if major >= 8:
            logger.info("GPU arch: Ampere+ (SM %d.%d) — using bfloat16", cap[0], cap[1])
            return "ampere+"
        else:
            logger.info("GPU arch: older (SM %d.%d) — using float16", cap[0], cap[1])
            return "older"
    except Exception:
        return "cpu"


def get_torch_precision():
    """Return the correct torch dtype based on GPU architecture."""
    import torch
    arch = detect_gpu_arch()
    if arch == "ampere+":
        return torch.bfloat16
    elif arch == "older":
        return torch.float16
    else:
        return torch.float32


def get_device(settings: Settings) -> str:
    """Return 'cuda' or 'cpu' based on settings and availability."""
    if settings.use_cuda and detect_cuda():
        return "cuda"
    return "cpu"


# ---------------------------------------------------------------------------
# Engine metadata
# ---------------------------------------------------------------------------

# Display label and category for each engine. Used by the Settings dropdown
# and anywhere an engine name is rendered to the user.
ENGINE_LABELS = {
    "kokoro":     "Kokoro (fast, offline, 54 voices)",
    "voxcpm_05b": "VoxCPM 0.5B (cloning, 16 kHz, low VRAM)",
    "voxcpm_2b":  "VoxCPM 2B (cloning, 48 kHz, top quality)",
    "omnivoice":  "OmniVoice (cloning, 600+ languages)",
}


def engine_label(engine_id: str) -> str:
    """Return the display label for an engine ID, or the ID itself if unknown."""
    return ENGINE_LABELS.get(engine_id, engine_id)


def engine_id_from_label(label: str) -> str:
    """Map a display label back to its engine ID (inverse of engine_label)."""
    for eid, lbl in ENGINE_LABELS.items():
        if lbl == label:
            return eid
    return "kokoro"
