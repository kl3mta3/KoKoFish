"""
FishTalk — Application settings and configuration.

Handles persistent settings (JSON), CUDA detection, GPU architecture
probing, and Fish-Speech path validation.
"""

import json
import logging
import os
import sys

logger = logging.getLogger("FishTalk.settings")

# Default settings values
DEFAULTS = {
    "fish_speech_path": "",  # Auto-detected at startup
    "checkpoint_name": "checkpoints/fish-speech-1.4",
    "engine": "kokoro",           # "fish14" | "fish15" | "kokoro"
    "kokoro_voice": "af_bella",  # Active Kokoro preset voice ID
    "use_cuda": False,
    "memory_saver": False,
    "silent_mode": False,
    "whisper_model_size": "base",
    "default_voice": "",
    "speed": 1.0,
    "volume": 80,
    "cadence": 50,
    "window_geometry": "1280x800",
    # TTS generation quality parameters
    "tts_temperature": 0.7,
    "tts_top_p": 0.7,
    "tts_repetition_penalty": 1.2,
    "tts_chunk_length": 150,
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
                return cls(data)
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
# Fish-Speech path validation
# ---------------------------------------------------------------------------

def get_bundled_fish_speech_path() -> str:
    """Return the path to the bundled fish-speech directory."""
    return os.path.join(_get_app_dir(), "fish-speech")


def validate_fish_speech_path(path: str) -> dict:
    """
    Validate that a directory looks like a Fish-Speech repo.

    Returns dict with:
        valid: bool
        message: str
        details: dict of found/missing components
    """
    result = {
        "valid": False,
        "message": "",
        "details": {}
    }

    if not path or not os.path.isdir(path):
        result["message"] = "Directory does not exist."
        return result

    # Check for key directories
    checks = {
        "fish_speech_pkg": os.path.isdir(os.path.join(path, "fish_speech")),
    }

    # Support either v1.4.3 or v1.5 file structure
    has_1_4_logic = os.path.isfile(os.path.join(path, "tools", "llama", "generate.py"))
    has_1_5_logic = os.path.isfile(os.path.join(path, "fish_speech", "models", "text2semantic", "inference.py"))
    
    checks["inference_engine (1.4 or 1.5)"] = has_1_4_logic or has_1_5_logic

    result["details"] = checks
    missing = [k for k, v in checks.items() if not v]

    if not missing:
        result["valid"] = True
        result["message"] = "Fish-Speech installation looks good!"
    else:
        result["message"] = f"Missing components: {', '.join(missing)}"

    return result


def find_checkpoints(fish_speech_path: str) -> list:
    """List available checkpoint directories inside the Fish-Speech repo."""
    ckpt_dir = os.path.join(fish_speech_path, "checkpoints")
    if not os.path.isdir(ckpt_dir):
        return []
    return [
        d for d in os.listdir(ckpt_dir)
        if os.path.isdir(os.path.join(ckpt_dir, d))
    ]
