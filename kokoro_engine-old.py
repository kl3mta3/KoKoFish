"""
KoKoFish — Kokoro Fast CPU Engine.

Wraps kokoro-onnx (v1.0 int8 quantized) to provide real-time TTS on CPU
with 54 preset voices. No voice cloning — use Fish-Speech 1.4/1.5 for that.

Matches the same on_chunk / on_complete / on_error callback interface
as tts_engine.py so the Read Aloud player code is identical.
"""

import logging
import os
import re
import subprocess
import sys
import tempfile
import threading
from typing import Callable, Optional

CREATE_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0x08000000)

logger = logging.getLogger("KoKoFish.kokoro")

# ---------------------------------------------------------------------------
# Model file paths (bundled inside the repo)
# ---------------------------------------------------------------------------

_APP_DIR = os.path.dirname(os.path.abspath(__file__))
_MODEL_DIR = os.path.join(_APP_DIR, "kokoro_models")
KOKORO_ONNX_PATH = os.path.join(_MODEL_DIR, "kokoro-v1.0.int8.onnx")
KOKORO_VOICES_PATH = os.path.join(_MODEL_DIR, "voices-v1.0.bin")


# ---------------------------------------------------------------------------
# Preset voice catalogue
# ---------------------------------------------------------------------------

# fmt: off
KOKORO_VOICES: dict[str, str] = {
    # ── American English Female ───────────────────────────────────────────
    "🇺🇸 Heart (F)":        "af_heart",
    "🇺🇸 Bella (F)":        "af_bella",
    "🇺🇸 Nicole (F)":       "af_nicole",
    "🇺🇸 Sarah (F)":        "af_sarah",
    "🇺🇸 Sky (F)":          "af_sky",
    "🇺🇸 Alloy (F)":        "af_alloy",
    "🇺🇸 Echo (F)":         "af_echo",
    "🇺🇸 Nova (F)":         "af_nova",
    "🇺🇸 River (F)":        "af_river",
    "🇺🇸 Aoede (F)":        "af_aoede",
    "🇺🇸 Jessica (F)":      "af_jessica",
    "🇺🇸 Kore (F)":         "af_kore",
    # ── American English Male ─────────────────────────────────────────────
    "🇺🇸 Adam (M)":         "am_adam",
    "🇺🇸 Michael (M)":      "am_michael",
    "🇺🇸 Eric (M)":         "am_eric",
    "🇺🇸 Fenrir (M)":       "am_fenrir",
    "🇺🇸 Liam (M)":         "am_liam",
    "🇺🇸 Onyx (M)":         "am_onyx",
    "🇺🇸 Puck (M)":         "am_puck",
    "🇺🇸 Santa (M)":        "am_santa",
    # ── British English Female ────────────────────────────────────────────
    "🇬🇧 Emma (F)":         "bf_emma",
    "🇬🇧 Isabella (F)":     "bf_isabella",
    "🇬🇧 Alice (F)":        "bf_alice",
    "🇬🇧 Lily (F)":         "bf_lily",
    # ── British English Male ──────────────────────────────────────────────
    "🇬🇧 George (M)":       "bm_george",
    "🇬🇧 Lewis (M)":        "bm_lewis",
    "🇬🇧 Daniel (M)":       "bm_daniel",
    "🇬🇧 Fable (M)":        "bm_fable",
    # ── Japanese Female ───────────────────────────────────────────────────
    "🇯🇵 Alpha (F)":        "jf_alpha",
    "🇯🇵 Gongitsune (F)":   "jf_gongitsune",
    "🇯🇵 Nezuko (F)":       "jf_nezuko",
    "🇯🇵 Tebukuro (F)":     "jf_tebukuro",
    # ── Japanese Male ─────────────────────────────────────────────────────
    "🇯🇵 Kumo (M)":         "jm_kumo",
    # ── Spanish Female ────────────────────────────────────────────────────
    "🇪🇸 Dora (F)":         "ef_dora",
    # ── Spanish Male ──────────────────────────────────────────────────────
    "🇪🇸 Alex (M)":         "em_alex",
    "🇪🇸 Santa (M)":        "em_santa",
    # ── French Female ─────────────────────────────────────────────────────
    "🇫🇷 Siwis (F)":        "ff_siwis",
    # ── Hindi Female ──────────────────────────────────────────────────────
    "🇮🇳 Alpha (F)":        "hf_alpha",
    "🇮🇳 Beta (F)":         "hf_beta",
    # ── Hindi Male ────────────────────────────────────────────────────────
    "🇮🇳 Omega (M)":        "hm_omega",
    "🇮🇳 Psi (M)":          "hm_psi",
    # ── Italian Female ────────────────────────────────────────────────────
    "🇮🇹 Sara (F)":         "if_sara",
    # ── Italian Male ──────────────────────────────────────────────────────
    "🇮🇹 Nicola (M)":       "im_nicola",
    # ── Brazilian Portuguese Female ───────────────────────────────────────
    "🇧🇷 Dora (F)":         "pf_dora",
    # ── Brazilian Portuguese Male ─────────────────────────────────────────
    "🇧🇷 Alex (M)":         "pm_alex",
    "🇧🇷 Santa (M)":        "pm_santa",
    # ── Mandarin Chinese Female ───────────────────────────────────────────
    "🇨🇳 Xiaobei (F)":      "zf_xiaobei",
    "🇨🇳 Xiaoni (F)":       "zf_xiaoni",
    "🇨🇳 Xiaoxiao (F)":     "zf_xiaoxiao",
    "🇨🇳 Xiaoyi (F)":       "zf_xiaoyi",
    # ── Mandarin Chinese Male ─────────────────────────────────────────────
    "🇨🇳 Yunjian (M)":      "zm_yunjian",
    "🇨🇳 Yunxi (M)":        "zm_yunxi",
    "🇨🇳 Yunxia (M)":       "zm_yunxia",
    "🇨🇳 Yunyang (M)":      "zm_yunyang",
    # ── Korean Female ─────────────────────────────────────────────────────
    "🇰🇷 Bella (F)":        "kf_bella",
    "🇰🇷 Heart (F)":        "kf_heart",
}
# fmt: on

DEFAULT_VOICE = "af_bella"
DEFAULT_VOICE_DISPLAY = "🇺🇸 Bella (F)"
VOICE_ID_FROM_NAME = {v: k for k, v in KOKORO_VOICES.items()}

# Map voice ID prefix (first 2 chars) → language group display label.
# Adding a new language to KOKORO_VOICES auto-creates its radio button.
_LANG_FROM_PREFIX: dict[str, str] = {
    "af": "🇺🇸 American",
    "am": "🇺🇸 American",
    "bf": "🇬🇧 British",
    "bm": "🇬🇧 British",
    "jf": "🇯🇵 Japanese",
    "jm": "🇯🇵 Japanese",
    "ef": "🇪🇸 Spanish",
    "em": "🇪🇸 Spanish",
    "ff": "🇫🇷 French",
    "fm": "🇫🇷 French",
    "hf": "🇮🇳 Hindi",
    "hm": "🇮🇳 Hindi",
    "if": "🇮🇹 Italian",
    "im": "🇮🇹 Italian",
    "pf": "🇧🇷 Portuguese",
    "pm": "🇧🇷 Portuguese",
    "zf": "🇨🇳 Mandarin",
    "zm": "🇨🇳 Mandarin",
    "kf": "🇰🇷 Korean",
    "km": "🇰🇷 Korean",
}

# Auto-derive language groups from voice ID prefixes — preserves insertion order.
from collections import defaultdict as _defaultdict
_grp: dict[str, list[str]] = _defaultdict(list)
for _dn, _vid in KOKORO_VOICES.items():
    _lang = _LANG_FROM_PREFIX.get(_vid[:2])
    if _lang:
        _grp[_lang].append(_dn)
KOKORO_LANGUAGE_GROUPS: dict[str, list[str]] = dict(_grp)
del _grp, _dn, _vid, _lang

KOKORO_DEFAULT_LANG = "🇺🇸 American"

# voice_id → plain language name used in translation prompts.
# English voices return "English" so translate can skip them (no-op).
_PREFIX_TO_LANGNAME: dict[str, str] = {
    "af": "English",           "am": "English",
    "bf": "English",           "bm": "English",
    "jf": "Japanese",          "jm": "Japanese",
    "ef": "Spanish",           "em": "Spanish",
    "ff": "French",            "fm": "French",
    "hf": "Hindi",             "hm": "Hindi",
    "if": "Italian",           "im": "Italian",
    "pf": "Brazilian Portuguese", "pm": "Brazilian Portuguese",
    "zf": "Mandarin Chinese",  "zm": "Mandarin Chinese",
    "kf": "Korean",            "km": "Korean",
}

# voice ID prefix → kokoro-onnx lang code passed to create()
_LANG_CODE_FROM_PREFIX: dict[str, str] = {
    "af": "en-us", "am": "en-us",
    "bf": "en-gb", "bm": "en-gb",
    "jf": "ja",    "jm": "ja",
    "ef": "es",    "em": "es",
    "ff": "fr",    "fm": "fr",
    "hf": "hi",    "hm": "hi",
    "if": "it",    "im": "it",
    "pf": "pt-br", "pm": "pt-br",
    "zf": "zh",    "zm": "zh",
    "kf": "ko",    "km": "ko",
}

# CJK and other phoneme-dense scripts need smaller chunks to stay under the
# 510-phoneme limit. English/Latin scripts are fine at 350 chars.
_MAX_CHARS_FROM_PREFIX: dict[str, int] = {
    "af": 350, "am": 350,
    "bf": 350, "bm": 350,
    "jf": 100, "jm": 100,   # Japanese: ~3-5 phonemes/char
    "ef": 280, "em": 280,
    "ff": 280, "fm": 280,
    "hf": 200, "hm": 200,
    "if": 280, "im": 280,
    "pf": 280, "pm": 280,
    "zf": 80,  "zm": 80,    # Mandarin: most phoneme-dense
    "kf": 150, "km": 150,
}
KOKORO_VOICE_LANG: dict[str, str] = {
    vid: _PREFIX_TO_LANGNAME[vid[:2]]
    for vid in KOKORO_VOICES.values()
    if vid[:2] in _PREFIX_TO_LANGNAME
}


def voice_display(voice_id: str) -> str:
    return VOICE_ID_FROM_NAME.get(voice_id, voice_id)


# ---------------------------------------------------------------------------
# Dependency helpers
# ---------------------------------------------------------------------------

def _is_kokoro_installed() -> bool:
    try:
        import kokoro_onnx  # noqa
        return True
    except ImportError:
        return False


def _model_files_ready() -> bool:
    """Return True if both model files exist on disk."""
    return os.path.isfile(KOKORO_ONNX_PATH) and os.path.isfile(KOKORO_VOICES_PATH)


def _get_pip() -> list:
    """Return the pip command list for the venv (or system pip as fallback)."""
    venv_pip = os.path.join(_APP_DIR, "venv", "Scripts", "pip.exe")
    if os.path.isfile(venv_pip):
        return [venv_pip]
    return [sys.executable, "-m", "pip"]


def _get_packages_dir() -> str:
    return os.path.join(_APP_DIR, "packages")


def switch_onnxruntime(
    use_cuda: bool,
    on_progress: Optional[Callable[[str], None]] = None,
    on_complete: Optional[Callable[[bool, str], None]] = None,
):
    """
    Swap onnxruntime ↔ onnxruntime-gpu in the venv without internet if
    local wheels are present in the packages/ folder.

    use_cuda=True  → uninstall onnxruntime,     install onnxruntime-gpu
    use_cuda=False → uninstall onnxruntime-gpu, install onnxruntime

    Keeps both wheels in packages/ so the swap can happen offline.
    on_complete(ok: bool, message: str) is called when done.
    """
    install_pkg = "onnxruntime-gpu" if use_cuda else "onnxruntime"
    remove_pkg  = "onnxruntime"     if use_cuda else "onnxruntime-gpu"

    def _worker():
        pip       = _get_pip()
        pkg_dir   = _get_packages_dir()

        # ── Step 1: uninstall the outgoing package ────────────────────────
        if on_progress:
            on_progress(f"Removing {remove_pkg}…")
        subprocess.run(
            pip + ["uninstall", remove_pkg, "-y"],
            capture_output=True, creationflags=CREATE_NO_WINDOW,
        )

        # ── Step 2: install the target, local wheel first ─────────────────
        if on_progress:
            on_progress(f"Installing {install_pkg}…")

        # Try local packages/ folder (no internet needed)
        result = subprocess.run(
            pip + ["install", install_pkg,
                   "--find-links", pkg_dir, "--no-index", "--quiet"],
            capture_output=True, creationflags=CREATE_NO_WINDOW,
        )

        if result.returncode != 0:
            # Local wheel missing — download and cache it
            if on_progress:
                on_progress(f"Downloading {install_pkg} (one-time)…")
            dl = subprocess.run(
                pip + ["download", install_pkg, "-d", pkg_dir, "--quiet"],
                capture_output=True, creationflags=CREATE_NO_WINDOW,
            )
            if dl.returncode == 0:
                result = subprocess.run(
                    pip + ["install", install_pkg,
                           "--find-links", pkg_dir, "--no-index", "--quiet"],
                    capture_output=True, creationflags=CREATE_NO_WINDOW,
                )
            else:
                # Final fallback: straight PyPI install
                result = subprocess.run(
                    pip + ["install", install_pkg, "--quiet"],
                    capture_output=True, creationflags=CREATE_NO_WINDOW,
                )

        ok  = result.returncode == 0
        msg = (
            f"Switched to {install_pkg}. Restart KoKoFish to apply."
            if ok else
            f"Could not install {install_pkg}:\n"
            + (result.stderr.decode(errors="replace")[-400:] if result.stderr else "unknown error")
        )
        logger.info("switch_onnxruntime(use_cuda=%s): %s", use_cuda, msg)
        if on_complete:
            on_complete(ok, msg)

    threading.Thread(target=_worker, daemon=True, name="OrtSwitch").start()


def install_kokoro(
    on_progress: Optional[Callable[[str], None]] = None,
    on_complete: Optional[Callable[[bool, str], None]] = None,
):
    """Install kokoro-onnx in a background thread (fallback, rarely needed)."""
    def _worker():
        pip = _get_pip()
        try:
            if on_progress:
                on_progress("Installing Kokoro TTS…")
            result = subprocess.run(
                pip + ["install", "kokoro-onnx", "misaki[en]", "--upgrade", "--quiet"],
                capture_output=True, text=True, timeout=600,
                creationflags=CREATE_NO_WINDOW
            )
            if result.returncode != 0:
                msg = result.stderr[-500:] if result.stderr else "Unknown error"
                if on_complete:
                    on_complete(False, f"Installation failed:\n{msg}")
                return

            if on_complete:
                on_complete(True, "Kokoro installed! Restart KoKoFish to use it.")
        except Exception as exc:
            if on_complete:
                on_complete(False, f"Error: {exc}")

    threading.Thread(target=_worker, daemon=True, name="Kokoro-Install").start()


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class KokoroEngine:
    """
    Real-time TTS using Kokoro 82M (ONNX v1.0 int8).

    Text is split into sentences; on_chunk(audio_np, sample_rate) fires after
    each sentence so playback starts within ~1 second even for long chapters.
    on_complete(wav_path) fires after the full audio is saved to disk.
    """

    SAMPLE_RATE = 24000

    def __init__(self, use_cuda: bool = True):
        self._kokoro = None
        self._lock = threading.Lock()
        self._cancel_event = threading.Event()
        self._worker_thread: Optional[threading.Thread] = None
        self.is_loaded = False
        self._use_cuda = use_cuda
        self.provider = "cpu"   # updated to "cuda" if CUDA provider loads successfully

    # ------------------------------------------------------------------
    # Load
    # ------------------------------------------------------------------

    def load(self):
        """Load the Kokoro ONNX model from bundled kokoro_models/ folder."""
        if self.is_loaded:
            return
        with self._lock:
            if self.is_loaded:
                return
            if not _model_files_ready():
                raise FileNotFoundError(
                    f"Kokoro model files not found.\n"
                    f"Expected:\n  {KOKORO_ONNX_PATH}\n  {KOKORO_VOICES_PATH}"
                )
            logger.info("Loading Kokoro ONNX model from %s", KOKORO_ONNX_PATH)
            from kokoro_onnx import Kokoro

            # Try CUDA if enabled in settings, fall back to CPU.
            # We verify onnxruntime-gpu is actually installed before attempting
            # CUDA — the CPU package also lists CUDAExecutionProvider in
            # get_available_providers() but the DLLs aren't present, causing a
            # silent fall-through that still ends up on CPU.
            _cuda_loaded = False
            if self._use_cuda:
                # Check whether onnxruntime-gpu (not the CPU-only package) is installed
                _ort_is_gpu = False
                try:
                    from importlib.metadata import version as _pkg_ver, PackageNotFoundError as _PNF
                    _pkg_ver("onnxruntime-gpu")
                    _ort_is_gpu = True
                except Exception:
                    pass

                if not _ort_is_gpu:
                    logger.warning(
                        "onnxruntime-gpu is not installed — Kokoro will run on CPU. "
                        "Toggle CUDA off and on in Settings to trigger the swap."
                    )
                else:
                    try:
                        import onnxruntime as _ort
                        _available = _ort.get_available_providers()
                        logger.info("onnxruntime providers: %s", _available)
                        if "CUDAExecutionProvider" in _available:
                            # kokoro-onnx reads ONNX_PROVIDER env var to select provider
                            os.environ["ONNX_PROVIDER"] = "CUDAExecutionProvider"
                            try:
                                self._kokoro = Kokoro(KOKORO_ONNX_PATH, KOKORO_VOICES_PATH)
                                _cuda_loaded = True
                                self.provider = "cuda"
                                logger.info("Kokoro loaded with CUDAExecutionProvider.")
                            finally:
                                os.environ.pop("ONNX_PROVIDER", None)
                        else:
                            logger.warning(
                                "CUDAExecutionProvider not in provider list even though "
                                "onnxruntime-gpu is installed. Falling back to CPU."
                            )
                    except Exception as _e:
                        logger.warning("Kokoro CUDA load failed (%s), falling back to CPU.", _e, exc_info=True)
            else:
                logger.info("Kokoro: use_cuda=False, loading on CPU.")

            if not _cuda_loaded:
                self._kokoro = Kokoro(KOKORO_ONNX_PATH, KOKORO_VOICES_PATH)
                self.provider = "cpu"
                logger.info("Kokoro loaded on CPU.")

            self.is_loaded = True
            logger.info("Kokoro ready. Available voices: %s", len(self._kokoro.get_voices()))

    def load_model(
        self,
        on_progress: Optional[Callable] = None,
        on_ready: Optional[Callable] = None,
        on_error: Optional[Callable] = None,
    ):
        """
        Async wrapper for load() matching the Fish-Speech load_model() interface.
        Fires on_ready() when done, on_error(exc) on failure.
        """
        def _worker():
            try:
                if on_progress:
                    on_progress("Loading Kokoro model…", 0.3)
                self.load()
                if on_progress:
                    on_progress("Ready", 1.0)
                if on_ready:
                    on_ready()
            except Exception as exc:
                logger.error("Kokoro load_model failed: %s", exc, exc_info=True)
                if on_error:
                    on_error(exc)

        threading.Thread(target=_worker, daemon=True, name="Kokoro-Load").start()

    def unload(self):
        with self._lock:
            self._kokoro = None
            self.is_loaded = False
            import gc; gc.collect()
            logger.info("Kokoro unloaded.")

    # ------------------------------------------------------------------
    # Text splitting
    # ------------------------------------------------------------------

    @staticmethod
    def _split_sentences(text: str, max_chars: int = 350) -> list:
        """
        Split text into TTS-ready chunks for sentence-level streaming.

        Handles wrapped text files correctly:
          - Single newlines (soft wraps) → joined as a space
          - Double newlines (paragraph breaks) → sentence boundary
          - Splits only on .!? punctuation, not on every newline
          - Merges short fragments so Kokoro always gets complete thoughts
        """
        import re

        # 1. Normalize line endings
        text = text.strip()
        text = re.sub(r'\r\n|\r', '\n', text)

        # 2. Mark paragraph breaks, then join soft-wrapped lines
        text = re.sub(r'\n{2,}', '\x00', text)   # double newline → sentinel
        text = re.sub(r'\n', ' ', text)           # single newline → space
        text = re.sub(r'\x00', ' . ', text)       # paragraph → sentence end
        text = re.sub(r'\s+', ' ', text).strip()  # collapse spaces

        # 3. Split on sentence-ending punctuation
        raw = re.split(r'(?<=[.!?])\s+', text)

        # 4. Merge short chunks together and split oversized ones at commas
        merged: list = []
        buf = ""
        for chunk in raw:
            chunk = chunk.strip()
            if not chunk:
                continue
            if not buf:
                buf = chunk
            elif len(buf) + 1 + len(chunk) <= max_chars:
                buf = buf + " " + chunk
            else:
                merged.append(buf)
                buf = chunk
        if buf:
            merged.append(buf)

        # 5. Sub-divide any still-oversized chunks at commas/semicolons
        final: list = []
        for sentence in merged:
            if len(sentence) <= max_chars:
                final.append(sentence)
            else:
                parts = re.split(r'(?<=[,;:])\s+', sentence)
                batch = ""
                for part in parts:
                    if batch and len(batch) + 1 + len(part) > max_chars:
                        final.append(batch.strip())
                        batch = part
                    else:
                        batch = (batch + " " + part).strip() if batch else part
                if batch:
                    final.append(batch.strip())

        return [s for s in final if s.strip()]


    # ------------------------------------------------------------------
    # Generate
    # ------------------------------------------------------------------

    def build_blend_voice(self, voice_id: str, blend_voice: str = "", blend_ratio: float = 0.5):
        """
        Return a blended style vector (numpy array) for kokoro_onnx.create(), or
        just the voice_id string when no blend is requested.

        blend_ratio is the weight of voice_id (0.0–1.0).
        kokoro_onnx.create() accepts either a str voice key or an NDArray style vector.
        """
        if not blend_voice or blend_voice == voice_id or self._kokoro is None:
            return voice_id
        try:
            available = self._kokoro.get_voices()
            if blend_voice not in available:
                logger.warning("Blend voice '%s' not available, skipping blend", blend_voice)
                return voice_id
            ratio = max(0.0, min(1.0, blend_ratio))
            style1 = self._kokoro.get_voice_style(voice_id)
            style2 = self._kokoro.get_voice_style(blend_voice)
            return (style1 * ratio + style2 * (1.0 - ratio)).astype(style1.dtype)
        except Exception as exc:
            logger.warning("Voice blend failed (%s), using single voice: %s", exc, voice_id)
            return voice_id

    def generate(
        self,
        text: str,
        voice_id: str = DEFAULT_VOICE,
        blend_voice: str = "",
        blend_ratio: float = 0.5,
        speed: float = 1.0,
        output_path: Optional[str] = None,
        on_chunk: Optional[Callable] = None,
        on_complete: Optional[Callable] = None,
        on_error: Optional[Callable] = None,
        on_progress: Optional[Callable] = None,
        # Accept (and ignore) fish-speech-specific kwargs for uniform call sites
        reference_wav: Optional[str] = None,
        reference_tokens=None,
        prompt_text: Optional[str] = None,
    ):
        """
        Generate speech sentence-by-sentence, streaming on_chunk immediately.

        on_chunk(audio_np, sample_rate) — fired after each sentence (real-time)
        on_complete(wav_path)           — fired when full audio WAV is saved
        on_error(exception)             — fired on failure
        on_progress(message, fraction)  — status updates
        """
        if self._worker_thread and self._worker_thread.is_alive():
            self.cancel()

        self._cancel_event.clear()

        # Clamp speed to Kokoro's allowed range
        speed = max(0.5, min(2.0, float(speed)))

        def _worker():
            nonlocal output_path
            try:
                import numpy as np
                import soundfile as sf
                from utils import normalize_text

                # Normalize text — strip Fish Speech tags Kokoro doesn't understand
                text_norm = normalize_text(text, engine="kokoro")

                if not self.is_loaded:
                    if on_progress:
                        on_progress("Loading Kokoro model…", 0.05)
                    self.load()

                if self._cancel_event.is_set():
                    return

                # Validate voice ID then build blend vector (or plain ID) if requested
                available = self._kokoro.get_voices()
                _vid = voice_id if voice_id in available else (DEFAULT_VOICE if DEFAULT_VOICE in available else available[0])
                if _vid != voice_id:
                    logger.warning("Voice '%s' not found, using %s", voice_id, _vid)

                # blend_voice also needs to be a valid raw ID
                _bv = blend_voice if blend_voice in available else ""
                vid = self.build_blend_voice(_vid, _bv, blend_ratio)
                logger.info("Kokoro voice: %s", vid)

                # Determine lang code and safe chunk size from voice ID prefix
                _prefix    = _vid[:2]
                _lang_code = _LANG_CODE_FROM_PREFIX.get(_prefix, "en-us")
                _max_chars = _MAX_CHARS_FROM_PREFIX.get(_prefix, 350)

                # Split into sentences for real-time streaming
                sentences = self._split_sentences(text_norm, max_chars=_max_chars)
                if not sentences:
                    sentences = [text_norm]

                total = len(sentences)
                all_audio = []
                sr = self.SAMPLE_RATE

                logger.info("Kokoro: %d sentence(s) from %d chars (lang=%s, max_chars=%d)",
                            total, len(text_norm), _lang_code, _max_chars)

                for i, sentence in enumerate(sentences):
                    if self._cancel_event.is_set():
                        break

                    if on_progress:
                        on_progress(
                            f"Sentence {i + 1}/{total}",
                            0.1 + 0.85 * (i / total),
                        )

                    samples, sr = self._kokoro.create(
                        sentence, voice=vid, speed=speed, lang=_lang_code
                    )
                    audio_np = samples.astype(np.float32)
                    all_audio.append(audio_np)

                    # Fire immediately — playback starts on first sentence
                    if on_chunk:
                        on_chunk(audio_np, sr)

                if self._cancel_event.is_set():
                    return

                # Save full concatenated WAV
                if output_path is None:
                    import time as _time
                    output_path = os.path.join(
                        tempfile.gettempdir(),
                        f"kokofish_tts_{int(_time.time())}.wav",
                    )

                full_audio = np.concatenate(all_audio) if all_audio else np.zeros(0, dtype=np.float32)
                sf.write(output_path, full_audio, sr)
                logger.info("Kokoro saved WAV: %s", output_path)

                if on_progress:
                    on_progress("Done", 1.0)

                if on_complete:
                    on_complete(output_path)

            except Exception as exc:
                logger.error("Kokoro generation error: %s", exc, exc_info=True)
                if on_error:
                    on_error(exc)

        self._worker_thread = threading.Thread(target=_worker, daemon=True, name="Kokoro-Gen")
        self._worker_thread.start()

    def cancel(self):
        self._cancel_event.set()
        if self._worker_thread:
            self._worker_thread.join(timeout=3)
