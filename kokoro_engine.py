"""
FishTalk — Kokoro Fast CPU Engine.

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

logger = logging.getLogger("FishTalk.kokoro")

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
    # American Female
    "🇺🇸 Heart (F)":    "af_heart",
    "🇺🇸 Bella (F)":    "af_bella",
    "🇺🇸 Nicole (F)":   "af_nicole",
    "🇺🇸 Sarah (F)":    "af_sarah",
    "🇺🇸 Sky (F)":      "af_sky",
    "🇺🇸 Alloy (F)":    "af_alloy",
    "🇺🇸 Echo (F)":     "af_echo",
    "🇺🇸 Nova (F)":     "af_nova",
    "🇺🇸 River (F)":    "af_river",
    # American Male
    "🇺🇸 Adam (M)":     "am_adam",
    "🇺🇸 Michael (M)":  "am_michael",
    "🇺🇸 Eric (M)":     "am_eric",
    "🇺🇸 Fenrir (M)":   "am_fenrir",
    "🇺🇸 Liam (M)":     "am_liam",
    "🇺🇸 Onyx (M)":     "am_onyx",
    "🇺🇸 Puck (M)":     "am_puck",
    "🇺🇸 Santa (M)":    "am_santa",
    # British Female
    "🇬🇧 Emma (F)":     "bf_emma",
    "🇬🇧 Isabella (F)": "bf_isabella",
    "🇬🇧 Alice (F)":    "bf_alice",
    "🇬🇧 Lily (F)":     "bf_lily",
    # British Male
    "🇬🇧 George (M)":   "bm_george",
    "🇬🇧 Lewis (M)":    "bm_lewis",
    "🇬🇧 Daniel (M)":   "bm_daniel",
    "🇬🇧 Fable (M)":    "bm_fable",
}
# fmt: on

DEFAULT_VOICE = "af_bella"
DEFAULT_VOICE_DISPLAY = "🇺🇸 Bella (F)"
VOICE_ID_FROM_NAME = {v: k for k, v in KOKORO_VOICES.items()}


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


def install_kokoro(
    on_progress: Optional[Callable[[str], None]] = None,
    on_complete: Optional[Callable[[bool, str], None]] = None,
):
    """Install kokoro-onnx in a background thread (fallback, rarely needed)."""
    venv_pip = os.path.join(_APP_DIR, "venv", "Scripts", "pip.exe")
    if not os.path.isfile(venv_pip):
        venv_pip = f"{sys.executable} -m pip"

    def _worker():
        try:
            if on_progress:
                on_progress("Installing Kokoro TTS…")
            result = subprocess.run(
                [venv_pip, "install", "kokoro-onnx", "misaki[en]", "--upgrade", "--quiet"],
                capture_output=True, text=True, timeout=600,
                creationflags=CREATE_NO_WINDOW
            )
            if result.returncode != 0:
                msg = result.stderr[-500:] if result.stderr else "Unknown error"
                if on_complete:
                    on_complete(False, f"Installation failed:\n{msg}")
                return
            if on_complete:
                on_complete(True, "Kokoro installed! Restart FishTalk to use it.")
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

    def __init__(self):
        self._kokoro = None
        self._lock = threading.Lock()
        self._cancel_event = threading.Event()
        self._worker_thread: Optional[threading.Thread] = None
        self.is_loaded = False

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
            self._kokoro = Kokoro(KOKORO_ONNX_PATH, KOKORO_VOICES_PATH)
            self.is_loaded = True
            logger.info("Kokoro loaded. Available voices: %s", len(self._kokoro.get_voices()))

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

    @staticmethod
    def build_blend_voice(voice_id: str, blend_voice: str = "", blend_ratio: float = 0.5) -> str:
        """
        Construct a Kokoro voice blend string.

        blend_ratio is the weight of voice_id (0.0–1.0).
        e.g. voice_id="af_sky", blend_voice="af_nicole", blend_ratio=0.7
             → "af_sky.7+af_nicole.3"
        """
        if not blend_voice or blend_voice == voice_id:
            return voice_id
        ratio = max(0.1, min(0.9, blend_ratio))
        w1 = round(ratio * 10)
        w2 = 10 - w1
        return f"{voice_id}.{w1}+{blend_voice}.{w2}"

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
                text = normalize_text(text, engine="kokoro")

                if not self.is_loaded:
                    if on_progress:
                        on_progress("Loading Kokoro model…", 0.05)
                    self.load()

                if self._cancel_event.is_set():
                    return

                # Validate voice ID and build blend string if requested
                available = self._kokoro.get_voices()
                if voice_id not in available:
                    logger.warning("Voice '%s' not found, using %s", voice_id, DEFAULT_VOICE)
                    voice_id = DEFAULT_VOICE if DEFAULT_VOICE in available else available[0]

                vid = self.build_blend_voice(voice_id, blend_voice, blend_ratio)
                logger.info("Kokoro voice: %s", vid)

                # Split into sentences for real-time streaming
                sentences = self._split_sentences(text)
                if not sentences:
                    sentences = [text]

                total = len(sentences)
                all_audio = []
                sr = self.SAMPLE_RATE

                logger.info("Kokoro: %d sentence(s) from %d chars", total, len(text))

                for i, sentence in enumerate(sentences):
                    if self._cancel_event.is_set():
                        break

                    if on_progress:
                        on_progress(
                            f"Sentence {i + 1}/{total}",
                            0.1 + 0.85 * (i / total),
                        )

                    samples, sr = self._kokoro.create(
                        sentence, voice=vid, speed=speed, lang="en-us"
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
                        f"fishtalk_tts_{int(_time.time())}.wav",
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
