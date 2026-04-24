"""
KoKoFish — OmniVoice Engine.

Wraps k2-fsa/OmniVoice to provide zero-shot voice cloning across 600+
languages. Matches the same on_chunk / on_complete / on_error callback
interface as kokoro_engine.py and tts_engine.py so the Read Aloud player
code is identical.
"""

import logging
import os
import tempfile
import threading
import time
from typing import Callable, Optional

logger = logging.getLogger("KoKoFish.omnivoice")

OMNIVOICE_SAMPLE_RATE = 24000

_APP_DIR = os.path.dirname(os.path.abspath(__file__))
_MODEL_REPO = "k2-fsa/OmniVoice"


class OmniVoiceEngine:
    """
    Zero-shot multilingual voice cloning via k2-fsa/OmniVoice (~1.2B params).

    Reference audio + transcript are required — OmniVoice has no preset
    voice mode. on_chunk(audio_np, sample_rate) fires once when the full
    wav is ready (OmniVoice is non-streaming). on_complete(wav_path) fires
    after the WAV is saved to disk.
    """

    SAMPLE_RATE = OMNIVOICE_SAMPLE_RATE

    def __init__(self, use_cuda: bool = True):
        self._model = None
        self._lock = threading.Lock()
        self._cancel_event = threading.Event()
        self._worker_thread: Optional[threading.Thread] = None
        self._loaded = False
        self._use_cuda = use_cuda
        self._device = "cpu"
        self._dtype = None
        self.provider = "cpu"
        self.load_warnings: list[str] = []

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def is_busy(self) -> bool:
        return bool(self._worker_thread and self._worker_thread.is_alive())

    # ------------------------------------------------------------------
    # Load
    # ------------------------------------------------------------------

    def load_model(
        self,
        on_progress: Optional[Callable[[str, float], None]] = None,
        on_ready: Optional[Callable] = None,
        on_error: Optional[Callable[[Exception], None]] = None,
    ):
        """
        Load OmniVoice in a background thread.

        Args:
            on_progress: Callback(status_text, fraction) for splash screen.
            on_ready:    Called when the model is ready.
            on_error:    Called on failure.
        """
        def _worker():
            try:
                if on_progress:
                    on_progress("Checking device…", 0.05)

                import torch

                cuda_available = torch.cuda.is_available()
                if self._use_cuda and not cuda_available:
                    msg = "CUDA requested but not available — OmniVoice will run on CPU."
                    logger.warning(msg)
                    self.load_warnings.append(msg)
                    self._device = "cpu"
                elif self._use_cuda:
                    self._device = "cuda:0"
                else:
                    self._device = "cpu"

                self._dtype = torch.float16 if self._device.startswith("cuda") else torch.float32
                self.provider = "cuda" if self._device.startswith("cuda") else "cpu"

                if cuda_available:
                    try:
                        torch.set_float32_matmul_precision("high")
                        torch.backends.cuda.matmul.allow_tf32 = True
                        torch.backends.cudnn.allow_tf32 = True
                        torch.backends.cudnn.benchmark = True
                    except Exception as _e:
                        logger.debug("TF32/cuDNN tuning skipped: %s", _e)

                if on_progress:
                    on_progress("Loading OmniVoice model…", 0.3)

                from omnivoice import OmniVoice

                with self._lock:
                    self._model = OmniVoice.from_pretrained(
                        _MODEL_REPO,
                        device_map=self._device,
                        dtype=self._dtype,
                    )
                    try:
                        if os.environ.get("TORCH_COMPILE_DISABLE") != "1" \
                                and cuda_available and hasattr(torch, "compile"):
                            logger.info("OmniVoice: applying torch.compile (first run will be slow).")
                            self._model = torch.compile(
                                self._model, mode="reduce-overhead",
                                dynamic=True, fullgraph=False,
                            )
                    except Exception as _ce:
                        logger.warning("torch.compile wrap failed: %s", _ce)
                    self._loaded = True

                logger.info("OmniVoice loaded on %s (%s).", self._device, self._dtype)

                if on_progress:
                    on_progress("Ready", 1.0)
                if on_ready:
                    on_ready()
            except Exception as exc:
                logger.error("OmniVoice load_model failed: %s", exc, exc_info=True)
                if on_error:
                    on_error(exc)

        thread = threading.Thread(target=_worker, daemon=True, name="OmniVoice-Load")
        thread.start()
        return thread

    def unload_model(self):
        with self._lock:
            self._model = None
            self._loaded = False
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass
            import gc; gc.collect()
            logger.info("OmniVoice unloaded.")

    # ------------------------------------------------------------------
    # Generate
    # ------------------------------------------------------------------

    def generate(
        self,
        text: str,
        reference_wav: Optional[str] = None,
        prompt_text: Optional[str] = None,
        speed: float = 1.0,
        cadence: float = 0.0,
        output_path: Optional[str] = None,
        on_progress: Optional[Callable[[str, float], None]] = None,
        on_chunk: Optional[Callable] = None,
        on_complete: Optional[Callable[[str], None]] = None,
        on_error: Optional[Callable[[Exception], None]] = None,
        **_ignored,
    ):
        """
        Generate cloned speech in a background thread.

        Args:
            text:          The text to synthesize.
            reference_wav: Path to reference .wav (3-10s recommended). Required.
            prompt_text:   Transcript of the reference audio. Required.
            speed:         Playback speed multiplier (0.5–2.0).
            cadence:       Reserved for parity with other engines (unused here).
            output_path:   Path to save the .wav. Auto-generated if None.
            on_progress:   Callback(status, fraction).
            on_chunk:      Callback(audio_np, sample_rate) fired once when ready.
            on_complete:   Callback(output_wav_path) when done.
            on_error:      Callback(Exception) on failure.
        """
        if not self._loaded:
            exc = RuntimeError("OmniVoice model is not loaded.")
            if on_error:
                on_error(exc)
            return None

        if not reference_wav or not prompt_text:
            exc = ValueError(
                "OmniVoice requires reference_wav and prompt_text for voice cloning."
            )
            if on_error:
                on_error(exc)
            return None

        if self._worker_thread and self._worker_thread.is_alive():
            self.cancel()

        self._cancel_event.clear()
        speed = max(0.5, min(2.0, float(speed)))

        def _worker():
            nonlocal output_path
            try:
                import numpy as np
                import soundfile as sf
                import torch

                if on_progress:
                    on_progress("Preparing generation…", 0.1)

                if self._cancel_event.is_set():
                    return

                with self._lock:
                    model_ref = self._model

                if model_ref is None:
                    raise RuntimeError("OmniVoice model is not loaded.")

                if on_progress:
                    on_progress("Generating speech…", 0.3)

                audio = model_ref.generate(
                    text=text,
                    ref_audio=reference_wav,
                    ref_text=prompt_text,
                )

                if self._cancel_event.is_set():
                    return

                audio_np = np.asarray(audio[0], dtype=np.float32)
                sample_rate = OMNIVOICE_SAMPLE_RATE

                if abs(speed - 1.0) > 0.02:
                    import torchaudio as _ta
                    _t = torch.from_numpy(audio_np).unsqueeze(0)
                    _t = _ta.functional.resample(
                        _t,
                        orig_freq=int(sample_rate * speed),
                        new_freq=sample_rate,
                    )
                    audio_np = _t.squeeze(0).numpy()

                if on_progress:
                    on_progress("Saving audio…", 0.85)

                if output_path is None:
                    out_dir = os.path.join(_APP_DIR, "temp", "audio")
                    os.makedirs(out_dir, exist_ok=True)
                    output_path = os.path.join(
                        out_dir, f"omnivoice_{int(time.time())}.wav"
                    )
                else:
                    os.makedirs(os.path.dirname(os.path.abspath(output_path)) or ".", exist_ok=True)

                sf.write(output_path, audio_np, sample_rate)
                logger.info("OmniVoice saved WAV: %s", output_path)

                if on_chunk:
                    on_chunk(audio_np, sample_rate)

                if on_progress:
                    on_progress("Done", 1.0)
                if on_complete:
                    on_complete(output_path)

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            except Exception as exc:
                logger.error("OmniVoice generation error: %s", exc, exc_info=True)
                if on_error:
                    on_error(exc)

        self._worker_thread = threading.Thread(
            target=_worker, daemon=True, name="OmniVoice-Gen"
        )
        self._worker_thread.start()
        return self._worker_thread

    def cancel(self):
        self._cancel_event.set()
        if self._worker_thread:
            self._worker_thread.join(timeout=3)
