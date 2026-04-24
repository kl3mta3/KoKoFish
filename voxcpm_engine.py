"""
KoKoFish — VoxCPM Engine.

Wraps the VoxCPM TTS library (0.5B and 2B variants) to provide voice-cloning
TTS via reference wav files. Matches the callback interface of kokoro_engine.py
and tts_engine.py so the Read Aloud player code is identical.
"""

import logging
import os
import tempfile
import threading
import time
from typing import Callable, Optional

logger = logging.getLogger("KoKoFish.voxcpm")

_APP_DIR = os.path.dirname(os.path.abspath(__file__))

_VARIANT_REPOS: dict[str, str] = {
    "0.5B": "openbmb/VoxCPM-0.5B",
    "2B":   "openbmb/VoxCPM2",
}

_VARIANT_SAMPLE_RATES: dict[str, int] = {
    "0.5B": 16000,
    "2B":   48000,
}


class VoxCPMEngine:
    """
    Voice-cloning TTS using VoxCPM (0.5B or 2B variant).

    VoxCPM is non-streaming: generate() produces a single audio buffer that
    is emitted via on_chunk(audio_np, sample_rate) and written to disk.
    on_complete(wav_path) fires after the WAV is saved.
    """

    def __init__(self, variant: str = "0.5B", use_cuda: bool = True):
        if variant not in _VARIANT_REPOS:
            raise ValueError(f"Unsupported VoxCPM variant: {variant!r}. Use '0.5B' or '2B'.")
        self.variant = variant
        self.repo_id = _VARIANT_REPOS[variant]
        self.sample_rate = _VARIANT_SAMPLE_RATES[variant]
        self._use_cuda = use_cuda
        self._model = None
        self._loaded = False
        self._lock = threading.Lock()
        self._cancel_event = threading.Event()
        self._worker_thread: Optional[threading.Thread] = None
        self.load_warnings: list[str] = []

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    # ------------------------------------------------------------------
    # Load
    # ------------------------------------------------------------------

    def load(self):
        """Load the VoxCPM model from HuggingFace (cached after first download)."""
        if self._loaded:
            return
        with self._lock:
            if self._loaded:
                return
            logger.info("Loading VoxCPM variant=%s repo=%s", self.variant, self.repo_id)
            # Free GPU optimizations (safe, no first-run compile cost):
            # - TF32 matmul: ~20-30% faster FP32 matmul on Ampere+ at negligible accuracy loss.
            # - cuDNN benchmark: auto-pick fastest conv algorithm per input shape.
            try:
                import torch
                if torch.cuda.is_available():
                    torch.set_float32_matmul_precision("high")
                    torch.backends.cuda.matmul.allow_tf32 = True
                    torch.backends.cudnn.allow_tf32 = True
                    torch.backends.cudnn.benchmark = True
            except Exception as _e:
                logger.debug("TF32/cuDNN tuning skipped: %s", _e)
            from voxcpm import VoxCPM

            try:
                self._model = VoxCPM.from_pretrained(self.repo_id, load_denoiser=False)
            except TypeError:
                # Older voxcpm versions may not accept load_denoiser kwarg
                self._model = VoxCPM.from_pretrained(self.repo_id)

            # Opt-in torch.compile: wrap the inner LLM module if the setting is on
            # AND triton is importable. First run will be slow (~10 min) as kernels
            # compile; subsequent runs reuse the on-disk Inductor cache.
            try:
                import os as _os
                if _os.environ.get("TORCH_COMPILE_DISABLE") != "1":
                    import torch
                    inner = getattr(self._model, "llm", None) or getattr(self._model, "model", None)
                    if inner is not None and hasattr(torch, "compile"):
                        logger.info("VoxCPM: applying torch.compile to inner model (first run will be slow).")
                        compiled = torch.compile(
                            inner, mode="reduce-overhead", dynamic=True, fullgraph=False,
                        )
                        # Replace in-place on whichever attribute held the module.
                        if getattr(self._model, "llm", None) is not None:
                            self._model.llm = compiled
                        else:
                            self._model.model = compiled
            except Exception as _ce:
                logger.warning("torch.compile wrap failed (continuing eager): %s", _ce)

            self._loaded = True
            logger.info("VoxCPM ready (sr=%d).", self.sample_rate)

    def load_model(
        self,
        on_progress: Optional[Callable[[str, float], None]] = None,
        on_ready: Optional[Callable] = None,
        on_error: Optional[Callable[[Exception], None]] = None,
    ) -> threading.Thread:
        """
        Load VoxCPM in a background thread.

        Args:
            on_progress: Callback(status_text, fraction) for splash screen.
            on_ready:    Called when model is ready.
            on_error:    Called with the exception on failure.
        """
        def _worker():
            try:
                if on_progress:
                    on_progress(f"Loading VoxCPM {self.variant}…", 0.3)
                self.load()
                if on_progress:
                    on_progress("Ready", 1.0)
                if on_ready:
                    on_ready()
            except Exception as exc:
                logger.error("VoxCPM load_model failed: %s", exc, exc_info=True)
                if on_error:
                    on_error(exc)

        thread = threading.Thread(target=_worker, daemon=True, name="VoxCPM-Load")
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
            logger.info("VoxCPM unloaded.")

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
        Generate speech from text in a background thread.

        Args:
            text:          The text to synthesize.
            reference_wav: Path to reference .wav for voice cloning.
            prompt_text:   Transcript of the reference audio (enables ultimate cloning).
            speed:         Playback speed multiplier (0.5–2.0).
            cadence:       Accepted for API parity; ignored (VoxCPM is single-chunk).
            output_path:   Path to save the .wav. Auto-generated under temp/audio/ if None.
            on_progress:   Callback(status, fraction).
            on_chunk:      Callback(audio_np, sample_rate) fired once with full audio.
            on_complete:   Callback(output_wav_path) when done.
            on_error:      Callback(Exception) on failure.
        """
        if self._worker_thread and self._worker_thread.is_alive():
            self.cancel()

        self._cancel_event.clear()
        speed = max(0.5, min(2.0, float(speed)))

        def _worker():
            nonlocal output_path
            try:
                import numpy as np
                import soundfile as sf
                from utils import normalize_text

                if not self._loaded:
                    if on_progress:
                        on_progress(f"Loading VoxCPM {self.variant}…", 0.05)
                    self.load()

                if self._cancel_event.is_set():
                    return

                text_norm = normalize_text(text, engine="voxcpm")

                if on_progress:
                    on_progress("Generating speech…", 0.3)

                gen_kwargs = dict(
                    text=text_norm,
                    cfg_value=2.0,
                    inference_timesteps=10,
                )

                has_ref = reference_wav and os.path.isfile(reference_wav)
                is_v2 = self.variant == "2B"
                if has_ref and prompt_text:
                    logger.info("VoxCPM cloning (prompt mode): ref=%s", reference_wav)
                    gen_kwargs["prompt_wav_path"] = reference_wav
                    gen_kwargs["prompt_text"] = prompt_text
                    if is_v2:
                        gen_kwargs["reference_wav_path"] = reference_wav
                elif has_ref:
                    if is_v2:
                        logger.info("VoxCPM2 reference-only cloning: ref=%s", reference_wav)
                        gen_kwargs["reference_wav_path"] = reference_wav
                    else:
                        logger.warning(
                            "VoxCPM 0.5B requires a prompt_text transcript for cloning; "
                            "falling back to plain TTS (ref=%s).", reference_wav,
                        )
                else:
                    logger.info("VoxCPM plain TTS (no reference).")

                with self._lock:
                    _model_ref = self._model
                audio_np = _model_ref.generate(**gen_kwargs)

                if self._cancel_event.is_set():
                    return

                audio_np = np.asarray(audio_np, dtype=np.float32)
                sr = self.sample_rate

                if abs(speed - 1.0) > 0.02:
                    import torch
                    import torchaudio as _ta
                    _t = torch.from_numpy(audio_np).unsqueeze(0)
                    _t = _ta.functional.resample(
                        _t,
                        orig_freq=int(sr * speed),
                        new_freq=sr,
                    )
                    audio_np = _t.squeeze(0).numpy()

                if output_path is None:
                    audio_dir = os.path.join(_APP_DIR, "temp", "audio")
                    os.makedirs(audio_dir, exist_ok=True)
                    output_path = os.path.join(
                        audio_dir, f"voxcpm_{int(time.time())}.wav"
                    )
                else:
                    os.makedirs(os.path.dirname(os.path.abspath(output_path)) or ".", exist_ok=True)

                sf.write(output_path, audio_np, sr)
                logger.info("VoxCPM saved WAV: %s", output_path)

                # VoxCPM is non-streaming — we emit one chunk so the UI sees it
                # like a streaming engine.
                if on_chunk:
                    on_chunk(audio_np, sr)

                if on_progress:
                    on_progress("Done", 1.0)
                if on_complete:
                    on_complete(output_path)

            except Exception as exc:
                logger.error("VoxCPM generation error: %s", exc, exc_info=True)
                if on_error:
                    on_error(exc)

        self._worker_thread = threading.Thread(
            target=_worker, daemon=True, name="VoxCPM-Gen"
        )
        self._worker_thread.start()

    def cancel(self):
        self._cancel_event.set()
        if self._worker_thread:
            self._worker_thread.join(timeout=3)

    def is_busy(self) -> bool:
        return bool(self._worker_thread and self._worker_thread.is_alive())
