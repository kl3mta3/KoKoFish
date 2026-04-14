"""
KoKoFish — STT (Speech-to-Text) Engine.

Wraps faster-whisper with background threading and real-time segment callbacks.
The GUI thread is never blocked.
"""

import gc
import logging
import os
import threading
from typing import Callable, Optional

logger = logging.getLogger("KoKoFish.stt")


class STTEngine:
    """
    Speech-to-Text engine using faster-whisper.

    All transcription runs in a background thread with segment-by-segment
    callbacks so the GUI can update in real time.
    """

    def __init__(
        self,
        model_size: str = "base",
        device: str = "cpu",
        compute_type: str = "int8",
    ):
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        self._model = None
        self._lock = threading.Lock()
        self._cancel_event = threading.Event()
        self._worker_thread: Optional[threading.Thread] = None

    # ------------------------------------------------------------------
    # Model lifecycle
    # ------------------------------------------------------------------

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def load_model(
        self,
        model_size: str = None,
        device: str = None,
        compute_type: str = None,
        on_ready: Callable = None,
        on_error: Callable = None,
    ):
        """
        Load the Whisper model in a background thread.

        Args:
            model_size:   tiny / base / small / medium / large
            device:       "cuda" or "cpu"
            compute_type: "float16", "int8", "int8_float16", etc.
            on_ready:     Callback when model is ready.
            on_error:     Callback(Exception) on failure.
        """
        if model_size:
            self.model_size = model_size
        if device:
            self.device = device
        if compute_type:
            self.compute_type = compute_type

        def _load():
            try:
                from faster_whisper import WhisperModel

                logger.info(
                    "Loading Whisper model: size=%s device=%s compute=%s",
                    self.model_size,
                    self.device,
                    self.compute_type,
                )
                with self._lock:
                    self._model = WhisperModel(
                        self.model_size,
                        device=self.device,
                        compute_type=self.compute_type,
                    )
                logger.info("Whisper model loaded successfully.")
                if on_ready:
                    on_ready()
            except Exception as exc:
                logger.error("Failed to load Whisper model: %s", exc)
                if on_error:
                    on_error(exc)

        t = threading.Thread(target=_load, daemon=True, name="STT-Load")
        t.start()
        return t

    def unload_model(self):
        """Unload the model and free memory."""
        with self._lock:
            if self._model is not None:
                del self._model
                self._model = None
                gc.collect()
                try:
                    import torch
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except Exception:
                    pass
                logger.info("Whisper model unloaded.")

    # ------------------------------------------------------------------
    # Transcription
    # ------------------------------------------------------------------

    def transcribe(
        self,
        audio_path: str,
        on_segment: Callable[[str, float, float], None] = None,
        on_progress: Callable[[float], None] = None,
        on_complete: Callable[[str, dict], None] = None,
        on_error: Callable[[Exception], None] = None,
        language: str = None,
    ):
        """
        Transcribe an audio file in a background thread.

        Args:
            audio_path:   Path to .wav or .mp3 file.
            on_segment:   Callback(text, start_time, end_time) per segment.
            on_progress:  Callback(fraction 0.0-1.0) for progress updates.
            on_complete:  Callback(full_text, info_dict) when done.
            on_error:     Callback(Exception) on failure.
            language:     Optional language code (e.g. "en"). None = auto-detect.
        """
        if not self.is_loaded:
            if on_error:
                on_error(RuntimeError("STT model is not loaded."))
            return

        self._cancel_event.clear()

        def _worker():
            try:
                logger.info("Starting transcription: %s", audio_path)
                with self._lock:
                    segments, info = self._model.transcribe(
                        audio_path,
                        vad_filter=True,
                        vad_parameters=dict(
                            min_silence_duration_ms=500,
                        ),
                        language=language,
                        beam_size=5,
                    )

                full_text_parts = []
                duration = info.duration if info.duration else 1.0

                for segment in segments:
                    if self._cancel_event.is_set():
                        logger.info("Transcription cancelled.")
                        break

                    text = segment.text.strip()
                    if text:
                        full_text_parts.append(text)
                        if on_segment:
                            on_segment(text, segment.start, segment.end)
                        if on_progress and duration > 0:
                            progress = min(segment.end / duration, 1.0)
                            on_progress(progress)

                if not self._cancel_event.is_set():
                    full_text = " ".join(full_text_parts)
                    info_dict = {
                        "language": info.language,
                        "language_probability": info.language_probability,
                        "duration": info.duration,
                    }
                    logger.info(
                        "Transcription complete. Language: %s (%.1f%%)",
                        info.language,
                        info.language_probability * 100,
                    )
                    if on_complete:
                        on_complete(full_text, info_dict)

            except Exception as exc:
                logger.error("Transcription error: %s", exc, exc_info=True)
                if on_error:
                    on_error(exc)

        self._worker_thread = threading.Thread(
            target=_worker, daemon=True, name="STT-Transcribe"
        )
        self._worker_thread.start()

    def cancel(self):
        """Cancel an ongoing transcription."""
        self._cancel_event.set()
        logger.info("Transcription cancel requested.")

    def is_busy(self) -> bool:
        """Check if a transcription is currently running."""
        return (
            self._worker_thread is not None
            and self._worker_thread.is_alive()
        )
