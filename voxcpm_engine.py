"""
KoKoFish — VoxCPM Engine.

Wraps the VoxCPM TTS library (0.5B and 2B variants) to provide voice-cloning
TTS via reference wav files. Matches the callback interface of kokoro_engine.py
and tts_engine.py so the Read Aloud player code is identical.
"""

import logging
import os
import queue
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
        # Persistent generation worker. All inference runs on this single
        # thread so torch.compile / CUDA-graph TLS stays valid across calls
        # — spawning a fresh thread per generate() was the reason main.py had
        # to install a cudagraph killswitch.
        self._job_queue: "queue.Queue" = queue.Queue()
        self._gen_thread: Optional[threading.Thread] = None
        self._processing: threading.Event = threading.Event()
        # Kept for legacy is_busy() callers that may still poll it.
        self._worker_thread: Optional[threading.Thread] = None
        self.load_warnings: list[str] = []
        # Set during load_model when the stop-head stride wrapper is
        # installed; lets the UI update the stride live without reload.
        self._stop_head_wrapper = None

    def set_cpu_skips(self, stride: int) -> None:
        """Update the stop-head sync stride on the live model.
        Safe to call before load (no-op if the wrapper isn't installed yet).
        """
        try:
            stride = max(1, min(6, int(stride)))
        except Exception:
            return
        w = self._stop_head_wrapper
        if w is not None:
            w.stride = stride

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

            # torchaudio 2.11+ dispatches `load()` through torchcodec, which
            # requires FFmpeg DLLs on PATH; VoxCPM 0.5B's `build_prompt_cache`
            # calls `torchaudio.load(prompt_wav_path)` and crashes when those
            # DLLs aren't present. Patch torchaudio.load with a soundfile
            # fallback — safe for all .wav/.flac reference audio we use.
            try:
                import torchaudio as _ta
                _orig_ta_load = _ta.load

                def _ta_load_sf_fallback(uri, *a, **kw):
                    try:
                        return _orig_ta_load(uri, *a, **kw)
                    except Exception as _ta_exc:
                        logger.warning(
                            "torchaudio.load failed (%s); using soundfile fallback for %s",
                            type(_ta_exc).__name__, uri,
                        )
                        import soundfile as _sf
                        import torch as _torch
                        import numpy as _np
                        data, sr = _sf.read(str(uri), dtype="float32", always_2d=True)
                        # soundfile → [frames, channels]; torchaudio expects [channels, frames].
                        tensor = _torch.from_numpy(_np.ascontiguousarray(data.T))
                        return tensor, sr

                _ta.load = _ta_load_sf_fallback
            except Exception as _ta_patch_exc:
                logger.debug("torchaudio.load patch skipped: %s", _ta_patch_exc)

            from voxcpm import VoxCPM

            try:
                self._model = VoxCPM.from_pretrained(self.repo_id, load_denoiser=False)
            except TypeError:
                # Older voxcpm versions may not accept load_denoiser kwarg
                self._model = VoxCPM.from_pretrained(self.repo_id)

            # VoxCPM ships its own optimize() that compiles the LM forward_step,
            # the feature encoder, and the CFM estimator with torch.compile.
            # Gated on the user's `torch_compile_enabled` setting — main.py sets
            # TORCHDYNAMO_DISABLE=1 when the setting is OFF (default), so we use
            # that as the signal. First call compiles for several minutes;
            # subsequent launches reuse the on-disk Inductor cache.
            # NOTE: main.py also installs a cudagraph killswitch (see comment in
            # main.py around the _install_cudagraph_killswitch helper) which
            # downgrades reduce-overhead → default mode. So the speedup here is
            # Inductor kernel fusion only, not CUDA graph replay. Lifting that
            # killswitch is a separate workstream (needs the generate worker on
            # a single persistent thread for CUDA-graph-trees TLS).
            try:
                import os as _os
                _compile_on = _os.environ.get("TORCHDYNAMO_DISABLE") != "1"
                if _compile_on:
                    tts_inner = getattr(self._model, "tts_model", None)
                    if tts_inner is not None:
                        # We avoid VoxCPM's built-in optimize() because it does
                        # all four compiles in a single try/except — when one
                        # fails, the rest are skipped. In the field, feat_encoder
                        # raises:
                        #   "cannot assign ... as child module 'feat_encoder'
                        #    (torch.nn.Module or None expected)"
                        # which silently aborts compilation of feat_decoder.estimator
                        # — the actual hot path. So we compile each piece
                        # independently and tolerate per-piece failure.
                        import torch as _torch
                        try:
                            import triton  # noqa: F401
                        except ImportError:
                            logger.warning("VoxCPM: triton not installed — skipping torch.compile.")
                            triton = None  # type: ignore
                        if triton is not None:
                            logger.info(
                                "VoxCPM: per-piece torch.compile starting "
                                "(reduce-overhead, fullgraph). First inference "
                                "compiles for several minutes; subsequent runs "
                                "reuse the on-disk Inductor cache."
                            )

                            def _safe_compile_method(owner, method_name, label,
                                                     fullgraph=True):
                                """Compile a bound method on `owner` and replace
                                the instance attribute. Used for both regular
                                methods (e.g. base_lm.forward_step) and the
                                forward() of nn.Modules. Replacing `forward`
                                via the parent's __setattr__ fails because
                                nn.Module rejects non-Module values for child
                                module slots, but assigning to the *instance*
                                method shadows the class method without
                                touching the parent's _modules dict.

                                Wraps the compiled callable in a one-shot
                                probe that logs the first invocation and
                                transparently falls back to the original
                                bound method if the first compiled call
                                raises. This keeps the engine functional
                                even when an outer-level compile (e.g. the
                                CFM forward with its Python solve_euler
                                loop) hits an op Inductor can't lower.
                                """
                                try:
                                    target = getattr(owner, method_name)
                                except AttributeError:
                                    return
                                try:
                                    compiled = _torch.compile(
                                        target, mode="reduce-overhead",
                                        fullgraph=fullgraph,
                                    )
                                    _probe_state = {"hit": False, "bad": False}

                                    def _probed(*a, _c=compiled, _t=target,
                                                _l=label, _s=_probe_state,
                                                _o=owner, _m=method_name, **kw):
                                        if _s["bad"]:
                                            return _t(*a, **kw)
                                        if not _s["hit"]:
                                            _s["hit"] = True
                                            logger.info(
                                                "VoxCPM: first call into "
                                                "compiled %s — kernel build "
                                                "starts now.", _l,
                                            )
                                            try:
                                                return _c(*a, **kw)
                                            except Exception as _e:
                                                _s["bad"] = True
                                                logger.warning(
                                                    "VoxCPM: compiled %s "
                                                    "raised on first call "
                                                    "(%s) — falling back to "
                                                    "eager for this piece.",
                                                    _l, _e,
                                                )
                                                object.__setattr__(_o, _m, _t)
                                                return _t(*a, **kw)
                                        return _c(*a, **kw)

                                    object.__setattr__(owner, method_name, _probed)
                                    logger.info("VoxCPM: compiled %s.", label)
                                except Exception as _exc:
                                    logger.warning(
                                        "VoxCPM: skip compile of %s — %s",
                                        label, _exc,
                                    )

                            base_lm = getattr(tts_inner, "base_lm", None)
                            if base_lm is not None:
                                _safe_compile_method(base_lm, "forward_step", "base_lm.forward_step")
                            residual_lm = getattr(tts_inner, "residual_lm", None)
                            if residual_lm is not None:
                                _safe_compile_method(residual_lm, "forward_step", "residual_lm.forward_step")
                            # Compile the encoder's forward() (the *method*,
                            # not the module). Wrapping the module itself fails
                            # because nn.Module.__setattr__ rejects the result.
                            feat_encoder = getattr(tts_inner, "feat_encoder", None)
                            if feat_encoder is not None:
                                _safe_compile_method(feat_encoder, "forward", "feat_encoder.forward")
                            # The big win: the CFM diffusion estimator runs
                            # ~10 steps per audio patch × thousands of patches.
                            # Compiling its forward() method captures the
                            # estimator under CUDA graphs.
                            feat_decoder = getattr(tts_inner, "feat_decoder", None)
                            estimator = getattr(feat_decoder, "estimator", None) if feat_decoder is not None else None
                            if estimator is not None:
                                _safe_compile_method(estimator, "forward", "feat_decoder.estimator.forward")
                            # NOTE: an outer-level compile of feat_decoder.forward
                            # (the UnifiedCFM wrapper) was tried but Inductor
                            # crashes on its first call with a weakref error
                            # inside the @torch.inference_mode() decorator,
                            # corrupting PythonDispatcherTLS so even the eager
                            # fallback raises. The four pieces above already
                            # cover ~all the speedup compile can give without
                            # touching the venv source.
                            logger.info(
                                "VoxCPM: per-piece compile dispatch complete "
                                "(actual kernel build happens on first inference)."
                            )

                            # ── Strided stop-head sync ────────────────────
                            # The patch loop in voxcpm2.py:_inference does:
                            #   stop_flag = self.stop_head(self.stop_actn(
                            #       self.stop_proj(lm_hidden)
                            #   )).argmax(dim=-1)[0].cpu().item()
                            # That .cpu().item() is a blocking GPU→CPU sync
                            # *every patch* — it stalls kernel-launch
                            # pipelining and caps per-patch throughput
                            # regardless of how well the kernels compile.
                            #
                            # Wrap stop_head so on N-1 of every N calls it
                            # returns a tiny pre-built CPU tensor whose
                            # argmax → 0 and whose .cpu().item() needs no
                            # sync. Worst case: model overshoots its
                            # natural stop by (stride − 1) patches of
                            # (mostly silent) tail audio.
                            try:
                                stop_head = getattr(tts_inner, "stop_head", None)
                                if stop_head is not None:
                                    # Stride from settings.json's cpu_skips
                                    # (1 = no skip, every patch synced).
                                    # Read straight from disk because the
                                    # engine doesn't take a Settings ref —
                                    # main.py instantiates with just variant
                                    # / use_cuda. Clamped to [1, 6] so a
                                    # corrupt value can't run the model away
                                    # from its stop signal.
                                    _stride = 2
                                    try:
                                        import json as _json2, os as _os2, sys as _sys2
                                        _root2 = (_os2.path.dirname(_sys2.executable)
                                                  if getattr(_sys2, "frozen", False)
                                                  else _os2.path.dirname(
                                                      _os2.path.abspath(__file__)))
                                        _sp = _os2.path.join(_root2, "settings.json")
                                        if _os2.path.isfile(_sp):
                                            with open(_sp, "r", encoding="utf-8") as _f2:
                                                _stride = int((_json2.load(_f2)
                                                               or {}).get("cpu_skips", 2))
                                    except Exception:
                                        _stride = 2
                                    _stride = max(1, min(6, _stride))
                                    # Shape [1, 2]: argmax(dim=-1)[0] → 0,
                                    # already on CPU so .cpu() is a no-op
                                    # and .item() is a Python int read.
                                    _no_stop = _torch.tensor([[1.0, 0.0]])

                                    class _StridedStopHead:
                                        __slots__ = ("real", "stride", "count")

                                        def __init__(self, real, stride):
                                            self.real = real
                                            self.stride = stride
                                            self.count = 0

                                        def __call__(self, x):
                                            self.count += 1
                                            if (self.count % self.stride) == 0:
                                                return self.real(x)
                                            return _no_stop

                                    _wrapper = _StridedStopHead(stop_head, _stride)
                                    object.__setattr__(
                                        tts_inner, "stop_head", _wrapper,
                                    )
                                    # Stash the wrapper on the engine so
                                    # the UI slider can update its stride
                                    # live (set_cpu_skips) without forcing
                                    # the user to reload the model.
                                    self._stop_head_wrapper = _wrapper
                                    logger.info(
                                        "VoxCPM: stop-head sync strided (1 of "
                                        "every %d patches) — eliminates "
                                        "per-patch GPU sync bottleneck.",
                                        _stride,
                                    )
                            except Exception as _shexc:
                                logger.warning(
                                    "VoxCPM: stop-head stride setup failed "
                                    "(continuing without it): %s", _shexc,
                                )
            except Exception as _ce:
                logger.warning("VoxCPM compile setup failed (continuing eager): %s", _ce)

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
        def _load_job():
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

        # Route load + internal warm-up through the persistent gen worker so
        # torch.compile + cudagraph_trees see the SAME thread for compile,
        # warm-up, and every subsequent generate() call. Splitting load onto
        # its own thread was the cause of the
        # `assert _is_key_in_tls(...)` AssertionError on first inference.
        self._ensure_worker()
        self._job_queue.put(_load_job)
        return self._gen_thread

    def unload_model(self):
        with self._lock:
            # Stop the persistent generation worker before tearing down the
            # model — otherwise an in-flight job would keep the model alive.
            self._cancel_event.set()
            try:
                # Drain any queued jobs.
                while True:
                    self._job_queue.get_nowait()
                    self._job_queue.task_done()
            except queue.Empty:
                pass
            if self._gen_thread and self._gen_thread.is_alive():
                self._job_queue.put(None)  # shutdown sentinel
                self._gen_thread.join(timeout=3)
            self._gen_thread = None
            self._worker_thread = None
            self._processing.clear()

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
    # Persistent worker (single thread for all inference)
    # ------------------------------------------------------------------

    def _ensure_worker(self):
        """Start the persistent generation worker if it isn't already running."""
        if self._gen_thread is None or not self._gen_thread.is_alive():
            self._gen_thread = threading.Thread(
                target=self._worker_loop, daemon=True, name="VoxCPM-Gen"
            )
            self._gen_thread.start()
            # Mirror to the legacy attribute so existing is_busy() pollers keep
            # seeing a live thread.
            self._worker_thread = self._gen_thread

    def _worker_loop(self):
        """Pull jobs (closures) off the queue and run them on this thread."""
        while True:
            job = self._job_queue.get()
            if job is None:
                # Shutdown sentinel from unload_model().
                self._job_queue.task_done()
                return
            self._processing.set()
            try:
                if not self._cancel_event.is_set():
                    job()
            except Exception as exc:
                logger.error("VoxCPM worker job error: %s", exc, exc_info=True)
            finally:
                self._processing.clear()
                self._job_queue.task_done()

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
        instruction: Optional[str] = None,
        cfg_value: float = 2.0,
        inference_timesteps: int = 10,
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
        # If a previous job is in flight or queued, abort it and clear the
        # backlog before submitting this one. The persistent worker thread
        # itself stays alive — only the work it's holding gets dropped.
        if self.is_busy():
            self.cancel()

        self._cancel_event.clear()
        self._ensure_worker()
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

                # VoxCPM2 Control Instruction is passed as a parenthetical
                # prefix on the `text` kwarg; no separate argument exists.
                # Only supported on the 2B variant — silently ignored on 0.5B.
                is_v2 = self.variant == "2B"
                if instruction and instruction.strip():
                    if is_v2:
                        _ins = instruction.strip().strip("()")
                        text_norm = f"({_ins}) {text_norm}"
                        logger.info("VoxCPM2 control instruction applied: %r", _ins[:80])
                    else:
                        logger.warning(
                            "VoxCPM 0.5B does not support Control Instruction — ignoring."
                        )

                if on_progress:
                    on_progress("Generating speech…", 0.3)

                gen_kwargs = dict(
                    text=text_norm,
                    cfg_value=float(cfg_value),
                    inference_timesteps=int(inference_timesteps),
                )

                has_ref = reference_wav and os.path.isfile(reference_wav)
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

                # --- Long-text chunking ---------------------------------
                # A single VoxCPM call materialises the whole audio tensor on
                # the GPU. At ~1500+ chars (≈ a minute of 48 kHz audio) the
                # activations start to blow VRAM. Split at sentence
                # boundaries only when the input is long — short texts keep
                # the original single-shot path so quality is untouched.
                CHUNK_CHAR_BUDGET = 1500

                def _split_sentences(s: str):
                    # Greedy pack sentences (split on .!? followed by space/EOL)
                    # into chunks up to CHUNK_CHAR_BUDGET. Falls back to
                    # whitespace splits if a single "sentence" is over-budget.
                    import re as _re
                    parts = _re.split(r"(?<=[.!?])\s+", s.strip())
                    out, buf = [], ""
                    for p in parts:
                        if not p:
                            continue
                        if len(p) > CHUNK_CHAR_BUDGET:
                            if buf:
                                out.append(buf); buf = ""
                            words = p.split()
                            tmp = ""
                            for w in words:
                                if len(tmp) + 1 + len(w) > CHUNK_CHAR_BUDGET:
                                    out.append(tmp.strip()); tmp = w
                                else:
                                    tmp = f"{tmp} {w}" if tmp else w
                            if tmp:
                                buf = tmp
                            continue
                        if len(buf) + 1 + len(p) > CHUNK_CHAR_BUDGET and buf:
                            out.append(buf); buf = p
                        else:
                            buf = f"{buf} {p}" if buf else p
                    if buf:
                        out.append(buf)
                    return out

                if len(text_norm) <= CHUNK_CHAR_BUDGET:
                    # Single-shot path has no chunk loop to hang an
                    # elapsed/ETA tick off, so spin a tiny heartbeat
                    # thread that updates the status every ~500ms while
                    # the model is busy. ETA is estimated from a coarse
                    # chars/sec rate (~120 cps on a 5060 Ti at 6 steps,
                    # tuned conservatively) so progress never sticks at 0.
                    def _fmt_hms(secs: float) -> str:
                        secs = max(0, int(secs))
                        h, r = divmod(secs, 3600)
                        m, s = divmod(r, 60)
                        if h:
                            return f"{h}h {m:02d}m"
                        if m:
                            return f"{m}m {s:02d}s"
                        return f"{s}s"

                    _hb_stop = threading.Event()
                    _hb_start = time.time()
                    _hb_chars = max(1, len(text_norm))
                    _est_cps = 120.0  # conservative ~bf16 5060 Ti baseline
                    _est_total = _hb_chars / _est_cps

                    def _heartbeat():
                        while not _hb_stop.wait(0.5):
                            if not on_progress:
                                continue
                            try:
                                el = time.time() - _hb_start
                                # Cap visible progress at 0.95 so we don't
                                # hit 100% before the call actually returns.
                                frac = 0.3 + 0.65 * min(0.95, el / _est_total)
                                eta = max(0.0, _est_total - el)
                                on_progress(
                                    f"Generating  ·  elapsed {_fmt_hms(el)}  "
                                    f"·  eta ~{_fmt_hms(eta)}",
                                    frac,
                                )
                            except Exception:
                                pass

                    _hb_thread = threading.Thread(
                        target=_heartbeat, name="VoxCPM-HB", daemon=True,
                    )
                    _hb_thread.start()
                    try:
                        audio_np = _model_ref.generate(**gen_kwargs)
                    finally:
                        _hb_stop.set()
                        _hb_thread.join(timeout=1.0)
                else:
                    # Multi-chunk path: build the prompt cache ONCE and reuse
                    # it across every chunk. The high-level .generate() wrapper
                    # would rebuild it per call (CPU-bound reference encode),
                    # which is the main reason long books were 10 min/chunk.
                    chunks = _split_sentences(text_norm)
                    logger.info(
                        "VoxCPM long-text: %d chars → %d chunks (shared prompt cache)",
                        len(text_norm), len(chunks),
                    )

                    import torch as _torch

                    tts_inner = _model_ref.tts_model

                    # Build prompt cache once (or None for plain TTS).
                    has_prompt_pair = bool(has_ref and prompt_text)
                    if is_v2 and (has_prompt_pair or has_ref):
                        shared_cache = tts_inner.build_prompt_cache(
                            prompt_text=(prompt_text if has_prompt_pair else None),
                            prompt_wav_path=(reference_wav if has_prompt_pair else None),
                            reference_wav_path=(reference_wav if has_ref else None),
                        )
                    elif has_prompt_pair:
                        shared_cache = tts_inner.build_prompt_cache(
                            prompt_text=prompt_text,
                            prompt_wav_path=reference_wav,
                        )
                    else:
                        shared_cache = None

                    def _autocast_ctx():
                        # bf16 autocast ~doubles tensor-core throughput on
                        # Ampere+ (5060 Ti is sm_120, fully supported). Safe
                        # to wrap generation; weights stay fp32 so nothing
                        # breaks downstream.
                        if _torch.cuda.is_available():
                            return _torch.autocast(
                                device_type="cuda",
                                dtype=_torch.bfloat16,
                                enabled=True,
                            )
                        class _NullCtx:
                            def __enter__(self): return None
                            def __exit__(self, *a): return False
                        return _NullCtx()

                    def _run_chunk(chunk_text: str):
                        gen = tts_inner._generate_with_prompt_cache(
                            target_text=chunk_text,
                            prompt_cache=shared_cache,
                            cfg_value=float(cfg_value),
                            inference_timesteps=int(inference_timesteps),
                            streaming=False,
                        )
                        # Non-streaming: one tuple (wav, tokens, feats).
                        for wav, _tok, _feat in gen:
                            return wav.squeeze(0).float().cpu().numpy()
                        return np.zeros(0, dtype=np.float32)

                    def _fmt_hms(secs: float) -> str:
                        secs = max(0, int(secs))
                        h, r = divmod(secs, 3600)
                        m, s = divmod(r, 60)
                        if h:
                            return f"{h}h {m:02d}m"
                        if m:
                            return f"{m}m {s:02d}s"
                        return f"{s}s"

                    pieces = []
                    _job_start = time.time()
                    _chunk_secs: list[float] = []
                    for i, chunk in enumerate(chunks):
                        if self._cancel_event.is_set():
                            return
                        if on_progress:
                            elapsed = time.time() - _job_start
                            # Use the median of recent chunk times as the ETA
                            # estimator — robust against the slow first chunk
                            # (cold compile) skewing the average.
                            if _chunk_secs:
                                recent = _chunk_secs[-min(5, len(_chunk_secs)):]
                                per_chunk = sorted(recent)[len(recent) // 2]
                                eta = per_chunk * (len(chunks) - i)
                                status = (
                                    f"Chunk {i + 1}/{len(chunks)}  "
                                    f"·  elapsed {_fmt_hms(elapsed)}  "
                                    f"·  eta {_fmt_hms(eta)}  "
                                    f"·  {per_chunk:.1f}s/chunk"
                                )
                            else:
                                status = (
                                    f"Chunk {i + 1}/{len(chunks)}  "
                                    f"·  elapsed {_fmt_hms(elapsed)}  "
                                    f"·  measuring…"
                                )
                            on_progress(
                                status,
                                0.3 + 0.65 * (i / max(1, len(chunks))),
                            )
                        _chunk_t0 = time.time()
                        try:
                            with _autocast_ctx():
                                part = _run_chunk(chunk)
                        except Exception as _exc:
                            try:
                                _torch.cuda.empty_cache()
                            except Exception:
                                pass
                            logger.warning(
                                "Chunk %d retry after error: %s", i + 1, _exc,
                            )
                            with _autocast_ctx():
                                part = _run_chunk(chunk)
                        _chunk_secs.append(time.time() - _chunk_t0)
                        pieces.append(np.asarray(part, dtype=np.float32))
                        del part
                        try:
                            _torch.cuda.synchronize()
                            _torch.cuda.empty_cache()
                        except Exception:
                            pass
                        import gc as _gc
                        _gc.collect()
                    audio_np = np.concatenate(pieces, axis=0) if pieces \
                        else np.zeros(0, dtype=np.float32)

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

        # Submit the job closure to the persistent worker. The closure already
        # captures every parameter and callback it needs, so the worker thread
        # can run it without any extra plumbing.
        self._job_queue.put(_worker)

    def cancel(self):
        # Signal the in-flight job to abort at its next checkpoint.
        self._cancel_event.set()
        # Drain any queued-but-not-yet-started jobs so they don't run after
        # the user asked to stop.
        try:
            while True:
                self._job_queue.get_nowait()
                self._job_queue.task_done()
        except queue.Empty:
            pass
        # Wait briefly for the running job to notice the cancel flag and exit.
        deadline = time.time() + 3.0
        while self._processing.is_set() and time.time() < deadline:
            time.sleep(0.05)

    def is_busy(self) -> bool:
        # Busy if a job is running OR jobs are sitting in the queue waiting.
        return self._processing.is_set() or not self._job_queue.empty()
