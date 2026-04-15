"""
KoKoFish — TTS (Text-to-Speech) Engine.

Wraps Fish-Speech using direct Python imports from the locally bundled repo.
Uses the 3-stage pipeline: encode reference → generate semantic tokens → decode to audio.
All inference runs in background threads.
"""

import gc
import logging
import os
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Callable, List, Optional

import numpy as np
import soundfile as sf

logger = logging.getLogger("KoKoFish.tts")


# ---------------------------------------------------------------------------
# Tokenizer for OpenAudio S1/S1-Mini checkpoints
# ---------------------------------------------------------------------------

class _S1MiniTokenizer:
    """
    Minimal FishTokenizer-compatible wrapper for OpenAudio S1/S1-Mini.

    S1Mini uses a tiktoken BPE vocabulary (tokenizer.tiktoken) plus a
    flat mapping of 4096 semantic tokens in special_tokens.json.
    The fish-speech-latest FishTokenizer wraps AutoTokenizer which fails
    because the checkpoint has no tokenizer_config.json.  This class
    bypasses that by loading tiktoken directly.
    """

    _TIKTOKEN_PATTERN = "|".join([
        r"(?i:'s|'t|'re|'ve|'m|'ll|'d)",
        r"\p{P}",
        r"[^\r\n\p{L}\p{N}]?\p{L}+",
        r"\p{N}",
        r" ?[^\s\p{L}\p{N}]+[\r\n]*",
        r"\s*[\r\n]+",
        r"\s+(?!\S)",
        r"\s+",
    ])

    def __init__(self, checkpoint_dir: str):
        import base64
        import json
        import torch as _torch
        import tiktoken

        # Load special token → id mapping from special_tokens.json
        special_path = os.path.join(checkpoint_dir, "special_tokens.json")
        with open(special_path, encoding="utf-8") as f:
            self._special: dict = json.load(f)

        # Load tiktoken BPE ranks from tokenizer.tiktoken
        tok_path = os.path.join(checkpoint_dir, "tokenizer.tiktoken")
        mergeable_ranks: dict = {}
        with open(tok_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                token_b64, rank = line.split()
                mergeable_ranks[base64.b64decode(token_b64)] = int(rank)

        self._enc = tiktoken.core.Encoding(
            name="s1mini",
            pat_str=self._TIKTOKEN_PATTERN,
            mergeable_ranks=mergeable_ranks,
            special_tokens=self._special,
        )

        # Build semantic token look-up tables
        self.semantic_id_to_token_id: dict = {}
        valid_ids = []
        for code_idx in range(4096):
            token = f"<|semantic:{code_idx}|>"
            if token in self._special:
                tid = self._special[token]
                self.semantic_id_to_token_id[code_idx] = tid
                valid_ids.append(tid)

        if not valid_ids:
            logger.error("_S1MiniTokenizer: no semantic tokens found — generation will fail")
            self.semantic_begin_id = 0
            self.semantic_end_id   = 0
            self.semantic_map_tensor = _torch.zeros(4096, dtype=_torch.long)
        else:
            self.semantic_begin_id = min(valid_ids)
            self.semantic_end_id   = max(valid_ids)
            self.semantic_map_tensor = _torch.zeros(4096, dtype=_torch.long)
            for k, v in self.semantic_id_to_token_id.items():
                self.semantic_map_tensor[k] = v

        logger.info(
            "_S1MiniTokenizer: semantic range %d–%d  (%d codes)",
            self.semantic_begin_id, self.semantic_end_id, len(valid_ids),
        )

    # --- FishTokenizer-compatible interface --------------------------------

    def get_token_id(self, token: str) -> int:
        if token in self._special:
            return self._special[token]
        try:
            return self._enc.encode_single_token(token)
        except Exception:
            return 0

    def encode(self, text: str, add_special_tokens: bool = False, **kwargs) -> list:
        return self._enc.encode(text, allowed_special="all")

    def decode(self, tokens, **kwargs) -> str:
        if hasattr(tokens, "tolist"):
            tokens = tokens.tolist()
        return self._enc.decode(tokens)

    # -----------------------------------------------------------------------


class TTSEngine:
    """
    Text-to-Speech engine using Fish-Speech.

    Integrates with the locally bundled fish-speech repo via direct Python
    imports. The 3-stage pipeline:
      1. Encode reference audio → VQ tokens (for voice cloning)
      2. Text → semantic tokens via the LLM
      3. Semantic tokens → audio via the DAC decoder
    """

    def __init__(
        self,
        fish_speech_path: str,
        device: str = "cpu",
        checkpoint_name: str = "checkpoints/fish-speech-1.4",
    ):
        self.fish_speech_path = fish_speech_path
        self.device = device
        self.checkpoint_name = checkpoint_name
        self._model = None
        self._decode_one_token = None
        self._codec = None
        self._lock = threading.Lock()
        self._cancel_event = threading.Event()
        self._worker_thread: Optional[threading.Thread] = None
        self._precision = None

        # Generation quality params — updated live from UI settings
        self.temperature = 0.7
        self.top_p = 0.7
        self.repetition_penalty = 1.2
        self.chunk_length = 150

        # Ensure fish-speech is on sys.path
        if fish_speech_path and fish_speech_path not in sys.path:
            sys.path.insert(0, fish_speech_path)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    @property
    def checkpoint_path(self) -> str:
        return os.path.join(self.fish_speech_path, self.checkpoint_name)

    @property
    def codec_checkpoint_path(self) -> str:
        return os.path.join(self.checkpoint_path, "codec.pth")

    @property
    def _is_openaudio_engine(self) -> bool:
        """True for OpenAudio S1/S1-Mini checkpoints (use fish-speech-latest + DAC codec)."""
        return os.path.isfile(os.path.join(self.checkpoint_path, "tokenizer.tiktoken"))

    @property
    def _openaudio_code_path(self) -> str:
        """Path to fish-speech-latest directory for OpenAudio S1/S1-Mini models."""
        app_dir = os.path.dirname(os.path.abspath(self.fish_speech_path))
        return os.path.join(app_dir, "fish-speech-latest")

    @property
    def _codec_device(self) -> str:
        """Device to use for codec operations.
        OpenAudio DAC codec runs on CPU to keep VRAM free for the LLM.
        Fish14 Firefly codec runs on the same device as the model."""
        return "cpu" if self._is_openaudio_engine else self.device

    @property
    def _codec_sample_rate(self) -> int:
        """Sample rate from loaded codec (works for both Firefly and DAC)."""
        if self._codec is None:
            return 44100
        # DAC exposes sample_rate directly; Firefly via spec_transform
        if hasattr(self._codec, "sample_rate"):
            return self._codec.sample_rate
        return self._codec.spec_transform.sample_rate

    # ------------------------------------------------------------------
    # Precision detection
    # ------------------------------------------------------------------

    def _detect_precision(self):
        """Determine the best precision for the current hardware."""
        import torch

        if self.device == "cpu":
            self._precision = torch.float32
            logger.info("Using float32 precision (CPU mode)")
            return

        try:
            cap = torch.cuda.get_device_capability(0)
            if cap[0] >= 8:  # Ampere+
                self._precision = torch.bfloat16
                logger.info("Using bfloat16 precision (Ampere+ GPU)")
            else:
                self._precision = torch.float16
                logger.info("Using float16 precision (pre-Ampere GPU)")
        except Exception:
            self._precision = torch.float32
            logger.info("Using float32 precision (fallback)")

    # ------------------------------------------------------------------
    # Model lifecycle
    # ------------------------------------------------------------------

    def load_model(
        self,
        on_progress: Callable[[str, float], None] = None,
        on_ready: Callable = None,
        on_error: Callable[[Exception], None] = None,
    ):
        """
        Load Fish-Speech models in a background thread.

        Args:
            on_progress: Callback(status_text, fraction) for splash screen.
            on_ready:    Called when models are ready.
            on_error:    Called on failure.
        """
        def _load():
            try:
                import torch
                import sys

                self._detect_precision()

                # Free any fragmented VRAM before loading large models
                try:
                    import torch as _torch
                    if _torch.cuda.is_available():
                        _torch.cuda.empty_cache()
                except Exception:
                    pass

                if on_progress:
                    on_progress("Loading Fish-Speech LLM...", 0.2)

                # v1.4.3 LLM loading — lives in tools/llama/generate.py
                # We need to add the fish-speech root to path for `tools` to be importable
                fs_root = self.fish_speech_path
                if fs_root not in sys.path:
                    sys.path.insert(0, fs_root)

                if self._is_openaudio_engine:
                    # ── OpenAudio S1/S1-Mini: fish-speech-latest code + DAC codec ──
                    oa_code = self._openaudio_code_path
                    if not os.path.isdir(oa_code):
                        raise RuntimeError(
                            "fish-speech-latest directory not found.\n"
                            "Re-run the KoKoFish installer to download it."
                        )
                    # Ensure fish-speech-latest is at the front of sys.path.
                    # Fish 1.4 code is also on sys.path and its fish_speech package
                    # has no text2semantic/inference.py — Python would pick the wrong
                    # one if 1.4 appears first.  Remove it, then re-insert at 0.
                    if oa_code in sys.path:
                        sys.path.remove(oa_code)
                    sys.path.insert(0, oa_code)

                    # Purge any previously-cached fish_speech modules so Python
                    # re-imports them from fish-speech-latest, not from fish-speech 1.4.
                    for _mod_key in list(sys.modules.keys()):
                        if _mod_key == "fish_speech" or _mod_key.startswith("fish_speech."):
                            del sys.modules[_mod_key]

                    from fish_speech.models.text2semantic.inference import init_model
                    import torch as _torch

                    def _load_on_device(dev):
                        return init_model(
                            checkpoint_path=self.checkpoint_path,
                            device=dev,
                            precision=self._precision,
                            compile=False,
                        )

                    logger.info("Loading S1/S1-Mini LLM from: %s (device=%s)", self.checkpoint_path, self.device)

                    # Check Flash Attention availability — its absence causes a
                    # 5-10x slowdown on long sequences with no visible error.
                    try:
                        import flash_attn  # noqa: F401
                        logger.info("flash_attn found — fast attention enabled.")
                    except ImportError:
                        logger.warning(
                            "flash_attn not installed — S1/S1-Mini will use the standard "
                            "attention kernel, which is significantly slower on long sequences. "
                            "Install flash-attn if you see slow generation times."
                        )
                        if on_progress:
                            on_progress(
                                "⚠  flash_attn not installed — generation may be slower than expected. "
                                "See Settings for install help.", 0.5,
                            )

                    try:
                        model, decode_fn = _load_on_device(self.device)
                    except RuntimeError as _oom:
                        if "out of memory" in str(_oom).lower() and self.device != "cpu":
                            logger.warning("VRAM OOM loading S1Mini — retrying on CPU")
                            _torch.cuda.empty_cache()
                            if on_progress:
                                on_progress(
                                    "⚠  VRAM full — falling back to CPU. "
                                    "Close other apps or lower VRAM usage to run on GPU.", 0.55,
                                )
                            model, decode_fn = _load_on_device("cpu")
                            self.device = "cpu"
                        else:
                            raise

                    # Inject a working tokenizer — the latest code's FishTokenizer
                    # wraps AutoTokenizer which fails on S1Mini (no tokenizer_config.json).
                    # We build a minimal but complete tokenizer from the checkpoint files.
                    model.tokenizer = _S1MiniTokenizer(self.checkpoint_path)
                    # Inject semantic range into model config so embeddings work correctly.
                    model.config.semantic_begin_id = model.tokenizer.semantic_begin_id
                    model.config.semantic_end_id   = model.tokenizer.semantic_end_id
                    logger.info(
                        "S1Mini tokenizer: semantic range %d–%d",
                        model.tokenizer.semantic_begin_id,
                        model.tokenizer.semantic_end_id,
                    )

                    with self._lock:
                        self._model = model
                        self._decode_one_token = decode_fn

                    if on_progress:
                        on_progress("Loading DAC audio codec...", 0.6)

                    # DAC codec (fish-speech-latest modded_dac_vq.yaml)
                    from omegaconf import OmegaConf
                    from hydra.utils import instantiate
                    dac_cfg_path = os.path.join(
                        oa_code, "fish_speech", "configs", "modded_dac_vq.yaml"
                    )
                    logger.info("Loading DAC codec config: %s", dac_cfg_path)
                    OmegaConf.register_new_resolver("eval", eval, replace=True)
                    dac_cfg = OmegaConf.load(dac_cfg_path)
                    codec = instantiate(dac_cfg)

                    try:
                        state_dict = _torch.load(
                            self.codec_checkpoint_path,
                            map_location=self.device,
                            weights_only=False,
                        )
                    except RuntimeError as _oom:
                        if "out of memory" in str(_oom).lower() and self.device != "cpu":
                            logger.warning("VRAM OOM loading DAC codec — retrying on CPU")
                            _torch.cuda.empty_cache()
                            self.device = "cpu"
                            if on_progress:
                                on_progress(
                                    "⚠  VRAM full loading codec — falling back to CPU mode.", 0.65,
                                )
                            state_dict = _torch.load(
                                self.codec_checkpoint_path,
                                map_location="cpu",
                                weights_only=False,
                            )
                        else:
                            raise
                    if "state_dict" in state_dict:
                        state_dict = state_dict["state_dict"]
                    if any("generator" in k for k in state_dict):
                        state_dict = {
                            k.replace("generator.", ""): v
                            for k, v in state_dict.items()
                            if "generator." in k
                        }
                    codec.load_state_dict(state_dict, strict=False)
                    codec.eval()
                    # DAC codec always runs on CPU for OpenAudio engines.
                    # The LLM transformer needs every byte of VRAM; the codec
                    # is ~200 MB and runs once per chunk — CPU is fast enough.
                    codec.to("cpu")
                    with self._lock:
                        self._codec = codec

                else:
                    # ── Fish-Speech 1.4 / Fish-Speech 1.5 ──────────────────────
                    try:
                        from tools.llama.generate import load_model as load_llm_model
                    except ImportError:
                        from fish_speech.models.text2semantic.inference import load_model as load_llm_model

                    logger.info("Loading Fish-Speech LLM from: %s", self.checkpoint_path)
                    model, decode_fn = load_llm_model(
                        checkpoint_path=self.checkpoint_path,
                        device=self.device,
                        precision=self._precision,
                        compile=False,
                    )

                    # Setup KV cache
                    with torch.device(self.device):
                        model.setup_caches(
                            max_batch_size=1,
                            max_seq_len=model.config.max_seq_len,
                            dtype=next(model.parameters()).dtype,
                        )

                    with self._lock:
                        self._model = model
                        self._decode_one_token = decode_fn

                    if on_progress:
                        on_progress("Loading audio codec...", 0.6)

                    # v1.4.3 codec — FireflyArchitecture via firefly_gan_vq.yaml
                    from omegaconf import OmegaConf
                    from hydra.utils import instantiate

                    config_path = os.path.join(
                        self.fish_speech_path,
                        "fish_speech", "configs", "firefly_gan_vq.yaml"
                    )
                    logger.info("Loading Firefly codec config: %s", config_path)
                    OmegaConf.register_new_resolver("eval", eval, replace=True)
                    cfg = OmegaConf.load(config_path)
                    codec = instantiate(cfg)

                    import torch as _torch
                    state_dict = _torch.load(
                        self.codec_checkpoint_path,
                        map_location=self.device,
                        weights_only=False,
                    )
                    if "state_dict" in state_dict:
                        state_dict = state_dict["state_dict"]
                    if any("generator" in k for k in state_dict):
                        state_dict = {
                            k.replace("generator.", ""): v
                            for k, v in state_dict.items()
                            if "generator." in k
                        }
                    codec.load_state_dict(state_dict, strict=False)
                    codec.eval()
                    codec.to(self.device)  # Fish14 Firefly codec stays on GPU
                    with self._lock:
                        self._codec = codec

                # torch.compile() — speeds up repeated inference calls on Fish14.
                # SKIP for OpenAudio (S1/S1-Mini): the compiled graph pre-allocates
                # CUDA workspace that can never be reclaimed, which exhausts VRAM on
                # 12 GB cards and also prevents CPU fallback during OOM.
                if hasattr(torch, "compile") and not self._is_openaudio_engine:
                    try:
                        if on_progress:
                            on_progress("Optimizing model (torch.compile)...", 0.85)
                        with self._lock:
                            self._codec = torch.compile(
                                self._codec,
                                mode="reduce-overhead",
                                fullgraph=False,
                            )
                        logger.info("torch.compile() applied to codec.")
                    except Exception as _ce:
                        logger.warning("torch.compile() skipped: %s", _ce)

                if on_progress:
                    on_progress("TTS engine ready!", 1.0)

                logger.info("Fish-Speech models loaded successfully.")
                if on_ready:
                    on_ready()

            except Exception as exc:
                logger.error("Failed to load Fish-Speech: %s", exc, exc_info=True)
                self.unload_model()
                if on_error:
                    on_error(exc)

        t = threading.Thread(target=_load, daemon=True, name="TTS-Load")
        t.start()
        return t

    def unload_model(self):
        """Unload all models and free GPU memory."""
        with self._lock:
            if self._model is not None:
                del self._model
                self._model = None
            if self._codec is not None:
                del self._codec
                self._codec = None
            self._decode_one_token = None

        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
        logger.info("TTS models unloaded.")

    # ------------------------------------------------------------------
    # Reference audio encoding (for voice cloning)
    # ------------------------------------------------------------------

    def encode_reference(self, wav_path: str) -> np.ndarray:
        """
        Encode a reference WAV file to VQ tokens for zero-shot cloning.

        Args:
            wav_path: Path to reference .wav file. Fish Speech supports up to
                      210s; longer clips give better cloning but use more RAM.

        Returns:
            numpy array of VQ token indices.
        """
        if not self.is_loaded or self._codec is None:
            raise RuntimeError("Models not loaded. Call load_model() first.")

        import torch
        import torchaudio

        logger.info("Encoding reference audio: %s", wav_path)
        with self._lock:
            # v1.4.3 encode: load audio, run through FireflyArchitecture.encode()
            wav, sr = torchaudio.load(wav_path)
            if wav.shape[0] > 1:
                wav = wav.mean(dim=0, keepdim=True)
            target_sr = self._codec_sample_rate
            if sr != target_sr:
                wav = torchaudio.functional.resample(wav, sr, target_sr)
            audios = wav.to(self._codec_device).unsqueeze(0)  # [1, 1, T]
            audio_lengths = torch.tensor([audios.shape[2]], device=self._codec_device, dtype=torch.long)
            indices = self._codec.encode(audios, audio_lengths)[0][0]  # [codebooks, T]

        npy_data = indices.cpu().detach().numpy()
        logger.info("Reference encoded: shape=%s", npy_data.shape)
        return npy_data

    # ------------------------------------------------------------------
    # Text-to-speech generation
    # ------------------------------------------------------------------

    def generate(
        self,
        text: str,
        reference_wav: str = None,
        reference_tokens: np.ndarray = None,
        prompt_text: str = None,
        speed: float = 1.0,
        cadence: float = 0.0,
        output_path: str = None,
        on_progress: Callable[[str, float], None] = None,
        on_chunk: Callable[[np.ndarray, int], None] = None,
        on_complete: Callable[[str], None] = None,
        on_error: Callable[[Exception], None] = None,
    ):
        """
        Generate speech from text in a background thread.

        Args:
            text:             The text to synthesize.
            reference_wav:    Path to reference .wav for voice cloning.
            reference_tokens: Pre-computed VQ tokens (numpy array).
            prompt_text:      Transcript of the reference audio.
            speed:            Playback speed multiplier (0.5–2.0).
            output_path:      Path to save the output .wav. Auto-generated if None.
            on_progress:      Callback(status, fraction).
            on_chunk:         Callback(audio_np_array, sample_rate) fired per decoded chunk.
                              Use this for streaming playback before full generation finishes.
            on_complete:      Callback(output_wav_path) when done.
            on_error:         Callback(Exception) on failure.
        """
        if not self.is_loaded:
            if on_error:
                on_error(RuntimeError("TTS model is not loaded."))
            return

        self._cancel_event.clear()

        def _worker():
            nonlocal output_path
            try:
                import torch
                import torchaudio
                from utils import normalize_text, preprocess_reference_audio

                # Free fragmented VRAM before starting — reserved-but-unallocated
                # memory from the model load can block codec allocations.
                try:
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        gc.collect()
                except Exception:
                    pass

                # Import generate_long from the matching code path
                if self._is_openaudio_engine:
                    oa_code = self._openaudio_code_path
                    if oa_code not in sys.path:
                        sys.path.insert(0, oa_code)
                    from fish_speech.models.text2semantic.inference import generate_long
                else:
                    try:
                        from tools.llama.generate import generate_long
                    except ImportError:
                        from fish_speech.models.text2semantic.inference import generate_long

                if on_progress:
                    on_progress("Preparing generation...", 0.1)

                # Normalize text — preserve Fish Speech emotion/prosody tags
                text_norm = normalize_text(text, engine="fish")

                # Prepare prompt tokens for voice cloning
                prompt_tokens_list = None
                prompt_text_list = None

                # RAM-based cap on reference audio tokens. Fish Speech supports up to
                # 210s of reference audio, but longer clips use more VRAM/RAM.
                # Raise this value if you have headroom; must stay below model
                # max_seq_len (4096) minus space for text context + new tokens.
                MAX_PROMPT_TOKENS = 2048

                if reference_wav and os.path.isfile(reference_wav):
                    logger.info("Using reference audio: %s", reference_wav)
                    # Pre-process: normalize, trim silence, optional denoise
                    _processed_ref = preprocess_reference_audio(reference_wav)
                    _ref_is_temp = (_processed_ref != reference_wav)
                    wav, sr = torchaudio.load(_processed_ref)
                    if wav.shape[0] > 1:
                        wav = wav.mean(dim=0, keepdim=True)
                    with self._lock:
                        target_sr = self._codec_sample_rate
                        if sr != target_sr:
                            wav = torchaudio.functional.resample(wav, sr, target_sr)
                        audios = wav.to(self._codec_device).unsqueeze(0)
                        audio_lengths = torch.tensor([audios.shape[2]], device=self._codec_device, dtype=torch.long)
                        tokens = self._codec.encode(audios, audio_lengths)[0][0]
                    if tokens.shape[1] > MAX_PROMPT_TOKENS:
                        logger.warning(
                            "Reference audio encodes to %d tokens, exceeds RAM cap of %d. "
                            "Truncating to first %d tokens. Raise MAX_PROMPT_TOKENS if you have the VRAM.",
                            tokens.shape[1], MAX_PROMPT_TOKENS, MAX_PROMPT_TOKENS,
                        )
                        tokens = tokens[:, :MAX_PROMPT_TOKENS]
                    prompt_tokens_list = [tokens.to(self.device)]
                    prompt_text_list = [prompt_text or ""]
                    # Delete the preprocessed temp file now that it's encoded
                    if _ref_is_temp:
                        try:
                            os.remove(_processed_ref)
                        except OSError:
                            pass
                elif reference_tokens is not None:
                    ref_tensor = torch.from_numpy(reference_tokens).to(self.device)
                    if ref_tensor.shape[1] > MAX_PROMPT_TOKENS:
                        logger.warning(
                            "Reference tokens too long (%d), truncating to %d.",
                            ref_tensor.shape[1], MAX_PROMPT_TOKENS,
                        )
                        ref_tensor = ref_tensor[:, :MAX_PROMPT_TOKENS]
                    prompt_tokens_list = [ref_tensor]
                    prompt_text_list = [prompt_text or ""]

                if self._cancel_event.is_set():
                    return

                if on_progress:
                    on_progress("Generating speech...", 0.3)

                # Run generation via generate_long (works for both Fish14 and S1Mini
                # — both yield GenerateResponse(action="sample", codes=...) chunks).
                # Accumulate decoded audio chunks rather than raw codes so we never
                # hold the entire chapter's VQ codes in memory or run a single
                # giant decode at the end (avoids OOM on long chapters).
                # Capture model/codec refs under a brief lock, then release.
                # Holding the lock for the entire generate_long loop (minutes)
                # blocks is_loaded() on the main thread and freezes the UI.
                with self._lock:
                    _model_ref          = self._model
                    _codec_ref          = self._codec
                    _decode_fn_ref      = self._decode_one_token
                    _is_openaudio       = self._is_openaudio_engine
                    _codec_dev          = self._codec_device

                # Build kwargs common to both Fish14 and S1Mini generate_long.
                # fish-speech-latest removed max_length and added top_k.
                import inspect as _inspect
                _gl_sig = _inspect.signature(generate_long).parameters
                _gen_kwargs = dict(
                    model=_model_ref,
                    device=self.device,
                    decode_one_token=_decode_fn_ref,
                    text=text_norm,
                    num_samples=1,
                    max_new_tokens=0,
                    top_p=self.top_p,
                    repetition_penalty=self.repetition_penalty,
                    temperature=self.temperature,
                    compile=False,
                    iterative_prompt=True,
                    chunk_length=int(self.chunk_length),
                    prompt_text=prompt_text_list,
                    prompt_tokens=prompt_tokens_list,
                )
                if "max_length" in _gl_sig:
                    _gen_kwargs["max_length"] = _model_ref.config.max_seq_len
                if "top_k" in _gl_sig:
                    _gen_kwargs["top_k"] = 30

                # Run generation WITHOUT holding the lock so is_loaded() and
                # other main-thread calls can proceed normally.
                audio_chunks = []
                generator = generate_long(**_gen_kwargs)

                for response in generator:
                    if self._cancel_event.is_set():
                        logger.info("TTS generation cancelled.")
                        return

                    if response.action == "sample":
                        if on_progress:
                            on_progress("Generating speech...", 0.6)

                        # Decode each chunk immediately — avoids holding the whole
                        # chapter's VQ codes in memory and lets streaming work.
                        try:
                            chunk_codes = response.codes.to(_codec_dev)
                            with torch.no_grad():
                                if _is_openaudio:
                                    # DAC: from_indices expects [B, N, T] — runs on CPU
                                    chunk_audio = _codec_ref.from_indices(
                                        chunk_codes[None]
                                    )[0, 0]
                                else:
                                    # Firefly: decode expects named args — runs on GPU
                                    chunk_lengths = torch.tensor(
                                        [chunk_codes.shape[1]],
                                        device=_codec_dev,
                                    )
                                    chunk_audio = _codec_ref.decode(
                                        indices=chunk_codes[None],
                                        feature_lengths=chunk_lengths,
                                    )[0].squeeze()
                            chunk_np = chunk_audio.cpu().detach().float().numpy()
                            audio_chunks.append(chunk_np)
                            if on_chunk:
                                on_chunk(chunk_np, self._codec_sample_rate)
                        except Exception as ce:
                            logger.warning("Chunk decode error (non-fatal): %s", ce)

                    elif response.action == "next":
                        break

                if not audio_chunks:
                    raise RuntimeError("No audio generated.")

                if on_progress:
                    on_progress("Assembling audio...", 0.8)

                sample_rate = self._codec_sample_rate

                # Apply cadence: insert silence between decoded chunks
                # cadence 0.0 = no pauses, 1.0 = ~600 ms pause between chunks
                if cadence > 0.01 and len(audio_chunks) > 1:
                    pause_samples = int(sample_rate * 0.6 * cadence)
                    silence = np.zeros(pause_samples, dtype=np.float32)
                    spaced = []
                    for i, chunk in enumerate(audio_chunks):
                        spaced.append(chunk)
                        if i < len(audio_chunks) - 1:
                            spaced.append(silence)
                    audio_np = np.concatenate(spaced)
                else:
                    audio_np = np.concatenate(audio_chunks)

                # Apply speed via resampling (pitch shifts slightly, acceptable for TTS)
                if abs(speed - 1.0) > 0.02:
                    import torchaudio as _ta
                    _t = torch.from_numpy(audio_np).unsqueeze(0)
                    _t = _ta.functional.resample(
                        _t,
                        orig_freq=int(sample_rate * speed),
                        new_freq=sample_rate,
                    )
                    audio_np = _t.squeeze(0).numpy()

                # Save output
                if output_path is None:
                    output_path = os.path.join(
                        tempfile.gettempdir(),
                        f"kokofish_tts_{int(time.time())}.wav"
                    )

                sf.write(output_path, audio_np, sample_rate)
                logger.info("TTS output saved: %s", output_path)

                if on_progress:
                    on_progress("Done!", 1.0)
                if on_complete:
                    on_complete(output_path)

                # Cleanup
                del audio_chunks
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            except Exception as exc:
                logger.error("TTS generation error: %s", exc, exc_info=True)
                if on_error:
                    on_error(exc)

        self._worker_thread = threading.Thread(
            target=_worker, daemon=True, name="TTS-Generate"
        )
        self._worker_thread.start()

    def generate_playlist(
        self,
        items: List[dict],
        voice_profile: dict = None,
        speed: float = 1.0,
        on_item_start: Callable[[int, str], None] = None,
        on_item_complete: Callable[[int, str], None] = None,
        on_playlist_complete: Callable = None,
        on_progress: Callable[[str, float], None] = None,
        on_error: Callable[[Exception], None] = None,
    ):
        """
        Generate TTS for a playlist of text items sequentially.

        Args:
            items: List of dicts with keys: 'name', 'text'
            voice_profile: Dict with 'wav_path' and/or 'tokens_path', 'prompt_text'
            speed: Playback speed multiplier.
            on_item_start:    Callback(index, name) when starting an item.
            on_item_complete: Callback(index, wav_path) when item is done.
            on_playlist_complete: Callback when all items are done.
            on_progress: Callback(status, fraction) for progress updates.
            on_error: Callback(Exception) on failure.
        """
        self._cancel_event.clear()

        def _worker():
            try:
                # Pre-encode the reference audio once for the whole playlist.
                # Without this, torchaudio.load + codec.encode would run for
                # every item — very slow for large playlists.
                ref_tokens = None
                prompt_text_val = None

                if voice_profile:
                    tokens_path = voice_profile.get("tokens_path")
                    wav_path = voice_profile.get("wav_path")
                    prompt_text_val = voice_profile.get("prompt_text", "")

                    if tokens_path and os.path.isfile(tokens_path):
                        ref_tokens = np.load(tokens_path)
                        logger.info("Playlist: loaded reference tokens from %s", tokens_path)
                    elif wav_path and os.path.isfile(wav_path):
                        logger.info("Playlist: encoding reference audio once from %s", wav_path)
                        ref_tokens = self.encode_reference(wav_path)

                for idx, item in enumerate(items):
                    if self._cancel_event.is_set():
                        break

                    name = item.get("name", f"Item {idx + 1}")
                    text = item.get("text", "")

                    if on_item_start:
                        on_item_start(idx, name)

                    # Synchronous generation within the thread
                    done_event = threading.Event()
                    result = {"path": None, "error": None}

                    def _on_done(path):
                        result["path"] = path
                        done_event.set()

                    def _on_err(exc):
                        result["error"] = exc
                        done_event.set()

                    self.generate(
                        text=text,
                        reference_tokens=ref_tokens,
                        prompt_text=prompt_text_val,
                        speed=speed,
                        on_progress=on_progress,
                        on_complete=_on_done,
                        on_error=_on_err,
                    )
                    done_event.wait()

                    if result["error"]:
                        if on_error:
                            on_error(result["error"])
                        break

                    if on_item_complete and result["path"]:
                        on_item_complete(idx, result["path"])

                if not self._cancel_event.is_set() and on_playlist_complete:
                    on_playlist_complete()

            except Exception as exc:
                logger.error("Playlist error: %s", exc, exc_info=True)
                if on_error:
                    on_error(exc)

        t = threading.Thread(target=_worker, daemon=True, name="TTS-Playlist")
        t.start()

    def cancel(self):
        """Cancel an ongoing generation."""
        self._cancel_event.set()
        logger.info("TTS cancel requested.")

    def is_busy(self) -> bool:
        return (
            self._worker_thread is not None
            and self._worker_thread.is_alive()
        )
