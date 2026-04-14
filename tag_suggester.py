"""
KoKoFish — Tag Suggester

Provides two levels of emotion/prosody tag suggestion for Fish Speech text:
  1. Rule-based (instant, no dependencies beyond vaderSentiment)
  2. LLM-based via Qwen 2.5 0.5B GGUF (requires llama-cpp-python, ~400 MB model)

Fish Speech tag reference:
  Emotion (parenthesis, before sentence):
    (excited) (happy) (sad) (angry) (surprised) (confused)
    (nervous) (confident) (satisfied) (fearful) (gentle) (serious)
  Voice effects (square bracket, inline):
    [laugh]  [whisper]  [breath]  [sigh]
"""

import logging
import os
import re
import subprocess
import sys
import threading
from typing import Optional, Callable

CREATE_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0x08000000)

logger = logging.getLogger("KoKoFish.tags")

APP_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(APP_DIR, "models")

# ---------------------------------------------------------------------------
# LLM model catalogue  (key = settings value, shown in UI)
# ---------------------------------------------------------------------------

LLM_MODELS: dict = {
    "Qwen 2.5 0.5B (default, ~400 MB)": {
        "filename":    "qwen2.5-0.5b-instruct-q4_k_m.gguf",
        "hf_repo":     "Qwen/Qwen2.5-0.5B-Instruct-GGUF",
        "hf_file":     "qwen2.5-0.5b-instruct-q4_k_m.gguf",
        "n_ctx":       2048,
        "chat_format": "chatml",
    },
    "Qwen 3 1.7B (~1 GB)": {
        "filename":    "qwen3-1.7b-q4_k_m.gguf",
        "hf_repo":     "Qwen/Qwen3-1.7B-GGUF",
        "hf_file":     "Qwen3-1.7B-Q4_K_M.gguf",
        "n_ctx":       4096,
        "chat_format": "chatml",
    },
    "Gemma 3 1B (~700 MB)": {
        "filename":    "gemma-3-1b-it-q4_k_m.gguf",
        "hf_repo":     "bartowski/gemma-3-1b-it-GGUF",
        "hf_file":     "gemma-3-1b-it-Q4_K_M.gguf",
        "n_ctx":       4096,
        "chat_format": "gemma",
    },
    "Gemma 3 4B (~2.5 GB)": {
        "filename":    "gemma-3-4b-it-q4_k_m.gguf",
        "hf_repo":     "bartowski/gemma-3-4b-it-GGUF",
        "hf_file":     "gemma-3-4b-it-Q4_K_M.gguf",
        "n_ctx":       4096,
        "chat_format": "gemma",
    },
    "Gemma 4 E2B Q4_0 (~3.4 GB)": {
        "filename":    "gemma-4-e2b-it-q4_0.gguf",
        "hf_repo":     "bartowski/google_gemma-4-E2B-it-GGUF",
        "hf_file":     "google_gemma-4-E2B-it-Q4_0.gguf",
        "n_ctx":       8192,
        "chat_format": "gemma",
    },
    # ── Uncensored / abliterated models — refusal direction removed ──────────
    # Better for creative writing: dark themes, villain POV, mature fiction, etc.
    "Qwen 2.5 0.5B Uncensored (~400 MB)": {
        "filename":    "Qwen2.5-0.5B-Instruct-abliterated.Q4_K_M.gguf",
        "hf_repo":     "mradermacher/Qwen2.5-0.5B-Instruct-abliterated-GGUF",
        "hf_file":     "Qwen2.5-0.5B-Instruct-abliterated.Q4_K_M.gguf",
        "n_ctx":       2048,
        "chat_format": "chatml",
    },
    "Qwen 2.5 1.5B Uncensored (~1 GB)": {
        "filename":    "Qwen2.5-1.5B-Instruct-abliterated.Q4_K_M.gguf",
        "hf_repo":     "mradermacher/Qwen2.5-1.5B-Instruct-abliterated-GGUF",
        "hf_file":     "Qwen2.5-1.5B-Instruct-abliterated.Q4_K_M.gguf",
        "n_ctx":       4096,
        "chat_format": "chatml",
    },
}

_LLM_SETTINGS_FILE = os.path.join(APP_DIR, "llm_model.txt")


def get_active_llm_key() -> str:
    """Return the currently selected LLM model key (display name)."""
    try:
        if os.path.isfile(_LLM_SETTINGS_FILE):
            with open(_LLM_SETTINGS_FILE, encoding="utf-8") as f:
                key = f.read().strip()
            if key in LLM_MODELS:
                return key
    except Exception:
        pass
    return next(iter(LLM_MODELS))  # default = first entry


def set_active_llm_key(key: str):
    """Persist the selected LLM model key and unload any loaded model."""
    global _llm
    if key not in LLM_MODELS:
        return
    with open(_LLM_SETTINGS_FILE, "w", encoding="utf-8") as f:
        f.write(key)
    # Unload any currently-loaded model so the new one is loaded fresh
    with _llm_lock:
        if _llm is not None:
            del _llm
            _llm = None
            import gc; gc.collect()


def get_active_model_cfg() -> dict:
    return LLM_MODELS[get_active_llm_key()]


# For backwards-compat: these now resolve dynamically from active model
@property
def _qwen_model_path_prop():
    return os.path.join(MODELS_DIR, get_active_model_cfg()["filename"])


QWEN_MODEL_FILENAME = "qwen2.5-0.5b-instruct-q4_k_m.gguf"   # kept for compat
QWEN_MODEL_PATH     = os.path.join(MODELS_DIR, QWEN_MODEL_FILENAME)
QWEN_HF_REPO        = "Qwen/Qwen2.5-0.5B-Instruct-GGUF"

# ---------------------------------------------------------------------------
# Tag catalogue (used by both the editor panel and the suggester)
# ---------------------------------------------------------------------------

FISH_TAGS = {
    "Emotion": [
        ("(excited)",    "Excited / High energy"),
        ("(happy)",      "Happy / Upbeat"),
        ("(satisfied)",  "Satisfied / Content"),
        ("(confident)",  "Confident / Assertive"),
        ("(gentle)",     "Gentle / Soft"),
        ("(serious)",    "Serious / Formal"),
        ("(sad)",        "Sad / Melancholy"),
        ("(angry)",      "Angry / Frustrated"),
        ("(nervous)",    "Nervous / Anxious"),
        ("(fearful)",    "Fearful / Scared"),
        ("(surprised)",  "Surprised / Shocked"),
        ("(confused)",   "Confused / Uncertain"),
    ],
    "Voice Effects": [
        ("[laugh]",      "Laughter"),
        ("[whisper]",    "Whispering"),
        ("[breath]",     "Breath sound"),
        ("[sigh]",       "Sighing"),
    ],
}

# ---------------------------------------------------------------------------
# Rule-based suggestion
# ---------------------------------------------------------------------------

# (regex, tag, placement)
# placement: "before" = insert tag at start of sentence containing match
#            "inline" = insert tag immediately before the matched word
_RULES = [
    # Excitement / energy
    (re.compile(r'[!]{2,}'),
     "(excited)", "before"),
    (re.compile(r'\b(amazing|incredible|fantastic|wonderful|awesome|unbelievable|extraordinary)\b', re.I),
     "(excited)", "before"),

    # Happiness
    (re.compile(r'\b(great|perfect|love|joy|delight|pleased|glad|thrilled|overjoyed)\b', re.I),
     "(happy)", "before"),

    # Sadness
    (re.compile(r'\b(sad|unfortunately|sorry|apologize|miss|lost|died|death|crying|tears|grief|mourning)\b', re.I),
     "(sad)", "before"),

    # Anger
    (re.compile(r'\b(angry|furious|rage|hate|frustrated|annoying|terrible|awful|outraged|infuriated)\b', re.I),
     "(angry)", "before"),

    # Fear / nervousness
    (re.compile(r'\b(afraid|terrified|scared|frightened|nervous|anxious|trembling|panic)\b', re.I),
     "(fearful)", "before"),

    # Confidence
    (re.compile(r'\b(certainly|absolutely|definitely|without doubt|I am sure|I know|clearly)\b', re.I),
     "(confident)", "before"),

    # Surprise
    (re.compile(r'\b(what\?|no way|unbelievable|shocking|suddenly|startled|gasp)\b', re.I),
     "(surprised)", "before"),

    # Laughter — inline
    (re.compile(r'\b(haha|hehe|lol|chuckled|giggled|laughed|cackled|snickered)\b', re.I),
     "[laugh]", "inline"),

    # Whisper — inline
    (re.compile(r'\b(whispered|quietly|softly|in a hushed|murmured|breathed)\b', re.I),
     "[whisper]", "inline"),
]


def suggest_tags(text: str) -> str:
    """
    Rule-based tag suggestion.  Scans the text sentence by sentence and
    inserts the most appropriate tag per sentence.  Returns the tagged text.

    Only one emotion tag is added per sentence (the first rule that matches).
    Inline [effects] can be added on top.
    """
    # Split into sentences preserving delimiters
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    result = []

    for sentence in sentences:
        emotion_added = False
        tagged = sentence

        for pattern, tag, placement in _RULES:
            if not pattern.search(sentence):
                continue

            if placement == "before" and not emotion_added:
                # Only one emotion tag per sentence
                if tag.startswith("("):
                    tagged = tag + " " + tagged
                    emotion_added = True
                else:
                    # bracket tag at start too
                    tagged = tag + " " + tagged

            elif placement == "inline":
                # Insert tag before first match
                match = pattern.search(tagged)
                if match:
                    tagged = tagged[:match.start()] + tag + " " + tagged[match.start():]

        result.append(tagged)

    return " ".join(result)


# ---------------------------------------------------------------------------
# Qwen 0.5B GGUF — LLM-based tagging
# ---------------------------------------------------------------------------

_llm = None          # llama_cpp.Llama instance (loaded on demand)
_llm_lock = threading.Lock()
_llm_use_gpu = False  # set True by set_llm_gpu_mode() when CUDA is available
_llm_on_gpu  = False  # tracks whether the currently-loaded LLM is using VRAM


def set_llm_gpu_mode(enabled: bool):
    """Call at startup to allow LLM to offload layers to GPU when available."""
    global _llm_use_gpu
    _llm_use_gpu = enabled


def is_llm_on_gpu() -> bool:
    """Return True if the currently loaded LLM is occupying VRAM."""
    return _llm_on_gpu and _llm is not None


def is_llm_available() -> bool:
    """Return True if llama-cpp-python is installed."""
    try:
        import llama_cpp  # noqa
        return True
    except ImportError:
        return False


def install_llama_cpp(
    on_line: Optional[Callable[[str], None]] = None,
    on_complete: Optional[Callable[[bool, str], None]] = None,
):
    """
    Install llama-cpp-python (CPU pre-built wheel) via pip in a background thread.

    on_line(text)          — called with each line of pip output
    on_complete(ok, msg)   — called when done; ok=True on success
    """
    def _worker():
        # Resolve pip inside the active venv if present
        app_dir = os.path.dirname(os.path.abspath(__file__))
        venv_pip = os.path.join(app_dir, "venv", "Scripts", "pip.exe")
        pip_cmd = [venv_pip] if os.path.isfile(venv_pip) else [sys.executable, "-m", "pip"]

        cmd = pip_cmd + [
            "install", "llama-cpp-python",
            "--prefer-binary",   # use pre-built wheel, never compile from source
            "--upgrade",
            "--quiet", "--progress-bar", "off",
        ]

        try:
            if on_line:
                on_line("Starting llama-cpp-python install…")

            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                creationflags=CREATE_NO_WINDOW,
            )

            for raw_line in proc.stdout:
                line = raw_line.rstrip()
                if line and on_line:
                    on_line(line)

            proc.wait()

            if proc.returncode == 0:
                logger.info("llama-cpp-python installed successfully.")
                if on_complete:
                    on_complete(True, "llama-cpp-python installed.")
            else:
                msg = "pip exited with code " + str(proc.returncode)
                logger.error("llama-cpp-python install failed: %s", msg)
                if on_complete:
                    on_complete(False, msg)

        except Exception as exc:
            logger.error("llama-cpp-python install error: %s", exc)
            if on_complete:
                on_complete(False, str(exc))

    threading.Thread(target=_worker, daemon=True, name="LlamaCppInstall").start()


def is_qwen_model_ready(key: str = None) -> bool:
    """Return True if the model file for *key* (or the active model) exists on disk."""
    if key is not None and key in LLM_MODELS:
        cfg = LLM_MODELS[key]
    else:
        cfg = get_active_model_cfg()
    path = os.path.join(MODELS_DIR, cfg["filename"])
    return os.path.isfile(path)


def download_qwen_model(
    on_progress: Optional[Callable[[str, float], None]] = None,
    on_complete: Optional[Callable[[bool, str], None]] = None,
):
    """Download the currently selected LLM model GGUF from HuggingFace in a background thread."""

    cfg      = get_active_model_cfg()
    filename = cfg["filename"]
    hf_repo  = cfg["hf_repo"]
    hf_file  = cfg["hf_file"]
    dest_path = os.path.join(MODELS_DIR, filename)

    def _worker():
        os.makedirs(MODELS_DIR, exist_ok=True)
        try:
            from huggingface_hub import hf_hub_download
        except ImportError:
            if on_complete:
                on_complete(False, "huggingface_hub not installed.")
            return

        try:
            if on_progress:
                on_progress(f"Downloading {filename}…", 0.05)
            logger.info("Downloading LLM model from %s / %s", hf_repo, hf_file)

            hf_hub_download(
                repo_id=hf_repo,
                filename=hf_file,
                local_dir=MODELS_DIR,
                local_dir_use_symlinks=False,
            )

            # hf_hub_download may nest inside a subfolder — move to flat MODELS_DIR
            nested = os.path.join(MODELS_DIR, hf_file.split("/")[-1])
            if not os.path.isfile(dest_path) and os.path.isfile(nested):
                import shutil
                shutil.move(nested, dest_path)

            if os.path.isfile(dest_path):
                logger.info("LLM model downloaded: %s", dest_path)
                if on_complete:
                    on_complete(True, f"{filename} ready.")
            else:
                if on_complete:
                    on_complete(False, "Download completed but model file not found.")
        except Exception as exc:
            logger.error("LLM download failed: %s", exc)
            if on_complete:
                on_complete(False, str(exc))

    threading.Thread(target=_worker, daemon=True, name="LLMDownload").start()


def _load_llm():
    """Load the active LLM GGUF model (call inside _llm_lock)."""
    global _llm, _llm_on_gpu
    if _llm is not None:
        return
    from llama_cpp import Llama
    cfg        = get_active_model_cfg()
    model_path = os.path.join(MODELS_DIR, cfg["filename"])
    chat_fmt   = cfg.get("chat_format", "chatml")
    # Offload all layers to GPU when CUDA is enabled — unload_llm() is called
    # before S1/S1mini TTS so VRAM is freed when those engines need it.
    n_gpu = -1 if _llm_use_gpu else 0
    logger.info("Loading LLM model: %s (chat_format=%s, gpu_layers=%s)",
                model_path, chat_fmt, "all" if n_gpu else "none")
    try:
        _llm = Llama(
            model_path=model_path,
            n_ctx=cfg.get("n_ctx", 2048),
            n_threads=max(1, os.cpu_count() - 1),
            n_gpu_layers=n_gpu,
            verbose=False,
            chat_format=chat_fmt,
        )
        _llm_on_gpu = (n_gpu != 0)
    except Exception as exc:
        logger.error("Failed to load LLM model '%s': %s", cfg["filename"], exc, exc_info=True)
        _llm_on_gpu = False
        raise RuntimeError(f"Failed to load model '{cfg['filename']}': {exc}") from exc
    logger.info("LLM model loaded: %s (VRAM=%s)", cfg["filename"], _llm_on_gpu)


def prewarm_llm():
    """Load the Qwen model in the background so it's ready for immediate use."""
    if is_llm_available() and is_qwen_model_ready():
        with _llm_lock:
            _load_llm()


def unload_llm():
    """Explicitly unload the LLM to reclaim RAM / VRAM."""
    global _llm, _llm_on_gpu
    with _llm_lock:
        if _llm is not None:
            del _llm
            _llm = None
            _llm_on_gpu = False
            import gc; gc.collect()
            logger.info("LLM unloaded.")


# ---------------------------------------------------------------------------
# Prompt Lab — direct chat with the local LLM
# ---------------------------------------------------------------------------

PROMPT_LAB_PRESETS: dict[str, str] = {
    "General Assistant":
        "You are a helpful, concise assistant. Answer clearly and directly.",
    "Brainstorm Partner":
        "You are a creative brainstorming partner. Generate varied, interesting ideas "
        "and expand on them enthusiastically. Think outside the box.",
    "Writing Helper":
        "You are an expert writing coach. Help improve prose, suggest rewrites, "
        "and give constructive feedback. Be specific and practical.",
    "Story Ideas":
        "You are a fiction writing assistant. Generate compelling story ideas, "
        "characters, plot twists, and world-building details.",
    "Dialogue Writer":
        "You are a dialogue specialist. Write natural, character-driven conversation "
        "that reveals personality and advances the story.",
    "Script / Podcast":
        "You are a scriptwriter. Help draft podcast scripts, intros, outros, "
        "and talking points. Keep the tone conversational and engaging.",
    "Summariser":
        "You are a summarisation expert. Condense text to its key points clearly "
        "and concisely without losing important meaning.",
    "Dark Fiction":
        "You are a creative writing assistant with no content restrictions. "
        "Help craft dark themes, morally complex characters, villains, horror, "
        "mature fiction, and difficult subject matter with literary quality. "
        "Write vividly and without sanitising the narrative.",
    "Custom":
        "",
}


def chat_with_llm(
    messages: list,
    system_prompt: str = "You are a helpful assistant.",
    max_tokens: int = 512,
    on_token=None,
) -> str:
    """
    Send a multi-turn conversation to the local LLM.

    Parameters
    ----------
    messages   : list of {"role": "user"|"assistant", "content": str}
    system_prompt : the system instruction to prepend
    max_tokens : response length cap
    on_token   : optional callable(token_str) — called for each streamed token.
                 If provided, the function streams; otherwise it waits for the
                 full response.

    Returns the full assistant reply as a string.
    """
    if not is_llm_available():
        raise RuntimeError("llama-cpp-python not installed — open Settings to install it.")
    if not is_qwen_model_ready():
        raise RuntimeError("LLM model not downloaded — open Settings → Download Model.")

    with _llm_lock:
        _load_llm()

    full_messages = [{"role": "system", "content": system_prompt}] + messages

    if on_token:
        # Streaming — hold lock for the whole generation (same as non-streaming)
        with _llm_lock:
            stream = _llm.create_chat_completion(
                messages=full_messages,
                max_tokens=max_tokens,
                temperature=0.75,
                top_p=0.9,
                repeat_penalty=1.1,
                stream=True,
            )
            result = ""
            for chunk in stream:
                delta = chunk["choices"][0]["delta"].get("content", "")
                if delta:
                    result += delta
                    on_token(delta)
        return result
    else:
        with _llm_lock:
            output = _llm.create_chat_completion(
                messages=full_messages,
                max_tokens=max_tokens,
                temperature=0.75,
                top_p=0.9,
                repeat_penalty=1.1,
            )
        return output["choices"][0]["message"]["content"].strip()


_GRAMMAR_SYSTEM_PROMPT = """You are a proofreading assistant preparing text for text-to-speech narration.
Your ONLY job is to fix spelling mistakes and grammar errors in the text provided.

Rules:
1. NEVER change the meaning, tone, or style of the writing.
2. NEVER alter proper nouns, character names, place names, invented words, or fictional terminology — even if they look unusual. If it could be a deliberate creative choice, leave it alone.
3. Fix only clear, unambiguous spelling errors and grammatical mistakes.
4. Preserve all punctuation, paragraph breaks, and formatting.
5. Return ONLY the corrected text. No explanations, no commentary, no extra formatting."""

_SYSTEM_PROMPT = """You are a TTS preprocessing assistant for Fish Speech.
Your ONLY job is to add emotion/prosody tags to text — never change, add, or remove words.

Available tags:
  Emotion (place at START of sentence): (excited) (happy) (satisfied) (confident)
    (gentle) (serious) (sad) (angry) (nervous) (fearful) (surprised) (confused)
  Voice effects (place INLINE before the word): [laugh] [whisper] [breath] [sigh]

Rules:
1. Never modify any words — only INSERT tags.
2. Be conservative — only tag sentences with a clear emotional tone.
3. One emotion tag per sentence maximum.
4. Return ONLY the tagged text. No explanations, no extra formatting."""


def generate_tags(
    text: str,
    on_progress: Optional[Callable[[str, float], None]] = None,
) -> str:
    """
    Use Qwen 0.5B to insert Fish Speech tags into text.

    Processes the text in paragraph-sized chunks to stay within context limits.
    Returns the fully tagged text.

    Raises RuntimeError if llama-cpp-python or the model file is missing.
    """
    if not is_llm_available():
        raise RuntimeError(
            "llama-cpp-python is not installed.\n"
            "Install it with: pip install llama-cpp-python"
        )
    if not is_qwen_model_ready():
        raise RuntimeError(
            f"Qwen model not found at {QWEN_MODEL_PATH}.\n"
            "Use the 'Download AI Tagger' button to fetch it."
        )

    with _llm_lock:
        _load_llm()

    # Split into manageable paragraphs (~500 chars each to stay in context)
    paragraphs = [p.strip() for p in re.split(r'\n{2,}', text) if p.strip()]
    if not paragraphs:
        return text

    tagged_parts = []
    for i, para in enumerate(paragraphs):
        if on_progress:
            on_progress(
                f"Tagging paragraph {i + 1}/{len(paragraphs)}…",
                0.1 + 0.85 * (i / len(paragraphs)),
            )

        # Sub-chunk if paragraph is very long
        chunks = _chunk_text(para, max_chars=600)
        para_result = []
        for chunk in chunks:
            with _llm_lock:
                output = _llm.create_chat_completion(
                    messages=[
                        {"role": "system", "content": _PROMPTS.get("tag_gen", _SYSTEM_PROMPT)},
                        {"role": "user", "content": chunk},
                    ],
                    # ~4 chars per token; tags add ~50% overhead; cap at 600
                    max_tokens=min(600, int(len(chunk) / 4 * 2) + 64),
                    temperature=0.2,
                    top_p=0.9,
                )
            tagged_chunk = output["choices"][0]["message"]["content"].strip()
            # Sanity check: result shouldn't be much longer than input
            # (if model added prose, fall back to the original)
            if len(tagged_chunk) > len(chunk) * 2.5:
                logger.warning("LLM output too long — using rule-based fallback for this chunk.")
                tagged_chunk = suggest_tags(chunk)
            para_result.append(tagged_chunk)

        tagged_parts.append(" ".join(para_result))

    if on_progress:
        on_progress("Done", 1.0)

    return "\n\n".join(tagged_parts)


def grammar_check(
    text: str,
    on_progress: Optional[Callable[[str, float], None]] = None,
) -> str:
    """
    Use Qwen 0.5B to fix spelling and grammar errors in text.

    Preserves proper nouns, fictional terms, and creative style choices.
    Returns the corrected text.

    Raises RuntimeError if llama-cpp-python or the model file is missing.
    """
    if not is_llm_available():
        raise RuntimeError(
            "llama-cpp-python is not installed.\n"
            "Install it via the Generate Tags (AI) button."
        )
    if not is_qwen_model_ready():
        raise RuntimeError(
            f"Qwen model not found at {QWEN_MODEL_PATH}.\n"
            "Use the Generate Tags (AI) button to download it."
        )

    with _llm_lock:
        _load_llm()

    paragraphs = [p.strip() for p in re.split(r'\n{2,}', text) if p.strip()]
    if not paragraphs:
        return text

    corrected_parts = []
    for i, para in enumerate(paragraphs):
        if on_progress:
            on_progress(
                f"Checking paragraph {i + 1}/{len(paragraphs)}…",
                0.1 + 0.85 * (i / len(paragraphs)),
            )

        chunks = _chunk_text(para, max_chars=600)
        para_result = []
        for chunk in chunks:
            with _llm_lock:
                output = _llm.create_chat_completion(
                    messages=[
                        {"role": "system", "content": _PROMPTS.get("grammar", _GRAMMAR_SYSTEM_PROMPT)},
                        {"role": "user",   "content": chunk},
                    ],
                    # ~4 chars per token; grammar output ≈ input length; cap at 500
                    max_tokens=min(500, int(len(chunk) / 4 * 1.5) + 64),
                    temperature=0.1,   # lower = more conservative corrections
                    top_p=0.9,
                )
            result = output["choices"][0]["message"]["content"].strip()
            # Sanity check: if model went off the rails, keep original chunk
            if len(result) > len(chunk) * 2.5 or len(result) < len(chunk) * 0.5:
                logger.warning("Grammar LLM output suspicious length — keeping original chunk.")
                result = chunk
            para_result.append(result)

        corrected_parts.append(" ".join(para_result))

    if on_progress:
        on_progress("Done", 1.0)

    return "\n\n".join(corrected_parts)


# ---------------------------------------------------------------------------
# Fish Speech 1.4 — baseline, conservative tag use
# ---------------------------------------------------------------------------
_ENHANCE_SYSTEM_PROMPT_FISH14 = """You are a TTS preparation assistant for Fish Speech 1.4, a neural voice engine.
Your job: improve the rhythm and pacing of text so it sounds natural when spoken aloud, with light emotional direction.

Rules:
- Add commas or em-dashes where a speaker would naturally breathe or pause
- Break overly long sentences into shorter ones at logical clause boundaries
- Spell out numerals and abbreviations in spoken form (e.g. "3" → "three", "Dr." → "Doctor")
- Where emotion is clearly and strongly present, insert ONE tag at the START of the sentence only:
    Emotions: (excited) (happy) (satisfied) (confident) (gentle) (serious) (sad) (angry) (nervous) (fearful) (surprised) (confused)
    Inline effects (place directly before the relevant word): [laugh] [breath] [sigh] [whisper]
- Be conservative — only tag sentences with unmistakable emotional content
- Preserve ALL meaning, names, proper nouns, and fictional/technical terminology exactly as written
- Match the tone, register, and style of the original — do not rewrite, only reformat
- Do NOT add, remove, or paraphrase any content
- Return ONLY the improved text, nothing else"""

# ---------------------------------------------------------------------------
# S1 Mini (0.5B) — compact model, simple clear instructions, conservative tags
# ---------------------------------------------------------------------------
_ENHANCE_SYSTEM_PROMPT_S1MINI = """You are a TTS preparation assistant for OpenAudio S1 Mini, a compact neural voice engine that understands Fish Speech prosody tags.
Your job: improve rhythm and pacing so the text sounds natural when spoken aloud by a human narrator.

Rules:
- Add commas or em-dashes where a speaker would naturally breathe or pause
- Break overly long sentences into shorter ones at logical clause boundaries
- Spell out numerals and abbreviations in spoken form (e.g. "3" → "three", "Dr." → "Doctor")
- Only tag sentences where emotion is unmistakably strong and obvious — insert ONE tag at the START of the sentence:
    Emotions: (excited) (happy) (satisfied) (confident) (gentle) (serious) (sad) (angry) (nervous) (fearful) (surprised) (confused)
    Inline effects (directly before the relevant word): [laugh] [breath] [sigh] [whisper]
- When in doubt, do NOT add a tag — plain text is better than a wrong tag
- Preserve ALL meaning, names, proper nouns, and fictional/technical terminology exactly as written
- Match the tone, register, and style of the original — do not rewrite, only reformat
- Do NOT add, remove, or paraphrase any content
- Return ONLY the improved text, nothing else"""

# ---------------------------------------------------------------------------
# S1 Full (4B) — high-capacity model, richer emotional direction encouraged
# ---------------------------------------------------------------------------
_ENHANCE_SYSTEM_PROMPT_S1 = """You are a TTS preparation assistant for OpenAudio S1, a high-quality neural voice engine with full Fish Speech prosody tag support.
Your job: shape the text for expressive, natural narration — improving both pacing and emotional delivery.

Rules:
- Add commas or em-dashes where a speaker would naturally breathe or pause
- Break overly long sentences into shorter ones at logical clause boundaries
- Spell out numerals and abbreviations in spoken form (e.g. "3" → "three", "Dr." → "Doctor")
- For each sentence where emotion is present, insert ONE tag at the START — choose the most fitting:
    Emotions: (excited) (happy) (satisfied) (confident) (gentle) (serious) (sad) (angry) (nervous) (fearful) (surprised) (confused)
    Inline effects (directly before the relevant word): [laugh] [breath] [sigh] [whisper]
- Use [breath] at natural inhale points in long dramatic or descriptive passages
- Be expressive but accurate — never force a tag that doesn't fit the actual content
- Preserve ALL meaning, names, proper nouns, and fictional/technical terminology exactly as written
- Match the tone, register, and style of the original — do not rewrite, only reformat
- Do NOT add, remove, or paraphrase any content
- Return ONLY the improved text, nothing else"""

# Keep a generic fish alias pointing at the 1.4 prompt for any legacy code paths
_ENHANCE_SYSTEM_PROMPT_FISH = _ENHANCE_SYSTEM_PROMPT_FISH14

# Kokoro does not parse Fish Speech tags — focus on punctuation and pacing only.
_ENHANCE_SYSTEM_PROMPT_KOKORO = """You are a TTS preparation assistant for Kokoro, a neural voice engine that reads plain text only.

Your job: improve the rhythm and pacing of text so it sounds natural when spoken aloud by a human narrator.

Rules:
- Add commas or em-dashes where a speaker would naturally breathe or pause
- Break overly long sentences into shorter ones at logical clause boundaries
- Spell out numerals and abbreviations in spoken form (e.g. "3" → "three", "Dr." → "Doctor")
- Preserve ALL meaning, names, proper nouns, and fictional/technical terminology exactly as written
- Match the tone, register, and style of the original — do not rewrite, only reformat
- Do NOT add ellipses, brackets, tags, markdown, or any special characters
- Do NOT add, remove, or paraphrase any content
- Return ONLY the improved text, nothing else"""

_TONE_SYSTEM_PROMPT_TEMPLATE = """You are a writing assistant. Rewrite the following text in a {tone} tone.

Rules:
1. Preserve ALL core information, events, and meaning.
2. Preserve all proper nouns, character names, place names, and fictional terminology exactly.
3. Keep roughly the same length — do not add new content or remove key details.
4. Return ONLY the rewritten text. No explanations, no preamble."""

# ---------------------------------------------------------------------------
# Mutable prompts dict — editable at runtime, persisted to JSON
# ---------------------------------------------------------------------------

import json as _json

_PROMPTS_FILE = os.path.join(APP_DIR, "prompts.json")

# Keys map to the prompt constants above.  Values can be replaced via
# the prompt editor window without touching this source file.
_PROMPTS: dict = {
    "tag_gen":    _SYSTEM_PROMPT,
    "grammar":    _GRAMMAR_SYSTEM_PROMPT,
    "fish14_af":  _ENHANCE_SYSTEM_PROMPT_FISH14,
    "s1mini_af":  _ENHANCE_SYSTEM_PROMPT_S1MINI,
    "s1_af":      _ENHANCE_SYSTEM_PROMPT_S1,
    "kokoro_af":  _ENHANCE_SYSTEM_PROMPT_KOKORO,
    "tone":       _TONE_SYSTEM_PROMPT_TEMPLATE,
    "translate":  (
        "You are a professional translator. "
        "If the text is already written in {target_language}, return it unchanged. "
        "Otherwise translate it to {target_language}{tone_clause}. "
        "Output ONLY the result — no explanations, no labels. "
        "Keep tags like (laugh) or [whisper] exactly as-is."
    ),
}

# Default snapshots for reset
_PROMPTS_DEFAULTS: dict = dict(_PROMPTS)


def get_prompt(key: str) -> str:
    return _PROMPTS.get(key, "")

def set_prompt(key: str, value: str) -> None:
    _PROMPTS[key] = value

def save_prompts() -> None:
    """Persist current prompts to prompts.json."""
    try:
        with open(_PROMPTS_FILE, "w", encoding="utf-8") as f:
            _json.dump(_PROMPTS, f, indent=2, ensure_ascii=False)
    except Exception as exc:
        logger.error("save_prompts failed: %s", exc)

def load_prompts() -> None:
    """Load custom prompts from prompts.json if it exists."""
    if not os.path.isfile(_PROMPTS_FILE):
        return
    try:
        with open(_PROMPTS_FILE, "r", encoding="utf-8") as f:
            custom = _json.load(f)
        for k, v in custom.items():
            if k in _PROMPTS and isinstance(v, str):
                _PROMPTS[k] = v
    except Exception as exc:
        logger.error("load_prompts failed: %s", exc)

def reset_prompts() -> None:
    """Reset all prompts to factory defaults and delete prompts.json."""
    _PROMPTS.update(_PROMPTS_DEFAULTS)
    try:
        if os.path.isfile(_PROMPTS_FILE):
            os.remove(_PROMPTS_FILE)
    except Exception as exc:
        logger.error("reset_prompts: could not delete %s: %s", _PROMPTS_FILE, exc)

# Load any saved customisations at import time
load_prompts()

TONE_OPTIONS = [
    "Neutral",
    "Casual / Conversational",
    "Formal / Professional",
    "Dramatic / Cinematic",
    "Energetic / Upbeat",
    "Calm / Soothing",
    "Humorous / Playful",
    "Narrative / Storytelling",
    "Tense / Suspenseful",
]


def enhance_for_tts(
    text: str,
    engine: str = "fish14",
    content_style: str = "None",
    on_progress: Optional[Callable[[str, float], None]] = None,
) -> str:
    """
    Use Qwen 0.5B to improve text for natural TTS delivery.

    engine: "kokoro" | "fish14" | "s1mini" | "s1"
    content_style: optional hint appended to the system prompt
                   (e.g. "Podcast", "Story — Fiction", "Formal")
    """
    if not is_llm_available():
        raise RuntimeError("llama-cpp-python is not installed.")
    if not is_qwen_model_ready():
        raise RuntimeError(f"Qwen model not found at {QWEN_MODEL_PATH}.")

    _key_map = {
        "kokoro": "kokoro_af",
        "s1mini": "s1mini_af",
        "s1":     "s1_af",
        "fish14": "fish14_af",
    }
    system_prompt = _PROMPTS.get(_key_map.get(engine, "fish14_af"), _PROMPTS["fish14_af"])
    if content_style and content_style not in ("None", ""):
        system_prompt += f"\nContent style context: {content_style}. Keep formatting choices appropriate to this style."

    with _llm_lock:
        _load_llm()

    paragraphs = [p.strip() for p in re.split(r'\n{2,}', text) if p.strip()]
    if not paragraphs:
        return text

    result_parts = []
    for i, para in enumerate(paragraphs):
        if on_progress:
            on_progress(f"Enhancing paragraph {i + 1}/{len(paragraphs)}…",
                        0.1 + 0.85 * (i / len(paragraphs)))
        chunks = _chunk_text(para, max_chars=600)
        para_result = []
        for chunk in chunks:
            with _llm_lock:
                output = _llm.create_chat_completion(
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user",   "content": chunk},
                    ],
                    # ~4 chars per token; Assisted Flow output ≈ input + TTS tags; cap at 700
                    max_tokens=min(700, int(len(chunk) / 4 * 2) + 100),
                    temperature=0.3,
                    top_p=0.9,
                )
            result = output["choices"][0]["message"]["content"].strip()
            if len(result) > len(chunk) * 2.5 or len(result) < len(chunk) * 0.4:
                logger.warning("enhance_for_tts: suspicious output length — keeping original.")
                result = chunk
            para_result.append(result)
        result_parts.append(" ".join(para_result))

    if on_progress:
        on_progress("Done", 1.0)
    return "\n\n".join(result_parts)


TRANSLATE_LANGUAGES = [
    "Japanese",
    "Mandarin Chinese",
    "Spanish",
    "French",
    "German",
    "Hindi",
    "Italian",
    "Brazilian Portuguese",
    "Korean",
    "Russian",
    "Arabic",
    "English",
]

TRANSLATE_TONES = ["Natural", "Formal", "Casual", "Professional"]


def translate_for_voice(
    text: str,
    target_language: str,
    tone: str = "Natural",
    content_style: str = "None",
) -> str:
    """
    Translate text into target_language using Qwen.

    target_language : plain name e.g. "Japanese", "Mandarin Chinese"
    tone            : "Natural" | "Formal" | "Casual" | "Professional"
    content_style   : optional genre hint ("Podcast", "Story — Fiction", …)
    """
    if not is_llm_available():
        logger.warning("translate_for_voice: llama-cpp-python not installed.")
        return text
    if not is_qwen_model_ready():
        logger.warning("translate_for_voice: Qwen model not found.")
        return text

    _tone_clause = f" with a {tone.lower()} tone" if tone and tone != "Natural" else ""
    _style_clause = f" The text is {content_style} content — preserve its style." if content_style and content_style not in ("None", "") else ""
    _tmpl = _PROMPTS.get("translate", _PROMPTS_DEFAULTS["translate"])
    system_prompt = _tmpl.format(target_language=target_language, tone_clause=_tone_clause) + _style_clause

    with _llm_lock:
        _load_llm()

    # Process paragraph by paragraph so large documents don't blow the context.
    paragraphs = [p.strip() for p in re.split(r'\n{2,}', text) if p.strip()]
    if not paragraphs:
        return text

    logger.info("translate_for_voice: translating %d chars -> %s [%s]", len(text), target_language, tone)

    result_parts = []
    for para in paragraphs:
        chunks = _chunk_text(para, max_chars=400)
        para_result = []
        for chunk in chunks:
            try:
                with _llm_lock:
                    # Translation output ≈ same length as input in tokens.
                    # ~4 chars per token, ×2 expansion headroom, capped at 600.
                    # The old formula (len*3 + 256) produced 1000+ tokens for a
                    # short paragraph, causing multi-minute waits on a 0.5B model.
                    _max_tok = min(600, int(len(chunk) / 4 * 2) + 100)
                    output = _llm.create_chat_completion(
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user",   "content": chunk},
                        ],
                        max_tokens=_max_tok,
                        temperature=0.1,
                        top_p=0.95,
                        repeat_penalty=1.1,
                    )
                result = output["choices"][0]["message"]["content"].strip()
                logger.info("translate_for_voice: chunk %d chars → %d chars result: %.80r",
                            len(chunk), len(result), result)
                if not result:
                    logger.warning("translate_for_voice: empty result, keeping original chunk")
                    result = chunk
            except Exception as exc:
                logger.warning("translate_for_voice chunk failed: %s", exc)
                result = chunk
            para_result.append(result)
        result_parts.append(" ".join(para_result))

    return "\n\n".join(result_parts)


def rewrite_tone(
    text: str,
    tone: str,
    on_progress: Optional[Callable[[str, float], None]] = None,
) -> str:
    """
    Use Qwen 0.5B to rewrite text in the given tone.
    tone should be one of TONE_OPTIONS.
    """
    if not is_llm_available():
        raise RuntimeError("llama-cpp-python is not installed.")
    if not is_qwen_model_ready():
        raise RuntimeError(f"Qwen model not found at {QWEN_MODEL_PATH}.")

    system_prompt = _PROMPTS.get("tone", _TONE_SYSTEM_PROMPT_TEMPLATE).format(tone=tone)

    with _llm_lock:
        _load_llm()

    paragraphs = [p.strip() for p in re.split(r'\n{2,}', text) if p.strip()]
    if not paragraphs:
        return text

    result_parts = []
    for i, para in enumerate(paragraphs):
        if on_progress:
            on_progress(f"Rewriting paragraph {i + 1}/{len(paragraphs)}…",
                        0.1 + 0.85 * (i / len(paragraphs)))
        chunks = _chunk_text(para, max_chars=600)
        para_result = []
        for chunk in chunks:
            with _llm_lock:
                output = _llm.create_chat_completion(
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user",   "content": chunk},
                    ],
                    # ~4 chars per token; tone rewrite ≈ same length; cap at 700
                    max_tokens=min(700, int(len(chunk) / 4 * 2) + 128),
                    temperature=0.7,
                    top_p=0.95,
                )
            result = output["choices"][0]["message"]["content"].strip()
            if len(result) > len(chunk) * 3.5 or len(result) < len(chunk) * 0.3:
                logger.warning("rewrite_tone: suspicious output length — keeping original.")
                result = chunk
            para_result.append(result)
        result_parts.append(" ".join(para_result))

    if on_progress:
        on_progress("Done", 1.0)
    return "\n\n".join(result_parts)


def _chunk_text(text: str, max_chars: int = 600) -> list:
    """Split text into sentence-boundary chunks no larger than max_chars."""
    sentences = re.split(r'(?<=[.!?])\s+', text)
    chunks, buf = [], ""
    for s in sentences:
        if buf and len(buf) + len(s) + 1 > max_chars:
            chunks.append(buf.strip())
            buf = s
        else:
            buf = (buf + " " + s).strip() if buf else s
    if buf:
        chunks.append(buf.strip())
    return chunks or [text]
