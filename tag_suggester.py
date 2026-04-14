"""
FishTalk — Tag Suggester

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
import threading
from typing import Optional, Callable

logger = logging.getLogger("FishTalk.tags")

APP_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(APP_DIR, "models")
QWEN_MODEL_FILENAME = "qwen2.5-0.5b-instruct-q4_k_m.gguf"
QWEN_MODEL_PATH = os.path.join(MODELS_DIR, QWEN_MODEL_FILENAME)
QWEN_HF_REPO = "Qwen/Qwen2.5-0.5B-Instruct-GGUF"

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


def is_llm_available() -> bool:
    """Return True if llama-cpp-python is installed."""
    try:
        import llama_cpp  # noqa
        return True
    except ImportError:
        return False


def is_qwen_model_ready() -> bool:
    """Return True if the Qwen GGUF model file exists on disk."""
    return os.path.isfile(QWEN_MODEL_PATH)


def download_qwen_model(
    on_progress: Optional[Callable[[str, float], None]] = None,
    on_complete: Optional[Callable[[bool, str], None]] = None,
):
    """Download the Qwen 2.5 0.5B Q4_K_M GGUF from HuggingFace in a background thread."""

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
                on_progress(f"Downloading {QWEN_MODEL_FILENAME} (~400 MB)…", 0.05)
            logger.info("Downloading Qwen model from %s", QWEN_HF_REPO)

            hf_hub_download(
                repo_id=QWEN_HF_REPO,
                filename=QWEN_MODEL_FILENAME,
                local_dir=MODELS_DIR,
                local_dir_use_symlinks=False,
            )

            # hf_hub_download may nest inside a subfolder — move to flat MODELS_DIR
            nested = os.path.join(MODELS_DIR, QWEN_MODEL_FILENAME.split("/")[-1])
            if not os.path.isfile(QWEN_MODEL_PATH) and os.path.isfile(nested):
                import shutil
                shutil.move(nested, QWEN_MODEL_PATH)

            if is_qwen_model_ready():
                logger.info("Qwen model downloaded: %s", QWEN_MODEL_PATH)
                if on_complete:
                    on_complete(True, "Qwen model ready.")
            else:
                if on_complete:
                    on_complete(False, "Download completed but model file not found.")
        except Exception as exc:
            logger.error("Qwen download failed: %s", exc)
            if on_complete:
                on_complete(False, str(exc))

    threading.Thread(target=_worker, daemon=True, name="QwenDownload").start()


def _load_llm():
    """Load the Qwen GGUF model (call inside _llm_lock)."""
    global _llm
    if _llm is not None:
        return
    from llama_cpp import Llama
    logger.info("Loading Qwen model: %s", QWEN_MODEL_PATH)
    _llm = Llama(
        model_path=QWEN_MODEL_PATH,
        n_ctx=2048,
        n_threads=max(1, os.cpu_count() - 1),
        n_gpu_layers=0,      # CPU only — keeps VRAM free for TTS
        verbose=False,
    )
    logger.info("Qwen model loaded.")


def unload_llm():
    """Explicitly unload the Qwen model to reclaim RAM."""
    global _llm
    with _llm_lock:
        if _llm is not None:
            del _llm
            _llm = None
            import gc; gc.collect()
            logger.info("Qwen model unloaded.")


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
                        {"role": "system", "content": _SYSTEM_PROMPT},
                        {"role": "user", "content": chunk},
                    ],
                    max_tokens=int(len(chunk) * 1.5) + 64,
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
