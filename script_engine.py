"""
KoKoFish — Script Lab engine.

Handles character profile management, script parsing,
AI script tagging, and multi-voice playback coordination.
"""

import json
import logging
import os
import re
import sys

logger = logging.getLogger("KoKoFish.script")

APP_DIR = (
    os.path.dirname(sys.executable)
    if getattr(sys, "frozen", False)
    else os.path.dirname(os.path.abspath(__file__))
)
PROFILES_DIR = os.path.join(APP_DIR, "scripts", "profiles")
SCRIPTS_DIR  = os.path.join(APP_DIR, "scripts")


# ---------------------------------------------------------------------------
# Profile management
# ---------------------------------------------------------------------------

def get_profiles_dir() -> str:
    os.makedirs(PROFILES_DIR, exist_ok=True)
    return PROFILES_DIR


def list_profiles() -> list:
    d = get_profiles_dir()
    return [f[:-5] for f in sorted(os.listdir(d)) if f.endswith(".json")]


def load_profile(name: str) -> dict:
    path = os.path.join(get_profiles_dir(), f"{name}.json")
    if os.path.isfile(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as exc:
            logger.warning("load_profile failed: %s", exc)
    return default_profile()


def save_profile(name: str, profile: dict):
    os.makedirs(get_profiles_dir(), exist_ok=True)
    path = os.path.join(get_profiles_dir(), f"{name}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(profile, f, indent=2, ensure_ascii=False)
    logger.info("Profile saved: %s", name)


def delete_profile(name: str):
    path = os.path.join(get_profiles_dir(), f"{name}.json")
    if os.path.isfile(path):
        os.remove(path)
        logger.info("Profile deleted: %s", name)


def default_profile() -> dict:
    return {
        "characters": [],
        "narrator": {"voice": "", "blend_voice": "", "blend_ratio": 0.0},
    }


# ---------------------------------------------------------------------------
# Script parsing
# ---------------------------------------------------------------------------

def parse_script(text: str) -> list:
    """
    Parse a [Character] tagged script into a list of dicts.
    Each dict: {character: str, text: str}
    Lines without a tag are assigned to Narrator.
    """
    segments = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        m = re.match(r'^\[([^\]]+)\]\s*(.*)', line)
        if m:
            segments.append({
                "character": m.group(1).strip(),
                "text":      m.group(2).strip(),
            })
        else:
            segments.append({"character": "Narrator", "text": line})
    return segments


def format_script(segments: list) -> str:
    """Convert parsed segments back to tagged script text."""
    return "\n".join(f"[{s['character']}] {s['text']}" for s in segments)


def read_source_file(path: str) -> str:
    """Read a source document (txt, pdf, docx, epub) to plain text."""
    ext = os.path.splitext(path)[1].lower()
    try:
        if ext == ".txt":
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                return f.read()
        elif ext == ".pdf":
            import pdfplumber
            pages = []
            with pdfplumber.open(path) as pdf:
                for page in pdf.pages:
                    t = page.extract_text()
                    if t:
                        pages.append(t)
            return "\n\n".join(pages)
        elif ext == ".docx":
            from docx import Document
            doc = Document(path)
            return "\n\n".join(p.text for p in doc.paragraphs if p.text.strip())
        elif ext == ".epub":
            import ebooklib
            from ebooklib import epub
            from html.parser import HTMLParser

            class _Strip(HTMLParser):
                def __init__(self):
                    super().__init__()
                    self.parts = []
                def handle_data(self, data):
                    self.parts.append(data)

            book = epub.read_epub(path)
            chapters = []
            for item in book.get_items_of_type(ebooklib.ITEM_DOCUMENT):
                p = _Strip()
                p.feed(item.get_content().decode("utf-8", errors="replace"))
                chapters.append("".join(p.parts))
            return "\n\n".join(chapters)
    except Exception as exc:
        raise RuntimeError(f"Could not read {os.path.basename(path)}: {exc}")
    return ""


# ---------------------------------------------------------------------------
# AI script tagging prompts
# ---------------------------------------------------------------------------

_KOKORO_PROMPT = (
    "You are a script formatter converting prose fiction into a voice-acted script.\n"
    "Rules:\n"
    "1. Find all dialogue (text in quotation marks) and identify the speaker from context.\n"
    "2. Format each dialogue line as: [SpeakerName] dialogue text\n"
    "3. Remove all attribution text (e.g. 'John said', 'she replied', 'he asked').\n"
    "4. Format all narration as: [Narrator] narration text\n"
    "5. Do NOT add any emotion or style tags — plain text only.\n"
    "6. If the speaker is unclear, use [Narrator].\n"
    "7. Known characters: {characters}\n"
    "Output ONLY the formatted script. No explanations, no commentary."
)

_FISH_PROMPT = (
    "You are a script formatter converting prose fiction into a voice-acted script.\n"
    "Rules:\n"
    "1. Find all dialogue (text in quotation marks) and identify the speaker from context.\n"
    "2. Format each dialogue line as: [SpeakerName] (emotion) dialogue text\n"
    "3. Remove all attribution text (e.g. 'John said excitedly', 'she whispered', 'he shouted').\n"
    "4. Format all narration as: [Narrator] narration text\n"
    "5. Add ONE emotion tag before the dialogue based on how it was spoken:\n"
    "   (excited) (happy) (satisfied) (confident) (gentle) (serious) (sad) (angry) (nervous) (fearful) (surprised) (confused)\n"
    "   Use [whisper] for whispered speech. Use (laugh) for laughing.\n"
    "6. If the speaker is unclear, use [Narrator].\n"
    "7. Known characters: {characters}\n"
    "Output ONLY the formatted script. No explanations, no commentary."
)


def tag_script_with_ai(
    source_text: str,
    engine: str,
    characters: list,
    on_progress=None,
) -> str:
    """
    Use the local LLM to convert prose to a [Character] tagged script.

    engine     : 'kokoro' | 'fish14' | 'fish15' | 's1mini' | 's1'
    characters : list of known character name strings
    on_progress: optional callable(message: str, fraction: float)

    Returns the tagged script as a string.
    """
    try:
        from tag_suggester import (
            _infer_chunk, is_llm_available, is_qwen_model_ready,
            _load_llm, _active_is_ollama, _llm_lock,
        )
    except ImportError as exc:
        raise RuntimeError(f"LLM not available: {exc}")

    if not is_llm_available():
        raise RuntimeError(
            "LLM runtime not installed.\n"
            "Install llama-cpp-python from Settings."
        )
    if not is_qwen_model_ready():
        raise RuntimeError(
            "No LLM model downloaded.\n"
            "Download a model from Settings."
        )

    char_list = ", ".join(characters) if characters else "unknown"
    system_prompt = (
        _KOKORO_PROMPT if engine == "kokoro" else _FISH_PROMPT
    ).format(characters=char_list)

    if not _active_is_ollama():
        with _llm_lock:
            _load_llm()

    paragraphs = [p.strip() for p in re.split(r'\n{2,}', source_text) if p.strip()]
    total = max(len(paragraphs), 1)
    result_parts = []

    for i, para in enumerate(paragraphs):
        if on_progress:
            on_progress(f"Tagging paragraph {i + 1} of {total}...", i / total)

        # Break very long paragraphs into ~600-char chunks
        chunks = [para[j:j + 600] for j in range(0, len(para), 600)]
        for chunk in chunks:
            try:
                max_tok = min(900, int(len(chunk) / 4 * 3) + 200)
                tagged = _infer_chunk(
                    system_prompt, chunk,
                    max_tokens=max_tok,
                    temperature=0.2,
                    top_p=0.95,
                    repeat_penalty=1.1,
                )
                if tagged and tagged.strip():
                    result_parts.append(tagged.strip())
                else:
                    result_parts.append(f"[Narrator] {chunk}")
            except Exception as exc:
                logger.warning("tag_script chunk failed: %s", exc)
                result_parts.append(f"[Narrator] {chunk}")

    if on_progress:
        on_progress("Done.", 1.0)

    return "\n".join(result_parts)
