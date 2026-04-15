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
    "1. Find all dialogue (text in quotation marks) and identify the speaker from surrounding context (e.g. 'said John', 'she replied').\n"
    "2. Format each dialogue line as: [SpeakerName] dialogue text\n"
    "3. Remove all attribution text (e.g. 'John said', 'she replied', 'he asked').\n"
    "4. Format all narration as: [Narrator] narration text\n"
    "5. Do NOT add any emotion or style tags — plain text only.\n"
    "6. If the speaker is unclear, use [Narrator].\n"
    "7. IMPORTANT: If a character's name appears inside quotation marks (e.g. 'She called out \"John!\"'), "
    "that is dialogue referencing the name — do NOT treat it as that character speaking.\n"
    "8. Match the emotional tone of each line to the flow of the conversation. "
    "Consider what was just said and how the speaker would naturally respond.\n"
    "9. Known characters: {characters}\n"
    "Output ONLY the formatted script. No explanations, no commentary."
)

_FISH_PROMPT = (
    "You are a script formatter converting prose fiction into a voice-acted script.\n"
    "Rules:\n"
    "1. Find all dialogue (text in quotation marks) and identify the speaker from surrounding context (e.g. 'said John', 'she replied').\n"
    "2. Format each dialogue line as: [SpeakerName] (emotion) dialogue text\n"
    "3. Remove all attribution text (e.g. 'John said excitedly', 'she whispered', 'he shouted').\n"
    "4. Format all narration as: [Narrator] narration text\n"
    "5. Add ONE emotion tag before the dialogue based on how it was spoken:\n"
    "   (excited) (happy) (satisfied) (confident) (gentle) (serious) (sad) (angry) (nervous) (fearful) (surprised) (confused)\n"
    "   Use [whisper] for whispered speech. Use (laugh) for laughing.\n"
    "6. IMPORTANT: If a character's name appears inside quotation marks (e.g. 'She called out \"John!\"'), "
    "that is dialogue referencing the name — do NOT treat it as that character speaking.\n"
    "7. Match emotion tags to the flow of conversation — consider what was said in the previous line "
    "and choose an emotion that makes sense as a natural response or continuation.\n"
    "8. If the speaker is unclear, use [Narrator].\n"
    "9. Known characters: {characters}\n"
    "Output ONLY the formatted script. No explanations, no commentary."
)

_ENHANCE_KOKORO_PROMPT = (
    "You are a script editor improving a voice-acted script for natural flow and emotional continuity.\n"
    "The script uses the format: [CharacterName] dialogue text\n"
    "Rules:\n"
    "1. Keep all [CharacterName] tags exactly as they are. Do NOT change speaker names.\n"
    "2. Improve the dialogue text for natural spoken delivery — fix awkward phrasing, "
    "remove written-prose habits, and make lines sound like they were actually spoken.\n"
    "3. Ensure emotional continuity: each line should feel like a natural response to the previous one. "
    "Characters should react to each other realistically.\n"
    "4. Do NOT add emotion tags — this is for the Kokoro engine which uses plain text.\n"
    "5. Do NOT summarize, shorten, or remove lines. Every line in must produce a line out.\n"
    "6. IMPORTANT: Names inside quotation marks within dialogue are references, not speakers — leave them.\n"
    "Output ONLY the improved script. No explanations, no commentary."
)

_ENHANCE_FISH_PROMPT = (
    "You are a script editor improving a voice-acted script for natural flow and emotional continuity.\n"
    "The script uses the format: [CharacterName] (emotion) dialogue text\n"
    "Rules:\n"
    "1. Keep all [CharacterName] tags exactly as they are. Do NOT change speaker names.\n"
    "2. Update or correct the (emotion) tag on each dialogue line so it matches the flow of conversation. "
    "Valid tags: (excited) (happy) (satisfied) (confident) (gentle) (serious) (sad) (angry) "
    "(nervous) (fearful) (surprised) (confused) (laugh) [whisper]\n"
    "3. Improve dialogue text for natural spoken delivery — fix awkward phrasing and written-prose habits.\n"
    "4. Ensure emotional continuity: each character's emotional state should react realistically to the previous line.\n"
    "5. Do NOT summarize, shorten, or remove lines. Every line in must produce a line out.\n"
    "6. IMPORTANT: Names inside quotation marks within dialogue are references, not speakers — leave them.\n"
    "Output ONLY the improved script. No explanations, no commentary."
)


def find_characters_in_script(script_text: str) -> list:
    """
    Scan script_text for all [Name] tags and return a sorted unique list.
    Always includes 'Narrator'. Names inside quotes are ignored by the parser
    because they follow the bracket format, not the quote format.
    """
    names = set()
    names.add("Narrator")
    for m in re.finditer(r'^\[([^\]]+)\]', script_text, re.MULTILINE):
        names.add(m.group(1).strip())
    return sorted(names)


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


def enhance_script_flow(
    script_text: str,
    engine: str,
    on_progress=None,
) -> str:
    """
    Run the already-tagged script through the LLM to improve emotional
    continuity, natural delivery, and conversation flow.

    engine     : 'kokoro' | 'fish14' | 'fish15' | 's1mini' | 's1'
    on_progress: optional callable(message: str, fraction: float)

    Returns the enhanced script as a string.
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

    system_prompt = _ENHANCE_KOKORO_PROMPT if engine == "kokoro" else _ENHANCE_FISH_PROMPT

    if not _active_is_ollama():
        with _llm_lock:
            _load_llm()

    # Split into ~30-line windows so the model sees enough context for continuity
    lines = [l for l in script_text.splitlines() if l.strip()]
    window = 30
    result_lines = []
    total = max(len(lines), 1)

    for i in range(0, len(lines), window):
        chunk_lines = lines[i:i + window]
        chunk = "\n".join(chunk_lines)
        if on_progress:
            on_progress(
                f"Enhancing lines {i + 1}–{min(i + window, total)} of {total}...",
                i / total,
            )
        try:
            max_tok = min(1200, int(len(chunk) / 4 * 3) + 300)
            enhanced = _infer_chunk(
                system_prompt, chunk,
                max_tokens=max_tok,
                temperature=0.3,
                top_p=0.95,
                repeat_penalty=1.1,
            )
            if enhanced and enhanced.strip():
                result_lines.extend(enhanced.strip().splitlines())
            else:
                result_lines.extend(chunk_lines)
        except Exception as exc:
            logger.warning("enhance_script_flow chunk failed: %s", exc)
            result_lines.extend(chunk_lines)

    if on_progress:
        on_progress("Enhancement complete.", 1.0)

    return "\n".join(result_lines)
