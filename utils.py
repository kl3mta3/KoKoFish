"""
FishTalk — Utility functions.

Handles ffmpeg detection, file reading (txt/pdf/docx), audio export,
document export, and system monitoring helpers.
"""

import os
import re
import sys
import shutil
import logging

import psutil
import pdfplumber
from docx import Document as DocxDocument
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer

logger = logging.getLogger("FishTalk.utils")

# ---------------------------------------------------------------------------
# FFmpeg setup
# ---------------------------------------------------------------------------

def get_app_dir() -> str:
    """Return the directory where the application lives."""
    if getattr(sys, "frozen", False):
        return os.path.dirname(sys.executable)
    return os.path.dirname(os.path.abspath(__file__))


def _download_ffmpeg(bin_dir: str, on_progress=None) -> bool:
    """
    Download a pre-built FFmpeg Windows binary from gyan.dev and extract
    ffmpeg.exe + ffprobe.exe into bin_dir.

    Returns True on success, False on failure.
    """
    import urllib.request
    import zipfile
    import tempfile

    # Essentials build: ~75 MB, contains ffmpeg.exe and ffprobe.exe
    URL = "https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-essentials.zip"

    os.makedirs(bin_dir, exist_ok=True)

    try:
        if on_progress:
            on_progress("Downloading FFmpeg...", 0.05)

        logger.info("Downloading FFmpeg from %s", URL)

        tmp_zip = os.path.join(tempfile.gettempdir(), "ffmpeg_download.zip")

        def _reporthook(block_num, block_size, total_size):
            if on_progress and total_size > 0:
                frac = min(block_num * block_size / total_size, 1.0)
                on_progress(f"Downloading FFmpeg... {int(frac * 100)}%", frac * 0.8)

        urllib.request.urlretrieve(URL, tmp_zip, reporthook=_reporthook)

        if on_progress:
            on_progress("Extracting FFmpeg...", 0.85)

        with zipfile.ZipFile(tmp_zip, "r") as zf:
            for member in zf.namelist():
                filename = os.path.basename(member)
                if filename in ("ffmpeg.exe", "ffprobe.exe"):
                    source = zf.open(member)
                    dest_path = os.path.join(bin_dir, filename)
                    with open(dest_path, "wb") as dest:
                        dest.write(source.read())
                    logger.info("Extracted %s -> %s", filename, dest_path)

        try:
            os.remove(tmp_zip)
        except OSError:
            pass

        if on_progress:
            on_progress("FFmpeg ready", 1.0)

        return os.path.isfile(os.path.join(bin_dir, "ffmpeg.exe"))

    except Exception as exc:
        logger.error("FFmpeg auto-download failed: %s", exc)
        return False


def setup_ffmpeg(on_progress=None) -> bool:
    """
    Locate FFmpeg and ensure it is available on the OS PATH so libraries
    like pydub can find it. If FFmpeg is not present, automatically
    downloads the official Windows build from gyan.dev into bin/.

    Accepts an optional on_progress(message, fraction) callback for
    splash screen integration.
    """
    app_dir = get_app_dir()
    bin_dir = os.path.join(app_dir, "bin")
    bundled_ffmpeg = os.path.join(bin_dir, "ffmpeg.exe")

    def _activate(ffmpeg_path: str) -> bool:
        """Put the containing folder on PATH and configure pydub."""
        folder = os.path.dirname(ffmpeg_path)
        current_path = os.environ.get("PATH", "")
        if folder not in current_path:
            os.environ["PATH"] = folder + os.pathsep + current_path
        try:
            from pydub import AudioSegment
            AudioSegment.converter = ffmpeg_path
        except ImportError:
            pass
        return True

    # 1. Bundled bin/ folder
    if os.path.isfile(bundled_ffmpeg):
        logger.info("FFmpeg found (bundled): %s", bundled_ffmpeg)
        return _activate(bundled_ffmpeg)

    # 2. System PATH
    path_ffmpeg = shutil.which("ffmpeg")
    if path_ffmpeg:
        logger.info("FFmpeg found (PATH): %s", path_ffmpeg)
        return _activate(path_ffmpeg)

    # 3. Auto-download
    logger.warning("FFmpeg not found -- attempting auto-download.")
    success = _download_ffmpeg(bin_dir, on_progress=on_progress)
    if success:
        logger.info("FFmpeg auto-download succeeded.")
        return _activate(bundled_ffmpeg)

    logger.error("FFmpeg could not be installed automatically.")
    return False




def is_ffmpeg_available() -> bool:
    """Quick check without mutating state."""
    bundled = os.path.join(get_app_dir(), "bin", "ffmpeg.exe")
    return os.path.isfile(bundled) or shutil.which("ffmpeg") is not None


# ---------------------------------------------------------------------------
# Fish-Speech auto-setup
# ---------------------------------------------------------------------------

def is_kokoro_ready() -> bool:
    """Return True if both Kokoro ONNX model files are present."""
    app_dir = os.path.dirname(os.path.abspath(__file__))
    model_dir = os.path.join(app_dir, "kokoro_models")
    return (
        os.path.isfile(os.path.join(model_dir, "kokoro-v1.0.int8.onnx")) and
        os.path.isfile(os.path.join(model_dir, "voices-v1.0.bin"))
    )


def setup_kokoro(on_progress=None) -> bool:
    """
    Download Kokoro ONNX model files from HuggingFace if missing.

    Files: kokoro-v1.0.int8.onnx (~310 MB) and voices-v1.0.bin (~20 MB)
    from hexgrad/Kokoro-82M.

    on_progress(message, fraction) is called throughout.
    Returns True if models are ready, False on failure.
    """
    app_dir = os.path.dirname(os.path.abspath(__file__))
    model_dir = os.path.join(app_dir, "kokoro_models")
    os.makedirs(model_dir, exist_ok=True)

    files = [
        ("kokoro-v1.0.int8.onnx", "onnx/kokoro-v1.0.int8.onnx"),
        ("voices-v1.0.bin", "voices-v1.0.bin"),
    ]

    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        logger.error("huggingface_hub not installed; cannot download Kokoro models.")
        return False

    for i, (local_name, repo_path) in enumerate(files):
        dest = os.path.join(model_dir, local_name)
        if os.path.isfile(dest):
            continue
        frac_base = i / len(files)
        if on_progress:
            on_progress(f"Downloading Kokoro {local_name}…", frac_base + 0.05)
        try:
            logger.info("Downloading Kokoro model: %s", repo_path)
            hf_hub_download(
                repo_id="hexgrad/Kokoro-82M",
                filename=repo_path,
                local_dir=model_dir,
                local_dir_use_symlinks=False,
            )
            # hf_hub_download saves into a subfolder matching repo_path structure;
            # move it to the flat model_dir if needed.
            downloaded = os.path.join(model_dir, repo_path)
            if os.path.isfile(downloaded) and downloaded != dest:
                import shutil as _shutil
                _shutil.move(downloaded, dest)
            logger.info("Kokoro model saved: %s", dest)
        except Exception as exc:
            logger.error("Kokoro model download failed (%s): %s", local_name, exc)
            return False

    if on_progress:
        on_progress("Kokoro models ready", 1.0)
    return is_kokoro_ready()


# ---------------------------------------------------------------------------
# Fish-Speech engine config
# ---------------------------------------------------------------------------

# Maps engine key → (checkpoint folder name, HuggingFace repo ID, needs HF token)
FISH_ENGINE_CONFIG = {
    "fish14": ("fish-speech-1.4", "fishaudio/fish-speech-1.4", False),
    "s1mini":  ("openaudio-s1-mini", "fishaudio/openaudio-s1-mini", True),
    "s1":      ("openaudio-s1",      "fishaudio/openaudio-s1",      True),
}

# All Fish Speech engines share the same v1.4.3 code directory
FISH_CODE_DIR_NAME = "fish-speech"
FISH_CODE_URL = "https://github.com/fishaudio/fish-speech/archive/refs/tags/v1.4.3.zip"


def is_fish_speech_ready(engine: str = "fish14") -> bool:
    """Return True if the Fish-Speech code and checkpoints for the engine are present."""
    app_dir = os.path.dirname(os.path.abspath(__file__))
    code_dir = os.path.join(app_dir, FISH_CODE_DIR_NAME)
    ckpt_name = FISH_ENGINE_CONFIG.get(engine, FISH_ENGINE_CONFIG["fish14"])[0]
    ckpt_dir  = os.path.join(code_dir, "checkpoints", ckpt_name)

    if not os.path.isdir(code_dir) or not os.listdir(code_dir):
        return False
    if not os.path.isdir(ckpt_dir):
        return False
    return any(
        f.endswith((".pth", ".bin", ".safetensors"))
        for f in os.listdir(ckpt_dir)
    )


def setup_fish_speech(engine: str = "fish14", on_progress=None, hf_token: str = "") -> bool:
    """
    Ensure Fish-Speech code and model checkpoints are present for the given engine.

    engine: "fish14" | "s1mini" | "s1"

    Steps:
      1. Downloads Fish-Speech v1.4.3 source from GitHub if not present (~20 MB).
      2. Downloads the engine's checkpoints from HuggingFace (~1–2 GB).
         S1 Mini and S1 require a HuggingFace token (gated models).

    on_progress(message, fraction) is called throughout.
    Returns True if ready, False on failure.
    """
    import urllib.request
    import zipfile
    import tempfile

    cfg = FISH_ENGINE_CONFIG.get(engine, FISH_ENGINE_CONFIG["fish14"])
    ckpt_name, hf_repo, needs_token = cfg

    app_dir  = os.path.dirname(os.path.abspath(__file__))
    code_dir = os.path.join(app_dir, FISH_CODE_DIR_NAME)
    ckpt_dir = os.path.join(code_dir, "checkpoints", ckpt_name)

    # --- Step 1: Fish-Speech code (shared across all engines) ---
    if not os.path.isdir(code_dir) or not os.listdir(code_dir):
        try:
            if on_progress:
                on_progress("Downloading Fish-Speech code (~20 MB)...", 0.05)
            logger.info("Downloading Fish-Speech source from %s", FISH_CODE_URL)
            tmp_zip = os.path.join(tempfile.gettempdir(), "fish_speech_src.zip")

            def _hook(b, bs, total):
                if on_progress and total > 0:
                    frac = min(b * bs / total, 1.0) * 0.2
                    on_progress(f"Downloading Fish-Speech code... {int(frac * 500)}%", frac)

            urllib.request.urlretrieve(FISH_CODE_URL, tmp_zip, reporthook=_hook)
            if on_progress:
                on_progress("Extracting Fish-Speech code...", 0.22)

            os.makedirs(code_dir, exist_ok=True)
            with zipfile.ZipFile(tmp_zip, "r") as zf:
                for member in zf.infolist():
                    parts = member.filename.split("/", 1)
                    if len(parts) < 2 or not parts[1]:
                        continue
                    target = os.path.join(code_dir, parts[1])
                    if member.is_dir():
                        os.makedirs(target, exist_ok=True)
                    else:
                        os.makedirs(os.path.dirname(target), exist_ok=True)
                        with zf.open(member) as src, open(target, "wb") as dst:
                            dst.write(src.read())
            try:
                os.remove(tmp_zip)
            except OSError:
                pass
            logger.info("Fish-Speech code extracted to %s", code_dir)
        except Exception as exc:
            logger.error("Fish-Speech code download failed: %s", exc)
            return False

    # --- Step 2: Model checkpoints ---
    has_weights = (
        os.path.isdir(ckpt_dir) and any(
            f.endswith((".pth", ".bin", ".safetensors"))
            for f in os.listdir(ckpt_dir)
        )
    ) if os.path.isdir(ckpt_dir) else False

    if not has_weights:
        try:
            from huggingface_hub import snapshot_download
        except ImportError:
            logger.error("huggingface_hub not installed; cannot download checkpoints.")
            return False

        if on_progress:
            on_progress(f"Downloading {ckpt_name} checkpoints (~1.5 GB)...", 0.25)
        logger.info("Downloading checkpoints from HuggingFace: %s", hf_repo)

        token_arg = hf_token.strip() if (needs_token and hf_token) else None
        try:
            snapshot_download(
                repo_id=hf_repo,
                local_dir=ckpt_dir,
                ignore_patterns=["*.md", "*.txt"],
                token=token_arg,
            )
            logger.info("Checkpoints downloaded to %s", ckpt_dir)
        except Exception as exc:
            logger.error("Checkpoint download failed for %s: %s", ckpt_name, exc)
            return False

    if on_progress:
        on_progress("Fish-Speech ready", 1.0)
    return True




# ---------------------------------------------------------------------------
# Text normalization
# ---------------------------------------------------------------------------

# Emotion/prosody tags Fish Speech understands — preserve these for Fish, strip for Kokoro
_FISH_EMOTION_TAGS = re.compile(
    r'\((?:excited|happy|sad|angry|surprised|confused|nervous|confident|'
    r'satisfied|fearful|whisper|laughing|crying|shouting|serious|gentle)\)',
    re.IGNORECASE,
)
_FISH_BRACKET_TAGS = re.compile(r'\[[^\]]{1,40}\]')


def normalize_text(text: str, engine: str = "fish") -> str:
    """
    Normalize text before TTS to improve pronunciation and quality.

    engine: "fish" / "fish14" / "s1mini" / "s1" — preserve Fish Speech tags
            "kokoro" — strip all inline tags (Kokoro reads plain text only)

    - Strips unknown/broken markup
    - Converts numbers to words (requires num2words)
    - Expands common abbreviations
    - Strips URLs
    """
    # Strip Fish Speech tags only for Kokoro — all Fish Speech engines preserve them
    if engine == "kokoro":
        text = _FISH_BRACKET_TAGS.sub(' ', text)
        text = _FISH_EMOTION_TAGS.sub(' ', text)

    # 1. Strip URLs
    text = re.sub(r'https?://\S+|www\.\S+', '', text)

    # 2. Common abbreviations — order matters (longer matches first)
    abbrevs = [
        (r'\bDr\.(?=\s)', 'Doctor'),
        (r'\bMr\.(?=\s)', 'Mister'),
        (r'\bMrs\.(?=\s)', 'Missus'),
        (r'\bMs\.(?=\s)', 'Miss'),
        (r'\bProf\.(?=\s)', 'Professor'),
        (r'\bSgt\.(?=\s)', 'Sergeant'),
        (r'\bCpt\.(?=\s)', 'Captain'),
        (r'\bLt\.(?=\s)', 'Lieutenant'),
        (r'\bSt\.(?=\s)', 'Saint'),
        (r'\bAve\.', 'Avenue'),
        (r'\bBlvd\.', 'Boulevard'),
        (r'\bvs\.', 'versus'),
        (r'\betc\.', 'etcetera'),
        (r'\be\.g\.', 'for example'),
        (r'\bi\.e\.', 'that is'),
    ]
    for pattern, replacement in abbrevs:
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)

    # 3. Numbers to words
    try:
        from num2words import num2words

        def _replace_number(m):
            raw = m.group(0).replace(',', '')
            try:
                if '.' in raw:
                    return num2words(float(raw))
                return num2words(int(raw))
            except Exception:
                return m.group(0)

        # Match numbers with optional thousands-commas (e.g. 1,234 or 3.14)
        text = re.sub(r'\b\d{1,3}(?:,\d{3})*(?:\.\d+)?\b|\b\d+(?:\.\d+)?\b',
                      _replace_number, text)
    except ImportError:
        pass  # num2words not installed; skip silently

    # 4. Collapse whitespace
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


# ---------------------------------------------------------------------------
# Reference audio pre-processing
# ---------------------------------------------------------------------------

def preprocess_reference_audio(wav_path: str, denoise: bool = True) -> str:
    """
    Clean up a reference audio clip for better voice cloning quality.

    Steps applied:
      1. Convert to mono 16-bit PCM
      2. Trim leading/trailing silence (threshold -40 dBFS, 150 ms padding)
      3. Normalize peak volume to -1 dBFS
      4. Optional light denoising via noisereduce (stationary noise only)

    Returns the path to a processed temp WAV (caller should delete when done).
    If pydub is unavailable the original path is returned unchanged.
    """
    import tempfile
    import os

    try:
        from pydub import AudioSegment
        from pydub.silence import detect_leading_silence
        from pydub.effects import normalize
    except ImportError:
        logger.warning("pydub not available; skipping reference audio preprocessing.")
        return wav_path

    try:
        audio = AudioSegment.from_file(wav_path)

        # Mono
        if audio.channels > 1:
            audio = audio.set_channels(1)

        # Trim silence
        start_trim = detect_leading_silence(audio, silence_threshold=-40)
        end_trim   = detect_leading_silence(audio.reverse(), silence_threshold=-40)
        duration   = len(audio)
        if start_trim + end_trim < duration:
            pad = 150  # ms of silence to keep at each end
            audio = audio[max(0, start_trim - pad): duration - max(0, end_trim - pad)]

        # Normalize volume
        audio = normalize(audio)

        # Optional denoise
        if denoise:
            try:
                import numpy as np
                import noisereduce as nr

                samples = np.array(audio.get_array_of_samples(), dtype=np.float32)
                sr = audio.frame_rate
                reduced = nr.reduce_noise(y=samples, sr=sr, stationary=True, prop_decrease=0.75)
                reduced_int16 = np.clip(reduced, -32768, 32767).astype(np.int16)
                audio = AudioSegment(
                    reduced_int16.tobytes(),
                    frame_rate=sr,
                    sample_width=2,
                    channels=1,
                )
            except ImportError:
                pass  # noisereduce not installed; skip
            except Exception as exc:
                logger.warning("Denoising failed (non-fatal): %s", exc)

        # Write to temp file
        tmp_path = tempfile.mktemp(suffix="_ref_processed.wav")
        audio.export(tmp_path, format="wav")
        logger.info("Reference audio preprocessed: %s -> %s", wav_path, tmp_path)
        return tmp_path

    except Exception as exc:
        logger.warning("Reference audio preprocessing failed (non-fatal): %s", exc)
        return wav_path  # Fall back to original


# ---------------------------------------------------------------------------
# File readers
# ---------------------------------------------------------------------------

def read_txt(path: str) -> str:
    """Read a plain text file."""
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        return f.read()


def read_pdf(path: str) -> str:
    """Extract text from a PDF using pdfplumber."""
    pages = []
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                pages.append(text)
    return "\n\n".join(pages)


def read_docx(path: str) -> str:
    """Extract text from a DOCX file."""
    doc = DocxDocument(path)
    return "\n".join(p.text for p in doc.paragraphs if p.text.strip())


def read_epub(path: str) -> str:
    """
    Extract text from an EPUB file.

    Reads chapters in spine order, strips HTML tags, and joins paragraphs
    with double newlines so the sentence splitter sees proper boundaries.
    """
    import html
    import re
    from html.parser import HTMLParser

    class _TextExtractor(HTMLParser):
        """Minimal HTML-to-text converter that preserves paragraph breaks."""
        BLOCK_TAGS = {"p", "div", "h1", "h2", "h3", "h4", "h5", "h6",
                      "li", "tr", "br", "blockquote", "section", "article"}

        def __init__(self):
            super().__init__()
            self._parts: list = []
            self._current = []
            self._skip = False

        def handle_starttag(self, tag, attrs):
            if tag in ("script", "style", "head"):
                self._skip = True
            if tag in self.BLOCK_TAGS and self._current:
                self._flush()

        def handle_endtag(self, tag):
            if tag in ("script", "style", "head"):
                self._skip = False
                self._current.clear()
            if tag in self.BLOCK_TAGS:
                self._flush()

        def handle_data(self, data):
            if not self._skip:
                data = data.strip()
                if data:
                    self._current.append(data)

        def handle_entityref(self, name):
            self._current.append(html.unescape(f"&{name};"))

        def handle_charref(self, name):
            self._current.append(html.unescape(f"&#{name};"))

        def _flush(self):
            text = " ".join(self._current).strip()
            if text:
                self._parts.append(text)
            self._current.clear()

        def get_text(self) -> str:
            self._flush()
            return "\n\n".join(self._parts)

    try:
        import ebooklib
        from ebooklib import epub
    except ImportError:
        raise ImportError(
            "ebooklib is required for EPUB support.\n"
            "Install it with: pip install ebooklib"
        )

    book = epub.read_epub(path, options={"ignore_ncx": True})
    chapters: list = []

    for item in book.get_items_of_type(ebooklib.ITEM_DOCUMENT):
        content = item.get_content()
        if isinstance(content, bytes):
            content = content.decode("utf-8", errors="replace")
        extractor = _TextExtractor()
        extractor.feed(content)
        text = extractor.get_text().strip()
        if text:
            chapters.append(text)

    if not chapters:
        raise ValueError("No readable text found in EPUB.")

    return "\n\n".join(chapters)


def read_file(path: str) -> str:

    """
    Dispatch to the correct reader based on file extension.
    Supported: .txt, .pdf, .docx, .epub
    """
    ext = os.path.splitext(path)[1].lower()
    readers = {
        ".txt":  read_txt,
        ".pdf":  read_pdf,
        ".docx": read_docx,
        ".epub": read_epub,
    }
    reader = readers.get(ext)
    if reader is None:
        raise ValueError(f"Unsupported file type: {ext}")
    return reader(path)



# ---------------------------------------------------------------------------
# Audio export
# ---------------------------------------------------------------------------

def export_mp3(wav_path: str, out_path: str) -> str:
    """Convert a WAV file to MP3 using pydub + ffmpeg."""
    from pydub import AudioSegment

    if not is_ffmpeg_available():
        raise RuntimeError("ffmpeg is not available. Cannot export to MP3.")
    setup_ffmpeg()
    audio = AudioSegment.from_wav(wav_path)
    audio.export(out_path, format="mp3", bitrate="192k")
    logger.info("Exported MP3: %s", out_path)
    return out_path


def apply_speed(wav_path: str, speed: float, out_path: str = None) -> str:
    """
    Adjust playback speed of a WAV file.
    speed < 1.0 = slower, speed > 1.0 = faster.
    """
    from pydub import AudioSegment

    if abs(speed - 1.0) < 0.01:
        return wav_path

    if out_path is None:
        base, ext = os.path.splitext(wav_path)
        out_path = f"{base}_speed{ext}"

    audio = AudioSegment.from_wav(wav_path)
    # Change frame rate for speed adjustment, then restore sample rate
    adjusted = audio._spawn(
        audio.raw_data,
        overrides={"frame_rate": int(audio.frame_rate * speed)}
    ).set_frame_rate(audio.frame_rate)
    adjusted.export(out_path, format="wav")
    return out_path


def apply_volume(wav_path: str, volume_db: float, out_path: str = None) -> str:
    """Adjust volume of a WAV file by the given dB amount."""
    from pydub import AudioSegment

    if abs(volume_db) < 0.1:
        return wav_path

    if out_path is None:
        base, ext = os.path.splitext(wav_path)
        out_path = f"{base}_vol{ext}"

    audio = AudioSegment.from_wav(wav_path)
    adjusted = audio + volume_db
    adjusted.export(out_path, format="wav")
    return out_path


# ---------------------------------------------------------------------------
# Document export
# ---------------------------------------------------------------------------

def export_txt(text: str, path: str) -> str:
    """Save text to a .txt file."""
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    logger.info("Exported TXT: %s", path)
    return path


def export_docx(text: str, path: str) -> str:
    """Save text to a .docx file."""
    doc = DocxDocument()
    for para in text.split("\n"):
        doc.add_paragraph(para)
    doc.save(path)
    logger.info("Exported DOCX: %s", path)
    return path


def export_pdf(text: str, path: str) -> str:
    """Save text to a .pdf file using ReportLab."""
    doc = SimpleDocTemplate(path, pagesize=letter)
    styles = getSampleStyleSheet()
    story = []
    for para in text.split("\n"):
        if para.strip():
            story.append(Paragraph(para, styles["Normal"]))
            story.append(Spacer(1, 6))
    if not story:
        story.append(Paragraph("(empty)", styles["Normal"]))
    doc.build(story)
    logger.info("Exported PDF: %s", path)
    return path


def export_epub(text: str, path: str, title: str = "Document") -> str:
    """Save text to an EPUB file using ebooklib."""
    try:
        from ebooklib import epub
    except ImportError:
        raise ImportError("ebooklib is required for EPUB export.\nInstall it with: pip install ebooklib")

    book = epub.EpubBook()
    book.set_title(title)
    book.set_language("en")

    # Split on double newlines to create paragraphs
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    if not paragraphs:
        paragraphs = [text.strip() or "(empty)"]

    html_body = "\n".join(f"<p>{p}</p>" for p in paragraphs)
    chapter = epub.EpubHtml(title=title, file_name="content.xhtml", lang="en")
    chapter.content = (
        "<?xml version='1.0' encoding='utf-8'?>"
        "<html xmlns='http://www.w3.org/1999/xhtml'>"
        f"<head><title>{title}</title></head>"
        f"<body>{html_body}</body></html>"
    )

    book.add_item(chapter)
    book.add_item(epub.EpubNcx())
    book.add_item(epub.EpubNav())
    book.spine = ["nav", chapter]

    epub.write_epub(path, book)
    logger.info("Exported EPUB: %s", path)
    return path


# ---------------------------------------------------------------------------
# System monitoring
# ---------------------------------------------------------------------------

def get_ram_usage() -> dict:
    """
    Return RAM usage info for the current process and system.

    Returns dict with keys:
      - process_mb: float — RAM used by this process
      - system_percent: float — system-wide RAM usage percentage
      - system_used_gb: float — system RAM used in GB
      - system_total_gb: float — total system RAM in GB
    """
    proc = psutil.Process(os.getpid())
    mem_info = proc.memory_info()
    sys_mem = psutil.virtual_memory()
    return {
        "process_mb": mem_info.rss / (1024 * 1024),
        "system_percent": sys_mem.percent,
        "system_used_gb": sys_mem.used / (1024 ** 3),
        "system_total_gb": sys_mem.total / (1024 ** 3),
    }

def get_vram_usage() -> dict:
    """
    Return system-wide VRAM usage if a CUDA GPU is available.
    
    Returns array or None:
      - used_gb: float — system-wide GPU VRAM used in GB
      - total_gb: float — GPU VRAM total in GB
      - percent: float — VRAM usage percentage
    """
    try:
        import torch
        if torch.cuda.is_available():
            # torch.cuda.mem_get_info() returns (free, total) for the system
            free, total = torch.cuda.mem_get_info()
            used = total - free
            return {
                "used_gb": used / (1024**3),
                "total_gb": total / (1024**3),
                "percent": (used / total) * 100 if total > 0 else 0.0,
            }
    except Exception:
        pass
    return None
