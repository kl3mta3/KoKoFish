"""
FishTalk — Main entry point.

Launches the splash screen, initializes engines, and shows the main UI.
"""

import logging
import os
import sys
import threading
import time

# ---------------------------------------------------------------------------
# Logging setup — do this first before any other imports
# ---------------------------------------------------------------------------
log_handlers = []
if sys.stdout is None:
    # Running under pythonw / no console. Redirect stdout/stderr to log file to prevent crashes.
    log_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fishtalk_error.log")
    f = open(log_path, "a")
    sys.stdout = f
    sys.stderr = f
    log_handlers.append(logging.FileHandler(log_path))
else:
    log_handlers.append(logging.StreamHandler(sys.stdout))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    handlers=log_handlers,
)
logger = logging.getLogger("FishTalk")

# ---------------------------------------------------------------------------
# App directory setup
# ---------------------------------------------------------------------------
APP_DIR = os.path.dirname(os.path.abspath(__file__))
VOICES_DIR = os.path.join(APP_DIR, "voices")
os.makedirs(VOICES_DIR, exist_ok=True)
os.makedirs(os.path.join(APP_DIR, "bin"), exist_ok=True)

# ---------------------------------------------------------------------------
# Ensure fish-speech is importable
# ---------------------------------------------------------------------------
import json as _json
_settings_path = os.path.join(APP_DIR, "settings.json")
# Default fallback: the original fish-speech folder (1.4 code lives here)
FISH_SPEECH_DIR = os.path.join(APP_DIR, "fish-speech")
if os.path.isfile(_settings_path):
    try:
        with open(_settings_path, "r", encoding="utf-8") as _f:
            _sdata = _json.load(_f)
            _saved_path = _sdata.get("fish_speech_path", "")
            # Only use the saved path if the directory actually exists on disk
            if _saved_path and os.path.isdir(_saved_path):
                FISH_SPEECH_DIR = _saved_path
    except Exception:
        pass

if os.path.isdir(FISH_SPEECH_DIR) and FISH_SPEECH_DIR not in sys.path:
    sys.path.insert(0, FISH_SPEECH_DIR)
logger.info("Fish-Speech framework loaded from: %s", FISH_SPEECH_DIR)

# ---------------------------------------------------------------------------
# Pre-flight dependency check
# ---------------------------------------------------------------------------
def _check_dependencies():
    """Verify critical dependencies are installed before proceeding."""
    missing = []
    for mod in ["customtkinter", "psutil", "pdfplumber", "docx", "soundfile"]:
        try:
            __import__(mod)
        except ImportError:
            missing.append(mod)

    if missing:
        print("\n" + "=" * 60)
        print("  FishTalk — Missing Dependencies")
        print("=" * 60)
        print(f"\n  The following packages are not installed:")
        for m in missing:
            print(f"    - {m}")
        print(f"\n  Please run FishTalk.bat (or FishTalk.ps1) to")
        print(f"  automatically install everything.")
        print(f"\n  Or install manually:")
        print(f"    pip install -r requirements.txt")
        print("=" * 60 + "\n")
        sys.exit(1)

_check_dependencies()

# ---------------------------------------------------------------------------
# GUI imports
# ---------------------------------------------------------------------------
import customtkinter as ctk

# Try to enable drag-and-drop
DND_AVAILABLE = False
try:
    from tkinterdnd2 import TkinterDnD
    DND_AVAILABLE = True
except ImportError:
    logger.warning("tkinterdnd2 not available — drag-and-drop disabled.")

from settings import (
    Settings,
    detect_cuda,
    get_bundled_fish_speech_path,
    validate_fish_speech_path,
    get_device,
)
from utils import setup_ffmpeg, is_kokoro_ready, setup_kokoro
from tts_engine import TTSEngine
from kokoro_engine import KokoroEngine
from stt_engine import STTEngine
from voice_manager import VoiceManager
from ui import FishTalkUI, COLORS, FONT_FAMILY


# ============================================================================
# SPLASH SCREEN
# ============================================================================

class SplashScreen:
    """Borderless splash/loading screen shown during startup."""

    def __init__(self, root):
        self.root = root
        self.root.overrideredirect(True)
        self.root.configure(fg_color="#0a0a18")

        # Center on screen
        w, h = 520, 340
        sw = self.root.winfo_screenwidth()
        sh = self.root.winfo_screenheight()
        x = (sw - w) // 2
        y = (sh - h) // 2
        self.root.geometry(f"{w}x{h}+{x}+{y}")

        # Keep on top
        self.root.attributes("-topmost", True)

        # Content
        frame = ctk.CTkFrame(
            self.root,
            fg_color="#0f0f1a",
            corner_radius=16,
            border_color="#2a2a4a",
            border_width=2,
        )
        frame.pack(fill="both", expand=True, padx=2, pady=2)

        # Logo / title
        ctk.CTkLabel(
            frame,
            text="🐟",
            font=(FONT_FAMILY, 56),
        ).pack(pady=(40, 5))

        ctk.CTkLabel(
            frame,
            text="FishTalk",
            font=(FONT_FAMILY, 32, "bold"),
            text_color="#6c83f7",
        ).pack(pady=(0, 2))

        ctk.CTkLabel(
            frame,
            text="TTS/STT Voice Studio",
            font=(FONT_FAMILY, 13),
            text_color="#9a9ab0",
        ).pack(pady=(0, 25))

        # Progress bar
        self.progress = ctk.CTkProgressBar(
            frame,
            progress_color="#4361ee",
            fg_color="#16213e",
            height=6,
            corner_radius=3,
            width=360,
        )
        self.progress.pack(pady=(0, 10))
        self.progress.set(0)

        # Status label
        self.status = ctk.CTkLabel(
            frame,
            text="Initializing...",
            font=(FONT_FAMILY, 11),
            text_color="#5a5a7a",
        )
        self.status.pack(pady=(0, 20))

        # Version
        ctk.CTkLabel(
            frame,
            text="v1.4.5",
            font=(FONT_FAMILY, 10),
            text_color="#3a3a5a",
        ).pack(side="bottom", pady=10)

    def update(self, status: str, progress: float):
        """Update splash screen status (thread-safe)."""
        try:
            self.status.configure(text=status)
            self.progress.set(progress)
            self.root.update_idletasks()
        except Exception:
            pass  # Window may have been destroyed

    def destroy(self):
        try:
            self.root.destroy()
        except Exception:
            pass


# ============================================================================
# APP BOOTSTRAP
# ============================================================================

class FishTalkApp:
    """Main application controller."""

    def __init__(self):
        self.settings = Settings.load()
        self.tts: TTSEngine = None
        self.stt: STTEngine = None
        self.voice_manager: VoiceManager = None
        self.ui: FishTalkUI = None

    def run(self):
        """Start the application."""
        # Force Windows to treat this as a distinct app so it uses our icon
        # in the taskbar instead of the generic pythonw.exe logo
        if sys.platform == "win32":
            try:
                import ctypes
                myappid = "Fishtalk.LocalAIVoiceStudio.1.0"
                ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(myappid)
            except Exception:
                pass

        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")

        # --- Phase 1: Create main root (with DnD if available) ---
        if DND_AVAILABLE:
            class DnDRoot(ctk.CTk, TkinterDnD.DnDWrapper):
                def __init__(self, *args, **kwargs):
                    super().__init__(*args, **kwargs)
                    self.TkdndVersion = TkinterDnD._require(self)

            try:
                main_root = DnDRoot()
            except Exception:
                logger.warning("DnD root creation failed, falling back to standard root")
                main_root = ctk.CTk()
        else:
            main_root = ctk.CTk()

        # Configure as splash screen initially
        main_root.overrideredirect(True)
        main_root.configure(fg_color="#0a0a18")

        w, h = 520, 340
        sw = main_root.winfo_screenwidth()
        sh = main_root.winfo_screenheight()
        x = (sw - w) // 2
        y = (sh - h) // 2
        main_root.geometry(f"{w}x{h}+{x}+{y}")
        main_root.attributes("-topmost", True)

        # Build splash content in a container we can remove later
        splash_container = ctk.CTkFrame(
            main_root, fg_color="#0f0f1a", corner_radius=16,
            border_color="#2a2a4a", border_width=2,
        )
        splash_container.pack(fill="both", expand=True, padx=2, pady=2)

        ctk.CTkLabel(splash_container, text="🐟", font=(FONT_FAMILY, 56)).pack(pady=(40, 5))
        ctk.CTkLabel(splash_container, text="FishTalk", font=(FONT_FAMILY, 32, "bold"),
                     text_color="#6c83f7").pack(pady=(0, 2))
        ctk.CTkLabel(splash_container, text="TTS/STT Studio",
                     font=(FONT_FAMILY, 13), text_color="#9a9ab0").pack(pady=(0, 25))

        splash_progress = ctk.CTkProgressBar(
            splash_container, progress_color="#4361ee", fg_color="#16213e",
            height=6, corner_radius=3, width=360,
        )
        splash_progress.pack(pady=(0, 10))
        splash_progress.set(0)

        splash_status = ctk.CTkLabel(
            splash_container, text="Initializing...",
            font=(FONT_FAMILY, 11), text_color="#5a5a7a",
        )
        splash_status.pack(pady=(0, 20))

        ctk.CTkLabel(splash_container, text="v1.4.5", font=(FONT_FAMILY, 10),
                     text_color="#3a3a5a").pack(side="bottom", pady=10)

        def update_splash(status, progress):
            try:
                splash_status.configure(text=status)
                splash_progress.set(progress)
                main_root.update_idletasks()
            except Exception:
                pass

        main_root.update()

        # --- Phase 2: Background initialization ---
        init_done = threading.Event()
        init_error = [None]

        def _init():
            try:
                update_splash("Checking FFmpeg...", 0.1)
                setup_ffmpeg(on_progress=update_splash)

                _engine = getattr(self.settings, 'engine', 'kokoro')

                if _engine == 'kokoro':
                    if not is_kokoro_ready():
                        update_splash("Downloading Kokoro models (~330 MB)…", 0.15)
                        ok = setup_kokoro(on_progress=update_splash)
                        if ok:
                            logger.info("Kokoro models downloaded successfully.")
                        else:
                            logger.warning("Kokoro model download failed; engine may not load.")
                    else:
                        update_splash("Kokoro engine ready.", 0.2)
                        logger.info("Engine: Kokoro (models present)")
                else:
                    # Fish Speech — only validate/download the selected version.
                    _fs_version = "1.5" if _engine == "fish15" else "1.4"
                    update_splash(f"Validating Fish-Speech {_fs_version}...", 0.15)

                    bundled_path = os.path.join(APP_DIR, "fish-speech" if _fs_version == "1.4" else "fish-speech-1.5")
                    saved_path = self.settings.fish_speech_path

                    if saved_path and os.path.isdir(saved_path):
                        fs_path = saved_path
                    elif os.path.isdir(bundled_path) and os.listdir(bundled_path):
                        fs_path = bundled_path
                        self.settings.fish_speech_path = fs_path
                        self.settings.save()
                        logger.info("Auto-resolved Fish-Speech path: %s", fs_path)
                    else:
                        # Not present — download now (user deliberately chose this engine)
                        update_splash(f"Fish-Speech {_fs_version} not found. Downloading (~1.5 GB)...", 0.16)
                        ok = setup_fish_speech(
                            dest_dir=bundled_path,
                            on_progress=update_splash,
                            version=_fs_version,
                        )
                        fs_path = bundled_path
                        if ok:
                            self.settings.fish_speech_path = fs_path
                            self.settings.checkpoint_name = f"checkpoints/fish-speech-{_fs_version}"
                            self.settings.save()
                            logger.info("Fish-Speech %s download complete: %s", _fs_version, fs_path)
                        else:
                            logger.warning("Fish-Speech %s download failed; engine may not work.", _fs_version)

                    result = validate_fish_speech_path(fs_path)
                    if result["valid"]:
                        self.settings.fish_speech_path = fs_path
                        logger.info("Fish-Speech path validated: %s", fs_path)
                    else:
                        logger.warning("Fish-Speech not found at %s: %s", fs_path, result["message"])

                update_splash("Initializing voice manager...", 0.2)
                # Ensure all engine voice subdirectories exist on every launch
                # so switching engines never hits a missing-folder error.
                for _subdir in ("fish14", "fish15", "kokoro"):
                    os.makedirs(os.path.join(VOICES_DIR, _subdir), exist_ok=True)
                _eng = getattr(self.settings, 'engine', 'fish14')
                voices_subdir = os.path.join(VOICES_DIR, _eng)
                self.voice_manager = VoiceManager(voices_subdir)


                update_splash("Initializing STT engine...", 0.3)
                device = get_device(self.settings)
                compute_type = "float16" if device == "cuda" else "int8"
                self.stt = STTEngine(
                    model_size=self.settings.whisper_model_size,
                    device=device,
                    compute_type=compute_type,
                )

                update_splash("Initializing TTS engine...", 0.4)
                _engine_type = getattr(self.settings, 'engine', 'fish14')
                if _engine_type == 'kokoro':
                    update_splash("Booting Kokoro engine...", 0.4)
                    self.tts = KokoroEngine()
                else:
                    update_splash("Booting Fish-Speech engine...", 0.4)
                    self.tts = TTSEngine(
                        fish_speech_path=self.settings.fish_speech_path or get_bundled_fish_speech_path(),
                        device=device,
                        checkpoint_name=self.settings.checkpoint_name,
                    )

                update_splash("Ready!", 1.0)
                time.sleep(0.5)

            except Exception as exc:
                logger.error("Initialization failed: %s", exc, exc_info=True)
                init_error[0] = exc
            finally:
                init_done.set()

        init_thread = threading.Thread(target=_init, daemon=True, name="Init")
        init_thread.start()

        # Wait for init (keep splash responsive)
        while not init_done.is_set():
            main_root.update()
            time.sleep(0.05)

        if init_error[0]:
            logger.error("Fatal initialization error: %s", init_error[0])
            import tkinter.messagebox as mb
            mb.showerror(
                "FishTalk — Startup Error",
                f"An error occurred during initialization:\n\n{init_error[0]}\n\n"
                "The app will launch but some features may not work."
            )

        # --- Phase 3: Transform splash into main window ---
        splash_container.destroy()
        
        # Windows requires withdraw/deiconify when toggling overrideredirect
        main_root.withdraw()
        main_root.overrideredirect(False)
        main_root.attributes("-topmost", False)
        main_root.title("FishTalk — TTS/STT Studio")
        main_root.geometry(self.settings.window_geometry)
        main_root.minsize(1024, 680)
        main_root.configure(fg_color=COLORS["bg_dark"])
        
        # Restore window
        main_root.deiconify()

        # Set window icon using PhotoImage (avoids .ico format rejection bugs)
        try:
            import tkinter as tk
            icon_png_path = os.path.join(APP_DIR, "icon.png")
            if os.path.isfile(icon_png_path):
                icon_img = tk.PhotoImage(file=icon_png_path)
                main_root.iconphoto(False, icon_img)
        except Exception:
            pass

        # Build UI
        self.ui = FishTalkUI(
            root=main_root,
            settings=self.settings,
            tts_engine=self.tts,
            stt_engine=self.stt,
            voice_manager=self.voice_manager,
        )

        # Update engine status labels
        fs_path = self.settings.fish_speech_path or get_bundled_fish_speech_path()
        fs_valid = validate_fish_speech_path(fs_path)
        if fs_valid["valid"]:
            self.ui.update_tts_status("✅  Engine ready (model loads on first use)", COLORS["success"])
        else:
            self.ui.update_tts_status(
                "⚠  Fish-Speech not found — check Settings",
                COLORS["warning"],
            )

        self.ui.update_stt_status("✅  Engine ready (model loads on first use)", COLORS["success"])

        # Save settings on exit
        def on_close():
            self.settings.window_geometry = main_root.geometry()
            self.settings.save()
            # Cleanup
            if self.tts:
                self.tts.cancel()
            if self.stt:
                self.stt.cancel()
            main_root.destroy()

        main_root.protocol("WM_DELETE_WINDOW", on_close)

        logger.info("FishTalk is ready!")
        main_root.mainloop()


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    app = FishTalkApp()
    app.run()
