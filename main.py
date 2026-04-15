"""
KoKoFish — Main entry point.

Launches the splash screen, initializes engines, and shows the main UI.
"""

import logging
import os
import sys
import threading
import time

# Reduce VRAM fragmentation. expandable_segments requires CUDA 11.8+; use
# cudaMallocAsync (available since CUDA 11.4) as a safer cross-version option.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "backend:cudaMallocAsync")

# ---------------------------------------------------------------------------
# Logging setup — do this first before any other imports
# ---------------------------------------------------------------------------
log_handlers = []
if sys.stdout is None:
    # Running under pythonw / no console. Redirect stdout/stderr to log file to prevent crashes.
    log_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "kokofish_error.log")
    f = open(log_path, "a", encoding="utf-8")
    sys.stdout = f
    sys.stderr = f
    log_handlers.append(logging.FileHandler(log_path, encoding="utf-8"))
else:
    # Force UTF-8 on the console stream so Unicode characters in log messages
    # (arrows, emoji, non-Latin script from translations, etc.) never crash the
    # logger on Windows systems whose console uses cp1252.
    import io
    _utf8_stdout = io.TextIOWrapper(
        sys.stdout.buffer, encoding="utf-8", errors="replace", line_buffering=True
    )
    log_handlers.append(logging.StreamHandler(_utf8_stdout))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    handlers=log_handlers,
)
logger = logging.getLogger("KoKoFish")

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
        print("  KoKoFish — Missing Dependencies")
        print("=" * 60)
        print(f"\n  The following packages are not installed:")
        for m in missing:
            print(f"    - {m}")
        print(f"\n  Please run KoKoFish.bat (or KoKoFish.ps1) to")
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
from utils import setup_ffmpeg, is_kokoro_ready, setup_kokoro, is_fish_speech_ready, FISH_ENGINE_CONFIG
from tts_engine import TTSEngine
from kokoro_engine import KokoroEngine
from stt_engine import STTEngine
from voice_manager import VoiceManager
from ui import KoKoFishUI, COLORS, FONT_FAMILY


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
            text="KoKoFish",
            font=(FONT_FAMILY, 32, "bold"),
            text_color="#6c83f7",
        ).pack(pady=(0, 2))

        ctk.CTkLabel(
            frame,
            text="Audiobook Studio",
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

class KoKoFishApp:
    """Main application controller."""

    def __init__(self):
        self.settings = Settings.load()
        self.tts: TTSEngine = None
        self.stt: STTEngine = None
        self.voice_manager: VoiceManager = None
        self.ui: KoKoFishUI = None

    def run(self):
        """Start the application."""
        # Force Windows to treat this as a distinct app so it uses our icon
        # in the taskbar instead of the generic pythonw.exe logo
        if sys.platform == "win32":
            try:
                import ctypes
                myappid = "KoKoFish.LocalAIVoiceStudio.1.0"
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
        ctk.CTkLabel(splash_container, text="KoKoFish", font=(FONT_FAMILY, 32, "bold"),
                     text_color="#6c83f7").pack(pady=(0, 2))
        ctk.CTkLabel(splash_container, text="Audiobook Studio",
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

        import queue as _queue
        _splash_q = _queue.Queue()

        def update_splash(status, progress):
            """Push a splash update from any thread — never touches tkinter directly."""
            try:
                _splash_q.put_nowait((status, progress))
            except Exception:
                pass

        def _pulse(label: str, from_val: float, to_val: float,
                   stop_evt: threading.Event, interval: float = 0.22):
            """Slowly push progress updates while a slow operation runs."""
            val = from_val
            step = (to_val - from_val) / 14
            while not stop_evt.is_set() and val < to_val - step:
                val += step
                update_splash(label, val)
                stop_evt.wait(interval)

        main_root.update()

        # --- Phase 2: Background initialization ---
        init_done = threading.Event()
        init_error = [None]

        def _init():
            try:
                # ── CPU thread limit ─────────────────────────────────────────
                _cpu_threads = getattr(self.settings, "cpu_threads", 0)
                if _cpu_threads > 0:
                    try:
                        import torch as _torch
                        _torch.set_num_threads(_cpu_threads)
                        _torch.set_num_interop_threads(max(1, _cpu_threads // 2))
                        logger.info("CPU threads limited to %d", _cpu_threads)
                    except Exception as _te:
                        logger.warning("Could not set CPU thread limit: %s", _te)

                # ── FFmpeg ───────────────────────────────────────────────────
                update_splash("Checking FFmpeg…", 0.03)
                setup_ffmpeg(on_progress=update_splash)
                update_splash("FFmpeg ready", 0.07)

                _engine = getattr(self.settings, 'engine', 'kokoro')

                # ── Engine model check / download ────────────────────────────
                if _engine == 'kokoro':
                    if not is_kokoro_ready():
                        update_splash("Downloading Kokoro models (~330 MB)…", 0.10)
                        ok = setup_kokoro(on_progress=update_splash)
                        if ok:
                            logger.info("Kokoro models downloaded successfully.")
                            update_splash("Kokoro models downloaded", 0.22)
                        else:
                            logger.warning("Kokoro model download failed; engine may not load.")
                            update_splash("Kokoro download failed — will retry on use", 0.22)
                    else:
                        update_splash("Kokoro models found", 0.12)
                        logger.info("Engine: Kokoro (models present)")
                else:
                    # OpenAudio engines (s1mini, s1) use fish-speech-latest; Fish14 uses fish-speech
                    _code_dir  = "fish-speech-latest" if _engine in ("s1mini", "s1") else "fish-speech"
                    _fs_path   = os.path.join(APP_DIR, _code_dir)
                    _ckpt_name = FISH_ENGINE_CONFIG.get(_engine, FISH_ENGINE_CONFIG["fish14"])[0]
                    _eng_labels = {"fish14": "Fish-Speech 1.4", "s1mini": "S1 Mini", "s1": "S1 Full"}
                    _eng_label  = _eng_labels.get(_engine, _engine)
                    update_splash(f"Checking {_eng_label} models…", 0.10)

                    if is_fish_speech_ready(_engine):
                        self.settings.fish_speech_path = _fs_path
                        self.settings.checkpoint_name  = f"checkpoints/{_ckpt_name}"
                        self.settings.save()
                        update_splash(f"{_eng_label} models found", 0.18)
                        logger.info("Engine: %s (checkpoints present)", _engine)
                    else:
                        update_splash(f"{_eng_label} not downloaded — select in Settings to get it", 0.18)
                        logger.info("Engine: %s — not yet downloaded (first-use download).", _engine)

                # ── Voice folders ────────────────────────────────────────────
                update_splash("Setting up voice folders…", 0.23)
                for _subdir in ("kokoro", "fish14", "s1mini", "s1"):
                    os.makedirs(os.path.join(VOICES_DIR, _subdir), exist_ok=True)
                _eng = getattr(self.settings, 'engine', 'kokoro')
                voices_subdir = os.path.join(VOICES_DIR, _eng)
                self.voice_manager = VoiceManager(voices_subdir)
                update_splash("Voice profiles loaded", 0.27)

                # ── STT engine (Whisper — can take a moment to init) ─────────
                update_splash("Starting speech recognition…", 0.30)
                _stt_stop = threading.Event()
                _stt_pulse = threading.Thread(
                    target=_pulse,
                    args=("Starting speech recognition…", 0.30, 0.44, _stt_stop),
                    daemon=True,
                )
                _stt_pulse.start()
                device = get_device(self.settings)
                compute_type = "float16" if device == "cuda" else "int8"
                self.stt = STTEngine(
                    model_size=self.settings.whisper_model_size,
                    device=device,
                    compute_type=compute_type,
                )
                _stt_stop.set()
                _stt_pulse.join()
                update_splash("Speech recognition ready", 0.45)

                # ── TTS engine (ONNX / Fish-Speech — can take several seconds) ─
                _engine_type = getattr(self.settings, 'engine', 'fish14')
                if _engine_type == 'kokoro':
                    update_splash("Loading Kokoro voice engine…", 0.48)
                    _tts_stop = threading.Event()
                    _tts_pulse = threading.Thread(
                        target=_pulse,
                        args=("Loading Kokoro voice engine…", 0.48, 0.62, _tts_stop),
                        daemon=True,
                    )
                    _tts_pulse.start()
                    self.tts = KokoroEngine()
                    _tts_stop.set()
                    _tts_pulse.join()
                    update_splash("Kokoro engine ready", 0.63)
                else:
                    _tts_label = _eng_labels.get(_engine_type, "TTS")
                    update_splash(f"Loading {_tts_label} engine…", 0.48)
                    _tts_stop = threading.Event()
                    _tts_pulse = threading.Thread(
                        target=_pulse,
                        args=(f"Loading {_tts_label} engine…", 0.48, 0.62, _tts_stop),
                        daemon=True,
                    )
                    _tts_pulse.start()
                    self.tts = TTSEngine(
                        fish_speech_path=self.settings.fish_speech_path or get_bundled_fish_speech_path(),
                        device=device,
                        checkpoint_name=self.settings.checkpoint_name,
                    )
                    _tts_stop.set()
                    _tts_pulse.join()
                    update_splash(f"{_tts_label} engine ready", 0.63)

                # ── AI features (llama-cpp-python + Qwen) ───────────────────
                from tag_suggester import (
                    is_llm_available as _llm_avail,
                    is_qwen_model_ready as _qwen_ready,
                    install_llama_cpp as _install_llama,
                    download_qwen_model as _dl_qwen,
                    set_llm_gpu_mode as _set_llm_gpu,
                )
                # Allow LLM to use GPU when CUDA is enabled.
                # S1/S1mini TTS will call unload_llm() before generation to free VRAM.
                _set_llm_gpu(bool(getattr(self.settings, "use_cuda", False)))

                if not _llm_avail():
                    update_splash("Installing AI features (~60 MB)…", 0.66)
                    _llama_done = threading.Event()
                    _llama_ok = [False]

                    def _on_llama_complete(ok, msg, _ev=_llama_done, _res=_llama_ok):
                        _res[0] = ok
                        _ev.set()

                    _install_llama(on_complete=_on_llama_complete)
                    _llama_done.wait()
                    if _llama_ok[0]:
                        logger.info("llama-cpp-python installed successfully.")
                        update_splash("AI features installed", 0.70)
                    else:
                        logger.warning("llama-cpp-python install failed; AI features unavailable.")
                        update_splash("AI install failed — Assisted Flow unavailable", 0.70)
                else:
                    update_splash("AI features ready", 0.68)

                if _llm_avail() and not _qwen_ready():
                    update_splash("Downloading Qwen AI model (~400 MB)…", 0.72)
                    _qwen_done = threading.Event()
                    _qwen_ok = [False]

                    def _on_qwen_progress(status, frac):
                        update_splash(status, 0.72 + frac * 0.22)

                    def _on_qwen_complete(ok, msg, _ev=_qwen_done, _res=_qwen_ok):
                        _res[0] = ok
                        _ev.set()

                    _dl_qwen(on_progress=_on_qwen_progress, on_complete=_on_qwen_complete)
                    _qwen_done.wait()
                    if _qwen_ok[0]:
                        logger.info("Qwen model downloaded successfully.")
                        update_splash("Qwen model ready", 0.94)
                    else:
                        logger.warning("Qwen model download failed; AI features unavailable.")
                        update_splash("Qwen download failed — AI features unavailable", 0.94)
                elif _llm_avail():
                    update_splash("Qwen model ready", 0.94)

                update_splash("Ready!", 1.0)
                time.sleep(0.4)

            except Exception as exc:
                logger.error("Initialization failed: %s", exc, exc_info=True)
                init_error[0] = exc
            finally:
                init_done.set()

        init_thread = threading.Thread(target=_init, daemon=True, name="Init")
        init_thread.start()

        # Wait for init — drain the splash queue and keep tkinter responsive
        while not init_done.is_set():
            # Apply every pending splash update (show each one briefly)
            try:
                while True:
                    _s, _p = _splash_q.get_nowait()
                    splash_status.configure(text=_s)
                    splash_progress.set(_p)
                    main_root.update()
                    time.sleep(0.08)   # pause so each message is readable
            except _queue.Empty:
                pass
            main_root.update()
            time.sleep(0.03)

        if init_error[0]:
            logger.error("Fatal initialization error: %s", init_error[0])
            import tkinter.messagebox as mb
            mb.showerror(
                "KoKoFish — Startup Error",
                f"An error occurred during initialization:\n\n{init_error[0]}\n\n"
                "The app will launch but some features may not work."
            )

        # --- Phase 3: Transform splash into main window ---
        splash_container.destroy()
        
        # Windows requires withdraw/deiconify when toggling overrideredirect
        main_root.withdraw()
        main_root.overrideredirect(False)
        main_root.attributes("-topmost", False)
        main_root.title("KoKoFish — Audiobook Studio")
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
        self.ui = KoKoFishUI(
            root=main_root,
            settings=self.settings,
            tts_engine=self.tts,
            stt_engine=self.stt,
            voice_manager=self.voice_manager,
        )

        # Update engine status labels
        _active_engine = getattr(self.settings, 'engine', 'kokoro')
        if _active_engine == 'kokoro' or is_fish_speech_ready(_active_engine):
            self.ui.update_tts_status("✅  Engine ready (model loads on first use)", COLORS["success"])
        else:
            _eng_labels = {"fish14": "Fish-Speech 1.4", "s1mini": "S1 Mini", "s1": "S1"}
            _label = _eng_labels.get(_active_engine, _active_engine)
            self.ui.update_tts_status(
                f"⬇  {_label} not downloaded — select it in Settings to download",
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

        # Windows DPI / multi-monitor fix: force a redraw whenever the window
        # is moved or resized.  Without this, moving to a monitor with a
        # different DPI scaling causes the background to go transparent.
        _last_pos = [None]
        def _on_configure(event):
            if event.widget is not main_root:
                return
            pos = (event.x, event.y)
            if pos != _last_pos[0]:
                _last_pos[0] = pos
                main_root.configure(fg_color=COLORS["bg_dark"])
                main_root.update_idletasks()

        main_root.bind("<Configure>", _on_configure)

        logger.info("KoKoFish is ready!")
        main_root.mainloop()


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    app = KoKoFishApp()
    app.run()
