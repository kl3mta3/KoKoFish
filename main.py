"""
KoKoFish — Main entry point.

Launches the splash screen, initializes engines, and shows the main UI.
"""

import logging
import os
import sys
import threading
import time

# Reduce VRAM fragmentation. The native caching allocator + expandable_segments
# is compatible with torch.compile / Triton; `cudaMallocAsync` is NOT (it lacks
# checkPoolLiveAllocations and blows up during graph capture).
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

# Prevent huggingface_hub from storing/reading tokens via Windows Credential
# Manager (keyring). The HF token is stored in settings.json instead.
os.environ["HF_HUB_DISABLE_CREDENTIAL_STORAGE"] = "1"

# torch.compile / TorchDynamo is opt-in. When disabled (default), we set the
# env vars before torch is imported so VoxCPM / OmniVoice skip the compile
# path entirely — avoiding the 10-min first-run compile and the compiler
# subprocess storm on Windows without a full build toolchain. MUST be set
# before torch is imported anywhere.
try:
    import json as _json
    _settings_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "settings.json")
    if os.path.isfile(_settings_path):
        with open(_settings_path, "r", encoding="utf-8") as _sf:
            _compile_on = bool(_json.load(_sf).get("torch_compile_enabled", False))
    else:
        _compile_on = False
except Exception:
    _compile_on = False
if not _compile_on:
    os.environ["TORCHDYNAMO_DISABLE"] = "1"
    os.environ["TORCH_COMPILE_DISABLE"] = "1"

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
# Localisation — load language preference early, before any UI strings
# ---------------------------------------------------------------------------
from lang import t, load_language
load_language()

# Fish-Speech is no longer bundled — VoxCPM and OmniVoice install on demand
# via pip + HuggingFace cache. Nothing to put on sys.path here.

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
    get_device,
    VALID_ENGINES,
    engine_label,
)
from utils import (
    setup_ffmpeg,
    setup_python_deps,
    is_kokoro_ready,
    setup_kokoro,
    is_voxcpm_ready,
    setup_voxcpm,
    is_omnivoice_ready,
    setup_omnivoice,
    VOXCPM_VARIANTS,
)
from kokoro_engine import KokoroEngine
from voxcpm_engine import VoxCPMEngine
from omnivoice_engine import OmniVoiceEngine
from stt_engine import STTEngine
from voice_manager import VoiceManager
from ui import KoKoFishUI, COLORS, FONT_FAMILY


def _instantiate_engine(engine_id: str, use_cuda: bool):
    """Factory: return a fresh engine instance for the given engine_id."""
    if engine_id == "kokoro":
        return KokoroEngine(use_cuda=use_cuda)
    if engine_id == "voxcpm_05b":
        return VoxCPMEngine(variant="0.5B", use_cuda=use_cuda)
    if engine_id == "voxcpm_2b":
        return VoxCPMEngine(variant="2B", use_cuda=use_cuda)
    if engine_id == "omnivoice":
        return OmniVoiceEngine(use_cuda=use_cuda)
    logger.warning("Unknown engine_id %r — falling back to Kokoro.", engine_id)
    return KokoroEngine(use_cuda=use_cuda)


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
            text=t("LAUNCHER_SUBTITLE"),
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
            text=t("MAIN_SPLASH_INITIALIZING"),
            font=(FONT_FAMILY, 11),
            text_color="#5a5a7a",
        )
        self.status.pack(pady=(0, 20))

        # Version
        ctk.CTkLabel(
            frame,
            text="v1.4.7",
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
        # Sync llm_model into the Settings object so Settings.save() never
        # overwrites the user's choice back to "" when other settings change.
        try:
            from tag_suggester import get_active_llm_key
            self.settings.llm_model = get_active_llm_key()
        except Exception:
            pass
        self.tts = None
        self.stt: STTEngine = None
        self.voice_manager: VoiceManager = None
        self.ui: KoKoFishUI = None

        # Ensure onnxruntime package matches CUDA setting on startup.
        # Runs in background so it never delays launch.
        if getattr(self.settings, "engine", "kokoro") == "kokoro":
            threading.Thread(target=self._ensure_ort_package, daemon=True,
                             name="OrtStartupCheck").start()

    def _ensure_ort_package(self):
        """Background: swap onnxruntime ↔ onnxruntime-gpu if it doesn't match use_cuda."""
        use_cuda = getattr(self.settings, "use_cuda", False)
        try:
            from importlib.metadata import version as _pkg_ver, PackageNotFoundError as _PNF
            gpu_installed = True
            try:
                _pkg_ver("onnxruntime-gpu")
            except _PNF:
                gpu_installed = False

            if use_cuda and not gpu_installed:
                logger.info("CUDA enabled but onnxruntime-gpu not installed — swapping now.")
                from kokoro_engine import switch_onnxruntime
                switch_onnxruntime(True)
            elif not use_cuda and gpu_installed:
                logger.info("CUDA disabled but onnxruntime-gpu is installed — swapping to CPU.")
                from kokoro_engine import switch_onnxruntime
                switch_onnxruntime(False)
            else:
                logger.info("onnxruntime package matches CUDA setting (gpu=%s, use_cuda=%s).",
                            gpu_installed, use_cuda)
        except Exception as exc:
            logger.warning("OrtStartupCheck failed: %s", exc)

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
        ctk.CTkLabel(splash_container, text=t("LAUNCHER_SUBTITLE"),
                     font=(FONT_FAMILY, 13), text_color="#9a9ab0").pack(pady=(0, 25))

        splash_progress = ctk.CTkProgressBar(
            splash_container, progress_color="#4361ee", fg_color="#16213e",
            height=6, corner_radius=3, width=360,
        )
        splash_progress.pack(pady=(0, 10))
        splash_progress.set(0)

        splash_status = ctk.CTkLabel(
            splash_container, text=t("MAIN_SPLASH_INITIALIZING"),
            font=(FONT_FAMILY, 11), text_color="#5a5a7a",
        )
        splash_status.pack(pady=(0, 20))

        ctk.CTkLabel(splash_container, text="v1.4.7", font=(FONT_FAMILY, 10),
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
                update_splash(t("MAIN_SPLASH_CHECKING_FFMPEG"), 0.03)
                setup_ffmpeg(on_progress=update_splash)
                update_splash(t("MAIN_SPLASH_FFMPEG_READY"), 0.07)

                # ── Python runtime deps (noisereduce, scipy, …) ──────────────
                update_splash(t("MAIN_SPLASH_CHECKING_DEPS"), 0.08)
                setup_python_deps(on_progress=update_splash)
                update_splash("Dependencies ready", 0.09)

                _engine = getattr(self.settings, 'engine', 'kokoro')
                if _engine not in VALID_ENGINES:
                    logger.warning("Unknown engine %r in settings — resetting to kokoro.", _engine)
                    _engine = 'kokoro'
                    self.settings.engine = 'kokoro'
                    self.settings.save()

                _eng_label = engine_label(_engine)

                # ── Engine model check / download ────────────────────────────
                # Kokoro is always installed at first launch (it's the default).
                # VoxCPM / OmniVoice install on first use — but if the user last
                # picked one and the weights are still cached, we just load.
                if _engine == 'kokoro':
                    if not is_kokoro_ready():
                        update_splash(t("MAIN_SPLASH_DOWNLOADING_KOKORO"), 0.10)
                        ok = setup_kokoro(on_progress=update_splash)
                        if ok:
                            update_splash(t("MAIN_SPLASH_KOKORO_DOWNLOADED"), 0.22)
                        else:
                            update_splash(t("MAIN_SPLASH_KOKORO_FAILED"), 0.22)
                    else:
                        update_splash("Kokoro models found", 0.12)
                elif _engine in ("voxcpm_05b", "voxcpm_2b"):
                    _variant = "0.5B" if _engine == "voxcpm_05b" else "2B"
                    update_splash(f"Checking {_eng_label}…", 0.10)
                    if not is_voxcpm_ready(_variant):
                        update_splash(f"Installing {_eng_label}…", 0.12)
                        ok = setup_voxcpm(_variant, on_progress=update_splash)
                        if not ok:
                            update_splash(f"{_eng_label} install failed — see log", 0.22)
                            logger.warning("VoxCPM %s setup failed on startup.", _variant)
                        else:
                            update_splash(f"{_eng_label} ready", 0.22)
                    else:
                        update_splash(f"{_eng_label} ready", 0.18)
                elif _engine == 'omnivoice':
                    update_splash(f"Checking {_eng_label}…", 0.10)
                    if not is_omnivoice_ready():
                        update_splash(f"Installing {_eng_label}…", 0.12)
                        ok = setup_omnivoice(on_progress=update_splash)
                        if not ok:
                            update_splash(f"{_eng_label} install failed — see log", 0.22)
                            logger.warning("OmniVoice setup failed on startup.")
                        else:
                            update_splash(f"{_eng_label} ready", 0.22)
                    else:
                        update_splash(f"{_eng_label} ready", 0.18)

                # ── Voice folders ────────────────────────────────────────────
                update_splash("Setting up voice folders…", 0.23)
                for _subdir in VALID_ENGINES:
                    os.makedirs(os.path.join(VOICES_DIR, _subdir), exist_ok=True)
                voices_subdir = os.path.join(VOICES_DIR, _engine)
                self.voice_manager = VoiceManager(voices_subdir)
                update_splash("Voice profiles loaded", 0.27)

                # ── STT engine (Whisper — can take a moment to init) ─────────
                update_splash(t("MAIN_SPLASH_STARTING_STT"), 0.30)
                _stt_stop = threading.Event()
                _stt_pulse = threading.Thread(
                    target=_pulse,
                    args=(t("MAIN_SPLASH_STARTING_STT"), 0.30, 0.44, _stt_stop),
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
                update_splash(t("MAIN_SPLASH_STT_READY"), 0.45)

                # ── TTS engine (can take several seconds) ─
                _engine_type = getattr(self.settings, 'engine', 'kokoro')
                if _engine_type == 'kokoro':
                    _loading_msg = t("MAIN_SPLASH_LOADING_KOKORO")
                    _ready_msg = t("MAIN_SPLASH_KOKORO_READY")
                else:
                    _tts_label = engine_label(_engine_type)
                    _loading_msg = t("MAIN_SPLASH_LOADING_ENGINE", engine_name=_tts_label)
                    _ready_msg = t("MAIN_SPLASH_ENGINE_READY", engine_name=_tts_label)
                update_splash(_loading_msg, 0.48)
                _tts_stop = threading.Event()
                _tts_pulse = threading.Thread(
                    target=_pulse,
                    args=(_loading_msg, 0.48, 0.62, _tts_stop),
                    daemon=True,
                )
                _tts_pulse.start()
                self.tts = _instantiate_engine(_engine_type, self.settings.use_cuda)
                _tts_stop.set()
                _tts_pulse.join()
                update_splash(_ready_msg, 0.63)

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
                    update_splash(t("MAIN_SPLASH_INSTALLING_AI"), 0.66)
                    _llama_done = threading.Event()
                    _llama_ok = [False]

                    def _on_llama_complete(ok, msg, _ev=_llama_done, _res=_llama_ok):
                        _res[0] = ok
                        _ev.set()

                    _install_llama(on_complete=_on_llama_complete)
                    _llama_done.wait()
                    if _llama_ok[0]:
                        logger.info("llama-cpp-python installed successfully.")
                        update_splash(t("MAIN_SPLASH_AI_INSTALLED"), 0.70)
                    else:
                        logger.warning("llama-cpp-python install failed; AI features unavailable.")
                        update_splash(t("MAIN_SPLASH_AI_FAILED"), 0.70)
                else:
                    update_splash(t("MAIN_SPLASH_AI_INSTALLED"), 0.68)

                if _llm_avail() and not _qwen_ready():
                    update_splash(t("MAIN_SPLASH_DOWNLOADING_QWEN"), 0.72)
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
                        update_splash(t("MAIN_SPLASH_QWEN_READY"), 0.94)
                    else:
                        logger.warning("Qwen model download failed; AI features unavailable.")
                        update_splash(t("MAIN_SPLASH_QWEN_FAILED"), 0.94)
                elif _llm_avail():
                    update_splash(t("MAIN_SPLASH_QWEN_READY"), 0.94)

                update_splash(t("MAIN_SPLASH_READY"), 1.0)
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
                f"KoKoFish — {t('MAIN_STARTUP_ERROR_TITLE')}",
                t("MAIN_STARTUP_ERROR_BODY", error=init_error[0])
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
        _engine_ready = (
            _active_engine == 'kokoro'
            or (_active_engine == 'voxcpm_05b' and is_voxcpm_ready('0.5B'))
            or (_active_engine == 'voxcpm_2b' and is_voxcpm_ready('2B'))
            or (_active_engine == 'omnivoice' and is_omnivoice_ready())
        )
        if _engine_ready:
            self.ui.update_tts_status(t("MAIN_ENGINE_READY"), COLORS["success"])
        else:
            self.ui.update_tts_status(
                t("MAIN_ENGINE_NOT_DOWNLOADED", label=engine_label(_active_engine)),
                COLORS["warning"],
            )

        self.ui.update_stt_status(t("MAIN_ENGINE_READY"), COLORS["success"])

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
