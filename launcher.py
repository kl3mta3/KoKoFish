"""
KoKoFish — Auto-Launcher & Setup UI.

This file uses ONLY built-in Python libraries (tkinter, os, subprocess)
so it can run on a completely bare Python installation.

It serves as the main entry point:
1. Shows a clean GUI Splash Screen
2. Checks if venv exists
3. (First Run) Creates venv and quietly installs from local packages/
4. Re-launches the main app (main.py) inside the new venv.
"""

# ===========================================================================
# LOGGING — absolute first thing, before any other imports or logic.
# ===========================================================================
import os
import sys
import datetime

if getattr(sys, 'frozen', False):
    APP_DIR = os.path.dirname(sys.executable)
else:
    APP_DIR = os.path.dirname(os.path.abspath(__file__))

INSTALL_LOG = os.path.join(APP_DIR, "installation_log.txt")

def _log(msg: str) -> None:
    """Append a timestamped line to installation_log.txt. Never raises."""
    try:
        with open(INSTALL_LOG, "a", encoding="utf-8") as _f:
            _f.write(f"[{datetime.datetime.now():%Y-%m-%d %H:%M:%S}] {msg}\n")
    except Exception:
        pass

_log("=" * 60)
_log("KoKoFish launcher started")
_log(f"Log       : {INSTALL_LOG}")
_log(f"APP_DIR   : {APP_DIR}")
_log(f"Executable: {sys.executable}")
_log(f"Python    : {sys.version}")
_log(f"Frozen    : {getattr(sys, 'frozen', False)}")

# ===========================================================================
# Rest of imports
# ===========================================================================
import subprocess
import threading
import traceback
import math
import tkinter as tk
import tkinter.ttk as ttk
from tkinter import messagebox
from lang import t, load_language

VENV_DIR     = os.path.join(APP_DIR, "venv")
SETUP_MARKER = os.path.join(APP_DIR, ".setup_complete")
PACKAGES_DIR = os.path.join(APP_DIR, "packages")

_log(f"VENV_DIR    : {VENV_DIR}  (exists={os.path.exists(VENV_DIR)})")
_log(f"SETUP_MARKER: {SETUP_MARKER}  (exists={os.path.exists(SETUP_MARKER)})")
_log(f"PACKAGES_DIR: {PACKAGES_DIR}  (exists={os.path.exists(PACKAGES_DIR)})")

if sys.platform == "win32":
    PYTHON_EXE = os.path.join(VENV_DIR, "Scripts", "pythonw.exe")
    PIP_EXE    = os.path.join(VENV_DIR, "Scripts", "pip.exe")
else:
    PYTHON_EXE = os.path.join(VENV_DIR, "bin", "python")
    PIP_EXE    = os.path.join(VENV_DIR, "bin", "pip")

_log(f"PYTHON_EXE  : {PYTHON_EXE}  (exists={os.path.exists(PYTHON_EXE)})")
_log(f"PIP_EXE     : {PIP_EXE}  (exists={os.path.exists(PIP_EXE)})")


def _run_logged(label: str, cmd: list, **kwargs):
    """Run a subprocess, log the full command and all output."""
    _log(f">>> {label}")
    _log(f"    cmd: {' '.join(str(c) for c in cmd)}")
    r = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        **kwargs,
    )
    for line in (r.stdout or "").strip().splitlines():
        _log(f"    {line}")
    _log(f"    exit code: {r.returncode}")
    return r


def _run_with_progress(label: str, cmd: list, on_line=None, **kwargs):
    """Run a subprocess with real-time line callback for progress tracking."""
    _log(f">>> {label}")
    _log(f"    cmd: {' '.join(str(c) for c in cmd)}")
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        **kwargs,
    )
    for line in proc.stdout:
        line = line.rstrip()
        _log(f"    {line}")
        if on_line:
            on_line(line)
    proc.wait()
    _log(f"    exit code: {proc.returncode}")
    return proc


def launch_main_app():
    """Launch the real application."""
    main_script = os.path.join(APP_DIR, "main.py")
    _log(f"Launching main app: {main_script}")

    flags = 0
    if sys.platform == "win32":
        flags = 0x00000008 | 0x00000200

    env = os.environ.copy()
    env.pop("TCL_LIBRARY", None)
    env.pop("TK_LIBRARY", None)

    subprocess.Popen([PYTHON_EXE, main_script], cwd=APP_DIR, env=env, creationflags=flags)
    sys.exit(0)


# ---------------------------------------------------------------------------
# Count installable lines in requirements.txt for progress calculation
# ---------------------------------------------------------------------------
def _count_requirements(req_file: str) -> int:
    count = 0
    try:
        with open(req_file, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and not line.startswith("-"):
                    count += 1
    except Exception:
        count = 30  # fallback estimate
    return max(count, 1)


class InstallerGUI(tk.Tk):
    # Progress milestones (%)
    _P_VENV    = 5
    _P_PIP     = 10
    _P_TORCH   = 35
    _P_LLAMA   = 45
    _P_REQ_END = 99

    def __init__(self):
        super().__init__()

        self.title("KoKoFish " + t("LAUNCHER_WINDOW_TITLE"))
        self.geometry("450x340")
        self.configure(bg="#0f0f1a")

        self.update_idletasks()
        x = (self.winfo_screenwidth() - 450) // 2
        y = (self.winfo_screenheight() - 340) // 2
        self.geometry(f"+{x}+{y}")
        self.overrideredirect(True)

        # ── Title ──────────────────────────────────────────────────────────
        tk.Label(
            self, text="🐟 KoKoFish", font=("Segoe UI", 28, "bold"),
            bg="#0f0f1a", fg="#6c83f7"
        ).pack(pady=(28, 4))

        tk.Label(
            self, text=t("LAUNCHER_SUBTITLE"), font=("Segoe UI", 11),
            bg="#0f0f1a", fg="#9a9ab0"
        ).pack()

        # ── Main status ────────────────────────────────────────────────────
        self.status_var = tk.StringVar(value=t("LAUNCHER_STATUS_INIT"))
        tk.Label(
            self, textvariable=self.status_var, font=("Segoe UI", 10),
            bg="#0f0f1a", fg="#e8e8f0"
        ).pack(pady=(14, 4))

        # ── Progress bar ───────────────────────────────────────────────────
        style = ttk.Style(self)
        style.theme_use("default")
        style.configure(
            "KoKo.Horizontal.TProgressbar",
            troughcolor="#1a1a2e",
            background="#6c83f7",
            darkcolor="#6c83f7",
            lightcolor="#8a9af9",
            bordercolor="#0f0f1a",
            thickness=10,
        )
        self._progress = tk.DoubleVar(value=0)
        ttk.Progressbar(
            self, style="KoKo.Horizontal.TProgressbar",
            variable=self._progress, maximum=100, length=390
        ).pack(pady=(0, 4))

        # ── Package detail label ───────────────────────────────────────────
        self._detail_var = tk.StringVar(value="")
        tk.Label(
            self, textvariable=self._detail_var, font=("Segoe UI", 8),
            bg="#0f0f1a", fg="#5a5a7a"
        ).pack()

        # ── Bubble animation canvas ────────────────────────────────────────
        self._canvas = tk.Canvas(
            self, width=450, height=60,
            bg="#0f0f1a", highlightthickness=0
        )
        self._canvas.pack(side="bottom", pady=(4, 8))

        # Fish label drawn on canvas
        self._canvas.create_text(
            195, 42, text="🐟", font=("Segoe UI", 18), anchor="e",
            tags="fish"
        )

        # Animation state
        self._anim_step  = 0
        self._anim_id    = None
        self._anim_active = True
        self._tick()

    # ── Animation ─────────────────────────────────────────────────────────
    def _tick(self):
        if not self._anim_active:
            return
        c = self._canvas
        c.delete("bubble")

        CYCLE   = 40          # frames for one bubble to travel top-to-bottom
        OFFSETS = [0, 13, 26] # stagger the 3 bubbles evenly across the cycle
        FISH_X  = 200
        FISH_Y  = 44
        RISE    = 48          # total pixels bubbles travel upward

        s = self._anim_step
        for offset in OFFSETS:
            phase = ((s + offset) % CYCLE) / CYCLE   # 0.0 → 1.0
            x     = FISH_X + 10 + OFFSETS.index(offset) * 7
            y     = FISH_Y - int(phase * RISE)
            r     = 2 + int(phase * 7)               # radius 2 → 9
            # Colour lightens as bubble rises
            blue  = min(255, 0x6c + int(phase * 140))
            green = min(255, 0x83 + int(phase * 80))
            color = f"#{0x6c:02x}{green:02x}{blue:02x}"
            c.create_oval(
                x - r, y - r, x + r, y + r,
                fill=color, outline="", tags="bubble"
            )

        self._anim_step += 1
        self._anim_id = self.after(55, self._tick)

    def _stop_animation(self):
        self._anim_active = False
        if self._anim_id:
            self.after_cancel(self._anim_id)

    # ── Public update helpers ──────────────────────────────────────────────
    def update_status(self, text: str):
        self.status_var.set(text)
        self.update()

    def update_progress(self, pct: float, detail: str = ""):
        self._progress.set(min(pct, 100))
        if detail:
            self._detail_var.set(detail)
        self.update()

    # ── Spawn main app after install ───────────────────────────────────────
    def _spawn_and_linger(self):
        self._stop_animation()
        self._canvas.delete("bubble")
        # Draw a little celebration — full bar
        self._progress.set(100)
        env = os.environ.copy()
        env.pop("TCL_LIBRARY", None)
        env.pop("TK_LIBRARY", None)
        flags = 0x00000008 | 0x00000200 if sys.platform == "win32" else 0
        main_script = os.path.join(APP_DIR, "main.py")
        subprocess.Popen([PYTHON_EXE, main_script], cwd=APP_DIR, env=env, creationflags=flags)
        self.update_status(t("LAUNCHER_LINGER_MSG"))
        self._detail_var.set("")
        self.after(15000, self.destroy)

    # ── Main setup logic ───────────────────────────────────────────────────
    def run_setup(self):
        import shutil
        import urllib.request
        import tempfile

        def _find_valid_python():
            _log("Searching for Python 3.12...")
            candidates = []
            local_app = os.environ.get("LOCALAPPDATA", "")
            for v in ["312", "313", "311"]:
                p = os.path.join(local_app, "Programs", "Python", f"Python{v}", "python.exe")
                _log(f"  Candidate: {p}  exists={os.path.exists(p)}")
                if os.path.exists(p):
                    candidates.append(p)
            sys_py = shutil.which("python")
            _log(f"  which(python): {sys_py}")
            if sys_py:
                candidates.append(sys_py)

            for calc in candidates:
                if not calc or not os.path.exists(calc):
                    continue
                try:
                    out = subprocess.run(
                        [calc, "-c", "import sys; print(sys.version_info[:2])"],
                        capture_output=True, text=True,
                        creationflags=subprocess.CREATE_NO_WINDOW
                    )
                    ver = out.stdout.strip()
                    _log(f"  {calc} -> {ver}")
                    if "(3, 12)" in ver:
                        _log(f"  Selected: {calc}")
                        return calc
                except Exception as ex:
                    _log(f"  Error probing {calc}: {ex}")
            _log("  No valid Python 3.12 found")
            return None

        _log("--- run_setup() called ---")
        system_python = _find_valid_python()
        wants_install  = False

        if not system_python:
            _log("Python 3.12 not found — asking user")
            wants_install = messagebox.askyesno(
                t("LAUNCHER_PYTHON_REQUIRED_TITLE"),
                "KoKoFish requires Python 3.12 which was not found on your system.\n\n"
                "Would you like KoKoFish to download and install Python 3.12 for you right now?\n"
                "(It installs cleanly to your local user folder without requiring Admin privileges)"
            )
            if not wants_install:
                _log("User declined — aborting")
                messagebox.showerror(
                    t("LAUNCHER_SETUP_FAILED_TITLE"),
                    "Python 3.12 is required. Please install it from python.org and try again."
                )
                self.destroy()
                return

        def _worker():
            try:
                nonlocal system_python

                if wants_install:
                    installer_filename = "python-3.12.9-amd64.exe"
                    installer_path = os.path.join(APP_DIR, "bin", installer_filename)
                    _log(f"Bundled installer: {installer_path}  exists={os.path.exists(installer_path)}")

                    if not os.path.exists(installer_path):
                        installer_path = os.path.join(
                            tempfile.gettempdir(), "python-3.12.9-amd64-kokofish.exe"
                        )
                        _log(f"Downloading Python installer to: {installer_path}")
                        self.update_status("Downloading Python 3.12... (This may take a minute)")
                        try:
                            urllib.request.urlretrieve(
                                "https://www.python.org/ftp/python/3.12.9/python-3.12.9-amd64.exe",
                                installer_path
                            )
                            _log("Download complete")
                        except Exception as e:
                            raise Exception(f"Failed to download Python installer: {e}")
                    else:
                        self.update_status("Installing Python 3.12...")

                    self.update_status("Installing Python in the background...")
                    _log("Running Python installer...")
                    subprocess.run(
                        [installer_path,
                         "/passive", "InstallAllUsers=0", "PrependPath=1",
                         "Include_test=0", "Include_doc=0", "Include_launcher=0"],
                        check=True
                    )
                    _log("Python installer finished")
                    system_python = _find_valid_python()
                    if not system_python:
                        raise Exception("Automated Python installation failed. Please install manually.")

                # ── Create required folders ────────────────────────────────
                for _folder in [
                    PACKAGES_DIR,
                    os.path.join(APP_DIR, "voices"),
                    os.path.join(APP_DIR, "temp"),
                    os.path.join(APP_DIR, "outputs"),
                    os.path.join(APP_DIR, "scripts", "profiles"),
                ]:
                    if not os.path.exists(_folder):
                        os.makedirs(_folder, exist_ok=True)
                        _log(f"Created folder: {_folder}")

                # ── Step 1: venv ───────────────────────────────────────────
                _log(f"Step 1/5: venv (exists={os.path.exists(VENV_DIR)})")
                self.update_status(t("LAUNCHER_STEP_VENV"))
                self.update_progress(0, "Setting up virtual environment...")
                if not os.path.exists(VENV_DIR):
                    result = _run_logged("create venv",
                        [system_python, "-m", "venv", VENV_DIR],
                        check=True, creationflags=subprocess.CREATE_NO_WINDOW,
                    )
                    if result.returncode != 0:
                        raise Exception("Failed to create virtual environment.")
                self.update_progress(self._P_VENV)

                # ── Step 2: upgrade pip ────────────────────────────────────
                _log("Step 2/5: upgrade pip")
                self.update_status(t("LAUNCHER_STEP_PIP"))
                self.update_progress(self._P_VENV, "Upgrading pip...")
                _run_logged("upgrade pip",
                    [PIP_EXE, "install", "--upgrade", "pip"],
                    check=False, creationflags=subprocess.CREATE_NO_WINDOW,
                )
                self.update_progress(self._P_PIP)

                # ── Step 3: PyTorch CPU ────────────────────────────────────
                _log("Step 3/5: PyTorch CPU")
                self.update_status(t("LAUNCHER_STEP_PYTORCH"))
                self.update_progress(self._P_PIP, "Installing PyTorch (CPU)...")
                torch_local = _run_logged("torch local wheels",
                    [PIP_EXE, "install", "--find-links", PACKAGES_DIR,
                     "--timeout", "120", "torch", "torchaudio"],
                    check=False, creationflags=subprocess.CREATE_NO_WINDOW,
                )
                if torch_local.returncode != 0:
                    _log("Local torch failed — trying PyPI CPU index")
                    self.update_status(t("LAUNCHER_STEP_PYTORCH_DL"))
                    self.update_progress(self._P_PIP, "Downloading PyTorch from pytorch.org...")
                    _run_logged("torch PyPI CPU",
                        [PIP_EXE, "install", "--timeout", "120",
                         "torch", "torchaudio",
                         "--index-url", "https://download.pytorch.org/whl/cpu"],
                        check=False, creationflags=subprocess.CREATE_NO_WINDOW,
                    )
                self.update_progress(self._P_TORCH)

                # ── Step 4: llama-cpp-python (wheel only) ──────────────────
                _log("Step 4/5: llama-cpp-python (wheel only)")
                self.update_status("Installing AI components (4/5)...")
                self.update_progress(self._P_TORCH, "Installing llama-cpp-python...")
                llama_result = _run_logged("llama-cpp-python (wheel only)",
                    [PIP_EXE, "install",
                     "--only-binary=llama_cpp_python",
                     "--find-links", PACKAGES_DIR,
                     "--extra-index-url", "https://abetlen.github.io/llama-cpp-python/whl/cpu",
                     "--timeout", "120",
                     "llama-cpp-python==0.3.19"],
                    check=False, creationflags=subprocess.CREATE_NO_WINDOW,
                )
                if llama_result.returncode != 0:
                    raise Exception(
                        "Could not install llama-cpp-python — no prebuilt wheel found for your Python version.\n"
                        "Please check https://github.com/abetlen/llama-cpp-python/releases"
                    )
                self.update_progress(self._P_LLAMA)

                # ── Step 5: requirements.txt with per-package progress ─────
                req_file = os.path.join(APP_DIR, "requirements.txt")
                _log(f"Step 5/5: requirements.txt (exists={os.path.exists(req_file)})")
                self.update_status(t("LAUNCHER_STEP_COMPONENTS"))

                total_pkgs   = _count_requirements(req_file)
                done_pkgs    = 0
                range_pct    = self._P_REQ_END - self._P_LLAMA  # remaining %

                def _on_pip_line(line: str):
                    nonlocal done_pkgs
                    # Detect package start
                    if line.startswith("Collecting ") or line.startswith("Requirement already satisfied:"):
                        pkg = line.split()[1].split("==")[0].split(">=")[0].split("[")[0]
                        if line.startswith("Collecting "):
                            done_pkgs += 1
                        pct = self._P_LLAMA + (done_pkgs / total_pkgs) * range_pct
                        label = f"{'Installing' if line.startswith('Collecting') else 'Already installed'}: {pkg}"
                        self.update_progress(pct, label)

                proc = _run_with_progress(
                    "requirements.txt",
                    [PIP_EXE, "install", "--find-links", PACKAGES_DIR,
                     "--timeout", "120", "-r", req_file],
                    on_line=_on_pip_line,
                    check=False, creationflags=subprocess.CREATE_NO_WINDOW,
                )

                if proc.returncode != 0:
                    raise Exception(
                        "Package installation failed.\n"
                        f"Check your internet connection and try again.\n\n"
                        f"Full details saved to:\n{INSTALL_LOG}"
                    )

                _log("=== Installation completed successfully ===")
                with open(SETUP_MARKER, "w") as f:
                    f.write("setup_complete")

                self.update_progress(self._P_REQ_END, "")
                self.update_status(t("LAUNCHER_STEP_COMPLETE"))
                self.after(1000, self._spawn_and_linger)

            except Exception as e:
                _log(f"EXCEPTION: {e}")
                _log(traceback.format_exc())
                self.update_status(t("LAUNCHER_SETUP_FAILED_TITLE"))
                messagebox.showerror(
                    t("LAUNCHER_SETUP_FAILED_TITLE"),
                    f"Setup failed:\n{e}\n\nFull log:\n{INSTALL_LOG}"
                )
                self.after(2000, self.destroy)

        threading.Thread(target=_worker, daemon=True).start()


if __name__ == "__main__":
    load_language()
    _log(f"__main__: setup_done={os.path.exists(SETUP_MARKER)}  venv_python={os.path.exists(PYTHON_EXE)}")
    if os.path.exists(SETUP_MARKER) and os.path.exists(PYTHON_EXE):
        _log("Already set up — launching main app directly")
        launch_main_app()
    else:
        _log("First run — opening installer GUI")
        gui = InstallerGUI()
        gui.after(1000, gui.run_setup)
        gui.mainloop()
