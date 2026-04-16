"""
KoKoFish — Auto-Launcher & Setup UI.

This file uses ONLY built-in Python libraries (tkinter, os, subprocess)
so it can run on a completely bare Python installation.

It serves as the main entry point:
1. On first run: shows a language picker, then the installer GUI.
2. On subsequent runs: launches main.py directly inside the venv.
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
import math
import subprocess
import threading
import traceback
import tkinter as tk
import tkinter.ttk as ttk
from tkinter import messagebox
from lang import t, load_language, save_language_pref, get_languages

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
        count = 30
    return max(count, 1)


# ===========================================================================
# Language Picker — shown once on first run before the installer opens
# ===========================================================================
class LanguagePickerDialog(tk.Tk):
    """
    Tiny styled window that lets the user pick their language before
    the installer begins.  Saves the preference via save_language_pref()
    so both the installer and the main app start in the chosen language.
    """

    def __init__(self):
        super().__init__()

        self.title(t("LAUNCHER_LANG_PICKER_TITLE"))
        self.configure(bg="#0f0f1a")
        self.overrideredirect(True)
        self.resizable(False, False)

        W, H = 320, 200
        self.update_idletasks()
        x = (self.winfo_screenwidth()  - W) // 2
        y = (self.winfo_screenheight() - H) // 2
        self.geometry(f"{W}x{H}+{x}+{y}")

        # ── Fish header ────────────────────────────────────────────────────
        tk.Label(
            self, text="🐟 KoKoFish", font=("Segoe UI", 20, "bold"),
            bg="#0f0f1a", fg="#6c83f7"
        ).pack(pady=(22, 2))

        # ── Label ──────────────────────────────────────────────────────────
        tk.Label(
            self, text=t("LAUNCHER_LANG_PICKER_LABEL"),
            font=("Segoe UI", 10), bg="#0f0f1a", fg="#9a9ab0"
        ).pack(pady=(8, 4))

        # ── Dropdown ───────────────────────────────────────────────────────
        languages = get_languages()   # [(code, display_name), ...]
        self._lang_codes = [code for code, _ in languages]
        display_names    = [name for _, name in languages]

        self._selected = tk.StringVar(value=display_names[0] if display_names else "English")

        style = ttk.Style(self)
        style.theme_use("default")
        style.configure(
            "KoKo.TCombobox",
            fieldbackground="#1a1a2e",
            background="#1a1a2e",
            foreground="#e8e8f0",
            selectbackground="#6c83f7",
            selectforeground="#ffffff",
            bordercolor="#6c83f7",
            arrowcolor="#6c83f7",
        )
        self._combo = ttk.Combobox(
            self, textvariable=self._selected,
            values=display_names, state="readonly",
            style="KoKo.TCombobox", width=26, font=("Segoe UI", 10)
        )
        self._combo.pack(pady=(0, 14))

        # ── Continue button ────────────────────────────────────────────────
        btn_frame = tk.Frame(self, bg="#0f0f1a")
        btn_frame.pack()
        tk.Button(
            btn_frame,
            text=t("LAUNCHER_LANG_PICKER_BTN"),
            font=("Segoe UI", 10, "bold"),
            bg="#6c83f7", fg="#ffffff",
            activebackground="#8a9af9", activeforeground="#ffffff",
            relief="flat", padx=24, pady=6,
            cursor="hand2",
            command=self._confirm,
        ).pack()

        self._chosen_code = "en"

    def _confirm(self):
        selected_name = self._selected.get()
        # Map display name back to language code
        languages = get_languages()
        for code, name in languages:
            if name == selected_name:
                self._chosen_code = code
                break
        _log(f"Language chosen: {self._chosen_code}")
        save_language_pref(self._chosen_code)
        load_language(self._chosen_code)
        self.destroy()

    @classmethod
    def pick(cls) -> str:
        """Show the picker and return the chosen language code."""
        picker = cls()
        picker.mainloop()
        return picker._chosen_code


# ===========================================================================
# Main Installer GUI
# ===========================================================================
class InstallerGUI(tk.Tk):
    # Progress milestones (%)
    _P_VENV      = 5
    _P_PIP       = 10
    _P_TORCH     = 35
    _P_LLAMA     = 45
    _P_REQ_END   = 85   # requirements.txt finishes here
    _P_KOKORO_END = 99  # Kokoro model download finishes here

    def __init__(self):
        super().__init__()

        self.title("KoKoFish " + t("LAUNCHER_WINDOW_TITLE"))
        self.geometry("450x340")
        self.configure(bg="#0f0f1a")

        self.update_idletasks()
        x = (self.winfo_screenwidth()  - 450) // 2
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

        # Swimming fish state — _tick draws everything each frame
        self._fish_x      = 50.0   # start near left edge
        self._fish_dir    = 1      # +1 = right, -1 = left

        self._anim_step   = 0
        self._anim_id     = None
        self._anim_active = True
        self._tick()

    # ── Animation ─────────────────────────────────────────────────────────
    def _tick(self):
        if not self._anim_active:
            return
        c = self._canvas
        c.delete("fish")
        c.delete("bubble")

        # ── Fish swim constants ────────────────────────────────────────────
        SPEED     = 1.2        # px per frame
        FISH_MIN  = 36         # left turn-around x
        FISH_MAX  = 414        # right turn-around x
        FY        = 38         # fish centre y (canvas is 60px tall)
        FH        = 9          # half-height of body
        FW        = 14         # half-width of body

        # ── Bubble constants (slightly slower than before) ─────────────────
        CYCLE     = 52         # frames per bubble cycle  (was 40)
        OFFSETS   = [0, 17, 34]
        RISE      = 50

        s  = self._anim_step
        dr = self._fish_dir

        # Move fish
        self._fish_x += dr * SPEED
        if self._fish_x >= FISH_MAX:
            self._fish_x = FISH_MAX
            self._fish_dir = -1
            dr = -1
        elif self._fish_x <= FISH_MIN:
            self._fish_x = FISH_MIN
            self._fish_dir = 1
            dr = 1

        cx = int(self._fish_x)

        # Gentle tail waggle via sine — tip swings ±4px
        waggle = int(4 * math.sin(s * 0.30))

        # ── Draw fish ─────────────────────────────────────────────────────
        if dr == -1:   # facing LEFT — tail on right, mouth on left
            # Tail
            c.create_polygon(
                cx + FW, FY - FH,
                cx + FW + 12, FY + waggle,
                cx + FW, FY + FH,
                fill="#6c83f7", outline="", tags="fish"
            )
            # Body
            c.create_oval(
                cx - FW, FY - FH, cx + FW, FY + FH,
                fill="#6c83f7", outline="", tags="fish"
            )
            # Pectoral fin (lighter blue, below body)
            c.create_polygon(
                cx - 2, FY + 4,
                cx + 7, FY + FH,
                cx - 7, FY + FH,
                fill="#8a9af9", outline="", tags="fish"
            )
            # Eye near mouth (left side)
            c.create_oval(
                cx - FW + 3, FY - 3,
                cx - FW + 8, FY + 2,
                fill="#0f0f1a", outline="", tags="fish"
            )
            mouth_x = cx - FW   # bubbles rise from here

        else:          # facing RIGHT — tail on left, mouth on right
            # Tail
            c.create_polygon(
                cx - FW, FY - FH,
                cx - FW - 12, FY + waggle,
                cx - FW, FY + FH,
                fill="#6c83f7", outline="", tags="fish"
            )
            # Body
            c.create_oval(
                cx - FW, FY - FH, cx + FW, FY + FH,
                fill="#6c83f7", outline="", tags="fish"
            )
            # Pectoral fin
            c.create_polygon(
                cx + 2, FY + 4,
                cx - 7, FY + FH,
                cx + 7, FY + FH,
                fill="#8a9af9", outline="", tags="fish"
            )
            # Eye near mouth (right side)
            c.create_oval(
                cx + FW - 8, FY - 3,
                cx + FW - 3, FY + 2,
                fill="#0f0f1a", outline="", tags="fish"
            )
            mouth_x = cx + FW   # bubbles rise from here

        # ── Draw bubbles (cluster of 3 rising from mouth) ─────────────────
        for i, offset in enumerate(OFFSETS):
            phase = ((s + offset) % CYCLE) / CYCLE
            # Scatter bubbles a little horizontally around the mouth
            bx = mouth_x + (i - 1) * 5   # –5, 0, +5 px spread
            by = FY - int(phase * RISE)
            r  = 2 + int(phase * 7)
            blue  = min(255, 0x6c + int(phase * 140))
            green = min(255, 0x83 + int(phase * 80))
            color = f"#{0x6c:02x}{green:02x}{blue:02x}"
            c.create_oval(bx - r, by - r, bx + r, by + r,
                          fill=color, outline="", tags="bubble")

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
        self._canvas.delete("fish")
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
                t("LAUNCHER_PYTHON_REQUIRED_MSG")
            )
            if not wants_install:
                _log("User declined — aborting")
                messagebox.showerror(
                    t("LAUNCHER_SETUP_FAILED_TITLE"),
                    t("LAUNCHER_PYTHON_REQUIRED_ABORT")
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
                        self.update_status(t("LAUNCHER_PYTHON_DOWNLOADING"))
                        try:
                            urllib.request.urlretrieve(
                                "https://www.python.org/ftp/python/3.12.9/python-3.12.9-amd64.exe",
                                installer_path
                            )
                            _log("Download complete")
                        except Exception as e:
                            raise Exception(f"{t('LAUNCHER_PYTHON_DOWNLOADING')}: {e}")
                    else:
                        self.update_status(t("LAUNCHER_PYTHON_INSTALLING_STATUS"))

                    self.update_status(t("LAUNCHER_PYTHON_INSTALLING_BG"))
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
                        raise Exception(t("LAUNCHER_PYTHON_INSTALL_FAILED"))

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
                _log(f"Step 1/6: venv (exists={os.path.exists(VENV_DIR)})")
                self.update_status(t("LAUNCHER_STEP_VENV"))
                self.update_progress(0, t("LAUNCHER_STEP_VENV_DETAIL"))
                if not os.path.exists(VENV_DIR):
                    result = _run_logged("create venv",
                        [system_python, "-m", "venv", VENV_DIR],
                        check=True, creationflags=subprocess.CREATE_NO_WINDOW,
                    )
                    if result.returncode != 0:
                        raise Exception("Failed to create virtual environment.")
                self.update_progress(self._P_VENV)

                # ── Step 2: upgrade pip ────────────────────────────────────
                _log("Step 2/6: upgrade pip")
                self.update_status(t("LAUNCHER_STEP_PIP"))
                self.update_progress(self._P_VENV, t("LAUNCHER_STEP_PIP_DETAIL"))
                _run_logged("upgrade pip",
                    [PIP_EXE, "install", "--upgrade", "pip"],
                    check=False, creationflags=subprocess.CREATE_NO_WINDOW,
                )
                self.update_progress(self._P_PIP)

                # ── Step 3: PyTorch CPU ────────────────────────────────────
                _log("Step 3/6: PyTorch CPU")
                self.update_status(t("LAUNCHER_STEP_PYTORCH"))
                self.update_progress(self._P_PIP, t("LAUNCHER_STEP_PYTORCH_DETAIL"))
                torch_local = _run_logged("torch local wheels",
                    [PIP_EXE, "install", "--find-links", PACKAGES_DIR,
                     "--timeout", "120", "torch", "torchaudio"],
                    check=False, creationflags=subprocess.CREATE_NO_WINDOW,
                )
                if torch_local.returncode != 0:
                    _log("Local torch failed — trying PyPI CPU index")
                    self.update_status(t("LAUNCHER_STEP_PYTORCH_DL"))
                    self.update_progress(self._P_PIP, t("LAUNCHER_STEP_PYTORCH_DL_DETAIL"))
                    _run_logged("torch PyPI CPU",
                        [PIP_EXE, "install", "--timeout", "120",
                         "torch", "torchaudio",
                         "--index-url", "https://download.pytorch.org/whl/cpu"],
                        check=False, creationflags=subprocess.CREATE_NO_WINDOW,
                    )
                self.update_progress(self._P_TORCH)

                # ── Step 4: llama-cpp-python (wheel only) ──────────────────
                _log("Step 4/6: llama-cpp-python (wheel only)")
                self.update_status(t("LAUNCHER_STEP_LLAMA"))
                self.update_progress(self._P_TORCH, t("LAUNCHER_STEP_LLAMA_DETAIL"))
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
                    raise Exception(t("LAUNCHER_LLAMA_FAILED"))
                self.update_progress(self._P_LLAMA)

                # ── Step 5: requirements.txt with per-package progress ─────
                req_file = os.path.join(APP_DIR, "requirements.txt")
                _log(f"Step 5/6: requirements.txt (exists={os.path.exists(req_file)})")
                self.update_status(t("LAUNCHER_STEP_COMPONENTS"))

                total_pkgs = _count_requirements(req_file)
                done_pkgs  = 0
                range_pct  = self._P_REQ_END - self._P_LLAMA

                def _on_pip_line(line: str):
                    nonlocal done_pkgs
                    if line.startswith("Collecting "):
                        pkg = line.split()[1].split("==")[0].split(">=")[0].split("[")[0]
                        done_pkgs += 1
                        pct   = self._P_LLAMA + (done_pkgs / total_pkgs) * range_pct
                        label = t("LAUNCHER_PKG_INSTALLING", pkg=pkg)
                        self.update_progress(pct, label)
                    elif line.startswith("Requirement already satisfied:"):
                        pkg = line.split()[3].split("==")[0].split(">=")[0].split("[")[0]
                        pct = self._P_LLAMA + (done_pkgs / total_pkgs) * range_pct
                        self.update_progress(pct, t("LAUNCHER_PKG_ALREADY", pkg=pkg))

                proc = _run_with_progress(
                    "requirements.txt",
                    [PIP_EXE, "install", "--find-links", PACKAGES_DIR,
                     "--timeout", "120", "-r", req_file],
                    on_line=_on_pip_line,
                    creationflags=subprocess.CREATE_NO_WINDOW,
                )

                if proc.returncode != 0:
                    raise Exception(
                        t("LAUNCHER_PACKAGES_FAILED", log=INSTALL_LOG)
                    )

                self.update_progress(self._P_REQ_END, "")

                # ── Step 6: Kokoro voice model files ──────────────────────
                _log("Step 6/6: Kokoro model files")
                self.update_status(t("LAUNCHER_STEP_KOKORO"))

                kokoro_dir   = os.path.join(APP_DIR, "kokoro_models")
                onnx_path    = os.path.join(kokoro_dir, "kokoro-v1.0.int8.onnx")
                voices_path  = os.path.join(kokoro_dir, "voices-v1.0.bin")
                os.makedirs(kokoro_dir, exist_ok=True)

                def _file_ok(p, min_mb):
                    return os.path.isfile(p) and os.path.getsize(p) > min_mb * 1_000_000

                if _file_ok(onnx_path, 50) and _file_ok(voices_path, 5):
                    _log("Kokoro model files already present — skipping")
                    self.update_progress(self._P_KOKORO_END, t("LAUNCHER_KOKORO_SKIP"))
                else:
                    _log("Kokoro model files missing — downloading via huggingface_hub")
                    # Use the venv's console python (python.exe, not pythonw.exe)
                    # so stdout can be captured for progress reporting.
                    venv_py_console = os.path.join(VENV_DIR, "Scripts", "python.exe")
                    if not os.path.isfile(venv_py_console):
                        venv_py_console = PYTHON_EXE  # fallback

                    # APP_DIR path — escape backslashes for the inline script
                    _adir = APP_DIR.replace("\\", "\\\\")
                    inline = (
                        "import sys; sys.path.insert(0, r'" + APP_DIR + "'); "
                        "from utils import setup_kokoro, is_kokoro_ready; "
                        "ok = is_kokoro_ready() or setup_kokoro("
                        "    lambda msg, frac: print(f'PROG:{msg}|{frac:.3f}', flush=True)"
                        "); "
                        "sys.exit(0 if ok else 1)"
                    )

                    _kok_range = self._P_KOKORO_END - self._P_REQ_END

                    def _on_kokoro_line(line: str):
                        if line.startswith("PROG:"):
                            parts = line[5:].rsplit("|", 1)
                            msg  = parts[0].strip()
                            try:
                                frac = float(parts[1])
                            except Exception:
                                frac = 0.0
                            pct = self._P_REQ_END + frac * _kok_range
                            self.update_progress(pct, msg)

                    kok_result = _run_with_progress(
                        "kokoro model download",
                        [venv_py_console, "-c", inline],
                        on_line=_on_kokoro_line,
                        cwd=APP_DIR,
                        creationflags=subprocess.CREATE_NO_WINDOW,
                    )

                    if kok_result.returncode != 0:
                        _log("WARNING: Kokoro model download failed — app will prompt on first use")
                        self.update_progress(self._P_KOKORO_END, t("LAUNCHER_KOKORO_DL_FAILED"))
                    else:
                        _log("Kokoro model files ready")
                        self.update_progress(self._P_KOKORO_END, t("LAUNCHER_KOKORO_SKIP"))

                _log("=== Installation completed successfully ===")
                with open(SETUP_MARKER, "w") as f:
                    f.write("setup_complete")

                self.update_status(t("LAUNCHER_STEP_COMPLETE"))
                self.after(1000, self._spawn_and_linger)

            except Exception as e:
                _log(f"EXCEPTION: {e}")
                _log(traceback.format_exc())
                self.update_status(t("LAUNCHER_SETUP_FAILED_TITLE"))
                messagebox.showerror(
                    t("LAUNCHER_SETUP_FAILED_TITLE"),
                    t("LAUNCHER_SETUP_FAILED_MSG", error=e, log=INSTALL_LOG)
                )
                self.after(2000, self.destroy)

        threading.Thread(target=_worker, daemon=True).start()

    # ── Kokoro-only repair (for existing installs missing model files) ─────
    def run_kokoro_only(self):
        """
        Download Kokoro model files only — used when venv is already set up
        but the model files are missing (e.g. upgrade from pre-step-6 install).
        """
        def _worker():
            try:
                kokoro_dir  = os.path.join(APP_DIR, "kokoro_models")
                onnx_path   = os.path.join(kokoro_dir, "kokoro-v1.0.int8.onnx")
                voices_path = os.path.join(kokoro_dir, "voices-v1.0.bin")
                os.makedirs(kokoro_dir, exist_ok=True)

                def _file_ok(p, min_mb):
                    return os.path.isfile(p) and os.path.getsize(p) > min_mb * 1_000_000

                if _file_ok(onnx_path, 50) and _file_ok(voices_path, 5):
                    _log("run_kokoro_only: files already present")
                    self.update_progress(100, t("LAUNCHER_KOKORO_SKIP"))
                    self.after(800, self._spawn_and_linger)
                    return

                _log("run_kokoro_only: downloading missing Kokoro model files")
                self.update_status(t("LAUNCHER_STEP_KOKORO"))
                self.update_progress(0, "")

                venv_py_console = os.path.join(VENV_DIR, "Scripts", "python.exe")
                if not os.path.isfile(venv_py_console):
                    venv_py_console = PYTHON_EXE

                inline = (
                    "import sys; sys.path.insert(0, r'" + APP_DIR + "'); "
                    "from utils import setup_kokoro, is_kokoro_ready; "
                    "ok = is_kokoro_ready() or setup_kokoro("
                    "    lambda msg, frac: print(f'PROG:{msg}|{frac:.3f}', flush=True)"
                    "); "
                    "sys.exit(0 if ok else 1)"
                )

                def _on_line(line: str):
                    if line.startswith("PROG:"):
                        parts = line[5:].rsplit("|", 1)
                        msg  = parts[0].strip()
                        try:
                            frac = float(parts[1])
                        except Exception:
                            frac = 0.0
                        self.update_progress(frac * 99, msg)

                result = _run_with_progress(
                    "kokoro model download (repair)",
                    [venv_py_console, "-c", inline],
                    on_line=_on_line,
                    cwd=APP_DIR,
                    creationflags=subprocess.CREATE_NO_WINDOW,
                )

                if result.returncode != 0:
                    _log("run_kokoro_only: download failed — launching anyway")
                    self.update_progress(99, t("LAUNCHER_KOKORO_DL_FAILED"))
                else:
                    _log("run_kokoro_only: download complete")
                    self.update_progress(99, t("LAUNCHER_KOKORO_SKIP"))

                self.update_status(t("LAUNCHER_STEP_COMPLETE"))
                self.after(1000, self._spawn_and_linger)

            except Exception as e:
                _log(f"run_kokoro_only EXCEPTION: {e}")
                # Non-fatal — just launch the app anyway
                self.after(0, launch_main_app)

        threading.Thread(target=_worker, daemon=True).start()


# ===========================================================================
# Entry point
# ===========================================================================
if __name__ == "__main__":
    # Load whatever language was previously saved (default: English)
    load_language()

    _log(f"__main__: setup_done={os.path.exists(SETUP_MARKER)}  venv_python={os.path.exists(PYTHON_EXE)}")

    if os.path.exists(SETUP_MARKER) and os.path.exists(PYTHON_EXE):
        # Venv is ready — but check whether Kokoro model files exist too.
        _kokoro_dir    = os.path.join(APP_DIR, "kokoro_models")
        _onnx_path     = os.path.join(_kokoro_dir, "kokoro-v1.0.int8.onnx")
        _voices_path   = os.path.join(_kokoro_dir, "voices-v1.0.bin")
        _onnx_ok   = os.path.isfile(_onnx_path)   and os.path.getsize(_onnx_path)   > 50 * 1_000_000
        _voices_ok = os.path.isfile(_voices_path) and os.path.getsize(_voices_path) >  5 * 1_000_000
        _log(f"Kokoro check: onnx_ok={_onnx_ok}  voices_ok={_voices_ok}")

        if _onnx_ok and _voices_ok:
            _log("All files present — launching main app directly")
            launch_main_app()
        else:
            # Venv exists but Kokoro models are missing (e.g. upgrade from
            # a pre-step-6 install, or a partial first run).
            _log("Kokoro models missing — opening installer for Kokoro-only repair")
            gui = InstallerGUI()
            gui.after(500, gui.run_kokoro_only)
            gui.mainloop()
    else:
        # First run — show language picker, then full installer
        _log("First run — showing language picker")
        LanguagePickerDialog.pick()
        # Language is now saved and active; open the installer
        _log("Language selected — opening installer GUI")
        gui = InstallerGUI()
        gui.after(1000, gui.run_setup)
        gui.mainloop()
