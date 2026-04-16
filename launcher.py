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
import os
import sys
import subprocess
import threading
import tkinter as tk
from tkinter import messagebox
from lang import t, load_language

if getattr(sys, 'frozen', False):
    # Running as compiled PyInstaller executable
    APP_DIR = os.path.dirname(sys.executable)
else:
    # Running as standard Python script
    APP_DIR = os.path.dirname(os.path.abspath(__file__))

VENV_DIR = os.path.join(APP_DIR, "venv")
SETUP_MARKER = os.path.join(APP_DIR, ".setup_complete")
PACKAGES_DIR = os.path.join(APP_DIR, "packages")

# Resolve correct python/pip within venv
if sys.platform == "win32":
    PYTHON_EXE = os.path.join(VENV_DIR, "Scripts", "pythonw.exe") # Use pythonw to prevent cmd window
    PIP_EXE = os.path.join(VENV_DIR, "Scripts", "pip.exe")
else:
    PYTHON_EXE = os.path.join(VENV_DIR, "bin", "python")
    PIP_EXE = os.path.join(VENV_DIR, "bin", "pip")

def launch_main_app():
    """Launch the real application."""
    main_script = os.path.join(APP_DIR, "main.py")

    # Detach process so the main app doesn't close when the launcher closes
    flags = 0
    if sys.platform == "win32":
        flags = 0x00000008 | 0x00000200 # DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP

    # CRITICAL FIX: PyInstaller sets TCL/TK_LIBRARY to a temp folder that gets
    # deleted as soon as this launcher exits. We must remove these so the actual
    # pythonw process falls back to its own built-in TCL/TK libraries.
    env = os.environ.copy()
    env.pop("TCL_LIBRARY", None)
    env.pop("TK_LIBRARY", None)

    subprocess.Popen([PYTHON_EXE, main_script], cwd=APP_DIR, env=env, creationflags=flags)
    sys.exit(0)

class InstallerGUI(tk.Tk):
    def __init__(self):
        super().__init__()

        self.title("KoKoFish " + t("LAUNCHER_WINDOW_TITLE"))
        self.geometry("450x250")
        self.configure(bg="#0f0f1a")

        # Center window
        self.update_idletasks()
        x = (self.winfo_screenwidth() - 450) // 2
        y = (self.winfo_screenheight() - 250) // 2
        self.geometry(f"+{x}+{y}")
        self.overrideredirect(True) # Remove windows border

        # UI Elements
        tk.Label(
            self, text="🐟 KoKoFish", font=("Segoe UI", 28, "bold"),
            bg="#0f0f1a", fg="#6c83f7"
        ).pack(pady=(40, 5))

        tk.Label(
            self, text=t("LAUNCHER_SUBTITLE"), font=("Segoe UI", 12),
            bg="#0f0f1a", fg="#9a9ab0"
        ).pack(pady=(0, 20))

        self.status_var = tk.StringVar(value=t("LAUNCHER_STATUS_INIT"))
        self.status_label = tk.Label(
            self, textvariable=self.status_var, font=("Segoe UI", 10),
            bg="#0f0f1a", fg="#e8e8f0"
        )
        self.status_label.pack()

    def update_status(self, text):
        self.status_var.set(text)
        self.update()

    def _spawn_and_linger(self):
        """Spawn main.py then keep this splash visible while it loads."""
        env = os.environ.copy()
        env.pop("TCL_LIBRARY", None)
        env.pop("TK_LIBRARY", None)
        flags = 0x00000008 | 0x00000200 if sys.platform == "win32" else 0
        main_script = os.path.join(APP_DIR, "main.py")
        subprocess.Popen([PYTHON_EXE, main_script], cwd=APP_DIR, env=env, creationflags=flags)
        self.update_status(t("LAUNCHER_LINGER_MSG"))
        self.after(15000, self.destroy)

    def run_setup(self):
        """Run the offline installation process."""
        import shutil
        import urllib.request
        import tempfile

        def _find_valid_python():
            # Prefer Python 3.12 exactly — bundled torch wheels are cp312
            candidates = []
            local_app = os.environ.get("LOCALAPPDATA", "")
            for v in ["312", "313", "311"]:  # 312 first
                p = os.path.join(local_app, "Programs", "Python", f"Python{v}", "python.exe")
                if os.path.exists(p):
                    candidates.append(p)
            sys_py = shutil.which("python")
            if sys_py:
                candidates.append(sys_py)

            for calc in candidates:
                if not calc or not os.path.exists(calc): continue
                try:
                    out = subprocess.run(
                        [calc, "-c", "import sys; print(sys.version_info[:2] == (3, 12))"],
                        capture_output=True, text=True, creationflags=subprocess.CREATE_NO_WINDOW
                    )
                    if "True" in out.stdout:
                        return calc
                except Exception:
                    pass
            return None

        # Check for Python on the main UI thread safely
        system_python = _find_valid_python()
        wants_install = False

        if not system_python:
            wants_install = messagebox.askyesno(
                t("LAUNCHER_PYTHON_REQUIRED_TITLE"),
                "KoKoFish requires Python 3.12 which was not found on your system.\n\n"
                "Would you like KoKoFish to download and install Python 3.12 for you right now?\n"
                "(It installs cleanly to your local user folder without requiring Admin privileges)"
            )
            if not wants_install:
                messagebox.showerror(t("LAUNCHER_SETUP_FAILED_TITLE"), "Python 3.12 is required. Please install it from python.org and try again.")
                self.destroy()
                return

        def _worker():
            try:
                nonlocal system_python

                if wants_install:
                    installer_filename = "python-3.12.9-amd64.exe"
                    installer_path = os.path.join(APP_DIR, "bin", installer_filename)

                    if not os.path.exists(installer_path):
                        # Fall back to temp folder download
                        installer_path = os.path.join(tempfile.gettempdir(), "python-3.12.9-amd64-kokofish.exe")
                        self.update_status("Downloading Python 3.12... (This may take a minute)")
                        installer_url = "https://www.python.org/ftp/python/3.12.9/python-3.12.9-amd64.exe"
                        try:
                            urllib.request.urlretrieve(installer_url, installer_path)
                        except Exception as e:
                            raise Exception(f"Failed to download Python installer: {e}")
                    else:
                        self.update_status("Installing Python 3.12...")

                    self.update_status("Installing Python in the background...")
                    flags = ["/passive", "InstallAllUsers=0", "PrependPath=1", "Include_test=0", "Include_doc=0", "Include_launcher=0"]
                    subprocess.run([installer_path] + flags, check=True)

                    system_python = _find_valid_python()
                    if not system_python:
                        raise Exception("Automated Python installation failed. Please install manually.")


                import datetime, traceback

                INSTALL_LOG = os.path.join(APP_DIR, "installation_error.txt")

                def _log(msg: str):
                    """Append a timestamped line to installation_error.txt."""
                    try:
                        with open(INSTALL_LOG, "a", encoding="utf-8") as _f:
                            _f.write(f"[{datetime.datetime.now():%Y-%m-%d %H:%M:%S}] {msg}\n")
                    except Exception:
                        pass

                def _run_logged(label, cmd, **kwargs):
                    """Run a subprocess, capture output, log on failure, return CompletedProcess."""
                    _log(f"Running: {label}")
                    r = subprocess.run(
                        cmd,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT,
                        text=True,
                        **kwargs,
                    )
                    if r.returncode != 0:
                        _log(f"FAILED (exit {r.returncode}): {label}")
                        _log(r.stdout or "(no output)")
                    else:
                        _log(f"OK: {label}")
                    return r

                _log("=== KoKoFish installation started ===")

                # 1. Create VENV
                if not os.path.exists(VENV_DIR):
                    self.update_status(t("LAUNCHER_STEP_VENV"))
                    _run_logged("create venv",
                        [system_python, "-m", "venv", VENV_DIR],
                        check=True, creationflags=subprocess.CREATE_NO_WINDOW,
                    )

                # 2. Upgrade pip inside the venv first
                self.update_status(t("LAUNCHER_STEP_PIP"))
                _run_logged("upgrade pip",
                    [PIP_EXE, "install", "--upgrade", "pip"],
                    check=False, creationflags=subprocess.CREATE_NO_WINDOW,
                )

                # 3. PyTorch CPU — try local wheels first, fall back to official CPU index
                self.update_status(t("LAUNCHER_STEP_PYTORCH"))
                torch_local = _run_logged("torch (local wheels)",
                    [PIP_EXE, "install", "--find-links", PACKAGES_DIR,
                     "torch", "torchaudio"],
                    check=False, creationflags=subprocess.CREATE_NO_WINDOW,
                )
                if torch_local.returncode != 0:
                    self.update_status(t("LAUNCHER_STEP_PYTORCH_DL"))
                    _run_logged("torch (PyPI CPU index)",
                        [PIP_EXE, "install",
                         "torch", "torchaudio",
                         "--index-url", "https://download.pytorch.org/whl/cpu"],
                        check=False, creationflags=subprocess.CREATE_NO_WINDOW,
                    )

                # 4. All other dependencies — local wheels first, PyPI fallback
                self.update_status(t("LAUNCHER_STEP_COMPONENTS"))
                req_file = os.path.join(APP_DIR, "requirements.txt")
                result = _run_logged("requirements.txt",
                    [PIP_EXE, "install", "--find-links", PACKAGES_DIR,
                     "-r", req_file],
                    check=False, creationflags=subprocess.CREATE_NO_WINDOW,
                )

                if result.returncode != 0:
                    _log("Step 4/4 failed — raising exception")
                    raise Exception(
                        "Package installation failed.\n"
                        f"Check your internet connection and try again.\n\n"
                        f"Full details saved to:\n{INSTALL_LOG}"
                    )

                _log("=== Installation completed successfully ===")

                # Mark setup complete only after a successful install
                with open(SETUP_MARKER, "w") as f:
                    f.write("setup_complete")

                self.update_status(t("LAUNCHER_STEP_COMPLETE"))
                self.after(1000, self._spawn_and_linger)

            except Exception as e:
                # Write full traceback to installation_error.txt
                try:
                    INSTALL_LOG = os.path.join(APP_DIR, "installation_error.txt")
                    import datetime, traceback
                    with open(INSTALL_LOG, "a", encoding="utf-8") as _f:
                        _f.write(f"\n[{datetime.datetime.now():%Y-%m-%d %H:%M:%S}] EXCEPTION:\n")
                        _f.write(traceback.format_exc())
                        _f.write("\n")
                except Exception:
                    pass
                self.update_status(t("LAUNCHER_SETUP_FAILED_TITLE"))
                messagebox.showerror(
                    t("LAUNCHER_SETUP_FAILED_TITLE"),
                    f"Setup failed:\n{e}\n\nSee installation_error.txt for details."
                )
                self.after(2000, self.destroy)

        threading.Thread(target=_worker, daemon=True).start()


if __name__ == "__main__":
    load_language()
    # If setup is already done, launch immediately
    if os.path.exists(SETUP_MARKER) and os.path.exists(PYTHON_EXE):
        launch_main_app()
    else:
        # First-time setup required
        gui = InstallerGUI()
        gui.after(1000, gui.run_setup)
        gui.mainloop()
