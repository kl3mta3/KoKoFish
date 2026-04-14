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
        
        self.title("KoKoFish Setup")
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
            self, text="TTS/STT Studio", font=("Segoe UI", 12),
            bg="#0f0f1a", fg="#9a9ab0"
        ).pack(pady=(0, 20))
        
        self.status_var = tk.StringVar(value="Initializing first-time setup...")
        self.status_label = tk.Label(
            self, textvariable=self.status_var, font=("Segoe UI", 10),
            bg="#0f0f1a", fg="#e8e8f0"
        )
        self.status_label.pack()

    def update_status(self, text):
        self.status_var.set(text)
        self.update()
        
    def run_setup(self):
        """Run the offline installation process."""
        import shutil
        import urllib.request
        import tempfile
        
        def _find_valid_python():
            candidates = [shutil.which("python")]
            local_app = os.environ.get("LOCALAPPDATA", "")
            for v in ["312", "311", "313"]:
                p = os.path.join(local_app, "Programs", "Python", f"Python{v}", "python.exe")
                if os.path.exists(p):
                    candidates.append(p)
                    
            for calc in candidates:
                if not calc or not os.path.exists(calc): continue
                try:
                    out = subprocess.run(
                        [calc, "-c", "import sys; print(sys.version_info >= (3, 11))"],
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
                "Python Required", 
                "KoKoFish requires Python (v3.11+) which was not found on your system.\n\n"
                "Would you like KoKoFish to smoothly download and install Python 3.11 for you right now?\n"
                "(It installs cleanly to your local user folder without requiring Admin privileges)"
            )
            if not wants_install:
                messagebox.showerror("Setup Failed", "Python 3.11+ is required to execute the Fish-Speech offline engine natively.")
                self.destroy()
                return

        def _worker():
            try:
                nonlocal system_python
                
                if wants_install:
                    self.update_status("Downloading Python 3.11... (This may take a minute)")
                    installer_url = "https://www.python.org/ftp/python/3.11.9/python-3.11.9-amd64.exe"
                    installer_path = os.path.join(tempfile.gettempdir(), "python-3.11.9-amd64-kokofish.exe")
                    
                    try:
                        urllib.request.urlretrieve(installer_url, installer_path)
                    except Exception as e:
                        raise Exception(f"Failed to download Python installer: {e}")
                        
                    self.update_status("Installing Python in the background...")
                    flags = ["/passive", "InstallAllUsers=0", "PrependPath=1", "Include_test=0", "Include_doc=0", "Include_launcher=0"]
                    subprocess.run([installer_path] + flags, check=True)
                    
                    system_python = _find_valid_python()
                    if not system_python:
                        raise Exception("Automated Python installation failed. Please install manually.")

                    
                # 1. Create VENV
                if not os.path.exists(VENV_DIR):
                    self.update_status(f"Creating virtual environment (1/3)...")
                    subprocess.run([system_python, "-m", "venv", VENV_DIR], check=True, creationflags=subprocess.CREATE_NO_WINDOW)
                
                # 2. PyTorch Install (Offline)
                self.update_status("Installing PyTorch Core (2/3)...")
                subprocess.run([
                    PIP_EXE, "install", "--no-index", "--find-links", PACKAGES_DIR,
                    "torch", "torchaudio"
                ], check=False, creationflags=subprocess.CREATE_NO_WINDOW)
                
                # 3. Dependencies Install (Offline)
                self.update_status("Installing application components (3/3)...")
                req_file = os.path.join(APP_DIR, "requirements.txt")
                subprocess.run([
                    PIP_EXE, "install", "--no-index", "--find-links", PACKAGES_DIR,
                    "-r", req_file
                ], check=False, creationflags=subprocess.CREATE_NO_WINDOW)
                
                # Done
                with open(SETUP_MARKER, "w") as f:
                    f.write("setup_complete")
                
                self.update_status("Setup complete! Starting KoKoFish...")
                self.after(1000, launch_main_app)
                
            except Exception as e:
                self.update_status("Installation Error!")
                messagebox.showerror("Setup Failed", f"Setup failed:\n{e}")
                self.after(2000, self.destroy)

        threading.Thread(target=_worker, daemon=True).start()


if __name__ == "__main__":
    # If setup is already done, launch immediately
    if os.path.exists(SETUP_MARKER) and os.path.exists(PYTHON_EXE):
        launch_main_app()
    else:
        # First-time setup required
        gui = InstallerGUI()
        gui.after(1000, gui.run_setup)
        gui.mainloop()
