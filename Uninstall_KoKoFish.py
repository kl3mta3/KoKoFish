"""
KoKoFish — Uninstaller

Deletes the venv and other heavy files to free up space.
Gives the user the option to either:
1. Re-run setup
2. Completely remove everything (including settings/voices)
"""
import os
import sys
import shutil
import tkinter as tk
from tkinter import messagebox

APP_DIR = os.path.dirname(os.path.abspath(__file__))

class UninstallerGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        
        self.title("Uninstall KoKoFish")
        self.geometry("450x250")
        self.configure(bg="#0f0f1a")
        
        # Center window
        self.update_idletasks()
        x = (self.winfo_screenwidth() - 450) // 2
        y = (self.winfo_screenheight() - 250) // 2
        self.geometry(f"+{x}+{y}")
        self.overrideredirect(True)
        
        # UI Elements
        tk.Label(
            self, text="🐟 KoKoFish Uninstaller", font=("Segoe UI", 24, "bold"),
            bg="#0f0f1a", fg="#ef476f"
        ).pack(pady=(30, 5))
        
        tk.Label(
            self, text="Do you want to completely remove KoKoFish?", font=("Segoe UI", 12),
            bg="#0f0f1a", fg="#e8e8f0"
        ).pack(pady=(0, 20))
        
        btn_frame = tk.Frame(self, bg="#0f0f1a")
        btn_frame.pack(pady=10)
        
        tk.Button(
            btn_frame, text="Uninstall Everything", font=("Segoe UI", 11, "bold"),
            bg="#ef476f", fg="white", relief="flat", width=20,
            command=self.full_uninstall
        ).pack(side="left", padx=10)
        
        tk.Button(
            btn_frame, text="Cancel", font=("Segoe UI", 11),
            bg="#2a2a4a", fg="white", relief="flat", width=10,
            command=self.destroy
        ).pack(side="left", padx=10)

    def force_delete_dir(self, dir_path):
        if os.path.exists(dir_path):
            try:
                shutil.rmtree(dir_path, ignore_errors=True)
            except:
                pass
                
    def full_uninstall(self):
        confirm = messagebox.askyesno(
            "Confirm Uninstall",
            "This will delete ALL KoKoFish files in this folder, including:\n"
            "- The ~3GB Virtual Environment\n"
            "- Downloaded AI Models\n"
            "- Your custom Voice Profiles\n"
            "- Settings files\n\n"
            "Are you sure you want to proceed?"
        )
        if not confirm:
            return
            
        # Delete heavy folders
        self.force_delete_dir(os.path.join(APP_DIR, "venv"))
        self.force_delete_dir(os.path.join(APP_DIR, "packages"))
        self.force_delete_dir(os.path.join(APP_DIR, "bin"))
        self.force_delete_dir(os.path.join(APP_DIR, "voices"))
        
        # Delete Fish-Speech weights if downloaded
        weights = os.path.join(APP_DIR, "fish-speech", "checkpoints")
        self.force_delete_dir(weights)
        
        # Delete settings & setup cache
        for f in ["settings.json", ".setup_complete", "requirements.txt", "KoKoFish.bat", "KoKoFish.ps1", "KoKoFish.exe", "launcher.py", "Uninstall_KoKoFish.py", "main.py", "ui.py", "tts_engine.py", "stt_engine.py", "voice_manager.py", "settings.py", "utils.py", "cuda_setup.py", "setup_ffmpeg.py"]:
            path = os.path.join(APP_DIR, f)
            if os.path.exists(path):
                try:
                    os.remove(path)
                except:
                    pass
                    
        messagebox.showinfo("Uninstall Complete", "KoKoFish has been successfully uninstalled.\n\nYou can now delete this folder.")
        self.destroy()


if __name__ == "__main__":
    app = UninstallerGUI()
    app.mainloop()
