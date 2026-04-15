"""
Top-level relay launcher for KoKoFish distribution.

Compiled to KoKoFish.exe and placed next to the KoKoFish\ folder.
Launches KoKoFish\KoKoFish.exe relative to its own location so the
zip is fully portable — no hardcoded paths.
"""
import os
import sys
import subprocess

here   = os.path.dirname(sys.executable if getattr(sys, "frozen", False) else os.path.abspath(__file__))
target = os.path.join(here, "KoKoFish", "KoKoFish.exe")

if not os.path.isfile(target):
    import ctypes
    ctypes.windll.user32.MessageBoxW(
        0,
        f"Could not find KoKoFish\\KoKoFish.exe next to this launcher.\n\nExpected:\n{target}",
        "KoKoFish — Launch Error",
        0x10,  # MB_ICONERROR
    )
    sys.exit(1)

subprocess.Popen([target], cwd=os.path.dirname(target))
