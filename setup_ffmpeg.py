"""
KoKoFish — ffmpeg auto-download helper.

Downloads ffmpeg essentials and extracts ffmpeg.exe to bin/.
Called by the launcher during first-run setup.
"""

import os
import shutil
import sys
import urllib.request
import zipfile

APP_DIR = os.path.dirname(os.path.abspath(__file__))
BIN_DIR = os.path.join(APP_DIR, "bin")
FFMPEG_EXE = os.path.join(BIN_DIR, "ffmpeg.exe")
FFMPEG_URL = "https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-essentials.zip"


def download_ffmpeg():
    os.makedirs(BIN_DIR, exist_ok=True)
    
    FFPROBE_EXE = os.path.join(BIN_DIR, "ffprobe.exe")

    if os.path.isfile(FFMPEG_EXE) and os.path.isfile(FFPROBE_EXE):
        print("  ffmpeg toolkit already fully present.")
        return True

    zip_path = os.path.join(BIN_DIR, "ffmpeg_download.zip")

    try:
        print("  Downloading ffmpeg toolkit (~80 MB)...")
        urllib.request.urlretrieve(FFMPEG_URL, zip_path)

        print("  Extracting binaries...")
        with zipfile.ZipFile(zip_path, "r") as zf:
            for name in zf.namelist():
                if name.endswith("bin/ffmpeg.exe") or name.endswith("bin\\ffmpeg.exe"):
                    zf.extract(name, BIN_DIR)
                    shutil.copy2(os.path.join(BIN_DIR, name), FFMPEG_EXE)
                elif name.endswith("bin/ffprobe.exe") or name.endswith("bin\\ffprobe.exe"):
                    zf.extract(name, BIN_DIR)
                    shutil.copy2(os.path.join(BIN_DIR, name), FFPROBE_EXE)

        # Cleanup zip and extracted artifact directory structure
        os.remove(zip_path)
        for d in os.listdir(BIN_DIR):
            full = os.path.join(BIN_DIR, d)
            if os.path.isdir(full):
                shutil.rmtree(full)

        if os.path.isfile(FFMPEG_EXE) and os.path.isfile(FFPROBE_EXE):
            print("  ffmpeg toolkit ready.")
            return True
        else:
            print("  ERROR: required binaries not found in download.")
            return False

    except Exception as e:
        print(f"  ERROR: {e}")
        # Cleanup partial download
        if os.path.isfile(zip_path):
            os.remove(zip_path)
        return False


if __name__ == "__main__":
    success = download_ffmpeg()
    sys.exit(0 if success else 1)
