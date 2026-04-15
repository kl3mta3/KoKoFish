# Getting Started

## System Requirements

- **OS**: Windows 10 or later (64-bit)
- **Python**: 3.12 — the launcher installs this automatically if missing
- **Internet**: Required on first run to download models (Kokoro ~300 MB, Qwen ~400 MB). Subsequent launches are instant.
- **GPU**: Optional but recommended for Fish-Speech engines. See [TTS Engines & Hardware](TTS-Engines-and-Hardware) for specifics.

---

## Installation

### Easy Install (KoKoFish-Lite)

The Lite release is a small zip (~20 MB). It downloads everything it needs on first run.

1. Download and extract **KoKoFish-Lite**
2. Run `KoKoFish.exe`
3. If Python 3.12 is not found on your system, the launcher will offer to install it automatically
4. The app creates a virtual environment, installs all packages, then downloads the Kokoro voice model and Qwen AI model
5. First run takes 5–15 minutes depending on your connection speed. Subsequent launches open in seconds.

> **Note:** Do not close the setup window while installation is in progress.

---

### Full Offline Install (KoKoFish-Full)

The Full release (~1.2 GB) includes all Python packages, the Kokoro model, the Qwen 0.5B model, and FFmpeg pre-bundled. After extracting, the only thing the launcher needs to install is Python 3.12 (if not already present) and create the virtual environment — everything else is already there.

1. Download and extract **KoKoFish-Full**
2. Run `KoKoFish.exe`
3. If Python 3.12 is missing, the launcher installs it automatically
4. Virtual environment creation and package installation completes in under a minute (all offline)
5. The app opens immediately after setup

---

### Developer / Advanced Install

For running directly from source:

```
git clone https://github.com/kl3mta3/KoKoFish.git
cd KoKoFish
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python main.py
```

> Python 3.12 is required. PyTorch installs as CPU-only by default. To enable CUDA, use the CUDA toggle in Settings after first launch.

---

## First Launch

On first launch you will see a splash screen while the app initializes. What gets downloaded:

| Item | Size | When |
|------|------|------|
| Kokoro voice model | ~300 MB | First launch |
| Qwen 0.5B LLM | ~400 MB | First launch |
| FFmpeg | ~100 MB | First launch (if not bundled) |
| Fish-Speech 1.4 | ~1.5 GB | First time you switch to that engine |
| Fish-Speech 1.5 | ~1.5 GB | First time you switch to that engine |
| Whisper model | 50 MB–3 GB | First time you use STT (size depends on selected model) |

All downloads happen automatically in the background with progress shown on the splash screen.

---

## Choosing an Engine

When the app opens, go to **Settings** and choose your TTS engine:

- **Kokoro** — Best for quick, lightweight audiobook creation. No GPU needed. 54 preset voices. Fast.
- **Fish-Speech 1.4** — Adds voice cloning. Good quality. Requires a GPU for practical use.
- **Fish-Speech 1.5** — Higher quality output with voice cloning. Same GPU requirements as 1.4.

If you are just getting started, **Kokoro** is the recommended choice. It works well on any modern PC, sounds great, and has voices in 10 languages.

---

## What Stays Where

```
KoKoFish/
  KoKoFish.exe       ← run this
  main.py
  settings.json      ← your saved preferences
  models/            ← downloaded AI models (Kokoro, Qwen, etc.)
  voices/            ← your cloned voice profiles (Fish-Speech)
  bin/               ← FFmpeg
  venv/              ← Python environment (created on first run)
```

> **Distributing or moving the app:** You can zip the folder and share it, but exclude `venv/` — virtual environments have hardcoded paths and won't work on other machines. The launcher recreates it automatically on any machine.
