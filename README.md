# KoKoFish — Audiobook Studio

<p align="center">
  <img src="https://github.com/kl3mta3/KoKoFish/blob/main/Images/icon.png" alt="KoKoFish Icon" width="300"/>
</p>

<h1 align="center">KoKoFish</h1>

KoKoFish is a free, offline-first Audiobook Studio for Windows — built for writers, readers, and anyone who wants to bring text to life.

Turn any document into a full audiobook with a single drag and drop. Choose from 54 built-in voices across 10 languages, clone your own voice, blend two voices together, or let the AI rewrite and enhance your text before it ever hits the speaker. Transcribe audio back to text, translate between languages, convert file formats, and combine chapters into a finished M4B — all without sending a single byte to the cloud.

Everything runs locally on your machine. No subscriptions. No API keys. No internet required after the first setup.

---

## Features

**Read Aloud (TTS)**

- Drag and drop .txt, .pdf, .docx, and .epub files into a playlist
- Real-time sentence-by-sentence audio playback as text is generated
- Adjustable speed, volume, and cadence controls
- Silent mode — generate audio files without playing them aloud
- Save output as MP3 or WAV

**Multiple AI Engines**

- Kokoro — 82M parameter ONNX model, fast CPU inference, 54 built-in preset voices across multiple languages
- Fish-Speech 1.4 — compact model with voice cloning support
- Fish-Speech 1.5 — higher quality output with voice cloning support

**Voice Cloning (Fish-Speech only)**

- Record or upload a 15–180 second reference audio clip
- Voice profiles are stored per engine so 1.4 and 1.5 libraries stay separate
- Pre-computes VQ tokens for faster inference at playback time

**Voice Lab**

- Create, rename, and delete voice profiles
- Reference audio is automatically trimmed to 180 seconds
- Disabled automatically when Kokoro engine is active

**Speech to Text (STT)**

- Transcribe audio files using Whisper
- Supports .wav, .mp3, .m4a, .flac, and .ogg
- Export transcription as .txt, .docx, or .pdf

**Script Lab — Multi-Voice Audiobook Production**

- Write or generate a multi-character script using `[CharacterName] dialogue` format
- Assign a voice to each character in a reusable Character Profile
- AI script generation from raw prose — automatically identifies speakers, strips attribution text, and formats the script (with optional emotion tags for Fish-Speech)
- **Find in Script** — scan any tagged script and auto-populate the character list with every name found
- **Enhance Script** — LLM pass over the finished script to improve conversation flow, natural delivery, and emotional continuity between lines
- Play the full script with each character's assigned voice, switching voices per segment automatically
- Export the script as a `.txt` transcript or the audio as a `.wav`

**AI Writing Tools (powered by local LLM)**

- Grammar check and correction
- Tone rewriting (Casual, Formal, Dramatic, and more)
- Translation to any language
- TTS enhancement — rewrites text for more natural spoken delivery
- AI tag suggester — adds pacing and emotion tags for Fish-Speech and Kokoro
- **Assisted Flow** — an optional per-item pipeline that automatically runs grammar → translation → TTS enhancement before each item plays, so your playlist just sounds right
- All AI features run locally using a small on-device model (no cloud, no API keys)

**Supported AI Models for Writing Tools**

- Qwen 2.5 0.5B (default, ~400 MB) — fastest, lowest memory
- Gemma 3 1B (~700 MB)
- Gemma 3 1B Heretic Abliterated (~900 MB)
- Gemma 3 4B Abliterated (~2.5 GB)
- Ollama models (e.g. huihui_ai/gemma-4-abliterated:e2b) — requires Ollama installed

**File Format Support**

- .txt — plain text
- .pdf — text extraction via pdfplumber
- .docx — Microsoft Word documents
- .epub — ebook format, chapters extracted in spine order

---

## Screenshots

<table>
  <tr>
    <td align="center"><b>Speech Lab — Kokoro</b></td>
    <td align="center"><b>Speech Lab — Fish-Speech</b></td>
  </tr>
  <tr>
    <td><img src="https://github.com/kl3mta3/KoKoFish/blob/main/Images/Speech_Lab-Kokoro.png" alt="Speech Lab Kokoro" width="400"/></td>
    <td><img src="https://github.com/kl3mta3/KoKoFish/blob/main/Images/Speech_Lab-Fish-Speech.png" alt="Speech Lab Fish-Speech" width="400"/></td>
  </tr>
  <tr>
    <td align="center"><b>Text Editor</b></td>
    <td align="center"><b>Voice Lab</b></td>
  </tr>
  <tr>
    <td><img src="https://github.com/kl3mta3/KoKoFish/blob/main/Images/Text_Editor.png" alt="Text Editor" width="400"/></td>
    <td><img src="https://github.com/kl3mta3/KoKoFish/blob/main/Images/Voice_Lab.png" alt="Voice Lab" width="400"/></td>
  </tr>
  <tr>
    <td align="center"><b>Text Lab (STT)</b></td>
    <td align="center"><b>File Lab</b></td>
  </tr>
  <tr>
    <td><img src="https://github.com/kl3mta3/KoKoFish/blob/main/Images/Text_Lab.png" alt="Text Lab" width="400"/></td>
    <td><img src="https://github.com/kl3mta3/KoKoFish/blob/main/Images/File_Lab.png" alt="File Lab" width="400"/></td>
  </tr>
  <tr>
    <td align="center"><b>Listen Lab</b></td>
    <td align="center"><b>Prompt Lab</b></td>
  </tr>
  <tr>
    <td><img src="https://github.com/kl3mta3/KoKoFish/blob/main/Images/Listen_Lab.png" alt="Listen Lab" width="400"/></td>
    <td><img src="https://github.com/kl3mta3/KoKoFish/blob/main/Images/Prompt_Lab.png" alt="Prompt Lab" width="400"/></td>
  </tr>
  <tr>
    <td align="center"><b>TTS Enhancement</b></td>
    <td align="center"><b>Tone Rewrite</b></td>
  </tr>
  <tr>
    <td><img src="https://github.com/kl3mta3/KoKoFish/blob/main/Images/Enhanced_TTS1.png" alt="TTS Enhancement" width="400"/></td>
    <td><img src="https://github.com/kl3mta3/KoKoFish/blob/main/Images/Retone1.png" alt="Tone Rewrite" width="400"/></td>
  </tr>
  <tr>
    <td align="center"><b>Translation</b></td>
    <td align="center"><b>AI Tag Generation</b></td>
  </tr>
  <tr>
    <td><img src="https://github.com/kl3mta3/KoKoFish/blob/main/Images/Translate1.png" alt="Translation" width="400"/></td>
    <td><img src="https://github.com/kl3mta3/KoKoFish/blob/main/Images/Generate_Style_Tags.png" alt="Generate Style Tags" width="400"/></td>
  </tr>
  <tr>
    <td align="center"><b>TTS Settings</b></td>
    <td align="center"><b>LLM Model Options</b></td>
  </tr>
  <tr>
    <td><img src="https://github.com/kl3mta3/KoKoFish/blob/main/Images/TTS_Options.png" alt="TTS Options" width="400"/></td>
    <td><img src="https://github.com/kl3mta3/KoKoFish/blob/main/Images/LLM_Options.png" alt="LLM Options" width="400"/></td>
  </tr>
</table>

---

## Requirements

- Windows 10 or later
- Python 3.12 — automatically installed by the launcher if not found
- FFmpeg — automatically downloaded on first run
- NVIDIA GPU with CUDA is optional but significantly speeds up Fish-Speech engines

---

## Installation

### Easy Install (Lite)
1. Download and extract **KoKoFish-Lite**
2. Run `KoKoFish.exe`
3. The launcher checks for Python 3.12, installs it if missing, then sets up the app
4. Kokoro and Qwen models download automatically on first run (~700 MB total)

### Full Offline Install
1. Download and extract **KoKoFish-Full**
2. Run `KoKoFish.exe`
3. Python 3.12 is still required (launcher installs it if missing) — everything else is already included

```
The Full release includes PyTorch CPU wheels, Kokoro model, Qwen 0.5B model, and FFmpeg.
Fish-Speech models download automatically the first time you switch to those engines.
```

### Advanced / Developer Install
1. Clone the repository:

```
git clone https://github.com/kl3mta3/KoKoFish.git
cd KoKoFish
```

2. Create a virtual environment and install dependencies:

```
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

3. Run the application:

```
python main.py
```

On first launch the app will automatically download anything missing:

- FFmpeg (~100 MB, one-time)
- Kokoro model files (~300 MB, one-time)
- Qwen 0.5B model (~400 MB, one-time)
- Fish-Speech 1.4 source + checkpoints (~1.5 GB, one-time, only if you use that engine)
- Fish-Speech 1.5 source + checkpoints (~1.5 GB, one-time, only if you use that engine)

**AI Writing Tools (llama-cpp-python):**
The AI features (grammar check, tone rewriting, translation, tag generation, Prompt Lab) require `llama-cpp-python`. The app will prompt you to install it from Settings when you first use an AI feature. Pre-built CPU wheels install automatically with no compiler needed. If you want CUDA-accelerated LLM inference, enable CUDA in Settings first — the correct build will be selected automatically.

---

## Switching Engines

Select an engine from the Settings tab. The app saves your selection and restarts automatically. Each engine has its own voice library:

- Fish-Speech 1.4 voices are stored in `voices/fish14/`
- Fish-Speech 1.5 voices are stored in `voices/fish15/`
- Kokoro uses built-in preset voices — no voice library needed

---

## CUDA Support

CUDA is optional and only applies to Fish-Speech engines. Enabling it requires a compatible NVIDIA GPU and will trigger an automatic installation of the CUDA-enabled PyTorch build.

To enable CUDA, go to Settings and toggle the CUDA option. The app will restart and use GPU acceleration for Fish-Speech inference.

---

## Project Structure

```
KoKoFish/
  launcher.py          -- auto-setup and launch entry point
  main.py              -- application startup and splash screen
  ui.py                -- main window, tabs, and event handlers
  tts_engine.py        -- Fish-Speech TTS engine wrapper
  kokoro_engine.py     -- Kokoro ONNX engine with sentence-level streaming
  stt_engine.py        -- Whisper speech-to-text engine
  tag_suggester.py     -- AI writing tools (grammar, tone, translate, tags)
  voice_manager.py     -- voice profile creation and management
  settings.py          -- persistent settings backed by settings.json
  script_engine.py     -- Script Lab: profile management, script parsing, AI tagging
  utils.py             -- file readers, audio export, FFmpeg utilities
  bin/                 -- bundled ffmpeg.exe
  scripts/
    profiles/          -- character profiles (JSON) for Script Lab
  models/              -- downloaded LLM and Kokoro model files
  voices/
    fish14/            -- voice profiles for Fish-Speech 1.4
    fish15/            -- voice profiles for Fish-Speech 1.5
  fish-speech/         -- Fish-Speech repository (downloaded on first use)
  packages/            -- pre-downloaded Python wheels (Full release only)
```

---

## Acknowledgments

- [Fish-Speech](https://github.com/fishaudio/fish-speech) by fishaudio
- [Kokoro TTS](https://github.com/hexgrad/kokoro) by hexgrad
- [kokoro-onnx](https://github.com/thewh1teagle/kokoro-onnx) by thewh1teagle
- [Whisper](https://github.com/openai/whisper) by OpenAI
- [pdfplumber](https://github.com/jsvine/pdfplumber)
- [ebooklib](https://github.com/aerkalov/ebooklib)

---

## License

See LICENSE for details.
