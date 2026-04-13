# FishTalk

FishTalk is a desktop text-to-speech application built for Windows. It supports multiple AI engines, voice cloning, real-time audio playback, and speech-to-text transcription. It is designed to work entirely offline once the models are downloaded.

---

## Features

**Text to Speech**

- Drag and drop .txt, .pdf, .docx, and .epub files into a playlist
- Real-time sentence-by-sentence audio playback as text is generated
- Adjustable speed, volume, and cadence controls
- Work Silent mode to generate audio files without playing them aloud
- Save output as MP3 or WAV via a standard save dialog

**Multiple AI Engines**

- Fish-Speech 1.4 -- smaller model, good for quick generation with voice cloning
- Fish-Speech 1.5 -- higher quality output with voice cloning support
- Kokoro Fast CPU -- 82M parameter ONNX model optimized for real-time CPU inference with 54 built-in preset voices

**Voice Cloning (Fish-Speech only)**

- Record or upload a 15-30 second reference audio clip, Longer audio will be trimmed to 30 seconds
- Voice profiles are stored per engine so 1.4 and 1.5 voice libraries stay separate
- Pre-computes VQ tokens for faster inference at playback time

**Voice Lab**

- Create, rename, and delete voice profiles
- Reference audio is automatically trimmed to 30 seconds to prevent out-of-memory errors
- Voice Lab is automatically disabled when the Kokoro engine is active

**Speech to Text**

- Transcribe audio files using Whisper
- Supports .wav, .mp3, .m4a, .flac, and .ogg
- Export transcription as .txt, .docx, or .pdf

**File Format Support**

- .txt -- plain text with soft line-wrap handling
- .pdf -- text extraction via pdfplumber
- .docx -- Microsoft Word documents
- .epub -- ebook format, chapters extracted in spine order

---

## Requirements

- Windows 10 or later
- Python 3.11 or later. (Will install if not found.)
- FFmpeg (Will install if not found.)
- Fish-Speech repository (Will install if not found.) cloned into the project directory for Fish-Speech engines
- NVIDIA GPU with CUDA support is optional but improves Fish-Speech generation speed significantly

---

## Setup

#Easy Install:
1. Download and extract FiahTalk-Lite
2. Run FishTalk.exe.  This will install all needed dependencies


#Offline Install:
1. Download and extract FishTalk-Kokoro_Only
2. Download and extract Model-fish-speech-1.4 inside the FishTalk folder
3. Download and extract Model-fish-speech-1.5 inside the FishTalk folder
4. Run FishTalk.exe. It will still install Python from the included installer
(If you miss any of these and have an internet connection it will grab them.)

#Advanced Install:
1. Clone this repository:

```
git clone https://github.com/kl3mta3/FishTalk.git
cd FishTalk
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

On first launch the app will automatically download anything that is missing:

- FFmpeg (~75 MB, one-time)
- Kokoro model files (~115 MB, one-time)
- Fish-Speech v1.4.3 source code (~20 MB, one-time)
- Fish-Speech 1.4 model checkpoints (~1.5 GB, one-time)
- Fish-Speech 1.5 model checkpoints (~1.5 GB, one-time)

All downloads show progress on the startup screen. Subsequent launches are instant. If you are distributing the app as a zip, include the fish-speech/ and bin/ folders to skip these downloads entirely.



---

## Switching Engines

Select an engine from the Settings tab. The application will save the selection and prompt for a restart. Each engine has its own voice library:

- Fish-Speech 1.4 voices are stored in voices/fish14/
- Fish-Speech 1.5 voices are stored in voices/fish15/
- Kokoro uses built-in preset voices and does not use the voice library

---

## CUDA Support

CUDA is optional and only applies to the Fish-Speech engines. Enabling it requires a compatible NVIDIA GPU and will prompt an automatic installation of the CUDA-enabled PyTorch build.

To enable CUDA, go to Settings and toggle the CUDA option. The application will restart and use GPU acceleration for all Fish-Speech inference.

---

## Project Structure

```
FishTalk/
  main.py              -- application entry point and startup logic
  ui.py                -- main window, tabs, and event handlers
  tts_engine.py        -- Fish-Speech TTS engine wrapper
  kokoro_engine.py     -- Kokoro ONNX engine with sentence-level streaming
  stt_engine.py        -- Whisper speech-to-text engine
  voice_manager.py     -- voice profile creation and management
  settings.py          -- persistent settings backed by settings.json
  utils.py             -- file readers, audio export, FFmpeg utilities
  voices/
    fish14/            -- voice profiles for Fish-Speech 1.4
    fish15/            -- voice profiles for Fish-Speech 1.5
  kokoro_models/
    kokoro-v1.0.int8.onnx
    voices-v1.0.bin
  fish-speech/         -- Fish-Speech repository (user-provided)
  bin/                 -- optional location for ffmpeg.exe
```

---

## Acknowledgments

- Fish-Speech by fishaudio (https://github.com/fishaudio/fish-speech)
- Kokoro TTS by hexgrad (https://github.com/hexgrad/kokoro)
- kokoro-onnx by thewh1teagle (https://github.com/thewh1teagle/kokoro-onnx)
- Whisper by OpenAI (https://github.com/openai/whisper)
- pdfplumber (https://github.com/jsvine/pdfplumber)
- ebooklib (https://github.com/aerkalov/ebooklib)

---

## License

See LICENSE for details.
