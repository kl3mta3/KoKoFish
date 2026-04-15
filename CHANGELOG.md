# Changelog

All notable changes to KoKoFish are documented here.

---

## [1.4.6] - 2026-04-14

### Added
- Full GitHub Wiki with documentation for every tab, feature, and model
- GitHub issue templates for bug reports and feature requests
- Python 3.12 installer bundled in `bin/` for fully offline first-run setup
- CUDA PyTorch install now checks local `packages/` folder first — toggling CUDA on/off is instant in the Full release without re-downloading
- Gemma 3 1B Heretic Abliterated and Gemma 3 4B Abliterated LLM models added
- Ollama backend support — use any Ollama model (e.g. `huihui_ai/gemma-4-abliterated:e2b`) for all AI writing tools
- LICENSE file (MIT) with full third-party model license notices
- CHANGELOG

### Changed
- Rebranded from "TTS/STT Studio" to **Audiobook Studio**
- Python target version updated from 3.11 to 3.12 across both launchers (matches bundled cp312 PyTorch wheels)
- Launcher now shows splash screen for 15 seconds after setup completes so the app has time to load before the splash disappears
- `.setup_complete` marker only written after all packages install successfully (previously written even on failed installs)
- Removed `--no-index` from pip install steps — local `packages/` folder is tried first with PyPI as fallback
- Gemma 3 1B and 4B switched from gated bartowski repos to ungated lmstudio-community mirrors (fixes 401 download errors)
- `packages/` folder cleaned up — removed duplicate wheel versions and unused packages (IPython, TensorBoard, Matplotlib stacks)
- README fully rewritten with 2-column screenshot grid, accurate feature descriptions, and llama-cpp-python install note

### Fixed
- `local_dir_use_symlinks` deprecation warning removed from HuggingFace Hub downloads
- Fish-Speech path in settings no longer hardcoded to developer machine path
- Lite version no longer loads Fish-Speech from the main repo path
- `reportlab` and `huggingface_hub` added to auto-install dependency check so missing packages are caught on startup

---

## [1.4.5] - 2026-04

### Added
- Listen Lab — transcribe audio, translate, and re-read in a target language in one pipeline
- File Lab — document format conversion (.txt, .pdf, .docx, .epub) and audio conversion (MP3, WAV, M4B, FLAC)
- Audiobook Combiner — merge multiple WAV files into a single M4B/MP3 with chapter markers
- S1 Mini and S1 Full engine support
- CUDA toggle in Settings with automatic PyTorch CUDA install

### Changed
- Tabs renamed to Labs (Speech Lab, Voice Lab, Text Lab, etc.)

---

## [1.4.0] - 2026-03

### Added
- Translation support in Text Editor (12+ languages via local LLM)
- Tone rewriting in Text Editor (9 tone options)
- Assisted Flow pipeline — grammar → translation → TTS enhancement per playlist item
- Qwen 2.5 0.5B as default local LLM for all AI features
- Gemma 3 1B and 4B model options
- Prompt Lab — direct chat with local LLM

### Changed
- AI tag generation now uses Qwen instead of rule-based system
- TTS enhancement prompts are now engine-aware (different behavior for Kokoro vs Fish-Speech)

---

## [1.3.0] - 2026-02

### Added
- AI tag generation and suggestion in Text Editor
- Grammar check in Text Editor
- TTS Enhancement (rewrite text for natural spoken delivery)
- Voice blending for Kokoro (mix two preset voices at any ratio)
- Per-playlist-item voice selection

---

## [1.2.0] - 2026-01

### Added
- EPUB support — chapters extracted in spine order
- Kokoro multilanguage voices (Japanese, Spanish, French, Hindi, Italian, Portuguese, Mandarin, Korean)
- Text Editor with tag insertion for Fish-Speech emotion and effect tags
- Silent mode — generate audio without playing it aloud
- Save as MP3 or WAV

### Fixed
- Performance improvements for sentence-level streaming
- Large file handling for long chapters

---

## [1.1.0] - 2025-12

### Added
- Fish-Speech 1.5 engine support
- Voice Lab — record or upload reference audio to create cloned voice profiles
- Whisper speech-to-text (Text Lab)
- Export transcription as .txt, .docx, or .pdf
- Auto-download for Fish-Speech models on first engine selection

---

## [1.0.0] - 2025-11

### Initial Release
- Kokoro TTS engine with 54 preset voices
- Fish-Speech 1.4 TTS engine with voice cloning
- Drag and drop .txt, .pdf, .docx files
- Real-time sentence-by-sentence audio playback
- Auto-setup launcher — installs Python, venv, and dependencies on first run
- FFmpeg auto-download
