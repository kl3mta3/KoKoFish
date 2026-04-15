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

## [Pre-release Development History]

Prior to v1.4.6 the project was in active development without a formal changelog. Features built during that period include:

- Kokoro TTS with 54 preset voices across 10 languages
- Fish-Speech 1.4, 1.5, S1 Mini, and S1 Full engine support
- Voice cloning and Voice Lab
- Voice blending (Kokoro)
- Whisper speech-to-text (Text Lab)
- File Lab with document and audio format conversion
- Audiobook Combiner (multi-chapter M4B export)
- Listen Lab with transcribe, translate, and re-read pipeline
- Text Editor with AI grammar check, tone rewriting, TTS enhancement, and tag generation
- Assisted Flow per-playlist-item AI preprocessing pipeline
- Prompt Lab (local LLM chat)
- CUDA toggle with automatic PyTorch install
- Auto-setup launcher with Python detection and venv creation
