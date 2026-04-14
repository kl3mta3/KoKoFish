"""
KoKoFish — Voice Manager.

Manages voice profiles stored in the /voices directory.
Handles zero-shot voice cloning by saving reference WAV files and
pre-computing VQ tokens.
"""

import logging
import os
import shutil
from typing import Dict, List, Optional

import numpy as np

logger = logging.getLogger("KoKoFish.voices")


class VoiceManager:
    """
    Manages voice profiles for TTS voice cloning.

    Each voice profile is a subdirectory under voices_dir containing:
        - reference.wav  — the reference audio clip
        - tokens.npy     — pre-computed VQ token indices (optional, cached)
        - meta.txt       — optional metadata (prompt text, description)
    """

    def __init__(self, voices_dir: str):
        self.voices_dir = voices_dir
        os.makedirs(voices_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Listing & querying
    # ------------------------------------------------------------------

    def list_voices(self) -> List[str]:
        """Return sorted list of voice profile names."""
        if not os.path.isdir(self.voices_dir):
            return []
        voices = []
        for name in sorted(os.listdir(self.voices_dir)):
            profile_dir = os.path.join(self.voices_dir, name)
            if os.path.isdir(profile_dir):
                ref_wav = os.path.join(profile_dir, "reference.wav")
                if os.path.isfile(ref_wav):
                    voices.append(name)
        return voices

    def get_voice(self, name: str) -> Optional[Dict]:
        """
        Get voice profile details.

        Returns dict with keys:
            name:         str
            wav_path:     str — path to reference.wav
            tokens_path:  str or None — path to tokens.npy (if pre-computed)
            prompt_text:  str — reference audio transcript (if available)
        """
        profile_dir = os.path.join(self.voices_dir, name)
        ref_wav = os.path.join(profile_dir, "reference.wav")

        if not os.path.isfile(ref_wav):
            return None

        tokens_path = os.path.join(profile_dir, "tokens.npy")
        if not os.path.isfile(tokens_path):
            tokens_path = None

        prompt_text = ""
        meta_path = os.path.join(profile_dir, "meta.txt")
        if os.path.isfile(meta_path):
            try:
                with open(meta_path, "r", encoding="utf-8") as f:
                    prompt_text = f.read().strip()
            except Exception:
                pass

        return {
            "name": name,
            "wav_path": ref_wav,
            "tokens_path": tokens_path,
            "prompt_text": prompt_text,
        }

    def get_voice_names(self) -> List[str]:
        """Return list of voice names for dropdown population."""
        return ["Default (Random)"] + self.list_voices()

    # ------------------------------------------------------------------
    # Voice creation (zero-shot cloning)
    # ------------------------------------------------------------------

    def clone_voice(
        self,
        name: str,
        reference_wav_path: str,
        tts_engine=None,
        prompt_text: str = "",
    ) -> str:
        """
        Create a new voice profile from a reference WAV file.

        This is ZERO-SHOT only — the reference WAV is used directly
        for conditioning at inference time. No fine-tuning occurs.

        Args:
            name:               Name for the voice profile.
            reference_wav_path: Path to the 30–180 second reference WAV.
            tts_engine:         Optional TTSEngine to pre-compute tokens.
            prompt_text:        Transcript of the reference audio.

        Returns:
            Path to the created voice profile directory.
        """
        # Sanitize name
        safe_name = "".join(
            c for c in name if c.isalnum() or c in (" ", "-", "_")
        ).strip()
        if not safe_name:
            raise ValueError("Invalid voice name.")

        profile_dir = os.path.join(self.voices_dir, safe_name)
        os.makedirs(profile_dir, exist_ok=True)

        # Pre-process reference audio: denoise, normalize, trim silence, then save
        dest_wav = os.path.join(profile_dir, "reference.wav")
        try:
            from utils import preprocess_reference_audio
            from pydub import AudioSegment

            # First pass: format conversion + 180s trim
            audio = AudioSegment.from_file(reference_wav_path)
            if len(audio) > 180000:
                logger.info("Audio is %.1fs — trimming to first 180s.", len(audio) / 1000)
                audio = audio[:180000]
            tmp_wav = dest_wav + ".tmp.wav"
            audio.export(tmp_wav, format="wav")

            # Second pass: normalize + trim silence + denoise
            processed = preprocess_reference_audio(tmp_wav, denoise=True)
            if processed != tmp_wav:
                import shutil as _shutil
                _shutil.move(processed, dest_wav)
                try:
                    os.remove(tmp_wav)
                except OSError:
                    pass
            else:
                os.replace(tmp_wav, dest_wav)

            logger.info("Preprocessed reference WAV saved: %s", dest_wav)
        except ImportError:
            shutil.copy2(reference_wav_path, dest_wav)
            logger.info("Copied reference WAV (no preprocessing): %s", dest_wav)
        except Exception as e:
            logger.error("Audio processing failed: %s. Rolling back.", e)
            shutil.rmtree(profile_dir)
            raise ValueError(f"Could not process audio file: {e}")

        # Save prompt text metadata
        if prompt_text:
            meta_path = os.path.join(profile_dir, "meta.txt")
            with open(meta_path, "w", encoding="utf-8") as f:
                f.write(prompt_text)

        # Pre-compute VQ tokens if TTS engine is available and loaded
        if tts_engine is not None and tts_engine.is_loaded:
            try:
                tokens = tts_engine.encode_reference(dest_wav)
                tokens_path = os.path.join(profile_dir, "tokens.npy")
                np.save(tokens_path, tokens)
                logger.info("Pre-computed VQ tokens: %s", tokens_path)
                
                # Tag the voice profile with the engine version that generated the tokens
                # Since v1.4 and v1.5 math is fundamentally incompatible, we must separate them.
                from settings import Settings
                active_settings = Settings.load()
                engine_sig = getattr(active_settings, 'engine', 'fish14')
                with open(os.path.join(profile_dir, "engine.txt"), "w") as ef:
                    ef.write(engine_sig)
            except Exception as exc:
                logger.warning(
                    "Could not pre-compute tokens (will encode at runtime): %s",
                    exc,
                )

        logger.info("Voice profile created: %s", safe_name)
        return profile_dir

    # ------------------------------------------------------------------
    # Voice management
    # ------------------------------------------------------------------

    def delete_voice(self, name: str) -> bool:
        """Delete a voice profile."""
        profile_dir = os.path.join(self.voices_dir, name)
        if os.path.isdir(profile_dir):
            shutil.rmtree(profile_dir)
            logger.info("Deleted voice profile: %s", name)
            return True
        return False

    def rename_voice(self, old_name: str, new_name: str) -> bool:
        """Rename a voice profile."""
        old_dir = os.path.join(self.voices_dir, old_name)
        safe_name = "".join(
            c for c in new_name if c.isalnum() or c in (" ", "-", "_")
        ).strip()
        new_dir = os.path.join(self.voices_dir, safe_name)

        if os.path.isdir(old_dir) and not os.path.exists(new_dir):
            os.rename(old_dir, new_dir)
            logger.info("Renamed voice: %s → %s", old_name, safe_name)
            return True
        return False

    def voice_exists(self, name: str) -> bool:
        """Check if a voice profile exists."""
        return os.path.isdir(os.path.join(self.voices_dir, name))
