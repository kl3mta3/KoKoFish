"""
FishTalk — User Interface.

CustomTkinter dark-mode GUI with 4 tabs:
  Tab 1: Read Aloud (TTS)
  Tab 2: Transcribe (STT)
  Tab 3: Voice Lab
  Tab 4: Settings
"""

import logging
import os
import threading
import tkinter as tk
from tkinter import filedialog, messagebox
from typing import Optional

import customtkinter as ctk
import numpy as np

from settings import Settings, detect_cuda, get_bundled_fish_speech_path, validate_fish_speech_path
from cuda_setup import has_nvidia_gpu, get_nvidia_gpu_name, is_cuda_torch_installed, install_cuda_pytorch, revert_to_cpu_pytorch
from kokoro_engine import KOKORO_VOICES, DEFAULT_VOICE, DEFAULT_VOICE_DISPLAY, install_kokoro, _is_kokoro_installed
from utils import (
    get_ram_usage,
    get_vram_usage,
    is_ffmpeg_available,
    read_file,
    export_mp3,
    export_txt,
    export_docx,
    export_pdf,
)
from voice_manager import VoiceManager

import re
import time

logger = logging.getLogger("FishTalk.ui")

# ---------------------------------------------------------------------------
# App paths
# ---------------------------------------------------------------------------
APP_DIR = os.path.dirname(os.path.abspath(__file__))
AUDIO_TEMP_DIR = os.path.join(APP_DIR, "temp", "audio")

# ---------------------------------------------------------------------------
# Color palette
# ---------------------------------------------------------------------------
COLORS = {
    "bg_dark":       "#0f0f1a",
    "bg_card":       "#1a1a2e",
    "bg_card_hover": "#222240",
    "bg_input":      "#16213e",
    "accent":        "#4361ee",
    "accent_hover":  "#3a56d4",
    "accent_light":  "#6c83f7",
    "success":       "#06d6a0",
    "warning":       "#ffd166",
    "danger":        "#ef476f",
    "text_primary":  "#e8e8f0",
    "text_secondary":"#9a9ab0",
    "text_muted":    "#5a5a7a",
    "border":        "#2a2a4a",
}

FONT_FAMILY = "Segoe UI"


# ============================================================================
# MAIN UI CLASS
# ============================================================================

class FishTalkUI:
    """Builds and manages the entire FishTalk user interface."""

    def __init__(
        self,
        root: ctk.CTk,
        settings: Settings,
        tts_engine,
        stt_engine,
        voice_manager: VoiceManager,
    ):
        self.root = root
        self.settings = settings
        self.tts = tts_engine
        self.stt = stt_engine
        self.voices = voice_manager

        # Playback state
        self._playlist_items = []   # List of dicts: {name, text, path}
        self._current_playing = -1
        self._is_playing = False
        self._is_paused = False
        self._playback_stream = None

        # Audio playback
        self._audio_data = None
        self._audio_sr = None
        self._play_position = 0

        # Item preview playback (play completed items from the playlist)
        self._preview_idx = -1
        self._preview_paused = False
        self._preview_stream = None
        self._preview_audio = None
        self._preview_pos = 0
        self._preview_sr = None

        # Most recently completed WAV (used by manual Save MP3)
        self._last_wav_path = None

        # Push persisted generation settings into engine on startup
        self._sync_gen_settings_to_engine()

        self._build_ui()
        self._start_ram_monitor()
        threading.Thread(
            target=self._cleanup_old_temp_audio,
            daemon=True,
            name="TempCleanup",
        ).start()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self):
        """Build the full tabbed interface."""
        self.root.configure(fg_color=COLORS["bg_dark"])

        # Header
        header = ctk.CTkFrame(self.root, fg_color=COLORS["bg_dark"], height=60)
        header.pack(fill="x", padx=20, pady=(15, 5))
        header.pack_propagate(False)

        title_label = ctk.CTkLabel(
            header,
            text="🐟  FishTalk",
            font=(FONT_FAMILY, 28, "bold"),
            text_color=COLORS["accent_light"],
        )
        title_label.pack(side="left", pady=10)

        subtitle = ctk.CTkLabel(
            header,
            text="TTS/STT Studio",
            font=(FONT_FAMILY, 13),
            text_color=COLORS["text_secondary"],
        )
        subtitle.pack(side="left", padx=(12, 0), pady=(18, 10))

        # Tab view
        self.tabview = ctk.CTkTabview(
            self.root,
            fg_color=COLORS["bg_card"],
            segmented_button_fg_color=COLORS["bg_input"],
            segmented_button_selected_color=COLORS["accent"],
            segmented_button_selected_hover_color=COLORS["accent_hover"],
            segmented_button_unselected_color=COLORS["bg_input"],
            segmented_button_unselected_hover_color=COLORS["bg_card_hover"],
            text_color=COLORS["text_primary"],
            corner_radius=12,
        )
        self.tabview.pack(fill="both", expand=True, padx=20, pady=(0, 20))

        # Create tabs
        self.tab_tts = self.tabview.add("📖  Read Aloud")
        self.tab_stt = self.tabview.add("🎙  Transcribe")
        self.tab_voices = self.tabview.add("🧬  Voice Lab")
        self.tab_settings = self.tabview.add("⚙  Settings")

        self._build_tts_tab()
        self._build_stt_tab()
        self._build_voice_lab_tab()
        self._build_settings_tab()

        # Lock Voice Lab tab when Kokoro engine is active
        if getattr(self.settings, 'engine', 'fish14') == 'kokoro':
            self._lock_voice_lab()

        # Bind tab change for memory saver
        self.tabview.configure(command=self._on_tab_changed)

    # ==================================================================
    # TAB 1: Read Aloud (TTS)
    # ==================================================================

    def _build_tts_tab(self):
        tab = self.tab_tts

        # Top row — Drop zone + controls
        top = ctk.CTkFrame(tab, fg_color="transparent")
        top.pack(fill="x", padx=10, pady=(10, 5))

        # Drop zone
        self.tts_drop = ctk.CTkFrame(
            top,
            fg_color=COLORS["bg_input"],
            border_color=COLORS["border"],
            border_width=2,
            corner_radius=10,
            height=100,
        )
        self.tts_drop.pack(fill="x", pady=(0, 10))
        self.tts_drop.pack_propagate(False)

        self.tts_drop_label = ctk.CTkLabel(
            self.tts_drop,
            text="📄  Drag & drop .txt, .pdf, .docx, or .epub files here\nor click to browse",
            font=(FONT_FAMILY, 13),
            text_color=COLORS["text_secondary"],
            justify="center",
        )
        self.tts_drop_label.pack(expand=True)
        self.tts_drop_label.bind("<Button-1>", self._tts_browse_file)

        # Register drag-and-drop
        try:
            from tkinterdnd2 import DND_FILES
            self.tts_drop.drop_target_register(DND_FILES)
            self.tts_drop.dnd_bind("<<Drop>>", self._tts_on_drop)
        except Exception as e:
            logger.warning("Drag-and-drop not available: %s", e)

        # Controls row
        controls = ctk.CTkFrame(tab, fg_color="transparent")
        controls.pack(fill="x", padx=10, pady=5)

        # Voice selector
        voice_frame = ctk.CTkFrame(controls, fg_color="transparent")
        voice_frame.pack(side="left", padx=(0, 15))

        ctk.CTkLabel(
            voice_frame,
            text="Voice",
            font=(FONT_FAMILY, 11),
            text_color=COLORS["text_secondary"],
        ).pack(anchor="w")

        # --- Voice selector (context-sensitive) ---
        is_kokoro = getattr(self.settings, 'engine', 'fish14') == 'kokoro'

        if is_kokoro:
            # Kokoro mode: show preset voice dropdown
            kokoro_display_names = list(KOKORO_VOICES.keys())
            saved_kokoro_id = getattr(self.settings, 'kokoro_voice', DEFAULT_VOICE)
            # Find display name matching saved voice ID
            saved_display = next(
                (k for k, v in KOKORO_VOICES.items() if v == saved_kokoro_id),
                kokoro_display_names[0]
            )
            self.tts_voice_var = ctk.StringVar(value=saved_display)
            self.tts_voice_menu = ctk.CTkOptionMenu(
                voice_frame,
                values=kokoro_display_names,
                variable=self.tts_voice_var,
                width=200,
                fg_color=COLORS["bg_input"],
                button_color=COLORS["accent"],
                button_hover_color=COLORS["accent_hover"],
                dropdown_fg_color=COLORS["bg_card"],
                dropdown_hover_color=COLORS["bg_card_hover"],
                font=(FONT_FAMILY, 12),
                command=self._on_kokoro_voice_change,
            )
        else:
            # Fish-Speech mode: show cloned voice profiles
            voice_names = self.voices.get_voice_names()
            self.tts_voice_var = ctk.StringVar(value=voice_names[0] if voice_names else "Default (Random)")
            self.tts_voice_menu = ctk.CTkOptionMenu(
                voice_frame,
                values=voice_names if voice_names else ["Default (Random)"],
                variable=self.tts_voice_var,
                width=180,
                fg_color=COLORS["bg_input"],
                button_color=COLORS["accent"],
                button_hover_color=COLORS["accent_hover"],
                dropdown_fg_color=COLORS["bg_card"],
                dropdown_hover_color=COLORS["bg_card_hover"],
                font=(FONT_FAMILY, 12),
            )
        self.tts_voice_menu.pack()

        # Speed slider
        speed_frame = ctk.CTkFrame(controls, fg_color="transparent")
        speed_frame.pack(side="left", padx=15)

        self.speed_label = ctk.CTkLabel(
            speed_frame,
            text="Speed: 1.0x",
            font=(FONT_FAMILY, 11),
            text_color=COLORS["text_secondary"],
        )
        self.speed_label.pack(anchor="w")

        self.speed_slider = ctk.CTkSlider(
            speed_frame,
            from_=0.5,
            to=2.0,
            number_of_steps=30,
            width=140,
            progress_color=COLORS["accent"],
            button_color=COLORS["accent_light"],
            button_hover_color=COLORS["accent"],
        )
        self.speed_slider.set(self.settings.speed)
        self.speed_slider.configure(command=self._on_speed_change)
        self.speed_slider.pack()

        # Volume slider
        vol_frame = ctk.CTkFrame(controls, fg_color="transparent")
        vol_frame.pack(side="left", padx=15)

        self.vol_label = ctk.CTkLabel(
            vol_frame,
            text=f"Volume: {self.settings.volume}%",
            font=(FONT_FAMILY, 11),
            text_color=COLORS["text_secondary"],
        )
        self.vol_label.pack(anchor="w")

        self.vol_slider = ctk.CTkSlider(
            vol_frame,
            from_=0,
            to=100,
            number_of_steps=100,
            width=140,
            progress_color=COLORS["success"],
            button_color=COLORS["success"],
        )
        self.vol_slider.set(self.settings.volume)
        self.vol_slider.configure(command=self._on_volume_change)
        self.vol_slider.pack()

        # Cadence slider
        cad_frame = ctk.CTkFrame(controls, fg_color="transparent")
        cad_frame.pack(side="left", padx=15)

        self.cad_label = ctk.CTkLabel(
            cad_frame,
            text=f"Cadence: {self.settings.cadence}%",
            font=(FONT_FAMILY, 11),
            text_color=COLORS["text_secondary"],
        )
        self.cad_label.pack(anchor="w")

        self.cad_slider = ctk.CTkSlider(
            cad_frame,
            from_=0,
            to=100,
            number_of_steps=100,
            width=140,
            progress_color=COLORS["warning"],
            button_color=COLORS["warning"],
        )
        self.cad_slider.set(self.settings.cadence)
        self.cad_slider.configure(command=self._on_cadence_change)
        self.cad_slider.pack()

        # ── Playlist header row ──────────────────────────────────────────
        playlist_header = ctk.CTkFrame(tab, fg_color="transparent")
        playlist_header.pack(fill="x", padx=15, pady=(10, 2))

        ctk.CTkLabel(
            playlist_header,
            text="📋  Playlist",
            font=(FONT_FAMILY, 14, "bold"),
            text_color=COLORS["text_primary"],
        ).pack(side="left")

        # Transport buttons — right side of header
        hdr_btn = {"font": (FONT_FAMILY, 12, "bold"), "corner_radius": 7, "height": 32, "width": 90}

        self.btn_save_mp3 = ctk.CTkButton(
            playlist_header,
            text="💾 Save",
            fg_color=COLORS["accent"],
            hover_color=COLORS["accent_hover"],
            command=self._tts_save_mp3,
            **hdr_btn,
        )
        self.btn_save_mp3.pack(side="right", padx=(4, 0))

        self.btn_stop = ctk.CTkButton(
            playlist_header,
            text="⏹ Stop",
            fg_color=COLORS["danger"],
            hover_color="#d43d62",
            command=self._tts_stop,
            **hdr_btn,
        )
        self.btn_stop.pack(side="right", padx=(4, 0))

        self.btn_pause = ctk.CTkButton(
            playlist_header,
            text="⏸ Pause",
            fg_color=COLORS["warning"],
            hover_color="#e6bc5c",
            text_color="#1a1a2e",
            command=self._tts_pause,
            **hdr_btn,
        )
        self.btn_pause.pack(side="right", padx=(4, 0))

        self.btn_play = ctk.CTkButton(
            playlist_header,
            text="▶ Read",
            fg_color=COLORS["success"],
            hover_color="#05b890",
            command=self._tts_play,
            **hdr_btn,
        )
        self.btn_play.pack(side="right", padx=(4, 0))

        # Work Silent + Auto Save toggles — between title and buttons
        self.silent_mode_var = ctk.BooleanVar(value=getattr(self.settings, 'silent_mode', False))
        silent_frame = ctk.CTkFrame(playlist_header, fg_color="transparent")
        silent_frame.pack(side="right", padx=(0, 16))
        ctk.CTkLabel(
            silent_frame, text="🔇", font=(FONT_FAMILY, 13),
            text_color=COLORS["text_secondary"],
        ).pack(side="left", padx=(0, 3))
        self.silent_switch = ctk.CTkSwitch(
            silent_frame, text="", variable=self.silent_mode_var,
            width=40, progress_color=COLORS["accent"],
            command=self._on_silent_toggle,
        )
        self.silent_switch.pack(side="left")

        self.auto_save_var = ctk.BooleanVar(value=False)
        auto_save_frame = ctk.CTkFrame(playlist_header, fg_color="transparent")
        auto_save_frame.pack(side="right", padx=(0, 10))
        ctk.CTkLabel(
            auto_save_frame, text="📂", font=(FONT_FAMILY, 13),
            text_color=COLORS["text_secondary"],
        ).pack(side="left", padx=(0, 3))
        self.auto_save_switch = ctk.CTkSwitch(
            auto_save_frame, text="", variable=self.auto_save_var,
            width=40, progress_color=COLORS["success"],
        )
        self.auto_save_switch.pack(side="left")

        # ── Playlist scrollable frame ────────────────────────────────────
        self.playlist_frame = ctk.CTkScrollableFrame(
            tab,
            fg_color=COLORS["bg_input"],
            corner_radius=8,
            height=180,
            scrollbar_button_color=COLORS["accent"],
        )
        self.playlist_frame.pack(fill="both", expand=True, padx=15, pady=5)

        self.playlist_empty_label = ctk.CTkLabel(
            self.playlist_frame,
            text="No files in queue. Drop files above to add them.",
            font=(FONT_FAMILY, 12),
            text_color=COLORS["text_muted"],
        )
        self.playlist_empty_label.pack(pady=30)

        # ── Progress / status ────────────────────────────────────────────
        self.tts_progress = ctk.CTkProgressBar(
            tab,
            progress_color=COLORS["accent"],
            fg_color=COLORS["bg_input"],
            height=6,
            corner_radius=3,
        )
        self.tts_progress.pack(fill="x", padx=15, pady=(5, 2))
        self.tts_progress.set(0)

        self.tts_status = ctk.CTkLabel(
            tab,
            text="Ready",
            font=(FONT_FAMILY, 11),
            text_color=COLORS["text_muted"],
        )
        self.tts_status.pack(anchor="w", padx=15)

        # ── Bottom selection bar ─────────────────────────────────────────
        sel_bar = ctk.CTkFrame(tab, fg_color="transparent")
        sel_bar.pack(fill="x", padx=15, pady=(5, 10))

        sel_btn = {"font": (FONT_FAMILY, 12, "bold"), "corner_radius": 8, "height": 34}

        ctk.CTkButton(
            sel_bar, text="🔊  TTS Selected",
            fg_color=COLORS["success"], hover_color="#05b890",
            command=self._tts_selected,
            width=140, **sel_btn,
        ).pack(side="left", padx=(0, 6))

        ctk.CTkButton(
            sel_bar, text="▶  Play Selected",
            fg_color=COLORS["accent"], hover_color=COLORS["accent_hover"],
            command=self._play_selected,
            width=140, **sel_btn,
        ).pack(side="left", padx=(0, 16))

        ctk.CTkButton(
            sel_bar, text="☑ All",
            fg_color=COLORS["bg_input"], hover_color=COLORS["bg_card_hover"],
            border_color=COLORS["border"], border_width=1,
            command=self._select_all,
            width=70, **sel_btn,
        ).pack(side="left", padx=(0, 4))

        ctk.CTkButton(
            sel_bar, text="☐ None",
            fg_color=COLORS["bg_input"], hover_color=COLORS["bg_card_hover"],
            border_color=COLORS["border"], border_width=1,
            command=self._deselect_all,
            width=70, **sel_btn,
        ).pack(side="left", padx=(0, 4))

        ctk.CTkButton(
            sel_bar, text="🗑 Clear",
            fg_color=COLORS["bg_input"], hover_color=COLORS["bg_card_hover"],
            border_color=COLORS["border"], border_width=1,
            command=self._tts_clear_playlist,
            width=80, **sel_btn,
        ).pack(side="right")

    # ==================================================================
    # TAB 2: Transcribe (STT)
    # ==================================================================

    def _build_stt_tab(self):
        tab = self.tab_stt

        # Top row — drop zone + controls
        top = ctk.CTkFrame(tab, fg_color="transparent")
        top.pack(fill="x", padx=10, pady=(10, 5))

        # Drop zone
        self.stt_drop = ctk.CTkFrame(
            top,
            fg_color=COLORS["bg_input"],
            border_color=COLORS["border"],
            border_width=2,
            corner_radius=10,
            height=80,
        )
        self.stt_drop.pack(fill="x", pady=(0, 10))
        self.stt_drop.pack_propagate(False)

        self.stt_drop_label = ctk.CTkLabel(
            self.stt_drop,
            text="🎧  Drag & drop .wav or .mp3 audio files here\nor click to browse",
            font=(FONT_FAMILY, 13),
            text_color=COLORS["text_secondary"],
            justify="center",
        )
        self.stt_drop_label.pack(expand=True)
        self.stt_drop_label.bind("<Button-1>", self._stt_browse_file)

        # Register drag-and-drop
        try:
            from tkinterdnd2 import DND_FILES
            self.stt_drop.drop_target_register(DND_FILES)
            self.stt_drop.dnd_bind("<<Drop>>", self._stt_on_drop)
        except Exception as e:
            logger.warning("Drag-and-drop not available for STT: %s", e)

        # Controls row
        stt_controls = ctk.CTkFrame(tab, fg_color="transparent")
        stt_controls.pack(fill="x", padx=10, pady=5)

        # Model size selector
        model_frame = ctk.CTkFrame(stt_controls, fg_color="transparent")
        model_frame.pack(side="left")

        ctk.CTkLabel(
            model_frame,
            text="Whisper Model",
            font=(FONT_FAMILY, 11),
            text_color=COLORS["text_secondary"],
        ).pack(anchor="w")

        self.stt_model_var = ctk.StringVar(value=self.settings.whisper_model_size)
        self.stt_model_menu = ctk.CTkOptionMenu(
            model_frame,
            values=["tiny", "base", "small", "medium", "large-v3"],
            variable=self.stt_model_var,
            width=150,
            fg_color=COLORS["bg_input"],
            button_color=COLORS["accent"],
            button_hover_color=COLORS["accent_hover"],
            dropdown_fg_color=COLORS["bg_card"],
            dropdown_hover_color=COLORS["bg_card_hover"],
            font=(FONT_FAMILY, 12),
            command=self._on_stt_model_change,
        )
        self.stt_model_menu.pack()

        # Transcribe button
        self.btn_transcribe = ctk.CTkButton(
            stt_controls,
            text="🎙  Transcribe",
            fg_color=COLORS["accent"],
            hover_color=COLORS["accent_hover"],
            font=(FONT_FAMILY, 13, "bold"),
            height=38,
            width=140,
            command=self._stt_transcribe,
        )
        self.btn_transcribe.pack(side="left", padx=15)

        self.btn_stt_cancel = ctk.CTkButton(
            stt_controls,
            text="⏹  Cancel",
            fg_color=COLORS["danger"],
            hover_color="#d43d62",
            font=(FONT_FAMILY, 13, "bold"),
            height=38,
            width=100,
            command=self._stt_cancel,
        )
        self.btn_stt_cancel.pack(side="left")

        # File info label
        self.stt_file_label = ctk.CTkLabel(
            stt_controls,
            text="No file loaded",
            font=(FONT_FAMILY, 11),
            text_color=COLORS["text_muted"],
        )
        self.stt_file_label.pack(side="right", padx=10)

        # Progress bar
        self.stt_progress = ctk.CTkProgressBar(
            tab,
            progress_color=COLORS["accent"],
            fg_color=COLORS["bg_input"],
            height=6,
            corner_radius=3,
        )
        self.stt_progress.pack(fill="x", padx=15, pady=(5, 5))
        self.stt_progress.set(0)

        # Transcription output
        self.stt_textbox = ctk.CTkTextbox(
            tab,
            fg_color=COLORS["bg_input"],
            text_color=COLORS["text_primary"],
            font=(FONT_FAMILY, 13),
            corner_radius=8,
            border_color=COLORS["border"],
            border_width=1,
            wrap="word",
        )
        self.stt_textbox.pack(fill="both", expand=True, padx=15, pady=5)
        self.stt_textbox.insert("1.0", "Transcription will appear here in real time...")
        self.stt_textbox.configure(state="disabled")

        # Export buttons
        export_frame = ctk.CTkFrame(tab, fg_color="transparent")
        export_frame.pack(fill="x", padx=15, pady=(5, 10))

        exp_style = {
            "font": (FONT_FAMILY, 12),
            "corner_radius": 8,
            "height": 34,
            "width": 130,
            "fg_color": COLORS["bg_input"],
            "hover_color": COLORS["bg_card_hover"],
            "border_color": COLORS["border"],
            "border_width": 1,
        }

        ctk.CTkButton(
            export_frame, text="📄  Save .txt",
            command=lambda: self._stt_export("txt"), **exp_style,
        ).pack(side="left", padx=(0, 8))

        ctk.CTkButton(
            export_frame, text="📝  Save .docx",
            command=lambda: self._stt_export("docx"), **exp_style,
        ).pack(side="left", padx=(0, 8))

        ctk.CTkButton(
            export_frame, text="📑  Save .pdf",
            command=lambda: self._stt_export("pdf"), **exp_style,
        ).pack(side="left", padx=(0, 8))

        # Store current audio path for transcription
        self._stt_audio_path = None

    # ==================================================================
    # TAB 3: Voice Lab
    # ==================================================================

    def _build_voice_lab_tab(self):
        tab = self.tab_voices

        # Header
        header = ctk.CTkFrame(tab, fg_color="transparent")
        header.pack(fill="x", padx=10, pady=(10, 5))

        ctk.CTkLabel(
            header,
            text="🧬  Voice Profiles",
            font=(FONT_FAMILY, 18, "bold"),
            text_color=COLORS["text_primary"],
        ).pack(side="left")

        self.btn_clone = ctk.CTkButton(
            header,
            text="➕  Clone Voice",
            fg_color=COLORS["accent"],
            hover_color=COLORS["accent_hover"],
            font=(FONT_FAMILY, 13, "bold"),
            height=38,
            width=150,
            corner_radius=8,
            command=self._voice_clone,
        )
        self.btn_clone.pack(side="right")

        self.btn_refresh_voices = ctk.CTkButton(
            header,
            text="🔄  Refresh",
            fg_color=COLORS["bg_input"],
            hover_color=COLORS["bg_card_hover"],
            border_color=COLORS["border"],
            border_width=1,
            font=(FONT_FAMILY, 12),
            height=34,
            width=100,
            corner_radius=8,
            command=self._refresh_voice_grid,
        )
        self.btn_refresh_voices.pack(side="right", padx=(0, 10))

        ctk.CTkLabel(
            tab,
            text="Upload a WAV reference clip to create a new voice. "
                 "Zero-shot cloning — no fine-tuning required.",
            font=(FONT_FAMILY, 11),
            text_color=COLORS["text_muted"],
            justify="left",
        ).pack(anchor="w", padx=15, pady=(0, 6))

        # Generation quality settings
        gen_frame = ctk.CTkFrame(tab, fg_color=COLORS["bg_card"], corner_radius=8)
        gen_frame.pack(fill="x", padx=15, pady=(0, 10))

        ctk.CTkLabel(
            gen_frame,
            text="⚙  Generation Settings",
            font=(FONT_FAMILY, 11, "bold"),
            text_color=COLORS["text_secondary"],
        ).pack(anchor="w", padx=12, pady=(8, 4))

        sliders_row = ctk.CTkFrame(gen_frame, fg_color="transparent")
        sliders_row.pack(fill="x", padx=12, pady=(0, 10))

        # Map setting key → engine attribute name
        _engine_attr = {
            "tts_temperature": "temperature",
            "tts_top_p": "top_p",
            "tts_repetition_penalty": "repetition_penalty",
            "tts_chunk_length": "chunk_length",
        }

        def _make_gen_slider(parent, label, from_, to, initial, resolution, setting_key, fmt="{:.2f}"):
            col = ctk.CTkFrame(parent, fg_color="transparent")
            col.pack(side="left", expand=True, fill="x", padx=(0, 16))
            lbl = ctk.CTkLabel(
                col,
                text=f"{label}: {fmt.format(initial)}",
                font=(FONT_FAMILY, 11),
                text_color=COLORS["text_secondary"],
            )
            lbl.pack(anchor="w")
            def _on_change(v, _sk=setting_key):
                val = round(float(v) / resolution) * resolution
                lbl.configure(text=f"{label}: {fmt.format(val)}")
                setattr(self.settings, _sk, val)
                self.settings.save()
                # Push live to engine so it takes effect on next generation
                attr = _engine_attr.get(_sk)
                if attr and self.tts and hasattr(self.tts, attr):
                    setattr(self.tts, attr, val)
            sl = ctk.CTkSlider(
                col,
                from_=from_, to=to,
                number_of_steps=round((to - from_) / resolution),
                command=_on_change,
                progress_color=COLORS["accent"],
                button_color=COLORS["accent_light"],
                height=16,
            )
            sl.set(initial)
            sl.pack(fill="x", pady=(2, 0))
            return sl

        _make_gen_slider(sliders_row, "Temperature", 0.1, 1.0,
                         getattr(self.settings, "tts_temperature", 0.7), 0.05, "tts_temperature")
        _make_gen_slider(sliders_row, "Top-P", 0.1, 1.0,
                         getattr(self.settings, "tts_top_p", 0.7), 0.05, "tts_top_p")
        _make_gen_slider(sliders_row, "Repetition Penalty", 1.0, 1.8,
                         getattr(self.settings, "tts_repetition_penalty", 1.2), 0.05, "tts_repetition_penalty")
        _make_gen_slider(sliders_row, "Chunk Length", 50, 300,
                         getattr(self.settings, "tts_chunk_length", 150), 10, "tts_chunk_length", fmt="{:.0f}")

        # Voice grid (scrollable)
        self.voice_grid_frame = ctk.CTkScrollableFrame(
            tab,
            fg_color=COLORS["bg_input"],
            corner_radius=8,
            scrollbar_button_color=COLORS["accent"],
        )
        self.voice_grid_frame.pack(fill="both", expand=True, padx=15, pady=(0, 10))

        self._refresh_voice_grid()

    def _refresh_voice_grid(self):
        """Rebuild the voice cards grid."""
        for widget in self.voice_grid_frame.winfo_children():
            widget.destroy()

        voice_list = self.voices.list_voices()

        if not voice_list:
            ctk.CTkLabel(
                self.voice_grid_frame,
                text="No voice profiles yet.\nClick 'Clone Voice' to create one.",
                font=(FONT_FAMILY, 13),
                text_color=COLORS["text_muted"],
                justify="center",
            ).pack(pady=40)
            return

        # Grid layout
        cols = 3
        for idx, name in enumerate(voice_list):
            row = idx // cols
            col = idx % cols

            card = ctk.CTkFrame(
                self.voice_grid_frame,
                fg_color=COLORS["bg_card"],
                border_color=COLORS["border"],
                border_width=1,
                corner_radius=10,
                width=220,
                height=120,
            )
            card.grid(row=row, column=col, padx=8, pady=8, sticky="nsew")
            card.grid_propagate(False)

            # Voice icon + name
            ctk.CTkLabel(
                card,
                text="🎤",
                font=(FONT_FAMILY, 28),
            ).pack(pady=(12, 2))

            ctk.CTkLabel(
                card,
                text=name,
                font=(FONT_FAMILY, 13, "bold"),
                text_color=COLORS["text_primary"],
            ).pack()

            # Action buttons
            btn_row = ctk.CTkFrame(card, fg_color="transparent")
            btn_row.pack(pady=(5, 8))

            ctk.CTkButton(
                btn_row,
                text="▶",
                width=35,
                height=28,
                corner_radius=6,
                fg_color=COLORS["success"],
                hover_color="#05b890",
                font=(FONT_FAMILY, 15),
                command=lambda n=name: self._voice_test(n),
            ).pack(side="left", padx=3)

            ctk.CTkButton(
                btn_row,
                text="🗑",
                width=35,
                height=28,
                corner_radius=6,
                fg_color=COLORS["danger"],
                hover_color="#d43d62",
                font=(FONT_FAMILY, 13),
                command=lambda n=name: self._voice_delete(n),
            ).pack(side="left", padx=3)

        # Configure grid weights
        for c in range(cols):
            self.voice_grid_frame.grid_columnconfigure(c, weight=1)

        # Also refresh the TTS dropdown
        self._refresh_tts_voice_dropdown()

    def _refresh_tts_voice_dropdown(self):
        """Update the voice dropdown in the TTS tab."""
        names = self.voices.get_voice_names()
        self.tts_voice_menu.configure(values=names)
        if self.tts_voice_var.get() not in names:
            self.tts_voice_var.set(names[0] if names else "Default (Random)")

    # ==================================================================
    # TAB 4: Settings
    # ==================================================================

    def _build_settings_tab(self):
        tab = self.tab_settings

        main = ctk.CTkScrollableFrame(
            tab,
            fg_color="transparent",
            scrollbar_button_color=COLORS["accent"],
        )
        main.pack(fill="both", expand=True, padx=5, pady=5)

        def section_header(parent, text):
            ctk.CTkLabel(
                parent,
                text=text,
                font=(FONT_FAMILY, 15, "bold"),
                text_color=COLORS["accent_light"],
            ).pack(anchor="w", pady=(15, 8), padx=10)

        def setting_row(parent):
            f = ctk.CTkFrame(parent, fg_color=COLORS["bg_card"], corner_radius=8)
            f.pack(fill="x", padx=10, pady=3)
            return f

        # --- GPU / Acceleration ---
        section_header(main, "🖥  GPU Acceleration")

        cuda_row = setting_row(main)
        ctk.CTkLabel(
            cuda_row,
            text="NVIDIA CUDA Acceleration",
            font=(FONT_FAMILY, 13),
            text_color=COLORS["text_primary"],
        ).pack(side="left", padx=15, pady=12)

        has_gpu = has_nvidia_gpu()
        cuda_installed = is_cuda_torch_installed()
        self.cuda_var = ctk.BooleanVar(value=self.settings.use_cuda and cuda_installed)
        self.cuda_switch = ctk.CTkSwitch(
            cuda_row,
            text="",
            variable=self.cuda_var,
            onvalue=True,
            offvalue=False,
            progress_color=COLORS["success"],
            command=self._on_cuda_toggle,
        )
        self.cuda_switch.pack(side="right", padx=15, pady=12)

        self.cuda_status_label = ctk.CTkLabel(
            cuda_row,
            text="",
            font=(FONT_FAMILY, 11),
        )
        self.cuda_status_label.pack(side="right", padx=5, pady=12)

        if not has_gpu:
            self.cuda_switch.configure(state="disabled")
            self.cuda_status_label.configure(
                text="⚠  No NVIDIA GPU detected",
                text_color=COLORS["warning"],
            )
        elif cuda_installed:
            gpu_name = get_nvidia_gpu_name()
            self.cuda_status_label.configure(
                text=f"✅  {gpu_name}",
                text_color=COLORS["success"],
            )
        else:
            gpu_name = get_nvidia_gpu_name()
            self.cuda_status_label.configure(
                text=f"🖥  {gpu_name} — enable to download CUDA support",
                text_color=COLORS["text_secondary"],
            )
            
        # CUDA Notice
        notice_row = setting_row(main)
        ctk.CTkLabel(
            notice_row,
            text="Note:CUDA is the best option — enabling it allows much faster and quality results",
            font=(FONT_FAMILY, 13),
            text_color=COLORS["warning"],
        ).pack(side="left", padx=15, pady=12)


        # Memory saver
        mem_row = setting_row(main)
        ctk.CTkLabel(
            mem_row,
            text="Memory Saver Mode",
            font=(FONT_FAMILY, 13),
            text_color=COLORS["text_primary"],
        ).pack(side="left", padx=15, pady=12)

        ctk.CTkLabel(
            mem_row,
            text="Unload inactive model to save RAM",
            font=(FONT_FAMILY, 11),
            text_color=COLORS["text_muted"],
        ).pack(side="left", padx=(0, 10), pady=12)

        self.memsave_var = ctk.BooleanVar(value=self.settings.memory_saver)
        self.memsave_switch = ctk.CTkSwitch(
            mem_row,
            text="",
            variable=self.memsave_var,
            onvalue=True,
            offvalue=False,
            progress_color=COLORS["success"],
            command=self._on_memsave_toggle,
        )
        self.memsave_switch.pack(side="right", padx=15, pady=12)

        _is_kokoro = getattr(self.settings, 'engine', 'fish14') == 'kokoro'

        if not _is_kokoro:
            # --- Fish-Speech Path (hidden in Kokoro mode) ---
            section_header(main, "🐟  Fish-Speech Engine")

            path_row = setting_row(main)

            ctk.CTkLabel(
                path_row,
                text="Model Path",
                font=(FONT_FAMILY, 13),
                text_color=COLORS["text_primary"],
            ).pack(side="left", padx=15, pady=12)

            self.fish_path_var = ctk.StringVar(
                value=self.settings.fish_speech_path or get_bundled_fish_speech_path()
            )
            self.fish_path_entry = ctk.CTkEntry(
                path_row,
                textvariable=self.fish_path_var,
                width=400,
                fg_color=COLORS["bg_input"],
                border_color=COLORS["border"],
                font=(FONT_FAMILY, 11),
            )
            self.fish_path_entry.pack(side="left", padx=5, pady=12)

            ctk.CTkButton(
                path_row,
                text="📂  Browse",
                width=90,
                height=30,
                corner_radius=6,
                fg_color=COLORS["bg_input"],
                hover_color=COLORS["bg_card_hover"],
                border_color=COLORS["border"],
                border_width=1,
                font=(FONT_FAMILY, 11),
                command=self._browse_fish_path,
            ).pack(side="left", padx=5, pady=12)

            # Validation indicator
            self.fish_status_label = ctk.CTkLabel(
                path_row,
                text="",
                font=(FONT_FAMILY, 11),
            )
            self.fish_status_label.pack(side="right", padx=15, pady=12)
            self._validate_fish_path()
        else:
            # --- Kokoro Engine Info ---
            section_header(main, "🎙  Kokoro Engine")
            kokoro_row = setting_row(main)
            ctk.CTkLabel(
                kokoro_row,
                text="Model",
                font=(FONT_FAMILY, 13),
                text_color=COLORS["text_primary"],
            ).pack(side="left", padx=15, pady=12)
            ctk.CTkLabel(
                kokoro_row,
                text="kokoro-v1.0 (int8 quantized)  •  54 preset voices  •  24 kHz",
                font=(FONT_FAMILY, 11),
                text_color=COLORS["text_secondary"],
            ).pack(side="left", padx=5, pady=12)
            ctk.CTkLabel(
                kokoro_row,
                text="✅  Ready",
                font=(FONT_FAMILY, 11),
                text_color=COLORS["success"],
            ).pack(side="right", padx=15, pady=12)



        # Engine Selection
        engine_row = setting_row(main)
        ctk.CTkLabel(
            engine_row,
            text="Engine Architecture",
            font=(FONT_FAMILY, 13),
            text_color=COLORS["text_primary"],
        ).pack(side="left", padx=15, pady=12)

        ctk.CTkLabel(
            engine_row,
            text="Core neural logic framework",
            font=(FONT_FAMILY, 11),
            text_color=COLORS["text_muted"],
        ).pack(side="left", padx=(0, 10), pady=12)

        engine_options = [
            "Fish-Speech 1.4 — Medium W/Cloning (1-4Gb Ram)",
            "Fish-Speech 1.5 — High W/Cloning (8+Gb Ram)",
            "Kokoro — Fast(best) No Cloning (1-2Gb Ram)",
        ]

        # Determine current engine from settings.engine field
        _eng = getattr(self.settings, 'engine', 'fish14')
        if _eng == 'kokoro':
            current_engine = engine_options[2]
        elif _eng == 'fish15':
            current_engine = engine_options[1]
        else:
            current_engine = engine_options[0]



        self.engine_var = ctk.StringVar(value=current_engine)
        self.engine_menu = ctk.CTkOptionMenu(
            engine_row,
            variable=self.engine_var,
            values=engine_options,
            width=230,
            fg_color=COLORS["bg_input"],
            button_color=COLORS["bg_input"],
            button_hover_color=COLORS["bg_card_hover"],
            dropdown_fg_color=COLORS["bg_card"],
            dropdown_hover_color=COLORS["accent"],
            font=(FONT_FAMILY, 12),
            command=self._on_engine_select,
        )
        self.engine_menu.pack(side="right", padx=15, pady=12)

        # --- Status ---
        section_header(main, "📊  System Status")

        # RAM readout
        ram_row = setting_row(main)
        ctk.CTkLabel(
            ram_row,
            text="RAM Usage",
            font=(FONT_FAMILY, 13),
            text_color=COLORS["text_primary"],
        ).pack(side="left", padx=15, pady=12)

        self.ram_label = ctk.CTkLabel(
            ram_row,
            text="Calculating...",
            font=(FONT_FAMILY, 12),
            text_color=COLORS["text_secondary"],
        )
        self.ram_label.pack(side="right", padx=15, pady=12)

        # VRAM readout
        vram_row = setting_row(main)
        ctk.CTkLabel(
            vram_row,
            text="VRAM Usage",
            font=(FONT_FAMILY, 13),
            text_color=COLORS["text_primary"],
        ).pack(side="left", padx=15, pady=12)

        self.vram_label = ctk.CTkLabel(
            vram_row,
            text="Calculating...",
            font=(FONT_FAMILY, 12),
            text_color=COLORS["text_secondary"],
        )
        self.vram_label.pack(side="right", padx=15, pady=12)

        # ffmpeg status
        ffmpeg_row = setting_row(main)
        ctk.CTkLabel(
            ffmpeg_row,
            text="FFmpeg Status",
            font=(FONT_FAMILY, 13),
            text_color=COLORS["text_primary"],
        ).pack(side="left", padx=15, pady=12)

        ffmpeg_ok = is_ffmpeg_available()
        ffmpeg_text = "✅  Found" if ffmpeg_ok else "❌  Not found — install ffmpeg for MP3 export"
        ffmpeg_color = COLORS["success"] if ffmpeg_ok else COLORS["danger"]
        ctk.CTkLabel(
            ffmpeg_row,
            text=ffmpeg_text,
            font=(FONT_FAMILY, 12),
            text_color=ffmpeg_color,
        ).pack(side="right", padx=15, pady=12)

        # TTS model status
        tts_status_row = setting_row(main)
        ctk.CTkLabel(
            tts_status_row,
            text="TTS Engine",
            font=(FONT_FAMILY, 13),
            text_color=COLORS["text_primary"],
        ).pack(side="left", padx=15, pady=12)

        self.tts_status_label = ctk.CTkLabel(
            tts_status_row,
            text="⏳  Not loaded",
            font=(FONT_FAMILY, 12),
            text_color=COLORS["text_muted"],
        )
        self.tts_status_label.pack(side="right", padx=15, pady=12)

        # STT model status
        stt_status_row = setting_row(main)
        ctk.CTkLabel(
            stt_status_row,
            text="STT Engine",
            font=(FONT_FAMILY, 13),
            text_color=COLORS["text_primary"],
        ).pack(side="left", padx=15, pady=12)

        self.stt_status_label = ctk.CTkLabel(
            stt_status_row,
            text="⏳  Not loaded",
            font=(FONT_FAMILY, 12),
            text_color=COLORS["text_muted"],
        )
        self.stt_status_label.pack(side="right", padx=15, pady=12)

        # --- Credits ---
        section_header(main, "ℹ  Credits")

        credits_frame = setting_row(main)
        credits_text = (
            "FishTalk uses the following open-source libraries:\n"
            "• Fish-Speech (Apache 2.0) — TTS engine by Fish Audio\n"
            "• faster-whisper (MIT) — STT engine by SYSTRAN\n"
            "• CustomTkinter (MIT) — GUI framework by Tom Schimansky\n"
            "• PyTorch (BSD 3-Clause) — ML backend by Meta\n"
            "• pydub (MIT) — Audio processing\n"
            "• See CREDITS.txt for full license details."
        )
        ctk.CTkLabel(
            credits_frame,
            text=credits_text,
            font=(FONT_FAMILY, 11),
            text_color=COLORS["text_muted"],
            justify="left",
        ).pack(padx=15, pady=12, anchor="w")

    # ==================================================================
    # EVENT HANDLERS — TTS Tab
    # ==================================================================

    def _tts_on_drop(self, event):
        """Handle file drop on the TTS drop zone."""
        paths = self._parse_drop_data(event.data)
        for path in paths:
            self._tts_add_file(path)

    def _tts_browse_file(self, event=None):
        """Open file dialog to add a file to the playlist."""
        paths = filedialog.askopenfilenames(
            title="Select text files",
            filetypes=[
                ("Text files", "*.txt *.pdf *.docx *.epub"),
                ("All files", "*.*"),
            ],
        )
        for path in paths:
            self._tts_add_file(path)

    def _tts_add_file(self, path: str):
        """Add a file to the playlist."""
        ext = os.path.splitext(path)[1].lower()
        if ext not in (".txt", ".pdf", ".docx", ".epub"):
            logger.warning("Unsupported file type: %s", ext)
            return

        try:
            text = read_file(path)
            name = os.path.basename(path)
            # Per-item voice: inherit from the last item, or fall back to global default
            if self._playlist_items:
                default_voice = self._playlist_items[-1].get("voice", self.tts_voice_var.get())
            else:
                default_voice = self.tts_voice_var.get()
            item = {
                "name": name,
                "text": text,
                "path": path,
                "selected": True,
                "voice": default_voice,
            }
            self._playlist_items.append(item)
            self._rebuild_playlist_ui()
            logger.info("Added to playlist: %s (%d chars)", name, len(text))
        except Exception as exc:
            logger.error("Failed to read file: %s", exc)
            messagebox.showerror("Error", f"Could not read file:\n{exc}")

    def _rebuild_playlist_ui(self):
        """Rebuild the playlist list in the scrollable frame."""
        for widget in self.playlist_frame.winfo_children():
            widget.destroy()

        if not self._playlist_items:
            self.playlist_empty_label = ctk.CTkLabel(
                self.playlist_frame,
                text="No files in queue. Drop files above to add them.",
                font=(FONT_FAMILY, 12),
                text_color=COLORS["text_muted"],
            )
            self.playlist_empty_label.pack(pady=30)
            return

        is_kokoro = getattr(self.settings, 'engine', 'fish14') == 'kokoro'
        if is_kokoro:
            voice_options = list(KOKORO_VOICES.keys())
        else:
            voice_options = self.voices.get_voice_names() or ["Default (Random)"]

        for idx, item in enumerate(self._playlist_items):
            is_active = (idx == self._current_playing)
            row_color = COLORS["accent"] if is_active else COLORS["bg_card"]
            row = ctk.CTkFrame(
                self.playlist_frame,
                fg_color=row_color,
                corner_radius=6,
                height=36,
            )
            row.pack(fill="x", pady=2, padx=4)
            row.pack_propagate(False)

            txt_color = "#ffffff" if is_active else COLORS["text_primary"]

            # ── Left: checkbox ───────────────────────────────────────────
            sel_var = ctk.BooleanVar(value=item.get("selected", True))

            def _on_toggle(v=sel_var, i=idx):
                self._playlist_items[i]["selected"] = v.get()

            ctk.CTkCheckBox(
                row,
                text="",
                variable=sel_var,
                command=_on_toggle,
                width=20,
                height=20,
                checkbox_width=16,
                checkbox_height=16,
                corner_radius=3,
                border_color=COLORS["border"],
                checkmark_color="#ffffff",
            ).pack(side="left", padx=(8, 2))

            # ── Status dot ───────────────────────────────────────────────
            audio_path = item.get("audio_path", "")
            has_audio = bool(audio_path and os.path.isfile(audio_path))
            dot = "✅" if has_audio else ("⏳" if is_active else "○")
            ctk.CTkLabel(
                row, text=dot, font=(FONT_FAMILY, 11), width=20,
            ).pack(side="left", padx=(0, 4))

            # ── Index + filename + char count ────────────────────────────
            ctk.CTkLabel(
                row,
                text=f"{idx + 1}. {item['name']}",
                font=(FONT_FAMILY, 12),
                text_color=txt_color,
                anchor="w",
            ).pack(side="left", padx=(0, 4))

            ctk.CTkLabel(
                row,
                text=f"({len(item['text']):,})",
                font=(FONT_FAMILY, 10),
                text_color=COLORS["text_muted"],
            ).pack(side="left")

            # ── Right: remove button ─────────────────────────────────────
            ctk.CTkButton(
                row,
                text="✕",
                width=26, height=22, corner_radius=4,
                fg_color=COLORS["danger"], hover_color="#d43d62",
                font=(FONT_FAMILY, 11),
                command=lambda i=idx: self._tts_remove_item(i),
            ).pack(side="right", padx=(0, 6))

            # ── Preview play/pause ───────────────────────────────────────
            if has_audio:
                is_previewing = (idx == self._preview_idx and not self._preview_paused)
                ctk.CTkButton(
                    row,
                    text="⏸" if is_previewing else "▶",
                    width=26, height=22, corner_radius=4,
                    fg_color=COLORS["success"] if is_previewing else COLORS["bg_input"],
                    hover_color="#05b890" if is_previewing else COLORS["bg_card_hover"],
                    border_color=COLORS["success"], border_width=1,
                    font=(FONT_FAMILY, 11),
                    command=lambda i=idx: self._preview_item(i),
                ).pack(side="right", padx=(0, 3))

            # ── Per-item voice dropdown ──────────────────────────────────
            item_voice = item.get("voice", voice_options[0] if voice_options else "")
            # Validate stored voice is still in current options
            if item_voice not in voice_options:
                item_voice = voice_options[0] if voice_options else item_voice

            voice_var = ctk.StringVar(value=item_voice)

            def _on_voice_change(v, i=idx, var=voice_var):
                self._playlist_items[i]["voice"] = var.get()

            ctk.CTkOptionMenu(
                row,
                values=voice_options,
                variable=voice_var,
                width=150,
                height=24,
                fg_color=COLORS["bg_input"],
                button_color=COLORS["accent"],
                button_hover_color=COLORS["accent_hover"],
                dropdown_fg_color=COLORS["bg_card"],
                dropdown_hover_color=COLORS["bg_card_hover"],
                font=(FONT_FAMILY, 11),
                command=lambda v, i=idx, var=voice_var: _on_voice_change(v, i, var),
            ).pack(side="right", padx=(0, 4))

    def _tts_remove_item(self, index: int):
        if self._preview_idx == index:
            self._stop_preview()
        if 0 <= index < len(self._playlist_items):
            self._playlist_items.pop(index)
            self._rebuild_playlist_ui()

    def _preview_item(self, idx: int):
        """Play or pause the generated audio for a completed playlist item."""
        import sounddevice as _sd
        import soundfile as _sf

        if idx < 0 or idx >= len(self._playlist_items):
            return
        item = self._playlist_items[idx]
        audio_path = item.get("audio_path", "")
        if not audio_path or not os.path.isfile(audio_path):
            return

        # Toggle pause if this item is already active
        if self._preview_idx == idx:
            self._preview_paused = not self._preview_paused
            self._rebuild_playlist_ui()
            return

        # Stop whatever was playing before and start fresh
        self._stop_preview()

        data, sr = _sf.read(audio_path, dtype="float32")
        if data.ndim > 1:
            data = data[:, 0]

        self._preview_audio = data
        self._preview_sr = sr
        self._preview_pos = 0
        self._preview_idx = idx
        self._preview_paused = False
        self._rebuild_playlist_ui()

        def _callback(outdata, frames, time_info, status):
            if self._preview_paused:
                outdata[:] = 0
                return
            vol = self.vol_slider.get() / 100.0
            remaining = len(self._preview_audio) - self._preview_pos
            if remaining <= 0:
                outdata[:] = 0
                raise _sd.CallbackStop()
            take = min(frames, remaining)
            chunk = (self._preview_audio[self._preview_pos:self._preview_pos + take] * vol).astype(np.float32)
            outdata[:take, 0] = chunk
            if outdata.shape[1] > 1:
                outdata[:take, 1] = chunk
            if take < frames:
                outdata[take:] = 0
            self._preview_pos += take

        def _on_finished():
            self._preview_idx = -1
            self._preview_paused = False
            self._preview_stream = None
            self.root.after(0, self._rebuild_playlist_ui)

        stream = _sd.OutputStream(
            samplerate=sr,
            channels=1,
            dtype="float32",
            blocksize=2048,
            callback=_callback,
            finished_callback=_on_finished,
        )
        self._preview_stream = stream
        stream.start()

    def _stop_preview(self):
        """Stop any active item preview."""
        if self._preview_stream:
            try:
                self._preview_stream.stop()
                self._preview_stream.close()
            except Exception:
                pass
            self._preview_stream = None
        self._preview_idx = -1
        self._preview_paused = False

    def _sync_gen_settings_to_engine(self):
        """Push persisted generation quality settings into the TTS engine."""
        if not self.tts:
            return
        mapping = {
            "tts_temperature": "temperature",
            "tts_top_p": "top_p",
            "tts_repetition_penalty": "repetition_penalty",
            "tts_chunk_length": "chunk_length",
        }
        for setting_key, attr in mapping.items():
            val = getattr(self.settings, setting_key, None)
            if val is not None and hasattr(self.tts, attr):
                setattr(self.tts, attr, val)

    def _cleanup_old_temp_audio(self, max_age_days: int = 7):
        """Delete temp WAVs older than max_age_days from AUDIO_TEMP_DIR."""
        if not os.path.isdir(AUDIO_TEMP_DIR):
            return
        cutoff = time.time() - max_age_days * 86400
        removed = 0
        for fname in os.listdir(AUDIO_TEMP_DIR):
            if not fname.lower().endswith(".wav"):
                continue
            fpath = os.path.join(AUDIO_TEMP_DIR, fname)
            try:
                if os.path.getmtime(fpath) < cutoff:
                    os.remove(fpath)
                    removed += 1
            except Exception:
                pass
        if removed:
            logger.info("Cleaned %d temp audio file(s) older than %d days.", removed, max_age_days)

    @staticmethod
    def _delete_temp_wav(path: str):
        """Silently delete a temp WAV file."""
        try:
            if path and os.path.isfile(path):
                os.remove(path)
                logger.info("Deleted temp WAV: %s", path)
        except Exception as e:
            logger.warning("Could not delete temp WAV %s: %s", path, e)

    def _tts_clear_playlist(self):
        self._playlist_items.clear()
        self._current_playing = -1
        self._rebuild_playlist_ui()

    def _on_speed_change(self, value):
        self.speed_label.configure(text=f"Speed: {value:.1f}x")
        self.settings.speed = round(value, 1)

    def _on_volume_change(self, value):
        self.vol_label.configure(text=f"Volume: {int(value)}%")
        self.settings.volume = int(value)

    def _on_cadence_change(self, value):
        self.cad_label.configure(text=f"Cadence: {int(value)}%")
        self.settings.cadence = int(value)

    def _ensure_tts_loaded(self, on_success):
        """Helper to lazy load TTS with a beautiful popup overlay."""
        if self.tts.is_loaded:
            on_success()
            return
            
        # Create themed popup
        popup = ctk.CTkToplevel(self.root)
        popup.title("Starting AI Engine")
        popup.geometry("350x140")
        popup.resizable(False, False)
        popup.transient(self.root)
        popup.grab_set()
        popup.configure(fg_color=COLORS["bg_card"])
        
        # Center popup over the main window
        self.root.update_idletasks()
        x = self.root.winfo_x() + (self.root.winfo_width() // 2) - 175
        y = self.root.winfo_y() + (self.root.winfo_height() // 2) - 70
        popup.geometry(f"+{x}+{y}")
        
        engine_name = "Kokoro" if getattr(self.settings, 'engine', 'fish14') == 'kokoro' else "Fish-Speech"
        lbl_msg = ctk.CTkLabel(popup, text=f"Booting {engine_name} Engine...", font=(FONT_FAMILY, 14, "bold"), text_color=COLORS["text_primary"])
        lbl_msg.pack(pady=(20, 10))
        
        pb = ctk.CTkProgressBar(popup, progress_color=COLORS["accent"], fg_color=COLORS["bg_input"], width=280, height=8, corner_radius=4)
        pb.pack(pady=5)
        pb.set(0)
        
        status = ctk.CTkLabel(popup, text="Initializing...", font=(FONT_FAMILY, 11), text_color=COLORS["text_muted"])
        status.pack(pady=5)

        def on_progress(text: str, frac: float):
            self.root.after(0, lambda: status.configure(text=text))
            self.root.after(0, lambda: pb.set(frac))

        def on_ready():
            def _ready_ui():
                popup.grab_release()
                popup.destroy()
                self.update_tts_status("✅  Engine ready", COLORS["success"])
                try:
                    self.tts_progress.set(0)
                    for btn in (self.btn_play, self.btn_pause, self.btn_save_mp3):
                        btn.configure(state="normal")
                except AttributeError:
                    pass
                on_success()
            self.root.after(0, _ready_ui)

        def on_error(exc):
            def _err_ui():
                popup.grab_release()
                popup.destroy()
                messagebox.showerror("Error", f"Failed to load engine:\n{exc}")
                self.update_tts_status("❌  Load failed", COLORS["danger"])
            self.root.after(0, _err_ui)

        self.tts.load_model(on_progress=on_progress, on_ready=on_ready, on_error=on_error)

    def _tts_play(self):
        """Read all playlist items in order, skipping those already generated."""
        if not self._playlist_items:
            messagebox.showinfo("FishTalk", "Add files to the playlist first.")
            return

        if not self.tts.is_loaded:
            for btn in (self.btn_play, self.btn_pause, self.btn_save_mp3):
                btn.configure(state="disabled")
            self._ensure_tts_loaded(self._tts_play)
            return

        if self._is_paused:
            self._is_paused = False
            self._resume_audio()
            return

        # Find first item without audio, skip already-done ones
        start_idx = 0
        for i, item in enumerate(self._playlist_items):
            ap = item.get("audio_path", "")
            if not ap or not os.path.isfile(ap):
                start_idx = i
                break
        else:
            # All items already have audio — ask to re-read from top
            if messagebox.askyesno(
                "All Done",
                "All items have already been generated.\nRe-read from the beginning?",
            ):
                start_idx = 0
            else:
                return

        self._stop_preview()
        self._is_playing = True
        self._current_playing = start_idx
        self._rebuild_playlist_ui()
        self._play_current_item()

    def _select_all(self):
        for item in self._playlist_items:
            item["selected"] = True
        self._rebuild_playlist_ui()

    def _deselect_all(self):
        for item in self._playlist_items:
            item["selected"] = False
        self._rebuild_playlist_ui()

    def _tts_selected(self):
        """Generate TTS for all checked items. Ask before re-generating completed ones."""
        selected = [i for i, it in enumerate(self._playlist_items) if it.get("selected", True)]
        if not selected:
            messagebox.showinfo("FishTalk", "No items selected.")
            return

        # Check if any already have audio
        done = [i for i in selected
                if self._playlist_items[i].get("audio_path") and
                os.path.isfile(self._playlist_items[i]["audio_path"])]
        pending = [i for i in selected if i not in done]

        if done and not pending:
            # All selected are already done
            if not messagebox.askyesno(
                "Re-generate?",
                f"All {len(done)} selected item(s) already have audio.\nRe-generate them?",
            ):
                return
            indices = selected
        elif done:
            answer = messagebox.askyesno(
                "Re-generate?",
                f"{len(done)} selected item(s) already have audio.\n"
                f"Re-generate those too, or skip them?\n\n"
                "Yes = re-generate all  |  No = skip already-done",
            )
            indices = selected if answer else pending
        else:
            indices = pending

        if not self.tts.is_loaded:
            for btn in (self.btn_play, self.btn_pause, self.btn_save_mp3):
                btn.configure(state="disabled")
            self._ensure_tts_loaded(lambda: self._run_tts_for_indices(indices))
            return

        self._run_tts_for_indices(indices)

    def _run_tts_for_indices(self, indices: list):
        """Generate TTS sequentially for the given list of item indices."""
        if not indices:
            return
        self._stop_preview()
        self._is_playing = True
        self._tts_queue = list(indices)
        self._current_playing = self._tts_queue[0]
        self._rebuild_playlist_ui()
        self._play_queued_item()

    def _play_queued_item(self):
        """Advance through _tts_queue and generate each item."""
        if not getattr(self, "_tts_queue", None):
            self._is_playing = False
            self._current_playing = -1
            self._rebuild_playlist_ui()
            self.tts_status.configure(text="Done")
            return
        self._current_playing = self._tts_queue[0]
        self._rebuild_playlist_ui()
        self._play_current_item(advance_fn=self._advance_queued)

    def _advance_queued(self):
        """Pop the front of the queue and continue."""
        if getattr(self, "_tts_queue", None):
            self._tts_queue.pop(0)
        self.root.after(300, self._play_queued_item)

    def _play_selected(self):
        """Play already-generated audio for all checked items in sequence."""
        ready = [
            i for i, it in enumerate(self._playlist_items)
            if it.get("selected", True)
            and it.get("audio_path")
            and os.path.isfile(it["audio_path"])
        ]
        if not ready:
            messagebox.showinfo("FishTalk", "No selected items have generated audio yet.")
            return
        self._stop_preview()
        self._preview_queue = list(ready)
        self._run_preview_queue()

    def _run_preview_queue(self):
        if not getattr(self, "_preview_queue", None):
            self._preview_idx = -1
            self._rebuild_playlist_ui()
            return
        idx = self._preview_queue[0]
        item = self._playlist_items[idx]

        import soundfile as _sf
        import sounddevice as _sd
        import numpy as _np

        data, sr = _sf.read(item["audio_path"], dtype="float32")
        if data.ndim > 1:
            data = data[:, 0]

        self._preview_audio = data
        self._preview_sr = sr
        self._preview_pos = 0
        self._preview_idx = idx
        self._preview_paused = False
        self._rebuild_playlist_ui()

        def _callback(outdata, frames, time_info, status):
            vol = self.vol_slider.get() / 100.0
            remaining = len(self._preview_audio) - self._preview_pos
            if remaining <= 0:
                outdata[:] = 0
                raise _sd.CallbackStop()
            take = min(frames, remaining)
            chunk = (self._preview_audio[self._preview_pos:self._preview_pos + take] * vol).astype(_np.float32)
            outdata[:take, 0] = chunk
            if outdata.shape[1] > 1:
                outdata[:take, 1] = chunk
            if take < frames:
                outdata[take:] = 0
            self._preview_pos += take

        def _on_finished():
            if getattr(self, "_preview_queue", None):
                self._preview_queue.pop(0)
            self.root.after(300, self._run_preview_queue)

        try:
            self._preview_stream = _sd.OutputStream(
                samplerate=sr, channels=2, dtype="float32",
                callback=_callback, finished_callback=_on_finished,
            )
            self._preview_stream.start()
        except Exception as exc:
            logger.error("Play Selected stream error: %s", exc)

    def _play_current_item(self, advance_fn=None):
        """Generate and play the current playlist item."""
        if self._current_playing < 0 or self._current_playing >= len(self._playlist_items):
            self._is_playing = False
            self._current_playing = -1
            self._rebuild_playlist_ui()
            self.tts_status.configure(text="Playlist complete")
            return

        item = self._playlist_items[self._current_playing]
        self.tts_status.configure(text=f"Generating: {item['name']}...")
        self._rebuild_playlist_ui()

        # Use per-item voice, fall back to global selector
        voice_name = item.get("voice") or self.tts_voice_var.get()
        profile = None
        if voice_name != "Default (Random)":
            profile = self.voices.get_voice(voice_name)

        speed = self.speed_slider.get()

        # Compute a deterministic output path in our managed temp folder.
        os.makedirs(AUDIO_TEMP_DIR, exist_ok=True)
        safe_stem = re.sub(r"[^\w\-]", "_", os.path.splitext(item["name"])[0])
        _output_path = os.path.join(AUDIO_TEMP_DIR, f"{safe_stem}_{int(time.time())}.wav")

        def on_progress(status, frac):
            self.root.after(0, lambda: self.tts_progress.set(frac))
            self.root.after(0, lambda: self.tts_status.configure(text=status))

        # --- Streaming chunk playback via continuous OutputStream ---
        # Volume is read live from the slider inside the audio callback
        # so moving the slider mid-playback takes effect immediately.
        import queue as _queue
        import numpy as _np
        import sounddevice as _sd

        _sample_queue = _queue.Queue()
        _SENTINEL = object()
        _stream = [None]

        def _stream_callback(outdata, frames, time_info, status):
            vol = self.vol_slider.get() / 100.0
            remaining = frames
            offset = 0
            outdata[:] = 0  # default silence
            while remaining > 0:
                if _stream_callback._buf is None or len(_stream_callback._buf) == 0:
                    try:
                        item = _sample_queue.get_nowait()
                        if item is _SENTINEL:
                            return
                        _stream_callback._buf = item
                    except _queue.Empty:
                        return  # underrun — silence already written
                take = min(remaining, len(_stream_callback._buf))
                out_slice = (_stream_callback._buf[:take] * vol).astype(_np.float32)
                outdata[offset:offset+take, 0] = out_slice
                if outdata.shape[1] > 1:
                    outdata[offset:offset+take, 1] = out_slice
                _stream_callback._buf = _stream_callback._buf[take:]
                offset += take
                remaining -= take
        _stream_callback._buf = None


        def on_chunk(chunk_np, sr):
            """Push decoded samples into the playback stream (skipped in silent mode)."""
            if self.silent_mode_var.get():
                return  # Work Silent: generate only, no output
            if _stream[0] is None:
                s = _sd.OutputStream(
                    samplerate=sr, channels=1, dtype='float32',
                    blocksize=2048, callback=_stream_callback,
                )
                s.start()
                _stream[0] = s
            _sample_queue.put(chunk_np.astype(_np.float32))

        def on_complete(wav_path):
            # Store for preview button and manual Save MP3
            item["audio_path"] = wav_path
            self._last_wav_path = wav_path
            _sample_queue.put(_SENTINEL)
            def _finish():
                import time as _t
                if self.silent_mode_var.get() or _stream[0] is None:
                    # Silent mode (or stream never started): nothing is consuming
                    # the queue so drain it ourselves to avoid an infinite wait.
                    while not _sample_queue.empty():
                        try:
                            _sample_queue.get_nowait()
                        except Exception:
                            break
                else:
                    # Wait until the audio stream callback consumes everything
                    while not _sample_queue.empty():
                        _t.sleep(0.1)
                    # Give it a moment to finish playing the final buffer
                    _t.sleep(0.5)

                if _stream[0]:
                    try:
                        _stream[0].stop()
                        _stream[0].close()
                    except Exception:
                        pass

                # Auto-save to outputs/<stem>/<stem>.mp3 if toggle is on
                if self.auto_save_var.get() and wav_path and os.path.isfile(wav_path):
                    try:
                        stem = os.path.splitext(item["name"])[0]
                        out_dir = os.path.join(APP_DIR, "outputs", stem)
                        os.makedirs(out_dir, exist_ok=True)
                        out_mp3 = os.path.join(out_dir, stem + ".mp3")
                        export_mp3(wav_path, out_mp3)
                        logger.info("Auto-saved: %s", out_mp3)
                        # Point preview to the MP3 and delete the temp WAV
                        item["audio_path"] = out_mp3
                        self._last_wav_path = None
                        self._delete_temp_wav(wav_path)
                        self.root.after(0, lambda p=out_mp3: self.tts_status.configure(
                            text=f"✅ Saved: outputs/{stem}/{stem}.mp3"
                        ))
                    except Exception as _e:
                        logger.error("Auto-save failed: %s", _e)
                        self.root.after(0, lambda e=_e: self.tts_status.configure(
                            text=f"⚠️ Auto-save failed: {e}"
                        ))
                else:
                    self.root.after(0, lambda: self.tts_status.configure(
                        text=f"✅ Done: {self._playlist_items[self._current_playing]['name']}"
                    ))

                self.root.after(0, lambda: self.tts_progress.set(1.0))
                self.root.after(0, self._rebuild_playlist_ui)
                if advance_fn:
                    self.root.after(300, advance_fn)
                else:
                    self._current_playing += 1
                    self.root.after(500, self._play_current_item)
            threading.Thread(target=_finish, daemon=True).start()

        def on_error(exc):
            _sample_queue.put(_SENTINEL)
            if _stream[0]:
                try: _stream[0].stop(); _stream[0].close()
                except Exception: pass
            self.root.after(0, lambda: self.tts_status.configure(text=f"Error: {exc}"))

        # Route to correct engine based on settings
        _is_kokoro_mode = getattr(self.settings, 'engine', 'fish14') == 'kokoro'

        if _is_kokoro_mode:
            # voice_name may be display name or raw ID — handle both
            voice_id = KOKORO_VOICES.get(voice_name, voice_name if voice_name in KOKORO_VOICES.values() else DEFAULT_VOICE)
            self.tts.generate(
                text=item["text"],
                voice_id=voice_id,
                speed=speed,
                output_path=_output_path,
                on_progress=on_progress,
                on_chunk=on_chunk,
                on_complete=on_complete,
                on_error=on_error,
            )
        else:
            self.tts.generate(
                text=item["text"],
                reference_wav=profile["wav_path"] if profile else None,
                reference_tokens=None,
                prompt_text=profile["prompt_text"] if profile else None,
                speed=speed,
                output_path=_output_path,
                on_progress=on_progress,
                on_chunk=on_chunk,
                on_complete=on_complete,
                on_error=on_error,
            )

    def _play_audio(self, wav_path: str):
        """Play generated audio through speakers and auto-advance."""
        try:
            import soundfile as sf_lib
            import sounddevice as sd

            data, sr = sf_lib.read(wav_path)

            # Apply volume
            vol = self.vol_slider.get() / 100.0
            data = data * vol

            self._audio_data = data
            self._audio_sr = sr
            self.tts_status.configure(text=f"Playing: {self._playlist_items[self._current_playing]['name']}")

            # Stop any existing playback
            sd.stop()

            def _play_thread():
                import time as t
                try:
                    sd.play(data, sr)
                    # Poll in small slices so pause/stop can interrupt
                    duration = len(data) / sr if sr else 1
                    start = t.time()
                    while sd.get_stream() is not None and sd.get_stream().active:
                        if not self._is_playing or self._is_paused:
                            sd.stop()
                            return
                        elapsed = t.time() - start
                        progress = min(elapsed / duration, 1.0) if duration > 0 else 1.0
                        self.root.after(0, lambda p=progress: self.tts_progress.set(p))
                        t.sleep(0.05)
                    # Natural end — advance to next item
                    if self._is_playing and not self._is_paused:
                        self._current_playing += 1
                        self.root.after(300, self._play_current_item)
                except Exception as exc:
                    logger.error("Playback thread error: %s", exc)

            threading.Thread(target=_play_thread, daemon=True).start()

        except Exception as exc:
            logger.error("Audio playback error: %s", exc)
            self.tts_status.configure(text=f"Playback error: {exc}")

    def _resume_audio(self):
        """Resume paused audio playback."""
        try:
            import sounddevice as sd
            # sounddevice doesn't support true pause/resume easily,
            # so this is a simplified implementation
            self.tts_status.configure(text="Resumed")
        except Exception:
            pass

    def _tts_pause(self):
        """Pause playback."""
        self._is_paused = True
        try:
            import sounddevice as sd
            sd.stop()
        except Exception:
            pass
        self.tts_status.configure(text="Paused")

    def _tts_stop(self):
        """Stop playback and cancel any pending generation."""
        self._stop_preview()
        self._is_playing = False
        self._is_paused = False
        self._current_playing = -1
        self.tts.cancel()
        try:
            import sounddevice as sd
            sd.stop()
        except Exception:
            pass
        self.tts_progress.set(0)
        self.tts_status.configure(text="Stopped")
        self._rebuild_playlist_ui()

    def _tts_save_mp3(self):
        """Export the last generated audio to MP3."""
        if not is_ffmpeg_available():
            messagebox.showwarning(
                "FishTalk",
                "ffmpeg is not installed.\nPlease install ffmpeg or place ffmpeg.exe in the bin/ folder."
            )
            return

        src = self._last_wav_path
        if not src or not os.path.isfile(src):
            messagebox.showinfo("FishTalk", "No audio to export. Generate speech first.")
            return

        path = filedialog.asksaveasfilename(
            title="Save as MP3",
            defaultextension=".mp3",
            filetypes=[("MP3 files", "*.mp3"), ("WAV files", "*.wav")],
            initialdir=os.path.expanduser("~/Documents"),
        )
        if not path:
            return

        try:
            export_mp3(src, path)
            # Remove temp WAV now that it has been permanently saved
            self._delete_temp_wav(src)
            self._last_wav_path = None
            messagebox.showinfo("FishTalk", f"Saved to:\n{path}")
        except Exception as exc:
            messagebox.showerror("Error", f"Export failed:\n{exc}")

    # ==================================================================
    # EVENT HANDLERS — STT Tab
    # ==================================================================

    def _stt_on_drop(self, event):
        """Handle file drop on the STT drop zone."""
        paths = self._parse_drop_data(event.data)
        if paths:
            self._stt_set_file(paths[0])

    def _stt_browse_file(self, event=None):
        """Open file dialog to select an audio file."""
        path = filedialog.askopenfilename(
            title="Select audio file",
            filetypes=[
                ("Audio files", "*.wav *.mp3 *.m4a *.flac *.ogg"),
                ("All files", "*.*"),
            ],
        )
        if path:
            self._stt_set_file(path)

    def _stt_set_file(self, path: str):
        """Set the audio file for transcription."""
        ext = os.path.splitext(path)[1].lower()
        if ext not in (".wav", ".mp3", ".m4a", ".flac", ".ogg"):
            messagebox.showwarning("FishTalk", f"Unsupported audio format: {ext}")
            return
        self._stt_audio_path = path
        name = os.path.basename(path)
        self.stt_file_label.configure(text=f"📎  {name}", text_color=COLORS["success"])
        self.stt_drop_label.configure(text=f"✅  {name}")

    def _stt_transcribe(self):
        """Start transcription."""
        if not self._stt_audio_path:
            messagebox.showinfo("FishTalk", "Drop or select an audio file first.")
            return

        if not self.stt.is_loaded:
            # Load model first
            device = "cuda" if self.cuda_var.get() else "cpu"
            compute_type = "float16" if device == "cuda" else "int8"
            model_size = self.stt_model_var.get()

            self.stt_textbox.configure(state="normal")
            self.stt_textbox.delete("1.0", "end")
            self.stt_textbox.insert("1.0", "Loading Whisper model...\n")
            self.stt_textbox.configure(state="disabled")

            def on_ready():
                self.root.after(0, self._stt_run_transcription)
                self.root.after(
                    0,
                    lambda: self.stt_status_label.configure(
                        text="✅  Loaded",
                        text_color=COLORS["success"],
                    )
                )

            def on_error(exc):
                self.root.after(
                    0,
                    lambda: self._stt_append_text(f"\nError loading model: {exc}")
                )

            self.stt.load_model(
                model_size=model_size,
                device=device,
                compute_type=compute_type,
                on_ready=on_ready,
                on_error=on_error,
            )
        else:
            self._stt_run_transcription()

    def _stt_run_transcription(self):
        """Run the actual transcription."""
        self.stt_textbox.configure(state="normal")
        self.stt_textbox.delete("1.0", "end")
        self.stt_textbox.configure(state="disabled")
        self.stt_progress.set(0)

        def on_segment(text, start, end):
            timestamp = f"[{start:.1f}s → {end:.1f}s]"
            self.root.after(
                0,
                lambda: self._stt_append_text(f"{timestamp}  {text}\n")
            )

        def on_progress(frac):
            self.root.after(0, lambda: self.stt_progress.set(frac))

        def on_complete(full_text, info):
            self.root.after(0, lambda: self.stt_progress.set(1.0))
            lang = info.get("language", "unknown")
            prob = info.get("language_probability", 0.0) * 100
            self.root.after(
                0,
                lambda: self._stt_append_text(
                    f"\n--- Done! Language: {lang} ({prob:.0f}% confidence) ---"
                )
            )

        def on_error(exc):
            self.root.after(
                0,
                lambda: self._stt_append_text(f"\n⚠ Error: {exc}")
            )

        self.stt.transcribe(
            audio_path=self._stt_audio_path,
            on_segment=on_segment,
            on_progress=on_progress,
            on_complete=on_complete,
            on_error=on_error,
        )

    def _stt_append_text(self, text: str):
        """Append text to the STT textbox (thread-safe via root.after)."""
        self.stt_textbox.configure(state="normal")
        self.stt_textbox.insert("end", text)
        self.stt_textbox.see("end")
        self.stt_textbox.configure(state="disabled")

    def _stt_cancel(self):
        """Cancel ongoing transcription."""
        self.stt.cancel()
        self._stt_append_text("\n--- Cancelled ---")

    def _stt_export(self, fmt: str):
        """Export transcription to a file."""
        self.stt_textbox.configure(state="normal")
        text = self.stt_textbox.get("1.0", "end").strip()
        self.stt_textbox.configure(state="disabled")

        if not text or text == "Transcription will appear here in real time...":
            messagebox.showinfo("FishTalk", "No transcription to export.")
            return

        exts = {"txt": ".txt", "docx": ".docx", "pdf": ".pdf"}
        ftypes = {
            "txt": [("Text files", "*.txt")],
            "docx": [("Word documents", "*.docx")],
            "pdf": [("PDF files", "*.pdf")],
        }

        path = filedialog.asksaveasfilename(
            title=f"Save as {fmt.upper()}",
            defaultextension=exts[fmt],
            filetypes=ftypes[fmt],
        )
        if not path:
            return

        try:
            {"txt": export_txt, "docx": export_docx, "pdf": export_pdf}[fmt](text, path)
            messagebox.showinfo("FishTalk", f"Saved to:\n{path}")
        except Exception as exc:
            messagebox.showerror("Error", f"Export failed:\n{exc}")

    def _on_stt_model_change(self, value):
        """Handle Whisper model size change."""
        self.settings.whisper_model_size = value
        # Force model reload on next transcription
        if self.stt.is_loaded:
            self.stt.unload_model()
            self.stt_status_label.configure(
                text="⏳  Model changed — will reload on next use",
                text_color=COLORS["warning"],
            )

    # ==================================================================
    # EVENT HANDLERS — Voice Lab
    # ==================================================================

    def _voice_clone(self):
        """Open dialog to clone a new voice."""
        path = filedialog.askopenfilename(
            title="Select reference audio (15–30 sec WAV)",
            filetypes=[
                ("WAV files", "*.wav"),
                ("All audio", "*.wav *.mp3 *.flac"),
            ],
        )
        if not path:
            return

        # Ask for voice name
        dialog = ctk.CTkInputDialog(
            text="Enter a name for this voice profile:",
            title="Clone Voice",
        )
        name = dialog.get_input()
        if not name or not name.strip():
            return

        name = name.strip()
        if self.voices.voice_exists(name):
            messagebox.showwarning("FishTalk", f"Voice '{name}' already exists.")
            return

        # Auto-transcribe the reference audio so Fish Speech gets proper text
        # context for voice conditioning (empty transcript = poor clone quality).
        def _do_clone(transcript: str):
            try:
                tts = self.tts if self.tts.is_loaded else None
                self.voices.clone_voice(
                    name=name,
                    reference_wav_path=path,
                    tts_engine=tts,
                    prompt_text=transcript,
                )
                self.root.after(0, self._refresh_voice_grid)
                self.root.after(0, lambda: messagebox.showinfo(
                    "FishTalk", f"Voice '{name}' created successfully!"
                    + (f"\n\nTranscript saved:\n\"{transcript[:120]}{'…' if len(transcript) > 120 else ''}\"" if transcript else "")
                ))
            except Exception as exc:
                self.root.after(0, lambda e=exc: messagebox.showerror(
                    "Error", f"Voice cloning failed:\n{e}"
                ))

        def _transcribe_and_clone():
            transcript_result = [None]
            done = threading.Event()

            def _on_transcript(text, _info):
                transcript_result[0] = text.strip()
                done.set()

            def _on_stt_error(_exc):
                logger.warning("STT transcription failed during voice clone: %s", _exc)
                transcript_result[0] = ""
                done.set()

            def _on_stt_ready():
                self.stt.transcribe(
                    audio_path=path,
                    on_complete=_on_transcript,
                    on_error=_on_stt_error,
                )
                done.wait(timeout=120)
                _do_clone(transcript_result[0] or "")

            if self.stt.is_loaded:
                _on_stt_ready()
            else:
                # Load Whisper first, then transcribe
                self.stt.load_model(on_ready=_on_stt_ready, on_error=_on_stt_error)
                done.wait(timeout=180)

        self.root.after(0, lambda: self.tts_status.configure(
            text="Transcribing reference audio for voice conditioning…"
        ))
        threading.Thread(target=_transcribe_and_clone, daemon=True, name="VoiceClone").start()

    def _voice_delete(self, name: str):
        """Delete a voice profile."""
        confirm = messagebox.askyesno(
            "Delete Voice",
            f"Are you sure you want to delete '{name}'?"
        )
        if confirm:
            self.voices.delete_voice(name)
            self._refresh_voice_grid()

    def _voice_test(self, name: str):
        """Generate and play a sample of the selected voice."""
        if not self.tts.is_loaded:
            self._ensure_tts_loaded(lambda: self._voice_test(name))
            return
            
        profile = self.voices.get_voice(name)
        if not profile:
            return
            
        test_text = "Hello, how do I sound?"
        
        # We need a small loading indicator. We can use the status bar of TTS tab or similar
        self.update_tts_status(f"⏳ Testing {name}...", COLORS["warning"])
        
        def on_complete(wav_path):
            self.root.after(0, lambda: self.update_tts_status("✅ Engine ready (test complete)", COLORS["success"]))
            def _play():
                try:
                    import soundfile as sf_lib
                    import sounddevice as sd
                    data, sr = sf_lib.read(wav_path)
                    sd.stop()
                    sd.play(data, sr)
                    sd.wait()
                except Exception as e:
                    logger.error("Voice test playback error: %s", e)
            threading.Thread(target=_play, daemon=True).start()
                
        def on_error(exc):
            self.root.after(0, lambda: self.update_tts_status("✅ Engine ready", COLORS["success"]))
            self.root.after(0, lambda: messagebox.showerror("Test Failed", f"Could not generate sample:\n{exc}"))

        ref_wav = profile.get("wav_path")
        tokens_path = profile.get("tokens_path")
        ref_tokens = np.load(tokens_path) if tokens_path and os.path.isfile(tokens_path) else None
        
        self.tts.generate(
            text=test_text,
            reference_wav=ref_wav,
            reference_tokens=ref_tokens,
            prompt_text=profile.get("prompt_text", ""),
            on_complete=on_complete,
            on_error=on_error
        )

    # ==================================================================
    # EVENT HANDLERS — Settings
    # ==================================================================

    def _on_cuda_toggle(self):
        """Handle CUDA toggle — download CUDA PyTorch on demand."""
        wants_cuda = self.cuda_var.get()

        if wants_cuda and not is_cuda_torch_installed():
            # User is enabling CUDA but doesn't have CUDA PyTorch yet
            if not has_nvidia_gpu():
                messagebox.showwarning(
                    "FishTalk",
                    "No NVIDIA GPU detected.\nCUDA acceleration is not available on this system."
                )
                self.cuda_var.set(False)
                return

            gpu_name = get_nvidia_gpu_name()
            confirm = messagebox.askyesno(
                "Download CUDA Support",
                f"GPU detected: {gpu_name}\n\n"
                "To enable GPU acceleration, FishTalk needs to download\n"
                "the CUDA version of PyTorch (~2.5 GB).\n\n"
                "This is a one-time download. The app will continue\n"
                "working while it downloads in the background.\n\n"
                "Download now?"
            )

            if not confirm:
                self.cuda_var.set(False)
                return

            # Start download
            self.cuda_switch.configure(state="disabled")
            self.cuda_status_label.configure(
                text="⏳  Downloading CUDA PyTorch...",
                text_color=COLORS["warning"],
            )

            def on_progress(status):
                self.root.after(0, lambda: self.cuda_status_label.configure(
                    text=f"⏳  {status}",
                    text_color=COLORS["warning"],
                ))

            def on_complete(success, message):
                def _update():
                    self.cuda_switch.configure(state="normal")
                    if success:
                        self.cuda_status_label.configure(
                            text=f"✅  {gpu_name} — CUDA ready",
                            text_color=COLORS["success"],
                        )
                        self.settings.use_cuda = True
                        self.settings.save()
                        messagebox.showinfo(
                            "CUDA Ready",
                            f"{message}\n\nRestart FishTalk to use GPU acceleration."
                        )
                    else:
                        self.cuda_var.set(False)
                        self.cuda_status_label.configure(
                            text="❌  CUDA download failed",
                            text_color=COLORS["danger"],
                        )
                        messagebox.showerror("CUDA Setup Failed", message)
                self.root.after(0, _update)

            install_cuda_pytorch(on_progress=on_progress, on_complete=on_complete)

        elif not wants_cuda and is_cuda_torch_installed():
            # User is disabling CUDA — offer to revert to save space
            revert = messagebox.askyesno(
                "Revert to CPU",
                "Would you like to remove CUDA PyTorch and switch back\n"
                "to CPU-only mode? This frees ~2 GB of disk space.\n\n"
                "(Choose 'No' to keep CUDA installed but disabled)"
            )

            self.settings.use_cuda = False
            self.settings.save()

            if revert:
                self.cuda_switch.configure(state="disabled")
                self.cuda_status_label.configure(
                    text="⏳  Reverting to CPU...",
                    text_color=COLORS["warning"],
                )

                def on_revert_complete(success, message):
                    def _update():
                        self.cuda_switch.configure(state="normal")
                        self.cuda_status_label.configure(
                            text="✅  Reverted to CPU mode",
                            text_color=COLORS["success"],
                        )
                        messagebox.showinfo("Done", message)
                    self.root.after(0, _update)

                revert_to_cpu_pytorch(on_complete=on_revert_complete)
            else:
                self.cuda_status_label.configure(
                    text="💤  CUDA installed but disabled",
                    text_color=COLORS["text_muted"],
                )
        else:
            # Simple toggle (CUDA already installed, or already CPU)
            self.settings.use_cuda = wants_cuda
            self.settings.save()

    def _on_memsave_toggle(self):
        self.settings.memory_saver = self.memsave_var.get()
        self.settings.save()

    def _on_silent_toggle(self):
        self.settings.silent_mode = self.silent_mode_var.get()
        self.settings.save()
        state = "enabled — audio will be generated but not played" if self.settings.silent_mode else "disabled"
        self.tts_status.configure(text=f"🔇 Work Silent {state}")

    def _lock_voice_lab(self):
        """Disable the Voice Lab tab when Kokoro engine is active."""
        try:
            # Destroy all existing children so voice cards don't show through
            for widget in self.tab_voices.winfo_children():
                widget.destroy()
            # Show centred notice
            notice_frame = ctk.CTkFrame(self.tab_voices, fg_color="transparent")
            notice_frame.pack(expand=True, fill="both")
            ctk.CTkLabel(
                notice_frame,
                text="🔒",
                font=(FONT_FAMILY, 48),
                text_color=COLORS["text_muted"],
            ).pack(expand=True, pady=(80, 8))
            ctk.CTkLabel(
                notice_frame,
                text="Voice Lab is not available in Kokoro mode.\n\n"
                     "Kokoro uses 54 built-in preset voices — no cloning needed.\n"
                     "Switch to Fish-Speech 1.4 or 1.5 in Settings to clone voices.",
                font=(FONT_FAMILY, 14),
                text_color=COLORS["text_muted"],
                justify="center",
            ).pack(pady=(0, 80))
        except Exception as exc:
            logger.warning("Could not lock Voice Lab: %s", exc)


    def _on_kokoro_voice_change(self, display_name: str):
        """Save the selected Kokoro voice to settings."""
        voice_id = KOKORO_VOICES.get(display_name, DEFAULT_VOICE)
        self.settings.kokoro_voice = voice_id
        self.settings.save()

    def _on_engine_select(self, new_val: str):
        """Update the active AI engine, downloading models if needed, then restart."""
        from utils import is_fish_speech_ready, setup_fish_speech, is_kokoro_ready, setup_kokoro

        # Determine which engine was chosen
        if "Kokoro" in new_val:
            new_engine = "kokoro"
            fs_version = None
        elif "1.5" in new_val:
            new_engine = "fish15"
            fs_version = "1.5"
        else:
            new_engine = "fish14"
            fs_version = "1.4"

        # No change
        if new_engine == getattr(self.settings, 'engine', 'kokoro'):
            return

        # Save engine choice immediately
        if new_engine == "kokoro":
            self.settings.engine = 'kokoro'
        elif new_engine == "fish15":
            self.settings.fish_speech_path = os.path.join(APP_DIR, "fish-speech-1.5")
            self.settings.checkpoint_name = "checkpoints/fish-speech-1.5"
            self.settings.engine = 'fish15'
        else:
            self.settings.fish_speech_path = os.path.join(APP_DIR, "fish-speech")
            self.settings.checkpoint_name = "checkpoints/fish-speech-1.4"
            self.settings.engine = 'fish14'
        self.settings.save()

        # For Fish Speech, check if models are present and download if not
        if fs_version and not is_fish_speech_ready(fs_version):
            download = messagebox.askyesno(
                "Download Required",
                f"Fish-Speech {fs_version} models are not installed (~1.5 GB).\n\n"
                "Download now? FishTalk will restart when complete.",
            )
            if not download:
                # Revert setting
                self.settings.engine = getattr(self.settings, '_prev_engine', 'kokoro')
                self.settings.save()
                self._update_engine_dropdown()
                return

            # Show download progress in status bar and run in background
            dest = os.path.join(APP_DIR, "fish-speech" if fs_version == "1.4" else "fish-speech-1.5")

            def _on_progress(msg, _frac=None):
                self.root.after(0, lambda m=msg: self.update_tts_status(f"⬇ {m}", COLORS["warning"]))

            def _download():
                ok = setup_fish_speech(dest_dir=dest, on_progress=_on_progress, version=fs_version)
                if ok:
                    self.root.after(0, lambda: self.update_tts_status(
                        f"✅ Fish-Speech {fs_version} ready — restarting…", COLORS["success"]
                    ))
                    self.root.after(1500, self._restart_app)
                else:
                    self.root.after(0, lambda: self.update_tts_status(
                        f"❌ Download failed — check your connection", COLORS["danger"]
                    ))
                    self.root.after(0, self._update_engine_dropdown)

            threading.Thread(target=_download, daemon=True, name="FishDownload").start()
            return

        # For Kokoro, check if model files are present and download if not
        if new_engine == "kokoro" and not is_kokoro_ready():
            download = messagebox.askyesno(
                "Download Required",
                "Kokoro model files are not installed (~330 MB).\n\n"
                "Download now? FishTalk will restart when complete.",
            )
            if not download:
                self.settings.engine = getattr(self.settings, '_prev_engine', 'fish14')
                self.settings.save()
                self._update_engine_dropdown()
                return

            def _on_progress(msg, _frac=None):
                self.root.after(0, lambda m=msg: self.update_tts_status(f"⬇ {m}", COLORS["warning"]))

            def _download_kokoro():
                ok = setup_kokoro(on_progress=_on_progress)
                if ok:
                    self.root.after(0, lambda: self.update_tts_status(
                        "✅ Kokoro models ready — restarting…", COLORS["success"]
                    ))
                    self.root.after(1500, self._restart_app)
                else:
                    self.root.after(0, lambda: self.update_tts_status(
                        "❌ Download failed — check your connection", COLORS["danger"]
                    ))
                    self.root.after(0, self._update_engine_dropdown)

            threading.Thread(target=_download_kokoro, daemon=True, name="KokoroDownload").start()
            return

        # Models present — just restart
        confirm = messagebox.askyesno(
            "Restart Required",
            f"Switched to: {new_val}\n\nFishTalk needs to restart to load the new engine.\n\nRestart now?",
        )
        if confirm:
            self._restart_app()
        else:
            self._update_engine_dropdown()

    def _restart_app(self):
        """Save settings and restart the process."""
        self.settings.save()
        import sys
        os.execv(sys.executable, [sys.executable] + sys.argv)

    def _update_engine_dropdown(self):
        """Revert the engine dropdown to match the saved setting."""
        _eng = getattr(self.settings, 'engine', 'kokoro')
        if _eng == 'kokoro':
            label = "Kokoro — Fast(best) No Cloning (1-2Gb Ram)"
        elif _eng == 'fish15':
            label = "Fish-Speech 1.5 — High W/Cloning (8+Gb Ram)"
        else:
            label = "Fish-Speech 1.4 — Medium W/Cloning (1-4Gb Ram)"
        self.engine_var.set(label)


    def _browse_fish_path(self):
        path = filedialog.askdirectory(title="Select Fish-Speech directory")
        if path:
            self.fish_path_var.set(path)
            self.settings.fish_speech_path = path
            self._validate_fish_path()
            self.settings.save()

    def _validate_fish_path(self):
        path = self.fish_path_var.get()
        result = validate_fish_speech_path(path)
        if result["valid"]:
            self.fish_status_label.configure(
                text="✅  Valid",
                text_color=COLORS["success"],
            )
        else:
            self.fish_status_label.configure(
                text=f"❌  {result['message']}",
                text_color=COLORS["danger"],
            )

    def _on_tab_changed(self):
        """Handle tab switching for Memory Saver mode."""
        if not self.settings.memory_saver:
            return

        current = self.tabview.get()

        if "Read Aloud" in current:
            # On TTS tab — unload STT, load TTS
            if self.stt.is_loaded:
                self.stt.unload_model()
                self.stt_status_label.configure(
                    text="💤  Unloaded (Memory Saver)",
                    text_color=COLORS["text_muted"],
                )
        elif "Transcribe" in current:
            # On STT tab — unload TTS, keep STT
            if self.tts.is_loaded:
                self.tts.unload_model()
                self.tts_status_label.configure(
                    text="💤  Unloaded (Memory Saver)",
                    text_color=COLORS["text_muted"],
                )

    # ==================================================================
    # RAM monitoring
    # ==================================================================

    def _start_ram_monitor(self):
        """Start periodic RAM usage updates."""
        self._update_ram()

    def _update_ram(self):
        """Update RAM and VRAM readout labels."""
        try:
            ram = get_ram_usage()
            text = (
                f"App: {ram['process_mb']:.0f} MB  |  "
                f"System: {ram['system_used_gb']:.1f} / "
                f"{ram['system_total_gb']:.1f} GB ({ram['system_percent']:.0f}%)"
            )
            self.ram_label.configure(text=text)
        except Exception:
            self.ram_label.configure(text="Unable to read")

        try:
            vram = get_vram_usage()
            if vram:
                text = (
                    f"System: {vram['used_gb']:.1f} / "
                    f"{vram['total_gb']:.1f} GB ({vram['percent']:.0f}%)"
                )
                self.vram_label.configure(text=text)
            else:
                self.vram_label.configure(text="N/A (No CUDA GPU)")
        except Exception:
            self.vram_label.configure(text="Unable to read")

        # Schedule next update
        self.root.after(5000, self._update_ram)

    # ==================================================================
    # Helpers
    # ==================================================================

    def _parse_drop_data(self, data: str) -> list:
        """Parse tkinterdnd2 drop event data into file paths."""
        # Windows wraps paths with spaces in {}
        paths = []
        if "{" in data:
            import re
            paths = re.findall(r"\{([^}]+)\}", data)
            remaining = re.sub(r"\{[^}]+\}", "", data).strip()
            if remaining:
                paths.extend(remaining.split())
        else:
            paths = data.strip().split()
        return [p for p in paths if os.path.isfile(p)]

    def update_tts_status(self, text: str, color: str = None):
        """Update TTS engine status label in Settings tab."""
        self.tts_status_label.configure(
            text=text,
            text_color=color or COLORS["text_secondary"],
        )

    def update_stt_status(self, text: str, color: str = None):
        """Update STT engine status label in Settings tab."""
        self.stt_status_label.configure(
            text=text,
            text_color=color or COLORS["text_secondary"],
        )
