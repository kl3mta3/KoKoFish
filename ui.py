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
from kokoro_engine import KOKORO_VOICES, DEFAULT_VOICE, DEFAULT_VOICE_DISPLAY, install_kokoro, _is_kokoro_installed, KOKORO_LANGUAGE_GROUPS, KOKORO_DEFAULT_LANG, KOKORO_VOICE_LANG
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

        # Single-item convert queue (Option A — items waiting for their turn)
        self._single_item_queue = []   # list of playlist indices to process in order

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
    # Tooltip helper
    # ==================================================================

    @staticmethod
    def _make_tooltip(widget, text: str):
        """Attach a themed tooltip to *widget* that appears after a short hover delay."""
        tip_win = None
        _after_id = None

        def _show(event=None):
            nonlocal tip_win, _after_id
            _after_id = None
            if tip_win:
                return
            x = widget.winfo_rootx() + 10
            y = widget.winfo_rooty() + widget.winfo_height() + 4
            tip_win = tk.Toplevel(widget)
            tip_win.wm_overrideredirect(True)
            tip_win.wm_geometry(f"+{x}+{y}")
            tip_win.configure(bg="#1a1a2e")
            tk.Label(
                tip_win, text=text,
                background="#1a1a2e", foreground="#c8c8e8",
                font=("Segoe UI", 9), padx=7, pady=4,
                relief="flat",
            ).pack()

        def _schedule(event=None):
            nonlocal _after_id
            _cancel(None)
            _after_id = widget.after(550, _show)

        def _cancel(event=None):
            nonlocal tip_win, _after_id
            if _after_id:
                try:
                    widget.after_cancel(_after_id)
                except Exception:
                    pass
                _after_id = None
            if tip_win:
                try:
                    tip_win.destroy()
                except Exception:
                    pass
                tip_win = None

        widget.bind("<Enter>", _schedule, add="+")
        widget.bind("<Leave>", _cancel, add="+")
        widget.bind("<ButtonPress>", _cancel, add="+")

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

        # --- Kokoro language filter (American / British) ---
        is_kokoro = getattr(self.settings, 'engine', 'fish14') == 'kokoro'
        self._kokoro_lang_var = ctk.StringVar(value=KOKORO_DEFAULT_LANG)

        # --- Voice variable (hidden — each playlist item has its own voice dropdown) ---
        if is_kokoro:
            kokoro_display_names = list(KOKORO_VOICES.keys())
            saved_kokoro_id = getattr(self.settings, 'kokoro_voice', DEFAULT_VOICE)
            saved_display = next(
                (k for k, v in KOKORO_VOICES.items() if v == saved_kokoro_id),
                kokoro_display_names[0]
            )
            self.tts_voice_var = ctk.StringVar(value=saved_display)
        else:
            voice_names = self.voices.get_voice_names()
            self.tts_voice_var = ctk.StringVar(value=voice_names[0] if voice_names else "Default (Random)")

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
        self._make_tooltip(self.speed_label,  "Playback speed — 1.0x is normal, 0.5x is half speed, 2.0x is double")
        self._make_tooltip(self.speed_slider, "Playback speed — 1.0x is normal, 0.5x is half speed, 2.0x is double")

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
        self._make_tooltip(self.vol_label,  "Playback volume — audio is peak-normalized before this is applied")
        self._make_tooltip(self.vol_slider, "Playback volume — audio is peak-normalized before this is applied")

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
        self._make_tooltip(self.cad_label,  "Cadence — adds a brief pause between decoded speech chunks (Fish Speech only). 0% = no pause, 100% = ~600 ms gap. Has no effect on Kokoro.")
        self._make_tooltip(self.cad_slider, "Cadence — adds a brief pause between decoded speech chunks (Fish Speech only). 0% = no pause, 100% = ~600 ms gap. Has no effect on Kokoro.")

        # ── Playlist header row ──────────────────────────────────────────
        playlist_header = ctk.CTkFrame(tab, fg_color="transparent")
        playlist_header.pack(fill="x", padx=15, pady=(10, 2))

        ctk.CTkLabel(
            playlist_header,
            text="📋  Playlist",
            font=(FONT_FAMILY, 14, "bold"),
            text_color=COLORS["text_primary"],
        ).pack(side="left")
        
        ctk.CTkLabel(
        playlist_header,
        text="(Double-Click File Name to open in Editor.)",
        font=(FONT_FAMILY, 10),
        text_color=COLORS["text_secondary"],
        ).pack(side="left", padx=(5, 0))

        # ── Kokoro language filter dropdown ───────────────────────────────
        if is_kokoro:
            lang_frame = ctk.CTkFrame(playlist_header, fg_color="transparent")
            lang_frame.pack(side="left", padx=(16, 0))
            ctk.CTkLabel(
                lang_frame,
                text="Accent:",
                font=(FONT_FAMILY, 11),
                text_color=COLORS["text_muted"],
            ).pack(side="left", padx=(0, 4))
            ctk.CTkOptionMenu(
                lang_frame,
                variable=self._kokoro_lang_var,
                values=list(KOKORO_LANGUAGE_GROUPS.keys()),
                width=145,
                height=26,
                fg_color=COLORS["bg_input"],
                button_color=COLORS["accent"],
                button_hover_color=COLORS["accent_hover"],
                dropdown_fg_color=COLORS["bg_card"],
                dropdown_hover_color=COLORS["bg_card_hover"],
                font=(FONT_FAMILY, 11),
                command=lambda _: self._rebuild_playlist_ui(),
            ).pack(side="left")

        # Read All button — only batch control in the header now
        self.btn_play = ctk.CTkButton(
            playlist_header,
            text="▶ Read All",
            fg_color=COLORS["success"],
            hover_color="#05b890",
            font=(FONT_FAMILY, 12, "bold"),
            corner_radius=7, height=32, width=100,
            command=self._tts_play,
        )
        self.btn_play.pack(side="right", padx=(4, 0))
        self._make_tooltip(self.btn_play, "Convert and play all items in the playlist")

        # Work Silent + Auto Save toggles
        self.silent_mode_var = ctk.BooleanVar(value=getattr(self.settings, 'silent_mode', False))
        silent_frame = ctk.CTkFrame(playlist_header, fg_color="transparent")
        silent_frame.pack(side="right", padx=(0, 16))
        ctk.CTkLabel(
            silent_frame, text="🔇 Work Silently", font=(FONT_FAMILY, 13),
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
            auto_save_frame, text="📂 Auto Save", font=(FONT_FAMILY, 13),
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

        # ── Bottom action bar ────────────────────────────────────────────
        sel_bar = ctk.CTkFrame(tab, fg_color="transparent")
        sel_bar.pack(fill="x", padx=15, pady=(5, 10))

        _big  = {"font": (FONT_FAMILY, 12, "bold"), "corner_radius": 8, "height": 34}
        _mini = {"font": (FONT_FAMILY, 13),         "corner_radius": 6, "height": 34, "width": 36}
        _util = {"font": (FONT_FAMILY, 12, "bold"), "corner_radius": 8, "height": 34}

        # TTS Selected + its Pause / Stop
        _btn = ctk.CTkButton(
            sel_bar, text="🔊 TTS Selected",
            fg_color=COLORS["success"], hover_color="#05b890",
            command=self._tts_selected, width=140, **_big,
        )
        _btn.pack(side="left", padx=(0, 2))
        self._make_tooltip(_btn, "Convert selected items to speech")

        self.btn_pause = ctk.CTkButton(
            sel_bar, text="⏸",
            fg_color=COLORS["warning"], hover_color="#e6bc5c",
            text_color="#1a1a2e",
            command=self._tts_pause, **_mini,
        )
        self.btn_pause.pack(side="left", padx=(0, 2))
        self._make_tooltip(self.btn_pause, "Pause conversion")

        self.btn_stop = ctk.CTkButton(
            sel_bar, text="⏹",
            fg_color=COLORS["danger"], hover_color="#d43d62",
            command=self._tts_stop, **_mini,
        )
        self.btn_stop.pack(side="left", padx=(0, 12))
        self._make_tooltip(self.btn_stop, "Stop / cancel conversion")

        # Play Selected + its Pause / Stop
        _btn = ctk.CTkButton(
            sel_bar, text="▶ Play Selected",
            fg_color=COLORS["accent"], hover_color=COLORS["accent_hover"],
            command=self._play_selected, width=140, **_big,
        )
        _btn.pack(side="left", padx=(0, 2))
        self._make_tooltip(_btn, "Play audio for selected items")

        _btn = ctk.CTkButton(
            sel_bar, text="⏸",
            fg_color=COLORS["warning"], hover_color="#e6bc5c",
            text_color="#1a1a2e",
            command=self._stop_preview, **_mini,
        )
        _btn.pack(side="left", padx=(0, 2))
        self._make_tooltip(_btn, "Pause playback")

        _btn = ctk.CTkButton(
            sel_bar, text="⏹",
            fg_color=COLORS["danger"], hover_color="#d43d62",
            command=self._stop_preview, **_mini,
        )
        _btn.pack(side="left", padx=(0, 12))
        self._make_tooltip(_btn, "Stop playback")

        # Save Selected
        self.btn_save_mp3 = ctk.CTkButton(
            sel_bar, text="💾 Save Selected",
            fg_color=COLORS["accent"], hover_color=COLORS["accent_hover"],
            command=self._tts_save_mp3, width=140, **_big,
        )
        self.btn_save_mp3.pack(side="left", padx=(0, 16))
        self._make_tooltip(self.btn_save_mp3, "Export selected items as MP3")

        # Selection helpers
        _btn = ctk.CTkButton(
            sel_bar, text="☑ All",
            fg_color=COLORS["bg_input"], hover_color=COLORS["bg_card_hover"],
            border_color=COLORS["border"], border_width=1,
            command=self._select_all, width=60, **_util,
        )
        _btn.pack(side="left", padx=(0, 4))
        self._make_tooltip(_btn, "Select all items")

        _btn = ctk.CTkButton(
            sel_bar, text="☐ None",
            fg_color=COLORS["bg_input"], hover_color=COLORS["bg_card_hover"],
            border_color=COLORS["border"], border_width=1,
            command=self._deselect_all, width=60, **_util,
        )
        _btn.pack(side="left", padx=(0, 4))
        self._make_tooltip(_btn, "Deselect all items")

        # Clear Selected — far right
        _btn = ctk.CTkButton(
            sel_bar, text="🗑 Clear Selected",
            fg_color=COLORS["bg_input"], hover_color=COLORS["bg_card_hover"],
            border_color=COLORS["border"], border_width=1,
            command=self._tts_clear_playlist, width=120, **_util,
        )
        _btn.pack(side="right")
        self._make_tooltip(_btn, "Remove selected items from the playlist")

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

        def _make_gen_slider(parent, label, from_, to, initial, resolution, setting_key,
                             fmt="{:.2f}", tooltip=""):
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
            if tooltip:
                self._make_tooltip(lbl, tooltip)
                self._make_tooltip(sl, tooltip)
            return sl

        _make_gen_slider(sliders_row, "Temperature", 0.1, 1.0,
                         getattr(self.settings, "tts_temperature", 0.7), 0.05, "tts_temperature",
                         tooltip="Randomness of generation — lower = more consistent/monotone, higher = more expressive/varied. Default 0.7.")
        _make_gen_slider(sliders_row, "Top-P", 0.1, 1.0,
                         getattr(self.settings, "tts_top_p", 0.7), 0.05, "tts_top_p",
                         tooltip="Nucleus sampling — limits which tokens are considered each step. Lower = safer/safer, higher = more creative. Default 0.7.")
        _make_gen_slider(sliders_row, "Repetition Penalty", 1.0, 1.8,
                         getattr(self.settings, "tts_repetition_penalty", 1.2), 0.05, "tts_repetition_penalty",
                         tooltip="Penalizes repeating the same sounds or patterns. Raise if you hear looping/stuttering. Default 1.2.")
        _make_gen_slider(sliders_row, "Chunk Length", 50, 300,
                         getattr(self.settings, "tts_chunk_length", 150), 10, "tts_chunk_length", fmt="{:.0f}",
                         tooltip="Tokens generated per speech chunk — smaller = lower latency but choppier; larger = smoother but slower to start. Default 150.")

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
        """Update the voice variable default when voices change (per-item dropdowns rebuilt via _rebuild_playlist_ui)."""
        names = self.voices.get_voice_names()
        if self.tts_voice_var.get() not in names:
            self.tts_voice_var.set(names[0] if names else "Default (Random)")
        self._rebuild_playlist_ui()

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
            "Kokoro — Fast, No Cloning (1-2 GB RAM)",
            "Fish-Speech 1.4 — Voice Cloning, No Account (1-4 GB RAM)",
            "S1 Mini — Voice Cloning, HF Account Required (2-4 GB RAM)",
            "S1 Full — High Quality Cloning, HF Account Required (6-8 GB RAM)",
        ]

        # Determine current engine from settings.engine field
        _eng = getattr(self.settings, 'engine', 'kokoro')
        _eng_to_option = {
            "kokoro": engine_options[0],
            "fish14": engine_options[1],
            "s1mini": engine_options[2],
            "s1":     engine_options[3],
        }
        current_engine = _eng_to_option.get(_eng, engine_options[0])

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

        # --- AI Features ---
        section_header(main, "🤖  AI Features (Qwen 0.5B)")

        from tag_suggester import is_llm_available as _llm_avail, is_qwen_model_ready as _qwen_ready

        # llama-cpp-python row
        llama_row = setting_row(main)
        ctk.CTkLabel(
            llama_row,
            text="llama-cpp-python",
            font=(FONT_FAMILY, 13),
            text_color=COLORS["text_primary"],
        ).pack(side="left", padx=15, pady=12)

        ctk.CTkLabel(
            llama_row,
            text="Required for AI tags, grammar check, Assisted Flow",
            font=(FONT_FAMILY, 11),
            text_color=COLORS["text_muted"],
        ).pack(side="left", padx=(0, 10), pady=12)

        _llama_installed = _llm_avail()
        self.llama_status_label = ctk.CTkLabel(
            llama_row,
            text="✅  Installed" if _llama_installed else "❌  Not installed",
            font=(FONT_FAMILY, 11),
            text_color=COLORS["success"] if _llama_installed else COLORS["danger"],
        )
        self.llama_status_label.pack(side="right", padx=(5, 15), pady=12)

        if not _llama_installed:
            self.llama_install_btn = ctk.CTkButton(
                llama_row,
                text="⬇  Install",
                width=90,
                height=28,
                corner_radius=6,
                fg_color=COLORS["accent"],
                hover_color=COLORS["accent_hover"],
                font=(FONT_FAMILY, 11),
                command=self._install_llama_cpp,
            )
            self.llama_install_btn.pack(side="right", padx=(0, 5), pady=12)

        # Qwen model row
        qwen_row = setting_row(main)
        ctk.CTkLabel(
            qwen_row,
            text="Qwen 2.5 0.5B Model",
            font=(FONT_FAMILY, 13),
            text_color=COLORS["text_primary"],
        ).pack(side="left", padx=15, pady=12)

        ctk.CTkLabel(
            qwen_row,
            text="~400 MB — downloaded on first use",
            font=(FONT_FAMILY, 11),
            text_color=COLORS["text_muted"],
        ).pack(side="left", padx=(0, 10), pady=12)

        _qwen_installed = _qwen_ready()
        self.qwen_status_label = ctk.CTkLabel(
            qwen_row,
            text="✅  Ready" if _qwen_installed else ("⏳  Install llama-cpp first" if not _llama_installed else "❌  Not downloaded"),
            font=(FONT_FAMILY, 11),
            text_color=COLORS["success"] if _qwen_installed else COLORS["text_muted"],
        )
        self.qwen_status_label.pack(side="right", padx=(5, 15), pady=12)

        # Install log (hidden until install starts)
        self._llama_log_row = setting_row(main)
        self._llama_log_label = ctk.CTkLabel(
            self._llama_log_row,
            text="",
            font=(FONT_FAMILY, 10),
            text_color=COLORS["text_secondary"],
            justify="left",
            anchor="w",
            wraplength=560,
        )
        self._llama_log_label.pack(fill="x", padx=15, pady=8)
        self._llama_log_row.pack_forget()  # hidden until needed

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
                "blend_voice": "",
                "blend_ratio": 0.5,
                "assisted_flow": False,
                "translate": False,
                "translate_lang": "",
                "translate_tone": "Natural",
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
            # Filter by the selected language group radio
            selected_lang = getattr(self, '_kokoro_lang_var', None)
            lang_key = selected_lang.get() if selected_lang else KOKORO_DEFAULT_LANG
            voice_options = KOKORO_LANGUAGE_GROUPS.get(lang_key, list(KOKORO_VOICES.keys()))
        else:
            voice_options = self.voices.get_voice_names() or ["Default (Random)"]

        from tag_suggester import is_llm_available as _llm_avail
        _af_llm_ok = _llm_avail()

        for idx, item in enumerate(self._playlist_items):
            is_active = (idx == self._current_playing)
            is_queued = (idx in self._single_item_queue)
            if is_active:
                row_color = COLORS["accent"]
            elif is_queued:
                row_color = COLORS["bg_card_hover"]
            else:
                row_color = COLORS["bg_card"]
            row = ctk.CTkFrame(
                self.playlist_frame,
                fg_color=row_color,
                corner_radius=6,
                height=44,
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
            if has_audio:
                dot = "✅"
            elif is_active:
                dot = "⏳"
            elif is_queued:
                dot = "🕐"
            else:
                dot = "○"
            ctk.CTkLabel(
                row, text=dot, font=(FONT_FAMILY, 11), width=20,
            ).pack(side="left", padx=(0, 2))

            # ── Index + filename (double-click opens editor) ─────────────
            name_lbl = ctk.CTkLabel(
                row,
                text=f"{idx + 1}. {item['name']}",
                font=(FONT_FAMILY, 12),
                text_color=txt_color,
                anchor="w",
                cursor="hand2",
            )
            name_lbl.pack(side="left", padx=(0, 2))
            name_lbl.bind("<Double-Button-1>", lambda e, i=idx: self._open_editor(i))

            ctk.CTkLabel(
                row,
                text=f"({len(item['text']):,})",
                font=(FONT_FAMILY, 10),
                text_color=COLORS["text_muted"],
            ).pack(side="left", padx=(0, 6))

            # ── Per-item voice dropdown (after title) ────────────────────
            item_voice = item.get("voice", voice_options[0] if voice_options else "")
            if item_voice not in voice_options:
                item_voice = voice_options[0] if voice_options else item_voice
                # Keep the dict in sync so generation uses the correct voice
                self._playlist_items[idx]["voice"] = item_voice

            voice_var = ctk.StringVar(value=item_voice)

            def _on_voice_change(v, i=idx, var=voice_var):
                self._playlist_items[i]["voice"] = var.get()

            ctk.CTkOptionMenu(
                row,
                values=voice_options,
                variable=voice_var,
                width=150,
                height=26,
                fg_color=COLORS["bg_input"],
                button_color=COLORS["accent"],
                button_hover_color=COLORS["accent_hover"],
                dropdown_fg_color=COLORS["bg_card"],
                dropdown_hover_color=COLORS["bg_card_hover"],
                font=(FONT_FAMILY, 11),
                command=lambda v, i=idx, var=voice_var: _on_voice_change(v, i, var),
            ).pack(side="left", padx=(0, 2))

            # ── Kokoro blend voice + ratio slider ────────────────────────
            if is_kokoro:
                ctk.CTkLabel(
                    row, text="+", font=(FONT_FAMILY, 10),
                    text_color=COLORS["text_muted"], width=10,
                ).pack(side="left")

                blend_options = ["— blend —"] + voice_options
                item_blend = item.get("blend_voice", "") or "— blend —"
                if item_blend not in blend_options:
                    item_blend = "— blend —"

                blend_var = ctk.StringVar(value=item_blend)

                def _on_blend_change(v, i=idx, bvar=blend_var):
                    val = bvar.get()
                    self._playlist_items[i]["blend_voice"] = "" if val == "— blend —" else val

                ctk.CTkOptionMenu(
                    row,
                    values=blend_options,
                    variable=blend_var,
                    width=110,
                    height=26,
                    fg_color=COLORS["bg_input"],
                    button_color=COLORS["bg_card_hover"],
                    button_hover_color=COLORS["accent"],
                    dropdown_fg_color=COLORS["bg_card"],
                    dropdown_hover_color=COLORS["bg_card_hover"],
                    font=(FONT_FAMILY, 10),
                    command=lambda v, i=idx, bvar=blend_var: _on_blend_change(v, i, bvar),
                ).pack(side="left", padx=(0, 2))

                # Blend ratio slider + % label
                blend_ratio_val = item.get("blend_ratio", 0.5)

                blend_ratio_label = ctk.CTkLabel(
                    row,
                    text=f"{int(blend_ratio_val * 100)}%",
                    font=(FONT_FAMILY, 9),
                    text_color=COLORS["text_muted"],
                    width=28,
                )

                def _on_blend_ratio(v, i=idx, lbl=blend_ratio_label):
                    ratio = round(float(v), 2)
                    self._playlist_items[i]["blend_ratio"] = ratio
                    lbl.configure(text=f"{int(ratio * 100)}%")

                _blend_slider = ctk.CTkSlider(
                    row,
                    from_=0.1, to=0.9,
                    number_of_steps=8,
                    width=72,
                    height=14,
                    progress_color=COLORS["accent"],
                    button_color=COLORS["accent_light"],
                    button_hover_color=COLORS["accent"],
                    command=lambda v, i=idx, lbl=blend_ratio_label: _on_blend_ratio(v, i, lbl),
                )
                _blend_slider.set(blend_ratio_val)
                _blend_slider.pack(side="left", padx=(0, 1))
                self._make_tooltip(_blend_slider, "Blend ratio — left = more primary voice, right = more blend voice")
                blend_ratio_label.pack(side="left", padx=(0, 4))

            # ── Assisted Flow toggle ─────────────────────────────────────
            af_var = ctk.BooleanVar(value=item.get("assisted_flow", False) if _af_llm_ok else False)

            def _on_af_toggle(v=af_var, i=idx):
                self._playlist_items[i]["assisted_flow"] = v.get()

            ctk.CTkLabel(
                row, text="Assisted Flow",
                font=(FONT_FAMILY, 9),
                text_color=COLORS["accent_light"] if _af_llm_ok else COLORS["text_muted"],
                width=18,
            ).pack(side="left", padx=(4, 1))

            _af_switch = ctk.CTkSwitch(
                row,
                text="",
                variable=af_var,
                command=_on_af_toggle,
                state="normal" if _af_llm_ok else "disabled",
                width=36,
                height=20,
                switch_width=32,
                switch_height=16,
                progress_color=COLORS["accent"],
                button_color=COLORS["accent_light"] if _af_llm_ok else COLORS["text_muted"],
                button_hover_color=COLORS["accent"],
            )
            _af_switch.pack(side="left", padx=(0, 2))
            self._make_tooltip(_af_switch, "Assisted Flow — AI enhances text phrasing before TTS")

            if not _af_llm_ok:
                ctk.CTkLabel(
                    row,
                    text="⚠ Settings",
                    font=(FONT_FAMILY, 8),
                    text_color=COLORS["text_muted"],
                ).pack(side="left", padx=(0, 2))
            else:
                ctk.CTkFrame(row, width=2, fg_color="transparent").pack(side="left")

            # ── Translate toggle ──────────────────────────────────────────
            # Kokoro: auto-detects language from voice ID (no dropdown needed)
            # Fish engines: dropdown lets user pick target language
            if _af_llm_ok:
                tr_var = ctk.BooleanVar(value=item.get("translate", False))

                def _on_tr_toggle(v=tr_var, i=idx):
                    self._playlist_items[i]["translate"] = v.get()

                ctk.CTkLabel(
                    row, text="Translate",
                    font=(FONT_FAMILY, 9),
                    text_color=COLORS["accent_light"],
                    width=18,
                ).pack(side="left", padx=(4, 1))

                _tr_switch = ctk.CTkSwitch(
                    row,
                    text="",
                    variable=tr_var,
                    command=_on_tr_toggle,
                    width=36,
                    height=20,
                    switch_width=32,
                    switch_height=16,
                    progress_color="#f4a261",
                    button_color="#e76f51",
                    button_hover_color="#f4a261",
                )
                _tr_switch.pack(side="left", padx=(0, 2))

                if is_kokoro:
                    self._make_tooltip(
                        _tr_switch,
                        "Translate — AI translates text into the voice's language before generating speech",
                    )
                else:
                    # Non-Kokoro: show language dropdown next to the toggle
                    from tag_suggester import TRANSLATE_LANGUAGES, TRANSLATE_TONES
                    saved_lang = item.get("translate_lang", "") or TRANSLATE_LANGUAGES[0]
                    tr_lang_var = ctk.StringVar(value=saved_lang)

                    def _on_tr_lang(v, i=idx, lvar=tr_lang_var):
                        self._playlist_items[i]["translate_lang"] = lvar.get()

                    _tr_lang_menu = ctk.CTkOptionMenu(
                        row,
                        variable=tr_lang_var,
                        values=TRANSLATE_LANGUAGES,
                        width=130,
                        height=22,
                        fg_color=COLORS["bg_input"],
                        button_color="#e76f51",
                        button_hover_color="#f4a261",
                        dropdown_fg_color=COLORS["bg_card"],
                        dropdown_hover_color=COLORS["bg_card_hover"],
                        font=(FONT_FAMILY, 10),
                        command=lambda v, i=idx, lvar=tr_lang_var: _on_tr_lang(v, i, lvar),
                    )
                    _tr_lang_menu.pack(side="left", padx=(0, 2))
                    self._make_tooltip(
                        _tr_switch,
                        "Translate — AI translates text into the selected language before generating speech",
                    )

            # ── Per-item buttons (icon-only, 32×32) ──────────────────────
            _ib = {"width": 32, "height": 32, "corner_radius": 5, "font": (FONT_FAMILY, 14)}

            # Always rightmost: remove ✕
            _rb = ctk.CTkButton(
                row, text="✕",
                fg_color=COLORS["danger"], hover_color="#d43d62",
                command=lambda i=idx: self._tts_remove_item(i),
                **_ib,
            )
            _rb.pack(side="right", padx=(0, 6))
            self._make_tooltip(_rb, "Remove item from playlist")

            # After audio exists (near ✕): play/pause 🔊, save 📁
            if has_audio:
                is_previewing = (idx == self._preview_idx and not self._preview_paused)
                _sb = ctk.CTkButton(
                    row, text="📁",
                    fg_color=COLORS["bg_input"], hover_color=COLORS["bg_card_hover"],
                    border_color=COLORS["border"], border_width=1,
                    command=lambda i=idx: self._save_item_audio(i),
                    **_ib,
                )
                _sb.pack(side="right", padx=(0, 2))
                self._make_tooltip(_sb, "Save audio to file")
                _pb = ctk.CTkButton(
                    row, text="⏸" if is_previewing else "▶",
                    fg_color=COLORS["success"] if is_previewing else COLORS["bg_input"],
                    hover_color="#05b890" if is_previewing else COLORS["bg_card_hover"],
                    border_color=COLORS["success"], border_width=1,
                    command=lambda i=idx: self._preview_item(i),
                    **_ib,
                )
                _pb.pack(side="right", padx=(0, 2))
                self._make_tooltip(_pb, "Pause preview" if is_previewing else "Play audio preview")

            # After voice dropdowns:
            #  Active  → ⏸ pause  ⊘ cancel
            #  Queued  → ⊘ remove from queue
            #  Idle    → ▶ convert  (accent = not done, muted = re-convert)
            if is_active:
                _pb = ctk.CTkButton(
                    row, text="⏸",
                    fg_color=COLORS["warning"], hover_color="#e6bc5c",
                    text_color="#1a1a2e",
                    command=self._tts_pause,
                    **_ib,
                )
                _pb.pack(side="left", padx=(4, 2))
                self._make_tooltip(_pb, "Pause conversion")
                _cb = ctk.CTkButton(
                    row, text="⊘",
                    fg_color=COLORS["danger"], hover_color="#d43d62",
                    command=lambda i=idx: self._cancel_item(i),
                    **_ib,
                )
                _cb.pack(side="left", padx=(0, 2))
                self._make_tooltip(_cb, "Cancel conversion")
            elif is_queued:
                _cb = ctk.CTkButton(
                    row, text="⊘",
                    fg_color=COLORS["danger"], hover_color="#d43d62",
                    command=lambda i=idx: self._cancel_item(i),
                    **_ib,
                )
                _cb.pack(side="left", padx=(4, 2))
                self._make_tooltip(_cb, "Remove from queue")
            else:
                _cvt = ctk.CTkButton(
                    row, text="▶",
                    fg_color=COLORS["accent"] if not has_audio else COLORS["bg_input"],
                    hover_color=COLORS["accent_hover"] if not has_audio else COLORS["bg_card_hover"],
                    border_color=COLORS["accent"], border_width=1,
                    command=lambda i=idx: self._process_single_item(i),
                    **_ib,
                )
                _cvt.pack(side="left", padx=(4, 2))
                self._make_tooltip(_cvt, "Re-convert to speech" if has_audio else "Convert to speech")

    def _open_editor(self, index: int):
        """Open the text editor window for a playlist item."""
        if index < 0 or index >= len(self._playlist_items):
            return
        from text_editor_window import TextEditorWindow
        engine = getattr(self.settings, 'engine', 'fish14')
        TextEditorWindow(
            parent=self.root,
            item=self._playlist_items[index],
            engine=engine,
            on_save=self._rebuild_playlist_ui,
        )

    def _process_single_item(self, index: int):
        """Convert/generate TTS for one item — queues automatically if another is running."""
        if index < 0 or index >= len(self._playlist_items):
            return
        item = self._playlist_items[index]
        has_audio = bool(item.get("audio_path") and os.path.isfile(item.get("audio_path", "")))

        # If already in queue or actively running, treat ▶ as a dequeue/cancel request
        if index == self._current_playing:
            return  # already running — ⊘ handles cancel
        if index in self._single_item_queue:
            self._single_item_queue.remove(index)
            self._rebuild_playlist_ui()
            return

        if has_audio:
            if not messagebox.askyesno(
                "Re-generate",
                f"'{item['name']}' already has audio.\nRe-generate it?",
                parent=self.root,
            ):
                return

        # If something is running, add to queue and bail
        if self._is_playing:
            self._single_item_queue.append(index)
            self._rebuild_playlist_ui()
            self.tts_status.configure(text=f"Queued: {item['name']}")
            return

        # Nothing running — start immediately
        if not self.tts.is_loaded:
            self.tts_status.configure(text="Loading model…")
            for btn in (self.btn_play, self.btn_pause, self.btn_save_mp3):
                btn.configure(state="disabled")
            self._ensure_tts_loaded(lambda: self._process_single_item(index))
            return

        self._is_playing = True
        self._current_playing = index
        self._tts_queue = []
        self._rebuild_playlist_ui()
        self._play_current_item(advance_fn=self._finish_single_item)

    def _finish_single_item(self):
        """Called when a single-item conversion finishes — auto-advances queue."""
        self._is_playing = False
        self._current_playing = -1
        if self._single_item_queue:
            next_idx = self._single_item_queue.pop(0)
            self.root.after(200, lambda: self._process_single_item(next_idx))
        else:
            self._rebuild_playlist_ui()
            self.tts_status.configure(text="Done")

    def _cancel_item(self, index: int):
        """Cancel an actively generating item or remove it from the queue."""
        if index == self._current_playing:
            # Stop generation, then advance queue
            self.tts.cancel()
            self._is_playing = False
            self._current_playing = -1
            if self._single_item_queue:
                next_idx = self._single_item_queue.pop(0)
                self.root.after(200, lambda: self._process_single_item(next_idx))
            else:
                self._rebuild_playlist_ui()
                self.tts_status.configure(text="Cancelled")
        elif index in self._single_item_queue:
            self._single_item_queue.remove(index)
            self._rebuild_playlist_ui()
            self.tts_status.configure(text=f"Removed from queue: {self._playlist_items[index]['name']}")

    def _save_item_audio(self, index: int):
        """Save the generated audio for a single playlist item to a user-chosen path."""
        if index < 0 or index >= len(self._playlist_items):
            return
        item = self._playlist_items[index]
        src = item.get("audio_path", "")
        if not src or not os.path.isfile(src):
            messagebox.showwarning("Save Audio", "No generated audio for this item yet.", parent=self.root)
            return
        from tkinter.filedialog import asksaveasfilename
        stem = os.path.splitext(item["name"])[0]
        dest = asksaveasfilename(
            parent=self.root,
            defaultextension=".mp3",
            filetypes=[("MP3 audio", "*.mp3"), ("WAV audio", "*.wav"), ("All files", "*.*")],
            initialfile=f"{stem}.mp3",
            title="Save audio as…",
        )
        if not dest:
            return
        try:
            from pydub import AudioSegment
            audio = AudioSegment.from_wav(src)
            fmt = os.path.splitext(dest)[1].lstrip(".").lower() or "mp3"
            audio.export(dest, format=fmt)
            self.tts_status.configure(text=f"Saved: {os.path.basename(dest)}")
        except Exception as exc:
            messagebox.showerror("Save Audio", f"Export failed:\n{exc}", parent=self.root)

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
        self._single_item_queue.clear()
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

    def _play_current_item(self, advance_fn=None, _text_override=None):
        """Generate and play the current playlist item."""
        if self._current_playing < 0 or self._current_playing >= len(self._playlist_items):
            self._is_playing = False
            self._current_playing = -1
            self._rebuild_playlist_ui()
            self.tts_status.configure(text="Playlist complete")
            return

        item = self._playlist_items[self._current_playing]

        # ── Preprocessing: Translate → Assisted Flow (chain in one thread) ─
        _is_kokoro_mode = getattr(self.settings, 'engine', 'fish14') == 'kokoro'
        _needs_translate = item.get("translate", False)
        _needs_af        = item.get("assisted_flow", False)

        # Resolve target language now (on main thread) before spawning worker
        if _needs_translate:
            if _is_kokoro_mode:
                _item_voice_id  = KOKORO_VOICES.get(item.get("voice", ""), "")
                _target_lang    = KOKORO_VOICE_LANG.get(_item_voice_id, "English")
            else:
                _target_lang = item.get("translate_lang", "") or "Japanese"
        else:
            _target_lang   = "English"
            _item_voice_id = ""

        # Skip translate if target is English (no-op) or nothing to do
        _needs_translate = _needs_translate and _target_lang != "English"

        if _text_override is None and (_needs_translate or _needs_af):
            from tag_suggester import is_llm_available, is_qwen_model_ready
            if is_llm_available() and is_qwen_model_ready():
                self._rebuild_playlist_ui()
                _pre_engine = getattr(self.settings, 'engine', 'fish14')

                def _preprocess(
                    _eng=_pre_engine,
                    _lang=_target_lang,
                    _do_tr=_needs_translate,
                    _do_af=_needs_af,
                ):
                    current_text = item["text"]

                    # ── Step 1: Translate ────────────────────────────────
                    if _do_tr:
                        try:
                            from tag_suggester import translate_for_voice
                            self.root.after(0, lambda l=_lang: self.tts_status.configure(
                                text=f"Translating {item['name']} → {l}…"
                            ))
                            translated = translate_for_voice(current_text, _lang)
                            if translated and translated.strip():
                                current_text = translated
                        except Exception as exc:
                            logger.warning("Translate step failed: %s", exc)

                    # ── Step 2: Assisted Flow ────────────────────────────
                    if _do_af:
                        try:
                            from tag_suggester import enhance_for_tts
                            self.root.after(0, lambda: self.tts_status.configure(
                                text=f"AI Flow: enhancing {item['name']}…"
                            ))
                            enhanced = enhance_for_tts(current_text, engine=_eng)
                            if enhanced and enhanced.strip():
                                current_text = enhanced
                        except Exception as exc:
                            logger.warning("Assisted Flow failed: %s", exc)

                    self.root.after(0, lambda t=current_text: self._play_current_item(
                        advance_fn=advance_fn, _text_override=t
                    ))

                threading.Thread(target=_preprocess, daemon=True, name="Preprocess").start()
                return

        text_to_use = _text_override if _text_override is not None else item["text"]

        self.tts_status.configure(text=f"Generating: {item['name']}...")
        self._rebuild_playlist_ui()

        # Use per-item voice, fall back to global selector
        voice_name = item.get("voice") or self.tts_voice_var.get()
        profile = None
        if voice_name != "Default (Random)":
            profile = self.voices.get_voice(voice_name)

        speed = self.speed_slider.get()
        cadence = self.cad_slider.get() / 100.0  # 0.0–1.0

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
            # Per-item blend voice (also resolve display name → ID)
            blend_name = item.get("blend_voice", "")
            blend_voice_id = KOKORO_VOICES.get(blend_name, blend_name) if blend_name else ""
            self.tts.generate(
                text=text_to_use,
                voice_id=voice_id,
                blend_voice=blend_voice_id,
                blend_ratio=item.get("blend_ratio", 0.5),
                speed=speed,
                output_path=_output_path,
                on_progress=on_progress,
                on_chunk=on_chunk,
                on_complete=on_complete,
                on_error=on_error,
            )
        else:
            self.tts.generate(
                text=text_to_use,
                reference_wav=profile["wav_path"] if profile else None,
                reference_tokens=None,
                prompt_text=profile["prompt_text"] if profile else None,
                speed=speed,
                cadence=cadence,
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
        """Stop playback, cancel any pending generation, and clear the queue."""
        self._stop_preview()
        self._is_playing = False
        self._is_paused = False
        self._current_playing = -1
        self._single_item_queue.clear()
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
        """Export all selected items that have audio to a chosen folder."""
        ready = [
            it for it in self._playlist_items
            if it.get("selected", True)
            and it.get("audio_path")
            and os.path.isfile(it["audio_path"])
        ]
        if not ready:
            messagebox.showinfo("Save Selected", "No generated audio found for selected items.\nGenerate speech first.", parent=self.root)
            return

        dest_dir = filedialog.askdirectory(
            title=f"Save {len(ready)} audio file(s) to folder…",
            initialdir=os.path.expanduser("~/Documents"),
            parent=self.root,
        )
        if not dest_dir:
            return

        errors = []
        saved = 0
        for it in ready:
            src = it["audio_path"]
            stem = os.path.splitext(it["name"])[0]
            dest = os.path.join(dest_dir, stem + ".mp3")
            # Avoid overwriting: append index if name already taken
            counter = 1
            while os.path.exists(dest):
                dest = os.path.join(dest_dir, f"{stem}_{counter}.mp3")
                counter += 1
            try:
                from pydub import AudioSegment
                AudioSegment.from_wav(src).export(dest, format="mp3")
                saved += 1
            except Exception as exc:
                errors.append(f"{it['name']}: {exc}")

        if errors:
            messagebox.showwarning(
                "Save Selected",
                f"Saved {saved}/{len(ready)} files.\n\nErrors:\n" + "\n".join(errors),
                parent=self.root,
            )
        else:
            self.tts_status.configure(text=f"✅ Saved {saved} file(s) to {os.path.basename(dest_dir)}")

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
            for widget in self.tab_voices.winfo_children():  # noqa
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
                text=f"Voice Lab is not available in Kokoro mode.\n\n"
                     f"Kokoro uses {len(KOKORO_VOICES)} built-in preset voices across {len(KOKORO_LANGUAGE_GROUPS)} languages — no cloning needed.\n"
                     f"Switch to Fish-Speech 1.4, S1 Mini, or S1 Full in Settings to clone voices.",
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
        from utils import is_fish_speech_ready, setup_fish_speech, is_kokoro_ready, setup_kokoro, FISH_ENGINE_CONFIG

        # Map display label → engine key
        if "Kokoro" in new_val:
            new_engine = "kokoro"
        elif "1.4" in new_val:
            new_engine = "fish14"
        elif "S1 Mini" in new_val:
            new_engine = "s1mini"
        elif "S1 Full" in new_val:
            new_engine = "s1"
        else:
            new_engine = "kokoro"

        prev_engine = getattr(self.settings, 'engine', 'kokoro')

        # No change
        if new_engine == prev_engine:
            return

        # Save engine choice immediately
        self.settings.engine = new_engine
        if new_engine in FISH_ENGINE_CONFIG:
            ckpt_folder, _repo, _needs_token = FISH_ENGINE_CONFIG[new_engine]
            self.settings.fish_speech_path = os.path.join(APP_DIR, "fish-speech")
            self.settings.checkpoint_name = f"checkpoints/{ckpt_folder}"
        self.settings.save()

        # Fish Speech engines — check presence, download on first use
        if new_engine in FISH_ENGINE_CONFIG:
            if not is_fish_speech_ready(new_engine):
                _ckpt_folder, _hf_repo, needs_token = FISH_ENGINE_CONFIG[new_engine]
                if needs_token:
                    # Show HF token dialog; it handles download + restart on success
                    self._show_hf_token_dialog(new_engine, prev_engine)
                    return
                else:
                    # fish14 — no account required
                    download = messagebox.askyesno(
                        "Download Required",
                        f"Fish-Speech 1.4 models are not installed (~1.5 GB).\n\n"
                        "Download now? FishTalk will restart when complete.",
                    )
                    if not download:
                        self.settings.engine = prev_engine
                        self.settings.save()
                        self._update_engine_dropdown()
                        return

                    def _on_progress(msg, _frac=None):
                        self.root.after(0, lambda m=msg: self.update_tts_status(f"⬇ {m}", COLORS["warning"]))

                    def _download(_eng=new_engine):
                        ok = setup_fish_speech(engine=_eng, on_progress=_on_progress)
                        if ok:
                            self.root.after(0, lambda: self.update_tts_status(
                                "✅ Fish-Speech 1.4 ready — restarting…", COLORS["success"]
                            ))
                            self.root.after(1500, self._restart_app)
                        else:
                            self.root.after(0, lambda: self.update_tts_status(
                                "❌ Download failed — check your connection", COLORS["danger"]
                            ))
                            self.root.after(0, self._update_engine_dropdown)

                    threading.Thread(target=_download, daemon=True, name="FishDownload").start()
                    return

            # Models already present — just restart
            confirm = messagebox.askyesno(
                "Restart Required",
                f"Switched to: {new_val}\n\nFishTalk needs to restart to load the new engine.\n\nRestart now?",
            )
            if confirm:
                self._restart_app()
            else:
                self._update_engine_dropdown()
            return

        # Kokoro — check presence, download if missing
        if new_engine == "kokoro":
            if not is_kokoro_ready():
                download = messagebox.askyesno(
                    "Download Required",
                    "Kokoro model files are not installed (~330 MB).\n\n"
                    "Download now? FishTalk will restart when complete.",
                )
                if not download:
                    self.settings.engine = prev_engine
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

            confirm = messagebox.askyesno(
                "Restart Required",
                f"Switched to: {new_val}\n\nFishTalk needs to restart to load the new engine.\n\nRestart now?",
            )
            if confirm:
                self._restart_app()
            else:
                self._update_engine_dropdown()

    def _show_hf_token_dialog(self, engine: str, prev_engine: str):
        """Show a dialog prompting the user for their HuggingFace token to download a gated model."""
        from utils import setup_fish_speech, FISH_ENGINE_CONFIG
        import webbrowser

        _ckpt_folder, hf_repo, _needs_token = FISH_ENGINE_CONFIG[engine]
        engine_display = "S1 Mini" if engine == "s1mini" else "S1 Full"
        model_size = "~2.5 GB" if engine == "s1mini" else "~6 GB"

        dialog = ctk.CTkToplevel(self.root)
        dialog.title(f"HuggingFace Token Required — {engine_display}")
        dialog.geometry("520x400")
        dialog.grab_set()
        dialog.resizable(False, False)
        dialog.configure(fg_color=COLORS["bg_card"])

        ctk.CTkLabel(
            dialog,
            text=f"Download {engine_display} ({model_size})",
            font=(FONT_FAMILY, 16, "bold"),
            text_color=COLORS["text_primary"],
        ).pack(pady=(24, 4))

        ctk.CTkLabel(
            dialog,
            text=(
                f"This model is hosted on HuggingFace and requires you to\n"
                f"agree to its license and provide an access token.\n\n"
                f"1. Click the link below to visit the model page\n"
                f"2. Log in and click \"Agree\" to accept the license\n"
                f"3. Go to huggingface.co/settings/tokens and create a token\n"
                f"4. Paste it below"
            ),
            font=(FONT_FAMILY, 12),
            text_color=COLORS["text_secondary"],
            justify="left",
        ).pack(padx=24, pady=(0, 8))

        link_btn = ctk.CTkButton(
            dialog,
            text=f"Open Model Page: {hf_repo}",
            font=(FONT_FAMILY, 12),
            fg_color="transparent",
            text_color=COLORS["accent"],
            hover_color=COLORS["bg_card_hover"],
            command=lambda: webbrowser.open(f"https://huggingface.co/{hf_repo}"),
        )
        link_btn.pack(padx=24, pady=(0, 12))

        ctk.CTkLabel(
            dialog,
            text="HuggingFace Token:",
            font=(FONT_FAMILY, 12),
            text_color=COLORS["text_primary"],
            anchor="w",
        ).pack(padx=24, fill="x")

        token_entry = ctk.CTkEntry(
            dialog,
            placeholder_text="hf_...",
            font=(FONT_FAMILY, 12),
            fg_color=COLORS["bg_input"],
            show="*",
        )
        token_entry.pack(padx=24, fill="x", pady=(4, 4))

        # Pre-fill if we already have a saved token
        saved_token = getattr(self.settings, 'hf_token', '')
        if saved_token:
            token_entry.insert(0, saved_token)

        status_label = ctk.CTkLabel(
            dialog, text="", font=(FONT_FAMILY, 11), text_color=COLORS["text_muted"]
        )
        status_label.pack(padx=24)

        def _cancel():
            self.settings.engine = prev_engine
            self.settings.save()
            self._update_engine_dropdown()
            dialog.destroy()

        def _start_download():
            token = token_entry.get().strip()
            if not token:
                status_label.configure(text="Please enter a token.", text_color=COLORS["danger"])
                return
            self.settings.hf_token = token
            self.settings.save()
            download_btn.configure(state="disabled", text="Downloading…")
            status_label.configure(text="Starting download…", text_color=COLORS["warning"])

            def _on_progress(msg, _frac=None):
                dialog.after(0, lambda m=msg: status_label.configure(
                    text=m, text_color=COLORS["warning"]
                ))

            def _download(_eng=engine, _tok=token):
                ok = setup_fish_speech(engine=_eng, on_progress=_on_progress, hf_token=_tok)
                if ok:
                    dialog.after(0, lambda: status_label.configure(
                        text=f"✅ {engine_display} ready — restarting…", text_color=COLORS["success"]
                    ))
                    dialog.after(1500, lambda: (dialog.destroy(), self._restart_app()))
                else:
                    dialog.after(0, lambda: status_label.configure(
                        text="❌ Download failed — check token or connection.", text_color=COLORS["danger"]
                    ))
                    dialog.after(0, lambda: download_btn.configure(state="normal", text="Retry"))

            threading.Thread(target=_download, daemon=True, name="HFDownload").start()

        btn_row = ctk.CTkFrame(dialog, fg_color="transparent")
        btn_row.pack(fill="x", padx=24, pady=(12, 20))

        ctk.CTkButton(
            btn_row,
            text="Cancel",
            width=100,
            fg_color=COLORS["bg_input"],
            hover_color=COLORS["bg_card_hover"],
            command=_cancel,
        ).pack(side="left")

        download_btn = ctk.CTkButton(
            btn_row,
            text=f"Save Token & Download",
            width=180,
            fg_color=COLORS["accent"],
            command=_start_download,
        )
        download_btn.pack(side="right")

    def _restart_app(self):
        """Save settings and restart the process."""
        self.settings.save()
        import sys
        os.execv(sys.executable, [sys.executable] + sys.argv)

    def _update_engine_dropdown(self):
        """Revert the engine dropdown to match the saved setting."""
        _eng = getattr(self.settings, 'engine', 'kokoro')
        _eng_to_option = {
            "kokoro": "Kokoro — Fast, No Cloning (1-2 GB RAM)",
            "fish14": "Fish-Speech 1.4 — Voice Cloning, No Account (1-4 GB RAM)",
            "s1mini": "S1 Mini — Voice Cloning, HF Account Required (2-4 GB RAM)",
            "s1":     "S1 Full — High Quality Cloning, HF Account Required (6-8 GB RAM)",
        }
        self.engine_var.set(_eng_to_option.get(_eng, _eng_to_option["kokoro"]))


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

    def _install_llama_cpp(self):
        """Install llama-cpp-python from the Settings tab with inline log output."""
        from tag_suggester import install_llama_cpp, download_qwen_model, is_qwen_model_ready

        # Show log row and disable button
        self._llama_log_row.pack(fill="x", padx=10, pady=2)
        if hasattr(self, "llama_install_btn"):
            self.llama_install_btn.configure(state="disabled", text="Installing…")
        self._llama_log_label.configure(text="Starting install…", text_color=COLORS["text_secondary"])

        def _on_line(text):
            self.root.after(0, lambda t=text: self._llama_log_label.configure(text=t))

        def _on_complete(ok, msg):
            def _ui():
                if ok:
                    self.llama_status_label.configure(text="✅  Installed", text_color=COLORS["success"])
                    self._llama_log_label.configure(
                        text="✅  Installed. Downloading Qwen model next…",
                        text_color=COLORS["success"],
                    )
                    if hasattr(self, "llama_install_btn"):
                        self.llama_install_btn.pack_forget()
                    # Now download Qwen if not already present
                    if not is_qwen_model_ready():
                        self._download_qwen_from_settings()
                    else:
                        self.qwen_status_label.configure(text="✅  Ready", text_color=COLORS["success"])
                        self._rebuild_playlist_ui()
                else:
                    self.llama_status_label.configure(text="❌  Install failed", text_color=COLORS["danger"])
                    self._llama_log_label.configure(
                        text=f"❌  {msg}",
                        text_color=COLORS["danger"],
                    )
                    if hasattr(self, "llama_install_btn"):
                        self.llama_install_btn.configure(state="normal", text="⬇  Retry")
            self.root.after(0, _ui)

        install_llama_cpp(on_line=_on_line, on_complete=_on_complete)

    def _download_qwen_from_settings(self):
        """Download Qwen model and update the settings tab status labels."""
        from tag_suggester import download_qwen_model

        self.qwen_status_label.configure(text="⬇  Downloading…", text_color=COLORS["warning"])

        def _progress(status, frac):
            self.root.after(0, lambda s=status: self._llama_log_label.configure(
                text=s, text_color=COLORS["text_secondary"]
            ))

        def _complete(ok, msg):
            def _ui():
                if ok:
                    self.qwen_status_label.configure(text="✅  Ready", text_color=COLORS["success"])
                    self._llama_log_label.configure(
                        text="✅  Qwen model ready. AI Features are now available.",
                        text_color=COLORS["success"],
                    )
                    self._rebuild_playlist_ui()
                else:
                    self.qwen_status_label.configure(text="❌  Download failed", text_color=COLORS["danger"])
                    self._llama_log_label.configure(
                        text=f"❌  {msg}",
                        text_color=COLORS["danger"],
                    )
            self.root.after(0, _ui)

        download_qwen_model(on_progress=_progress, on_complete=_complete)

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
