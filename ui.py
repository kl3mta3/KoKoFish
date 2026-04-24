"""
KoKoFish — User Interface.

CustomTkinter dark-mode GUI with 4 tabs:
  Tab 1: Read Aloud (TTS)
  Tab 2: Transcribe (STT)
  Tab 3: Voice Lab
  Tab 4: Settings
"""

import logging
import os
import subprocess
import threading
import tkinter as tk
from tkinter import filedialog, messagebox
from typing import Optional

# Suppress console windows on Windows for all subprocess calls
_NO_WIN = getattr(subprocess, "CREATE_NO_WINDOW", 0x08000000)

import customtkinter as ctk
import numpy as np

from settings import (
    Settings,
    detect_cuda,
    VALID_ENGINES,
    ENGINE_LABELS,
    engine_label,
    engine_id_from_label,
)
from cuda_setup import has_nvidia_gpu, get_nvidia_gpu_name, is_cuda_torch_installed, install_cuda_pytorch, revert_to_cpu_pytorch
from kokoro_engine import KOKORO_VOICES, DEFAULT_VOICE, DEFAULT_VOICE_DISPLAY, install_kokoro, _is_kokoro_installed, KOKORO_LANGUAGE_GROUPS, KOKORO_DEFAULT_LANG, KOKORO_VOICE_LANG
from utils import (
    get_ram_usage,
    get_vram_usage,
    get_cpu_usage,
    is_ffmpeg_available,
    read_file,
    export_mp3,
    export_txt,
    export_docx,
    export_pdf,
)
from voice_manager import VoiceManager
from lang import t

import re
import time

logger = logging.getLogger("KoKoFish.ui")

# ---------------------------------------------------------------------------
# Dropdown translation maps — (english_value, translation_key) pairs.
# Used to display translated names in OptionMenus while keeping English
# values for API calls and settings persistence.
# ---------------------------------------------------------------------------
_LANG_KEY_MAP = [
    ("Japanese",             "PROMPT_LAB_TRANSLATE_LANG_JAPANESE"),
    ("Mandarin Chinese",     "PROMPT_LAB_TRANSLATE_LANG_MANDARIN"),
    ("Spanish",              "PROMPT_LAB_TRANSLATE_LANG_SPANISH"),
    ("French",               "PROMPT_LAB_TRANSLATE_LANG_FRENCH"),
    ("German",               "PROMPT_LAB_TRANSLATE_LANG_GERMAN"),
    ("Hindi",                "PROMPT_LAB_TRANSLATE_LANG_HINDI"),
    ("Italian",              "PROMPT_LAB_TRANSLATE_LANG_ITALIAN"),
    ("Brazilian Portuguese", "PROMPT_LAB_TRANSLATE_LANG_PORTUGUESE"),
    ("Korean",               "PROMPT_LAB_TRANSLATE_LANG_KOREAN"),
    ("Russian",              "PROMPT_LAB_TRANSLATE_LANG_RUSSIAN"),
    ("Arabic",               "PROMPT_LAB_TRANSLATE_LANG_ARABIC"),
    ("English",              "PROMPT_LAB_TRANSLATE_LANG_ENGLISH"),
]
_TONE_KEY_MAP = [
    ("Natural",      "PROMPT_LAB_TRANSLATE_TONE_NATURAL"),
    ("Formal",       "PROMPT_LAB_TRANSLATE_TONE_FORMAL"),
    ("Casual",       "PROMPT_LAB_TRANSLATE_TONE_CASUAL"),
    ("Professional", "PROMPT_LAB_TRANSLATE_TONE_PROFESSIONAL"),
]
_PRESET_KEY_MAP = [
    ("General Assistant",  "PROMPT_LAB_PRESET_GENERAL"),
    ("Brainstorm Partner", "PROMPT_LAB_PRESET_BRAINSTORM"),
    ("Writing Helper",     "PROMPT_LAB_PRESET_WRITING"),
    ("Story Ideas",        "PROMPT_LAB_PRESET_STORY"),
    ("Dialogue Writer",    "PROMPT_LAB_PRESET_DIALOGUE"),
    ("Script / Podcast",   "PROMPT_LAB_PRESET_SCRIPT"),
    ("Summariser",         "PROMPT_LAB_PRESET_SUMMARISE"),
    ("Dark Fiction",       "PROMPT_LAB_PRESET_DARK"),
    ("Custom",             "PROMPT_LAB_PRESET_CUSTOM"),
]
_CONTENT_STYLE_KEY_MAP = [
    ("None",               "SPEECH_LAB_CONTENT_STYLE_NONE"),
    ("Formal",             "SPEECH_LAB_CONTENT_STYLE_FORMAL"),
    ("Casual",             "SPEECH_LAB_CONTENT_STYLE_CASUAL"),
    ("Scientific",         "SPEECH_LAB_CONTENT_STYLE_SCIENTIFIC"),
    ("Professional",       "SPEECH_LAB_CONTENT_STYLE_PROFESSIONAL"),
    ("Story — Fiction",    "SPEECH_LAB_CONTENT_STYLE_STORY_FICTION"),
    ("Story — Non-Fiction","SPEECH_LAB_CONTENT_STYLE_STORY_NONFICTION"),
    ("Podcast",            "SPEECH_LAB_CONTENT_STYLE_PODCAST"),
]


def _display_names(key_map):
    """Return list of translated display names for a key_map."""
    return [t(tk) for _, tk in key_map]


def _en_from_display(key_map, display_name):
    """Map a translated display name back to its English value."""
    for en, tk in key_map:
        if t(tk) == display_name:
            return en
    return display_name  # fallback: return as-is (handles English already)


def _display_from_en(key_map, en_name):
    """Map an English value to its translated display name."""
    for en, tk in key_map:
        if en == en_name:
            return t(tk)
    return en_name  # fallback

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

class KoKoFishUI:
    """Builds and manages the entire KoKoFish user interface."""

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

        # TTS generation streaming stream (real-time chunk playback during generation)
        self._tts_gen_stream = None      # sounddevice OutputStream for live chunk playback
        self._tts_gen_queue  = None      # the sample queue feeding that stream
        self._tts_gen_active = False     # True while a generation stream is running

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
            text="🐟  KoKoFish",
            font=(FONT_FAMILY, 28, "bold"),
            text_color=COLORS["accent_light"],
        )
        title_label.pack(side="left", pady=10)

        self._subtitle_label = ctk.CTkLabel(
            header,
            text=t("LAUNCHER_SUBTITLE"),
            font=(FONT_FAMILY, 13),
            text_color=COLORS["text_secondary"],
        )
        self._subtitle_label.pack(side="left", padx=(12, 0), pady=(18, 10))

        # Settings button — top right
        self._settings_window = None
        self._settings_btn = ctk.CTkButton(
            header,
            text=t("SETTINGS_TAB_LABEL"),
            width=110,
            height=32,
            font=(FONT_FAMILY, 12),
            fg_color=COLORS["bg_input"],
            hover_color=COLORS["bg_card_hover"],
            text_color=COLORS["text_primary"],
            corner_radius=8,
            command=self._open_settings_window,
        )
        self._settings_btn.pack(side="right", pady=10)

        # Language selector — to the left of Settings button
        from lang import get_languages, get_current_language
        _langs = get_languages()
        _lang_names = [name for _, name in _langs]
        _current_code = get_current_language()
        _current_name = next((name for code, name in _langs if code == _current_code), _lang_names[0] if _lang_names else "English")
        self._lang_var = ctk.StringVar(value=_current_name)
        self._lang_menu = ctk.CTkOptionMenu(
            header,
            variable=self._lang_var,
            values=_lang_names,
            width=145,
            height=26,
            font=(FONT_FAMILY, 10),
            fg_color=COLORS["bg_input"],
            button_color=COLORS["bg_card_hover"],
            button_hover_color=COLORS["accent"],
            text_color=COLORS["text_primary"],
            dropdown_fg_color=COLORS["bg_card"],
            dropdown_hover_color=COLORS["bg_card_hover"],
            dropdown_text_color=COLORS["text_primary"],
            corner_radius=8,
            command=self._on_language_changed,
        )
        self._lang_menu.pack(side="right", padx=(0, 8), pady=10)
        ctk.CTkLabel(header, text="Language:", font=(FONT_FAMILY, 10),
                     text_color=COLORS["text_muted"]).pack(side="right", padx=(0, 2), pady=10)

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
        self.tab_tts      = self.tabview.add(t("MAIN_TAB_SPEECH_LAB"))
        self.tab_voices   = self.tabview.add(t("MAIN_TAB_VOICE_LAB"))
        self.tab_stt      = self.tabview.add(t("MAIN_TAB_TEXT_LAB"))
        self.tab_convert  = self.tabview.add(t("MAIN_TAB_FILE_LAB"))
        self.tab_listen   = self.tabview.add(t("MAIN_TAB_LISTEN_LAB"))
        self.tab_script   = self.tabview.add(t("MAIN_TAB_SCRIPT_LAB"))
        self.tab_chat     = self.tabview.add(t("MAIN_TAB_PROMPT_LAB"))

        # Listen Lab state
        self._listen_items: list = []          # {path, name, selected}
        self._listen_preview_idx: int = -1
        self._listen_preview_paused: bool = False
        self._listen_preview_stream = None
        self._listen_preview_audio = None
        self._listen_vol_var = tk.IntVar(value=100)
        self._listen_drag_source: int = -1
        self._listen_drag_target: int = -1
        self._listen_drag_rows:   list = []
        self._listen_preview_pos: int = 0
        self._listen_preview_sr = None
        self._listen_ffmpeg_proc = None           # subprocess when streaming via ffmpeg
        self._listen_ffmpeg_path: str = ""        # path of file being streamed via ffmpeg
        self._listen_sf_handle = None             # SoundFile handle when streaming natively
        self._listen_is_ffmpeg: bool = False      # True = ffmpeg pipe, False = soundfile
        self._listen_preview_pos_sec: float = 0.0 # position in seconds (for ffmpeg seeks)
        self._listen_run_plain_on_done = None     # callback fired when _listen_run_plain finishes
        self._listen_stream_gen: int = 0  # incremented each stream start; finished callbacks check this to avoid stale resets
        self._listen_speed_var = tk.DoubleVar(value=1.0)  # playback speed multiplier

        self._build_tts_tab()
        self._build_voice_lab_tab()
        self._build_stt_tab()
        self._build_convert_tab()
        self._build_listen_lab_tab()
        self._build_script_lab_tab()
        self._build_prompt_lab_tab()

        # Lock Voice Lab tab when Kokoro engine is active
        if getattr(self.settings, 'engine', 'kokoro') == 'kokoro':
            self._lock_voice_lab()

        # Bind tab change for memory saver
        self.tabview.configure(command=self._on_tab_changed)

    # ==================================================================
    # Language switching
    # ==================================================================

    def _on_language_changed(self, lang_name: str):
        """Called when the user picks a new language from the dropdown."""
        from lang import get_languages, save_language_pref, load_language
        langs = get_languages()
        code = next((c for c, n in langs if n == lang_name), "en")
        save_language_pref(code)
        load_language(code)
        self._rebuild_all_tabs()

    def _rebuild_all_tabs(self):
        """Destroy and recreate the entire tabview so tab names translate too."""
        # ── Remember which tab was active so we can restore it ──────────────
        # Tab order is fixed: 0=Speech, 1=Voice, 2=Text, 3=File, 4=Listen, 5=Script, 6=Prompt
        _TAB_KEYS = [
            "MAIN_TAB_SPEECH_LAB", "MAIN_TAB_VOICE_LAB", "MAIN_TAB_TEXT_LAB",
            "MAIN_TAB_FILE_LAB",   "MAIN_TAB_LISTEN_LAB", "MAIN_TAB_SCRIPT_LAB",
            "MAIN_TAB_PROMPT_LAB",
        ]
        _saved_tab_idx = 0
        try:
            current_name = self.tabview.get()
            # Map current (old-language) tab name to its index by position
            for i, key in enumerate(_TAB_KEYS):
                if self.tabview.tab(t(key)):
                    pass  # will raise if not found — we rely on index order instead
            # Simpler: find index by iterating the actual tab list
            _all_tabs = self.tabview._tab_dict  # internal CTkTabview dict
            for i, name in enumerate(_all_tabs):
                if name == current_name:
                    _saved_tab_idx = i
                    break
        except Exception:
            _saved_tab_idx = 0

        # Update the Settings button label while we're here
        if hasattr(self, "_settings_btn"):
            self._settings_btn.configure(text=t("SETTINGS_TAB_LABEL"))
        if hasattr(self, "_subtitle_label"):
            self._subtitle_label.configure(text=t("LAUNCHER_SUBTITLE"))

        # ── Destroy and rebuild ──────────────────────────────────────────────
        self.tabview.destroy()

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

        # Recreate tabs with translated names
        self.tab_tts      = self.tabview.add(t("MAIN_TAB_SPEECH_LAB"))
        self.tab_voices   = self.tabview.add(t("MAIN_TAB_VOICE_LAB"))
        self.tab_stt      = self.tabview.add(t("MAIN_TAB_TEXT_LAB"))
        self.tab_convert  = self.tabview.add(t("MAIN_TAB_FILE_LAB"))
        self.tab_listen   = self.tabview.add(t("MAIN_TAB_LISTEN_LAB"))
        self.tab_script   = self.tabview.add(t("MAIN_TAB_SCRIPT_LAB"))
        self.tab_chat     = self.tabview.add(t("MAIN_TAB_PROMPT_LAB"))

        # Rebuild each tab's content
        self._build_tts_tab()
        self._build_voice_lab_tab()
        self._build_stt_tab()
        self._build_convert_tab()
        self._build_listen_lab_tab()
        self._build_script_lab_tab()
        self._build_prompt_lab_tab()

        # Re-bind tab change for memory saver
        self.tabview.configure(command=self._on_tab_changed)

        # ── Restore the previously active tab ───────────────────────────────
        try:
            target_tab_name = t(_TAB_KEYS[_saved_tab_idx])
            self.tabview.set(target_tab_name)
        except Exception:
            pass

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
            text=t("SPEECH_LAB_DROP_ZONE"),
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
        is_kokoro = getattr(self.settings, 'engine', 'kokoro') == 'kokoro'
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
            text=t("SPEECH_LAB_SPEED_LABEL", value=f"{self.settings.speed:.1f}"),
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
        self._make_tooltip(self.speed_label,  t("SPEECH_LAB_TOOLTIP_SPEED"))
        self._make_tooltip(self.speed_slider, t("SPEECH_LAB_TOOLTIP_SPEED"))

        # Volume slider
        vol_frame = ctk.CTkFrame(controls, fg_color="transparent")
        vol_frame.pack(side="left", padx=15)

        self.vol_label = ctk.CTkLabel(
            vol_frame,
            text=t("SPEECH_LAB_VOLUME_LABEL", value=self.settings.volume),
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
        self._make_tooltip(self.vol_label,  t("SPEECH_LAB_TOOLTIP_VOLUME"))
        self._make_tooltip(self.vol_slider, t("SPEECH_LAB_TOOLTIP_VOLUME"))

        # Cadence slider
        cad_frame = ctk.CTkFrame(controls, fg_color="transparent")
        cad_frame.pack(side="left", padx=15)

        self.cad_label = ctk.CTkLabel(
            cad_frame,
            text=t("SPEECH_LAB_CADENCE_LABEL", value=self.settings.cadence),
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
        self._make_tooltip(self.cad_label,  t("SPEECH_LAB_TOOLTIP_CADENCE"))
        self._make_tooltip(self.cad_slider, t("SPEECH_LAB_TOOLTIP_CADENCE"))

        # ── Content Style dropdown ────────────────────────────────────────
        style_frame = ctk.CTkFrame(controls, fg_color="transparent")
        style_frame.pack(side="left", padx=15)
        ctk.CTkLabel(
            style_frame,
            text=t("SPEECH_LAB_CONTENT_STYLE_LABEL"),
            font=(FONT_FAMILY, 11),
            text_color=COLORS["text_secondary"],
        ).pack(anchor="w")
        _CONTENT_STYLES_DISPLAY = _display_names(_CONTENT_STYLE_KEY_MAP)
        saved_style = getattr(self.settings, "content_style", "None")
        # saved_style is stored as English; convert to display name
        saved_style_display = _display_from_en(_CONTENT_STYLE_KEY_MAP, saved_style)
        self._content_style_var = ctk.StringVar(value=saved_style_display)
        def _on_style_change(v):
            self.settings.content_style = _en_from_display(_CONTENT_STYLE_KEY_MAP, v)
        ctk.CTkOptionMenu(
            style_frame,
            variable=self._content_style_var,
            values=_CONTENT_STYLES_DISPLAY,
            width=160,
            height=26,
            fg_color=COLORS["bg_input"],
            button_color="#9b59b6",
            button_hover_color="#8e44ad",
            dropdown_fg_color=COLORS["bg_card"],
            dropdown_hover_color=COLORS["bg_card_hover"],
            font=(FONT_FAMILY, 11),
            command=_on_style_change,
        ).pack(anchor="w")
        self._make_tooltip(style_frame, t("SPEECH_LAB_CONTENT_STYLE_TOOLTIP"))

        # ── Playlist header row ──────────────────────────────────────────
        playlist_header = ctk.CTkFrame(tab, fg_color="transparent")
        playlist_header.pack(fill="x", padx=15, pady=(10, 2))

        ctk.CTkLabel(
            playlist_header,
            text=t("SPEECH_LAB_HEADER_PLAYLIST"),
            font=(FONT_FAMILY, 14, "bold"),
            text_color=COLORS["text_primary"],
        ).pack(side="left")

        ctk.CTkLabel(
        playlist_header,
        text=t("SPEECH_LAB_HINT_DBLCLICK"),
        font=(FONT_FAMILY, 10),
        text_color=COLORS["text_secondary"],
        ).pack(side="left", padx=(5, 0))

        # ── Kokoro language filter dropdown ───────────────────────────────
        if is_kokoro:
            lang_frame = ctk.CTkFrame(playlist_header, fg_color="transparent")
            lang_frame.pack(side="left", padx=(16, 0))
            ctk.CTkLabel(
                lang_frame,
                text=t("SPEECH_LAB_LANG_FILTER_LABEL"),
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
            text=t("SPEECH_LAB_BTN_CONVERT"),
            fg_color=COLORS["success"],
            hover_color="#05b890",
            font=(FONT_FAMILY, 12, "bold"),
            corner_radius=7, height=32, width=100,
            command=self._tts_play,
        )
        self.btn_play.pack(side="right", padx=(4, 0))
        self._make_tooltip(self.btn_play, t("SPEECH_LAB_TOOLTIP_PLAY_ALL"))

        # Work Silent + Auto Save toggles
        self.silent_mode_var = ctk.BooleanVar(value=getattr(self.settings, 'silent_mode', False))
        silent_frame = ctk.CTkFrame(playlist_header, fg_color="transparent")
        silent_frame.pack(side="right", padx=(0, 16))
        ctk.CTkLabel(
            silent_frame, text=t("SPEECH_LAB_BTN_SILENT_MODE"), font=(FONT_FAMILY, 13),
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
            auto_save_frame, text=t("SPEECH_LAB_BTN_AUTO_SAVE"), font=(FONT_FAMILY, 13),
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
            text=t("SPEECH_LAB_PLAYLIST_EMPTY"),
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
            text=t("SPEECH_LAB_STATUS_READY"),
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

        # TTS Selected + its Pause / Stop (pause/stop hidden until converting)
        _btn_convert = ctk.CTkButton(
            sel_bar, text=t("SPEECH_LAB_BTN_CONVERT_SELECTED"),
            fg_color=COLORS["success"], hover_color="#05b890",
            command=self._tts_selected, width=140, **_big,
        )
        _btn_convert.pack(side="left", padx=(0, 2))
        self._make_tooltip(_btn_convert, t("SPEECH_LAB_TOOLTIP_CONVERT_SELECTED"))

        self.btn_pause = ctk.CTkButton(
            sel_bar, text="⏸",
            fg_color=COLORS["warning"], hover_color="#e6bc5c",
            text_color="#1a1a2e",
            command=self._tts_pause, **_mini,
        )
        self._make_tooltip(self.btn_pause, t("SPEECH_LAB_TOOLTIP_PAUSE_CONV"))
        # Not packed yet — shown when conversion starts

        self.btn_stop = ctk.CTkButton(
            sel_bar, text="⏹",
            fg_color=COLORS["danger"], hover_color="#d43d62",
            command=self._tts_stop, **_mini,
        )
        self._make_tooltip(self.btn_stop, t("SPEECH_LAB_TOOLTIP_STOP_CONV"))
        # Not packed yet — shown when conversion starts

        # Play Selected + its Pause / Stop (pause/stop hidden until playing)
        _btn_play_sel = ctk.CTkButton(
            sel_bar, text=t("SPEECH_LAB_BTN_PLAY_SELECTED"),
            fg_color=COLORS["accent"], hover_color=COLORS["accent_hover"],
            command=self._play_selected, width=140, **_big,
        )
        _btn_play_sel.pack(side="left", padx=(0, 2))
        self._make_tooltip(_btn_play_sel, t("SPEECH_LAB_TOOLTIP_PLAY_SELECTED"))

        self.btn_play_pause = ctk.CTkButton(
            sel_bar, text="⏸",
            fg_color=COLORS["warning"], hover_color="#e6bc5c",
            text_color="#1a1a2e",
            command=self._stop_preview, **_mini,
        )
        self._make_tooltip(self.btn_play_pause, t("SPEECH_LAB_TOOLTIP_PAUSE_PB"))
        # Not packed yet

        self.btn_play_stop = ctk.CTkButton(
            sel_bar, text="⏹",
            fg_color=COLORS["danger"], hover_color="#d43d62",
            command=self._stop_preview, **_mini,
        )
        self._make_tooltip(self.btn_play_stop, t("SPEECH_LAB_TOOLTIP_STOP_PB"))
        # Not packed yet

        # Store sel_bar reference so we can re-pack buttons with correct order
        self._sel_bar = sel_bar
        self._btn_convert = _btn_convert
        self._btn_play_sel = _btn_play_sel

        # Save Selected
        self.btn_save_mp3 = ctk.CTkButton(
            sel_bar, text=t("SPEECH_LAB_BTN_SAVE_SELECTED"),
            fg_color=COLORS["accent"], hover_color=COLORS["accent_hover"],
            command=self._tts_save_mp3, width=140, **_big,
        )
        self.btn_save_mp3.pack(side="left", padx=(0, 4))
        self._make_tooltip(self.btn_save_mp3, t("SPEECH_LAB_TOOLTIP_EXPORT_MP3"))

        # Export Audiobook
        _btn_ab = ctk.CTkButton(
            sel_bar, text=t("SPEECH_LAB_BTN_AUDIOBOOK"),
            fg_color="#2d6a4f", hover_color="#1b4332",
            command=self._tts_export_audiobook, width=120, **_big,
        )
        _btn_ab.pack(side="left", padx=(0, 16))
        self._make_tooltip(_btn_ab, t("SPEECH_LAB_TOOLTIP_AUDIOBOOK"))

        # Selection helpers
        _btn = ctk.CTkButton(
            sel_bar, text=t("SPEECH_LAB_BTN_SELECT_ALL"),
            fg_color=COLORS["bg_input"], hover_color=COLORS["bg_card_hover"],
            border_color=COLORS["border"], border_width=1,
            command=self._select_all, width=60, **_util,
        )
        _btn.pack(side="left", padx=(0, 4))
        self._make_tooltip(_btn, t("SPEECH_LAB_TOOLTIP_SELECT_ALL"))

        _btn = ctk.CTkButton(
            sel_bar, text=t("SPEECH_LAB_BTN_SELECT_NONE"),
            fg_color=COLORS["bg_input"], hover_color=COLORS["bg_card_hover"],
            border_color=COLORS["border"], border_width=1,
            command=self._deselect_all, width=60, **_util,
        )
        _btn.pack(side="left", padx=(0, 4))
        self._make_tooltip(_btn, t("SPEECH_LAB_TOOLTIP_DESELECT_ALL"))

        # Clear Selected — far right
        _btn = ctk.CTkButton(
            sel_bar, text=t("SPEECH_LAB_BTN_CLEAR_SELECTED"),
            fg_color=COLORS["bg_input"], hover_color=COLORS["bg_card_hover"],
            border_color=COLORS["border"], border_width=1,
            command=self._tts_clear_playlist, width=120, **_util,
        )
        _btn.pack(side="right")
        self._make_tooltip(_btn, t("SPEECH_LAB_TOOLTIP_REMOVE_SEL"))

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
            text=t("TEXT_LAB_DROP_ZONE"),
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
            text=t("TEXT_LAB_MODEL_LABEL"),
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
            text=t("TEXT_LAB_BTN_TRANSCRIBE"),
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
            text=t("TEXT_LAB_BTN_CANCEL"),
            fg_color=COLORS["danger"],
            hover_color="#d43d62",
            font=(FONT_FAMILY, 13, "bold"),
            height=38,
            width=100,
            command=self._stt_cancel,
        )
        self.btn_stt_cancel.pack(side="left")

        # Timestamp toggle
        _ts_frame = ctk.CTkFrame(stt_controls, fg_color="transparent")
        _ts_frame.pack(side="left", padx=(16, 0))
        self._stt_timestamps_var = ctk.BooleanVar(value=True)
        ctk.CTkLabel(
            _ts_frame,
            text=t("TEXT_LAB_TIMESTAMPS_TOGGLE"),
            font=(FONT_FAMILY, 11),
            text_color=COLORS["text_secondary"],
        ).pack(anchor="w")
        ctk.CTkSwitch(
            _ts_frame,
            text="",
            variable=self._stt_timestamps_var,
            width=44,
            height=22,
            switch_width=40,
            switch_height=18,
            progress_color=COLORS["accent"],
            button_color=COLORS["accent_light"],
            button_hover_color=COLORS["accent"],
        ).pack()

        # Translate controls (next to timestamps)
        _tr_frame = ctk.CTkFrame(stt_controls, fg_color="transparent")
        _tr_frame.pack(side="left", padx=(20, 0))
        ctk.CTkLabel(
            _tr_frame,
            text=t("TEXT_LAB_TRANSLATE_TO_LABEL"),
            font=(FONT_FAMILY, 11),
            text_color=COLORS["text_secondary"],
        ).pack(anchor="w")
        _tr_inner = ctk.CTkFrame(_tr_frame, fg_color="transparent")
        _tr_inner.pack()
        _stt_lang_display = _display_names(_LANG_KEY_MAP)
        self._stt_translate_lang_var = ctk.StringVar(value=_stt_lang_display[0])
        ctk.CTkOptionMenu(
            _tr_inner,
            variable=self._stt_translate_lang_var,
            values=_stt_lang_display,
            width=130, height=26,
            fg_color=COLORS["bg_input"],
            button_color="#e76f51",
            button_hover_color="#f4a261",
            dropdown_fg_color=COLORS["bg_card"],
            dropdown_hover_color=COLORS["bg_card_hover"],
            font=(FONT_FAMILY, 11),
        ).pack(side="left", padx=(0, 4))
        self._stt_translate_btn = ctk.CTkButton(
            _tr_inner,
            text=t("TEXT_LAB_BTN_TRANSLATE"),
            fg_color="#e76f51",
            hover_color="#f4a261",
            font=(FONT_FAMILY, 11, "bold"),
            height=26, width=110,
            corner_radius=6,
            command=self._stt_translate,
        )
        self._stt_translate_btn.pack(side="left")

        # File info label
        self.stt_file_label = ctk.CTkLabel(
            stt_controls,
            text=t("TEXT_LAB_FILE_LABEL_NONE"),
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
        self.stt_textbox.insert("1.0", t("TEXT_LAB_TEXTBOX_PLACEHOLDER"))
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
            export_frame, text=t("TEXT_LAB_BTN_EXPORT_TXT"),
            command=lambda: self._stt_export("txt"), **exp_style,
        ).pack(side="left", padx=(0, 8))

        ctk.CTkButton(
            export_frame, text=t("TEXT_LAB_BTN_EXPORT_DOCX"),
            command=lambda: self._stt_export("docx"), **exp_style,
        ).pack(side="left", padx=(0, 8))

        ctk.CTkButton(
            export_frame, text=t("TEXT_LAB_BTN_EXPORT_PDF"),
            command=lambda: self._stt_export("pdf"), **exp_style,
        ).pack(side="left", padx=(0, 8))

        ctk.CTkButton(
            export_frame, text=t("TEXT_LAB_BTN_EXPORT_EPUB"),
            command=lambda: self._stt_export("epub"), **exp_style,
        ).pack(side="left", padx=(0, 8))

        ctk.CTkButton(
            export_frame,
            text=t("TEXT_LAB_BTN_SEND_SPEECH"),
            command=self._stt_send_to_speech_lab,
            fg_color=COLORS["success"],
            hover_color="#05b886",
            text_color="#0a0a18",
            font=(FONT_FAMILY, 12, "bold"),
            corner_radius=8,
            height=34,
            width=185,
        ).pack(side="right")

        # Store current audio path for transcription
        self._stt_audio_path = None

    # ==================================================================
    # TAB 5: Convert
    # ==================================================================

    def _build_convert_tab(self):
        tab = self.tab_convert

        scroll = ctk.CTkScrollableFrame(tab, fg_color="transparent")
        scroll.pack(fill="both", expand=True, padx=10, pady=10)

        # ── shared card style ──────────────────────────────────────────
        def _card(parent, title, icon):
            f = ctk.CTkFrame(parent, fg_color=COLORS["bg_card"], corner_radius=12)
            f.pack(fill="x", pady=(0, 14))
            ctk.CTkLabel(
                f, text=f"{icon}  {title}",
                font=(FONT_FAMILY, 13, "bold"),
                text_color=COLORS["text_primary"],
            ).pack(anchor="w", padx=14, pady=(12, 6))
            return f

        # ── TEXT CONVERSION ────────────────────────────────────────────
        txt_card = _card(scroll, t("FILE_LAB_TEXT_CONVERSION_HEADER"), "📄")

        _TEXT_FORMATS = ["TXT", "PDF", "DOCX", "EPUB"]
        _TEXT_EXT = {
            "TXT":  (".txt",  [("Text file", "*.txt")]),
            "PDF":  (".pdf",  [("PDF document", "*.pdf")]),
            "DOCX": (".docx", [("Word document", "*.docx")]),
            "EPUB": (".epub", [("EPUB ebook", "*.epub")]),
        }
        _TEXT_READ_EXT = "*.txt *.pdf *.docx *.epub"

        self._conv_text_in_path = None
        self._conv_text_src_lbl = None

        def _txt_browse(event=None):
            p = filedialog.askopenfilename(
                title=t("FILE_LAB_DIALOG_OPEN_DOCUMENT"),
                filetypes=[
                    ("Supported documents", _TEXT_READ_EXT),
                    ("All files", "*.*"),
                ],
                parent=self.root,
            )
            if p:
                _txt_set_file(p)

        def _txt_set_file(p):
            self._conv_text_in_path = p
            name = os.path.basename(p)
            ext  = os.path.splitext(p)[1].upper().lstrip(".")
            self._conv_text_src_lbl.configure(text=f"📎  {name}", text_color=COLORS["success"])
            # Set "from" label and filter "to" so it excludes the source format
            fmt = ext if ext in _TEXT_FORMATS else "?"
            _txt_from_var.set(fmt if fmt != "?" else _TEXT_FORMATS[0])
            _txt_update_to_options(fmt)

        def _txt_update_to_options(src_fmt):
            opts = [f for f in _TEXT_FORMATS if f != src_fmt]
            _txt_to_menu.configure(values=opts)
            if _txt_to_var.get() == src_fmt or _txt_to_var.get() not in opts:
                _txt_to_var.set(opts[0])

        def _txt_convert():
            src = self._conv_text_in_path
            if not src or not os.path.isfile(src):
                messagebox.showinfo(t("FILE_LAB_CONVERT_TITLE"), t("FILE_LAB_MSG_NO_INPUT"), parent=self.root)
                return
            to_fmt = _txt_to_var.get()
            ext, ftypes = _TEXT_EXT[to_fmt]
            stem = os.path.splitext(os.path.basename(src))[0]
            out = filedialog.asksaveasfilename(
                title=t("FILE_LAB_DIALOG_SAVE_AS", fmt=to_fmt),
                defaultextension=ext,
                filetypes=ftypes,
                initialfile=f"{stem}{ext}",
                parent=self.root,
            )
            if not out:
                return

            _txt_status.configure(text=t("FILE_LAB_STATUS_CONVERTING"), text_color=COLORS["warning"])
            _txt_btn.configure(state="disabled")

            def _run():
                try:
                    from utils import read_file, export_txt, export_docx, export_pdf, export_epub
                    text = read_file(src)
                    title = stem
                    if to_fmt == "TXT":
                        export_txt(text, out)
                    elif to_fmt == "DOCX":
                        export_docx(text, out)
                    elif to_fmt == "PDF":
                        export_pdf(text, out)
                    elif to_fmt == "EPUB":
                        export_epub(text, out, title=title)
                    self.root.after(0, lambda: _txt_status.configure(
                        text=t("FILE_LAB_STATUS_SAVED", fmt=to_fmt, filename=os.path.basename(out)),
                        text_color=COLORS["success"],
                    ))
                except Exception as exc:
                    logger.error("Text convert failed: %s", exc)
                    self.root.after(0, lambda e=str(exc): _txt_status.configure(
                        text=t("FILE_LAB_STATUS_ERROR", error=e), text_color=COLORS["danger"]
                    ))
                finally:
                    self.root.after(0, lambda: _txt_btn.configure(state="normal"))

            threading.Thread(target=_run, daemon=True, name="TextConvert").start()

        # Drop zone
        _txt_drop = ctk.CTkFrame(
            txt_card, fg_color=COLORS["bg_input"],
            border_color=COLORS["border"], border_width=2, corner_radius=8, height=60,
        )
        _txt_drop.pack(fill="x", padx=14, pady=(0, 8))
        _txt_drop.pack_propagate(False)
        self._conv_text_src_lbl = ctk.CTkLabel(
            _txt_drop,
            text=t("FILE_LAB_DROP_ZONE"),
            font=(FONT_FAMILY, 11), text_color=COLORS["text_secondary"],
        )
        self._conv_text_src_lbl.pack(expand=True)
        self._conv_text_src_lbl.bind("<Button-1>", _txt_browse)
        _txt_drop.bind("<Button-1>", _txt_browse)
        try:
            from tkinterdnd2 import DND_FILES
            _txt_drop.drop_target_register(DND_FILES)
            _txt_drop.dnd_bind("<<Drop>>", lambda e: _txt_set_file(self._parse_drop_data(e.data)[0]) if self._parse_drop_data(e.data) else None)
        except Exception:
            pass

        # From / To row
        _fmt_row = ctk.CTkFrame(txt_card, fg_color="transparent")
        _fmt_row.pack(fill="x", padx=14, pady=(0, 8))

        _txt_from_var = ctk.StringVar(value="TXT")
        ctk.CTkLabel(_fmt_row, text=t("TEXT_LAB_FROM_LABEL"), font=(FONT_FAMILY, 11),
                     text_color=COLORS["text_secondary"]).pack(side="left", padx=(0, 4))
        ctk.CTkLabel(_fmt_row, textvariable=_txt_from_var,
                     font=(FONT_FAMILY, 11, "bold"),
                     text_color=COLORS["accent_light"]).pack(side="left", padx=(0, 12))

        ctk.CTkLabel(_fmt_row, text=t("TEXT_LAB_TO_LABEL"), font=(FONT_FAMILY, 11),
                     text_color=COLORS["text_secondary"]).pack(side="left", padx=(0, 4))
        _txt_to_var = ctk.StringVar(value="PDF")
        _txt_to_menu = ctk.CTkOptionMenu(
            _fmt_row, variable=_txt_to_var,
            values=[f for f in _TEXT_FORMATS if f != "TXT"],
            width=100, height=28,
            fg_color=COLORS["bg_input"],
            button_color=COLORS["accent"], button_hover_color=COLORS["accent_hover"],
            dropdown_fg_color=COLORS["bg_card"], dropdown_hover_color=COLORS["bg_card_hover"],
            font=(FONT_FAMILY, 11),
        )
        _txt_to_menu.pack(side="left", padx=(0, 12))

        _txt_btn = ctk.CTkButton(
            _fmt_row, text=t("TEXT_LAB_BTN_CONVERT_FMT"),
            fg_color=COLORS["accent"], hover_color=COLORS["accent_hover"],
            font=(FONT_FAMILY, 11, "bold"), height=28, width=90,
            command=_txt_convert,
        )
        _txt_btn.pack(side="left")

        _txt_status = ctk.CTkLabel(
            txt_card, text="",
            font=(FONT_FAMILY, 10), text_color=COLORS["text_secondary"],
        )
        _txt_status.pack(anchor="w", padx=14, pady=(0, 10))

        # ── AUDIO CONVERSION ───────────────────────────────────────────
        aud_card = _card(scroll, t("FILE_LAB_AUDIO_HEADER"), "🎵")

        _AUDIO_FORMATS = {
            "MP3":  (".mp3",  [("MP3 audio",          "*.mp3")],  "libmp3lame", "192k"),
            "WAV":  (".wav",  [("WAV audio",           "*.wav")],  "pcm_s16le",  None),
            "M4B":  (".m4b",  [("Audiobook M4B",       "*.m4b")],  "aac",        "128k"),
            "MP4":  (".mp4",  [("MP4 audio",           "*.mp4")],  "aac",        "128k"),
            "FLAC": (".flac", [("FLAC lossless audio", "*.flac")], "flac",       None),
        }
        _AUDIO_READ_EXT = "*.mp3 *.wav *.m4b *.m4a *.mp4 *.flac *.ogg *.aac *.weba *.webm *.opus *.wma *.amr"
        _AUDIO_NOTE = {
            "M4B":  t("FILE_LAB_AUDIO_NOTE_CHAPTERS"),
            "MP4":  t("FILE_LAB_AUDIO_NOTE_CHAPTERS"),
            "MP3":  t("FILE_LAB_AUDIO_NOTE_MP3"),
            "WAV":  t("FILE_LAB_AUDIO_NOTE_WAV"),
            "FLAC": t("FILE_LAB_AUDIO_NOTE_FLAC"),
        }

        self._conv_audio_in_path = None
        self._conv_audio_src_lbl = None

        def _aud_browse(event=None):
            p = filedialog.askopenfilename(
                title="Open audio file…",
                filetypes=[
                    ("Audio files", _AUDIO_READ_EXT),
                    ("All files", "*.*"),
                ],
                parent=self.root,
            )
            if p:
                _aud_set_file(p)

        def _aud_set_file(p):
            self._conv_audio_in_path = p
            name = os.path.basename(p)
            self._conv_audio_src_lbl.configure(text=f"🎵  {name}", text_color=COLORS["success"])
            src_ext = os.path.splitext(p)[1].upper().lstrip(".")
            opts = list(_AUDIO_FORMATS.keys())
            # Filter out the exact same format
            if src_ext in opts:
                opts = [f for f in opts if f != src_ext]
            _aud_to_menu.configure(values=opts)
            if _aud_to_var.get() not in opts:
                _aud_to_var.set(opts[0])

        def _aud_convert():
            src = self._conv_audio_in_path
            if not src or not os.path.isfile(src):
                messagebox.showinfo(t("FILE_LAB_AUDIO_HEADER"), t("FILE_LAB_MSG_NO_INPUT_AUDIO"), parent=self.root)
                return
            if not is_ffmpeg_available():
                messagebox.showerror(t("FILE_LAB_AUDIO_HEADER"), t("FILE_LAB_MSG_NO_FFMPEG_CONV"), parent=self.root)
                return

            to_fmt = _aud_to_var.get()
            ext, ftypes, acodec, bitrate = _AUDIO_FORMATS[to_fmt]
            stem = os.path.splitext(os.path.basename(src))[0]
            out = filedialog.asksaveasfilename(
                title=f"Save as {to_fmt}…",
                defaultextension=ext,
                filetypes=ftypes,
                initialfile=f"{stem}{ext}",
                parent=self.root,
            )
            if not out:
                return

            _aud_status.configure(text=t("FILE_LAB_STATUS_CONVERTING_AUDIO"), text_color=COLORS["warning"])
            _aud_progress.set(0)
            _aud_progress.pack(anchor="w", fill="x", padx=14, pady=(0, 2))
            _aud_btn.configure(state="disabled")

            _aud_cancel = threading.Event()

            def _tick(start, _id=[None]):
                if _aud_cancel.is_set():
                    return
                elapsed = time.time() - start
                _aud_status.configure(
                    text=f"Converting…  {int(elapsed // 60)}:{int(elapsed % 60):02d} elapsed",
                    text_color=COLORS["warning"],
                )
                _aud_progress.set(min(elapsed / 60, 0.95))  # crawls toward 95% (no true duration)
                _id[0] = self.root.after(500, lambda: _tick(start, _id))

            def _run():
                import subprocess
                start = time.time()
                self.root.after(0, lambda: _tick(start))
                try:
                    cmd = ["ffmpeg", "-y", "-i", src, "-c:a", acodec]
                    if bitrate:
                        cmd += ["-b:a", bitrate]
                    if to_fmt in ("M4B", "MP4"):
                        cmd += ["-map_metadata", "0", "-movflags", "+faststart"]
                    elif to_fmt == "MP3":
                        cmd += ["-map_metadata", "0", "-id3v2_version", "3"]
                    cmd.append(out)
                    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, creationflags=_NO_WIN)
                    elapsed = time.time() - start
                    size_mb = os.path.getsize(out) / (1024 * 1024)
                    _aud_cancel.set()
                    self.root.after(0, lambda: _aud_progress.set(1.0))
                    self.root.after(0, lambda: _aud_status.configure(
                        text=f"✅ {os.path.basename(out)} ({size_mb:.1f} MB) — {elapsed:.0f}s",
                        text_color=COLORS["success"],
                    ))
                    self.root.after(1500, lambda: _aud_progress.pack_forget())
                except Exception as exc:
                    _aud_cancel.set()
                    logger.error("Audio convert failed: %s", exc)
                    self.root.after(0, lambda: _aud_progress.pack_forget())
                    self.root.after(0, lambda e=str(exc): _aud_status.configure(
                        text=f"⚠ {e}", text_color=COLORS["danger"],
                    ))
                finally:
                    self.root.after(0, lambda: _aud_btn.configure(state="normal"))

            threading.Thread(target=_run, daemon=True, name="AudioConvert").start()

        def _aud_show_note(*_):
            fmt = _aud_to_var.get()
            note = _AUDIO_NOTE.get(fmt, "")
            _aud_note_lbl.configure(text=note)

        # Drop zone
        _aud_drop = ctk.CTkFrame(
            aud_card, fg_color=COLORS["bg_input"],
            border_color=COLORS["border"], border_width=2, corner_radius=8, height=60,
        )
        _aud_drop.pack(fill="x", padx=14, pady=(0, 8))
        _aud_drop.pack_propagate(False)
        self._conv_audio_src_lbl = ctk.CTkLabel(
            _aud_drop,
            text=t("FILE_LAB_AUDIO_DROP_HINT"),
            font=(FONT_FAMILY, 11), text_color=COLORS["text_secondary"],
        )
        self._conv_audio_src_lbl.pack(expand=True)
        self._conv_audio_src_lbl.bind("<Button-1>", _aud_browse)
        _aud_drop.bind("<Button-1>", _aud_browse)
        try:
            from tkinterdnd2 import DND_FILES
            _aud_drop.drop_target_register(DND_FILES)
            _aud_drop.dnd_bind("<<Drop>>", lambda e: _aud_set_file(self._parse_drop_data(e.data)[0]) if self._parse_drop_data(e.data) else None)
        except Exception:
            pass

        # To format row
        _aud_row = ctk.CTkFrame(aud_card, fg_color="transparent")
        _aud_row.pack(fill="x", padx=14, pady=(0, 4))

        ctk.CTkLabel(_aud_row, text=t("FILE_LAB_CONVERT_TO_LABEL"), font=(FONT_FAMILY, 11),
                     text_color=COLORS["text_secondary"]).pack(side="left", padx=(0, 8))
        _aud_to_var = ctk.StringVar(value="MP3")
        _aud_to_menu = ctk.CTkOptionMenu(
            _aud_row, variable=_aud_to_var,
            values=list(_AUDIO_FORMATS.keys()),
            width=100, height=28,
            fg_color=COLORS["bg_input"],
            button_color="#2d6a4f", button_hover_color="#1b4332",
            dropdown_fg_color=COLORS["bg_card"], dropdown_hover_color=COLORS["bg_card_hover"],
            font=(FONT_FAMILY, 11),
            command=_aud_show_note,
        )
        _aud_to_menu.pack(side="left", padx=(0, 12))

        _aud_btn = ctk.CTkButton(
            _aud_row, text=t("FILE_LAB_BTN_CONVERT_AUDIO"),
            fg_color="#2d6a4f", hover_color="#1b4332",
            font=(FONT_FAMILY, 11, "bold"), height=28, width=90,
            command=_aud_convert,
        )
        _aud_btn.pack(side="left")

        _aud_note_lbl = ctk.CTkLabel(
            aud_card, text=_AUDIO_NOTE.get("MP3", ""),
            font=(FONT_FAMILY, 9), text_color=COLORS["text_muted"],
        )
        _aud_note_lbl.pack(anchor="w", padx=14, pady=(0, 4))

        _aud_progress = ctk.CTkProgressBar(
            aud_card, progress_color=COLORS["accent"],
            fg_color=COLORS["bg_input"], height=8, corner_radius=4,
        )
        _aud_progress.set(0)
        # Not packed yet — shown dynamically during conversion

        _aud_status = ctk.CTkLabel(
            aud_card, text="",
            font=(FONT_FAMILY, 10), text_color=COLORS["text_secondary"],
        )
        _aud_status.pack(anchor="w", padx=14, pady=(0, 10))

        # ── COMBINE AUDIO FILES → AUDIOBOOK ────────────────────────────
        comb_card = _card(scroll, t("FILE_LAB_COMBINE_HEADER"), "📚")

        ctk.CTkLabel(
            comb_card,
            text=t("FILE_LAB_COMBINER_HINT"),
            font=(FONT_FAMILY, 10), text_color=COLORS["text_secondary"],
            justify="left",
        ).pack(anchor="w", padx=14, pady=(0, 6))

        # File list
        _comb_files: list = []  # list of dicts: {path, name}

        _list_frame = ctk.CTkScrollableFrame(
            comb_card, fg_color=COLORS["bg_input"],
            corner_radius=6, height=160,
        )
        _list_frame.pack(fill="x", padx=14, pady=(0, 6))

        _comb_progress = ctk.CTkProgressBar(
            comb_card, progress_color=COLORS["accent"],
            fg_color=COLORS["bg_input"], height=8, corner_radius=4,
        )
        _comb_progress.set(0)
        # Not packed yet — shown during combine

        _comb_status = ctk.CTkLabel(
            comb_card, text=t("FILE_LAB_COMBINER_EMPTY"),
            font=(FONT_FAMILY, 10), text_color=COLORS["text_muted"],
        )
        _comb_status.pack(anchor="w", padx=14, pady=(0, 4))

        # drag state for the combine list
        _comb_drag: dict = {"src": None, "rows": []}

        def _comb_rebuild():
            for w in _list_frame.winfo_children():
                w.destroy()
            _comb_drag["rows"].clear()

            if not _comb_files:
                ctk.CTkLabel(
                    _list_frame,
                    text=t("FILE_LAB_COMBINER_EMPTY_BTN"),
                    font=(FONT_FAMILY, 10),
                    text_color=COLORS["text_muted"],
                ).pack(pady=10)
                _comb_status.configure(text=t("FILE_LAB_COMBINER_EMPTY"))
                return

            total_s = 0.0
            for i, f in enumerate(_comb_files):
                row = ctk.CTkFrame(_list_frame, fg_color=COLORS["bg_card"], corner_radius=4, height=32)
                row.pack(fill="x", pady=1)
                row.pack_propagate(False)
                _comb_drag["rows"].append((i, row))

                # drag handle
                _h = ctk.CTkLabel(row, text="⠿", font=(FONT_FAMILY, 14),
                                   text_color=COLORS["text_muted"], width=18, cursor="fleur")
                _h.pack(side="left", padx=(4, 0))
                _h.bind("<ButtonPress-1>",  lambda e, si=i: _comb_drag_start(e, si))
                _h.bind("<B1-Motion>",       lambda e, si=i: _comb_drag_motion(e, si))
                _h.bind("<ButtonRelease-1>", lambda e, si=i: _comb_drag_end(e, si))

                ctk.CTkLabel(
                    row,
                    text=f"{i+1}. {f['name']}",
                    font=(FONT_FAMILY, 10),
                    text_color=COLORS["text_primary"],
                    anchor="w",
                ).pack(side="left", fill="x", expand=True, padx=6)

                ctk.CTkButton(
                    row, text="✕", width=24, height=22, corner_radius=4,
                    fg_color=COLORS["danger"], hover_color="#d43d62",
                    font=(FONT_FAMILY, 11),
                    command=lambda fi=i: (_comb_files.pop(fi), _comb_rebuild()),
                ).pack(side="right", padx=4)

            n = len(_comb_files)
            _comb_status.configure(text=f"{n} file{'s' if n != 1 else ''} ready to combine")

        def _comb_drag_start(event, i):
            _comb_drag["src"] = i
            event.widget.grab_set()

        def _comb_drag_motion(event, _i):
            if _comb_drag["src"] is None:
                return
            x = event.widget.winfo_rootx() + event.x
            y = event.widget.winfo_rooty() + event.y
            for row_i, row_w in _comb_drag["rows"]:
                try:
                    rx, ry = row_w.winfo_rootx(), row_w.winfo_rooty()
                    if rx <= x <= rx + row_w.winfo_width() and ry <= y <= ry + row_w.winfo_height():
                        _comb_drag["over"] = row_i
                        break
                except Exception:
                    pass

        def _comb_drag_end(event, _i):
            if _comb_drag["src"] is None:
                return
            try:
                event.widget.grab_release()
            except Exception:
                pass
            x = event.widget.winfo_rootx() + event.x
            y = event.widget.winfo_rooty() + event.y
            target = None
            for row_i, row_w in _comb_drag["rows"]:
                try:
                    rx, ry = row_w.winfo_rootx(), row_w.winfo_rooty()
                    if rx <= x <= rx + row_w.winfo_width() and ry <= y <= ry + row_w.winfo_height():
                        target = row_i
                        break
                except Exception:
                    pass
            src = _comb_drag["src"]
            _comb_drag["src"] = None
            if target is not None and target != src:
                item = _comb_files.pop(src)
                _comb_files.insert(target, item)
                _comb_rebuild()

        def _comb_add():
            paths = filedialog.askopenfilenames(
                title="Add audio files…",
                filetypes=[("Audio files", "*.mp3 *.wav *.m4b *.m4a *.mp4 *.flac *.ogg")],
                parent=self.root,
            )
            for p in paths:
                _comb_files.append({"path": p, "name": os.path.basename(p)})
            _comb_rebuild()

        def _comb_clear():
            _comb_files.clear()
            _comb_rebuild()

        def _comb_export():
            if not _comb_files:
                messagebox.showinfo(t("FILE_LAB_COMBINE_HEADER"), t("FILE_LAB_MSG_NO_FILES_COMBINE"), parent=self.root)
                return
            if not is_ffmpeg_available():
                messagebox.showerror(t("FILE_LAB_COMBINE_HEADER"), t("FILE_LAB_MSG_NO_FFMPEG_COMBINE"), parent=self.root)
                return
            out = filedialog.asksaveasfilename(
                title="Save combined audiobook as…",
                defaultextension=".m4b",
                filetypes=[
                    ("Audiobook M4B — chapters supported", "*.m4b"),
                    ("MP4 audio — chapters supported",     "*.mp4"),
                    ("WAV — lossless, no chapters",        "*.wav"),
                    ("MP3 — ID3v2 chapters",               "*.mp3"),
                ],
                initialfile="combined.m4b",
                parent=self.root,
            )
            if not out:
                return
            _comb_status.configure(text=t("FILE_LAB_STATUS_COMBINING"), text_color=COLORS["warning"])
            _comb_progress.set(0)
            _comb_progress.pack(anchor="w", fill="x", padx=14, pady=(0, 2))

            _n_files = len(_comb_files)
            _comb_cancel = threading.Event()

            def _comb_tick(start, _id=[None]):
                if _comb_cancel.is_set():
                    return
                elapsed = time.time() - start
                _id[0] = self.root.after(500, lambda: _comb_tick(start, _id))

            def _run():
                import subprocess, tempfile
                start = time.time()
                self.root.after(0, lambda: _comb_tick(start))
                tmp = tempfile.mkdtemp(prefix="kokofish_comb_")
                try:
                    wav_files, durations_ms = [], []
                    for fi, f in enumerate(_comb_files):
                        norm = os.path.join(tmp, f"t{fi:04d}.wav")
                        frac = fi / max(_n_files, 1) * 0.7
                        elapsed = time.time() - start
                        self.root.after(0, lambda fr=frac, fi2=fi, el=elapsed: (
                            _comb_progress.set(fr),
                            _comb_status.configure(
                                text=f"Normalising {fi2+1}/{_n_files}…  {el:.0f}s elapsed",
                                text_color=COLORS["warning"],
                            ),
                        ))
                        subprocess.run(
                            ["ffmpeg", "-y", "-i", f["path"], "-ar", "44100", "-ac", "2", norm],
                            check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                            creationflags=_NO_WIN,
                        )
                        probe = subprocess.run(
                            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
                             "-of", "default=noprint_wrappers=1:nokey=1", norm],
                            capture_output=True, text=True, creationflags=_NO_WIN,
                        )
                        dur_s = float(probe.stdout.strip() or "0")
                        durations_ms.append(int(dur_s * 1000))
                        wav_files.append(norm)
                    elapsed_now = time.time() - start
                    self.root.after(0, lambda el=elapsed_now: (
                        _comb_progress.set(0.75),
                        _comb_status.configure(text=f"Building chapters…  {el:.0f}s elapsed",
                                               text_color=COLORS["warning"]),
                    ))

                    concat_txt = os.path.join(tmp, "concat.txt")
                    with open(concat_txt, "w", encoding="utf-8") as fh:
                        for w in wav_files:
                            fh.write(f"file '{w.replace(chr(92), '/')}'\n")

                    meta_txt = os.path.join(tmp, "chapters.txt")
                    with open(meta_txt, "w", encoding="utf-8") as fh:
                        fh.write(";FFMETADATA1\ntitle=Combined Audiobook\nartist=KoKoFish\n\n")
                        cursor = 0
                        for f, dur in zip(_comb_files, durations_ms):
                            title = os.path.splitext(f["name"])[0]
                            fh.write(f"[CHAPTER]\nTIMEBASE=1/1000\nSTART={cursor}\nEND={cursor+dur}\ntitle={title}\n\n")
                            cursor += dur

                    ext = os.path.splitext(out)[1].lower()
                    if ext == ".wav":
                        cmd = ["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", concat_txt, "-c", "copy", out]
                    elif ext == ".mp3":
                        cmd = ["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", concat_txt, "-i", meta_txt,
                               "-map_metadata", "1", "-c:a", "libmp3lame", "-b:a", "192k", "-id3v2_version", "3", out]
                    else:
                        cmd = ["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", concat_txt, "-i", meta_txt,
                               "-map_metadata", "1", "-c:a", "aac", "-b:a", "128k", "-movflags", "+faststart", out]
                    self.root.after(0, lambda: (
                        _comb_progress.set(0.9),
                        _comb_status.configure(text=t("FILE_LAB_STATUS_ENCODING"), text_color=COLORS["warning"]),
                    ))
                    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, creationflags=_NO_WIN)

                    total_s   = sum(durations_ms) / 1000
                    elapsed   = time.time() - start
                    m, s = divmod(int(total_s), 60)
                    h, m = divmod(m, 60)
                    dur_str = f"{h}h {m}m {s}s" if h else f"{m}m {s}s"
                    _comb_cancel.set()
                    self.root.after(0, lambda: _comb_progress.set(1.0))
                    self.root.after(0, lambda ds=dur_str, el=elapsed: _comb_status.configure(
                        text=f"✅ Saved — {len(_comb_files)} chapters, {ds}  ({el:.0f}s total)",
                        text_color=COLORS["success"],
                    ))
                    self.root.after(1500, lambda: _comb_progress.pack_forget())
                except Exception as exc:
                    _comb_cancel.set()
                    logger.error("Combine export failed: %s", exc)
                    self.root.after(0, lambda: _comb_progress.pack_forget())
                    self.root.after(0, lambda e=str(exc): _comb_status.configure(
                        text=f"⚠ {e}", text_color=COLORS["danger"],
                    ))
                finally:
                    import shutil
                    shutil.rmtree(tmp, ignore_errors=True)

            threading.Thread(target=_run, daemon=True, name="CombineExport").start()

        # Also support drag-and-drop into _list_frame
        try:
            from tkinterdnd2 import DND_FILES
            _list_frame.drop_target_register(DND_FILES)
            def _on_drop(e):
                for p in self._parse_drop_data(e.data):
                    if os.path.isfile(p):
                        _comb_files.append({"path": p, "name": os.path.basename(p)})
                _comb_rebuild()
            _list_frame.dnd_bind("<<Drop>>", _on_drop)
        except Exception:
            pass

        # Buttons row
        _comb_btn_row = ctk.CTkFrame(comb_card, fg_color="transparent")
        _comb_btn_row.pack(fill="x", padx=14, pady=(0, 12))
        _bs = {"font": (FONT_FAMILY, 11, "bold"), "corner_radius": 6, "height": 28}
        ctk.CTkButton(_comb_btn_row, text=t("FILE_LAB_BTN_ADD_FILES"),
                      fg_color=COLORS["accent"], hover_color=COLORS["accent_hover"],
                      command=_comb_add, width=110, **_bs).pack(side="left", padx=(0, 6))
        ctk.CTkButton(_comb_btn_row, text=t("FILE_LAB_BTN_CLEAR_ALL"),
                      fg_color=COLORS["bg_input"], hover_color=COLORS["bg_card_hover"],
                      border_color=COLORS["border"], border_width=1,
                      command=_comb_clear, width=80, **_bs).pack(side="left", padx=(0, 12))
        ctk.CTkButton(_comb_btn_row, text=t("FILE_LAB_BTN_EXPORT_AUDIOBOOK"),
                      fg_color="#2d6a4f", hover_color="#1b4332",
                      command=_comb_export, width=160, **_bs).pack(side="left")

        _comb_rebuild()

    # ==================================================================
    # TAB 5: Listen Lab
    # ==================================================================

    def _build_listen_lab_tab(self):
        tab = self.tab_listen

        # ── Header ───────────────────────────────────────────────────────────
        hdr = ctk.CTkFrame(tab, fg_color="transparent")
        hdr.pack(fill="x", padx=10, pady=(10, 4))
        ctk.CTkLabel(hdr, text=t("LISTEN_LAB_HEADER"),
                     font=(FONT_FAMILY, 18, "bold"),
                     text_color=COLORS["text_primary"]).pack(side="left")

        # ── Translate panel ──────────────────────────────────────────────────
        self._listen_translate_var = tk.BooleanVar(value=False)
        # Default to translated display name for "Spanish"
        self._listen_translate_lang_var = ctk.StringVar(value=_display_from_en(_LANG_KEY_MAP, "Spanish"))
        self._listen_translate_voice_var = ctk.StringVar(
            value=getattr(self, "tts_voice_var", ctk.StringVar()).get()
            if hasattr(self, "tts_voice_var") else "Default (Random)"
        )

        tr_card = ctk.CTkFrame(tab, fg_color=COLORS["bg_card"], corner_radius=10)
        tr_card.pack(fill="x", padx=10, pady=(0, 4))

        tr_top = ctk.CTkFrame(tr_card, fg_color="transparent")
        tr_top.pack(fill="x", padx=12, pady=(8, 4))

        tr_switch = ctk.CTkSwitch(
            tr_top, text=t("LISTEN_LAB_BTN_TRANSLATE_REREAD"),
            variable=self._listen_translate_var,
            font=(FONT_FAMILY, 12, "bold"),
            text_color=COLORS["text_primary"],
            progress_color=COLORS["accent"],
            button_color=COLORS["accent_light"],
            command=self._listen_translate_toggled,
        )
        tr_switch.pack(side="left")
        self._make_tooltip(tr_switch, t("LISTEN_LAB_TOOLTIP_TR_SWITCH"))

        # Controls shown only when translate is ON
        self._listen_tr_controls = ctk.CTkFrame(tr_card, fg_color="transparent")
        self._listen_tr_controls.pack(fill="x", padx=12, pady=(0, 8))

        _lf = {"font": (FONT_FAMILY, 11), "text_color": COLORS["text_muted"], "anchor": "w"}
        _ef = {"fg_color": COLORS["bg_input"], "button_color": COLORS["accent"],
               "text_color": COLORS["text_primary"], "font": (FONT_FAMILY, 11),
               "height": 30, "dynamic_resizing": False}

        # Language
        lang_col = ctk.CTkFrame(self._listen_tr_controls, fg_color="transparent")
        lang_col.pack(side="left", padx=(0, 16))
        ctk.CTkLabel(lang_col, text=t("LISTEN_LAB_TARGET_LANG_LABEL"), **_lf).pack(anchor="w")
        lang_menu = ctk.CTkOptionMenu(
            lang_col, variable=self._listen_translate_lang_var,
            values=_display_names(_LANG_KEY_MAP), width=180, **_ef)
        lang_menu.pack()
        self._make_tooltip(lang_menu, t("LISTEN_LAB_TOOLTIP_LANG_MENU"))

        # Voice
        voice_col = ctk.CTkFrame(self._listen_tr_controls, fg_color="transparent")
        voice_col.pack(side="left", padx=(0, 16))
        ctk.CTkLabel(voice_col, text=t("LISTEN_LAB_TTS_VOICE_LABEL"), **_lf).pack(anchor="w")
        _engine = getattr(self.settings, "engine", "kokoro")
        if _engine == "kokoro":
            from kokoro_engine import KOKORO_VOICES
            _voice_names = list(KOKORO_VOICES.keys())
        else:
            _voice_names = self.voices.get_voice_names() if hasattr(self, "voices") else ["Default (Random)"]
        self._listen_voice_menu = ctk.CTkOptionMenu(
            voice_col, variable=self._listen_translate_voice_var,
            values=_voice_names, width=200, **_ef)
        self._listen_voice_menu.pack()
        self._make_tooltip(self._listen_voice_menu, t("LISTEN_LAB_TOOLTIP_VOICE_MENU"))

        # Hide controls initially (translate is OFF by default)
        self._listen_tr_controls.pack_forget()

        # ── Drop-zone hint + Add button ──────────────────────────────────────
        _hint_row = ctk.CTkFrame(tab, fg_color="transparent")
        _hint_row.pack(fill="x", padx=10, pady=(2, 0))
        ctk.CTkButton(
            _hint_row, text=t("LISTEN_LAB_BTN_ADD_FILES"), width=110, height=28,
            corner_radius=7, font=(FONT_FAMILY, 11),
            fg_color=COLORS["bg_input"], hover_color=COLORS["bg_card_hover"],
            command=self._listen_browse_add,
        ).pack(side="left", padx=(0, 10))
        ctk.CTkLabel(
            _hint_row,
            text=t("LISTEN_LAB_DROP_HINT"),
            font=(FONT_FAMILY, 11),
            text_color=COLORS["text_muted"],
        ).pack(side="left")

        # ── Scrollable playlist ──────────────────────────────────────────────
        self._listen_scroll = ctk.CTkScrollableFrame(
            tab, fg_color=COLORS["bg_card"], corner_radius=10)
        self._listen_scroll.pack(fill="both", expand=True, padx=10, pady=(6, 4))

        # ── Bottom bar ───────────────────────────────────────────────────────
        bot = ctk.CTkFrame(tab, fg_color=COLORS["bg_card"], corner_radius=10, height=52)
        bot.pack(fill="x", padx=10, pady=(0, 10))
        bot.pack_propagate(False)

        _bs = {"height": 34, "corner_radius": 7, "font": (FONT_FAMILY, 12)}
        self._btn_listen_play = ctk.CTkButton(
            bot, text=t("LISTEN_LAB_BTN_PLAY_SELECTED"), width=140,
            fg_color=COLORS["accent"], hover_color=COLORS["accent_hover"],
            command=self._listen_play_selected, **_bs)
        self._btn_listen_play.pack(side="left", padx=(10, 4), pady=9)
        self._make_tooltip(self._btn_listen_play, t("LISTEN_LAB_TOOLTIP_PLAY_ALL"))

        _prev_btn = ctk.CTkButton(bot, text="⏮", width=40,
                      fg_color=COLORS["bg_input"], hover_color=COLORS["bg_card_hover"],
                      command=self._listen_prev, **_bs)
        _prev_btn.pack(side="left", padx=(0, 2), pady=9)
        self._make_tooltip(_prev_btn, t("LISTEN_LAB_TOOLTIP_PREV"))

        _next_btn = ctk.CTkButton(bot, text="⏭", width=40,
                      fg_color=COLORS["bg_input"], hover_color=COLORS["bg_card_hover"],
                      command=self._listen_next, **_bs)
        _next_btn.pack(side="left", padx=(0, 8), pady=9)
        self._make_tooltip(_next_btn, t("LISTEN_LAB_TOOLTIP_NEXT"))

        _rm_sel = ctk.CTkButton(bot, text=t("LISTEN_LAB_BTN_REMOVE_SELECTED"), width=150,
                      fg_color=COLORS["danger"], hover_color="#d43d62",
                      command=self._listen_remove_selected, **_bs)
        _rm_sel.pack(side="left", padx=(0, 12), pady=9)
        self._make_tooltip(_rm_sel, t("LISTEN_LAB_TOOLTIP_REMOVE_SEL"))

        _all_btn = ctk.CTkButton(bot, text=t("LISTEN_LAB_BTN_SELECT_ALL"), width=70,
                      fg_color=COLORS["bg_input"], hover_color=COLORS["bg_card_hover"],
                      command=lambda: self._listen_select_all(True), **_bs)
        _all_btn.pack(side="left", padx=(0, 4), pady=9)
        self._make_tooltip(_all_btn, t("LISTEN_LAB_TOOLTIP_SELECT_ALL"))

        _none_btn = ctk.CTkButton(bot, text=t("LISTEN_LAB_BTN_SELECT_NONE"), width=70,
                      fg_color=COLORS["bg_input"], hover_color=COLORS["bg_card_hover"],
                      command=lambda: self._listen_select_all(False), **_bs)
        _none_btn.pack(side="left", padx=(0, 4), pady=9)
        self._make_tooltip(_none_btn, t("LISTEN_LAB_TOOLTIP_DESELECT_ALL"))

        self._listen_status = ctk.CTkLabel(bot, text="",
                                           font=(FONT_FAMILY, 11),
                                           text_color=COLORS["text_secondary"])
        self._listen_status.pack(side="right", padx=(0, 12))

        # Speed control
        _spd_frame = ctk.CTkFrame(bot, fg_color="transparent")
        _spd_frame.pack(side="right", padx=(0, 4))
        ctk.CTkLabel(_spd_frame, text=t("LISTEN_LAB_SPEED_LABEL"),
                     font=(FONT_FAMILY, 10), text_color=COLORS["text_muted"],
                     width=38).pack(side="left")
        self._listen_speed_label = ctk.CTkLabel(_spd_frame, text="1.0×",
                     font=(FONT_FAMILY, 10), text_color=COLORS["text_secondary"],
                     width=34)
        self._listen_speed_label.pack(side="right")
        ctk.CTkSlider(
            _spd_frame,
            from_=0.5, to=2.0,
            variable=self._listen_speed_var,
            width=100, height=16,
            progress_color=COLORS["accent"],
            button_color=COLORS["accent_light"],
            button_hover_color=COLORS["accent"],
            command=self._listen_speed_changed,
        ).pack(side="left", padx=4)

        # Volume control
        _vol_frame = ctk.CTkFrame(bot, fg_color="transparent")
        _vol_frame.pack(side="right", padx=(0, 8))
        ctk.CTkLabel(_vol_frame, text=t("LISTEN_LAB_VOL_LABEL"),
                     font=(FONT_FAMILY, 10), text_color=COLORS["text_muted"],
                     width=24).pack(side="left")
        self._listen_vol_label = ctk.CTkLabel(_vol_frame, text="100%",
                     font=(FONT_FAMILY, 10), text_color=COLORS["text_secondary"],
                     width=34)
        self._listen_vol_label.pack(side="right")
        ctk.CTkSlider(
            _vol_frame,
            from_=0, to=100,
            variable=self._listen_vol_var,
            width=110, height=16,
            progress_color=COLORS["accent"],
            button_color=COLORS["accent_light"],
            button_hover_color=COLORS["accent"],
            command=lambda v: self._listen_vol_label.configure(text=f"{int(v)}%"),
        ).pack(side="left", padx=4)

        # ── Drag-and-drop ────────────────────────────────────────────────────
        try:
            import tkinterdnd2 as _dnd
            tab.drop_target_register(_dnd.DND_FILES)
            tab.dnd_bind("<<Drop>>", self._listen_on_drop)
            self._listen_scroll.drop_target_register(_dnd.DND_FILES)
            self._listen_scroll.dnd_bind("<<Drop>>", self._listen_on_drop)
        except Exception:
            pass

        self._rebuild_listen_ui()
        self._listen_update_playhead()

    # ------------------------------------------------------------------
    # Listen Lab — drag-to-reorder
    # ------------------------------------------------------------------

    def _listen_drag_start(self, idx: int):
        self._listen_drag_source = idx
        self._listen_drag_target = idx

    def _listen_drag_motion(self, event):
        if self._listen_drag_source < 0:
            return
        y_root = event.y_root
        new_tgt = self._listen_drag_source
        for i, r in enumerate(self._listen_drag_rows):
            try:
                ry = r.winfo_rooty()
                rh = r.winfo_height()
                if ry <= y_root < ry + rh:
                    new_tgt = i
                    break
            except Exception:
                pass
        if new_tgt == self._listen_drag_target:
            return
        self._listen_drag_target = new_tgt
        # Update row colours without a full rebuild
        for i, r in enumerate(self._listen_drag_rows):
            if i == new_tgt:
                r.configure(fg_color=COLORS["accent"])
            elif i == self._listen_drag_source:
                r.configure(fg_color=COLORS["bg_card_hover"])
            else:
                sel = self._listen_items[i]["selected"]
                r.configure(fg_color=COLORS["bg_input"] if sel else COLORS["bg_card"])

    def _listen_drag_end(self, event):
        if self._listen_drag_source < 0:
            return
        src = self._listen_drag_source
        dst = self._listen_drag_target
        self._listen_drag_source = -1
        self._listen_drag_target = -1
        if src != dst:
            item = self._listen_items.pop(src)
            self._listen_items.insert(dst, item)
        self._rebuild_listen_ui()

    def _listen_translate_toggled(self):
        if self._listen_translate_var.get():
            self._listen_tr_controls.pack(fill="x", padx=12, pady=(0, 8))
        else:
            self._listen_tr_controls.pack_forget()

    def _listen_browse_add(self):
        """Open file dialog to add audio files to the Listen Lab playlist."""
        paths = filedialog.askopenfilenames(
            title="Add Audio Files",
            filetypes=[
                ("Audio files", "*.mp3 *.wav *.m4a *.m4b *.flac *.ogg *.opus *.aac *.wma *.weba *.webm *.amr"),
                ("All files", "*.*"),
            ],
            parent=self.root,
        )
        added = 0
        for p in paths:
            if os.path.isfile(p):
                self._listen_items.append({"path": p, "name": os.path.basename(p), "selected": False})
                added += 1
        if added:
            self._rebuild_listen_ui()
            self._listen_status.configure(text=f"Added {added} file(s)")

    def _listen_on_drop(self, event):
        _AUDIO_EXTS = {
            ".mp3", ".wav", ".m4a", ".m4b", ".flac", ".ogg",
            ".opus", ".aac", ".wma", ".weba", ".webm", ".amr",
        }
        paths = self._parse_drop_data(event.data)
        added = 0
        for p in paths:
            ext = os.path.splitext(p)[1].lower()
            # Accept known audio extensions OR any file (let ffmpeg decide)
            if ext in _AUDIO_EXTS or ext == "":
                self._listen_items.append({"path": p, "name": os.path.basename(p), "selected": False})
                added += 1
                logger.info("Listen Lab: added %s", p)
            else:
                logger.warning("Listen Lab: skipped drop (unknown ext %s): %s", ext, p)
        if added:
            self._rebuild_listen_ui()
            self._listen_status.configure(text=f"Added {added} file(s)")
        elif paths:
            self._listen_status.configure(text=t("LISTEN_LAB_MSG_NO_AUDIO_DROP"))

    def _rebuild_listen_ui(self):
        for w in self._listen_scroll.winfo_children():
            w.destroy()
        self._listen_drag_rows = []

        if not self._listen_items:
            ctk.CTkLabel(self._listen_scroll,
                         text=t("LISTEN_LAB_MSG_NO_FILES"),
                         font=(FONT_FAMILY, 12),
                         text_color=COLORS["text_muted"]).pack(pady=30)
            return

        _ib = {"width": 32, "height": 32, "corner_radius": 5, "font": (FONT_FAMILY, 14)}
        translate_on = getattr(self, "_listen_translate_var", None) and self._listen_translate_var.get()

        for idx, item in enumerate(self._listen_items):
            tl_status  = item.get("tl_status")   # None|transcribing|translating|converting|done|error
            tl_progress = item.get("tl_progress", 0.0)
            tl_path    = item.get("tl_path")
            is_active  = tl_status in ("transcribing", "translating", "converting")
            is_done    = tl_status == "done"
            is_error   = tl_status == "error"
            is_playing = (idx == self._listen_preview_idx and not self._listen_preview_paused)
            is_paused_play = (idx == self._listen_preview_idx and self._listen_preview_paused)

            _stripe = "#1a1a2e" if idx % 2 == 0 else "#16162a"
            row_color = COLORS["bg_input"] if item["selected"] else _stripe
            row = ctk.CTkFrame(self._listen_scroll, fg_color=row_color, corner_radius=7)
            row.pack(fill="x", padx=4, pady=2)
            self._listen_drag_rows.append(row)

            # ── Top line ─────────────────────────────────────────────────
            top = ctk.CTkFrame(row, fg_color="transparent", height=40)
            top.pack(fill="x")
            top.pack_propagate(False)

            # Drag handle
            _handle = ctk.CTkLabel(
                top, text="⠿", width=18,
                font=(FONT_FAMILY, 16), text_color=COLORS["text_muted"],
                cursor="hand2",
            )
            _handle.pack(side="left", padx=(6, 0))
            _handle.bind("<ButtonPress-1>",   lambda e, i=idx: self._listen_drag_start(i))
            _handle.bind("<B1-Motion>",        self._listen_drag_motion)
            _handle.bind("<ButtonRelease-1>",  self._listen_drag_end)

            # Checkbox
            var = tk.BooleanVar(value=item["selected"])
            def _on_check(v=var, i=idx):
                self._listen_items[i]["selected"] = v.get()
                self._rebuild_listen_ui()
            cb = ctk.CTkCheckBox(top, text="", variable=var, width=24,
                                 command=_on_check, fg_color=COLORS["accent"],
                                 hover_color=COLORS["accent_hover"])
            cb.pack(side="left", padx=(4, 4))

            # ── Right buttons (packed right before name label) ─────────────
            # Remove ✕
            _rb = ctk.CTkButton(top, text="✕",
                                fg_color=COLORS["danger"], hover_color="#d43d62",
                                command=lambda i=idx: self._listen_remove(i), **_ib)
            _rb.pack(side="right", padx=(0, 6))
            self._make_tooltip(_rb, t("LISTEN_LAB_TOOLTIP_REMOVE_ITEM"))

            if is_active:
                # Cancel ⊘
                _cb = ctk.CTkButton(top, text="⊘",
                                    fg_color=COLORS["warning"], hover_color="#e6bc5c",
                                    text_color="#1a1a2e",
                                    command=lambda i=idx: self._listen_cancel_pipeline(i), **_ib)
                _cb.pack(side="right", padx=(0, 2))
                self._make_tooltip(_cb, t("LISTEN_LAB_TOOLTIP_CANCEL_CONV"))

                # Pause ⏸ (only during TTS conversion — earlier stages can't pause)
                if tl_status == "converting":
                    _tl_paused = item.get("tl_paused", False)
                    _pause_btn = ctk.CTkButton(
                        top,
                        text="▶" if _tl_paused else "⏸",
                        fg_color=COLORS["warning"] if _tl_paused else COLORS["bg_input"],
                        hover_color="#e6bc5c" if _tl_paused else COLORS["bg_card_hover"],
                        border_color=COLORS["warning"], border_width=1,
                        command=lambda i=idx: self._listen_pause_pipeline(i), **_ib)
                    _pause_btn.pack(side="right", padx=(0, 2))
                    self._make_tooltip(_pause_btn, t("LISTEN_LAB_TOOLTIP_RESUME") if _tl_paused else t("LISTEN_LAB_TOOLTIP_PAUSE_CONV"))

            elif is_done and tl_path and os.path.isfile(tl_path):
                # Metadata ⓘ (of translated audio)
                _mb = ctk.CTkButton(top, text="ⓘ",
                                    fg_color=COLORS["bg_input"], hover_color=COLORS["bg_card_hover"],
                                    border_color=COLORS["border"], border_width=1,
                                    command=lambda p=tl_path: self._open_audio_meta_editor(p),
                                    **_ib)
                _mb.pack(side="right", padx=(0, 2))
                self._make_tooltip(_mb, t("LISTEN_LAB_TOOLTIP_EDIT_META"))

                # Save 💾
                _sb = ctk.CTkButton(top, text="💾",
                                    fg_color=COLORS["bg_input"], hover_color=COLORS["bg_card_hover"],
                                    border_color=COLORS["border"], border_width=1,
                                    command=lambda p=tl_path, n=item["name"]: self._listen_save_translated(p, n),
                                    **_ib)
                _sb.pack(side="right", padx=(0, 2))
                self._make_tooltip(_sb, t("LISTEN_LAB_TOOLTIP_SAVE_AUDIO"))

                # Transport strip packed right-to-left: ✕ … [⏮][⏪][▶/⏸][⏩][⏭] … ⓘ
                # ⏭ next chapter (always visible) — rightmost of strip
                _nxt = ctk.CTkButton(top, text="⏭",
                                     fg_color=COLORS["bg_input"], hover_color=COLORS["bg_card_hover"],
                                     border_color=COLORS["border"], border_width=1,
                                     command=self._listen_next, **_ib)
                _nxt.pack(side="right", padx=(0, 2))
                self._make_tooltip(_nxt, t("LISTEN_LAB_TOOLTIP_NEXT_CH"))
                # ⏩ forward 30 s (only when active)
                if is_playing or is_paused_play:
                    _fwd = ctk.CTkButton(top, text="⏩",
                                         fg_color=COLORS["bg_input"], hover_color=COLORS["bg_card_hover"],
                                         border_color=COLORS["border"], border_width=1,
                                         command=lambda: self._listen_seek(30), **_ib)
                    _fwd.pack(side="right", padx=(0, 2))
                    self._make_tooltip(_fwd, t("LISTEN_LAB_TOOLTIP_SKIP_FWD"))
                # ▶/⏸ — centre of strip
                _pp = ctk.CTkButton(
                    top,
                    text="⏸" if is_playing else "▶",
                    fg_color=COLORS["success"] if is_playing else COLORS["bg_input"],
                    hover_color="#05b890" if is_playing else COLORS["bg_card_hover"],
                    border_color=COLORS["success"], border_width=1,
                    command=lambda i=idx: self._listen_toggle_play(i), **_ib)
                _pp.pack(side="right", padx=(0, 2))
                self._make_tooltip(_pp, t("LISTEN_LAB_TOOLTIP_PAUSE") if is_playing else t("LISTEN_LAB_TOOLTIP_PLAY_TRANSLATED"))
                # ⏪ rewind 30 s (only when active)
                if is_playing or is_paused_play:
                    _rwd = ctk.CTkButton(top, text="⏪",
                                         fg_color=COLORS["bg_input"], hover_color=COLORS["bg_card_hover"],
                                         border_color=COLORS["border"], border_width=1,
                                         command=lambda: self._listen_seek(-30), **_ib)
                    _rwd.pack(side="right", padx=(0, 2))
                    self._make_tooltip(_rwd, t("LISTEN_LAB_TOOLTIP_REWIND"))
                # ⏮ previous chapter — leftmost of strip
                _prv = ctk.CTkButton(top, text="⏮",
                                     fg_color=COLORS["bg_input"], hover_color=COLORS["bg_card_hover"],
                                     border_color=COLORS["border"], border_width=1,
                                     command=self._listen_prev, **_ib)
                _prv.pack(side="right", padx=(0, 2))
                self._make_tooltip(_prv, t("LISTEN_LAB_TOOLTIP_PREV_CH"))

            else:
                # Normal: metadata ⓘ on original file
                _mb = ctk.CTkButton(top, text="ⓘ",
                                    fg_color=COLORS["bg_input"], hover_color=COLORS["bg_card_hover"],
                                    border_color=COLORS["border"], border_width=1,
                                    command=lambda p=item["path"]: self._open_audio_meta_editor(p),
                                    **_ib)
                _mb.pack(side="right", padx=(0, 2))
                self._make_tooltip(_mb, t("LISTEN_LAB_TOOLTIP_EDIT_META"))

                # Transport strip packed right-to-left: ✕ … [⏮][⏪][▶/⏸][⏩][⏭] … ⓘ
                # ⏭ next chapter (always visible) — rightmost of strip
                _nxt = ctk.CTkButton(top, text="⏭",
                                     fg_color=COLORS["bg_input"], hover_color=COLORS["bg_card_hover"],
                                     border_color=COLORS["border"], border_width=1,
                                     command=self._listen_next, **_ib)
                _nxt.pack(side="right", padx=(0, 2))
                self._make_tooltip(_nxt, t("LISTEN_LAB_TOOLTIP_NEXT_CH"))
                # ⏩ forward 30 s (only when active)
                if is_playing or is_paused_play:
                    _fwd = ctk.CTkButton(top, text="⏩",
                                         fg_color=COLORS["bg_input"], hover_color=COLORS["bg_card_hover"],
                                         border_color=COLORS["border"], border_width=1,
                                         command=lambda: self._listen_seek(30), **_ib)
                    _fwd.pack(side="right", padx=(0, 2))
                    self._make_tooltip(_fwd, t("LISTEN_LAB_TOOLTIP_SKIP_FWD"))
                # ▶/⏸ — centre of strip
                _pp = ctk.CTkButton(
                    top,
                    text="⏸" if is_playing else "▶",
                    fg_color=COLORS["success"] if is_playing else COLORS["bg_input"],
                    hover_color="#05b890" if is_playing else COLORS["bg_card_hover"],
                    border_color=COLORS["success"], border_width=1,
                    command=lambda i=idx: self._listen_toggle_play(i), **_ib)
                _pp.pack(side="right", padx=(0, 2))
                self._make_tooltip(_pp, t("LISTEN_LAB_TOOLTIP_PAUSE") if is_playing else t("LISTEN_LAB_TOOLTIP_PLAY_AUDIO"))
                # ⏪ rewind 30 s (only when active)
                if is_playing or is_paused_play:
                    _rwd = ctk.CTkButton(top, text="⏪",
                                         fg_color=COLORS["bg_input"], hover_color=COLORS["bg_card_hover"],
                                         border_color=COLORS["border"], border_width=1,
                                         command=lambda: self._listen_seek(-30), **_ib)
                    _rwd.pack(side="right", padx=(0, 2))
                    self._make_tooltip(_rwd, t("LISTEN_LAB_TOOLTIP_REWIND"))
                # ⏮ previous chapter — leftmost of strip
                _prv = ctk.CTkButton(top, text="⏮",
                                     fg_color=COLORS["bg_input"], hover_color=COLORS["bg_card_hover"],
                                     border_color=COLORS["border"], border_width=1,
                                     command=self._listen_prev, **_ib)
                _prv.pack(side="right", padx=(0, 2))
                self._make_tooltip(_prv, t("LISTEN_LAB_TOOLTIP_PREV_CH"))

            # Name / status label — no expand so waveform can fill remaining space
            if is_active:
                _status_map = {
                    "transcribing": t("LISTEN_LAB_STATUS_TRANSCRIBING"),
                    "translating":  t("LISTEN_LAB_STATUS_TRANSLATING"),
                    "converting":   t("LISTEN_LAB_STATUS_CONVERTING"),
                }
                ctk.CTkLabel(top, text=_status_map.get(tl_status, tl_status),
                             font=(FONT_FAMILY, 11, "italic"),
                             text_color=COLORS["warning"], anchor="w").pack(
                                 side="left", padx=4)
            elif is_error:
                err_short = str(item.get("tl_error", "Error"))[:60]
                ctk.CTkLabel(top, text=f"⚠  {err_short}",
                             font=(FONT_FAMILY, 11), text_color=COLORS["danger"],
                             anchor="w").pack(side="left", padx=4)
            else:
                suffix = "  ✓" if is_done else ""
                # Truncate long names so the waveform always gets space
                _name_str = item["name"] + suffix
                if len(_name_str) > 36:
                    _name_str = _name_str[:34] + "…"
                ctk.CTkLabel(top, text=_name_str,
                             font=(FONT_FAMILY, 12),
                             text_color=COLORS["success"] if is_done else COLORS["text_primary"],
                             anchor="w").pack(side="left", padx=4)

            # ── Waveform inline (right of name, inside the top row) ───────
            waveform = item.get("waveform")
            duration  = item.get("duration_sec", 0)
            if waveform and duration > 0:
                wf_canvas = tk.Canvas(
                    top, height=28, bg="#0f0f1a",
                    highlightthickness=1, highlightbackground="#1e2a3a", bd=0,
                    cursor="hand2",
                )
                wf_canvas.pack(side="left", fill="x", expand=True, padx=(6, 6), pady=6)
                item["_waveform_canvas"] = wf_canvas
                item["_waveform_data"]   = waveform
                item["_waveform_dur"]    = duration

                def _draw_waveform(c=wf_canvas, wf=waveform, it=item, i=idx):
                    c.update_idletasks()
                    w = c.winfo_width()
                    h = c.winfo_height()
                    if w < 2 or h < 2:
                        return
                    c.delete("all")
                    n = len(wf)
                    if n == 0:
                        return
                    mid = h // 2
                    peak = max(wf) or 1.0
                    bar_w = max(1, w / n)
                    for k, amp in enumerate(wf):
                        x = k * bar_w
                        bar_h = int((amp / peak) * mid * 0.9)
                        x0, x1 = int(x), max(int(x + bar_w) - 1, int(x) + 1)
                        c.create_rectangle(x0, mid - bar_h, x1, mid + bar_h,
                                           fill="#3a5f8a", outline="", tags="wave")
                    # Playhead
                    dur = it.get("duration_sec", 0)
                    pos = it.get("_playhead_sec", 0.0)
                    if dur > 0:
                        px = int((pos / dur) * w)
                        c.create_line(px, 0, px, h, fill="white", width=2, tags="playhead")

                def _on_wf_seek(event, c=wf_canvas, it=item, i=idx):
                    w = c.winfo_width()
                    dur = it.get("duration_sec", 0)
                    if w > 0 and dur > 0:
                        ratio = max(0.0, min(1.0, event.x / w))
                        target_sec = ratio * dur
                        self._listen_seek_abs(i, target_sec)

                wf_canvas.bind("<Configure>", lambda e, d=_draw_waveform: d())
                wf_canvas.bind("<ButtonPress-1>", _on_wf_seek)
                wf_canvas.bind("<B1-Motion>",     _on_wf_seek)
                wf_canvas.after(50, _draw_waveform)
            else:
                item["_waveform_canvas"] = None
                if not item.get("waveform") and not item.get("_wf_computing"):
                    item["_wf_computing"] = True
                    self._listen_compute_waveform(idx, item["path"])

            # ── Progress bar (only during TTS conversion) ─────────────────
            if tl_status == "converting":
                pb_frame = ctk.CTkFrame(row, fg_color="transparent", height=18)
                pb_frame.pack(fill="x", padx=12, pady=(0, 4))
                pb_frame.pack_propagate(False)
                pb = ctk.CTkProgressBar(pb_frame, progress_color=COLORS["accent"],
                                        fg_color=COLORS["bg_input"], height=10,
                                        corner_radius=5)
                pb.set(tl_progress)
                pb.pack(side="left", fill="x", expand=True, pady=4)
                pct_lbl = ctk.CTkLabel(pb_frame, text=f"{int(tl_progress*100)}%",
                                       font=(FONT_FAMILY, 9),
                                       text_color=COLORS["text_muted"], width=32)
                pct_lbl.pack(side="left", padx=(4, 0))
                # Store for in-place updates (avoids full rebuild on every tick)
                item["_tl_pb"] = pb
                item["_tl_pct"] = pct_lbl

    # ── Listen Lab playback / pipeline ────────────────────────────────

    def _listen_toggle_play(self, idx: int):
        import sounddevice as _sd
        import soundfile as _sf

        if idx < 0 or idx >= len(self._listen_items):
            return
        item = self._listen_items[idx]
        translate_on = getattr(self, "_listen_translate_var", None) and self._listen_translate_var.get()

        # Translate mode: route through pipeline
        if translate_on:
            tl_status = item.get("tl_status")
            if tl_status == "done" and item.get("tl_path") and os.path.isfile(item["tl_path"]):
                pass  # fall through to normal play with tl_path
            elif tl_status in ("transcribing", "translating", "converting"):
                self._listen_pause_pipeline(idx)
                return
            else:
                self._listen_run_pipeline(idx)
                return

        # Normal play (or translated-done playback)
        play_path = item.get("tl_path") if (translate_on and item.get("tl_status") == "done") else item["path"]
        if not play_path or not os.path.isfile(play_path):
            messagebox.showwarning(t("MAIN_TAB_LISTEN_LAB"), t("LISTEN_LAB_MSG_FILE_NOT_FOUND", path=play_path), parent=self.root)
            return

        # ── Same item tapped: toggle pause ────────────────────────────────
        if self._listen_preview_idx == idx:
            if self._listen_is_ffmpeg:
                # ffmpeg streaming: pause = stop + remember position; resume = restart
                if self._listen_preview_paused:
                    # Resume — restart ffmpeg from saved position
                    resume_sec = self._listen_preview_pos_sec
                    self._listen_preview_paused = False
                    self._rebuild_listen_ui()
                    self._listen_start_stream(idx, play_path, start_sec=resume_sec)
                else:
                    # Pause — save position and kill ffmpeg
                    self._listen_preview_pos_sec = self._listen_preview_pos / (self._listen_preview_sr or 44100)
                    self._listen_preview_paused = True
                    self._listen_kill_ffmpeg()
                    if self._listen_preview_stream:
                        try:
                            self._listen_preview_stream.stop()
                            self._listen_preview_stream.close()
                        except Exception:
                            pass
                        self._listen_preview_stream = None
                    self._rebuild_listen_ui()
            else:
                # soundfile streaming: simple callback-level pause
                self._listen_preview_paused = not self._listen_preview_paused
                self._rebuild_listen_ui()
            return

        # ── New item: stop previous and start fresh ───────────────────────
        self._listen_stop_preview()
        self._listen_start_stream(idx, play_path, start_sec=0.0)

    def _listen_start_stream(self, idx: int, path: str, start_sec: float = 0.0):
        """Start streaming playback of *path* beginning at *start_sec* seconds."""
        import sounddevice as _sd
        import soundfile as _sf

        self._listen_preview_idx = idx
        self._listen_preview_paused = False
        self._listen_preview_pos = 0
        self._listen_preview_pos_sec = start_sec
        self._listen_stream_gen += 1

        # ── Determine whether soundfile can open this format ──────────────
        _sf_ok = False
        try:
            with _sf.SoundFile(path) as _t:
                _sr = _t.samplerate
            _sf_ok = True
        except Exception:
            pass

        if _sf_ok:
            # ── Native streaming via soundfile (WAV, FLAC, OGG, AIFF…) ───
            self._listen_is_ffmpeg = False
            sf_handle = _sf.SoundFile(path)
            sr = sf_handle.samplerate
            if start_sec > 0:
                sf_handle.seek(int(start_sec * sr))
            self._listen_sf_handle = sf_handle
            self._listen_preview_sr = sr
            self._listen_preview_pos = sf_handle.tell()
            self._rebuild_listen_ui()

            def _cb(outdata, frames, time_info, status):
                if self._listen_preview_paused:
                    outdata[:] = 0
                    return
                vol = self._listen_vol_var.get() / 100.0
                data = sf_handle.read(frames, dtype="float32", always_2d=True)
                n = len(data)
                if n == 0:
                    outdata[:] = 0
                    raise _sd.CallbackStop()
                chunk = (data[:, 0] if data.ndim > 1 else data.ravel()) * vol
                outdata[:n, 0] = chunk
                if n < frames:
                    outdata[n:] = 0
                _spd = round(float(self._listen_speed_var.get()), 2)
                if abs(_spd - 1.0) > 0.02 and n > 0:
                    # crude speed: advance position by speed factor
                    extra = int(n * (_spd - 1.0))
                    if extra > 0:
                        sf_handle.seek(min(sf_handle.tell() + extra, sf_handle.frames - 1))
                    elif extra < 0:
                        sf_handle.seek(max(sf_handle.tell() + extra, 0))
                self._listen_preview_pos = sf_handle.tell()
                self._listen_preview_pos_sec = self._listen_preview_pos / sr

            _my_gen_sf = self._listen_stream_gen

            def _finished_sf():
                try:
                    sf_handle.close()
                except Exception:
                    pass
                self._listen_sf_handle = None
                self._listen_preview_stream = None
                # If paused intentionally or a newer stream has started, don't reset state
                if self._listen_preview_paused or self._listen_stream_gen != _my_gen_sf:
                    self.root.after(0, self._rebuild_listen_ui)
                    return
                self._listen_preview_idx = -1
                self.root.after(0, self._rebuild_listen_ui)
                _cb_done = getattr(self, "_listen_run_plain_on_done", None)
                self._listen_run_plain_on_done = None
                if _cb_done:
                    self.root.after(50, _cb_done)
                elif getattr(self, "_listen_queue", []):
                    self.root.after(50, self._listen_play_next)

            stream = _sd.OutputStream(
                samplerate=sr, channels=1, dtype="float32",
                callback=_cb, finished_callback=_finished_sf,
            )
            self._listen_preview_stream = stream
            stream.start()

        else:
            # ── ffmpeg pipe streaming (M4A, MP3, AAC, WMA, OPUS…) ─────────
            self._listen_is_ffmpeg = True
            self._listen_ffmpeg_path = path
            SR = 44100
            self._listen_preview_sr = SR
            self._listen_preview_pos = int(start_sec * SR)

            # Locate ffmpeg (bundled first, then PATH)
            _app_dir = os.path.dirname(os.path.abspath(__file__))
            _ffmpeg = os.path.join(_app_dir, "bin", "ffmpeg.exe")
            if not os.path.isfile(_ffmpeg):
                import shutil as _sh
                _ffmpeg = _sh.which("ffmpeg") or "ffmpeg"

            _speed = round(float(self._listen_speed_var.get()), 2)
            cmd = [_ffmpeg]
            if start_sec > 0:
                cmd += ["-ss", f"{start_sec:.3f}"]
            cmd += ["-i", path]
            if abs(_speed - 1.0) > 0.02:
                # clamp atempo to 0.5–2.0 range
                _tempo = max(0.5, min(2.0, _speed))
                cmd += ["-filter:a", f"atempo={_tempo:.2f}"]
            cmd += ["-f", "s16le", "-ac", "1", "-ar", str(SR), "pipe:1"]

            try:
                proc = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.DEVNULL,
                    creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                )
            except Exception as exc:
                messagebox.showerror(t("MAIN_TAB_LISTEN_LAB"), t("LISTEN_LAB_MSG_FFMPEG_ERROR", error=exc), parent=self.root)
                self._listen_preview_idx = -1
                return

            self._listen_ffmpeg_proc = proc
            self._rebuild_listen_ui()

            def _cb_ff(outdata, frames, time_info, status):
                if self._listen_preview_paused:
                    outdata[:] = 0
                    return
                vol = self._listen_vol_var.get() / 100.0
                raw = proc.stdout.read(frames * 2)  # s16le = 2 bytes/sample
                if not raw:
                    outdata[:] = 0
                    raise _sd.CallbackStop()
                samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
                n = len(samples)
                if n < frames:
                    outdata[:n, 0] = samples * vol
                    outdata[n:] = 0
                else:
                    outdata[:, 0] = samples[:frames] * vol
                self._listen_preview_pos += n
                self._listen_preview_pos_sec = self._listen_preview_pos / SR

            _my_gen_ff = self._listen_stream_gen

            def _finished_ff():
                self._listen_kill_ffmpeg()
                self._listen_preview_stream = None
                # If paused intentionally or a newer stream has started, don't reset state
                if self._listen_preview_paused or self._listen_stream_gen != _my_gen_ff:
                    self.root.after(0, self._rebuild_listen_ui)
                    return
                self._listen_preview_idx = -1
                self.root.after(0, self._rebuild_listen_ui)
                _cb_done = getattr(self, "_listen_run_plain_on_done", None)
                self._listen_run_plain_on_done = None
                if _cb_done:
                    self.root.after(50, _cb_done)
                elif getattr(self, "_listen_queue", []):
                    self.root.after(50, self._listen_play_next)

            stream = _sd.OutputStream(
                samplerate=SR, channels=1, dtype="float32",
                callback=_cb_ff, finished_callback=_finished_ff,
            )
            self._listen_preview_stream = stream
            stream.start()

    def _listen_kill_ffmpeg(self):
        """Kill the ffmpeg subprocess if one is running."""
        proc = getattr(self, "_listen_ffmpeg_proc", None)
        if proc:
            try:
                proc.kill()
                proc.wait(timeout=2)
            except Exception:
                pass
            self._listen_ffmpeg_proc = None

    def _listen_compute_waveform(self, idx: int, path: str):
        """Compute a downsampled waveform for the given item in a background thread."""
        def _worker():
            try:
                _app_dir = os.path.dirname(os.path.abspath(__file__))
                _ffmpeg = os.path.join(_app_dir, "bin", "ffmpeg.exe")
                if not os.path.isfile(_ffmpeg):
                    import shutil as _sh
                    _ffmpeg = _sh.which("ffmpeg") or "ffmpeg"

                proc = subprocess.Popen(
                    [_ffmpeg, "-i", path, "-f", "s16le", "-ac", "1", "-ar", "4000", "pipe:1"],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.DEVNULL,
                    creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                )
                raw = proc.stdout.read()
                proc.wait()
                if not raw:
                    return

                samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
                n_blocks = 300
                block_size = max(1, len(samples) // n_blocks)
                waveform = []
                for i in range(0, len(samples) - block_size + 1, block_size):
                    block = samples[i:i + block_size]
                    waveform.append(float(np.sqrt(np.mean(block ** 2))))

                duration_sec = len(samples) / 4000.0

                if 0 <= idx < len(self._listen_items):
                    self._listen_items[idx]["waveform"] = waveform
                    self._listen_items[idx]["duration_sec"] = duration_sec
                    self.root.after(0, self._rebuild_listen_ui)
            except Exception as exc:
                logger.debug("Waveform compute failed for %s: %s", path, exc)

        threading.Thread(target=_worker, daemon=True, name=f"WaveformCompute-{idx}").start()

    def _listen_update_playhead(self):
        """Periodic timer — updates the waveform playhead without rebuilding the full UI."""
        try:
            if self._listen_preview_idx >= 0:
                idx = self._listen_preview_idx
                if 0 <= idx < len(self._listen_items):
                    item = self._listen_items[idx]
                    dur = item.get("duration_sec", 0)
                    canvas = item.get("_waveform_canvas")
                    if canvas and dur > 0:
                        try:
                            if not canvas.winfo_exists():
                                raise RuntimeError("gone")
                            pos = self._listen_preview_pos_sec
                            item["_playhead_sec"] = pos
                            w = canvas.winfo_width()
                            h = canvas.winfo_height()
                            if w > 1 and h > 1:
                                px = int(min(1.0, pos / dur) * w)
                                canvas.delete("playhead")
                                canvas.create_line(px, 0, px, h,
                                                   fill="white", width=2, tags="playhead")
                        except Exception:
                            pass
        except Exception:
            pass
        self.root.after(500, self._listen_update_playhead)

    def _listen_stop_preview(self):
        if self._listen_preview_stream:
            try:
                self._listen_preview_stream.stop()
                self._listen_preview_stream.close()
            except Exception:
                pass
            self._listen_preview_stream = None
        self._listen_kill_ffmpeg()
        sf_h = getattr(self, "_listen_sf_handle", None)
        if sf_h:
            try:
                sf_h.close()
            except Exception:
                pass
            self._listen_sf_handle = None
        self._listen_preview_idx = -1
        self._listen_preview_paused = False
        self._listen_is_ffmpeg = False

    def _listen_seek(self, seconds: float):
        """Seek forward (positive) or backward (negative) in the currently playing audio."""
        if self._listen_preview_idx < 0:
            return
        if self._listen_is_ffmpeg:
            # ffmpeg: stop and restart at new position
            idx = self._listen_preview_idx
            path = self._listen_ffmpeg_path
            new_sec = max(0.0, self._listen_preview_pos_sec + seconds)
            was_paused = self._listen_preview_paused
            self._listen_stop_preview()
            self._listen_preview_idx = idx  # restore so UI shows correctly
            if not was_paused:
                self.root.after(50, lambda: self._listen_start_stream(idx, path, start_sec=new_sec))
            else:
                # Paused: update position without restarting
                self._listen_preview_pos_sec = new_sec
                self._listen_preview_paused = True
                self._listen_preview_idx = idx
                self._rebuild_listen_ui()
        else:
            # soundfile: seek directly on the open file handle
            sf_h = getattr(self, "_listen_sf_handle", None)
            if sf_h and self._listen_preview_sr:
                delta = int(seconds * self._listen_preview_sr)
                new_pos = max(0, self._listen_preview_pos + delta)
                try:
                    sf_h.seek(new_pos)
                    self._listen_preview_pos = sf_h.tell()
                    self._listen_preview_pos_sec = self._listen_preview_pos / self._listen_preview_sr
                except Exception:
                    pass

    def _listen_seek_abs(self, idx: int, target_sec: float):
        """Seek to an absolute position (seconds) for the given item."""
        if self._listen_preview_idx != idx:
            # Item not playing — just update playhead visually
            if 0 <= idx < len(self._listen_items):
                self._listen_items[idx]["_playhead_sec"] = target_sec
            return
        delta = target_sec - self._listen_preview_pos_sec
        self._listen_seek(delta)

    def _listen_prev(self):
        """Stop current playback and play the previous item."""
        prev_idx = self._listen_preview_idx - 1
        if prev_idx < 0:
            prev_idx = 0
        self._listen_stop_preview()
        self.root.after(50, lambda: self._listen_toggle_play(prev_idx))
        self._rebuild_listen_ui()

    def _listen_next(self):
        """Stop current playback and play the next item."""
        next_idx = self._listen_preview_idx + 1
        if next_idx >= len(self._listen_items):
            return
        self._listen_stop_preview()
        self.root.after(50, lambda: self._listen_toggle_play(next_idx))
        self._rebuild_listen_ui()

    def _listen_speed_changed(self, value):
        """Called when the speed slider moves. Updates label and restarts stream if playing."""
        speed = round(float(value), 2)
        # Snap to 1.0 when close
        if abs(speed - 1.0) < 0.05:
            speed = 1.0
            self._listen_speed_var.set(1.0)
        lbl = getattr(self, "_listen_speed_label", None)
        if lbl:
            lbl.configure(text=f"{speed:.1f}×")
        # Restart stream at current position with new speed if something is playing
        if self._listen_preview_idx >= 0 and not self._listen_preview_paused:
            idx = self._listen_preview_idx
            path = (self._listen_ffmpeg_path if self._listen_is_ffmpeg
                    else self._listen_items[idx]["path"] if 0 <= idx < len(self._listen_items) else None)
            if path and os.path.isfile(path):
                pos_sec = self._listen_preview_pos_sec
                self._listen_stop_preview()
                self.root.after(80, lambda: self._listen_start_stream(idx, path, start_sec=pos_sec))

    def _listen_play_selected(self):
        """Play/pipeline all selected items in sequence."""
        selected = [i for i, it in enumerate(self._listen_items) if it["selected"]]
        if not selected:
            self._listen_status.configure(text=t("LISTEN_LAB_MSG_NO_SELECTION"))
            return
        translate_on = getattr(self, "_listen_translate_var", None) and self._listen_translate_var.get()
        if translate_on:
            # Run pipeline queue: start first, chain via on_done callback
            self._listen_queue = list(selected)
            self._listen_pipeline_next()
        else:
            self._listen_queue = list(selected)
            self._listen_play_next()

    def _listen_play_next(self):
        """Advance normal (no-translate) playback queue."""
        if not getattr(self, "_listen_queue", []):
            return
        nxt = self._listen_queue.pop(0)
        self._listen_toggle_play(nxt)

    def _listen_pipeline_next(self):
        """Advance translate pipeline queue."""
        if not getattr(self, "_listen_queue", []):
            return
        nxt = self._listen_queue[0]  # don't pop yet — pipeline callback will pop
        if nxt < 0 or nxt >= len(self._listen_items):
            self._listen_queue.pop(0)
            self._listen_pipeline_next()
            return
        item = self._listen_items[nxt]
        # If already done, just play it and chain
        if item.get("tl_status") == "done" and item.get("tl_path"):
            self._listen_queue.pop(0)
            def _chain():
                if getattr(self, "_listen_queue", []):
                    self.root.after(300, self._listen_pipeline_next)
            self._listen_run_plain(nxt, on_done=_chain)
        else:
            self._listen_run_pipeline(nxt, on_done=lambda: (
                self._listen_queue.pop(0) if self._listen_queue and self._listen_queue[0] == nxt else None,
                self.root.after(300, self._listen_pipeline_next),
            ))

    # ── Listen Lab plain playback helper ──────────────────────────────

    def _listen_run_plain(self, idx: int, on_done=None):
        """Play a single item (original or translated) without pipeline.

        Delegates to _listen_start_stream so all formats (M4A, MP3, WAV…)
        and all file sizes work without loading into memory.
        on_done() is called via _listen_queue auto-advance when the track ends.
        """
        if idx < 0 or idx >= len(self._listen_items):
            if on_done:
                on_done()
            return
        item = self._listen_items[idx]
        play_path = item.get("tl_path") if item.get("tl_status") == "done" else item["path"]
        if not play_path or not os.path.isfile(play_path):
            if on_done:
                on_done()
            return

        self._listen_stop_preview()
        # Store on_done so _listen_play_next can chain it after playback ends
        self._listen_run_plain_on_done = on_done
        self._listen_start_stream(idx, play_path, start_sec=0.0)

    # ── Listen Lab translate pipeline ─────────────────────────────────

    def _listen_run_pipeline(self, idx: int, on_done=None):
        """STT → Qwen translate → TTS, streaming playback of TTS chunks."""
        if idx < 0 or idx >= len(self._listen_items):
            return
        item = self._listen_items[idx]

        # Check LLM availability up-front
        from tag_suggester import is_llm_available, is_qwen_model_ready
        if not is_llm_available() or not is_qwen_model_ready():
            messagebox.showwarning(
                t("MAIN_TAB_LISTEN_LAB"),
                t("LISTEN_LAB_MSG_QWEN_NOT_READY"),
                parent=self.root,
            )
            return

        self._listen_stop_preview()
        item["tl_status"]   = "transcribing"
        item["tl_progress"] = 0.0
        item["tl_path"]     = None
        item["tl_cancel"]   = False
        item["tl_paused"]   = False
        self._rebuild_listen_ui()
        self._listen_status.configure(text=f"Transcribing: {item['name']}")

        src_path = item["path"]
        lang     = _en_from_display(_LANG_KEY_MAP, self._listen_translate_lang_var.get())
        voice_name = self._listen_translate_voice_var.get()
        engine   = getattr(self.settings, "engine", "kokoro")

        def _cancelled():
            return item.get("tl_cancel", False)

        # ── Step 2+3: translate then TTS (runs in thread) ─────────────
        def _translate_and_tts(transcript: str):
            if _cancelled():
                return
            # Translate
            item["tl_status"] = "translating"
            self.root.after(0, lambda: (
                self._rebuild_listen_ui(),
                self._listen_status.configure(text=f"Translating → {lang}…"),
            ))
            try:
                from tag_suggester import translate_for_voice
                translated = translate_for_voice(transcript, lang)
                if not translated or not translated.strip():
                    translated = transcript
            except Exception as exc:
                logger.warning("Listen pipeline translate failed: %s", exc)
                translated = transcript

            if _cancelled():
                return

            # TTS
            item["tl_status"]   = "converting"
            item["tl_progress"] = 0.0
            self.root.after(0, lambda: (
                self._rebuild_listen_ui(),
                self._listen_status.configure(text=f"Converting: {item['name']}…"),
            ))

            os.makedirs(AUDIO_TEMP_DIR, exist_ok=True)
            safe = re.sub(r"[^\w\-]", "_", os.path.splitext(item["name"])[0])
            out_path = os.path.join(AUDIO_TEMP_DIR, f"listen_tl_{safe}_{int(time.time())}.wav")

            import queue as _q
            import sounddevice as _sd2
            _sq = _q.Queue()
            _SENT = object()
            _stream = [None]

            def _stream_cb(outdata, frames, _t, _st):
                if item.get("tl_paused", False):
                    outdata[:] = 0
                    return
                vol = (self.vol_slider.get() / 100.0) if hasattr(self, "vol_slider") else 1.0
                remaining = frames
                offset = 0
                outdata[:] = 0
                while remaining > 0:
                    if _stream_cb._buf is None or len(_stream_cb._buf) == 0:
                        try:
                            chunk = _sq.get_nowait()
                            if chunk is _SENT:
                                return
                            _stream_cb._buf = chunk
                        except _q.Empty:
                            return
                    take = min(remaining, len(_stream_cb._buf))
                    out = (_stream_cb._buf[:take] * vol).astype(np.float32)
                    outdata[offset:offset+take, 0] = out
                    _stream_cb._buf = _stream_cb._buf[take:]
                    offset += take
                    remaining -= take
            _stream_cb._buf = None

            def _on_chunk(chunk_np, sr):
                if _cancelled():
                    return
                if _stream[0] is None:
                    s = _sd2.OutputStream(samplerate=sr, channels=1, dtype="float32",
                                          blocksize=2048, callback=_stream_cb)
                    s.start()
                    _stream[0] = s
                    self._listen_preview_stream = s
                    self._listen_preview_idx  = idx
                    self._listen_preview_paused = False
                    self.root.after(0, self._rebuild_listen_ui)
                _sq.put(chunk_np.astype(np.float32))

            def _on_progress(status_txt, frac):
                item["tl_progress"] = frac
                # Update progress bar in-place if widget still alive
                pb = item.get("_tl_pb")
                lbl = item.get("_tl_pct")
                def _upd():
                    try:
                        if pb and pb.winfo_exists():
                            pb.set(frac)
                        if lbl and lbl.winfo_exists():
                            lbl.configure(text=f"{int(frac*100)}%")
                    except Exception:
                        pass
                self.root.after(0, _upd)

            def _on_complete(wav_path):
                item["tl_path"]   = wav_path
                item["tl_status"] = "done"
                _sq.put(_SENT)

                def _finish():
                    import time as _t
                    while not _sq.empty():
                        _t.sleep(0.05)
                    _t.sleep(0.3)
                    if _stream[0]:
                        try:
                            _stream[0].stop()
                            _stream[0].close()
                        except Exception:
                            pass
                    self._listen_preview_stream = None
                    self._listen_preview_idx    = -1
                    self.root.after(0, self._rebuild_listen_ui)
                    self.root.after(0, lambda: self._listen_status.configure(
                        text=f"Done: {item['name']}"))
                    if on_done:
                        self.root.after(100, on_done)

                threading.Thread(target=_finish, daemon=True, name="ListenFinish").start()

            def _on_error(exc):
                item["tl_status"] = "error"
                item["tl_error"]  = str(exc)
                self.root.after(0, self._rebuild_listen_ui)
                if on_done:
                    self.root.after(0, on_done)

            _is_kokoro = (engine == "kokoro")
            if _is_kokoro:
                from kokoro_engine import KOKORO_VOICES, DEFAULT_VOICE
                vid = KOKORO_VOICES.get(voice_name, DEFAULT_VOICE)
                self.tts.generate(
                    text=translated, voice_id=vid,
                    speed=self.speed_slider.get(),
                    output_path=out_path,
                    on_progress=_on_progress, on_chunk=_on_chunk,
                    on_complete=_on_complete, on_error=_on_error,
                )
            else:
                profile = self.voices.get_voice(voice_name) if voice_name != "Default (Random)" else None
                self.tts.generate(
                    text=translated,
                    reference_wav=profile["wav_path"]    if profile else None,
                    reference_tokens=None,
                    prompt_text=profile["prompt_text"] if profile else None,
                    speed=self.speed_slider.get(),
                    cadence=self.cad_slider.get() / 100.0,
                    output_path=out_path,
                    on_progress=_on_progress, on_chunk=_on_chunk,
                    on_complete=_on_complete, on_error=_on_error,
                )

        # ── Step 1: STT transcribe ────────────────────────────────────
        def _on_stt_done(text, _info):
            if _cancelled():
                return
            threading.Thread(
                target=_translate_and_tts,
                args=(text.strip(),),
                daemon=True, name="ListenTranslateTTS",
            ).start()

        def _on_stt_error(exc):
            logger.warning("Listen pipeline STT failed: %s", exc)
            # Fall back: translate empty text → TTS the original transcript
            threading.Thread(
                target=_translate_and_tts,
                args=("",),
                daemon=True, name="ListenTranslateTTS",
            ).start()

        def _start_stt():
            self.stt.transcribe(
                audio_path=src_path,
                on_complete=_on_stt_done,
                on_error=_on_stt_error,
            )

        if self.stt.is_loaded:
            threading.Thread(target=_start_stt, daemon=True, name="ListenSTT").start()
        else:
            self.stt.load_model(
                on_ready=lambda: threading.Thread(target=_start_stt, daemon=True,
                                                  name="ListenSTT").start(),
                on_error=_on_stt_error,
            )

    def _listen_cancel_pipeline(self, idx: int):
        if 0 <= idx < len(self._listen_items):
            self._listen_items[idx]["tl_cancel"] = True
            self._listen_items[idx]["tl_status"] = None
            self._listen_stop_preview()
            self._rebuild_listen_ui()

    def _listen_pause_pipeline(self, idx: int):
        if 0 <= idx < len(self._listen_items):
            item = self._listen_items[idx]
            item["tl_paused"] = not item.get("tl_paused", False)
            # Also pause/resume the preview stream
            self._listen_preview_paused = item["tl_paused"]
            self._rebuild_listen_ui()

    def _listen_save_translated(self, tl_path: str, name: str):
        if not tl_path or not os.path.isfile(tl_path):
            return
        from tkinter.filedialog import asksaveasfilename
        stem = os.path.splitext(name)[0]
        dest = asksaveasfilename(
            parent=self.root,
            defaultextension=".mp3",
            filetypes=[("MP3 audio", "*.mp3"), ("WAV audio", "*.wav"), ("All files", "*.*")],
            initialfile=f"{stem}_translated.mp3",
            title="Save translated audio as…",
        )
        if dest:
            try:
                from pydub import AudioSegment
                audio = AudioSegment.from_wav(tl_path)
                fmt = os.path.splitext(dest)[1].lstrip(".").lower() or "mp3"
                audio.export(dest, format=fmt)
                self._listen_status.configure(text=f"Saved: {os.path.basename(dest)}")
            except Exception as exc:
                messagebox.showerror(t("COMMON_BTN_SAVE"), t("LISTEN_LAB_MSG_SAVE_FAILED", error=exc), parent=self.root)

    def _listen_remove(self, idx: int):
        if self._listen_preview_idx == idx:
            self._listen_stop_preview()
        if 0 <= idx < len(self._listen_items):
            self._listen_items[idx]["tl_cancel"] = True
            self._listen_items.pop(idx)
            self._rebuild_listen_ui()

    def _listen_remove_selected(self):
        self._listen_stop_preview()
        for it in self._listen_items:
            if it.get("selected"):
                it["tl_cancel"] = True
        self._listen_items = [it for it in self._listen_items if not it["selected"]]
        self._rebuild_listen_ui()

    def _listen_select_all(self, state: bool):
        for it in self._listen_items:
            it["selected"] = state
        self._rebuild_listen_ui()

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
            text=t("VOICE_LAB_TITLE"),
            font=(FONT_FAMILY, 18, "bold"),
            text_color=COLORS["text_primary"],
        ).pack(side="left")

        self.btn_clone = ctk.CTkButton(
            header,
            text=t("VOICE_LAB_BTN_CLONE"),
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
            text=t("VOICE_LAB_BTN_REFRESH"),
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
            text=t("VOICE_LAB_UPLOAD_HINT"),
            font=(FONT_FAMILY, 11),
            text_color=COLORS["text_muted"],
            justify="left",
        ).pack(anchor="w", padx=15, pady=(0, 6))

        # ── Sub-tab container ──────────────────────────────────────────
        sub_tabs = ctk.CTkTabview(
            tab,
            fg_color=COLORS["bg_card"],
            segmented_button_selected_color=COLORS["accent"],
            segmented_button_selected_hover_color=COLORS["accent_hover"],
        )
        sub_tabs.pack(fill="both", expand=True, padx=10, pady=(0, 10))
        tab_design = sub_tabs.add("Voice Design")
        tab_cloning = sub_tabs.add("Voice Cloning")

        # ── Voice Design tab ───────────────────────────────────────────
        ctk.CTkLabel(
            tab_design,
            text="Kokoro preset voices (54). Select one and click Use to make it the default.",
            font=(FONT_FAMILY, 11),
            text_color=COLORS["text_muted"],
            justify="left",
        ).pack(anchor="w", padx=8, pady=(6, 6))

        self._kokoro_voice_rows: dict[str, ctk.CTkFrame] = {}

        design_scroll = ctk.CTkScrollableFrame(
            tab_design,
            fg_color=COLORS["bg_input"],
            corner_radius=8,
            scrollbar_button_color=COLORS["accent"],
        )
        design_scroll.pack(fill="both", expand=True, padx=8, pady=(0, 8))

        def _make_kokoro_row(display_name: str, voice_id: str):
            lang = KOKORO_VOICE_LANG.get(voice_id, "")
            is_active = (getattr(self.settings, "kokoro_voice", "") == voice_id)
            row = ctk.CTkFrame(
                design_scroll,
                fg_color=COLORS["bg_card"],
                border_color=COLORS["accent"] if is_active else COLORS["border"],
                border_width=2 if is_active else 1,
                corner_radius=8,
                height=48,
            )
            row.pack(fill="x", padx=4, pady=3)
            row.pack_propagate(False)

            ctk.CTkLabel(
                row,
                text=f"{display_name}",
                font=(FONT_FAMILY, 12, "bold"),
                text_color=COLORS["text_primary"],
                width=180,
                anchor="w",
            ).pack(side="left", padx=(12, 4))

            ctk.CTkLabel(
                row,
                text=voice_id,
                font=(FONT_FAMILY, 11),
                text_color=COLORS["text_muted"],
                width=110,
                anchor="w",
            ).pack(side="left", padx=(0, 4))

            ctk.CTkLabel(
                row,
                text=lang,
                font=(FONT_FAMILY, 11),
                text_color=COLORS["text_secondary"],
                width=140,
                anchor="w",
            ).pack(side="left", padx=(0, 8))

            ctk.CTkButton(
                row,
                text="Use as Default",
                width=120,
                height=28,
                corner_radius=6,
                fg_color=COLORS["accent"],
                hover_color=COLORS["accent_hover"],
                font=(FONT_FAMILY, 11, "bold"),
                command=lambda vid=voice_id: self._set_kokoro_default(vid),
            ).pack(side="right", padx=(4, 8))

            ctk.CTkButton(
                row,
                text="Preview",
                width=80,
                height=28,
                corner_radius=6,
                fg_color=COLORS["bg_input"],
                hover_color=COLORS["bg_card_hover"],
                border_color=COLORS["border"],
                border_width=1,
                font=(FONT_FAMILY, 11),
                command=lambda vid=voice_id: self._preview_kokoro_voice(vid),
            ).pack(side="right", padx=(4, 0))

            self._kokoro_voice_rows[voice_id] = row

        for _display, _vid in KOKORO_VOICES.items():
            _make_kokoro_row(_display, _vid)

        # ── Voice Cloning tab ──────────────────────────────────────────
        if getattr(self.settings, "engine", "") == "kokoro":
            banner = ctk.CTkFrame(
                tab_cloning,
                fg_color=COLORS["bg_card"],
                border_color=COLORS["warning"],
                border_width=1,
                corner_radius=8,
            )
            banner.pack(fill="x", padx=8, pady=(6, 4))
            ctk.CTkLabel(
                banner,
                text="Switch to VoxCPM or OmniVoice in Settings to use cloned voices.",
                font=(FONT_FAMILY, 11, "bold"),
                text_color=COLORS["warning"],
            ).pack(anchor="w", padx=10, pady=6)

        ctk.CTkLabel(
            tab_cloning,
            text="Clone voices from reference audio. Requires VoxCPM or OmniVoice (not Kokoro).",
            font=(FONT_FAMILY, 11),
            text_color=COLORS["text_muted"],
            justify="left",
        ).pack(anchor="w", padx=8, pady=(6, 6))

        # Mic recorder card
        self._build_mic_recorder_card(tab_cloning)

        # Voice grid (scrollable)
        self.voice_grid_frame = ctk.CTkScrollableFrame(
            tab_cloning,
            fg_color=COLORS["bg_input"],
            corner_radius=8,
            scrollbar_button_color=COLORS["accent"],
        )
        self.voice_grid_frame.pack(fill="both", expand=True, padx=8, pady=(0, 8))

        self._refresh_voice_grid()

    def _set_kokoro_default(self, voice_id: str):
        """Set the active Kokoro preset voice and refresh row highlights."""
        self.settings.kokoro_voice = voice_id
        self.settings.default_voice = voice_id
        try:
            self.settings.save()
        except Exception as exc:
            logger.warning("Failed to save settings: %s", exc)
        # Refresh row styling
        rows = getattr(self, "_kokoro_voice_rows", {})
        for vid, row in rows.items():
            active = (vid == voice_id)
            try:
                row.configure(
                    border_color=COLORS["accent"] if active else COLORS["border"],
                    border_width=2 if active else 1,
                )
            except Exception:
                pass

    def _preview_kokoro_voice(self, voice_id: str):
        """Generate a short sample with the given Kokoro voice."""
        from tkinter import messagebox
        if getattr(self.settings, "engine", "") != "kokoro":
            messagebox.showinfo(
                "Preview unavailable",
                "Switch the active engine to Kokoro in Settings to preview preset voices.",
            )
            return
        if not self.tts or not getattr(self.tts, "is_loaded", False):
            messagebox.showinfo(
                "Preview unavailable",
                "Kokoro engine is not loaded yet. Open the Speech Lab tab or wait for startup to finish.",
            )
            return

        def _on_complete(wav_path):
            try:
                import sounddevice as sd
                import soundfile as sf
                data, sr = sf.read(wav_path, dtype="float32")
                sd.play(data, sr)
            except Exception as exc:
                logger.warning("Kokoro preview playback failed: %s", exc)

        def _on_error(exc):
            logger.error("Kokoro preview generation failed: %s", exc)

        try:
            self.tts.generate(
                text="The quick brown fox jumps over the lazy dog.",
                voice_id=voice_id,
                on_complete=_on_complete,
                on_error=_on_error,
            )
            logger.info("Kokoro preview requested: %s", voice_id)
        except Exception as exc:
            logger.error("Kokoro preview failed to start: %s", exc)
            messagebox.showinfo("Preview", f"Could not start preview: {exc}")

    def _build_mic_recorder_card(self, parent):
        """Build the microphone recording section for voice cloning."""
        import time as _time

        MAX_SEC = 180

        # Instance state
        self._mic_rec_stream     = None
        self._mic_rec_frames     = []
        self._mic_rec_start      = 0.0
        self._mic_rec_running    = False
        self._mic_rec_path       = None
        self._mic_preview_stream = None
        self._mic_preview_audio  = None
        self._mic_preview_pos    = 0
        self._mic_preview_sr     = 44100
        self._mic_preview_paused = False
        self._mic_timer_id       = None

        # ── Card ────────────────────────────────────────────────────────
        card = ctk.CTkFrame(parent, fg_color=COLORS["bg_card"], corner_radius=10)
        card.pack(fill="x", padx=15, pady=(0, 8))

        ctk.CTkLabel(card, text=t("VOICE_LAB_RECORD_HEADER"),
                     font=(FONT_FAMILY, 13, "bold"),
                     text_color=COLORS["text_secondary"]).pack(anchor="w", padx=14, pady=(10, 6))

        # ── Mic selector ─────────────────────────────────────────────────
        mic_row = ctk.CTkFrame(card, fg_color="transparent")
        mic_row.pack(fill="x", padx=14, pady=(0, 6))
        ctk.CTkLabel(mic_row, text=t("VOICE_LAB_MICROPHONE_LABEL"),
                     font=(FONT_FAMILY, 11), text_color=COLORS["text_muted"],
                     width=90, anchor="w").pack(side="left")

        try:
            import sounddevice as _sd_q
            _devs = _sd_q.query_devices()
            _input_devs = [(i, d["name"]) for i, d in enumerate(_devs) if d["max_input_channels"] > 0]
        except Exception:
            _input_devs = []
        _dev_names = [d[1] for d in _input_devs] or ["Default"]
        _mic_var = ctk.StringVar(value=_dev_names[0])
        ctk.CTkOptionMenu(mic_row, variable=_mic_var, values=_dev_names,
                          fg_color=COLORS["bg_input"], button_color=COLORS["accent"],
                          text_color=COLORS["text_primary"], font=(FONT_FAMILY, 11),
                          height=30, dynamic_resizing=False).pack(
                              side="left", fill="x", expand=True, padx=(8, 0))

        # ── Record / Stop buttons ─────────────────────────────────────────
        btn_row = ctk.CTkFrame(card, fg_color="transparent")
        btn_row.pack(fill="x", padx=14, pady=(0, 4))
        _bs = {"height": 34, "corner_radius": 7, "font": (FONT_FAMILY, 12, "bold")}

        btn_record = ctk.CTkButton(btn_row, text=t("VOICE_LAB_BTN_RECORD"), width=120,
                                   fg_color=COLORS["danger"], hover_color="#d43d62",
                                   command=lambda: _start_recording(), **_bs)
        btn_record.pack(side="left", padx=(0, 8))
        btn_stop = ctk.CTkButton(btn_row, text=t("VOICE_LAB_BTN_STOP_RECORD"), width=100,
                                 fg_color=COLORS["bg_input"], hover_color=COLORS["bg_card_hover"],
                                 border_color=COLORS["border"], border_width=1,
                                 state="disabled", command=lambda: _stop_recording(), **_bs)
        btn_stop.pack(side="left")
        rec_status = ctk.CTkLabel(btn_row, text=t("VOICE_LAB_STATUS_NO_RECORDING"),
                                  font=(FONT_FAMILY, 11), text_color=COLORS["text_muted"])
        rec_status.pack(side="left", padx=(14, 0))

        # ── Progress bar canvas ──────────────────────────────────────────
        # Extra height for marker labels above and time below the bar.
        canvas = tk.Canvas(card, height=58, bg=COLORS["bg_card"],
                           highlightthickness=0, bd=0)
        canvas.pack(fill="x", padx=14, pady=(4, 2))

        _MARKERS = [
            (15,  "minimum",     COLORS["warning"]),
            (30,  "recommended", COLORS["success"]),
            (90,  "ideal",       COLORS["accent_light"]),
        ]
        _BAR_Y  = 26   # top of bar
        _BAR_H  = 14   # height of bar

        def _draw_progress(elapsed: float = 0.0):
            canvas.delete("all")
            w = canvas.winfo_width() or 460

            # Background track
            canvas.create_rectangle(0, _BAR_Y, w, _BAR_Y + _BAR_H,
                                    fill="#2a2a4a", outline="")

            # Filled portion — colour shifts as duration grows
            fill_w = min(elapsed / MAX_SEC, 1.0) * w
            fill_color = (COLORS["danger"]  if elapsed < 15
                          else COLORS["warning"] if elapsed < 30
                          else COLORS["success"])
            if fill_w > 1:
                canvas.create_rectangle(0, _BAR_Y, fill_w, _BAR_Y + _BAR_H,
                                        fill=fill_color, outline="")

            # Marker tick lines + labels above bar
            for sec, label, color in _MARKERS:
                x = (sec / MAX_SEC) * w
                canvas.create_line(x, _BAR_Y - 12, x, _BAR_Y + _BAR_H + 2,
                                   fill=color, width=2, dash=(3, 2))
                canvas.create_text(x, _BAR_Y - 14, text=label,
                                   fill=color, font=("Segoe UI", 8),
                                   anchor="s")

            # Elapsed / max time (bottom-right)
            e_str = f"{int(elapsed // 60)}:{int(elapsed % 60):02d}"
            m_str = f"{MAX_SEC // 60}:{MAX_SEC % 60:02d}"
            canvas.create_text(w - 2, _BAR_Y + _BAR_H + 12,
                               text=f"{e_str} / {m_str}",
                               fill=COLORS["text_muted"],
                               font=("Segoe UI", 9), anchor="e")

        canvas.bind("<Configure>", lambda e: _draw_progress(
            _time.time() - self._mic_rec_start if self._mic_rec_running else 0.0
        ))
        _draw_progress(0.0)

        # ── Playback / result row ─────────────────────────────────────────
        res_row = ctk.CTkFrame(card, fg_color="transparent")
        res_row.pack(fill="x", padx=14, pady=(4, 12))

        rec_file_lbl = ctk.CTkLabel(res_row, text=t("VOICE_LAB_STATUS_NO_RECORDING_YET"),
                                    font=(FONT_FAMILY, 11),
                                    text_color=COLORS["text_muted"], anchor="w")
        rec_file_lbl.pack(side="left", fill="x", expand=True)

        _ib = {"width": 34, "height": 34, "corner_radius": 6, "font": (FONT_FAMILY, 13)}
        btn_preview = ctk.CTkButton(res_row, text="▶",
                                    fg_color=COLORS["bg_input"],
                                    hover_color=COLORS["bg_card_hover"],
                                    border_color=COLORS["border"], border_width=1,
                                    state="disabled",
                                    command=lambda: _toggle_preview(), **_ib)
        btn_preview.pack(side="left", padx=(8, 4))
        self._make_tooltip(btn_preview, t("VOICE_LAB_TOOLTIP_PREVIEW_REC"))

        btn_save_rec = ctk.CTkButton(res_row, text="💾",
                                     fg_color=COLORS["bg_input"],
                                     hover_color=COLORS["bg_card_hover"],
                                     border_color=COLORS["border"], border_width=1,
                                     state="disabled",
                                     command=lambda: _save_recording(), **_ib)
        btn_save_rec.pack(side="left", padx=(0, 16))
        self._make_tooltip(btn_save_rec, t("VOICE_LAB_TOOLTIP_SAVE_REC"))

        btn_clone_rec = ctk.CTkButton(res_row, text=t("VOICE_LAB_BTN_CLONE_FROM_REC"),
                                      width=150, height=36, corner_radius=8,
                                      fg_color=COLORS["accent"],
                                      hover_color=COLORS["accent_hover"],
                                      font=(FONT_FAMILY, 13, "bold"),
                                      state="disabled",
                                      command=lambda: _clone_from_recording())
        btn_clone_rec.pack(side="right")

        # ──────────────────────────────────────────────────────────────────
        # Inner logic (closures referencing local widgets above)
        # ──────────────────────────────────────────────────────────────────

        def _get_device_index():
            sel = _mic_var.get()
            for idx, name in _input_devs:
                if name == sel:
                    return idx
            return None  # sounddevice default

        def _tick():
            if not self._mic_rec_running:
                return
            elapsed = _time.time() - self._mic_rec_start
            _draw_progress(elapsed)
            rec_status.configure(text=f"Recording  {int(elapsed // 60)}:{int(elapsed % 60):02d}")
            if elapsed >= MAX_SEC:
                _stop_recording()
                return
            self._mic_timer_id = parent.after(100, _tick)

        def _start_recording():
            import sounddevice as _sd2
            if self._mic_rec_running:
                return
            _stop_preview_internal()
            self._mic_rec_frames   = []
            self._mic_rec_running  = True
            self._mic_rec_start    = _time.time()
            self._mic_rec_path     = None
            btn_record.configure(state="disabled")
            btn_stop.configure(state="normal")
            btn_preview.configure(state="disabled")
            btn_save_rec.configure(state="disabled")
            btn_clone_rec.configure(state="disabled")
            rec_status.configure(text=t("VOICE_LAB_STATUS_RECORDING_ACTIVE"), text_color=COLORS["danger"])
            rec_file_lbl.configure(text=t("VOICE_LAB_STATUS_RECORDING_ACTIVE"), text_color=COLORS["danger"])

            SR = 44100
            dev_idx = _get_device_index()

            def _audio_cb(indata, frames, _t, _st):
                if self._mic_rec_running:
                    self._mic_rec_frames.append(indata.copy())

            try:
                self._mic_rec_stream = _sd2.InputStream(
                    samplerate=SR, channels=1, dtype="float32",
                    device=dev_idx, callback=_audio_cb,
                )
                self._mic_rec_stream.start()
                self._mic_preview_sr = SR
                self._mic_timer_id = parent.after(100, _tick)
            except Exception as exc:
                self._mic_rec_running = False
                btn_record.configure(state="normal")
                btn_stop.configure(state="disabled")
                rec_status.configure(text=f"Mic error: {exc}",
                                     text_color=COLORS["danger"])

        def _stop_recording():
            if not self._mic_rec_running:
                return
            self._mic_rec_running = False
            if self._mic_timer_id:
                try:
                    parent.after_cancel(self._mic_timer_id)
                except Exception:
                    pass
                self._mic_timer_id = None
            if self._mic_rec_stream:
                try:
                    self._mic_rec_stream.stop()
                    self._mic_rec_stream.close()
                except Exception:
                    pass
                self._mic_rec_stream = None

            elapsed = _time.time() - self._mic_rec_start
            _draw_progress(elapsed)
            btn_record.configure(state="normal")
            btn_stop.configure(state="disabled")

            if not self._mic_rec_frames:
                rec_status.configure(text=t("VOICE_LAB_STATUS_NOTHING_CAPTURED"),
                                     text_color=COLORS["text_muted"])
                rec_file_lbl.configure(text=t("VOICE_LAB_STATUS_NO_RECORDING_YET"),
                                       text_color=COLORS["text_muted"])
                return

            import soundfile as _sf2
            audio_data = np.concatenate(self._mic_rec_frames, axis=0).flatten().astype(np.float32)
            self._mic_preview_audio = audio_data
            self._mic_preview_pos   = 0

            os.makedirs(AUDIO_TEMP_DIR, exist_ok=True)
            tmp_path = os.path.join(AUDIO_TEMP_DIR, f"mic_rec_{int(_time.time())}.wav")
            _sf2.write(tmp_path, audio_data, self._mic_preview_sr)
            self._mic_rec_path = tmp_path

            dur = len(audio_data) / self._mic_preview_sr
            rec_status.configure(text=f"Done — {dur:.1f}s recorded",
                                  text_color=COLORS["success"])
            rec_file_lbl.configure(text=os.path.basename(tmp_path),
                                   text_color=COLORS["text_primary"])
            btn_preview.configure(state="normal")
            btn_save_rec.configure(state="normal")
            btn_clone_rec.configure(state="normal")

        def _stop_preview_internal():
            if self._mic_preview_stream:
                try:
                    self._mic_preview_stream.stop()
                    self._mic_preview_stream.close()
                except Exception:
                    pass
                self._mic_preview_stream = None
            self._mic_preview_paused = False

        def _toggle_preview():
            import sounddevice as _sd2
            # Toggle pause if already streaming
            if self._mic_preview_stream:
                self._mic_preview_paused = not self._mic_preview_paused
                btn_preview.configure(text="▶" if self._mic_preview_paused else "⏸")
                return
            if self._mic_preview_audio is None:
                return
            self._mic_preview_pos    = 0
            self._mic_preview_paused = False

            def _cb(outdata, frames, _t, _st):
                if self._mic_preview_paused:
                    outdata[:] = 0
                    return
                remaining = len(self._mic_preview_audio) - self._mic_preview_pos
                if remaining <= 0:
                    outdata[:] = 0
                    raise _sd2.CallbackStop()
                take = min(frames, remaining)
                chunk = self._mic_preview_audio[
                    self._mic_preview_pos:self._mic_preview_pos + take
                ].astype(np.float32)
                outdata[:take, 0] = chunk
                if take < frames:
                    outdata[take:] = 0
                self._mic_preview_pos += take

            def _done():
                self._mic_preview_stream = None
                self._mic_preview_paused = False
                parent.after(0, lambda: btn_preview.configure(text="▶"))

            stream = _sd2.OutputStream(
                samplerate=self._mic_preview_sr, channels=1,
                dtype="float32", callback=_cb, finished_callback=_done,
            )
            self._mic_preview_stream = stream
            stream.start()
            btn_preview.configure(text="⏸")

        def _save_recording():
            if not self._mic_rec_path or not os.path.isfile(self._mic_rec_path):
                return
            from tkinter.filedialog import asksaveasfilename
            dest = asksaveasfilename(
                parent=self.root,
                defaultextension=".wav",
                filetypes=[("WAV audio", "*.wav"), ("All files", "*.*")],
                initialfile=os.path.basename(self._mic_rec_path),
                title="Save recording as…",
            )
            if dest:
                import shutil
                shutil.copy2(self._mic_rec_path, dest)

        def _clone_from_recording():
            if not self._mic_rec_path or not os.path.isfile(self._mic_rec_path):
                messagebox.showwarning(t("VOICE_LAB_BTN_CLONE"), t("VOICE_LAB_MSG_CLONE_NO_RECORDING"), parent=self.root)
                return
            _stop_preview_internal()
            dialog = ctk.CTkInputDialog(
                text=t("VOICE_LAB_DIALOG_CLONE_PROMPT"),
                title=t("VOICE_LAB_DIALOG_CLONE_TITLE"),
            )
            name = dialog.get_input()
            if not name or not name.strip():
                return
            name = name.strip()
            if self.voices.voice_exists(name):
                messagebox.showwarning("KoKoFish", t("VOICE_LAB_MSG_CLONE_NAME_EXISTS", name=name), parent=self.root)
                return

            rec_path = self._mic_rec_path

            def _do_clone(transcript: str):
                try:
                    tts = self.tts if self.tts.is_loaded else None
                    self.voices.clone_voice(
                        name=name,
                        reference_wav_path=rec_path,
                        tts_engine=tts,
                        prompt_text=transcript,
                    )
                    self.root.after(0, self._refresh_voice_grid)
                    self.root.after(0, lambda: messagebox.showinfo(
                        "KoKoFish",
                        t("VOICE_LAB_MSG_CLONE_SUCCESS", name=name),
                    ))
                except Exception as exc:
                    self.root.after(0, lambda e=exc: messagebox.showerror(
                        "KoKoFish", t("VOICE_LAB_MSG_CLONE_FAILED", error=e)
                    ))

            def _transcribe_and_clone():
                transcript_result = [None]
                done_evt = threading.Event()

                def _on_transcript(text, _info):
                    transcript_result[0] = text.strip()
                    done_evt.set()

                def _on_stt_error(_exc):
                    transcript_result[0] = ""
                    done_evt.set()

                def _on_stt_ready():
                    self.stt.transcribe(
                        audio_path=rec_path,
                        on_complete=_on_transcript,
                        on_error=_on_stt_error,
                    )
                    done_evt.wait(timeout=120)
                    _do_clone(transcript_result[0] or "")

                if self.stt.is_loaded:
                    _on_stt_ready()
                else:
                    self.stt.load_model(on_ready=_on_stt_ready, on_error=_on_stt_error)
                    done_evt.wait(timeout=180)

            self.root.after(0, lambda: self.tts_status.configure(
                text=t("VOICE_LAB_STATUS_TRANSCRIBING_REC")
            ))
            threading.Thread(target=_transcribe_and_clone, daemon=True, name="MicClone").start()

    def _refresh_voice_grid(self):
        """Rebuild the voice cards grid."""
        for widget in self.voice_grid_frame.winfo_children():
            widget.destroy()

        voice_list = self.voices.list_voices()

        if not voice_list:
            ctk.CTkLabel(
                self.voice_grid_frame,
                text=t("VOICE_LAB_EMPTY_MSG"),
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
    # TAB: Script Lab
    # ==================================================================

    def _build_script_lab_tab(self):
        import threading
        from script_engine import (
            default_profile, list_profiles, load_profile, save_profile,
            delete_profile, parse_script, format_script,
            read_source_file, tag_script_with_ai,
            find_characters_in_script, enhance_script_flow,
        )

        tab = self.tab_script

        # ---- State -------------------------------------------------------
        self._script_profile      = default_profile()
        self._script_profile_name = tk.StringVar(value="")
        self._script_playing      = False
        self._script_stop_flag    = False
        self._script_char_rows    = []   # list of {name_var, voice_var, frame}
        self._script_audio_chunks = []   # accumulated audio for export

        # ---- Helpers -----------------------------------------------------
        def _get_voice_options():
            engine = getattr(self.settings, "engine", "kokoro")
            if engine == "kokoro":
                try:
                    from kokoro_engine import KOKORO_VOICES
                    return [""] + list(KOKORO_VOICES.keys())
                except Exception:
                    return [""]
            else:
                try:
                    from voice_manager import VoiceManager
                    vm = VoiceManager(engine)
                    return [""] + vm.list_voices()
                except Exception:
                    return [""]

        def _voice_for_character(char_name: str):
            if char_name == "Narrator":
                return self._script_profile.get("narrator", {}).get("voice", "")
            for ch in self._script_profile.get("characters", []):
                if ch["name"] == char_name:
                    return ch.get("voice", "")
            return ""

        def _refresh_char_list():
            for row in self._script_char_rows:
                row["frame"].destroy()
            self._script_char_rows.clear()
            voices = _get_voice_options()
            for ch in self._script_profile.get("characters", []):
                _add_char_row(ch["name"], ch.get("voice", ""), voices)

        def _add_char_row(name="", voice="", voices=None):
            if voices is None:
                voices = _get_voice_options()
            row_frame = ctk.CTkFrame(char_scroll, fg_color=COLORS["bg_input"], corner_radius=6)
            row_frame.pack(fill="x", padx=4, pady=3)

            name_var  = tk.StringVar(value=name)
            voice_var = tk.StringVar(value=voice)

            ctk.CTkEntry(
                row_frame, textvariable=name_var, width=120,
                fg_color=COLORS["bg_dark"], text_color=COLORS["text_primary"],
                placeholder_text="Character name",
            ).pack(side="left", padx=(6, 4), pady=6)

            ctk.CTkOptionMenu(
                row_frame, variable=voice_var, values=voices, width=160,
                fg_color=COLORS["bg_dark"], text_color=COLORS["text_primary"],
                button_color=COLORS["accent"], button_hover_color=COLORS["accent_hover"],
            ).pack(side="left", padx=4, pady=6)

            def _remove(rf=row_frame, nv=name_var):
                self._script_profile["characters"] = [
                    c for c in self._script_profile["characters"]
                    if c["name"] != nv.get()
                ]
                rf.destroy()
                self._script_char_rows = [r for r in self._script_char_rows if r["frame"].winfo_exists()]

            ctk.CTkButton(
                row_frame, text="✕", width=28, height=28,
                fg_color=COLORS["bg_dark"], hover_color="#3a1a1a",
                text_color=COLORS["text_secondary"], command=_remove,
            ).pack(side="right", padx=6, pady=6)

            entry = {"frame": row_frame, "name_var": name_var, "voice_var": voice_var}
            self._script_char_rows.append(entry)

        def _collect_profile():
            chars = []
            for row in self._script_char_rows:
                if not row["frame"].winfo_exists():
                    continue
                n = row["name_var"].get().strip()
                v = row["voice_var"].get()
                if n:
                    chars.append({"name": n, "voice": v, "blend_voice": "", "blend_ratio": 0.0})
            self._script_profile["characters"] = chars
            self._script_profile["narrator"]["voice"] = narrator_voice_var.get()

        def _cur_engine():
            return getattr(self.settings, "engine", "kokoro")

        def _save_profile():
            _collect_profile()
            name = self._script_profile_name.get().strip()
            if not name:
                from tkinter import simpledialog
                name = simpledialog.askstring("Save Profile", "Profile name:", parent=self.root)
                if not name:
                    return
                self._script_profile_name.set(name)
            save_profile(name, self._script_profile, _cur_engine())
            _refresh_profile_menu()
            status_var.set(t("SCRIPT_LAB_STATUS_PROFILE_SAVED", name=name))

        def _load_profile_by_name(name):
            if not name:
                return
            self._script_profile = load_profile(name, _cur_engine())
            self._script_profile_name.set(name)
            narrator_voice_var.set(self._script_profile.get("narrator", {}).get("voice", ""))
            _refresh_char_list()
            status_var.set(t("SCRIPT_LAB_STATUS_PROFILE_LOADED", name=name))

        def _refresh_profile_menu():
            profiles = list_profiles(_cur_engine())
            profile_menu.configure(values=profiles if profiles else [""])

        def _delete_profile():
            name = self._script_profile_name.get().strip()
            if not name:
                return
            delete_profile(name, _cur_engine())
            self._script_profile = default_profile()
            self._script_profile_name.set("")
            _refresh_char_list()
            _refresh_profile_menu()
            status_var.set(t("SCRIPT_LAB_STATUS_PROFILE_DELETED"))

        # ---- Layout ------------------------------------------------------
        outer = ctk.CTkFrame(tab, fg_color="transparent")
        outer.pack(fill="both", expand=True, padx=8, pady=8)
        outer.columnconfigure(0, weight=0, minsize=300)
        outer.columnconfigure(1, weight=1)
        outer.rowconfigure(0, weight=1)

        # == LEFT: Character Profiles ======================================
        left = ctk.CTkFrame(outer, fg_color=COLORS["bg_card"], corner_radius=10)
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 6))

        ctk.CTkLabel(
            left, text=t("SCRIPT_LAB_CHAR_PROFILES_HEADER"),
            font=(FONT_FAMILY, 14, "bold"), text_color=COLORS["text_primary"],
        ).pack(anchor="w", padx=12, pady=(12, 4))

        # Profile selector row
        prof_row = ctk.CTkFrame(left, fg_color="transparent")
        prof_row.pack(fill="x", padx=8, pady=(0, 6))

        def _new_profile():
            """Clear everything to start a fresh profile."""
            self._script_profile = default_profile()
            self._script_profile_name.set("")
            narrator_voice_var.set("")
            _refresh_char_list()
            status_var.set(t("SCRIPT_LAB_STATUS_NEW_PROFILE"))

        profiles_now = list_profiles(_cur_engine())
        profile_menu = ctk.CTkOptionMenu(
            prof_row,
            variable=self._script_profile_name,
            values=profiles_now if profiles_now else [""],
            width=150,
            fg_color=COLORS["bg_input"], text_color=COLORS["text_primary"],
            button_color=COLORS["accent"], button_hover_color=COLORS["accent_hover"],
            command=_load_profile_by_name,
        )
        profile_menu.pack(side="left", padx=(0, 4))

        ctk.CTkButton(
            prof_row, text=t("SCRIPT_LAB_BTN_NEW_PROFILE"), width=48, height=30,
            fg_color=COLORS["bg_input"], hover_color=COLORS["bg_card_hover"],
            text_color=COLORS["text_primary"],
            command=_new_profile,
        ).pack(side="left", padx=2)

        ctk.CTkButton(
            prof_row, text=t("COMMON_BTN_SAVE"), width=48, height=30,
            fg_color=COLORS["accent"], hover_color=COLORS["accent_hover"],
            command=_save_profile,
        ).pack(side="left", padx=2)

        ctk.CTkButton(
            prof_row, text=t("COMMON_BTN_DELETE"), width=52, height=30,
            fg_color="#5a2020", hover_color="#7a3030",
            command=_delete_profile,
        ).pack(side="left", padx=2)

        # Narrator voice
        ctk.CTkLabel(
            left, text=t("SCRIPT_LAB_NARRATOR_VOICE_LABEL"),
            font=(FONT_FAMILY, 12, "bold"), text_color=COLORS["text_secondary"],
        ).pack(anchor="w", padx=12, pady=(8, 2))

        narrator_voice_var = tk.StringVar(value="")
        narrator_voice_menu = ctk.CTkOptionMenu(
            left, variable=narrator_voice_var,
            values=_get_voice_options(), width=260,
            fg_color=COLORS["bg_input"], text_color=COLORS["text_primary"],
            button_color=COLORS["accent"], button_hover_color=COLORS["accent_hover"],
        )
        narrator_voice_menu.pack(anchor="w", padx=12, pady=(0, 8))

        def _refresh_voices_for_engine():
            """Refresh narrator + all character voice dropdowns to match the active engine."""
            voices = _get_voice_options()
            narrator_voice_menu.configure(values=voices)
            # Keep current selection if it's still valid, else clear
            if narrator_voice_var.get() not in voices:
                narrator_voice_var.set("")
            _refresh_char_list()
            _refresh_profile_menu()
            self._script_profile_name.set("")
            self._script_profile = default_profile()

        # Expose so _on_tab_changed can call it when engine switches
        self._script_refresh_voices = _refresh_voices_for_engine
        self._script_last_engine    = _cur_engine()

        # Characters header
        ctk.CTkLabel(
            left, text=t("SCRIPT_LAB_CHARACTERS_HEADER"),
            font=(FONT_FAMILY, 12, "bold"), text_color=COLORS["text_secondary"],
        ).pack(anchor="w", padx=12, pady=(4, 2))

        char_scroll = ctk.CTkScrollableFrame(
            left, fg_color="transparent",
            scrollbar_button_color=COLORS["accent"],
        )
        char_scroll.pack(fill="both", expand=True, padx=8, pady=(0, 4))

        def _find_characters():
            """Scan the script for [Name] tags and add any new ones to the profile."""
            # script_text and status_var are defined later in the outer scope;
            # Python closures are late-binding so they'll be available at call time.
            found = find_characters_in_script(script_text.get("1.0", "end"))
            existing = {r["name_var"].get().strip() for r in self._script_char_rows
                        if r["frame"].winfo_exists()}
            voices = _get_voice_options()
            added = 0
            for name in found:
                if name and name not in existing:
                    _add_char_row(name, "", voices)
                    added += 1
            if added:
                status_var.set(t("SCRIPT_LAB_STATUS_FOUND_CHARS", count=added))
            else:
                status_var.set(t("SCRIPT_LAB_STATUS_NO_NEW_CHARS"))

        add_row = ctk.CTkFrame(left, fg_color="transparent")
        add_row.pack(fill="x", padx=12, pady=(4, 12))

        ctk.CTkButton(
            add_row, text=t("SCRIPT_LAB_BTN_ADD_CHARACTER"), height=32,
            fg_color=COLORS["bg_input"], hover_color=COLORS["bg_card_hover"],
            text_color=COLORS["text_primary"],
            command=lambda: _add_char_row(),
        ).pack(side="left", fill="x", expand=True, padx=(0, 4))

        ctk.CTkButton(
            add_row, text=t("SCRIPT_LAB_BTN_FIND_IN_SCRIPT"), height=32,
            fg_color=COLORS["bg_input"], hover_color=COLORS["bg_card_hover"],
            text_color=COLORS["text_secondary"],
            command=_find_characters,
        ).pack(side="left", fill="x", expand=True)

        # == RIGHT: Script Editor ==========================================
        right = ctk.CTkFrame(outer, fg_color=COLORS["bg_card"], corner_radius=10)
        right.grid(row=0, column=1, sticky="nsew")

        # Toolbar
        toolbar = ctk.CTkFrame(right, fg_color="transparent")
        toolbar.pack(fill="x", padx=10, pady=(10, 4))

        ctk.CTkLabel(
            toolbar, text=t("SCRIPT_LAB_EDITOR_HEADER"),
            font=(FONT_FAMILY, 14, "bold"), text_color=COLORS["text_primary"],
        ).pack(side="left")

        # Toolbar buttons - right side
        btn_cfg = dict(height=30, corner_radius=6, font=(FONT_FAMILY, 12))

        def _export_audio():
            if not self._script_audio_chunks:
                status_var.set(t("SCRIPT_LAB_STATUS_NO_AUDIO_EXPORT"))
                return
            import numpy as np
            from tkinter import filedialog
            path = filedialog.asksaveasfilename(
                defaultextension=".wav",
                filetypes=[("WAV", "*.wav"), ("MP3", "*.mp3")],
                title=t("SCRIPT_LAB_DIALOG_EXPORT_AUDIO"),
            )
            if not path:
                return
            try:
                import soundfile as sf
                combined = np.concatenate(self._script_audio_chunks)
                sf.write(path, combined, self._script_audio_sr)
                status_var.set(t("SCRIPT_LAB_STATUS_AUDIO_EXPORTED", filename=os.path.basename(path)))
            except Exception as exc:
                status_var.set(t("SCRIPT_LAB_STATUS_EXPORT_FAILED", error=exc))

        def _export_script():
            from tkinter import filedialog
            text = script_text.get("1.0", "end").strip()
            if not text:
                status_var.set(t("SCRIPT_LAB_STATUS_SCRIPT_EMPTY"))
                return
            path = filedialog.asksaveasfilename(
                defaultextension=".txt",
                filetypes=[("Text", "*.txt"), ("All files", "*.*")],
                title=t("SCRIPT_LAB_DIALOG_EXPORT_SCRIPT"),
            )
            if not path:
                return
            try:
                with open(path, "w", encoding="utf-8") as f:
                    f.write(text)
                status_var.set(t("SCRIPT_LAB_STATUS_SCRIPT_EXPORTED", filename=os.path.basename(path)))
            except Exception as exc:
                status_var.set(t("SCRIPT_LAB_STATUS_EXPORT_FAILED", error=exc))

        def _load_script_file():
            from tkinter import filedialog
            path = filedialog.askopenfilename(
                filetypes=[("Text files", "*.txt"), ("All files", "*.*")],
                title=t("SCRIPT_LAB_DIALOG_LOAD_SCRIPT"),
            )
            if not path:
                return
            try:
                with open(path, "r", encoding="utf-8", errors="replace") as f:
                    script_text.delete("1.0", "end")
                    script_text.insert("1.0", f.read())
                status_var.set(t("SCRIPT_LAB_STATUS_SCRIPT_LOADED", filename=os.path.basename(path)))
            except Exception as exc:
                status_var.set(t("SCRIPT_LAB_STATUS_LOAD_FAILED", error=exc))

        # ── State extras beyond what __init__ sets up ─────────────────
        self._script_audio_sr      = 44100   # sample rate of generated audio
        self._script_combined      = None    # numpy array — full rendered audio
        self._script_pb_thread     = None    # playback thread
        self._script_pb_stop       = threading.Event()

        silent_var = tk.BooleanVar(value=False)  # False = play live; True = silent

        # ── Button references (created below, used in state helpers) ───
        _btn_refs = {}

        def _set_state(state):
            """Switch toolbar to idle / generating / ready / playing / paused."""
            self._script_state = state
            no_audio  = self._script_combined is None
            genning   = state == "generating"
            ready     = state in ("ready", "playing", "paused")
            playing   = state == "playing"

            _btn_refs["generate"].configure(
                state="disabled" if genning else "normal",
                fg_color=COLORS["accent"] if not genning else COLORS["bg_input"],
                hover_color=COLORS["accent_hover"] if not genning else COLORS["bg_input"],
            )
            # Cancel: show only while generating, hidden otherwise
            if genning:
                _btn_refs["cancel"].pack(side="right", padx=(3, 0))
            else:
                _btn_refs["cancel"].pack_forget()

            _btn_refs["play_pause"].configure(
                text=t("SCRIPT_LAB_BTN_PAUSE") if playing else t("SCRIPT_LAB_BTN_PLAY"),
                state="normal" if (ready and not genning) else "disabled",
                fg_color=COLORS["accent"] if ready else COLORS["bg_card"],
                text_color=COLORS["text_primary"] if ready else COLORS["text_secondary"],
            )
            _btn_refs["save_audio"].configure(
                state="normal" if (ready and not no_audio) else "disabled",
                fg_color=COLORS["bg_input"] if (ready and not no_audio) else COLORS["bg_card"],
                text_color=COLORS["text_primary"] if (ready and not no_audio) else COLORS["text_secondary"],
            )

        def _cancel_generation():
            self._script_stop_flag = True
            self._script_pb_stop.set()
            status_var.set(t("SCRIPT_LAB_STATUS_CANCELLED"))
            self.root.after(200, lambda: _set_state("idle"))

        def _do_playback():
            """Play the already-generated combined audio in a background thread."""
            if self._script_combined is None:
                return
            import sounddevice as sd
            self._script_pb_stop.clear()
            _set_state("playing")

            def _worker():
                try:
                    vol = script_vol_var.get() / 100.0
                    sd.play(self._script_combined * vol, self._script_audio_sr)
                    # Poll until done or stopped
                    while sd.get_stream().active:
                        if self._script_pb_stop.is_set():
                            sd.stop()
                            break
                        threading.Event().wait(0.05)
                except Exception as exc:
                    logger.warning("Script playback error: %s", exc)
                finally:
                    self.root.after(0, lambda: _set_state("ready"))

            self._script_pb_thread = threading.Thread(target=_worker, daemon=True)
            self._script_pb_thread.start()

        def _toggle_play_pause():
            if self._script_state == "playing":
                # Pause
                self._script_pb_stop.set()
                import sounddevice as sd
                sd.stop()
                _set_state("paused")
            elif self._script_state == "paused":
                # Resume — restart from beginning (full replay)
                _do_playback()
            else:
                # First play
                _do_playback()

        def _generate_script():
            if self._script_state == "generating":
                return
            _collect_profile()
            text = script_text.get("1.0", "end").strip()
            if not text:
                status_var.set(t("SCRIPT_LAB_STATUS_SCRIPT_EMPTY"))
                return
            segments = parse_script(text)
            if not segments:
                status_var.set(t("SCRIPT_LAB_STATUS_NO_SEGMENTS"))
                return

            self._script_stop_flag = False
            self._script_audio_chunks = []
            self._script_combined = None
            _set_state("generating")
            status_var.set(t("SCRIPT_LAB_STATUS_GENERATING"))
            live = not silent_var.get()

            def _worker():
                import numpy as np
                import queue as _q
                engine = getattr(self.settings, "engine", "kokoro")
                sr = 44100

                def _synth_blocking(text, voice_name):
                    """
                    Wrap self.tts.generate() (callback-based) into a blocking
                    generator that yields (audio_np, sample_rate) tuples.
                    Works for both KokoroEngine and TTSEngine.
                    """
                    chunk_q  = _q.Queue()
                    done_evt = threading.Event()
                    error_holder = [None]

                    def on_chunk(audio, chunk_sr):
                        chunk_q.put((audio, chunk_sr))

                    def on_complete(_path=None):
                        done_evt.set()
                        # Surface GPU/CPU provider in TTS status after first load
                        if engine == "kokoro":
                            _prov = getattr(self.tts, "provider", "")
                            if _prov == "cuda":
                                _ready_msg = t("SPEECH_LAB_STATUS_ENGINE_READY_GPU")
                            elif _prov == "cpu":
                                _ready_msg = t("SPEECH_LAB_STATUS_ENGINE_READY_CPU")
                            else:
                                _ready_msg = t("SPEECH_LAB_STATUS_ENGINE_READY")
                            self.root.after(0, lambda msg=_ready_msg: self.update_tts_status(
                                msg, COLORS["success"]
                            ))

                    def on_error(exc):
                        error_holder[0] = exc
                        done_evt.set()

                    if engine == "kokoro":
                        from kokoro_engine import KOKORO_VOICES, DEFAULT_VOICE
                        vid = KOKORO_VOICES.get(voice_name, DEFAULT_VOICE) if voice_name else DEFAULT_VOICE

                        def _kokoro_progress(msg, _frac):
                            # Only surface loading messages, not per-sentence counters
                            if "load" in msg.lower() or "ready" in msg.lower():
                                self.root.after(0, lambda m=msg: status_var.set(m))

                        self.tts.generate(
                            text, voice_id=vid,
                            on_chunk=on_chunk, on_complete=on_complete, on_error=on_error,
                            on_progress=_kokoro_progress,
                        )
                    else:
                        if not self.tts.is_loaded:
                            raise RuntimeError(
                                "TTS engine not loaded — load it in Speech Lab first."
                            )
                        voice_info = self.voices.get_voice(voice_name) if voice_name else None
                        ref_wav    = voice_info["wav_path"] if voice_info else None
                        ref_tokens = None
                        if voice_info and voice_info.get("tokens_path") and \
                                os.path.isfile(voice_info["tokens_path"]):
                            ref_tokens = np.load(voice_info["tokens_path"])
                        self.tts.generate(
                            text,
                            reference_wav=ref_wav,
                            reference_tokens=ref_tokens,
                            prompt_text=voice_info.get("prompt_text", "") if voice_info else "",
                            on_chunk=on_chunk, on_complete=on_complete, on_error=on_error,
                        )

                    # Drain the queue until generation finishes
                    while not done_evt.is_set() or not chunk_q.empty():
                        if self._script_stop_flag:
                            self.tts.cancel()
                            return
                        try:
                            yield chunk_q.get(timeout=0.05)
                        except _q.Empty:
                            continue

                    if error_holder[0]:
                        raise error_holder[0]

                # Target RMS for loudness normalisation (~-18 dBFS)
                _TARGET_RMS = 0.08

                def _rms_normalize(audio):
                    """Scale audio so its RMS matches _TARGET_RMS. Skips silence."""
                    rms = np.sqrt(np.mean(audio.astype(np.float32) ** 2))
                    if rms < 1e-6:
                        return audio
                    return (audio.astype(np.float32) * (_TARGET_RMS / rms)).clip(-1.0, 1.0)

                try:
                    for i, seg in enumerate(segments):
                        if self._script_stop_flag:
                            break
                        char  = seg["character"]
                        stext = seg["text"]
                        if not stext.strip():
                            continue
                        voice = _voice_for_character(char)
                        self.root.after(0, lambda c=char, i=i, n=len(segments): status_var.set(
                            t("SCRIPT_LAB_STATUS_GENERATING_SEGMENT", index=i+1, total=n, character=c)
                        ))
                        try:
                            for audio, chunk_sr in _synth_blocking(stext, voice):
                                if self._script_stop_flag:
                                    break
                                sr = chunk_sr
                                self._script_audio_sr = chunk_sr
                                audio = _rms_normalize(audio)
                                self._script_audio_chunks.append(audio)
                                if live:
                                    import sounddevice as sd
                                    vol = script_vol_var.get() / 100.0
                                    sd.play(audio * vol, sr)
                                    while sd.get_stream().active:
                                        if self._script_stop_flag:
                                            sd.stop()
                                            break
                                        threading.Event().wait(0.02)
                        except Exception as exc:
                            logger.warning("Script segment [%s] failed: %s", char, exc)
                            self.root.after(0, lambda e=str(exc): status_var.set(t("SCRIPT_LAB_STATUS_SEGMENT_ERROR", error=e)))

                    if self._script_audio_chunks and not self._script_stop_flag:
                        self._script_combined = np.concatenate(self._script_audio_chunks)
                        self._script_audio_sr = sr
                        self.root.after(0, lambda: status_var.set(
                            t("SCRIPT_LAB_STATUS_DONE")
                        ))
                        self.root.after(0, lambda: _set_state("ready"))
                    else:
                        self.root.after(0, lambda: status_var.set(t("SCRIPT_LAB_STATUS_CANCELLED") if self._script_stop_flag else t("SCRIPT_LAB_STATUS_NO_AUDIO")))
                        self.root.after(0, lambda: _set_state("idle"))
                except Exception as exc:
                    logger.error("Script generation failed: %s", exc)
                    self.root.after(0, lambda e=str(exc): status_var.set(t("SCRIPT_LAB_STATUS_GENERATION_FAILED", error=e)))
                    self.root.after(0, lambda: _set_state("idle"))

            threading.Thread(target=_worker, daemon=True).start()

        # ── Toolbar: save buttons (top right) + vol/silent (top left) ─
        ctk.CTkButton(
            toolbar, text=t("SCRIPT_LAB_BTN_SAVE_SCRIPT"), width=90,
            fg_color=COLORS["bg_input"], hover_color=COLORS["bg_card_hover"],
            text_color=COLORS["text_primary"], command=_export_script, **btn_cfg,
        ).pack(side="right", padx=3)

        _btn_refs["save_audio"] = ctk.CTkButton(
            toolbar, text=t("SCRIPT_LAB_BTN_SAVE_AUDIO"), width=85,
            fg_color=COLORS["bg_card"], hover_color=COLORS["bg_card_hover"],
            text_color=COLORS["text_secondary"], command=_export_audio,
            state="disabled", **btn_cfg,
        )
        _btn_refs["save_audio"].pack(side="right", padx=3)

        # Volume slider
        script_vol_var = tk.IntVar(value=self.settings.volume)
        ctk.CTkLabel(
            toolbar, text=t("SCRIPT_LAB_VOL_LABEL"),
            font=(FONT_FAMILY, 11), text_color=COLORS["text_secondary"],
        ).pack(side="left", padx=(10, 2))
        ctk.CTkSlider(
            toolbar, from_=0, to=100, number_of_steps=100,
            variable=script_vol_var, width=90,
            progress_color=COLORS["accent"], button_color=COLORS["accent"],
        ).pack(side="left", padx=(0, 6))

        ctk.CTkSwitch(
            toolbar, text=t("SCRIPT_LAB_SILENT_LABEL"), variable=silent_var,
            onvalue=True, offvalue=False,
            font=(FONT_FAMILY, 11), text_color=COLORS["text_secondary"],
            progress_color=COLORS["accent"],
        ).pack(side="left", padx=(6, 2))

        # AI Tag section
        ai_bar = ctk.CTkFrame(right, fg_color=COLORS["bg_input"], corner_radius=8)
        ai_bar.pack(fill="x", padx=10, pady=(0, 6))

        ctk.CTkLabel(
            ai_bar, text=t("SCRIPT_LAB_AI_TAG_LABEL"),
            font=(FONT_FAMILY, 12), text_color=COLORS["text_secondary"],
        ).pack(side="left", padx=(10, 6), pady=6)

        ai_file_var = tk.StringVar(value=t("SCRIPT_LAB_AI_DROP_ZONE"))
        ctk.CTkLabel(
            ai_bar, textvariable=ai_file_var,
            font=(FONT_FAMILY, 11), text_color=COLORS["text_secondary"],
            width=300, anchor="w",
        ).pack(side="left", padx=4, pady=6)

        self._ai_source_path = None

        def _browse_source():
            from tkinter import filedialog
            path = filedialog.askopenfilename(
                filetypes=[
                    ("Supported files", "*.txt *.pdf *.docx *.epub"),
                    ("All files", "*.*"),
                ],
                title="Select source document",
            )
            if path:
                self._ai_source_path = path
                ai_file_var.set(os.path.basename(path))

        def _run_ai_tag():
            if not self._ai_source_path:
                status_var.set(t("SCRIPT_LAB_STATUS_BROWSE_FIRST"))
                return
            _collect_profile()
            char_names = [c["name"] for c in self._script_profile.get("characters", [])]
            engine = getattr(self.settings, "engine", "kokoro")
            status_var.set(t("SCRIPT_LAB_STATUS_READING_FILE"))

            def _worker():
                try:
                    source = read_source_file(self._ai_source_path)
                    def _prog(msg, _frac):
                        self.root.after(0, lambda m=msg: status_var.set(m))
                    tagged = tag_script_with_ai(source, engine, char_names, on_progress=_prog)
                    def _apply():
                        script_text.delete("1.0", "end")
                        script_text.insert("1.0", tagged)
                        status_var.set(t("SCRIPT_LAB_STATUS_AI_TAG_COMPLETE"))
                    self.root.after(0, _apply)
                except Exception as exc:
                    self.root.after(0, lambda e=exc: status_var.set(t("SCRIPT_LAB_STATUS_AI_TAG_FAILED", error=e)))

            threading.Thread(target=_worker, daemon=True).start()

        ctk.CTkButton(
            ai_bar, text=t("COMMON_BTN_BROWSE"), width=70, height=28,
            fg_color=COLORS["bg_dark"], hover_color=COLORS["bg_card_hover"],
            text_color=COLORS["text_primary"], command=_browse_source,
        ).pack(side="left", padx=4, pady=6)

        ctk.CTkButton(
            ai_bar, text=t("SCRIPT_LAB_BTN_GENERATE_SCRIPT"), width=120, height=28,
            fg_color=COLORS["accent"], hover_color=COLORS["accent_hover"],
            command=_run_ai_tag,
        ).pack(side="left", padx=4, pady=6)

        def _run_enhance():
            text = script_text.get("1.0", "end").strip()
            if not text:
                status_var.set(t("SCRIPT_LAB_STATUS_EMPTY_ENHANCE"))
                return
            engine = getattr(self.settings, "engine", "kokoro")
            status_var.set(t("SCRIPT_LAB_STATUS_ENHANCING"))

            def _worker():
                try:
                    def _prog(msg, _frac):
                        self.root.after(0, lambda m=msg: status_var.set(m))
                    enhanced = enhance_script_flow(text, engine, on_progress=_prog)
                    def _apply():
                        script_text.delete("1.0", "end")
                        script_text.insert("1.0", enhanced)
                        status_var.set(t("SCRIPT_LAB_STATUS_ENHANCE_COMPLETE"))
                    self.root.after(0, _apply)
                except Exception as exc:
                    self.root.after(0, lambda e=exc: status_var.set(t("SCRIPT_LAB_STATUS_ENHANCE_FAILED", error=e)))

            threading.Thread(target=_worker, daemon=True).start()

        ctk.CTkButton(
            ai_bar, text=t("SCRIPT_LAB_BTN_ENHANCE_SCRIPT"), width=140, height=28,
            fg_color=COLORS["bg_dark"], hover_color=COLORS["bg_card_hover"],
            text_color=COLORS["text_primary"], font=(FONT_FAMILY, 12),
            command=_run_enhance,
        ).pack(side="left", padx=4, pady=6)

        #ctk.CTkLabel(
          #  ai_bar,
           # text=t("SCRIPT_LAB_AI_TAG_INFO"),
           # font=(FONT_FAMILY, 10), text_color=COLORS["text_secondary"],
        #).pack(side="right", padx=10, pady=6)

        # Script text area
        script_text = ctk.CTkTextbox(
            right,
            fg_color=COLORS["bg_input"],
            text_color=COLORS["text_primary"],
            font=(FONT_FAMILY, 13),
            wrap="word",
            corner_radius=8,
            scrollbar_button_color=COLORS["accent"],
        )
        script_text.pack(fill="both", expand=True, padx=10, pady=(0, 6))

        # Status bar
        status_var = tk.StringVar(value=t("SCRIPT_LAB_STATUS_READY"))

        # Drag-and-drop into the script editor
        def _on_script_drop(event):
            raw = event.data.strip()
            # tkinterdnd2 wraps paths with spaces in braces: {C:/my path/file.txt}
            paths = []
            if raw.startswith("{"):
                import re as _re
                paths = _re.findall(r'\{([^}]+)\}', raw)
            if not paths:
                paths = raw.split()
            path = paths[0] if paths else ""
            if not path or not os.path.isfile(path):
                status_var.set(t("SCRIPT_LAB_STATUS_DROP_HINT"))
                return
            ext = os.path.splitext(path)[1].lower()
            if ext == ".txt":
                # Load as plain text — may already be a tagged script or raw prose
                try:
                    with open(path, "r", encoding="utf-8", errors="replace") as f:
                        content = f.read()
                    script_text.delete("1.0", "end")
                    script_text.insert("1.0", content)
                    # Register as source so "Generate Script" can AI-tag it
                    self._ai_source_path = path
                    ai_file_var.set(os.path.basename(path))
                    status_var.set(t("SCRIPT_LAB_STATUS_SCRIPT_LOADED", filename=os.path.basename(path)))
                except Exception as exc:
                    status_var.set(t("SCRIPT_LAB_STATUS_LOAD_FAILED", error=exc))
            elif ext in (".pdf", ".docx", ".epub"):
                # Extract prose — user will then run AI tagging
                status_var.set(t("SCRIPT_LAB_STATUS_READING_FILE"))
                def _load(p=path):
                    try:
                        content = read_source_file(p)
                        def _apply():
                            script_text.delete("1.0", "end")
                            script_text.insert("1.0", content)
                            self._ai_source_path = p
                            ai_file_var.set(os.path.basename(p))
                            status_var.set(t("SCRIPT_LAB_STATUS_LOADED_PROSE", filename=os.path.basename(p)))
                        self.root.after(0, _apply)
                    except Exception as exc:
                        self.root.after(0, lambda e=exc: status_var.set(t("SCRIPT_LAB_STATUS_LOAD_FAILED", error=e)))
                threading.Thread(target=_load, daemon=True).start()
            else:
                status_var.set(t("SCRIPT_LAB_STATUS_UNSUPPORTED_FILE"))

        try:
            from tkinterdnd2 import DND_FILES
            script_text.drop_target_register(DND_FILES)
            script_text.dnd_bind("<<Drop>>", _on_script_drop)
        except Exception as _dnd_err:
            logger.warning("Script Lab drag-and-drop unavailable: %s", _dnd_err)

        # ── Bottom action bar (Generate / Cancel / Play) ───────────────
        bottom_bar = ctk.CTkFrame(right, fg_color="transparent")
        bottom_bar.pack(fill="x", padx=10, pady=(4, 2))

        _btn_refs["play_pause"] = ctk.CTkButton(
            bottom_bar, text=t("SCRIPT_LAB_BTN_PLAY"), width=80,
            fg_color=COLORS["bg_card"], hover_color=COLORS["accent_hover"],
            text_color=COLORS["text_secondary"], command=_toggle_play_pause,
            state="disabled", **btn_cfg,
        )
        _btn_refs["play_pause"].pack(side="right", padx=(3, 0))

        # Cancel lives in the bottom bar but starts hidden; _set_state shows/hides it
        _btn_refs["cancel"] = ctk.CTkButton(
            bottom_bar, text=t("COMMON_BTN_CANCEL"), width=80,
            fg_color="#5a2020", hover_color="#7a3030",
            text_color=COLORS["text_primary"], command=_cancel_generation,
            **btn_cfg,
        )
        # (not packed yet — _set_state("idle") will leave it hidden)

        _btn_refs["generate"] = ctk.CTkButton(
            bottom_bar, text=t("SCRIPT_LAB_BTN_GENERATE"), width=100,
            fg_color=COLORS["accent"], hover_color=COLORS["accent_hover"],
            command=_generate_script, **btn_cfg,
        )
        _btn_refs["generate"].pack(side="right", padx=3)

        # Initialise button states
        _set_state("idle")

        ctk.CTkLabel(
            right, textvariable=status_var,
            font=(FONT_FAMILY, 11), text_color=COLORS["text_secondary"],
            anchor="w",
        ).pack(fill="x", padx=12, pady=(0, 8))

    # ==================================================================
    # TAB: Prompt Lab
    # ==================================================================

    def _build_prompt_lab_tab(self):
        """Chat interface for direct conversation with the local LLM."""
        from tag_suggester import PROMPT_LAB_PRESETS
        tab = self.tab_chat

        # Chat state
        self._chat_history: list = []
        self._chat_busy = False

        outer = ctk.CTkFrame(tab, fg_color="transparent")
        outer.pack(fill="both", expand=True, padx=12, pady=10)

        # ── Top bar ──────────────────────────────────────────────────────
        top_bar = ctk.CTkFrame(outer, fg_color="transparent")
        top_bar.pack(fill="x", pady=(0, 8))

        ctk.CTkLabel(
            top_bar, text=t("PROMPT_LAB_HEADER"),
            font=(FONT_FAMILY, 16, "bold"),
            text_color=COLORS["accent_light"],
        ).pack(side="left")

        self._chat_model_badge = ctk.CTkLabel(
            top_bar, text="",
            font=(FONT_FAMILY, 10),
            text_color=COLORS["text_muted"],
        )
        self._chat_model_badge.pack(side="left", padx=(10, 0))
        self._chat_refresh_model_badge()

        ctk.CTkButton(
            top_bar, text=t("PROMPT_LAB_BTN_CLEAR_CHAT"),
            width=110, height=28, corner_radius=6,
            fg_color=COLORS["bg_input"],
            hover_color=COLORS["bg_card_hover"],
            border_color=COLORS["border"], border_width=1,
            font=(FONT_FAMILY, 11),
            command=self._chat_clear,
        ).pack(side="right")

        # ── Context depth control ─────────────────────────────────────────
        # How many recent messages (pairs) to include in each request.
        # Small models overflow quickly — 10 pairs = 20 messages is a safe cap.
        self._chat_ctx_var = ctk.IntVar(value=10)   # pairs → × 2 for user+asst

        _ctx_right = ctk.CTkFrame(top_bar, fg_color="transparent")
        _ctx_right.pack(side="right", padx=(0, 12))

        ctk.CTkLabel(
            _ctx_right, text=t("PROMPT_LAB_CONTEXT_LABEL"),
            font=(FONT_FAMILY, 10),
            text_color=COLORS["text_muted"],
        ).pack(side="left", padx=(0, 4))

        self._chat_ctx_lbl = ctk.CTkLabel(
            _ctx_right,
            text="10 turns",
            font=(FONT_FAMILY, 10),
            text_color=COLORS["text_secondary"],
            width=52,
        )
        self._chat_ctx_lbl.pack(side="right", padx=(4, 0))

        def _on_ctx_slide(v):
            n = int(float(v))
            self._chat_ctx_var.set(n)
            self._chat_ctx_lbl.configure(
                text="all" if n == 0 else f"{n} turns"
            )

        _ctx_slider = ctk.CTkSlider(
            _ctx_right,
            from_=0, to=25, number_of_steps=25,
            variable=self._chat_ctx_var,
            width=110, height=14,
            command=_on_ctx_slide,
            button_color=COLORS["accent"],
            progress_color=COLORS["accent"],
        )
        _ctx_slider.pack(side="left")
        self._make_tooltip(_ctx_slider, t("PROMPT_LAB_TOOLTIP_CTX"))

        # ── Persona / system-prompt bar ──────────────────────────────────
        preset_bar = ctk.CTkFrame(outer, fg_color=COLORS["bg_card"], corner_radius=8)
        preset_bar.pack(fill="x", pady=(0, 8))

        ctk.CTkLabel(
            preset_bar, text=t("PROMPT_LAB_PERSONA_LABEL"),
            font=(FONT_FAMILY, 11),
            text_color=COLORS["text_secondary"],
        ).pack(side="left", padx=(10, 6), pady=6)

        _preset_display = _display_names(_PRESET_KEY_MAP)
        self._chat_preset_var = ctk.StringVar(value=_preset_display[0])

        def _on_preset_change(v):
            en_name = _en_from_display(_PRESET_KEY_MAP, v)
            self._chat_system_box.configure(state="normal")
            self._chat_system_box.delete("1.0", "end")
            preset_text = PROMPT_LAB_PRESETS.get(en_name, "")
            self._chat_system_box.insert("1.0", preset_text)
            if en_name != "Custom":
                self._chat_system_box.configure(state="disabled")

        ctk.CTkOptionMenu(
            preset_bar,
            variable=self._chat_preset_var,
            values=_preset_display,
            width=180, height=26,
            fg_color=COLORS["bg_input"],
            button_color=COLORS["accent"],
            button_hover_color=COLORS["accent_hover"],
            dropdown_fg_color=COLORS["bg_card"],
            dropdown_hover_color=COLORS["bg_card_hover"],
            font=(FONT_FAMILY, 11),
            command=_on_preset_change,
        ).pack(side="left", padx=(0, 8), pady=6)

        ctk.CTkLabel(
            preset_bar, text=t("PROMPT_LAB_SYSTEM_PROMPT_LABEL"),
            font=(FONT_FAMILY, 10),
            text_color=COLORS["text_muted"],
        ).pack(side="left", padx=(0, 4))

        _first_preset_en = _en_from_display(_PRESET_KEY_MAP, _preset_display[0])
        self._chat_system_box = ctk.CTkTextbox(
            preset_bar,
            height=28, wrap="none",
            fg_color=COLORS["bg_input"],
            text_color=COLORS["text_secondary"],
            font=(FONT_FAMILY, 10),
        )
        self._chat_system_box.pack(side="left", fill="x", expand=True, padx=(0, 10), pady=4)
        self._chat_system_box.insert("1.0", PROMPT_LAB_PRESETS[_first_preset_en])
        self._chat_system_box.configure(state="disabled")  # editable only on Custom

        # ── Chat display ─────────────────────────────────────────────────
        self._chat_display = ctk.CTkTextbox(
            outer,
            fg_color=COLORS["bg_card"],
            text_color=COLORS["text_primary"],
            font=(FONT_FAMILY, 13),
            wrap="word",
            state="disabled",
            corner_radius=8,
        )
        self._chat_display.pack(fill="both", expand=True, pady=(0, 8))

        # CTkTextbox wraps tk.Text — use ._textbox directly for tag operations
        _tb = self._chat_display._textbox
        _tb.tag_config("user_label",
            foreground=COLORS["accent_light"],
            font=(FONT_FAMILY, 12, "bold"),
        )
        _tb.tag_config("user_text",
            foreground="#c8d6f7",
            font=(FONT_FAMILY, 13),
        )
        _tb.tag_config("asst_label",
            foreground=COLORS["success"],
            font=(FONT_FAMILY, 12, "bold"),
        )
        _tb.tag_config("asst_text",
            foreground=COLORS["text_primary"],
            font=(FONT_FAMILY, 13),
        )
        _tb.tag_config("muted",
            foreground=COLORS["text_muted"],
            font=(FONT_FAMILY, 10),
        )

        # Build a welcome message that identifies the loaded model
        try:
            from tag_suggester import get_active_model_cfg, get_active_llm_key, is_llm_available, is_qwen_model_ready
            if not is_llm_available():
                _welcome = t("PROMPT_LAB_WELCOME_NO_LLM")
            elif not is_qwen_model_ready():
                _welcome = t("PROMPT_LAB_WELCOME_NO_MODEL")
            else:
                _cfg = get_active_model_cfg()
                _welcome = t("PROMPT_LAB_WELCOME_READY", model=get_active_llm_key(), filename=_cfg.get('filename', ''))
        except Exception:
            _welcome = t("PROMPT_LAB_WELCOME_FALLBACK")

        self._chat_append("muted", _welcome)

        # ── Input area ───────────────────────────────────────────────────
        self._chat_attachment = None   # {"type": "text"|"image", "path": str, "content": str|bytes}

        input_frame = ctk.CTkFrame(outer, fg_color=COLORS["bg_card"], corner_radius=8)
        input_frame.pack(fill="x")

        # Attachment bar (hidden until a file is attached)
        self._chat_attach_bar = ctk.CTkFrame(input_frame, fg_color=COLORS["bg_input"], corner_radius=6)
        # Packed dynamically in _chat_set_attachment()

        _attach_icon = ctk.CTkLabel(
            self._chat_attach_bar, text="📎",
            font=(FONT_FAMILY, 13),
        )
        _attach_icon.pack(side="left", padx=(8, 4), pady=4)

        self._chat_attach_name = ctk.CTkLabel(
            self._chat_attach_bar, text="",
            font=(FONT_FAMILY, 11),
            text_color=COLORS["text_secondary"],
        )
        self._chat_attach_name.pack(side="left", padx=(0, 8), pady=4)

        ctk.CTkButton(
            self._chat_attach_bar, text="✕ Remove",
            width=70, height=22, corner_radius=4,
            fg_color=COLORS["bg_card"], hover_color=COLORS["danger"],
            font=(FONT_FAMILY, 10),
            command=self._chat_clear_attachment,
        ).pack(side="right", padx=8, pady=4)

        # Text input + button column
        input_row = ctk.CTkFrame(input_frame, fg_color="transparent")
        input_row.pack(fill="x")

        self._chat_input = ctk.CTkTextbox(
            input_row,
            height=80,
            fg_color=COLORS["bg_input"],
            text_color=COLORS["text_primary"],
            font=(FONT_FAMILY, 13),
            wrap="word",
        )
        self._chat_input.pack(side="left", fill="both", expand=True, padx=(8, 6), pady=8)
        self._chat_input.bind("<Control-Return>", lambda e: self._chat_send())

        # Enable drag-and-drop if tkinterdnd2 is available
        try:
            from tkinterdnd2 import DND_FILES
            self._chat_input.drop_target_register(DND_FILES)
            self._chat_input.dnd_bind("<<Drop>>", self._chat_on_drop)
            self._chat_display.drop_target_register(DND_FILES)
            self._chat_display.dnd_bind("<<Drop>>", self._chat_on_drop)
        except Exception:
            pass  # tkinterdnd2 not installed — use the attach button instead

        btn_col = ctk.CTkFrame(input_row, fg_color="transparent")
        btn_col.pack(side="right", padx=(0, 8), pady=8)

        self._chat_send_btn = ctk.CTkButton(
            btn_col,
            text=t("PROMPT_LAB_BTN_SEND"),
            width=100, height=36, corner_radius=8,
            fg_color=COLORS["accent"],
            hover_color=COLORS["accent_hover"],
            font=(FONT_FAMILY, 13, "bold"),
            command=self._chat_send,
        )
        self._chat_send_btn.pack()

        ctk.CTkButton(
            btn_col,
            text=t("PROMPT_LAB_BTN_ATTACH"),
            width=100, height=28, corner_radius=6,
            fg_color=COLORS["bg_input"],
            hover_color=COLORS["bg_card_hover"],
            border_color=COLORS["border"], border_width=1,
            font=(FONT_FAMILY, 11),
            command=self._chat_attach_file,
        ).pack(pady=(6, 0))

        self._chat_status = ctk.CTkLabel(
            btn_col, text="",
            font=(FONT_FAMILY, 9),
            text_color=COLORS["text_muted"],
            wraplength=100,
        )
        self._chat_status.pack(pady=(4, 0))

        self._make_tooltip(self._chat_send_btn, t("PROMPT_LAB_TOOLTIP_SEND"))

    # ── Attachment helpers ────────────────────────────────────────────────

    _CHAT_TEXT_EXTS = {".txt", ".md", ".csv", ".log", ".rst", ".json", ".xml", ".html"}

    def _chat_attach_file(self):
        """Open a file picker and attach a text file to the next message."""
        path = filedialog.askopenfilename(
            title=t("PROMPT_LAB_DIALOG_ATTACH"),
            filetypes=[
                ("Text files", "*.txt *.md *.csv *.log *.rst *.json *.xml *.html"),
                ("All files",  "*.*"),
            ],
        )
        if path:
            self._chat_load_attachment(path)

    def _chat_on_drop(self, event):
        """Handle files dropped onto the chat input or display area."""
        raw = event.data.strip()
        # tkinterdnd2 wraps multi-word paths in braces
        if raw.startswith("{") and raw.endswith("}"):
            raw = raw[1:-1]
        # Take the first file if multiple were dropped
        path = raw.split("} {")[0].strip("{}")
        if os.path.isfile(path):
            self._chat_load_attachment(path)

    def _chat_load_attachment(self, path: str):
        """Read the file at *path* and store it as the pending attachment."""
        ext = os.path.splitext(path)[1].lower()
        name = os.path.basename(path)

        if ext in self._CHAT_TEXT_EXTS:
            try:
                content = open(path, encoding="utf-8", errors="replace").read()
            except Exception as exc:
                messagebox.showerror(t("COMMON_ERROR"), t("SPEECH_LAB_MSG_FILE_ERROR_BODY", error=exc))
                return
            size_kb = len(content) / 1024
            # Warn if file is very large — small models have limited context
            if len(content) > 6000:
                if not messagebox.askyesno(
                    t("PROMPT_LAB_MSG_LARGE_FILE_TITLE"),
                    t("PROMPT_LAB_MSG_LARGE_FILE_BODY", name=name, size_kb=f"{size_kb:.0f}", chars=f"{len(content):,}"),
                    parent=self.root,
                ):
                    return
            self._chat_set_attachment({
                "type": "text", "path": path, "content": content,
            })
        else:
            messagebox.showinfo(
                t("PROMPT_LAB_MSG_UNSUPPORTED_FILE_TITLE"),
                t("PROMPT_LAB_MSG_UNSUPPORTED_FILE_BODY", ext=ext),
                parent=self.root,
            )

    def _chat_set_attachment(self, att: dict):
        """Store attachment and update the attachment bar."""
        self._chat_attachment = att
        name = os.path.basename(att["path"])
        size = f"{len(att['content']):,} chars"
        self._chat_attach_name.configure(text=f"📄 {name}  ({size})")
        self._chat_attach_bar.pack(fill="x", padx=8, pady=(6, 0), before=self._chat_input.master)

    def _chat_clear_attachment(self):
        """Remove the pending attachment."""
        self._chat_attachment = None
        self._chat_attach_bar.pack_forget()
        self._chat_attach_name.configure(text="")

    def _chat_refresh_model_badge(self):
        """Update the model-name badge shown next to the Prompt Lab title."""
        try:
            from tag_suggester import get_active_model_cfg, get_active_llm_key, is_llm_available, is_qwen_model_ready
            if not is_llm_available():
                self._chat_model_badge.configure(
                    text=t("PROMPT_LAB_BADGE_NO_LLM"), text_color=COLORS["danger"]
                )
            elif not is_qwen_model_ready():
                self._chat_model_badge.configure(
                    text=t("PROMPT_LAB_BADGE_NO_MODEL"), text_color=COLORS["warning"]
                )
            else:
                _active_key = get_active_llm_key()
                self._chat_model_badge.configure(
                    text=t("PROMPT_LAB_MODEL_BADGE", model=_active_key),
                    text_color=COLORS["success"],
                )
        except Exception:
            pass

    def _chat_append(self, tag: str, text: str):
        """Append styled text to the chat display (call from main thread)."""
        self._chat_display.configure(state="normal")
        # Use underlying tk.Text directly for reliable tag support
        self._chat_display._textbox.insert("end", text, tag)
        self._chat_display._textbox.see("end")
        self._chat_display.configure(state="disabled")

    def _chat_send(self):
        """Send the user message (with any attachment) and stream the LLM response."""
        if self._chat_busy:
            return
        user_text = self._chat_input.get("1.0", "end").strip()
        att = self._chat_attachment

        if not user_text and att is None:
            return

        from tag_suggester import is_llm_available, is_qwen_model_ready, chat_with_llm

        if not is_llm_available():
            messagebox.showerror(
                t("MAIN_TAB_PROMPT_LAB"),
                t("PROMPT_LAB_MSG_NO_LLM"),
            )
            return
        if not is_qwen_model_ready():
            messagebox.showerror(
                t("MAIN_TAB_PROMPT_LAB"),
                t("PROMPT_LAB_MSG_NO_MODEL"),
            )
            return

        # Build the actual message sent to the LLM
        # Text attachment: prepend file content so the model can reason about it
        display_text = user_text or "(see attached file)"
        if att and att["type"] == "text":
            att_name = os.path.basename(att["path"])
            # Truncate very large files to the first ~4000 chars so small models don't choke
            snippet = att["content"]
            truncated = len(snippet) > 4000
            if truncated:
                snippet = snippet[:4000]
            file_block = (
                f"[Attached file: {att_name}]\n"
                f"{'(truncated to first 4000 chars) ' if truncated else ''}\n"
                f"{snippet}\n\n"
            )
            llm_text = file_block + (user_text if user_text else "Summarise or describe this file.")
        else:
            llm_text = user_text

        # Read system prompt
        self._chat_system_box.configure(state="normal")
        system_prompt = self._chat_system_box.get("1.0", "end").strip()
        _preset = self._chat_preset_var.get()
        if _preset != "Custom":
            self._chat_system_box.configure(state="disabled")

        # Clear input + attachment, lock UI
        self._chat_input.delete("1.0", "end")
        self._chat_clear_attachment()
        self._chat_busy = True
        self._chat_send_btn.configure(state="disabled", text="…")
        self._chat_status.configure(text=t("PROMPT_LAB_STATUS_THINKING"))

        # Show user bubble (display only the typed text, not the whole file dump)
        self._chat_append("user_label", t("PROMPT_LAB_LABEL_YOU") + "\n")
        if att:
            att_note = f"[📎 {os.path.basename(att['path'])}]  " if att else ""
            self._chat_append("muted", att_note)
        self._chat_append("user_text", f"{display_text}\n\n")

        # Record in history with the full content (file + prompt)
        self._chat_history.append({"role": "user", "content": llm_text})

        # Start assistant bubble header
        self._chat_append("asst_label", t("PROMPT_LAB_LABEL_ASSISTANT") + "\n")

        def _on_token(token: str):
            self.root.after(0, lambda t=token: self._chat_append("asst_text", t))

        # Slice history according to context-depth slider
        # Each "turn" = 1 user msg + 1 assistant msg = 2 entries.
        # The current user message was just appended, so it's always included.
        _ctx_turns = self._chat_ctx_var.get()
        if _ctx_turns == 0:
            _history_to_send = self._chat_history          # send everything
        else:
            # Keep the last N turns (pairs) worth of history.
            # _ctx_turns * 2 gives the max messages; the current user message
            # is already at the end of _chat_history so it's naturally included.
            _history_to_send = self._chat_history[-(_ctx_turns * 2):]

        def _run():
            try:
                reply = chat_with_llm(
                    messages=_history_to_send,
                    system_prompt=system_prompt,
                    max_tokens=700,
                    on_token=_on_token,
                )
                self._chat_history.append({"role": "assistant", "content": reply})
                self.root.after(0, lambda: self._chat_append("asst_text", "\n\n"))
            except Exception as exc:
                self.root.after(0, lambda: self._chat_append(
                    "muted", f"[Error: {exc}]\n\n"
                ))
            finally:
                self.root.after(0, self._chat_send_done)

        threading.Thread(target=_run, daemon=True, name="PromptLab").start()

    def _chat_send_done(self):
        """Re-enable the send button after response completes."""
        self._chat_busy = False
        self._chat_send_btn.configure(state="normal", text=t("PROMPT_LAB_BTN_SEND"))
        self._chat_status.configure(text="")

    def _chat_clear(self):
        """Clear conversation history and display."""
        self._chat_history.clear()
        self._chat_display.configure(state="normal")
        self._chat_display.delete("1.0", "end")
        self._chat_display.configure(state="disabled")
        self._chat_append(
            "muted",
            "Chat cleared. Start a new conversation below.\n\n",
        )

    # ==================================================================
    # TAB 4: Settings
    # ==================================================================

    def _open_settings_window(self):
        """Open the Settings window, or bring it to front if already open."""
        if self._settings_window is not None and self._settings_window.winfo_exists():
            self._settings_window.lift()
            self._settings_window.focus_force()
            return

        win = ctk.CTkToplevel(self.root)
        win.title("KoKoFish — Settings")
        win.geometry("780x820")
        win.configure(fg_color=COLORS["bg_dark"])
        win.resizable(True, True)
        win.transient(self.root)   # keep it on top of the main window

        # Center over main window
        self.root.update_idletasks()
        rx = self.root.winfo_x() + (self.root.winfo_width() - 780) // 2
        ry = self.root.winfo_y() + (self.root.winfo_height() - 820) // 2
        win.geometry(f"+{rx}+{ry}")

        # Bring to front — after() gives Tk time to fully map the window first
        win.after(50, lambda: (win.lift(), win.focus_force()) if win.winfo_exists() else None)

        self._settings_window = win
        def _on_settings_close():
            self._settings_window = None
            # Clear label refs so update loops don't touch destroyed widgets
            for attr in ("ram_label", "vram_label", "cpu_label",
                         "tts_status_label", "stt_status_label"):
                setattr(self, attr, None)
            win.destroy()
        win.protocol("WM_DELETE_WINDOW", _on_settings_close)

        self._build_settings_tab(win)

    def _build_settings_tab(self, parent=None):
        tab = parent if parent is not None else getattr(self, "tab_settings", None)
        if tab is None:
            return

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
        section_header(main, t("SETTINGS_HEADER_GPU"))

        cuda_row = setting_row(main)
        ctk.CTkLabel(
            cuda_row,
            text=t("SETTINGS_GPU_CUDA_LABEL"),
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
                text=t("SETTINGS_GPU_NO_GPU"),
                text_color=COLORS["warning"],
            )
        elif cuda_installed:
            gpu_name = get_nvidia_gpu_name()
            self.cuda_status_label.configure(
                text=t("SETTINGS_GPU_DETECTED", gpu_name=gpu_name),
                text_color=COLORS["success"],
            )
        else:
            gpu_name = get_nvidia_gpu_name()
            self.cuda_status_label.configure(
                text=t("SETTINGS_GPU_ENABLE_CUDA", gpu_name=gpu_name),
                text_color=COLORS["text_secondary"],
            )
            
        # CUDA Notice
        notice_row = setting_row(main)
        ctk.CTkLabel(
            notice_row,
            text=t("SETTINGS_GPU_CUDA_NOTE"),
            font=(FONT_FAMILY, 13),
            text_color=COLORS["warning"],
        ).pack(side="left", padx=15, pady=12)


        # Memory saver
        mem_row = setting_row(main)
        ctk.CTkLabel(
            mem_row,
            text=t("SETTINGS_GPU_MEM_SAVER_LABEL"),
            font=(FONT_FAMILY, 13),
            text_color=COLORS["text_primary"],
        ).pack(side="left", padx=15, pady=12)

        ctk.CTkLabel(
            mem_row,
            text=t("SETTINGS_GPU_MEM_SAVER_DESC"),
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

        _active_engine_id = getattr(self.settings, 'engine', 'kokoro')
        _is_kokoro = _active_engine_id == 'kokoro'

        # --- Active Engine Info (read-only) ---
        section_header(main, t("SETTINGS_HEADER_KOKORO_ENGINE"))

        engine_info_row = setting_row(main)
        ctk.CTkLabel(
            engine_info_row,
            text=ENGINE_LABELS.get(_active_engine_id, _active_engine_id),
            font=(FONT_FAMILY, 13),
            text_color=COLORS["text_primary"],
        ).pack(side="left", padx=15, pady=12)

        # Install status
        try:
            from utils import is_kokoro_ready
            if _active_engine_id == "kokoro":
                _ready = is_kokoro_ready()
            elif _active_engine_id == "voxcpm_05b":
                from utils import is_voxcpm_ready
                _ready = is_voxcpm_ready("0.5B")
            elif _active_engine_id == "voxcpm_2b":
                from utils import is_voxcpm_ready
                _ready = is_voxcpm_ready("2B")
            elif _active_engine_id == "omnivoice":
                from utils import is_omnivoice_ready
                _ready = is_omnivoice_ready()
            else:
                _ready = False
        except Exception:
            _ready = False

        ctk.CTkLabel(
            engine_info_row,
            text=(t("SETTINGS_KOKORO_READY") if _ready else "Not installed"),
            font=(FONT_FAMILY, 11),
            text_color=(COLORS["success"] if _ready else COLORS["warning"]),
        ).pack(side="right", padx=15, pady=12)



        # Engine Selection
        engine_row = setting_row(main)
        ctk.CTkLabel(
            engine_row,
            text=t("SETTINGS_ENGINE_LABEL"),
            font=(FONT_FAMILY, 13),
            text_color=COLORS["text_primary"],
        ).pack(side="left", padx=15, pady=12)

        ctk.CTkLabel(
            engine_row,
            text=t("SETTINGS_ENGINE_DESC"),
            font=(FONT_FAMILY, 11),
            text_color=COLORS["text_muted"],
        ).pack(side="left", padx=(0, 10), pady=12)

        engine_options = [ENGINE_LABELS[k] for k in VALID_ENGINES]

        # Determine current engine from settings.engine field
        _eng = getattr(self.settings, 'engine', 'kokoro')
        current_engine = engine_label(_eng) if _eng in VALID_ENGINES else engine_label('kokoro')

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
        section_header(main, t("SETTINGS_HEADER_SYSTEM_STATUS"))

        # RAM readout
        ram_row = setting_row(main)
        ctk.CTkLabel(
            ram_row,
            text=t("SETTINGS_SYSTEM_RAM_LABEL"),
            font=(FONT_FAMILY, 13),
            text_color=COLORS["text_primary"],
        ).pack(side="left", padx=15, pady=12)

        self.ram_label = ctk.CTkLabel(
            ram_row,
            text=t("SETTINGS_SYSTEM_CALCULATING"),
            font=(FONT_FAMILY, 12),
            text_color=COLORS["text_secondary"],
        )
        self.ram_label.pack(side="right", padx=15, pady=12)

        # VRAM readout
        vram_row = setting_row(main)
        ctk.CTkLabel(
            vram_row,
            text=t("SETTINGS_SYSTEM_VRAM_LABEL"),
            font=(FONT_FAMILY, 13),
            text_color=COLORS["text_primary"],
        ).pack(side="left", padx=15, pady=12)

        self.vram_label = ctk.CTkLabel(
            vram_row,
            text=t("SETTINGS_SYSTEM_CALCULATING"),
            font=(FONT_FAMILY, 12),
            text_color=COLORS["text_secondary"],
        )
        self.vram_label.pack(side="right", padx=15, pady=12)

        # CPU usage readout
        cpu_row = setting_row(main)
        ctk.CTkLabel(
            cpu_row,
            text=t("SETTINGS_SYSTEM_CPU_LABEL"),
            font=(FONT_FAMILY, 13),
            text_color=COLORS["text_primary"],
        ).pack(side="left", padx=15, pady=12)

        self.cpu_label = ctk.CTkLabel(
            cpu_row,
            text=t("SETTINGS_SYSTEM_CALCULATING"),
            font=(FONT_FAMILY, 12),
            text_color=COLORS["text_secondary"],
        )
        self.cpu_label.pack(side="right", padx=15, pady=12)

        # CPU thread limit slider
        import os as _os
        _logical_cores = __import__("psutil").cpu_count(logical=True) or 8
        _cur_threads = getattr(self.settings, "cpu_threads", 0)
        cpu_thread_row = setting_row(main)
        ctk.CTkLabel(
            cpu_thread_row,
            text=t("SETTINGS_SYSTEM_CPU_THREADS_LABEL"),
            font=(FONT_FAMILY, 13),
            text_color=COLORS["text_primary"],
        ).pack(side="left", padx=15, pady=10)
        ctk.CTkLabel(
            cpu_thread_row,
            text=t("SETTINGS_SYSTEM_CPU_THREADS_DESC"),
            font=(FONT_FAMILY, 11),
            text_color=COLORS["text_muted"],
        ).pack(side="left", padx=(0, 10), pady=10)

        _thread_lbl = ctk.CTkLabel(
            cpu_thread_row,
            text=f"{_cur_threads} (auto)" if _cur_threads == 0 else str(_cur_threads),
            font=(FONT_FAMILY, 12),
            text_color=COLORS["text_secondary"],
            width=70,
        )
        _thread_lbl.pack(side="right", padx=(0, 5), pady=10)

        def _on_thread_change(v):
            val = int(round(float(v)))
            _thread_lbl.configure(text=f"{val} (auto)" if val == 0 else str(val))
            self.settings.cpu_threads = val
            self.settings.save()
            # Apply immediately
            if val > 0:
                try:
                    import torch as _t
                    _t.set_num_threads(val)
                    _t.set_num_interop_threads(max(1, val // 2))
                except Exception:
                    pass
            else:
                try:
                    import torch as _t
                    import psutil as _ps
                    _t.set_num_threads(_ps.cpu_count(logical=True) or 4)
                except Exception:
                    pass

        _thread_sl = ctk.CTkSlider(
            cpu_thread_row,
            from_=0, to=_logical_cores,
            number_of_steps=_logical_cores,
            command=_on_thread_change,
            progress_color=COLORS["accent"],
            button_color=COLORS["accent_light"],
            width=200,
            height=16,
        )
        _thread_sl.set(_cur_threads)
        _thread_sl.pack(side="right", padx=15, pady=10)

        # ffmpeg status
        ffmpeg_row = setting_row(main)
        ctk.CTkLabel(
            ffmpeg_row,
            text=t("SETTINGS_SYSTEM_FFMPEG_LABEL"),
            font=(FONT_FAMILY, 13),
            text_color=COLORS["text_primary"],
        ).pack(side="left", padx=15, pady=12)

        ffmpeg_ok = is_ffmpeg_available()
        ffmpeg_text = t("SETTINGS_SYSTEM_FFMPEG_FOUND") if ffmpeg_ok else t("SETTINGS_SYSTEM_FFMPEG_NOT_FOUND")
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
            text=t("SETTINGS_SYSTEM_TTS_LABEL"),
            font=(FONT_FAMILY, 13),
            text_color=COLORS["text_primary"],
        ).pack(side="left", padx=15, pady=12)

        # Reflect actual engine state rather than always showing "Not loaded"
        if self.tts and self.tts.is_loaded:
            _prov = getattr(self.tts, "provider", "")
            if _prov == "cuda":
                _tts_text = t("SETTINGS_SYSTEM_TTS_LOADED_GPU")
            elif _prov == "cpu":
                _tts_text = t("SETTINGS_SYSTEM_TTS_LOADED_CPU")
            else:
                _tts_text = t("SETTINGS_SYSTEM_TTS_LOADED")
            _tts_color = COLORS["success"]
        else:
            _tts_text  = t("SETTINGS_SYSTEM_NOT_LOADED")
            _tts_color = COLORS["text_muted"]

        self.tts_status_label = ctk.CTkLabel(
            tts_status_row,
            text=_tts_text,
            font=(FONT_FAMILY, 12),
            text_color=_tts_color,
        )
        self.tts_status_label.pack(side="right", padx=15, pady=12)

        # STT model status
        stt_status_row = setting_row(main)
        ctk.CTkLabel(
            stt_status_row,
            text=t("SETTINGS_SYSTEM_STT_LABEL"),
            font=(FONT_FAMILY, 13),
            text_color=COLORS["text_primary"],
        ).pack(side="left", padx=15, pady=12)

        _stt_text  = t("SETTINGS_SYSTEM_STT_LOADED") if (self.stt and self.stt.is_loaded) else t("SETTINGS_SYSTEM_NOT_LOADED")
        _stt_color = COLORS["success"] if (self.stt and self.stt.is_loaded) else COLORS["text_muted"]
        self.stt_status_label = ctk.CTkLabel(
            stt_status_row,
            text=_stt_text,
            font=(FONT_FAMILY, 12),
            text_color=_stt_color,
        )
        self.stt_status_label.pack(side="right", padx=15, pady=12)

        # --- AI Features ---
        section_header(main, t("SETTINGS_HEADER_AI_FEATURES"))

        from tag_suggester import (
            is_llm_available as _llm_avail,
            is_qwen_model_ready as _qwen_ready,
            LLM_MODELS as _LLM_MODELS,
            get_active_llm_key as _get_llm_key,
            set_active_llm_key as _set_llm_key,
        )

        # llama-cpp-python row
        llama_row = setting_row(main)
        ctk.CTkLabel(
            llama_row, text=t("SETTINGS_AI_LLAMA_LABEL"),
            font=(FONT_FAMILY, 13), text_color=COLORS["text_primary"],
        ).pack(side="left", padx=15, pady=12)
        ctk.CTkLabel(
            llama_row, text=t("SETTINGS_AI_LLAMA_DESC"),
            font=(FONT_FAMILY, 11), text_color=COLORS["text_muted"],
        ).pack(side="left", padx=(0, 10), pady=12)

        _llama_installed = _llm_avail()
        self.llama_status_label = ctk.CTkLabel(
            llama_row,
            text=t("SETTINGS_AI_INSTALLED") if _llama_installed else t("SETTINGS_AI_NOT_INSTALLED"),
            font=(FONT_FAMILY, 11),
            text_color=COLORS["success"] if _llama_installed else COLORS["danger"],
        )
        self.llama_status_label.pack(side="right", padx=(5, 15), pady=12)
        if not _llama_installed:
            self.llama_install_btn = ctk.CTkButton(
                llama_row, text=t("SETTINGS_BTN_INSTALL"), width=90, height=28, corner_radius=6,
                fg_color=COLORS["accent"], hover_color=COLORS["accent_hover"],
                font=(FONT_FAMILY, 11), command=self._install_llama_cpp,
            )
            self.llama_install_btn.pack(side="right", padx=(0, 5), pady=12)

        # LLM model selector row
        llm_sel_row = setting_row(main)
        ctk.CTkLabel(
            llm_sel_row, text=t("SETTINGS_AI_LLM_MODEL_LABEL"),
            font=(FONT_FAMILY, 13), text_color=COLORS["text_primary"],
        ).pack(side="left", padx=15, pady=12)

        _llm_key_var = ctk.StringVar(value=_get_llm_key())

        def _on_llm_change(key):
            _set_llm_key(key)
            self.settings.llm_model = key  # keep Settings in sync so save() doesn't overwrite
            _is_ollama_key = key in _LLM_MODELS and _LLM_MODELS[key].get("backend") == "ollama"
            # Update the llama-cpp status row to reflect the backend in use
            if _is_ollama_key:
                self.llama_status_label.configure(
                    text=t("SETTINGS_AI_NOT_NEEDED_OLLAMA"),
                    text_color=COLORS["text_muted"],
                )
            else:
                self.llama_status_label.configure(
                    text=t("SETTINGS_AI_INSTALLED") if _llama_installed else t("SETTINGS_AI_NOT_INSTALLED"),
                    text_color=COLORS["success"] if _llama_installed else COLORS["danger"],
                )
            # Pass the key directly so we check the new model's file,
            # not whatever get_active_llm_key() happens to return right now.
            _qw_installed = _qwen_ready(key)
            _not_ready_label = t("SETTINGS_AI_NOT_PULLED") if _is_ollama_key else t("SETTINGS_AI_NOT_DOWNLOADED")
            self.qwen_status_label.configure(
                text=t("SETTINGS_AI_READY") if _qw_installed else _not_ready_label,
                text_color=COLORS["success"] if _qw_installed else COLORS["danger"],
            )
            # If model not present, offer to download / pull immediately
            if not _qw_installed and (_llama_installed or _is_ollama_key):
                _action = "pull" if _is_ollama_key else "download"
                if messagebox.askyesno(
                    t("SETTINGS_AI_MODEL_NOT_AVAILABLE_TITLE"),
                    t("SETTINGS_AI_MODEL_NOT_AVAILABLE_BODY", key=key, action=_action),
                    parent=self.root,
                ):
                    self._download_qwen_from_settings()

        ctk.CTkOptionMenu(
            llm_sel_row,
            variable=_llm_key_var,
            values=list(_LLM_MODELS.keys()),
            width=280, height=30,
            fg_color=COLORS["bg_input"], button_color=COLORS["accent"],
            dropdown_fg_color=COLORS["bg_card"], dropdown_hover_color=COLORS["bg_card_hover"],
            text_color=COLORS["text_primary"], font=(FONT_FAMILY, 11),
            dynamic_resizing=False,
            command=_on_llm_change,
        ).pack(side="left", padx=(8, 0), pady=10)

        # Qwen/LLM model status + download row
        qwen_row = setting_row(main)
        ctk.CTkLabel(
            qwen_row, text=t("SETTINGS_AI_MODEL_STATUS_LABEL"),
            font=(FONT_FAMILY, 13), text_color=COLORS["text_primary"],
        ).pack(side="left", padx=15, pady=12)

        _qwen_installed = _qwen_ready()
        self.qwen_status_label = ctk.CTkLabel(
            qwen_row,
            text=t("SETTINGS_AI_READY") if _qwen_installed else (t("SETTINGS_AI_INSTALL_LLAMA_FIRST") if not _llama_installed else t("SETTINGS_AI_NOT_DOWNLOADED")),
            font=(FONT_FAMILY, 11),
            text_color=COLORS["success"] if _qwen_installed else COLORS["text_muted"],
        )
        self.qwen_status_label.pack(side="right", padx=(5, 15), pady=12)

        ctk.CTkButton(
            qwen_row, text=t("SETTINGS_AI_BTN_EDIT_PROMPTS"), width=110, height=28, corner_radius=6,
            fg_color="#5a3e8a", hover_color="#7b5ea7", font=(FONT_FAMILY, 11),
            command=self._open_prompt_editor,
        ).pack(side="right", padx=(0, 5), pady=12)

        if not _qwen_installed and _llama_installed:
            ctk.CTkButton(
                qwen_row, text=t("SETTINGS_AI_BTN_DOWNLOAD_MODEL"), width=140, height=28, corner_radius=6,
                fg_color=COLORS["accent"], hover_color=COLORS["accent_hover"],
                font=(FONT_FAMILY, 11), command=self._download_qwen_from_settings,
            ).pack(side="right", padx=(0, 5), pady=12)

        # Install log (hidden until install starts)
        self._llama_log_row = setting_row(main)
        self._llama_log_label = ctk.CTkLabel(
            self._llama_log_row, text="",
            font=(FONT_FAMILY, 10), text_color=COLORS["text_secondary"],
            justify="left", anchor="w", wraplength=560,
        )
        self._llama_log_label.pack(fill="x", padx=15, pady=8)
        self._llama_log_row.pack_forget()  # hidden until needed

        # --- Credits ---
        section_header(main, t("SETTINGS_HEADER_CREDITS"))

        credits_frame = setting_row(main)
        ctk.CTkLabel(
            credits_frame,
            text=t("SETTINGS_CREDITS_TEXT"),
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
            title=t("SPEECH_LAB_DIALOG_FILE_SELECT"),
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
            messagebox.showerror(t("SPEECH_LAB_MSG_FILE_ERROR_TITLE"), t("SPEECH_LAB_MSG_FILE_ERROR_BODY", error=exc))

    def _rebuild_playlist_ui(self):
        """Rebuild the playlist list in the scrollable frame."""
        for widget in self.playlist_frame.winfo_children():
            widget.destroy()

        self._drag_row_widgets: list = []  # [(idx, row_widget), ...]

        if not self._playlist_items:
            self.playlist_empty_label = ctk.CTkLabel(
                self.playlist_frame,
                text=t("SPEECH_LAB_PLAYLIST_EMPTY"),
                font=(FONT_FAMILY, 12),
                text_color=COLORS["text_muted"],
            )
            self.playlist_empty_label.pack(pady=30)
            return

        is_kokoro = getattr(self.settings, 'engine', 'kokoro') == 'kokoro'
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
            self._drag_row_widgets.append((idx, row))

            txt_color = "#ffffff" if is_active else COLORS["text_primary"]

            # ── Drag handle ──────────────────────────────────────────────
            _handle = ctk.CTkLabel(
                row,
                text="⠿",
                font=(FONT_FAMILY, 16),
                text_color=COLORS["text_muted"],
                width=20,
                cursor="fleur",
            )
            _handle.pack(side="left", padx=(4, 0))
            _handle.bind("<ButtonPress-1>",   lambda e, i=idx: self._drag_start(e, i))
            _handle.bind("<B1-Motion>",        lambda e, i=idx: self._drag_motion(e, i))
            _handle.bind("<ButtonRelease-1>",  lambda e, i=idx: self._drag_end(e, i))

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
            _MAX_NAME = 32
            _full_name = item['name']
            _short_name = (
                _full_name[:_MAX_NAME] + "…"
                if len(_full_name) > _MAX_NAME else _full_name
            )
            name_lbl = ctk.CTkLabel(
                row,
                text=f"{idx + 1}. {_short_name}",
                font=(FONT_FAMILY, 12),
                text_color=txt_color,
                anchor="w",
                cursor="hand2",
                width=240,
            )
            name_lbl.pack(side="left", padx=(0, 2))
            name_lbl.bind("<Double-Button-1>", lambda e, i=idx: self._open_editor(i))
            if len(_full_name) > _MAX_NAME:
                self._make_tooltip(name_lbl, _full_name)

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

                blend_options = [t("SPEECH_LAB_BLEND_PLACEHOLDER")] + voice_options
                item_blend = item.get("blend_voice", "") or t("SPEECH_LAB_BLEND_PLACEHOLDER")
                if item_blend not in blend_options:
                    item_blend = t("SPEECH_LAB_BLEND_PLACEHOLDER")

                blend_var = ctk.StringVar(value=item_blend)

                def _on_blend_change(v, i=idx, bvar=blend_var):
                    val = bvar.get()
                    self._playlist_items[i]["blend_voice"] = "" if val == t("SPEECH_LAB_BLEND_PLACEHOLDER") else val

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
                self._make_tooltip(_blend_slider, t("SPEECH_LAB_TOOLTIP_BLEND_RATIO"))
                blend_ratio_label.pack(side="left", padx=(0, 4))

            # ── Assisted Flow toggle ─────────────────────────────────────
            af_var = ctk.BooleanVar(value=item.get("assisted_flow", False) if _af_llm_ok else False)

            def _on_af_toggle(v=af_var, i=idx):
                self._playlist_items[i]["assisted_flow"] = v.get()

            ctk.CTkLabel(
                row, text=t("SPEECH_LAB_ASSISTED_FLOW_LABEL"),
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
            self._make_tooltip(_af_switch, t("SPEECH_LAB_TOOLTIP_ASSISTED_FLOW"))

            if not _af_llm_ok:
                ctk.CTkLabel(
                    row,
                    text=t("SPEECH_LAB_WARN_SETTINGS"),
                    font=(FONT_FAMILY, 8),
                    text_color=COLORS["text_muted"],
                ).pack(side="left", padx=(0, 2))
            else:
                ctk.CTkFrame(row, width=2, fg_color="transparent").pack(side="left")

            # ── Translate toggle ──────────────────────────────────────────
            # Kokoro: auto-translates based on voice language (no toggle)
            # Fish engines: toggle + language dropdown + tone dropdown
            if _af_llm_ok:
                if is_kokoro:
                    # Show a small "Auto" badge — translation is always automatic
                    _item_vid = KOKORO_VOICES.get(item.get("voice", ""), "")
                    _item_lang = KOKORO_VOICE_LANG.get(_item_vid, "English")
                    if _item_lang != "English":
                        _auto_lbl = ctk.CTkLabel(
                            row,
                            text=t("SPEECH_LAB_AUTO_TRANSLATE_BADGE", lang=_item_lang),
                            font=(FONT_FAMILY, 8),
                            text_color="#f4a261",
                        )
                        _auto_lbl.pack(side="left", padx=(4, 2))
                        self._make_tooltip(
                            _auto_lbl,
                            t("SPEECH_LAB_AUTO_TRANSLATE_TOOLTIP", lang=_item_lang),
                        )
                else:
                    # Non-Kokoro: toggle + language dropdown + tone dropdown
                    tr_var = ctk.BooleanVar(value=item.get("translate", False))

                    def _on_tr_toggle(v=tr_var, i=idx):
                        self._playlist_items[i]["translate"] = v.get()

                    ctk.CTkLabel(
                        row, text=t("SPEECH_LAB_TRANSLATE_LABEL"),
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
                    self._make_tooltip(
                        _tr_switch,
                        t("SPEECH_LAB_TOOLTIP_TRANSLATE"),
                    )

                    _lang_display_opts = _display_names(_LANG_KEY_MAP)
                    saved_lang_en = item.get("translate_lang", "") or "Japanese"
                    saved_lang_disp = _display_from_en(_LANG_KEY_MAP, saved_lang_en)
                    tr_lang_var = ctk.StringVar(value=saved_lang_disp)

                    def _on_tr_lang(v, i=idx, lvar=tr_lang_var):
                        self._playlist_items[i]["translate_lang"] = _en_from_display(_LANG_KEY_MAP, lvar.get())

                    ctk.CTkOptionMenu(
                        row,
                        variable=tr_lang_var,
                        values=_lang_display_opts,
                        width=110,
                        height=22,
                        fg_color=COLORS["bg_input"],
                        button_color="#e76f51",
                        button_hover_color="#f4a261",
                        dropdown_fg_color=COLORS["bg_card"],
                        dropdown_hover_color=COLORS["bg_card_hover"],
                        font=(FONT_FAMILY, 10),
                        command=lambda v, i=idx, lvar=tr_lang_var: _on_tr_lang(v, i, lvar),
                    ).pack(side="left", padx=(0, 2))

                    _tone_display_opts = _display_names(_TONE_KEY_MAP)
                    saved_tone_en = item.get("translate_tone", "Natural")
                    saved_tone_disp = _display_from_en(_TONE_KEY_MAP, saved_tone_en)
                    tr_tone_var = ctk.StringVar(value=saved_tone_disp)

                    def _on_tr_tone(v, i=idx, tvar=tr_tone_var):
                        self._playlist_items[i]["translate_tone"] = _en_from_display(_TONE_KEY_MAP, tvar.get())

                    ctk.CTkOptionMenu(
                        row,
                        variable=tr_tone_var,
                        values=_tone_display_opts,
                        width=90,
                        height=22,
                        fg_color=COLORS["bg_input"],
                        button_color="#9b59b6",
                        button_hover_color="#8e44ad",
                        dropdown_fg_color=COLORS["bg_card"],
                        dropdown_hover_color=COLORS["bg_card_hover"],
                        font=(FONT_FAMILY, 10),
                        command=lambda v, i=idx, tvar=tr_tone_var: _on_tr_tone(v, i, tvar),
                    ).pack(side="left", padx=(0, 2))

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
            self._make_tooltip(_rb, t("SPEECH_LAB_TOOLTIP_REMOVE_ITEM"))

            # After audio exists (near ✕): play/pause 🔊, save 📁, info ⓘ
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
                self._make_tooltip(_sb, t("SPEECH_LAB_TOOLTIP_SAVE_AUDIO"))
                _pb = ctk.CTkButton(
                    row, text="⏸" if is_previewing else "▶",
                    fg_color=COLORS["success"] if is_previewing else COLORS["bg_input"],
                    hover_color="#05b890" if is_previewing else COLORS["bg_card_hover"],
                    border_color=COLORS["success"], border_width=1,
                    command=lambda i=idx: self._preview_item(i),
                    **_ib,
                )
                _pb.pack(side="right", padx=(0, 2))
                self._make_tooltip(_pb, t("SPEECH_LAB_TOOLTIP_PAUSE_PREVIEW") if is_previewing else t("SPEECH_LAB_TOOLTIP_PLAY_PREVIEW"))
                _ib_btn = ctk.CTkButton(
                    row, text="ⓘ",
                    fg_color=COLORS["bg_input"], hover_color=COLORS["bg_card_hover"],
                    border_color=COLORS["border"], border_width=1,
                    command=lambda i=idx: self._open_audio_meta_editor(
                        self._playlist_items[i].get("audio_path", "")
                    ),
                    **_ib,
                )
                _ib_btn.pack(side="right", padx=(0, 2))
                self._make_tooltip(_ib_btn, t("SPEECH_LAB_TOOLTIP_EDIT_METADATA"))

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
                self._make_tooltip(_pb, t("SPEECH_LAB_TOOLTIP_PAUSE_CONVERT"))
                _cb = ctk.CTkButton(
                    row, text="⊘",
                    fg_color=COLORS["danger"], hover_color="#d43d62",
                    command=lambda i=idx: self._cancel_item(i),
                    **_ib,
                )
                _cb.pack(side="left", padx=(0, 2))
                self._make_tooltip(_cb, t("SPEECH_LAB_TOOLTIP_CANCEL_CONVERT"))
            elif is_queued:
                _cb = ctk.CTkButton(
                    row, text="⊘",
                    fg_color=COLORS["danger"], hover_color="#d43d62",
                    command=lambda i=idx: self._cancel_item(i),
                    **_ib,
                )
                _cb.pack(side="left", padx=(4, 2))
                self._make_tooltip(_cb, t("SPEECH_LAB_TOOLTIP_REMOVE_QUEUE"))
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
                self._make_tooltip(_cvt, t("SPEECH_LAB_TOOLTIP_RECONVERT") if has_audio else t("SPEECH_LAB_TOOLTIP_CONVERT"))

    def _open_editor(self, index: int):
        """Open the text editor window for a playlist item."""
        if index < 0 or index >= len(self._playlist_items):
            return
        from text_editor_window import TextEditorWindow
        engine = getattr(self.settings, 'engine', 'kokoro')
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
                t("SPEECH_LAB_MSG_REGEN_TITLE"),
                t("SPEECH_LAB_MSG_REGEN_BODY", name=item['name']),
                parent=self.root,
            ):
                return

        # If something is running, add to queue and bail
        if self._is_playing:
            self._single_item_queue.append(index)
            self._rebuild_playlist_ui()
            self.tts_status.configure(text=t("SPEECH_LAB_STATUS_QUEUED", name=item['name']))
            return

        # Nothing running — start immediately
        if not self.tts.is_loaded:
            self.tts_status.configure(text=t("SPEECH_LAB_STATUS_LOADING_MODEL"))
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
            self.tts_status.configure(text=t("SPEECH_LAB_STATUS_DONE"))

    def _cancel_item(self, index: int):
        """Cancel an actively generating item or remove it from the queue."""
        if index == self._current_playing:
            # Stop generation engine AND kill the live audio stream immediately
            self.tts.cancel()
            self._stop_tts_gen_stream()
            self._is_playing = False
            self._current_playing = -1
            if self._single_item_queue:
                next_idx = self._single_item_queue.pop(0)
                self.root.after(200, lambda: self._process_single_item(next_idx))
            else:
                self._rebuild_playlist_ui()
                self.tts_status.configure(text=t("SPEECH_LAB_STATUS_CANCELLED"))
        elif index in self._single_item_queue:
            self._single_item_queue.remove(index)
            self._rebuild_playlist_ui()
            self.tts_status.configure(text=t("SPEECH_LAB_STATUS_REMOVED_FROM_QUEUE", name=self._playlist_items[index]['name']))

    def _save_item_audio(self, index: int):
        """Save the generated audio for a single playlist item to a user-chosen path."""
        if index < 0 or index >= len(self._playlist_items):
            return
        item = self._playlist_items[index]
        src = item.get("audio_path", "")
        if not src or not os.path.isfile(src):
            messagebox.showwarning(t("SPEECH_LAB_MSG_SAVE_AUDIO_TITLE"), t("SPEECH_LAB_MSG_SAVE_AUDIO_NO_GEN"), parent=self.root)
            return
        from tkinter.filedialog import asksaveasfilename
        stem = os.path.splitext(item["name"])[0]
        dest = asksaveasfilename(
            parent=self.root,
            defaultextension=".mp3",
            filetypes=[("MP3 audio", "*.mp3"), ("WAV audio", "*.wav"), ("All files", "*.*")],
            initialfile=f"{stem}.mp3",
            title=t("SPEECH_LAB_DIALOG_SAVE_AUDIO_AS"),
        )
        if not dest:
            return
        try:
            from pydub import AudioSegment
            audio = AudioSegment.from_wav(src)
            fmt = os.path.splitext(dest)[1].lstrip(".").lower() or "mp3"
            audio.export(dest, format=fmt)
            self.tts_status.configure(text=t("SPEECH_LAB_STATUS_SAVED", filename=os.path.basename(dest)))
        except Exception as exc:
            messagebox.showerror(t("SPEECH_LAB_MSG_SAVE_AUDIO_TITLE"), t("SPEECH_LAB_MSG_SAVE_AUDIO_FAILED", error=exc), parent=self.root)

    def _open_audio_meta_editor(self, audio_path: str):
        """Open a window to view/edit audio file metadata using FFmpeg."""
        if not audio_path or not os.path.isfile(audio_path):
            messagebox.showwarning(t("METADATA_MSG_NO_FILE_TITLE"), t("METADATA_MSG_NO_FILE_BODY"), parent=self.root)
            return
        if not is_ffmpeg_available():
            messagebox.showerror(t("METADATA_MSG_NO_FILE_TITLE"), t("METADATA_MSG_NO_FFMPEG"), parent=self.root)
            return

        import subprocess, json, shutil, tempfile

        # ── Read existing tags via ffprobe ──────────────────────────────
        existing: dict = {}
        try:
            result = subprocess.run(
                ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_format", audio_path],
                capture_output=True, text=True, timeout=15, creationflags=_NO_WIN,
            )
            info = json.loads(result.stdout)
            existing = {k.lower(): v for k, v in info.get("format", {}).get("tags", {}).items()}
        except Exception as exc:
            logger.warning("ffprobe failed: %s", exc)

        # ── Build editor window ─────────────────────────────────────────
        win = ctk.CTkToplevel(self.root)
        win.title(t("METADATA_EDITOR_TITLE", filename=os.path.basename(audio_path)))
        win.geometry("480x400")
        win.configure(fg_color=COLORS["bg_dark"])
        win.grab_set()
        win.lift()
        win.focus_force()

        _lf = {"font": (FONT_FAMILY, 12), "text_color": COLORS["text_secondary"],
               "anchor": "w", "width": 100}
        _ef = {"font": (FONT_FAMILY, 12), "fg_color": COLORS["bg_input"],
               "text_color": COLORS["text_primary"], "border_color": COLORS["border"],
               "border_width": 1, "width": 300, "height": 34}

        fields_cfg = [
            ("title",   t("METADATA_FIELD_TITLE")),
            ("artist",  t("METADATA_FIELD_ARTIST")),
            ("album",   t("METADATA_FIELD_ALBUM")),
            ("date",    t("METADATA_FIELD_DATE")),
            ("genre",   t("METADATA_FIELD_GENRE")),
            ("comment", t("METADATA_FIELD_COMMENT")),
        ]

        entries: dict = {}
        form = ctk.CTkFrame(win, fg_color=COLORS["bg_card"], corner_radius=10)
        form.pack(fill="both", expand=True, padx=20, pady=(20, 10))

        for tag, label in fields_cfg:
            row = ctk.CTkFrame(form, fg_color="transparent")
            row.pack(fill="x", padx=16, pady=4)
            ctk.CTkLabel(row, text=label, **_lf).pack(side="left")
            ent = ctk.CTkEntry(row, **_ef)
            ent.insert(0, existing.get(tag, ""))
            ent.pack(side="left", padx=(8, 0))
            entries[tag] = ent

        def _save_meta():
            meta_args = []
            for tag, _ in fields_cfg:
                val = entries[tag].get().strip()
                meta_args += ["-metadata", f"{tag}={val}"]

            ext = os.path.splitext(audio_path)[1].lower()
            tmp_fd, tmp_path = tempfile.mkstemp(suffix=ext)
            os.close(tmp_fd)
            try:
                cmd = (
                    ["ffmpeg", "-y", "-i", audio_path]
                    + meta_args
                    + ["-c", "copy", tmp_path]
                )
                r = subprocess.run(cmd, capture_output=True, text=True, timeout=60, creationflags=_NO_WIN)
                if r.returncode != 0:
                    raise RuntimeError(r.stderr[-600:])
                shutil.move(tmp_path, audio_path)
                messagebox.showinfo(t("METADATA_MSG_NO_FILE_TITLE"), t("METADATA_MSG_SAVED"), parent=win)
                win.destroy()
            except Exception as exc:
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass
                messagebox.showerror(t("METADATA_MSG_NO_FILE_TITLE"), t("METADATA_MSG_FAILED", error=exc), parent=win)

        btn_row = ctk.CTkFrame(win, fg_color="transparent")
        btn_row.pack(pady=(0, 16))
        ctk.CTkButton(btn_row, text=t("COMMON_BTN_SAVE"), width=120, height=36,
                      fg_color=COLORS["accent"], hover_color=COLORS["accent_hover"],
                      font=(FONT_FAMILY, 13, "bold"), command=_save_meta).pack(side="left", padx=6)
        ctk.CTkButton(btn_row, text=t("COMMON_BTN_CANCEL"), width=100, height=36,
                      fg_color=COLORS["bg_input"], hover_color=COLORS["bg_card_hover"],
                      font=(FONT_FAMILY, 13), command=win.destroy).pack(side="left", padx=6)

    def _tts_remove_item(self, index: int):
        if self._preview_idx == index:
            self._stop_preview()
        if self._current_playing == index:
            # Item is actively generating — cancel engine and kill audio immediately
            self.tts.cancel()
            self._stop_tts_gen_stream()
            self._is_playing = False
            self._current_playing = -1
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

    def _stop_tts_gen_stream(self):
        """Immediately kill the real-time TTS generation audio stream and drain its queue."""
        self._tts_gen_active = False
        q = self._tts_gen_queue
        if q is not None:
            # Drain so the _finish thread doesn't block
            while not q.empty():
                try: q.get_nowait()
                except Exception: break
            self._tts_gen_queue = None
        s = self._tts_gen_stream
        if s is not None:
            try:
                s.stop()
                s.close()
            except Exception:
                pass
            self._tts_gen_stream = None

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
        self._preview_queue = []
        self._show_play_btns(False)

    def _sync_gen_settings_to_engine(self):
        """Push persisted generation quality settings into the TTS engine."""
        pass

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
        self.speed_label.configure(text=t("SPEECH_LAB_SPEED_LABEL", value=f"{value:.1f}"))
        self.settings.speed = round(value, 1)

    def _on_volume_change(self, value):
        self.vol_label.configure(text=t("SPEECH_LAB_VOLUME_LABEL", value=int(value)))
        self.settings.volume = int(value)

    def _on_cadence_change(self, value):
        self.cad_label.configure(text=t("SPEECH_LAB_CADENCE_LABEL", value=int(value)))
        self.settings.cadence = int(value)

    def _ensure_tts_loaded(self, on_success):
        """Helper to lazy load TTS with a beautiful popup overlay."""
        if self.tts.is_loaded:
            on_success()
            return
            
        # Create themed popup
        popup = ctk.CTkToplevel(self.root)
        popup.title(t("SPEECH_LAB_ENGINE_BOOT_TITLE"))
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
        
        _eng_id = getattr(self.settings, 'engine', 'kokoro')
        engine_name = ENGINE_LABELS.get(_eng_id, _eng_id)
        lbl_msg = ctk.CTkLabel(popup, text=t("SPEECH_LAB_ENGINE_BOOTING", engine_name=engine_name), font=(FONT_FAMILY, 14, "bold"), text_color=COLORS["text_primary"])
        lbl_msg.pack(pady=(20, 10))
        
        pb = ctk.CTkProgressBar(popup, progress_color=COLORS["accent"], fg_color=COLORS["bg_input"], width=280, height=8, corner_radius=4)
        pb.pack(pady=5)
        pb.set(0)
        
        status = ctk.CTkLabel(popup, text=t("COMMON_INITIALIZING"), font=(FONT_FAMILY, 11), text_color=COLORS["text_muted"])
        status.pack(pady=5)

        def on_progress(text: str, frac: float):
            self.root.after(0, lambda: status.configure(text=text))
            self.root.after(0, lambda: pb.set(frac))

        def on_ready():
            def _ready_ui():
                popup.grab_release()
                popup.destroy()
                warnings = getattr(self.tts, "load_warnings", [])
                if warnings:
                    self.update_tts_status(f"⚠  {warnings[0]}", COLORS["warning"])
                else:
                    _prov = getattr(self.tts, "provider", "")
                    if _prov == "cuda":
                        self.update_tts_status(t("SPEECH_LAB_STATUS_ENGINE_READY_GPU"), COLORS["success"])
                    elif _prov == "cpu":
                        self.update_tts_status(t("SPEECH_LAB_STATUS_ENGINE_READY_CPU"), COLORS["success"])
                    else:
                        self.update_tts_status(t("SPEECH_LAB_STATUS_ENGINE_READY"), COLORS["success"])
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
                messagebox.showerror(t("COMMON_ERROR"), t("SPEECH_LAB_ENGINE_LOAD_FAILED_BODY", error=exc))
                self.update_tts_status(t("SPEECH_LAB_STATUS_LOAD_FAILED"), COLORS["danger"])
            self.root.after(0, _err_ui)

        self.tts.load_model(on_progress=on_progress, on_ready=on_ready, on_error=on_error)

    def _tts_play(self):
        """Read all playlist items in order, skipping those already generated."""
        if not self._playlist_items:
            messagebox.showinfo("KoKoFish", t("SPEECH_LAB_MSG_ADD_FILES"))
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
                t("SPEECH_LAB_MSG_ALL_DONE_TITLE"),
                t("SPEECH_LAB_MSG_ALL_DONE_BODY"),
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
            messagebox.showinfo("KoKoFish", t("SPEECH_LAB_MSG_NO_SELECTED"))
            return

        # Check if any already have audio
        done = [i for i in selected
                if self._playlist_items[i].get("audio_path") and
                os.path.isfile(self._playlist_items[i]["audio_path"])]
        pending = [i for i in selected if i not in done]

        if done and not pending:
            # All selected are already done
            if not messagebox.askyesno(
                t("SPEECH_LAB_MSG_REGEN_ALL_TITLE"),
                t("SPEECH_LAB_MSG_REGEN_ALL_BODY", count=len(done)),
            ):
                return
            indices = selected
        elif done:
            answer = messagebox.askyesno(
                t("SPEECH_LAB_MSG_REGEN_ALL_TITLE"),
                t("SPEECH_LAB_MSG_REGEN_SOME_BODY", count=len(done)),
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

    # ── Playlist drag-to-reorder ─────────────────────────────────────────

    def _drag_start(self, event, idx: int):
        self._drag_src = idx
        self._drag_over = idx
        event.widget.grab_set()

    def _drag_motion(self, event, _idx: int):
        if not hasattr(self, "_drag_src") or self._drag_src is None:
            return
        x = event.widget.winfo_rootx() + event.x
        y = event.widget.winfo_rooty() + event.y
        target = self._row_at(x, y)
        if target is not None and target != self._drag_over:
            self._drag_over = target
            self.tts_status.configure(
                text=t("SPEECH_LAB_STATUS_DROP_AT_POSITION", position=target + 1)
            )

    def _drag_end(self, event, _idx: int):
        if not hasattr(self, "_drag_src") or self._drag_src is None:
            return
        try:
            event.widget.grab_release()
        except Exception:
            pass
        x = event.widget.winfo_rootx() + event.x
        y = event.widget.winfo_rooty() + event.y
        target = self._row_at(x, y)
        src = self._drag_src
        self._drag_src = None
        self._drag_over = None
        if target is not None and target != src:
            item = self._playlist_items.pop(src)
            self._playlist_items.insert(target, item)
            self._rebuild_playlist_ui()
        self.tts_status.configure(text=t("SPEECH_LAB_STATUS_READY"))

    def _row_at(self, x_screen: int, y_screen: int):
        """Return the playlist item index under screen coords (x, y), or None."""
        for item_idx, row_widget in getattr(self, "_drag_row_widgets", []):
            try:
                rx = row_widget.winfo_rootx()
                ry = row_widget.winfo_rooty()
                rw = row_widget.winfo_width()
                rh = row_widget.winfo_height()
                if rx <= x_screen <= rx + rw and ry <= y_screen <= ry + rh:
                    return item_idx
            except Exception:
                pass
        return None

    def _show_convert_btns(self, show: bool):
        """Show or hide the pause/stop buttons for Convert Selected."""
        if show:
            self.btn_pause.pack(side="left", padx=(0, 2), after=self._btn_convert)
            self.btn_stop.pack(side="left", padx=(0, 12), after=self.btn_pause)
        else:
            self.btn_pause.pack_forget()
            self.btn_stop.pack_forget()

    def _show_play_btns(self, show: bool):
        """Show or hide the pause/stop buttons for Play Selected."""
        if show:
            self.btn_play_pause.pack(side="left", padx=(0, 2), after=self._btn_play_sel)
            self.btn_play_stop.pack(side="left", padx=(0, 12), after=self.btn_play_pause)
        else:
            self.btn_play_pause.pack_forget()
            self.btn_play_stop.pack_forget()

    def _run_tts_for_indices(self, indices: list):
        """Generate TTS sequentially for the given list of item indices."""
        if not indices:
            return
        self._stop_preview()
        self._is_playing = True
        self._show_convert_btns(True)
        self._tts_queue = list(indices)
        self._current_playing = self._tts_queue[0]
        self._rebuild_playlist_ui()
        self._play_queued_item()

    def _play_queued_item(self):
        """Advance through _tts_queue and generate each item."""
        if not getattr(self, "_tts_queue", None):
            self._is_playing = False
            self._current_playing = -1
            self._show_convert_btns(False)
            self._rebuild_playlist_ui()
            self.tts_status.configure(text=t("SPEECH_LAB_STATUS_DONE"))
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
            messagebox.showinfo("KoKoFish", t("SPEECH_LAB_MSG_NO_AUDIO_SEL"))
            return
        self._stop_preview()
        self._preview_queue = list(ready)
        self._show_play_btns(True)
        self._run_preview_queue()

    def _run_preview_queue(self):
        if not getattr(self, "_preview_queue", None):
            self._preview_idx = -1
            self._show_play_btns(False)
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
            self.tts_status.configure(text=t("SPEECH_LAB_STATUS_PLAYLIST_COMPLETE"))
            return

        item = self._playlist_items[self._current_playing]

        # ── Preprocessing: Translate → Assisted Flow (chain in one thread) ─
        _is_kokoro_mode = getattr(self.settings, 'engine', 'kokoro') == 'kokoro'
        _needs_af       = item.get("assisted_flow", False)

        # Resolve target language and translation need now (on main thread)
        if _is_kokoro_mode:
            # Kokoro: only translate when the per-item toggle is ON and voice is non-English.
            # The LLM prompt auto-detects source language and skips if already correct.
            _item_voice_id = KOKORO_VOICES.get(item.get("voice", ""), "")
            _target_lang   = KOKORO_VOICE_LANG.get(_item_voice_id, "English")
            _needs_translate = item.get("translate", False) and _target_lang != "English"
            _tone = "Natural"
        else:
            _needs_translate = item.get("translate", False)
            _target_lang     = item.get("translate_lang", "") or "Japanese"
            _tone            = item.get("translate_tone", "Natural")
            _needs_translate = _needs_translate and _target_lang != "English"

        if _text_override is None and (_needs_translate or _needs_af):
            from tag_suggester import is_llm_available, is_qwen_model_ready
            if not is_llm_available():
                self.root.after(0, lambda: self.tts_status.configure(
                    text=t("SPEECH_LAB_WARN_NO_LLM"),
                    text_color=COLORS["warning"],
                ))
            elif not is_qwen_model_ready():
                self.root.after(0, lambda: self.tts_status.configure(
                    text=t("SPEECH_LAB_WARN_NO_MODEL"),
                    text_color=COLORS["warning"],
                ))
            if is_llm_available() and is_qwen_model_ready():
                self._rebuild_playlist_ui()
                _pre_engine = getattr(self.settings, 'engine', 'kokoro')

                _global_style = getattr(self._content_style_var, "get", lambda: "None")()

                def _preprocess(
                    _eng=_pre_engine,
                    _lang=_target_lang,
                    _do_tr=_needs_translate,
                    _do_af=_needs_af,
                    _tr_tone=_tone,
                    _style=_global_style,
                ):
                    current_text = item["text"]

                    # ── Step 1: Translate ────────────────────────────────
                    if _do_tr:
                        try:
                            from tag_suggester import translate_for_voice
                            self.root.after(0, lambda l=_lang: self.tts_status.configure(
                                text=t("SPEECH_LAB_STATUS_TRANSLATING", name=item['name'], lang=l)
                            ))
                            translated = translate_for_voice(current_text, _lang, tone=_tr_tone, content_style=_style)
                            if translated and translated.strip():
                                current_text = translated
                        except Exception as exc:
                            logger.warning("Translate step failed: %s", exc)

                    # ── Step 2: Assisted Flow ────────────────────────────
                    if _do_af:
                        try:
                            from tag_suggester import enhance_for_tts
                            self.root.after(0, lambda: self.tts_status.configure(
                                text=t("SPEECH_LAB_STATUS_AI_FLOW", name=item['name'])
                            ))
                            enhanced = enhance_for_tts(current_text, engine=_eng, content_style=_style)
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

        # ── Free VRAM before GPU-heavy TTS engines ───────────────────────
        # VoxCPM 2B and OmniVoice can use significant VRAM; unload the LLM
        # first if it was loaded onto the GPU. It will reload automatically
        # on the next Prompt Lab / Translate / Assisted-Flow request.
        _engine_now = getattr(self.settings, "engine", "kokoro")
        if _engine_now in ("voxcpm_2b", "omnivoice") and getattr(self.settings, "use_cuda", False):
            try:
                from tag_suggester import is_llm_on_gpu, unload_llm
                if is_llm_on_gpu():
                    logger.info("Unloading LLM from VRAM before %s TTS generation", _engine_now)
                    unload_llm()
            except Exception as _ue:
                logger.warning("LLM VRAM unload failed: %s", _ue)

        self.tts_status.configure(text=t("SPEECH_LAB_STATUS_GENERATING", name=item['name']))
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
        self._tts_gen_queue  = _sample_queue
        self._tts_gen_active = True

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
            if not self._tts_gen_active:
                return  # cancelled — discard chunk
            if self.silent_mode_var.get():
                return  # Work Silent: generate only, no output
            if _stream[0] is None:
                s = _sd.OutputStream(
                    samplerate=sr, channels=1, dtype='float32',
                    blocksize=2048, callback=_stream_callback,
                )
                s.start()
                _stream[0] = s
                self._tts_gen_stream = s
            _sample_queue.put(chunk_np.astype(_np.float32))

        def on_complete(wav_path):
            # Store for preview button and manual Save MP3
            item["audio_path"] = wav_path
            self._last_wav_path = wav_path
            _sample_queue.put(_SENTINEL)
            def _finish():
                import time as _t
                if not self._tts_gen_active:
                    # Cancelled while finishing — just clean up and return
                    while not _sample_queue.empty():
                        try: _sample_queue.get_nowait()
                        except Exception: break
                    return
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
                    while not _sample_queue.empty() and self._tts_gen_active:
                        _t.sleep(0.1)
                    # Give it a moment to finish playing the final buffer
                    if self._tts_gen_active:
                        _t.sleep(0.5)

                if _stream[0]:
                    try:
                        _stream[0].stop()
                        _stream[0].close()
                    except Exception:
                        pass
                self._tts_gen_stream = None
                self._tts_gen_active = False

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
                            text=t("SPEECH_LAB_STATUS_AUTO_SAVED", stem=stem)
                        ))
                    except Exception as _e:
                        logger.error("Auto-save failed: %s", _e)
                        self.root.after(0, lambda e=_e: self.tts_status.configure(
                            text=t("SPEECH_LAB_STATUS_AUTO_SAVE_FAILED", error=e)
                        ))
                else:
                    self.root.after(0, lambda: self.tts_status.configure(
                        text=t("SPEECH_LAB_STATUS_DONE_ITEM", name=self._playlist_items[self._current_playing]['name'])
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
            self._tts_gen_stream = None
            self._tts_gen_active = False
            self.root.after(0, lambda: self.tts_status.configure(text=t("SPEECH_LAB_STATUS_ERROR", error=exc)))

        # Route to correct engine based on settings
        _is_kokoro_mode = getattr(self.settings, 'engine', 'kokoro') == 'kokoro'

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
            self.tts_status.configure(text=t("SPEECH_LAB_STATUS_PLAYING", name=self._playlist_items[self._current_playing]['name']))

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
            self.tts_status.configure(text=t("SPEECH_LAB_STATUS_PLAYBACK_ERROR", error=exc))

    def _resume_audio(self):
        """Resume paused audio playback."""
        try:
            import sounddevice as sd
            # sounddevice doesn't support true pause/resume easily,
            # so this is a simplified implementation
            self.tts_status.configure(text=t("SPEECH_LAB_STATUS_RESUMED"))
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
        self.tts_status.configure(text=t("SPEECH_LAB_STATUS_PAUSED"))

    def _tts_stop(self):
        """Stop playback, cancel any pending generation, and clear the queue."""
        self._stop_preview()
        self._is_playing = False
        self._is_paused = False
        self._current_playing = -1
        self._single_item_queue.clear()
        self._show_convert_btns(False)
        self.tts.cancel()
        try:
            import sounddevice as sd
            sd.stop()
        except Exception:
            pass
        self.tts_progress.set(0)
        self.tts_status.configure(text=t("SPEECH_LAB_STATUS_STOPPED"))
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
            messagebox.showinfo(t("SPEECH_LAB_MSG_SAVE_SEL_TITLE"), t("SPEECH_LAB_MSG_SAVE_SEL_NO_AUDIO"), parent=self.root)
            return

        dest_dir = filedialog.askdirectory(
            title=t("SPEECH_LAB_DIALOG_SAVE_FOLDER", count=len(ready)),
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
                t("SPEECH_LAB_MSG_SAVE_SEL_TITLE"),
                t("SPEECH_LAB_MSG_SAVE_SEL_ERRORS", saved=saved, total=len(ready), errors="\n".join(errors)),
                parent=self.root,
            )
        else:
            self.tts_status.configure(text=t("SPEECH_LAB_STATUS_SAVED_BATCH", count=saved, folder=os.path.basename(dest_dir)))

    def _tts_export_audiobook(self):
        """Merge selected items with audio into a single M4B audiobook with chapter marks."""
        ready = [
            it for it in self._playlist_items
            if it.get("selected", True)
            and it.get("audio_path")
            and os.path.isfile(it["audio_path"])
        ]
        if not ready:
            messagebox.showinfo(
                t("SPEECH_LAB_MSG_EXPORT_AB_TITLE"),
                t("SPEECH_LAB_MSG_EXPORT_AB_NO_AUDIO"),
                parent=self.root,
            )
            return

        if not is_ffmpeg_available():
            messagebox.showerror(
                t("SPEECH_LAB_MSG_EXPORT_AB_TITLE"),
                t("SPEECH_LAB_MSG_EXPORT_AB_NO_FFMPEG"),
                parent=self.root,
            )
            return

        out_path = filedialog.asksaveasfilename(
            title=t("SPEECH_LAB_DIALOG_SAVE_AUDIOBOOK"),
            defaultextension=".m4b",
            filetypes=[
                (t("SPEECH_LAB_DIALOG_AUDIOBOOK_FT_M4B"), "*.m4b"),
                (t("SPEECH_LAB_DIALOG_AUDIOBOOK_FT_MP4"), "*.mp4"),
                (t("SPEECH_LAB_DIALOG_AUDIOBOOK_FT_WAV"), "*.wav"),
                (t("SPEECH_LAB_DIALOG_AUDIOBOOK_FT_MP3"), "*.mp3"),
            ],
            initialfile="audiobook.m4b",
            parent=self.root,
        )
        if not out_path:
            return

        self.tts_status.configure(text=t("SPEECH_LAB_STATUS_BUILDING_AUDIOBOOK"))
        self.root.update_idletasks()

        def _build():
            import subprocess, tempfile

            tmp_dir = tempfile.mkdtemp(prefix="kokofish_ab_")
            try:
                # ── 1. Normalise all inputs to wav (ffmpeg handles wav/mp3) ──
                wav_files = []
                durations_ms = []
                for idx, it in enumerate(ready):
                    src = it["audio_path"]
                    norm_wav = os.path.join(tmp_dir, f"track_{idx:04d}.wav")
                    subprocess.run(
                        ["ffmpeg", "-y", "-i", src, "-ar", "44100", "-ac", "2", norm_wav],
                        check=True,
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                        creationflags=_NO_WIN,
                    )
                    # Get duration via ffprobe
                    probe = subprocess.run(
                        ["ffprobe", "-v", "error", "-show_entries",
                         "format=duration", "-of", "default=noprint_wrappers=1:nokey=1", norm_wav],
                        capture_output=True, text=True, creationflags=_NO_WIN,
                    )
                    dur_s = float(probe.stdout.strip() or "0")
                    durations_ms.append(int(dur_s * 1000))
                    wav_files.append(norm_wav)

                # ── 2. Write FFmpeg concat list ──────────────────────────────
                concat_txt = os.path.join(tmp_dir, "concat.txt")
                with open(concat_txt, "w", encoding="utf-8") as f:
                    for w in wav_files:
                        # FFmpeg concat list: paths use forward slashes, single-quoted
                        safe = w.replace("\\", "/").replace("'", "'\\''")
                        f.write(f"file '{safe}'\n")

                # ── 3. Write FFmpeg chapter metadata ─────────────────────────
                meta_txt = os.path.join(tmp_dir, "chapters.txt")
                with open(meta_txt, "w", encoding="utf-8") as f:
                    f.write(";FFMETADATA1\ntitle=Audiobook\nartist=KoKoFish\n\n")
                    cursor_ms = 0
                    for it, dur_ms in zip(ready, durations_ms):
                        title = os.path.splitext(it["name"])[0]
                        f.write("[CHAPTER]\n")
                        f.write("TIMEBASE=1/1000\n")
                        f.write(f"START={cursor_ms}\n")
                        f.write(f"END={cursor_ms + dur_ms}\n")
                        f.write(f"title={title}\n\n")
                        cursor_ms += dur_ms

                # ── 4. Concat + embed chapters + encode ──────────────────────
                ext = os.path.splitext(out_path)[1].lower()
                if ext == ".wav":
                    # WAV: simple concat, no chapter metadata, no re-encode
                    cmd = [
                        "ffmpeg", "-y",
                        "-f", "concat", "-safe", "0", "-i", concat_txt,
                        "-c", "copy", out_path,
                    ]
                elif ext == ".mp3":
                    # Embed ID3v2 chapter tags so data survives a later re-encode
                    cmd = [
                        "ffmpeg", "-y",
                        "-f", "concat", "-safe", "0", "-i", concat_txt,
                        "-i", meta_txt,
                        "-map_metadata", "1",
                        "-c:a", "libmp3lame", "-b:a", "192k",
                        "-id3v2_version", "3",
                        out_path,
                    ]
                else:
                    # M4B / MP4: chapters + AAC
                    cmd = [
                        "ffmpeg", "-y",
                        "-f", "concat", "-safe", "0", "-i", concat_txt,
                        "-i", meta_txt,
                        "-map_metadata", "1",
                        "-c:a", "aac", "-b:a", "128k",
                        "-movflags", "+faststart",
                        out_path,
                    ]
                subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, creationflags=_NO_WIN)

                total_s = sum(durations_ms) / 1000
                m, s = divmod(int(total_s), 60)
                h, m = divmod(m, 60)
                dur_str = f"{h}h {m}m {s}s" if h else f"{m}m {s}s"
                self.root.after(0, lambda: self.tts_status.configure(
                    text=t("SPEECH_LAB_STATUS_AUDIOBOOK_SAVED", chapters=len(ready), duration=dur_str)
                ))

            except Exception as exc:
                logger.error("Audiobook export failed: %s", exc)
                self.root.after(0, lambda e=str(exc): messagebox.showerror(
                    t("SPEECH_LAB_MSG_EXPORT_AB_FAILED_TITLE"), e, parent=self.root
                ))
                self.root.after(0, lambda: self.tts_status.configure(text=t("SPEECH_LAB_STATUS_AUDIOBOOK_FAILED")))
            finally:
                # Clean up temp files
                import shutil
                try:
                    shutil.rmtree(tmp_dir, ignore_errors=True)
                except Exception:
                    pass

        threading.Thread(target=_build, daemon=True, name="AudiobookExport").start()

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
                ("Audio files", "*.wav *.mp3 *.m4a *.flac *.ogg *.weba *.webm *.opus *.wma *.amr *.aac"),
                ("All files", "*.*"),
            ],
        )
        if path:
            self._stt_set_file(path)

    def _stt_set_file(self, path: str):
        """Set the audio file for transcription."""
        ext = os.path.splitext(path)[1].lower()
        if ext not in (".wav", ".mp3", ".m4a", ".flac", ".ogg", ".weba", ".webm", ".opus", ".wma", ".amr", ".aac"):
            messagebox.showwarning("KoKoFish", t("STT_MSG_UNSUPPORTED_FORMAT", ext=ext))
            return
        self._stt_audio_path = path
        name = os.path.basename(path)
        self.stt_file_label.configure(text=f"📎  {name}", text_color=COLORS["success"])
        self.stt_drop_label.configure(text=f"✅  {name}")

    def _stt_transcribe(self):
        """Start transcription."""
        if not self._stt_audio_path:
            messagebox.showinfo("KoKoFish", t("TEXT_LAB_MSG_NO_FILE"))
            return

        if not self.stt.is_loaded:
            # Load model first
            device = "cuda" if self.cuda_var.get() else "cpu"
            compute_type = "float16" if device == "cuda" else "int8"
            model_size = self.stt_model_var.get()

            self.stt_textbox.configure(state="normal")
            self.stt_textbox.delete("1.0", "end")
            self.stt_textbox.insert("1.0", t("TEXT_LAB_STATUS_LOADING_MODEL") + "\n")
            self.stt_textbox.configure(state="disabled")

            def on_ready():
                self.root.after(0, self._stt_run_transcription)
                self.root.after(
                    0,
                    lambda: self.update_stt_status(t("TEXT_LAB_STATUS_LOADED"), COLORS["success"])
                )

            def on_error(exc):
                self.root.after(
                    0,
                    lambda: self._stt_append_text(t("TEXT_LAB_STATUS_ERROR_LOADING", error=exc))
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
            if getattr(self, "_stt_timestamps_var", None) and self._stt_timestamps_var.get():
                line = f"[{start:.1f}s → {end:.1f}s]  {text}\n"
            else:
                line = f"{text} "
            self.root.after(0, lambda l=line: self._stt_append_text(l))

        def on_progress(frac):
            self.root.after(0, lambda: self.stt_progress.set(frac))

        def on_complete(full_text, info):
            self.root.after(0, lambda: self.stt_progress.set(1.0))
            lang = info.get("language", "unknown")
            prob = info.get("language_probability", 0.0) * 100
            self.root.after(
                0,
                lambda: self._stt_append_text(
                    t("TEXT_LAB_STATUS_DONE", lang=lang, prob=f"{prob:.0f}")
                )
            )

        def on_error(exc):
            self.root.after(
                0,
                lambda: self._stt_append_text(t("TEXT_LAB_STATUS_TRANSCRIBE_ERROR", error=exc))
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
        self._stt_append_text(t("TEXT_LAB_STATUS_CANCELLED"))

    def _stt_export(self, fmt: str):
        """Export transcription to a file."""
        self.stt_textbox.configure(state="normal")
        text = self.stt_textbox.get("1.0", "end").strip()
        self.stt_textbox.configure(state="disabled")

        if not text or text == t("TEXT_LAB_TEXTBOX_PLACEHOLDER"):
            messagebox.showinfo("KoKoFish", t("TEXT_LAB_MSG_NO_TRANSCRIPT_EXPORT"))
            return

        exts   = {"txt": ".txt", "docx": ".docx", "pdf": ".pdf", "epub": ".epub"}
        ftypes = {
            "txt":  [("Text files",      "*.txt")],
            "docx": [("Word documents",  "*.docx")],
            "pdf":  [("PDF files",       "*.pdf")],
            "epub": [("EPUB ebooks",     "*.epub")],
        }

        path = filedialog.asksaveasfilename(
            title=t("TEXT_LAB_DIALOG_SAVE_AS", fmt=fmt.upper()),
            defaultextension=exts[fmt],
            filetypes=ftypes[fmt],
        )
        if not path:
            return

        try:
            if fmt == "epub":
                # Strip timestamps — they're meaningless in an ebook reader
                clean = re.sub(r'\[\d+\.\d+s\s*→\s*\d+\.\d+s\]\s*', '', text)
                clean = re.sub(r'--- .* ---\n?', '', clean).strip()
                from utils import export_epub
                stem = os.path.splitext(os.path.basename(
                    self._stt_audio_path or "transcription"
                ))[0]
                export_epub(clean, path, title=stem)
            else:
                {"txt": export_txt, "docx": export_docx, "pdf": export_pdf}[fmt](text, path)
            messagebox.showinfo("KoKoFish", t("TEXT_LAB_MSG_SAVED_TO", path=path))
        except Exception as exc:
            messagebox.showerror(t("COMMON_ERROR"), t("TEXT_LAB_MSG_EXPORT_FAILED", error=exc))

    def _stt_strip_timestamps(self, text: str) -> str:
        """Remove [X.Xs → Y.Ys] markers and status lines from a transcript."""
        clean = re.sub(r'\[\d+\.\d+s\s*[→>]\s*\d+\.\d+s\]\s*', '', text)
        clean = re.sub(r'---[^\n]*---\n?', '', clean)
        clean = re.sub(r'\n{3,}', '\n\n', clean)
        return clean.strip()

    def _stt_translate(self):
        """Translate the current transcript and show a comparison dialog."""
        self.stt_textbox.configure(state="normal")
        text = self.stt_textbox.get("1.0", "end").strip()
        self.stt_textbox.configure(state="disabled")

        if not text or text == t("TEXT_LAB_TEXTBOX_PLACEHOLDER"):
            messagebox.showinfo("KoKoFish", t("TEXT_LAB_MSG_NO_TRANSCRIPT_TRANSLATE"))
            return

        from tag_suggester import is_llm_available, is_qwen_model_ready
        if not is_llm_available():
            messagebox.showerror("KoKoFish", t("SPEECH_LAB_WARN_NO_LLM"))
            return
        if not is_qwen_model_ready():
            messagebox.showerror("KoKoFish", t("SPEECH_LAB_WARN_NO_MODEL"))
            return

        clean = self._stt_strip_timestamps(text)
        lang  = _en_from_display(_LANG_KEY_MAP, self._stt_translate_lang_var.get())
        self._stt_translate_btn.configure(state="disabled", text=t("TEXT_LAB_STATUS_TRANSLATING"))

        def _run():
            try:
                from tag_suggester import translate_for_voice
                translated = translate_for_voice(clean, lang)
                self.root.after(0, lambda: _show(translated))
            except Exception as exc:
                self.root.after(0, lambda: messagebox.showerror(t("TEXT_LAB_MSG_TRANSLATE_FAILED_TITLE"), str(exc)))
            finally:
                self.root.after(0, lambda: self._stt_translate_btn.configure(
                    state="normal", text=t("TEXT_LAB_BTN_TRANSLATE")
                ))

        def _show(translated):
            self._stt_show_translate_dialog(clean, translated, lang)

        threading.Thread(target=_run, daemon=True, name="STT-Translate").start()

    def _stt_show_translate_dialog(self, original: str, translated: str, lang: str):
        """Show side-by-side original vs translated transcript with save/send options."""
        win = ctk.CTkToplevel(self.root)
        win.title(t("STT_TR_DIALOG_TITLE", lang=lang))
        win.geometry("1060x640")
        win.configure(fg_color=COLORS["bg_dark"])
        win.grab_set()

        ctk.CTkLabel(
            win,
            text=t("STT_TR_DIALOG_HEADING", lang=lang),
            font=(FONT_FAMILY, 16, "bold"),
            text_color=COLORS["accent_light"],
        ).pack(pady=(15, 5))

        # Two-column area
        cols = ctk.CTkFrame(win, fg_color="transparent")
        cols.pack(fill="both", expand=True, padx=15, pady=5)

        for title, content, editable in [
            (t("STT_TR_COLUMN_ORIGINAL"), original, False),
            (t("STT_TR_COLUMN_TRANSLATED", lang=lang), translated, True),
        ]:
            card = ctk.CTkFrame(cols, fg_color=COLORS["bg_card"], corner_radius=8)
            card.pack(side="left", fill="both", expand=True, padx=6)
            ctk.CTkLabel(
                card, text=title,
                font=(FONT_FAMILY, 12, "bold"),
                text_color=COLORS["text_secondary"],
            ).pack(anchor="w", padx=10, pady=(8, 0))
            box = ctk.CTkTextbox(
                card, fg_color=COLORS["bg_input"],
                text_color=COLORS["text_primary"],
                font=(FONT_FAMILY, 12), wrap="word",
            )
            box.pack(fill="both", expand=True, padx=8, pady=8)
            box.insert("1.0", content)
            if not editable:
                box.configure(state="disabled")
            if editable:
                self._stt_tr_translated_box = box  # keep ref for save

        # Bottom bar
        btns = ctk.CTkFrame(win, fg_color="transparent")
        btns.pack(fill="x", padx=15, pady=(5, 15))

        stem = os.path.splitext(os.path.basename(
            self._stt_audio_path or "transcript"
        ))[0]
        save_dir = os.path.dirname(self._stt_audio_path) if self._stt_audio_path else APP_DIR

        def _save_translated():
            tr_text = self._stt_tr_translated_box.get("1.0", "end").strip()
            path = filedialog.asksaveasfilename(
                title=t("STT_TR_SAVE_DIALOG_TITLE"),
                defaultextension=".txt",
                initialfile=f"{stem} (Original-{lang}).txt",
                filetypes=[("Text files", "*.txt")],
                parent=win,
            )
            if not path:
                return
            with open(path, "w", encoding="utf-8") as f:
                f.write(tr_text)
            messagebox.showinfo(t("STT_TR_SAVED_TITLE"), t("STT_TR_SAVED_BODY", path=path), parent=win)

        def _send(content, suffix):
            filename = f"{stem}{suffix}.txt"
            path = os.path.join(save_dir, filename)
            try:
                with open(path, "w", encoding="utf-8") as f:
                    f.write(content)
                self._tts_add_file(path)
                self.tabview.set("🔊  Speech Lab")
                win.destroy()
            except Exception as exc:
                messagebox.showerror("Error", t("STT_TR_SEND_ERROR", error=exc), parent=win)

        _btn = {"font": (FONT_FAMILY, 12), "corner_radius": 8, "height": 34}

        ctk.CTkButton(
            btns, text=t("STT_TR_BTN_SAVE"), width=160,
            fg_color=COLORS["bg_input"], hover_color=COLORS["bg_card_hover"],
            border_color=COLORS["border"], border_width=1,
            command=_save_translated, **_btn,
        ).pack(side="left", padx=(0, 8))

        ctk.CTkButton(
            btns, text=t("STT_TR_BTN_SEND_ORIGINAL"), width=220,
            fg_color=COLORS["bg_input"], hover_color=COLORS["bg_card_hover"],
            border_color=COLORS["border"], border_width=1,
            command=lambda: _send(original, ""),
            **_btn,
        ).pack(side="left", padx=(0, 8))

        ctk.CTkButton(
            btns, text=t("STT_TR_BTN_SEND_TRANSLATED"), width=240,
            fg_color=COLORS["success"], hover_color="#05b886",
            text_color="#0a0a18",
            command=lambda: _send(
                self._stt_tr_translated_box.get("1.0", "end").strip(),
                f" (Original-{lang})",
            ),
            **_btn,
        ).pack(side="left", padx=(0, 8))

        ctk.CTkButton(
            btns, text=t("COMMON_BTN_CLOSE"), width=90,
            fg_color=COLORS["bg_input"], hover_color=COLORS["bg_card_hover"],
            border_color=COLORS["border"], border_width=1,
            command=win.destroy, **_btn,
        ).pack(side="right")

    def _stt_send_to_speech_lab(self):
        """Save clean transcript (no timestamps) and add it to the Speech Lab playlist."""
        self.stt_textbox.configure(state="normal")
        text = self.stt_textbox.get("1.0", "end").strip()
        self.stt_textbox.configure(state="disabled")

        if not text or text == t("TEXT_LAB_TEXTBOX_PLACEHOLDER"):
            messagebox.showinfo("KoKoFish", t("TEXT_LAB_MSG_NO_TRANSCRIPT_SEND"))
            return

        clean = self._stt_strip_timestamps(text)
        stem  = os.path.splitext(os.path.basename(
            self._stt_audio_path or "transcript"
        ))[0]
        save_dir = os.path.dirname(self._stt_audio_path) if self._stt_audio_path else APP_DIR
        path = os.path.join(save_dir, f"{stem}.txt")

        try:
            with open(path, "w", encoding="utf-8") as f:
                f.write(clean)
            self._tts_add_file(path)
            self.tabview.set(t("MAIN_TAB_SPEECH_LAB"))
        except Exception as exc:
            messagebox.showerror(t("COMMON_ERROR"), t("TEXT_LAB_MSG_SEND_FAILED", error=exc))

    def _open_prompt_editor(self):
        """Open a window to view and edit all Qwen AI prompts."""
        from tag_suggester import get_prompt, set_prompt, save_prompts, reset_prompts

        _PROMPT_TABS = [
            ("Script → Kokoro",   "script_kokoro"),
            ("Script → Fish",     "script_fish"),
            ("Kokoro Flow",       "kokoro_af"),
            ("Translation",       "translate"),
            ("Tone Rewrite",      "tone"),
            ("Tag Gen",           "tag_gen"),
            ("Grammar",           "grammar"),
        ]

        win = ctk.CTkToplevel(self.root)
        win.title(t("PROMPT_EDITOR_TITLE"))
        win.geometry("980x620")
        win.resizable(True, True)
        win.transient(self.root)
        win.configure(fg_color=COLORS["bg_dark"])

        ctk.CTkLabel(
            win,
            text=t("PROMPT_EDITOR_TITLE"),
            font=(FONT_FAMILY, 15, "bold"),
            text_color=COLORS["text_primary"],
        ).pack(anchor="w", padx=20, pady=(16, 4))

        ctk.CTkLabel(
            win,
            text=t("PROMPT_EDITOR_HINT"),
            font=(FONT_FAMILY, 10),
            text_color=COLORS["text_muted"],
            justify="left",
        ).pack(anchor="w", padx=20, pady=(0, 10))

        tabs = ctk.CTkTabview(
            win,
            fg_color=COLORS["bg_card"],
            segmented_button_fg_color=COLORS["bg_input"],
            segmented_button_selected_color="#5a3e8a",
            segmented_button_selected_hover_color="#7b5ea7",
            segmented_button_unselected_color=COLORS["bg_input"],
            segmented_button_unselected_hover_color=COLORS["bg_card_hover"],
            text_color=COLORS["text_primary"],
        )
        tabs.pack(fill="both", expand=True, padx=20, pady=(0, 8))

        _textboxes: dict = {}
        for label, key in _PROMPT_TABS:
            tab = tabs.add(label)
            tb = ctk.CTkTextbox(
                tab,
                fg_color=COLORS["bg_input"],
                text_color=COLORS["text_primary"],
                font=("Consolas", 11),
                corner_radius=6,
                border_color=COLORS["border"],
                border_width=1,
                wrap="word",
            )
            tb.pack(fill="both", expand=True, padx=8, pady=8)
            tb.insert("1.0", get_prompt(key))
            _textboxes[key] = tb

        btn_row = ctk.CTkFrame(win, fg_color="transparent")
        btn_row.pack(fill="x", padx=20, pady=(0, 16))

        def _save():
            for key, tb in _textboxes.items():
                set_prompt(key, tb.get("1.0", "end-1c"))
            save_prompts()
            self.tts_status.configure(text=t("PROMPT_EDITOR_STATUS_SAVED"))

        def _reset():
            from tkinter import messagebox as _mb
            if not _mb.askyesno("Reset Prompts", "Reset ALL prompts to factory defaults?\nThis cannot be undone.", parent=win):
                return
            reset_prompts()
            for key, tb in _textboxes.items():
                tb.delete("1.0", "end")
                tb.insert("1.0", get_prompt(key))
            self.tts_status.configure(text=t("PROMPT_EDITOR_STATUS_RESET"))

        ctk.CTkButton(
            btn_row, text=t("PROMPT_EDITOR_BTN_SAVE"),
            fg_color="#5a3e8a", hover_color="#7b5ea7",
            font=(FONT_FAMILY, 12, "bold"), height=32, width=120,
            command=_save,
        ).pack(side="left", padx=(0, 8))

        ctk.CTkButton(
            btn_row, text=t("PROMPT_EDITOR_BTN_RESET"),
            fg_color=COLORS["bg_input"], hover_color=COLORS["bg_card_hover"],
            border_color=COLORS["border"], border_width=1,
            font=(FONT_FAMILY, 12), height=32, width=160,
            command=_reset,
        ).pack(side="left")

    def _on_stt_model_change(self, value):
        """Handle Whisper model size change."""
        self.settings.whisper_model_size = value
        # Force model reload on next transcription
        if self.stt.is_loaded:
            self.stt.unload_model()
            self.update_stt_status("⏳  Model changed — will reload on next use", COLORS["warning"])

    # ==================================================================
    # EVENT HANDLERS — Voice Lab
    # ==================================================================

    def _voice_clone(self):
        """Open dialog to clone a new voice."""
        path = filedialog.askopenfilename(
            title=t("VOICE_LAB_DIALOG_SELECT_AUDIO"),
            filetypes=[
                ("Audio files", "*.wav *.mp3 *.flac *.ogg *.m4a *.weba *.webm *.opus *.wma *.amr *.aac"),
                ("All files", "*.*"),
            ],
        )
        if not path:
            return

        # Ask for voice name
        dialog = ctk.CTkInputDialog(
            text=t("VOICE_LAB_DIALOG_ENTER_NAME"),
            title=t("VOICE_LAB_DIALOG_CLONE_TITLE"),
        )
        name = dialog.get_input()
        if not name or not name.strip():
            return

        name = name.strip()
        if self.voices.voice_exists(name):
            messagebox.showwarning("KoKoFish", t("VOICE_LAB_MSG_ALREADY_EXISTS", name=name))
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
                    "KoKoFish", t("VOICE_LAB_MSG_CLONE_SUCCESS", name=name)
                    + (f"\n\nTranscript saved:\n\"{transcript[:120]}{'…' if len(transcript) > 120 else ''}\"" if transcript else "")
                ))
            except Exception as exc:
                self.root.after(0, lambda e=exc: messagebox.showerror(
                    t("COMMON_ERROR"), t("VOICE_LAB_MSG_CLONE_FAILED", error=e)
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
            text=t("VOICE_LAB_STATUS_TRANSCRIBING")
        ))
        threading.Thread(target=_transcribe_and_clone, daemon=True, name="VoiceClone").start()

    def _voice_delete(self, name: str):
        """Delete a voice profile."""
        confirm = messagebox.askyesno(
            t("VOICE_LAB_DIALOG_DELETE_TITLE"),
            t("VOICE_LAB_MSG_DELETE_CONFIRM", name=name)
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
        _start_time = [time.time()]
        _timer_id   = [None]

        def _tick():
            """Update elapsed time every second while the test runs."""
            elapsed = int(time.time() - _start_time[0])
            self.update_tts_status(
                t("VOICE_LAB_STATUS_TESTING", name=name, elapsed=elapsed),
                COLORS["warning"],
            )
            _timer_id[0] = self.root.after(1000, _tick)

        def _stop_timer():
            if _timer_id[0]:
                self.root.after_cancel(_timer_id[0])
                _timer_id[0] = None

        self.update_tts_status(t("VOICE_LAB_STATUS_TESTING", name=name, elapsed=0), COLORS["warning"])
        _timer_id[0] = self.root.after(1000, _tick)

        def on_complete(wav_path):
            self.root.after(0, _stop_timer)
            self.root.after(0, lambda: self.update_tts_status(
                t("VOICE_LAB_STATUS_TEST_COMPLETE"), COLORS["success"]
            ))
            def _play():
                try:
                    import soundfile as sf_lib
                    import sounddevice as sd
                    data, sr = sf_lib.read(wav_path)
                    sd.stop()
                    sd.play(data, sr)
                    sd.wait()
                    self.root.after(0, lambda: self.update_tts_status(
                        t("SPEECH_LAB_STATUS_ENGINE_READY"), COLORS["success"]
                    ))
                except Exception as e:
                    logger.error("Voice test playback error: %s", e)
            threading.Thread(target=_play, daemon=True).start()

        def on_error(exc):
            self.root.after(0, _stop_timer)
            self.root.after(0, lambda: self.update_tts_status(
                t("SPEECH_LAB_STATUS_ENGINE_READY"), COLORS["success"]
            ))
            self.root.after(0, lambda: messagebox.showerror(
                t("VOICE_LAB_MSG_TEST_FAILED_TITLE"), t("VOICE_LAB_MSG_TEST_FAILED_BODY", error=exc)
            ))

        ref_wav     = profile.get("wav_path")
        tokens_path = profile.get("tokens_path")
        ref_tokens  = np.load(tokens_path) if tokens_path and os.path.isfile(tokens_path) else None

        self.tts.generate(
            text=test_text,
            reference_wav=ref_wav,
            reference_tokens=ref_tokens,
            prompt_text=profile.get("prompt_text", ""),
            on_complete=on_complete,
            on_error=on_error,
        )

    # ==================================================================
    # EVENT HANDLERS — Settings
    # ==================================================================

    def _on_cuda_toggle(self):
        """Handle CUDA toggle — download CUDA PyTorch on demand."""
        wants_cuda = self.cuda_var.get()
        _is_kokoro = getattr(self.settings, "engine", "kokoro") == "kokoro"

        def _swap_ort(use_cuda: bool):
            """Switch onnxruntime ↔ onnxruntime-gpu for the Kokoro engine."""
            if not _is_kokoro:
                return
            from kokoro_engine import switch_onnxruntime
            def _ort_progress(msg):
                self.root.after(0, lambda m=msg: self.cuda_status_label.configure(
                    text=f"⏳  {m}", text_color=COLORS["warning"]
                ))
            def _ort_done(ok, msg):
                logger.info("onnxruntime swap: %s", msg)
            switch_onnxruntime(use_cuda, on_progress=_ort_progress, on_complete=_ort_done)

        if wants_cuda and not is_cuda_torch_installed():
            # User is enabling CUDA but doesn't have CUDA PyTorch yet
            if not has_nvidia_gpu():
                messagebox.showwarning(
                    "KoKoFish",
                    t("SETTINGS_CUDA_NO_GPU_TITLE")
                )
                self.cuda_var.set(False)
                return

            gpu_name = get_nvidia_gpu_name()
            confirm = messagebox.askyesno(
                t("SETTINGS_CUDA_DOWNLOAD_TITLE"),
                t("SETTINGS_CUDA_DOWNLOAD_BODY", gpu_name=gpu_name)
            )

            if not confirm:
                self.cuda_var.set(False)
                return

            # Start download
            self.cuda_switch.configure(state="disabled")
            self.cuda_status_label.configure(
                text=t("SETTINGS_CUDA_DOWNLOADING"),
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
                            text=t("SETTINGS_CUDA_READY", gpu_name=gpu_name),
                            text_color=COLORS["success"],
                        )
                        self.settings.use_cuda = True
                        self.settings.save()
                        # Also swap onnxruntime → onnxruntime-gpu for Kokoro
                        _swap_ort(True)
                        messagebox.showinfo(
                            t("SETTINGS_CUDA_READY_TITLE"),
                            t("SETTINGS_CUDA_READY_BODY", message=message)
                        )
                    else:
                        self.cuda_var.set(False)
                        self.cuda_status_label.configure(
                            text=t("SETTINGS_CUDA_DOWNLOAD_FAILED"),
                            text_color=COLORS["danger"],
                        )
                        messagebox.showerror(t("SETTINGS_CUDA_SETUP_FAILED_TITLE"), message)
                self.root.after(0, _update)

            install_cuda_pytorch(on_progress=on_progress, on_complete=on_complete)

        elif not wants_cuda and is_cuda_torch_installed():
            # User is disabling CUDA — offer to revert to save space
            revert = messagebox.askyesno(
                t("SETTINGS_CUDA_REVERT_TITLE"),
                t("SETTINGS_CUDA_REVERT_BODY")
            )

            self.settings.use_cuda = False
            self.settings.save()
            # Swap onnxruntime-gpu → onnxruntime (CPU) for Kokoro
            _swap_ort(False)

            if revert:
                self.cuda_switch.configure(state="disabled")
                self.cuda_status_label.configure(
                    text=t("SETTINGS_CUDA_REVERTING"),
                    text_color=COLORS["warning"],
                )

                def on_revert_complete(success, message):
                    def _update():
                        self.cuda_switch.configure(state="normal")
                        self.cuda_status_label.configure(
                            text=t("SETTINGS_GPU_REVERTED"),
                            text_color=COLORS["success"],
                        )
                        messagebox.showinfo(t("COMMON_DONE"), message)
                    self.root.after(0, _update)

                revert_to_cpu_pytorch(on_complete=on_revert_complete)
            else:
                self.cuda_status_label.configure(
                    text=t("SETTINGS_GPU_DISABLED"),
                    text_color=COLORS["text_muted"],
                )
        else:
            # Simple toggle (CUDA already installed, or already CPU)
            self.settings.use_cuda = wants_cuda
            self.settings.save()
            # Swap onnxruntime for Kokoro to match new CUDA state
            _swap_ort(wants_cuda)

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
                text=t("VOICE_LAB_LOCKED_MSG", voice_count=len(KOKORO_VOICES), lang_count=len(KOKORO_LANGUAGE_GROUPS)),
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
        from utils import is_kokoro_ready, setup_kokoro

        # Map display label → engine id
        new_engine = engine_id_from_label(new_val)
        if new_engine not in VALID_ENGINES:
            new_engine = "kokoro"

        prev_engine = getattr(self.settings, 'engine', 'kokoro')

        # No change
        if new_engine == prev_engine:
            return

        # Save engine choice immediately
        self.settings.engine = new_engine
        self.settings.save()

        def _revert():
            self.settings.engine = prev_engine
            self.settings.save()
            self._update_engine_dropdown()

        def _on_progress(msg, _frac=None):
            self.root.after(0, lambda m=msg: self.update_tts_status(f"⬇ {m}", COLORS["warning"]))

        def _prompt_restart():
            confirm = messagebox.askyesno(
                t("SETTINGS_RESTART_REQUIRED_TITLE"),
                t("SETTINGS_RESTART_REQUIRED_BODY", engine=new_val),
            )
            if confirm:
                self._restart_app()
            else:
                self._update_engine_dropdown()

        def _run_install(install_fn, ready_msg):
            def _worker():
                try:
                    ok = install_fn()
                except Exception as exc:
                    logger.warning("Engine install failed: %s", exc)
                    ok = False
                if ok:
                    self.root.after(0, lambda: self.update_tts_status(
                        ready_msg, COLORS["success"]
                    ))
                    self.root.after(1500, self._restart_app)
                else:
                    self.root.after(0, lambda: self.update_tts_status(
                        t("SETTINGS_DOWNLOAD_FAILED"), COLORS["danger"]
                    ))
                    self.root.after(0, _revert)
            threading.Thread(target=_worker, daemon=True, name="EngineInstall").start()

        # Kokoro
        if new_engine == "kokoro":
            if not is_kokoro_ready():
                download = messagebox.askyesno(
                    t("SETTINGS_DOWNLOAD_REQUIRED_TITLE"),
                    t("SETTINGS_KOKORO_DOWNLOAD_BODY"),
                )
                if not download:
                    _revert()
                    return
                _run_install(
                    lambda: setup_kokoro(on_progress=_on_progress),
                    t("SETTINGS_KOKORO_DOWNLOAD_READY"),
                )
                return
            _prompt_restart()
            return

        # VoxCPM 0.5B / 2B
        if new_engine in ("voxcpm_05b", "voxcpm_2b"):
            from utils import is_voxcpm_ready, setup_voxcpm
            variant = "0.5B" if new_engine == "voxcpm_05b" else "2B"
            if not is_voxcpm_ready(variant):
                size_txt = "~1 GB" if variant == "0.5B" else "~4 GB"
                body = (
                    f"Download VoxCPM {variant} ({size_txt})? "
                    "This will install the voxcpm package and download model weights."
                )
                download = messagebox.askyesno(
                    t("SETTINGS_DOWNLOAD_REQUIRED_TITLE"),
                    body,
                )
                if not download:
                    _revert()
                    return
                _run_install(
                    lambda v=variant: setup_voxcpm(v, on_progress=_on_progress),
                    f"VoxCPM {variant} ready. Restarting…",
                )
                return
            _prompt_restart()
            return

        # OmniVoice
        if new_engine == "omnivoice":
            from utils import is_omnivoice_ready, setup_omnivoice
            if not is_omnivoice_ready():
                body = (
                    "Download OmniVoice (~2 GB)? "
                    "This will install the omnivoice package and download model weights."
                )
                download = messagebox.askyesno(
                    t("SETTINGS_DOWNLOAD_REQUIRED_TITLE"),
                    body,
                )
                if not download:
                    _revert()
                    return
                _run_install(
                    lambda: setup_omnivoice(on_progress=_on_progress),
                    "OmniVoice ready. Restarting…",
                )
                return
            _prompt_restart()
            return

        # Unknown engine — revert
        _revert()

    def _restart_app(self):
        """Save settings and restart the process."""
        self.settings.save()
        import sys
        os.execv(sys.executable, [sys.executable] + sys.argv)

    def _update_engine_dropdown(self):
        """Revert the engine dropdown to match the saved setting."""
        _eng = getattr(self.settings, 'engine', 'kokoro')
        if _eng not in ENGINE_LABELS:
            _eng = 'kokoro'
        self.engine_var.set(ENGINE_LABELS[_eng])

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
        # Make sure the log row is visible so progress/errors are readable
        if hasattr(self, "_llama_log_row"):
            self._llama_log_row.pack(fill="x", padx=10, pady=2)

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
        """Handle tab switching — refresh Prompt Lab badge, Memory Saver logic."""
        current = self.tabview.get()

        # Refresh the model badge whenever Prompt Lab is opened
        if "Prompt Lab" in current:
            try:
                self._chat_refresh_model_badge()
            except Exception:
                pass

        # Refresh Script Lab voices/profiles when engine has changed since last visit
        if "Script Lab" in current:
            cur_eng = getattr(self.settings, "engine", "kokoro")
            if cur_eng != getattr(self, "_script_last_engine", None):
                self._script_last_engine = cur_eng
                try:
                    self._script_refresh_voices()
                except Exception:
                    pass

        if not self.settings.memory_saver:
            return

        if "Read Aloud" in current:
            # On TTS tab — unload STT, load TTS
            if self.stt.is_loaded:
                self.stt.unload_model()
                self.update_stt_status("💤  Unloaded (Memory Saver)", COLORS["text_muted"])
        elif "Transcribe" in current:
            # On STT tab — unload TTS, keep STT
            if self.tts.is_loaded:
                self.tts.unload_model()
                self.update_tts_status("💤  Unloaded (Memory Saver)", COLORS["text_muted"])

    # ==================================================================
    # RAM monitoring
    # ==================================================================

    def _start_ram_monitor(self):
        """Start periodic RAM usage updates."""
        self._update_ram()

    @staticmethod
    def _lbl_set(label, text, color=None):
        """Configure a label only if its widget still exists."""
        try:
            if label.winfo_exists():
                if color:
                    label.configure(text=text, text_color=color)
                else:
                    label.configure(text=text)
        except Exception:
            pass

    def _update_ram(self):
        """Update RAM and VRAM readout labels."""
        ram_lbl  = getattr(self, "ram_label",  None)
        vram_lbl = getattr(self, "vram_label", None)
        cpu_lbl  = getattr(self, "cpu_label",  None)

        if ram_lbl:
            try:
                ram = get_ram_usage()
                text = (
                    f"App: {ram['process_mb']:.0f} MB  |  "
                    f"System: {ram['system_used_gb']:.1f} / "
                    f"{ram['system_total_gb']:.1f} GB ({ram['system_percent']:.0f}%)"
                )
                self._lbl_set(ram_lbl, text)
            except Exception:
                self._lbl_set(ram_lbl, "Unable to read")

        if vram_lbl:
            try:
                vram = get_vram_usage()
                if vram:
                    text = (
                        f"System: {vram['used_gb']:.1f} / "
                        f"{vram['total_gb']:.1f} GB ({vram['percent']:.0f}%)"
                    )
                    self._lbl_set(vram_lbl, text)
                else:
                    self._lbl_set(vram_lbl, "N/A (No CUDA GPU)")
            except Exception:
                self._lbl_set(vram_lbl, "Unable to read")

        if cpu_lbl:
            try:
                cpu = get_cpu_usage()
                _thread_info = (
                    f"  |  Threads: {cpu['threads_torch']}/{cpu['cores']}"
                    if cpu.get("threads_torch") else f"  |  Cores: {cpu['cores']}"
                )
                _color = (
                    COLORS["danger"] if cpu["percent"] > 90
                    else COLORS["warning"] if cpu["percent"] > 70
                    else COLORS["text_secondary"]
                )
                self._lbl_set(cpu_lbl, f"{cpu['percent']:.0f}%{_thread_info}", _color)
            except Exception:
                self._lbl_set(cpu_lbl, "Unable to read")

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
        """Update TTS engine status label in Settings tab (only if open)."""
        lbl = getattr(self, "tts_status_label", None)
        if lbl:
            self._lbl_set(lbl, text, color or COLORS["text_secondary"])

    def update_stt_status(self, text: str, color: str = None):
        """Update STT engine status label in Settings tab (only if open)."""
        lbl = getattr(self, "stt_status_label", None)
        if lbl:
            self._lbl_set(lbl, text, color or COLORS["text_secondary"])
