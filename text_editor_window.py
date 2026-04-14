"""
FishTalk — Playlist Item Text Editor Window

Opens on double-click of a playlist item.  Provides:
  - Full text editor (edits item["text"] in-memory; original file untouched)
  - Emotion / prosody tag panel (Fish Speech) or info panel (Kokoro)
  - Grammar Check   (LanguageTool via language_tool_python)
  - Suggest Tags    (rule-based, instant)
  - Generate Tags   (Qwen 0.5B GGUF via llama-cpp-python)
"""

import logging
import threading
import tkinter as tk
from tkinter import messagebox

import customtkinter as ctk

from ui import COLORS, FONT_FAMILY
from tag_suggester import (
    FISH_TAGS,
    TONE_OPTIONS,
    TRANSLATE_LANGUAGES,
    suggest_tags,
    generate_tags,
    grammar_check,
    enhance_for_tts,
    rewrite_tone,
    translate_for_voice,
    is_llm_available,
    is_qwen_model_ready,
    install_llama_cpp,
    download_qwen_model,
    prewarm_llm,
    unload_llm,
)

logger = logging.getLogger("FishTalk.editor")


class TextEditorWindow(ctk.CTkToplevel):
    """
    Modal-ish editor window for a single playlist item.

    Parameters
    ----------
    parent      : root Tk window
    item        : playlist item dict (edited in-place on Save)
    engine      : "kokoro" | "fish14" | "s1mini" | "s1"
    on_save     : optional callback() fired after user clicks Save
    """

    def __init__(self, parent, item: dict, engine: str = "fish14", on_save=None):
        super().__init__(parent)
        self.item = item
        self.engine = engine
        self.on_save = on_save
        self._spell_checker = None  # SpellChecker instance (lazy-loaded)
        self._tag_thread: threading.Thread = None

        self.title(f"Edit — {item.get('name', 'Untitled')}")
        self.geometry("1100x680")
        self.minsize(800, 500)
        self.configure(fg_color=COLORS["bg_dark"])
        self.grab_set()   # modal
        self.focus_force()

        self._build_ui()

        # Pre-warm Qwen in background so AI buttons respond instantly
        self.protocol("WM_DELETE_WINDOW", self._on_editor_close)
        if is_llm_available() and is_qwen_model_ready():
            threading.Thread(target=prewarm_llm, daemon=True, name="QwenPrewarm").start()

    def _on_editor_close(self):
        """Unload Qwen to free RAM when the editor is closed."""
        threading.Thread(target=unload_llm, daemon=True, name="QwenUnload").start()
        self.destroy()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self):
        is_fish = self.engine != "kokoro"

        # ── Top bar ──────────────────────────────────────────────────
        top = ctk.CTkFrame(self, fg_color=COLORS["bg_card"], corner_radius=0, height=44)
        top.pack(fill="x")
        top.pack_propagate(False)

        ctk.CTkLabel(
            top,
            text=f"✏  {self.item.get('name', 'Untitled')}",
            font=(FONT_FAMILY, 13, "bold"),
            text_color=COLORS["text_primary"],
        ).pack(side="left", padx=14, pady=10)

        char_lbl_text = f"{len(self.item.get('text','')  ):,} chars"
        self._char_label = ctk.CTkLabel(
            top,
            text=char_lbl_text,
            font=(FONT_FAMILY, 11),
            text_color=COLORS["text_muted"],
        )
        self._char_label.pack(side="left", padx=6)

        # ── Main content: editor + tag panel ─────────────────────────
        content = ctk.CTkFrame(self, fg_color="transparent")
        content.pack(fill="both", expand=True, padx=10, pady=(8, 4))
        content.grid_columnconfigure(0, weight=1)
        content.grid_columnconfigure(1, weight=0)
        content.grid_rowconfigure(0, weight=1)

        # Text editor
        self._textbox = ctk.CTkTextbox(
            content,
            fg_color=COLORS["bg_input"],
            text_color=COLORS["text_primary"],
            font=(FONT_FAMILY, 13),
            wrap="word",
            corner_radius=8,
            border_color=COLORS["border"],
            border_width=1,
        )
        self._textbox.grid(row=0, column=0, sticky="nsew", padx=(0, 6))
        self._textbox.insert("1.0", self.item.get("text", ""))
        self._textbox.bind("<KeyRelease>", self._on_text_change)

        # Right panel — tabbed
        tabs = ctk.CTkTabview(
            content,
            fg_color=COLORS["bg_card"],
            segmented_button_fg_color=COLORS["bg_input"],
            segmented_button_selected_color=COLORS["accent"],
            segmented_button_selected_hover_color=COLORS["accent_hover"],
            segmented_button_unselected_color=COLORS["bg_input"],
            segmented_button_unselected_hover_color=COLORS["bg_card_hover"],
            text_color=COLORS["text_primary"],
            width=240,
            corner_radius=8,
        )
        tabs.grid(row=0, column=1, sticky="nsew")

        if is_fish:
            tabs.add("🏷 Tags")
            self._build_fish_tag_panel(tabs.tab("🏷 Tags"))
        else:
            tabs.add("ℹ Kokoro")
            self._build_kokoro_panel(tabs.tab("ℹ Kokoro"))

        tabs.add("✨ Enhance")
        self._build_enhance_tab(tabs.tab("✨ Enhance"))

        tabs.add("🎭 Tone")
        self._build_tone_tab(tabs.tab("🎭 Tone"))

        tabs.add("🌐 Translate")
        self._build_translate_tab(tabs.tab("🌐 Translate"))

        # ── Bottom action bar ─────────────────────────────────────────
        self._build_bottom_bar()

    def _build_fish_tag_panel(self, parent):
        ctk.CTkLabel(
            parent,
            text="🏷  Tags",
            font=(FONT_FAMILY, 13, "bold"),
            text_color=COLORS["text_primary"],
        ).pack(anchor="w", padx=10, pady=(10, 6))

        ctk.CTkLabel(
            parent,
            text="Click to insert at cursor",
            font=(FONT_FAMILY, 10),
            text_color=COLORS["text_muted"],
        ).pack(anchor="w", padx=10, pady=(0, 8))

        for category, tags in FISH_TAGS.items():
            ctk.CTkLabel(
                parent,
                text=category,
                font=(FONT_FAMILY, 11, "bold"),
                text_color=COLORS["accent_light"],
            ).pack(anchor="w", padx=10, pady=(6, 2))

            for tag, tooltip in tags:
                row = ctk.CTkFrame(parent, fg_color="transparent")
                row.pack(fill="x", padx=8, pady=1)

                ctk.CTkButton(
                    row,
                    text=tag,
                    font=(FONT_FAMILY, 11),
                    fg_color=COLORS["bg_input"],
                    hover_color=COLORS["bg_card_hover"],
                    border_color=COLORS["accent"],
                    border_width=1,
                    height=26,
                    corner_radius=5,
                    command=lambda t=tag: self._insert_tag(t),
                ).pack(side="left", fill="x", expand=True)

        # Download AI tagger button (shown when model missing)
        self._ai_download_frame = ctk.CTkFrame(parent, fg_color="transparent")
        self._ai_download_frame.pack(fill="x", padx=8, pady=(16, 4))
        self._refresh_ai_status_panel()

    def _build_kokoro_panel(self, parent):
        ctk.CTkLabel(
            parent,
            text="ℹ  Kokoro",
            font=(FONT_FAMILY, 13, "bold"),
            text_color=COLORS["text_primary"],
        ).pack(anchor="w", padx=10, pady=(10, 6))

        ctk.CTkLabel(
            parent,
            text=(
                "Kokoro uses style vectors,\n"
                "not inline text tags.\n\n"
                "Use the voice dropdown\n"
                "and blend slider to\n"
                "adjust speaking style.\n\n"
                "Grammar Check and\n"
                "text cleanup still work."
            ),
            font=(FONT_FAMILY, 11),
            text_color=COLORS["text_secondary"],
            justify="left",
        ).pack(anchor="w", padx=10)

    def _build_enhance_tab(self, parent):
        ctk.CTkLabel(
            parent, text="Improve for TTS",
            font=(FONT_FAMILY, 12, "bold"),
            text_color=COLORS["text_primary"],
        ).pack(anchor="w", padx=10, pady=(10, 2))

        ctk.CTkLabel(
            parent,
            text=(
                "Rewrites the text to sound\n"
                "more natural when spoken:\n\n"
                "• Adds natural pauses\n"
                "• Breaks long sentences\n"
                "• Expands abbreviations\n"
                "• Improves TTS pacing\n\n"
                "Result shown as a preview\n"
                "— accept or discard."
            ),
            font=(FONT_FAMILY, 10),
            text_color=COLORS["text_secondary"],
            justify="left",
        ).pack(anchor="w", padx=10, pady=(0, 10))

        ctk.CTkButton(
            parent,
            text="✨ Enhance for TTS",
            fg_color=COLORS["accent"],
            hover_color=COLORS["accent_hover"],
            font=(FONT_FAMILY, 11, "bold"),
            height=30,
            corner_radius=6,
            command=self._enhance_for_tts,
        ).pack(fill="x", padx=10, pady=(0, 4))

    def _build_tone_tab(self, parent):
        ctk.CTkLabel(
            parent, text="Rewrite Tone",
            font=(FONT_FAMILY, 12, "bold"),
            text_color=COLORS["text_primary"],
        ).pack(anchor="w", padx=10, pady=(10, 2))

        ctk.CTkLabel(
            parent,
            text="Choose a tone and preview\na rewrite. Accept or discard.",
            font=(FONT_FAMILY, 10),
            text_color=COLORS["text_secondary"],
            justify="left",
        ).pack(anchor="w", padx=10, pady=(0, 8))

        self._tone_var = ctk.StringVar(value=TONE_OPTIONS[0])
        ctk.CTkOptionMenu(
            parent,
            values=TONE_OPTIONS,
            variable=self._tone_var,
            fg_color=COLORS["bg_input"],
            button_color=COLORS["accent"],
            button_hover_color=COLORS["accent_hover"],
            dropdown_fg_color=COLORS["bg_card"],
            dropdown_hover_color=COLORS["bg_card_hover"],
            font=(FONT_FAMILY, 11),
            height=28,
        ).pack(fill="x", padx=10, pady=(0, 6))

        ctk.CTkButton(
            parent,
            text="🎭 Preview Rewrite",
            fg_color=COLORS["accent"],
            hover_color=COLORS["accent_hover"],
            font=(FONT_FAMILY, 11, "bold"),
            height=30,
            corner_radius=6,
            command=self._rewrite_tone,
        ).pack(fill="x", padx=10, pady=(0, 4))

        ctk.CTkLabel(
            parent,
            text=(
                "The original text is never\n"
                "changed unless you click\n"
                "Accept in the preview."
            ),
            font=(FONT_FAMILY, 9),
            text_color=COLORS["text_muted"],
            justify="left",
        ).pack(anchor="w", padx=10, pady=(6, 0))

    def _build_translate_tab(self, parent):
        ctk.CTkLabel(
            parent, text="Translate Text",
            font=(FONT_FAMILY, 12, "bold"),
            text_color=COLORS["text_primary"],
        ).pack(anchor="w", padx=10, pady=(10, 2))

        ctk.CTkLabel(
            parent,
            text="Choose a target language,\ntranslate, then accept or\ndiscard the result.",
            font=(FONT_FAMILY, 10),
            text_color=COLORS["text_secondary"],
            justify="left",
        ).pack(anchor="w", padx=10, pady=(0, 8))

        self._translate_lang_var = ctk.StringVar(value=TRANSLATE_LANGUAGES[0])
        ctk.CTkOptionMenu(
            parent,
            values=TRANSLATE_LANGUAGES,
            variable=self._translate_lang_var,
            fg_color=COLORS["bg_input"],
            button_color="#e76f51",
            button_hover_color="#f4a261",
            dropdown_fg_color=COLORS["bg_card"],
            dropdown_hover_color=COLORS["bg_card_hover"],
            font=(FONT_FAMILY, 11),
            height=28,
        ).pack(fill="x", padx=10, pady=(0, 6))

        self._translate_btn = ctk.CTkButton(
            parent,
            text="🌐 Translate",
            fg_color="#e76f51",
            hover_color="#f4a261",
            font=(FONT_FAMILY, 11, "bold"),
            height=30,
            corner_radius=6,
            command=self._translate_text,
            state="normal" if (is_llm_available() and is_qwen_model_ready()) else "disabled",
        )
        self._translate_btn.pack(fill="x", padx=10, pady=(0, 4))

        if not (is_llm_available() and is_qwen_model_ready()):
            ctk.CTkLabel(
                parent,
                text="⚠ AI features not available.\nInstall via Settings.",
                font=(FONT_FAMILY, 9),
                text_color=COLORS["danger"],
                justify="left",
            ).pack(anchor="w", padx=10, pady=(4, 0))

        ctk.CTkLabel(
            parent,
            text=(
                "The original text is never\n"
                "changed unless you click\n"
                "Accept in the preview."
            ),
            font=(FONT_FAMILY, 9),
            text_color=COLORS["text_muted"],
            justify="left",
        ).pack(anchor="w", padx=10, pady=(8, 0))

    def _translate_text(self):
        text = self._textbox.get("1.0", "end-1c").strip()
        if not text:
            self._set_status("Nothing to translate.", COLORS["warning"])
            return

        target_lang = self._translate_lang_var.get()
        self._translate_btn.configure(state="disabled", text="Translating…")
        self._set_status(f"Translating → {target_lang}…", COLORS["warning"])

        def _progress(msg, _frac=None):
            self.after(0, lambda m=msg: self._set_status(m, COLORS["warning"]))

        def _run():
            try:
                result = translate_for_voice(text, target_lang)
            except Exception as exc:
                logger.warning("Editor translate failed: %s", exc)
                result = None

            def _show():
                self._translate_btn.configure(state="normal", text="🌐 Translate")
                if not result or not result.strip():
                    self._set_status("Translation returned empty — try again.", COLORS["danger"])
                    return
                self._set_status("")
                self._show_diff_dialog(text, result, source=f"🌐 Translate → {target_lang}")
                threading.Thread(target=unload_llm, daemon=True).start()

            self.after(0, _show)

        threading.Thread(target=_run, daemon=True, name="EditorTranslate").start()

    def _build_bottom_bar(self):
        bar = ctk.CTkFrame(self, fg_color=COLORS["bg_card"], corner_radius=0, height=52)
        bar.pack(fill="x", side="bottom")
        bar.pack_propagate(False)

        btn = {"font": (FONT_FAMILY, 12, "bold"), "corner_radius": 7, "height": 34}

        # Left: action buttons
        ctk.CTkButton(
            bar,
            text="📝 Grammar Check",
            fg_color=COLORS["bg_input"],
            hover_color=COLORS["bg_card_hover"],
            border_color=COLORS["border"],
            border_width=1,
            width=150,
            command=self._grammar_check,
            **btn,
        ).pack(side="left", padx=(10, 6), pady=9)

        suggest_btn = ctk.CTkButton(
            bar,
            text="💡 Suggest Tags",
            fg_color=COLORS["bg_input"],
            hover_color=COLORS["bg_card_hover"],
            border_color=COLORS["border"],
            border_width=1,
            width=130,
            command=self._suggest_tags,
            **btn,
        )
        if self.engine == "kokoro":
            suggest_btn.configure(state="disabled", text="💡 Suggest Tags (Fish only)")
        suggest_btn.pack(side="left", padx=(0, 6), pady=9)

        self._gen_btn = ctk.CTkButton(
            bar,
            text="🤖 Generate Tags (AI)",
            fg_color=COLORS["accent"],
            hover_color=COLORS["accent_hover"],
            width=160,
            command=self._generate_tags_ai,
            **btn,
        )
        if self.engine == "kokoro":
            self._gen_btn.configure(state="disabled")
        self._gen_btn.pack(side="left", padx=(0, 6), pady=9)

        self._status_label = ctk.CTkLabel(
            bar,
            text="",
            font=(FONT_FAMILY, 11),
            text_color=COLORS["text_muted"],
        )
        self._status_label.pack(side="left", padx=6)

        # Right: save / cancel
        ctk.CTkButton(
            bar,
            text="✕ Cancel",
            fg_color=COLORS["bg_input"],
            hover_color=COLORS["bg_card_hover"],
            border_color=COLORS["border"],
            border_width=1,
            width=90,
            command=self.destroy,
            **btn,
        ).pack(side="right", padx=(6, 10), pady=9)

        ctk.CTkButton(
            bar,
            text="✔ Save",
            fg_color=COLORS["success"],
            hover_color="#05b890",
            width=90,
            command=self._save,
            **btn,
        ).pack(side="right", padx=(0, 6), pady=9)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _insert_tag(self, tag: str):
        """Insert tag text at the current cursor position in the textbox."""
        try:
            idx = self._textbox.index(tk.INSERT)
            self._textbox.insert(idx, tag + " ")
            self._textbox.focus_set()
            self._on_text_change()
        except Exception:
            self._textbox.insert("end", tag + " ")

    def _on_text_change(self, _event=None):
        text = self._textbox.get("1.0", "end-1c")
        self._char_label.configure(text=f"{len(text):,} chars")

    def _set_status(self, msg: str, color: str = None):
        self._status_label.configure(
            text=msg,
            text_color=color or COLORS["text_muted"],
        )

    def _refresh_ai_status_panel(self):
        for w in self._ai_download_frame.winfo_children():
            w.destroy()

        if not is_llm_available():
            ctk.CTkLabel(
                self._ai_download_frame,
                text="⚠ llama-cpp-python\nnot installed",
                font=(FONT_FAMILY, 10),
                text_color=COLORS["warning"],
                justify="left",
            ).pack(anchor="w")
        elif not is_qwen_model_ready():
            ctk.CTkLabel(
                self._ai_download_frame,
                text="Qwen model not downloaded\n(~400 MB)",
                font=(FONT_FAMILY, 10),
                text_color=COLORS["text_muted"],
                justify="left",
            ).pack(anchor="w", pady=(0, 4))
            ctk.CTkButton(
                self._ai_download_frame,
                text="⬇ Download AI Model",
                fg_color=COLORS["accent"],
                hover_color=COLORS["accent_hover"],
                font=(FONT_FAMILY, 11),
                height=28,
                command=self._download_qwen,
            ).pack(fill="x")
        else:
            ctk.CTkLabel(
                self._ai_download_frame,
                text="✅ AI tagger ready",
                font=(FONT_FAMILY, 10),
                text_color=COLORS["success"],
            ).pack(anchor="w")

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------

    def _save(self):
        self.item["text"] = self._textbox.get("1.0", "end-1c")
        if self.on_save:
            self.on_save()
        self.destroy()

    def _grammar_check(self):
        """Grammar + spell check using Qwen 0.5B — same model as Generate Tags."""
        if not is_llm_available():
            self._show_install_dialog(
                title="Install AI Grammar Checker",
                message=(
                    "Grammar Check uses the same Qwen 0.5B model as Generate Tags (~60 MB package + ~400 MB model).\n"
                    "After install, the model will also be downloaded.\n\n"
                    "Install now?"
                ),
                next_step=self._ensure_qwen_then_grammar,
                install_fn=self._run_llama_install,
            )
            return
        self._ensure_qwen_then_grammar()

    def _ensure_qwen_then_grammar(self):
        if not is_qwen_model_ready():
            self._show_download_dialog(
                title="Download Qwen Model",
                message="The Qwen 0.5B model (~400 MB) is needed for Grammar Check.\n\nDownload now?",
                next_step=lambda: self._run_grammar_check_thread(use_llm=True),
                download_fn=self._run_qwen_download,
            )
            return
        self._run_grammar_check_thread(use_llm=True)

    def _run_grammar_check_thread(self, use_llm: bool = True):
        label = "grammar & spelling" if use_llm else "spelling"
        self._set_status(f"Checking {label}…", COLORS["warning"])
        threading.Thread(
            target=self._run_grammar_check,
            args=(use_llm,),
            daemon=True,
        ).start()

    def _run_grammar_check(self, use_llm: bool = True):
        text = self._textbox.get("1.0", "end-1c")
        try:
            if use_llm:
                def _progress(msg, _frac):
                    self.after(0, lambda m=msg: self._set_status(m, COLORS["warning"]))

                corrected = grammar_check(text, on_progress=_progress)
                source = "Qwen grammar & spell check"
            else:
                corrected = self._spellcheck_fallback(text)
                source = "Spell check (fallback)"

            if corrected.strip() == text.strip():
                self.after(0, lambda: self._set_status("✅ No issues found.", COLORS["success"]))
                if use_llm:
                    threading.Thread(target=unload_llm, daemon=True).start()
                return

            def _show(orig=text, corr=corrected, src=source):
                self._show_diff_dialog(orig, corr, source=src)
                self._set_status("")
                if use_llm:
                    threading.Thread(target=unload_llm, daemon=True).start()
            self.after(0, _show)

        except Exception as exc:
            err = repr(exc) if not str(exc) or str(exc) == "None" else str(exc)
            self.after(0, lambda e=err: self._set_status(f"Check error: {e}", COLORS["danger"]))

    def _spellcheck_fallback(self, text: str) -> str:
        """Basic spell correction using pyspellchecker when Qwen is unavailable."""
        import re
        from spellchecker import SpellChecker
        if self._spell_checker is None:
            self._spell_checker = SpellChecker()
        spell = self._spell_checker

        corrected = text
        for m in reversed(list(re.finditer(r"\b[a-z]{3,}\b", text))):
            word = m.group()
            if spell.unknown([word]):
                suggestion = spell.correction(word)
                if suggestion and suggestion != word:
                    corrected = corrected[:m.start()] + suggestion + corrected[m.end():]
        return corrected

    def _suggest_tags(self):
        text = self._textbox.get("1.0", "end-1c")
        self._set_status("Suggesting tags…", COLORS["warning"])
        try:
            tagged = suggest_tags(text)
            self._show_diff_dialog(text, tagged, source="Rule-based suggestions")
            self._set_status("")
        except Exception as exc:
            self._set_status(f"Error: {exc}", COLORS["danger"])

    def _generate_tags_ai(self):
        """Entry point for Generate Tags button — auto-installs deps if needed."""
        if not is_llm_available():
            self._show_install_dialog(
                title="Install AI Tagger",
                message=(
                    "The AI tagger requires llama-cpp-python (~60 MB).\n"
                    "After install, the Qwen 0.5B model (~400 MB) will also be downloaded.\n\n"
                    "Install now?"
                ),
                next_step=self._ensure_qwen_then_generate,
                install_fn=self._run_llama_install,
            )
            return

        self._ensure_qwen_then_generate()

    def _ensure_qwen_then_generate(self):
        """Check model file; download if missing, then run generation."""
        if not is_qwen_model_ready():
            self._show_download_dialog(
                title="Download Qwen 0.5B Model",
                message="The Qwen 0.5B model (~400 MB) is needed for AI tagging.\n\nDownload now?",
                next_step=self._run_generation,
                download_fn=self._run_qwen_download,
            )
            return
        self._run_generation()

    def _run_generation(self):
        """Run Qwen inference — both deps are confirmed present."""
        self._gen_btn.configure(state="disabled", text="🤖 Generating…")
        self._set_status("Loading Qwen model…", COLORS["warning"])
        text = self._textbox.get("1.0", "end-1c")

        def _worker():
            try:
                def _progress(msg, _frac):
                    self.after(0, lambda m=msg: self._set_status(m, COLORS["warning"]))

                tagged = generate_tags(text, on_progress=_progress)

                def _show():
                    self._show_diff_dialog(text, tagged, source="Qwen AI suggestions")
                    self._set_status("")
                    self._gen_btn.configure(state="normal", text="🤖 Generate Tags (AI)")
                    threading.Thread(target=unload_llm, daemon=True).start()

                self.after(0, _show)
            except Exception as exc:
                def _err():
                    self._set_status(f"Error: {exc}", COLORS["danger"])
                    self._gen_btn.configure(state="normal", text="🤖 Generate Tags (AI)")
                self.after(0, _err)

        threading.Thread(target=_worker, daemon=True, name="GenTags").start()

    def _enhance_for_tts(self):
        """Enhance tab: improve text for natural TTS delivery."""
        if not is_llm_available():
            self._show_install_dialog(
                title="Install AI Model",
                message="Enhance for TTS uses Qwen 0.5B (~60 MB package + ~400 MB model).\n\nInstall now?",
                next_step=self._ensure_qwen_then_enhance,
                install_fn=self._run_llama_install,
            )
            return
        self._ensure_qwen_then_enhance()

    def _ensure_qwen_then_enhance(self):
        if not is_qwen_model_ready():
            self._show_download_dialog(
                title="Download Qwen Model",
                message="The Qwen 0.5B model (~400 MB) is needed.\n\nDownload now?",
                next_step=self._run_enhance,
                download_fn=self._run_qwen_download,
            )
            return
        self._run_enhance()

    def _run_enhance(self):
        self._set_status("Enhancing for TTS…", COLORS["warning"])
        text = self._textbox.get("1.0", "end-1c")

        def _worker():
            try:
                def _progress(msg, _frac):
                    self.after(0, lambda m=msg: self._set_status(m, COLORS["warning"]))
                result = enhance_for_tts(text, engine=self.engine, on_progress=_progress)

                def _show():
                    self._show_diff_dialog(text, result, source="✨ Enhance for TTS")
                    self._set_status("")
                    threading.Thread(target=unload_llm, daemon=True).start()
                self.after(0, _show)
            except Exception as exc:
                self.after(0, lambda e=str(exc): self._set_status(f"Error: {e}", COLORS["danger"]))

        threading.Thread(target=_worker, daemon=True, name="EnhanceTTS").start()

    def _rewrite_tone(self):
        """Tone tab: rewrite text in the selected tone."""
        if not is_llm_available():
            self._show_install_dialog(
                title="Install AI Model",
                message="Tone Rewrite uses Qwen 0.5B (~60 MB package + ~400 MB model).\n\nInstall now?",
                next_step=self._ensure_qwen_then_tone,
                install_fn=self._run_llama_install,
            )
            return
        self._ensure_qwen_then_tone()

    def _ensure_qwen_then_tone(self):
        if not is_qwen_model_ready():
            self._show_download_dialog(
                title="Download Qwen Model",
                message="The Qwen 0.5B model (~400 MB) is needed.\n\nDownload now?",
                next_step=self._run_tone_rewrite,
                download_fn=self._run_qwen_download,
            )
            return
        self._run_tone_rewrite()

    def _run_tone_rewrite(self):
        tone = getattr(self, "_tone_var", None)
        tone = tone.get() if tone else "Neutral"
        self._set_status(f"Rewriting as {tone}…", COLORS["warning"])
        text = self._textbox.get("1.0", "end-1c")

        def _worker():
            try:
                def _progress(msg, _frac):
                    self.after(0, lambda m=msg: self._set_status(m, COLORS["warning"]))
                result = rewrite_tone(text, tone=tone, on_progress=_progress)

                def _show():
                    self._show_diff_dialog(text, result, source=f"🎭 Tone: {tone}")
                    self._set_status("")
                    threading.Thread(target=unload_llm, daemon=True).start()
                self.after(0, _show)
            except Exception as exc:
                self.after(0, lambda e=str(exc): self._set_status(f"Error: {e}", COLORS["danger"]))

        threading.Thread(target=_worker, daemon=True, name="ToneRewrite").start()

    def _download_qwen(self):
        """Download button in the side panel — shows inline status."""
        self._set_status("Downloading Qwen model (~400 MB)…", COLORS["warning"])

        def _progress(msg, _frac):
            self.after(0, lambda m=msg: self._set_status(m, COLORS["warning"]))

        def _complete(ok, msg):
            def _ui():
                if ok:
                    self._set_status("✅ Qwen model ready.", COLORS["success"])
                    self._refresh_ai_status_panel()
                else:
                    self._set_status(f"Download failed: {msg}", COLORS["danger"])
            self.after(0, _ui)

        download_qwen_model(on_progress=_progress, on_complete=_complete)

    # ------------------------------------------------------------------
    # Install / download progress dialogs
    # ------------------------------------------------------------------

    def _show_install_dialog(self, title, message, next_step, install_fn):
        """Yes/No prompt → progress dialog → calls next_step on success."""
        if not messagebox.askyesno(title, message, parent=self):
            return
        self._show_progress_dialog(
            title=title,
            start_fn=install_fn,
            on_success=next_step,
        )

    def _show_download_dialog(self, title, message, next_step, download_fn):
        """Yes/No prompt → progress dialog → calls next_step on success."""
        if not messagebox.askyesno(title, message, parent=self):
            return
        self._show_progress_dialog(
            title=title,
            start_fn=download_fn,
            on_success=next_step,
        )

    def _show_progress_dialog(self, title: str, start_fn, on_success):
        """
        Generic progress dialog with log output and indeterminate bar.
        start_fn receives (on_line, on_complete) callbacks.
        on_success() is called on the main thread if completed successfully.
        """
        dlg = ctk.CTkToplevel(self)
        dlg.title(title)
        dlg.geometry("520x320")
        dlg.configure(fg_color=COLORS["bg_dark"])
        dlg.grab_set()
        dlg.resizable(False, False)

        ctk.CTkLabel(
            dlg, text=title,
            font=(FONT_FAMILY, 13, "bold"),
            text_color=COLORS["text_primary"],
        ).pack(pady=(16, 4))

        bar = ctk.CTkProgressBar(
            dlg,
            progress_color=COLORS["accent"],
            fg_color=COLORS["bg_input"],
            height=8, corner_radius=4, width=460,
            mode="indeterminate",
        )
        bar.pack(pady=(4, 8))
        bar.start()

        log = ctk.CTkTextbox(
            dlg,
            fg_color=COLORS["bg_input"],
            text_color=COLORS["text_secondary"],
            font=(FONT_FAMILY, 10),
            height=160, width=480,
            corner_radius=6,
            state="normal",
        )
        log.pack(padx=16, pady=(0, 8))

        status_lbl = ctk.CTkLabel(
            dlg, text="Working…",
            font=(FONT_FAMILY, 11),
            text_color=COLORS["text_muted"],
        )
        status_lbl.pack()

        close_btn = ctk.CTkButton(
            dlg, text="Close", state="disabled",
            fg_color=COLORS["success"], hover_color="#05b890",
            font=(FONT_FAMILY, 12, "bold"), width=100, height=32,
            command=dlg.destroy,
        )
        close_btn.pack(pady=(6, 12))

        def _append(line: str):
            def _ui():
                log.configure(state="normal")
                log.insert("end", line + "\n")
                log.see("end")
                log.configure(state="disabled")
                status_lbl.configure(text=line[:80])
            dlg.after(0, _ui)

        def _done(ok: bool, msg: str):
            def _ui():
                bar.stop()
                bar.configure(mode="determinate")
                bar.set(1.0 if ok else 0.0)
                bar.configure(progress_color=COLORS["success"] if ok else COLORS["danger"])
                status_lbl.configure(
                    text="✅ Complete!" if ok else f"❌ Failed: {msg}",
                    text_color=COLORS["success"] if ok else COLORS["danger"],
                )
                close_btn.configure(state="normal")
                if ok:
                    dlg.after(800, dlg.destroy)
                    dlg.after(900, on_success)
            dlg.after(0, _ui)

        start_fn(on_line=_append, on_complete=_done)

    def _run_llama_install(self, on_line, on_complete):
        install_llama_cpp(on_line=on_line, on_complete=on_complete)

    def _run_spellchecker_install(self, on_line, on_complete):
        import subprocess, sys, os
        app_dir = os.path.dirname(os.path.abspath(__file__))
        venv_pip = os.path.join(app_dir, "venv", "Scripts", "pip.exe")
        pip_cmd = [venv_pip] if os.path.isfile(venv_pip) else [sys.executable, "-m", "pip"]
        cmd = pip_cmd + ["install", "pyspellchecker", "--upgrade",
                         "--quiet", "--progress-bar", "off"]
        CREATE_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0x08000000)
        def _worker():
            try:
                if on_line:
                    on_line("Installing pyspellchecker…")
                proc = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                                        stderr=subprocess.STDOUT, text=True,
                                        creationflags=CREATE_NO_WINDOW)
                for raw in proc.stdout:
                    if raw.strip() and on_line:
                        on_line(raw.rstrip())
                proc.wait()
                if proc.returncode == 0:
                    if on_complete:
                        on_complete(True, "pyspellchecker installed.")
                else:
                    if on_complete:
                        on_complete(False, f"pip exited with code {proc.returncode}")
            except Exception as exc:
                if on_complete:
                    on_complete(False, str(exc))
        import threading as _t
        _t.Thread(target=_worker, daemon=True, name="SpellCheckInstall").start()

    def _run_qwen_download(self, on_line, on_complete):
        def _progress(msg, frac):
            on_line(f"[{int(frac*100):3d}%] {msg}")

        def _done(ok, msg):
            on_complete(ok, msg)

        download_qwen_model(on_progress=_progress, on_complete=_done)

    # ------------------------------------------------------------------
    # Diff / review dialog
    # ------------------------------------------------------------------

    # Sources that are tag operations vs text-change operations
    _TAG_SOURCES = {"Rule-based suggestions", "Qwen AI suggestions"}

    def _show_diff_dialog(self, original: str, suggested: str, source: str = ""):
        """Show a side-by-side diff dialog with word-level highlights. Accept or discard."""
        if original.strip() == suggested.strip():
            messagebox.showinfo("No Changes", "No changes were suggested.", parent=self)
            return

        is_tag_op = source in self._TAG_SOURCES
        window_title = f"Review Tags — {source}" if is_tag_op else f"Review Changes — {source}"
        right_label  = "With Tags" if is_tag_op else "Suggested"

        dlg = ctk.CTkToplevel(self)
        dlg.title(window_title)
        dlg.geometry("920x560")
        dlg.configure(fg_color=COLORS["bg_dark"])
        dlg.grab_set()

        ctk.CTkLabel(
            dlg,
            text="Review the suggested changes. Accept to apply, Discard to keep the original.",
            font=(FONT_FAMILY, 12),
            text_color=COLORS["text_secondary"],
        ).pack(padx=14, pady=(10, 6))

        panels = ctk.CTkFrame(dlg, fg_color="transparent")
        panels.pack(fill="both", expand=True, padx=10)
        panels.grid_columnconfigure(0, weight=1)
        panels.grid_columnconfigure(1, weight=1)
        panels.grid_rowconfigure(1, weight=1)

        # Column headers with a small legend
        hdr_orig = ctk.CTkFrame(panels, fg_color="transparent")
        hdr_orig.grid(row=0, column=0, sticky="w", padx=4, pady=(0, 2))
        ctk.CTkLabel(hdr_orig, text="Original", font=(FONT_FAMILY, 11, "bold"),
                     text_color=COLORS["text_muted"]).pack(side="left")
        ctk.CTkLabel(hdr_orig, text="  removed", font=(FONT_FAMILY, 10),
                     text_color="#ef476f").pack(side="left", padx=(8, 0))

        hdr_sugg = ctk.CTkFrame(panels, fg_color="transparent")
        hdr_sugg.grid(row=0, column=1, sticky="w", padx=4, pady=(0, 2))
        ctk.CTkLabel(hdr_sugg, text=right_label, font=(FONT_FAMILY, 11, "bold"),
                     text_color=COLORS["accent_light"]).pack(side="left")
        ctk.CTkLabel(hdr_sugg, text="  added", font=(FONT_FAMILY, 10),
                     text_color="#06d6a0").pack(side="left", padx=(8, 0))

        orig_box = ctk.CTkTextbox(panels, fg_color=COLORS["bg_input"], font=(FONT_FAMILY, 12),
                                   wrap="word", corner_radius=6)
        orig_box.grid(row=1, column=0, sticky="nsew", padx=(0, 4))
        orig_box.insert("1.0", original)

        tagged_box = ctk.CTkTextbox(panels, fg_color=COLORS["bg_input"], font=(FONT_FAMILY, 12),
                                     wrap="word", corner_radius=6)
        tagged_box.grid(row=1, column=1, sticky="nsew", padx=(4, 0))
        tagged_box.insert("1.0", suggested)

        # Apply word-level diff highlights, then lock the original pane
        self._apply_diff_highlights(orig_box, tagged_box, original, suggested)
        orig_box.configure(state="disabled")

        btn_bar = ctk.CTkFrame(dlg, fg_color="transparent")
        btn_bar.pack(fill="x", padx=10, pady=8)

        def _accept():
            new_text = tagged_box.get("1.0", "end-1c")
            self._textbox.delete("1.0", "end")
            self._textbox.insert("1.0", new_text)
            self._on_text_change()
            dlg.destroy()

        ctk.CTkButton(btn_bar, text="✔ Accept", fg_color=COLORS["success"],
                      hover_color="#05b890", width=120, height=34,
                      font=(FONT_FAMILY, 12, "bold"), command=_accept).pack(side="left", padx=(0, 8))
        ctk.CTkButton(btn_bar, text="✕ Discard", fg_color=COLORS["danger"],
                      hover_color="#d43d62", width=120, height=34,
                      font=(FONT_FAMILY, 12, "bold"), command=dlg.destroy).pack(side="left")

    @staticmethod
    def _apply_diff_highlights(orig_ctk: ctk.CTkTextbox, sugg_ctk: ctk.CTkTextbox,
                                original: str, suggested: str):
        """
        Word-level diff highlighting using difflib.

        Original pane  — removed/changed words: red + underline
        Suggested pane — added/changed words:   green + subtle background
        """
        import difflib
        import re

        orig_tw = orig_ctk._textbox
        sugg_tw = sugg_ctk._textbox

        orig_tw.tag_configure("removed", foreground="#ef476f", underline=True)
        sugg_tw.tag_configure("added",   foreground="#06d6a0", background="#0d2a1c")

        # Tokenize preserving whitespace so offsets stay accurate
        tokens_orig = re.findall(r'\S+|\s+', original)
        tokens_sugg = re.findall(r'\S+|\s+', suggested)

        matcher = difflib.SequenceMatcher(None, tokens_orig, tokens_sugg, autojunk=False)

        orig_off = 0
        sugg_off = 0

        for op, i1, i2, j1, j2 in matcher.get_opcodes():
            o_len = sum(len(tokens_orig[k]) for k in range(i1, i2))
            s_len = sum(len(tokens_sugg[k]) for k in range(j1, j2))

            if op in ("replace", "delete") and o_len:
                s = orig_tw.index(f"1.0 + {orig_off}c")
                e = orig_tw.index(f"1.0 + {orig_off + o_len}c")
                orig_tw.tag_add("removed", s, e)

            if op in ("replace", "insert") and s_len:
                s = sugg_tw.index(f"1.0 + {sugg_off}c")
                e = sugg_tw.index(f"1.0 + {sugg_off + s_len}c")
                sugg_tw.tag_add("added", s, e)

            orig_off += o_len
            sugg_off += s_len
