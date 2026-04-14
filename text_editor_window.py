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
    suggest_tags,
    generate_tags,
    is_llm_available,
    is_qwen_model_ready,
    download_qwen_model,
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
    engine      : "kokoro" | "fish14" | "fish15"
    on_save     : optional callback() fired after user clicks Save
    """

    def __init__(self, parent, item: dict, engine: str = "fish14", on_save=None):
        super().__init__(parent)
        self.item = item
        self.engine = engine
        self.on_save = on_save
        self._grammar_tool = None   # LanguageTool instance (lazy)
        self._tag_thread: threading.Thread = None

        self.title(f"Edit — {item.get('name', 'Untitled')}")
        self.geometry("1100x680")
        self.minsize(800, 500)
        self.configure(fg_color=COLORS["bg_dark"])
        self.grab_set()   # modal
        self.focus_force()

        self._build_ui()

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

        # Right panel
        right = ctk.CTkScrollableFrame(
            content,
            fg_color=COLORS["bg_card"],
            corner_radius=8,
            width=220,
            scrollbar_button_color=COLORS["accent"],
        )
        right.grid(row=0, column=1, sticky="nsew")

        if is_fish:
            self._build_fish_tag_panel(right)
        else:
            self._build_kokoro_panel(right)

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
        self._set_status("Running grammar check…", COLORS["warning"])
        threading.Thread(target=self._run_grammar_check, daemon=True).start()

    def _run_grammar_check(self):
        text = self._textbox.get("1.0", "end-1c")
        try:
            import language_tool_python
            if self._grammar_tool is None:
                self.after(0, lambda: self._set_status("Loading LanguageTool (first run may take a moment)…", COLORS["warning"]))
                self._grammar_tool = language_tool_python.LanguageTool("en-US")
            matches = self._grammar_tool.check(text)
            if not matches:
                self.after(0, lambda: self._set_status("✅ No issues found.", COLORS["success"]))
                return
            corrected = language_tool_python.utils.correct(text, matches)
            def _apply():
                self._textbox.delete("1.0", "end")
                self._textbox.insert("1.0", corrected)
                self._on_text_change()
                self._set_status(f"✅ Fixed {len(matches)} issue(s).", COLORS["success"])
            self.after(0, _apply)
        except ImportError:
            self.after(0, lambda: messagebox.showwarning(
                "Grammar Check",
                "language_tool_python is not installed.\n\nInstall it with:\n  pip install language-tool-python",
                parent=self,
            ))
            self.after(0, lambda: self._set_status(""))
        except Exception as exc:
            self.after(0, lambda: self._set_status(f"Grammar check error: {exc}", COLORS["danger"]))

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
        if not is_llm_available():
            messagebox.showwarning(
                "AI Tagger",
                "llama-cpp-python is not installed.\n\nInstall it with:\n  pip install llama-cpp-python",
                parent=self,
            )
            return
        if not is_qwen_model_ready():
            messagebox.showinfo(
                "AI Tagger",
                "Qwen model not downloaded yet.\nUse the 'Download AI Model' button in the tag panel.",
                parent=self,
            )
            return

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
                    # Unload model to free RAM after use
                    threading.Thread(target=unload_llm, daemon=True).start()

                self.after(0, _show)
            except Exception as exc:
                def _err():
                    self._set_status(f"Error: {exc}", COLORS["danger"])
                    self._gen_btn.configure(state="normal", text="🤖 Generate Tags (AI)")
                self.after(0, _err)

        threading.Thread(target=_worker, daemon=True, name="GenTags").start()

    def _download_qwen(self):
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
    # Diff / review dialog
    # ------------------------------------------------------------------

    def _show_diff_dialog(self, original: str, suggested: str, source: str = ""):
        """Show a side-by-side before/after dialog. User can accept or discard."""
        if original.strip() == suggested.strip():
            messagebox.showinfo("Tag Suggestion", "No tags were added — no clear emotional cues found.", parent=self)
            return

        dlg = ctk.CTkToplevel(self)
        dlg.title(f"Review Tags — {source}")
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

        ctk.CTkLabel(panels, text="Original", font=(FONT_FAMILY, 11, "bold"),
                     text_color=COLORS["text_muted"]).grid(row=0, column=0, sticky="w", padx=4)
        ctk.CTkLabel(panels, text="With Tags", font=(FONT_FAMILY, 11, "bold"),
                     text_color=COLORS["accent_light"]).grid(row=0, column=1, sticky="w", padx=4)

        orig_box = ctk.CTkTextbox(panels, fg_color=COLORS["bg_input"], font=(FONT_FAMILY, 12),
                                   wrap="word", corner_radius=6)
        orig_box.grid(row=1, column=0, sticky="nsew", padx=(0, 4))
        orig_box.insert("1.0", original)
        orig_box.configure(state="disabled")

        tagged_box = ctk.CTkTextbox(panels, fg_color=COLORS["bg_input"], font=(FONT_FAMILY, 12),
                                     wrap="word", corner_radius=6)
        tagged_box.grid(row=1, column=1, sticky="nsew", padx=(4, 0))
        tagged_box.insert("1.0", suggested)

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
