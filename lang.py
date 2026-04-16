"""
lang.py — KoKoFish Localisation Engine

Purely stdlib (json, os) so it can be imported by launcher.py before
the venv exists, as well as by the main application.

Usage:
    from lang import t, load_language, save_language_pref, get_languages

    # At app start (after determining APP_DIR):
    load_language()

    # In UI code:
    button_label = t("COMMON_BTN_SAVE")
    status_msg   = t("SPEECH_LAB_CONVERTING_STATUS", name=item_name)

    # When user switches language:
    save_language_pref("fr")
    load_language("fr")
    # then rebuild / refresh the UI
"""

import json
import os

# ---------------------------------------------------------------------------
# Internal state
# ---------------------------------------------------------------------------
_strings:      dict = {}          # key → {lang: str, dynamic: bool, ...}
_current_lang: str  = "en"
_available:    list = ["en"]      # filled from translations.json meta

# Resolve paths relative to this file's location (works for both
# the installed app and a dev checkout).
_HERE             = os.path.dirname(os.path.abspath(__file__))
TRANSLATIONS_FILE = os.path.join(_HERE, "translations.json")
LANG_PREF_FILE    = os.path.join(_HERE, "lang_pref.json")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_language(lang: str | None = None) -> str:
    """
    Load translations.json and activate *lang*.

    If *lang* is None the saved preference is used; if no preference
    exists, English is used.

    Returns the active language code.
    """
    global _strings, _current_lang, _available

    try:
        with open(TRANSLATIONS_FILE, encoding="utf-8") as fh:
            data = json.load(fh)
        _strings   = data.get("strings", {})
        _available = data.get("meta", {}).get("languages", ["en"])
    except Exception:
        # translations.json missing or corrupt — fall back to key passthrough
        _strings   = {}
        _available = ["en"]

    if lang is None:
        lang = _load_pref()

    # Make sure the requested language actually exists
    if lang not in _available:
        lang = "en"

    _current_lang = lang
    return lang


def save_language_pref(lang: str) -> None:
    """Persist the chosen language and update the active language."""
    global _current_lang
    _current_lang = lang
    try:
        with open(LANG_PREF_FILE, "w", encoding="utf-8") as fh:
            json.dump({"lang": lang}, fh)
    except Exception:
        pass


def get_current_language() -> str:
    """Return the active language code (e.g. 'en', 'fr', 'ja')."""
    return _current_lang


def get_languages() -> list[tuple[str, str]]:
    """
    Return all available languages as a list of (code, display_name) tuples.
    E.g. [('en', 'English'), ('fr', 'French'), ...]
    """
    try:
        with open(TRANSLATIONS_FILE, encoding="utf-8") as fh:
            data = json.load(fh)
        names = data.get("meta", {}).get("language_names", {})
        return [(code, names.get(code, code)) for code in data.get("meta", {}).get("languages", ["en"])]
    except Exception:
        return [("en", "English")]


def t(key: str, **kwargs) -> str:
    """
    Look up *key* in the current language.

    Falls back to English if the key has no translation for the active
    language.  Falls back to the bare key name if the key is missing
    entirely (so the UI never crashes, just shows the raw key).

    Dynamic strings use Python str.format() style placeholders:
        t("SPEECH_LAB_CONVERTING_STATUS", name="Chapter 1")
        # → "Converting: Chapter 1…"
    """
    entry = _strings.get(key)
    if entry is None:
        # Key not found — return the key itself as a visible fallback
        return key

    # Prefer current language, fall back to English.
    # The extraction agent stored English under "value" key; support both.
    english = entry.get("en") or entry.get("value") or key
    text = entry.get(_current_lang) or english

    # Substitute any runtime variables
    if kwargs and isinstance(text, str):
        try:
            text = text.format(**kwargs)
        except (KeyError, ValueError, IndexError):
            # If substitution fails just return the raw template
            pass

    return text


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _load_pref() -> str:
    """Read the saved language preference; return 'en' if not set."""
    try:
        with open(LANG_PREF_FILE, encoding="utf-8") as fh:
            return json.load(fh).get("lang", "en")
    except Exception:
        return "en"
