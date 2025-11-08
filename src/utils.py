"""
Utility functions
"""
import os
import json
from pathlib import Path
from typing import Dict, List, Optional


def ensure_dir(path: str):
    """Ensure directory exists"""
    Path(path).mkdir(parents=True, exist_ok=True)


def save_json(data: dict, path: str):
    """Save data to JSON file"""
    ensure_dir(Path(path).parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def load_json(path: str) -> dict:
    """Load data from JSON file"""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_language_code(language: str) -> str:
    """Get NLLB language code for a language"""
    lang_code_map = {
        "Hindi": "hin_Deva",
        "Bhili": "bhi_Deva",
        "Bhilli": "bhi_Deva",  # Alternative spelling
        "Mundari": "unr_Deva",
        "Gondi": "gon_Deva",
        "Gondari": "gon_Deva",  # Alternative spelling
        "English": "eng_Latn",
        "Santali": "sat_Olck"
    }
    return lang_code_map.get(language, "eng_Latn")


def validate_language_pair(source_lang: str, target_lang: str) -> bool:
    """Validate that language pair is supported"""
    valid_pairs = [
        ("Hindi", "Bhili"),
        ("Hindi", "Mundari"),
        ("Hindi", "Gondi"),
        ("English", "Santali"),
        ("Bhili", "Hindi"),
        ("Mundari", "Hindi"),
        ("Gondi", "Hindi"),
        ("Santali", "English")
    ]
    return (source_lang, target_lang) in valid_pairs

