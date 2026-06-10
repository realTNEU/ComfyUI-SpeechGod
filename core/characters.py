"""Character database: characters/<name>.json + <name>.voice

<name>.json   - all profile parameters (age, gender, tone, sliders, style...)
<name>.voice  - reference audio stored as a torch file {waveform, sample_rate}
                (this doubles as the 'voice embedding' for cloning engines,
                 which consume reference audio directly)
"""
import json
import os
import re

import torch

PKG_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHAR_DIR = os.path.join(PKG_DIR, "characters")
os.makedirs(CHAR_DIR, exist_ok=True)

PROFILE_KEYS = [
    "character_name", "gender", "age", "tone", "energy", "speech_speed",
    "accent_strength", "style_prompt", "scene_context", "reference_text",
    "laugh", "whisper", "nervousness", "confidence", "dramatic_intensity",
    "sarcasm", "cuteness",
]


def _slug(name: str) -> str:
    return re.sub(r"[^a-z0-9_\-]", "_", name.strip().lower()) or "unnamed"


def save_character(character: dict) -> str:
    name = _slug(character.get("character_name", "unnamed"))
    profile = {k: character.get(k) for k in PROFILE_KEYS if k in character}
    with open(os.path.join(CHAR_DIR, f"{name}.json"), "w", encoding="utf-8") as f:
        json.dump(profile, f, indent=2, ensure_ascii=False)
    ref = character.get("voice_reference")
    if ref is not None:
        torch.save(
            {"waveform": ref["waveform"].cpu(), "sample_rate": ref["sample_rate"]},
            os.path.join(CHAR_DIR, f"{name}.voice"),
        )
    return name


def load_character(name: str) -> dict:
    name = _slug(name.replace(".json", "").replace(".voice", ""))
    path = os.path.join(CHAR_DIR, f"{name}.json")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Speech-God: character '{name}' not found in {CHAR_DIR}. "
            f"Save it first with the 'Save Character' node."
        )
    with open(path, encoding="utf-8") as f:
        character = json.load(f)
    vpath = os.path.join(CHAR_DIR, f"{name}.voice")
    if os.path.exists(vpath):
        data = torch.load(vpath, map_location="cpu", weights_only=True)
        character["voice_reference"] = {
            "waveform": data["waveform"], "sample_rate": data["sample_rate"],
        }
    return character


def list_characters():
    return sorted(
        f[:-5] for f in os.listdir(CHAR_DIR) if f.endswith(".json")
    ) or ["(none saved)"]


def evolve(base: dict, target: dict | None, progress: float) -> dict:
    """Voice evolution: interpolate numeric traits (and crossfade reference
    audio amplitude) from base toward target. With no target, applies an
    'aging' drift driven by progress."""
    out = dict(base)
    p = max(0.0, min(1.0, progress))
    numeric = ["energy", "speech_speed", "accent_strength", "laugh", "whisper",
               "nervousness", "confidence", "dramatic_intensity", "sarcasm", "cuteness"]
    if target is not None:
        for k in numeric:
            a, b = float(base.get(k, 0) or 0), float(target.get(k, 0) or 0)
            out[k] = type(base.get(k, 0))(a + (b - a) * p) if base.get(k) is not None else b * p
        ages = ["child", "teen", "young_adult", "adult", "elderly"]
        ai = ages.index(base.get("age", "adult"))
        bi = ages.index(target.get("age", "adult"))
        out["age"] = ages[round(ai + (bi - ai) * p)]
        if p >= 0.5 and target.get("voice_reference") is not None:
            out["voice_reference"] = target["voice_reference"]
        out["_evolve_pitch"] = 0.0
    else:
        # natural aging drift: pitch falls, tempo slows slightly
        out["_evolve_pitch"] = -2.5 * p
        out["speech_speed"] = float(base.get("speech_speed", 1.0)) * (1.0 - 0.08 * p)
    return out
