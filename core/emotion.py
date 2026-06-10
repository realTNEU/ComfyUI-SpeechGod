"""Emotion / age / tone -> delivery-parameter mapping.

Every control resolves to a small set of concrete, engine-agnostic knobs:
  pitch    : semitones added after generation (DSP)
  tempo    : playback tempo multiplier (pitch-preserving)
  gain_db  : output gain
  marker   : inline emotion marker (Fish Speech / OpenAudio understands these)
  style    : natural-language style fragment fed to prompt-aware engines
"""

TONES = ["neutral", "happy", "sad", "excited", "angry", "scared", "confused",
         "curious", "wise", "evil", "heroic", "narrator"]

TONE_MAP = {
    #            pitch  tempo  gain  marker        style fragment
    "neutral":  (0.0,   1.00,  0.0, "",           "a natural, even delivery"),
    "happy":    (1.5,   1.06,  1.0, "(happy)",     "a bright, warm, smiling delivery"),
    "sad":      (-1.5,  0.90, -2.0, "(sad)",       "a soft, downcast, heavy-hearted delivery"),
    "excited":  (2.5,   1.14,  2.5, "(excited)",   "an enthusiastic, fast, energetic delivery"),
    "angry":    (-0.5,  1.08,  3.0, "(angry)",     "a sharp, forceful, clipped delivery"),
    "scared":   (2.0,   1.12, -1.0, "(scared)",    "a trembling, breathy, urgent delivery"),
    "confused": (0.5,   0.94,  0.0, "(confused)",  "a hesitant delivery with rising intonation"),
    "curious":  (1.0,   1.00,  0.0, "(curious)",   "an inquisitive, lilting delivery"),
    "wise":     (-2.0,  0.88,  0.0, "",            "a slow, measured, thoughtful delivery"),
    "evil":     (-3.0,  0.92,  1.0, "(sinister)",  "a low, menacing, deliberate delivery"),
    "heroic":   (-1.0,  1.02,  2.0, "",            "a bold, confident, projected delivery"),
    "narrator": (-1.0,  0.96,  0.0, "",            "a clear, articulate storyteller delivery"),
}

AGES = ["child", "teen", "young_adult", "adult", "elderly"]

AGE_MAP = {
    #               pitch  tempo  style fragment
    "child":       (4.0,   1.10, "a young child's voice, light and playful"),
    "teen":        (2.0,   1.05, "a teenager's voice, youthful and lively"),
    "young_adult": (0.5,   1.00, "a young adult voice"),
    "adult":       (0.0,   1.00, "an adult voice"),
    "elderly":     (-2.0,  0.88, "an elderly voice, slower and slightly frail"),
}

GENDERS = ["female", "male", "neutral"]
GENDER_PITCH = {"female": 1.0, "male": -1.0, "neutral": 0.0}

MICRO_STYLES = {
    # slider name -> (style fragment, marker for fish-speech)
    "laugh":              ("with light laughter woven in", "(laughing)"),
    "whisper":            ("in a hushed whisper", "(whispering)"),
    "nervousness":        ("nervous and slightly stumbling", "(nervous)"),
    "confidence":         ("assured and self-confident", ""),
    "dramatic_intensity": ("with theatrical, dramatic intensity", ""),
    "sarcasm":            ("dripping with sarcasm", "(sarcastic)"),
    "cuteness":           ("adorably cute and endearing", ""),
}


def blend_tones(tone_a: str, weight_a: float, tone_b: str, weight_b: float):
    """Linear blend of two tone parameter sets. Returns a synthetic TONE tuple."""
    a, b = TONE_MAP[tone_a], TONE_MAP[tone_b]
    total = max(weight_a + weight_b, 1e-6)
    wa, wb = weight_a / total, weight_b / total
    pitch = a[0] * wa + b[0] * wb
    tempo = a[1] * wa + b[1] * wb
    gain = a[2] * wa + b[2] * wb
    marker = a[3] if wa >= wb else b[3]
    style = f"{int(wa * 100)}% {tone_a} and {int(wb * 100)}% {tone_b}: {a[4]}, blended with {b[4]}"
    return (pitch, tempo, gain, marker, style)


def resolve_delivery(character: dict, emotion_blend=None):
    """Collapse a character profile into concrete delivery parameters."""
    tone = character.get("tone", "neutral")
    tone_params = emotion_blend if emotion_blend is not None else TONE_MAP.get(tone, TONE_MAP["neutral"])
    age_params = AGE_MAP.get(character.get("age", "adult"), AGE_MAP["adult"])
    energy = character.get("energy", 50) / 100.0          # 0..1
    accent = character.get("accent_strength", 0) / 100.0

    pitch = tone_params[0] + age_params[0] + GENDER_PITCH.get(character.get("gender", "neutral"), 0.0)
    tempo = tone_params[1] * age_params[1]
    # energy: 0 -> subdued (-3 dB, -4% tempo), 1 -> intense (+3 dB, +6% tempo)
    gain_db = tone_params[2] + (energy - 0.5) * 6.0
    tempo *= 1.0 + (energy - 0.5) * 0.10
    tempo *= character.get("speech_speed", 1.0)

    style_bits = [age_params[2], tone_params[4]]
    markers = [tone_params[3]] if tone_params[3] else []
    for key, (frag, marker) in MICRO_STYLES.items():
        val = character.get(key, 0)
        if val >= 25:
            style_bits.append(frag)
            if marker and val >= 50:
                markers.append(marker)
    if character.get("style_prompt"):
        style_bits.append(character["style_prompt"])
    if character.get("scene_context"):
        style_bits.append(f"scene: {character['scene_context']}")
    if accent > 0:
        style_bits.append(f"with a {'strong' if accent > 0.6 else 'mild'} accent")

    return {
        "pitch": pitch,
        "tempo": max(0.5, min(2.0, tempo)),
        "gain_db": gain_db,
        "markers": markers,
        "style_text": ", ".join(s for s in style_bits if s),
    }
