"""Speech-God node suite.

Pipeline:  Loader -> Character Definition -> Dialogue Parser -> Emotion
Processor -> Voice Generation Engine -> Post Processing -> Audio Export
"""
import os

import torch

import folder_paths

from .core import characters as chardb
from .core import dsp
from .core import engines
from .core.emotion import AGES, GENDERS, TONES, blend_tones, resolve_delivery
from .core.parser import parse_dialogue

CATEGORY = "Speech-God"


def _audio(wave, sr):
    """[C,T] -> ComfyUI AUDIO dict."""
    if wave.dim() == 2:
        wave = wave.unsqueeze(0)
    return {"waveform": wave.cpu(), "sample_rate": sr}


# ===================================================================== Loader
class SpeechGodLoader:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "engine": (["f5-tts", "fish-speech"], {"default": "f5-tts"}),
            "device": (["auto", "cuda", "cpu"], {"default": "auto"}),
            "precision": (["auto", "fp16", "fp32"], {"default": "auto"}),
        }}

    RETURN_TYPES = ("SPEECHGOD_ENGINE",)
    RETURN_NAMES = ("engine",)
    FUNCTION = "load"
    CATEGORY = CATEGORY

    def load(self, engine, device, precision):
        return ({"name": engine, "device": device, "precision": precision},)


# ============================================================== Character def
class SpeechGodCharacter:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "character_name": ("STRING", {"default": "Character"}),
                "gender": (GENDERS, {"default": "neutral"}),
                "age": (AGES, {"default": "adult"}),
                "tone": (TONES, {"default": "neutral"}),
                "energy": ("INT", {"default": 50, "min": 0, "max": 100}),
                "speech_speed": ("FLOAT", {"default": 1.0, "min": 0.5, "max": 2.0, "step": 0.05}),
                "accent_strength": ("INT", {"default": 0, "min": 0, "max": 100}),
            },
            "optional": {
                "voice_reference": ("AUDIO",),
                "emotion_reference": ("AUDIO",),
                "emotion_blend": ("SPEECHGOD_EMOTION",),
                "style_prompt": ("STRING", {"default": "", "multiline": True}),
                "scene_context": ("STRING", {"default": ""}),
                "reference_text": ("STRING", {"default": "", "multiline": True,
                                              "tooltip": "Transcript of the voice reference (improves cloning; F5 auto-transcribes when empty)"}),
                "laugh": ("INT", {"default": 0, "min": 0, "max": 100}),
                "whisper": ("INT", {"default": 0, "min": 0, "max": 100}),
                "nervousness": ("INT", {"default": 0, "min": 0, "max": 100}),
                "confidence": ("INT", {"default": 0, "min": 0, "max": 100}),
                "dramatic_intensity": ("INT", {"default": 0, "min": 0, "max": 100}),
                "sarcasm": ("INT", {"default": 0, "min": 0, "max": 100}),
                "cuteness": ("INT", {"default": 0, "min": 0, "max": 100}),
            },
        }

    RETURN_TYPES = ("SPEECHGOD_CHARACTER",)
    RETURN_NAMES = ("character",)
    FUNCTION = "build"
    CATEGORY = CATEGORY

    def build(self, character_name, gender, age, tone, energy, speech_speed,
              accent_strength, voice_reference=None, emotion_reference=None,
              emotion_blend=None, style_prompt="", scene_context="",
              reference_text="", laugh=0, whisper=0, nervousness=0,
              confidence=0, dramatic_intensity=0, sarcasm=0, cuteness=0):
        character = {
            "character_name": character_name, "gender": gender, "age": age,
            "tone": tone, "energy": energy, "speech_speed": speech_speed,
            "accent_strength": accent_strength, "style_prompt": style_prompt,
            "scene_context": scene_context, "reference_text": reference_text,
            "laugh": laugh, "whisper": whisper, "nervousness": nervousness,
            "confidence": confidence, "dramatic_intensity": dramatic_intensity,
            "sarcasm": sarcasm, "cuteness": cuteness,
            "emotion_blend": emotion_blend,
        }
        if voice_reference is not None:
            character["voice_reference"] = {
                "waveform": voice_reference["waveform"],
                "sample_rate": voice_reference["sample_rate"],
            }
        if emotion_reference is not None:
            character["emotion_reference"] = {
                "waveform": emotion_reference["waveform"],
                "sample_rate": emotion_reference["sample_rate"],
            }
        return (character,)


# ============================================================== Emotion blend
class SpeechGodEmotionBlend:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "emotion_a": (TONES, {"default": "happy"}),
            "weight_a": ("INT", {"default": 70, "min": 0, "max": 100}),
            "emotion_b": (TONES, {"default": "excited"}),
            "weight_b": ("INT", {"default": 30, "min": 0, "max": 100}),
        }}

    RETURN_TYPES = ("SPEECHGOD_EMOTION",)
    RETURN_NAMES = ("emotion_blend",)
    FUNCTION = "blend"
    CATEGORY = CATEGORY

    def blend(self, emotion_a, weight_a, emotion_b, weight_b):
        return (blend_tones(emotion_a, weight_a / 100.0, emotion_b, weight_b / 100.0),)


# ============================================================ Character DB IO
class SpeechGodSaveCharacter:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"character": ("SPEECHGOD_CHARACTER",)}}

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("saved_as",)
    FUNCTION = "save"
    CATEGORY = CATEGORY
    OUTPUT_NODE = True

    def save(self, character):
        name = chardb.save_character(character)
        return (f"characters/{name}.json (+ .voice if reference attached)",)


class SpeechGodLoadCharacter:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "character_file": (chardb.list_characters(),),
        }}

    RETURN_TYPES = ("SPEECHGOD_CHARACTER",)
    RETURN_NAMES = ("character",)
    FUNCTION = "load"
    CATEGORY = CATEGORY

    def load(self, character_file):
        return (chardb.load_character(character_file),)


class SpeechGodVoiceEvolution:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "character": ("SPEECHGOD_CHARACTER",),
                "progress": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
            },
            "optional": {"evolve_toward": ("SPEECHGOD_CHARACTER",)},
        }

    RETURN_TYPES = ("SPEECHGOD_CHARACTER",)
    RETURN_NAMES = ("character",)
    FUNCTION = "evolve"
    CATEGORY = CATEGORY

    def evolve(self, character, progress, evolve_toward=None):
        return (chardb.evolve(character, evolve_toward, progress),)


# ======================================================================= Cast
class SpeechGodCast:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {"character_1": ("SPEECHGOD_CHARACTER",)},
            "optional": {
                "character_2": ("SPEECHGOD_CHARACTER",),
                "character_3": ("SPEECHGOD_CHARACTER",),
                "character_4": ("SPEECHGOD_CHARACTER",),
                "cast": ("SPEECHGOD_CAST", {"tooltip": "Chain another Cast node here for unlimited characters"}),
            },
        }

    RETURN_TYPES = ("SPEECHGOD_CAST",)
    RETURN_NAMES = ("cast",)
    FUNCTION = "build"
    CATEGORY = CATEGORY

    def build(self, character_1, character_2=None, character_3=None,
              character_4=None, cast=None):
        out = dict(cast) if cast else {}
        for c in (character_1, character_2, character_3, character_4):
            if c is not None:
                out[c["character_name"].strip().upper()] = c
        return (out,)


# ================================================================== Generate
class SpeechGodGenerate:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "engine": ("SPEECHGOD_ENGINE",),
                "cast": ("SPEECHGOD_CAST",),
                "dialogue": ("STRING", {"default": "Hello, my name is Timmy.", "multiline": True}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFF,
                                 "control_after_generate": True}),
                "variation_strength": ("INT", {"default": 20, "min": 0, "max": 100}),
                "takes": ("INT", {"default": 1, "min": 1, "max": 8}),
                "merge_lines": ("BOOLEAN", {"default": True}),
                "pause_between_lines_ms": ("INT", {"default": 350, "min": 0, "max": 5000}),
            },
        }

    RETURN_TYPES = ("AUDIO", "STRING")
    RETURN_NAMES = ("audio", "report")
    FUNCTION = "generate"
    CATEGORY = CATEGORY

    def generate(self, engine, cast, dialogue, seed, variation_strength,
                 takes, merge_lines, pause_between_lines_ms):
        eng = engines.get_engine(engine["name"], engine["device"], engine["precision"])
        lines = parse_dialogue(dialogue)
        if not lines:
            raise ValueError("Speech-God: dialogue is empty")
        default_char = next(iter(cast.values()))
        variation = variation_strength / 100.0
        target_sr = 44100
        report, take_waves = [], []

        for take in range(takes):
            rendered = []
            for li, (speaker, text) in enumerate(lines):
                character = cast.get((speaker or "").strip().upper(), default_char)
                wave, sr = self._render_line(
                    eng, character, text,
                    seed=seed + take * 99991 + li * 17,
                    variation=variation if (takes > 1 or variation > 0) else 0.0,
                    take=take,
                )
                wave = dsp.resample(wave, sr, target_sr)
                rendered.append(wave)
                report.append(
                    f"take {take + 1} | line {li + 1} | "
                    f"{character['character_name']}: {text[:60]}"
                )
            take_waves.append(
                dsp.concat_with_pauses(rendered, target_sr, pause_between_lines_ms)
                if merge_lines else
                dsp.concat_with_pauses(rendered, target_sr, pause_between_lines_ms)
            )

        if takes > 1:
            # batch the takes (pad to the longest)
            longest = max(w.shape[-1] for w in take_waves)
            ch = max(w.shape[0] for w in take_waves)
            batch = torch.zeros(takes, ch, longest)
            for i, w in enumerate(take_waves):
                if w.shape[0] < ch:
                    w = w.repeat(ch, 1)
                batch[i, :, : w.shape[-1]] = w
            audio = {"waveform": batch, "sample_rate": target_sr}
        else:
            audio = _audio(take_waves[0], target_sr)
        return (audio, "\n".join(report))

    @staticmethod
    def _render_line(eng, character, text, seed, variation, take):
        delivery = resolve_delivery(character, character.get("emotion_blend"))

        ref = character.get("voice_reference")
        if ref is not None:
            ref_wave = dsp.to_ct(ref["waveform"])
            ref_sr = ref["sample_rate"]
        else:
            ref_wave, ref_sr = engines.default_voice(
                character.get("gender", "neutral"), character.get("age", "adult"))

        # emotion reference: prepend its pacing/emotion to the cloning reference
        emo = character.get("emotion_reference")
        if emo is not None and ref_wave is not None:
            emo_wave = dsp.resample(dsp.to_ct(emo["waveform"]), emo["sample_rate"], ref_sr)
            ref_wave = torch.cat([emo_wave, dsp.silence(ref_sr, 200, emo_wave.shape[0]),
                                  ref_wave], dim=-1)

        # alternate takes: jitter delivery slightly, scaled by variation
        g = torch.Generator().manual_seed(seed)
        jitter = lambda scale: (torch.rand(1, generator=g).item() - 0.5) * 2 * scale * variation

        wave, sr = eng.synthesize(
            text=text, ref_wave=ref_wave, ref_sr=ref_sr,
            ref_text=character.get("reference_text", ""),
            style_text=delivery["style_text"], markers=delivery["markers"],
            speed=delivery["tempo"] * (1.0 + jitter(0.08)),
            seed=seed, variation=variation,
        )
        wave = dsp.to_ct(wave)

        # engine-agnostic delivery shaping
        pitch = delivery["pitch"] + character.get("_evolve_pitch", 0.0) + jitter(0.8)
        wave = dsp.pitch_shift(wave, sr, pitch)
        if eng.name == "f5-tts":
            pass  # tempo already applied via native speed parameter
        else:
            wave = dsp.change_tempo(wave, sr, delivery["tempo"])
        wave = dsp.gain_db(wave, delivery["gain_db"])
        # whisper slider: soften + high-pass feel via mid/high EQ
        if character.get("whisper", 0) >= 50:
            wave = dsp.eq_3band(wave, sr, low_db=-6.0, high_db=2.0)
            wave = dsp.gain_db(wave, -4.0)
        return dsp.peak_normalize(wave.clamp(-1, 1), 0.95), sr


# ============================================================ Post processing
class SpeechGodPostProcess:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "audio": ("AUDIO",),
            "noise_reduction": ("BOOLEAN", {"default": False}),
            "loudness_normalize": ("BOOLEAN", {"default": True}),
            "target_db": ("FLOAT", {"default": -18.0, "min": -36.0, "max": -6.0, "step": 0.5}),
            "compressor": ("BOOLEAN", {"default": True}),
            "limiter": ("BOOLEAN", {"default": True}),
            "eq": ("BOOLEAN", {"default": False}),
            "eq_low_db": ("FLOAT", {"default": 0.0, "min": -12.0, "max": 12.0, "step": 0.5}),
            "eq_mid_db": ("FLOAT", {"default": 0.0, "min": -12.0, "max": 12.0, "step": 0.5}),
            "eq_high_db": ("FLOAT", {"default": 0.0, "min": -12.0, "max": 12.0, "step": 0.5}),
            "de_breath": ("BOOLEAN", {"default": False}),
            "trim_silence": ("BOOLEAN", {"default": True}),
        }}

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "process"
    CATEGORY = CATEGORY

    def process(self, audio, noise_reduction, loudness_normalize, target_db,
                compressor, limiter, eq, eq_low_db, eq_mid_db, eq_high_db,
                de_breath, trim_silence):
        sr = audio["sample_rate"]
        batch = audio["waveform"]
        outs = []
        for b in range(batch.shape[0]):
            w = dsp.to_ct(batch[b])
            if trim_silence:
                w = dsp.trim_silence(w, sr)
            if noise_reduction:
                w = dsp.noise_reduction(w, sr)
            if de_breath:
                w = dsp.de_breath(w, sr)
            if eq:
                w = dsp.eq_3band(w, sr, eq_low_db, eq_mid_db, eq_high_db)
            if compressor:
                w = dsp.compressor(w)
            if loudness_normalize:
                w = dsp.rms_normalize(w, target_db)
            if limiter:
                w = dsp.limiter(w)
            outs.append(w)
        longest = max(w.shape[-1] for w in outs)
        ch = max(w.shape[0] for w in outs)
        out = torch.zeros(len(outs), ch, longest)
        for i, w in enumerate(outs):
            if w.shape[0] < ch:
                w = w.repeat(ch, 1)
            out[i, :, : w.shape[-1]] = w
        return ({"waveform": out, "sample_rate": sr},)


# ===================================================================== Export
class SpeechGodExport:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "audio": ("AUDIO",),
            "filename_prefix": ("STRING", {"default": "speech-god/dialogue"}),
            "format": (["wav", "flac", "mp3"], {"default": "wav"}),
            "sample_rate": (["44100", "48000", "22050"], {"default": "44100"}),
        }}

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("saved_files",)
    FUNCTION = "export"
    CATEGORY = CATEGORY
    OUTPUT_NODE = True

    def export(self, audio, filename_prefix, format, sample_rate):
        import torchaudio
        sr_out = int(sample_rate)
        full_dir, filename, counter, subfolder, _ = folder_paths.get_save_image_path(
            filename_prefix, folder_paths.get_output_directory())
        saved = []
        batch = audio["waveform"]
        for b in range(batch.shape[0]):
            w = dsp.resample(dsp.to_ct(batch[b]), audio["sample_rate"], sr_out)
            name = f"{filename}_{counter + b:05}.{format}"
            path = os.path.join(full_dir, name)
            if format == "mp3":
                try:
                    torchaudio.save(path, w, sr_out, format="mp3")
                except Exception:
                    # fallback: write wav then convert with ffmpeg (bundled with ComfyUI's av)
                    tmp = path[:-4] + ".tmp.wav"
                    torchaudio.save(tmp, w, sr_out)
                    import subprocess
                    subprocess.run(["ffmpeg", "-y", "-i", tmp, "-b:a", "192k", path],
                                   capture_output=True)
                    os.unlink(tmp)
            else:
                torchaudio.save(path, w, sr_out, format=format)
            saved.append(os.path.join(subfolder, name))
        return ("\n".join(saved),)


# ================================================================= Mappings
NODE_CLASS_MAPPINGS = {
    "SpeechGodLoader": SpeechGodLoader,
    "SpeechGodCharacter": SpeechGodCharacter,
    "SpeechGodEmotionBlend": SpeechGodEmotionBlend,
    "SpeechGodSaveCharacter": SpeechGodSaveCharacter,
    "SpeechGodLoadCharacter": SpeechGodLoadCharacter,
    "SpeechGodVoiceEvolution": SpeechGodVoiceEvolution,
    "SpeechGodCast": SpeechGodCast,
    "SpeechGodGenerate": SpeechGodGenerate,
    "SpeechGodPostProcess": SpeechGodPostProcess,
    "SpeechGodExport": SpeechGodExport,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SpeechGodLoader": "Speech-God Loader",
    "SpeechGodCharacter": "Speech-God Character",
    "SpeechGodEmotionBlend": "Speech-God Emotion Blend",
    "SpeechGodSaveCharacter": "Speech-God Save Character",
    "SpeechGodLoadCharacter": "Speech-God Load Character",
    "SpeechGodVoiceEvolution": "Speech-God Voice Evolution",
    "SpeechGodCast": "Speech-God Cast",
    "SpeechGodGenerate": "Speech-God Generate",
    "SpeechGodPostProcess": "Speech-God Post Process",
    "SpeechGodExport": "Speech-God Export",
}
