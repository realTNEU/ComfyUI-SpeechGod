# Speech-God 🎙

Expressive character-dialogue TTS suite for ComfyUI. Fully local, ElevenLabs-style
controls, two interchangeable engines:

| Engine | Strengths | Install weight |
|---|---|---|
| **F5-TTS** (default) | excellent zero-shot voice cloning, native speed control, auto-transcribes references | light (`pip install f5-tts`) |
| **Fish Speech / OpenAudio S1** | inline emotion markers `(excited)`, `(whispering)`, temperature-based variation | heavier (source install) |

Both are **voice-cloning** models: every voice starts from reference audio.
Speech-God layers character parameters (age, gender, tone, energy, micro-expressions)
on top via prompt conditioning + DSP so each character is distinct and repeatable.

---

## Architecture

```
Speech-God Loader        (engine dropdown: f5-tts / fish-speech)
        ↓
Character Definition     (name, gender, age, tone, energy, speed, accent,
        ↓                 style prompt, scene context, 7 micro-expression sliders,
        ↓                 optional: voice ref, emotion ref, emotion blend)
Dialogue Parser          ([NAME] tags / NAME: lines / plain text → routed lines)
        ↓
Emotion Processor        (tone+age+gender+sliders → pitch/tempo/gain/markers/style)
        ↓
Voice Generation Engine  (F5-TTS or Fish Speech, per line, seeded takes)
        ↓
Post Processing          (NR, normalize, compressor, limiter, EQ, de-breath, trim)
        ↓
Audio Export             (wav / flac / mp3 @ 22050 / 44100 / 48000)
```

## Nodes

| Node | Purpose |
|---|---|
| **Speech-God Loader** | engine/device/precision selector |
| **Speech-God Character** | full character profile; optional AUDIO sockets for voice/emotion reference |
| **Speech-God Emotion Blend** | e.g. 70% happy + 30% excited → plug into Character |
| **Speech-God Cast** | groups characters; chain Cast→Cast for unlimited characters |
| **Speech-God Generate** | parses dialogue, routes lines, renders takes (1–8), merges with pauses |
| **Speech-God Save / Load Character** | character DB: `characters/<name>.json` + `<name>.voice` |
| **Speech-God Voice Evolution** | interpolate a voice toward another profile, or age it (progress 0–1) |
| **Speech-God Post Process** | all modules toggleable |
| **Speech-God Export** | wav/flac/mp3, 3 sample rates, batch-aware (each take = one file) |

---

## Installation

1. Dependencies (into the ComfyUI venv):
   ```
   C:\Users\TNEU\ComfyUI-Installs\ComfyUI\ComfyUI\.venv\Scripts\pip.exe install -r ^
     C:\Users\TNEU\ComfyUI-Installs\ComfyUI\ComfyUI\custom_nodes\ComfyUI-SpeechGod\requirements.txt
   ```
2. Restart ComfyUI. The 10 Speech-God nodes appear under category **Speech-God**.
3. Open `workflows/Speech-God.json` (also copied to your user workflows folder).

### F5-TTS (recommended first)
`pip install f5-tts` is all you need. Models (~1.4 GB) auto-download from
HuggingFace **SWivid/F5-TTS** on first generation into your HF cache.

### Fish Speech (optional second engine)
```
git clone https://github.com/fishaudio/fish-speech
cd fish-speech
C:\...\ComfyUI\.venv\Scripts\pip.exe install -e .
```
Download the model from **huggingface.co/fishaudio/openaudio-s1-mini** into:
```
ComfyUI/models/checkpoints/openaudio-s1-mini/
```
If you keep fish-speech outside the venv, set env var `FISH_SPEECH_DIR` to the checkout.

### Default voices
Cloning engines always need *some* reference voice. Speech-God falls back to the
reference clip bundled with `f5-tts`, so **parameter-only characters work out of
the box** — every character with no Voice Reference shares that neutral voice,
differentiated by pitch/tempo/energy. For distinct default voices, drop 5–15 s
WAV seed clips into `assets/default_voices/`:
```
female_child.wav   male_elderly.wav   neutral_adult.wav   default.wav   ...
```
Priority: `<gender>_<age>.wav` → `<gender>.wav` → `neutral_adult.wav` →
`default.wav` → the bundled f5-tts clip. (For truly distinct characters, connect a
Voice Reference per character — that's what cloning is for.)

### Audio I/O note (Windows / torchaudio ≥ 2.8)
Recent torchaudio routes all `load`/`save` through **torchcodec**, which needs
FFmpeg shared libraries that many Windows installs lack (`Could not load
libtorchcodec`). Speech-God detects this at engine-load time and transparently
shims `torchaudio.load`/`save` with **soundfile** (libsndfile, no FFmpeg needed),
which also fixes F5-TTS's internal reference loading. No action required; if your
torchcodec works, the shim stays out of the way.

---

## Model download locations

| Model | Source | Size | Goes to |
|---|---|---|---|
| F5-TTS v1 Base | HF `SWivid/F5-TTS` | ~1.4 GB | auto (HF cache) |
| Vocos vocoder (F5 dep) | HF `charactr/vocos-mel-24khz` | ~50 MB | auto |
| Whisper (F5 auto-transcribe) | HF `openai/whisper-large-v3-turbo` | ~1.6 GB | auto, first time ref_text is empty |
| OpenAudio S1-mini | HF `fishaudio/openaudio-s1-mini` | ~2 GB | `models/checkpoints/openaudio-s1-mini/` |

## VRAM estimates

| Engine | Load | Per-line generation peak |
|---|---|---|
| F5-TTS fp16 | ~2.5 GB | 3–4 GB |
| F5-TTS + Whisper transcribe | +1.5 GB (first call per reference) | 5–6 GB |
| Fish Speech S1-mini fp16 | ~3.5 GB | 4–6 GB |

Post-processing is CPU-only. Generation memory is independent of script length
(lines render sequentially).

### RTX 3060 Ti profile (8 GB)
- Loader: `precision: fp16`, `device: auto`
- Use **one engine per session** (engines stay cached; don't alternate per queue)
- Fill `reference_text` on Characters → avoids loading Whisper (saves ~1.5 GB)
- If you run Speech-God alongside LTX-2.3 video: generate audio **first**, then
  free VRAM (ComfyUI → Free model and node cache) before video sampling
- takes ≤ 4 per queue

### RTX 3080 Ti profile (12 GB)
- Loader: `precision: fp16`
- Both engines can stay resident if you don't co-run a video model
- Whisper auto-transcribe is fine; takes up to 8
- Batch hundreds of lines in one Generate call (sequential, VRAM-flat)

---

## Usage notes

**Dialogue formats** (mix freely):
```
[SQUIRREL]              SQUIRREL: Hey elephant!        Hello, my name is Timmy.
Hey elephant!           [0:03] Zumboo: "..."           (single voice = whole text)
```
Untagged lines inherit the previous speaker (or the first cast character).

**Alternate takes**: `takes = N` renders N seeded variants; `variation_strength`
scales delivery jitter (speed/pitch) and Fish Speech sampling temperature.
Output is a batched AUDIO; Export writes `take_00001.wav`, `take_00002.wav`, ...

**Character consistency across episodes**: Save Character once → `characters/squirrel.json`
(+ `.voice` with the reference audio embedded) → Load Character anywhere.

**Voice Evolution**: connect a second character to `evolve_toward` and sweep
`progress` 0→1 across episodes (gradual aging/growth), or leave it empty for a
natural aging drift (pitch falls, tempo slows).

**Emotion reference**: a short clip whose *pacing and emotion* you want; it is
prepended to the cloning reference so the engine absorbs its delivery.

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| `F5-TTS is not installed` | install requirements.txt into **the venv**, not system python |
| `Could not load libtorchcodec` | handled automatically (soundfile shim). If it still appears, `pip install soundfile` into the venv |
| `no voice reference available` | only if f5-tts isn't importable; otherwise the bundled default voice is used. Connect LoadAudio or add seed wavs to `assets/default_voices/` for distinct voices |
| Very slow / "process exited" right after install | f5-tts pulls a large dependency train (gradio/fastapi/bitsandbytes) that ComfyUI scans on cold start; none are needed for inference — safe to `pip uninstall` them. Add a Defender exclusion for the ComfyUI folder |
| `fish-speech model folder not found` | download openaudio-s1-mini into `models/checkpoints/openaudio-s1-mini/` |
| Robotic / wrong-pitch output | lower the micro sliders; extreme age+tone combos stack pitch (child+excited ≈ +7 st) — reduce energy or pick `young_adult` |
| Cloned voice ignores emotion | F5 follows the reference's emotion: use an Emotion Reference clip or switch to fish-speech (markers) |
| First run very slow | model download + (F5) Whisper transcription of the reference; fill `reference_text` to skip |
| MP3 export fails | ffmpeg not on PATH — use wav/flac, or install ffmpeg |
| CUDA OOM on 8 GB | see 3060 Ti profile; ensure no video model is resident |
| Lines routed to wrong character | character_name must match the script tag (case-insensitive): `[SQUIRREL]` ↔ name `SQUIRREL` |

## Example workflows (in `workflows/`)

- `Speech-God.json` — flagship: 3 characters, cloning sockets, emotion blend, full chain
- `example_single_character.json`
- `example_multi_character.json`
- `example_voice_cloning.json` (+ Save Character)
- `example_cartoon_characters.json` (2 takes, high-cuteness profiles)
- `example_narrator.json` (storyteller/audiobook)
