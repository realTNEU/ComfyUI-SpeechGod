"""TTS engine wrappers - common interface for F5-TTS and Fish Speech.

Both engines are reference-audio voice-cloning models. The common contract:

    engine.synthesize(text, ref_wave, ref_sr, ref_text, style_text,
                      markers, speed, seed, variation) -> (wave [C,T], sr)

If no reference audio is supplied, a default voice is looked up in
assets/default_voices/<gender>_<age>.wav and, failing that, a clear error
explains what to install/provide. Engines are lazy-loaded and cached.
"""
import gc
import os
import random
import tempfile

import torch
import torchaudio

PKG_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_VOICES = os.path.join(PKG_DIR, "assets", "default_voices")

_ENGINE_CACHE = {}


# --------------------------------------------------------------- audio I/O
# torchaudio >= 2.8 routes ALL load/save through torchcodec, which needs FFmpeg
# shared libraries on the system. On many Windows ComfyUI installs those DLLs
# are absent, so torchaudio.load/save raise "Could not load libtorchcodec".
# soundfile (libsndfile, bundled in its wheel) has no such dependency, so we
# read/write WAV via soundfile and, when torchcodec is broken, transparently
# shim torchaudio.load/save with soundfile-backed versions. This also fixes
# F5-TTS, which calls torchaudio.load() on the reference internally.
_IO_PATCHED = False


def _sf_load(filepath, frame_offset=0, num_frames=-1, normalize=True,
             channels_first=True, format=None, buffer_size=4096, backend=None):
    import soundfile as sf
    start = int(frame_offset) or 0
    stop = None if (num_frames is None or num_frames < 0) else start + int(num_frames)
    data, sr = sf.read(str(filepath), start=start, stop=stop,
                       dtype="float32", always_2d=True)  # [T, C]
    wave = torch.from_numpy(data.copy())
    wave = wave.t().contiguous() if channels_first else wave   # -> [C, T]
    return wave, sr


def _sf_save(filepath, src, sample_rate, channels_first=True, format=None,
             encoding=None, bits_per_sample=None, buffer_size=4096, backend=None,
             compression=None):
    import numpy as np  # noqa: F401
    import soundfile as sf
    arr = src.detach().cpu().float().numpy()
    if arr.ndim == 1:
        arr = arr[None, :]
    if channels_first:
        arr = arr.T                                            # [C,T] -> [T,C]
    sf.write(str(filepath), arr, int(sample_rate))


def ensure_audio_io():
    """Install soundfile-backed torchaudio.load/save shims iff torchcodec is
    broken. Idempotent; safe no-op when torchaudio I/O already works."""
    global _IO_PATCHED
    if _IO_PATCHED:
        return
    _IO_PATCHED = True
    try:
        from torchcodec.decoders import AudioDecoder  # noqa: F401 - triggers DLL load
        return  # torchcodec works -> leave native torchaudio in place
    except Exception:
        pass
    try:
        import soundfile  # noqa: F401 - confirm fallback is available
    except Exception:
        return  # nothing we can do; let the native error surface
    torchaudio.load = _sf_load
    torchaudio.save = _sf_save


def get_engine(name: str, device: str = "auto", precision: str = "auto"):
    key = (name, device, precision)
    if key not in _ENGINE_CACHE:
        if name == "f5-tts":
            _ENGINE_CACHE[key] = F5Engine(device, precision)
        elif name == "fish-speech":
            _ENGINE_CACHE[key] = FishEngine(device, precision)
        else:
            raise ValueError(f"Speech-God: unknown engine '{name}'")
    return _ENGINE_CACHE[key]


def unload_all():
    _ENGINE_CACHE.clear()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _resolve_device(device):
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


def _bundled_reference():
    """F5-TTS ships an English reference clip with a known transcript. We use it
    as a last-resort default so parameter-only characters render *something*
    instead of crashing. Returning the known transcript skips Whisper
    auto-transcription (avoids a ~1.6 GB download and a slow first run)."""
    try:
        import importlib.util
        spec = importlib.util.find_spec("f5_tts")
        locs = list(getattr(spec, "submodule_search_locations", None) or [])
        if not locs:  # fall back to a submodule that has a real __file__
            import f5_tts.api as _a
            locs = [os.path.dirname(os.path.abspath(_a.__file__))]
        wav = os.path.join(locs[0], "infer", "examples", "basic", "basic_ref_en.wav")
        if os.path.exists(wav):
            wave, sr = _sf_load(wav)
            return wave, sr, "Some call me nature, others call me mother nature."
    except Exception:
        pass
    return None, None, ""


def default_voice(gender: str, age: str):
    """Find a default reference voice for parameter-only characters.

    Priority: user-supplied wavs in assets/default_voices/  ->  F5-TTS's
    bundled example clip. Returns (wave [C,T], sr, ref_text). ref_text is ""
    for user wavs (engine will auto-transcribe) and the known transcript for
    the bundled fallback."""
    candidates = [
        f"{gender}_{age}.wav", f"{gender}.wav", "neutral_adult.wav", "default.wav",
    ]
    for c in candidates:
        p = os.path.join(DEFAULT_VOICES, c)
        if os.path.exists(p):
            wave, sr = _sf_load(p)
            return wave, sr, ""
    return _bundled_reference()


def _temp_wav(wave, sr):
    f = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    f.close()
    _sf_save(f.name, wave.cpu(), sr)
    return f.name


class BaseEngine:
    name = "base"

    def __init__(self, device="auto", precision="auto"):
        self.device = _resolve_device(device)
        self.precision = precision
        self._model = None

    def synthesize(self, text, ref_wave, ref_sr, ref_text="", style_text="",
                   markers=None, speed=1.0, seed=0, variation=0.0):
        raise NotImplementedError

    def _seed(self, seed):
        seed = int(seed) % (2 ** 31)
        torch.manual_seed(seed)
        random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        return seed


class F5Engine(BaseEngine):
    """F5-TTS (SWivid/F5-TTS, pip package `f5-tts`). Reference-based cloning;
    ref_text='' triggers automatic transcription of the reference."""
    name = "f5-tts"

    def _load(self):
        if self._model is None:
            ensure_audio_io()  # F5-TTS reads the reference via torchaudio.load
            try:
                from f5_tts.api import F5TTS
            except ImportError as e:
                raise RuntimeError(
                    "Speech-God: F5-TTS is not installed in the ComfyUI venv.\n"
                    "Install with:  <ComfyUI>\\.venv\\Scripts\\pip.exe install f5-tts\n"
                    "Models auto-download from HuggingFace (SWivid/F5-TTS) on first run."
                ) from e
            self._model = F5TTS(model="F5TTS_v1_Base", device=self.device)
        return self._model

    def synthesize(self, text, ref_wave, ref_sr, ref_text="", style_text="",
                   markers=None, speed=1.0, seed=0, variation=0.0):
        model = self._load()
        seed = self._seed(seed)
        if ref_wave is None:
            raise RuntimeError(
                "Speech-God [f5-tts]: no voice reference available. Connect a "
                "'Voice Reference' (LoadAudio), load a saved character, or drop "
                f"default wavs into {DEFAULT_VOICES}\\<gender>_<age>.wav"
            )
        ref_path = _temp_wav(ref_wave, ref_sr)
        try:
            wav, sr, _ = model.infer(
                ref_file=ref_path,
                ref_text=ref_text or "",
                gen_text=text,
                seed=seed,
                speed=float(max(0.5, min(2.0, speed))),
            )
        finally:
            try:
                os.unlink(ref_path)
            except OSError:
                pass
        wave = torch.from_numpy(wav).float()
        if wave.dim() == 1:
            wave = wave.unsqueeze(0)
        return wave, sr


class FishEngine(BaseEngine):
    """Fish Speech / OpenAudio (fishaudio/fish-speech). Supports inline
    emotion markers like (excited), (whispering) inside the text.

    Loading strategies, in order:
      1. `fish_speech` python package installed into the ComfyUI venv
      2. FISH_SPEECH_DIR env var pointing at a fish-speech checkout
    """
    name = "fish-speech"

    def _load(self):
        if self._model is None:
            ensure_audio_io()
            try:
                self._model = self._load_package()
            except ImportError as e:
                raise RuntimeError(
                    "Speech-God: Fish Speech is not installed.\n"
                    "Option A: <ComfyUI>\\.venv\\Scripts\\pip.exe install fish-speech\n"
                    "Option B: git clone https://github.com/fishaudio/fish-speech, "
                    "pip install -e ., and set env var FISH_SPEECH_DIR to the checkout.\n"
                    "Model: huggingface.co/fishaudio/openaudio-s1-mini -> "
                    "models/checkpoints/openaudio-s1-mini/"
                ) from e
        return self._model

    def _load_package(self):
        import importlib
        if os.environ.get("FISH_SPEECH_DIR"):
            import sys
            sys.path.insert(0, os.environ["FISH_SPEECH_DIR"])
        mod = importlib.import_module("fish_speech")  # noqa: F841 - probe import
        from fish_speech.inference_engine import TTSInferenceEngine
        from fish_speech.models.dac.inference import load_model as load_decoder_model
        from fish_speech.models.text2semantic.inference import launch_thread_safe_queue

        ckpt = self._find_checkpoint()
        half = self.precision != "fp32" and self.device == "cuda"
        llama_queue = launch_thread_safe_queue(
            checkpoint_path=ckpt, device=self.device,
            precision=torch.half if half else torch.float32, compile=False,
        )
        decoder = load_decoder_model(
            config_name="modded_dac_vq",
            checkpoint_path=os.path.join(ckpt, "codec.pth"),
            device=self.device,
        )
        return TTSInferenceEngine(
            llama_queue=llama_queue, decoder_model=decoder,
            precision=torch.half if half else torch.float32, compile=False,
        )

    @staticmethod
    def _find_checkpoint():
        try:
            import folder_paths
            roots = folder_paths.get_folder_paths("checkpoints")
        except Exception:
            roots = []
        for root in roots:
            for name in ("openaudio-s1-mini", "fish-speech-1.5", "fish-speech"):
                p = os.path.join(root, name)
                if os.path.isdir(p):
                    return p
        raise RuntimeError(
            "Speech-God [fish-speech]: model folder not found. Download "
            "huggingface.co/fishaudio/openaudio-s1-mini into "
            "ComfyUI/models/checkpoints/openaudio-s1-mini/"
        )

    def synthesize(self, text, ref_wave, ref_sr, ref_text="", style_text="",
                   markers=None, speed=1.0, seed=0, variation=0.0):
        engine = self._load()
        seed = self._seed(seed)
        from fish_speech.utils.schema import ServeReferenceAudio, ServeTTSRequest

        # fish speech understands inline markers -> prepend them to the text
        marked = " ".join(markers or []) + (" " if markers else "") + text
        references = []
        if ref_wave is not None:
            import io
            buf = io.BytesIO()
            torchaudio.save(buf, ref_wave.cpu(), ref_sr, format="wav")
            references = [ServeReferenceAudio(audio=buf.getvalue(), text=ref_text or "")]

        temperature = 0.6 + 0.35 * float(variation)      # variation -> sampling temp
        req = ServeTTSRequest(
            text=marked, references=references, reference_id=None,
            max_new_tokens=2048, chunk_length=300,
            top_p=0.8, repetition_penalty=1.1,
            temperature=min(1.0, temperature), seed=seed, format="wav",
        )
        result_wave, result_sr = None, 44100
        for chunk in engine.inference(req):
            if chunk.code == "final":
                result_sr, audio = chunk.audio
                result_wave = torch.from_numpy(audio).float()
        if result_wave is None:
            raise RuntimeError("Speech-God [fish-speech]: engine returned no audio")
        if result_wave.dim() == 1:
            result_wave = result_wave.unsqueeze(0)
        return result_wave, result_sr
