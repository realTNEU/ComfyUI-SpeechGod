"""Audio DSP utilities - pure torch/torchaudio, Windows-safe (no sox binary).

All functions take/return mono-or-multichannel float tensors shaped [C, T]
at a given sample rate, values in [-1, 1].
"""
import math

import torch
import torchaudio
import torchaudio.functional as AF


# ---------------------------------------------------------------- basics
def to_ct(wave: torch.Tensor) -> torch.Tensor:
    """Accept [T], [C,T] or [B,C,T] (first batch item) -> [C,T] float32 cpu."""
    if wave.dim() == 3:
        wave = wave[0]
    if wave.dim() == 1:
        wave = wave.unsqueeze(0)
    return wave.float().cpu()


def resample(wave, sr_from, sr_to):
    if sr_from == sr_to:
        return wave
    return AF.resample(wave, sr_from, sr_to)


def gain_db(wave, db):
    return wave * (10.0 ** (db / 20.0))


def peak_normalize(wave, peak=0.97):
    m = wave.abs().max()
    return wave if m < 1e-8 else wave * (peak / m)


# ---------------------------------------------------------------- pitch / tempo
def pitch_shift(wave, sr, semitones):
    if abs(semitones) < 0.05:
        return wave
    return AF.pitch_shift(wave, sr, n_steps=float(semitones))


def change_tempo(wave, sr, factor):
    """Pitch-preserving tempo change: resample (changes speed+pitch),
    then pitch-shift back. factor > 1 = faster."""
    if abs(factor - 1.0) < 0.01:
        return wave
    new_len = int(wave.shape[-1] / factor)
    sped = torch.nn.functional.interpolate(
        wave.unsqueeze(0), size=new_len, mode="linear", align_corners=False
    ).squeeze(0)
    semis = -12.0 * math.log2(1.0 / factor)  # compensate the pitch change
    return AF.pitch_shift(sped, sr, n_steps=-semis)


# ---------------------------------------------------------------- post chain
def trim_silence(wave, sr, threshold_db=-45.0, pad_ms=60):
    env = wave.abs().max(dim=0).values
    thresh = 10.0 ** (threshold_db / 20.0)
    idx = torch.nonzero(env > thresh)
    if idx.numel() == 0:
        return wave
    pad = int(sr * pad_ms / 1000)
    start = max(0, int(idx[0]) - pad)
    end = min(wave.shape[-1], int(idx[-1]) + pad)
    return wave[:, start:end]


def rms_normalize(wave, target_db=-18.0):
    rms = wave.pow(2).mean().sqrt()
    if rms < 1e-8:
        return wave
    target = 10.0 ** (target_db / 20.0)
    out = wave * (target / rms)
    return out.clamp(-1.0, 1.0)


def compressor(wave, threshold_db=-20.0, ratio=3.0, makeup_db=2.0):
    """Simple static soft compressor on the sample envelope."""
    thresh = 10.0 ** (threshold_db / 20.0)
    mag = wave.abs()
    over = mag > thresh
    compressed = torch.where(
        over, thresh * (mag / thresh).pow(1.0 / ratio), mag
    )
    out = torch.sign(wave) * compressed
    return gain_db(out, makeup_db).clamp(-1.0, 1.0)


def limiter(wave, ceiling_db=-1.0):
    ceiling = 10.0 ** (ceiling_db / 20.0)
    return torch.tanh(wave / ceiling) * ceiling


def eq_3band(wave, sr, low_db=0.0, mid_db=0.0, high_db=0.0):
    out = wave
    if abs(low_db) > 0.1:
        out = AF.equalizer_biquad(out, sr, center_freq=120.0, gain=low_db, Q=0.7)
    if abs(mid_db) > 0.1:
        out = AF.equalizer_biquad(out, sr, center_freq=1200.0, gain=mid_db, Q=0.8)
    if abs(high_db) > 0.1:
        out = AF.equalizer_biquad(out, sr, center_freq=8000.0, gain=high_db, Q=0.7)
    return out.clamp(-1.0, 1.0)


def de_breath(wave, sr, amount=0.5):
    """Downward expander on quiet, high-frequency-dominant frames."""
    frame = max(256, int(sr * 0.02))
    n = wave.shape[-1] // frame
    if n < 2:
        return wave
    out = wave.clone()
    body = wave[:, : n * frame].reshape(wave.shape[0], n, frame)
    rms = body.pow(2).mean(dim=-1).sqrt().mean(dim=0)          # [n]
    quiet = rms < rms.median() * 0.35
    scale = torch.ones(n)
    scale[quiet] = 1.0 - 0.8 * amount
    scale = scale.repeat_interleave(frame).unsqueeze(0)
    out[:, : n * frame] = body.reshape(wave.shape[0], -1) * scale
    return out


def noise_reduction(wave, sr, amount=0.7):
    """Spectral-gate noise reduction. Uses `noisereduce` if available,
    falls back to a simple spectral floor gate."""
    try:
        import noisereduce as nr
        import numpy as np
        arr = wave.numpy()
        out = np.stack([
            nr.reduce_noise(y=arr[c], sr=sr, prop_decrease=amount, stationary=True)
            for c in range(arr.shape[0])
        ])
        return torch.from_numpy(out).float()
    except Exception:
        spec = torch.stft(wave, n_fft=1024, hop_length=256,
                          window=torch.hann_window(1024), return_complex=True)
        mag = spec.abs()
        floor = mag.median(dim=-1, keepdim=True).values * (0.5 + amount)
        mask = (mag > floor).float()
        mask = torch.nn.functional.avg_pool2d(
            mask.unsqueeze(1), kernel_size=3, stride=1, padding=1
        ).squeeze(1)
        spec = spec * (mask * (1 - 0.1) + 0.1)
        out = torch.istft(spec, n_fft=1024, hop_length=256,
                          window=torch.hann_window(1024), length=wave.shape[-1])
        return out


# ---------------------------------------------------------------- assembly
def silence(sr, ms, channels=1):
    return torch.zeros(channels, int(sr * ms / 1000.0))


def concat_with_pauses(waves, sr, pause_ms=350):
    """waves: list of [C,T] at the same sr -> single [C,T]."""
    if not waves:
        return silence(sr, 100)
    ch = max(w.shape[0] for w in waves)
    parts = []
    gap = silence(sr, pause_ms, ch)
    for i, w in enumerate(waves):
        if w.shape[0] < ch:
            w = w.repeat(ch, 1)
        parts.append(w)
        if i < len(waves) - 1:
            parts.append(gap)
    return torch.cat(parts, dim=-1)
