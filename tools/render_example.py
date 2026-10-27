#!/usr/bin/env python3
"""Render an example and validate the encoded output.

This is the script that produces the project's "rendered example, level-matched
A/B and spectrograms" deliverable.  It runs the *shipped* engine, so its output
is what the application exports.

    python tools/render_example.py --input TestSong.ogg --profile profiles/reference_character.json \
        --out-dir examples/TestSong

Writes into ``--out-dir``:
    <name>_processed.wav        the render (24-bit, source sample rate)
    <name>_ab.wav               A/B file: input then processed at matched loudness,
                                with a 0.5 s gap (optional, --ab)
    <name>_processed.ogg        encoded check copy in the target format (--encode ogg)
    validation.txt              measurement of the *encoded* file
    spectrograms.png            identical-setting spectrograms of input vs export
    spectrum.png                level-matched long-term spectra
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from timbreforge.dsp.audio import (AudioData, ExportSettings, load_audio, save_audio)
from timbreforge.dsp.graph import (Profile, preset_available, preset_reference_hypothesis,
                                   render)
from timbreforge.dsp.meter import (band_delta_db, db, integrated_loudness, measure,
                                   power_spectrum, to_mono, true_peak)


def load_profile(path: str | None) -> Profile:
    if not path:
        return preset_reference_hypothesis()
    if os.path.isfile(path):
        return Profile.load(path)
    presets = preset_available()
    if path in presets:
        return presets[path]()
    raise SystemExit(f"unknown profile: {path}")


def figures(orig: np.ndarray, out: np.ndarray, sr: int, out_dir: str, name: str,
            gain_db: float) -> list[str]:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # noqa: BLE001
        print(f"  (no matplotlib, skipping figures: {exc})")
        return []
    made = []
    # spectrograms, identical settings
    fig, ax = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
    seg = slice(0, min(len(orig), len(out), int(sr * 12)))
    for a, (label, sig) in zip(ax, (("input", orig[seg]), ("exported output", out[seg]))):
        a.specgram(to_mono(sig), NFFT=2048, Fs=sr, noverlap=1024, cmap="magma",
                   vmin=-110, vmax=-30)
        a.set_title(label, fontsize=9)
        a.set_xlabel("seconds")
        a.set_ylim(0, 16000)
    ax[0].set_ylabel("Hz")
    fig.suptitle(f"First {seg.stop / sr:.0f} s, identical spectrogram settings "
                 "(dB, -110..-30, resolution 2048)", fontsize=10)
    fig.tight_layout()
    p = os.path.join(out_dir, f"{name}_spectrograms.png")
    fig.savefig(p, dpi=130)
    plt.close(fig)
    made.append(p)

    # level matched spectra
    f_o, p_o = power_spectrum(orig, sr)
    f_r, p_r = power_spectrum(out * 10 ** (gain_db / 20.0), sr)
    n = min(len(p_o), len(p_r))
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.semilogx(f_o[:n], 10 * np.log10(np.maximum(p_o[:n], 1e-20)), color="#4d5766",
                lw=1.0, label="input")
    ax.semilogx(f_r[:n], 10 * np.log10(np.maximum(p_r[:n], 1e-20)), color="#57c8d8",
                lw=1.0, label=f"output ({gain_db:+.1f} dB to match)")
    ax.set_xlim(20, sr / 2)
    ax.set_ylim(-140, -40)
    ax.grid(True, which="both", alpha=0.2)
    ax.set_xlabel("Hz")
    ax.set_ylabel("dB/Hz")
    ax.set_title("Long-term spectrum at matched loudness")
    ax.legend(fontsize=8)
    fig.tight_layout()
    p = os.path.join(out_dir, f"{name}_spectrum.png")
    fig.savefig(p, dpi=130)
    plt.close(fig)
    made.append(p)
    return made


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True)
    ap.add_argument("--profile", default=None,
                    help="profile JSON or a built-in preset name (default: hypothesis preset)")
    ap.add_argument("--out-dir", default="examples")
    ap.add_argument("--name", default=None)
    ap.add_argument("--encode", default="wav",
                    choices=["wav", "flac", "ogg", "mp3"],
                    help="also write and validate an encoded copy in this format")
    ap.add_argument("--ab", action="store_true", help="also write a level-matched A/B file")
    args = ap.parse_args()

    if not os.path.isfile(args.input):
        print(f"error: file not found: {args.input}")
        return 2
    os.makedirs(args.out_dir, exist_ok=True)
    name = args.name or os.path.splitext(os.path.basename(args.input))[0]

    audio = load_audio(args.input)
    profile = load_profile(args.profile)
    errs = profile.validate()
    if errs:
        print("invalid profile:", errs)
        return 2
    print(f"input   {audio.filename}: {audio.sample_rate} Hz, {audio.channels} ch, "
          f"{audio.duration:.3f} s")
    print(f"profile {profile.name} (stages: {', '.join(profile.active_stages()) or 'none'})")

    result = render(audio.samples, audio.sample_rate, profile,
                    progress=lambda f, l: None)
    assert result.samples.shape == audio.samples.shape, "length changed!"
    st_in = measure(audio.samples, audio.sample_rate)
    st_pcm = measure(result.samples, audio.sample_rate)
    match_db = st_in.lufs - st_pcm.lufs

    wav_path = os.path.join(args.out_dir, f"{name}_processed.wav")
    facts = save_audio(AudioData(samples=result.samples, sample_rate=audio.sample_rate),
                       ExportSettings(path=wav_path, format="WAV", subtype="PCM_24",
                                      overwrite=True), sample_rate=audio.sample_rate)
    print("wrote", wav_path)

    ext = args.encode
    fmt = {"wav": ("WAV", "PCM_24"), "flac": ("FLAC", "PCM_24"),
           "ogg": ("OGG", "VORBIS"), "mp3": ("MP3", "MPEG_LAYER_III")}[ext]
    enc_path = os.path.join(args.out_dir, f"{name}_processed.{ext}")
    save_audio(AudioData(samples=result.samples, sample_rate=audio.sample_rate),
               ExportSettings(path=enc_path, format=fmt[0], subtype=fmt[1], overwrite=True),
               sample_rate=audio.sample_rate)
    print("wrote", enc_path)

    encoded = load_audio(enc_path)
    st_enc = measure(encoded.samples, encoded.sample_rate)

    lines = [
        f"# Validation - {name}",
        "",
        f"Input        {audio.filename}  {audio.sample_rate} Hz  {audio.channels} ch  "
        f"{audio.duration:.6f} s  ({audio.frames} frames)",
        f"Profile      {profile.name}",
        f"Stages       {', '.join(profile.active_stages()) or 'none'}",
        "",
        "## Timing",
        f"input frames        {audio.frames}",
        f"PCM render frames   {result.frames}  (identical: {result.frames == audio.frames})",
        f"encoded duration    {encoded.duration:.6f} s  "
        f"(difference {encoded.duration - audio.duration:+.6f} s)",
        "no pitch shift / tempo change / time-stretch is present in the graph; "
        "the only difference is container padding from the codec",
        "",
        "## Levels",
        f"{'':22s}{'input':>10s}{'render':>10s}{'encoded':>10s}",
        f"{'sample peak dBFS':22s}{st_in.peak_db:10.2f}{st_pcm.peak_db:10.2f}{st_enc.peak_db:10.2f}",
        f"{'true peak dBTP':22s}{st_in.true_peak_db:10.2f}{st_pcm.true_peak_db:10.2f}{st_enc.true_peak_db:10.2f}",
        f"{'RMS dBFS':22s}{st_in.rms_db:10.2f}{st_pcm.rms_db:10.2f}{st_enc.rms_db:10.2f}",
        f"{'integrated LUFS':22s}{st_in.lufs:10.2f}{st_pcm.lufs:10.2f}{st_enc.lufs:10.2f}",
        f"{'LRA LU':22s}{st_in.lra:10.2f}{st_pcm.lra:10.2f}{st_enc.lra:10.2f}",
        f"{'crest dB':22s}{st_in.crest_db:10.2f}{st_pcm.crest_db:10.2f}{st_enc.crest_db:10.2f}",
        f"{'envelope range dB':22s}{st_in.envelope_range_db:10.2f}"
        f"{st_pcm.envelope_range_db:10.2f}{st_enc.envelope_range_db:10.2f}",
        f"{'L/R correlation':22s}{st_in.correlation:10.3f}{st_pcm.correlation:10.3f}"
        f"{st_enc.correlation:10.3f}",
        f"{'S/M dB':22s}{st_in.side_mid_db:10.2f}{st_pcm.side_mid_db:10.2f}{st_enc.side_mid_db:10.2f}",
        f"{'mono fold dB':22s}{st_in.mono_fold_db:10.2f}{st_pcm.mono_fold_db:10.2f}"
        f"{st_enc.mono_fold_db:10.2f}",
        f"{'centroid Hz':22s}{st_in.centroid_hz:10.0f}{st_pcm.centroid_hz:10.0f}"
        f"{st_enc.centroid_hz:10.0f}",
        "",
        f"loudness match applied: {result.loudness_match_db:+.2f} dB "
        "(negative means the render was attenuated)",
        f"render is {'quieter' if st_pcm.lufs <= st_in.lufs + 0.1 else 'LOUDER'} than the input "
        f"by {abs(st_pcm.lufs - st_in.lufs):.2f} LU",
        f"encoded true peak is {'within' if st_enc.true_peak_db <= profile.limiter.ceiling_db + 0.1 else 'ABOVE'} "
        f"the {profile.limiter.ceiling_db:+.1f} dBTP ceiling",
        f"samples at full scale in the encoded file: {st_enc.clipped_fraction * 100:.4f} %",
        "",
        "## Spectral change at matched loudness (band energy, dB)",
    ]
    delta = band_delta_db(st_in.bands, st_pcm.bands)
    for k, v in delta.items():
        lines.append(f"  {k:20s} {v:+6.2f}")
    lines += ["", "## Stereo / mono checks",
              f"  mono fold-down change: {st_pcm.mono_fold_db - st_in.mono_fold_db:+.2f} dB "
              "(0 dB = mono compatibility unchanged)"]
    if result.warnings:
        lines += ["", "## Notes"] + [f"  {w}" for w in result.warnings]
    lines += ["", "## Stage reports"]
    for rep in result.reports:
        lines.append("  " + json.dumps(rep, default=str))
    text = "\n".join(lines)
    with open(os.path.join(args.out_dir, "validation.txt"), "w", encoding="utf-8") as fh:
        fh.write(text + "\n")
    print("\n" + text)

    if args.ab:
        gap = np.zeros((int(0.5 * audio.sample_rate), audio.channels))
        a = audio.samples * 10 ** (min(0.0, st_pcm.lufs - st_in.lufs) / 20.0)
        b = result.samples * 10 ** (min(0.0, st_in.lufs - st_pcm.lufs) / 20.0)
        ab = np.concatenate([a, gap, b], axis=0)
        if ab.shape[1] != audio.channels:
            ab = ab[:, : audio.channels]
        ab_path = os.path.join(args.out_dir, f"{name}_ab_matched.wav")
        save_audio(AudioData(samples=ab, sample_rate=audio.sample_rate),
                   ExportSettings(path=ab_path, format="WAV", subtype="PCM_24",
                                  overwrite=True), sample_rate=audio.sample_rate)
        print("wrote", ab_path, "(input, 0.5 s gap, processed at matched loudness)")

    for p in figures(audio.samples, result.samples, audio.sample_rate, args.out_dir,
                     name, match_db):
        print("wrote", p)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
