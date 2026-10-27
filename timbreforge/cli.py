"""Headless command line front end (same engine as the desktop app).

Useful for batch rendering, CI and for reproducing the validation numbers.

    python -m timbreforge.cli render  --input song.ogg --profile profiles/reference_character.json \
        --output song_processed.ogg
    python -m timbreforge.cli measure --input song.ogg [--loudness]
    python -m timbreforge.cli profiles
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

from . import APP_NAME, __version__
from .dsp.audio import AudioData, ExportSettings, AudioError, format_for_extension, load_audio, save_audio
from .dsp.graph import Profile, preset_available, render
from .dsp.meter import band_delta_db, measure


def _load_profile(spec: str | None) -> Profile:
    if not spec:
        return Profile()
    if os.path.isfile(spec):
        return Profile.load(spec)
    presets = preset_available()
    by_key = {k.lower(): v for k, v in presets.items()}
    by_name = {v().name.lower(): v for v in presets.values()}
    key = spec.lower()
    if key in by_key:
        return by_key[key]()
    if key in by_name:
        return by_name[key]()
    raise AudioError(f"Unknown profile '{spec}'. Use a JSON file or one of: "
                     + ", ".join(presets))


def cmd_render(args) -> int:
    audio = load_audio(args.input)
    profile = _load_profile(args.profile)
    errs = profile.validate()
    if errs:
        print("invalid profile:\n  " + "\n  ".join(errs))
        return 2
    result = render(audio.samples, audio.sample_rate, profile,
                    progress=lambda f, l: None)
    out = AudioData(samples=result.samples, sample_rate=audio.sample_rate,
                    title=audio.title, artist=audio.artist)
    fmt, subtype, _ = format_for_extension(args.output)
    facts = save_audio(out, ExportSettings(path=args.output, format=fmt, subtype=subtype,
                                           sample_rate=args.sample_rate,
                                           overwrite=args.force),
                       sample_rate=audio.sample_rate)
    written = load_audio(args.output)
    st_in = measure(audio.samples, audio.sample_rate)
    st_out = measure(written.samples, written.sample_rate)
    print(f"{APP_NAME} {__version__}")
    print(f"  input     {audio.filename}  {audio.sample_rate} Hz  {audio.channels} ch  "
          f"{audio.duration:.4f} s")
    print(f"  profile   {profile.name}  [{', '.join(profile.active_stages()) or 'none'}]")
    print(f"  output    {facts['path']}  {facts['format']}/{facts['subtype']}  "
          f"{written.duration:.4f} s")
    print(f"  frames    input {audio.frames}  pcm {result.frames}  "
          f"encoded {written.frames} (padding/codec only)")
    print(f"  loudness  {st_in.lufs:.2f} -> {st_out.lufs:.2f} LUFS "
          f"({st_out.lufs - st_in.lufs:+.2f}, match applied {result.loudness_match_db:+.2f} dB)")
    print(f"  true peak {st_in.true_peak_db:+.2f} -> {st_out.true_peak_db:+.2f} dBTP "
          f"(ceiling {profile.limiter.ceiling_db:+.1f})")
    print(f"  crest     {st_in.crest_db:.2f} -> {st_out.crest_db:.2f} dB")
    if st_out.true_peak_db > profile.limiter.ceiling_db + 0.1:
        print("  WARNING: encoded true peak is above the ceiling (lossy overshoot)")
    if st_out.lufs > st_in.lufs + 0.3:
        print("  WARNING: output is louder than the input")
    if args.json:
        print(json.dumps({"facts": facts, "input": st_in.to_dict(),
                          "output": st_out.to_dict(),
                          "warnings": result.warnings}, indent=2, default=str))
    return 0


def cmd_measure(args) -> int:
    audio = load_audio(args.input)
    st = measure(audio.samples, audio.sample_rate)
    print(f"{audio.filename}: {audio.sample_rate} Hz, {audio.channels} ch, "
          f"{audio.duration:.6f} s, {audio.format}/{audio.subtype}")
    for line in st.summary_lines():
        print("  " + line)
    if args.bands:
        print("  bands:")
        for k, v in st.bands.items():
            print(f"    {k:20s} {v:8.2f} dB")
    if args.json:
        print(json.dumps(st.to_dict(), indent=2))
    return 0


def cmd_compare(args) -> int:
    a = load_audio(args.a)
    b = load_audio(args.b)
    if a.sample_rate != b.sample_rate:
        print("sample rates differ; resample first")
        return 2
    sa = measure(a.samples, a.sample_rate)
    sb = measure(b.samples, b.sample_rate)
    print(f"A: {a.filename}\nB: {b.filename}\n")
    print(f"{'':16s}{'A':>10s}{'B':>10s}{'delta':>10s}")
    rows = [("true peak dBTP", "true_peak_db"), ("RMS dBFS", "rms_db"),
            ("LUFS", "lufs"), ("crest dB", "crest_db"),
            ("env range dB", "envelope_range_db"), ("centroid Hz", "centroid_hz"),
            ("corr", "correlation"), ("S/M dB", "side_mid_db")]
    for label, key in rows:
        va, vb = getattr(sa, key), getattr(sb, key)
        print(f"{label:16s}{va:10.2f}{vb:10.2f}{vb - va:+10.2f}")
    print("\nband delta (B relative to A):")
    for k, v in band_delta_db(sa.bands, sb.bands).items():
        print(f"  {k:20s} {v:+7.2f} dB")
    return 0


def cmd_profiles(args) -> int:
    for key, fn in preset_available().items():
        p = fn()
        print(f"{p.name}\n  {p.description}\n  stages: "
              f"{', '.join(p.active_stages()) or 'none'}\n")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="timbreforge",
                                 description=f"{APP_NAME} - headless renderer")
    ap.add_argument("--version", action="version", version=f"{APP_NAME} {__version__}")
    sub = ap.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("render", help="render a file through a profile")
    r.add_argument("--input", required=True)
    r.add_argument("--output", required=True)
    r.add_argument("--profile", default=None)
    r.add_argument("--sample-rate", type=int, default=None)
    r.add_argument("--force", action="store_true", help="overwrite the destination")
    r.add_argument("--json", action="store_true")
    r.set_defaults(func=cmd_render)

    m = sub.add_parser("measure", help="measure a file")
    m.add_argument("--input", required=True)
    m.add_argument("--bands", action="store_true")
    m.add_argument("--json", action="store_true")
    m.set_defaults(func=cmd_measure)

    c = sub.add_parser("compare", help="compare two files monotonically measured")
    c.add_argument("--a", required=True)
    c.add_argument("--b", required=True)
    c.set_defaults(func=cmd_compare)

    p = sub.add_parser("profiles", help="list built-in profiles")
    p.set_defaults(func=cmd_profiles)

    args = ap.parse_args(argv)
    try:
        return args.func(args)
    except AudioError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
