# Audacity macros recovered from a pair of files

`ModifiedFile.txt` reproduces `originalfile.ogg` → `ModifiedFile.ogg`:

```
ChangePitch: Percentage=5.18
ChangeTempo: Percentage=5.06
```

Import it with **Tools ▸ Macros… ▸ New**, then **Import…** and pick the file (or
type the two lines in the Macro editor with **Insert**), and apply it to the whole
track. The measurement, its uncertainty and the evidence behind it are in
`../docs/PAIR_MATCH.md`.

To recover the same numbers for a different pair, use **Analysis ▸ Pitch/tempo of
two files…** in the app (Ctrl+T, or the *Pitch/tempo* button in the top bar),
which can also save a macro file for you, or run:

```bash
python3 tools/pair_match.py original.ogg modified.ogg --out-dir analysis/pair
```

The modified file is also about 2.9 dB quieter than the original; add
`Amplify: Amplification=-2.90` if you want that matched as well.
