"""Fast waveform envelopes for drawing.

Drawing a waveform needs one number per screen pixel, but the audio may hold
tens of millions of samples.  Scanning the whole array on every repaint is what
made the window stall while the playhead moved, so the scan is done once, on a
coarse block grid, and every later repaint only reads that grid.

The grid is a *peak* grid (the largest absolute sample per block), not an average,
so the drawing keeps transients visible when the whole file is on screen.  When
the view is zoomed in far enough that one block would cover several pixels the
code falls back to scanning the visible slice directly, which is cheap because
the slice is short.

This module has no Qt dependency on purpose: it is unit-tested directly.
"""

from __future__ import annotations

import numpy as np

BLOCK = 256


def mono_peaks(x: np.ndarray, block: int = BLOCK) -> np.ndarray:
    """Largest absolute sample per ``block`` samples, in bounded memory.

    ``min``/``max`` are reduced together, so peak transients survive the
    decimation and no full-length temporary of ``|x|`` is ever allocated.  The
    array is walked in chunks of about two million samples, which keeps the
    memory cost flat regardless of file length (a 10-minute stereo file needs
    ~15 MB here instead of the ~350 MB a naive ``np.abs(x).max(axis=1)`` costs).
    """
    n = int(x.shape[0])
    block = max(1, int(block))
    if n == 0:
        return np.zeros(0, dtype=np.float32)
    nb = n // block
    out = np.zeros(max(1, nb), dtype=np.float32)
    step = max(1, (1 << 21) // block)
    for b0 in range(0, nb, step):
        b1 = min(nb, b0 + step)
        seg = x[b0 * block:b1 * block]
        if seg.ndim > 1:
            r = seg.reshape(b1 - b0, block, seg.shape[1])
            out[b0:b1] = np.maximum(r.max(axis=(1, 2)), -r.min(axis=(1, 2)))
        else:
            r = seg.reshape(b1 - b0, block)
            out[b0:b1] = np.maximum(r.max(axis=1), -r.min(axis=1))
    if nb * block < n:
        tail = x[nb * block:]
        last = float(np.abs(tail).max()) if tail.size else 0.0
        out[-1] = max(float(out[-1]), last)
    return out


class PeakCache:
    """Block peaks for one array plus a small cache of recent envelopes."""

    def __init__(self, block: int = BLOCK):
        self.block = int(block)
        self.sr = 44100
        self.frames = 0
        self._peaks: np.ndarray | None = None
        self._src: np.ndarray | None = None
        self._cache: dict[tuple, np.ndarray] = {}

    # -- data ---------------------------------------------------------
    def set(self, x: np.ndarray | None, sr: int):
        if x is None or len(x) == 0:
            self._src = None
            self._peaks = None
            self.frames = 0
            self._cache.clear()
            return
        if x is self._src:
            self.sr = int(sr) if sr else self.sr
            return
        self._src = x
        self.sr = int(sr) if sr else self.sr
        self.frames = int(x.shape[0])
        self._peaks = mono_peaks(x, self.block)
        self._cache.clear()

    # -- query --------------------------------------------------------
    def envelope(self, t0: float, t1: float, width: int) -> np.ndarray:
        """One peak value per pixel for the time range ``[t0, t1]``."""
        width = max(2, int(width))
        if self._src is None or not self.frames:
            return np.zeros(width, dtype=np.float32)
        sr = max(1, self.sr)
        i0 = int(np.clip(t0 * sr, 0, self.frames - 1))
        i1 = int(np.clip(t1 * sr, i0 + 1, self.frames))
        key = (i0, i1, width)
        hit = self._cache.get(key)
        if hit is not None:
            return hit

        span = i1 - i0
        # Zoomed in far enough that a block would span more than a pixel: scan
        # the (short) visible slice for exact peaks.  Otherwise the block grid
        # already has at least one entry per pixel.
        if span <= width or span < width * self.block:
            out = self._direct(i0, i1, width)
        else:
            out = self._from_peaks(i0, i1, width)
        if len(self._cache) > 8:
            self._cache.clear()
        self._cache[key] = out
        return out

    # -- internals ----------------------------------------------------
    def _direct(self, i0: int, i1: int, width: int) -> np.ndarray:
        """Exact per-pixel peaks over the visible slice (used when zoomed in)."""
        seg = self._src[i0:i1]
        flat = seg.reshape(-1) if seg.flags.c_contiguous else np.ravel(seg)
        n = flat.size
        if n <= 0:
            return np.zeros(width, dtype=np.float32)
        if n <= width:
            pk = np.abs(flat)
            return np.interp(np.linspace(0, pk.size - 1, width),
                             np.arange(pk.size), pk).astype(np.float32)
        edges = np.linspace(0, n, width + 1).astype(np.int64)
        edges = np.clip(edges, 0, n - 1)
        # reduceat on the flat (channel-interleaved) view: two cheap O(n) passes,
        # no strided axis-1 reduction and no full |x| temporary.
        hi = np.maximum.reduceat(flat, edges[:-1])
        lo = -np.minimum.reduceat(flat, edges[:-1])
        return np.maximum(hi, lo).astype(np.float32)

    def _from_peaks(self, i0: int, i1: int, width: int) -> np.ndarray:
        peaks = self._peaks
        b0 = i0 // self.block
        b1 = min(peaks.size, (i1 + self.block - 1) // self.block)
        if b1 <= b0:
            return self._direct(i0, i1, width)
        seg = peaks[b0:b1]
        if seg.size <= width:
            return np.interp(np.linspace(0, seg.size - 1, width),
                             np.arange(seg.size), seg).astype(np.float32)
        edges = np.linspace(0, seg.size, width + 1).astype(np.int64)
        edges = np.clip(edges, 0, seg.size - 1)
        return np.maximum.reduceat(seg, edges[:-1]).astype(np.float32)
