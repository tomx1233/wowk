"""Test double for sounddevice.

Part of the headless GUI simulation (tests/test_gui_sim.py).  It exists so the
real widget/worker code can be executed on a machine with no Qt platform
libraries; it is never imported by the application itself.
"""
"""Fake PortAudio backend: opens, pushes one block through the callback, closes."""
import numpy as np

default = type("D", (), {"device": (0, 0)})()

DTOKEN = {"float32": np.float32, "int16": np.int16, "int32": np.int32}


def query_devices(kind=None, device=None):
    return {"name": "Fake Output", "default_samplerate": 48000.0,
            "max_output_channels": 2, "max_input_channels": 0}


def check_output_settings(*a, **k):
    return None


class OutputStream:
    def __init__(self, samplerate, channels, dtype, blocksize=1024, callback=None,
                 device=None, **k):
        self.samplerate = samplerate
        self.channels = channels
        self.dtype = dtype
        self.blocksize = blocksize
        self.callback = callback
        self.started = False
        self.blocks = 0

    def start(self):
        self.started = True
        for _ in range(3):
            out = np.zeros((self.blocksize, self.channels), dtype=DTOKEN[self.dtype])
            self.callback(out, self.blocksize, None, None)
            self.blocks += 1

    def stop(self):
        self.started = False

    def close(self):
        pass
