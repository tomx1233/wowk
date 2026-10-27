"""Timbreforge - a standalone desktop audio-character processor.

The application imports an audio file, renders it through one documented,
non-destructive processing graph, previews the result at matched loudness and
exports it.  It never substitutes or mixes in a reference recording, never
changes pitch or tempo, and is not a loudness maximiser.
"""

__version__ = "1.0.0"
APP_NAME = "Timbreforge"
APP_SUBTITLE = "Audio character processor"
SCHEMA_VERSION = 1

__all__ = ["__version__", "APP_NAME", "APP_SUBTITLE", "SCHEMA_VERSION"]
