"""Test double for the PySide6 package.

Part of the headless GUI simulation (tests/test_gui_sim.py).  It exists so the
real widget/worker code can be executed on a machine with no Qt platform
libraries; it is never imported by the application itself.
"""
"""Minimal stand-in for PySide6, used only to execute GUI logic in a sandbox
without Qt platform libraries.  Not shipped, not imported by the app."""
__version__ = "0.0-fake"
