"""Test double for PySide6.QtGui.

Part of the headless GUI simulation (tests/test_gui_sim.py).  It exists so the
real widget/worker code can be executed on a machine with no Qt platform
libraries; it is never imported by the application itself.
"""
from .QtCore import QObject, Signal, _Const, _Namespace, QPointF, QRectF


class _Thing:
    def __init__(self, *a, **k):
        pass

    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        return lambda *a, **k: None


class QColor(_Thing):
    def __init__(self, *a, **k):
        self._a = a

    def lighter(self, f=100):
        return self

    def darker(self, f=100):
        return self

    def name(self):
        return "#000000"


class QPen(_Thing):
    def __init__(self, *a, **k):
        pass


class QBrush(_Thing):
    pass


class QFont(_Thing):
    pass


class QPainterPath(_Thing):
    pass


class QPolygonF(_Thing):
    pass


class QLinearGradient(_Thing):
    pass


class QPalette(_Thing):
    pass


class QPixmap(_Thing):
    def __init__(self, *a, **k):
        pass

    def fill(self, *a):
        pass

    def isNull(self):
        return False


class QIcon(_Thing):
    def __init__(self, *a, **k):
        pass


class QPainter(_Thing):
    RenderHint = _Namespace()
    Antialiasing = _Const(1)

    def __init__(self, *a, **k):
        pass

    def setRenderHint(self, *a, **k):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *e):
        return False


class QAction(QObject):
    triggered = Signal(bool)
    toggled = Signal(bool)

    def __init__(self, *a, **k):
        QObject.__init__(self)
        self._shortcut = None
        self._checkable = False

    def setShortcut(self, key):
        self._shortcut = key

    def setCheckable(self, on):
        self._checkable = bool(on)

    def setChecked(self, on):
        pass


class QKeySequence(_Thing):
    Open = 1
    Quit = 2
    Save = 3


class QGuiApplication(_Thing):
    @staticmethod
    def setOverrideCursor(cursor):
        pass

    @staticmethod
    def restoreOverrideCursor():
        pass
