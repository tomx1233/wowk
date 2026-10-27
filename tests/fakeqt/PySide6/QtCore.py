"""Test double for PySide6.QtCore.

Part of the headless GUI simulation (tests/test_gui_sim.py).  It exists so the
real widget/worker code can be executed on a machine with no Qt platform
libraries; it is never imported by the application itself.
"""
import itertools

_counter = itertools.count(1)


class _Const(int):
    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        return _Const(next(_counter))


class _Namespace:
    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        return _Const(next(_counter))


Qt = _Namespace()


class Signal:
    def __init__(self, *types):
        self._types = types

    def __set_name__(self, owner, name):
        self._name = f"_sig_{name}"

    def __get__(self, obj, cls=None):
        if obj is None:
            return self
        store = obj.__dict__.setdefault(self._name, [])
        class _Bound:
            def __init__(self, slots):
                self._slots = slots

            def connect(self, slot):
                self._slots.append(slot)

            def disconnect(self, slot=None):
                if slot is None:
                    self._slots.clear()
                elif slot in self._slots:
                    self._slots.remove(slot)

            def emit(self, *args):
                for slot in list(self._slots):
                    slot(*args)
        return _Bound(store)


class QObject:
    def __init__(self, parent=None, *a, **k):
        self._parent = parent
        self._children = []

    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(name)
        return lambda *a, **k: None

    def deleteLater(self):
        pass

    def blockSignals(self, flag):
        return False


class QSignalBlocker:
    def __init__(self, obj):
        self.obj = obj

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class QTimer(QObject):
    timeout = Signal()

    def __init__(self, parent=None):
        QObject.__init__(self, parent)
        self._interval = 0
        self._single = False
        self._active = False

    def setInterval(self, ms):
        self._interval = int(ms)

    def interval(self):
        return self._interval

    def setSingleShot(self, on):
        self._single = bool(on)

    def start(self, ms=None):
        if ms is not None:
            self._interval = int(ms)
        self._active = True

    def stop(self):
        self._active = False

    def isActive(self):
        return self._active

    @staticmethod
    def singleShot(ms, fn):
        fn()


class QRectF:
    def __init__(self, *a):
        a = list(a) + [0.0] * (4 - len(a))
        self._x, self._y, self._w, self._h = (float(v) for v in a[:4])

    def left(self): return self._x
    def top(self): return self._y
    def right(self): return self._x + self._w
    def bottom(self): return self._y + self._h
    def width(self): return self._w
    def height(self): return self._h
    def center(self): return QPointF(self._x + self._w / 2, self._y + self._h / 2)
    def adjusted(self, *a): return QRectF(self._x, self._y, self._w, self._h)
    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(name)
        return lambda *a, **k: None


class QPointF:
    def __init__(self, x=0.0, y=0.0):
        self._x, self._y = float(x), float(y)

    def x(self): return self._x
    def y(self): return self._y

    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(name)
        return lambda *a, **k: None


class QIODevice(QObject):
    class OpenModeFlag:
        ReadOnly = 1
        WriteOnly = 2

    def open(self, mode):
        return True

    def close(self):
        pass

    def readData(self, maxlen):
        return b""

    def writeData(self, data):
        return 0


class QThread(QObject):
    started = Signal()
    finished = Signal()

    def __init__(self, parent=None):
        QObject.__init__(self, parent)
        self._running = False

    def start(self, *a):
        self._running = True
        self.started.emit()
        try:
            self.run()
        finally:
            self._running = False
            self.finished.emit()

    def run(self):
        pass

    def isRunning(self):
        return self._running

    def wait(self, ms=None):
        return True

    def msleep(self, ms):
        pass
