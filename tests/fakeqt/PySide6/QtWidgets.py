"""Test double for PySide6.QtWidgets.

Part of the headless GUI simulation (tests/test_gui_sim.py).  It exists so the
real widget/worker code can be executed on a machine with no Qt platform
libraries; it is never imported by the application itself.
"""
from .QtCore import QObject, QRectF, Signal, QTimer, _Const, _Namespace
from .QtGui import QAction, QIcon


class _Widget(QObject):
    def __init__(self, *a, **k):
        parent = a[0] if a and isinstance(a[0], QObject) else k.get("parent")
        QObject.__init__(self, parent)
        self._object_name = ""
        self._text = ""
        self._enabled = True
        self._visible = True
        self._tooltip = ""
        self._width = self._height = 0
        if a and isinstance(a[0], str):
            self._text = a[0]

    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(name)
        return lambda *a, **k: None

    # -- common members used by the app -------------------------------
    def setObjectName(self, name):
        self._object_name = name

    def objectName(self):
        return self._object_name

    def setToolTip(self, t):
        self._tooltip = t

    def toolTip(self):
        return self._tooltip

    def setEnabled(self, on):
        self._enabled = bool(on)

    def isEnabled(self):
        return self._enabled

    def setVisible(self, on):
        self._visible = bool(on)

    def setHidden(self, on):
        self._visible = not on

    def isVisible(self):
        return self._visible

    def show(self):
        self._visible = True

    def hide(self):
        self._visible = False

    def setFixedWidth(self, w):
        self._width = int(w)

    def setFixedHeight(self, h):
        self._height = int(h)

    def width(self):
        return self._width or 100

    def height(self):
        return self._height or 30

    def resize(self, w, h=None):
        if h is None:
            w, h = w.width(), w.height()
        self._width, self._height = int(w), int(h)

    def update(self, *args):
        # Qt accepts update() and update(x, y, w, h) for a partial repaint; the
        # app uses both, so the double has to accept both.
        pass

    def rect(self):
        return QRectF(0, 0, self.width(), self.height())

    def setLayout(self, lay):
        self._layout = lay

    def deleteLater(self):
        pass

    # Qt's own event handlers, so that super().<handler>() works
    def paintEvent(self, e):
        pass

    def resizeEvent(self, e):
        pass

    def mousePressEvent(self, e):
        pass

    def mouseMoveEvent(self, e):
        pass

    def mouseReleaseEvent(self, e):
        pass

    def mouseDoubleClickEvent(self, e):
        pass

    def wheelEvent(self, e):
        pass

    def keyPressEvent(self, e):
        pass

    def closeEvent(self, e):
        pass

    def dragEnterEvent(self, e):
        pass

    def dragMoveEvent(self, e):
        pass

    def dropEvent(self, e):
        pass

    def showEvent(self, e):
        pass

    def enterEvent(self, e):
        pass

    def leaveEvent(self, e):
        pass

    def contextMenuEvent(self, e):
        pass

    def sizeHint(self):
        return _Size(100, 30)


class _Size:
    def __init__(self, w, h):
        self._w, self._h = w, h

    def width(self):
        return self._w

    def height(self):
        return self._h

    def expandedTo(self, other):
        return self


class QLabel(_Widget):
    def setText(self, text):
        self._text = str(text)

    def text(self):
        return self._text


class QPushButton(_Widget):
    clicked = Signal(bool)
    toggled = Signal(bool)

    def __init__(self, *a, **k):
        _Widget.__init__(self, *a, **k)
        if a and isinstance(a[0], str):
            self._text = a[0]
        self._checkable = False
        self._checked = False

    def setText(self, t):
        self._text = str(t)

    def text(self):
        return self._text

    def setCheckable(self, on):
        self._checkable = bool(on)

    def isCheckable(self):
        return self._checkable

    def setChecked(self, on):
        on = bool(on)
        if on != self._checked:
            self._checked = on
            if self._checkable:
                self.toggled.emit(on)

    def isChecked(self):
        return self._checked

    def click(self):
        self.clicked.emit(False)
        if self._checkable:
            self.setChecked(not self._checked)


class QToolButton(QPushButton):
    pass


class QCheckBox(_Widget):
    toggled = Signal(bool)
    stateChanged = Signal(int)

    def __init__(self, *a, **k):
        _Widget.__init__(self, *a, **k)
        self._checked = False

    def setChecked(self, on):
        on = bool(on)
        if on != self._checked:
            self._checked = on
            self.toggled.emit(on)

    def isChecked(self):
        return self._checked

    def setText(self, t):
        self._text = str(t)

    def text(self):
        return self._text


class QLineEdit(_Widget):
    textChanged = Signal(str)
    editingFinished = Signal()

    def __init__(self, *a, **k):
        _Widget.__init__(self, *a, **k)
        self._text = ""

    def setText(self, t):
        self._text = str(t)

    def text(self):
        return self._text

    def setPlaceholderText(self, t):
        pass


class QComboBox(_Widget):
    currentIndexChanged = Signal(int)
    activated = Signal(int)

    def __init__(self, *a, **k):
        _Widget.__init__(self, *a, **k)
        self._items = []
        self._index = -1

    def addItem(self, text, data=None):
        self._items.append((text, data))
        if self._index < 0:
            self._index = 0

    def count(self):
        return len(self._items)

    def currentData(self):
        if 0 <= self._index < len(self._items):
            return self._items[self._index][1]
        return None

    def currentText(self):
        if 0 <= self._index < len(self._items):
            return self._items[self._index][0]
        return ""

    def findData(self, data):
        for i, (_t, d) in enumerate(self._items):
            if d == data:
                return i
        return -1

    def findText(self, text):
        for i, (t, _d) in enumerate(self._items):
            if t == text:
                return i
        return -1

    def setCurrentIndex(self, i):
        i = int(i)
        if i != self._index and 0 <= i < len(self._items):
            self._index = i
            self.currentIndexChanged.emit(i)

    def currentIndex(self):
        return self._index


class QSlider(_Widget):
    valueChanged = Signal(int)

    def __init__(self, *a, **k):
        _Widget.__init__(self, *a, **k)
        self._v = 0
        self._min = 0
        self._max = 100

    def setRange(self, lo, hi):
        self._min, self._max = int(lo), int(hi)

    def setMinimum(self, lo):
        self._min = int(lo)

    def setMaximum(self, hi):
        self._max = int(hi)

    def setSingleStep(self, s):
        pass

    def setPageStep(self, s):
        pass

    def setValue(self, v):
        v = max(self._min, min(self._max, int(v)))
        if v != self._v:
            self._v = v
            self.valueChanged.emit(v)

    def value(self):
        return self._v


class QDoubleSpinBox(_Widget):
    valueChanged = Signal(float)

    def __init__(self, *a, **k):
        _Widget.__init__(self, *a, **k)
        self._v = 0.0
        self._lo, self._hi = 0.0, 99.0

    def setRange(self, lo, hi):
        self._lo, self._hi = float(lo), float(hi)

    def setDecimals(self, d):
        pass

    def setSingleStep(self, s):
        pass

    def setSuffix(self, s):
        pass

    def setValue(self, v):
        v = max(self._lo, min(self._hi, float(v)))
        if abs(v - self._v) > 1e-12:
            self._v = v
            self.valueChanged.emit(v)

    def value(self):
        return self._v

    def setKeyboardTracking(self, on):
        pass

    def setAlignment(self, a):
        pass


class QProgressBar(_Widget):
    def setRange(self, lo, hi):
        pass

    def setValue(self, v):
        pass


class QFrame(_Widget):
    HLine = 4
    VLine = 5

    def setFrameShape(self, shape):
        pass


class QScrollArea(_Widget):
    def setWidgetResizable(self, on):
        pass

    def setWidget(self, w):
        self._widget = w

    def setHorizontalScrollBarPolicy(self, p):
        pass

    def setVerticalScrollBarPolicy(self, p):
        pass


class QStackedWidget(_Widget):
    def __init__(self, *a, **k):
        _Widget.__init__(self, *a, **k)
        self._pages = []
        self._index = 0

    def addWidget(self, w):
        self._pages.append(w)
        return len(self._pages) - 1

    def setCurrentIndex(self, i):
        self._index = int(i)

    def currentIndex(self):
        return self._index

    def currentWidget(self):
        return self._pages[self._index] if self._pages else None


class _Layout(QObject):
    def __init__(self, parent=None):
        QObject.__init__(self)
        self.items = []

    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(name)
        return lambda *a, **k: None

    def addWidget(self, w, stretch=0, *a, **k):
        self.items.append(w)

    def addLayout(self, lay, stretch=0):
        self.items.append(lay)

    def addStretch(self, n=0):
        pass

    def setContentsMargins(self, *a):
        pass

    def setSpacing(self, s):
        pass

    def setAlignment(self, *a):
        pass

    def count(self):
        return len(self.items)

    def itemAt(self, i):
        return self.items[i] if 0 <= i < len(self.items) else None


class QHBoxLayout(_Layout):
    pass


class QVBoxLayout(_Layout):
    pass


class QGridLayout(_Layout):
    def addWidget(self, w, r=0, c=0, *a, **k):
        self.items.append(w)

    def setHorizontalSpacing(self, s):
        pass

    def setVerticalSpacing(self, s):
        pass


class QFormLayout(_Layout):
    def addRow(self, *a, **k):
        pass


class QSplitter(_Widget):
    def __init__(self, *a, **k):
        _Widget.__init__(self, *a, **k)
        self._widgets = []

    def addWidget(self, w):
        self._widgets.append(w)

    def setStretchFactor(self, i, f):
        pass

    def setSizes(self, sizes):
        pass

    def setChildrenCollapsible(self, on):
        pass


class QSizePolicy:
    Expanding = _Const(1)
    Preferred = _Const(2)
    Fixed = _Const(3)
    Maximum = _Const(4)
    Minimum = _Const(5)


class QMenu(_Widget):
    def addAction(self, action):
        pass

    def addSeparator(self):
        pass

    def addMenu(self, title):
        return QMenu()


class QMenuBar(_Widget):
    def addMenu(self, title):
        return QMenu()


class QStatusBar(_Widget):
    def showMessage(self, msg, timeout=0):
        self._message = msg

    def addPermanentWidget(self, w, stretch=0):
        pass

    def setSizeGripEnabled(self, on):
        pass


class QWidget(_Widget):
    pass


class QMainWindow(QWidget):
    def __init__(self):
        QWidget.__init__(self)
        self._menubar = QMenuBar()
        self._statusbar = QStatusBar()
        self._central = None

    def menuBar(self):
        return self._menubar

    def statusBar(self):
        return self._statusbar

    def setCentralWidget(self, w):
        self._central = w

    def setWindowTitle(self, t):
        self._title = t

    def setWindowIcon(self, i):
        pass

    def setAcceptDrops(self, on):
        pass

    def close(self):
        self.closeEvent(None)


class QDialog(QWidget):
    Accepted = 1
    Rejected = 0

    def exec(self):
        return self.Rejected

    def accept(self):
        pass

    def reject(self):
        pass


class QDialogButtonBox(_Widget):
    Ok = 1
    Cancel = 2
    Save = 3

    def __init__(self, *a, **k):
        _Widget.__init__(self, *a, **k)
        self.accepted = Signal()
        self.rejected = Signal()


class QMessageBox(QDialog):
    Open = 16
    Close = 33
    Information = 1
    Warning = 2
    Critical = 3
    Yes = 4
    No = 5

    StandardButton = _Namespace()

    def __init__(self, *a, **k):
        QDialog.__init__(self)
        self._detailed = ""
        self._result = QMessageBox.Close

    def setIcon(self, i):
        pass

    def setText(self, t):
        pass

    def setInformativeText(self, t):
        pass

    def setDetailedText(self, t):
        self._detailed = t

    def detailedText(self):
        return self._detailed

    def setStandardButtons(self, *a):
        pass

    def button(self, which):
        return QPushButton("ok")

    def exec(self):
        return self._result

    def setWindowTitle(self, t):
        pass

    @staticmethod
    def information(*a, **k):
        return QMessageBox.Information

    @staticmethod
    def warning(*a, **k):
        return QMessageBox.Warning

    @staticmethod
    def critical(*a, **k):
        return QMessageBox.Critical

    @staticmethod
    def question(*a, **k):
        return QMessageBox.Yes


class QPlainTextEdit(_Widget):
    def setPlainText(self, t):
        self._text = t

    def toPlainText(self):
        return getattr(self, "_text", "")

    def appendPlainText(self, t):
        pass


class QFileDialog(QDialog):
    @staticmethod
    def getOpenFileName(*a, **k):
        return ("", "")

    @staticmethod
    def getSaveFileName(*a, **k):
        return ("", "")

    @staticmethod
    def getExistingDirectory(*a, **k):
        return ""


class QButtonGroup(QObject):
    def __init__(self, parent=None):
        QObject.__init__(self, parent)
        self._buttons = []

    def addButton(self, b):
        self._buttons.append(b)

    def setExclusive(self, on):
        pass

    def checkedButton(self):
        for b in self._buttons:
            if getattr(b, "_checked", False):
                return b
        return None


class QApplication(QObject):
    def __init__(self, *a, **k):
        QObject.__init__(self)
        QApplication._instance = self

    _instance = None

    @staticmethod
    def instance():
        return QApplication._instance

    @staticmethod
    def processEvents(*a, **k):
        pass

    def exec(self):
        return 0

    @staticmethod
    def setHighDpiScaleFactorRoundingPolicy(policy):
        pass

    def setApplicationName(self, n):
        pass

    def setApplicationDisplayName(self, n):
        pass

    def setApplicationVersion(self, v):
        pass

    def setOrganizationName(self, n):
        pass

    def setWindowIcon(self, i):
        pass

    def setStyleSheet(self, s):
        pass
