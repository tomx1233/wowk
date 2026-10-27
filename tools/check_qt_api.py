"""Static check that every Qt member the GUI calls actually exists.

The sandbox this project is developed in has no display stack (no libGL, no
fonts, no ICU), so ``QApplication`` cannot be created and the GUI cannot be
executed.  This script is the substitute: it parses the PySide6 ``.pyi`` stubs
that ship with the installed wheel and verifies that

* every Qt class we import exists in the stub for its module,
* every method/property we call on an object we created exists on that class
  (walking the stubs' base classes, including bases in other Qt modules),
* every enum member we read (``Qt.Key_Space``, ``QMessageBox.Open`` …) exists,
  including PySide's flattened enum aliases.

It is deliberately conservative: when it cannot infer the type of an object it
skips the call instead of guessing, so a clean run means "no known-bad call",
not "the UI definitely renders".  It has zero false positives on purpose, which
is what makes it usable as a pre-commit gate.

Run:  python tools/check_qt_api.py
"""

from __future__ import annotations

import ast
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

try:
    import PySide6

    QT_DIR = os.path.dirname(PySide6.__file__)
except Exception:  # noqa: BLE001
    QT_DIR = None


class ClassInfo:
    __slots__ = ("module", "bases", "own", "nested", "nested_members")

    def __init__(self, module: str):
        self.module = module
        self.bases: list[tuple[str, str | None]] = []
        self.own: set[str] = set()
        self.nested: set[str] = set()
        self.nested_members: set[str] = set()


class Registry:
    """Class table built from the PySide6 ``.pyi`` stubs, loaded on demand."""

    def __init__(self):
        #: module -> class name -> info.  Lookup must be module-aware: several Qt
        #: modules define unrelated classes with the same name (QAction lives in
        #: QtGui *and* Qt3DInput).
        self.by_module: dict[str, dict[str, ClassInfo]] = {}
        self.by_name: dict[str, ClassInfo] = {}
        self.loaded_modules: set[str] = set()
        self.missing_modules: set[str] = set()

    @property
    def classes(self) -> dict[str, ClassInfo]:
        return self.by_name

    def _register(self, name: str, info: ClassInfo):
        self.by_module.setdefault(info.module, {}).setdefault(name, info)
        self.by_name.setdefault(name, info)

    def load(self, module: str) -> bool:
        if module in self.loaded_modules:
            return True
        if module in self.missing_modules or not QT_DIR:
            return False
        path = os.path.join(QT_DIR, f"{module}.pyi")
        if not os.path.isfile(path):
            self.missing_modules.add(module)
            return False
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            tree = ast.parse(fh.read(), filename=path)
        self.loaded_modules.add(module)

        def collect(node: ast.ClassDef, info: ClassInfo):
            for item in node.body:
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    info.own.add(item.name)
                elif isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name):
                    info.own.add(item.target.id)
                elif isinstance(item, ast.Assign):
                    for tgt in item.targets:
                        if isinstance(tgt, ast.Name):
                            info.own.add(tgt.id)
                elif isinstance(item, ast.ClassDef):
                    sub = ClassInfo(module)
                    collect(item, sub)
                    info.nested.add(item.name)
                    info.nested_members |= sub.own | sub.nested | sub.nested_members
                    self._register(item.name, sub)

        for node in tree.body:
            if isinstance(node, ast.ClassDef):
                info = ClassInfo(module)
                collect(node, info)
                for base in node.bases:
                    # stubs qualify every base: ``PySide6.QtWidgets.QFrame``
                    if isinstance(base, ast.Name):
                        info.bases.append((base.id, module))
                    elif isinstance(base, ast.Attribute):
                        parts = []
                        cur = base
                        while isinstance(cur, ast.Attribute):
                            parts.append(cur.attr)
                            cur = cur.value
                        if isinstance(cur, ast.Name):
                            parts.append(cur.id)
                        parts.reverse()
                        info.bases.append((parts[-1], parts[-2] if len(parts) > 1 else module))
                self._register(node.name, info)
        return True

    def find(self, cls: str, module: str | None = None) -> ClassInfo | None:
        if module:
            self.load(module)
            info = self.by_module.get(module, {}).get(cls)
            if info is not None:
                return info
        info = self.by_name.get(cls)
        if info is not None:
            return info
        # last resort: parse every stub on disk and look again
        if QT_DIR:
            for name in sorted(os.listdir(QT_DIR)):
                if name.endswith(".pyi"):
                    self.load(name[:-4])
            return self.by_name.get(cls)
        return None

    def members(self, cls: str, module: str | None = None, seen=None) -> set[str]:
        seen = seen if seen is not None else set()
        key = f"{module}:{cls}"
        if key in seen:
            return set()
        seen.add(key)
        info = self.find(cls, module)
        if info is None:
            return set()
        out = set(info.own) | set(info.nested) | set(info.nested_members)
        for base, base_mod in info.bases:
            out |= self.members(base, base_mod, seen)
        return out


class Checker:
    def __init__(self):
        self.reg = Registry()
        self.problems: list[str] = []
        self._seen: set[str] = set()

    def report(self, path: str, node: ast.AST, msg: str):
        line = getattr(node, "lineno", "?")
        key = f"{path}:{line}:{msg}"
        if key in self._seen:
            return
        self._seen.add(key)
        self.problems.append(f"{os.path.relpath(path, ROOT)}:{line}: {msg}")

    # -- resolution ----------------------------------------------------
    @staticmethod
    def ctor_name(call: ast.Call, imports, modules) -> str | None:
        fn = call.func
        if isinstance(fn, ast.Name) and fn.id in imports:
            return fn.id
        if isinstance(fn, ast.Attribute) and isinstance(fn.value, ast.Name):
            if fn.value.id in modules:
                return fn.attr
        return None

    def check_member(self, path, node, cls, member, module=None) -> None:
        if member.startswith("__") or member in ("connect", "disconnect", "emit"):
            return
        info = self.reg.find(cls, module)
        if info is None:
            self.report(path, node, f"Qt class {cls} not found in the PySide6 stubs")
            return
        if member not in self.reg.members(cls, info.module):
            self.report(path, node, f"{cls}.{member} is not in the PySide6 stubs")

    # -- per file ------------------------------------------------------
    def check_file(self, path: str):
        with open(path, "r", encoding="utf-8") as fh:
            tree = ast.parse(fh.read(), filename=path)

        imports: dict[str, str] = {}          # name -> Qt module
        modules: dict[str, str] = {}          # alias -> Qt module
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith("PySide6"):
                mod = node.module.split(".")[-1]
                self.reg.load(mod)
                for alias in node.names:
                    imports[alias.asname or alias.name] = mod
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.startswith("PySide6"):
                        mod = alias.name.split(".")[-1]
                        self.reg.load(mod)
                        modules[alias.asname or mod] = mod

        for fn in ast.walk(tree):
            if isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
                self.check_scope(path, fn, imports, modules)

    def check_scope(self, path, fn, imports, modules):
        types: dict[str, tuple[str, str | None]] = {}

        def note(target: str, value):
            if isinstance(value, ast.Call):
                cls = self.ctor_name(value, imports, modules)
                types[target] = (cls, imports.get(cls) if cls else None)
            else:
                types.pop(target, None)

        def walk(stmts):
            for st in stmts:
                if isinstance(st, ast.Assign):
                    for tgt in st.targets:
                        if isinstance(tgt, ast.Name):
                            note(tgt.id, st.value)
                        elif (isinstance(tgt, ast.Attribute)
                              and isinstance(tgt.value, ast.Name)
                              and tgt.value.id == "self"):
                            note(f"self.{tgt.attr}", st.value)
                elif isinstance(st, ast.AnnAssign) and isinstance(st.target, ast.Name):
                    ann = st.annotation
                    if isinstance(ann, ast.Name) and ann.id in imports:
                        types[st.target.id] = (ann.id, imports[ann.id])
                for sub in ast.walk(st):
                    if isinstance(sub, ast.Call):
                        self.check_call(path, sub, types, imports, modules)
                if isinstance(st, ast.If):
                    walk(st.body); walk(st.orelse)
                elif isinstance(st, (ast.For, ast.While)):
                    walk(st.body); walk(st.orelse)
                elif isinstance(st, (ast.With, ast.Try)):
                    walk(st.body)
                    for h in getattr(st, "handlers", []):
                        walk(h.body)
                    walk(getattr(st, "orelse", []))
                    walk(getattr(st, "finalbody", []))

        walk(fn.body)

    def check_call(self, path, call, types, imports, modules):
        fn = call.func
        if isinstance(fn, ast.Name):
            if fn.id in imports:
                self.reg.find(fn.id, imports[fn.id])
            return
        if not isinstance(fn, ast.Attribute):
            return
        owner = fn.value
        resolved = None
        if isinstance(owner, ast.Name):
            if owner.id in types:
                resolved = types[owner.id]
            elif owner.id in imports:
                resolved = (owner.id, imports[owner.id])
        elif (isinstance(owner, ast.Attribute) and isinstance(owner.value, ast.Name)
              and owner.value.id == "self"):
            resolved = types.get(f"self.{owner.attr}")
        if resolved and resolved[0]:
            self.check_member(path, call, resolved[0], fn.attr, resolved[1])

    def check_value_chain(self, path, expr, imports):
        """Check enum/member reads such as ``Qt.AlignmentFlag.AlignRight``.

        PySide flattens nested enums onto their owner, so every attribute in the
        chain is validated against the *root* class's member set.
        """
        for node in ast.walk(expr):
            if not isinstance(node, ast.Attribute):
                continue
            chain = []
            cur = node
            while isinstance(cur, ast.Attribute):
                chain.append(cur.attr)
                cur = cur.value
            if not isinstance(cur, ast.Name) or cur.id not in imports:
                continue
            cls, module = cur.id, imports[cur.id]
            self.reg.load(module)
            for attr in reversed(chain):
                self.check_member(path, node, cls, attr, module)


def main() -> int:
    if not QT_DIR:
        print("PySide6 is not installed; nothing to check.")
        return 0
    checker = Checker()
    gui_dir = os.path.join(ROOT, "timbreforge", "gui")
    files = sorted(os.path.join(gui_dir, f) for f in os.listdir(gui_dir)
                   if f.endswith(".py"))
    for path in files:
        checker.check_file(path)
        with open(path, "r", encoding="utf-8") as fh:
            tree = ast.parse(fh.read(), filename=path)
        imports = {}
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith("PySide6"):
                mod = node.module.split(".")[-1]
                for alias in node.names:
                    imports[alias.asname or alias.name] = mod
        for node in ast.walk(tree):
            checker.check_value_chain(path, node, imports) if isinstance(node, ast.Expr) else None
        # enum reads appear as arguments and comparands, not only as statements
        for node in ast.walk(tree):
            if isinstance(node, (ast.Call, ast.Assign, ast.Compare, ast.Return)):
                checker.check_value_chain(path, node, imports)

    if checker.problems:
        print(f"{len(checker.problems)} Qt API problem(s):")
        for p in checker.problems:
            print("  " + p)
        return 1
    print(f"Qt API check passed for {len(files)} GUI modules "
          f"({len(checker.reg.classes)} stubbed classes).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
