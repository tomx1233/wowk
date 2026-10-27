"""Cross-module wiring check: the attributes each GUI object uses really exist.

``main_window`` talks to ``ProcessingPanel``, ``AudioPlayer``, the worker classes
and the ``RenderPayload`` dataclass by attribute name.  A rename on one side and
not the other is a guaranteed ``AttributeError`` at runtime - and it is invisible
to both the compiler and to a Qt API check.

This script parses the modules (it never imports the GUI, so it also works where
Qt cannot start) and verifies every ``self.panel.x`` / ``self.player.x`` /
``payload.x`` / ``worker.x`` reference against the members actually defined on the
target class.

Run:  python tools/check_wiring.py
"""

from __future__ import annotations

import ast
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GUI = os.path.join(ROOT, "timbreforge", "gui")


def parse(name: str) -> ast.Module:
    with open(os.path.join(GUI, name), "r", encoding="utf-8") as fh:
        return ast.parse(fh.read(), filename=name)


def class_members(tree: ast.Module, cls: str) -> set[str]:
    """Every attribute a class defines: assignments, defs and annotations."""
    out: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == cls:
            for item in node.body:
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    out.add(item.name)
                elif isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name):
                    out.add(item.target.id)
                elif isinstance(item, ast.Assign):
                    for tgt in item.targets:
                        if isinstance(tgt, ast.Name):
                            out.add(tgt.id)
            for sub in ast.walk(node):
                if (isinstance(sub, ast.Attribute) and isinstance(sub.value, ast.Name)
                        and sub.value.id == "self" and isinstance(sub.ctx, ast.Store)):
                    out.add(sub.attr)
            # properties are methods with @property; decorators are covered above
            out |= {m for m in out}
    return out


def protobuf_members(tree: ast.Module, cls: str) -> set[str]:
    """Fields of a dataclass from its annotations."""
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == cls:
            return {i.target.id for i in node.body
                    if isinstance(i, ast.AnnAssign) and isinstance(i.target, ast.Name)}
    return set()


def usages(tree: ast.Module, pattern: tuple[str, ...]) -> dict[str, int]:
    """Collect ``a.b.<attr>`` reads for a dotted prefix such as ('self','panel')."""
    found: dict[str, int] = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Attribute):
            continue
        cur, parts = node.value, []
        while isinstance(cur, ast.Attribute):
            parts.append(cur.attr)
            cur = cur.value
        if isinstance(cur, ast.Name):
            parts.append(cur.id)
        parts.reverse()
        if tuple(parts) == pattern:
            found.setdefault(node.attr, node.lineno)
    return found


def main() -> int:
    problems: list[str] = []

    main_tree = parse("main_window.py")
    panel_tree = parse("panel.py")
    player_tree = parse("player.py")
    workers_tree = parse("workers.py")

    panel_members = class_members(panel_tree, "ProcessingPanel")
    # Methods inherited from QWidget/QObject are legitimate calls on the panel;
    # their existence is verified by tools/check_qt_api.py, not here.
    qwidget_members = {
        "setMinimumWidth", "setMaximumWidth", "setMinimumHeight", "setMaximumHeight",
        "setEnabled", "setVisible", "setFocus", "update", "show", "hide", "close",
        "setSizePolicy", "sizeHint", "setLayout", "deleteLater", "blockSignals",
    }
    for attr, line in usages(main_tree, ("self", "panel")).items():
        if attr not in panel_members and attr not in qwidget_members:
            problems.append(f"main_window.py:{line}: ProcessingPanel has no '{attr}'")

    player_members = class_members(player_tree, "AudioPlayer")
    qobject_members = {"connect", "disconnect", "parent", "deleteLater", "blockSignals"}
    for attr, line in usages(main_tree, ("self", "player")).items():
        if attr not in player_members and attr not in qobject_members:
            problems.append(f"main_window.py:{line}: AudioPlayer has no '{attr}'")

    payload_fields = (protobuf_members(workers_tree, "RenderPayload")
                      | class_members(workers_tree, "RenderPayload"))
    for pattern in (("payload",), ("self", "_payload")):
        for attr, line in usages(main_tree, pattern).items():
            if attr not in payload_fields:
                problems.append(f"main_window.py:{line}: RenderPayload has no '{attr}'")

    worker_members = (class_members(workers_tree, "RenderWorker")
                      | class_members(workers_tree, "_CancellableThread")
                      | {"isRunning", "wait", "start", "finished", "deleteLater"})
    for attr, line in usages(main_tree, ("self", "_worker")).items():
        if attr not in worker_members:
            problems.append(f"main_window.py:{line}: RenderWorker has no '{attr}'")
    for attr, line in usages(main_tree, ("self", "_export_worker")).items():
        if attr not in worker_members:
            problems.append(f"main_window.py:{line}: ExportWorker has no '{attr}'")

    # workers -> dsp contracts
    graph_tree = parse_tree(os.path.join(ROOT, "timbreforge", "dsp", "graph.py"))
    result_fields = protobuf_members(graph_tree, "RenderResult")
    for attr, line in usages(workers_tree, ("result",)).items():
        if attr not in result_fields:
            problems.append(f"workers.py:{line}: RenderResult has no '{attr}'")

    device_fields = protobuf_members(parse_tree(
        os.path.join(ROOT, "timbreforge", "dsp", "playback.py")), "DeviceFormat")
    for attr, line in usages(main_tree, ("self", "player", "device_format")).items():
        if attr not in device_fields:
            problems.append(f"main_window.py:{line}: DeviceFormat has no '{attr}'")

    if problems:
        print(f"{len(problems)} wiring problem(s):")
        for p in problems:
            print("  " + p)
        return 1
    print("Wiring check passed: panel/player/worker/payload attributes all exist.")
    return 0


def parse_tree(path: str) -> ast.Module:
    with open(path, "r", encoding="utf-8") as fh:
        return ast.parse(fh.read(), filename=path)


if __name__ == "__main__":
    raise SystemExit(main())
