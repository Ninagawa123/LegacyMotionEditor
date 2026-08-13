"""
File Name: LegacyMotionEditor_CodeEditor.py
Description: Internal Python code editor for LegacyMotionEditor.

Author      : Izumi Ninagawa
License     : MIT License
Copyright (c) 2026 Izumi Ninagawa
"""

import ast
import re

from PySide6 import QtCore, QtGui, QtWidgets


# ---------------------------------------------------------------------------
# Syntax highlighter
# ---------------------------------------------------------------------------
class PythonHighlighter(QtGui.QSyntaxHighlighter):
    """Syntax highlighter for Python code (dark theme)."""

    KEYWORDS = [
        "False", "None", "True", "and", "as", "assert", "async", "await",
        "break", "class", "continue", "def", "del", "elif", "else", "except",
        "finally", "for", "from", "global", "if", "import", "in", "is",
        "lambda", "nonlocal", "not", "or", "pass", "raise", "return", "try",
        "while", "with", "yield",
    ]

    def __init__(self, document):
        super().__init__(document)

        self._rules = []

        def _fmt(color, bold=False, italic=False):
            f = QtGui.QTextCharFormat()
            f.setForeground(QtGui.QColor(color))
            if bold:
                f.setFontWeight(QtGui.QFont.Bold)
            if italic:
                f.setFontItalic(True)
            return f

        kw_fmt = _fmt("#569cd6", bold=True)
        for kw in self.KEYWORDS:
            self._rules.append((re.compile(r"\b" + kw + r"\b"), kw_fmt))

        # Function / class names after def/class
        self._rules.append((re.compile(r"\bdef\s+(\w+)"), _fmt("#dcdcaa")))
        self._rules.append((re.compile(r"\bclass\s+(\w+)"), _fmt("#4ec9b0")))

        # Built-ins
        builtin_fmt = _fmt("#9cdcfe")
        for bi in ["print", "len", "range", "int", "float", "str", "list",
                   "dict", "set", "tuple", "bool", "type", "isinstance",
                   "hasattr", "getattr", "setattr", "enumerate", "zip",
                   "map", "filter", "sorted", "reversed", "sum", "min", "max",
                   "abs", "round", "open", "input", "super", "object"]:
            self._rules.append((re.compile(r"\b" + bi + r"\b"), builtin_fmt))

        # Decorators
        self._rules.append((re.compile(r"@\w+"), _fmt("#c586c0")))

        # Numbers
        self._rules.append((re.compile(r"\b\d+\.?\d*\b"), _fmt("#b5cea8")))

        # Single-line strings (both ' and ")
        self._rules.append((re.compile(r"\"[^\"\\]*(\\.[^\"\\]*)*\""), _fmt("#ce9178")))
        self._rules.append((re.compile(r"'[^'\\]*(\\.[^'\\]*)*'"), _fmt("#ce9178")))

        # Comments (must come after strings so # inside string isn't highlighted)
        self._rules.append((re.compile(r"#[^\n]*"), _fmt("#6a9955", italic=True)))

        # Multi-line string state formats
        self._ml_fmt = _fmt("#ce9178")

        self._tri_double = re.compile(r'"""')
        self._tri_single = re.compile(r"'''")

    def highlightBlock(self, text):
        # Apply single-line rules
        for pattern, fmt in self._rules:
            for m in pattern.finditer(text):
                # For def/class patterns, highlight the captured group (name) too
                start = m.start(1) if m.lastindex else m.start()
                length = len(m.group(1)) if m.lastindex else len(m.group())
                # Also highlight the keyword itself for def/class
                if m.lastindex:
                    self.setFormat(m.start(), m.end() - m.start(), fmt)
                else:
                    self.setFormat(start, length, fmt)

        # Multi-line string handling (triple-double-quote)
        self._handle_multiline(text, self._tri_double, 1, self._ml_fmt)

    def _handle_multiline(self, text, delimiter, state_in, fmt):
        if self.previousBlockState() == state_in:
            start = 0
            add = 0
        else:
            m = delimiter.search(text)
            if not m:
                self.setCurrentBlockState(0)
                return
            start = m.start()
            add = m.end() - m.start()

        while start >= 0:
            m = delimiter.search(text, start + add)
            if m:
                length = m.end() - start
                self.setFormat(start, length, fmt)
                self.setCurrentBlockState(0)
                start = m.end()
                add = 0
            else:
                self.setCurrentBlockState(state_in)
                length = len(text) - start
                self.setFormat(start, length, fmt)
                break


# ---------------------------------------------------------------------------
# Default template inserted when Code editor is opened on empty project code
# ---------------------------------------------------------------------------
CODE_DEFAULT_TEMPLATE = '''\
# ── User Code ─────────────────────────────────────────────────────────────────
# This block is injected into the exported cartridge after MOTION_* definitions.
# LME will also exec() this code for 3D preview and Valkey send.
#
# Controllers (WalkParams, LinkParams, WalkController, …) belong in this file
# if the project uses computed motion. Pose-only projects can omit them.
#
# Available from SYSTEM AREA (cartridge) / LMEMotionRuntime (LME preview):
#   run_default(r, s)        — default motion playback
#   Pad(s)                   — gamepad helper (pad.l1, pad.tri, pad.lx, pad.ly)
#   set_angle(s, name, deg)  — write joint angle to s_meridim
#   get_angle(r, name)       — read joint angle from r_meridim [deg]
#   get_imu(r)               — {ax,ay,az,gx,gy,gz,roll,pitch,yaw}
#   _bhv_commander           — Commander stub (motion state machine)
#   PAD_REGISTER_VALUES      — {Pad_Ly, Pad_Lx, …} current pad state (LME preview)
#   logger                   — standard Python logger
#
# MOTION_BOOT, MOTION_BASE (and any other MOTION_* lists from LME) are
# declared above this block and can be referenced freely here.
#
# Define user_setup() and/or user_logic() to override the auto-generated
# versions.  Omitting either falls back to the default.
#
# ── Computed motion example ────────────────────────────────────────────────────
# Define a class with a compute() method; assign it to _bhv_commander.controller.
# Name the step function anything — it must match the function JumpNode in LME.
#
# class MyController:
#     def __init__(self):
#         self.data = [0.0] * 90   # Meridim90 angle array
#     def compute(self):
#         ...                       # fill self.data with joint angles
#     def stop(self):
#         pass
#     @property
#     def busy(self):
#         return False
#
# def motion_ik_step() -> None:
#     """Called every frame by LME (preview) and by MotionPlayer (cartridge)."""
#     _bhv_commander.controller.compute()
#
# ── Pass-through with default motion control ───────────────────────────────────
#
# def user_logic(r: list, s: list) -> list:
#     return run_default(r, s)
'''


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------
def get_function_names(code: str) -> list[str]:
    """Return top-level function names defined in the given Python source."""
    if not code or not code.strip():
        return []
    try:
        tree = ast.parse(code)
        return [
            node.name
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef) and isinstance(
                getattr(node, "col_offset", 0), int
            ) and node.col_offset == 0
        ]
    except SyntaxError:
        return []


# ---------------------------------------------------------------------------
# Code editor window
# ---------------------------------------------------------------------------
class CodeEditorWindow(QtWidgets.QWidget):
    """Standalone internal Python code editor (dark theme)."""

    code_saved = QtCore.Signal(str)  # emitted when Apply is clicked

    def __init__(self, parent=None):
        super().__init__(parent, QtCore.Qt.Window)
        self.setWindowTitle("Code Editor")
        self.resize(720, 520)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Editor area
        self._editor = QtWidgets.QPlainTextEdit()
        self._editor.setStyleSheet(
            "QPlainTextEdit {"
            "  background-color: #1e1e1e;"
            "  color: #d4d4d4;"
            "  font-family: 'Menlo', 'Monaco', 'Courier New', monospace;"
            "  font-size: 13px;"
            "  selection-background-color: #264f78;"
            "  border: none;"
            "  padding: 8px;"
            "}"
        )
        font = QtGui.QFont("Menlo", 13)
        font.setStyleHint(QtGui.QFont.Monospace)
        self._editor.setFont(font)
        self._editor.setTabStopDistance(
            QtGui.QFontMetricsF(font).horizontalAdvance(" ") * 4
        )
        self._editor.setLineWrapMode(QtWidgets.QPlainTextEdit.NoWrap)

        self._highlighter = PythonHighlighter(self._editor.document())

        layout.addWidget(self._editor, 1)

        # Button bar
        btn_bar = QtWidgets.QWidget()
        btn_bar.setStyleSheet("background-color: #252526;")
        btn_layout = QtWidgets.QHBoxLayout(btn_bar)
        btn_layout.setContentsMargins(8, 6, 8, 6)
        btn_layout.setSpacing(8)

        self._status_label = QtWidgets.QLabel("")
        self._status_label.setStyleSheet("color: #888; font-size: 11px;")
        btn_layout.addWidget(self._status_label, 1)

        apply_btn = QtWidgets.QPushButton("Apply")
        apply_btn.setFixedWidth(80)
        apply_btn.setStyleSheet(
            "QPushButton { background-color: #0e639c; color: white; border: none; "
            "  padding: 4px 12px; border-radius: 3px; }"
            "QPushButton:hover { background-color: #1177bb; }"
            "QPushButton:pressed { background-color: #0a4f7e; }"
        )
        apply_btn.clicked.connect(self._on_apply)
        btn_layout.addWidget(apply_btn)

        close_btn = QtWidgets.QPushButton("Close")
        close_btn.setFixedWidth(70)
        close_btn.setStyleSheet(
            "QPushButton { background-color: #3c3c3c; color: #d4d4d4; border: none; "
            "  padding: 4px 12px; border-radius: 3px; }"
            "QPushButton:hover { background-color: #505050; }"
        )
        close_btn.clicked.connect(self.hide)
        btn_layout.addWidget(close_btn)

        layout.addWidget(btn_bar)

    # ------------------------------------------------------------------
    def set_code(self, code: str):
        self._editor.setPlainText(code or "")
        self._status_label.setText("")

    def get_code(self) -> str:
        return self._editor.toPlainText()

    def _on_apply(self):
        code = self.get_code()
        try:
            ast.parse(code if code.strip() else "pass")
            self._status_label.setText("")
        except SyntaxError as e:
            self._status_label.setText(f"SyntaxError: {e}")
            return
        self.code_saved.emit(code)
        self._status_label.setText("Saved.")
