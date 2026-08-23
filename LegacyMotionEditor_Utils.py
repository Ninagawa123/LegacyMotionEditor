"""
File Name: LegacyMotionEditor_Utils.py
Description: Utility functions, constants, and helper classes for LegacyMotionEditor.

Author      : Izumi Ninagawa
Created On  : Feb 4, 2026
Version     : 0.0.1
License     : MIT License
Copyright (c) 2026 Izumi Ninagawa
"""

import os
import sys
import math
import logging
import xml.etree.ElementTree as ET
import ast
import datetime
import json
import copy
import re
import shutil
import subprocess
import tempfile
import threading
import time
import numpy as np
import vtk

_log = logging.getLogger(__name__)
from vtk.util.numpy_support import vtk_to_numpy

from Qt import QtWidgets, QtCore, QtGui
from collections import deque
from dataclasses import dataclass, field

# =============================================================================
# Cross-platform helpers (macOS / Windows / Linux)
# =============================================================================

def primary_mod_held(modifiers) -> bool:
    """True if the platform primary shortcut modifier is held.

    - macOS: Command (Meta) or Control
    - Windows / Linux: Control (Ctrl)
    """
    ctrl = bool(modifiers & QtCore.Qt.ControlModifier)
    meta = bool(modifiers & QtCore.Qt.MetaModifier)
    if sys.platform == "darwin":
        return ctrl or meta
    return ctrl


def code_editor_placeholder() -> str:
    """Settings UI placeholder for external editor path."""
    if sys.platform == "darwin":
        return "/Applications/Visual Studio Code.app"
    if sys.platform == "win32":
        return r"C:\Users\<you>\AppData\Local\Programs\Microsoft VS Code\Code.exe"
    return "code  (or /usr/bin/code)"


def code_editor_browse_start_dir() -> str:
    """Initial directory for the external-editor file dialog."""
    if sys.platform == "darwin":
        for cand in (
            os.path.expanduser("~/Applications"),
            "/Applications",
            os.path.expanduser("~"),
        ):
            if os.path.isdir(cand):
                return cand
        return os.path.expanduser("~")
    if sys.platform == "win32":
        for key in ("ProgramFiles", "ProgramFiles(x86)", "LOCALAPPDATA"):
            cand = os.environ.get(key, "")
            if cand and os.path.isdir(cand):
                return cand
        return os.path.expanduser("~")
    for cand in ("/usr/bin", "/usr/local/bin", os.path.expanduser("~")):
        if os.path.isdir(cand):
            return cand
    return os.path.expanduser("~")


def code_editor_browse_filter() -> str:
    if sys.platform == "darwin":
        return "Applications (*.app);;All Files (*)"
    if sys.platform == "win32":
        return "Executables (*.exe);;All Files (*)"
    return "All Files (*)"


def resolve_external_editor_path(app_path: str) -> str | None:
    """Resolve a user-entered editor path or PATH command (e.g. ``code``)."""
    app_path = (app_path or "").strip().strip('"')
    if not app_path:
        return None
    app_path = os.path.expanduser(app_path)
    if os.path.exists(app_path):
        return app_path
    found = shutil.which(app_path)
    return found


def launch_external_editor(app_path: str, file_path: str) -> None:
    """Open ``file_path`` with an external editor. Raises OSError on failure."""
    resolved = resolve_external_editor_path(app_path)
    if not resolved:
        raise FileNotFoundError(
            f"Editor not found: {app_path!r} (set a full path or a command on PATH)"
        )
    file_path = os.path.abspath(file_path)
    if sys.platform == "darwin" and resolved.endswith(".app"):
        subprocess.Popen(
            ["open", "-a", resolved, file_path],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        return
    if sys.platform == "win32":
        # .exe / .cmd / .bat and PATH shims (code.cmd) — avoid shell=True.
        subprocess.Popen(
            [resolved, file_path],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            shell=False,
        )
        return
    subprocess.Popen(
        [resolved, file_path],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def install_signal_handlers(handler) -> None:
    """Install Ctrl+C / terminate handlers in a cross-platform way."""
    import signal as _signal
    _signal.signal(_signal.SIGINT, handler)
    for name in ("SIGTERM", "SIGBREAK"):  # SIGBREAK = Ctrl+Break on Windows
        sig = getattr(_signal, name, None)
        if sig is None:
            continue
        try:
            _signal.signal(sig, handler)
        except (ValueError, OSError):
            pass


# =============================================================================
# Undo / Redo Stack
# =============================================================================

class LMEUndoStack:
    """Snapshot-based undo/redo stack for LegacyMotionEditor."""

    def __init__(self, max_size=100):
        self._undo = deque(maxlen=max_size)
        self._redo = deque(maxlen=max_size)

    def push(self, snapshot):
        """Push a before-state snapshot; clears redo."""
        self._undo.append(snapshot)
        self._redo.clear()

    def do_undo(self, current_snapshot):
        """Pop previous state; push current to redo. Returns previous or None."""
        if not self._undo:
            return None
        self._redo.append(current_snapshot)
        return self._undo.pop()

    def do_redo(self, current_snapshot):
        """Pop next state; push current to undo. Returns next or None."""
        if not self._redo:
            return None
        self._undo.append(current_snapshot)
        return self._redo.pop()

    def can_undo(self):
        return bool(self._undo)

    def can_redo(self):
        return bool(self._redo)

    def clear(self):
        self._undo.clear()
        self._redo.clear()

    def set_max_size(self, max_size):
        self._undo = deque(self._undo, maxlen=max_size)
        self._redo = deque(self._redo, maxlen=max_size)

    @property
    def max_size(self):
        return self._undo.maxlen

# =============================================================================
# Application Settings
# =============================================================================

_APP_SETTINGS_ORG = "LegacyMotionEditor"
_APP_SETTINGS_APP = "LegacyMotionEditor"
_DEFAULT_HZ_FPS = 100  # Hz(FPS) - Pose更新レートと再生FPSの共通デフォルト値


def _app_settings():
    return QtCore.QSettings(_APP_SETTINGS_ORG, _APP_SETTINGS_APP)


def get_default_hz_fps():
    """Hz(FPS)の統合された既定値を取得"""
    v = _app_settings().value("motion/hz_fps", _DEFAULT_HZ_FPS)
    try:
        return max(1, min(1000, int(v)))
    except (TypeError, ValueError):
        return _DEFAULT_HZ_FPS


# Default node offset values
_DEFAULT_NODE_OFFSET_X = 170
_DEFAULT_NODE_OFFSET_Y = 100


def get_node_offset_x():
    """新規ノード追加時のXオフセットを取得"""
    v = _app_settings().value("node/offset_x", _DEFAULT_NODE_OFFSET_X)
    try:
        return int(v)
    except (TypeError, ValueError):
        return _DEFAULT_NODE_OFFSET_X


def get_node_offset_y():
    """新規ノード追加時のYオフセットを取得"""
    v = _app_settings().value("node/offset_y", _DEFAULT_NODE_OFFSET_Y)
    try:
        return int(v)
    except (TypeError, ValueError):
        return _DEFAULT_NODE_OFFSET_Y


def set_node_offset(x, y):
    """新規ノード追加時のオフセットを保存"""
    settings = _app_settings()
    settings.setValue("node/offset_x", int(x))
    settings.setValue("node/offset_y", int(y))


# =============================================================================
# Debug Logger
# =============================================================================

_LME_DIR = os.path.dirname(os.path.abspath(__file__))
DEBUG_LOG_FILE = os.path.join(_LME_DIR, "debug.txt")
# Autosave / crash-recovery session (relative to package: save/_lme_session.xml)
SAVE_DIR = os.path.join(_LME_DIR, "save")
APP_SETTINGS_FILE = os.path.join(SAVE_DIR, "LegacyMotionEditor_Settings.json")
# Pre-relocation path (package root); still accepted on load for compatibility.
_APP_SETTINGS_FILE_LEGACY = os.path.join(_LME_DIR, "LegacyMotionEditor_Settings.json")
_LEGACY_SETTINGS_FILE = os.path.join(_LME_DIR, "LegacyMotionEditor_settings.json")
_OLD_SETTINGS_FILE = os.path.join(_LME_DIR, "app_settings.json")
SESSION_FILE_PATH = os.path.join(SAVE_DIR, "_lme_session.xml")
# Pre-relocation path (package root); still accepted on load for compatibility.
SESSION_FILE_PATH_LEGACY = os.path.join(_LME_DIR, "_lme_session.xml")

_SETTINGS_PATH_KEYS = (
    "last_xml_path",
    "last_model_path",
    "last_project_path",
    "last_cartridge_export_dir",
)


def path_for_project_save(path, save_filepath=None):
    """Persist ``path`` as POSIX-relative (LME package first, else save-file dir)."""
    if not path:
        return ""
    abs_path = os.path.abspath(os.path.expanduser(str(path)))
    bases = [_LME_DIR]
    if save_filepath:
        bases.append(os.path.dirname(os.path.abspath(save_filepath)))
    inside = []
    outside = []
    for base in bases:
        try:
            rel = os.path.relpath(abs_path, base).replace("\\", "/")
        except ValueError:
            continue
        if rel == ".":
            rel = "./"
        elif not rel.startswith(".") and not os.path.isabs(rel):
            rel = "./" + rel
        if rel.startswith("../") or rel == "..":
            outside.append(rel)
        else:
            inside.append(rel)
    if inside:
        return inside[0]
    if outside:
        return outside[0]
    return abs_path.replace("\\", "/")


def resolve_project_path(path, save_filepath=None):
    """Resolve a saved relative/absolute path to an existing file when possible."""
    if not path:
        return ""
    raw = str(path).strip()
    expanded = os.path.expanduser(raw)
    candidates = []
    if os.path.isabs(expanded):
        candidates.append(os.path.normpath(expanded))
    if save_filepath:
        candidates.append(os.path.normpath(
            os.path.join(os.path.dirname(os.path.abspath(save_filepath)), raw)))
    candidates.append(os.path.normpath(os.path.join(_LME_DIR, raw)))
    if not os.path.isabs(expanded):
        candidates.append(os.path.abspath(expanded))
    seen = set()
    for cand in candidates:
        if not cand or cand in seen:
            continue
        seen.add(cand)
        if os.path.exists(cand):
            return cand
    return candidates[0] if candidates else raw


def ensure_session_save_dir() -> str:
    """Create save/ if needed; return SESSION_FILE_PATH."""
    os.makedirs(SAVE_DIR, exist_ok=True)
    return SESSION_FILE_PATH


def resolve_session_file_for_load():
    """Prefer save/_lme_session.xml; fall back to legacy package-root path."""
    if os.path.exists(SESSION_FILE_PATH):
        return SESSION_FILE_PATH
    if os.path.exists(SESSION_FILE_PATH_LEGACY):
        return SESSION_FILE_PATH_LEGACY
    return None


class DebugLogger:
    """デバッグログをファイルに書き出すクラス"""
    _instance = None
    _file = None

    @classmethod
    def init(cls):
        """起動時に呼び出し、ログファイルをクリアして初期化"""
        cls._file = open(DEBUG_LOG_FILE, "w", encoding="utf-8")
        cls.log("[INIT]", "Debug logger initialized")

    @classmethod
    def log(cls, tag, message, **kwargs):
        """ログを書き出す"""
        if cls._file is None:
            return
        timestamp = datetime.datetime.now().strftime("%H:%M:%S.%f")[:-3]
        extra = ""
        if kwargs:
            extra = " | " + " | ".join(f"{k}={v}" for k, v in kwargs.items())
        line = f"{timestamp} {tag} {message}{extra}\n"
        cls._file.write(line)
        cls._file.flush()

    @classmethod
    def close(cls):
        """終了時に呼び出し"""
        if cls._file:
            cls.log("[CLOSE]", "Debug logger closed")
            cls._file.close()
            cls._file = None


# Tags omitted from debug.txt (hot-path / noisy diagnostics)
_LME_DEBUG_SUPPRESSED_TAGS = frozenset({
    "[Branch]", "[VirtualGraph]", "[TRIGGER]", "[SAVE]", "[LOAD]", "[NODE]",
})


def dbg(tag, message, **kwargs):
    """Write to debug.txt unless tag is in the quiet list."""
    if tag in _LME_DEBUG_SUPPRESSED_TAGS:
        return
    DebugLogger.log(tag, message, **kwargs)


# =============================================================================
# Console output (quiet by default)
# =============================================================================

import builtins

_ORIGINAL_PRINT = builtins.print
_LME_QUIET_CONSOLE = True
_LME_PRINT_FILTER_INSTALLED = False

_LME_SUPPRESSED_PREFIXES = (
    "[Playback]", "[Branch]", "[DEBUG", "[Motion]", "[Duplicate]",
    "[Auto]", "[PORT POSITION]", "[JumpCallback]", "[JumpFunc]",
    "[FooNode]", "[JumpNode]", "[PoseNode]", "[DefineNode]",
    "[BranchingNode]", "[CommandNode]", "[MixNode]",
    "[VTK]", "[LOOP]", "[ModelImport]", "[URDFParser]", "[MJCFRobotModel]",
    "[MJCF", "[Normalize]", "[Session]", "[ProjectXML]", "[MotionJSON]",
    "[Start Over]", "[Paste]", "[Cut]", "[clear_graph]",
    "[BaseLinkNode]", "[LongPress]", "[JointEditor]", "[Cleanup]",
    "[ExportCartridge]", "[Project]", "[Select All]",
    "[draw_path]", "[Graph]", "[Registered", "[JumpEdit]",
    "[VirtualGraph]",
    "[LMEValkey] write", "[LMEValkey] joint mapping", "[LMEValkey] unmatched",
    "[WalkPlayback]",
)

_LME_ALWAYS_SHOW_SUBSTRINGS = (
    "an error occurred", "traceback", "exception:", " failed", "failed:",
    "warning:", " unavailable:", "cannot import", "cannot delete",
    "importer unavailable", "not found", "connected ", "disconnected",
    "evaluation error",
)


def configure_lme_console(*, quiet: bool | None = None, verbose: bool = False) -> None:
    """Toggle filtered console output (verbose=True prints everything)."""
    global _LME_QUIET_CONSOLE
    if verbose:
        _LME_QUIET_CONSOLE = False
    elif quiet is not None:
        _LME_QUIET_CONSOLE = bool(quiet)


def _lme_should_print_console(message: str) -> bool:
    if not _LME_QUIET_CONSOLE:
        return True
    msg_lower = message.lower()
    for needle in _LME_ALWAYS_SHOW_SUBSTRINGS:
        if needle in msg_lower:
            return True
    stripped = message.lstrip()
    for prefix in _LME_SUPPRESSED_PREFIXES:
        if stripped.startswith(prefix):
            return False
    noisy_exact = (
        "Registered node type:",
        "Base Link node created successfully",
        "Input validators setup completed",
        "Node positions recalculated",
        "3D view updated",
        "No node selected",
    )
    for prefix in noisy_exact:
        if stripped.startswith(prefix):
            return False
    return True


def _lme_filtered_print(*args, **kwargs) -> None:
    if not args:
        _ORIGINAL_PRINT(*args, **kwargs)
        return
    try:
        message = " ".join(str(a) for a in args)
    except Exception:
        _ORIGINAL_PRINT(*args, **kwargs)
        return
    if _lme_should_print_console(message):
        _ORIGINAL_PRINT(*args, **kwargs)


def install_lme_quiet_console(*, quiet: bool = True, verbose: bool = False) -> None:
    """Install global print filter for LME (call once at startup)."""
    global _LME_PRINT_FILTER_INSTALLED
    configure_lme_console(quiet=quiet, verbose=verbose)
    if not _LME_PRINT_FILTER_INSTALLED:
        builtins.print = _lme_filtered_print
        _LME_PRINT_FILTER_INSTALLED = True


def load_app_settings():
    """アプリ設定をJSONから読み込む。壊れている場合は空設定として扱う。"""
    if not os.path.exists(APP_SETTINGS_FILE):
        import shutil
        for legacy_path in (_APP_SETTINGS_FILE_LEGACY, _LEGACY_SETTINGS_FILE, _OLD_SETTINGS_FILE):
            if os.path.exists(legacy_path):
                try:
                    os.makedirs(SAVE_DIR, exist_ok=True)
                    shutil.copy2(legacy_path, APP_SETTINGS_FILE)
                    print(f"[Settings] Migrated settings to {APP_SETTINGS_FILE}")
                except Exception as e:
                    print(f"[Settings] Migration failed: {e}")
                break
    try:
        if os.path.exists(APP_SETTINGS_FILE):
            with open(APP_SETTINGS_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                for key in _SETTINGS_PATH_KEYS:
                    if data.get(key):
                        data[key] = resolve_project_path(data[key])
                return data
    except Exception as e:
        print(f"[Settings] Failed to load app settings: {e}")
    return {}


def save_app_settings(settings):
    """アプリ設定をJSONへ保存する。"""
    try:
        os.makedirs(SAVE_DIR, exist_ok=True)
        out = dict(settings) if isinstance(settings, dict) else settings
        if isinstance(out, dict):
            out = dict(out)
            for key in _SETTINGS_PATH_KEYS:
                if out.get(key):
                    out[key] = path_for_project_save(out[key])
        with open(APP_SETTINGS_FILE, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2, ensure_ascii=False)
    except Exception as e:
        print(f"[Settings] Failed to save app settings: {e}")


# =============================================================================
# Valkey / Meridim Constants
# =============================================================================

MERIDIM_SIZE = 90
MASTER_CMD_RESET = 5556.0
# LME → Valkey パケット識別（Commander 再書き込みエコーと区別）
LME_PACKET_MARKER_SLOT = 89
LME_PACKET_MARKER_VALUE = 1.0
VALKEY_DEFAULT_HOST = "127.0.0.1"
VALKEY_DEFAULT_PORT = 6379
VALKEY_DEFAULT_WRITE_KEY = "merikey_psclon_sub"
VALKEY_DEFAULT_READ_KEY  = "merikey_psclon_pub"

# =============================================================================
# Valkey availability / Windows Docker auto-start
# =============================================================================
# On the dev machines, Valkey runs in a Docker container named
# "physicalon-valkey" (see docker-compose / manual `docker run` setup). On
# macOS/Ubuntu that container (or a native valkey-server) is expected to
# already be running, so no auto-start is attempted there. On Windows, Docker
# Desktop is often not started yet when LME launches, so we best-effort bring
# it up automatically instead of just failing to connect.

DOCKER_VALKEY_CONTAINER = "physicalon-valkey"
DOCKER_AUTOSTART_TIMEOUT_S = 60.0
_CREATE_NO_WINDOW = 0x08000000 if os.name == "nt" else 0


def valkey_available(host: str, port: int, timeout_s: float = 0.35) -> bool:
    """Return True if a Valkey/Redis server answers PING."""
    try:
        import valkey
        client = valkey.Valkey(
            host=str(host), port=int(port),
            socket_connect_timeout=timeout_s, socket_timeout=timeout_s)
        client.ping()
        try:
            client.close()
        except Exception:
            pass
        return True
    except Exception:
        return False


def _docker_daemon_ready(timeout_s: float = 2.0) -> bool:
    """Return True if `docker info` succeeds (daemon is up and reachable)."""
    try:
        r = subprocess.run(
            ["docker", "info"], capture_output=True, timeout=timeout_s,
            creationflags=_CREATE_NO_WINDOW)
        return r.returncode == 0
    except Exception:
        return False


def _launch_docker_desktop() -> bool:
    """Best-effort launch of Docker Desktop so its daemon comes up. Windows only."""
    candidates = [
        os.path.expandvars(r"%ProgramFiles%\Docker\Docker\Docker Desktop.exe"),
        os.path.expandvars(r"%ProgramFiles(x86)%\Docker\Docker\Docker Desktop.exe"),
        os.path.expandvars(r"%LocalAppData%\Docker\Docker Desktop.exe"),
    ]
    for exe in candidates:
        if os.path.isfile(exe):
            try:
                subprocess.Popen([exe], creationflags=subprocess.DETACHED_PROCESS)
                _log.info("Launched Docker Desktop: %s", exe)
                return True
            except Exception as e:
                _log.warning("Failed to launch Docker Desktop (%s): %s", exe, e)
                return False
    _log.warning("Docker Desktop executable not found in default locations")
    return False


def ensure_valkey_container_running(host: str, port: int) -> bool:
    """Best-effort auto-start of the physicalon-valkey Docker container (Windows).

    If Valkey isn't already reachable, brings up Docker Desktop (if needed) and
    starts the `physicalon-valkey` container, then waits for Valkey to answer
    PING. Meant to run off the main thread — it blocks up to
    DOCKER_AUTOSTART_TIMEOUT_S seconds. Returns True once Valkey is reachable.
    No-op (returns False) on macOS/Ubuntu, where Valkey is expected to already
    be running.
    """
    if valkey_available(host, port):
        return True
    if os.name != "nt":
        return False

    deadline = time.monotonic() + DOCKER_AUTOSTART_TIMEOUT_S

    if not _docker_daemon_ready():
        _log.info("Docker daemon not running — launching Docker Desktop")
        _launch_docker_desktop()
        while time.monotonic() < deadline and not _docker_daemon_ready():
            time.sleep(2.0)

    if _docker_daemon_ready():
        try:
            subprocess.run(
                ["docker", "start", DOCKER_VALKEY_CONTAINER],
                capture_output=True, timeout=15, creationflags=_CREATE_NO_WINDOW)
            _log.info("docker start %s issued", DOCKER_VALKEY_CONTAINER)
        except Exception as e:
            _log.warning("docker start %s failed: %s", DOCKER_VALKEY_CONTAINER, e)
    else:
        _log.warning("Docker daemon did not become ready within %.0fs",
                      DOCKER_AUTOSTART_TIMEOUT_S)

    while time.monotonic() < deadline:
        if valkey_available(host, port):
            return True
        time.sleep(1.0)
    return valkey_available(host, port)

# LME URDF joint name → (meridim_index, multiplier)
# Keys are URDF joint names used by LegacyMotionEditor (roid1.urdf).
# multiplier matches PhysicalOn joint_to_meridis: +1.0 = symmetric, -1.0 = axis-inverted R joints.
# write_angles() converts: deg / mul * 100 → Meridim *100 wire format.
# Reader divides by 100 then multiplies by mul to recover MJCF ctrl.
JOINT_TO_MERIDIM = {
    # === RobotLabelBridge canonical names (primary) ===
    # Head / Chest
    "c_neck_zy":                            (21,  1.0),
    "c_spine_01_yp":                        (51,  1.0),
    # Left arm
    "l_shoulder_yp":                        (23,  1.0),
    "l_shoulder_xr":                        (25,  1.0),
    "l_elbow_zy":                           (27,  1.0),
    "l_elbow_yp":                           (29,  1.0),
    # Left leg
    "l_hipjoint_zy":                         (31,  1.0),
    "l_hipjoint_xr":                        (33,  1.0),
    "l_hipjoint_yp":                        (35,  1.0),
    "l_knee_yp":                            (37,  1.0),
    "l_ankle_yp":                           (39,  1.0),
    "l_ankle_xr":                           (41,  1.0),
    # Right arm
    "r_shoulder_yp":                        (53,  1.0),
    "r_shoulder_xr":                        (55, -1.0),
    "r_elbow_zy":                           (57, -1.0),
    "r_elbow_yp":                           (59,  1.0),
    # Right leg
    "r_hipjoint_zy":                         (61, -1.0),
    "r_hipjoint_xr":                        (63, -1.0),
    "r_hipjoint_yp":                        (65,  1.0),
    "r_knee_yp":                            (67,  1.0),
    "r_ankle_yp":                           (69,  1.0),
    "r_ankle_xr":                           (71, -1.0),
    # === Original MJCF joint names (leven_mjcf / roid1.urdf raw names) ===
    "l_hipjoint_zy":                             (31,  1.0),
    "l_hipjoint_xr":                             (33,  1.0),
    "l_hipjoint_yp":                             (35,  1.0),
    "r_hipjoint_zy":                             (61, -1.0),
    "r_hipjoint_xr":                             (63, -1.0),
    "r_hipjoint_yp":                             (65,  1.0),
    # === Legacy MJCF link-chain aliases (backward compatibility) ===
    "c_chest_to_c_head":                    (21,  1.0),
    "c_waist_to_c_chest":                   (51,  1.0),
    "c_chest_to_l_shoulder":                (23,  1.0),
    "l_shoulder_to_l_arm_upper":            (25,  1.0),
    "l_arm_upper_to_l_elbow":              (27,  1.0),
    "l_elbow_to_l_arm_lower":              (29,  1.0),
    "c_waist_to_l_hipjoint_upper":          (31,  1.0),
    "l_hipjoint_upper_to_l_hipjoint_lower": (33,  1.0),
    "l_hipjoint_lower_to_l_leg_upper":      (35,  1.0),
    "l_leg_upper_to_l_leg_lower":           (37,  1.0),
    "l_leg_lower_to_l_ankle":              (39,  1.0),
    "l_ankle_to_l_foot":                   (41,  1.0),
    "c_chest_to_r_shoulder":                (53,  1.0),
    "r_shoulder_to_r_arm_upper":            (55, -1.0),
    "r_arm_upper_to_r_elbow":              (57, -1.0),
    "r_elbow_to_r_arm_lower":              (59,  1.0),
    "c_waist_to_r_hipjoint_upper":          (61, -1.0),
    "r_hipjoint_upper_to_r_hipjoint_lower": (63, -1.0),
    "r_hipjoint_lower_to_r_leg_upper":      (65,  1.0),
    "r_leg_upper_to_r_leg_lower":           (67,  1.0),
    "r_leg_lower_to_r_ankle":              (69,  1.0),
    "r_ankle_to_r_foot":                   (71, -1.0),
    # === arthropod (multi-legged robots: up to 8 legs per side) ===
    # _xr = abduction/adduction axis, _yp = protraction/retraction axis.
    # Indices overlap with humanoid — safe because only one morphology loads at a time.
    # Multipliers default 1.0; adjust per hardware servo orientation if needed.
    "l_leg_01_xr":  (21,  1.0),  "l_leg_01_yp":  (23,  1.0),
    "l_leg_02_xr":  (25,  1.0),  "l_leg_02_yp":  (27,  1.0),
    "l_leg_03_xr":  (29,  1.0),  "l_leg_03_yp":  (31,  1.0),
    "l_leg_04_xr":  (33,  1.0),  "l_leg_04_yp":  (35,  1.0),
    "l_leg_05_xr":  (37,  1.0),  "l_leg_05_yp":  (39,  1.0),
    "l_leg_06_xr":  (41,  1.0),  "l_leg_06_yp":  (43,  1.0),
    "l_leg_07_xr":  (45,  1.0),  "l_leg_07_yp":  (47,  1.0),
    "l_leg_08_xr":  (49,  1.0),  "l_leg_08_yp":  (51,  1.0),
    "r_leg_01_xr":  (51, -1.0),  "r_leg_01_yp":  (53,  1.0),
    "r_leg_02_xr":  (55, -1.0),  "r_leg_02_yp":  (57,  1.0),
    "r_leg_03_xr":  (59, -1.0),  "r_leg_03_yp":  (61,  1.0),
    "r_leg_04_xr":  (63, -1.0),  "r_leg_04_yp":  (65,  1.0),
    "r_leg_05_xr":  (67, -1.0),  "r_leg_05_yp":  (69,  1.0),
    "r_leg_06_xr":  (71, -1.0),  "r_leg_06_yp":  (73,  1.0),
    "r_leg_07_xr":  (75, -1.0),  "r_leg_07_yp":  (77,  1.0),
    "r_leg_08_xr":  (79, -1.0),  "r_leg_08_yp":  (81,  1.0),
    # === quadruped ===
    "l_fore_scapula_xr":   (21,  1.0),
    "l_fore_shoulder_xr":  (23,  1.0),
    "l_fore_shoulder_yp":  (25,  1.0),
    "l_fore_shoulder_zy":  (27,  1.0),
    "l_fore_elbow_yp":     (29,  1.0),
    "l_fore_carpus_yp":    (31,  1.0),
    "l_fore_paw_xr":       (33,  1.0),
    "l_hind_hipjoint_xr":  (35,  1.0),
    "l_hind_hipjoint_yp":  (37,  1.0),
    "l_hind_hipjoint_zy":  (39,  1.0),
    "l_hind_stifle_yp":    (41,  1.0),
    "l_hind_tarsus_yp":    (43,  1.0),
    "l_hind_paw_xr":       (45,  1.0),
    "r_fore_scapula_xr":   (47, -1.0),
    "r_fore_shoulder_xr":  (49, -1.0),
    "r_fore_shoulder_yp":  (51,  1.0),
    "r_fore_shoulder_zy":  (53, -1.0),
    "r_fore_elbow_yp":     (55,  1.0),
    "r_fore_carpus_yp":    (57,  1.0),
    "r_fore_paw_xr":       (59, -1.0),
    "r_hind_hipjoint_xr":  (61, -1.0),
    "r_hind_hipjoint_yp":  (63,  1.0),
    "r_hind_hipjoint_zy":  (65, -1.0),
    "r_hind_stifle_yp":    (67,  1.0),
    "r_hind_tarsus_yp":    (69,  1.0),
    "r_hind_paw_xr":       (71, -1.0),
    "l_fore_lower_limb_yp": (31,  1.0),
    "r_fore_lower_limb_yp": (57,  1.0),
    "l_hind_upper_limb_yp": (37,  1.0),
    "l_hind_lower_limb_yp": (43,  1.0),
    "r_hind_upper_limb_yp": (63,  1.0),
    "r_hind_lower_limb_yp": (69,  1.0),
    # === avian ===
    "l_wing_shoulder_xr":  (21,  1.0),
    "l_wing_shoulder_yp":  (23,  1.0),
    "l_wing_shoulder_zy":  (25,  1.0),
    "l_wing_elbow_yp":     (27,  1.0),
    "l_wing_wrist_yp":     (29,  1.0),
    "l_alula_yp":          (31,  1.0),
    "l_wing_upper_zy":     (33,  1.0),
    "l_wing_fore_yp":      (35,  1.0),
    "l_wing_hand_yp":      (37,  1.0),
    "r_wing_shoulder_xr":  (39, -1.0),
    "r_wing_shoulder_yp":  (41,  1.0),
    "r_wing_shoulder_zy":  (43, -1.0),
    "r_wing_elbow_yp":     (45,  1.0),
    "r_wing_wrist_yp":     (47,  1.0),
    "r_alula_yp":          (49,  1.0),
    "r_wing_upper_zy":     (51, -1.0),
    "r_wing_fore_yp":      (53,  1.0),
    "r_wing_hand_yp":      (55,  1.0),
    # === generic_vertebrate (neck / tail chains) ===
    "c_neck_01_yp":  (21,  1.0),  "c_neck_02_yp":  (23,  1.0),
    "c_neck_03_yp":  (25,  1.0),  "c_neck_04_yp":  (27,  1.0),
    "c_neck_05_yp":  (29,  1.0),  "c_neck_06_yp":  (31,  1.0),
    "c_tail_01_yp":  (33,  1.0),  "c_tail_01_zy":  (35,  1.0),
    "c_tail_02_yp":  (37,  1.0),  "c_tail_02_zy":  (39,  1.0),
    "c_tail_03_yp":  (41,  1.0),  "c_tail_03_zy":  (43,  1.0),
    "c_tail_04_yp":  (45,  1.0),  "c_tail_04_zy":  (47,  1.0),
    "c_tail_05_yp":  (49,  1.0),  "c_tail_05_zy":  (51,  1.0),
    "c_tail_06_yp":  (53,  1.0),  "c_tail_06_zy":  (55,  1.0),
    "c_tail_07_yp":  (57,  1.0),  "c_tail_07_zy":  (59,  1.0),
    "c_tail_08_yp":  (61,  1.0),  "c_tail_08_zy":  (63,  1.0),
    "c_tail_09_yp":  (65,  1.0),  "c_tail_09_zy":  (67,  1.0),
    "c_tail_10_yp":  (69,  1.0),  "c_tail_10_zy":  (71,  1.0),
}


# Canonical MJCF joint names (leven_mjcf / RobotLabelBridge).
# MERIDIM_JOINT_MAP must use these — PhysicalOn resolves p1_{name} in MuJoCo.
CANONICAL_MJCF_JOINTS: tuple[str, ...] = (
    "c_neck_zy", "c_spine_01_yp",
    "l_shoulder_yp", "l_shoulder_xr", "l_elbow_zy", "l_elbow_yp",
    "l_hipjoint_zy", "l_hipjoint_xr", "l_hipjoint_yp",
    "l_knee_yp", "l_ankle_yp", "l_ankle_xr",
    "r_shoulder_yp", "r_shoulder_xr", "r_elbow_zy", "r_elbow_yp",
    "r_hipjoint_zy", "r_hipjoint_xr", "r_hipjoint_yp",
    "r_knee_yp", "r_ankle_yp", "r_ankle_xr",
)

# Logic_cartridge_sample.py naming convention.
MJCF_JOINT_TO_SHORTNAME: dict[str, str] = {
    "c_neck_zy":      "C_HEAD",
    "c_spine_01_yp":  "C_CHEST",
    "l_shoulder_yp":  "L_SHOULDER_PITCH",
    "l_shoulder_xr":  "L_SHOULDER_ROLL",
    "l_elbow_zy":     "L_ELBOW_YAW",
    "l_elbow_yp":     "L_ELBOW_PITCH",
    "l_hipjoint_zy":  "L_HIPJOINT_ZY",
    "l_hipjoint_xr":  "L_HIPJOINT_XR",
    "l_hipjoint_yp":  "L_HIPJOINT_YP",
    "l_knee_yp":      "L_KNEE_YP",
    "l_ankle_yp":     "L_ANKLE_YP",
    "l_ankle_xr":     "L_ANKLE_XR",
    "r_shoulder_yp":  "R_SHOULDER_PITCH",
    "r_shoulder_xr":  "R_SHOULDER_ROLL",
    "r_elbow_zy":     "R_ELBOW_YAW",
    "r_elbow_yp":     "R_ELBOW_PITCH",
    "r_hipjoint_zy":  "R_HIPJOINT_ZY",
    "r_hipjoint_xr":  "R_HIPJOINT_XR",
    "r_hipjoint_yp":  "R_HIPJOINT_YP",
    "r_knee_yp":      "R_KNEE_YP",
    "r_ankle_yp":     "R_ANKLE_YP",
    "r_ankle_xr":     "R_ANKLE_XR",
}


def resolve_joint_meridim(joint_name: str) -> tuple[int, float] | None:
    """Return (meridim_index, sign_mul) for any known joint alias."""
    entry = JOINT_TO_MERIDIM.get(joint_name)
    if entry is None:
        return None
    idx, mul = entry
    return int(idx), float(mul)


def meridim_angle_from_joint(joint_name: str, angle_deg: float) -> tuple[int, float] | None:
    """Convert LME MJCF-space angle [deg] to (meridim_idx, meridim_angle)."""
    resolved = resolve_joint_meridim(joint_name)
    if resolved is None:
        return None
    idx, mul = resolved
    if mul == 0.0:
        return None
    return idx, float(angle_deg) / mul


def _cartridge_joint_used(mjcf_name: str, joints_used: set[str]) -> bool:
    if mjcf_name in joints_used:
        return True
    target_entry = JOINT_TO_MERIDIM.get(mjcf_name)
    if target_entry is None:
        return False
    for name in joints_used:
        entry = JOINT_TO_MERIDIM.get(name)
        if entry is not None and entry[0] == target_entry[0]:
            return True
    return False


def build_joints_dict(
    joints_used: set[str] | None = None,
    joints: tuple[str, ...] | None = None,
) -> dict[str, int]:
    """{SHORT_NAME: meridim_idx} for exported cartridge JOINTS dict."""
    result: dict[str, int] = {}
    for mjcf_name in (joints if joints is not None else CANONICAL_MJCF_JOINTS):
        entry = JOINT_TO_MERIDIM.get(mjcf_name)
        if entry is None:
            continue
        if joints_used is not None and not _cartridge_joint_used(mjcf_name, joints_used):
            continue
        short = MJCF_JOINT_TO_SHORTNAME.get(mjcf_name, mjcf_name.upper())
        meridim_idx, _sign = entry
        result[short] = int(meridim_idx)
    return result


def build_meridim_joint_map(
    joints: tuple[str, ...] | None = None,
    role: str = "servo",
) -> list[dict]:
    """MERIDIM_JOINT_MAP entries for PhysicalOn (canonical MJCF joint names).

    Pass ``joints`` (from robot_model.joint_order) to use the loaded model's
    actual joint list instead of the humanoid-only CANONICAL_MJCF_JOINTS.
    """
    entries: list[dict] = []
    for mjcf_name in (joints if joints is not None else CANONICAL_MJCF_JOINTS):
        entry = JOINT_TO_MERIDIM.get(mjcf_name)
        if entry is None:
            continue
        meridim_idx, sign = entry
        entries.append({
            "joint": mjcf_name,
            "meridim": int(meridim_idx),
            "sign": float(sign),
            "role": role,
        })
    return entries


# =============================================================================
# =============================================================================
# Color Constants
# =============================================================================

# ノード接続線の色設定
MINT_GREEN_COLOR = (0, 25, 225)
BRANCH_POINT_COLOR = (255, 0, 0)
BRANCH_LINE_COLOR = (255, 0, 0)

# パレット色設定 (RGB)
PALETTE_WINDOW = (200, 200, 200)
PALETTE_WINDOW_TEXT = (20, 20, 20)
PALETTE_BASE = (250, 250, 250)
PALETTE_ALTERNATE_BASE = (66, 66, 66)
PALETTE_TOOLTIP_BASE = (255, 255, 255)
PALETTE_TOOLTIP_TEXT = (255, 255, 255)
PALETTE_TEXT = (20, 20, 20)
PALETTE_BUTTON = (53, 53, 53)
PALETTE_BUTTON_TEXT = (255, 255, 255)
PALETTE_BRIGHT_TEXT = (255, 0, 0)
PALETTE_HIGHLIGHT = (42, 130, 218)
PALETTE_HIGHLIGHTED_TEXT = (0, 0, 0)

# ノードグラフ設定 (RGB)
NODE_GRAPH_BG_COLOR = (225, 225, 225)
NODE_GRAPH_GRID_COLOR = (200, 200, 200)
NODE_GRAPH_GRID_SNAP_SIZE = 50
NODE_GRAPH_FOCUS_COLOR = (20, 20, 20)
NODE_GRAPH_FOCUS_TEXT_COLOR = (0, 0, 0)

# ノードの色設定 (RGB 0-255)
NODE_COLOR_DEFAULT = (180, 180, 180)

# --- BaseLinkNode (スタートノード) ---
NODE_START_TITLE_COLOR = (55, 55, 58)
NODE_START_TITLE_BG_COLOR = (226, 226, 230)
NODE_START_PANEL_BG_COLOR = (180, 180, 180)
NODE_START_TEXT_COLOR = (20, 20, 20)
NODE_START_INPUT_PORT_COLOR = (40, 40, 40)
NODE_START_INPUT_PORT_BORDER_COLOR = (30, 30, 30)
NODE_START_OUTPUT_PORT_COLOR = (225, 225, 225)
NODE_START_OUTPUT_PORT_BORDER_COLOR = (80, 80, 80)
NODE_START_TITLE_HIGHLIGHT_COLOR = (40, 40, 44)
NODE_START_TITLE_BG_HIGHLIGHT_COLOR = (205, 208, 212)
NODE_START_PANEL_BG_HIGHLIGHT_COLOR = (200, 200, 200)
NODE_START_INPUT_PORT_HIGHLIGHT_COLOR = (225, 225, 225)
NODE_START_INPUT_PORT_HIGHLIGHT_BORDER_COLOR = (80, 80, 80)
NODE_START_OUTPUT_PORT_HIGHLIGHT_COLOR = NODE_START_OUTPUT_PORT_COLOR
NODE_START_OUTPUT_PORT_HIGHLIGHT_BORDER_COLOR = NODE_START_OUTPUT_PORT_BORDER_COLOR

# --- FooNode (基本ノード) ---
NODE_BASIC_TITLE_COLOR = (250, 250, 250)
NODE_BASIC_TITLE_BG_COLOR = (120, 120, 120)
NODE_BASIC_PANEL_BG_COLOR = (180, 180, 180)
NODE_BASIC_TEXT_COLOR = (20, 20, 20)
NODE_BASIC_INPUT_PORT_COLOR = (180, 80, 0)
NODE_BASIC_INPUT_PORT_BORDER_COLOR = (120, 50, 0)
NODE_BASIC_OUTPUT_PORT_COLOR = NODE_START_OUTPUT_PORT_COLOR
NODE_BASIC_OUTPUT_PORT_BORDER_COLOR = NODE_START_OUTPUT_PORT_BORDER_COLOR
NODE_BASIC_TITLE_HIGHLIGHT_COLOR = (0, 0, 0)
NODE_BASIC_TITLE_BG_HIGHLIGHT_COLOR = (100, 150, 255)
NODE_BASIC_PANEL_BG_HIGHLIGHT_COLOR = (250, 250, 250)
NODE_BASIC_INPUT_PORT_HIGHLIGHT_COLOR = (225, 225, 225)
NODE_BASIC_INPUT_PORT_HIGHLIGHT_BORDER_COLOR = (80, 80, 80)
NODE_BASIC_OUTPUT_PORT_HIGHLIGHT_COLOR = NODE_START_OUTPUT_PORT_COLOR
NODE_BASIC_OUTPUT_PORT_HIGHLIGHT_BORDER_COLOR = NODE_START_OUTPUT_PORT_BORDER_COLOR

# --- PoseNode (ポーズノード) ---
NODE_POSE_TITLE_COLOR = (250, 250, 250)
NODE_POSE_TITLE_BG_COLOR = (120, 120, 120)
NODE_POSE_PANEL_BG_COLOR = (180, 180, 180)
NODE_POSE_TEXT_COLOR = (20, 20, 20)
NODE_POSE_INPUT_PORT_COLOR = (180, 80, 0)
NODE_POSE_INPUT_PORT_BORDER_COLOR = (120, 50, 0)
NODE_POSE_OUTPUT_PORT_COLOR = NODE_START_OUTPUT_PORT_COLOR
NODE_POSE_OUTPUT_PORT_BORDER_COLOR = NODE_START_OUTPUT_PORT_BORDER_COLOR
NODE_POSE_BRANCH_TO_PORT_COLOR = BRANCH_POINT_COLOR
NODE_POSE_BRANCH_TO_PORT_BORDER_COLOR = (180, 0, 0)
NODE_POSE_BRANCH_OTHERWISE_PORT_COLOR = (47, 128, 237)
NODE_POSE_BRANCH_OTHERWISE_PORT_BORDER_COLOR = (20, 90, 180)
NODE_POSE_TITLE_HIGHLIGHT_COLOR = (0, 0, 0)
NODE_POSE_TITLE_BG_HIGHLIGHT_COLOR = (100, 150, 255)
NODE_POSE_PANEL_BG_HIGHLIGHT_COLOR = (200, 220, 255)
NODE_POSE_INPUT_PORT_HIGHLIGHT_COLOR = (225, 225, 225)
NODE_POSE_INPUT_PORT_HIGHLIGHT_BORDER_COLOR = (80, 80, 80)
NODE_POSE_OUTPUT_PORT_HIGHLIGHT_COLOR = NODE_START_OUTPUT_PORT_COLOR
NODE_POSE_OUTPUT_PORT_HIGHLIGHT_BORDER_COLOR = NODE_START_OUTPUT_PORT_BORDER_COLOR

# --- DefineNode ---
NODE_DEFINE_TITLE_COLOR = NODE_POSE_TITLE_COLOR
NODE_DEFINE_TITLE_BG_COLOR = (100, 100, 105)
NODE_DEFINE_PANEL_BG_COLOR = (195, 195, 200)
NODE_DEFINE_TITLE_HIGHLIGHT_COLOR = NODE_POSE_TITLE_HIGHLIGHT_COLOR
NODE_DEFINE_TITLE_BG_HIGHLIGHT_COLOR = (90, 130, 210)
NODE_DEFINE_PANEL_BG_HIGHLIGHT_COLOR = (205, 215, 235)

# --- WaitNode ---
NODE_WAIT_TITLE_COLOR = NODE_POSE_TITLE_COLOR
NODE_WAIT_TITLE_BG_COLOR = NODE_POSE_TITLE_BG_COLOR   # same beige as PoseNode
NODE_WAIT_PANEL_BG_COLOR = (200, 202, 198)             # light gray panel
NODE_WAIT_TITLE_HIGHLIGHT_COLOR = NODE_POSE_TITLE_HIGHLIGHT_COLOR
NODE_WAIT_TITLE_BG_HIGHLIGHT_COLOR = NODE_POSE_TITLE_BG_HIGHLIGHT_COLOR
NODE_WAIT_PANEL_BG_HIGHLIGHT_COLOR = (220, 222, 218)

# --- BranchingNode ---
NODE_BRANCH_TITLE_COLOR = (255, 252, 248)
NODE_BRANCH_TITLE_BG_COLOR = (115, 85, 55)
NODE_BRANCH_PANEL_BG_COLOR = (215, 198, 172)
NODE_BRANCH_TITLE_HIGHLIGHT_COLOR = (45, 30, 15)
NODE_BRANCH_TITLE_BG_HIGHLIGHT_COLOR = (165, 125, 80)
NODE_BRANCH_PANEL_BG_HIGHLIGHT_COLOR = (235, 223, 200)

# --- MixNode ---
NODE_MIX_TITLE_COLOR = (255, 255, 250)
NODE_MIX_TITLE_BG_COLOR = (180, 100, 50)
NODE_MIX_PANEL_BG_COLOR = (240, 200, 160)
NODE_MIX_INPUT_PORT_COLOR = (200, 120, 40)
NODE_MIX_INPUT_PORT_BORDER_COLOR = (150, 80, 20)
NODE_MIX_OUTPUT_PORT_COLOR = (220, 140, 60)
NODE_MIX_OUTPUT_PORT_BORDER_COLOR = (170, 100, 30)
NODE_MIX_TITLE_HIGHLIGHT_COLOR = (60, 30, 10)
NODE_MIX_TITLE_BG_HIGHLIGHT_COLOR = (220, 140, 70)
NODE_MIX_PANEL_BG_HIGHLIGHT_COLOR = (255, 230, 200)
NODE_MIX_INPUT_PORT_HIGHLIGHT_COLOR = (255, 220, 180)
NODE_MIX_INPUT_PORT_HIGHLIGHT_BORDER_COLOR = (180, 120, 60)
NODE_MIX_OUTPUT_PORT_HIGHLIGHT_COLOR = (255, 200, 140)
NODE_MIX_OUTPUT_PORT_HIGHLIGHT_BORDER_COLOR = (200, 140, 70)

# --- CommandNode ---
NODE_COMMAND_TITLE_COLOR = (255, 255, 250)
NODE_COMMAND_TITLE_BG_COLOR = (100, 80, 150)
NODE_COMMAND_PANEL_BG_COLOR = (200, 190, 220)
NODE_COMMAND_INPUT_PORT_COLOR = (120, 100, 180)
NODE_COMMAND_INPUT_PORT_BORDER_COLOR = (80, 60, 130)
NODE_COMMAND_OUTPUT_PORT_COLOR = (140, 120, 200)
NODE_COMMAND_OUTPUT_PORT_BORDER_COLOR = (100, 80, 150)
NODE_COMMAND_TITLE_HIGHLIGHT_COLOR = (30, 20, 50)
NODE_COMMAND_TITLE_BG_HIGHLIGHT_COLOR = (140, 120, 190)
NODE_COMMAND_PANEL_BG_HIGHLIGHT_COLOR = (230, 220, 250)
NODE_COMMAND_INPUT_PORT_HIGHLIGHT_COLOR = (200, 180, 255)
NODE_COMMAND_INPUT_PORT_HIGHLIGHT_BORDER_COLOR = (140, 120, 200)
NODE_COMMAND_OUTPUT_PORT_HIGHLIGHT_COLOR = (180, 160, 240)
NODE_COMMAND_OUTPUT_PORT_HIGHLIGHT_BORDER_COLOR = (140, 120, 200)

# Command types for servo control
SERVO_COMMAND_TYPES = [
    (0, "Torq Off", "Torque off"),
    (1, "Torq On", "Torque on"),
    (50, "Stretch1", "Holding force at small angles (P gain, KONDO)"),
    (51, "Stretch2", "Holding force at all angles except Stretch1 (P gain, KONDO)"),
    (52, "Speed", "Servo power (P gain, KONDO)"),
    (53, "Punch", "Servo power in small range (KONDO)"),
    (54, "DeadBand", "Insensitive zone"),
    (55, "Damping", "Spring characteristics (PI gain, KONDO/FUTABA)"),
    (56, "Compliance Margin", "Target position tolerance (FUTABA)"),
    (57, "Compliance Slope", "Torque to return to target (I gain, FUTABA)"),
]

# --- JumpNode ---
NODE_JUMP_TITLE_COLOR = (248, 252, 255)
NODE_JUMP_TITLE_BG_COLOR = (65, 82, 110)
NODE_JUMP_PANEL_BG_COLOR = (78, 96, 128)
NODE_JUMP_TITLE_HIGHLIGHT_COLOR = (255, 255, 255)
NODE_JUMP_TITLE_BG_HIGHLIGHT_COLOR = (95, 118, 160)
NODE_JUMP_PANEL_BG_HIGHLIGHT_COLOR = (105, 128, 170)

# その他の色設定
COLOR_SAMPLE_DEFAULT = (255, 255, 255)
LABEL_TEXT_COLOR = (20, 20, 20)
MESH_HIGHLIGHT_COLOR = (0.5, 0.8, 1.0)
MESH_HIGHLIGHT_BLINK_INTERVAL = 500
PLAYBACK_HIGHLIGHT_COLOR = (50, 205, 50)
PLAYBACK_HIGHLIGHT_BORDER_WIDTH = 4
PLAYBACK_INCOMPLETE_COLOR = (255, 200, 50)
PLAYBACK_INCOMPLETE_BORDER_WIDTH = 3
MESH_DRAG_SENSITIVITY = 0.5
MESH_WHEEL_SENSITIVITY = 0.1


# =============================================================================
# Layout Constants
# =============================================================================

LEFT_PANEL_WIDTH = 140
NODE_INSPECTOR_MIN_WIDTH = 400
JOINT_EDITOR_WIDTH = 900
VTK_DISPLAY_MIN_WIDTH = 100
VTK_DISPLAY_MIN_HEIGHT = 100
SPLITTER_NODE_GRAPH_WIDTH = 500
SPLITTER_3DVIEW_WIDTH = 300
SPLITTER_JOINT_EDITOR_WIDTH = 500
NODE_GRAPH_MIN_WIDTH = 200
RIGHT_PANEL_MIN_WIDTH = 200

# 3Dビュー設定
VTK_BACKGROUND_COLOR = "#1a1a1a"
VTK_BG_SLIDER_DEFAULT = 50
VTK_BG_COLOR_A = [0.1, 0.1, 0.1]
VTK_BG_COLOR_B = [0.3, 0.3, 0.3]
VTK_BG_GRADIENT_TYPE = "vertical"


# =============================================================================
# PAD Constants
# =============================================================================

PAD_BUTTON_NAMES = (
    "L1", "L2", "DPad_Up", "DPad_Left", "DPad_Right", "DPad_Down",
    "Select", "Start", "Triangle", "Square", "Circle", "Cross", "R1", "R2"
)
PAD_AXIS_NAMES = ("Lx", "Ly", "Rx", "Ry", "L2v", "R2v")

PAD_REGISTER_VALUES = {
    "Pad_btn": 0,
    "Pad_L1": 0, "Pad_L2": 0,
    "Pad_DPad_Up": 0, "Pad_DPad_Left": 0, "Pad_DPad_Right": 0, "Pad_DPad_Down": 0,
    "Pad_Select": 0, "Pad_Start": 0,
    "Pad_Triangle": 0, "Pad_Square": 0, "Pad_Circle": 0, "Pad_Cross": 0,
    "Pad_R1": 0, "Pad_R2": 0,
    "Pad_Lx": 0, "Pad_Ly": 0, "Pad_Rx": 0, "Pad_Ry": 0,
    "Pad_L2v": 0, "Pad_R2v": 0,
}

PAD_REGISTER_ALIASES = {
    "pad_buttons": "Pad_btn", "pad_btn": "Pad_btn", "pad_bit": "Pad_btn",
    "padbuttons": "Pad_btn", "padbtn": "Pad_btn",
    "pad_lx": "Pad_Lx", "pad_stick_lx": "Pad_Lx",
    "pad_ly": "Pad_Ly", "pad_stick_ly": "Pad_Ly",
    "pad_rx": "Pad_Rx", "pad_stick_rx": "Pad_Rx",
    "pad_ry": "Pad_Ry", "pad_stick_ry": "Pad_Ry",
    "pad_l2v": "Pad_L2v", "pad_trigger_l": "Pad_L2v",
    "pad_r2v": "Pad_R2v", "pad_trigger_r": "Pad_R2v",
}

PAD_BUTTON_BIT_VALUES = {
    "pad_btn_r_up": 4096, "pad_btn_r_right": 8192,
    "pad_btn_r_down": 16384, "pad_btn_r_left": 32768,
    "pad_btn_l_up": 16, "pad_btn_l_down": 64,
    "pad_btn_l_left": 128, "pad_btn_l_right": 32,
    "pad_btn_l1": 1024, "pad_btn_l2": 256,
    "pad_btn_r1": 2048, "pad_btn_r2": 512,
    "pad_btn_select": 1, "pad_btn_start": 8,
    "pad_btn_none": 0, "pad_btn_0": 0,
}

# Branching PAD mode: button choices and mapping to PAD_REGISTER_VALUES keys
PAD_IF_BUTTON_CHOICES = (
    "L1", "L2", "R1", "R2",
    "TRI", "CIR", "SQR", "CRS",
    "UP", "DOWN", "LEFT", "RIGHT",
    "SELECT", "START",
)
PAD_IF_BUTTON_TO_PAD_KEY = {
    "L1":       "Pad_L1",
    "L2":       "Pad_L2",
    "R1":       "Pad_R1",
    "R2":       "Pad_R2",
    "TRI":      "Pad_Triangle",
    "CIR":      "Pad_Circle",
    "SQR":      "Pad_Square",
    "CRS":      "Pad_Cross",
    "UP":   "Pad_DPad_Up",
    "DOWN": "Pad_DPad_Down",
    "LEFT": "Pad_DPad_Left",
    "RIGHT":"Pad_DPad_Right",
    "SELECT":   "Pad_Select",
    "START":    "Pad_Start",
}

# PAD analog axis branching (sticks: -127~+127, triggers: 0~255)
PAD_IF_ANALOG_AXIS_CHOICES = ("Lx", "Ly", "Rx", "Ry", "L2v", "R2v")
PAD_IF_ANALOG_OP_CHOICES = (">=", "<=")
PAD_IF_ANALOG_AXIS_TO_PAD_KEY = {
    "Lx":  "Pad_Lx",
    "Ly":  "Pad_Ly",
    "Rx":  "Pad_Rx",
    "Ry":  "Pad_Ry",
    "L2v": "Pad_L2v",
    "R2v": "Pad_R2v",
}
PAD_IF_ANALOG_AXIS_RANGE = {
    "Lx": (-127, 127), "Ly": (-127, 127),
    "Rx": (-127, 127), "Ry": (-127, 127),
    "L2v": (0, 255),   "R2v": (0, 255),
}

# =============================================================================
# User Value Session
# =============================================================================

USER_VALUE_SESSION_COUNT = 64


def default_user_value_session():
    return [{"kind": "literal", "value": 0} for _ in range(USER_VALUE_SESSION_COUNT)]


def normalize_user_value_session(raw):
    """UserVal_0〜63 用（エディタセッション内のみ）。"""
    out = default_user_value_session()
    if not isinstance(raw, list):
        return out
    for i in range(USER_VALUE_SESSION_COUNT):
        if i >= len(raw) or not isinstance(raw[i], dict):
            continue
        d = raw[i]
        k = d.get("kind", "literal")
        if k == "register":
            nm = str(d.get("name", "")).strip()
            if nm:
                out[i] = {"kind": "register", "name": nm}
        else:
            try:
                v = int(d.get("value", 0))
            except (TypeError, ValueError):
                v = 0
            v = max(-32768, min(32767, v))
            out[i] = {"kind": "literal", "value": v}
    return out


# =============================================================================
# Branch Register Functions
# =============================================================================

_BRANCH_REGISTER_LEFT_DEFAULT = "merged_motion_condition_variable_map\u306e\u30b3\u30d2\u309a\u30fc.numbers"
_BRANCH_REGISTER_RIGHT_DEFAULT = "merged_motion_condition_variable_map\u306e\u30b3\u30d2\u309a\u30fc3.numbers"
_BRANCH_REGISTER_CACHE = {}

_BRANCH_IF_OPERATOR_CHOICES = ("==", "!=", ">", ">=", "<", "<=", "and", "or")


def normalize_branch_if_op_stored(op):
    """古い保存形式の '=' を '==' に揃える。"""
    if op == "=":
        return "=="
    return op


def _branch_register_env_path(side):
    key = "LEGACY_BRANCH_IF_LEFT_NUMBERS" if side == "left" else "LEGACY_BRANCH_IF_RIGHT_NUMBERS"
    p = (os.environ.get(key) or "").strip()
    return os.path.expanduser(p) if p else ""


def _branch_register_default_numbers_path(side):
    base = os.path.join(os.path.expanduser("~"), "Documents")
    name = _BRANCH_REGISTER_LEFT_DEFAULT if side == "left" else _BRANCH_REGISTER_RIGHT_DEFAULT
    return os.path.join(base, name)


def _branch_register_candidate_paths(side):
    primary = _branch_register_env_path(side) or _branch_register_default_numbers_path(side)
    out = [primary]
    if primary.endswith(".numbers"):
        out.append(primary[: -len(".numbers")] + ".csv")
    return out


def _load_branch_register_csv_first_column(path):
    import csv
    out = []
    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        for i, row in enumerate(reader):
            if not row or not str(row[0]).strip():
                continue
            cell = str(row[0]).strip()
            if i == 0 and cell == "generic_variable_name":
                continue
            out.append(cell)
    return out


def _load_branch_register_numbers_first_column(path):
    try:
        from numbers_parser import Document
    except ImportError:
        return []
    try:
        doc = Document(path)
    except Exception:
        return []
    if not doc.sheets or not doc.sheets[0].tables:
        return []
    table = doc.sheets[0].tables[0]
    start = 0
    try:
        h = table.cell(0, 0).value
        if h is not None and str(h).strip() == "generic_variable_name":
            start = 1
    except Exception:
        start = 0
    out = []
    seen = set()
    for r in range(start, table.num_rows):
        try:
            v = table.cell(r, 0).value
        except Exception:
            continue
        if v is None:
            continue
        s = str(v).strip()
        if not s or s in seen:
            continue
        seen.add(s)
        out.append(s)
    return out


def _load_register_items_from_resolved_path(path):
    if not path or not os.path.isfile(path):
        return []
    try:
        mtime = os.path.getmtime(path)
    except OSError:
        return []
    hit = _BRANCH_REGISTER_CACHE.get(path)
    if hit and hit[0] == mtime:
        return list(hit[1])
    items = []
    if path.endswith(".numbers"):
        items = _load_branch_register_numbers_first_column(path)
        if not items:
            csv_p = path[: -len(".numbers")] + ".csv"
            if os.path.isfile(csv_p):
                try:
                    items = _load_branch_register_csv_first_column(csv_p)
                except Exception:
                    items = []
    elif path.endswith(".csv"):
        try:
            items = _load_branch_register_csv_first_column(path)
        except Exception:
            items = []
    if items:
        _BRANCH_REGISTER_CACHE[path] = (mtime, tuple(items))
    return list(items)


def load_branch_register_items_for_side(side):
    for p in _branch_register_candidate_paths(side):
        items = _load_register_items_from_resolved_path(p)
        if items:
            return items
    return []


# =============================================================================
# Math Functions
# =============================================================================

def _rpy_to_rotation_matrix(roll, pitch, yaw):
    """RPY角(radian)から3x3回転行列を生成"""
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    R = np.array([
        [cy*cp, cy*sp*sr - sy*cr, cy*sp*cr + sy*sr],
        [sy*cp, sy*sp*sr + cy*cr, sy*sp*cr - cy*sr],
        [-sp,   cp*sr,            cp*cr           ]
    ])
    return R


def _axis_angle_to_rotation_matrix(axis, angle_rad):
    """軸角度表現から3x3回転行列を生成 (Rodrigues)"""
    ax = np.array(axis, dtype=float)
    norm = np.linalg.norm(ax)
    if norm < 1e-12:
        return np.eye(3)
    ax = ax / norm
    K = np.array([
        [0, -ax[2], ax[1]],
        [ax[2], 0, -ax[0]],
        [-ax[1], ax[0], 0]
    ])
    R = np.eye(3) + math.sin(angle_rad) * K + (1 - math.cos(angle_rad)) * (K @ K)
    return R


def _make_4x4(R, t):
    """3x3回転行列 + 3ベクトルから4x4変換行列を生成"""
    M = np.eye(4)
    M[:3, :3] = R
    M[:3, 3] = t
    return M


def rpy_to_matrix(roll, pitch, yaw):
    """Convert RPY angles (radians) to 4x4 transformation matrix."""
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)

    R = np.array([
        [cy*cp, cy*sp*sr - sy*cr, cy*sp*cr + sy*sr, 0],
        [sy*cp, sy*sp*sr + cy*cr, sy*sp*cr - cy*sr, 0],
        [-sp,   cp*sr,            cp*cr,            0],
        [0,     0,                0,                1]
    ])
    return R


def make_transform_matrix(xyz, rpy):
    """Create 4x4 homogeneous transformation matrix from xyz translation and rpy rotation.

    The transformation matrix combines rotation and translation:
    T = | R   t |
        | 0   1 |

    Where R is the 3x3 rotation matrix from RPY and t is the translation vector.
    """
    # Get rotation matrix (4x4 with identity in bottom-right)
    R = rpy_to_matrix(rpy[0], rpy[1], rpy[2])
    # Set translation in the right column
    R[0, 3] = xyz[0]
    R[1, 3] = xyz[1]
    R[2, 3] = xyz[2]
    return R


def make_scale_matrix(scale):
    """Create 4x4 scale matrix from xyz scale."""
    S = np.eye(4)
    if scale and len(scale) >= 3:
        S[0, 0] = scale[0]
        S[1, 1] = scale[1]
        S[2, 2] = scale[2]
    return S


def quat_to_rpy_xyzw(quat):
    """Convert MJCF quaternion [w, x, y, z] to RPY radians.

    Uses rotation matrix extraction to correctly handle gimbal lock (pitch = ±90°).
    """
    if not quat or len(quat) < 4:
        return [0.0, 0.0, 0.0]
    w, x, y, z = quat[:4]

    # Normalize quaternion (MuJoCo allows non-normalized quaternions like "1 0 -1 0")
    norm = math.sqrt(w*w + x*x + y*y + z*z)
    if norm > 1e-10:
        w, x, y, z = w/norm, x/norm, y/norm, z/norm

    # Build rotation matrix from quaternion
    R00 = 1 - 2*(y*y + z*z)
    R01 = 2*(x*y - w*z)
    R02 = 2*(x*z + w*y)
    R10 = 2*(x*y + w*z)
    R11 = 1 - 2*(x*x + z*z)
    R12 = 2*(y*z - w*x)
    R20 = 2*(x*z - w*y)
    R21 = 2*(y*z + w*x)
    R22 = 1 - 2*(x*x + y*y)

    # Extract RPY from rotation matrix (ZYX Euler convention)
    # R = Rz(yaw) * Ry(pitch) * Rx(roll)
    sy = math.sqrt(R00*R00 + R10*R10)
    singular = sy < 1e-6

    if not singular:
        roll = math.atan2(R21, R22)
        pitch = math.atan2(-R20, sy)
        yaw = math.atan2(R10, R00)
    else:
        # Gimbal lock: pitch = ±90°, combine roll and yaw
        roll = math.atan2(-R12, R11)
        pitch = math.atan2(-R20, sy)
        yaw = 0.0

    return [roll, pitch, yaw]


def _win_spin_arrow_icon_paths():
    """Render small up/down triangle PNGs to a cache dir and return their paths.

    Qt's QSS `image: url(data:...)` did not reliably load in testing on this
    Qt/PySide build, so we render real files once and reference them by path
    instead (a plain file path is the well-supported QSS pattern).
    """
    cache_dir = os.path.join(tempfile.gettempdir(), "lme_spin_icons")
    os.makedirs(cache_dir, exist_ok=True)
    up_path = os.path.join(cache_dir, "spin_up.png")
    down_path = os.path.join(cache_dir, "spin_down.png")
    if not (os.path.isfile(up_path) and os.path.isfile(down_path)):
        for path, points in (
            (up_path, [(1, 6), (7, 6), (4, 1)]),
            (down_path, [(1, 1), (7, 1), (4, 6)]),
        ):
            pixmap = QtGui.QPixmap(8, 8)
            pixmap.fill(QtCore.Qt.transparent)
            painter = QtGui.QPainter(pixmap)
            painter.setRenderHint(QtGui.QPainter.Antialiasing, True)
            painter.setPen(QtCore.Qt.NoPen)
            painter.setBrush(QtGui.QColor(30, 30, 30))
            painter.drawPolygon(QtGui.QPolygon([QtCore.QPoint(x, y) for x, y in points]))
            painter.end()
            pixmap.save(path, "PNG")
    return up_path.replace("\\", "/"), down_path.replace("\\", "/")


class _WinSpinBoxStyler(QtCore.QObject):
    """Windows-only: assigns the Fusion QStyle to every QAbstractSpinBox.

    The native "windowsvista" style ignores QSS subcontrol customization
    (::up-button/::down-button/::up-arrow/::down-arrow) for spin boxes
    entirely, so the narrow/visible-arrow QSS rules in apply_dark_theme()
    have no effect unless the widget itself uses Fusion (which fully honors
    QSS subcontrol styling). Rather than touching every QSpinBox/
    QDoubleSpinBox construction site in the app, this filter catches each
    spin box's first Polish event and switches just that widget to Fusion —
    every other widget type keeps the native Windows look untouched.
    """

    def __init__(self, parent=None):
        super(_WinSpinBoxStyler, self).__init__(parent)
        self._fusion_style = QtWidgets.QStyleFactory.create("Fusion")

    def eventFilter(self, obj, event):
        if (event.type() == QtCore.QEvent.Polish
                and isinstance(obj, QtWidgets.QAbstractSpinBox)
                and self._fusion_style is not None):
            obj.setStyle(self._fusion_style)
        return False


def apply_dark_theme(app):
    """Apply dark theme palette to the application."""
    QPalette = QtGui.QPalette
    QColor = QtGui.QColor
    dark_palette = QPalette()
    dark_palette.setColor(QPalette.Window, QColor(*PALETTE_WINDOW))
    dark_palette.setColor(QPalette.WindowText, QColor(*PALETTE_WINDOW_TEXT))
    dark_palette.setColor(QPalette.Base, QColor(*PALETTE_BASE))
    dark_palette.setColor(QPalette.AlternateBase, QColor(*PALETTE_ALTERNATE_BASE))
    dark_palette.setColor(QPalette.ToolTipBase, QColor(*PALETTE_TOOLTIP_BASE))
    dark_palette.setColor(QPalette.ToolTipText, QColor(*PALETTE_TOOLTIP_TEXT))
    dark_palette.setColor(QPalette.Text, QColor(*PALETTE_TEXT))
    dark_palette.setColor(QPalette.Button, QColor(*PALETTE_BUTTON))
    dark_palette.setColor(QPalette.ButtonText, QColor(*PALETTE_BUTTON_TEXT))
    dark_palette.setColor(QPalette.BrightText, QColor(*PALETTE_BRIGHT_TEXT))
    dark_palette.setColor(QPalette.Highlight, QColor(*PALETTE_HIGHLIGHT))
    dark_palette.setColor(QPalette.HighlightedText, QColor(*PALETTE_HIGHLIGHTED_TEXT))
    app.setPalette(dark_palette)
    # ネイティブスタイル（特にWindows）はQPalette.ButtonTextを無視してボタン背景を
    # 明るいまま描画することがあり、白文字が読めなくなるためQSSで明示的に黒文字に固定する。
    # 個別にsetStyleSheet()済みのボタンはウィジェット側の指定が優先されるため影響しない。
    extra_qss = "\nQPushButton, QToolButton { color: black; }"
    if sys.platform == "win32":
        # Windows native style (windowsvista) draws QAbstractSpinBox's up/down
        # buttons via the OS visual-styles theme but the tiny arrow glyph via
        # QPalette.ButtonText — which this app sets to white for the dark
        # QPalette.Button above. The button chrome itself stays native light
        # grey (same ignoring-the-palette behavior as QPushButton, see note
        # above), so a white arrow on a light button is invisible. The native
        # theme also sizes the button subcontrols generously, which eats into
        # the value text at the small fixed widths used for joint spin boxes
        # (e.g. digits get clipped). macOS/Linux styles don't have either
        # problem, so this is Windows-only: Qt-draw the buttons ourselves with
        # a narrow fixed width and an explicit dark arrow colour, still
        # stacked top/bottom like the native look on other platforms.
        styler = _WinSpinBoxStyler(app)
        app.installEventFilter(styler)
        app._lme_win_spinbox_styler = styler  # keep alive (installEventFilter doesn't own it)

        up_icon, down_icon = _win_spin_arrow_icon_paths()
        extra_qss += f"""
QAbstractSpinBox {{
    padding-right: 14px;
}}
QAbstractSpinBox::up-button, QAbstractSpinBox::down-button {{
    subcontrol-origin: border;
    width: 14px;
    background-color: #e0e0e0;
    border-left: 1px solid #a0a0a0;
}}
QAbstractSpinBox::up-button {{
    subcontrol-position: top right;
    height: 12px;
    border-top-right-radius: 2px;
}}
QAbstractSpinBox::down-button {{
    subcontrol-position: bottom right;
    height: 12px;
    border-bottom-right-radius: 2px;
}}
QAbstractSpinBox::up-button:hover, QAbstractSpinBox::down-button:hover {{
    background-color: #cfcfcf;
}}
QAbstractSpinBox::up-button:pressed, QAbstractSpinBox::down-button:pressed {{
    background-color: #b8b8b8;
}}
QAbstractSpinBox::up-arrow {{
    image: url({up_icon});
    width: 7px;
    height: 7px;
}}
QAbstractSpinBox::down-arrow {{
    image: url({down_icon});
    width: 7px;
    height: 7px;
}}
"""
    app.setStyleSheet((app.styleSheet() or "") + extra_qss)


# =============================================================================
# Easing Functions
# =============================================================================

def _ease_out_bounce(x):
    n1 = 7.5625
    d1 = 2.75
    if x < 1 / d1:
        return n1 * x * x
    if x < 2 / d1:
        x -= 1.5 / d1
        return n1 * x * x + 0.75
    if x < 2.5 / d1:
        x -= 2.25 / d1
        return n1 * x * x + 0.9375
    x -= 2.625 / d1
    return n1 * x * x + 0.984375


def _ease_in_bounce(x):
    return 1 - _ease_out_bounce(1 - x)


def _ease_in_out_bounce(x):
    if x < 0.5:
        return (1 - _ease_out_bounce(1 - 2 * x)) / 2
    return (1 + _ease_out_bounce(2 * x - 1)) / 2


def _ease_in_elastic(x):
    if x == 0:
        return 0
    if x == 1:
        return 1
    c4 = (2 * math.pi) / 3
    return -(2 ** (10 * x - 10)) * math.sin((x * 10 - 10.75) * c4)


def _ease_out_elastic(x):
    if x == 0:
        return 0
    if x == 1:
        return 1
    c4 = (2 * math.pi) / 3
    return (2 ** (-10 * x)) * math.sin((x * 10 - 0.75) * c4) + 1


def _ease_in_out_elastic(x):
    if x == 0:
        return 0
    if x == 1:
        return 1
    c5 = (2 * math.pi) / 4.5
    if x < 0.5:
        return -((2 ** (20 * x - 10)) * math.sin((20 * x - 11.125) * c5)) / 2
    return ((2 ** (-20 * x + 10)) * math.sin((20 * x - 11.125) * c5)) / 2 + 1


def _ease_in_back(x):
    c1 = 1.70158
    c3 = c1 + 1
    return c3 * x * x * x - c1 * x * x


def _ease_out_back(x):
    c1 = 1.70158
    c3 = c1 + 1
    return 1 + c3 * ((x - 1) ** 3) + c1 * ((x - 1) ** 2)


def _ease_in_out_back(x):
    c1 = 1.70158
    c2 = c1 * 1.525
    if x < 0.5:
        return ((2 * x) ** 2 * ((c2 + 1) * 2 * x - c2)) / 2
    return (((2 * x - 2) ** 2) * ((c2 + 1) * (x * 2 - 2) + c2) + 2) / 2


def _ease_in_expo(x):
    return 0 if x == 0 else 2 ** (10 * x - 10)


def _ease_out_expo(x):
    return 1 if x == 1 else 1 - 2 ** (-10 * x)


def _ease_in_out_expo(x):
    if x == 0:
        return 0
    if x == 1:
        return 1
    if x < 0.5:
        return (2 ** (20 * x - 10)) / 2
    return (2 - 2 ** (-20 * x + 10)) / 2


EASING_PRESETS = [
    ("linear", lambda x: x),
    ("easeInSine", lambda x: 1 - math.cos((x * math.pi) / 2)),
    ("easeOutSine", lambda x: math.sin((x * math.pi) / 2)),
    ("easeInOutSine", lambda x: -(math.cos(math.pi * x) - 1) / 2),
    ("easeInQuad", lambda x: x * x),
    ("easeOutQuad", lambda x: 1 - (1 - x) * (1 - x)),
    ("easeInOutQuad", lambda x: 2 * x * x if x < 0.5 else 1 - ((-2 * x + 2) ** 2) / 2),
    ("easeInCubic", lambda x: x * x * x),
    ("easeOutCubic", lambda x: 1 - ((1 - x) ** 3)),
    ("easeInOutCubic", lambda x: 4 * x * x * x if x < 0.5 else 1 - ((-2 * x + 2) ** 3) / 2),
    ("easeInQuart", lambda x: x ** 4),
    ("easeOutQuart", lambda x: 1 - ((1 - x) ** 4)),
    ("easeInOutQuart", lambda x: 8 * (x ** 4) if x < 0.5 else 1 - ((-2 * x + 2) ** 4) / 2),
    ("easeInQuint", lambda x: x ** 5),
    ("easeOutQuint", lambda x: 1 - ((1 - x) ** 5)),
    ("easeInOutQuint", lambda x: 16 * (x ** 5) if x < 0.5 else 1 - ((-2 * x + 2) ** 5) / 2),
    ("easeInExpo", _ease_in_expo),
    ("easeOutExpo", _ease_out_expo),
    ("easeInOutExpo", _ease_in_out_expo),
    ("easeInCirc", lambda x: 1 - math.sqrt(1 - x * x)),
    ("easeOutCirc", lambda x: math.sqrt(1 - ((x - 1) ** 2))),
    ("easeInOutCirc", lambda x: (1 - math.sqrt(1 - ((2 * x) ** 2))) / 2 if x < 0.5 else (math.sqrt(1 - ((-2 * x + 2) ** 2)) + 1) / 2),
    ("easeInBack", _ease_in_back),
    ("easeOutBack", _ease_out_back),
    ("easeInOutBack", _ease_in_out_back),
    ("easeInElastic", _ease_in_elastic),
    ("easeOutElastic", _ease_out_elastic),
    ("easeInOutElastic", _ease_in_out_elastic),
    ("easeInBounce", _ease_in_bounce),
    ("easeOutBounce", _ease_out_bounce),
    ("easeInOutBounce", _ease_in_out_bounce),
]

EASING_OPTIONS = [f"{idx}: {name}" for idx, (name, _) in enumerate(EASING_PRESETS)]
EASING_NAME_TO_INDEX = {name: idx for idx, (name, _) in enumerate(EASING_PRESETS)}


def easing_index(interpolation):
    key = str(interpolation or "0: linear")
    idx = 0
    if ":" in key:
        prefix = key.split(":", 1)[0].strip()
        if prefix.isdigit():
            idx = int(prefix)
    elif key in EASING_NAME_TO_INDEX:
        idx = EASING_NAME_TO_INDEX[key]
    elif key == "ease":
        idx = EASING_NAME_TO_INDEX.get("easeInOutSine", 3)
    if idx < 0 or idx >= len(EASING_PRESETS):
        idx = 0
    return idx


def easing_option(interpolation):
    return EASING_OPTIONS[easing_index(interpolation)]


def easing_value(interpolation, x):
    """easings.net準拠の補間率を返す。xは0.0-1.0の進行度。"""
    x = max(0.0, min(1.0, float(x)))
    idx = easing_index(interpolation)
    return EASING_PRESETS[idx][1](x)


# =============================================================================
# Helper Functions
# =============================================================================

def create_label(text):
    """スタイル付きQLabelを作成"""
    label = QtWidgets.QLabel(text)
    label.setStyleSheet(f"color: rgb({LABEL_TEXT_COLOR[0]},{LABEL_TEXT_COLOR[1]},{LABEL_TEXT_COLOR[2]});")
    return label


# メイン下部の H/V コンボ用
_MAIN_WINDOW_COMBO_TEXT_STYLE = (
    "QComboBox { color: black; } QComboBox QAbstractItemView { color: black; }"
)


# =============================================================================
# Virtual Node Classes (for cross-action playback)
# =============================================================================

class VirtualPort:
    """Virtual port for cross-action playback connections."""

    def __init__(self, node, port_name):
        self._node = node
        self._port_name = port_name
        self._connected_ports = []

    def node(self):
        return self._node

    def name(self):
        return self._port_name

    def connected_ports(self):
        return self._connected_ports

    def connect_to(self, other_port):
        if other_port not in self._connected_ports:
            self._connected_ports.append(other_port)


class VirtualBaseLinkNode:
    """Virtual StartNode for cross-action playback."""

    def __init__(self, node_data):
        self._name = node_data.get("name", "Start")
        self._id = node_data.get("id", "start")
        self._output_ports = [VirtualPort(self, "out")]

    def name(self):
        return self._name

    def output_ports(self):
        return self._output_ports

    def input_ports(self):
        return []


class VirtualPoseNode:
    """Virtual PoseNode for cross-action playback."""

    def __init__(self, node_data):
        self._name = node_data.get("name", "Pose")
        self._id = node_data.get("id", "")
        self.pose_name = node_data.get("name", "Pose")
        self.angles_deg = dict(node_data.get("angles_deg", {}))
        self.duration = float(node_data.get("duration", 1.0))
        self.frames = int(node_data.get("frames", get_default_hz_fps()))
        self.joint_easings = dict(node_data.get("joint_easings", {}))
        self.branching_enabled = node_data.get("branching_enabled", False)
        self.branch_outputs_swapped = node_data.get("branch_outputs_swapped", False)
        self.branch_if_left = node_data.get("branch_if_left", "UserVal_0")
        self.branch_if_op = node_data.get("branch_if_op", "==")
        self.branch_if_right = node_data.get("branch_if_right", "UserVal_1")
        self.branch_if_uv_enabled = node_data.get("branch_if_uv_enabled", True)
        self.branch_if_formula_enabled = node_data.get("branch_if_formula_enabled", False)
        self.branch_if_formula = node_data.get("branch_if_formula", "Form1:foo")
        self.branch_if_pad_enabled = node_data.get("branch_if_pad_enabled", False)
        self.branch_if_pad_button = node_data.get("branch_if_pad_button", "L1")
        self.branch_if_pad_analog_enabled = node_data.get("branch_if_pad_analog_enabled", False)
        self.branch_if_pad_analog_axis = node_data.get("branch_if_pad_analog_axis", "Lx")
        self.branch_if_pad_analog_op = node_data.get("branch_if_pad_analog_op", ">=")
        self.branch_if_pad_analog_threshold = int(node_data.get("branch_if_pad_analog_threshold", 0))
        self.out_port_labels = list(node_data.get("out_port_labels", ["default"]))
        self.out_port_priorities = list(node_data.get("out_port_priorities", [0]))
        # Create output ports
        self._output_ports = []
        for i, label in enumerate(self.out_port_labels):
            self._output_ports.append(VirtualPort(self, f"out_{i}"))
        if not self._output_ports:
            self._output_ports.append(VirtualPort(self, "out"))
        self._input_ports = [VirtualPort(self, "in")]

    def name(self):
        return self._name

    def output_ports(self):
        return self._output_ports

    def input_ports(self):
        return self._input_ports


class VirtualDefineNode:
    """Virtual DefineNode for cross-action playback."""

    def __init__(self, node_data):
        self._name = node_data.get("name", "Define")
        self._id = node_data.get("id", "")
        self.define_uv_index = int(node_data.get("define_uv_index", 0))
        self.define_memo = node_data.get("define_memo", "")
        self.define_kind = node_data.get("define_kind", "literal")
        self.define_literal = node_data.get("define_literal", 0)
        self.define_register_name = node_data.get("define_register_name", "")
        self._output_ports = [VirtualPort(self, "out")]
        self._input_ports = [VirtualPort(self, "in")]

    def name(self):
        return self._name

    def output_ports(self):
        return self._output_ports

    def input_ports(self):
        return self._input_ports


class VirtualWaitNode:
    """Virtual WaitNode for cross-action playback."""

    def __init__(self, node_data):
        self._name = node_data.get("name", "Wait")
        self._id = node_data.get("id", "")
        self.wait_name = node_data.get("name", "Wait")
        self.frames = int(node_data.get("frames", 0))
        self.duration = float(node_data.get("duration", 0.0))
        self.out_port_labels = list(node_data.get("out_port_labels", ["default"]))
        self.out_port_priorities = list(node_data.get("out_port_priorities", [0]))
        self._output_ports = [VirtualPort(self, "out")]
        self._input_ports = [VirtualPort(self, "in")]

    def name(self):
        return self._name

    def output_ports(self):
        return self._output_ports

    def input_ports(self):
        return self._input_ports


class VirtualBranchingNode:
    """Virtual BranchingNode for cross-action playback."""

    def __init__(self, node_data):
        self._name = node_data.get("name", "Branch")
        self._id = node_data.get("id", "")
        self.branching_enabled = node_data.get("branching_enabled", False)
        self.branch_outputs_swapped = node_data.get("branch_outputs_swapped", False)
        self.branch_if_left = node_data.get("branch_if_left", "UserVal_0")
        self.branch_if_op = node_data.get("branch_if_op", "==")
        self.branch_if_right = node_data.get("branch_if_right", "UserVal_1")
        self.branch_if_uv_enabled = node_data.get("branch_if_uv_enabled", True)
        self.branch_if_formula_enabled = node_data.get("branch_if_formula_enabled", False)
        self.branch_if_formula = node_data.get("branch_if_formula", "Form1:foo")
        self.branch_if_pad_enabled = node_data.get("branch_if_pad_enabled", False)
        self.branch_if_pad_button = node_data.get("branch_if_pad_button", "L1")
        self.branch_if_pad_analog_enabled = node_data.get("branch_if_pad_analog_enabled", False)
        self.branch_if_pad_analog_axis = node_data.get("branch_if_pad_analog_axis", "Lx")
        self.branch_if_pad_analog_op = node_data.get("branch_if_pad_analog_op", ">=")
        self.branch_if_pad_analog_threshold = int(node_data.get("branch_if_pad_analog_threshold", 0))
        self.out_port_labels = list(node_data.get("out_port_labels", ["then", "else"]))
        self.out_port_priorities = list(node_data.get("out_port_priorities", [0, 0]))
        # Create output ports
        self._output_ports = []
        for i, label in enumerate(self.out_port_labels):
            self._output_ports.append(VirtualPort(self, f"out_{i}"))
        if len(self._output_ports) < 2:
            self._output_ports.append(VirtualPort(self, "out_0"))
            self._output_ports.append(VirtualPort(self, "out_1"))
        self._input_ports = [VirtualPort(self, "in")]

    def name(self):
        return self._name

    def output_ports(self):
        return self._output_ports

    def input_ports(self):
        return self._input_ports


class VirtualMixNode:
    """Virtual MixNode for cross-action playback."""

    def __init__(self, node_data):
        self._name = node_data.get("name", "Mix")
        self._id = node_data.get("id", "")
        self.mix_name = node_data.get("name", "Mix")
        self.frames = int(node_data.get("frames", 1))
        # mix_settings: {joint_name: {enabled: bool, input_source: str, gain: float}}
        self.mix_settings = dict(node_data.get("mix_settings", {}))
        self._output_ports = [VirtualPort(self, "out")]
        self._input_ports = [VirtualPort(self, "in")]

    def name(self):
        return self._name

    def output_ports(self):
        return self._output_ports

    def input_ports(self):
        return self._input_ports


class VirtualCommandNode:
    """Virtual CommandNode for cross-action playback."""

    def __init__(self, node_data):
        self._name = node_data.get("name", "Command")
        self._id = node_data.get("id", "")
        self.command_name = node_data.get("name", "Command")
        self.frames = int(node_data.get("frames", 1))
        # command_settings: {joint_name: {command_type: int, value: int}}
        self.command_settings = dict(node_data.get("command_settings", {}))
        self._output_ports = [VirtualPort(self, "out")]
        self._input_ports = [VirtualPort(self, "in")]

    def name(self):
        return self._name

    def output_ports(self):
        return self._output_ports

    def input_ports(self):
        return self._input_ports


class VirtualJumpNode:
    """Virtual JumpNode for cross-action playback."""

    def __init__(self, node_data):
        self._name = node_data.get("name", "Jump")
        self._id = node_data.get("id", "")
        self.jump_target_action_index = int(node_data.get("jump_target_action_index", 0))
        self.jump_type = node_data.get("jump_type", "action")
        self.jump_target_function = node_data.get("jump_target_function", "")
        self.out_port_labels = list(node_data.get("out_port_labels", ["default"]))
        self.out_port_priorities = list(node_data.get("out_port_priorities", [0]))
        self._output_ports = [VirtualPort(self, "out")]
        self._input_ports = [VirtualPort(self, "in")]

    def name(self):
        return self._name

    def output_ports(self):
        return self._output_ports

    def input_ports(self):
        return self._input_ports


def build_virtual_graph_from_action_data(action_data):
    """Build virtual nodes and connections from action data.

    Args:
        action_data: Dictionary with 'nodes' and 'edges' keys

    Returns:
        VirtualBaseLinkNode (StartNode) or None if failed
    """
    nodes_data = action_data.get("nodes", [])
    edges_data = action_data.get("edges", [])

    if not nodes_data:
        return None

    # Create virtual nodes
    virtual_nodes = {}
    start_node = None

    # First, create StartNode (BaseLinkNode equivalent)
    start_node = VirtualBaseLinkNode({"name": "Start", "id": "start"})
    virtual_nodes["start"] = start_node

    # Create other nodes
    for nd in nodes_data:
        node_id = nd.get("id", "")
        node_type = nd.get("node_type", "")

        if node_type == "pose":
            virtual_nodes[node_id] = VirtualPoseNode(nd)
        elif node_type == "define":
            virtual_nodes[node_id] = VirtualDefineNode(nd)
        elif node_type == "wait":
            virtual_nodes[node_id] = VirtualWaitNode(nd)
        elif node_type == "branch":
            virtual_nodes[node_id] = VirtualBranchingNode(nd)
        elif node_type == "jump":
            virtual_nodes[node_id] = VirtualJumpNode(nd)
        elif node_type == "mix":
            virtual_nodes[node_id] = VirtualMixNode(nd)
        elif node_type == "command":
            virtual_nodes[node_id] = VirtualCommandNode(nd)

    # Build connections
    # Edge format: {"from": src_id, "to": tgt_id, "from_port": port_idx, "label": str, ...}
    for edge in edges_data:
        src_id = edge.get("from") or edge.get("source")
        tgt_id = edge.get("to") or edge.get("target")
        raw_fp = edge.get("from_port", edge.get("source_port", None))

        src_node = virtual_nodes.get(src_id)
        tgt_node = virtual_nodes.get(tgt_id)

        if src_node and tgt_node:
            out_ports = src_node.output_ports()
            in_ports = tgt_node.input_ports()
            if out_ports and in_ports:
                edge_label = edge.get("label", "")
                node_labels = getattr(src_node, "out_port_labels", [])
                port_idx = 0
                if edge_label:
                    # Label matching is always correct; from_port may be stale/wrong in saved data
                    matched = False
                    for i, lbl in enumerate(node_labels):
                        if lbl == edge_label and i < len(out_ports):
                            port_idx = i
                            matched = True
                            break
                    if not matched and raw_fp is not None:
                        # Label not found: fall back to from_port
                        port_idx = min(int(raw_fp), len(out_ports) - 1)
                elif raw_fp is not None:
                    # No label: use from_port directly
                    port_idx = min(int(raw_fp), len(out_ports) - 1)
                out_ports[port_idx].connect_to(in_ports[0])

    return start_node


# =============================================================================
# Generic UI Components
# =============================================================================

class OffscreenRenderer:
    """Offscreen VTK rendering to QLabel for macOS compatibility."""

    def __init__(self, render_window, renderer, render_lock=None):
        self.render_window = render_window
        self.renderer = renderer
        # Guards the actual VTK render+readback call below against the actor/
        # mesh data it reads being mutated concurrently (e.g. by a background
        # IK worker thread). Optional: pass None to skip locking entirely.
        self._render_lock = render_lock
        self._is_rendering = False

    def render_to_qpixmap(self):
        """Render VTK scene offscreen and return QPixmap."""
        if self._is_rendering:
            return None

        self._is_rendering = True
        try:
            if self._render_lock is not None:
                with self._render_lock:
                    return self._do_render_to_qpixmap()
            return self._do_render_to_qpixmap()
        finally:
            self._is_rendering = False

    def _do_render_to_qpixmap(self):
        try:
            self.render_window.Render()

            w2i = vtk.vtkWindowToImageFilter()
            w2i.SetInput(self.render_window)
            w2i.ReadFrontBufferOff()
            w2i.ShouldRerenderOn()
            w2i.Update()

            vtk_image = w2i.GetOutput()
            width, height, _ = vtk_image.GetDimensions()
            vtk_array = vtk_image.GetPointData().GetScalars()
            components = vtk_array.GetNumberOfComponents()

            arr = vtk_to_numpy(vtk_array)
            arr = arr.reshape(height, width, components)
            arr = np.flip(arr, axis=0)
            arr = np.ascontiguousarray(arr)

            if components == 3:
                qimage = QtGui.QImage(arr.data, width, height, width * 3, QtGui.QImage.Format_RGB888)
            else:
                qimage = QtGui.QImage(arr.data, width, height, width * 4, QtGui.QImage.Format_RGBA8888)

            pixmap = QtGui.QPixmap.fromImage(qimage.copy())
            return pixmap
        except Exception as e:
            print(f"[OffscreenRenderer] Error: {e}")
            return None

    def update_display(self, qlabel_widget):
        """Render and update QLabel display."""
        pixmap = self.render_to_qpixmap()
        if pixmap:
            scaled = pixmap.scaled(
                qlabel_widget.size(),
                QtCore.Qt.KeepAspectRatio,
                QtCore.Qt.SmoothTransformation
            )
            qlabel_widget.setPixmap(scaled)


class ArithmeticDoubleSpinBox(QtWidgets.QDoubleSpinBox):
    """Returnで四則演算の入力を計算できるDoubleSpinBox"""

    _ALLOWED_BINOPS = {
        ast.Add: lambda a, b: a + b,
        ast.Sub: lambda a, b: a - b,
        ast.Mult: lambda a, b: a * b,
        ast.Div: lambda a, b: a / b,
    }
    _ALLOWED_UNARYOPS = {
        ast.UAdd: lambda a: a,
        ast.USub: lambda a: -a,
    }

    def validate(self, text, pos):
        result = super(ArithmeticDoubleSpinBox, self).validate(text, pos)
        state = result[0] if isinstance(result, tuple) else result
        if state == QtGui.QValidator.Acceptable:
            return result
        expr = text.strip()
        if expr and all(ch in "0123456789.+-*/() \t" for ch in expr):
            return (QtGui.QValidator.Intermediate, text, pos)
        return result

    def keyPressEvent(self, event):
        if event.key() in (QtCore.Qt.Key_Return, QtCore.Qt.Key_Enter):
            if self._apply_expression_text():
                event.accept()
                self.editingFinished.emit()
                return
        super(ArithmeticDoubleSpinBox, self).keyPressEvent(event)

    def _apply_expression_text(self):
        text = self.lineEdit().text().strip()
        if not any(op in text for op in ("+", "-", "*", "/")):
            return False
        try:
            value = self._eval_expr(text)
        except Exception:
            return False
        value = max(self.minimum(), min(self.maximum(), value))
        self.setValue(value)
        self.lineEdit().setText(self.text())
        return True

    def _eval_expr(self, text):
        tree = ast.parse(text, mode="eval")
        return float(self._eval_node(tree.body))

    def _eval_node(self, node):
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return float(node.value)
        if isinstance(node, ast.Num):
            return float(node.n)
        if isinstance(node, ast.BinOp) and type(node.op) in self._ALLOWED_BINOPS:
            left = self._eval_node(node.left)
            right = self._eval_node(node.right)
            return self._ALLOWED_BINOPS[type(node.op)](left, right)
        if isinstance(node, ast.UnaryOp) and type(node.op) in self._ALLOWED_UNARYOPS:
            return self._ALLOWED_UNARYOPS[type(node.op)](self._eval_node(node.operand))
        raise ValueError("unsupported expression")


class ExportMotionDialog(QtWidgets.QDialog):
    """モーションCSVを表示してコピーするダイアログ"""

    def __init__(self, text, parent=None):
        super(ExportMotionDialog, self).__init__(parent)
        self.setWindowTitle("Export Motion")
        self.setMinimumWidth(760)
        self.setMinimumHeight(520)
        self.setModal(False)

        layout = QtWidgets.QVBoxLayout(self)
        top_layout = QtWidgets.QHBoxLayout()
        top_layout.addStretch()
        copy_button = QtWidgets.QPushButton("Copy")
        copy_button.clicked.connect(self._copy_text)
        top_layout.addWidget(copy_button)
        layout.addLayout(top_layout)

        self.text_edit = QtWidgets.QPlainTextEdit()
        self.text_edit.setPlainText(text)
        self.text_edit.setLineWrapMode(QtWidgets.QPlainTextEdit.NoWrap)
        layout.addWidget(self.text_edit, stretch=1)

    def _copy_text(self):
        QtWidgets.QApplication.clipboard().setText(self.text_edit.toPlainText())


class SingleJointDialog(QtWidgets.QDialog):
    """単体ジョイント編集用モーダルウィンドウ"""

    angle_changed = QtCore.Signal(str, float)  # joint_name, angle_deg

    def __init__(self, joint_name, joint_info, current_angle, parent=None):
        super(SingleJointDialog, self).__init__(parent)
        self.joint_name = joint_name
        self.joint_info = joint_info
        self._updating = False

        self.setWindowTitle(f"Joint: {joint_name}")
        self.setMinimumWidth(300)
        self.setModal(False)  # 非モーダル（3Dビューを操作可能）
        self.setWindowOpacity(0.9)  # 90%不透明（10%透明）

        layout = QtWidgets.QVBoxLayout(self)

        # ジョイント名ラベル
        name_label = QtWidgets.QLabel(f"<b>{joint_name}</b>")
        name_label.setStyleSheet("color: black; font-size: 14px;")
        layout.addWidget(name_label)

        # 角度範囲の表示（degree） - limit_lower/upperは既に度数
        limit_lower = joint_info.limit_lower
        limit_upper = joint_info.limit_upper
        range_label = QtWidgets.QLabel(f"Range: {limit_lower:.2f}° ~ {limit_upper:.2f}°")
        range_label.setStyleSheet("color: gray;")
        layout.addWidget(range_label)

        # スライダー行
        slider_layout = QtWidgets.QHBoxLayout()

        self.slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.slider.setMinimum(int(limit_lower * 100))
        self.slider.setMaximum(int(limit_upper * 100))
        self.slider.setValue(int(current_angle * 100))
        slider_layout.addWidget(self.slider, stretch=1)

        # 数値入力フィールド（degree、小数点2桁）
        self.spinbox = QtWidgets.QDoubleSpinBox()
        self.spinbox.setRange(limit_lower, limit_upper)
        self.spinbox.setDecimals(2)
        self.spinbox.setSuffix("°")
        self.spinbox.setValue(current_angle)
        self.spinbox.setFixedWidth(90)
        slider_layout.addWidget(self.spinbox)

        layout.addLayout(slider_layout)

        # 閉じるボタン
        close_btn = QtWidgets.QPushButton("Close")
        close_btn.clicked.connect(self.close)
        layout.addWidget(close_btn)

        # シグナル接続
        self.slider.valueChanged.connect(self._on_slider_changed)
        self.spinbox.valueChanged.connect(self._on_spinbox_changed)

    def _on_slider_changed(self, val):
        if self._updating:
            return
        self._updating = True
        angle = val / 100.0
        self.spinbox.setValue(angle)
        self.angle_changed.emit(self.joint_name, angle)
        self._updating = False

    def _on_spinbox_changed(self, val):
        if self._updating:
            return
        self._updating = True
        self.slider.setValue(int(val * 100))
        self.angle_changed.emit(self.joint_name, val)
        self._updating = False

    def update_angle(self, angle):
        """外部から角度を更新"""
        self._updating = True
        self.slider.setValue(int(angle * 100))
        self.spinbox.setValue(angle)
        self._updating = False

    def event(self, event):
        """ウィンドウの外をクリックした時に閉じる"""
        if event.type() == QtCore.QEvent.WindowDeactivate:
            self.close()
        return super(SingleJointDialog, self).event(event)


# ==============================================================================
# ColorPicker Classes
# ==============================================================================

class CustomColorDialog(QtWidgets.QColorDialog):
    """カスタムカラーボックスの選択機能を持つカラーダイアログ"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.selected_custom_color_index = 0
        self.custom_color_well_array = None
        self._setup_done = False

    def showEvent(self, event):
        super().showEvent(event)
        if not self._setup_done:
            QtCore.QTimer.singleShot(300, self._setup_custom_color_boxes)

    def _setup_custom_color_boxes(self):
        def find_custom_well_array(widget, depth=0):
            class_name = widget.metaObject().className()
            if class_name == 'QtPrivate::QWellArray':
                size = widget.size()
                if size.height() == 48:
                    self.custom_color_well_array = widget
                    widget.installEventFilter(self)
                    return True
            for child in widget.children():
                if isinstance(child, QtWidgets.QWidget):
                    if find_custom_well_array(child, depth + 1):
                        return True
            return False

        if find_custom_well_array(self):
            self._setup_done = True
            self._draw_selection_border()
            self._setup_add_button()
        else:
            if not self._setup_done:
                QtCore.QTimer.singleShot(500, self._setup_custom_color_boxes)

    def _setup_add_button(self):
        buttons = self.findChildren(QtWidgets.QPushButton)
        for button in buttons:
            if button.text() or True:
                if self.custom_color_well_array:
                    well_array_geo = self.custom_color_well_array.geometry()
                    button_geo = button.geometry()
                    if abs(button_geo.y() - (well_array_geo.y() + well_array_geo.height())) < 50:
                        if button_geo.width() > 100:
                            try:
                                button.clicked.disconnect()
                            except:
                                pass
                            button.clicked.connect(self._add_custom_color)
                            self._add_button = button
                            break

    def eventFilter(self, obj, event):
        if obj == self.custom_color_well_array:
            if event.type() == QtCore.QEvent.MouseButtonPress:
                pos = event.position().toPoint()
                width = self.custom_color_well_array.width()
                height = self.custom_color_well_array.height()
                cell_width = width / 8.0
                cell_height = height / 2.0
                col = int(pos.x() / cell_width)
                row = int(pos.y() / cell_height)
                col = max(0, min(7, col))
                row = max(0, min(1, row))
                index = col * 2 + row
                self.selected_custom_color_index = index
                self._draw_selection_border()
            elif event.type() == QtCore.QEvent.Paint:
                QtCore.QTimer.singleShot(0, self._draw_selection_border)
        return super().eventFilter(obj, event)

    def _add_custom_color(self):
        current_color = self.currentColor()
        QtWidgets.QColorDialog.setCustomColor(self.selected_custom_color_index, current_color)
        if self.custom_color_well_array:
            self.custom_color_well_array.update()
            self.custom_color_well_array.repaint()

    def _draw_selection_border(self):
        if not self.custom_color_well_array:
            return
        width = self.custom_color_well_array.width()
        height = self.custom_color_well_array.height()
        cell_width = width / 8.0
        cell_height = height / 2.0
        col = self.selected_custom_color_index // 2
        row = self.selected_custom_color_index % 2
        BORDER_WIDTH = 2
        x = int(col * cell_width) + BORDER_WIDTH
        y = int(row * cell_height) + BORDER_WIDTH
        next_x = int((col + 1) * cell_width)
        next_y = int((row + 1) * cell_height)
        frame_width = next_x - x - BORDER_WIDTH
        frame_height = next_y - y - BORDER_WIDTH

        if not hasattr(self, '_selection_frame'):
            self._selection_frame = QtWidgets.QFrame(self.custom_color_well_array)
            self._selection_frame.setStyleSheet("border: 3px solid #4080FF; background: transparent;")
            self._selection_frame.setAttribute(QtCore.Qt.WA_TransparentForMouseEvents)

        self._selection_frame.setGeometry(x, y, frame_width, frame_height)
        self._selection_frame.show()
        self._selection_frame.raise_()


class ColorPicker:
    """カラーピッカーウィジェット（RGB入力とプレビュー付き）"""

    def __init__(self, parent_widget, initial_color=None, on_color_changed=None):
        self.parent_widget = parent_widget
        self.on_color_changed = on_color_changed

        if initial_color is None:
            initial_color = [1.0, 1.0, 1.0]
        self.current_color = list(initial_color)[:3]

        self.color_inputs = []
        self.color_sample = None
        self.pick_button = None
        self._create_widgets()

    def _create_widgets(self):
        from PySide6.QtGui import QDoubleValidator
        from PySide6.QtCore import QLocale

        for i in range(3):
            color_input = QtWidgets.QLineEdit()
            color_input.setFixedWidth(50)
            color_input.setText(f"{self.current_color[i]:.2f}")
            validator = QDoubleValidator(0.0, 1.0, 2)
            validator.setLocale(QLocale.c())
            color_input.setValidator(validator)
            color_input.textChanged.connect(self._on_input_changed)
            self.color_inputs.append(color_input)

        self.color_sample = QtWidgets.QLabel()
        self.color_sample.setFixedSize(24, 20)
        self._update_color_sample()

        self.pick_button = QtWidgets.QPushButton("Pick")
        self.pick_button.setFixedWidth(40)
        self.pick_button.setAutoDefault(False)
        self.pick_button.setDefault(False)
        self.pick_button.clicked.connect(self.show_color_picker)

    def add_to_layout(self, layout):
        for color_input in self.color_inputs:
            layout.addWidget(color_input)
        layout.addWidget(self.color_sample)
        layout.addWidget(self.pick_button)

    def _update_color_sample(self):
        try:
            rgb = [max(0, min(255, int(c * 255))) for c in self.current_color]
            self.color_sample.setStyleSheet(
                f"background-color: rgb({rgb[0]},{rgb[1]},{rgb[2]}); border: 1px solid black;"
            )
        except (ValueError, IndexError):
            pass

    def _on_input_changed(self):
        try:
            new_color = [float(inp.text()) for inp in self.color_inputs]
            new_color = [max(0.0, min(1.0, v)) for v in new_color]
            self.current_color = new_color
            self._update_color_sample()
            if self.on_color_changed:
                self.on_color_changed(self.current_color)
        except ValueError:
            pass

    def show_color_picker(self):
        from PySide6.QtGui import QColor
        try:
            current_qcolor = QColor(*[min(255, max(0, int(c * 255))) for c in self.current_color])
        except (ValueError, IndexError):
            current_qcolor = QColor(255, 255, 255)

        dialog = CustomColorDialog(current_qcolor, self.parent_widget)
        dialog.setOption(QtWidgets.QColorDialog.DontUseNativeDialog, True)

        if dialog.exec() == QtWidgets.QDialog.Accepted:
            color = dialog.currentColor()
            if color.isValid():
                new_color = [color.red() / 255.0, color.green() / 255.0, color.blue() / 255.0]
                for i in range(3):
                    self.color_inputs[i].setText(f"{new_color[i]:.2f}")
                self.current_color = new_color
                self._update_color_sample()
                if self.on_color_changed:
                    self.on_color_changed(self.current_color)

    def get_color(self):
        return self.current_color.copy()

    def set_color(self, color):
        self.current_color = list(color)[:3]
        for i in range(3):
            self.color_inputs[i].setText(f"{self.current_color[i]:.2f}")
        self._update_color_sample()

    def set_enabled(self, enabled):
        """ウィジェットの有効/無効を切り替え"""
        for inp in self.color_inputs:
            inp.setEnabled(enabled)
        self.pick_button.setEnabled(enabled)
        if enabled:
            self.color_sample.setStyleSheet(
                f"background-color: rgb({int(self.current_color[0]*255)},{int(self.current_color[1]*255)},{int(self.current_color[2]*255)}); border: 1px solid black;"
            )
        else:
            self.color_sample.setStyleSheet("background-color: gray; border: 1px solid gray;")


# ==============================================================================
# Joint Helper Constants and Functions
# ==============================================================================

# Joint speed presets: (model_name, max_speed_rad_s)
# Conversions: sec/60° → deg/s = 60/(sec/60°);  RPM → deg/s = RPM × 6
JOINT_SPEED_PRESETS = [
    # Kondo KRS series
    ("KRS2552",         math.radians(428.57)),   # 0.14s/60°
    ("KRS2572",         math.radians(428.57)),   # 0.14s/60°
    ("KRS4034",         math.radians(352.94)),   # 0.17s/60°
    # Futaba RS series
    ("RS304MD",         math.radians(375.0)),    # 0.16s/60°
    ("RS303MR",         math.radians(545.45)),   # 0.11s/60°
    ("RS305CR",         math.radians(545.45)),   # 0.11s/60°
    # Tower Pro / generic
    ("SG-90",           math.radians(500.0)),    # 0.12s/60° @4.8V
    # Dynamixel X series
    ("XL330-M288-T",    math.radians(618.0)),    # 103 RPM @5V
    ("XC330-T181",      math.radians(624.0)),    # 104 RPM @11.1V
    ("XD540-T150",      math.radians(420.0)),    # 70 RPM
    ("XD540-T270",      math.radians(234.0)),    # 39 RPM
    ("XM540-W270",      math.radians(180.0)),    # 30 RPM
    # Feetech STS series
    ("STS3215-C046",    math.radians(672.0)),    # ~112 RPM est. (1:147 gear)
    ("STS3215-C044",    math.radians(516.0)),    # 86 RPM (1:191 gear)
    ("STS3032",         math.radians(666.0)),    # 111 RPM
    # RobStride QDD actuators
    ("RobStride-RS00",  math.radians(1890.0)),   # 315 RPM
    ("RobStride-RS01",  math.radians(1890.0)),   # 315 RPM
    ("RobStride-RS02",  math.radians(1890.0)),   # 315 RPM
    ("RobStride-RS03",  math.radians(1170.0)),   # 195 RPM
    ("RobStride-RS04",  math.radians(1200.0)),   # 200 RPM
    ("RobStride-RS05",  math.radians(2880.0)),   # 480 RPM
    ("RobStride-RS06",  math.radians(2880.0)),   # 480 RPM
]
# Joint Settings max speed: stored in rad/s internally
# Default: 300 deg/s = 5.24 rad/s
DEFAULT_JOINT_SPEED = math.radians(300.0)  # 5.24 rad/s
JOINT_DIRECTIONS = ("CW", "CCW")   # kept for migration only
DEFAULT_JOINT_DIRECTION = "CW"      # kept for migration only
DEFAULT_JOINT_REV = False


def joints_matching_right_yaw_roll_mirror(joint_names):
    """r_ で始まり _xr / _zy で終わる関節名（右側 yaw / roll ミラー対象）。"""
    matched = []
    for jname in joint_names:
        jn = jname.lower()
        if jn.startswith("r_") and (jn.endswith("_xr") or jn.endswith("_zy")):
            matched.append(jname)
    return sorted(matched)


def get_joint_speed_presets():
    """Joint Settings preset list: (model_name, rad/s).

    Returns built-in JOINT_SPEED_PRESETS if nothing is saved.
    If a saved list exists, merges it with built-ins so that any new built-in
    models not yet in the saved list are appended (preserving user edits).
    """
    raw = _app_settings().value("joint/speed_presets_json", "")
    if not raw:
        return list(JOINT_SPEED_PRESETS)
    try:
        data = json.loads(str(raw))
        saved = []
        for item in data:
            if isinstance(item, (list, tuple)) and len(item) >= 2:
                saved.append((str(item[0]), float(item[1])))
        if not saved:
            return list(JOINT_SPEED_PRESETS)
        # Append built-in entries whose name is not already in the saved list.
        saved_names = {n for n, _ in saved}
        for name, speed in JOINT_SPEED_PRESETS:
            if name not in saved_names:
                saved.append((name, speed))
        return saved
    except (TypeError, ValueError, json.JSONDecodeError):
        return list(JOINT_SPEED_PRESETS)


def save_joint_speed_presets(pairs):
    """Save (model_name, rad/s) list to QSettings as JSON."""
    data = [[n, float(v)] for n, v in pairs]
    _app_settings().setValue(
        "joint/speed_presets_json", json.dumps(data, ensure_ascii=False))


# Default frame preset values
DEFAULT_FRAME_PRESETS = [10, 20, 50, 100]


def get_frame_presets():
    """Get frame preset values (list of 4 integers)."""
    raw = _app_settings().value("motion/frame_presets_json", "")
    if not raw:
        return list(DEFAULT_FRAME_PRESETS)
    try:
        data = json.loads(str(raw))
        if isinstance(data, list) and len(data) == 4:
            return [int(v) for v in data]
        return list(DEFAULT_FRAME_PRESETS)
    except (TypeError, ValueError, json.JSONDecodeError):
        return list(DEFAULT_FRAME_PRESETS)


def save_frame_presets(values):
    """Save frame preset values (list of 4 integers)."""
    data = [int(v) for v in values[:4]]
    while len(data) < 4:
        data.append(DEFAULT_FRAME_PRESETS[len(data)])
    _app_settings().setValue(
        "motion/frame_presets_json", json.dumps(data))


def _joint_preset_item_data_parts(data):
    """Extract (model_name, rad/s) from QComboBox itemData (UserRole).
    Handles both list and tuple from PySide QVariant.
    """
    if data is None:
        return None, None
    if isinstance(data, (list, tuple)) and len(data) >= 2:
        try:
            return str(data[0]), float(data[1])
        except (TypeError, ValueError):
            return None, None
    try:
        return None, float(data)
    except (TypeError, ValueError):
        return None, None


# ==============================================================================
# AxisPad2D - 2D joystick-like input widget for Lx/Ly or Rx/Ry
# ==============================================================================

class AxisPad2D(QtWidgets.QWidget):
    """2D pad widget for analog stick input (X: -127 to 127, Y: -127 to 127)."""

    value_changed = QtCore.Signal(int, int)  # (x, y)

    _SPRING_DECAY = 0.65   # per tick (16 ms) → ~8 ticks / ~128 ms to settle
    _SPRING_INTERVAL = 16  # ms

    def __init__(self, label="L", parent=None, size=76):
        super(AxisPad2D, self).__init__(parent)
        self._label = label
        self._x = 0
        self._y = 0
        self._dragging = False
        self._enabled = True
        self.setFixedSize(size, size)
        self.setAttribute(QtCore.Qt.WA_TranslucentBackground)
        self.setStyleSheet("background: transparent;")
        self.setCursor(QtCore.Qt.CrossCursor)
        self._spring_timer = QtCore.QTimer(self)
        self._spring_timer.setInterval(self._SPRING_INTERVAL)
        self._spring_timer.timeout.connect(self._spring_step)

    def set_values(self, x, y):
        """Set x/y values programmatically (e.g., from PC pad polling)."""
        self._x = max(-127, min(127, x))
        self._y = max(-127, min(127, y))
        self.update()

    def get_values(self):
        return self._x, self._y

    def set_enabled(self, enabled):
        self._enabled = enabled
        self.setCursor(QtCore.Qt.CrossCursor if enabled else QtCore.Qt.ArrowCursor)
        self.update()

    def paintEvent(self, event):
        from PySide6 import QtGui
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.Antialiasing)

        w, h = self.width(), self.height()
        cx, cy = w // 2, h // 2
        radius = min(cx, cy) - 4

        # Draw outer circle (background)
        bg_color = QtGui.QColor("#1C222C") if self._enabled else QtGui.QColor("#14171C")
        painter.setBrush(bg_color)
        painter.setPen(QtGui.QPen(QtGui.QColor("#3A4454"), 1.5))
        painter.drawEllipse(cx - radius, cy - radius, radius * 2, radius * 2)

        # Draw inner well
        inner = radius - 7
        painter.setBrush(QtGui.QColor("#12151A"))
        painter.setPen(QtCore.Qt.NoPen)
        painter.drawEllipse(cx - inner, cy - inner, inner * 2, inner * 2)

        # Draw crosshairs
        painter.setPen(QtGui.QPen(QtGui.QColor("#2A3140"), 1))
        painter.drawLine(cx - inner, cy, cx + inner, cy)
        painter.drawLine(cx, cy - inner, cx, cy + inner)

        # Calculate knob position (+Y = up)
        knob_r = 6
        knob_x = cx + int(self._x / 127.0 * (radius - 7))
        knob_y = cy - int(self._y / 127.0 * (radius - 7))

        # Draw knob
        knob_color = QtGui.QColor("#5BA8C9") if self._enabled else QtGui.QColor("#3A4558")
        painter.setBrush(knob_color)
        painter.setPen(QtGui.QPen(QtGui.QColor("#8EC8DC") if self._enabled else QtGui.QColor("#454B56"), 1.5))
        painter.drawEllipse(knob_x - knob_r, knob_y - knob_r, knob_r * 2, knob_r * 2)

        painter.end()

    def mousePressEvent(self, event):
        if self._enabled and event.button() == QtCore.Qt.LeftButton:
            self._spring_timer.stop()
            self._dragging = True
            self._update_from_mouse(event.pos())

    def mouseMoveEvent(self, event):
        if self._dragging and self._enabled:
            self._update_from_mouse(event.pos())

    def mouseReleaseEvent(self, event):
        if event.button() == QtCore.Qt.LeftButton:
            self._dragging = False
            if self._x != 0 or self._y != 0:
                self._spring_timer.start()

    def _spring_step(self):
        """指数減衰でレバーをセンターへ戻す (PS4 バネ感)."""
        self._x = int(self._x * self._SPRING_DECAY)
        self._y = int(self._y * self._SPRING_DECAY)
        if abs(self._x) < 2 and abs(self._y) < 2:
            self._x = 0
            self._y = 0
            self._spring_timer.stop()
        self.update()
        self.value_changed.emit(self._x, self._y)

    def _update_from_mouse(self, pos):
        w, h = self.width(), self.height()
        cx, cy = w // 2, h // 2
        radius = min(cx, cy) - 4

        # Convert mouse position to -127..127 range (+Y = up)
        dx = pos.x() - cx
        dy = cy - pos.y()

        # Clamp to circle
        dist = (dx * dx + dy * dy) ** 0.5
        max_dist = radius - 8
        if dist > max_dist and dist > 0:
            dx = dx * max_dist / dist
            dy = dy * max_dist / dist

        self._x = int(dx / max_dist * 127)
        self._y = int(dy / max_dist * 127)
        self._x = max(-127, min(127, self._x))
        self._y = max(-127, min(127, self._y))

        self.update()
        self.value_changed.emit(self._x, self._y)


# ==============================================================================
# PadMonitorDialog - PS3-compatible gamepad input monitor
# ==============================================================================


class _RescanningComboBox(QtWidgets.QComboBox):
    """showPopup() 直前に外部コールバックで再スキャン → 項目更新するコンボ。
    再スキャン後もアイテムが「placeholder ひとつだけ」(UserRole が None) の
    場合は、popup を出さずに reopen_callback を呼ぶ (ウィンドウ閉じ開きで
    ネイティブレベルの再列挙をトリガーするため)。"""

    def __init__(self, rescan_callback, reopen_callback=None, parent=None):
        super().__init__(parent)
        self._rescan_callback = rescan_callback
        self._reopen_callback = reopen_callback

    def showPopup(self):
        try:
            self._rescan_callback()
        except Exception:
            pass
        # 再スキャン後も「No controller」placeholder しかなければ、popup は出さず
        # 呼び出し元 (Pad ウィンドウ) を閉じて開き直す。
        if (self._reopen_callback is not None
                and self.count() == 1
                and self.itemData(0, QtCore.Qt.UserRole) is None):
            try:
                self._reopen_callback()
            except Exception:
                pass
            return
        super().showPopup()


class PadMonitorDialog(QtWidgets.QDialog):
    """PS3準拠のボタン入力値を確認するモニタ."""

    use_pc_pad_changed = QtCore.Signal(bool)
    play_requested = QtCore.Signal()
    stop_requested = QtCore.Signal()
    home_requested = QtCore.Signal()
    zero_requested = QtCore.Signal()
    open_mujoco_requested = QtCore.Signal()
    respawn_requested = QtCore.Signal()

    BUTTON_LAYOUT = [
        ("L1", 42, 32, 1024),
        ("L2", 42, 6, 256),
        ("DPad Up", 80, 58, 16),
        ("DPad Left", 50, 84, 128),
        ("DPad Right", 110, 84, 32),
        ("DPad Down", 80, 110, 64),
        ("Select", 160, 84, 1),
        ("Start", 206, 84, 8),
        ("Triangle", 290, 58, 4096),
        ("Square", 260, 84, 32768),
        ("Circle", 320, 84, 8192),
        ("Cross", 290, 110, 16384),
        ("R1", 328, 32, 2048),
        ("R2", 328, 6, 512),
    ]
    PAD_REFERENCE_SIZE = (396, 148)
    BUTTON_SIZE = 24
    BACKGROUND_COLOR = "#12151A"
    COLOR_TEXT = "#E6EAF0"
    COLOR_MUTED = "#8B95A5"
    COLOR_ACCENT = "#5BA8C9"
    COLOR_ACCENT_HI = "#8EC8DC"
    COLOR_BTN = "#2A3140"
    COLOR_BTN_HOVER = "#3A4558"
    COLOR_PAD = "#252B36"
    COLOR_PAD_BORDER = "#3A4454"
    AXIS_NAMES = ("Lx", "Ly", "Rx", "Ry", "L2v", "R2v")
    _FACE_BUTTONS = frozenset({"Triangle", "Square", "Circle", "Cross"})

    @classmethod
    def _theme_qss(cls):
        bg, text = cls.BACKGROUND_COLOR, cls.COLOR_TEXT
        accent, hi = cls.COLOR_ACCENT, cls.COLOR_ACCENT_HI
        btn, hover, pad_b = cls.COLOR_BTN, cls.COLOR_BTN_HOVER, cls.COLOR_PAD_BORDER
        return f"""
            QDialog {{ background-color: {bg}; }}
            QLabel {{ color: {text}; background: transparent; }}
            QSpinBox {{
                color: {text}; background-color: {btn};
                border: 1px solid {pad_b}; border-radius: 4px;
                padding: 0px 2px; font-size: 11px;
                selection-background-color: {accent};
            }}
            QSlider::groove:horizontal {{
                height: 3px; background: #1C222C; border-radius: 2px;
            }}
            QSlider::sub-page:horizontal {{
                background: {accent}; border-radius: 2px;
            }}
            QSlider::handle:horizontal {{
                width: 12px; height: 12px; margin: -5px 0;
                background: {hi}; border: none; border-radius: 6px;
            }}
            QCheckBox#PadOption {{
                color: {text}; spacing: 6px; font-size: 11px;
            }}
            QCheckBox#PadOption::indicator {{
                width: 12px; height: 12px; border-radius: 3px;
                background: {btn}; border: 1px solid {pad_b};
            }}
            QCheckBox#PadOption::indicator:checked {{
                background: {accent}; border: 1px solid {hi};
            }}
            QPushButton#PadAction {{
                color: {text}; background-color: {btn};
                border: 1px solid {pad_b}; border-radius: 6px;
                padding: 2px 8px; font-size: 11px; font-weight: 600;
            }}
            QPushButton#PadAction:hover {{ background-color: {hover}; border-color: {accent}; }}
            QPushButton#PadAction:pressed {{ background-color: #1A1E26; }}
            QPushButton#PadAction:disabled {{
                color: #6A7380; background-color: #1A1E26;
                border: 1px solid #2A3140;
            }}
            QPushButton#PadActionPlaying {{
                color: #12151A; background-color: {hi};
                border: 1px solid {text}; border-radius: 6px;
                padding: 2px 8px; font-size: 11px; font-weight: 600;
            }}
            QPushButton#PadActionPlaying:hover {{ background-color: #A8D4E4; }}
            QToolButton#PadIconAction {{
                color: {text}; background-color: {btn};
                border: 1px solid {pad_b}; border-radius: 6px;
                padding: 0px; font-size: 11px; font-weight: 600;
            }}
            QToolButton#PadIconAction:hover {{ background-color: {hover}; border-color: {accent}; }}
            QToolButton#PadIconAction:pressed {{ background-color: #1A1E26; }}
            QToolButton#PadIconActionPlaying {{
                color: #12151A; background-color: {hi};
                border: 1px solid {text}; border-radius: 6px;
                padding: 0px; font-size: 11px; font-weight: 600;
            }}
            QToolButton#PadIconActionPlaying:hover {{ background-color: #A8D4E4; }}
        """

    def __init__(self, parent=None):
        super(PadMonitorDialog, self).__init__(parent)
        self.setWindowTitle("Pad Button Input")
        self.setModal(False)
        # Don't steal app activation (would raise LME main over MuJoCoStudio).
        self.setAttribute(QtCore.Qt.WA_ShowWithoutActivating, True)

        self._pygame = None
        self._sdl_controller = None  # pygame._sdl2.controller.Controller（あれば）
        self._sdl_mod = None
        self._raw_joystick = None  # Game Controller 非対応時の生ジョイスティック
        self._pad_layout = "ps3"  # "ps3" | "ps4" — 生ジョイスティック時のボタン割当
        self._hotplug_ticks = 0
        self._darwin_display_hack_applied = False
        self._pygame_error = ""
        # macOS: pygame/SDL が 0 台でも Apple GameController が拾えることがある（Bluetooth DUALSHOCK 等）
        self._gc_controller_obj = None
        self._gc_extended_pad = None
        self._gc_discovery_started = False
        self._gc_backend_error = ""
        self._gc_unavailable = False
        self._gc_monitor_ready = False
        self._gc_value_handler = None
        self._gc_handler_pad = None
        self._gc_connect_observer = None
        self._gc_disconnect_observer = None
        self._gc_connect_block = None
        self._gc_disconnect_block = None
        self._button_widgets = {}
        self._button_positions = {}
        self._button_bits = {}
        self._axis_sliders = {}
        self._axis_inputs = {}
        self._axis_pads = {}  # AxisPad2D widgets for L and R sticks
        self._axis_labels = {}  # Labels for Lx/Ly/Rx/Ry values
        self._updating_axes = False
        # Controller プルダウンで指定された joystick インデックス。
        # ユーザが Controller1..N から選ぶと _switch_to_controller で更新される。
        self._preferred_joystick_index = 0
        self._suppress_controller_combo = False  # populate 中の re-entrance ガード
        settings = load_app_settings()
        self.always_on_top = bool(settings.get("pad_monitor_always_on_top", True))

        self._poll_timer = QtCore.QTimer(self)
        self._poll_timer.setInterval(33)
        self._poll_timer.timeout.connect(self._poll_pc_pad)

        self._setup_ui()
        self._mujoco_running_checker = None
        self._mujoco_btn_timer = QtCore.QTimer(self)
        self._mujoco_btn_timer.setInterval(400)
        self._mujoco_btn_timer.timeout.connect(self._refresh_open_mujoco_btn)
        self._mujoco_btn_timer.start()
        self._apply_always_on_top(restore_visible=False)
        self._fit_pad_window()

    def _setup_ui(self):
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(10, 8, 10, 10)
        layout.setSpacing(8)
        layout.setSizeConstraint(QtWidgets.QLayout.SetMinimumSize)
        self.setStyleSheet(self._theme_qss())

        # Top: L2v / R2v only — pad actions live in the footer
        trigger_row = QtWidgets.QHBoxLayout()
        trigger_row.setSpacing(6)

        l2v_layout = QtWidgets.QHBoxLayout()
        l2v_layout.setSpacing(4)
        l2v_label = create_label("L2v")
        l2v_label.setStyleSheet(f"color: {self.COLOR_MUTED}; font-size: 11px; font-weight: 600;")
        l2v_label.setFixedWidth(26)
        l2v_input = QtWidgets.QSpinBox()
        l2v_input.setAlignment(QtCore.Qt.AlignCenter)
        l2v_input.setButtonSymbols(QtWidgets.QAbstractSpinBox.NoButtons)
        l2v_input.setFixedWidth(38)
        l2v_input.setRange(0, 255)
        l2v_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        l2v_slider.setRange(0, 255)
        l2v_slider.setValue(0)
        l2v_slider.setFixedWidth(72)
        l2v_slider.valueChanged.connect(lambda value: self._on_axis_slider_changed("L2v", value))
        l2v_slider.sliderReleased.connect(lambda: self._on_trigger_slider_released("L2v"))
        l2v_input.valueChanged.connect(lambda value: self._on_axis_input_changed("L2v", value))
        l2v_layout.addWidget(l2v_label)
        l2v_layout.addWidget(l2v_input)
        l2v_layout.addWidget(l2v_slider)
        self._axis_inputs["L2v"] = l2v_input
        self._axis_sliders["L2v"] = l2v_slider
        trigger_row.addLayout(l2v_layout)
        trigger_row.addStretch()
        self._open_mujoco_btn = QtWidgets.QPushButton("Open MuJoCo")
        self._open_mujoco_btn.setObjectName("PadAction")
        self._open_mujoco_btn.setToolTip("Launch MuJoCo Studio (raises the window if already open)")
        self._open_mujoco_btn.setFixedHeight(22)
        self._open_mujoco_btn.clicked.connect(self.open_mujoco_requested.emit)
        trigger_row.addWidget(self._open_mujoco_btn)
        trigger_row.addStretch()

        r2v_layout = QtWidgets.QHBoxLayout()
        r2v_layout.setSpacing(4)
        r2v_label = create_label("R2v")
        r2v_label.setStyleSheet(f"color: {self.COLOR_MUTED}; font-size: 11px; font-weight: 600;")
        r2v_label.setFixedWidth(26)
        r2v_input = QtWidgets.QSpinBox()
        r2v_input.setAlignment(QtCore.Qt.AlignCenter)
        r2v_input.setButtonSymbols(QtWidgets.QAbstractSpinBox.NoButtons)
        r2v_input.setFixedWidth(38)
        r2v_input.setRange(0, 255)
        r2v_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        r2v_slider.setRange(0, 255)
        r2v_slider.setValue(0)
        r2v_slider.setFixedWidth(72)
        r2v_slider.valueChanged.connect(lambda value: self._on_axis_slider_changed("R2v", value))
        r2v_slider.sliderReleased.connect(lambda: self._on_trigger_slider_released("R2v"))
        r2v_input.valueChanged.connect(lambda value: self._on_axis_input_changed("R2v", value))
        r2v_layout.addWidget(r2v_label)
        r2v_layout.addWidget(r2v_input)
        r2v_layout.addWidget(r2v_slider)
        self._axis_inputs["R2v"] = r2v_input
        self._axis_sliders["R2v"] = r2v_slider
        trigger_row.addLayout(r2v_layout)
        layout.addLayout(trigger_row)

        pad_area = QtWidgets.QFrame()
        self._pad_area = pad_area
        pad_area.setFixedHeight(self.PAD_REFERENCE_SIZE[1])
        pad_area.setStyleSheet(
            "QFrame { background-color: #1C222C; border: 1px solid #3A4454; border-radius: 10px; }"
        )
        layout.addWidget(pad_area)

        self.value_decimal_label = create_label("0")
        self.value_decimal_label.setParent(pad_area)
        self.value_decimal_label.setAlignment(QtCore.Qt.AlignCenter)
        self.value_decimal_label.setStyleSheet(
            f"color: {self.COLOR_TEXT}; font-size: 16px; font-weight: 600;"
        )

        self.value_binary_label = create_label("0000 0000 0000 0000")
        self.value_binary_label.setParent(pad_area)
        self.value_binary_label.setAlignment(QtCore.Qt.AlignCenter)
        self.value_binary_label.setStyleSheet(
            f"color: {self.COLOR_MUTED}; font-size: 10px; font-family: Menlo, Monaco, Consolas, monospace;"
        )

        for name, x, y, bit in self.BUTTON_LAYOUT:
            check = QtWidgets.QCheckBox(pad_area)
            check.setObjectName("PadKey")
            check.setToolTip(name)
            check.setText("")
            check.setFixedSize(self.BUTTON_SIZE, self.BUTTON_SIZE)
            check.toggled.connect(self._update_value_labels)
            self._button_widgets[name] = check
            self._button_positions[name] = (x, y)
            self._button_bits[name] = bit

        # アクティブなリモコン (pygame joystick) を Controller1..N として選択するプルダウン。
        # pad_area の絶対座標 (reference 系) で配置し _layout_pad_buttons でスケール。
        # No controller placeholder のときにクリックすると _reopen_for_rescan が
        # ウィンドウを閉じ開きして pygame の joystick サブシステムを再列挙する。
        self._controller_combo = _RescanningComboBox(
            self._populate_controller_combo,
            reopen_callback=self._reopen_for_rescan,
            parent=pad_area,
        )
        self._controller_combo.setObjectName("PadControllerCombo")
        self._controller_combo.setToolTip("Active controller")
        # macOS ネイティブメニューは QSS を無視するため、QListView に差し替えて
        # Qt 描画の popup にする (テキストの見切れとテーマ崩れの両方を防ぐ)。
        _combo_view = QtWidgets.QListView(self._controller_combo)
        _combo_view.setUniformItemSizes(True)
        # QListView 自身の白枠を消す (macOS で目立つ)
        _combo_view.setFrameShape(QtWidgets.QFrame.NoFrame)
        self._controller_combo.setView(_combo_view)
        # popup コンテナ (QComboBoxPrivateContainer) の白枠を消す:
        # frame を消して背景を pad テーマに合わせる。setView 直後は view.parent()
        # がその container を返す。
        _popup_container = _combo_view.parent()
        if isinstance(_popup_container, QtWidgets.QFrame):
            _popup_container.setFrameShape(QtWidgets.QFrame.NoFrame)
            _popup_container.setStyleSheet(
                f"background-color: {self.COLOR_PAD};"
                f" border: 1px solid {self.COLOR_PAD_BORDER};"
            )
        # popup の最小幅を確保 (Controller10+ でも切れないように)
        self._controller_combo.view().setMinimumWidth(140)
        self._controller_combo.setSizeAdjustPolicy(
            QtWidgets.QComboBox.AdjustToMinimumContentsLengthWithIcon
        )
        self._controller_combo.setStyleSheet(
            f"QComboBox {{ color: {self.COLOR_TEXT}; background-color: {self.COLOR_BTN};"
            f" border: 1px solid {self.COLOR_PAD_BORDER}; border-radius: 4px;"
            f" padding: 2px 14px 2px 6px; font-size: 11px; }}"
            f"QComboBox:hover {{ border-color: {self.COLOR_ACCENT}; }}"
            # drop-down: 区切り線を出さず、幅も最小限にしてテキスト領域を広く取る
            f"QComboBox::drop-down {{ subcontrol-origin: padding;"
            f" subcontrol-position: top right; width: 14px;"
            f" border: none; background: transparent; }}"
            f"QComboBox QAbstractItemView {{ color: {self.COLOR_TEXT};"
            f" background-color: {self.COLOR_PAD};"
            f" border: 1px solid {self.COLOR_PAD_BORDER};"
            f" selection-background-color: {self.COLOR_ACCENT};"
            f" selection-color: {self.COLOR_TEXT};"
            f" padding: 2px; outline: 0; }}"
            f"QComboBox QAbstractItemView::item {{ min-height: 20px; padding: 2px 6px; }}"
        )
        # activated: ユーザが明示的に項目を選んだときのみ発火 (同一項目の再選択でも発火する)。
        # currentIndexChanged だと同じ Controller を再度クリックしても走らないため、
        # 「Via PC OFF のとき Controller1 を再選択して有効化」の要件を満たせない。
        # 反対に activated は _populate_controller_combo 内の setCurrentIndex では発火しない
        # ので、_suppress_controller_combo ガードなしでも安全。
        self._controller_combo.activated.connect(
            self._on_controller_combo_changed
        )
        QtCore.QTimer.singleShot(0, self._layout_pad_buttons)

        # Sticks + play/stop. Axis numbers sit outside each stick.
        # Play/Stop are placed by geometry (not VBox) so macOS native
        # QPushButton chrome cannot collapse them on top of each other.
        _play_w, _play_h, _play_gap = 36, 28, 14
        _play_col_h = _play_h * 2 + _play_gap
        axis_host = QtWidgets.QWidget()
        axis_host.setFixedHeight(max(76, _play_col_h))
        axis_row = QtWidgets.QHBoxLayout(axis_host)
        axis_row.setContentsMargins(0, 0, 0, 0)
        axis_row.setSpacing(6)
        layout.addWidget(axis_host)
        axis_row.addStretch()

        left_caption = create_label("Lx 0\nLy 0")
        left_caption.setStyleSheet(f"color: {self.COLOR_MUTED}; font-size: 10px;")
        left_caption.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
        left_caption.setFixedWidth(48)
        self._axis_labels["L"] = left_caption
        axis_row.addWidget(left_caption, 0, QtCore.Qt.AlignVCenter)

        left_pad = AxisPad2D("L", self, size=76)
        left_pad.value_changed.connect(self._on_left_pad_changed)
        self._axis_pads["L"] = left_pad
        axis_row.addWidget(left_pad)

        playback_wrap = QtWidgets.QWidget()
        playback_wrap.setFixedSize(_play_w, _play_col_h)
        self._play_btn = QtWidgets.QToolButton(playback_wrap)
        # 3Dビュー下の ▶︎_ (play_full_btn) と同じラベル/機能 (Boot から再生)。
        # 接続は main() の _wire_pad_playback_buttons 経由で on_play_full に張られる。
        self._play_btn.setText("▶︎_")
        self._play_btn.setObjectName("PadIconAction")
        self._play_btn.setToolTip("Play from Boot Action (same as 3D view ▶︎_)")
        self._play_btn.setAutoRaise(False)
        self._play_btn.setFocusPolicy(QtCore.Qt.NoFocus)
        self._play_btn.setGeometry(0, 0, _play_w, _play_h)
        self._play_btn.clicked.connect(self.play_requested.emit)
        self._play_active = False
        self._stop_btn = QtWidgets.QToolButton(playback_wrap)
        self._stop_btn.setText("■")
        self._stop_btn.setObjectName("PadIconAction")
        self._stop_btn.setToolTip("Stop playback")
        self._stop_btn.setAutoRaise(False)
        self._stop_btn.setFocusPolicy(QtCore.Qt.NoFocus)
        self._stop_btn.setGeometry(0, _play_h + _play_gap, _play_w, _play_h)
        self._stop_btn.clicked.connect(self.stop_requested.emit)
        axis_row.addWidget(playback_wrap, 0, QtCore.Qt.AlignVCenter)

        right_pad = AxisPad2D("R", self, size=76)
        right_pad.value_changed.connect(self._on_right_pad_changed)
        self._axis_pads["R"] = right_pad
        axis_row.addWidget(right_pad)

        right_caption = create_label("Rx 0\nRy 0")
        right_caption.setStyleSheet(f"color: {self.COLOR_MUTED}; font-size: 10px;")
        right_caption.setAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter)
        right_caption.setFixedWidth(48)
        self._axis_labels["R"] = right_caption
        axis_row.addWidget(right_caption, 0, QtCore.Qt.AlignVCenter)
        axis_row.addStretch()

        # Home / Zero + options on one footer row (not on the pad face)
        use_pad_row = QtWidgets.QHBoxLayout()
        use_pad_row.setSpacing(6)
        self._home_btn = QtWidgets.QPushButton("Home")
        self._home_btn.setObjectName("PadAction")
        self._home_btn.setToolTip("Send Home position")
        self._home_btn.setFixedHeight(22)
        self._home_btn.setMinimumWidth(52)
        self._home_btn.clicked.connect(self.home_requested.emit)
        self._zero_btn = QtWidgets.QPushButton("Zero")
        self._zero_btn.setObjectName("PadAction")
        self._zero_btn.setToolTip("Send Zero position")
        self._zero_btn.setFixedHeight(22)
        self._zero_btn.setMinimumWidth(52)
        self._zero_btn.clicked.connect(self.zero_requested.emit)
        self._respawn_btn = QtWidgets.QPushButton("Respawn")
        self._respawn_btn.setObjectName("PadAction")
        self._respawn_btn.setToolTip("Respawn model in MuJoCo Studio (same as R key)")
        self._respawn_btn.setFixedHeight(22)
        self._respawn_btn.setMinimumWidth(64)
        self._respawn_btn.clicked.connect(self.respawn_requested.emit)
        use_pad_row.addWidget(self._home_btn)
        use_pad_row.addWidget(self._zero_btn)
        use_pad_row.addWidget(self._respawn_btn)
        use_pad_row.addStretch()
        self.use_pc_pad_checkbox = QtWidgets.QCheckBox("Use Pad via PC")
        self.use_pc_pad_checkbox.setObjectName("PadOption")
        self.use_pc_pad_checkbox.toggled.connect(self._on_use_pc_pad_toggled)
        use_pad_row.addWidget(self.use_pc_pad_checkbox)
        use_pad_row.addSpacing(12)
        self.always_on_top_checkbox = QtWidgets.QCheckBox("Always on Top")
        self.always_on_top_checkbox.setObjectName("PadOption")
        self.always_on_top_checkbox.setChecked(self.always_on_top)
        self.always_on_top_checkbox.toggled.connect(self._on_always_on_top_toggled)
        use_pad_row.addWidget(self.always_on_top_checkbox)
        layout.addLayout(use_pad_row)

        # Status line always reserved so Connected text never overlaps footer buttons
        self._pad_status_label = QtWidgets.QLabel()
        self._pad_status_label.setWordWrap(False)
        self._pad_status_label.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
        self._pad_status_label.setStyleSheet(f"color: {self.COLOR_MUTED}; font-size: 11px;")
        self._pad_status_label.setFixedHeight(18)
        self._pad_status_label.setMinimumWidth(0)
        self._pad_status_label.setSizePolicy(
            QtWidgets.QSizePolicy.Ignored, QtWidgets.QSizePolicy.Fixed)
        layout.addWidget(self._pad_status_label)
        self._refresh_pad_status_short()
        self.setStyleSheet(self._theme_qss())

    def _fit_pad_window(self):
        """setWindowFlags がサイズを潰すので、中身の高さに合わせて掛け直す。"""
        lay = self.layout()
        if lay is None:
            return
        lay.activate()
        hint = lay.sizeHint()
        w = max(416, int(hint.width()))
        h = max(348, int(hint.height()))
        self.setMinimumSize(w, h)
        self.resize(w, h)

    def _pad_key_style(self, name, indicator_size):
        if name in self._FACE_BUTTONS:
            radius = indicator_size // 2
        elif name in ("L1", "L2", "R1", "R2"):
            radius = max(6, indicator_size // 4)
        else:
            radius = max(6, indicator_size // 5)
        pad, border = self.COLOR_PAD, self.COLOR_PAD_BORDER
        accent, hi = self.COLOR_ACCENT, self.COLOR_ACCENT_HI
        return (
            f"QCheckBox::indicator {{ width: {indicator_size}px; height: {indicator_size}px; "
            f"border-radius: {radius}px; }}"
            f"QCheckBox::indicator:unchecked {{ background-color: {pad}; border: 1px solid {border}; }}"
            f"QCheckBox::indicator:checked {{ background-color: {accent}; border: 1px solid {hi}; }}"
            f"QCheckBox::indicator:hover {{ border: 1px solid {accent}; }}"
        )

    def _layout_pad_buttons(self):
        ref_w, ref_h = self.PAD_REFERENCE_SIZE
        area_w = max(1, self._pad_area.width())
        area_h = max(1, self._pad_area.height())
        scale = min(area_w / ref_w, area_h / ref_h)
        offset_x = (area_w - ref_w * scale) / 2.0
        offset_y = (area_h - ref_h * scale) / 2.0
        button_size = max(20, int(round(self.BUTTON_SIZE * scale)))

        indicator_size = max(16, button_size - 2)
        for name, widget in self._button_widgets.items():
            x, y = self._button_positions[name]
            widget.setFixedSize(button_size, button_size)
            widget.setStyleSheet(self._pad_key_style(name, indicator_size))
            widget.move(int(round(offset_x + x * scale)), int(round(offset_y + y * scale)))

        # Fixed readable type: do not scale text boxes with the pad.
        value_w = 148
        value_x = int(round((area_w - value_w) / 2.0))
        y_num = int(round(offset_y + 18 * scale))
        self.value_decimal_label.setGeometry(value_x, y_num, value_w, 20)
        self.value_binary_label.setGeometry(value_x, y_num + 23, value_w, 16)

        # Controller プルダウン: face ボタン下・スティック直前の中央帯 (reference 座標)
        # DPad Down 右端 (104) と Cross 左端 (290) の間に「Controller1 ▼」がちょうど収まる幅で中央配置
        combo = getattr(self, "_controller_combo", None)
        if combo is not None:
            combo_ref_w, combo_ref_h = 116, 20
            combo_ref_x = int(round((ref_w - combo_ref_w) / 2.0))
            combo_ref_y = 120
            combo.setGeometry(
                int(round(offset_x + combo_ref_x * scale)),
                int(round(offset_y + combo_ref_y * scale)),
                max(96, int(round(combo_ref_w * scale))),
                max(16, int(round(combo_ref_h * scale))),
            )

    def _scan_controllers_pygame(self):
        """pygame が現在検出しているジョイスティックを [(idx, name), ...] で返す。
        pygame 未初期化なら空リスト。現在使用中の index は既存ハンドルの名前を
        再利用し、他所で握られているハンドルを壊さない。"""
        try:
            if not self._ensure_pygame():
                return []
        except Exception:
            return []
        pg = self._pygame
        result = []

        # 現在アクティブな index と保持済み名前を先に控えておく
        active_indices = {}  # idx -> name
        if self._raw_joystick is not None:
            try:
                active_indices[int(self._raw_joystick.get_id())] = (
                    self._raw_joystick.get_name() or ""
                )
            except Exception:
                pass
        if self._sdl_controller is not None:
            try:
                joy_wrap = self._sdl_controller.as_joystick()
                active_indices.setdefault(
                    int(joy_wrap.get_id()), joy_wrap.get_name() or ""
                )
            except Exception:
                pass

        try:
            n = pg.joystick.get_count()
            # Windows 対策: 1 台の物理パッドが XInput / DirectInput / HIDAPI で
            # 重複列挙されるケースを dedupe。GUID が返らないドライバがあるので
            # GUID → name → 順序 の順に fallback。GUID が空文字列だと安全側に
            # 倒して残す (別デバイスの可能性)。
            seen_guids = set()
            seen_names = set()
            for i in range(n):
                if i in active_indices:
                    name = active_indices[i] or f"Joystick {i}"
                    result.append((i, name))
                    seen_names.add(name.lower())
                    continue
                name = ""
                guid = ""
                # SDL Controller の方が正規化名を返しやすいので先に試す
                if self._sdl_mod is not None:
                    try:
                        if self._sdl_mod.is_controller(i):
                            c = self._sdl_mod.Controller(i)
                            try:
                                jw = c.as_joystick()
                                name = jw.get_name() or ""
                                try:
                                    guid = jw.get_guid() or ""
                                except Exception:
                                    guid = ""
                            finally:
                                try:
                                    c.quit()
                                except Exception:
                                    pass
                    except Exception:
                        pass
                if not name:
                    try:
                        joy = pg.joystick.Joystick(i)
                        joy.init()
                        try:
                            name = joy.get_name() or ""
                            try:
                                guid = joy.get_guid() or ""
                            except Exception:
                                guid = ""
                        finally:
                            try:
                                joy.quit()
                            except Exception:
                                pass
                    except Exception:
                        name = ""
                # 同じ GUID (0 以外) or 同じ name が既に result に居るなら Windows の
                # 重複列挙とみなしスキップ
                if guid and guid not in ("00000000000000000000000000000000",):
                    if guid in seen_guids:
                        continue
                    seen_guids.add(guid)
                nm_key = (name or "").lower()
                if nm_key and nm_key in seen_names:
                    continue
                if nm_key:
                    seen_names.add(nm_key)
                result.append((i, name or f"Joystick {i}"))
        except Exception:
            return result
        return result

    def _populate_controller_combo(self):
        """プルダウンを最新のジョイスティック一覧で更新する。表示は
        Controller1..N、tooltip に実名。現在使用中の index を選択状態にする。"""
        combo = getattr(self, "_controller_combo", None)
        if combo is None:
            return
        entries = self._scan_controllers_pygame()
        self._suppress_controller_combo = True
        try:
            combo.blockSignals(True)
            combo.clear()
            if not entries:
                # placeholder を enabled のまま置いておく。ユーザがクリックすると
                # showPopup → _reopen_for_rescan (ウィンドウ閉じ開き + pygame 再列挙)
                combo.addItem("No controller")
                combo.setEnabled(True)
                combo.setToolTip("Click to rescan (window will briefly reopen)")
            else:
                combo.setEnabled(True)
                names = []
                for order, (idx, name) in enumerate(entries):
                    combo.addItem(f"Controller{order + 1}")
                    combo.setItemData(order, idx, QtCore.Qt.UserRole)
                    combo.setItemData(order, name or f"Joystick {idx}", QtCore.Qt.ToolTipRole)
                    names.append(f"Controller{order + 1}: {name}")
                combo.setToolTip("\n".join(names))
                # 現在使用中の index を選択
                cur_idx = self._preferred_joystick_index
                sel_order = 0
                for order, (idx, _n) in enumerate(entries):
                    if idx == cur_idx:
                        sel_order = order
                        break
                combo.setCurrentIndex(sel_order)
            combo.blockSignals(False)
        finally:
            self._suppress_controller_combo = False

    def _reopen_for_rescan(self):
        """No controller placeholder クリック時: Pad ウィンドウを一度閉じて開き直し、
        pygame の joystick サブシステムを quit()+init() で再列挙する。
        ネイティブ/SDL レベルで hot-plug 検出が更新されるため、その後の
        _populate_controller_combo で新しく繋がったリモコンが見える。"""
        # Windows 対策: hide() する前にコンボの popup を明示的に閉じる。
        # Qt バージョンによってはウィンドウ側 hide のみだと popup が残像化する。
        combo = getattr(self, "_controller_combo", None)
        if combo is not None:
            try:
                combo.hidePopup()
            except Exception:
                pass
        # Windows 対策: quit()/init() の前に必ず handle を閉じる。
        # 通常 No controller 状態では handle は無いが、race で残っていると
        # SDL 内部で dangling reference になり quit で crash する OS がある。
        try:
            self._close_pad_device_handles()
        except Exception:
            pass
        self.hide()

        def _do_reopen():
            # pygame joystick の強制再列挙。Windows/macOS 両対応。
            # quit → init を独立 try で囲み、どちらか失敗しても片方は試す。
            if self._pygame is not None:
                try:
                    self._pygame.joystick.quit()
                except Exception:
                    pass
                try:
                    self._pygame.joystick.init()
                except Exception:
                    pass
                try:
                    if self._sdl_mod is not None and not self._sdl_mod.get_init():
                        self._sdl_mod.init()
                except Exception:
                    pass
            # macOS: hidden display hack で HID 検出を蹴る
            # (Windows では該当なし)
            try:
                if sys.platform == "darwin":
                    self._try_darwin_hidden_display_for_joysticks()
            except Exception:
                pass
            self.show()
            self.raise_()

        # 100ms 遅延させて OS レベルの close 処理と popup 消滅を挟む
        QtCore.QTimer.singleShot(100, _do_reopen)

    def _on_controller_combo_changed(self, order):
        """ユーザがプルダウンを操作したとき: 選んだ order → joystick index に変換して切替。
        「リモコンを選ぶ」= 使う意思とみなし、Use Pad via PC が OFF なら自動で ON にする。"""
        if self._suppress_controller_combo:
            return
        combo = getattr(self, "_controller_combo", None)
        if combo is None or order < 0:
            return
        idx = combo.itemData(order, QtCore.Qt.UserRole)
        if idx is None:
            return
        # Use Pad via PC OFF なら先に ON にして polling/handles を有効化。
        # (setChecked → _on_use_pc_pad_toggled → _open_pad_device が macOS で
        # Apple GC を掴む可能性があるが、直後の _switch_to_controller が閉じて
        # 目的の pygame index を強制的に開き直す。)
        if not self.use_pc_pad_checkbox.isChecked():
            self.use_pc_pad_checkbox.setChecked(True)
        self._switch_to_controller(int(idx))

    def _switch_to_controller(self, idx: int):
        """既存ハンドルを閉じて、指定 index の pygame joystick を開き直す。
        macOS Apple GameController パスは明示切替で意味を成さないため使わない。"""
        self._close_pad_device_handles()
        self._preferred_joystick_index = idx

        if not self._ensure_pygame():
            self._refresh_pad_status_short()
            return
        pg = self._pygame
        try:
            n = pg.joystick.get_count()
            if not (0 <= idx < n):
                self._refresh_pad_status_short()
                return
            # SDL Controller が使えるなら優先 (軸マッピングが正規化されている)
            if self._sdl_mod is not None:
                try:
                    if self._sdl_mod.is_controller(idx):
                        self._sdl_controller = self._sdl_mod.Controller(idx)
                        self.use_pc_pad_checkbox.setToolTip(
                            self._sdl_controller.as_joystick().get_name() or ""
                        )
                        self._hotplug_ticks = 0
                        self._refresh_pad_status_short()
                        return
                except Exception:
                    self._sdl_controller = None
            joy = pg.joystick.Joystick(idx)
            joy.init()
            self._raw_joystick = joy
            self._pad_layout = self._detect_pad_layout(joy)
            self.use_pc_pad_checkbox.setToolTip(joy.get_name() or "")
            self._hotplug_ticks = 0
        except Exception as e:
            self._pygame_error = str(e)
        self._refresh_pad_status_short()

    def resizeEvent(self, event):
        super(PadMonitorDialog, self).resizeEvent(event)
        self._layout_pad_buttons()

    def set_play_active(self, playing: bool):
        """再生中は ▶︎_ をライトブルー、停止時は通常色に戻す。"""
        playing = bool(playing)
        if getattr(self, "_play_active", False) == playing:
            return
        self._play_active = playing
        self._play_btn.setObjectName("PadIconActionPlaying" if playing else "PadIconAction")
        self._play_btn.style().unpolish(self._play_btn)
        self._play_btn.style().polish(self._play_btn)
        self._play_btn.update()

    def set_mujoco_running_checker(self, fn):
        self._mujoco_running_checker = fn
        self._refresh_open_mujoco_btn()

    def _refresh_open_mujoco_btn(self):
        btn = getattr(self, "_open_mujoco_btn", None)
        if btn is None:
            return
        # 起動中でもクリックできるようにする（クリック時に最前面化される）。
        # ラベルとツールチップで状態を伝える。
        running = False
        checker = getattr(self, "_mujoco_running_checker", None)
        if callable(checker):
            try:
                running = bool(checker())
            except Exception:
                running = False
        btn.setEnabled(True)
        if running:
            btn.setText("Raise MuJoCo")
            btn.setToolTip("MuJoCo Studio is already running — click to bring the window to front")
        else:
            btn.setText("Open MuJoCo")
            btn.setToolTip("Launch MuJoCo Studio (raises the window if already open)")

    def showEvent(self, event):
        super(PadMonitorDialog, self).showEvent(event)
        lay = self.layout()
        if lay is not None:
            lay.activate()
            hint = lay.sizeHint()
            need_w = max(416, int(hint.width()))
            need_h = max(348, int(hint.height()))
            self.setMinimumSize(need_w, need_h)
            if self.width() < need_w or self.height() < need_h:
                self.resize(max(self.width(), need_w), max(self.height(), need_h))
        self._refresh_open_mujoco_btn()
        # 開いた瞬間にアクティブなリモコンを検索してプルダウンを反映。
        # (プルダウンを開いた瞬間の再スキャンは _RescanningComboBox.showPopup で対応)
        self._populate_controller_combo()

    def _on_always_on_top_toggled(self, checked):
        self.always_on_top = bool(checked)
        settings = load_app_settings()
        settings["pad_monitor_always_on_top"] = self.always_on_top
        save_app_settings(settings)
        self._apply_always_on_top(restore_visible=True)

    def _apply_always_on_top(self, restore_visible=True):
        was_visible = self.isVisible()
        geometry = self.geometry()
        flags = self.windowFlags()
        # Keep as a normal tool window so StaysOnTop works on macOS/Qt.
        flags |= QtCore.Qt.Window
        if self.always_on_top:
            flags |= QtCore.Qt.WindowStaysOnTopHint
        else:
            flags &= ~QtCore.Qt.WindowStaysOnTopHint
        self.setWindowFlags(flags)
        if restore_visible and was_visible:
            self.setGeometry(geometry)
            self.show()
            if self.always_on_top:
                self.raise_()
        else:
            self._fit_pad_window()

    def _on_use_pc_pad_toggled(self, checked):
        if checked:
            self._ensure_mac_gc_monitor()
            opened = self._open_pad_device(gc_wait_sec=0.4)
            self._set_manual_axis_enabled(not opened)
            self._poll_timer.start()
            if opened:
                self._poll_pc_pad()
            else:
                self.use_pc_pad_checkbox.setToolTip(
                    self._pygame_error or self._gc_backend_error or "Looking for pad — press the PS button"
                )
        else:
            self._poll_timer.stop()
            self._close_pad_device_handles()
            self._set_manual_axis_enabled(True)
            self._clear_inputs()
        self._refresh_pad_status_short()
        self.use_pc_pad_changed.emit(self.use_pc_pad_checkbox.isChecked())

    def _refresh_pad_status_short(self):
        """ユーザーが「効いてるか」をラベルだけで判断できるようにする。"""
        if not getattr(self, "_pad_status_label", None):
            return
        if not self.use_pc_pad_checkbox.isChecked():
            self._pad_status_label.setText("")
            return
        if self._gc_controller_obj is not None:
            nm = ""
            try:
                o = self._gc_controller_obj
                if o is not None:
                    for attr in ("productCategory", "vendorName", "localizedName"):
                        if hasattr(o, attr):
                            try:
                                m = getattr(o, attr)
                                nm = str(m() if callable(m) else m)
                                if nm and nm != "None":
                                    break
                            except Exception:
                                continue
            except Exception:
                nm = ""
            self._pad_status_label.setText(f"Connected: Apple GameController — {nm or 'DUALSHOCK etc.'}")
            return
        if sys.platform == "darwin" and self._sdl_controller is None and self._raw_joystick is None:
            if getattr(self, "_gc_unavailable", False) and self._gc_backend_error:
                self._pad_status_label.setText(self._gc_backend_error)
            else:
                self._pad_status_label.setText("Looking for pad — press the PS button once")
            return
        if self._pygame is None:
            self._pad_status_label.setText("pygame not loaded. Run: pip install pygame")
            return
        if self._sdl_controller is None and self._raw_joystick is None:
            try:
                n = self._pygame.joystick.get_count()
            except Exception:
                n = -1
            self._pad_status_label.setText(f"No pad connected (pygame.joystick.get_count()={n})")
            return
        if self._sdl_controller is not None:
            try:
                nm = self._sdl_controller.as_joystick().get_name() or ""
            except Exception:
                nm = ""
            self._pad_status_label.setText(f"Connected: Game Controller — {nm or '(unknown)'}")
            return
        nm = (self._raw_joystick.get_name() or "") if self._raw_joystick else ""
        self._pad_status_label.setText(f"Connected: Joystick — {nm} ({self._pad_layout} map)")

    @staticmethod
    def _configure_sdl_joystick_env():
        """macOS で Bluetooth の DUALSHOCK / DualSense を SDL に確実に渡すためのヒント。"""
        if sys.platform != "darwin":
            return
        os.environ.setdefault("SDL_JOYSTICK_HIDAPI", "1")
        os.environ.setdefault("SDL_JOYSTICK_HIDAPI_PS4", "1")
        os.environ.setdefault("SDL_JOYSTICK_HIDAPI_PS5", "1")

    def _ensure_pygame(self):
        if self._pygame is None:
            self._configure_sdl_joystick_env()
            try:
                import pygame  # type: ignore[reportMissingImports]
                pygame.init()
                pygame.joystick.init()
                self._pygame = pygame
            except Exception as e:
                self._pygame_error = f"install pygame ({e})"
                return False
            try:
                from pygame._sdl2 import controller as sdl_controller  # type: ignore

                if not sdl_controller.get_init():
                    sdl_controller.init()
                self._sdl_mod = sdl_controller
            except Exception:
                self._sdl_mod = None

        return True

    def _try_darwin_hidden_display_for_joysticks(self):
        """macOS/Linux: video 未初期化だと joystick が 0 台になることがあるため最小面を隠す。

        Windows では通常不要（余計なウィンドウが出るだけなのでスキップ）。
        """
        if sys.platform not in ("darwin", "linux") or self._pygame is None:
            return
        if self._darwin_display_hack_applied:
            return
        self._darwin_display_hack_applied = True
        try:
            pg = self._pygame
            if not pg.display.get_init():
                pg.display.init()
            if pg.display.get_surface() is None:
                hidden = getattr(pg, "HIDDEN", 8)
                try:
                    pg.display.set_mode((1, 1), hidden)
                except Exception:
                    pg.display.set_mode((1, 1))
            pg.joystick.quit()
            pg.joystick.init()
        except Exception:
            pass

    def _close_pad_device_handles(self):
        if self._sdl_controller is not None:
            try:
                self._sdl_controller.quit()
            except Exception:
                pass
            self._sdl_controller = None
        if self._raw_joystick is not None:
            try:
                self._raw_joystick.quit()
            except Exception:
                pass
            self._raw_joystick = None
        try:
            pad = self._gc_extended_pad
            if pad is not None and hasattr(pad, "setValueChangedHandler_"):
                pad.setValueChangedHandler_(None)
        except Exception:
            pass
        self._gc_controller_obj = None
        self._gc_extended_pad = None
        self._gc_handler_pad = None

    def _mac_gc_release_pygame_joystick(self):
        """GameController が取れたら SDL/HIDAPI の独占を外す。"""
        if self._sdl_controller is not None:
            try:
                self._sdl_controller.quit()
            except Exception:
                pass
            self._sdl_controller = None
        if self._raw_joystick is not None:
            try:
                self._raw_joystick.quit()
            except Exception:
                pass
            self._raw_joystick = None
        try:
            if self._pygame is not None:
                self._pygame.joystick.quit()
        except Exception:
            pass

    def _ensure_mac_gc_monitor(self):
        """Qt が前面でなくても DualShock の値を受け取れるようにする。"""
        if sys.platform != "darwin" or getattr(self, "_gc_monitor_ready", False):
            return
        try:
            from GameController import GCController  # type: ignore
        except ImportError:
            self._gc_unavailable = True
            self._gc_backend_error = (
                "pyobjc-framework-GameController が要る（pip install pyobjc-framework-GameController）"
            )
            return
        try:
            if hasattr(GCController, "setShouldMonitorBackgroundEvents_"):
                GCController.setShouldMonitorBackgroundEvents_(True)
            elif hasattr(GCController, "shouldMonitorBackgroundEvents"):
                GCController.shouldMonitorBackgroundEvents = True
        except Exception:
            pass
        try:
            if hasattr(GCController, "startWirelessControllerDiscoveryWithCompletionHandler_"):
                GCController.startWirelessControllerDiscoveryWithCompletionHandler_(None)
            self._gc_discovery_started = True
        except Exception:
            pass
        self._mac_gc_install_connect_observer()
        self._gc_monitor_ready = True

    def _mac_gc_install_connect_observer(self):
        if getattr(self, "_gc_connect_observer", None) is not None:
            return
        try:
            from Foundation import NSNotificationCenter  # type: ignore
        except Exception:
            return
        try:
            from GameController import (  # type: ignore
                GCControllerDidConnectNotification,
                GCControllerDidDisconnectNotification,
            )
        except Exception:
            GCControllerDidConnectNotification = "GCControllerDidConnectNotification"
            GCControllerDidDisconnectNotification = "GCControllerDidDisconnectNotification"

        def _did_connect(_notif):
            if not self.use_pc_pad_checkbox.isChecked():
                return
            try:
                ctl = _notif.object() if _notif is not None else None
            except Exception:
                ctl = None
            if ctl is None:
                self._try_open_mac_apple_gamepad(wait_sec=0.0)
                return
            pad = self._mac_gc_resolve_pad(ctl)
            if pad is None:
                return
            self._gc_controller_obj = ctl
            self._gc_extended_pad = pad
            self._mac_gc_attach_handlers(pad)
            self._mac_gc_release_pygame_joystick()
            self._set_manual_axis_enabled(False)
            self._hotplug_ticks = 0
            self._refresh_pad_status_short()

        def _did_disconnect(_notif):
            try:
                obj = _notif.object() if _notif is not None else None
            except Exception:
                obj = None
            if obj is not None and obj is not self._gc_controller_obj:
                return
            self._gc_controller_obj = None
            self._gc_extended_pad = None
            self._gc_handler_pad = None
            if self.use_pc_pad_checkbox.isChecked():
                self._open_pad_device(gc_wait_sec=0.0, allow_pygame=False)
            self._refresh_pad_status_short()

        # PyObjC のブロックは Python 側で参照を持たないと消える
        self._gc_connect_block = _did_connect
        self._gc_disconnect_block = _did_disconnect
        try:
            center = NSNotificationCenter.defaultCenter()
            self._gc_connect_observer = center.addObserverForName_object_queue_usingBlock_(
                GCControllerDidConnectNotification, None, None, _did_connect
            )
            self._gc_disconnect_observer = center.addObserverForName_object_queue_usingBlock_(
                GCControllerDidDisconnectNotification, None, None, _did_disconnect
            )
        except Exception:
            self._gc_connect_observer = None
            self._gc_disconnect_observer = None

    def _mac_gc_remove_observers(self):
        try:
            from Foundation import NSNotificationCenter  # type: ignore
            center = NSNotificationCenter.defaultCenter()
            for obs in (self._gc_connect_observer, self._gc_disconnect_observer):
                if obs is not None:
                    center.removeObserver_(obs)
        except Exception:
            pass
        self._gc_connect_observer = None
        self._gc_disconnect_observer = None
        self._gc_connect_block = None
        self._gc_disconnect_block = None
        self._gc_monitor_ready = False

    def _mac_gc_attach_handlers(self, pad):
        """valueChangedHandler を付けないと extendedGamepad の value() が 0 のままになる。"""
        if pad is None:
            return
        ctl = self._gc_controller_obj
        try:
            if ctl is not None and hasattr(ctl, "setPlayerIndex_"):
                ctl.setPlayerIndex_(0)
        except Exception:
            pass
        if getattr(self, "_gc_handler_pad", None) is pad:
            return

        def _on_change(_gp, _el):
            try:
                if not self.use_pc_pad_checkbox.isChecked():
                    return
                self._update_from_apple_gc()
                self._update_pad_registers()
            except Exception:
                pass

        self._gc_value_handler = _on_change
        try:
            if hasattr(pad, "setValueChangedHandler_"):
                pad.setValueChangedHandler_(_on_change)
            else:
                pad.valueChangedHandler = _on_change
            self._gc_handler_pad = pad
        except Exception:
            self._gc_handler_pad = None

    def _mac_gc_button_pressed(self, btn):
        if btn is None:
            return False
        try:
            if btn.isPressed():
                return True
        except Exception:
            pass
        try:
            return float(btn.value()) > 0.5
        except Exception:
            return False

    def _mac_gc_axis_value(self, axis, default=0.0):
        if axis is None:
            return default
        try:
            return float(axis.value())
        except Exception:
            try:
                return float(axis.value)
            except Exception:
                return default

    def _mac_gc_pump_runloop(self, seconds=0.0):
        """ターミナルや単発呼び出しでは CF ランループが進まず controllers が空のままになることがある。"""
        try:
            from CoreFoundation import CFRunLoopRunInMode, kCFRunLoopDefaultMode  # type: ignore
        except Exception:
            if seconds > 0:
                QtCore.QThread.msleep(int(min(seconds, 0.5) * 1000))
            return
        if seconds <= 0:
            try:
                CFRunLoopRunInMode(kCFRunLoopDefaultMode, 0.0, True)
            except Exception:
                pass
            try:
                QtCore.QCoreApplication.processEvents()
            except Exception:
                pass
            return
        import time as _time

        deadline = _time.time() + min(seconds, 1.0)
        while _time.time() < deadline:
            try:
                CFRunLoopRunInMode(kCFRunLoopDefaultMode, 0.05, False)
            except Exception:
                break
        try:
            QtCore.QCoreApplication.processEvents()
        except Exception:
            pass

    def _mac_gc_poll_controller_list(self, GCController, total_wait_sec=0.4):
        """ワイヤレス検索後、controllers が埋まるまで短く待つ（UI を固めない）。"""
        import time as _time

        deadline = _time.time() + max(0.0, float(total_wait_sec))
        while True:
            try:
                ctrls = GCController.controllers()
                seq = list(ctrls) if ctrls else []
                if seq:
                    return seq
            except Exception:
                seq = []
            if _time.time() >= deadline:
                break
            self._mac_gc_pump_runloop(0.05)
        try:
            ctrls = GCController.controllers()
            return list(ctrls) if ctrls else []
        except Exception:
            return []

    def _try_open_mac_apple_gamepad(self, wait_sec=0.4):
        """pygame が 0 台のとき、macOS の GameController.framework で読む。"""
        self._gc_backend_error = ""
        if sys.platform != "darwin":
            return False
        self._ensure_mac_gc_monitor()
        try:
            from GameController import GCController  # type: ignore  # pyobjc-framework-GameController
        except ImportError:
            self._gc_unavailable = True
            self._gc_backend_error = "pyobjc-framework-GameController が要る（pip install pyobjc-framework-GameController）"
            return False

        try:
            seq = self._mac_gc_poll_controller_list(GCController, total_wait_sec=wait_sec)
        except Exception as e:
            self._gc_backend_error = str(e)
            return False

        for ctl in seq:
            pad = self._mac_gc_resolve_pad(ctl)
            if pad is None:
                continue
            self._gc_controller_obj = ctl
            self._gc_extended_pad = pad
            self._mac_gc_attach_handlers(pad)
            self.use_pc_pad_checkbox.setToolTip("Apple GameController")
            self._hotplug_ticks = 0
            self._refresh_pad_status_short()
            return True

        self._gc_backend_error = (
            "GCController が空。PS ボタンを1回押す／Bluetooth を一度オフ→オン／"
            "Steam のゲームパッド設定オフ／他アプリ終了後に再チェック"
        )
        return False

    def _mac_gc_resolve_pad(self, ctl):
        """GCController から現在有効な gamepad / extendedGamepad を取り直す（参照スタレ対策）。"""
        if ctl is None:
            return None
        pad = None
        try:
            if hasattr(ctl, "extendedGamepad"):
                eg = ctl.extendedGamepad
                pad = eg() if callable(eg) else eg
        except Exception:
            pad = None
        if pad is None:
            try:
                if hasattr(ctl, "gamepad"):
                    gp = ctl.gamepad
                    pad = gp() if callable(gp) else gp
            except Exception:
                pad = None
        return pad

    def _update_from_apple_gc(self):
        ctl = self._gc_controller_obj
        if ctl is None:
            return
        pad = self._mac_gc_resolve_pad(ctl)
        if pad is None:
            return
        self._gc_extended_pad = pad

        def stick_pair(stick):
            if stick is None:
                return 0, 0
            try:
                x_axis = stick.xAxis
                x_axis = x_axis() if callable(x_axis) else x_axis
                y_axis = stick.yAxis
                y_axis = y_axis() if callable(y_axis) else y_axis
            except Exception:
                return 0, 0
            vx = max(-1.0, min(1.0, self._mac_gc_axis_value(x_axis)))
            vy = max(-1.0, min(1.0, self._mac_gc_axis_value(y_axis)))
            return int(round(vx * 127)), int(round(vy * 127))

        try:
            bA = pad.buttonA
            bA = bA() if callable(bA) else bA
        except Exception:
            bA = None
        try:
            bB = pad.buttonB
            bB = bB() if callable(bB) else bB
        except Exception:
            bB = None
        try:
            bX = pad.buttonX
            bX = bX() if callable(bX) else bX
        except Exception:
            bX = None
        try:
            bY = pad.buttonY
            bY = bY() if callable(bY) else bY
        except Exception:
            bY = None

        self._set_button("Cross", self._mac_gc_button_pressed(bA))
        self._set_button("Circle", self._mac_gc_button_pressed(bB))
        self._set_button("Square", self._mac_gc_button_pressed(bX))
        self._set_button("Triangle", self._mac_gc_button_pressed(bY))

        for attr, logical in (
            ("leftShoulder", "L1"),
            ("rightShoulder", "R1"),
        ):
            try:
                b = getattr(pad, attr, None)
                b = b() if callable(b) else b
            except Exception:
                b = None
            self._set_button(logical, self._mac_gc_button_pressed(b))

        try:
            lt = getattr(pad, "leftTrigger", None)
            lt = lt() if callable(lt) else lt
        except Exception:
            lt = None
        try:
            rt = getattr(pad, "rightTrigger", None)
            rt = rt() if callable(rt) else rt
        except Exception:
            rt = None
        l2v = int(round(max(0.0, min(1.0, self._mac_gc_axis_value(lt))) * 255))
        r2v = int(round(max(0.0, min(1.0, self._mac_gc_axis_value(rt))) * 255))
        self._set_axis_value("L2v", l2v)
        self._set_axis_value("R2v", r2v)
        self._set_button("L2", l2v > 48)
        self._set_button("R2", r2v > 48)

        opt = getattr(pad, "buttonOptions", None)
        opt = opt() if callable(opt) else opt
        menu = getattr(pad, "buttonMenu", None)
        menu = menu() if callable(menu) else menu
        self._set_button("Select", self._mac_gc_button_pressed(opt))
        self._set_button("Start", self._mac_gc_button_pressed(menu))

        try:
            d = pad.dpad
            d = d() if callable(d) else d
        except Exception:
            d = None
        if d is not None:
            for attr, logical in (
                ("up", "DPad Up"),
                ("down", "DPad Down"),
                ("left", "DPad Left"),
                ("right", "DPad Right"),
            ):
                try:
                    b = getattr(d, attr, None)
                    b = b() if callable(b) else b
                except Exception:
                    b = None
                self._set_button(logical, self._mac_gc_button_pressed(b))

        try:
            ls = getattr(pad, "leftThumbstick", None)
            ls = ls() if callable(ls) else ls
        except Exception:
            ls = None
        try:
            rs = getattr(pad, "rightThumbstick", None)
            rs = rs() if callable(rs) else rs
        except Exception:
            rs = None
        lx, ly = stick_pair(ls)
        rx, ry = stick_pair(rs)
        self._set_axis_value("Lx", lx)
        self._set_axis_value("Ly", ly)
        self._set_axis_value("Rx", rx)
        self._set_axis_value("Ry", ry)
        self._update_value_labels()

    def _detect_pad_layout(self, joy):
        """生 pygame.joystick 用: 名前から PS4 系かどうか推定（macOS Bluetooth 等）。"""
        name = (joy.get_name() or "").lower()
        ps4_hints = (
            "dualshock",
            "dualsense",
            "wireless controller",
            "sony interactive",
            "playstation 4",
            "ps4",
            "054c",  # USB VID が名前に出る場合
        )
        if any(h in name for h in ps4_hints):
            return "ps4"
        return "ps3"

    def _open_pad_device(self, gc_wait_sec=0.35, allow_pygame=None):
        """macOS は Apple GameController を先に（pygame.init が GC と干渉することがある）。その後 SDL / 生ジョイスティック。"""
        if self._gc_controller_obj is not None and self._mac_gc_resolve_pad(self._gc_controller_obj) is not None:
            self._mac_gc_attach_handlers(self._gc_extended_pad)
            return True
        if self._sdl_controller is not None:
            try:
                if not hasattr(self._sdl_controller, "attached") or self._sdl_controller.attached():
                    return True
            except Exception:
                return True
        if self._raw_joystick is not None:
            try:
                if self._raw_joystick.get_init():
                    return True
            except Exception:
                pass

        self._close_pad_device_handles()

        if allow_pygame is None:
            allow_pygame = (
                sys.platform != "darwin"
                or getattr(self, "_gc_unavailable", False)
                or self._hotplug_ticks >= 90
            )

        if sys.platform == "darwin" and self._try_open_mac_apple_gamepad(wait_sec=gc_wait_sec):
            self._mac_gc_release_pygame_joystick()
            return True

        if not allow_pygame:
            self._refresh_pad_status_short()
            return False

        pygame_ok = self._ensure_pygame()
        if not pygame_ok:
            if sys.platform == "darwin" and self._try_open_mac_apple_gamepad(wait_sec=0.0):
                return True
            self._refresh_pad_status_short()
            return False

        pg = self._pygame
        try:
            n = pg.joystick.get_count()
            if n <= 0:
                self._try_darwin_hidden_display_for_joysticks()
                n = pg.joystick.get_count()
            if n > 0 and self._sdl_mod is not None:
                for i in range(n):
                    try:
                        if self._sdl_mod.is_controller(i):
                            self._sdl_controller = self._sdl_mod.Controller(i)
                            tip = self._sdl_controller.as_joystick().get_name()
                            self.use_pc_pad_checkbox.setToolTip(tip)
                            self._hotplug_ticks = 0
                            self._refresh_pad_status_short()
                            return True
                    except Exception:
                        continue
            if n > 0:
                joy = pg.joystick.Joystick(0)
                joy.init()
                self._raw_joystick = joy
                self._pad_layout = self._detect_pad_layout(joy)
                self.use_pc_pad_checkbox.setToolTip(joy.get_name())
                self._hotplug_ticks = 0
                self._refresh_pad_status_short()
                return True
        except Exception as e:
            self._pygame_error = str(e)

        if sys.platform == "darwin" and self._try_open_mac_apple_gamepad(wait_sec=0.0):
            self._mac_gc_release_pygame_joystick()
            return True

        self.use_pc_pad_checkbox.setToolTip("PC pad is not connected")
        self._refresh_pad_status_short()
        return False

    def _maybe_hotplug_rescan(self):
        """Bluetooth 接続直後などで enumerate が遅れる場合に再スキャン。"""
        self._hotplug_ticks += 1
        if self._hotplug_ticks != 1 and self._hotplug_ticks % 30 != 0:
            return
        allow_pygame = (
            sys.platform != "darwin"
            or getattr(self, "_gc_unavailable", False)
            or self._hotplug_ticks >= 90
        )
        if allow_pygame:
            try:
                if self._pygame is not None:
                    self._pygame.joystick.quit()
                    self._pygame.joystick.init()
                    if self._sdl_mod is not None and not self._sdl_mod.get_init():
                        self._sdl_mod.init()
            except Exception:
                pass
        self._open_pad_device(gc_wait_sec=0.0, allow_pygame=allow_pygame)

    _poll_debug_counter = 0

    def _poll_pc_pad(self):
        if not self.use_pc_pad_checkbox.isChecked():
            return

        # Debug: print every 30 polls (~1 second) when hidden
        if not self.isVisible():
            PadMonitorDialog._poll_debug_counter += 1
            if PadMonitorDialog._poll_debug_counter % 30 == 0:
                btn = PAD_REGISTER_VALUES.get("Pad_btn", 0)
                print(f"[PadMonitor] Background polling... Pad_btn={btn}")

        if self._gc_controller_obj is not None:
            try:
                QtCore.QCoreApplication.processEvents()
                self._mac_gc_pump_runloop(0.0)
                pad = self._mac_gc_resolve_pad(self._gc_controller_obj)
                if pad is not None:
                    self._gc_extended_pad = pad
                    self._mac_gc_attach_handlers(pad)
                self._update_from_apple_gc()
            except Exception as e:
                self.use_pc_pad_checkbox.setToolTip(str(e))
                try:
                    self._open_pad_device(gc_wait_sec=0.0, allow_pygame=False)
                except Exception:
                    pass
            # Always update PAD_REGISTER_VALUES even when window is hidden
            self._update_pad_registers()
            self._refresh_pad_status_short()
            return

        if sys.platform == "darwin" and self._sdl_controller is None and self._raw_joystick is None:
            if not getattr(self, "_gc_unavailable", False):
                self._maybe_hotplug_rescan()
                if self._gc_controller_obj is not None:
                    try:
                        self._update_from_apple_gc()
                        self._update_pad_registers()
                    except Exception:
                        pass
                    self._refresh_pad_status_short()
                    return
                self._refresh_pad_status_short()
                if self._hotplug_ticks < 90:
                    return

        if not self._ensure_pygame():
            return

        if self._sdl_controller is None and self._raw_joystick is None:
            self._clear_inputs()
            self._maybe_hotplug_rescan()
            self._refresh_pad_status_short()
            return

        try:
            # Windows 対策: pump() 自体が例外を投げる稀ケース (SDL 内部エラー)
            # を分離キャッチ。失敗した場合は handle を疑って閉じる。
            try:
                self._pygame.event.pump()
            except Exception:
                self._close_pad_device_handles()
                self._clear_inputs()
                self._refresh_pad_status_short()
                return

            # 物理切断ガード: pump() で SDL_JOYDEVICEREMOVED が処理された後、
            # get_count() が 0 (もしくは我々の id 以下) なら基底オブジェクトが
            # 破棄されている可能性が高く、joy.get_button() などが C レベルで
            # segfault する。read する前にここで検出して安全に閉じる。
            # (joy.get_init() は quit() を呼ばない限り True のままなので判定に使えない)
            pg = self._pygame
            try:
                _count = pg.joystick.get_count()
            except Exception:
                _count = 0
            _joy_id = None
            _id_lookup_failed = False
            if self._raw_joystick is not None:
                try:
                    _joy_id = int(self._raw_joystick.get_id())
                except Exception:
                    _id_lookup_failed = True
            if self._sdl_controller is not None and _joy_id is None and not _id_lookup_failed:
                try:
                    _joy_id = int(self._sdl_controller.as_joystick().get_id())
                except Exception:
                    _id_lookup_failed = True
            # Windows 対策: get_id() 例外は「ハンドルが既に無効」の強い兆候。
            # 一部ドライバでは id が返らないので、この時点で切断扱いにする。
            _lost = (
                _count <= 0
                or _id_lookup_failed
                or (_joy_id is not None and _joy_id >= _count)
            )
            if _lost:
                # 物理切断とみなして handle を安全に閉じ、入力をクリア。
                # 再接続はチェックボックスの再チェック or プルダウン再選択に委ねる。
                self._close_pad_device_handles()
                self._clear_inputs()
                self._refresh_pad_status_short()
                return

            if self._sdl_controller is not None:
                # attached() でも切断検出。ここでは reopen せず handle を閉じるだけ
                # (以前は _open_pad_device を呼んでいたが、切断直後に呼ぶと
                # 上流の hotplug 処理と競合して segfault の温床になる)
                if hasattr(self._sdl_controller, "attached") and not self._sdl_controller.attached():
                    self._close_pad_device_handles()
                    self._clear_inputs()
                    self._refresh_pad_status_short()
                    return
                self._update_from_sdl_controller()
            else:
                if self._raw_joystick is not None and not self._raw_joystick.get_init():
                    self._close_pad_device_handles()
                    self._clear_inputs()
                    self._refresh_pad_status_short()
                    return
                self._update_axes_from_joystick()
                self._update_buttons_from_joystick()
            # Always update PAD_REGISTER_VALUES even when window is hidden
            self._update_pad_registers()
        except Exception as e:
            # Python-level 例外のみキャッチ。念のため handle も閉じておく
            self.use_pc_pad_checkbox.setToolTip(str(e))
            try:
                self._close_pad_device_handles()
                self._clear_inputs()
            except Exception:
                pass
        self._refresh_pad_status_short()

    def _update_from_sdl_controller(self):
        """SDL Game Controller（ Xbox 配置名だが PS4 でも同じ意味のボタンに正規化される）。"""
        c = self._sdl_controller
        m = self._sdl_mod
        if c is None or m is None:
            return

        # pygame-ce removed named CONTROLLER_* constants; fall back to SDL2 integer values.
        BTN_A  = getattr(m, "CONTROLLER_BUTTON_A",            0)
        BTN_B  = getattr(m, "CONTROLLER_BUTTON_B",            1)
        BTN_X  = getattr(m, "CONTROLLER_BUTTON_X",            2)
        BTN_Y  = getattr(m, "CONTROLLER_BUTTON_Y",            3)
        BTN_BK = getattr(m, "CONTROLLER_BUTTON_BACK",         4)
        BTN_ST = getattr(m, "CONTROLLER_BUTTON_START",        6)
        BTN_LS = getattr(m, "CONTROLLER_BUTTON_LEFTSHOULDER", 9)
        BTN_RS = getattr(m, "CONTROLLER_BUTTON_RIGHTSHOULDER", 10)
        BTN_DU = getattr(m, "CONTROLLER_BUTTON_DPAD_UP",      11)
        BTN_DD = getattr(m, "CONTROLLER_BUTTON_DPAD_DOWN",    12)
        BTN_DL = getattr(m, "CONTROLLER_BUTTON_DPAD_LEFT",    13)
        BTN_DR = getattr(m, "CONTROLLER_BUTTON_DPAD_RIGHT",   14)
        AX_LX  = getattr(m, "CONTROLLER_AXIS_LEFTX",          0)
        AX_LY  = getattr(m, "CONTROLLER_AXIS_LEFTY",          1)
        AX_RX  = getattr(m, "CONTROLLER_AXIS_RIGHTX",         2)
        AX_RY  = getattr(m, "CONTROLLER_AXIS_RIGHTY",         3)
        AX_TL  = getattr(m, "CONTROLLER_AXIS_TRIGGERLEFT",    4)
        AX_TR  = getattr(m, "CONTROLLER_AXIS_TRIGGERRIGHT",   5)

        def stick_127(axis_id):
            v = int(c.get_axis(axis_id))
            v = max(-32768, min(32767, v))
            fv = v / 32767.0
            fv = max(-1.0, min(1.0, fv))
            return int(round(fv * 127))

        def trig_255(axis_id):
            v = max(0, int(c.get_axis(axis_id)))
            fv = min(1.0, v / 32767.0)
            return int(round(fv * 255))

        self._set_button("Cross",    bool(c.get_button(BTN_A)))
        self._set_button("Circle",   bool(c.get_button(BTN_B)))
        self._set_button("Square",   bool(c.get_button(BTN_X)))
        self._set_button("Triangle", bool(c.get_button(BTN_Y)))
        self._set_button("L1",       bool(c.get_button(BTN_LS)))
        self._set_button("R1",       bool(c.get_button(BTN_RS)))
        self._set_button("Select",   bool(c.get_button(BTN_BK)))
        self._set_button("Start",    bool(c.get_button(BTN_ST)))

        l2v = trig_255(AX_TL)
        r2v = trig_255(AX_TR)
        self._set_axis_value("L2v", l2v)
        self._set_axis_value("R2v", r2v)
        self._set_button("L2", l2v > 48)
        self._set_button("R2", r2v > 48)

        self._set_button("DPad Up",    bool(c.get_button(BTN_DU)))
        self._set_button("DPad Down",  bool(c.get_button(BTN_DD)))
        self._set_button("DPad Left",  bool(c.get_button(BTN_DL)))
        self._set_button("DPad Right", bool(c.get_button(BTN_DR)))

        self._set_axis_value("Lx", stick_127(AX_LX))
        # SDL axis Y is negative when pushed up; invert to +Y = up
        self._set_axis_value("Ly", -stick_127(AX_LY))
        self._set_axis_value("Rx", stick_127(AX_RX))
        self._set_axis_value("Ry", -stick_127(AX_RY))
        self._update_value_labels()

    def _update_buttons_from_joystick(self):
        joy = self._raw_joystick
        if joy is None:
            return

        if self._pad_layout == "ps4":
            # macOS + Bluetooth DUALSHOCK 4 で多い生ボタン番号（SDL ジョイスティック層）
            index_to_logical = {
                0: "Square",
                1: "Cross",
                2: "Circle",
                3: "Triangle",
                4: "Select",
                6: "Start",
                9: "L1",
                10: "R1",
            }
        else:
            index_to_logical = {
                0: "Cross",
                1: "Circle",
                2: "Square",
                3: "Triangle",
                4: "L1",
                5: "R1",
                6: "Select",
                7: "Start",
                8: "L2",
                9: "R2",
            }

        button_count = joy.get_numbuttons()
        for index, name in index_to_logical.items():
            if index < button_count:
                self._set_button(name, bool(joy.get_button(index)))

        try:
            hat_x, hat_y = joy.get_hat(0) if joy.get_numhats() else (0, 0)
        except Exception:
            hat_x, hat_y = (0, 0)
        self._set_button("DPad Left", hat_x < 0)
        self._set_button("DPad Right", hat_x > 0)
        self._set_button("DPad Up", hat_y > 0)
        self._set_button("DPad Down", hat_y < 0)
        self._update_value_labels()

    def _normalize_trigger_axis(self, value):
        """トリガーが -1..1（離し=-1）か 0..1 かを雑に判別して 0..255 にする。"""
        v = float(value)
        if v < 0.0:
            norm = (v + 1.0) * 0.5
        else:
            norm = v
        norm = max(0.0, min(1.0, norm))
        return int(round(norm * 255))

    def _update_axes_from_joystick(self):
        joy = self._raw_joystick
        if joy is None:
            return

        axis_map = {
            "Lx": 0,
            "Ly": 1,
            "Rx": 2,
            "Ry": 3,
            "L2v": 4,
            "R2v": 5,
        }
        axis_count = joy.get_numaxes()
        for name, index in axis_map.items():
            value = 0.0
            if index < axis_count:
                value = float(joy.get_axis(index))
            if name in ("L2v", "R2v"):
                int_value = self._normalize_trigger_axis(value)
            else:
                if name in ("Ly", "Ry"):
                    value = -value  # SDL axis Y is negative when pushed up
                value = max(-1.0, min(1.0, value))
                int_value = int(round(value * 127))
            self._set_axis_value(name, int_value)

        if self._pad_layout == "ps4":
            l2v = self._axis_sliders["L2v"].value()
            r2v = self._axis_sliders["R2v"].value()
            self._set_button("L2", l2v > 48)
            self._set_button("R2", r2v > 48)

        self._update_value_labels()

    def _set_manual_axis_enabled(self, enabled):
        # Enable/disable 2D pads
        for pad in self._axis_pads.values():
            pad.set_enabled(enabled)
        # Enable/disable L2v/R2v sliders only (Lx/Ly/Rx/Ry are handled by 2D pads)
        for name in ("L2v", "R2v"):
            self._axis_sliders[name].setEnabled(enabled)
            self._axis_inputs[name].setEnabled(enabled)

    def _set_axis_value(self, name, value):
        self._updating_axes = True
        # Handle 2D pad axes (Lx/Ly/Rx/Ry)
        if name in ("Lx", "Ly"):
            pad = self._axis_pads.get("L")
            if pad:
                x, y = pad.get_values()
                if name == "Lx":
                    pad.set_values(value, y)
                else:
                    pad.set_values(x, value)
            self._refresh_stick_caption("L")
        elif name in ("Rx", "Ry"):
            pad = self._axis_pads.get("R")
            if pad:
                x, y = pad.get_values()
                if name == "Rx":
                    pad.set_values(value, y)
                else:
                    pad.set_values(x, value)
            self._refresh_stick_caption("R")
        else:
            # L2v/R2v use sliders
            self._axis_sliders[name].setValue(value)
            self._axis_inputs[name].setValue(value)
        self._updating_axes = False
        # Update global Pad register for this axis
        reg_name = "Pad_" + name
        PAD_REGISTER_VALUES[reg_name] = value

    def _on_axis_slider_changed(self, name, value):
        if self._updating_axes:
            return
        self._set_axis_value(name, value)
        if name in ("L2v", "R2v"):
            self._set_button(name[:2], value > 48)

    def _on_trigger_slider_released(self, name):
        """マウスを離したらトリガーを 0 に戻す（実機トリガー相当）。"""
        if not self._axis_sliders[name].isEnabled():
            return
        self._set_axis_value(name, 0)
        self._set_button(name[:2], False)

    def _on_axis_input_changed(self, name, value):
        if self._updating_axes:
            return
        self._set_axis_value(name, value)

    def _on_left_pad_changed(self, x, y):
        if self._updating_axes:
            return
        PAD_REGISTER_VALUES["Pad_Lx"] = x
        PAD_REGISTER_VALUES["Pad_Ly"] = y
        self._refresh_stick_caption("L")

    def _on_right_pad_changed(self, x, y):
        if self._updating_axes:
            return
        PAD_REGISTER_VALUES["Pad_Rx"] = x
        PAD_REGISTER_VALUES["Pad_Ry"] = y
        self._refresh_stick_caption("R")

    def _set_button(self, name, checked):
        widget = self._button_widgets.get(name)
        if widget:
            widget.setChecked(checked)
            # Update global Pad register for this button
            reg_name = "Pad_" + name.replace(" ", "_")
            PAD_REGISTER_VALUES[reg_name] = 1 if checked else 0
            # Update combined button value
            PAD_REGISTER_VALUES["Pad_btn"] = self._current_button_value()

    def _current_button_value(self):
        value = 0
        for name, widget in self._button_widgets.items():
            if widget.isChecked():
                value |= self._button_bits.get(name, 0)
        return value

    def _refresh_stick_caption(self, side):
        pad = self._axis_pads.get(side)
        label = self._axis_labels.get(side)
        if pad is None or label is None:
            return
        x, y = pad.get_values()
        prefix = "L" if side == "L" else "R"
        label.setText(f"{prefix}x {x}\n{prefix}y {y}")

    def _update_value_labels(self, _checked=None):
        value = self._current_button_value()
        self.value_decimal_label.setText(str(value))
        bits = f"{value:016b}"
        self.value_binary_label.setText(" ".join(bits[i:i + 4] for i in range(0, 16, 4)))
        # Also update global Pad registers
        self._update_pad_registers()

    def _update_pad_registers(self):
        """Update global PAD_REGISTER_VALUES with current Pad state."""
        old_btn = PAD_REGISTER_VALUES.get("Pad_btn", 0)

        # Combined button value (16-bit)
        PAD_REGISTER_VALUES["Pad_btn"] = self._current_button_value()

        # Individual button values (0 or 1)
        for name, widget in self._button_widgets.items():
            # Convert button names like "DPad Up" to "Pad_DPad_Up"
            reg_name = "Pad_" + name.replace(" ", "_")
            PAD_REGISTER_VALUES[reg_name] = 1 if widget.isChecked() else 0

        # Axis values - 2D pads for Lx/Ly/Rx/Ry, sliders for L2v/R2v
        if "L" in self._axis_pads:
            lx, ly = self._axis_pads["L"].get_values()
            PAD_REGISTER_VALUES["Pad_Lx"] = lx
            PAD_REGISTER_VALUES["Pad_Ly"] = ly
        if "R" in self._axis_pads:
            rx, ry = self._axis_pads["R"].get_values()
            PAD_REGISTER_VALUES["Pad_Rx"] = rx
            PAD_REGISTER_VALUES["Pad_Ry"] = ry
        for name in ("L2v", "R2v"):
            if name in self._axis_sliders:
                PAD_REGISTER_VALUES["Pad_" + name] = self._axis_sliders[name].value()

        # Debug: print when button value changes
        new_btn = PAD_REGISTER_VALUES.get("Pad_btn", 0)
        if new_btn != old_btn:
            print(f"[PadMonitor] Pad_btn changed: {old_btn} -> {new_btn} (0x{new_btn:04X})")

    def _clear_inputs(self):
        for check in self._button_widgets.values():
            check.setChecked(False)
        for name in self.AXIS_NAMES:
            self._set_axis_value(name, 0)

    def keyPressEvent(self, event):
        if event.key() in (QtCore.Qt.Key_Return, QtCore.Qt.Key_Enter):
            event.accept()
            return
        super(PadMonitorDialog, self).keyPressEvent(event)

    def accept(self):
        pass

    def shutdown(self):
        """Force-close on LME quit (bypasses hide-while-polling)."""
        self._force_quit = True
        try:
            self._poll_timer.stop()
        except Exception:
            pass
        try:
            self._mac_gc_remove_observers()
        except Exception:
            pass
        try:
            self._close_pad_device_handles()
        except Exception:
            pass
        self.close()

    def closeEvent(self, event):
        # Hide instead of closing to keep polling active
        # Polling continues in background to update PAD_REGISTER_VALUES
        if getattr(self, "_force_quit", False):
            try:
                self._poll_timer.stop()
                self._close_pad_device_handles()
            except Exception:
                pass
            super(PadMonitorDialog, self).closeEvent(event)
            return
        if self.use_pc_pad_checkbox.isChecked():
            # Keep polling in background
            event.ignore()
            self.hide()
            print("[PadMonitor] Hidden but still polling in background")
        else:
            # Actually close if not polling
            self._poll_timer.stop()
            self._close_pad_device_handles()
            super(PadMonitorDialog, self).closeEvent(event)


# ==============================================================================
# ValueListDialog - User value table display
# ==============================================================================

class ValueListDialog(QtWidgets.QDialog):
    """Value List: UserVal_0-63をテーブル表示。

    Shows a table with:
    - UserVal: UserVal_0, UserVal_1, etc. (all 64 entries)
    - 現在値: Current value from user_value_session
    - 定義元: Dropdown to select which Define node to use (node title)
    - メモ: Memo content from the selected Define node
    """

    # Class variable to track the active instance for refresh
    _active_instance = None

    def __init__(self, graph, parent=None):
        super(ValueListDialog, self).__init__(parent)
        self.graph = graph
        self.setWindowTitle("Value List")
        self.setModal(False)  # Non-modal for live updates
        self.resize(650, 500)

        # Register as active instance
        ValueListDialog._active_instance = self

        root = QtWidgets.QVBoxLayout(self)

        # Create table (always 64 rows)
        self._table = QtWidgets.QTableWidget()
        self._table.setColumnCount(4)
        self._table.setHorizontalHeaderLabels(["UserVal", "Value", "Source", "Memo"])
        self._table.verticalHeader().setVisible(False)  # Hide row numbers
        self._table.horizontalHeader().setSectionResizeMode(0, QtWidgets.QHeaderView.ResizeToContents)
        self._table.horizontalHeader().setSectionResizeMode(1, QtWidgets.QHeaderView.ResizeToContents)
        self._table.horizontalHeader().resizeSection(2, 200)  # 定義元: 200px
        self._table.horizontalHeader().setSectionResizeMode(3, QtWidgets.QHeaderView.Stretch)  # メモ: stretch
        self._table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self._table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self._table.setRowCount(USER_VALUE_SESSION_COUNT)

        self._source_combos = {}  # Map from row to combo (or None if no defines)

        # Initialize all 64 rows
        for row in range(USER_VALUE_SESSION_COUNT):
            # Column 0: UserVal
            uv_item = QtWidgets.QTableWidgetItem(f"UserVal_{row}")
            uv_item.setTextAlignment(QtCore.Qt.AlignCenter)
            self._table.setItem(row, 0, uv_item)

            # Column 1: 現在値 (placeholder, will be filled by refresh)
            value_item = QtWidgets.QTableWidgetItem("0")
            value_item.setTextAlignment(QtCore.Qt.AlignCenter)
            self._table.setItem(row, 1, value_item)

            # Column 2: 定義元 - will be populated by refresh (combo or empty)
            # Column 3: メモ (placeholder, will be filled by refresh)
            memo_item = QtWidgets.QTableWidgetItem("")
            self._table.setItem(row, 3, memo_item)

        root.addWidget(self._table)

        # Buttons
        bbox = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.StandardButton.Close)
        bbox.rejected.connect(self.reject)
        root.addWidget(bbox)

        # Initial population
        self.refresh()

    def closeEvent(self, event):
        """Clear active instance on close."""
        if ValueListDialog._active_instance is self:
            ValueListDialog._active_instance = None
        super().closeEvent(event)

    def refresh(self):
        """Refresh the table with current values and defines."""
        # Scan all Actions for DefineNodes
        define_map = self._scan_all_defines()

        # Get current user_value_session
        slots = normalize_user_value_session(getattr(self.graph, "user_value_session", None))

        # Block signals during update
        for combo in self._source_combos.values():
            if combo:
                combo.blockSignals(True)

        for row in range(USER_VALUE_SESSION_COUNT):
            # Update Column 1: 現在値
            slot = slots[row]
            if slot.get("kind") == "literal":
                current_value = str(slot.get("value", 0))
            else:
                current_value = f"→{slot.get('name', '?')}"
            value_item = self._table.item(row, 1)
            if value_item:
                value_item.setText(current_value)

            # Update Column 2: 定義元
            defines = define_map.get(row, [])
            existing_combo = self._source_combos.get(row)

            if defines:
                # Has defines - show dropdown
                if existing_combo is None:
                    # Create new combo
                    combo = QtWidgets.QComboBox()
                    combo.setStyleSheet(
                        "QComboBox { color: black; } "
                        "QComboBox QAbstractItemView { color: black; }"
                    )
                    combo.currentIndexChanged.connect(
                        lambda idx, r=row: self._on_source_changed(r, idx)
                    )
                    self._table.setCellWidget(row, 2, combo)
                    self._source_combos[row] = combo
                else:
                    combo = existing_combo
                    combo.blockSignals(True)

                combo.clear()
                current_selection = -1
                for i, define_info in enumerate(defines):
                    action_name = define_info["action_name"]
                    node_name = define_info["node_name"]
                    label = f"{action_name}: {node_name}"
                    combo.addItem(label, define_info)
                    if self._is_current_source(slot, define_info):
                        current_selection = i
                if current_selection >= 0:
                    combo.setCurrentIndex(current_selection)
                combo.blockSignals(False)

                # Update Column 3: メモ
                selected_idx = combo.currentIndex()
                if selected_idx >= 0:
                    selected_info = combo.itemData(selected_idx)
                    memo_text = selected_info.get("define_memo", "") if selected_info else ""
                else:
                    memo_text = ""
            else:
                # No defines - clear cell
                if existing_combo is not None:
                    self._table.removeCellWidget(row, 2)
                    self._source_combos[row] = None
                # Set empty item for 定義元
                empty_item = QtWidgets.QTableWidgetItem("")
                self._table.setItem(row, 2, empty_item)
                memo_text = ""

            # Update memo column
            memo_item = self._table.item(row, 3)
            if memo_item:
                memo_item.setText(memo_text)

        # Unblock signals
        for combo in self._source_combos.values():
            if combo:
                combo.blockSignals(False)

    @classmethod
    def refresh_active(cls):
        """Refresh the active ValueListDialog instance if exists."""
        if cls._active_instance is not None:
            cls._active_instance.refresh()

    def _scan_all_defines(self):
        """Scan all Actions for DefineNodes and return a mapping.

        Returns:
            dict: {res_index: [{"action_idx": int, "action_name": str,
                               "define_memo": str, "define_kind": str,
                               "define_literal": int, "define_register_name": str}, ...]}
        """
        define_map = {}  # res_index -> list of define infos

        mas = getattr(self.graph, "motion_action_state", None)
        if not mas:
            # No action state - scan current graph only
            self._scan_defines_in_graph(define_map, self.graph, 0, "Action_1")
            return define_map

        items = mas.get("items", [])
        current_idx = mas.get("current", 0)

        for action_idx, entry in enumerate(items):
            action_title = entry.get("title", "") or ""
            action_name = f"Action_{action_idx + 1}"
            if action_title:
                action_name = f"Action_{action_idx + 1}: {action_title}"

            if action_idx == current_idx:
                # Current action - scan nodes directly from graph
                self._scan_defines_in_graph(define_map, self.graph, action_idx, action_name)
            else:
                # Other actions - scan from saved data
                data = entry.get("data")
                if data:
                    self._scan_defines_in_data(define_map, data, action_idx, action_name)

        return define_map

    def _scan_defines_in_graph(self, define_map, graph, action_idx, action_name):
        """Scan DefineNodes in the current graph."""
        for node in graph.all_nodes():
            # Use string-based type check to avoid circular import
            if node.__class__.__name__ == "DefineNode":
                uv_idx = getattr(node, "define_uv_index", 0)
                if uv_idx not in define_map:
                    define_map[uv_idx] = []
                define_map[uv_idx].append({
                    "action_idx": action_idx,
                    "action_name": action_name,
                    "node_name": node.name(),
                    "define_memo": getattr(node, "define_memo", "") or "",
                    "define_kind": getattr(node, "define_kind", "literal"),
                    "define_literal": getattr(node, "define_literal", 0),
                    "define_register_name": getattr(node, "define_register_name", "") or "",
                })

    def _scan_defines_in_data(self, define_map, data, action_idx, action_name):
        """Scan DefineNodes in saved action data."""
        nodes = data.get("nodes", [])
        for nd in nodes:
            if nd.get("node_type") == "define":
                uv_idx = nd.get("define_uv_index", 0)
                if uv_idx not in define_map:
                    define_map[uv_idx] = []
                define_map[uv_idx].append({
                    "action_idx": action_idx,
                    "action_name": action_name,
                    "node_name": nd.get("name", f"define UserVal_{uv_idx}"),
                    "define_memo": nd.get("define_memo", "") or "",
                    "define_kind": nd.get("define_kind", "literal"),
                    "define_literal": nd.get("define_literal", 0),
                    "define_register_name": nd.get("define_register_name", "") or "",
                })

    def _is_current_source(self, slot, define_info):
        """Check if the slot matches the define info."""
        if slot.get("kind") == "literal" and define_info["define_kind"] == "literal":
            return slot.get("value", 0) == define_info["define_literal"]
        elif slot.get("kind") == "register" and define_info["define_kind"] == "register":
            return slot.get("name", "") == define_info["define_register_name"]
        return False

    def _on_source_changed(self, row, combo_idx):
        """Handle source dropdown change."""
        combo = self._source_combos.get(row)
        if not combo:
            return

        define_info = combo.itemData(combo_idx)
        if not define_info:
            return

        # row == uv_idx (since we have 64 rows for 64 UserVals)
        uv_idx = row

        # Update user_value_session
        slots = normalize_user_value_session(getattr(self.graph, "user_value_session", None))

        if define_info["define_kind"] == "literal":
            slots[uv_idx] = {"kind": "literal", "value": define_info["define_literal"]}
            new_value = str(define_info["define_literal"])
        else:
            slots[uv_idx] = {"kind": "register", "name": define_info["define_register_name"]}
            new_value = f"→{define_info['define_register_name']}"

        self.graph.user_value_session = slots

        # Update value display in table
        value_item = self._table.item(row, 1)
        if value_item:
            value_item.setText(new_value)

        # Update memo column
        memo_item = self._table.item(row, 3)
        if memo_item:
            memo_item.setText(define_info.get("define_memo", ""))

        print(f"[ValueList] UserVal_{uv_idx} updated to: {new_value}")


# ==============================================================================
# MotionFormsDialog - Formula editing for Pose Branching
# ==============================================================================

class MotionFormsDialog(QtWidgets.QDialog):
    """Pose Branching 用 Formula 一覧の編集モーダル。"""

    _DEFAULT_BODY = "\n".join(["(Not Available Now)"] * 5)

    def __init__(self, graph, parent=None):
        super(MotionFormsDialog, self).__init__(parent)
        self.graph = graph
        self.setWindowTitle("Forms")
        self.setModal(True)
        self.resize(440, 320)

        layout = QtWidgets.QVBoxLayout(self)

        row_name = QtWidgets.QHBoxLayout()
        row_name.addWidget(create_label("Formula 名:"))
        self._name_edit = QtWidgets.QLineEdit()
        self._name_edit.setPlaceholderText("例: foo")
        row_name.addWidget(self._name_edit)
        layout.addLayout(row_name)

        row_sel = QtWidgets.QHBoxLayout()
        row_sel.addWidget(create_label("選択:"))
        self._formula_select = QtWidgets.QComboBox()
        self._formula_select.setStyleSheet("color: black;")
        row_sel.addWidget(self._formula_select, 1)
        layout.addLayout(row_sel)

        btn_row = QtWidgets.QHBoxLayout()
        self._add_btn = QtWidgets.QPushButton("Add")
        self._del_btn = QtWidgets.QPushButton("Delete")
        self._add_btn.clicked.connect(self._on_add)
        self._del_btn.clicked.connect(self._on_delete)
        btn_row.addWidget(self._add_btn)
        btn_row.addWidget(self._del_btn)
        btn_row.addStretch()
        layout.addLayout(btn_row)

        layout.addWidget(create_label("Formula 本文:"))
        self._body = QtWidgets.QTextEdit()
        self._body.setMinimumHeight(130)
        self._body.setPlainText(self._DEFAULT_BODY)
        layout.addWidget(self._body)

        close_row = QtWidgets.QHBoxLayout()
        close_btn = QtWidgets.QPushButton("閉じる")
        close_btn.clicked.connect(self.accept)
        close_row.addStretch()
        close_row.addWidget(close_btn)
        layout.addLayout(close_row)

        self._populate_select()
        self._formula_select.currentTextChanged.connect(self._on_select_changed)
        self._body.textChanged.connect(self._on_body_edited)

    def _populate_select(self):
        self._formula_select.blockSignals(True)
        self._formula_select.clear()
        formulas = getattr(self.graph, "motion_formulas", None) or {}
        for k in formulas.keys():
            self._formula_select.addItem(k)
        self._formula_select.blockSignals(False)

    def _next_form_index(self):
        import re

        max_n = 0
        for k in getattr(self.graph, "motion_formulas", {}):
            m = re.match(r"^Form(\d+):", k)
            if m:
                max_n = max(max_n, int(m.group(1)))
        return max_n + 1

    def _save_current_body(self):
        if not self.graph or not hasattr(self.graph, "motion_formulas"):
            return
        key = self._formula_select.currentText()
        if key:
            self.graph.motion_formulas[key] = self._body.toPlainText()

    def _on_select_changed(self, key):
        if not self.graph or not hasattr(self.graph, "motion_formulas"):
            return
        self._body.blockSignals(True)
        if key and key in self.graph.motion_formulas:
            self._body.setPlainText(self.graph.motion_formulas[key])
        else:
            self._body.setPlainText(self._DEFAULT_BODY)
        self._body.blockSignals(False)

    def _on_body_edited(self):
        self._save_current_body()

    def _on_add(self):
        if not self.graph or not hasattr(self.graph, "motion_formulas"):
            return
        name = self._name_edit.text().strip()
        if not name:
            return
        self._save_current_body()
        idx = self._next_form_index()
        key = f"Form{idx}:{name}"
        while key in self.graph.motion_formulas:
            idx += 1
            key = f"Form{idx}:{name}"
        self.graph.motion_formulas[key] = self._DEFAULT_BODY
        self._populate_select()
        i = self._formula_select.findText(key)
        if i >= 0:
            self._formula_select.setCurrentIndex(i)
        self._name_edit.clear()

    def _on_delete(self):
        if not self.graph or not hasattr(self.graph, "motion_formulas"):
            return
        key = self._formula_select.currentText()
        if not key or key not in self.graph.motion_formulas:
            return
        del self.graph.motion_formulas[key]
        self._populate_select()
        if self._formula_select.count() > 0:
            self._formula_select.setCurrentIndex(0)
            self._on_select_changed(self._formula_select.currentText())
        else:
            self._body.blockSignals(True)
            self._body.setPlainText(self._DEFAULT_BODY)
            self._body.blockSignals(False)

    def done(self, code):
        self._save_current_body()
        super(MotionFormsDialog, self).done(code)


# =============================================================================
# Dialog Style Constants
# =============================================================================
# Shared background for view modals (Joint Sliders, Pose Duration labels, etc.)
_VIEW_MODAL_PANEL_BG = "#e8e8e8"

# Style for combos in main window (prevent white-on-white text)
_MAIN_WINDOW_COMBO_TEXT_STYLE = (
    "QComboBox { color: black; } QComboBox QAbstractItemView { color: black; }"
)


# =============================================================================
# JumpEditDialog
# =============================================================================
class JumpEditDialog(QtWidgets.QDialog):
    """JumpNode: ジャンプ先アクション or 関数を選択。"""

    def __init__(self, graph, node, parent=None):
        super(JumpEditDialog, self).__init__(parent)
        self.graph = graph
        self.node = node
        self.setWindowTitle("Jump")
        self.setModal(True)
        self.setMinimumWidth(280)

        root = QtWidgets.QVBoxLayout(self)
        root.setSpacing(8)

        # Type selector row
        type_row = QtWidgets.QHBoxLayout()
        type_label = QtWidgets.QLabel("Type:")
        type_label.setStyleSheet("color: black;")
        type_row.addWidget(type_label)
        self._radio_action = QtWidgets.QRadioButton("Action")
        self._radio_action.setStyleSheet("color: black;")
        self._radio_func = QtWidgets.QRadioButton("Function")
        self._radio_func.setStyleSheet("color: black;")
        type_row.addWidget(self._radio_action)
        type_row.addWidget(self._radio_func)
        type_row.addStretch()
        root.addLayout(type_row)

        # Action combo
        self._action_widget = QtWidgets.QWidget()
        action_layout = QtWidgets.QVBoxLayout(self._action_widget)
        action_layout.setContentsMargins(0, 0, 0, 0)
        action_layout.setSpacing(2)
        action_lbl = QtWidgets.QLabel("Jump To Action:")
        action_lbl.setStyleSheet("color: black;")
        action_layout.addWidget(action_lbl)
        self.combo = QtWidgets.QComboBox()
        self.combo.setStyleSheet(_MAIN_WINDOW_COMBO_TEXT_STYLE)
        mas = getattr(graph, "motion_action_state", None) or {"items": [{}]}
        items = mas.get("items", [{}])
        if not items:
            items = [{}]
        for i, item in enumerate(items):
            t = (item.get("title") or "").strip()
            label = f"Action_{i + 1}:{t}" if t else f"Action_{i + 1}:"
            self.combo.addItem(label, i)
        idx = int(getattr(node, "jump_target_action_index", 0))
        idx = max(0, min(idx, self.combo.count() - 1))
        self.combo.setCurrentIndex(idx)
        action_layout.addWidget(self.combo)
        root.addWidget(self._action_widget)

        # Function combo
        self._func_widget = QtWidgets.QWidget()
        func_layout = QtWidgets.QVBoxLayout(self._func_widget)
        func_layout.setContentsMargins(0, 0, 0, 0)
        func_layout.setSpacing(2)
        func_lbl = QtWidgets.QLabel("Call Function:")
        func_lbl.setStyleSheet("color: black;")
        func_layout.addWidget(func_lbl)
        self._func_combo = QtWidgets.QComboBox()
        self._func_combo.setStyleSheet(_MAIN_WINDOW_COMBO_TEXT_STYLE)
        self._populate_func_combo()
        func_layout.addWidget(self._func_combo)
        hint = QtWidgets.QLabel("Return value (int) = target action index")
        hint.setStyleSheet("color: gray; font-size: 10px;")
        func_layout.addWidget(hint)
        root.addWidget(self._func_widget)

        bb = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel
        )
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        root.addWidget(bb)

        # Init state
        current_type = getattr(node, "jump_type", "action")
        if current_type == "function":
            self._radio_func.setChecked(True)
        else:
            self._radio_action.setChecked(True)
        self._update_visibility()

        self._radio_action.toggled.connect(self._update_visibility)
        self._radio_func.toggled.connect(self._update_visibility)

    def _populate_func_combo(self):
        try:
            from LegacyMotionEditor_CodeEditor import get_function_names
        except ImportError:
            return
        code = getattr(self.graph, "project_code", "") or ""
        names = get_function_names(code)
        self._func_combo.clear()
        if not names:
            self._func_combo.addItem("(no functions defined)", "")
        else:
            for n in names:
                self._func_combo.addItem(n, n)
        # Pre-select current function
        current_func = getattr(self.node, "jump_target_function", "") or ""
        for i in range(self._func_combo.count()):
            if self._func_combo.itemData(i) == current_func:
                self._func_combo.setCurrentIndex(i)
                break

    def _update_visibility(self):
        is_action = self._radio_action.isChecked()
        self._action_widget.setVisible(is_action)
        self._func_widget.setVisible(not is_action)
        self.adjustSize()

    def accept(self):
        if self._radio_action.isChecked():
            self.node.jump_type = "action"
            self.node.jump_target_function = ""
            self.node.jump_target_action_index = int(
                self.combo.currentData() if self.combo.currentData() is not None
                else self.combo.currentIndex()
            )
        else:
            self.node.jump_type = "function"
            data = self._func_combo.currentData()
            self.node.jump_target_function = data if data else ""
        self.node.refresh_body_text()
        super(JumpEditDialog, self).accept()


# =============================================================================
# AddDefineShellDialog
# =============================================================================
class AddDefineShellDialog(QtWidgets.QDialog):
    """DefineNode 編集。Archived の要素のみ。見たしは User Value 同様（親は QMainWindow・ウィジェット単位の色指定のみ）。"""

    def __init__(self, graph, define_node, parent=None):
        super(AddDefineShellDialog, self).__init__(parent)
        self.graph = graph
        self.define_node = define_node
        self.setWindowTitle("Add Define")
        self.setModal(True)
        self.resize(480, 320)
        # ダイアログ全体 QSS / Archived のパレットは使わない（グラフ親との差分の元になるため）

        layout = QtWidgets.QVBoxLayout(self)

        row_uv = QtWidgets.QHBoxLayout()
        lbl_target = QtWidgets.QLabel("Target:")
        lbl_target.setStyleSheet("color: black;")
        row_uv.addWidget(lbl_target)
        self._uv_combo = QtWidgets.QComboBox()
        self._uv_combo.setStyleSheet("color: black;")
        for i in range(USER_VALUE_SESSION_COUNT):
            self._uv_combo.addItem(f"UserVal_{i}", i)
        row_uv.addWidget(self._uv_combo, 1)
        layout.addLayout(row_uv)

        row_val = QtWidgets.QHBoxLayout()
        lbl_value = QtWidgets.QLabel("Value:")
        lbl_value.setStyleSheet("color: black;")
        row_val.addWidget(lbl_value)
        self._value_edit = QtWidgets.QLineEdit()
        self._value_edit.setStyleSheet(
            "QLineEdit { color: black; padding-left: 3px; padding-top: 0px; padding-bottom: 0px; }"
        )
        self._value_edit.setValidator(
            QtGui.QIntValidator(-32768, 32767, self._value_edit)
        )
        self._value_edit.setFixedWidth(100)
        hint = QtWidgets.QLabel("(-32768 ~ 32767)")
        hint.setStyleSheet("color: gray;")
        row_val.addWidget(self._value_edit)
        row_val.addWidget(hint)
        row_val.addStretch()
        layout.addLayout(row_val)

        lbl_memo = QtWidgets.QLabel("Memo:")
        lbl_memo.setStyleSheet("color: black;")
        layout.addWidget(lbl_memo)
        self._memo = QtWidgets.QPlainTextEdit()
        self._memo.setStyleSheet("color: black;")
        self._memo.setPlaceholderText("メモ（任意）")
        self._memo.setFixedHeight(80)
        layout.addWidget(self._memo)

        bbox = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok
            | QtWidgets.QDialogButtonBox.StandardButton.Cancel
        )
        bbox.accepted.connect(self._on_accept)
        bbox.rejected.connect(self.reject)
        layout.addWidget(bbox)

        self._load_from_node()

    def _load_from_node(self):
        n = self.define_node
        idx = max(0, min(USER_VALUE_SESSION_COUNT - 1, int(getattr(n, "define_uv_index", 0))))
        self._uv_combo.setCurrentIndex(idx)
        self._memo.setPlainText(getattr(n, "define_memo", "") or "")
        v = int(getattr(n, "define_literal", 0))
        v = max(-32768, min(32767, v))
        self._value_edit.setText(str(v))

    def _on_accept(self):
        n = self.define_node
        n.define_uv_index = int(self._uv_combo.currentData())
        n.define_memo = self._memo.toPlainText()
        n.define_kind = "literal"
        txt = self._value_edit.text().strip()
        try:
            v = int(txt) if txt else 0
        except ValueError:
            v = 0
        n.define_literal = max(-32768, min(32767, v))
        n.define_register_name = ""
        uv = n.define_uv_index
        memo_short = (n.define_memo or "").strip().replace("\n", " ")[:20]
        label = f"define UserVal_{uv}"
        if memo_short:
            label = f"{label} {memo_short}"
        n.set_name(label)
        # Update user_value_session with the new value
        slots = normalize_user_value_session(getattr(self.graph, "user_value_session", None))
        slots[uv] = {"kind": "literal", "value": n.define_literal}
        self.graph.user_value_session = slots
        # Refresh ValueList if open
        ValueListDialog.refresh_active()
        self.accept()


# =============================================================================
# WaitEditDialog
# =============================================================================
class WaitEditDialog(QtWidgets.QDialog):
    """WaitNode 編集ダイアログ（Name + Frames）"""

    def __init__(self, graph, wait_node, fps=100, parent=None):
        super(WaitEditDialog, self).__init__(parent)
        self.graph = graph
        self.wait_node = wait_node
        self._fps = max(1, float(fps))
        self.setWindowTitle("Wait")
        self.setModal(True)
        self.resize(320, 160)

        layout = QtWidgets.QVBoxLayout(self)

        row_name = QtWidgets.QHBoxLayout()
        lbl_name = QtWidgets.QLabel("Name:")
        lbl_name.setStyleSheet("color: black;")
        row_name.addWidget(lbl_name)
        self._name_edit = QtWidgets.QLineEdit()
        self._name_edit.setStyleSheet(
            "QLineEdit { color: black; padding-left: 3px; }"
        )
        row_name.addWidget(self._name_edit, 1)
        layout.addLayout(row_name)

        row_frames = QtWidgets.QHBoxLayout()
        lbl_frames = QtWidgets.QLabel("Frames:")
        lbl_frames.setStyleSheet("color: black;")
        row_frames.addWidget(lbl_frames)
        self._frames_spin = QtWidgets.QSpinBox()
        self._frames_spin.setStyleSheet("QSpinBox { color: black; }")
        self._frames_spin.setMinimum(0)
        self._frames_spin.setMaximum(99999)
        self._frames_spin.setFixedWidth(80)
        row_frames.addWidget(self._frames_spin)
        self._dur_label = QtWidgets.QLabel("")
        self._dur_label.setStyleSheet("color: gray;")
        row_frames.addWidget(self._dur_label)
        row_frames.addStretch()
        layout.addLayout(row_frames)

        layout.addStretch()

        bbox = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok
            | QtWidgets.QDialogButtonBox.StandardButton.Cancel
        )
        bbox.accepted.connect(self._on_accept)
        bbox.rejected.connect(self.reject)
        layout.addWidget(bbox)

        self._frames_spin.valueChanged.connect(self._update_dur_label)
        self._load_from_node()

    def _update_dur_label(self, frames):
        dur = frames / self._fps
        self._dur_label.setText(f"= {dur:.3f} s")

    def _load_from_node(self):
        n = self.wait_node
        self._name_edit.setText(getattr(n, "wait_name", "") or "")
        frames = max(0, int(getattr(n, "frames", 0)))
        self._frames_spin.setValue(frames)
        self._update_dur_label(frames)

    def _on_accept(self):
        n = self.wait_node
        name = self._name_edit.text().strip() or "wait"
        frames = max(0, self._frames_spin.value())
        dur = frames / self._fps
        n.wait_name = name
        n.frames = frames
        n.duration = dur
        n.set_name(name)
        n.refresh_body_text()
        self.accept()


# =============================================================================
# BranchingDialog
# =============================================================================
class BranchingDialog(QtWidgets.QDialog):
    """Add Branching ダイアログ"""

    def __init__(self, graph, target_node, parent=None):
        super(BranchingDialog, self).__init__(parent)
        self.graph = graph
        self.target_node = target_node
        self.setWindowTitle("Branching")
        self.setModal(True)
        self.resize(520, 260)

        layout = QtWidgets.QVBoxLayout(self)
        name_row = QtWidgets.QHBoxLayout()
        name_lbl = QtWidgets.QLabel("Name:")
        name_lbl.setStyleSheet("font-weight: bold; font-size: 13px;")
        name_lbl.setFixedWidth(48)
        name_row.addWidget(name_lbl)
        self.name_edit = QtWidgets.QLineEdit()
        self.name_edit.setText(self.target_node.name() if self.target_node else "")
        self.name_edit.textChanged.connect(self._on_name_changed)
        name_row.addWidget(self.name_edit)
        layout.addLayout(name_row)

        if hasattr(self.target_node, "enable_branching_output"):
            self.target_node.branching_enabled = True
            self.target_node.enable_branching_output()

        branching_layout = QtWidgets.QVBoxLayout()
        branching_layout.setContentsMargins(8, 4, 0, 4)

        condition_layout = QtWidgets.QHBoxLayout()
        self.branch_uv_if_radio = QtWidgets.QRadioButton()
        self.branch_uv_if_radio.setToolTip("UserVal 比較条件を使う")
        condition_layout.addWidget(self.branch_uv_if_radio)
        condition_layout.addWidget(self._create_branch_label("IF"))
        self.branch_left_combo = self._create_branch_register_combo("left", "UserVal_0")
        self.branch_op_combo = QtWidgets.QComboBox()
        self.branch_op_combo.addItems(list(_BRANCH_IF_OPERATOR_CHOICES))
        self.branch_op_combo.setStyleSheet(
            "QComboBox { color: #141414; }"
            "QComboBox QAbstractItemView { color: #141414; background: #ffffff; }"
        )
        self.branch_right_combo = self._create_branch_register_combo("right", "UserVal_1")
        self.branch_left_combo.currentTextChanged.connect(self._apply_branching_condition)
        self.branch_op_combo.currentTextChanged.connect(self._apply_branching_condition)
        self.branch_right_combo.currentTextChanged.connect(self._apply_branching_condition)
        condition_layout.addWidget(self.branch_left_combo)
        condition_layout.addWidget(self.branch_op_combo)
        condition_layout.addWidget(self.branch_right_combo)
        condition_layout.addStretch()
        branching_layout.addLayout(condition_layout)

        formula_if_layout = QtWidgets.QHBoxLayout()
        self.branch_formula_if_radio = QtWidgets.QRadioButton()
        self.branch_formula_if_radio.setToolTip("Formula 条件を使う")
        formula_if_layout.addWidget(self.branch_formula_if_radio)
        formula_if_layout.addWidget(self._create_branch_label("IF"))
        self.branch_formula_combo = QtWidgets.QComboBox()
        self.branch_formula_combo.setStyleSheet(
            "QComboBox { color: #141414; }"
            "QComboBox QAbstractItemView { color: #141414; background: #ffffff; }"
        )
        self.branch_formula_combo.currentTextChanged.connect(
            self._apply_branch_formula_selection
        )
        formula_if_layout.addWidget(self.branch_formula_combo)
        self.branch_formula_is_true = QtWidgets.QLabel("is True")
        formula_if_layout.addWidget(self.branch_formula_is_true)
        self.branch_forms_button = QtWidgets.QPushButton("Forms")
        self.branch_forms_button.clicked.connect(self._open_forms_dialog)
        if not self.graph:
            self.branch_forms_button.setEnabled(False)
        formula_if_layout.addWidget(self.branch_forms_button)
        formula_if_layout.addStretch()
        branching_layout.addLayout(formula_if_layout)

        # PAD mode row: IF PAD || [button combo]
        pad_if_layout = QtWidgets.QHBoxLayout()
        self.branch_pad_if_radio = QtWidgets.QRadioButton()
        self.branch_pad_if_radio.setToolTip("Branch when a PAD button is pressed")
        pad_if_layout.addWidget(self.branch_pad_if_radio)
        pad_if_layout.addWidget(self._create_branch_label("IF PAD"))
        pad_if_layout.addWidget(self._create_branch_label("||"))
        self.branch_pad_button_combo = QtWidgets.QComboBox()
        self.branch_pad_button_combo.addItems(list(PAD_IF_BUTTON_CHOICES))
        self.branch_pad_button_combo.setStyleSheet(
            "QComboBox { color: #141414; }"
            "QComboBox QAbstractItemView { color: #141414; background: #ffffff; }"
        )
        self.branch_pad_button_combo.currentTextChanged.connect(self._apply_branch_pad_condition)
        pad_if_layout.addWidget(self.branch_pad_button_combo)
        pad_if_layout.addStretch()
        branching_layout.addLayout(pad_if_layout)

        # PAD analog mode row: IF PAD [axis] [>= / <=] [threshold spinbox]
        _combo_ss = (
            "QComboBox { color: #141414; }"
            "QComboBox QAbstractItemView { color: #141414; background: #ffffff; }"
        )
        pad_analog_layout = QtWidgets.QHBoxLayout()
        self.branch_pad_analog_if_radio = QtWidgets.QRadioButton()
        self.branch_pad_analog_if_radio.setToolTip("PAD analog axis threshold condition")
        pad_analog_layout.addWidget(self.branch_pad_analog_if_radio)
        pad_analog_layout.addWidget(self._create_branch_label("IF PAD"))
        self.branch_pad_analog_axis_combo = QtWidgets.QComboBox()
        self.branch_pad_analog_axis_combo.addItems(list(PAD_IF_ANALOG_AXIS_CHOICES))
        self.branch_pad_analog_axis_combo.setStyleSheet(_combo_ss)
        pad_analog_layout.addWidget(self.branch_pad_analog_axis_combo)
        self.branch_pad_analog_op_combo = QtWidgets.QComboBox()
        self.branch_pad_analog_op_combo.addItems(list(PAD_IF_ANALOG_OP_CHOICES))
        self.branch_pad_analog_op_combo.setStyleSheet(_combo_ss)
        pad_analog_layout.addWidget(self.branch_pad_analog_op_combo)
        self.branch_pad_analog_threshold_spin = QtWidgets.QSpinBox()
        self.branch_pad_analog_threshold_spin.setRange(-127, 127)
        self.branch_pad_analog_threshold_spin.setValue(0)
        self.branch_pad_analog_threshold_spin.setFixedWidth(70)
        pad_analog_layout.addWidget(self.branch_pad_analog_threshold_spin)
        pad_analog_layout.addStretch()
        branching_layout.addLayout(pad_analog_layout)

        self.branch_pad_analog_axis_combo.currentTextChanged.connect(self._on_analog_axis_changed)
        self.branch_pad_analog_op_combo.currentTextChanged.connect(self._apply_branch_pad_analog_condition)
        self.branch_pad_analog_threshold_spin.valueChanged.connect(self._apply_branch_pad_analog_condition)

        self._branch_if_mode_group = QtWidgets.QButtonGroup(self)
        self._branch_if_mode_group.setExclusive(True)
        self._branch_if_mode_group.addButton(self.branch_uv_if_radio, 0)
        self._branch_if_mode_group.addButton(self.branch_formula_if_radio, 1)
        self._branch_if_mode_group.addButton(self.branch_pad_if_radio, 2)
        self._branch_if_mode_group.addButton(self.branch_pad_analog_if_radio, 3)
        self._branch_if_mode_group.buttonClicked.connect(
            lambda _btn: self._apply_branch_if_ui_state()
        )
        self.branch_uv_if_radio.setChecked(True)

        to_layout = QtWidgets.QHBoxLayout()
        self.branch_to_label = self._create_branch_label("Then")
        self.branch_to_dot = self._create_color_dot("#EB5757")
        to_layout.addWidget(self.branch_to_label)
        to_layout.addWidget(self.branch_to_dot)
        to_layout.addStretch()
        branching_layout.addLayout(to_layout)

        otherwise_layout = QtWidgets.QHBoxLayout()
        self.branch_otherwise_label = self._create_branch_label("Else")
        self.branch_otherwise_dot = self._create_color_dot("#2F80ED")
        self.branch_swap_button = QtWidgets.QPushButton("swap")
        self.branch_swap_button.clicked.connect(self._swap_branch_outputs)
        otherwise_layout.addWidget(self.branch_otherwise_label)
        otherwise_layout.addWidget(self.branch_otherwise_dot)
        otherwise_layout.addWidget(self.branch_swap_button)
        otherwise_layout.addStretch()
        branching_layout.addLayout(otherwise_layout)

        layout.addLayout(branching_layout)

        bbox = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Close
        )
        bbox.rejected.connect(self.reject)
        layout.addWidget(bbox)

        self._sync_formula_combo_from_graph()
        self._load_branch_from_node(self.target_node)
        self._apply_branch_if_ui_state()
        self._update_branch_swap_ui()

    def reject(self):
        self._refresh_pose_inspector_if_needed()
        super(BranchingDialog, self).reject()

    def accept(self):
        self._refresh_pose_inspector_if_needed()
        super(BranchingDialog, self).accept()

    def _refresh_pose_inspector_if_needed(self):
        w = getattr(self.graph, "joint_editor", None)
        if w and getattr(w, "current_pose_node", None) is self.target_node:
            try:
                w._refresh_pose_meta_row()
            except Exception:
                pass

    def _on_name_changed(self, text):
        if self.target_node and text.strip():
            self.target_node.set_name(text.strip())

    def keyPressEvent(self, event):
        # Prevent Enter/Return from triggering autoDefault buttons (e.g. Forms)
        if event.key() in (QtCore.Qt.Key.Key_Return, QtCore.Qt.Key.Key_Enter):
            return
        super().keyPressEvent(event)

    def _create_branch_label(self, text):
        label = QtWidgets.QLabel(text)
        return label

    def _branch_if_register_choice_list(self, side):
        items = load_branch_register_items_for_side(side)
        if not items:
            items = [f"UserVal_{i}" for i in range(0, USER_VALUE_SESSION_COUNT)]
        # Append Pad registers (dynamic input values)
        pad_registers = list(PAD_REGISTER_VALUES.keys())
        # Append fixed PS3 button bit values
        pad_button_bits = list(PAD_BUTTON_BIT_VALUES.keys())
        return items + pad_registers + pad_button_bits

    def _create_branch_register_combo(self, side, fallback_default):
        combo = QtWidgets.QComboBox()
        combo.setStyleSheet(
            "QComboBox { color: #141414; }"
            "QComboBox QAbstractItemView { color: #141414; background: #ffffff; }"
        )
        self._reset_branch_register_combo(combo, side, fallback_default)
        return combo

    def _reset_branch_register_combo(self, combo, side, fallback_default):
        items = self._branch_if_register_choice_list(side)
        combo.blockSignals(True)
        combo.clear()
        combo.addItems(items)
        pick = fallback_default if fallback_default in items else items[0]
        idx = combo.findText(pick)
        combo.setCurrentIndex(max(0, idx))
        combo.blockSignals(False)

    def _ensure_branch_register_combo_value(self, combo, value, legacy_default):
        if combo.count() == 0:
            return
        use = (value if value else legacy_default) or combo.itemText(0)
        idx = combo.findText(use)
        if idx < 0:
            combo.blockSignals(True)
            combo.insertItem(0, use)
            combo.setCurrentIndex(0)
            combo.blockSignals(False)
        else:
            combo.setCurrentIndex(idx)

    def _ensure_branch_op_combo_value(self, op):
        if self.branch_op_combo.count() == 0:
            return
        op = normalize_branch_if_op_stored(op or "==")
        idx = self.branch_op_combo.findText(op)
        if idx < 0:
            self.branch_op_combo.blockSignals(True)
            self.branch_op_combo.insertItem(0, op)
            self.branch_op_combo.setCurrentIndex(0)
            self.branch_op_combo.blockSignals(False)
        else:
            self.branch_op_combo.setCurrentIndex(idx)

    def _set_color_dot(self, dot, color):
        dot.setStyleSheet(
            f"background-color: {color}; border: 1px solid {color}; border-radius: 6px;"
        )

    def _create_color_dot(self, color):
        dot = QtWidgets.QLabel()
        dot.setFixedSize(12, 12)
        self._set_color_dot(dot, color)
        return dot

    def _sync_formula_combo_from_graph(self):
        if not getattr(self, "graph", None) or not hasattr(
            self.graph, "motion_formulas"
        ):
            return
        self.branch_formula_combo.blockSignals(True)
        cur = self.branch_formula_combo.currentText()
        self.branch_formula_combo.clear()
        for k in self.graph.motion_formulas.keys():
            self.branch_formula_combo.addItem(k)
        if self.branch_formula_combo.count() == 0:
            d = "\n".join(["(Not Available Now)"] * 5)
            self.graph.motion_formulas["Form1:foo"] = d
            self.branch_formula_combo.addItem("Form1:foo")
        idx = self.branch_formula_combo.findText(cur)
        if idx >= 0:
            self.branch_formula_combo.setCurrentIndex(idx)
        self.branch_formula_combo.blockSignals(False)

    def _open_forms_dialog(self):
        if not self.graph:
            return
        dlg = MotionFormsDialog(self.graph, parent=self)
        dlg.exec()
        self._sync_formula_combo_from_graph()

    def _load_branch_from_node(self, node):
        self._ensure_branch_register_combo_value(
            self.branch_left_combo, getattr(node, "branch_if_left", None), "UserVal_0"
        )
        self._ensure_branch_op_combo_value(getattr(node, "branch_if_op", "=="))
        self._ensure_branch_register_combo_value(
            self.branch_right_combo, getattr(node, "branch_if_right", None), "UserVal_1"
        )
        self.branch_uv_if_radio.blockSignals(True)
        self.branch_formula_if_radio.blockSignals(True)
        self.branch_pad_if_radio.blockSignals(True)
        self.branch_pad_analog_if_radio.blockSignals(True)
        uv_en = getattr(node, "branch_if_uv_enabled", True)
        form_en = getattr(node, "branch_if_formula_enabled", False)
        pad_en = getattr(node, "branch_if_pad_enabled", False)
        analog_en = getattr(node, "branch_if_pad_analog_enabled", False)
        # Ensure exactly one mode is active
        active_count = sum([uv_en, form_en, pad_en, analog_en])
        if active_count != 1:
            uv_en, form_en, pad_en, analog_en = True, False, False, False
        self.branch_uv_if_radio.setChecked(uv_en)
        self.branch_formula_if_radio.setChecked(form_en)
        self.branch_pad_if_radio.setChecked(pad_en)
        self.branch_pad_analog_if_radio.setChecked(analog_en)
        self.branch_uv_if_radio.blockSignals(False)
        self.branch_formula_if_radio.blockSignals(False)
        self.branch_pad_if_radio.blockSignals(False)
        self.branch_pad_analog_if_radio.blockSignals(False)
        self._sync_formula_combo_from_graph()
        ft = getattr(node, "branch_if_formula", "Form1:foo")
        self.branch_formula_combo.blockSignals(True)
        idx = self.branch_formula_combo.findText(ft)
        if idx >= 0:
            self.branch_formula_combo.setCurrentIndex(idx)
        else:
            self.branch_formula_combo.insertItem(0, ft)
            self.branch_formula_combo.setCurrentIndex(0)
        self.branch_formula_combo.blockSignals(False)
        # Load PAD button selection
        pad_btn = getattr(node, "branch_if_pad_button", "L1")
        self.branch_pad_button_combo.blockSignals(True)
        idx = self.branch_pad_button_combo.findText(pad_btn)
        self.branch_pad_button_combo.setCurrentIndex(max(0, idx))
        self.branch_pad_button_combo.blockSignals(False)
        # Load PAD analog selection
        analog_axis = getattr(node, "branch_if_pad_analog_axis", "Lx")
        analog_op = getattr(node, "branch_if_pad_analog_op", ">=")
        analog_thr = int(getattr(node, "branch_if_pad_analog_threshold", 0))
        lo, hi = PAD_IF_ANALOG_AXIS_RANGE.get(analog_axis, (-127, 127))
        self.branch_pad_analog_axis_combo.blockSignals(True)
        self.branch_pad_analog_op_combo.blockSignals(True)
        self.branch_pad_analog_threshold_spin.blockSignals(True)
        self.branch_pad_analog_threshold_spin.setRange(lo, hi)
        idx = self.branch_pad_analog_axis_combo.findText(analog_axis)
        self.branch_pad_analog_axis_combo.setCurrentIndex(max(0, idx))
        idx = self.branch_pad_analog_op_combo.findText(analog_op)
        self.branch_pad_analog_op_combo.setCurrentIndex(max(0, idx))
        self.branch_pad_analog_threshold_spin.setValue(max(lo, min(hi, analog_thr)))
        self.branch_pad_analog_axis_combo.blockSignals(False)
        self.branch_pad_analog_op_combo.blockSignals(False)
        self.branch_pad_analog_threshold_spin.blockSignals(False)
        self._apply_branch_if_ui_state()

    def _update_branch_swap_ui(self):
        swapped = bool(
            self.target_node
            and getattr(self.target_node, "branch_outputs_swapped", False)
        )
        if swapped:
            self._set_color_dot(self.branch_to_dot, "#2F80ED")
            self._set_color_dot(self.branch_otherwise_dot, "#EB5757")
        else:
            self._set_color_dot(self.branch_to_dot, "#EB5757")
            self._set_color_dot(self.branch_otherwise_dot, "#2F80ED")

    def _swap_branch_outputs(self):
        if not self.target_node:
            return
        self.target_node.branch_outputs_swapped = not getattr(
            self.target_node, "branch_outputs_swapped", False
        )
        if hasattr(self.target_node, "_sync_branching_port_labels"):
            self.target_node._sync_branching_port_labels()
        # Update port colors (BranchingNode uses _apply_branch_output_colors, PoseNode uses _apply_pose_output_colors)
        if hasattr(self.target_node, "_apply_branch_output_colors"):
            self.target_node._apply_branch_output_colors()
        elif hasattr(self.target_node, "_apply_pose_output_colors"):
            self.target_node._apply_pose_output_colors()
        # Update port positions
        if hasattr(self.target_node, "_do_position_outputs"):
            self.target_node._do_position_outputs()
        # Update pipe colors for connected pipes
        self._update_connected_pipe_colors()
        # Trigger view update
        view = getattr(self.target_node, "view", None)
        if view:
            view.update()
            # Also update port views explicitly
            for port_view in getattr(view, 'outputs', []):
                port_view.update()
            # Update scene to ensure all changes are rendered
            scene = view.scene()
            if scene:
                scene.update()
        self._update_branch_swap_ui()

    def _update_connected_pipe_colors(self):
        """Update the colors of all pipes connected to the target node's output ports"""
        if not self.target_node:
            return
        for port in self.target_node.output_ports():
            # Get connected pipes from port model
            connected = getattr(port, 'connected_pipes', None)
            if connected:
                for pipe_model in connected:
                    # Get the pipe view (CustomPipe/PipeItem)
                    pipe_view = getattr(pipe_model, 'view', None)
                    if pipe_view and hasattr(pipe_view, '_update_pipe_color'):
                        pipe_view._update_pipe_color(port)
                        pipe_view.update()

    def _apply_branching_condition(self, *args):
        if self.target_node:
            self.target_node.branch_if_left = self.branch_left_combo.currentText()
            self.target_node.branch_if_op = normalize_branch_if_op_stored(
                self.branch_op_combo.currentText()
            )
            self.target_node.branch_if_right = self.branch_right_combo.currentText()

    def _apply_branch_if_ui_state(self, *args):
        uv_on = self.branch_uv_if_radio.isChecked()
        form_on = self.branch_formula_if_radio.isChecked()
        pad_on = self.branch_pad_if_radio.isChecked()
        analog_on = self.branch_pad_analog_if_radio.isChecked()
        self.branch_left_combo.setEnabled(uv_on)
        self.branch_op_combo.setEnabled(uv_on)
        self.branch_right_combo.setEnabled(uv_on)
        self.branch_formula_combo.setEnabled(form_on)
        self.branch_formula_is_true.setEnabled(form_on)
        self.branch_forms_button.setEnabled(bool(self.graph))
        self.branch_pad_button_combo.setEnabled(pad_on)
        self.branch_pad_analog_axis_combo.setEnabled(analog_on)
        self.branch_pad_analog_op_combo.setEnabled(analog_on)
        self.branch_pad_analog_threshold_spin.setEnabled(analog_on)
        if self.target_node:
            self.target_node.branch_if_uv_enabled = uv_on
            self.target_node.branch_if_formula_enabled = form_on
            self.target_node.branch_if_pad_enabled = pad_on
            self.target_node.branch_if_pad_analog_enabled = analog_on

    def _apply_branch_formula_selection(self, text):
        if self.target_node:
            self.target_node.branch_if_formula = text

    def _apply_branch_pad_condition(self, text):
        if self.target_node:
            self.target_node.branch_if_pad_button = text

    def _on_analog_axis_changed(self, axis):
        lo, hi = PAD_IF_ANALOG_AXIS_RANGE.get(axis, (-127, 127))
        cur = self.branch_pad_analog_threshold_spin.value()
        self.branch_pad_analog_threshold_spin.setRange(lo, hi)
        self.branch_pad_analog_threshold_spin.setValue(max(lo, min(hi, cur)))
        self._apply_branch_pad_analog_condition()

    def _apply_branch_pad_analog_condition(self, *args):
        if self.target_node:
            self.target_node.branch_if_pad_analog_axis = self.branch_pad_analog_axis_combo.currentText()
            self.target_node.branch_if_pad_analog_op = self.branch_pad_analog_op_combo.currentText()
            self.target_node.branch_if_pad_analog_threshold = self.branch_pad_analog_threshold_spin.value()


# Backward compatibility alias
BranchingShellDialog = BranchingDialog


# =============================================================================
# JointSettingsDialog
# =============================================================================
class JointSettingsDialog(QtWidgets.QDialog):
    """JointEditorの表示名と最高速度を編集するダイアログ

    User Value / Define と同様、親は QMainWindow（graph.widget.window）、
    全体 QSS は使わず QLabel＋コントロールへ個別指定する。
    """

    def __init__(self, joint_editor, parent=None):
        super(JointSettingsDialog, self).__init__(parent)
        self.joint_editor = joint_editor
        self.rows = {}
        self.setWindowTitle("Joint Settings")
        self.setMinimumWidth(920)
        self.setMinimumHeight(520)
        self.resize(960, 520)
        self.setModal(True)
        if getattr(joint_editor, "always_on_top", False):
            self.setWindowFlags(self.windowFlags() | QtCore.Qt.WindowStaysOnTopHint)
        self._setup_ui()

    def _setup_ui(self):
        layout = QtWidgets.QVBoxLayout(self)

        _le_style = (
            "QLineEdit { color: black; padding-left: 3px; padding-top: 0px; "
            "padding-bottom: 0px; }"
        )
        _num_style = "color: black;"

        bulk_title = QtWidgets.QLabel("Bulk Settings")
        bulk_title.setStyleSheet("color: black; font-weight: bold;")
        layout.addWidget(bulk_title)

        bulk_layout = QtWidgets.QHBoxLayout()
        self._all_preset_combos = []
        for group_key, group_title in (("L", "Line-L"), ("R", "Line-R"), ("C", "Line-C")):
            g_label = QtWidgets.QLabel(group_title)
            g_label.setStyleSheet("color: black;")
            bulk_layout.addWidget(g_label)
            combo = QtWidgets.QComboBox()
            combo.setStyleSheet("color: black;")
            self._fill_joint_preset_combo(combo)
            # activated は OS/スタイルによって発火しないことがあるため currentIndexChanged を使う
            combo.currentIndexChanged.connect(
                lambda idx, g=group_key, c=combo: self._on_bulk_preset_index_changed(
                    g, c, idx))
            bulk_layout.addWidget(combo)
            self._all_preset_combos.append(combo)
        bulk_layout.addStretch()
        btn_mirror_rr = QtWidgets.QPushButton("Mirror right-side yaw and roll.")
        btn_mirror_rr.setStyleSheet("color: black;")
        btn_mirror_rr.setToolTip(
            "Set Rev for all joints whose original name starts with r_ "
            "and ends with _xr or _zy."
        )
        btn_mirror_rr.clicked.connect(self._on_mirror_right_yaw_roll)
        bulk_layout.addWidget(btn_mirror_rr)
        layout.addLayout(bulk_layout)

        preset_title = QtWidgets.QLabel("Servo Model Preset")
        preset_title.setStyleSheet("color: black; font-weight: bold;")
        layout.addWidget(preset_title)
        pl_row = QtWidgets.QHBoxLayout()
        self._joint_preset_combo = QtWidgets.QComboBox()
        self._joint_preset_combo.setStyleSheet("color: black;")
        self._joint_preset_combo.setMinimumWidth(150)
        pl_row.addWidget(self._joint_preset_combo)
        btn_apply_all = QtWidgets.QPushButton("Apply to All")
        btn_apply_all.setStyleSheet("color: black;")
        btn_apply_all.setToolTip("Apply selected preset to all joints.")
        btn_apply_all.clicked.connect(self._on_apply_preset_to_all)
        pl_row.addWidget(btn_apply_all)
        lbl_nm = QtWidgets.QLabel("Add New:")
        lbl_nm.setStyleSheet("color: black;")
        pl_row.addWidget(lbl_nm)
        self._joint_preset_name_edit = QtWidgets.QLineEdit()
        self._joint_preset_name_edit.setPlaceholderText("e.g. MyServo")
        self._joint_preset_name_edit.setStyleSheet(_le_style)
        self._joint_preset_name_edit.setFixedWidth(100)
        pl_row.addWidget(self._joint_preset_name_edit)
        lbl_sp = QtWidgets.QLabel("deg/s:")
        lbl_sp.setStyleSheet("color: black;")
        pl_row.addWidget(lbl_sp)
        self._joint_preset_speed_spin = QtWidgets.QDoubleSpinBox()
        self._joint_preset_speed_spin.setRange(0.01, 999999.99)
        self._joint_preset_speed_spin.setDecimals(2)
        self._joint_preset_speed_spin.setValue(DEFAULT_JOINT_SPEED)
        self._joint_preset_speed_spin.setStyleSheet(_num_style)
        self._joint_preset_speed_spin.setFixedWidth(80)
        pl_row.addWidget(self._joint_preset_speed_spin)
        btn_add = QtWidgets.QPushButton("Add")
        btn_add.setStyleSheet("color: black;")
        btn_add.clicked.connect(self._on_joint_preset_add)
        pl_row.addWidget(btn_add)
        btn_rm = QtWidgets.QPushButton("Remove")
        btn_rm.setStyleSheet("color: black;")
        btn_rm.clicked.connect(self._on_joint_preset_remove)
        pl_row.addWidget(btn_rm)
        pl_row.addStretch()
        layout.addLayout(pl_row)
        self._refresh_joint_preset_list()

        rename_title = QtWidgets.QLabel("Canonical Name Conversion")
        rename_title.setStyleSheet("color: black; font-weight: bold;")
        layout.addWidget(rename_title)

        rename_row = QtWidgets.QHBoxLayout()
        self._last_rename_plan = None
        btn_convert = QtWidgets.QPushButton("Normalize Joint Names")
        btn_convert.setStyleSheet("color: black;")
        btn_convert.setToolTip(
            "Convert loaded joint names to canonical short servo names (e.g. l_shoulder_yp)."
        )
        btn_convert.clicked.connect(self._on_convert_joint_names)
        rename_row.addWidget(btn_convert)

        self._btn_overwrite_model = QtWidgets.QPushButton("Overwrite Model File")
        self._btn_overwrite_model.setStyleSheet("color: black;")
        self._btn_overwrite_model.setEnabled(False)
        self._btn_overwrite_model.setToolTip(
            "Write converted joint names back to the loaded URDF/MJCF file (.bak backup created)."
        )
        self._btn_overwrite_model.clicked.connect(self._on_overwrite_model_file)
        rename_row.addWidget(self._btn_overwrite_model)
        rename_row.addStretch()
        layout.addLayout(rename_row)

        scroll = QtWidgets.QScrollArea(self)
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)
        content = QtWidgets.QWidget()
        grid = QtWidgets.QGridLayout(content)
        grid.setContentsMargins(4, 4, 4, 4)
        grid.setHorizontalSpacing(8)
        grid.setVerticalSpacing(4)

        headers = ("Line", "Original Name", "Display Name", "Dir", "deg/s", "", "rad/s", "", "Preset")
        for col, text in enumerate(headers):
            label = QtWidgets.QLabel(text)
            label.setStyleSheet("color: black; font-weight: bold;")
            grid.addWidget(label, 0, col)

        robot_model = self.joint_editor.robot_model
        row = 1
        if robot_model:
            for jname in sorted(robot_model.joint_order):
                group = self.joint_editor._joint_group(jname)
                current = self.joint_editor.joint_settings.get(jname, {})
                display_name = current.get("display_name", jname)
                # migration: old "dir" field → "rev" bool
                if "rev" in current:
                    rev = bool(current["rev"])
                elif "dir" in current:
                    rev = current["dir"] == "CCW"
                else:
                    rev = jname.startswith("r_") and jname.endswith("_xr")
                max_speed_rads = float(current.get("max_speed_rad_s", DEFAULT_JOINT_SPEED))
                max_speed_degs = math.degrees(max_speed_rads)

                group_label = QtWidgets.QLabel(f"Line-{group}")
                group_label.setStyleSheet("color: black;")
                original_label = QtWidgets.QLabel(jname)
                original_label.setStyleSheet("color: black;")
                name_edit = QtWidgets.QLineEdit(display_name)
                name_edit.setMinimumWidth(160)
                name_edit.setAlignment(QtCore.Qt.AlignLeft)
                name_edit.setCursorPosition(0)
                name_edit.setStyleSheet(_le_style)
                name_edit.editingFinished.connect(
                    lambda edit=name_edit: edit.setCursorPosition(0))

                rev_checkbox = QtWidgets.QCheckBox("Rev")
                rev_checkbox.setStyleSheet("color: black;")
                rev_checkbox.setChecked(rev)

                speed_spin_degs = QtWidgets.QDoubleSpinBox()
                speed_spin_degs.setStyleSheet(_num_style)
                speed_spin_degs.setRange(0.0, 999999.99)
                speed_spin_degs.setDecimals(2)
                speed_spin_degs.setSingleStep(1.0)
                speed_spin_degs.setValue(max_speed_degs)
                speed_spin_degs.setFixedWidth(90)

                unit_label_degs = QtWidgets.QLabel("deg/s")
                unit_label_degs.setStyleSheet("color: black;")

                speed_spin_rads = QtWidgets.QDoubleSpinBox()
                speed_spin_rads.setStyleSheet(_num_style)
                speed_spin_rads.setRange(0.0, 999999.99)
                speed_spin_rads.setDecimals(4)
                speed_spin_rads.setSingleStep(0.1)
                speed_spin_rads.setValue(max_speed_rads)
                speed_spin_rads.setFixedWidth(90)

                unit_label_rads = QtWidgets.QLabel("rad/s")
                unit_label_rads.setStyleSheet("color: black;")

                # Mutual conversion between deg/s and rad/s
                def on_degs_changed(val, rads_spin=speed_spin_rads):
                    rads_spin.blockSignals(True)
                    rads_spin.setValue(math.radians(val))
                    rads_spin.blockSignals(False)

                def on_rads_changed(val, degs_spin=speed_spin_degs):
                    degs_spin.blockSignals(True)
                    degs_spin.setValue(math.degrees(val))
                    degs_spin.blockSignals(False)

                speed_spin_degs.valueChanged.connect(on_degs_changed)
                speed_spin_rads.valueChanged.connect(on_rads_changed)

                preset_combo = QtWidgets.QComboBox()
                preset_combo.setStyleSheet("color: black;")
                self._fill_joint_preset_combo(preset_combo)
                self._select_preset_combo_by_name(
                    preset_combo, str(current.get("speed_preset_name", "") or ""))
                preset_combo.currentIndexChanged.connect(
                    lambda idx, spin_rads=speed_spin_rads, spin_degs=speed_spin_degs, combo=preset_combo:
                    self._on_row_preset_index_changed(spin_rads, spin_degs, combo, idx))

                grid.addWidget(group_label, row, 0)
                grid.addWidget(original_label, row, 1)
                grid.addWidget(name_edit, row, 2)
                grid.addWidget(rev_checkbox, row, 3)
                grid.addWidget(speed_spin_degs, row, 4)
                grid.addWidget(unit_label_degs, row, 5)
                grid.addWidget(speed_spin_rads, row, 6)
                grid.addWidget(unit_label_rads, row, 7)
                grid.addWidget(preset_combo, row, 8)

                self.rows[jname] = {
                    "group": group,
                    "name_edit": name_edit,
                    "rev_checkbox": rev_checkbox,
                    "speed_spin_rads": speed_spin_rads,
                    "speed_spin_degs": speed_spin_degs,
                    "preset_combo": preset_combo,
                }
                self._all_preset_combos.append(preset_combo)
                row += 1

        scroll.setWidget(content)
        layout.addWidget(scroll, stretch=1)
        self._joint_scroll = scroll
        self._joint_grid_host = content

        bbox = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok
            | QtWidgets.QDialogButtonBox.StandardButton.Cancel
        )
        bbox.accepted.connect(self._apply)
        bbox.rejected.connect(self.reject)
        layout.addWidget(bbox)

    def _fill_joint_preset_combo(self, combo):
        combo.blockSignals(True)
        combo.clear()
        combo.addItem("Preset", None)
        for name, speed_rads in get_joint_speed_presets():
            speed_degs = math.degrees(speed_rads)
            combo.addItem(f"{name} ({speed_degs:.1f}°/s)", (name, float(speed_rads)))
        combo.blockSignals(False)

    def _select_preset_combo_by_name(self, combo, preset_name):
        combo.blockSignals(True)
        combo.setCurrentIndex(0)
        if preset_name:
            for i in range(1, combo.count()):
                data = combo.itemData(i)
                n0, _ = _joint_preset_item_data_parts(data)
                if n0 == preset_name:
                    combo.setCurrentIndex(i)
                    break
        combo.blockSignals(False)

    def _repopulate_all_joint_preset_combos(self):
        for idx, combo in enumerate(self._all_preset_combos):
            self._fill_joint_preset_combo(combo)
            if idx < 3:
                combo.blockSignals(True)
                combo.setCurrentIndex(0)
                combo.blockSignals(False)
            else:
                jnames = sorted(self.rows.keys())
                j_i = idx - 3
                if j_i < 0 or j_i >= len(jnames):
                    continue
                jname = jnames[j_i]
                pname = self.joint_editor.joint_settings.get(jname, {}).get(
                    "speed_preset_name", "") or ""
                self._select_preset_combo_by_name(combo, str(pname))

    def _refresh_joint_preset_list(self):
        self._joint_preset_combo.clear()
        for name, speed in get_joint_speed_presets():
            self._joint_preset_combo.addItem(f"{name}  ({math.degrees(speed):.1f} deg/s)", name)

    def _on_apply_preset_to_all(self):
        preset_name = self._joint_preset_combo.currentData()
        if not preset_name:
            return
        speed_rads = None
        for name, speed in get_joint_speed_presets():
            if name == preset_name:
                speed_rads = speed
                break
        if speed_rads is None:
            return
        for row in self.rows.values():
            row["speed_spin_rads"].setValue(float(speed_rads))
            self._select_preset_combo_by_name(row["preset_combo"], preset_name)

    def _on_joint_preset_add(self):
        name = self._joint_preset_name_edit.text().strip()
        if not name:
            return
        spd = float(self._joint_preset_speed_spin.value())
        pairs = list(get_joint_speed_presets())
        pairs = [(n, v) for n, v in pairs if n != name]
        pairs.append((name, spd))
        pairs.sort(key=lambda x: (x[0].lower(), x[0]))
        save_joint_speed_presets(pairs)
        self._refresh_joint_preset_list()
        self._repopulate_all_joint_preset_combos()

    def _on_joint_preset_remove(self):
        idx = self._joint_preset_combo.currentIndex()
        if idx < 0:
            return
        rm_name = self._joint_preset_combo.currentData()
        pairs = [(n, v) for n, v in get_joint_speed_presets() if n != rm_name]
        if not pairs:
            pairs = list(JOINT_SPEED_PRESETS)
        save_joint_speed_presets(pairs)
        for jname, jset in self.joint_editor.joint_settings.items():
            if str(jset.get("speed_preset_name", "") or "") == str(rm_name):
                jset["speed_preset_name"] = ""
        self._refresh_joint_preset_list()
        self._repopulate_all_joint_preset_combos()

    def _on_row_preset_index_changed(self, speed_spin_rads, speed_spin_degs, combo, index):
        if index < 0:
            return
        self._apply_row_preset(speed_spin_rads, speed_spin_degs, combo)

    def _on_bulk_preset_index_changed(self, group, combo, index):
        if index < 0:
            return
        self._apply_bulk_preset(group, combo)

    def _apply_row_preset(self, speed_spin_rads, speed_spin_degs, combo):
        _name, speed_rads = _joint_preset_item_data_parts(combo.currentData())
        if speed_rads is None:
            return
        # Set rad/s value; deg/s updates automatically via valueChanged signal
        speed_spin_rads.setValue(float(speed_rads))

    def _apply_bulk_preset(self, group, combo):
        _name, speed_rads = _joint_preset_item_data_parts(combo.currentData())
        if speed_rads is None:
            return
        preset_index = combo.currentIndex()
        for row in self.rows.values():
            if row["group"] == group:
                row["speed_spin_rads"].setValue(float(speed_rads))
                row["preset_combo"].setCurrentIndex(preset_index)

    def _on_mirror_right_yaw_roll(self):
        targets = joints_matching_right_yaw_roll_mirror(self.rows.keys())
        if not targets:
            QtWidgets.QMessageBox.information(
                self,
                "Mirror right-side yaw and roll.",
                (
                    "No matching joints found.\n\n"
                    "Original joint names must start with 'r_' and end with "
                    "'_xr' or '_zy'."
                ),
            )
            return
        for jname in targets:
            row = self.rows.get(jname)
            if row:
                row["rev_checkbox"].setChecked(True)
        joint_lines = "\n".join(f"  {jname}" for jname in targets)
        QtWidgets.QMessageBox.information(
            self,
            "Mirror right-side yaw and roll.",
            (
                f"Complete.\n\n"
                f"Set Rev for {len(targets)} joint(s):\n"
                f"{joint_lines}\n\n"
                f"Click OK on Joint Settings to save these changes."
            ),
        )

    def _apply(self):
        settings = {}
        for jname, row in self.rows.items():
            display_name = row["name_edit"].text().strip() or jname
            pdata = row["preset_combo"].currentData()
            pname, _spd = _joint_preset_item_data_parts(pdata)
            speed_preset_name = pname if pname is not None else ""
            settings[jname] = {
                "display_name": display_name,
                "rev": row["rev_checkbox"].isChecked(),
                "max_speed_rad_s": row["speed_spin_rads"].value(),
                "speed_preset_name": speed_preset_name,
            }
        self.joint_editor.set_joint_settings(settings)
        self.accept()

    def _show_rename_preview(self, plan):
        joint_map = plan.get("joint_map", {})
        joint_unchanged_map = plan.get("joint_unchanged_map", {})
        link_map = plan.get("link_map", {})
        preserved_link_map = plan.get("preserved_link_map", {})
        link_unchanged_map = plan.get("link_unchanged_map", {})
        unresolved = plan.get("unresolved", [])
        link_unresolved = plan.get("link_unresolved", [])
        joint_ambiguous = plan.get("joint_ambiguous", [])
        link_ambiguous = plan.get("link_ambiguous", [])
        skipped_fixed = plan.get("skipped_fixed", [])

        dialog = QtWidgets.QDialog(self)
        dialog.setWindowTitle("Confirm Canonical Name Conversion")
        dialog.setMinimumWidth(720)
        layout = QtWidgets.QVBoxLayout(dialog)

        summary = QtWidgets.QLabel(
            f"Model:\n{plan.get('model_path', '')}"
        )
        summary.setStyleSheet("color: black;")
        layout.addWidget(summary)

        text = QtWidgets.QPlainTextEdit()
        text.setReadOnly(True)
        text.setStyleSheet("color: black;")
        lines = []
        joint_plan_entries = dict(joint_unchanged_map)
        joint_plan_entries.update(joint_map)
        if joint_plan_entries:
            lines.append("Joint rename plan:")
            for old_name, new_name in sorted(joint_plan_entries.items()):
                lines.append(f"  {old_name}  ->  {new_name}")
        else:
            lines.append("No convertible joints found.")

        link_plan_entries = dict(preserved_link_map)
        link_plan_entries.update(link_unchanged_map)
        link_plan_entries.update(link_map)
        if link_plan_entries:
            lines.append("")
            lines.append("Link rename plan (optional):")
            for old_name, new_name in sorted(link_plan_entries.items()):
                lines.append(f"  {old_name}  ->  {new_name}")

        if joint_ambiguous:
            lines.append("")
            lines.append("Ambiguous joints (skipped on apply):")
            for old_name, reason in joint_ambiguous:
                lines.append(f"  {old_name}: {reason}")

        if link_ambiguous:
            lines.append("")
            lines.append("Ambiguous links (skipped on apply):")
            for old_name, reason in link_ambiguous:
                lines.append(f"  {old_name}: {reason}")

        if unresolved:
            lines.append("")
            lines.append("Unresolved joints:")
            for old_name, reason in unresolved:
                lines.append(f"  {old_name}: {reason}")

        if link_unresolved:
            lines.append("")
            lines.append("Unresolved links:")
            for old_name, reason in link_unresolved:
                lines.append(f"  {old_name}: {reason}")

        if skipped_fixed:
            lines.append("")
            lines.append(f"Skipped fixed joints: {len(skipped_fixed)}")

        text.setPlainText("\n".join(lines[:120]))
        layout.addWidget(text, stretch=1)

        include_links_cb = QtWidgets.QCheckBox(
            "リンク名も変換する (Rename links too)"
        )
        include_links_cb.setStyleSheet("color: black;")
        include_links_cb.setEnabled(bool(link_map))
        if not link_map:
            include_links_cb.setToolTip("変換可能なリンクがありません。")
        layout.addWidget(include_links_cb)

        bbox = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok
            | QtWidgets.QDialogButtonBox.StandardButton.Cancel
        )
        bbox.accepted.connect(dialog.accept)
        bbox.rejected.connect(dialog.reject)
        layout.addWidget(bbox)

        if dialog.exec() != QtWidgets.QDialog.DialogCode.Accepted:
            return False, False
        return True, include_links_cb.isChecked() and bool(link_map)

    def _reload_joint_rows(self):
        if hasattr(self, "_joint_scroll") and self._joint_scroll is not None:
            old_scroll = self._joint_scroll
            parent_layout = self.layout()
            idx = parent_layout.indexOf(old_scroll)
            old_scroll.setParent(None)
            old_scroll.deleteLater()

            scroll = QtWidgets.QScrollArea(self)
            scroll.setWidgetResizable(True)
            scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)
            content = QtWidgets.QWidget()
            grid = QtWidgets.QGridLayout(content)
            grid.setContentsMargins(4, 4, 4, 4)
            grid.setHorizontalSpacing(8)
            grid.setVerticalSpacing(4)

            _le_style = (
                "QLineEdit { color: black; padding-left: 3px; padding-top: 0px; "
                "padding-bottom: 0px; }"
            )
            _num_style = "color: black;"
            headers = ("Line", "Original Name", "Display Name", "Dir", "deg/s", "", "rad/s", "", "Preset")
            for col, text in enumerate(headers):
                label = QtWidgets.QLabel(text)
                label.setStyleSheet("color: black; font-weight: bold;")
                grid.addWidget(label, 0, col)

            self.rows = {}
            robot_model = self.joint_editor.robot_model
            row = 1
            if robot_model:
                for jname in sorted(robot_model.joint_order):
                    group = self.joint_editor._joint_group(jname)
                    current = self.joint_editor.joint_settings.get(jname, {})
                    display_name = current.get("display_name", jname)
                    # migration: old "dir" field → "rev" bool
                    if "rev" in current:
                        rev = bool(current["rev"])
                    elif "dir" in current:
                        rev = current["dir"] == "CCW"
                    else:
                        rev = jname.startswith("r_") and jname.endswith("_xr")
                    max_speed_rads = float(current.get("max_speed_rad_s", DEFAULT_JOINT_SPEED))
                    max_speed_degs = math.degrees(max_speed_rads)

                    group_label = QtWidgets.QLabel(f"Line-{group}")
                    group_label.setStyleSheet("color: black;")
                    original_label = QtWidgets.QLabel(jname)
                    original_label.setStyleSheet("color: black;")
                    name_edit = QtWidgets.QLineEdit(display_name)
                    name_edit.setMinimumWidth(160)
                    name_edit.setAlignment(QtCore.Qt.AlignLeft)
                    name_edit.setCursorPosition(0)
                    name_edit.setStyleSheet(_le_style)

                    rev_checkbox = QtWidgets.QCheckBox("Rev")
                    rev_checkbox.setStyleSheet("color: black;")
                    rev_checkbox.setChecked(rev)

                    speed_spin_degs = QtWidgets.QDoubleSpinBox()
                    speed_spin_degs.setStyleSheet(_num_style)
                    speed_spin_degs.setRange(0.0, 999999.99)
                    speed_spin_degs.setDecimals(2)
                    speed_spin_degs.setSingleStep(1.0)
                    speed_spin_degs.setValue(max_speed_degs)
                    speed_spin_degs.setFixedWidth(90)

                    unit_label_degs = QtWidgets.QLabel("deg/s")
                    unit_label_degs.setStyleSheet("color: black;")

                    speed_spin_rads = QtWidgets.QDoubleSpinBox()
                    speed_spin_rads.setStyleSheet(_num_style)
                    speed_spin_rads.setRange(0.0, 999999.99)
                    speed_spin_rads.setDecimals(4)
                    speed_spin_rads.setSingleStep(0.1)
                    speed_spin_rads.setValue(max_speed_rads)
                    speed_spin_rads.setFixedWidth(90)

                    unit_label_rads = QtWidgets.QLabel("rad/s")
                    unit_label_rads.setStyleSheet("color: black;")

                    def on_degs_changed(val, rads_spin=speed_spin_rads):
                        rads_spin.blockSignals(True)
                        rads_spin.setValue(math.radians(val))
                        rads_spin.blockSignals(False)

                    def on_rads_changed(val, degs_spin=speed_spin_degs):
                        degs_spin.blockSignals(True)
                        degs_spin.setValue(math.degrees(val))
                        degs_spin.blockSignals(False)

                    speed_spin_degs.valueChanged.connect(on_degs_changed)
                    speed_spin_rads.valueChanged.connect(on_rads_changed)

                    preset_combo = QtWidgets.QComboBox()
                    preset_combo.setStyleSheet("color: black;")
                    self._fill_joint_preset_combo(preset_combo)
                    self._select_preset_combo_by_name(
                        preset_combo, str(current.get("speed_preset_name", "") or ""))
                    preset_combo.currentIndexChanged.connect(
                        lambda idx, spin_rads=speed_spin_rads, spin_degs=speed_spin_degs, combo=preset_combo:
                        self._on_row_preset_index_changed(spin_rads, spin_degs, combo, idx))

                    grid.addWidget(group_label, row, 0)
                    grid.addWidget(original_label, row, 1)
                    grid.addWidget(name_edit, row, 2)
                    grid.addWidget(rev_checkbox, row, 3)
                    grid.addWidget(speed_spin_degs, row, 4)
                    grid.addWidget(unit_label_degs, row, 5)
                    grid.addWidget(speed_spin_rads, row, 6)
                    grid.addWidget(unit_label_rads, row, 7)
                    grid.addWidget(preset_combo, row, 8)

                    self.rows[jname] = {
                        "group": group,
                        "name_edit": name_edit,
                        "rev_checkbox": rev_checkbox,
                        "speed_spin_rads": speed_spin_rads,
                        "speed_spin_degs": speed_spin_degs,
                        "preset_combo": preset_combo,
                    }
                    row += 1

            scroll.setWidget(content)
            parent_layout.insertWidget(idx, scroll)
            self._joint_scroll = scroll
            self._joint_grid_host = content

    def _on_convert_joint_names(self):
        try:
            from RobotLabelBridge import (
                apply_joint_rename_plan,
                plan_joint_rename,
            )

            plan = plan_joint_rename(self.joint_editor)
            if not (
                plan.get("joint_map")
                or plan.get("joint_unchanged_map")
                or plan.get("joint_ambiguous")
            ):
                QtWidgets.QMessageBox.warning(
                    self,
                    "Joint Name Conversion",
                    "変換可能なジョイントがありません。",
                )
                return
            confirmed, include_links = self._show_rename_preview(plan)
            if not confirmed:
                return

            stats = apply_joint_rename_plan(
                self.joint_editor,
                plan,
                include_links=include_links,
            )
            self._last_rename_plan = plan

            self._btn_overwrite_model.setEnabled(True)
            self._reload_joint_rows()

            QtWidgets.QMessageBox.information(
                self,
                "Joint Name Conversion",
                (
                    f"変換完了\n"
                    f"renamed joints: {stats['renamed_joints']}\n"
                    f"renamed links: {stats['renamed_links']}\n"
                    f"pose nodes updated: {stats['pose_nodes_updated']}\n"
                    f"skipped ambiguous joints: {stats['joint_ambiguous']}\n"
                    f"skipped ambiguous links: {stats['link_ambiguous']}\n"
                    f"unresolved joints: {stats['unresolved']}\n"
                    f"unresolved links: {stats['link_unresolved']}"
                ),
            )
        except Exception as exc:
            QtWidgets.QMessageBox.critical(
                self,
                "Joint Name Conversion",
                str(exc),
            )

    def _on_overwrite_model_file(self):
        try:
            from RobotLabelBridge import (
                overwrite_loaded_model_file,
                plan_joint_rename,
            )

            plan = self._last_rename_plan or plan_joint_rename(self.joint_editor)
            if not plan.get("joint_map") and not (
                plan.get("include_links") and plan.get("link_map")
            ):
                QtWidgets.QMessageBox.warning(
                    self,
                    "Overwrite Model File",
                    "上書きする変換マップがありません。先に Convert Joint Names を実行してください。",
                )
                return

            model_path = plan.get("model_path", "")
            link_note = (
                "\nリンク名も上書きします。"
                if plan.get("include_links") and plan.get("link_map")
                else ""
            )
            answer = QtWidgets.QMessageBox.question(
                self,
                "Overwrite Model File",
                (
                    f"次のファイルを上書きします。\n{model_path}\n\n"
                    f".bak バックアップを作成してから更新します。{link_note}よろしいですか？"
                ),
                QtWidgets.QMessageBox.StandardButton.Yes
                | QtWidgets.QMessageBox.StandardButton.No,
            )
            if answer != QtWidgets.QMessageBox.StandardButton.Yes:
                return

            backup_path = overwrite_loaded_model_file(self.joint_editor, plan)
            QtWidgets.QMessageBox.information(
                self,
                "Overwrite Model File",
                f"モデルファイルを更新しました。\nbackup: {backup_path}",
            )
        except Exception as exc:
            QtWidgets.QMessageBox.critical(
                self,
                "Overwrite Model File",
                str(exc),
            )


# =============================================================================
# JointGroupDialog
# =============================================================================
class JointGroupDialog(QtWidgets.QDialog):
    """複数ジョイントをマスタースライダで連動させるダイアログ"""

    def __init__(self, joint_editor, parent=None):
        super(JointGroupDialog, self).__init__(parent)
        self.joint_editor = joint_editor
        self.rows = {}
        self._loading = False
        self.base_angles = {}
        self.presets = json.loads(json.dumps(joint_editor.get_joint_group_presets()))
        self.current_index = joint_editor.current_group_preset_index
        self.setWindowTitle("Joint Group")
        self.setMinimumWidth(680)
        self.setMinimumHeight(520)
        self.setModal(False)
        if getattr(joint_editor, 'always_on_top', False):
            self.setWindowFlags(self.windowFlags() | QtCore.Qt.WindowStaysOnTopHint)
        self._setup_ui()
        self._load_preset(self.current_index)

    def _ordered_joints(self):
        return self.joint_editor.get_ordered_joint_names()

    def _setup_ui(self):
        layout = QtWidgets.QVBoxLayout(self)

        top_layout = QtWidgets.QHBoxLayout()
        self.name_edit = QtWidgets.QLineEdit()
        self.name_edit.setPlaceholderText("Group name")
        self.name_edit.setMinimumWidth(150)
        self.name_edit.returnPressed.connect(self._on_name_return_pressed)
        top_layout.addWidget(self.name_edit)

        self.master_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.master_slider.setRange(-1800, 1800)
        self.master_slider.setValue(0)
        self.master_slider.valueChanged.connect(self._on_master_changed)
        top_layout.addWidget(self.master_slider, stretch=1)

        self.master_value_label = create_label("0.0 deg")
        self.master_value_label.setFixedWidth(70)
        top_layout.addWidget(self.master_value_label)

        self.preset_combo = QtWidgets.QComboBox()
        self.preset_combo.currentIndexChanged.connect(self._on_preset_changed)
        top_layout.addWidget(self.preset_combo)

        add_button = QtWidgets.QPushButton("Add")
        add_button.setAutoDefault(False)
        add_button.clicked.connect(self._add_preset)
        top_layout.addWidget(add_button)

        del_button = QtWidgets.QPushButton("Del")
        del_button.setAutoDefault(False)
        self.del_button = del_button
        del_button.clicked.connect(self._delete_preset)
        top_layout.addWidget(del_button)
        layout.addLayout(top_layout)

        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)
        content = QtWidgets.QWidget()
        columns_layout = QtWidgets.QHBoxLayout(content)
        columns_layout.setContentsMargins(0, 0, 0, 0)
        columns_layout.setSpacing(6)

        column_layouts = {}
        for key, title in (("L", "Line-L"), ("R", "Line-R"), ("C", "Line-C")):
            group_box = QtWidgets.QGroupBox(title)
            group_layout = QtWidgets.QVBoxLayout(group_box)
            group_layout.setContentsMargins(4, 4, 4, 4)
            group_layout.setSpacing(2)

            header = QtWidgets.QWidget()
            header_layout = QtWidgets.QHBoxLayout(header)
            header_layout.setContentsMargins(0, 0, 0, 0)
            header_layout.setSpacing(4)
            header_layout.addSpacing(22)
            slider_header = create_label("Slider")
            slider_header.setStyleSheet("color: black; font-weight: bold;")
            slider_header.setFixedWidth(95)
            header_layout.addWidget(slider_header)
            scale_header = create_label("Scale")
            scale_header.setStyleSheet("color: black; font-weight: bold;")
            scale_header.setFixedWidth(90)
            header_layout.addWidget(scale_header)
            group_layout.addWidget(header)

            column_layouts[key] = group_layout
            columns_layout.addWidget(group_box, stretch=1)

        for jname in self._ordered_joints():
            check = QtWidgets.QCheckBox()
            check.toggled.connect(self._on_member_changed)

            setting = self.joint_editor.joint_settings.get(jname, {})
            label = create_label(setting.get("display_name", jname))
            label.setFixedWidth(95)
            label.setToolTip(jname)

            scale = QtWidgets.QDoubleSpinBox()
            scale.setRange(-999.0, 999.0)
            scale.setDecimals(3)
            scale.setSingleStep(0.1)
            scale.setValue(1.0)
            scale.setFixedWidth(90)
            scale.valueChanged.connect(self._on_member_changed)

            row_widget = QtWidgets.QWidget()
            row_layout = QtWidgets.QHBoxLayout(row_widget)
            row_layout.setContentsMargins(0, 0, 0, 0)
            row_layout.setSpacing(4)
            row_layout.addWidget(check)
            row_layout.addWidget(label)
            row_layout.addWidget(scale)

            group = self.joint_editor.joint_display_groups.get(jname, self.joint_editor._joint_group(jname))
            if group not in column_layouts:
                group = self.joint_editor._joint_group(jname)
            column_layouts[group].addWidget(row_widget)

            self.rows[jname] = {
                "check": check,
                "scale": scale,
            }

        for group_layout in column_layouts.values():
            group_layout.addStretch()

        scroll.setWidget(content)
        layout.addWidget(scroll, stretch=1)

        button_layout = QtWidgets.QHBoxLayout()
        button_layout.addStretch()
        apply_button = QtWidgets.QPushButton("Apply")
        apply_button.setAutoDefault(False)
        apply_button.clicked.connect(self._apply)
        button_layout.addWidget(apply_button)
        close_button = QtWidgets.QPushButton("Close")
        close_button.setAutoDefault(False)
        close_button.clicked.connect(self.reject)
        button_layout.addWidget(close_button)
        layout.addLayout(button_layout)

        self._refresh_preset_combo()

    def _refresh_preset_combo(self):
        self._loading = True
        self.preset_combo.clear()
        self.preset_combo.addItem("Individual")
        for idx, preset in enumerate(self.presets):
            name = preset.get("name", "") or f"Preset {idx + 1}"
            self.preset_combo.addItem(name)
        self.preset_combo.setCurrentIndex(self.current_index + 1)
        self._loading = False

    def _default_preset(self, name="Group1"):
        return {
            "name": name,
            "master": 0.0,
            "members": {
                jname: {"enabled": False, "scale": 1.0}
                for jname in self._ordered_joints()
            },
        }

    def _current_preset_from_ui(self):
        return {
            "name": self.name_edit.text(),
            "master": self.master_slider.value() / 10.0,
            "members": {
                jname: {
                    "enabled": row["check"].isChecked(),
                    "scale": row["scale"].value(),
                }
                for jname, row in self.rows.items()
            },
        }

    def _store_current_preset(self):
        if 0 <= self.current_index < len(self.presets):
            self.presets[self.current_index] = self._current_preset_from_ui()

    def _on_name_return_pressed(self):
        """Update the current preset name when Enter is pressed in the name field."""
        if 0 <= self.current_index < len(self.presets):
            self._store_current_preset()
            self._refresh_preset_combo()

    def _load_preset(self, index):
        if index < -1 or index >= len(self.presets):
            return

        self._loading = True
        self.current_index = index
        is_individual = index == -1
        preset = self._default_preset("Individual") if is_individual else self.presets[index]
        self.name_edit.setText(preset.get("name", ""))
        self.master_slider.setValue(int(round(float(preset.get("master", 0.0)) * 10)))
        members = preset.get("members", {})
        for jname, row in self.rows.items():
            member = members.get(jname, {})
            row["check"].setChecked(bool(member.get("enabled", False)))
            row["scale"].setValue(float(member.get("scale", 1.0)))
            row["check"].setEnabled(not is_individual)
            row["scale"].setEnabled(not is_individual)
        self.name_edit.setEnabled(not is_individual)
        self.master_slider.setEnabled(not is_individual)
        self.del_button.setEnabled(not is_individual)
        self.master_value_label.setText(f"{self.master_slider.value() / 10.0:.1f} deg")
        self.base_angles = self.joint_editor.get_angles()
        self._loading = False
        self._apply_master_to_sliders()

    def _on_preset_changed(self, index):
        if self._loading:
            return
        self._store_current_preset()
        self._load_preset(index - 1)

    def _on_master_changed(self, value):
        snapped = self.joint_editor._snap_slider_value(self.master_slider, value)
        if snapped != value:
            self._loading = True
            self.master_slider.setValue(snapped)
            self._loading = False
            value = snapped
        self.master_value_label.setText(f"{value / 10.0:.1f} deg")
        if not self._loading:
            self._apply_master_to_sliders()

    def _on_member_changed(self, *args):
        if not self._loading:
            self._apply_master_to_sliders()

    def _apply_master_to_sliders(self):
        master = self.master_slider.value() / 10.0
        angles = dict(self.base_angles)
        for jname, row in self.rows.items():
            if row["check"].isChecked():
                base = self.base_angles.get(jname, 0.0)
                angles[jname] = base + (master * row["scale"].value())
        self.joint_editor.set_angles(angles)
        self.joint_editor._update_3dview()

    def _add_preset(self):
        self._store_current_preset()
        new_preset = json.loads(json.dumps(self._current_preset_from_ui()))
        new_preset["name"] = f"Preset {len(self.presets) + 1}"
        new_preset["master"] = 0.0
        self.presets.append(new_preset)
        self.current_index = len(self.presets) - 1
        self._refresh_preset_combo()
        self._load_preset(self.current_index)

    def _delete_preset(self):
        if self.current_index < 0 or not self.presets:
            return
        del self.presets[self.current_index]
        self.current_index = min(self.current_index, len(self.presets) - 1)
        if self.current_index < 0:
            self.current_index = -1
        self._refresh_preset_combo()
        self._load_preset(self.current_index)

    def _apply(self):
        self._store_current_preset()
        self.joint_editor.set_joint_group_presets(self.presets, self.current_index)
        self.joint_editor._save_to_node()
        self.accept()


# =============================================================================
# SettingsDialog
# =============================================================================
class SettingsDialog(QtWidgets.QDialog):
    """設定ダイアログ（背景色設定など）"""

    bg_color_changed = QtCore.Signal(list, list)  # color_a, color_b
    bg_gradient_changed = QtCore.Signal(str)  # gradient_type
    bg_slider_changed = QtCore.Signal(int)  # BG white mix slider value
    light_slider_changed = QtCore.Signal(int)  # Light intensity slider value
    motion_defaults_changed = QtCore.Signal(int, int)  # default_frames, default_fps
    joint_settings_requested = QtCore.Signal()  # Open Joint Settings dialog
    sliders_settings_requested = QtCore.Signal()  # Open Sliders Settings (Joint Sliders)
    link_group_settings_requested = QtCore.Signal()  # Open Link Group Settings
    frame_presets_changed = QtCore.Signal(list)  # Frame preset values changed
    set_home_position_requested = QtCore.Signal()  # Set current pose as Home
    valkey_changed = QtCore.Signal(dict)  # Valkey config changed: {enabled, host, port, write_key, read_key}
    normalize_joints_requested = QtCore.Signal()  # Normalize joint names in current graph
    clear_all_requested = QtCore.Signal()      # Clear all (models + nodes)
    clear_models_requested = QtCore.Signal()   # Clear imported models only
    clear_nodes_requested = QtCore.Signal()    # Clear graph nodes only
    undo_limit_changed = QtCore.Signal(int)    # Undo history limit changed

    def __init__(self, color_a, color_b, gradient_type="vertical", bg_slider_val=50, light_slider_val=70, parent=None):
        super(SettingsDialog, self).__init__(parent)
        self.setWindowTitle("Config")
        self.setFixedWidth(440)
        self.setFixedHeight(700)
        self.setModal(False)

        self.color_a = list(color_a)
        self.color_b = list(color_b)
        self.gradient_type = gradient_type
        self.bg_slider_val = bg_slider_val
        self.light_slider_val = light_slider_val

        # Outer layout: scroll area + close button
        outer_layout = QtWidgets.QVBoxLayout(self)
        outer_layout.setContentsMargins(0, 0, 0, 8)
        outer_layout.setSpacing(0)

        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        scroll.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)
        outer_layout.addWidget(scroll)

        content_widget = QtWidgets.QWidget()
        scroll.setWidget(content_widget)

        layout = QtWidgets.QVBoxLayout(content_widget)
        layout.setSpacing(4)
        layout.setContentsMargins(12, 12, 12, 12)

        # タイトル
        title = QtWidgets.QLabel("<b>3D View Background</b>")
        title.setStyleSheet("color: black; font-size: 14px;")
        layout.addWidget(title)

        # グラデーションタイプ選択
        gradient_layout = QtWidgets.QHBoxLayout()
        gradient_label = QtWidgets.QLabel("Gradient:")
        gradient_label.setStyleSheet("color: black;")
        gradient_label.setFixedWidth(100)
        gradient_layout.addWidget(gradient_label)

        self.gradient_combo = QtWidgets.QComboBox()
        self.gradient_combo.addItem("None", "none")
        self.gradient_combo.addItem("Vertical", "vertical")
        self.gradient_combo.setStyleSheet("color: black; background-color: white;")

        # 現在の値を選択
        for i in range(self.gradient_combo.count()):
            if self.gradient_combo.itemData(i) == self.gradient_type:
                self.gradient_combo.setCurrentIndex(i)
                break

        self.gradient_combo.currentIndexChanged.connect(self._on_gradient_changed)
        gradient_layout.addWidget(self.gradient_combo)
        gradient_layout.addStretch()
        layout.addLayout(gradient_layout)

        # Color B (グラデーション用/上) - UI上で上に配置
        color_b_layout = QtWidgets.QHBoxLayout()
        color_b_label = QtWidgets.QLabel("Top:")
        color_b_label.setStyleSheet("color: black;")
        color_b_label.setFixedWidth(100)
        color_b_layout.addWidget(color_b_label)

        self.color_picker_b = ColorPicker(
            self, initial_color=self.color_b,
            on_color_changed=self._on_color_b_changed
        )
        self.color_picker_b.add_to_layout(color_b_layout)
        color_b_layout.addStretch()
        layout.addLayout(color_b_layout)

        # Color Bはグラデーション時のみ有効
        self._update_color_b_enabled()

        # Color A (ベース色/下) - UI上で下に配置
        color_a_layout = QtWidgets.QHBoxLayout()
        color_a_label = QtWidgets.QLabel("Base:")
        color_a_label.setStyleSheet("color: black;")
        color_a_label.setFixedWidth(100)
        color_a_layout.addWidget(color_a_label)

        self.color_picker_a = ColorPicker(
            self, initial_color=self.color_a,
            on_color_changed=self._on_color_a_changed
        )
        self.color_picker_a.add_to_layout(color_a_layout)
        color_a_layout.addStretch()
        layout.addLayout(color_a_layout)

        # BG slider (white mix)
        bg_slider_layout = QtWidgets.QHBoxLayout()
        bg_slider_label = QtWidgets.QLabel("BG-Brightness:")
        bg_slider_label.setStyleSheet("color: black;")
        bg_slider_label.setFixedWidth(100)
        bg_slider_layout.addWidget(bg_slider_label)
        self.bg_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.bg_slider.setMinimum(0)
        self.bg_slider.setMaximum(100)
        self.bg_slider.setValue(self.bg_slider_val)
        self.bg_slider.valueChanged.connect(self._on_bg_slider_changed)
        bg_slider_layout.addWidget(self.bg_slider)
        layout.addLayout(bg_slider_layout)

        # Light slider
        light_slider_layout = QtWidgets.QHBoxLayout()
        light_slider_label = QtWidgets.QLabel("Light:")
        light_slider_label.setStyleSheet("color: black;")
        light_slider_label.setFixedWidth(100)
        light_slider_layout.addWidget(light_slider_label)
        self.light_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.light_slider.setMinimum(0)
        self.light_slider.setMaximum(100)
        self.light_slider.setValue(self.light_slider_val)
        self.light_slider.valueChanged.connect(self._on_light_slider_changed)
        light_slider_layout.addWidget(self.light_slider)
        layout.addLayout(light_slider_layout)

        # New Node Position section
        layout.addSpacing(16)
        node_pos_title = QtWidgets.QLabel("<b>New Node Position</b>")
        node_pos_title.setStyleSheet("color: black; font-size: 14px;")
        layout.addWidget(node_pos_title)

        node_offset_row = QtWidgets.QHBoxLayout()
        node_offset_label = QtWidgets.QLabel("Offset:")
        node_offset_label.setStyleSheet("color: black;")
        node_offset_label.setFixedWidth(100)
        node_offset_row.addWidget(node_offset_label)

        x_label = QtWidgets.QLabel("x:")
        x_label.setStyleSheet("color: black;")
        node_offset_row.addWidget(x_label)

        self.node_offset_x_spin = QtWidgets.QSpinBox()
        self.node_offset_x_spin.setRange(-1000, 1000)
        self.node_offset_x_spin.setValue(get_node_offset_x())
        self.node_offset_x_spin.setStyleSheet("color: black;")
        self.node_offset_x_spin.setFixedWidth(70)
        self.node_offset_x_spin.valueChanged.connect(self._on_node_offset_changed)
        node_offset_row.addWidget(self.node_offset_x_spin)

        y_label = QtWidgets.QLabel("y:")
        y_label.setStyleSheet("color: black;")
        node_offset_row.addWidget(y_label)

        self.node_offset_y_spin = QtWidgets.QSpinBox()
        self.node_offset_y_spin.setRange(-1000, 1000)
        self.node_offset_y_spin.setValue(get_node_offset_y())
        self.node_offset_y_spin.setStyleSheet("color: black;")
        self.node_offset_y_spin.setFixedWidth(70)
        self.node_offset_y_spin.valueChanged.connect(self._on_node_offset_changed)
        node_offset_row.addWidget(self.node_offset_y_spin)

        node_offset_row.addStretch()
        layout.addLayout(node_offset_row)

        layout.addSpacing(16)
        undo_title = QtWidgets.QLabel("<b>History / Undo</b>")
        undo_title.setStyleSheet("color: black; font-size: 14px;")
        layout.addWidget(undo_title)

        undo_limit_row = QtWidgets.QHBoxLayout()
        undo_limit_label = QtWidgets.QLabel("Undo limit:")
        undo_limit_label.setStyleSheet("color: black;")
        undo_limit_label.setFixedWidth(100)
        undo_limit_row.addWidget(undo_limit_label)
        self.undo_limit_spin = QtWidgets.QSpinBox()
        self.undo_limit_spin.setRange(1, 500)
        self.undo_limit_spin.setValue(load_app_settings().get("undo_limit", 100))
        self.undo_limit_spin.setStyleSheet("color: black;")
        self.undo_limit_spin.setFixedWidth(80)
        self.undo_limit_spin.setToolTip("Number of undo steps to keep in memory (1–500)")
        undo_limit_row.addWidget(self.undo_limit_spin)
        undo_steps_label = QtWidgets.QLabel("steps")
        undo_steps_label.setStyleSheet("color: black;")
        undo_limit_row.addWidget(undo_steps_label)
        undo_limit_row.addStretch()
        layout.addLayout(undo_limit_row)

        layout.addSpacing(16)
        motion_title = QtWidgets.QLabel("<b>Simulation Step Rate</b>")
        motion_title.setStyleSheet("color: black; font-size: 14px;")
        layout.addWidget(motion_title)

        hz_fps_row = QtWidgets.QHBoxLayout()
        hz_fps_label = QtWidgets.QLabel("Hz(FPS):")
        hz_fps_label.setStyleSheet("color: black;")
        hz_fps_label.setFixedWidth(100)
        hz_fps_row.addWidget(hz_fps_label)
        self.hz_fps_spin = QtWidgets.QSpinBox()
        self.hz_fps_spin.setRange(1, 1000)
        self.hz_fps_spin.setValue(get_default_hz_fps())
        self.hz_fps_spin.setStyleSheet("color: black;")
        hz_fps_row.addWidget(self.hz_fps_spin)
        hz_fps_row.addStretch()
        layout.addLayout(hz_fps_row)

        # Frame Set Preset
        frame_preset_row = QtWidgets.QHBoxLayout()
        frame_preset_label = QtWidgets.QLabel("Frame Set Preset:")
        frame_preset_label.setStyleSheet("color: black;")
        frame_preset_label.setFixedWidth(100)
        frame_preset_row.addWidget(frame_preset_label)

        current_presets = get_frame_presets()
        self.frame_preset_edits = []
        _fp_style = "color: black; padding: 2px;"
        for i in range(4):
            edit = QtWidgets.QLineEdit(str(current_presets[i]))
            edit.setFixedWidth(50)
            edit.setStyleSheet(_fp_style)
            edit.setAlignment(QtCore.Qt.AlignCenter)
            self.frame_preset_edits.append(edit)
            frame_preset_row.addWidget(edit)
            if i < 3:
                comma_label = QtWidgets.QLabel(",")
                comma_label.setStyleSheet("color: black;")
                frame_preset_row.addWidget(comma_label)
        frame_preset_row.addStretch()
        layout.addLayout(frame_preset_row)

        layout.addSpacing(16)

        # Joint/Sliders Settings section
        other_title = QtWidgets.QLabel("<b>Actuator</b>")
        other_title.setStyleSheet("color: black; font-size: 14px;")
        layout.addWidget(other_title)

        actuator_row = QtWidgets.QHBoxLayout()
        actuator_row.setSpacing(6)

        joint_settings_btn = QtWidgets.QPushButton("Joint Param Setting")
        joint_settings_btn.setAutoDefault(False)
        joint_settings_btn.setDefault(False)
        joint_settings_btn.clicked.connect(self._on_joint_settings_clicked)
        actuator_row.addWidget(joint_settings_btn)

        link_group_settings_btn = QtWidgets.QPushButton("Link Group Setting")
        link_group_settings_btn.setAutoDefault(False)
        link_group_settings_btn.setDefault(False)
        link_group_settings_btn.clicked.connect(self._on_link_group_settings_clicked)
        actuator_row.addWidget(link_group_settings_btn)

        layout.addLayout(actuator_row)

        layout.addSpacing(16)

        # Home Position section
        home_title = QtWidgets.QLabel("<b>Home Position</b>")
        home_title.setStyleSheet("color: black; font-size: 14px;")
        layout.addWidget(home_title)

        set_home_btn = QtWidgets.QPushButton("Set Now Pose as Home")
        set_home_btn.setAutoDefault(False)
        set_home_btn.setDefault(False)
        set_home_btn.clicked.connect(self._on_set_home_position_clicked)
        layout.addWidget(set_home_btn)

        layout.addSpacing(16)

        # Valkey / Meridim Output section
        vk_title = QtWidgets.QLabel("<b>Valkey / Meridim Output</b>")
        vk_title.setStyleSheet("color: black; font-size: 14px;")
        layout.addWidget(vk_title)

        _vk_settings = load_app_settings()
        _vk_label_style = "color: black;"
        _vk_input_style = "color: black; background-color: white;"

        # Host / Port row
        vk_host_row = QtWidgets.QHBoxLayout()
        vk_host_lbl = QtWidgets.QLabel("Host:")
        vk_host_lbl.setStyleSheet(_vk_label_style)
        vk_host_lbl.setFixedWidth(100)
        vk_host_row.addWidget(vk_host_lbl)
        self.valkey_host_edit = QtWidgets.QLineEdit(_vk_settings.get("valkey_host", VALKEY_DEFAULT_HOST))
        self.valkey_host_edit.setFixedWidth(120)
        self.valkey_host_edit.setStyleSheet(_vk_input_style)
        vk_host_row.addWidget(self.valkey_host_edit)
        vk_port_lbl = QtWidgets.QLabel("Port:")
        vk_port_lbl.setStyleSheet(_vk_label_style)
        vk_port_lbl.setContentsMargins(12, 0, 0, 0)
        vk_host_row.addWidget(vk_port_lbl)
        self.valkey_port_spin = QtWidgets.QSpinBox()
        self.valkey_port_spin.setRange(1, 65535)
        self.valkey_port_spin.setValue(int(_vk_settings.get("valkey_port", VALKEY_DEFAULT_PORT)))
        self.valkey_port_spin.setFixedWidth(70)
        self.valkey_port_spin.setStyleSheet(_vk_label_style)
        vk_host_row.addWidget(self.valkey_port_spin)
        vk_host_row.addStretch()
        layout.addLayout(vk_host_row)

        # Write key row
        vk_wkey_row = QtWidgets.QHBoxLayout()
        vk_wkey_lbl = QtWidgets.QLabel("Write Key:")
        vk_wkey_lbl.setStyleSheet(_vk_label_style)
        vk_wkey_lbl.setFixedWidth(100)
        vk_wkey_row.addWidget(vk_wkey_lbl)
        self.valkey_write_key_combo = QtWidgets.QComboBox()
        self.valkey_write_key_combo.setEditable(True)
        self.valkey_write_key_combo.setFixedWidth(200)
        self.valkey_write_key_combo.setStyleSheet(_vk_input_style)
        self.valkey_write_key_combo.addItem(
            _vk_settings.get("valkey_write_key", VALKEY_DEFAULT_WRITE_KEY))
        vk_wkey_row.addWidget(self.valkey_write_key_combo)
        vk_wkey_refresh_btn = QtWidgets.QPushButton("↻")
        vk_wkey_refresh_btn.setFixedWidth(28)
        vk_wkey_refresh_btn.setToolTip("Fetch existing keys from Valkey server")
        vk_wkey_refresh_btn.setAutoDefault(False)
        vk_wkey_refresh_btn.setDefault(False)
        vk_wkey_refresh_btn.clicked.connect(self._refresh_valkey_keys)
        vk_wkey_row.addWidget(vk_wkey_refresh_btn)
        vk_wkey_row.addStretch()
        layout.addLayout(vk_wkey_row)

        # Read key row
        vk_rkey_row = QtWidgets.QHBoxLayout()
        vk_rkey_lbl = QtWidgets.QLabel("Read Key:")
        vk_rkey_lbl.setStyleSheet(_vk_label_style)
        vk_rkey_lbl.setFixedWidth(100)
        vk_rkey_row.addWidget(vk_rkey_lbl)
        self.valkey_read_key_edit = QtWidgets.QLineEdit(
            _vk_settings.get("valkey_read_key", VALKEY_DEFAULT_READ_KEY))
        self.valkey_read_key_edit.setFixedWidth(200)
        self.valkey_read_key_edit.setStyleSheet(_vk_input_style)
        vk_rkey_row.addWidget(self.valkey_read_key_edit)
        vk_rkey_row.addStretch()
        layout.addLayout(vk_rkey_row)

        # Apply + Status row
        vk_apply_row = QtWidgets.QHBoxLayout()
        vk_apply_btn = QtWidgets.QPushButton("Apply")
        vk_apply_btn.setFixedWidth(60)
        vk_apply_btn.setAutoDefault(False)
        vk_apply_btn.setDefault(False)
        vk_apply_btn.clicked.connect(self._on_valkey_apply)
        vk_apply_row.addWidget(vk_apply_btn)
        self.valkey_status_lbl = QtWidgets.QLabel("---")
        self.valkey_status_lbl.setStyleSheet("color: #555555;")
        vk_apply_row.addWidget(self.valkey_status_lbl)
        vk_apply_row.addStretch()
        layout.addLayout(vk_apply_row)

        layout.addSpacing(16)

        # Joint Name Normalization section
        norm_title = QtWidgets.QLabel("<b>Joint Name Normalization</b>")
        norm_title.setStyleSheet("color: black; font-size: 14px;")
        layout.addWidget(norm_title)

        norm_desc = QtWidgets.QLabel(
            "Rename joints in the current graph to RobotLabelBridge\n"
            "canonical form (e.g. l_shoulder_yp) for Valkey compatibility."
        )
        norm_desc.setStyleSheet("color: #444444; font-size: 11px;")
        norm_desc.setWordWrap(True)
        layout.addWidget(norm_desc)

        norm_btn = QtWidgets.QPushButton("Normalize Joint Names in Graph")
        norm_btn.setAutoDefault(False)
        norm_btn.setDefault(False)
        norm_btn.clicked.connect(self.normalize_joints_requested.emit)
        layout.addWidget(norm_btn)

        layout.addSpacing(16)

        # Clear section
        clear_title = QtWidgets.QLabel("<b>Clear</b>")
        clear_title.setStyleSheet("color: black; font-size: 14px;")
        layout.addWidget(clear_title)

        clear_all_btn = QtWidgets.QPushButton("All Clear (Models + Nodes)")
        clear_all_btn.setAutoDefault(False)
        clear_all_btn.setDefault(False)
        clear_all_btn.setStyleSheet("color: white; background-color: #c0392b;")
        clear_all_btn.clicked.connect(self._on_clear_all_clicked)
        layout.addWidget(clear_all_btn)
        layout.addSpacing(4)

        clear_models_btn = QtWidgets.QPushButton("Clear Models")
        clear_models_btn.setAutoDefault(False)
        clear_models_btn.setDefault(False)
        clear_models_btn.setStyleSheet("color: black; background-color: #e0a060;")
        clear_models_btn.clicked.connect(self._on_clear_models_clicked)
        layout.addWidget(clear_models_btn)
        layout.addSpacing(4)

        clear_nodes_btn = QtWidgets.QPushButton("Clear Nodes")
        clear_nodes_btn.setAutoDefault(False)
        clear_nodes_btn.setDefault(False)
        clear_nodes_btn.setStyleSheet("color: black; background-color: #e0a060;")
        clear_nodes_btn.clicked.connect(self._on_clear_nodes_clicked)
        layout.addWidget(clear_nodes_btn)

        layout.addSpacing(16)

        # Code Editor section
        code_title = QtWidgets.QLabel("<b>Code Editor</b>")
        code_title.setStyleSheet("color: black; font-size: 14px;")
        layout.addWidget(code_title)

        _ce_settings = load_app_settings()
        _ce_mode = _ce_settings.get("code_editor_mode", "internal")

        code_mode_row = QtWidgets.QHBoxLayout()
        self._radio_internal = QtWidgets.QRadioButton("Built-in editor")
        self._radio_internal.setStyleSheet("color: black;")
        self._radio_external = QtWidgets.QRadioButton("External app")
        self._radio_external.setStyleSheet("color: black;")
        if _ce_mode == "external":
            self._radio_external.setChecked(True)
        else:
            self._radio_internal.setChecked(True)
        code_mode_row.addWidget(self._radio_internal)
        code_mode_row.addWidget(self._radio_external)
        code_mode_row.addStretch()
        layout.addLayout(code_mode_row)

        ext_path_row = QtWidgets.QHBoxLayout()
        self._ext_path_edit = QtWidgets.QLineEdit()
        self._ext_path_edit.setPlaceholderText(code_editor_placeholder())
        self._ext_path_edit.setStyleSheet("color: black; background-color: white;")
        self._ext_path_edit.setText(_ce_settings.get("code_editor_path", ""))
        ext_path_row.addWidget(self._ext_path_edit, 1)
        browse_btn = QtWidgets.QPushButton("Browse…")
        browse_btn.setAutoDefault(False)
        browse_btn.setDefault(False)
        browse_btn.setFixedWidth(70)
        browse_btn.setStyleSheet("color: black;")
        browse_btn.clicked.connect(self._on_browse_code_editor)
        ext_path_row.addWidget(browse_btn)
        layout.addLayout(ext_path_row)

        ce_apply_btn = QtWidgets.QPushButton("Apply Code Editor Setting")
        ce_apply_btn.setAutoDefault(False)
        ce_apply_btn.setDefault(False)
        ce_apply_btn.clicked.connect(self._on_code_editor_apply)
        layout.addWidget(ce_apply_btn)

        layout.addStretch()

        # Close button fixed outside the scroll area
        close_row = QtWidgets.QHBoxLayout()
        close_row.setContentsMargins(12, 4, 12, 0)
        close_row.addStretch()
        close_btn = QtWidgets.QPushButton("Close")
        close_btn.setFixedWidth(60)
        close_btn.setAutoDefault(False)
        close_btn.setDefault(False)
        close_btn.clicked.connect(self.close)
        close_row.addWidget(close_btn)
        outer_layout.addLayout(close_row)

    def _update_color_b_enabled(self):
        """グラデーションタイプに応じてColor Bの有効/無効を切り替え"""
        enabled = self.gradient_type != "none"
        self.color_picker_b.set_enabled(enabled)

    def _on_gradient_changed(self, index):
        self.gradient_type = self.gradient_combo.itemData(index)
        self._update_color_b_enabled()
        self.bg_gradient_changed.emit(self.gradient_type)

    def _on_color_a_changed(self, color):
        self.color_a = color
        self.bg_color_changed.emit(self.color_a, self.color_b)

    def _on_color_b_changed(self, color):
        self.color_b = color
        self.bg_color_changed.emit(self.color_a, self.color_b)

    def _on_bg_slider_changed(self, value):
        self.bg_slider_val = value
        self.bg_slider_changed.emit(value)

    def _on_light_slider_changed(self, value):
        self.light_slider_val = value
        self.light_slider_changed.emit(value)

    def _on_node_offset_changed(self):
        x = self.node_offset_x_spin.value()
        y = self.node_offset_y_spin.value()
        set_node_offset(x, y)

    def _on_joint_settings_clicked(self):
        self.joint_settings_requested.emit()

    def _on_sliders_settings_clicked(self):
        self.sliders_settings_requested.emit()

    def _on_link_group_settings_clicked(self):
        self.link_group_settings_requested.emit()

    def _on_set_home_position_clicked(self):
        self.set_home_position_requested.emit()

    def _on_valkey_apply(self):
        """Apply Valkey settings immediately and emit signal."""
        cfg = self._collect_valkey_cfg()
        self._save_valkey_settings(cfg)
        self.valkey_changed.emit(cfg)
        self.valkey_status_lbl.setText("Applied.")

    def _collect_valkey_cfg(self) -> dict:
        # enabled is controlled by the 3D view Valkey checkbox, preserve it from settings
        _saved = load_app_settings()
        return {
            "enabled":   _saved.get("valkey_enabled", False),
            "host":      self.valkey_host_edit.text().strip() or VALKEY_DEFAULT_HOST,
            "port":      int(self.valkey_port_spin.value()),
            "write_key": self.valkey_write_key_combo.currentText().strip() or VALKEY_DEFAULT_WRITE_KEY,
            "read_key":  self.valkey_read_key_edit.text().strip() or VALKEY_DEFAULT_READ_KEY,
        }

    def _save_valkey_settings(self, cfg: dict) -> None:
        settings = load_app_settings()
        # valkey_enabled is managed by the 3D view checkbox — do not overwrite here
        settings["valkey_host"]      = cfg["host"]
        settings["valkey_port"]      = cfg["port"]
        settings["valkey_write_key"] = cfg["write_key"]
        settings["valkey_read_key"]  = cfg["read_key"]
        save_app_settings(settings)

    def _refresh_valkey_keys(self) -> None:
        """Fetch all keys from the Valkey server and populate the write key dropdown."""
        try:
            import valkey as _vk_mod
            host = self.valkey_host_edit.text().strip() or VALKEY_DEFAULT_HOST
            port = int(self.valkey_port_spin.value())
            client = _vk_mod.Valkey(
                host=host, port=port,
                decode_responses=True,
                socket_connect_timeout=0.5, socket_timeout=0.5,
            )
            keys = sorted(client.keys("*"))
            client.close()
            current = self.valkey_write_key_combo.currentText()
            self.valkey_write_key_combo.clear()
            for k in keys:
                self.valkey_write_key_combo.addItem(k)
            if current:
                idx = self.valkey_write_key_combo.findText(current)
                if idx >= 0:
                    self.valkey_write_key_combo.setCurrentIndex(idx)
                else:
                    self.valkey_write_key_combo.setEditText(current)
            self.valkey_status_lbl.setText(f"Keys fetched: {len(keys)}")
        except ImportError:
            self.valkey_status_lbl.setText("valkey not installed")
        except Exception as e:
            self.valkey_status_lbl.setText(f"Fetch error: {e}")

    def set_valkey_status(self, text: str) -> None:
        """Called externally to update the connection status label."""
        self.valkey_status_lbl.setText(text)

    def _on_clear_all_clicked(self):
        reply = QtWidgets.QMessageBox.question(
            self, "All Clear",
            "Clear all models and nodes. This cannot be undone. Continue?",
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
            QtWidgets.QMessageBox.No,
        )
        if reply == QtWidgets.QMessageBox.Yes:
            self.clear_all_requested.emit()

    def _on_clear_models_clicked(self):
        reply = QtWidgets.QMessageBox.question(
            self, "Clear Models",
            "Clear all imported models (meshes and robot model). Continue?",
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
            QtWidgets.QMessageBox.No,
        )
        if reply == QtWidgets.QMessageBox.Yes:
            self.clear_models_requested.emit()

    def _on_clear_nodes_clicked(self):
        reply = QtWidgets.QMessageBox.question(
            self, "Clear Nodes",
            "Clear all nodes from the graph. Continue?",
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
            QtWidgets.QMessageBox.No,
        )
        if reply == QtWidgets.QMessageBox.Yes:
            self.clear_nodes_requested.emit()

    def _on_browse_code_editor(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Select External Editor",
            code_editor_browse_start_dir(),
            code_editor_browse_filter(),
        )
        if path:
            self._ext_path_edit.setText(path)

    def _on_code_editor_apply(self):
        mode = "external" if self._radio_external.isChecked() else "internal"
        path = self._ext_path_edit.text().strip()
        s = load_app_settings()
        s["code_editor_mode"] = mode
        s["code_editor_path"] = path
        save_app_settings(s)

    def showEvent(self, event):
        super(SettingsDialog, self).showEvent(event)
        self.hz_fps_spin.setValue(get_default_hz_fps())
        # Refresh Valkey fields from saved settings
        _vk = load_app_settings()
        self.valkey_host_edit.setText(_vk.get("valkey_host", VALKEY_DEFAULT_HOST))
        self.valkey_port_spin.setValue(int(_vk.get("valkey_port", VALKEY_DEFAULT_PORT)))
        self.valkey_write_key_combo.setEditText(_vk.get("valkey_write_key", VALKEY_DEFAULT_WRITE_KEY))
        self.valkey_read_key_edit.setText(_vk.get("valkey_read_key", VALKEY_DEFAULT_READ_KEY))
        # Auto-refresh key list from server
        self._refresh_valkey_keys()

    def closeEvent(self, event):
        s = _app_settings()
        hz_fps = int(self.hz_fps_spin.value())
        s.setValue("motion/hz_fps", hz_fps)
        # 後方互換性: 同じ値を両方に渡す
        self.motion_defaults_changed.emit(hz_fps, hz_fps)
        # Save frame presets
        new_presets = []
        for edit in self.frame_preset_edits:
            try:
                val = int(edit.text())
                if val < 1:
                    val = 1
                new_presets.append(val)
            except ValueError:
                new_presets.append(10)
        save_frame_presets(new_presets)
        self.frame_presets_changed.emit(new_presets)
        # Save and emit Valkey settings
        cfg = self._collect_valkey_cfg()
        self._save_valkey_settings(cfg)
        self.valkey_changed.emit(cfg)
        # Save and emit undo limit
        undo_limit = int(self.undo_limit_spin.value())
        _s = load_app_settings()
        _s["undo_limit"] = undo_limit
        save_app_settings(_s)
        self.undo_limit_changed.emit(undo_limit)
        super(SettingsDialog, self).closeEvent(event)


# =============================================================================
# MixEditorPanel - Editor for MixNode settings

# =============================================================================
# MixEditorPanel - Editor for MixNode settings
# =============================================================================

# =============================================================================

# Input source options for Mix node
MIX_INPUT_SOURCES = (
    ["Gyro_x", "Gyro_y", "Pad_Lx", "Pad_Ly", "Pad_Rx", "Pad_Ry"] +
    [f"UserVal_{i}" for i in range(64)]
)


class MixEditorPanel(QtWidgets.QWidget):
    """Editor panel for MixNode joint settings."""

    mix_changed = QtCore.Signal(dict)  # Emitted when mix settings change

    def __init__(self, parent=None):
        super(MixEditorPanel, self).__init__(parent)
        self.robot_model = None
        self.current_mix_node = None
        self.sliders = {}
        self.spinboxes = {}
        self.checkboxes = {}
        self.source_combos = {}
        self.gain_spins = {}
        self.joint_rows = {}
        self.joint_name_labels = {}
        self.joint_display_groups = {}
        self.joint_display_order = []
        self.column_layouts = {}
        self.column_widgets = {}
        self.graph = None
        self._updating = False
        settings = load_app_settings()
        self.always_on_top = settings.get("mix_editor_always_on_top", True)
        self._setup_ui()
        self._refresh_mix_meta_row()
        self._apply_always_on_top(restore_visible=False)

    def _setup_ui(self):
        self.main_layout = QtWidgets.QVBoxLayout(self)
        self.main_layout.setContentsMargins(4, 4, 4, 4)

        # Row 1: Name, Duration, Frames
        self.mix_meta_widget = QtWidgets.QWidget()
        meta_layout = QtWidgets.QHBoxLayout(self.mix_meta_widget)
        meta_layout.setContentsMargins(0, 4, 0, 6)
        meta_layout.setSpacing(8)

        # Name
        meta_layout.addWidget(self._create_label("Name:"))
        self.mix_name_edit = QtWidgets.QLineEdit()
        self.mix_name_edit.setFixedWidth(120)
        self.mix_name_edit.editingFinished.connect(self._apply_mix_name_from_ui)
        meta_layout.addWidget(self.mix_name_edit)

        # Duration
        duration_container = QtWidgets.QHBoxLayout()
        duration_container.setContentsMargins(0, 0, 0, 0)
        duration_container.setSpacing(0)
        duration_container.addWidget(self._create_label("Duration:"))
        self.mix_duration_label = QtWidgets.QLabel("—")
        self.mix_duration_label.setFixedWidth(70)
        self.mix_duration_label.setAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter)
        self.mix_duration_label.setToolTip("Duration = Frames / Simulation Step Rate (FPS)")
        self.mix_duration_label.setStyleSheet("color: black; padding: 2px 0px;")
        duration_container.addWidget(self.mix_duration_label)
        meta_layout.addLayout(duration_container)

        # Frames
        meta_layout.addWidget(self._create_label("Frames:"))
        self.mix_frames_spin = QtWidgets.QSpinBox()
        self.mix_frames_spin.setRange(1, 9999)
        self.mix_frames_spin.setValue(1)  # Default = 1
        self.mix_frames_spin.setFixedWidth(60)
        self.mix_frames_spin.valueChanged.connect(self._apply_mix_frames_from_ui)
        meta_layout.addWidget(self.mix_frames_spin)

        meta_layout.addStretch()
        self.main_layout.addWidget(self.mix_meta_widget)

        # Row 2: Stay on Top, Zero, Home
        options_layout = QtWidgets.QHBoxLayout()
        options_layout.setContentsMargins(0, 0, 0, 0)
        options_layout.setSpacing(12)

        dark_gray_style = "color: #555555;"

        self.always_on_top_checkbox = QtWidgets.QCheckBox("Stay on Top")
        self.always_on_top_checkbox.setStyleSheet(dark_gray_style)
        self.always_on_top_checkbox.setChecked(self.always_on_top)
        self.always_on_top_checkbox.toggled.connect(self._on_always_on_top_toggled)
        options_layout.addWidget(self.always_on_top_checkbox)

        options_layout.addStretch()

        # Zero button
        self.zero_button = QtWidgets.QPushButton("Zero")
        self.zero_button.clicked.connect(self._on_zero_button_clicked)
        options_layout.addWidget(self.zero_button)

        # Home button
        self.home_button = QtWidgets.QPushButton("Home")
        self.home_button.clicked.connect(self._on_home_button_clicked)
        options_layout.addWidget(self.home_button)

        self.main_layout.addLayout(options_layout)

        # Tab widget for Line-L, Line-R, Line-C columns
        self.tab_widget = QtWidgets.QTabWidget()
        self.tab_widget.setStyleSheet("color: black;")
        self.main_layout.addWidget(self.tab_widget)

    def _create_label(self, text):
        """Create a styled label."""
        label = QtWidgets.QLabel(text)
        label.setStyleSheet("color: black;")
        return label

    def set_robot_model(self, robot_model):
        """Set the robot model and populate joint sliders."""
        self.robot_model = robot_model
        self._populate_joints()

    def _populate_joints(self):
        """Populate joint rows based on robot model."""
        # Clear existing tabs
        while self.tab_widget.count() > 0:
            widget = self.tab_widget.widget(0)
            self.tab_widget.removeTab(0)
            widget.deleteLater()

        self.sliders.clear()
        self.spinboxes.clear()
        self.checkboxes.clear()
        self.source_combos.clear()
        self.gain_spins.clear()
        self.joint_rows.clear()
        self.column_layouts.clear()
        self.column_widgets.clear()

        # Get robot_model from graph's stl_viewer if not set
        robot_model = self.robot_model
        if not robot_model and self.graph:
            stl_viewer = getattr(self.graph, 'stl_viewer', None)
            if stl_viewer:
                robot_model = getattr(stl_viewer, 'robot_model', None)
        if not robot_model:
            return

        # Get joint layout from graph's joint_editor if available
        joint_editor = getattr(self.graph, 'joint_editor', None) if self.graph else None
        if joint_editor:
            self.joint_display_groups = dict(getattr(joint_editor, 'joint_display_groups', {}))
            self.joint_display_order = list(getattr(joint_editor, 'joint_display_order', []))
        else:
            self.joint_display_groups = {}
            self.joint_display_order = []

        # Get joints from robot model (joints is a dict, use joint_order for iteration)
        joints = []
        joint_order = getattr(robot_model, 'joint_order', [])
        print(f"[MixEditor] joint_order count: {len(joint_order)}")
        for jname in joint_order:
            joints.append(jname)
        print(f"[MixEditor] Found {len(joints)} joints")

        # Sort by display order
        if self.joint_display_order:
            ordered = [j for j in self.joint_display_order if j in joints]
            remaining = [j for j in joints if j not in ordered]
            joints = ordered + remaining

        # Group joints
        groups = {"L": [], "R": [], "C": []}
        for jname in joints:
            group = self.joint_display_groups.get(jname, self._detect_group(jname))
            groups[group].append(jname)

        # Create tabs for each group
        for group_key, group_label in [("L", "Line-L"), ("R", "Line-R"), ("C", "Line-C")]:
            if not groups[group_key]:
                continue

            scroll = QtWidgets.QScrollArea()
            scroll.setWidgetResizable(True)
            scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)

            container = QtWidgets.QWidget()
            layout = QtWidgets.QVBoxLayout(container)
            layout.setContentsMargins(4, 4, 4, 4)
            layout.setSpacing(2)

            self.column_layouts[group_key] = layout
            self.column_widgets[group_key] = container

            for jname in groups[group_key]:
                row = self._create_joint_row(jname)
                layout.addWidget(row)
                self.joint_rows[jname] = row

            layout.addStretch()
            scroll.setWidget(container)
            self.tab_widget.addTab(scroll, group_label)
            print(f"[MixEditor] Added tab '{group_label}' with {len(groups[group_key])} joints")

        print(f"[MixEditor] Total tabs: {self.tab_widget.count()}, sliders: {len(self.sliders)}")

    def _detect_group(self, joint_name):
        """Detect joint group from name."""
        name_lower = joint_name.lower()
        if '_l_' in name_lower or name_lower.endswith('_l') or name_lower.startswith('l_'):
            return "L"
        elif '_r_' in name_lower or name_lower.endswith('_r') or name_lower.startswith('r_'):
            return "R"
        return "C"

    def _create_joint_row(self, joint_name):
        """Create a row widget for a joint with checkbox, slider, source combo, and gain."""
        row = QtWidgets.QWidget()
        layout = QtWidgets.QHBoxLayout(row)
        layout.setContentsMargins(2, 1, 2, 1)
        layout.setSpacing(4)

        # Checkbox (enable/disable)
        checkbox = QtWidgets.QCheckBox()
        checkbox.setChecked(False)
        checkbox.stateChanged.connect(lambda state, jn=joint_name: self._on_checkbox_changed(jn, state))
        layout.addWidget(checkbox)
        self.checkboxes[joint_name] = checkbox

        # Joint name label
        name_label = QtWidgets.QLabel(joint_name)
        name_label.setFixedWidth(100)
        name_label.setStyleSheet("color: black;")
        layout.addWidget(name_label)
        self.joint_name_labels[joint_name] = name_label

        # Slider (for preview only, not saved)
        slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        slider.setRange(-1800, 1800)  # -180 to 180 degrees * 10
        slider.setValue(0)
        slider.valueChanged.connect(lambda v, jn=joint_name: self._on_slider_changed(jn, v))
        layout.addWidget(slider)
        self.sliders[joint_name] = slider

        # Spinbox (for preview only, not saved)
        spinbox = QtWidgets.QDoubleSpinBox()
        spinbox.setRange(-180.0, 180.0)
        spinbox.setDecimals(1)
        spinbox.setSingleStep(1.0)
        spinbox.setValue(0.0)
        spinbox.setFixedWidth(70)
        spinbox.valueChanged.connect(lambda v, jn=joint_name: self._on_spinbox_changed(jn, v))
        layout.addWidget(spinbox)
        self.spinboxes[joint_name] = spinbox

        # Source combo (Gyro_x, Gyro_y, Pad_*, UserVal_*)
        source_combo = QtWidgets.QComboBox()
        source_combo.addItems(MIX_INPUT_SOURCES)
        source_combo.setCurrentIndex(0)
        source_combo.setFixedWidth(100)
        source_combo.setStyleSheet("color: black;")
        source_combo.currentIndexChanged.connect(lambda idx, jn=joint_name: self._on_source_changed(jn, idx))
        layout.addWidget(source_combo)
        self.source_combos[joint_name] = source_combo

        # Gain spinbox
        gain_label = QtWidgets.QLabel("Gain:")
        gain_label.setStyleSheet("color: black;")
        layout.addWidget(gain_label)

        gain_spin = QtWidgets.QDoubleSpinBox()
        gain_spin.setRange(-100.0, 100.0)
        gain_spin.setDecimals(2)
        gain_spin.setSingleStep(0.1)
        gain_spin.setValue(1.0)
        gain_spin.setFixedWidth(70)
        gain_spin.valueChanged.connect(lambda v, jn=joint_name: self._on_gain_changed(jn, v))
        layout.addWidget(gain_spin)
        self.gain_spins[joint_name] = gain_spin

        return row

    def _on_checkbox_changed(self, joint_name, state):
        """Handle checkbox state change."""
        if self._updating:
            return
        self._save_to_node()

    def _on_slider_changed(self, joint_name, value):
        """Handle slider value change."""
        if self._updating:
            return
        self._updating = True
        self.spinboxes[joint_name].setValue(value / 10.0)
        self._updating = False
        self._save_to_node()
        self._update_3dview()

    def _on_spinbox_changed(self, joint_name, value):
        """Handle spinbox value change."""
        if self._updating:
            return
        self._updating = True
        self.sliders[joint_name].setValue(int(value * 10))
        self._updating = False
        self._save_to_node()
        self._update_3dview()

    def _on_source_changed(self, joint_name, index):
        """Handle source combo change."""
        if self._updating:
            return
        self._save_to_node()

    def _on_gain_changed(self, joint_name, value):
        """Handle gain value change."""
        if self._updating:
            return
        self._save_to_node()

    def _save_to_node(self):
        """Save current settings to the MixNode."""
        if not self.current_mix_node:
            return
        mix_settings = {}
        angles_deg = {}
        for jname in self.checkboxes:
            mix_settings[jname] = {
                "enabled": self.checkboxes[jname].isChecked(),
                "input_source": self.source_combos[jname].currentText(),
                "gain": self.gain_spins[jname].value()
            }
            if jname in self.spinboxes:
                angles_deg[jname] = self.spinboxes[jname].value()
        self.current_mix_node.mix_settings = mix_settings
        self.current_mix_node.angles_deg = angles_deg
        self.mix_changed.emit(mix_settings)

    def _update_3dview(self):
        """Update 3D view with current slider values."""
        if not self.graph:
            return
        stl_viewer = getattr(self.graph, 'stl_viewer', None)
        if not stl_viewer:
            return
        robot_model = getattr(stl_viewer, 'robot_model', None)
        if not robot_model:
            return
        angles = self.get_angles()
        # Apply to 3D view
        joint_editor = getattr(self.graph, 'joint_editor', None)
        if joint_editor:
            angles_for_3d = joint_editor.get_angles_for_3d(angles)
            robot_model.apply_joint_angles(angles_for_3d)
            stl_viewer.safe_render()

    def get_angles(self):
        """Get current slider angles (for preview)."""
        result = {}
        for jname, spin in self.spinboxes.items():
            result[jname] = spin.value()
        return result

    def set_angles(self, angles):
        """Set slider angles (for preview)."""
        self._updating = True
        for jname, value in angles.items():
            if jname in self.spinboxes:
                self.spinboxes[jname].setValue(value)
                self.sliders[jname].setValue(int(value * 10))
        self._updating = False

    def _on_zero_button_clicked(self):
        """Reset all sliders to zero."""
        angles = {jname: 0.0 for jname in self.spinboxes}
        self.set_angles(angles)
        self._update_3dview()

    def _on_home_button_clicked(self):
        """Move sliders to Home position."""
        joint_editor = getattr(self.graph, 'joint_editor', None) if self.graph else None
        if joint_editor:
            home_angles = getattr(joint_editor, 'home_position_angles', None)
            if home_angles:
                self.set_angles(home_angles)
                self._update_3dview()
                return
        self._on_zero_button_clicked()

    def _on_always_on_top_toggled(self, checked):
        """Handle Stay on Top checkbox toggle."""
        self.always_on_top = checked
        save_app_settings({"mix_editor_always_on_top": checked})
        self._apply_always_on_top()

    def _apply_always_on_top(self, restore_visible=True):
        """Apply the always on top setting."""
        parent_window = self.window()
        if parent_window and parent_window != self:
            was_visible = parent_window.isVisible()
            if self.always_on_top:
                parent_window.setWindowFlags(
                    parent_window.windowFlags() | QtCore.Qt.WindowStaysOnTopHint
                )
            else:
                parent_window.setWindowFlags(
                    parent_window.windowFlags() & ~QtCore.Qt.WindowStaysOnTopHint
                )
            if restore_visible and was_visible:
                parent_window.show()

    def set_mix_node(self, mix_node):
        """Set the current MixNode for editing."""
        print(f"[MixEditor] set_mix_node called, tab_count={self.tab_widget.count()}")
        self.current_mix_node = mix_node
        # Populate joints if tab widget is empty (always try if no tabs)
        if self.tab_widget.count() == 0:
            print("[MixEditor] Tab widget is empty, calling _populate_joints")
            self._populate_joints()
        self._refresh_mix_meta_row()
        self._load_from_node()

    def _refresh_mix_meta_row(self):
        """Refresh the name/frames/duration row."""
        node = self.current_mix_node
        self.mix_name_edit.blockSignals(True)
        self.mix_frames_spin.blockSignals(True)
        if node:
            self.mix_name_edit.setEnabled(True)
            self.mix_frames_spin.setEnabled(True)
            self.mix_name_edit.setText(getattr(node, 'mix_name', 'mix'))
            self.mix_frames_spin.setValue(getattr(node, 'frames', 1))
            self._update_duration_label()
        else:
            self.mix_name_edit.setEnabled(False)
            self.mix_frames_spin.setEnabled(False)
            self.mix_name_edit.clear()
            self.mix_frames_spin.setValue(1)
            self.mix_duration_label.setText("—")
        self.mix_name_edit.blockSignals(False)
        self.mix_frames_spin.blockSignals(False)

    def _update_duration_label(self):
        """Update duration label based on frames and FPS."""
        node = self.current_mix_node
        if not node:
            self.mix_duration_label.setText("—")
            return
        frames = getattr(node, 'frames', 1)
        fps = get_default_hz_fps()
        duration_sec = frames / fps if fps > 0 else 0
        self.mix_duration_label.setText(f"{duration_sec:.2f} sec")

    def _load_from_node(self):
        """Load settings from the current MixNode."""
        if not self.current_mix_node:
            return
        self._updating = True
        mix_settings = getattr(self.current_mix_node, 'mix_settings', {})
        angles_deg = getattr(self.current_mix_node, 'angles_deg', {})
        for jname in self.checkboxes:
            settings = mix_settings.get(jname, {})
            self.checkboxes[jname].setChecked(settings.get("enabled", False))
            source = settings.get("input_source", "Gyro_x")
            idx = self.source_combos[jname].findText(source)
            if idx >= 0:
                self.source_combos[jname].setCurrentIndex(idx)
            self.gain_spins[jname].setValue(settings.get("gain", 1.0))
            # Load slider/spinbox values from angles_deg
            angle = angles_deg.get(jname, 0.0)
            if jname in self.sliders:
                self.sliders[jname].setValue(int(angle * 10))
            if jname in self.spinboxes:
                self.spinboxes[jname].setValue(angle)
        self._updating = False
        try:
            self._update_3dview()
        except Exception as e:
            print(f"[MixEditor] Error updating 3D view: {e}")

    def _apply_mix_name_from_ui(self):
        """Apply name from UI to node."""
        if self.current_mix_node:
            self.current_mix_node.mix_name = self.mix_name_edit.text()
            self.current_mix_node.set_name(self.mix_name_edit.text())

    def _apply_mix_frames_from_ui(self):
        """Apply frames from UI to node."""
        if self.current_mix_node:
            self.current_mix_node.frames = self.mix_frames_spin.value()
            self._update_duration_label()


class CommandEditorPanel(QtWidgets.QWidget):
    """Editor panel for CommandNode servo command settings."""

    command_changed = QtCore.Signal(dict)  # Emitted when command settings change

    def __init__(self, parent=None):
        super(CommandEditorPanel, self).__init__(parent)
        self.robot_model = None
        self.current_command_node = None
        self.command_combos = {}
        self.value_spins = {}
        self.joint_rows = {}
        self.joint_name_labels = {}
        self.joint_display_groups = {}
        self.joint_display_order = []
        self.column_layouts = {}
        self.column_widgets = {}
        self.graph = None
        self._updating = False
        settings = load_app_settings()
        self.always_on_top = settings.get("command_editor_always_on_top", True)
        self._setup_ui()
        self._refresh_command_meta_row()
        self._apply_always_on_top(restore_visible=False)

    def _setup_ui(self):
        self.main_layout = QtWidgets.QVBoxLayout(self)
        self.main_layout.setContentsMargins(4, 4, 4, 4)

        # Row 1: Name, Duration, Frames
        self.command_meta_widget = QtWidgets.QWidget()
        meta_layout = QtWidgets.QHBoxLayout(self.command_meta_widget)
        meta_layout.setContentsMargins(0, 4, 0, 6)
        meta_layout.setSpacing(8)

        # Name
        meta_layout.addWidget(self._create_label("Name:"))
        self.command_name_edit = QtWidgets.QLineEdit()
        self.command_name_edit.setFixedWidth(120)
        self.command_name_edit.editingFinished.connect(self._apply_command_name_from_ui)
        meta_layout.addWidget(self.command_name_edit)

        # Duration
        duration_container = QtWidgets.QHBoxLayout()
        duration_container.setContentsMargins(0, 0, 0, 0)
        duration_container.setSpacing(0)
        duration_container.addWidget(self._create_label("Duration:"))
        self.command_duration_label = QtWidgets.QLabel("—")
        self.command_duration_label.setFixedWidth(70)
        self.command_duration_label.setAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter)
        self.command_duration_label.setToolTip("Duration = Frames / Simulation Step Rate (FPS)")
        self.command_duration_label.setStyleSheet("color: black; padding: 2px 0px;")
        duration_container.addWidget(self.command_duration_label)
        meta_layout.addLayout(duration_container)

        # Frames
        meta_layout.addWidget(self._create_label("Frames:"))
        self.command_frames_spin = QtWidgets.QSpinBox()
        self.command_frames_spin.setRange(1, 9999)
        self.command_frames_spin.setValue(1)
        self.command_frames_spin.setFixedWidth(60)
        self.command_frames_spin.valueChanged.connect(self._apply_command_frames_from_ui)
        meta_layout.addWidget(self.command_frames_spin)

        meta_layout.addStretch()
        self.main_layout.addWidget(self.command_meta_widget)

        # Row 2: Stay on Top
        options_layout = QtWidgets.QHBoxLayout()
        options_layout.setContentsMargins(0, 0, 0, 0)
        options_layout.setSpacing(12)

        dark_gray_style = "color: #555555;"

        self.always_on_top_checkbox = QtWidgets.QCheckBox("Stay on Top")
        self.always_on_top_checkbox.setStyleSheet(dark_gray_style)
        self.always_on_top_checkbox.setChecked(self.always_on_top)
        self.always_on_top_checkbox.toggled.connect(self._on_always_on_top_toggled)
        options_layout.addWidget(self.always_on_top_checkbox)

        options_layout.addStretch()
        self.main_layout.addLayout(options_layout)

        # Tab widget for Line-L, Line-R, Line-C columns
        self.tab_widget = QtWidgets.QTabWidget()
        self.tab_widget.setStyleSheet("color: black;")
        self.main_layout.addWidget(self.tab_widget)

    def _create_label(self, text):
        """Create a styled label."""
        label = QtWidgets.QLabel(text)
        label.setStyleSheet("color: black;")
        return label

    def set_robot_model(self, robot_model):
        """Set the robot model and populate joint rows."""
        self.robot_model = robot_model
        self._populate_joints()

    def _populate_joints(self):
        """Populate joint rows based on robot model."""
        # Clear existing tabs
        while self.tab_widget.count() > 0:
            widget = self.tab_widget.widget(0)
            self.tab_widget.removeTab(0)
            widget.deleteLater()

        self.command_combos.clear()
        self.value_spins.clear()
        self.joint_rows.clear()
        self.column_layouts.clear()
        self.column_widgets.clear()

        # Get robot_model from graph's stl_viewer if not set
        robot_model = self.robot_model
        if not robot_model and self.graph:
            stl_viewer = getattr(self.graph, 'stl_viewer', None)
            if stl_viewer:
                robot_model = getattr(stl_viewer, 'robot_model', None)
        if not robot_model:
            return

        # Get joint layout from graph's joint_editor if available
        joint_editor = getattr(self.graph, 'joint_editor', None) if self.graph else None
        if joint_editor:
            self.joint_display_groups = dict(getattr(joint_editor, 'joint_display_groups', {}))
            self.joint_display_order = list(getattr(joint_editor, 'joint_display_order', []))
        else:
            self.joint_display_groups = {}
            self.joint_display_order = []

        # Get joints from robot model (joints is a dict, use joint_order for iteration)
        joints = []
        joint_order = getattr(robot_model, 'joint_order', [])
        for jname in joint_order:
            joints.append(jname)

        # Sort by display order
        if self.joint_display_order:
            ordered = [j for j in self.joint_display_order if j in joints]
            remaining = [j for j in joints if j not in ordered]
            joints = ordered + remaining

        # Group joints
        groups = {"L": [], "R": [], "C": []}
        for jname in joints:
            group = self.joint_display_groups.get(jname, self._detect_group(jname))
            groups[group].append(jname)

        # Create tabs for each group
        for group_key, group_label in [("L", "Line-L"), ("R", "Line-R"), ("C", "Line-C")]:
            if not groups[group_key]:
                continue

            scroll = QtWidgets.QScrollArea()
            scroll.setWidgetResizable(True)
            scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)

            container = QtWidgets.QWidget()
            layout = QtWidgets.QVBoxLayout(container)
            layout.setContentsMargins(4, 4, 4, 4)
            layout.setSpacing(2)

            self.column_layouts[group_key] = layout
            self.column_widgets[group_key] = container

            for jname in groups[group_key]:
                row = self._create_joint_row(jname)
                layout.addWidget(row)
                self.joint_rows[jname] = row

            layout.addStretch()
            scroll.setWidget(container)
            self.tab_widget.addTab(scroll, group_label)

    def _detect_group(self, joint_name):
        """Detect joint group from name."""
        name_lower = joint_name.lower()
        if '_l_' in name_lower or name_lower.endswith('_l') or name_lower.startswith('l_'):
            return "L"
        elif '_r_' in name_lower or name_lower.endswith('_r') or name_lower.startswith('r_'):
            return "R"
        return "C"

    def _create_joint_row(self, joint_name):
        """Create a row widget for a joint with command type combo and value."""
        row = QtWidgets.QWidget()
        layout = QtWidgets.QHBoxLayout(row)
        layout.setContentsMargins(2, 1, 2, 1)
        layout.setSpacing(4)

        # Joint name label
        name_label = QtWidgets.QLabel(joint_name)
        name_label.setFixedWidth(120)
        name_label.setStyleSheet("color: black;")
        layout.addWidget(name_label)
        self.joint_name_labels[joint_name] = name_label

        # Command type combo
        command_combo = QtWidgets.QComboBox()
        for cmd_id, cmd_name, cmd_desc in SERVO_COMMAND_TYPES:
            command_combo.addItem(f"{cmd_id}: {cmd_name}", cmd_id)
        command_combo.setCurrentIndex(0)
        command_combo.setFixedWidth(180)
        command_combo.setStyleSheet("color: black;")
        command_combo.setToolTip("Select servo command type")
        command_combo.currentIndexChanged.connect(lambda idx, jn=joint_name: self._on_command_changed(jn, idx))
        layout.addWidget(command_combo)
        self.command_combos[joint_name] = command_combo

        # Value spinbox
        value_label = QtWidgets.QLabel("Value:")
        value_label.setStyleSheet("color: black;")
        layout.addWidget(value_label)

        value_spin = QtWidgets.QSpinBox()
        value_spin.setRange(0, 255)
        value_spin.setValue(0)
        value_spin.setFixedWidth(70)
        value_spin.valueChanged.connect(lambda v, jn=joint_name: self._on_value_changed(jn, v))
        layout.addWidget(value_spin)
        self.value_spins[joint_name] = value_spin

        layout.addStretch()

        return row

    def _on_command_changed(self, joint_name, index):
        """Handle command combo change."""
        if self._updating:
            return
        self._save_to_node()

    def _on_value_changed(self, joint_name, value):
        """Handle value change."""
        if self._updating:
            return
        self._save_to_node()

    def _save_to_node(self):
        """Save current settings to the CommandNode."""
        if not self.current_command_node:
            return
        command_settings = {}
        for jname in self.command_combos:
            command_settings[jname] = {
                "command_type": self.command_combos[jname].currentData(),
                "value": self.value_spins[jname].value()
            }
        self.current_command_node.command_settings = command_settings
        self.command_changed.emit(command_settings)

    def _on_always_on_top_toggled(self, checked):
        self.always_on_top = checked
        settings = load_app_settings()
        settings["command_editor_always_on_top"] = checked
        save_app_settings(settings)
        self._apply_always_on_top(restore_visible=True)

    def _apply_always_on_top(self, restore_visible=True):
        was_visible = self.isVisible()
        geometry = self.geometry()
        flags = self.windowFlags()
        if self.always_on_top:
            flags |= QtCore.Qt.WindowStaysOnTopHint
        else:
            flags &= ~QtCore.Qt.WindowStaysOnTopHint
        self.setWindowFlags(flags)
        if restore_visible and was_visible:
            self.setGeometry(geometry)
            self.show()
            if self.always_on_top:
                self.raise_()
        # Re-parent to main window if needed
        if self.graph:
            main_window = self.graph.widget.window() if hasattr(self.graph, 'widget') else None
            if main_window:
                parent_window = self.window()
                if parent_window != main_window:
                    self.setParent(main_window)
                    self.setWindowFlags(flags | QtCore.Qt.Window)
                if restore_visible and was_visible:
                    parent_window.setGeometry(geometry)
                    parent_window.show()

    def set_command_node(self, command_node):
        """Set the current CommandNode for editing."""
        self.current_command_node = command_node
        # Populate joints if tab widget is empty (always try if no tabs)
        if self.tab_widget.count() == 0:
            self._populate_joints()
        self._refresh_command_meta_row()
        self._load_from_node()

    def _refresh_command_meta_row(self):
        """Refresh the name/frames/duration row."""
        node = self.current_command_node
        self.command_name_edit.blockSignals(True)
        self.command_frames_spin.blockSignals(True)
        if node:
            self.command_name_edit.setEnabled(True)
            self.command_frames_spin.setEnabled(True)
            self.command_name_edit.setText(getattr(node, 'command_name', 'command'))
            self.command_frames_spin.setValue(getattr(node, 'frames', 1))
            self._update_duration_label()
        else:
            self.command_name_edit.setEnabled(False)
            self.command_frames_spin.setEnabled(False)
            self.command_name_edit.clear()
            self.command_frames_spin.setValue(1)
            self.command_duration_label.setText("—")
        self.command_name_edit.blockSignals(False)
        self.command_frames_spin.blockSignals(False)

    def _update_duration_label(self):
        """Update duration label based on frames and FPS."""
        node = self.current_command_node
        if not node:
            self.command_duration_label.setText("—")
            return
        frames = getattr(node, 'frames', 1)
        fps = get_default_hz_fps()
        duration_sec = frames / fps if fps > 0 else 0
        self.command_duration_label.setText(f"{duration_sec:.2f} sec")

    def _load_from_node(self):
        """Load settings from the current CommandNode."""
        if not self.current_command_node:
            return
        self._updating = True
        command_settings = getattr(self.current_command_node, 'command_settings', {})
        for jname in self.command_combos:
            settings = command_settings.get(jname, {})
            cmd_type = settings.get("command_type", 0)
            # Find index by data
            idx = self.command_combos[jname].findData(cmd_type)
            if idx >= 0:
                self.command_combos[jname].setCurrentIndex(idx)
            self.value_spins[jname].setValue(settings.get("value", 0))
        self._updating = False

    def _apply_command_name_from_ui(self):
        """Apply name from UI to node."""
        if self.current_command_node:
            self.current_command_node.command_name = self.command_name_edit.text()
            self.current_command_node.set_name(self.command_name_edit.text())

    def _apply_command_frames_from_ui(self):
        """Apply frames from UI to node."""
        if self.current_command_node:
            self.current_command_node.frames = self.command_frames_spin.value()
            self._update_duration_label()



# =============================================================================
# LME Motion Runtime — exec-based computed motion preview
# =============================================================================
# Generic engine: exec()s user ProjectCode and calls any named function to get
# joint angles for 3D preview + Valkey send.  Not walk-specific.
# =============================================================================

class LMECommanderStub:
    """Generic _bhv_commander stub injected into the ProjectCode exec namespace.

    Provides the same attribute surface as PhysicalOn's Commander so user code
    written for cartridge export works unchanged during LME preview.
    """

    def __init__(self):
        self._variables = {f"UserVal_{i}": 0 for i in range(64)}
        self._pad = [0.0] * 24
        self._pad_lock = threading.Lock()
        self._imu: dict = {}
        self.controller = None    # Optional ProjectCode controller (.data)
        self._last_output = [0.0] * 90

    @property
    def walk(self):
        """Alias for .controller (older ProjectCode)."""
        return self.controller

    @walk.setter
    def walk(self, value):
        self.controller = value

    def set_pad(self, pad: list) -> None:
        with self._pad_lock:
            self._pad = list(pad)

    def set_imu(self, r) -> None:
        self._imu = r if isinstance(r, dict) else {}

    def get_last_output(self) -> list:
        return list(self._last_output)

    def _publish(self, data: list) -> None:
        self._last_output = list(data)

    def poll_and_update(self) -> None:
        pass

    def _trigger_motion(self, *a, **kw) -> None:
        pass


class LMEMotionRuntime:
    """Exec-based computed motion preview engine.

    Exec's user ProjectCode in a namespace that mirrors the PhysicalOn cartridge
    environment (_bhv_commander, PAD_REGISTER_VALUES, math, numpy, …), then
    calls any named function (e.g. motion_ik_step, walk_ik_step) to obtain
    joint angles for the LME 3D preview and Valkey send.
    """

    PAD_AXIS_LX = 16
    PAD_AXIS_LY = 17
    PAD_AXIS_RX = 18
    PAD_AXIS_RY = 19

    def __init__(self):
        self._commander = LMECommanderStub()
        self._ns: dict | None = None
        self._last_code: str = ""
        self._warned_fns: set = set()

    def reset(self, project_code: str = "") -> None:
        self._commander = LMECommanderStub()
        self._ns = None
        self._last_code = ""
        self._warned_fns.clear()
        if project_code:
            self._rebuild_namespace(project_code)

    def sync_pad_from_registers(self, pad_values: dict) -> None:
        scale = 127.0
        pad = self._commander._pad[:]
        # LME の Pad_Ly は UI 座標（スティック前 = +）。
        # ProjectCode の _pad_axes は Meridim 生値前提で fwd = -pad[LY] する。
        # ここで UI → Meridim Y（前 = -1）に変換してから渡す。
        pad[self.PAD_AXIS_LX] =  pad_values.get("Pad_Lx", 0) / scale
        pad[self.PAD_AXIS_LY] = -pad_values.get("Pad_Ly", 0) / scale
        pad[self.PAD_AXIS_RX] =  pad_values.get("Pad_Rx", 0) / scale
        pad[self.PAD_AXIS_RY] =  pad_values.get("Pad_Ry", 0) / scale
        self._commander.set_pad(pad)
        if self._ns is not None:
            self._ns["PAD_REGISTER_VALUES"] = pad_values  # shared ref (ProjectCode may write flags)

    def call_function(
        self,
        func_name: str,
        project_code: str,
        pad_values: dict | None = None,
    ) -> bool:
        """Exec ProjectCode if changed, then call func_name(). Returns True on success."""
        if pad_values:
            self.sync_pad_from_registers(pad_values)
        if self._ns is None or project_code != self._last_code:
            self._rebuild_namespace(project_code)
        fn = self._ns.get(func_name) if self._ns else None
        if callable(fn):
            try:
                fn()
                return True
            except Exception as exc:
                _log.warning("[LMEMotionRuntime] %s() error: %s", func_name, exc)
                import traceback; traceback.print_exc()
        elif func_name not in self._warned_fns:
            _log.warning("[LMEMotionRuntime] function '%s' not found in ProjectCode", func_name)
            self._warned_fns.add(func_name)
        return False

    def get_angles_dict(self) -> dict:
        """Extract joint angles from commander state → {joint_name: angle_deg}.

        Prefers controller.data when a ProjectCode controller is attached; else _last_output.
        Returns LME/MJCF-space degrees (undoes Meridim sign convention).

        Only emits one name per Meridim index (canonical MJCF names preferred)
        so Valkey writers never see conflicting alias duplicates for the same slot.
        """
        ctrl = getattr(self._commander, "controller", None)
        if ctrl is not None and hasattr(ctrl, "data") and ctrl.data:
            data = ctrl.data
        else:
            data = self._commander.get_last_output()
        if not data:
            return {}
        result: dict = {}
        seen_idx: set = set()
        # Prefer canonical names first so aliases don't win the slot.
        ordered_names = list(CANONICAL_MJCF_JOINTS) + [
            n for n in JOINT_TO_MERIDIM if n not in CANONICAL_MJCF_JOINTS
        ]
        for joint_name in ordered_names:
            entry = JOINT_TO_MERIDIM.get(joint_name)
            if entry is None:
                continue
            idx, mul = entry
            if idx in seen_idx or idx >= len(data):
                continue
            seen_idx.add(idx)
            result[joint_name] = data[idx] * mul
        return result

    def _rebuild_namespace(self, project_code: str) -> None:
        # Hardware / protocol names that SYSTEM_AREA provides on cartridge export.
        # Gait feel (STICK_DEAD, strides, …) belongs in ProjectCode — do not inject.
        ns: dict = {
            "__builtins__": __builtins__,
            "math": math,
            "np": np,
            "numpy": np,
            "logging": logging,
            "logger": _log,
            "dataclass": dataclass,
            "field": field,
            "PAD_REGISTER_VALUES": PAD_REGISTER_VALUES,  # shared ref
            "PAD_BTN_TRI": 3,
            "PAD_BTN_CIR": 1,
            "PAD_BTN_CRS": 0,
            "PAD_BTN_SQR": 2,
            "PAD_BTN_L1": 9,
            "PAD_BTN_R1": 10,
            "PAD_AXIS_LX": self.PAD_AXIS_LX,
            "PAD_AXIS_LY": self.PAD_AXIS_LY,
            "PAD_AXIS_RX": self.PAD_AXIS_RX,
            "PAD_AXIS_RY": self.PAD_AXIS_RY,
            "PAD_HAT_Y": 23,
            "BTN_THRESH": 0.5,
            "IDX_L_SHOULDER_PITCH": 23, "IDX_L_SHOULDER_ROLL": 25,
            "IDX_L_ELBOW_YAW": 27, "IDX_L_ELBOW_PITCH": 29,
            "IDX_C_CHEST": 51,
            "IDX_R_SHOULDER_PITCH": 53, "IDX_R_SHOULDER_ROLL": 55,
            "IDX_R_ELBOW_YAW": 57, "IDX_R_ELBOW_PITCH": 59,
            "IDX_L_HIPJOINT_ZY": 31, "IDX_L_HIPJOINT_XR": 33, "IDX_L_HIPJOINT_YP": 35,
            "IDX_L_KNEE_YP": 37, "IDX_L_ANKLE_YP": 39, "IDX_L_ANKLE_XR": 41,
            "IDX_R_HIPJOINT_ZY": 61, "IDX_R_HIPJOINT_XR": 63, "IDX_R_HIPJOINT_YP": 65,
            "IDX_R_KNEE_YP": 67, "IDX_R_ANKLE_YP": 69, "IDX_R_ANKLE_XR": 71,
            "LOOP_HZ": 100,
            "MOTION_BOOT": [],
            "_bhv_commander": self._commander,
            "_decode_meridim_pad": lambda s: {},
        }
        try:
            exec(compile(project_code, "<ProjectCode>", "exec"), ns)
            _log.debug("[LMEMotionRuntime] ProjectCode exec OK, defined names: %s",
                       [k for k in ns if not k.startswith("__")])
        except Exception as exc:
            _log.warning("[LMEMotionRuntime] ProjectCode exec error: %s", exc)
            import traceback; traceback.print_exc()

        # Optional motion controller from ProjectCode (.data array).
        if self._commander.controller is None:
            inst = None
            conventional = []
            for name in ("MotionController", "WalkController", "Controller"):
                cls = ns.get(name)
                if isinstance(cls, type):
                    conventional.append(cls)
            others = []
            for name, obj in ns.items():
                if obj in conventional or not isinstance(obj, type) or name.startswith("_"):
                    continue
                if getattr(obj, "__dataclass_fields__", None):
                    continue
                others.append(obj)
            for cls in conventional + others:
                try:
                    cand = cls()
                except Exception as exc:
                    _log.warning("[LMEMotionRuntime] could not instantiate %s: %s", cls, exc)
                    continue
                if isinstance(getattr(cand, "data", None), list) or cls in conventional:
                    inst = cand
                    break
            self._commander.controller = inst

        self._ns = ns
        self._last_code = project_code


# =============================================================================
# Cartridge graph linearizer (bundled from cartridge_export/graph_to_lines.py)
# =============================================================================

from typing import Any


# ---------------------------------------------------------------------------
# LME → cartridge lexical translation tables
# ---------------------------------------------------------------------------

_PAD_TO_BTN_CODE = {
    "Pad_L1":        "L1",
    "Pad_R1":        "R1",
    "Pad_Triangle":  "TRI",
    "Pad_Circle":    "CIR",
    "Pad_Square":    "SQR",
    "Pad_Cross":     "CRS",
    "Pad_DPad_Up":   "UP",
    "Pad_DPad_Down": "DOWN",
    "Pad_DPad_R_UP": "UP",
    "Pad_DPad_R_DOWN": "DOWN",
    "Pad_Lx": "LX",
    "Pad_Ly": "LY",
    "Pad_Rx": "RX",
    "Pad_Ry": "RY",
}

# pad_btn_* bit name → cartridge button code (for Pad_btn bitmask conditions).
_PAD_BIT_TO_BTN_CODE = {
    "pad_btn_l1": "L1",
    "pad_btn_r1": "R1",
    "pad_btn_r_up": "TRI",
    "pad_btn_r_right": "CIR",
    "pad_btn_r_left": "SQR",
    "pad_btn_r_down": "CRS",
    "pad_btn_l_up": "UP",
    "pad_btn_l_down": "DOWN",
}

_LME_OP_TO_CARTRIDGE_OP = {
    "==": "EQ",
    "!=": "NE",
    ">":  "GT",
    ">=": "GE",
    "<":  "LT",
    "<=": "LE",
}


@dataclass
class LinearizeWarning:
    action_index: int
    node_id: str
    node_type: str
    message: str


@dataclass
class LinearizeResult:
    lines: list = field(default_factory=list)
    warnings: list = field(default_factory=list)
    joints_used: set = field(default_factory=set)


MOTION_REF_SENTINEL = "MOTION_REF"


def is_motion_ref(value: Any) -> bool:
    return (
        isinstance(value, tuple)
        and len(value) == 2
        and value[0] == MOTION_REF_SENTINEL
    )


def linearize_action_data(
    action_data: dict,
    action_index: int = 0,
) -> LinearizeResult:
    """Linearise a single Action's serialised graph into cartridge tuples."""
    result = LinearizeResult()
    if not isinstance(action_data, dict):
        return result

    playback_fps = int(action_data.get("playback", {}).get("fps", 100) or 100)

    nodes_by_id = {n["id"]: n for n in action_data.get("nodes", []) if isinstance(n, dict)}
    edges = action_data.get("edges", []) or []

    outgoing: dict[str, dict[int, list[str]]] = {}
    entry_id: str | None = None
    for edge in edges:
        if not isinstance(edge, dict):
            continue
        src = edge.get("from")
        dst = edge.get("to")
        if not src or not dst:
            continue
        port = int(edge.get("from_port", 0))
        priority = int(edge.get("priority", 0))
        if src == "start":
            entry_id = dst
            continue
        outgoing.setdefault(src, {}).setdefault(port, []).append((priority, dst))

    outgoing_sorted: dict[str, dict[int, list[str]]] = {}
    for src, per_port in outgoing.items():
        outgoing_sorted[src] = {
            p: [dst for _, dst in sorted(children, key=lambda x: x[0])]
            for p, children in per_port.items()
        }

    if entry_id is None or entry_id not in nodes_by_id:
        return result

    lines = result.lines
    node_to_line: dict[str, int] = {}
    pending_backrefs: list[tuple[int, str]] = []
    pending_end_jumps: list[int] = []
    in_progress: set[str] = set()

    def add_warning(node_id: str, node_type: str, message: str) -> None:
        result.warnings.append(LinearizeWarning(
            action_index=action_index,
            node_id=node_id,
            node_type=node_type,
            message=message,
        ))

    def first_child(node_id: str, port: int = 0) -> str | None:
        children = outgoing_sorted.get(node_id, {}).get(port, [])
        return children[0] if children else None

    def motion_end_line() -> int:
        return len(lines)

    def emit(tup) -> int:
        idx = len(lines)
        lines.append(tup)
        return idx

    def emit_jump_to_end() -> None:
        idx = emit(("JUMP", None))
        pending_end_jumps.append(idx)

    def visit(node_id: str) -> None:
        if node_id in node_to_line:
            emit(("JUMP", node_to_line[node_id]))
            return
        if node_id in in_progress:
            back_idx = emit(("JUMP", None))
            pending_backrefs.append((back_idx, node_id))
            return
        node = nodes_by_id.get(node_id)
        if node is None:
            add_warning(node_id, "?", "referenced node not found")
            return

        node_type = node.get("node_type")
        in_progress.add(node_id)
        try:
            if node_type == "pose":
                _emit_pose(node)
            elif node_type == "define":
                _emit_define(node)
            elif node_type == "wait":
                _emit_wait(node)
            elif node_type == "branch":
                _emit_branch(node)
            elif node_type == "jump":
                _emit_jump(node)
            elif node_type == "memo":
                # MemoNode は注釈用途のため Export では無視。警告も出さない。
                # 出力エッジが繋がっていれば後続だけ辿る。
                nxt = first_child(node_id, 0)
                if nxt is not None:
                    visit(nxt)
            elif node_type in ("command", "mix"):
                add_warning(
                    node_id, node_type,
                    f"{node_type.capitalize()}Node skipped by V1 exporter",
                )
                nxt = first_child(node_id, 0)
                if nxt is not None:
                    visit(nxt)
            else:
                add_warning(node_id, node_type or "?", f"unknown node_type '{node_type}'")
        finally:
            in_progress.discard(node_id)

    def _emit_pose(node: dict) -> None:
        joints_map: dict[int, float] = {}
        easing_map: dict[int, int] = {}
        angles = node.get("angles_deg") or {}
        node_easings = node.get("joint_easings") or {}
        joints_used_local: list[str] = []
        for jname, deg in angles.items():
            converted = meridim_angle_from_joint(jname, float(deg))
            if converted is None:
                add_warning(
                    node["id"], "pose",
                    f"joint '{jname}' has no meridim mapping — angle dropped",
                )
                continue
            m_idx, meridim_angle = converted
            joints_map[m_idx] = meridim_angle
            joints_used_local.append(jname)
            easing_str = node_easings.get(jname)
            if easing_str is not None:
                easing_map[m_idx] = _parse_easing_index(easing_str)

        duration_ms = _pose_duration_ms(node, playback_fps)
        if easing_map:
            line_idx = emit((joints_map, duration_ms, easing_map))
        else:
            line_idx = emit((joints_map, duration_ms))
        node_to_line[node["id"]] = line_idx
        for jname in joints_used_local:
            result.joints_used.add(jname)

        nxt = first_child(node["id"], 0)
        if nxt is not None:
            visit(nxt)

    def _emit_wait(node: dict) -> None:
        """WaitNode: N フレーム相当の時間、直前の目標値を保持したまま待つ。

        MotionPlayer の POS 命令は target joints が空 dict なら補間対象が無く、
        タイマーだけ進むので「その場で待つ」として機能する。Servo 側へは
        新しい指令を送らないため、直前の Pose の目標値がそのまま保持される。
        """
        duration_ms = _pose_duration_ms(node, playback_fps)
        line_idx = emit(({}, duration_ms))
        node_to_line[node["id"]] = line_idx

        nxt = first_child(node["id"], 0)
        if nxt is not None:
            visit(nxt)

    def _emit_define(node: dict) -> None:
        var = f"UserVal_{int(node.get('define_uv_index', 0))}"
        kind = node.get("define_kind", "literal")
        if kind == "literal":
            value = _int_or_zero(node.get("define_literal", 0))
        else:
            reg_name = node.get("define_register_name", "")
            value = _resolve_register_static(reg_name)
            add_warning(
                node["id"], "define",
                f"DefineNode kind='register' ('{reg_name}') statically resolved to {value}",
            )
        line_idx = emit(("SET", var, "MOV", value))
        node_to_line[node["id"]] = line_idx

        nxt = first_child(node["id"], 0)
        if nxt is not None:
            visit(nxt)

    _ANALOG_AXIS_TO_PAD_KEY = {
        "Lx": "LX", "Ly": "LY",
        "Rx": "RX", "Ry": "RY",
        "L2v": "Pad_L2v", "R2v": "Pad_R2v",
    }

    def _emit_branch(node: dict) -> None:
        if not node.get("branching_enabled", False):
            nxt = first_child(node["id"], 0)
            if nxt is not None:
                visit(nxt)
            return

        swapped = node.get("branch_outputs_swapped", False)
        invert = False

        if node.get("branch_if_pad_analog_enabled", False):
            axis = node.get("branch_if_pad_analog_axis", "Lx")
            op_str = node.get("branch_if_pad_analog_op", ">=")
            if axis == "Ly":
                raw = int(node.get("branch_if_pad_analog_threshold", 0))
                _t = abs(raw / 127.0)
                pad_key = "LY"
                cart_op = "LE" if op_str == ">=" else "GE"
                threshold = -_t if op_str == ">=" else _t
                cmp_idx = emit(("CMP", pad_key, cart_op, threshold, None, None))
            else:
                raw = int(node.get("branch_if_pad_analog_threshold", 0))
                scale = 255.0 if axis in ("L2v", "R2v") else 127.0
                threshold = raw / scale
                pad_key = _ANALOG_AXIS_TO_PAD_KEY.get(axis, "LX")
                cart_op = "GE" if op_str == ">=" else "LE"
                cmp_idx = emit(("CMP", pad_key, cart_op, threshold, None, None))
            node_to_line[node["id"]] = cmp_idx
        elif node.get("branch_if_pad_enabled", False):
            pad_btn = node.get("branch_if_pad_button", "L1")
            cmp_idx = emit(("CMP", pad_btn, "HELD", None, None))
            node_to_line[node["id"]] = cmp_idx
            # CMP TRUE = held → "to" port (action path). No invert needed;
            # to_port below already selects the "to" port correctly.
        else:
            left = node.get("branch_if_left", "UserVal_0")
            op = node.get("branch_if_op", "==")
            right = node.get("branch_if_right", "UserVal_1")

            held_cmp = _try_held_cmp(left, op, right)
            if held_cmp is not None:
                btn_src, invert = held_cmp
                cmp_idx = emit(("CMP", btn_src, "HELD", None, None))
                node_to_line[node["id"]] = cmp_idx
            else:
                src = _to_cmp_src(left)
                cart_op = _LME_OP_TO_CARTRIDGE_OP.get(op)
                if cart_op is None:
                    add_warning(node["id"], "branch", f"unsupported op '{op}' — using EQ 0")
                    cart_op = "EQ"
                value = _to_cmp_value(right)
                if value is None:
                    add_warning(
                        node["id"], "branch",
                        f"right-hand side '{right}' is not a static literal — using 0",
                    )
                    value = 0
                cmp_idx = emit(("CMP", src, cart_op, value, None, None))
                node_to_line[node["id"]] = cmp_idx

        # LME port convention: swapped=True → port-0="to", port-1="otherwise"
        #                      swapped=False → port-0="otherwise", port-1="to"
        to_port = 0 if swapped else 1
        other_port = 1 if swapped else 0

        to_children = outgoing_sorted.get(node["id"], {}).get(to_port, [])
        other_children = outgoing_sorted.get(node["id"], {}).get(other_port, [])

        to_line = len(lines)
        if to_children:
            visit(to_children[0])
            if other_children:
                emit_jump_to_end()

        other_line = len(lines)
        if other_children:
            visit(other_children[0])

        branch_line = to_line if to_children else motion_end_line()
        next_line = other_line if other_children else motion_end_line()

        if invert:
            branch_line, next_line = next_line, branch_line

        seg = lines[cmp_idx]
        if len(seg) == 5:
            lines[cmp_idx] = ("CMP", seg[1], seg[2], branch_line, next_line)
        else:
            lines[cmp_idx] = (seg[0], seg[1], seg[2], seg[3], branch_line, next_line)

    def _emit_jump(node: dict) -> None:
        if node.get("jump_type") == "function":
            func_name = node.get("jump_target_function", "")
            line_idx = emit(("CALL", func_name))
            node_to_line[node["id"]] = line_idx
            nxt = first_child(node["id"], 0)
            if nxt is not None:
                visit(nxt)
        else:
            target_action_idx = int(node.get("jump_target_action_index", 0))
            line_idx = emit(("JUMP", (MOTION_REF_SENTINEL, target_action_idx), 0))
            node_to_line[node["id"]] = line_idx

    visit(entry_id)

    for back_idx, target_id in pending_backrefs:
        target_line = node_to_line.get(target_id)
        if target_line is None:
            add_warning(target_id, "?", "unresolved back-edge target; falling back to line 0")
            target_line = 0
        lines[back_idx] = ("JUMP", target_line)

    motion_end = len(lines)
    for jmp_idx in pending_end_jumps:
        lines[jmp_idx] = ("JUMP", motion_end)

    return result


# ---------------------------------------------------------------------------
# Branch pattern helpers
# ---------------------------------------------------------------------------

def _try_held_cmp(left: str, op: str, right: str) -> tuple[str, bool] | None:
    """Detect LME pad conditions that map to cartridge HELD CMP.

    Returns (button_src, invert_branches) or None.
    *invert_branches* swaps branch_line / next_line (release vs hold).
    """
    left_btn = _PAD_TO_BTN_CODE.get(left)
    right_val = _to_cmp_value(right)

    # Pad_L1 == 1  /  Pad_L1 != 0  → HELD (hold branch)
    if left_btn:
        if op == "==" and right_val == 1:
            return left_btn, False
        if op == "==" and right_val == 0:
            return left_btn, True
        if op == "!=" and right_val == 0:
            return left_btn, False
        if op == "!=" and right_val == 1:
            return left_btn, True

    # Pad_btn == pad_btn_l1 (bitmask) → HELD on inferred button
    if left in ("Pad_btn", "pad_btn", "pad_buttons"):
        if op == "==" and right_val is not None:
            btn = _PAD_BIT_TO_BTN_CODE.get(str(right).lower())
            if btn:
                return btn, False
        if op == "!=" and right_val is not None:
            btn = _PAD_BIT_TO_BTN_CODE.get(str(right).lower())
            if btn:
                return btn, True

    # Right side is pad_btn_* bit name with Pad_btn on left
    if left in ("Pad_btn", "pad_btn", "pad_buttons") and op == "==":
        btn = _PAD_BIT_TO_BTN_CODE.get(str(right).lower())
        if btn:
            return btn, False

    return None


def _parse_easing_index(easing_str: str) -> int:
    """Parse LME easing option string ('0: linear') to integer index."""
    s = str(easing_str or "0").strip()
    if ":" in s:
        prefix = s.split(":", 1)[0].strip()
        if prefix.isdigit():
            return int(prefix)
    return 0


def _pose_duration_ms(node: dict, fps: int = 100) -> int:
    frames = node.get("frames")
    if frames is not None:
        try:
            return max(1, int(round(int(frames) / fps * 1000.0)))
        except (TypeError, ValueError):
            pass
    dur = node.get("duration")
    if dur is not None:
        try:
            return int(round(float(dur) * 1000.0))
        except (TypeError, ValueError):
            pass
    return 500


def _int_or_zero(v: Any) -> int:
    try:
        return int(v)
    except (TypeError, ValueError):
        try:
            return int(float(v))
        except (TypeError, ValueError):
            return 0


def _to_cmp_src(name: str) -> str:
    if not name:
        return "UserVal_0"
    if name in _PAD_TO_BTN_CODE:
        return _PAD_TO_BTN_CODE[name]
    if name.startswith("UserVal_"):
        return name
    return name


def _to_cmp_value(name_or_literal: Any) -> int | None:
    if isinstance(name_or_literal, (int, float)):
        return int(name_or_literal)
    if isinstance(name_or_literal, str):
        s = name_or_literal.strip()
        try:
            return int(s)
        except (TypeError, ValueError):
            pass
        if s.lower() in PAD_BUTTON_BIT_VALUES:
            return int(PAD_BUTTON_BIT_VALUES[s.lower()])
    return None


def _resolve_register_static(name: str) -> int:
    if not name:
        return 0
    if name in PAD_REGISTER_VALUES:
        return int(PAD_REGISTER_VALUES[name])
    low = name.lower()
    if low in PAD_REGISTER_ALIASES:
        return int(PAD_REGISTER_VALUES.get(PAD_REGISTER_ALIASES[low], 0))
    if low in PAD_BUTTON_BIT_VALUES:
        return int(PAD_BUTTON_BIT_VALUES[low])
    try:
        return int(name)
    except (TypeError, ValueError):
        return 0


# =============================================================================
# Logic Cartridge export — template + exporter (bundled for LME distribution)
# =============================================================================

HEADER_TEMPLATE = '''#!/usr/bin/env python3
"""Logic_cartridge_{robot_name}.py — auto-generated by LegacyMotionEditor.

Generated: {timestamp}
Source:    {source_project}

Loaded by Meridian_console / PhysicalOn as a Logic Cartridge and run at
{loop_hz} Hz (see LOOP_HZ). SYSTEM AREA = frozen MotionPlayer + Commander runtime.
USER WORK AREA / ProjectCode supplies controllers, params, and CALL helpers;
regenerated on every export.
"""

import json
import logging
import math
import os
import threading
import time
from dataclasses import dataclass, field, asdict

import numpy as np
'''


SYSTEM_AREA = r'''
# ══════════════════════════════════════════════════════════════════════════════
# SYSTEM AREA = frozen MotionPlayer + Commander runtime.
# Controllers / gait params / CALL helpers live in ProjectCode (USER WORK AREA).
# ══════════════════════════════════════════════════════════════════════════════

# ── Pad layout ────────────────────────────────────────────────────────────────
PAD_BTN_TRI  =  3   # △
PAD_BTN_CIR  =  1   # ○
PAD_BTN_CRS  =  0   # ×
PAD_BTN_SQR  =  2   # □
PAD_BTN_L1   =  9   # L1
PAD_BTN_R1   = 10   # R1
PAD_AXIS_LX  = 16   # left  stick X  (left=-1, right=+1)
PAD_AXIS_LY  = 17   # left  stick Y  (up=-1=fwd, down=+1=back)
PAD_AXIS_RX  = 18   # right stick X
PAD_AXIS_RY  = 19   # right stick Y
PAD_HAT_Y    = 23   # D-pad Y  (+1=up, -1=down)
BTN_THRESH   = 0.5

_BTN_NAME_TO_PAD_IDX: dict[str, int] = {
    "L1":  PAD_BTN_L1,  "R1":  PAD_BTN_R1,
    "TRI": PAD_BTN_TRI, "CIR": PAD_BTN_CIR,
    "SQR": PAD_BTN_SQR, "CRS": PAD_BTN_CRS,
}

# ── Meridim90 joint indices ───────────────────────────────────────────────────
IDX_L_SHOULDER_PITCH = 23;  IDX_L_SHOULDER_ROLL = 25
IDX_L_ELBOW_YAW      = 27;  IDX_L_ELBOW_PITCH   = 29
IDX_C_CHEST          = 51
IDX_R_SHOULDER_PITCH = 53;  IDX_R_SHOULDER_ROLL = 55
IDX_R_ELBOW_YAW      = 57;  IDX_R_ELBOW_PITCH   = 59
IDX_L_HIPJOINT_ZY = 31;  IDX_L_HIPJOINT_XR = 33;  IDX_L_HIPJOINT_YP = 35
IDX_L_KNEE_YP     = 37;  IDX_L_ANKLE_YP    = 39;  IDX_L_ANKLE_XR    = 41
IDX_R_HIPJOINT_ZY = 61;  IDX_R_HIPJOINT_XR = 63;  IDX_R_HIPJOINT_YP = 65
IDX_R_KNEE_YP     = 67;  IDX_R_ANKLE_YP    = 69;  IDX_R_ANKLE_XR    = 71

# ── Commander states ──────────────────────────────────────────────────────────
STATE_IDLE   = "IDLE"
STATE_MOTION = "MOTION"

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

# ── Pad decoder ───────────────────────────────────────────────────────────────
def _decode_meridim_pad(s: list) -> list:
    """s_meridim[15..18] → 24-element pad list."""
    pad = [0.0] * 24
    bitmask = int(float(s[15])) & 0xFFFF
    for bit, idx in {4096: 3, 8192: 1, 16384: 0, 32768: 2, 1024: 9, 2048: 10}.items():
        if bitmask & bit:
            pad[idx] = 1.0
    if bitmask & 16:    pad[23] =  1.0
    elif bitmask & 64:  pad[23] = -1.0
    if bitmask & 32:    pad[22] =  1.0
    elif bitmask & 128: pad[22] = -1.0

    def _unpack(v):
        v = int(float(v)) & 0xFFFF
        lo = v & 0xFF;          lo = lo - 256 if lo >= 128 else lo
        hi = (v >> 8) & 0xFF;   hi = hi - 256 if hi >= 128 else hi
        return lo, hi

    ly, lx = _unpack(s[16])
    ry, rx = _unpack(s[17])
    pad[16] = lx / 127.0
    pad[17] = ly / 127.0
    pad[18] = rx / 127.0
    pad[19] = ry / 127.0
    v18 = int(float(s[18])) & 0xFFFF
    pad[20] = (v18 & 0xFF) / 255.0
    pad[21] = ((v18 >> 8) & 0xFF) / 255.0
    return pad


# ── Pad helper ────────────────────────────────────────────────────────────────
class Pad:
    """パッド入力をわかりやすく取得するヘルパー。
    使い方: pad = Pad(s)   → pad.l1 / pad.tri / pad.lx / pad.ly
    """
    def __init__(self, s: list) -> None:
        raw     = _decode_meridim_pad(s)
        bitmask = int(float(s[15])) & 0xFFFF
        self.tri      = bool(bitmask & 4096)
        self.cir      = bool(bitmask & 8192)
        self.crs      = bool(bitmask & 16384)
        self.sqr      = bool(bitmask & 32768)
        self.l1       = bool(bitmask & 1024)
        self.r1       = bool(bitmask & 2048)
        self.hat_up   = bool(bitmask & 16)
        self.hat_down = bool(bitmask & 64)
        self.hat_r    = bool(bitmask & 32)
        self.hat_l    = bool(bitmask & 128)
        self.lx = raw[PAD_AXIS_LX]
        self.ly = raw[PAD_AXIS_LY]
        self.rx = raw[PAD_AXIS_RX]
        self.ry = raw[PAD_AXIS_RY]


# ── Joint helpers ─────────────────────────────────────────────────────────────
def get_angle(meridim: list, joint_name: str) -> float:
    """関節角度 [deg] を取得する。JOINTS の辞書キーを使う。"""
    idx = JOINTS.get(joint_name)
    if idx is None:
        raise KeyError(f"Joint '{joint_name}' not in JOINTS dict")
    return float(meridim[idx]) / 100.0


def set_angle(s: list, joint_name: str, deg: float) -> None:
    """関節角度コマンドを s_meridim に書き込む。トルクフラグも同時に立てる。"""
    idx = JOINTS.get(joint_name)
    if idx is None:
        raise KeyError(f"Joint '{joint_name}' not in JOINTS dict")
    s[idx - 1] = 1.0
    s[idx]     = float(deg) * 100.0


def get_imu(r: list) -> dict:
    """IMUデータを辞書で返す (PhysicalOn の MuJoCo 仮想IMU値)。"""
    return {
        "ax": float(r[2])  / 100.0, "ay": float(r[3])  / 100.0,
        "az": float(r[4])  / 100.0,
        "gx": float(r[5])  / 100.0, "gy": float(r[6])  / 100.0,
        "gz": float(r[7])  / 100.0,
        "roll":  float(r[12]) / 100.0,
        "pitch": float(r[13]) / 100.0,
        "yaw":   float(r[14]) / 100.0,
    }


# ── MotionPlayer ──────────────────────────────────────────────────────────────
def _apply_easing(idx: int, t: float) -> float:
    """Apply easing function by index (matches LME EASING_PRESETS order)."""
    t = max(0.0, min(1.0, t))
    if idx == 0:  return t
    if idx == 1:  return 1 - math.cos(t * math.pi / 2)
    if idx == 2:  return math.sin(t * math.pi / 2)
    if idx == 3:  return -(math.cos(math.pi * t) - 1) / 2
    if idx == 4:  return t * t
    if idx == 5:  return 1 - (1 - t) * (1 - t)
    if idx == 6:  return 2 * t * t if t < 0.5 else 1 - (-2 * t + 2) ** 2 / 2
    if idx == 7:  return t ** 3
    if idx == 8:  return 1 - (1 - t) ** 3
    if idx == 9:  return 4 * t ** 3 if t < 0.5 else 1 - (-2 * t + 2) ** 3 / 2
    if idx == 10: return t ** 4
    if idx == 11: return 1 - (1 - t) ** 4
    if idx == 12: return 8 * t ** 4 if t < 0.5 else 1 - (-2 * t + 2) ** 4 / 2
    if idx == 13: return t ** 5
    if idx == 14: return 1 - (1 - t) ** 5
    if idx == 15: return 16 * t ** 5 if t < 0.5 else 1 - (-2 * t + 2) ** 5 / 2
    if idx == 16: return 0.0 if t == 0 else (2 ** (10 * t - 10) if t < 1 else 1.0)
    if idx == 17: return 1.0 if t == 1 else 1 - 2 ** (-10 * t)
    if idx == 18:
        if t == 0: return 0.0
        if t == 1: return 1.0
        return 2 ** (20 * t - 10) / 2 if t < 0.5 else (2 - 2 ** (-20 * t + 10)) / 2
    if idx == 19: return 1 - math.sqrt(max(0.0, 1 - t * t))
    if idx == 20: return math.sqrt(max(0.0, 1 - (t - 1) ** 2))
    if idx == 21:
        return (1 - math.sqrt(max(0.0, 1 - (2 * t) ** 2))) / 2 if t < 0.5 \
            else (math.sqrt(max(0.0, 1 - (-2 * t + 2) ** 2)) + 1) / 2
    if idx == 22:  # easeInBack
        c1 = 1.70158; c3 = c1 + 1
        return c3 * t * t * t - c1 * t * t
    if idx == 23:  # easeOutBack
        c1 = 1.70158; c3 = c1 + 1
        return 1 + c3 * ((t - 1) ** 3) + c1 * ((t - 1) ** 2)
    if idx == 24:  # easeInOutBack
        c1 = 1.70158; c2 = c1 * 1.525
        if t < 0.5: return ((2 * t) ** 2 * ((c2 + 1) * 2 * t - c2)) / 2
        return (((2 * t - 2) ** 2) * ((c2 + 1) * (t * 2 - 2) + c2) + 2) / 2
    if idx == 25:  # easeInElastic
        if t == 0: return 0.0
        if t == 1: return 1.0
        return -(2 ** (10 * t - 10)) * math.sin((t * 10 - 10.75) * (2 * math.pi / 3))
    if idx == 26:  # easeOutElastic
        if t == 0: return 0.0
        if t == 1: return 1.0
        return (2 ** (-10 * t)) * math.sin((t * 10 - 0.75) * (2 * math.pi / 3)) + 1
    if idx == 27:  # easeInOutElastic
        if t == 0: return 0.0
        if t == 1: return 1.0
        c5 = 2 * math.pi / 4.5
        if t < 0.5: return -((2 ** (20 * t - 10)) * math.sin((20 * t - 11.125) * c5)) / 2
        return ((2 ** (-20 * t + 10)) * math.sin((20 * t - 11.125) * c5)) / 2 + 1
    if idx == 28:  # easeInBounce (= 1 - easeOutBounce(1-t))
        x = 1.0 - t; n1 = 7.5625; d1 = 2.75
        if x < 1 / d1: r = n1 * x * x
        elif x < 2 / d1: x -= 1.5 / d1; r = n1 * x * x + 0.75
        elif x < 2.5 / d1: x -= 2.25 / d1; r = n1 * x * x + 0.9375
        else: x -= 2.625 / d1; r = n1 * x * x + 0.984375
        return 1 - r
    if idx == 29:  # easeOutBounce
        n1 = 7.5625; d1 = 2.75
        if t < 1 / d1: return n1 * t * t
        if t < 2 / d1: t -= 1.5 / d1; return n1 * t * t + 0.75
        if t < 2.5 / d1: t -= 2.25 / d1; return n1 * t * t + 0.9375
        t -= 2.625 / d1; return n1 * t * t + 0.984375
    if idx == 30:  # easeInOutBounce
        n1 = 7.5625; d1 = 2.75
        if t < 0.5:
            x = 1 - 2 * t
            if x < 1 / d1: r = n1 * x * x
            elif x < 2 / d1: x -= 1.5 / d1; r = n1 * x * x + 0.75
            elif x < 2.5 / d1: x -= 2.25 / d1; r = n1 * x * x + 0.9375
            else: x -= 2.625 / d1; r = n1 * x * x + 0.984375
            return (1 - r) / 2
        x = 2 * t - 1
        if x < 1 / d1: r = n1 * x * x
        elif x < 2 / d1: x -= 1.5 / d1; r = n1 * x * x + 0.75
        elif x < 2.5 / d1: x -= 2.25 / d1; r = n1 * x * x + 0.9375
        else: x -= 2.625 / d1; r = n1 * x * x + 0.984375
        return (1 + r) / 2
    return 2 * t * t if t < 0.5 else 1 - (-2 * t + 2) ** 2 / 2  # fallback: easeInOutQuad

_MP_MAX_INSTANT = 100

class MotionPlayer:
    """POS/CMP/JUMP/SET セグメントによるキーフレームモーション再生エンジン。

    POS  ({joint_idx: angle_deg, ...}, time_ms)
    CMP  ("CMP", src, op, value, branch_line, next_line)
    JUMP ("JUMP", line_no)  /  ("JUMP", motion_ref, line_no)
    SET  ("SET", var_name, op, value)
    """

    def __init__(self, segments: list, base_data: list, loop_hz: int = 100,
                 variables: dict | None = None, start_line: int = 0) -> None:
        self._segments   = segments
        self._data       = list(base_data)
        self._frame_dt   = 1000.0 / loop_hz
        self._pc         = start_line
        self._last_pc    = -1
        self._tmr        = 0.0
        self._next_ms    = 0.0
        self._last: dict[int, float] = {}
        self._variables  = variables if variables is not None else {}
        self._jump_target: tuple | None = None

    @property
    def data(self) -> list:
        return self._data

    @property
    def jump_target(self) -> tuple | None:
        return self._jump_target

    @staticmethod
    def _ease_in_out_quad(t: float) -> float:
        t = max(0.0, min(1.0, t))
        return 2.0 * t * t if t < 0.5 else 1.0 - (-2.0 * t + 2.0) ** 2 / 2.0

    def is_done(self) -> bool:
        return self._pc >= len(self._segments)

    def _get_type(self, seg: tuple) -> str:
        return "POS" if isinstance(seg[0], dict) else seg[0]

    def _eval_src(self, src: str, pad: list | None) -> float:
        btn_idx = _BTN_NAME_TO_PAD_IDX.get(src)
        if btn_idx is not None:
            return float(pad[btn_idx]) if pad is not None else 0.0
        if src == "UP":
            return 1.0 if (pad is not None and pad[PAD_HAT_Y] >= 1.0) else 0.0
        if src == "DOWN":
            return 1.0 if (pad is not None and pad[PAD_HAT_Y] <= -1.0) else 0.0
        if src == "LX":  return float(pad[PAD_AXIS_LX]) if pad is not None else 0.0
        if src == "LY":  return float(pad[PAD_AXIS_LY]) if pad is not None else 0.0
        if src == "RX":  return float(pad[PAD_AXIS_RX]) if pad is not None else 0.0
        if src == "RY":  return float(pad[PAD_AXIS_RY]) if pad is not None else 0.0
        return self._variables.get(src, 0.0)

    def _eval_cmp(self, src: str, op: str, value: float, pad: list | None) -> bool:
        v = self._eval_src(src, pad)
        if op == "HELD":  return v >= BTN_THRESH
        if op == "EQ":    return v == value
        if op == "NE":    return v != value
        if op == "GT":    return v >  value
        if op == "GE":    return v >= value
        if op == "LT":    return v <  value
        if op == "LE":    return v <= value
        if op == "BIT":   return (int(v) & int(value)) != 0
        if op == "NBIT":  return (int(v) & int(value)) == 0
        return False

    def _apply_set(self, var: str, op: str, value: float) -> None:
        cur = self._variables.get(var, 0.0)
        if   op == "MOV": self._variables[var] = float(value)
        elif op == "ADD": self._variables[var] = cur + value
        elif op == "SUB": self._variables[var] = cur - value
        elif op == "MUL": self._variables[var] = cur * value
        elif op == "DIV":
            if value != 0: self._variables[var] = cur / value

    def _jump_to(self, line: int | None) -> None:
        self._pc      = (self._pc + 1) if line is None else line
        self._last_pc = -1

    def advance(self, pad: list | None = None) -> list:
        if self.is_done():
            return self._data

        instant = 0
        while not self.is_done() and instant < _MP_MAX_INSTANT:
            seg   = self._segments[self._pc]
            type_ = self._get_type(seg)

            if type_ == "POS":
                target_joints: dict = seg[0]
                duration_ms: float  = float(seg[1])
                if self._pc != self._last_pc:
                    self._tmr     = 0.0
                    self._next_ms = duration_ms
                    self._last    = {idx: self._data[idx] for idx in target_joints}
                    self._last_pc = self._pc
                # Map easing over N-1 intervals so the last frame naturally reaches target.
                # For single-frame segments (next_ms <= frame_dt), jump straight to t=1.
                _span = self._next_ms - self._frame_dt
                t = min(1.0, self._tmr / _span) if _span > 0 else 1.0
                joint_easing_ = seg[2] if len(seg) > 2 else {}
                for idx, angle_end in target_joints.items():
                    es_t = _apply_easing(joint_easing_.get(idx, 0) if joint_easing_ else 0, t)
                    self._data[idx]     = self._last[idx] + (angle_end - self._last[idx]) * es_t
                    self._data[idx - 1] = 1.0
                self._tmr += self._frame_dt
                if self._tmr >= self._next_ms:
                    for idx, angle in target_joints.items():
                        self._data[idx] = angle
                    self._pc      += 1
                    self._last_pc  = -1
                break

            elif type_ == "CMP":
                if len(seg) == 5:
                    _, src, op, branch_line, next_line = seg; value = None
                else:
                    _, src, op, value, branch_line, next_line = seg
                old_pc = self._pc
                if self._eval_cmp(src, op, value, pad):
                    self._jump_to(branch_line)
                else:
                    self._jump_to(next_line)
                if self._pc == old_pc:
                    break
                instant += 1

            elif type_ == "JUMP":
                if len(seg) == 2:
                    old_pc = self._pc
                    self._jump_to(seg[1])
                    if self._pc == old_pc:
                        break
                    instant += 1
                else:
                    self._jump_target = (seg[1], seg[2])
                    break

            elif type_ == "SET":
                _, var, op, value = seg
                self._apply_set(var, op, value)
                self._pc      += 1
                self._last_pc  = -1
                instant += 1

            elif type_ == "CALL":
                fn_name = seg[1] if len(seg) > 1 else ""
                if fn_name:
                    fn = globals().get(fn_name)
                    if callable(fn):
                        fn()
                    cmdr = globals().get("_bhv_commander")
                    if cmdr is not None:
                        pose = getattr(cmdr, "_pose_data", None)
                        if callable(pose):
                            self._data = list(pose())
                        elif getattr(cmdr, "controller", None) is not None and getattr(cmdr.controller, "data", None):
                            self._data = list(cmdr.controller.data)
                self._pc      += 1
                self._last_pc  = -1
                # One CALL per control tick (matches LME PlaybackController function JumpNodes).
                break

        if instant >= _MP_MAX_INSTANT:
            logger.warning("[MotionPlayer] instant step limit — possible infinite loop")

        return self._data


# ── Commander ─────────────────────────────────────────────────────────────────
class Commander:
    """モーション再生エンジン。コントローラは ProjectCode が任意で供給する。"""

    def __init__(self, loop_hz: int = 100) -> None:
        self._loop_hz     = loop_hz
        self._loop_dt     = 1.0 / loop_hz
        self._frame_budget_ms = self._loop_dt * 1000.0 * 0.85

        self._pad: list | None = None
        self._pad_lock          = threading.Lock()
        self._imu: list         = [0.0] * 90
        self._last_output: list = [0.0] * 90
        self._output_lock       = threading.Lock()
        self._pose              = [0.0] * 90

        self.controller = None
        for _cls_name in ("MotionController", "WalkController", "Controller"):
            _cls = globals().get(_cls_name)
            if isinstance(_cls, type):
                try:
                    self.controller = _cls()
                    break
                except Exception as exc:
                    logger.warning("%s() failed: %s", _cls_name, exc)
        _stop = getattr(self.controller, "stop", None) or getattr(self.controller, "stop_walk", None)
        if callable(_stop):
            _stop()

        self.state                         = STATE_IDLE
        self._prev_state                   = STATE_IDLE
        self._motion: MotionPlayer | None  = None
        self._variables: dict              = {}

    @property
    def walk(self):
        """Alias for .controller (older ProjectCode)."""
        return self.controller

    @walk.setter
    def walk(self, value):
        self.controller = value

    def set_pad(self, pad: list) -> None:
        with self._pad_lock:
            self._pad = list(pad)

    def set_imu(self, r: list) -> None:
        self._imu = list(r)

    def get_last_output(self) -> list:
        with self._output_lock:
            return list(self._last_output)

    def _pose_data(self) -> list:
        c = self.controller
        data = getattr(c, "data", None) if c is not None else None
        if isinstance(data, list) and data:
            return data
        return self._pose

    def _publish(self, data: list) -> None:
        out_mrd = list(data)
        for i in range(21, 81, 2):
            out_mrd[i] = round(float(out_mrd[i]) * 100.0, 2)
        with self._output_lock:
            self._last_output = out_mrd

    def _resolve_motion(self, motion_ref) -> list | None:
        if motion_ref is None:
            return None
        if isinstance(motion_ref, list):
            return motion_ref
        if isinstance(motion_ref, str):
            obj = globals().get(motion_ref)
            return obj if isinstance(obj, list) else None
        return None

    def _trigger_motion(self, segments: list, label: str) -> None:
        if self.state != STATE_MOTION:
            self._prev_state = self.state
        self.state   = STATE_MOTION
        self._motion = MotionPlayer(segments, list(self._pose_data()),
                                    loop_hz=self._loop_hz, variables=self._variables)
        logger.info(f"[CMD] → {label}")

    def poll_and_update(self) -> None:
        """Drive Boot/Base MOTION lists only. Stick/buttons are CMP inside Base."""
        with self._pad_lock:
            pad = list(self._pad) if self._pad is not None else None

        if self.state == STATE_MOTION and self._motion is not None and not self._motion.is_done():
            self._publish(self._motion.advance(pad=pad))
            jt = self._motion.jump_target
            if jt is not None:
                target_segs = self._resolve_motion(jt[0])
                start_line  = int(jt[1]) if jt[1] is not None else 0
                if target_segs is not None:
                    self._motion = MotionPlayer(
                        target_segs, self._motion.data,
                        loop_hz=self._loop_hz, variables=self._variables,
                        start_line=start_line,
                    )
                    logger.info(f"[CMD] → JUMP {jt[0]!r}:{start_line}")
                else:
                    logger.warning(f"[CMD] JUMP: unknown motion {jt[0]!r}")
                    self._motion = None
                    self._return_to_base(pad)
            return

        # Motion finished or never started → Base wait loop (not analog hijack).
        self._return_to_base(pad)

    def _return_to_base(self, pad: list | None) -> None:
        base = globals().get("MOTION_BASE")
        if isinstance(base, list) and base:
            _stop = getattr(self.controller, "stop", None) or getattr(self.controller, "stop_walk", None)
            if callable(_stop):
                _stop()
            self._trigger_motion(base, "Base")
            if self._motion is not None:
                self._publish(self._motion.advance(pad=pad))
            return
        self.state = STATE_IDLE
        self._motion = None
        self._publish(self._pose_data())


# ── BHV system functions ──────────────────────────────────────────────────────
_bhv_commander:        Commander | None = None
_bhv_user_logic_fn                      = None
_bhv_frame_count:      int              = 0
_bhv_frame_budget_ms:  float            = 8.5


def bhv_setup(user_setup_fn, user_logic_fn,
              loop_hz: int = 100) -> None:
    """BHVスタート時に1回呼ぶ。Commander を初期化して user_setup_fn を実行する。"""
    global _bhv_commander, _bhv_user_logic_fn, _bhv_frame_budget_ms
    _bhv_frame_budget_ms = (1.0 / loop_hz) * 1000.0 * 0.85
    _bhv_commander = Commander(loop_hz=loop_hz)
    _bhv_user_logic_fn = user_logic_fn
    user_setup_fn()
    logger.info(f"BHV ready  loop_hz={loop_hz}  budget={_bhv_frame_budget_ms:.1f}ms/frame")


def bhv_update(r: list, s: list) -> list:
    """毎フレーム呼ぶ。user_logic を実行し Late 検出を行う。"""
    global _bhv_frame_count
    if _bhv_commander is None or _bhv_user_logic_fn is None:
        return s
    if len(r) < 19 or len(s) <= 18:
        return s
    _bhv_frame_count += 1
    t0     = time.perf_counter()
    result = _bhv_user_logic_fn(r, s)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    if elapsed_ms > _bhv_frame_budget_ms:
        logger.warning(f"[LATE] frame={_bhv_frame_count}  elapsed={elapsed_ms:.1f}ms"
                       f"  budget={_bhv_frame_budget_ms:.1f}ms")
    return result if result is not None else s


def run_default(r: list, s: list) -> list:
    """Feed pad/IMU into Commander. Boot/Base MOTION lists drive all input.
    user_logic() の中で return run_default(r, s) と書くだけで使える。
    """
    if _bhv_commander is None:
        return s
    _bhv_commander.set_imu(r)
    _bhv_commander.set_pad(_decode_meridim_pad(s))
    _bhv_commander.poll_and_update()
    return _bhv_commander.get_last_output()
'''


@dataclass
class ExportResult:
    save_path: str
    action_count: int
    line_counts: list[int] = field(default_factory=list)
    warnings: list[LinearizeWarning] = field(default_factory=list)
    boot_action_idx: int | None = None
    base_action_idx: int | None = None


_BUTTON_ORDER = ["L1", "R1", "TRI", "CIR", "SQR", "CRS", "UP", "DOWN"]


def export_cartridge(
    motion_action_state: dict,
    save_path: str,
    robot_name: str = "robot",
    source_project: str = "",
    loop_hz: int = 100,
    boot_action_idx: int | None = None,
    base_action_idx: int | None = None,
    project_code: str = "",
    joints: tuple[str, ...] | None = None,
) -> ExportResult:
    """Write a Logic Cartridge to ``save_path``."""
    items = motion_action_state.get("items", []) if motion_action_state else []
    if not items:
        raise ValueError("motion_action_state has no items — nothing to export")

    loop_hz = max(1, min(1000, int(loop_hz)))

    # Clamp indices to valid range.
    n = len(items)
    if boot_action_idx is not None and not (0 <= boot_action_idx < n):
        boot_action_idx = None
    if base_action_idx is not None and not (0 <= base_action_idx < n):
        base_action_idx = None

    # Build symbol map: action_idx → Python variable name.
    motion_symbol_map: dict[int, str] = {}
    if boot_action_idx is not None:
        motion_symbol_map[boot_action_idx] = "MOTION_BOOT"
    if base_action_idx is not None:
        motion_symbol_map[base_action_idx] = "MOTION_BASE"

    # Linearise all actions.
    linearised: list[LinearizeResult] = []
    for i, item in enumerate(items):
        data = item.get("data")
        if not isinstance(data, dict):
            linearised.append(LinearizeResult())
            continue
        linearised.append(linearize_action_data(data, action_index=i))

    all_warnings: list[LinearizeWarning] = []
    joints_used: set[str] = set()
    for lr in linearised:
        all_warnings.extend(lr.warnings)
        joints_used.update(lr.joints_used)

    # Resolve cross-motion JUMP references, tracking referenced extras.
    referenced_extras: set[int] = set()

    def _resolve_motion_refs(lines: list, own_idx: int) -> list:
        out = []
        for tup in lines:
            if isinstance(tup, tuple) and len(tup) == 3 and tup[0] == "JUMP" and is_motion_ref(tup[1]):
                _, ref, target_line = tup
                _, action_idx = ref
                sym = motion_symbol_map.get(action_idx)
                if sym is None:
                    sym = _title_to_symbol(
                        (items[action_idx].get("title") or "") if action_idx < n else "",
                        action_idx,
                    )
                    if action_idx != own_idx:
                        referenced_extras.add(action_idx)
                        motion_symbol_map[action_idx] = sym
                if action_idx == own_idx:
                    # Self-referential JUMP: use 2-tuple (instant intra-motion) to avoid 1-tick overhead.
                    out.append(("JUMP", int(target_line)))
                else:
                    out.append(("JUMP", _RawSymbol(sym), int(target_line)))
            else:
                out.append(tup)
        return out

    resolved_lines: list[list] = [
        _resolve_motion_refs(lr.lines, i) for i, lr in enumerate(linearised)
    ]

    # Computed-motion exit → Base analog-dispatch line (skip button CMPs).
    if base_action_idx is not None:
        _patch_function_jump_base_exits(
            resolved_lines,
            items,
            base_action_idx,
            motion_symbol_map.get(base_action_idx),
        )

    # Register extras that were referenced but not yet in the symbol map.
    for extra_idx in list(referenced_extras):
        if extra_idx not in motion_symbol_map:
            sym = _title_to_symbol(
                (items[extra_idx].get("title") or "") if extra_idx < n else "",
                extra_idx,
            )
            motion_symbol_map[extra_idx] = sym

    # Auto-append JUMP to Base at end of Boot and referenced extras (if Base exists).
    base_symbol = motion_symbol_map.get(base_action_idx) if base_action_idx is not None else None

    def _ensure_jump_end(lines: list, target: str | None) -> list:
        """Append JUMP to target at end of lines if not already ending with JUMP."""
        if not lines:
            return lines
        last = lines[-1]
        if isinstance(last, tuple) and last[0] == "JUMP":
            return lines  # already ends with JUMP — respect user's intent
        if target is None:
            return lines
        return lines + [("JUMP", _RawSymbol(target), 0)]

    def _ensure_self_loop(lines: list) -> list:
        """Append JUMP 0 at end of Base if not already ending with JUMP."""
        if not lines:
            return lines
        last = lines[-1]
        if isinstance(last, tuple) and last[0] == "JUMP":
            return lines
        return lines + [("JUMP", 0)]

    if boot_action_idx is not None:
        resolved_lines[boot_action_idx] = _ensure_jump_end(
            resolved_lines[boot_action_idx], base_symbol
        )

    if base_action_idx is not None:
        resolved_lines[base_action_idx] = _ensure_self_loop(
            resolved_lines[base_action_idx]
        )

    # Auto-append JUMP MOTION_BASE to referenced extras.
    if base_symbol is not None:
        for extra_idx in referenced_extras:
            if extra_idx in (boot_action_idx, base_action_idx):
                continue
            resolved_lines[extra_idx] = _ensure_jump_end(
                resolved_lines[extra_idx], base_symbol
            )

    for i, lines in enumerate(resolved_lines):
        _validate_motion_lines(lines, action_index=i, warnings=all_warnings)

    joints_dict = build_joints_dict(joints_used=None, joints=joints)
    meridim_map = build_meridim_joint_map(joints=joints)
    if not meridim_map:
        raise ValueError("MERIDIM_JOINT_MAP is empty — JOINT_TO_MERIDIM has no canonical joints")

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    # HEADER_TEMPLATE embeds these directly inside the generated file's own
    # triple-quoted docstring, so any backslash they contain is parsed as a
    # Python escape sequence when the cartridge is (re)loaded. Windows paths
    # (e.g. "C:\Users\...") commonly contain "\U..." which Python reads as a
    # (truncated) \UXXXXXXXX unicode escape and fails to parse — macOS/Linux
    # paths use forward slashes and never hit this. Normalise to forward
    # slashes so the generated file stays valid Python on every platform.
    # Also strip absolute path: use relative path from _LME_DIR when possible,
    # otherwise fall back to filename only to avoid embedding user home dirs.
    def _sanitize_source_path(p: str) -> str:
        if not p:
            return "(unsaved)"
        abs_p = os.path.abspath(p)
        try:
            rel = os.path.relpath(abs_p, _LME_DIR)
        except ValueError:
            rel = None
        if rel and not rel.startswith(".."):
            return rel.replace("\\", "/")
        return os.path.basename(abs_p)

    header = HEADER_TEMPLATE.format(
        robot_name=str(robot_name).replace("\\", "/"),
        timestamp=timestamp,
        source_project=_sanitize_source_path(source_project),
        loop_hz=loop_hz,
    )

    item_titles = [it.get("title", "") for it in items]

    parts: list[str] = [header]
    parts.append(_render_joints_dict(joints_dict))
    parts.append(_render_meridim_joint_map(meridim_map))
    # Runtime first, then ProjectCode so user controllers win.
    parts.append(SYSTEM_AREA)
    parts.append(_render_bhv_aliases())
    parts.append(_render_user_work_area(
        resolved_lines,
        motion_symbol_map,
        loop_hz=loop_hz,
        item_titles=item_titles,
        boot_action_idx=boot_action_idx,
        base_action_idx=base_action_idx,
        referenced_extras=referenced_extras,
        n_items=n,
        project_code=project_code,
    ))

    text = "\n".join(parts)

    # Validate before writing so a bad cartridge never lands on disk.
    try:
        tree = ast.parse(text, filename=save_path)
    except SyntaxError as e:
        raise RuntimeError(f"Generated cartridge is not valid Python: {e}") from e

    _verify_meridim_literal(tree, save_path)

    out_dir = os.path.dirname(os.path.abspath(save_path))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(save_path, "w", encoding="utf-8") as f:
        f.write(text)

    return ExportResult(
        save_path=save_path,
        action_count=len(items),
        line_counts=[len(lines) for lines in resolved_lines],
        warnings=all_warnings,
        boot_action_idx=boot_action_idx,
        base_action_idx=base_action_idx,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class _RawSymbol:
    __slots__ = ("name",)

    def __init__(self, name: str) -> None:
        self.name = name


def _title_to_symbol(title: str, fallback_idx: int) -> str:
    """Convert an action title to a valid Python MOTION_* identifier."""
    if not title or not title.strip():
        return f"MOTION_ACTION_{fallback_idx + 1}"
    cleaned = re.sub(r"[^A-Za-z0-9]+", "_", title.strip()).strip("_").upper()
    if not cleaned:
        return f"MOTION_ACTION_{fallback_idx + 1}"
    return f"MOTION_{cleaned}"


def _base_analog_dispatch_line(base_lines: list) -> int:
    """Line index of the first Ly CMP in Base (stick dispatch, after button CMPs)."""
    for i, tup in enumerate(base_lines):
        if (
            isinstance(tup, tuple)
            and len(tup) >= 2
            and tup[0] == "CMP"
            and tup[1] == "LY"
        ):
            return i
    return 0


def _patch_function_jump_base_exits(
    resolved_lines: list[list],
    items: list,
    base_action_idx: int,
    base_symbol: str | None,
) -> None:
    """Route function-Jump action exit JUMP MOTION_BASE,0 → analog dispatch line."""
    if not base_symbol:
        return
    dispatch = _base_analog_dispatch_line(resolved_lines[base_action_idx])
    if dispatch <= 0:
        return
    for i, item in enumerate(items):
        if i == base_action_idx:
            continue
        data = item.get("data")
        if not isinstance(data, dict) or not any(
            isinstance(n, dict)
            and n.get("node_type") == "jump"
            and n.get("jump_type") == "function"
            for n in data.get("nodes", []) or []
        ):
            continue
        lines = resolved_lines[i]
        for j, tup in enumerate(lines):
            if not (isinstance(tup, tuple) and len(tup) == 3 and tup[0] == "JUMP"):
                continue
            ref, start_line = tup[1], int(tup[2])
            if start_line != 0:
                continue
            if isinstance(ref, _RawSymbol) and ref.name == base_symbol:
                lines[j] = ("JUMP", ref, dispatch)


def _validate_motion_lines(
    lines: list,
    action_index: int,
    warnings: list[LinearizeWarning],
) -> None:
    n = len(lines)
    for idx, tup in enumerate(lines):
        if not isinstance(tup, tuple) or not tup:
            continue
        if tup[0] != "CMP":
            continue
        for slot in (4, 5) if len(tup) >= 6 else (3, 4):
            if slot >= len(tup):
                continue
            target = tup[slot]
            if isinstance(target, int) and target > n:
                warnings.append(LinearizeWarning(
                    action_index=action_index,
                    node_id="",
                    node_type="cmp",
                    message=f"CMP line {idx} jump target {target} >= motion length {n}",
                ))


def _verify_meridim_literal(tree: ast.AST, save_path: str) -> None:
    for node in tree.body:
        value_node = None
        if isinstance(node, ast.Assign):
            if any(isinstance(t, ast.Name) and t.id == "MERIDIM_JOINT_MAP" for t in node.targets):
                value_node = node.value
        if value_node is not None:
            value = ast.literal_eval(value_node)
            if not value:
                raise RuntimeError(f"{save_path}: MERIDIM_JOINT_MAP is empty after export")
            return
    raise RuntimeError(f"{save_path}: MERIDIM_JOINT_MAP declaration missing")


def _format_value(v) -> str:
    if isinstance(v, _RawSymbol):
        return v.name
    if isinstance(v, bool):
        return "True" if v else "False"
    if isinstance(v, (int, float, str)) or v is None:
        return repr(v)
    if isinstance(v, dict):
        inner = ", ".join(f"{_format_value(k)}: {_format_value(val)}" for k, val in v.items())
        return "{" + inner + "}"
    if isinstance(v, tuple):
        inner = ", ".join(_format_value(x) for x in v)
        return "(" + inner + ("," if len(v) == 1 else "") + ")"
    if isinstance(v, list):
        inner = ", ".join(_format_value(x) for x in v)
        return "[" + inner + "]"
    return repr(v)


def _render_joints_dict(joints_dict: dict) -> str:
    ordered = sorted(joints_dict.items(), key=lambda kv: kv[1])
    lines = [
        "# ── Joint name dictionary (auto-generated) ──────────────────────",
        "JOINTS: dict[str, int] = {",
    ]
    for name, idx in ordered:
        lines.append(f"    {name!r:24s}: {idx},")
    lines.append("}")
    lines.append("")
    return "\n".join(lines)


def _render_meridim_joint_map(entries: list[dict]) -> str:
    lines = [
        "# ── Meridim joint map (read by PhysicalOn) ────────────────────",
        "MERIDIM_JOINT_MAP = [",
    ]
    for e in entries:
        joint = e["joint"]
        m = int(e["meridim"])
        sign = float(e["sign"])
        role = str(e["role"])
        lines.append(
            f'    {{"joint": {joint!r}, "meridim": {m:3d}, "sign": {sign:+.1f}, "role": {role!r}}},'
        )
    lines.append("]")
    lines.append("")
    return "\n".join(lines)


def _user_code_defines(code: str, name: str) -> bool:
    """Return True if ``code`` defines a top-level function named ``name``."""
    if not code or not code.strip():
        return False
    try:
        tree = ast.parse(code)
        return any(
            isinstance(n, ast.FunctionDef) and n.name == name and n.col_offset == 0
            for n in ast.walk(tree)
        )
    except SyntaxError:
        return False


def _render_user_work_area(
    resolved_lines: list[list],
    motion_symbol_map: dict[int, str],
    loop_hz: int,
    item_titles: list[str],
    boot_action_idx: int | None,
    base_action_idx: int | None,
    referenced_extras: set[int],
    n_items: int,
    project_code: str = "",
) -> str:
    out: list[str] = []
    out.append("")
    out.append("# " + "═" * 76)
    out.append("# ★★★  USER WORK AREA  ★★★  (regenerated on every export)")
    out.append("# " + "═" * 76)
    out.append("")
    out.append(f"LOOP_HZ = {loop_hz}")
    out.append("")

    def _title(idx: int) -> str:
        return (item_titles[idx] if idx < len(item_titles) else "").strip()

    # Collect ordered list of (idx, symbol): Boot → Base → referenced extras only.
    ordered: list[tuple[int, str]] = []
    _seen_syms: set[str] = set()

    def _add_ordered(idx: int, sym: str) -> None:
        if sym not in _seen_syms:
            _seen_syms.add(sym)
            ordered.append((idx, sym))

    if boot_action_idx is not None:
        _add_ordered(boot_action_idx, "MOTION_BOOT")
    if base_action_idx is not None:
        _add_ordered(base_action_idx, "MOTION_BASE")
    for i in sorted(referenced_extras):
        if i in (boot_action_idx, base_action_idx):
            continue
        sym = motion_symbol_map.get(i, f"MOTION_ACTION_{i + 1}")
        _add_ordered(i, sym)

    # Forward-declare all MOTION_* variables first so cross-motion JUMPs resolve.
    if ordered:
        out.append("# Forward declarations for cross-motion JUMP references.")
        for _, sym in ordered:
            out.append(f"{sym}: list = []")
        out.append("")

    # Now emit .extend() blocks for each motion.
    for idx, sym in ordered:
        title_note = f"  # {_title(idx)}" if _title(idx) else ""
        lines = resolved_lines[idx] if idx < len(resolved_lines) else []
        out.append(f"{sym}.extend([{title_note}")
        if not lines:
            out.append("    # (empty — no start edge or no reachable nodes)")
        else:
            for line_idx, tup in enumerate(lines):
                out.append(f"    {_format_value(tup)},  # line {line_idx}")
        out.append("])")
        out.append("")

    # Inject user Code block (from LME Code editor) before auto-generated functions.
    code = (project_code or "").strip()
    if code:
        out.append("# " + "─" * 76)
        out.append("# ── User Code (from LME Code editor) ────────────────────────────────────────")
        out.append("# " + "─" * 76)
        out.append("")
        out.append(code)
        out.append("")

    # Always start Boot (or Base), even when ProjectCode defines user_setup.
    out.append("def _lme_ensure_boot() -> None:")
    out.append("    if _bhv_commander is None:")
    out.append("        return")
    out.append("    if getattr(_bhv_commander, '_motion', None) is not None:")
    out.append("        return")
    if boot_action_idx is not None:
        out.append("    _ctrl = getattr(_bhv_commander, 'controller', None) or getattr(_bhv_commander, 'walk', None)")
        out.append("    if hasattr(_ctrl, 'reset_pose'):")
        out.append("        _ctrl.reset_pose()")
        out.append("    _bhv_commander._trigger_motion(MOTION_BOOT, 'Boot')")
    elif base_action_idx is not None:
        out.append("    _bhv_commander._trigger_motion(MOTION_BASE, 'Base')")
    out.append("")
    if _user_code_defines(code, "user_setup"):
        out.append("_lme_user_setup = user_setup")
        out.append("def user_setup() -> None:")
        out.append("    _lme_user_setup()")
        out.append("    _lme_ensure_boot()")
        out.append("")
    else:
        out.append("def user_setup() -> None:")
        out.append("    _lme_ensure_boot()")
        out.append('    logger.info("Logic cartridge: ready")')
        out.append("")
        out.append("")

    # user_logic — skip if user defined it in Code
    if not _user_code_defines(code, "user_logic"):
        out.append("def user_logic(r: list, s: list) -> list:")
        out.append("    return run_default(r, s)")
        out.append("")

    return "\n".join(out)


def _render_bhv_aliases() -> str:
    return (
        "\n"
        "# " + "═" * 76 + "\n"
        "# BHV interface (do not modify)\n"
        "# " + "═" * 76 + "\n"
        "def setup():         bhv_setup(user_setup, user_logic, loop_hz=LOOP_HZ)\n"
        "def update(r, s):    return bhv_update(r, s)\n"
    )
