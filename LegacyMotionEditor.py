"""
File Name: LegacyMotionEditor.py
Description: Node-graph motion editor for URDF/MJCF robots (LegacyMotionEditor).

Author      : Izumi Ninagawa
Created On  : Aug 22, 2026
Version     : 0.0.3
License     : MIT License
URL         : https://github.com/Ninagawa123/LegacyMotionEditor_alpha
Copyright (c) 2026 Izumi Ninagawa

pip install numpy
pip install PySide6
pip install vtk
pip install NodeGraphQt
"""

import sys
import os
import traceback
import ast

# Qt.py経由で統一（PySide6/PyQt5を自動選択）
# 環境変数 QT_PREFERRED_BINDING=PySide6 を推奨
from Qt import QtWidgets, QtCore, QtGui
# NodeGraphQt は NODE_GRAPH_FOCUS_COLOR で NodeEnum をパッチした後に import する（選択枠色を反映するため）
import vtk
from vtk.util.numpy_support import vtk_to_numpy

# Qt.pyからの再エクスポート（PySide6直接importを廃止）
QFileDialog = QtWidgets.QFileDialog
QPointF = QtCore.QPointF
QDoubleValidator = QtGui.QDoubleValidator
QIntValidator = QtGui.QIntValidator
QPalette = QtGui.QPalette
QColor = QtGui.QColor
import xml.etree.ElementTree as ET
import datetime
import json
import copy
import math
import unicodedata
import time
import numpy as np
import threading

try:
    import valkey as _valkey_lib
    _VALKEY_OK = True
except ImportError:
    _valkey_lib = None
    _VALKEY_OK = False

_LME_VERSION = "0.0.2"

_LEGACY_EDITOR_DIR = os.path.dirname(os.path.abspath(__file__))
if _LEGACY_EDITOR_DIR not in sys.path:
    sys.path.insert(0, _LEGACY_EDITOR_DIR)

# Utils module: constants, helper functions, easing, math functions
from LegacyMotionEditor_Utils import (
    # App settings
    _app_settings, get_default_hz_fps, get_node_offset_x, get_node_offset_y,
    # Debug logger
    DebugLogger, dbg, DEBUG_LOG_FILE, APP_SETTINGS_FILE,
    install_lme_quiet_console,
    load_app_settings, save_app_settings, SESSION_FILE_PATH,
    ensure_session_save_dir, resolve_session_file_for_load,
    path_for_project_save, resolve_project_path,
    # Valkey / Meridim constants
    MERIDIM_SIZE, JOINT_TO_MERIDIM,
    MASTER_CMD_RESET,
    LME_PACKET_MARKER_SLOT, LME_PACKET_MARKER_VALUE,
    VALKEY_DEFAULT_HOST, VALKEY_DEFAULT_PORT,
    VALKEY_DEFAULT_WRITE_KEY, VALKEY_DEFAULT_READ_KEY,
    ensure_valkey_container_running,
    ensure_valkey_running,
    # Color constants
    MINT_GREEN_COLOR, BRANCH_POINT_COLOR, BRANCH_LINE_COLOR,
    PALETTE_WINDOW, PALETTE_WINDOW_TEXT, PALETTE_BASE, PALETTE_ALTERNATE_BASE,
    PALETTE_TOOLTIP_BASE, PALETTE_TOOLTIP_TEXT, PALETTE_TEXT, PALETTE_BUTTON,
    PALETTE_BUTTON_TEXT, PALETTE_BRIGHT_TEXT, PALETTE_HIGHLIGHT, PALETTE_HIGHLIGHTED_TEXT,
    NODE_GRAPH_BG_COLOR, NODE_GRAPH_GRID_COLOR, NODE_GRAPH_GRID_SNAP_SIZE,
    NODE_GRAPH_FOCUS_COLOR, NODE_GRAPH_FOCUS_TEXT_COLOR,
    NODE_COLOR_DEFAULT,
    NODE_START_TITLE_COLOR, NODE_START_TITLE_BG_COLOR, NODE_START_PANEL_BG_COLOR,
    NODE_START_TEXT_COLOR, NODE_START_INPUT_PORT_COLOR, NODE_START_INPUT_PORT_BORDER_COLOR,
    NODE_START_OUTPUT_PORT_COLOR, NODE_START_OUTPUT_PORT_BORDER_COLOR,
    NODE_START_TITLE_HIGHLIGHT_COLOR, NODE_START_TITLE_BG_HIGHLIGHT_COLOR,
    NODE_START_PANEL_BG_HIGHLIGHT_COLOR, NODE_START_INPUT_PORT_HIGHLIGHT_COLOR,
    NODE_START_INPUT_PORT_HIGHLIGHT_BORDER_COLOR, NODE_START_OUTPUT_PORT_HIGHLIGHT_COLOR,
    NODE_START_OUTPUT_PORT_HIGHLIGHT_BORDER_COLOR,
    NODE_BASIC_TITLE_COLOR, NODE_BASIC_TITLE_BG_COLOR, NODE_BASIC_PANEL_BG_COLOR,
    NODE_BASIC_TEXT_COLOR, NODE_BASIC_INPUT_PORT_COLOR, NODE_BASIC_INPUT_PORT_BORDER_COLOR,
    NODE_BASIC_OUTPUT_PORT_COLOR, NODE_BASIC_OUTPUT_PORT_BORDER_COLOR,
    NODE_BASIC_TITLE_HIGHLIGHT_COLOR, NODE_BASIC_TITLE_BG_HIGHLIGHT_COLOR,
    NODE_BASIC_PANEL_BG_HIGHLIGHT_COLOR, NODE_BASIC_INPUT_PORT_HIGHLIGHT_COLOR,
    NODE_BASIC_INPUT_PORT_HIGHLIGHT_BORDER_COLOR, NODE_BASIC_OUTPUT_PORT_HIGHLIGHT_COLOR,
    NODE_BASIC_OUTPUT_PORT_HIGHLIGHT_BORDER_COLOR,
    NODE_POSE_TITLE_COLOR, NODE_POSE_TITLE_BG_COLOR, NODE_POSE_PANEL_BG_COLOR,
    NODE_POSE_TEXT_COLOR, NODE_POSE_INPUT_PORT_COLOR, NODE_POSE_INPUT_PORT_BORDER_COLOR,
    NODE_POSE_OUTPUT_PORT_COLOR, NODE_POSE_OUTPUT_PORT_BORDER_COLOR,
    NODE_POSE_BRANCH_TO_PORT_COLOR, NODE_POSE_BRANCH_TO_PORT_BORDER_COLOR,
    NODE_POSE_BRANCH_OTHERWISE_PORT_COLOR, NODE_POSE_BRANCH_OTHERWISE_PORT_BORDER_COLOR,
    NODE_POSE_TITLE_HIGHLIGHT_COLOR, NODE_POSE_TITLE_BG_HIGHLIGHT_COLOR,
    NODE_POSE_PANEL_BG_HIGHLIGHT_COLOR, NODE_POSE_INPUT_PORT_HIGHLIGHT_COLOR,
    NODE_POSE_INPUT_PORT_HIGHLIGHT_BORDER_COLOR, NODE_POSE_OUTPUT_PORT_HIGHLIGHT_COLOR,
    NODE_POSE_OUTPUT_PORT_HIGHLIGHT_BORDER_COLOR,
    NODE_DEFINE_TITLE_COLOR, NODE_DEFINE_TITLE_BG_COLOR, NODE_DEFINE_PANEL_BG_COLOR,
    NODE_DEFINE_TITLE_HIGHLIGHT_COLOR, NODE_DEFINE_TITLE_BG_HIGHLIGHT_COLOR,
    NODE_DEFINE_PANEL_BG_HIGHLIGHT_COLOR,
    NODE_WAIT_TITLE_COLOR, NODE_WAIT_TITLE_BG_COLOR, NODE_WAIT_PANEL_BG_COLOR,
    NODE_WAIT_TITLE_HIGHLIGHT_COLOR, NODE_WAIT_TITLE_BG_HIGHLIGHT_COLOR,
    NODE_WAIT_PANEL_BG_HIGHLIGHT_COLOR,
    NODE_BRANCH_TITLE_COLOR, NODE_BRANCH_TITLE_BG_COLOR, NODE_BRANCH_PANEL_BG_COLOR,
    NODE_BRANCH_TITLE_HIGHLIGHT_COLOR, NODE_BRANCH_TITLE_BG_HIGHLIGHT_COLOR,
    NODE_BRANCH_PANEL_BG_HIGHLIGHT_COLOR,
    NODE_MIX_TITLE_COLOR, NODE_MIX_TITLE_BG_COLOR, NODE_MIX_PANEL_BG_COLOR,
    NODE_MIX_INPUT_PORT_COLOR, NODE_MIX_INPUT_PORT_BORDER_COLOR,
    NODE_MIX_OUTPUT_PORT_COLOR, NODE_MIX_OUTPUT_PORT_BORDER_COLOR,
    NODE_MIX_TITLE_HIGHLIGHT_COLOR, NODE_MIX_TITLE_BG_HIGHLIGHT_COLOR,
    NODE_MIX_PANEL_BG_HIGHLIGHT_COLOR, NODE_MIX_INPUT_PORT_HIGHLIGHT_COLOR,
    NODE_MIX_INPUT_PORT_HIGHLIGHT_BORDER_COLOR, NODE_MIX_OUTPUT_PORT_HIGHLIGHT_COLOR,
    NODE_MIX_OUTPUT_PORT_HIGHLIGHT_BORDER_COLOR,
    NODE_COMMAND_TITLE_COLOR, NODE_COMMAND_TITLE_BG_COLOR, NODE_COMMAND_PANEL_BG_COLOR,
    NODE_COMMAND_INPUT_PORT_COLOR, NODE_COMMAND_INPUT_PORT_BORDER_COLOR,
    NODE_COMMAND_OUTPUT_PORT_COLOR, NODE_COMMAND_OUTPUT_PORT_BORDER_COLOR,
    NODE_COMMAND_TITLE_HIGHLIGHT_COLOR, NODE_COMMAND_TITLE_BG_HIGHLIGHT_COLOR,
    NODE_COMMAND_PANEL_BG_HIGHLIGHT_COLOR, NODE_COMMAND_INPUT_PORT_HIGHLIGHT_COLOR,
    NODE_COMMAND_INPUT_PORT_HIGHLIGHT_BORDER_COLOR, NODE_COMMAND_OUTPUT_PORT_HIGHLIGHT_COLOR,
    NODE_COMMAND_OUTPUT_PORT_HIGHLIGHT_BORDER_COLOR,
    SERVO_COMMAND_TYPES,
    NODE_JUMP_TITLE_COLOR, NODE_JUMP_TITLE_BG_COLOR, NODE_JUMP_PANEL_BG_COLOR,
    NODE_JUMP_TITLE_HIGHLIGHT_COLOR, NODE_JUMP_TITLE_BG_HIGHLIGHT_COLOR,
    NODE_JUMP_PANEL_BG_HIGHLIGHT_COLOR,
    COLOR_SAMPLE_DEFAULT, LABEL_TEXT_COLOR,
    MESH_HIGHLIGHT_COLOR, MESH_HIGHLIGHT_BLINK_INTERVAL,
    PLAYBACK_HIGHLIGHT_COLOR, PLAYBACK_HIGHLIGHT_BORDER_WIDTH,
    PLAYBACK_INCOMPLETE_COLOR, PLAYBACK_INCOMPLETE_BORDER_WIDTH,
    MESH_DRAG_SENSITIVITY, MESH_WHEEL_SENSITIVITY,
    # Layout constants
    LEFT_PANEL_WIDTH, NODE_INSPECTOR_MIN_WIDTH, JOINT_EDITOR_WIDTH,
    VTK_DISPLAY_MIN_WIDTH, VTK_DISPLAY_MIN_HEIGHT,
    SPLITTER_NODE_GRAPH_WIDTH, SPLITTER_3DVIEW_WIDTH, SPLITTER_JOINT_EDITOR_WIDTH,
    RIGHT_PANEL_MIN_WIDTH,
    VTK_BACKGROUND_COLOR, VTK_BG_SLIDER_DEFAULT, VTK_BG_COLOR_A, VTK_BG_COLOR_B,
    VTK_BG_GRADIENT_TYPE,
    # PAD constants
    PAD_BUTTON_NAMES, PAD_AXIS_NAMES, PAD_REGISTER_VALUES,
    PAD_REGISTER_ALIASES, PAD_BUTTON_BIT_VALUES,
    PAD_IF_BUTTON_CHOICES, PAD_IF_BUTTON_TO_PAD_KEY,
    PAD_IF_ANALOG_AXIS_TO_PAD_KEY, PAD_IF_ANALOG_AXIS_RANGE,
    # User value session
    USER_VALUE_SESSION_COUNT, default_user_value_session, normalize_user_value_session,
    # Branch register functions
    _BRANCH_IF_OPERATOR_CHOICES, normalize_branch_if_op_stored,
    load_branch_register_items_for_side,
    # Math functions
    _rpy_to_rotation_matrix, _axis_angle_to_rotation_matrix, _make_4x4,
    rpy_to_matrix, make_transform_matrix, make_scale_matrix, quat_to_rpy_xyzw,
    # Easing functions
    EASING_PRESETS, EASING_OPTIONS, EASING_NAME_TO_INDEX,
    easing_index, easing_option, easing_value,
    # Helper functions
    create_label, apply_dark_theme,
    # Virtual node classes (for cross-action playback)
    VirtualPort, VirtualBaseLinkNode, VirtualPoseNode, VirtualDefineNode, VirtualWaitNode,
    VirtualBranchingNode, VirtualMixNode, VirtualCommandNode, VirtualJumpNode, build_virtual_graph_from_action_data,
    # Generic UI components
    OffscreenRenderer, ArithmeticDoubleSpinBox, ExportMotionDialog, SingleJointDialog,
    # ColorPicker classes
    CustomColorDialog, ColorPicker,
    # Joint helper constants and functions
    JOINT_SPEED_PRESETS, DEFAULT_JOINT_SPEED, DEFAULT_JOINT_REV,
    get_joint_speed_presets, save_joint_speed_presets, _joint_preset_item_data_parts,
    # Frame preset functions
    get_frame_presets, save_frame_presets,
    # Pad monitor dialog
    PadMonitorDialog,
    # Value list dialog
    ValueListDialog,
    # Motion forms dialog
    MotionFormsDialog,
    # Dialog style constants
    _VIEW_MODAL_PANEL_BG, _MAIN_WINDOW_COMBO_TEXT_STYLE,
    # Dialog classes
    JumpEditDialog, AddDefineShellDialog, WaitEditDialog, BranchingDialog, BranchingShellDialog,
    JointSettingsDialog, JointGroupDialog, SettingsDialog,
    MixEditorPanel, MIX_INPUT_SOURCES,
    CommandEditorPanel,
    LMEUndoStack,
    # Cross-platform helpers
    primary_mod_held, install_signal_handlers, launch_external_editor,
)

_verbose_console = os.environ.get("LME_VERBOSE", "").lower() in ("1", "true", "yes")
install_lme_quiet_console(
    quiet=not _verbose_console,
    verbose=_verbose_console or bool(load_app_settings().get("lme_console_verbose", False)),
)

try:
    from LegacyMotionEditor_CodeEditor import CodeEditorWindow, get_function_names, CODE_DEFAULT_TEMPLATE
    _CODE_EDITOR_OK = True
except ImportError:
    _CODE_EDITOR_OK = False
    CodeEditorWindow = None
    def get_function_names(code): return []
    CODE_DEFAULT_TEMPLATE = ""

try:
    from LegacyMotionEditor_Importer import (
        select_and_parse_mjcf, select_and_parse_urdf, select_and_parse_model,
        parse_model_file,
        URDFJoint, URDFLink, URDFRobotModel,
        build_robot_model_from_mjcf, build_robot_model_from_urdf,
        classify_upper_lower_body_joints,
    )
except Exception as e:
    select_and_parse_mjcf = None
    select_and_parse_urdf = None
    select_and_parse_model = None
    parse_model_file = None
    URDFJoint = URDFLink = URDFRobotModel = None
    build_robot_model_from_mjcf = build_robot_model_from_urdf = None
    classify_upper_lower_body_joints = None
    print(f"[URDF/MJCF] LegacyMotionEditor_Importer unavailable: {e}")

# Mutable container for session save callback; mutated by main() setup code.
_session_cb = {"save": None}
_lme_quit_closers = []
_mujoco_studio_procs = []


def _mujoco_studio_is_running():
    """True if a MuJoCoStudio viewer process is alive (not the Valkey motor child)."""
    import subprocess
    alive = []
    for proc in list(_mujoco_studio_procs):
        try:
            if proc.poll() is None:
                alive.append(proc)
        except Exception:
            pass
    _mujoco_studio_procs[:] = alive
    if alive:
        return True
    if sys.platform == "win32":
        return False
    try:
        pids = subprocess.check_output(
            ["pgrep", "-f", "LegacyMotionEditor_MuJoCoStudio.py"],
            stderr=subprocess.DEVNULL, text=True,
        ).split()
        for pid in pids:
            try:
                args = subprocess.check_output(
                    ["ps", "-p", pid, "-o", "args="],
                    stderr=subprocess.DEVNULL, text=True,
                )
            except Exception:
                continue
            if "--shm-ctrl" in args:
                continue
            if "LegacyMotionEditor_MuJoCoStudio.py" in args:
                return True
    except Exception:
        pass
    return False


def _close_mujoco_studio_processes():
    """Terminate MuJoCoStudio subprocesses launched from this LME instance."""
    import subprocess
    import time
    for proc in list(_mujoco_studio_procs):
        try:
            if proc.poll() is None:
                proc.terminate()
        except Exception:
            pass
    deadline = time.time() + 0.6
    for proc in list(_mujoco_studio_procs):
        try:
            if proc.poll() is None:
                proc.wait(timeout=max(0.05, deadline - time.time()))
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass
    _mujoco_studio_procs.clear()
    try:
        if sys.platform == "win32":
            subprocess.run(
                ["taskkill", "/F", "/FI",
                 "WINDOWTITLE eq LegacyMotionEditor MuJoCoStudio*"],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=3,
            )
        else:
            subprocess.run(
                ["pkill", "-f", "LegacyMotionEditor_MuJoCoStudio.py"],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=3,
            )
    except Exception:
        pass


def _run_companion_shutdown():
    """Close Pad / editors / Studio when the main window quits."""
    for fn in list(_lme_quit_closers):
        try:
            fn()
        except Exception as e:
            print(f"[Quit] companion closer failed: {e}")
    _close_mujoco_studio_processes()


def _bring_mujoco_studio_to_front():
    """既に起動している MuJoCoStudio ウィンドウを最前面に持ってくる。

    - macOS: osascript で unix id (PID) 指定で activate
    - Windows: EnumWindows でタイトル一致の HWND を探して SetForegroundWindow
    - Linux: wmctrl → xdotool の順で試行
    ベストエフォート。成功したら True を返す。
    """
    import subprocess
    pids = [p.pid for p in _mujoco_studio_procs if p.poll() is None]
    title_needle = "LegacyMotionEditor MuJoCoStudio"
    try:
        if sys.platform == "darwin":
            if not pids:
                return False
            for pid in pids:
                try:
                    subprocess.run(
                        ["osascript", "-e",
                         f'tell application "System Events" to set frontmost '
                         f'of (first process whose unix id is {pid}) to true'],
                        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                        timeout=2,
                    )
                except Exception:
                    pass
            return True
        elif sys.platform.startswith("win"):
            import ctypes
            from ctypes import wintypes
            user32 = ctypes.windll.user32
            found = []

            @ctypes.WINFUNCTYPE(ctypes.c_bool, wintypes.HWND, wintypes.LPARAM)
            def _enum_cb(hwnd, _lparam):
                length = user32.GetWindowTextLengthW(hwnd)
                if length > 0:
                    buff = ctypes.create_unicode_buffer(length + 1)
                    user32.GetWindowTextW(hwnd, buff, length + 1)
                    if title_needle in buff.value:
                        found.append(hwnd)
                return True

            user32.EnumWindows(_enum_cb, 0)
            for hwnd in found:
                try:
                    user32.ShowWindow(hwnd, 9)  # SW_RESTORE
                    user32.SetForegroundWindow(hwnd)
                except Exception:
                    pass
            return bool(found)
        else:
            for cmd in (["wmctrl", "-a", title_needle],
                        ["xdotool", "search", "--name", title_needle,
                         "windowactivate"]):
                try:
                    result = subprocess.run(
                        cmd, stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL, timeout=2,
                    )
                    if result.returncode == 0:
                        return True
                except FileNotFoundError:
                    continue
                except Exception:
                    continue
            return False
    except Exception as _e:
        print(f"[MuJoCo] bring to front failed: {_e}")
        return False


# 親クラス(NodeGraphQt)の選択枠色を NODE_GRAPH_FOCUS_COLOR に合わせる
# NodeEnum.SELECTED_BORDER_COLOR を差し替え（node_base の import 前に実行すること）
import NodeGraphQt.constants as _ngq_const
from enum import Enum as _Enum
_NodeEnumItems = [
    (e.name, (*NODE_GRAPH_FOCUS_COLOR, 255) if e.name == 'SELECTED_BORDER_COLOR' else e.value)
    for e in _ngq_const.NodeEnum
]
_ngq_const.NodeEnum = _Enum('NodeEnum', _NodeEnumItems)

# パッチ適用後に NodeGraphQt を import（ここより前に import すると黄色のままになる）
from NodeGraphQt import NodeGraph, BaseNode
from NodeGraphQt.qgraphics.pipe import PipeItem
from NodeGraphQt.widgets.viewer import NodeViewer

# ============================================================================
# テスト用設定（本番では無効にすること）
# ============================================================================
DEBUG_AUTO_LOAD_URDF = False  # True: 起動時にURDFを自動読み込み, False: 無効
DEBUG_URDF_PATH = os.path.join(os.path.dirname(__file__), "roid1_description", "urdf", "roid1.urdf")


from NodeGraphQt.qgraphics.node_base import NodeItem
import NodeGraphQt.qgraphics.node_base as _node_base_mod
# node_base は import 時にすでに古い NodeEnum を参照しているので、モジュール内を上書きする
_node_base_mod.NodeEnum = _ngq_const.NodeEnum

from NodeGraphQt.constants import PortEnum

class CustomNodeItem(NodeItem):
    """カスタムノードアイテム - ポートを下部中央に配置"""

    def post_init(self, viewer=None, pos=None):
        """初期化後すぐにカスタムレイアウトを適用"""
        super(CustomNodeItem, self).post_init(viewer, pos)
        text_item = getattr(self, '_text_item', None)
        if text_item is not None:
            # 通常時は非表示にし、paint()で自前の色で描画。編集時（_title_editing）だけ表示
            self._title_editing = False
            text_item.setVisible(False)
        # NodeGraphQt 標準の左上アイコン (node_base.png) を一括非表示にする
        icon_item = getattr(self, '_icon_item', None)
        if icon_item is not None:
            icon_item.setVisible(False)
        self._align_ports(0.0)
        self._post_init_done = True

    def _align_ports(self, v_offset):
        """
        ポート配置の基本メソッドをオーバーライド
        常にカスタムレイアウトを適用
        """
        from NodeGraphQt.constants import LayoutDirectionEnum
        # レイアウト方向を確認（HORIZONTAL=0, VERTICAL=1）
        if self.layout_direction == LayoutDirectionEnum.HORIZONTAL.value:
            self._align_ports_horizontal(v_offset)
        else:
            self._align_ports_vertical(v_offset)

    def _set_base_size(self, add_w=0.0, add_h=0.0):
        """Branching用に横並びポートを使うノードは高さを固定する"""
        super(CustomNodeItem, self)._set_base_size(add_w, add_h)
        fixed_height = getattr(self, '_fixed_output_row_height', None)
        if fixed_height is not None:
            self._height = fixed_height

    def _align_ports_horizontal(self, v_offset):
        """
        水平レイアウトのポート配置をオーバーライド
        出力ポートを下部中央に配置
        """
        from NodeGraphQt.constants import PortEnum

        width = self._width
        txt_offset = PortEnum.CLICK_FALLOFF.value - 2
        spacing = 1

        # 入力ポート位置の調整（非表示ポートも含む - ライン接続のため）
        all_inputs = list(self.inputs)
        if all_inputs:
            if (
                getattr(self, "_hidden_input_at_panel_center", False)
                and len(all_inputs) == 1
            ):
                port = all_inputs[0]
                port_width = port.boundingRect().width()
                port_height = port.boundingRect().height()
                port.setPos(
                    width / 2.0 - port_width / 2.0,
                    self._height / 2.0 - port_height / 2.0,
                )
                port.update()
            else:
                port_width = all_inputs[0].boundingRect().width()
                port_height = all_inputs[0].boundingRect().height()
                port_x = (port_width / 2) * -1
                port_y = v_offset
                for port in all_inputs:
                    port.setPos(port_x, port_y)
                    port_y += port_height + spacing

        # 入力テキスト位置の調整
        for port, text in self._input_items.items():
            if port.isVisible():
                txt_x = port.boundingRect().width() / 2 - txt_offset
                text.setPos(txt_x, port.y() - 1.5)

        # 出力ポート位置の調整（カスタム：下部中央に横並び配置）
        # NOTE: 非表示ポートも位置を設定する（後でvisibleにした時に正しい位置になるように）
        all_outputs = list(self.outputs)
        if all_outputs:
            # MemoNode 用の特殊配置: タイトル帯の左端に出力ポートを 1 つ置く
            # (フラグ名は互換のため _memo_port_at_title_right のままだが、
            #  位置は左端)
            if getattr(self, "_memo_port_at_title_right", False):
                port = all_outputs[0]
                port_w = port.boundingRect().width()
                port_h = port.boundingRect().height()
                # タイトル帯の中心 Y を text_item から算出（無ければ大まかな 12px）
                text_item = getattr(self, "_text_item", None)
                if text_item is not None:
                    title_center_y = text_item.pos().y() + text_item.boundingRect().height() / 2.0
                else:
                    title_center_y = 12.0
                port.setPos(4, title_center_y - port_h / 2.0)
                port.update()
            else:
                port_width = all_outputs[0].boundingRect().width()
                port_height = all_outputs[0].boundingRect().height()
                spacing = 8

                total_width = (port_width * len(all_outputs)) + (spacing * (len(all_outputs) - 1))
                port_x = (width - total_width) / 2
                port_y = self._height - port_height - 5  # 下部のY座標

                for port in all_outputs:
                    port.setPos(port_x, port_y)
                    port.update()
                    port_x += port_width + spacing

        # 出力テキスト位置の調整（テキストを非表示に）
        for port, text in self._output_items.items():
            if port.isVisible():
                text.setVisible(False)  # テキストラベルを非表示

    def _draw_node_horizontal(self):
        """ノード描画時にもカスタムレイアウトを適用"""
        super(CustomNodeItem, self)._draw_node_horizontal()
        self._align_ports(0.0)

    def set_proxy_mode(self, mode):
        """親は paint 内で auto_switch_mode → set_proxy_mode により _text_item.setVisible(True) するため、
        通常時は _text_item を非表示のままにする（編集時のみ表示）。"""
        super(CustomNodeItem, self).set_proxy_mode(mode)
        text_item = getattr(self, '_text_item', None)
        if text_item is not None and not getattr(self, '_proxy_mode', False):
            text_item.setVisible(getattr(self, '_title_editing', False))

    def paint(self, painter, option, widget=None):
        """親の描画の後、タイトル背景とテキストを自前で描画"""
        super(CustomNodeItem, self).paint(painter, option, widget)

        # Draw playback highlight border (lime green) if this node is currently playing
        if getattr(self, '_is_playing', False):
            node_rect = self.boundingRect()
            pen = QtGui.QPen(QtGui.QColor(*PLAYBACK_HIGHLIGHT_COLOR), PLAYBACK_HIGHLIGHT_BORDER_WIDTH)
            pen.setJoinStyle(QtCore.Qt.RoundJoin)
            painter.setPen(pen)
            painter.setBrush(QtCore.Qt.NoBrush)
            inset = PLAYBACK_HIGHLIGHT_BORDER_WIDTH / 2.0
            highlight_rect = node_rect.adjusted(inset, inset, -inset, -inset)
            painter.drawRoundedRect(highlight_rect, 5.0, 5.0)

        # Draw incomplete highlight border (yellow-orange) as outer border if this node didn't reach target
        if getattr(self, '_is_incomplete', False):
            node_rect = self.boundingRect()
            pen = QtGui.QPen(QtGui.QColor(*PLAYBACK_INCOMPLETE_COLOR), PLAYBACK_INCOMPLETE_BORDER_WIDTH)
            pen.setJoinStyle(QtCore.Qt.RoundJoin)
            painter.setPen(pen)
            painter.setBrush(QtCore.Qt.NoBrush)
            # Draw outside the node (negative inset to expand outward)
            outset = PLAYBACK_INCOMPLETE_BORDER_WIDTH / 2.0
            incomplete_rect = node_rect.adjusted(-outset, -outset, outset, outset)
            painter.drawRoundedRect(incomplete_rect, 6.0, 6.0)
        text_item = getattr(self, '_text_item', None)
        if text_item is None or text_item.isVisible():
            self._restore_node_cache_mode()
            return
        selected = bool(option and (option.state & QtWidgets.QStyle.State_Selected))

        # タイトル背景を描画
        margin = 1.0
        padding = (3.0, 2.0)
        node_rect = self.boundingRect()
        text_rect = text_item.boundingRect()
        bg_rect = QtCore.QRectF(
            text_rect.x() + padding[0],
            node_rect.y() + margin + padding[1],
            node_rect.width() - padding[0] - margin * 2,
            text_rect.height() - (padding[1] * 2)
        )
        if selected:
            title_bg_rgb = getattr(self, '_highlight_title_bg', (100, 150, 255))
        else:
            title_bg_rgb = getattr(self, '_normal_title_bg', (0, 0, 0, 80))
        if len(title_bg_rgb) == 4:
            title_bg_color = QtGui.QColor(title_bg_rgb[0], title_bg_rgb[1], title_bg_rgb[2], title_bg_rgb[3])
        else:
            title_bg_color = QtGui.QColor(*title_bg_rgb)
        painter.setPen(QtCore.Qt.NoPen)
        painter.setBrush(title_bg_color)
        painter.drawRoundedRect(bg_rect, 3.0, 3.0)

        # タイトルテキストを描画
        if selected:
            rgb = getattr(self, '_highlight_title_color', NODE_GRAPH_FOCUS_TEXT_COLOR)
        else:
            rgb = getattr(self, '_normal_title_color', (20, 20, 20))
        text_color = QtGui.QColor(*rgb)
        text_draw_rect = QtCore.QRectF(
            text_item.pos().x(), text_item.pos().y(),
            text_item.boundingRect().width(), text_item.boundingRect().height()
        )
        painter.setPen(text_color)
        painter.setFont(text_item.font())
        painter.drawText(text_draw_rect, QtCore.Qt.AlignCenter, self.name)
        body = (getattr(self, "_body_text", None) or "").strip()
        if body and not getattr(self, "_title_editing", False):
            body_font = QtGui.QFont(text_item.font())
            ps = body_font.pointSize()
            if ps > 0:
                body_font.setPointSize(max(7, ps - 1))
            else:
                px = body_font.pixelSize()
                body_font.setPixelSize(max(9, px - 1))
            painter.setFont(body_font)
            painter.setPen(text_color)
            margin_side = 6.0
            body_top = text_draw_rect.bottom() + 5.0
            body_bottom_reserve = 22.0
            body_rect = QtCore.QRectF(
                node_rect.x() + margin_side,
                body_top,
                node_rect.width() - 2.0 * margin_side,
                max(18.0, node_rect.bottom() - body_top - body_bottom_reserve),
            )
            # 通常は中央寄せ。Memo など左寄せしたいノードは _body_align_left=True。
            if getattr(self, "_body_align_left", False):
                _align = QtCore.Qt.AlignLeft | QtCore.Qt.AlignTop
            else:
                _align = QtCore.Qt.AlignHCenter | QtCore.Qt.AlignTop
            painter.drawText(body_rect, _align, body)
            painter.setFont(text_item.font())

        # Draw the left-side badge. Pose shows the frame count, and
        # Branching shows a PAD button name or an analog axis name.
        # Only drawn when the node has a _compute_left_badge callable.
        badge_fn = getattr(self, "_compute_left_badge", None)
        badge_text = None
        if callable(badge_fn):
            try:
                raw = badge_fn()
                if raw is not None:
                    badge_text = str(raw).strip() or None
            except Exception:
                badge_text = None
        if badge_text and not getattr(self, "_title_editing", False):
            badge_font = QtGui.QFont(text_item.font())
            ps = badge_font.pointSize()
            if ps > 0:
                badge_font.setPointSize(max(7, ps - 1))
            else:
                px = badge_font.pixelSize()
                badge_font.setPixelSize(max(9, px - 1))
            painter.setFont(badge_font)
            painter.setPen(text_color)
            badge_margin = 4.0
            body_center_y = (text_draw_rect.bottom() + node_rect.bottom()) / 2.0
            badge_rect = QtCore.QRectF(
                node_rect.x() + badge_margin,
                body_center_y - 8.0,
                max(20.0, node_rect.width() / 2.0 - badge_margin),
                16.0,
            )
            painter.drawText(badge_rect, QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter, badge_text)
            painter.setFont(text_item.font())
        self._restore_node_cache_mode()

    def _restore_node_cache_mode(self):
        """選択変更で NoCache にしたノードのキャッシュを元に戻す"""
        if getattr(self, '_cache_mode_restore', None) is not None:
            self.setCacheMode(self._cache_mode_restore)
            self._cache_mode_restore = None

    def eventFilter(self, obj, event):
        """_text_item のフォーカス喪失で編集終了・非表示に戻す"""
        if obj is getattr(self, '_text_item', None) and event.type() == QtCore.QEvent.FocusOut:
            self._title_editing = False
            obj.setVisible(False)
        return super(CustomNodeItem, self).eventFilter(obj, event)

    def mouseDoubleClickEvent(self, event):
        """ダブルクリックでタイトル編集時は _title_editing を立てて _text_item を表示してから親の処理へ"""
        text_item = getattr(self, '_text_item', None)
        if text_item is not None and not self.disabled:
            self._title_editing = True
            text_item.setVisible(True)
        return super(CustomNodeItem, self).mouseDoubleClickEvent(event)

    def set_title_color(self, r, g, b, highlight_color=None):
        """ノードのタイトル色を保存（描画は paint() で行う）
        Args:
            r, g, b: 通常時のタイトル色
            highlight_color: ハイライト時のタイトル色 (r, g, b) のタプル
        """
        self._normal_title_color = (r, g, b)
        if highlight_color:
            self._highlight_title_color = highlight_color

    def set_highlight_colors(self, panel_bg=None, title_bg=None,
                              input_port=None, input_port_border=None,
                              output_port=None, output_port_border=None):
        """ハイライト時の各色を保存"""
        if panel_bg:
            self._highlight_panel_bg = panel_bg
        if title_bg:
            self._highlight_title_bg = title_bg
        if input_port:
            self._highlight_input_port = input_port
        if input_port_border:
            self._highlight_input_port_border = input_port_border
        if output_port:
            self._highlight_output_port = output_port
        if output_port_border:
            self._highlight_output_port_border = output_port_border

    def set_normal_colors(self, panel_bg=None, title_bg=None,
                          input_port=None, input_port_border=None,
                          output_port=None, output_port_border=None):
        """通常時の各色を保存"""
        if panel_bg:
            self._normal_panel_bg = panel_bg
        if title_bg:
            self._normal_title_bg = title_bg
        if input_port:
            self._normal_input_port = input_port
        if input_port_border:
            self._normal_input_port_border = input_port_border
        if output_port:
            self._normal_output_port = output_port
        if output_port_border:
            self._normal_output_port_border = output_port_border

    def _set_text_color(self, color):
        """親の_set_text_colorは無視（色は paint() で option.state に応じて描画）"""
        pass

    def itemChange(self, change, value):
        """選択変更でキャッシュを無効化して再描画し、各色を切り替える"""
        # 位置変更後にポートを再配置（初回移動時のみ）
        if change == QtWidgets.QGraphicsItem.GraphicsItemChange.ItemPositionHasChanged:
            if not getattr(self, '_ports_aligned_after_move', False):
                self._ports_aligned_after_move = True
                self._align_ports(0.0)
                self.update()
        if change == QtWidgets.QGraphicsItem.GraphicsItemChange.ItemSelectedChange:
            # 初期化完了前は処理をスキップ
            if not getattr(self, '_post_init_done', False):
                return super(CustomNodeItem, self).itemChange(change, value)
            try:
                self._cache_mode_restore = self.cacheMode()
                self.setCacheMode(QtWidgets.QGraphicsItem.NoCache)
            except Exception:
                self._cache_mode_restore = None

            will_be_selected = bool(value)

            # テキスト色変更
            text_item = getattr(self, '_text_item', None)
            if text_item is not None:
                if will_be_selected:
                    rgb = getattr(self, '_highlight_title_color', NODE_GRAPH_FOCUS_TEXT_COLOR)
                else:
                    rgb = getattr(self, '_normal_title_color', (20, 20, 20))
                text_item.setDefaultTextColor(QtGui.QColor(*rgb))

            # パネル背景色変更
            if will_be_selected:
                panel_bg = getattr(self, '_highlight_panel_bg', None)
            else:
                panel_bg = getattr(self, '_normal_panel_bg', None)
            if panel_bg:
                self.color = panel_bg

            # 入力ポート色変更
            for port_view in self.inputs:
                if will_be_selected:
                    port_color = getattr(self, '_highlight_input_port', None)
                    border_color = getattr(self, '_highlight_input_port_border', None)
                else:
                    port_color = getattr(self, '_normal_input_port', None)
                    border_color = getattr(self, '_normal_input_port_border', None)
                if port_color:
                    port_view.color = port_color
                if border_color:
                    port_view.border_color = border_color

            # 出力ポート色変更
            for port_view in self.outputs:
                if will_be_selected:
                    port_color = getattr(self, '_highlight_output_port', None)
                    border_color = getattr(self, '_highlight_output_port_border', None)
                else:
                    port_color = getattr(self, '_normal_output_port', None)
                    border_color = getattr(self, '_normal_output_port_border', None)
                if port_color:
                    port_view.color = port_color
                if border_color:
                    port_view.border_color = border_color

            branch_output_colors = getattr(self, '_branching_output_colors', None)
            if branch_output_colors:
                for i, port_view in enumerate(self.outputs):
                    if i < len(branch_output_colors):
                        port_view.color, port_view.border_color = branch_output_colors[i]

            self.update()
        return super(CustomNodeItem, self).itemChange(change, value)

class CustomPipe(PipeItem):
    """カスタムパイプクラス - ラインの始点・終点をポート（ぽっち）の中心にする"""

    def __init__(self, input_port=None, output_port=None):
        super(CustomPipe, self).__init__(input_port, output_port)
        # パイプをパネルの後ろに配置するためにZValueを低く設定
        self.setZValue(-1)

        self._color = self._pipe_color_for_port(output_port)
        self.set_pipe_styling(color=self._color, width=2)
        # Memo 由来のパイプは点線で描画
        if self._is_memo_output_port(output_port):
            try:
                pen = self.pen()
                pen.setStyle(QtCore.Qt.DotLine)
                pen.setCapStyle(QtCore.Qt.RoundCap)
                self.setPen(pen)
            except Exception:
                pass

    def _port_color_tuple(self, port):
        """NodeGraphQtのポート色をRGBタプルとして取得する"""
        if not port:
            return None
        color = getattr(port, 'color', None)
        if callable(color):
            try:
                color = color()
            except TypeError:
                return None
        if isinstance(color, QtGui.QColor):
            return (color.red(), color.green(), color.blue())
        if isinstance(color, (list, tuple)) and len(color) >= 3:
            return tuple(int(v) for v in color[:3])
        return None

    def _is_memo_output_port(self, output_port) -> bool:
        """出力ポートが MemoNode 由来かを判定する。"""
        if output_port is None:
            return False
        port_node = getattr(output_port, "node", None)
        if port_node is None:
            return False
        try:
            model_node = getattr(port_node, "node", None)
            if model_node is not None:
                if type(model_node).__name__ == "MemoNode":
                    return True
                if getattr(type(model_node), "NODE_NAME", "") == "MemoNode":
                    return True
        except Exception:
            pass
        # port.color がメモ用の色と完全一致する場合もメモ由来と判定
        if self._port_color_tuple(output_port) == NODE_MEMO_OUTPUT_PORT_COLOR:
            return True
        return False

    def _pipe_color_for_port(self, output_port):
        """出力ポートに応じてパイプ色を決める:
        - 赤ポッチ (Branch to): 赤
        - Memo (透明ポッチ): グレー
        - その他: ミントグリーン
        """
        if self._is_memo_output_port(output_port):
            return NODE_MEMO_PIPE_COLOR
        if self._port_color_tuple(output_port) == BRANCH_POINT_COLOR:
            return BRANCH_LINE_COLOR
        return MINT_GREEN_COLOR

    def _update_pipe_color(self, output_port=None, width=2, style=None):
        output_port = output_port or self.output_port
        self._color = self._pipe_color_for_port(output_port)
        if style is None:
            self.set_pipe_styling(color=self._color, width=width)
        else:
            self.set_pipe_styling(color=self._color, width=width, style=style)
        self._apply_direction_pointer_color(width=width)
        # Memo 由来のパイプは点線で描画する。set_pipe_styling で pen の style が
        # 上書きされるので、その後に DotLine に差し替える。
        if self._is_memo_output_port(output_port):
            try:
                pen = self.pen()
                pen.setStyle(QtCore.Qt.DotLine)
                pen.setCapStyle(QtCore.Qt.RoundCap)
                self.setPen(pen)
            except Exception:
                pass

    def _apply_direction_pointer_color(self, width=2):
        color = QtGui.QColor(*self._color)
        pen = self._dir_pointer.pen()
        pen.setColor(color)
        pen.setWidth(width)
        self._dir_pointer.setPen(pen)
        self._dir_pointer.setBrush(color.darker(120))

    def _port_center_scene_pos(self, port):
        """ポート（ぽっち）の中心をシーン座標で返す（非表示ポートも対応）"""
        br = port.boundingRect()
        node = port.node
        # 入力ポートが非表示の場合、出力ポート（下部中央のグリーンポッチ）の位置に接続
        if not port.isVisible() and node:
            # ノードの出力ポートを探す
            outputs = list(node.outputs)
            if outputs:
                # 最初の出力ポートの位置を使用
                out_port = outputs[0]
                out_sp = out_port.scenePos()
                out_br = out_port.boundingRect()
                return QtCore.QPointF(
                    out_sp.x() + out_br.width() / 2,
                    out_sp.y() + out_br.height() / 2
                )
        # 通常のポート位置取得
        sp = port.scenePos()
        return QtCore.QPointF(
            sp.x() + br.width() / 2,
            sp.y() + br.height() / 2
        )

    def draw_path(self, start_port, end_port=None, cursor_pos=None):
        """パイプのパスを描画 - 始点と終点をポート（ぽっち）の中心にする"""
        try:
            if not start_port:
                return

            # 常にoutput_port → input_portの方向で描画
            if self.output_port and self.input_port:
                actual_start_port = self.output_port
                actual_end_port = self.input_port
            else:
                actual_start_port = start_port
                actual_end_port = end_port

            self._update_pipe_color(actual_start_port, width=2)

            # 始点：ポートの中心（ぽっちの中心）
            pos1 = self._port_center_scene_pos(actual_start_port)

            # 終点の設定
            if cursor_pos:
                pos2 = cursor_pos
            elif actual_end_port:
                # 終点：ポートの中心（ぽっちの中心）
                pos2 = self._port_center_scene_pos(actual_end_port)
            else:
                return

            # 可視性チェック（入力ポートは非表示でもOK、ノードの可視性のみチェック）
            if self.input_port and self.output_port:
                is_visible = all([
                    self._input_port.node.isVisible(),
                    self._output_port.node.isVisible()
                ])
                self.setVisible(is_visible)
                if not is_visible:
                    return

            # パスの作成（直線）
            path = QtGui.QPainterPath()
            path.moveTo(pos1)
            path.lineTo(pos2)

            self.setPath(path)

            # ドラッグ中は矢印を非表示、接続完了時のみ表示
            if cursor_pos:
                self._dir_pointer.setVisible(False)
            else:
                self._draw_direction_pointer()

        except Exception as e:
            print(f"[draw_path] Exception: {e}")

    def _draw_direction_pointer(self):
        """
        矢印の描画 - パス上の中点に配置し、出力から入力への方向を指す
        """
        import math
        from Qt import QtGui
        from NodeGraphQt.constants import PipeEnum

        if not (self.input_port and self.output_port):
            self._dir_pointer.setVisible(False)
            return

        self._apply_direction_pointer_color(width=self.pen().width())

        if self.disabled():
            if not (self._active or self._highlight):
                color = QtGui.QColor(*PipeEnum.DISABLED_COLOR.value)
                pen = self._dir_pointer.pen()
                pen.setColor(color)
                self._dir_pointer.setPen(pen)
                self._dir_pointer.setBrush(color.darker(200))

        # パス上の位置を使用（座標系のずれを防止）
        path = self.path()
        if path.isEmpty():
            self._dir_pointer.setVisible(False)
            return

        loc_pt = path.pointAtPercent(0.49)
        tgt_pt = path.pointAtPercent(0.51)
        mid_pt = path.pointAtPercent(0.5)

        # 出力から入力への角度を計算
        dx = tgt_pt.x() - loc_pt.x()
        dy = tgt_pt.y() - loc_pt.y()

        radians = math.atan2(dy, dx)
        degrees = math.degrees(radians) - 90 + 180  # ポリゴンが上向きなので補正

        self._dir_pointer.setRotation(degrees)
        self._dir_pointer.setPos(mid_pt)
        self._dir_pointer.setVisible(True)

    def activate(self):
        """アクティブ時もポート由来の色を維持"""
        self._active = True
        self._update_pipe_color(width=3)

    def highlight(self):
        """ハイライト時もポート由来の色を維持"""
        self._highlight = True
        self._update_pipe_color(width=2)

    def reset(self):
        """リセット時もポート由来の色を維持"""
        self._active = False
        self._highlight = False
        self._update_pipe_color(width=2)
        self._draw_direction_pointer()

class CustomLivePipe(CustomPipe):
    """カスタムライブパイプクラス - ドラッグ中のラインも直線にする"""

    def __init__(self):
        super(CustomLivePipe, self).__init__()
        from NodeGraphQt.constants import PipeEnum, Z_VAL_NODE_WIDGET
        from NodeGraphQt.qgraphics.pipe import LivePipePolygonItem

        # LivePipeItemと同じ設定
        self.setZValue(Z_VAL_NODE_WIDGET + 1)
        # ミントグリーンの色を設定
        self.color = MINT_GREEN_COLOR
        self.style = PipeEnum.DRAW_TYPE_DASHED.value
        self.set_pipe_styling(color=self.color, width=3, style=self.style)
        self.shift_selected = False
        self._color = MINT_GREEN_COLOR

        # インデックスポインター（LivePipeItemと同じ）
        self._idx_pointer = LivePipePolygonItem(self)
        self._idx_pointer.setPolygon(self._poly)
        self._idx_pointer.setBrush(QtGui.QColor(*self.color).darker(300))
        pen = self._idx_pointer.pen()
        pen.setWidth(self.pen().width())
        pen.setColor(self.pen().color())
        pen.setJoinStyle(QtCore.Qt.MiterJoin)
        self._idx_pointer.setPen(pen)

        color = self.pen().color()
        color.setAlpha(80)
        from Qt import QtWidgets
        self._idx_text = QtWidgets.QGraphicsTextItem(self)
        self._idx_text.setDefaultTextColor(color)
        font = self._idx_text.font()
        font.setPointSize(7)
        self._idx_text.setFont(font)

    def hoverEnterEvent(self, event):
        """LivePipeItemと同じ動作"""
        from Qt import QtWidgets
        QtWidgets.QGraphicsPathItem.hoverEnterEvent(self, event)

    def draw_path(self, start_port, end_port=None, cursor_pos=None, color=None):
        """直線でパスを描画（LivePipeItem互換）"""
        # CustomPipeのdraw_pathを呼び出して直線描画
        super(CustomLivePipe, self).draw_path(start_port, end_port, cursor_pos)

        # インデックスポインターを更新（LivePipeItemと同じ処理）
        if cursor_pos:
            if self._port_color_tuple(start_port) == BRANCH_POINT_COLOR:
                color = self._color
            elif not color:
                color = self._color

            if color:
                pen = QtGui.QPen(QtGui.QColor(*color), 3)
                pen.setStyle(QtCore.Qt.DashLine)
                self.setPen(pen)
                self._idx_pointer.setBrush(QtGui.QColor(*color).darker(300))
                pen = self._idx_pointer.pen()
                pen.setColor(QtGui.QColor(*color))
                self._idx_pointer.setPen(pen)

            self._idx_pointer.setVisible(True)
            self._idx_pointer.setRotation(self._dir_pointer.rotation())
            self._idx_pointer.setPos(self._dir_pointer.pos())

    def draw_index_pointer(self, port, pos):
        """インデックスポインターを描画（LivePipeItemと互換）"""
        self._idx_pointer.setVisible(True)
        self._idx_pointer.setPos(pos)
        self._idx_text.setPlainText(str(port.connected_pipes))
        self._idx_text.setPos(pos.x() - 10.0, pos.y() - 20.0)

    def reset_path(self):
        """パスをリセット（LivePipeItemと互換）"""
        path = QtGui.QPainterPath(QtCore.QPointF(0.0, 0.0))
        self.setPath(path)
        self._idx_pointer.setVisible(False)
        self._idx_text.setPlainText('')

    def activate(self):
        """アクティブ時もポート由来の色を維持（CustomLivePipe用）"""
        self._active = True
        self._update_pipe_color(width=3, style=self.style)

    def highlight(self):
        """ハイライト時もポート由来の色を維持（CustomLivePipe用）"""
        self._highlight = True
        self._update_pipe_color(width=3, style=self.style)

    def reset(self):
        """リセット時もポート由来の色を維持（CustomLivePipe用）"""
        self._active = False
        self._highlight = False
        self._update_pipe_color(width=3, style=self.style)

class CustomViewer(NodeViewer):
    """カスタムビューアクラス - CustomPipeを使用"""

    # Class-level clipboard for copy/paste
    _clipboard = None
    _clipboard_connections = None

    def __init__(self, parent=None):
        super(CustomViewer, self).__init__(parent)

        # _LIVE_PIPEをCustomLivePipeに置き換え
        self.scene().removeItem(self._LIVE_PIPE)
        self._LIVE_PIPE = CustomLivePipe()
        self._LIVE_PIPE.setVisible(False)
        self.scene().addItem(self._LIVE_PIPE)

        # Graphへの参照（後で設定される）
        self._graph = None

        # Undo/redo hooks — set by build_motion_editor
        self._undo_push_fn = None       # push_undo()
        self._undo_push_raw_fn = None   # undo_stack.push(snap)
        self._capture_snap_fn = None    # _capture_undo_snapshot()
        self._pre_move_snap = None      # snapshot taken at mousePressEvent
        self._pre_move_positions = {}   # {node_id: (x, y)} at mousePressEvent

        # Shift+左ドラッグ パン
        self._shift_pan = False
        self._shift_pan_last = QtCore.QPoint()

    def set_graph(self, graph):
        """Graphオブジェクトへの参照を設定"""
        self._graph = graph

    def acyclic_check(self, start_port, end_port):
        """循環接続を許可（常にTrueを返す）"""
        # Motion graphでは循環接続を許可する
        # 再生時はvisitedセットで無限ループを防止
        return True

    def snap_to_grid(self, value):
        """値をグリッドにスナップ"""
        return round(value / NODE_GRAPH_GRID_SNAP_SIZE) * NODE_GRAPH_GRID_SNAP_SIZE

    def snap_selected_nodes_to_grid(self):
        """選択されているノードをグリッドにスナップ"""
        if not self._graph:
            return
        for node in self._graph.selected_nodes():
            pos = node.pos()
            snapped_x = self.snap_to_grid(pos[0])
            snapped_y = self.snap_to_grid(pos[1])
            if pos[0] != snapped_x or pos[1] != snapped_y:
                node.set_pos(snapped_x, snapped_y)

    def establish_connection(self, start_port, end_port):
        """
        カスタムパイプを使用して接続を確立
        """
        pipe = CustomPipe()
        self.scene().addItem(pipe)
        pipe.set_connections(start_port, end_port)
        pipe.draw_path(pipe.output_port, pipe.input_port)
        if start_port.node.selected or end_port.node.selected:
            pipe.highlight()
        if not start_port.node.visible or not end_port.node.visible:
            pipe.hide()
        self.scene().update()

    def mousePressEvent(self, event):
        """長押し検出/Shift複数選択/右クリックメニューを追加してから親クラスの処理へ"""
        cur = event.position().toPoint() if hasattr(event, "position") else event.pos()

        # 右クリック / macOS の Ctrl+左クリック でコンテキストメニュー
        _is_right = event.button() == QtCore.Qt.MouseButton.RightButton
        _is_ctrl_left_mac = (
            sys.platform == "darwin"
            and event.button() == QtCore.Qt.MouseButton.LeftButton
            and bool(event.modifiers() & QtCore.Qt.KeyboardModifier.ControlModifier)
        )
        if (_is_right or _is_ctrl_left_mac) and self._graph and hasattr(self._graph, '_show_context_menu'):
            scene_pos = self.mapToScene(cur)
            if hasattr(event, "globalPosition"):
                global_pos = event.globalPosition().toPoint()
            else:
                global_pos = event.globalPos()
            self._graph._show_context_menu(scene_pos, global_pos)
            return

        # Shift+左クリック: ノード上ならトグル選択、空白ならパンに専念
        if (event.button() == QtCore.Qt.MouseButton.LeftButton and
                event.modifiers() & QtCore.Qt.KeyboardModifier.ShiftModifier):
            if self._graph and hasattr(self._graph, '_node_from_scene_pos'):
                scene_pos = self.mapToScene(cur)
                _node = self._graph._node_from_scene_pos(scene_pos)
                if _node is not None:
                    try:
                        _node.set_selected(not _node.selected())
                    except Exception as _e:
                        print(f"[ShiftSelect] toggle failed: {_e}")
                    return
            self._shift_pan = True
            self._shift_pan_last = cur
            self.setCursor(QtCore.Qt.CursorShape.ClosedHandCursor)
            return
        # Capture pre-move snapshot for undo detection on node drag
        if event.button() == QtCore.Qt.MouseButton.LeftButton and self._graph and self._capture_snap_fn:
            self._pre_move_snap = self._capture_snap_fn()
            self._pre_move_positions = {
                node.id: node.pos() for node in self._graph.all_nodes()
            }
        super(CustomViewer, self).mousePressEvent(event)
        if event.button() == QtCore.Qt.MouseButton.LeftButton and self._graph:
            scene_pos = self.mapToScene(cur)
            self._graph._start_long_press(scene_pos, cur)

    def mouseMoveEvent(self, event):
        """ドラッグで長押しをキャンセル / Shift+ドラッグでパン"""
        cur = event.position().toPoint() if hasattr(event, "position") else event.pos()
        if self._shift_pan:
            # NodeViewer と同じ方式: scene 座標の差分で _set_viewer_pan を呼ぶ
            prev_scene = self.mapToScene(self._shift_pan_last)
            cur_scene = self.mapToScene(cur)
            delta = prev_scene - cur_scene
            self._set_viewer_pan(delta.x(), delta.y())
            self._shift_pan_last = cur
            return
        super(CustomViewer, self).mouseMoveEvent(event)
        if self._graph:
            self._graph._cancel_long_press_if_dragged(cur)

    def mouseReleaseEvent(self, event):
        """マウスリリース時、パネル全体を接続ターゲットとして扱う"""
        if event.button() == QtCore.Qt.MouseButton.LeftButton and self._shift_pan:
            self._shift_pan = False
            self.unsetCursor()
            return
        if event.button() == QtCore.Qt.MouseButton.LeftButton and self._graph:
            self._graph._cancel_long_press()

        # ライブパイプが表示されている場合（接続をドラッグ中）
        if self._LIVE_PIPE.isVisible() and self._start_port:
            # マウス位置のアイテムを取得
            pos = self.mapToScene(event.pos())
            items = self.scene().items(pos)

            # ノードアイテムを探す
            from NodeGraphQt.qgraphics.node_abstract import AbstractNodeItem
            target_node = None
            for item in items:
                if isinstance(item, AbstractNodeItem):
                    # 接続開始元のノードではないことを確認
                    if item != self._start_port.node:
                        target_node = item
                        break

            if target_node and hasattr(target_node, 'inputs') and len(target_node.inputs) > 0:
                # 最初の入力ポート（非表示だが存在する）を接続先として使用
                end_port = target_node.inputs[0]
                start_port = self._start_port

                if start_port and end_port and start_port != end_port:
                    # 接続可能かチェック
                    if self.acyclic_check(start_port, end_port):
                        # 切断ペアを収集（_detached_port + 既存パイプ）
                        disconnected = []
                        from NodeGraphQt.constants import PortTypeEnum

                        # _detached_port: mousePressEvent でパイプが削除済みだが
                        # ポートモデルに残っている古い接続先
                        if self._detached_port:
                            disconnected.append((start_port, self._detached_port))

                        # 出力ポートにまだ残っている既存パイプも削除
                        if start_port.port_type == PortTypeEnum.OUT.value:
                            existing_pipes = list(start_port.connected_pipes)
                            for pipe in existing_pipes:
                                in_p = getattr(pipe, 'input_port', None)
                                out_p = getattr(pipe, 'output_port', None)
                                if in_p and out_p:
                                    disconnected.append((out_p, in_p))
                                pipe.delete()

                        # ライブパイプを非表示に
                        self._LIVE_PIPE.setVisible(False)
                        self._LIVE_PIPE.shift_selected = False
                        # NodeGraphQt標準の mouseReleaseEvent と同様に、ドラッグ元
                        # ポートの hovered を明示的に解除する。これをしないと
                        # NodeItem.mousePressEvent が hovered==True のポートを
                        # 持つノードへのクリックを無視し続け、パネルが選択不能になる。
                        start_port.hovered = False
                        self._start_port = None
                        self._detached_port = None

                        # 切断 + 接続を1回のシグナルで送る。
                        # connection_changed → _on_connection_changed → PortConnectedCmd.redo()
                        # → source.view.connect_to() → establish_connection() が呼ばれるので
                        # ここで establish_connection を別途呼ぶ必要はない（二重パイプ防止）。
                        connected = [(start_port, end_port)]
                        self.connection_changed.emit(disconnected, connected)

                        self.LMB_state = False
                        return
            else:
                # ターゲットノードが見つからない場合、接続を切断
                if self._detached_port:
                    # Case 1: ポートから直接ドラッグ → mousePressEvent でパイプ削除済み
                    # _detached_port ペアでモデルのみ更新
                    disconnected_pairs = [(self._start_port, self._detached_port)]
                    self.connection_changed.emit(disconnected_pairs, [])
                else:
                    # Case 2: パイプ端点をドラッグ → パイプ残存、削除+モデル更新
                    start_port = self._start_port
                    if start_port:
                        existing_pipes = list(start_port.connected_pipes)
                        if existing_pipes:
                            disconnected_pairs = []
                            for pipe in existing_pipes:
                                in_p = getattr(pipe, 'input_port', None)
                                out_p = getattr(pipe, 'output_port', None)
                                if in_p and out_p:
                                    disconnected_pairs.append((out_p, in_p))
                                pipe.delete()
                            self.connection_changed.emit(disconnected_pairs, [])

                # ライブパイプをリセット
                self._LIVE_PIPE.setVisible(False)
                self._LIVE_PIPE.shift_selected = False
                # 上と同様、ドラッグ元ポートの hovered を解除してから外す。
                if self._start_port is not None:
                    self._start_port.hovered = False
                self._start_port = None
                self._detached_port = None
                self.LMB_state = False
                return

        super(CustomViewer, self).mouseReleaseEvent(event)

        # ノードをグリッドにスナップ
        self.snap_selected_nodes_to_grid()

        # Detect node movement and push undo if any node moved
        if (event.button() == QtCore.Qt.MouseButton.LeftButton
                and self._graph and self._pre_move_snap is not None
                and self._undo_push_raw_fn):
            moved = False
            for node in self._graph.all_nodes():
                old_pos = self._pre_move_positions.get(node.id)
                if old_pos is not None:
                    cur = node.pos()
                    if abs(cur[0] - old_pos[0]) > 1 or abs(cur[1] - old_pos[1]) > 1:
                        moved = True
                        break
            if moved:
                self._undo_push_raw_fn(self._pre_move_snap)
        self._pre_move_snap = None
        self._pre_move_positions = {}

    def keyPressEvent(self, event):
        """キーボードイベントを処理 - DeleteキーとBackspaceキーでノード削除、Ctrl+Dで複製"""
        # DeleteキーまたはBackspaceキーが押された場合
        if event.key() in (QtCore.Qt.Key_Delete, QtCore.Qt.Key_Backspace):
            if self._graph:
                # delete_selected_node関数を呼び出し
                selected_nodes = self._graph.selected_nodes()
                if selected_nodes:
                    if self._undo_push_fn:
                        self._undo_push_fn()
                    for node in selected_nodes:
                        # BaseLinkNodeは削除不可
                        if isinstance(node, BaseLinkNode):
                            print("Cannot delete Base Link node")
                            continue
                        self._graph.remove_node(node)
                    print(f"Deleted {len(selected_nodes)} node(s)")
                else:
                    print("No node selected for deletion")
                return

        # Ctrl+… (Cmd+… on macOS) — use primary_mod_held for Win/Linux/macOS
        _prim = primary_mod_held(event.modifiers())

        # Ctrl/Cmd+D でノード複製
        if event.key() == QtCore.Qt.Key_D and _prim:
            if self._graph:
                selected_nodes = self._graph.selected_nodes()
                if selected_nodes:
                    if self._undo_push_fn:
                        self._undo_push_fn()
                    self._duplicate_nodes(selected_nodes)
                else:
                    print("No node selected for duplication")
                return

        # Ctrl/Cmd+C でノードコピー
        if event.key() == QtCore.Qt.Key_C and _prim:
            if self._graph:
                selected_nodes = self._graph.selected_nodes()
                if selected_nodes:
                    self._copy_nodes(selected_nodes)
                else:
                    print("No node selected for copy")
                return

        # Ctrl/Cmd+V でノードペースト
        if event.key() == QtCore.Qt.Key_V and _prim:
            if self._graph:
                if self._undo_push_fn:
                    self._undo_push_fn()
                self._paste_nodes()
                return

        # Ctrl/Cmd+X でノードカット
        if event.key() == QtCore.Qt.Key_X and _prim:
            if self._graph:
                selected_nodes = self._graph.selected_nodes()
                if selected_nodes:
                    if self._undo_push_fn:
                        self._undo_push_fn()
                    self._cut_nodes(selected_nodes)
                else:
                    print("No node selected for cut")
                return

        # Ctrl/Cmd+A で全ノード選択
        if event.key() == QtCore.Qt.Key_A and _prim:
            if self._graph:
                self._select_all_nodes()
                return

        # 親クラスの処理を実行
        super(CustomViewer, self).keyPressEvent(event)

    def _select_all_nodes(self):
        """全ノードを選択"""
        if not self._graph:
            return
        all_nodes = self._graph.all_nodes()
        for node in all_nodes:
            node.set_selected(True)
        print(f"[Select All] Selected {len(all_nodes)} nodes")

    def _collect_graph_name_set(self):
        """現在グラフ内で使われているノード名 / pose_name / define_memo を集める。"""
        names = set()
        if not self._graph:
            return names
        for node in self._graph.all_nodes():
            try:
                n = node.name()
                if n:
                    names.add(n)
            except Exception:
                pass
            if isinstance(node, PoseNode):
                pn = getattr(node, "pose_name", None)
                if pn:
                    names.add(pn)
            if isinstance(node, DefineNode):
                memo = getattr(node, "define_memo", None) or ""
                if memo:
                    names.add(memo)
        return names

    @staticmethod
    def _next_unique_numbered_name(base_name, taken):
        """コピペ/デュプリケート用の次番号名を返す。

        - ``pose_0`` → 未使用なら ``pose_1``、埋まっていれば ``pose_2`` …
        - 末尾に ``_数字`` が無い名前は ``name_0``, ``name_1`` … を順に探す
        - 見つかった名前は ``taken`` に追加する（同一操作内の衝突回避）
        """
        name = (base_name or "").strip()
        if not name:
            name = "node"
        stem = name
        start_n = 0
        us = name.rfind("_")
        if us >= 0 and us < len(name) - 1 and name[us + 1:].isdigit():
            stem = name[:us]
            if not stem:
                stem = name
            else:
                start_n = int(name[us + 1:]) + 1
        n = start_n
        while True:
            candidate = f"{stem}_{n}"
            if candidate not in taken:
                taken.add(candidate)
                return candidate
            n += 1

    def _duplicate_nodes(self, nodes, offset_x=50, offset_y=50):
        """選択されたノードを複製（全データをコピー、接続も保持）"""
        if not self._graph:
            return []

        # Mapping from old node id to new node
        node_mapping = {}
        new_nodes = []
        old_node_ids = set(id(n) for n in nodes)
        taken_names = self._collect_graph_name_set()

        for node in nodes:
            new_node = None
            if isinstance(node, PoseNode):
                # PoseNodeの複製
                try:
                    new_node = self._graph.create_node('motion.nodes.PoseNode')
                    if new_node:
                        pos = node.pos()
                        new_node.set_pos(pos[0] + offset_x, pos[1] + offset_y)
                        new_node.set_color(*NODE_COLOR_DEFAULT)
                        new_node.pose_name = self._next_unique_numbered_name(
                            node.pose_name, taken_names)
                        new_node.set_name(new_node.pose_name)
                        new_node.duration = node.duration
                        new_node.frames = getattr(node, 'frames', get_default_hz_fps())
                        new_node.angles_deg = dict(node.angles_deg)
                        new_node.joint_easings = dict(getattr(node, 'joint_easings', {}))
                        new_node.branching_enabled = getattr(node, 'branching_enabled', False)
                        new_node.branch_outputs_swapped = getattr(node, 'branch_outputs_swapped', False)
                        new_node.branch_if_left = getattr(node, 'branch_if_left', "UserVal_0")
                        new_node.branch_if_op = getattr(node, 'branch_if_op', "==")
                        new_node.branch_if_right = getattr(node, 'branch_if_right', "UserVal_1")
                        new_node.branch_if_uv_enabled = getattr(node, 'branch_if_uv_enabled', True)
                        new_node.branch_if_formula_enabled = getattr(node, 'branch_if_formula_enabled', False)
                        new_node.branch_if_formula = getattr(node, 'branch_if_formula', "Form1:foo")
                        new_node.branch_if_pad_enabled = getattr(node, 'branch_if_pad_enabled', False)
                        new_node.branch_if_pad_button = getattr(node, 'branch_if_pad_button', "L1")
                        new_node.branch_if_pad_analog_enabled = getattr(node, 'branch_if_pad_analog_enabled', False)
                        new_node.branch_if_pad_analog_axis = getattr(node, 'branch_if_pad_analog_axis', "Lx")
                        new_node.branch_if_pad_analog_op = getattr(node, 'branch_if_pad_analog_op', ">=")
                        new_node.branch_if_pad_analog_threshold = getattr(node, 'branch_if_pad_analog_threshold', 0)
                        # ロード時と同じく、出力ポート追加前に高さをロックしないと
                        # add_output のたびに _set_base_size で高さが拡大する。
                        if new_node.branching_enabled:
                            new_node._lock_output_row_height()
                        while new_node.output_count < len(node.out_port_labels):
                            label_idx = new_node.output_count
                            new_node._add_pose_output(
                                node.out_port_labels[label_idx],
                                node.out_port_priorities[label_idx]
                                if label_idx < len(node.out_port_priorities) else 0
                            )
                        new_node.out_port_labels = list(node.out_port_labels)
                        new_node.out_port_priorities = list(node.out_port_priorities)
                        new_node._sync_branching_port_labels()
                        new_node._apply_pose_output_colors()
                        print(f"[Duplicate] Created copy of PoseNode: {new_node.pose_name}")
                except Exception as e:
                    print(f"[Duplicate] Error duplicating PoseNode: {e}")
                    traceback.print_exc()
            elif isinstance(node, DefineNode):
                try:
                    new_node = self._graph.create_node('motion.nodes.DefineNode')
                    if new_node:
                        pos = node.pos()
                        new_node.set_pos(pos[0] + offset_x, pos[1] + offset_y)
                        new_node.set_color(*NODE_DEFINE_PANEL_BG_COLOR)
                        new_node.define_uv_index = getattr(node, "define_uv_index", 0)
                        _memo = getattr(node, "define_memo", "") or ""
                        new_node.define_memo = (
                            self._next_unique_numbered_name(_memo, taken_names)
                            if _memo else "")
                        new_node.define_kind = getattr(node, "define_kind", "literal")
                        new_node.define_literal = getattr(node, "define_literal", 0)
                        new_node.define_register_name = getattr(node, "define_register_name", "") or ""
                        new_node.set_name(self._next_unique_numbered_name(
                            node.name(), taken_names))
                        print(f"[Duplicate] Created copy of DefineNode: {new_node.name()}")
                except Exception as e:
                    print(f"[Duplicate] Error duplicating DefineNode: {e}")
                    traceback.print_exc()
            elif isinstance(node, WaitNode):
                try:
                    new_node = self._graph.create_node('motion.nodes.WaitNode')
                    if new_node:
                        pos = node.pos()
                        new_node.set_pos(pos[0] + offset_x, pos[1] + offset_y)
                        new_node.set_color(*NODE_WAIT_PANEL_BG_COLOR)
                        orig_name = getattr(node, "wait_name", node.name())
                        new_node.wait_name = self._next_unique_numbered_name(orig_name, taken_names)
                        new_node.set_name(new_node.wait_name)
                        new_node.frames = getattr(node, "frames", 0)
                        new_node.duration = getattr(node, "duration", 0.0)
                        new_node.out_port_labels = list(getattr(node, "out_port_labels", ["default"]))
                        new_node.out_port_priorities = list(getattr(node, "out_port_priorities", [0]))
                        print(f"[Duplicate] Created copy of WaitNode: {new_node.name()}")
                except Exception as e:
                    print(f"[Duplicate] Error duplicating WaitNode: {e}")
                    traceback.print_exc()
            elif isinstance(node, BranchingNode):
                try:
                    new_node = self._graph.create_node("motion.nodes.BranchingNode")
                    if new_node:
                        pos = node.pos()
                        new_node.set_pos(pos[0] + offset_x, pos[1] + offset_y)
                        new_node.set_color(*NODE_BRANCH_PANEL_BG_COLOR)
                        new_node.branching_enabled = getattr(node, "branching_enabled", False)
                        new_node.branch_outputs_swapped = getattr(node, "branch_outputs_swapped", False)
                        new_node.branch_if_left = getattr(node, "branch_if_left", "UserVal_0")
                        new_node.branch_if_op = getattr(node, "branch_if_op", "==")
                        new_node.branch_if_right = getattr(node, "branch_if_right", "UserVal_1")
                        new_node.branch_if_uv_enabled = getattr(node, "branch_if_uv_enabled", True)
                        new_node.branch_if_formula_enabled = getattr(node, "branch_if_formula_enabled", False)
                        new_node.branch_if_formula = getattr(node, "branch_if_formula", "Form1:foo")
                        new_node.branch_if_pad_enabled = getattr(node, "branch_if_pad_enabled", False)
                        new_node.branch_if_pad_button = getattr(node, "branch_if_pad_button", "L1")
                        new_node.branch_if_pad_analog_enabled = getattr(node, "branch_if_pad_analog_enabled", False)
                        new_node.branch_if_pad_analog_axis = getattr(node, "branch_if_pad_analog_axis", "Lx")
                        new_node.branch_if_pad_analog_op = getattr(node, "branch_if_pad_analog_op", ">=")
                        new_node.branch_if_pad_analog_threshold = getattr(node, "branch_if_pad_analog_threshold", 0)
                        # 出力ポート追加前に高さをロック（拡大防止）。ロード時と同じ手順。
                        if new_node.branching_enabled:
                            new_node._lock_output_row_height()
                        while new_node.output_count < len(node.out_port_labels):
                            label_idx = new_node.output_count
                            new_node._add_branch_output(
                                node.out_port_labels[label_idx],
                                node.out_port_priorities[label_idx]
                                if label_idx < len(node.out_port_priorities) else 0,
                            )
                        new_node.out_port_labels = list(node.out_port_labels)
                        new_node.out_port_priorities = list(node.out_port_priorities)
                        new_node._sync_branching_port_labels()
                        new_node._apply_branch_output_colors()
                        new_node.set_name(self._next_unique_numbered_name(
                            node.name(), taken_names))
                        print(f"[Duplicate] Created copy of BranchingNode: {new_node.name()}")
                except Exception as e:
                    print(f"[Duplicate] Error duplicating BranchingNode: {e}")
                    traceback.print_exc()
            elif isinstance(node, JumpNode):
                try:
                    new_node = self._graph.create_node("motion.nodes.JumpNode")
                    if new_node:
                        pos = node.pos()
                        new_node.set_pos(pos[0] + offset_x, pos[1] + offset_y)
                        new_node.set_color(*NODE_JUMP_PANEL_BG_COLOR)
                        new_node.jump_target_action_index = int(getattr(node, "jump_target_action_index", 0))
                        new_node.jump_type = getattr(node, "jump_type", "action")
                        new_node.jump_target_function = getattr(node, "jump_target_function", "")
                        new_node.set_name("Jump to")
                        while new_node.output_count < len(node.out_port_labels):
                            li = new_node.output_count
                            new_node._add_jump_output(
                                node.out_port_labels[li],
                                node.out_port_priorities[li] if li < len(node.out_port_priorities) else 0,
                            )
                        new_node.out_port_labels = list(node.out_port_labels)
                        new_node.out_port_priorities = list(node.out_port_priorities)
                        new_node.refresh_body_text()
                        QtCore.QTimer.singleShot(15, new_node._apply_jump_node_colors)
                        print(f"[Duplicate] Created copy of JumpNode -> Action_{new_node.jump_target_action_index + 1}")
                except Exception as e:
                    print(f"[Duplicate] Error duplicating JumpNode: {e}")
                    traceback.print_exc()
            elif isinstance(node, MemoNode):
                try:
                    new_node = self._graph.create_node('motion.nodes.MemoNode')
                    if new_node:
                        pos = node.pos()
                        new_node.set_pos(pos[0] + offset_x, pos[1] + offset_y)
                        new_node.set_name("Memo")
                        new_node.memo_text = getattr(node, "memo_text", "") or ""
                        new_node.modal_width = int(getattr(node, "modal_width", 420) or 420)
                        new_node.modal_height = int(getattr(node, "modal_height", 320) or 320)
                        new_node.refresh_body_text()
                        print(f"[Duplicate] Created copy of MemoNode")
                except Exception as e:
                    print(f"[Duplicate] Error duplicating MemoNode: {e}")
                    traceback.print_exc()
            elif isinstance(node, FooNode):
                try:
                    new_node = self._graph.create_node('insilico.nodes.FooNode')
                    if new_node:
                        pos = node.pos()
                        new_node.set_pos(pos[0] + offset_x, pos[1] + offset_y)
                        new_node.set_name(self._next_unique_numbered_name(
                            node.name(), taken_names))
                        new_node.mass_value = node.mass_value
                        new_node.volume_value = getattr(node, 'volume_value', 0.0)
                        new_node.inertia = dict(node.inertia)
                        new_node.points = [dict(p) for p in node.points]
                        new_node.cumulative_coords = [dict(c) for c in node.cumulative_coords]
                        new_node.node_color = list(node.node_color)
                        new_node.rotation_axis = node.rotation_axis
                        new_node.stl_file = node.stl_file
                        print(f"[Duplicate] Created copy of FooNode: {new_node.name()}")
                except Exception as e:
                    print(f"[Duplicate] Error duplicating FooNode: {e}")
                    traceback.print_exc()
            else:
                print(f"[Duplicate] Unsupported node type: {type(node).__name__}")

            if new_node:
                node_mapping[id(node)] = new_node
                new_nodes.append(new_node)

        # Recreate connections between duplicated nodes
        for old_node in nodes:
            new_from_node = node_mapping.get(id(old_node))
            if not new_from_node:
                continue
            old_out_ports = old_node.output_ports()
            new_out_ports = new_from_node.output_ports()
            for port_idx, old_port in enumerate(old_out_ports):
                if port_idx >= len(new_out_ports):
                    break
                new_port = new_out_ports[port_idx]
                for connected_port in old_port.connected_ports():
                    target_node = connected_port.node()
                    # Only connect if target is also in the duplicated set
                    if id(target_node) in old_node_ids:
                        new_target = node_mapping.get(id(target_node))
                        if new_target:
                            target_in_ports = new_target.input_ports()
                            if target_in_ports:
                                try:
                                    new_port.connect_to(target_in_ports[0])
                                    print(f"[Duplicate] Connected {new_from_node.name()} -> {new_target.name()}")
                                except Exception as conn_e:
                                    print(f"[Duplicate] Connection error: {conn_e}")

        # Select the new nodes
        if new_nodes:
            self._graph.clear_selection()
            for n in new_nodes:
                n.set_selected(True)

        return new_nodes

    def _copy_nodes(self, nodes):
        """選択されたノードをクリップボードにコピー"""
        if not nodes:
            return

        # Store node data and connections.
        # BaseLinkNode (StartNode) は clipboard_data / connections の対象外なので、
        # インデックスマップも「BaseLinkNode を除いたリスト」で作る。
        # 以前は enumerate(nodes) で作っており、StartNode を選択に含めるだけで
        # 以降の Pose の index が +1 ずれて、ペースト時に接続が壊れていた。
        copyable_nodes = [n for n in nodes if not isinstance(n, BaseLinkNode)]
        clipboard_data = []
        old_node_ids = set(id(n) for n in copyable_nodes)
        node_id_to_index = {id(n): i for i, n in enumerate(copyable_nodes)}

        for node in copyable_nodes:

            node_data = {
                'type': type(node).__name__,
                'pos': node.pos(),
                'name': node.name(),
            }

            if isinstance(node, PoseNode):
                node_data.update({
                    'pose_name': node.pose_name,
                    'duration': node.duration,
                    'frames': getattr(node, 'frames', get_default_hz_fps()),
                    'angles_deg': dict(node.angles_deg),
                    'joint_easings': dict(getattr(node, 'joint_easings', {})),
                    'branching_enabled': getattr(node, 'branching_enabled', False),
                    'branch_outputs_swapped': getattr(node, 'branch_outputs_swapped', False),
                    'branch_if_left': getattr(node, 'branch_if_left', "UserVal_0"),
                    'branch_if_op': getattr(node, 'branch_if_op', "=="),
                    'branch_if_right': getattr(node, 'branch_if_right', "UserVal_1"),
                    'branch_if_uv_enabled': getattr(node, 'branch_if_uv_enabled', True),
                    'branch_if_formula_enabled': getattr(node, 'branch_if_formula_enabled', False),
                    'branch_if_formula': getattr(node, 'branch_if_formula', "Form1:foo"),
                    'branch_if_pad_enabled': getattr(node, 'branch_if_pad_enabled', False),
                    'branch_if_pad_button': getattr(node, 'branch_if_pad_button', "L1"),
                    'branch_if_pad_analog_enabled': getattr(node, 'branch_if_pad_analog_enabled', False),
                    'branch_if_pad_analog_axis': getattr(node, 'branch_if_pad_analog_axis', "Lx"),
                    'branch_if_pad_analog_op': getattr(node, 'branch_if_pad_analog_op', ">="),
                    'branch_if_pad_analog_threshold': int(getattr(node, 'branch_if_pad_analog_threshold', 0)),
                    'out_port_labels': list(node.out_port_labels),
                    'out_port_priorities': list(node.out_port_priorities),
                })
            elif isinstance(node, DefineNode):
                node_data.update({
                    'define_uv_index': getattr(node, "define_uv_index", 0),
                    'define_memo': getattr(node, "define_memo", ""),
                    'define_kind': getattr(node, "define_kind", "literal"),
                    'define_literal': getattr(node, "define_literal", 0),
                    'define_register_name': getattr(node, "define_register_name", ""),
                })
            elif isinstance(node, BranchingNode):
                node_data.update({
                    'branching_enabled': getattr(node, 'branching_enabled', False),
                    'branch_outputs_swapped': getattr(node, 'branch_outputs_swapped', False),
                    'branch_if_left': getattr(node, 'branch_if_left', "UserVal_0"),
                    'branch_if_op': getattr(node, 'branch_if_op', "=="),
                    'branch_if_right': getattr(node, 'branch_if_right', "UserVal_1"),
                    'branch_if_uv_enabled': getattr(node, 'branch_if_uv_enabled', True),
                    'branch_if_formula_enabled': getattr(node, 'branch_if_formula_enabled', False),
                    'branch_if_formula': getattr(node, 'branch_if_formula', "Form1:foo"),
                    'branch_if_pad_enabled': getattr(node, 'branch_if_pad_enabled', False),
                    'branch_if_pad_button': getattr(node, 'branch_if_pad_button', "L1"),
                    'branch_if_pad_analog_enabled': getattr(node, 'branch_if_pad_analog_enabled', False),
                    'branch_if_pad_analog_axis': getattr(node, 'branch_if_pad_analog_axis', "Lx"),
                    'branch_if_pad_analog_op': getattr(node, 'branch_if_pad_analog_op', ">="),
                    'branch_if_pad_analog_threshold': int(getattr(node, 'branch_if_pad_analog_threshold', 0)),
                    'out_port_labels': list(node.out_port_labels),
                    'out_port_priorities': list(node.out_port_priorities),
                })
            elif isinstance(node, JumpNode):
                node_data.update({
                    'jump_target_action_index': getattr(node, "jump_target_action_index", 0),
                    'jump_type': getattr(node, "jump_type", "action"),
                    'jump_target_function': getattr(node, "jump_target_function", ""),
                    'out_port_labels': list(node.out_port_labels),
                    'out_port_priorities': list(node.out_port_priorities),
                })
            elif isinstance(node, MemoNode):
                node_data.update({
                    'memo_text': getattr(node, "memo_text", "") or "",
                    'modal_width': int(getattr(node, "modal_width", 420) or 420),
                    'modal_height': int(getattr(node, "modal_height", 320) or 320),
                })

            clipboard_data.append(node_data)

        # Store connections (as indices into clipboard_data)
        connections = []
        for node in copyable_nodes:
            from_idx = node_id_to_index.get(id(node))
            if from_idx is None:
                continue
            for port_idx, port in enumerate(node.output_ports()):
                for connected_port in port.connected_ports():
                    target_node = connected_port.node()
                    if id(target_node) in old_node_ids:
                        to_idx = node_id_to_index.get(id(target_node))
                        if to_idx is not None:
                            connections.append((from_idx, port_idx, to_idx))

        CustomViewer._clipboard = clipboard_data
        CustomViewer._clipboard_connections = connections
        print(f"[Copy] Copied {len(clipboard_data)} node(s) with {len(connections)} connection(s)")

    def _paste_nodes(self):
        """クリップボードからノードをペースト"""
        if not CustomViewer._clipboard:
            print("[Paste] Clipboard is empty")
            return

        clipboard_data = CustomViewer._clipboard
        connections = CustomViewer._clipboard_connections or []

        # Calculate offset from original position
        offset_x = 50
        offset_y = 50
        taken_names = self._collect_graph_name_set()

        # Create nodes
        new_nodes = []
        for data in clipboard_data:
            new_node = None
            node_type = data['type']
            pos = data['pos']

            try:
                if node_type == 'PoseNode':
                    new_node = self._graph.create_node('motion.nodes.PoseNode')
                    if new_node:
                        new_node.set_pos(pos[0] + offset_x, pos[1] + offset_y)
                        new_node.set_color(*NODE_COLOR_DEFAULT)
                        new_node.pose_name = self._next_unique_numbered_name(
                            data.get('pose_name', 'pose'), taken_names)
                        new_node.set_name(new_node.pose_name)
                        new_node.duration = data.get('duration', 0.0)
                        new_node.frames = data.get('frames', get_default_hz_fps())
                        new_node.angles_deg = dict(data.get('angles_deg', {}))
                        new_node.joint_easings = dict(data.get('joint_easings', {}))
                        new_node.branching_enabled = data.get('branching_enabled', False)
                        new_node.branch_outputs_swapped = data.get('branch_outputs_swapped', False)
                        new_node.branch_if_left = data.get('branch_if_left', "UserVal_0")
                        new_node.branch_if_op = data.get('branch_if_op', "==")
                        new_node.branch_if_right = data.get('branch_if_right', "UserVal_1")
                        new_node.branch_if_uv_enabled = data.get('branch_if_uv_enabled', True)
                        new_node.branch_if_formula_enabled = data.get('branch_if_formula_enabled', False)
                        new_node.branch_if_formula = data.get('branch_if_formula', "Form1:foo")
                        new_node.branch_if_pad_enabled = data.get('branch_if_pad_enabled', False)
                        new_node.branch_if_pad_button = data.get('branch_if_pad_button', "L1")
                        new_node.branch_if_pad_analog_enabled = data.get('branch_if_pad_analog_enabled', False)
                        new_node.branch_if_pad_analog_axis = data.get('branch_if_pad_analog_axis', "Lx")
                        new_node.branch_if_pad_analog_op = data.get('branch_if_pad_analog_op', ">=")
                        new_node.branch_if_pad_analog_threshold = int(data.get('branch_if_pad_analog_threshold', 0))
                        out_labels = data.get('out_port_labels', ['default'])
                        out_priorities = data.get('out_port_priorities', [0])
                        # 出力ポート追加前に高さをロック（拡大防止）。
                        if new_node.branching_enabled:
                            new_node._lock_output_row_height()
                        while new_node.output_count < len(out_labels):
                            li = new_node.output_count
                            new_node._add_pose_output(
                                out_labels[li],
                                out_priorities[li] if li < len(out_priorities) else 0
                            )
                        new_node.out_port_labels = list(out_labels)
                        new_node.out_port_priorities = list(out_priorities)
                        new_node._sync_branching_port_labels()
                        new_node._apply_pose_output_colors()

                elif node_type == 'DefineNode':
                    new_node = self._graph.create_node('motion.nodes.DefineNode')
                    if new_node:
                        new_node.set_pos(pos[0] + offset_x, pos[1] + offset_y)
                        new_node.set_color(*NODE_DEFINE_PANEL_BG_COLOR)
                        new_node.define_uv_index = data.get('define_uv_index', 0)
                        _memo = data.get('define_memo', '') or ''
                        new_node.define_memo = (
                            self._next_unique_numbered_name(_memo, taken_names)
                            if _memo else '')
                        new_node.define_kind = data.get('define_kind', 'literal')
                        new_node.define_literal = data.get('define_literal', 0)
                        new_node.define_register_name = data.get('define_register_name', '')
                        new_node.set_name(self._next_unique_numbered_name(
                            data.get('name', 'define'), taken_names))

                elif node_type == 'BranchingNode':
                    new_node = self._graph.create_node('motion.nodes.BranchingNode')
                    if new_node:
                        new_node.set_pos(pos[0] + offset_x, pos[1] + offset_y)
                        new_node.set_color(*NODE_BRANCH_PANEL_BG_COLOR)
                        new_node.branching_enabled = data.get('branching_enabled', False)
                        new_node.branch_outputs_swapped = data.get('branch_outputs_swapped', False)
                        new_node.branch_if_left = data.get('branch_if_left', "UserVal_0")
                        new_node.branch_if_op = data.get('branch_if_op', "==")
                        new_node.branch_if_right = data.get('branch_if_right', "UserVal_1")
                        new_node.branch_if_uv_enabled = data.get('branch_if_uv_enabled', True)
                        new_node.branch_if_formula_enabled = data.get('branch_if_formula_enabled', False)
                        new_node.branch_if_formula = data.get('branch_if_formula', "Form1:foo")
                        new_node.branch_if_pad_enabled = data.get('branch_if_pad_enabled', False)
                        new_node.branch_if_pad_button = data.get('branch_if_pad_button', "L1")
                        new_node.branch_if_pad_analog_enabled = data.get('branch_if_pad_analog_enabled', False)
                        new_node.branch_if_pad_analog_axis = data.get('branch_if_pad_analog_axis', "Lx")
                        new_node.branch_if_pad_analog_op = data.get('branch_if_pad_analog_op', ">=")
                        new_node.branch_if_pad_analog_threshold = int(data.get('branch_if_pad_analog_threshold', 0))
                        out_labels = data.get('out_port_labels', ['default'])
                        out_priorities = data.get('out_port_priorities', [0])
                        # 出力ポート追加前に高さをロック（拡大防止）。
                        if new_node.branching_enabled:
                            new_node._lock_output_row_height()
                        while new_node.output_count < len(out_labels):
                            li = new_node.output_count
                            new_node._add_branch_output(
                                out_labels[li],
                                out_priorities[li] if li < len(out_priorities) else 0
                            )
                        new_node.out_port_labels = list(out_labels)
                        new_node.out_port_priorities = list(out_priorities)
                        new_node._sync_branching_port_labels()
                        new_node._apply_branch_output_colors()
                        new_node.set_name(self._next_unique_numbered_name(
                            data.get('name', 'branch'), taken_names))

                elif node_type == 'JumpNode':
                    new_node = self._graph.create_node('motion.nodes.JumpNode')
                    if new_node:
                        new_node.set_pos(pos[0] + offset_x, pos[1] + offset_y)
                        new_node.set_color(*NODE_JUMP_PANEL_BG_COLOR)
                        new_node.jump_target_action_index = data.get('jump_target_action_index', 0)
                        new_node.jump_type = data.get('jump_type', 'action')
                        new_node.jump_target_function = data.get('jump_target_function', '')
                        new_node.set_name("Jump to")
                        out_labels = data.get('out_port_labels', ['default'])
                        out_priorities = data.get('out_port_priorities', [0])
                        while new_node.output_count < len(out_labels):
                            li = new_node.output_count
                            new_node._add_jump_output(
                                out_labels[li],
                                out_priorities[li] if li < len(out_priorities) else 0
                            )
                        new_node.out_port_labels = list(out_labels)
                        new_node.out_port_priorities = list(out_priorities)
                        new_node.refresh_body_text()
                        QtCore.QTimer.singleShot(15, new_node._apply_jump_node_colors)

                elif node_type == 'MemoNode':
                    new_node = self._graph.create_node('motion.nodes.MemoNode')
                    if new_node:
                        new_node.set_pos(pos[0] + offset_x, pos[1] + offset_y)
                        new_node.set_name("Memo")
                        new_node.memo_text = data.get('memo_text', '') or ''
                        new_node.modal_width = int(data.get('modal_width', 420) or 420)
                        new_node.modal_height = int(data.get('modal_height', 320) or 320)
                        new_node.refresh_body_text()

            except Exception as e:
                print(f"[Paste] Error creating {node_type}: {e}")
                traceback.print_exc()

            new_nodes.append(new_node)

        # Recreate connections
        for from_idx, port_idx, to_idx in connections:
            if from_idx < len(new_nodes) and to_idx < len(new_nodes):
                from_node = new_nodes[from_idx]
                to_node = new_nodes[to_idx]
                if from_node and to_node:
                    out_ports = from_node.output_ports()
                    in_ports = to_node.input_ports()
                    if port_idx < len(out_ports) and in_ports:
                        try:
                            out_ports[port_idx].connect_to(in_ports[0])
                            print(f"[Paste] Connected {from_node.name()} -> {to_node.name()}")
                        except Exception as conn_e:
                            print(f"[Paste] Connection error: {conn_e}")

        # Select the new nodes
        valid_nodes = [n for n in new_nodes if n is not None]
        if valid_nodes:
            self._graph.clear_selection()
            for n in valid_nodes:
                n.set_selected(True)

        print(f"[Paste] Pasted {len(valid_nodes)} node(s)")

    def _cut_nodes(self, nodes):
        """選択されたノードをカット（コピーして削除）"""
        if not nodes:
            return

        # First copy the nodes
        self._copy_nodes(nodes)

        # Then delete them (except BaseLinkNode)
        deleted_count = 0
        for node in nodes:
            if isinstance(node, BaseLinkNode):
                print("[Cut] Cannot cut BaseLinkNode")
                continue
            try:
                self._graph.remove_node(node)
                deleted_count += 1
            except Exception as e:
                print(f"[Cut] Error deleting node: {e}")

        print(f"[Cut] Cut {deleted_count} node(s)")

    def copy_selected_nodes(self):
        """外部から呼び出し可能なコピー関数"""
        if self._graph:
            selected_nodes = self._graph.selected_nodes()
            if selected_nodes:
                self._copy_nodes(selected_nodes)
                return True
        return False

    def cut_selected_nodes(self):
        """外部から呼び出し可能なカット関数"""
        if self._graph:
            selected_nodes = self._graph.selected_nodes()
            if selected_nodes:
                self._cut_nodes(selected_nodes)
                return True
        return False

    def paste_nodes(self):
        """外部から呼び出し可能なペースト関数"""
        if self._graph:
            self._paste_nodes()
            return True
        return False

    def duplicate_selected_nodes(self):
        """外部から呼び出し可能な複製関数"""
        if self._graph:
            selected_nodes = self._graph.selected_nodes()
            if selected_nodes:
                self._duplicate_nodes(selected_nodes)
                return True
        return False


class BaseLinkNode(BaseNode):
    """Base link node class"""
    __identifier__ = 'insilico.nodes'
    NODE_NAME = 'BaseLinkNode'

    # カスタムビューを使用
    __view__ = CustomNodeItem

    def __init__(self):
        super(BaseLinkNode, self).__init__(CustomNodeItem)
        self.add_output('')  # 空文字列でラベル非表示

        self.volume_value = 0.0  # 追加
        self.mass_value = 0.0

        self.inertia = {
            'ixx': 0.0, 'ixy': 0.0, 'ixz': 0.0,
            'iyy': 0.0, 'iyz': 0.0, 'izz': 0.0
        }
        self.points = [{
            'name': 'base_link_point1',
            'type': 'fixed',
            'xyz': [0.0, 0.0, 0.0]
        }]
        self.cumulative_coords = [{
            'point_index': 0,
            'xyz': [0.0, 0.0, 0.0]
        }]

        self.stl_file = None

        # 色情報を追加
        self.node_color = [c / 255.0 for c in NODE_COLOR_DEFAULT]

        # 出力ポートを下部中央に配置
        self._position_output_port_center()
        # スタートノード用の色を適用（ビュー初期化後に実行）
        QtCore.QTimer.singleShot(20, self._apply_node_colors)

        self.pose_name = "start"
        self.duration = 0.0
        self.frames = 1
        self.joint_easings = {}
        self.out_port_labels = ["default"]
        self.out_port_priorities = [0]
        self.output_count = 1

        # ダブルクリックで Pose インスペクターは開かない（既定のタイトル操作のみ）
        self._original_double_click = self.view.mouseDoubleClickEvent
        self.view.mouseDoubleClickEvent = self._on_double_click

    def _on_double_click(self, event):
        if getattr(self, "_original_double_click", None):
            self._original_double_click(event)

    def _apply_node_colors(self):
        """スタートノードの色設定を適用"""
        v = getattr(self, 'view', None)
        if not v:
            return
        # タイトル文字色（通常時 + ハイライト時）
        if hasattr(v, 'set_title_color'):
            v.set_title_color(*NODE_START_TITLE_COLOR, highlight_color=NODE_START_TITLE_HIGHLIGHT_COLOR)
        # タイトル背景色
        if hasattr(v, '_title_bg_color'):
            v._title_bg_color = QtGui.QColor(*NODE_START_TITLE_BG_COLOR)
        # パネル背景色
        self.set_color(*NODE_START_PANEL_BG_COLOR)
        # ハイライト時/通常時の色をviewに保存
        if hasattr(v, 'set_normal_colors'):
            v.set_normal_colors(
                panel_bg=NODE_START_PANEL_BG_COLOR,
                title_bg=NODE_START_TITLE_BG_COLOR,
                input_port=NODE_START_INPUT_PORT_COLOR,
                input_port_border=NODE_START_INPUT_PORT_BORDER_COLOR,
                output_port=NODE_START_OUTPUT_PORT_COLOR,
                output_port_border=NODE_START_OUTPUT_PORT_BORDER_COLOR
            )
        if hasattr(v, 'set_highlight_colors'):
            v.set_highlight_colors(
                panel_bg=NODE_START_PANEL_BG_HIGHLIGHT_COLOR,
                title_bg=NODE_START_TITLE_BG_HIGHLIGHT_COLOR,
                input_port=NODE_START_INPUT_PORT_HIGHLIGHT_COLOR,
                input_port_border=NODE_START_INPUT_PORT_HIGHLIGHT_BORDER_COLOR,
                output_port=NODE_START_OUTPUT_PORT_HIGHLIGHT_COLOR,
                output_port_border=NODE_START_OUTPUT_PORT_HIGHLIGHT_BORDER_COLOR
            )
        # 出力ポート色
        for port in self.output_ports():
            port.color = NODE_START_OUTPUT_PORT_COLOR
            port.border_color = NODE_START_OUTPUT_PORT_BORDER_COLOR
        # 入力ポート色
        for port in self.input_ports():
            port.color = NODE_START_INPUT_PORT_COLOR
            port.border_color = NODE_START_INPUT_PORT_BORDER_COLOR

    def _position_output_port_center(self):
        """出力ポートを下部中央に配置"""
        try:
            # ノードビューが完全に初期化された後に実行
            QtCore.QTimer.singleShot(10, self._do_position_output_port)
        except Exception as e:
            print(f"Error scheduling port positioning: {e}")

    def _do_position_output_port(self):
        """実際にポートを配置"""
        try:
            if not self.view:
                print("[BaseLinkNode] View not available yet")
                return

            outputs = [p for p in self.view.outputs if p.isVisible()]
            if not outputs:
                print("[BaseLinkNode] No visible output ports")
                return

            port = outputs[0]
            node_width = self.view._width
            port_width = port.boundingRect().width()
            port_height = port.boundingRect().height()

            # 中央のX座標を計算
            port_x = (node_width - port_width) / 2
            # 下部のY座標を計算
            port_y = self.view._height - port_height - 5

            port.setPos(port_x, port_y)
            port.update()

            # テキストラベルを非表示に
            if port in self.view._output_items:
                self.view._output_items[port].setVisible(False)

        except Exception as e:
            print(f"[BaseLinkNode] Error positioning port: {e}")

        # 回転軸の初期値を追加
        self.rotation_axis = 0  # 0: X, 1: Y, 2: Z

        # ポート位置はCustomNodeItemの_align_ports_horizontalで自動設定されます

    def add_input(self, name='', **kwargs):
        # 入力ポートの追加を禁止
        print("Base Link node cannot have input ports")
        return None

    def add_output(self, name='out_1', **kwargs):
        # 出力ポートが既に存在する場合は追加しない
        if not self.has_output(name):
            return super(BaseLinkNode, self).add_output(name, **kwargs)
        return None

    def remove_output(self, port=None):
        # 出力ポートの削除を禁止
        print("Cannot remove output port from Base Link node")
        return None

    def has_output(self, name):
        """指定した名前の出力ポートが存在するかチェック"""
        return name in [p.name() for p in self.output_ports()]

    def _position_output_ports(self):
        """出力ポートを下部中央に配置"""
        if not hasattr(self, 'view'):
            return

        node_rect = self.view.boundingRect()
        node_width = node_rect.width()
        node_height = node_rect.height()
        output_ports = self.view.outputs

        if output_ports:
            center_x = node_width / 2
            bottom_y = node_height
            for port in output_ports:
                port_width = port.boundingRect().width()
                port_height = port.boundingRect().height()
                target_x = center_x - port_width / 2
                target_y = bottom_y - port_height - 5
                port.setPos(target_x, target_y)
                if hasattr(port, '_text'):
                    port._text.setVisible(False)

class FooNode(BaseNode):
    """General purpose node class"""
    __identifier__ = 'insilico.nodes'
    NODE_NAME = 'FooNode'

    # カスタムビューを使用
    __view__ = CustomNodeItem

    def __init__(self):
        super(FooNode, self).__init__(CustomNodeItem)
        # 入力ポートを追加するが非表示にする（空文字列でラベル非表示、複数接続許可）
        input_port = self.add_input('', color=NODE_BASIC_INPUT_PORT_COLOR, multi_input=True)
        # ポートを非表示にする
        if input_port:
            try:
                # ビューレベルでポートを非表示に
                self.view.inputs[0].setVisible(False)
                # テキストラベルも非表示に
                if hasattr(self.view.inputs[0], '_text'):
                    self.view.inputs[0]._text.setVisible(False)
            except:
                pass

        self.output_count = 0
        self.volume_value = 0.0  # 追加
        self.mass_value = 0.0

        # データ属性を初期化
        self.mass_value = 0.0
        self.inertia = {
            'ixx': 0.0, 'ixy': 0.0, 'ixz': 0.0,
            'iyy': 0.0, 'iyz': 0.0, 'izz': 0.0
        }
        self.points = []
        self.cumulative_coords = []
        self.stl_file = None

        # 色情報を追加
        self.node_color = [1.0, 1.0, 1.0]  # RGBの初期値（白）

        # 回転軸の初期値を追加
        self.rotation_axis = 0  # 0: X, 1: Y, 2: Z

        # 基本ノード用の色を適用（ビュー初期化後に実行）
        QtCore.QTimer.singleShot(20, self._apply_node_colors)

        # ポートタイプを追跡するリスト（'normal' or 'branch'）
        self.port_types = []

        # 出力ポートを追加
        self._add_output()

        self.set_port_deletion_allowed(True)
        self._original_double_click = self.view.mouseDoubleClickEvent
        self.view.mouseDoubleClickEvent = self.node_double_clicked

        # ポート位置はCustomNodeItemの_align_ports_horizontalで自動設定されます

    def _apply_node_colors(self):
        """基本ノードの色設定を適用"""
        v = getattr(self, 'view', None)
        if not v:
            return
        # タイトル文字色（通常時 + ハイライト時）
        if hasattr(v, 'set_title_color'):
            v.set_title_color(*NODE_BASIC_TITLE_COLOR, highlight_color=NODE_BASIC_TITLE_HIGHLIGHT_COLOR)
        # タイトル背景色
        if hasattr(v, '_title_bg_color'):
            v._title_bg_color = QtGui.QColor(*NODE_BASIC_TITLE_BG_COLOR)
        # パネル背景色
        self.set_color(*NODE_BASIC_PANEL_BG_COLOR)
        # ハイライト時/通常時の色をviewに保存
        if hasattr(v, 'set_normal_colors'):
            v.set_normal_colors(
                panel_bg=NODE_BASIC_PANEL_BG_COLOR,
                input_port=NODE_BASIC_INPUT_PORT_COLOR,
                input_port_border=NODE_BASIC_INPUT_PORT_BORDER_COLOR,
                output_port=NODE_BASIC_OUTPUT_PORT_COLOR,
                output_port_border=NODE_BASIC_OUTPUT_PORT_BORDER_COLOR
            )
        if hasattr(v, 'set_highlight_colors'):
            v.set_highlight_colors(
                panel_bg=NODE_BASIC_PANEL_BG_HIGHLIGHT_COLOR,
                input_port=NODE_BASIC_INPUT_PORT_HIGHLIGHT_COLOR,
                input_port_border=NODE_BASIC_INPUT_PORT_HIGHLIGHT_BORDER_COLOR,
                output_port=NODE_BASIC_OUTPUT_PORT_HIGHLIGHT_COLOR,
                output_port_border=NODE_BASIC_OUTPUT_PORT_HIGHLIGHT_BORDER_COLOR
            )
        # 出力ポート色
        for port in self.output_ports():
            port.color = NODE_BASIC_OUTPUT_PORT_COLOR
            port.border_color = NODE_BASIC_OUTPUT_PORT_BORDER_COLOR
        # 入力ポート色
        for port in self.input_ports():
            port.color = NODE_BASIC_INPUT_PORT_COLOR
            port.border_color = NODE_BASIC_INPUT_PORT_BORDER_COLOR

    def _add_output(self, _name=''):
        if self.output_count < 8:  # 最大8ポートまで
            self.output_count += 1
            port_name = f'out_{self.output_count}'
            # 一意のポート名を使用し、ラベルは非表示
            super(FooNode, self).add_output(port_name, display_name=False)

            # ポートタイプを追跡
            if not hasattr(self, 'port_types'):
                self.port_types = []
            self.port_types.append('normal')

            # ポイントデータの初期化
            if not hasattr(self, 'points'):
                self.points = []

            # 新しいポイントを[0,0,0]の座標で追加
            self.points.append({
                'name': f'point_{self.output_count}',
                'type': 'fixed',
                'xyz': [0.0, 0.0, 0.0]
            })

            # 累積座標の初期化
            if not hasattr(self, 'cumulative_coords'):
                self.cumulative_coords = []

            self.cumulative_coords.append({
                'point_index': self.output_count - 1,
                'xyz': [0.0, 0.0, 0.0]
            })

            # 出力ポートを下部中央に配置
            self._position_output_ports_center()

            print(f"Added output port '{port_name}' with zero coordinates")
            return port_name

    def _add_branch_output(self, _name=''):
        """赤色のブランチポイント（出力ポート）を追加"""
        if self.output_count < 8:  # 最大8ポートまで
            self.output_count += 1
            port_name = f'branch_{self.output_count}'

            # 赤色のポートを追加
            # RGBを0-255から0-1の範囲に変換
            red_color = (BRANCH_POINT_COLOR[0] / 255.0,
                        BRANCH_POINT_COLOR[1] / 255.0,
                        BRANCH_POINT_COLOR[2] / 255.0)
            # 一意のポート名を使用し、ラベルは非表示
            super(FooNode, self).add_output(port_name, display_name=False, color=red_color)

            # ポートタイプを追跡
            if not hasattr(self, 'port_types'):
                self.port_types = []
            self.port_types.append('branch')

            print(f"Added branch output port '{port_name}' with color: {red_color}")
            print(f"Current port_types: {self.port_types}")
            print(f"Total output count: {self.output_count}")

            # ポイントデータの初期化
            if not hasattr(self, 'points'):
                self.points = []

            # 新しいブランチポイントを[0,0,0]の座標で追加
            self.points.append({
                'name': f'branch_point_{self.output_count}',
                'type': 'branch',  # タイプをbranchに設定
                'xyz': [0.0, 0.0, 0.0]
            })

            # 累積座標の初期化
            if not hasattr(self, 'cumulative_coords'):
                self.cumulative_coords = []

            self.cumulative_coords.append({
                'point_index': self.output_count - 1,
                'xyz': [0.0, 0.0, 0.0]
            })

            # 出力ポートを配置
            self._position_output_ports_center()

            print(f"Added branch point '{port_name}' with red color")
            return port_name

    def _position_output_ports_center(self):
        """全ての出力ポートを下部中央に配置"""
        try:
            # ノードビューが完全に初期化された後に実行
            QtCore.QTimer.singleShot(10, self._do_position_output_ports)
        except Exception as e:
            print(f"Error scheduling port positioning: {e}")

    def _do_position_output_ports(self):
        """実際にポートを配置"""
        try:
            if not self.view:
                print("[FooNode] View not available yet")
                return

            # 全ての出力ポート（可視・不可視問わず）を取得
            outputs = self.view.outputs
            if not outputs:
                print("[FooNode] No output ports")
                return

            print(f"[FooNode] Total outputs: {len(outputs)}, port_types: {getattr(self, 'port_types', [])}")

            node_width = self.view._width

            # ポートタイプが初期化されていない場合は初期化
            if not hasattr(self, 'port_types'):
                self.port_types = ['normal'] * len(outputs)

            for i, port in enumerate(outputs):
                # ポートを可視化
                port.setVisible(True)

                port_width = port.boundingRect().width()
                port_height = port.boundingRect().height()

                # 中央のX座標を計算
                port_x = (node_width - port_width) / 2

                # ブランチポイントの場合は10px右にオフセット
                if i < len(self.port_types) and self.port_types[i] == 'branch':
                    port_x += 10
                    print(f"[FooNode] Branch port {i} ({port.name}): offset +10px to x={port_x}")
                else:
                    print(f"[FooNode] Normal port {i} ({port.name if hasattr(port, 'name') else 'unnamed'}): x={port_x}")

                # 下部のY座標を計算
                port_y = self.view._height - port_height - 5

                port.setPos(port_x, port_y)
                port.update()

                # テキストラベルを非表示に
                if port in self.view._output_items:
                    self.view._output_items[port].setVisible(False)

        except Exception as e:
            print(f"[FooNode] Error positioning port: {e}")
            import traceback
            traceback.print_exc()

    def remove_output(self):
        """出力ポートの削除（修正版）"""
        if self.output_count > 1:
            port_name = f'out_{self.output_count}'
            output_port = self.get_output(port_name)
            if output_port:
                try:
                    # 接続されているポートを処理
                    for connected_port in output_port.connected_ports():
                        try:
                            print(f"Disconnecting {port_name} from {connected_port.node().name()}.{connected_port.name()}")
                            # NodeGraphQtの標準メソッドを使用
                            self.graph.disconnect_node(self.id, port_name,
                                                     connected_port.node().id, connected_port.name())
                        except Exception as e:
                            print(f"Error during disconnection: {str(e)}")

                    # 対応するポイントデータを削除
                    if len(self.points) >= self.output_count:
                        self.points.pop()
                        print(f"Removed point data for port {port_name}")

                    # 累積座標を削除
                    if len(self.cumulative_coords) >= self.output_count:
                        self.cumulative_coords.pop()
                        print(f"Removed cumulative coordinates for port {port_name}")

                    # ポートの削除
                    self.delete_output(output_port)
                    self.output_count -= 1
                    print(f"Removed port {port_name}")

                    # ビューの更新
                    self.view.update()
                    
                except Exception as e:
                    print(f"Error removing port and associated data: {str(e)}")
                    traceback.print_exc()
            else:
                print(f"Output port {port_name} not found")
        else:
            print("Cannot remove the last output port")

    def _position_output_ports(self):
        """出力ポートを下部中央に配置"""
        if not hasattr(self, 'view'):
            return

        node_rect = self.view.boundingRect()
        node_width = node_rect.width()
        node_height = node_rect.height()
        output_ports = self.view.outputs

        if output_ports:
            center_x = node_width / 2
            bottom_y = node_height
            for port in output_ports:
                port_width = port.boundingRect().width()
                port_height = port.boundingRect().height()
                target_x = center_x - port_width / 2
                target_y = bottom_y - port_height - 5
                port.setPos(target_x, target_y)
                if hasattr(port, '_text'):
                    port._text.setVisible(False)

    def node_double_clicked(self, event):
        if hasattr(self.graph, 'show_inspector'):
            try:
                graph_view = self.graph.viewer()
                scene_pos = event.scenePos()
                view_pos = graph_view.mapFromScene(scene_pos)
                screen_pos = graph_view.mapToGlobal(view_pos)
                self.graph.show_inspector(self, screen_pos)
            except Exception as e:
                print(f"Error getting mouse position: {str(e)}")
                traceback.print_exc()
                self.graph.show_inspector(self)
        else:
            print("Error: graph does not have show_inspector method")

class InspectorWindow(QtWidgets.QWidget):
    
    def __init__(self, parent=None, stl_viewer=None):
        super(InspectorWindow, self).__init__(parent)
        self.setWindowTitle("Node Inspector")
        self.setMinimumWidth(NODE_INSPECTOR_MIN_WIDTH)
        self.setMinimumHeight(600)

        self.setWindowFlags(self.windowFlags() |
                            QtCore.Qt.WindowStaysOnTopHint)
        self.setAttribute(QtCore.Qt.WA_DeleteOnClose)

        self.current_node = None
        self.stl_viewer = stl_viewer
        self.port_widgets = []

        # UIの初期化
        self.setup_ui()

        # キーボードフォーカスを受け取れるように設定
        self.setFocusPolicy(QtCore.Qt.StrongFocus)

    def setup_ui(self):
        """UIの初期化"""
        main_layout = QtWidgets.QVBoxLayout(self)
        main_layout.setSpacing(10)  # 全体の余白を小さく
        main_layout.setContentsMargins(10, 5, 10, 5)  # 上下の余白も調整

        # スクロールエリアの設定
        scroll_area = QtWidgets.QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        scroll_area.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)

        # スクロールの中身となるウィジェット
        scroll_content = QtWidgets.QWidget()
        content_layout = QtWidgets.QVBoxLayout(scroll_content)
        content_layout.setSpacing(30)  # セクション間の間隔を小さく
        content_layout.setContentsMargins(5, 5, 5, 5)  # 余白を小さく

        # Node Name セクション（横一列）
        name_layout = QtWidgets.QHBoxLayout()
        name_layout.addWidget(create_label("Node Name:"))
        self.name_edit = QtWidgets.QLineEdit()
        self.name_edit.setPlaceholderText("Enter node name")
        self.name_edit.editingFinished.connect(self.update_node_name)
        name_layout.addWidget(self.name_edit)

        content_layout.addLayout(name_layout)
        content_layout.addSpacing(5)  # 追加の間隔

        # Physical Properties セクション（テキストを削除して詰める）
        physics_layout = QtWidgets.QGridLayout()
        physics_layout.addWidget(create_label("Volume:"), 0, 0)
        self.volume_input = QtWidgets.QLineEdit()
        self.volume_input.setReadOnly(True)
        physics_layout.addWidget(self.volume_input, 0, 1)

        physics_layout.addWidget(create_label("Mass:"), 1, 0)
        self.mass_input = QtWidgets.QLineEdit()
        self.mass_input.setValidator(QtGui.QDoubleValidator())
        physics_layout.addWidget(self.mass_input, 1, 1)
        content_layout.addLayout(physics_layout)

        # Rotation Axis セクション（横一列）
        rotation_layout = QtWidgets.QHBoxLayout()
        rotation_layout.addWidget(create_label("Rotation Axis:   "))
        self.axis_group = QtWidgets.QButtonGroup(self)
        for i, axis in enumerate(['X (Roll)', 'Y (Pitch)', 'Z (Yaw)', 'Fixed']):  # Fixedを追加
            radio = QtWidgets.QRadioButton(axis)
            self.axis_group.addButton(radio, i)  # iは0,1,2,3となる（3がFixed）
            rotation_layout.addWidget(radio)
        content_layout.addLayout(rotation_layout)

        # Color セクション
        color_layout = QtWidgets.QHBoxLayout()
        color_layout.addWidget(create_label("Color:"))

        # カラーサンプルチップ
        self.color_sample = QtWidgets.QLabel()
        self.color_sample.setFixedSize(20, 20)
        self.color_sample.setStyleSheet(
        f"background-color: rgb({COLOR_SAMPLE_DEFAULT[0]},{COLOR_SAMPLE_DEFAULT[1]},{COLOR_SAMPLE_DEFAULT[2]}); border: 1px solid black;")
        color_layout.addWidget(self.color_sample)

        # R,G,B入力
        color_layout.addWidget(create_label("   R:"))
        self.color_inputs = []
        for label in ['', 'G:', 'B:']:  # Rは既に追加したので空文字
            if label:  # G:とB:のみラベルを追加
                color_layout.addWidget(create_label(label))
            color_input = QtWidgets.QLineEdit("1.0")
            color_input.setFixedWidth(50)
            color_input.setValidator(QtGui.QDoubleValidator(0.0, 1.0, 3))
            self.color_inputs.append(color_input)
            color_layout.addWidget(color_input)

        # Applyボタン
        apply_button = QtWidgets.QPushButton("Set")
        apply_button.clicked.connect(self.apply_color_to_stl)
        apply_button.setFixedWidth(40)
        color_layout.addWidget(apply_button)
        color_layout.addStretch()
        content_layout.addLayout(color_layout)

        # Output Ports セクション
        ports_layout = QtWidgets.QVBoxLayout()
        self.ports_layout = QtWidgets.QVBoxLayout()  # 動的に追加されるポートのための親レイアウト
        ports_layout.addLayout(self.ports_layout)

        # SETボタンレイアウト
        set_button_layout = QtWidgets.QHBoxLayout()
        set_button_layout.addStretch()
        set_button = QtWidgets.QPushButton("SET")
        set_button.clicked.connect(self.apply_port_values)
        set_button_layout.addWidget(set_button)
        ports_layout.addLayout(set_button_layout)
        content_layout.addLayout(ports_layout)

        # ポートウィジェットを格納するリストを初期化
        self.port_widgets = []

        # Point Controls セクション（横一列にする）
        point_layout = QtWidgets.QHBoxLayout()
        point_layout.addWidget(create_label("Point Controls:"))
        self.add_point_btn = QtWidgets.QPushButton("[+] Add")
        self.remove_point_btn = QtWidgets.QPushButton("[-] Remove")
        point_layout.addWidget(self.add_point_btn)
        point_layout.addWidget(self.remove_point_btn)
        self.add_point_btn.clicked.connect(self.add_point)
        self.remove_point_btn.clicked.connect(self.remove_point)
        content_layout.addLayout(point_layout)

        # Branch Point Controls セクション
        branch_layout = QtWidgets.QHBoxLayout()
        branch_layout.addWidget(create_label("Branch Point:"))
        self.add_branch_point_btn = QtWidgets.QPushButton("Add Branch Point")
        branch_layout.addWidget(self.add_branch_point_btn)
        self.add_branch_point_btn.clicked.connect(self.add_branch_point)
        content_layout.addLayout(branch_layout)

        # Massless Decoration チェックボックス
        massless_layout = QtWidgets.QHBoxLayout()
        self.massless_checkbox = QtWidgets.QCheckBox("Massless Decoration")
        self.massless_checkbox.setChecked(False)
        massless_layout.addWidget(self.massless_checkbox)
        massless_layout.addStretch()
        content_layout.addLayout(massless_layout)

        # スクロールエリアにコンテンツをセット
        scroll_area.setWidget(scroll_content)
        main_layout.addWidget(scroll_area)

        # 既存のレイアウトにも spacing を設定
        name_layout.setSpacing(2)
        physics_layout.setSpacing(2)
        rotation_layout.setSpacing(2)
        color_layout.setSpacing(2)
        ports_layout.setSpacing(2)
        point_layout.setSpacing(2)

        # 既存のグリッドレイアウトの余白調整
        physics_layout.setVerticalSpacing(2)
        physics_layout.setHorizontalSpacing(2)

        for line_edit in self.findChildren(QtWidgets.QLineEdit):
            line_edit.setStyleSheet("QLineEdit { padding-left: 2px; padding-top: 0px; padding-bottom: 0px; }")

    def setup_validators(self):
        """数値入力フィールドにバリデータを設定"""
        try:
            # Mass入力フィールド用のバリデータ
            mass_validator = QtGui.QDoubleValidator()
            mass_validator.setBottom(0.0)  # 負の値を禁止
            self.mass_input.setValidator(mass_validator)

            # Volume入力フィールド用のバリデータ
            volume_validator = QtGui.QDoubleValidator()
            volume_validator.setBottom(0.0)  # 負の値を禁止
            self.volume_input.setValidator(volume_validator)

            # RGB入力フィールド用のバリデータ
            rgb_validator = QtGui.QDoubleValidator(
                0.0, 1.0, 3)  # 0.0から1.0まで、小数点以下3桁
            for color_input in self.color_inputs:
                color_input.setValidator(rgb_validator)

            # Output Ports用のバリデータ
            coord_validator = QtGui.QDoubleValidator()
            for port_widget in self.port_widgets:
                for input_field in port_widget.findChildren(QtWidgets.QLineEdit):
                    input_field.setValidator(coord_validator)

            print("Input validators setup completed")

        except Exception as e:
            print(f"Error setting up validators: {str(e)}")
            import traceback
            traceback.print_exc()

    def apply_color_to_stl(self):
        """選択された色をSTLモデルとカラーサンプルに適用"""
        if not self.current_node:
            print("No node selected")
            return
        
        try:
            # RGB値の取得（0-1の範囲）
            rgb_values = [float(input.text()) for input in self.color_inputs]
            
            # 値の範囲チェック
            rgb_values = [max(0.0, min(1.0, value)) for value in rgb_values]
            
            # ノードの色情報を更新
            self.current_node.node_color = rgb_values
            
            # カラーサンプルチップを必ず更新
            rgb_display = [int(v * 255) for v in rgb_values]
            self.color_sample.setStyleSheet(
                f"background-color: rgb({rgb_display[0]},{rgb_display[1]},{rgb_display[2]}); "
                f"border: 1px solid black;"
            )
            
            # STLモデルの色を変更
            if self.stl_viewer and hasattr(self.stl_viewer, 'stl_actors'):
                if self.current_node in self.stl_viewer.stl_actors:
                    actor = self.stl_viewer.stl_actors[self.current_node]
                    actor.GetProperty().SetColor(*rgb_values)
                    self.stl_viewer.safe_render()
                    print(f"Applied color: RGB({rgb_values[0]:.3f}, {rgb_values[1]:.3f}, {rgb_values[2]:.3f})")
                else:
                    print("No STL model found for this node")
            
        except ValueError as e:
            print(f"Error: Invalid color value - {str(e)}")
        except Exception as e:
            print(f"Error applying color: {str(e)}")
            traceback.print_exc()

    def update_color_sample(self):
        """カラーサンプルの表示を更新"""
        try:
            rgb_values = [min(255, max(0, int(float(input.text()) * 255))) 
                        for input in self.color_inputs]
            self.color_sample.setStyleSheet(
                f"background-color: rgb({rgb_values[0]},{rgb_values[1]},{rgb_values[2]}); "
                f"border: 1px solid black;"
            )
            
            if self.current_node:
                self.current_node.node_color = [float(input.text()) for input in self.color_inputs]
                
        except ValueError as e:
            print(f"Error updating color sample: {str(e)}")
            traceback.print_exc()

    def update_port_coordinate(self, port_index, coord_index, value):
        """ポート座標の更新"""
        try:
            if self.current_node and hasattr(self.current_node, 'points'):
                if 0 <= port_index < len(self.current_node.points):
                    try:
                        new_value = float(value)
                        self.current_node.points[port_index]['xyz'][coord_index] = new_value
                        print(
                            f"Updated port {port_index+1} coordinate {coord_index} to {new_value}")
                    except ValueError:
                        print("Invalid coordinate value")
        except Exception as e:
            print(f"Error updating coordinate: {str(e)}")

    def update_info(self, node):
        """ノード情報の更新"""
        self.current_node = node

        try:
            # Node Name
            self.name_edit.setText(node.name())

            # Volume & Mass
            if hasattr(node, 'volume_value'):
                self.volume_input.setText(f"{node.volume_value:.6f}")
                print(f"Volume set to: {node.volume_value}")

            if hasattr(node, 'mass_value'):
                self.mass_input.setText(f"{node.mass_value:.6f}")
                print(f"Mass set to: {node.mass_value}")

            # Rotation Axis - nodeのrotation_axis属性を確認して設定
            if hasattr(node, 'rotation_axis'):
                axis_button = self.axis_group.button(node.rotation_axis)
                if axis_button:
                    axis_button.setChecked(True)
                    print(f"Rotation axis set to: {node.rotation_axis}")
            else:
                # デフォルトでX軸を選択
                node.rotation_axis = 0
                if self.axis_group.button(0):
                    self.axis_group.button(0).setChecked(True)
                    print("Default rotation axis set to X (0)")

            # Massless Decoration の状態を設定
            if hasattr(node, 'massless_decoration'):
                self.massless_checkbox.setChecked(node.massless_decoration)
                print(f"Massless decoration set to: {node.massless_decoration}")
            else:
                node.massless_decoration = False
                self.massless_checkbox.setChecked(False)
                print("Default massless decoration set to False")

            # Color settings - nodeのnode_color属性を確認して設定
            if hasattr(node, 'node_color') and node.node_color:
                print(f"Setting color: {node.node_color}")
                for i, value in enumerate(node.node_color[:3]):
                    self.color_inputs[i].setText(f"{value:.3f}")
                
                # カラーサンプルチップの更新
                rgb_display = [int(v * 255) for v in node.node_color[:3]]
                self.color_sample.setStyleSheet(
                    f"background-color: rgb({rgb_display[0]},{rgb_display[1]},{rgb_display[2]}); "
                    f"border: 1px solid black;"
                )
                # STLモデルにも色を適用
                self.apply_color_to_stl()
            else:
                # デフォルトの色を設定（白）
                node.node_color = [1.0, 1.0, 1.0]
                for color_input in self.color_inputs:
                    color_input.setText("1.000")
                self.color_sample.setStyleSheet(
                    f"background-color: rgb({COLOR_SAMPLE_DEFAULT[0]},{COLOR_SAMPLE_DEFAULT[1]},{COLOR_SAMPLE_DEFAULT[2]}); border: 1px solid black;"
                )
                print("Default color set to white")

            # 回転軸の選択を更新するためのシグナルを接続
            for button in self.axis_group.buttons():
                button.clicked.connect(lambda checked, btn=button: self.update_rotation_axis(btn))

            # Output Ports
            self.update_output_ports(node)

            # ラジオボタンのイベントハンドラを設定
            self.axis_group.buttonClicked.connect(self.on_axis_selection_changed)

            # バリデータの設定
            self.setup_validators()

            print(f"Inspector window updated for node: {node.name()}")

        except Exception as e:
            print(f"Error updating inspector info: {str(e)}")
            traceback.print_exc()

    def update_rotation_axis(self, button):
        """回転軸の選択が変更されたときの処理"""
        if self.current_node:
            self.current_node.rotation_axis = self.axis_group.id(button)
            print(f"Updated rotation axis to: {self.current_node.rotation_axis}")

    def on_axis_selection_changed(self, button):
        """回転軸の選択が変更されたときのイベントハンドラ"""
        if self.current_node:
            # 現在のノードの変換情報を保存
            if self.stl_viewer and self.current_node in self.stl_viewer.transforms:
                current_transform = self.stl_viewer.transforms[self.current_node]
                current_position = current_transform.GetPosition()
            else:
                current_position = [0, 0, 0]

            # 回転軸の更新
            axis_id = self.axis_group.id(button)
            self.current_node.rotation_axis = axis_id

            # 軸のタイプを判定して表示
            axis_types = ['X (Roll)', 'Y (Pitch)', 'Z (Yaw)', 'Fixed']
            if 0 <= axis_id < len(axis_types):
                print(f"Rotation axis changed to: {axis_types[axis_id]}")
            else:
                print(f"Invalid rotation axis ID: {axis_id}")

            # STLモデルの更新
            if self.stl_viewer:
                # 変換の更新
                if self.current_node in self.stl_viewer.transforms:
                    transform = self.stl_viewer.transforms[self.current_node]
                    transform.Identity()  # 変換をリセット
                    transform.Translate(*current_position)  # 元の位置を維持
                    
                    # 回転軸に基づいて現在の角度を設定（必要な場合）
                    if hasattr(self.current_node, 'current_rotation'):
                        angle = self.current_node.current_rotation
                        if axis_id == 0:      # X軸
                            transform.RotateX(angle)
                        elif axis_id == 1:    # Y軸
                            transform.RotateY(angle)
                        elif axis_id == 2:    # Z軸
                            transform.RotateZ(angle)
                    
                    # 変換を適用
                    if self.current_node in self.stl_viewer.stl_actors:
                        self.stl_viewer.stl_actors[self.current_node].SetUserTransform(transform)
                        self.stl_viewer.safe_render()
                        print(f"Updated transform for node {self.current_node.name()} at position {current_position}")
                        
    def update_node_name(self):
        """ノード名の更新"""
        if self.current_node:
            new_name = self.name_edit.text()
            old_name = self.current_node.name()
            if new_name != old_name:
                self.current_node.set_name(new_name)
                print(f"Node name updated from '{old_name}' to '{new_name}'")

    def add_point(self):
        """ポイントの追加"""
        if self.current_node and hasattr(self.current_node, '_add_output'):
            new_port_name = self.current_node._add_output()
            if new_port_name:
                self.update_info(self.current_node)
                print(f"Added new port: {new_port_name}")

    def add_branch_point(self):
        """ブランチポイントの追加（赤色のポート）"""
        if self.current_node and hasattr(self.current_node, '_add_branch_output'):
            new_port_name = self.current_node._add_branch_output()
            if new_port_name:
                self.update_info(self.current_node)
                print(f"Added new branch point: {new_port_name}")

    def remove_point(self):
        """ポイントの削除"""
        if self.current_node and hasattr(self.current_node, 'remove_output'):
            self.current_node.remove_output()
            self.update_info(self.current_node)
            print("Removed last port")

    def closeEvent(self, event):
        """ウィンドウが閉じられるときのイベントを処理"""
        try:
            # 全てのウィジェットを明示的に削除
            for widget in self.findChildren(QtWidgets.QWidget):
                if widget is not self:
                    widget.setParent(None)
                    widget.deleteLater()

            # 参照のクリア
            self.current_node = None
            self.stl_viewer = None
            self.port_widgets.clear()

            # イベントを受け入れ
            event.accept()

        except Exception as e:
            print(f"Error in closeEvent: {str(e)}")
            event.accept()

    def apply_port_values(self):
        """Output Portsの値を適用"""
        if not self.current_node:
            print("No node selected")
            return

        try:
            # ポートウィジェットから値を取得して適用
            for i, port_widget in enumerate(self.port_widgets):
                # ポートの座標入力フィールドを検索
                coord_inputs = []
                for child in port_widget.findChildren(QtWidgets.QLineEdit):
                    coord_inputs.append(child)

                # 座標入力フィールドが3つ（X,Y,Z）あることを確認
                if len(coord_inputs) >= 3:
                    try:
                        # 座標値を取得
                        x = float(coord_inputs[0].text())
                        y = float(coord_inputs[1].text())
                        z = float(coord_inputs[2].text())

                        # ノードのポイントデータを更新
                        if hasattr(self.current_node, 'points') and i < len(self.current_node.points):
                            self.current_node.points[i]['xyz'] = [x, y, z]
                            print(
                                f"Updated point {i+1} coordinates to: ({x:.6f}, {y:.6f}, {z:.6f})")

                            # 累積座標も更新
                            if hasattr(self.current_node, 'cumulative_coords') and i < len(self.current_node.cumulative_coords):
                                if isinstance(self.current_node, BaseLinkNode):
                                    self.current_node.cumulative_coords[i]['xyz'] = [
                                        x, y, z]
                                else:
                                    # base_link以外のノードの場合は相対座標を保持
                                    self.current_node.cumulative_coords[i]['xyz'] = [
                                        0.0, 0.0, 0.0]

                    except ValueError:
                        print(f"Invalid numerical input for point {i+1}")
                        continue

            # ノードの位置を再計算（必要な場合）
            if hasattr(self.current_node, 'graph') and self.current_node.graph:
                self.current_node.graph.recalculate_all_positions()
                print("Node positions recalculated")

            # STLビューアの更新
            if self.stl_viewer:
                self.stl_viewer.safe_render()
                print("3D view updated")

        except Exception as e:
            print(f"Error applying port values: {str(e)}")
            import traceback
            traceback.print_exc()

    def create_port_widget(self, port_number, x=0.0, y=0.0, z=0.0):
        """Output Port用のウィジェットを作成"""
        port_layout = QtWidgets.QHBoxLayout()  # GridLayoutからHBoxLayoutに変更
        port_layout.setSpacing(5)
        port_layout.setContentsMargins(0, 1, 0, 1)

        # ポート番号
        port_name = create_label(f"out_{port_number}")
        port_name.setFixedWidth(45)
        port_layout.addWidget(port_name)

        # 座標入力のペアを作成
        coords = []
        for label, value in [('X:', x), ('Y:', y), ('Z:', z)]:
            # 各座標のペアをHBoxLayoutで作成
            coord_pair = QtWidgets.QHBoxLayout()
            coord_pair.setSpacing(2)
            
            # ラベル
            coord_label = create_label(label)
            coord_label.setFixedWidth(15)
            coord_pair.addWidget(coord_label)

            # 入力フィールド
            coord_input = QtWidgets.QLineEdit(f"{value:.6f}")
            coord_input.setFixedWidth(70)
            coord_input.setFixedHeight(20)
            coord_input.setStyleSheet("QLineEdit { padding-left: 2px; padding-top: 0px; padding-bottom: 0px; }")
            coord_input.setValidator(QtGui.QDoubleValidator())
            coord_input.textChanged.connect(
                lambda text, idx=port_number-1, coord=len(coords):
                self.update_port_coordinate(idx, coord, text))
            coord_pair.addWidget(coord_input)
            coords.append(coord_input)

            # ペアをメインレイアウトに追加
            port_layout.addLayout(coord_pair)
            
            # ペア間にスペースを追加
            if label != 'Z:':  # 最後のペア以外の後にスペースを追加
                port_layout.addSpacing(15)

        # 右側の余白
        port_layout.addStretch()

        # ウィジェットをラップ
        port_widget = QtWidgets.QWidget()
        port_widget.setFixedHeight(25)
        port_widget.setLayout(port_layout)
        return port_widget, coords

    def update_output_ports(self, node):
        """Output Portsセクションを更新"""
        # 既存のポートウィジェットをクリア
        for widget in self.port_widgets:
            self.ports_layout.removeWidget(widget)
            widget.setParent(None)
            widget.deleteLater()
        self.port_widgets.clear()

        # ノードの各ポートに対してウィジェットを作成
        if hasattr(node, 'points'):
            for i, point in enumerate(node.points):
                port_widget, _ = self.create_port_widget(
                    i + 1,
                    point['xyz'][0],
                    point['xyz'][1],
                    point['xyz'][2]
                )
                self.ports_layout.addWidget(port_widget)
                self.port_widgets.append(port_widget)

    def apply_color_to_stl(self):
        """選択された色をSTLモデルに適用"""
        if not self.current_node:
            return
        
        try:
            rgb_values = [float(input.text()) for input in self.color_inputs]
            rgb_values = [max(0.0, min(1.0, value)) for value in rgb_values]
            
            self.current_node.node_color = rgb_values
            
            if self.stl_viewer and hasattr(self.stl_viewer, 'stl_actors'):
                if self.current_node in self.stl_viewer.stl_actors:
                    actor = self.stl_viewer.stl_actors[self.current_node]
                    actor.GetProperty().SetColor(*rgb_values)
                    self.stl_viewer.safe_render()
        except ValueError as e:
            print(f"Error: Invalid color value - {str(e)}")

    def update_color_sample(self):
        """カラーサンプルの表示を更新"""
        try:
            rgb_values = [min(255, max(0, int(float(input.text()) * 255))) 
                        for input in self.color_inputs]
            self.color_sample.setStyleSheet(
                f"background-color: rgb({rgb_values[0]},{rgb_values[1]},{rgb_values[2]}); "
                f"border: 1px solid black;"
            )
        except ValueError:
            pass

    def moveEvent(self, event):
        """ウィンドウ移動イベントの処理"""
        super(InspectorWindow, self).moveEvent(event)
        # グラフオブジェクトが存在し、last_inspector_positionを保存可能な場合
        if hasattr(self, 'graph') and self.graph:
            self.graph.last_inspector_position = self.pos()

    def keyPressEvent(self, event):
        """キープレスイベントの処理"""
        # ESCキーが押されたかどうかを確認
        if event.key() == QtCore.Qt.Key.Key_Escape:
            self.close()
        else:
            # 他のキーイベントは通常通り処理
            super(InspectorWindow, self).keyPressEvent(event)

    def _calculate_base_inertia_tensor(self, poly_data, mass, center_of_mass, is_mirrored=False):
        """
        基本的な慣性テンソル計算のための共通実装。
        InspectorWindowクラスのメソッド。

        Args:
            poly_data: vtkPolyData オブジェクト
            mass: float 質量
            center_of_mass: list[float] 重心座標 [x, y, z]
            is_mirrored: bool ミラーリングモードかどうか

        Returns:
            numpy.ndarray: 3x3 慣性テンソル行列
        """
        # 体積を計算
        mass_properties = vtk.vtkMassProperties()
        mass_properties.SetInputData(poly_data)
        mass_properties.Update()
        total_volume = mass_properties.GetVolume()

        # 実際の質量から密度を逆算
        density = mass / total_volume
        print(f"Calculated density: {density:.6f} from mass: {mass:.6f} and volume: {total_volume:.6f}")

        # 慣性テンソルの初期化
        inertia_tensor = np.zeros((3, 3))
        num_cells = poly_data.GetNumberOfCells()
        print(f"Processing {num_cells} triangles for inertia tensor calculation...")

        for i in range(num_cells):
            cell = poly_data.GetCell(i)
            if cell.GetCellType() == vtk.VTK_TRIANGLE:
                # 三角形の頂点を取得（重心を原点とした座標系で）
                points = [np.array(cell.GetPoints().GetPoint(j)) - np.array(center_of_mass) for j in range(3)]

                # ミラーリングモードの場合、Y座標を反転
                if is_mirrored:
                    points = [[p[0], -p[1], p[2]] for p in points]

                # 三角形の面積と法線ベクトルを計算
                v1 = np.array(points[1]) - np.array(points[0])
                v2 = np.array(points[2]) - np.array(points[0])
                normal = np.cross(v1, v2)
                area = 0.5 * np.linalg.norm(normal)
                
                if area < 1e-10:  # 極小の三角形は無視
                    continue

                # 三角形の重心を計算
                tri_centroid = np.mean(points, axis=0)
                
                # 三角形の局所的な慣性テンソルを計算
                covariance = np.zeros((3, 3))
                for p in points:
                    r_squared = np.sum(p * p)
                    for a in range(3):
                        for b in range(3):
                            if a == b:
                                # 対角成分
                                covariance[a, a] += (r_squared - p[a] * p[a]) * area / 12.0
                            else:
                                # 非対角成分（オフセット項）
                                covariance[a, b] -= (p[a] * p[b]) * area / 12.0

                # 平行軸の定理を適用
                r_squared = np.sum(tri_centroid * tri_centroid)
                parallel_axis_term = np.zeros((3, 3))
                for a in range(3):
                    for b in range(3):
                        if a == b:
                            parallel_axis_term[a, a] = r_squared * area
                        else:
                            parallel_axis_term[a, b] = tri_centroid[a] * tri_centroid[b] * area

                # 局所的な慣性テンソルと平行軸の項を合成
                local_inertia = covariance + parallel_axis_term
                
                # 全体の慣性テンソルに加算
                inertia_tensor += local_inertia

        # 密度を考慮して最終的な慣性テンソルを計算
        inertia_tensor *= density

        # 数値誤差の処理
        threshold = 1e-10
        inertia_tensor[np.abs(inertia_tensor) < threshold] = 0.0

        # 対称性の確認と強制
        inertia_tensor = 0.5 * (inertia_tensor + inertia_tensor.T)

        # 対角成分が正であることを確認
        for i in range(3):
            if inertia_tensor[i, i] <= 0:
                print(f"Warning: Non-positive diagonal element detected at position ({i},{i})")
                inertia_tensor[i, i] = abs(inertia_tensor[i, i])

        return inertia_tensor

    def calculate_inertia_tensor(self):
        """
        通常モデルの慣性テンソルを計算。
        InspectorWindowクラスのメソッド。
        """
        if not self.current_node or not hasattr(self.current_node, 'stl_file'):
            print("No STL model is loaded.")
            return None

        try:
            # STLデータを取得
            if self.stl_viewer and self.current_node in self.stl_viewer.stl_actors:
                actor = self.stl_viewer.stl_actors[self.current_node]
                poly_data = actor.GetMapper().GetInput()
            else:
                print("No STL actor found for current node")
                return None

            # 体積と質量を取得
            mass_properties = vtk.vtkMassProperties()
            mass_properties.SetInputData(poly_data)
            mass_properties.Update()
            volume = mass_properties.GetVolume()
            density = float(self.density_input.text())
            mass = volume * density

            # 重心を取得
            com_filter = vtk.vtkCenterOfMass()
            com_filter.SetInputData(poly_data)
            com_filter.SetUseScalarsAsWeights(False)
            com_filter.Update()
            center_of_mass = np.array(com_filter.GetCenter())

            print("\nCalculating inertia tensor for normal model...")
            print(f"Volume: {volume:.6f}, Mass: {mass:.6f}")
            print(f"Center of Mass: {center_of_mass}")

            # 慣性テンソルを計算
            inertia_tensor = self._calculate_base_inertia_tensor(
                poly_data, mass, center_of_mass, is_mirrored=False)

            # URDFフォーマットに変換してUIを更新
            urdf_inertia = self.format_inertia_for_urdf(inertia_tensor)
            if hasattr(self, 'inertia_tensor_input'):
                self.inertia_tensor_input.setText(urdf_inertia)
                print("\nInertia tensor has been updated in UI")
            else:
                print("Warning: inertia_tensor_input not found")

            return inertia_tensor

        except Exception as e:
            print(f"Error calculating inertia tensor: {str(e)}")
            traceback.print_exc()
            return None


class FlatComboButton(QtWidgets.QPushButton):
    """Flat button that mimics QComboBox API but shows a QMenu on click (no arrow, no border)."""
    currentIndexChanged = QtCore.Signal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._items = []
        self._current_index = -1
        self.setFlat(True)
        self.setStyleSheet(
            "QPushButton { text-align: center; padding: 2px 4px; border: none; "
            "background: transparent; color: inherit; }"
            "QPushButton:hover { background: rgba(255,255,255,30); border-radius: 3px; }"
        )
        self.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)
        self.clicked.connect(self._show_menu)

    def addItem(self, text):
        self._items.append(text)
        if self._current_index == -1:
            self._set_index_silent(0)

    def clear(self):
        self._items = []
        self._current_index = -1
        self.setText("")

    def count(self):
        return len(self._items)

    def currentIndex(self):
        return self._current_index

    def _set_index_silent(self, index):
        """Update internal state and button text without emitting the signal."""
        if 0 <= index < len(self._items):
            self._current_index = index
            self.setText(self._items[index])

    def setCurrentIndex(self, index):
        if 0 <= index < len(self._items):
            old = self._current_index
            self._set_index_silent(index)
            if old != index and not self.signalsBlocked():
                self.currentIndexChanged.emit(index)

    def _show_menu(self):
        menu = QtWidgets.QMenu(self)
        menu.setStyleSheet("QMenu { color: black; }")
        for i, text in enumerate(self._items):
            action = menu.addAction(text)
            action.setCheckable(True)
            action.setChecked(i == self._current_index)
            action.setData(i)
        chosen = menu.exec(self.mapToGlobal(QtCore.QPoint(0, self.height())))
        if chosen is not None:
            self.setCurrentIndex(chosen.data())


# ホイール操作時の Step 値スナップ列 (昇順)。0.1〜1 は疎に、1〜10 は 5/1、
# 10 超は 5° 刻みで 180 まで。カスタム順序を単純化するためリテラル列挙。
STEP_VALUE_SEQUENCE = (
    [0.1, 0.2, 0.5, 1.0, 5.0] +
    [float(v) for v in range(10, 181, 5)]
)


class StepValueDialog(QtWidgets.QDialog):
    """3Dビュー「Step」右クリックで開く STEP 角度エディタ。

    - 数値入力: 0.1〜180.0 の任意値を受け付ける (即時反映)
    - マウスホイール: STEP_VALUE_SEQUENCE 上を 1 ステップずつ移動
      (例: 14→wheel up→15, 14→wheel down→10, 5→wheel down→1→0.5→0.2→0.1)
    - 0 にはならない (下限 0.1)
    """

    value_changed = QtCore.Signal(float)

    def __init__(self, initial_deg: float, parent=None):
        super().__init__(parent)
        self.setWindowTitle("STEP")
        # モードレスに近い挙動: フォーカスを失ったら閉じる。
        self.setWindowFlags(
            QtCore.Qt.Popup | QtCore.Qt.FramelessWindowHint
        )
        self.setStyleSheet("""
            QDialog { background-color: #ffffff; border: 1px solid #888888; }
            QLabel { color: #222222; font-size: 11px; }
            QDoubleSpinBox { color: #111111; background-color: #f5f5f5;
                             border: 1px solid #cccccc; padding: 2px 4px;
                             font-size: 12px; min-width: 80px; }
        """)

        layout = QtWidgets.QHBoxLayout(self)
        layout.setContentsMargins(8, 6, 8, 6)
        layout.setSpacing(6)

        layout.addWidget(QtWidgets.QLabel("STEP:"))

        self._spin = _StepSpinBox()
        self._spin.setRange(0.1, 180.0)
        self._spin.setDecimals(1)
        self._spin.setSingleStep(0.1)
        self._spin.setValue(max(0.1, min(180.0, float(initial_deg))))
        self._spin.valueChanged.connect(self._on_spin_changed)
        self._spin.setFocus()
        self._spin.selectAll()
        layout.addWidget(self._spin)

        layout.addWidget(QtWidgets.QLabel("deg"))

    def _on_spin_changed(self, v):
        self.value_changed.emit(float(v))

    def keyPressEvent(self, event):
        if event.key() in (QtCore.Qt.Key_Escape, QtCore.Qt.Key_Return,
                           QtCore.Qt.Key_Enter):
            self.close()
            return
        super().keyPressEvent(event)


class _StepSpinBox(QtWidgets.QDoubleSpinBox):
    """StepValueDialog 内で使う、ホイール時のみ STEP_VALUE_SEQUENCE を歩く SpinBox。"""

    def wheelEvent(self, event):
        delta = event.angleDelta().y()
        if delta == 0:
            event.accept()
            return
        cur = float(self.value())
        seq = STEP_VALUE_SEQUENCE
        if delta > 0:
            # 昇順で cur より真に大きい最初の要素
            nxt = next((s for s in seq if s > cur + 1e-9), seq[-1])
        else:
            # 降順で cur より真に小さい最初の要素
            nxt = next((s for s in reversed(seq) if s < cur - 1e-9), seq[0])
        self.setValue(nxt)
        event.accept()


class STLViewerWidget(QtWidgets.QWidget):
    # 関節ドラッグ終了時のシグナル
    joint_drag_ended = QtCore.Signal()
    # ダイアログからの角度変更シグナル (joint_name, angle_deg)
    joint_angle_changed = QtCore.Signal(str, float)
    # ホイール操作中: ノッチ毎に FK 空間の全関節角度を通知（Live 送信/スライダー反映用）
    wheel_angles_updated = QtCore.Signal(dict)
    # Valkey チェックボックス切替シグナル
    valkey_toggled = QtCore.Signal(bool)
    # Live チェックボックス切替シグナル（JointEditor 側と同期する）
    live_toggled = QtCore.Signal(bool)
    # reset_camera() 完了時: (focal_x, focal_y, focal_z, distance)
    camera_fitted = QtCore.Signal(float, float, float, float)

    def __init__(self, parent=None):
        self.joint_editor = None  # set after both widgets are created
        super(STLViewerWidget, self).__init__(parent)
        # ブレンダー風の 0-9 カメラショートカットを受け取るため
        # ウィジェット自身もフォーカスを受け取れるようにする
        self.setFocusPolicy(QtCore.Qt.StrongFocus)
        self.stl_actors = {}
        self.transforms = {}
        self.base_connected_node = None
        self.text_actors = []
        self._initialized = True
        # Guards VTK actor transform data (vtkTransform/vtkMatrix4x4 — plain CPU-side
        # objects, not GL calls) against being read by Render() on this (GUI) thread
        # while being mutated by the background computed-motion IK worker thread (see
        # main()). Unlike an earlier attempt at threading the *render* itself, the GL
        # context always stays on this thread here — only this CPU-side data is
        # shared, so no MakeCurrent() dance is needed.
        self._vtk_lock = threading.Lock()

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)

        # Use QLabel for offscreen rendering (macOS compatibility)
        self.vtk_display = QtWidgets.QLabel(self)
        self.vtk_display.setMinimumSize(VTK_DISPLAY_MIN_WIDTH, VTK_DISPLAY_MIN_HEIGHT)
        self.vtk_display.setStyleSheet(f"background-color: {VTK_BACKGROUND_COLOR};")
        self.vtk_display.setAlignment(QtCore.Qt.AlignCenter)
        self.vtk_display.setScaledContents(False)
        self.vtk_display.setMouseTracking(True)
        self.vtk_display.setFocusPolicy(QtCore.Qt.StrongFocus)
        # サイズポリシーを設定して伸縮可能にする
        self.vtk_display.setSizePolicy(
            QtWidgets.QSizePolicy.Expanding,
            QtWidgets.QSizePolicy.Expanding
        )
        layout.addWidget(self.vtk_display, stretch=1)

        # Setup offscreen VTK rendering
        self.render_window = vtk.vtkRenderWindow()
        self.render_window.SetOffScreenRendering(1)
        self.render_window.SetSize(800, 600)

        self.renderer = vtk.vtkRenderer()
        self.render_window.AddRenderer(self.renderer)

        # Initialize offscreen renderer
        self.offscreen_renderer = OffscreenRenderer(
            self.render_window, self.renderer, render_lock=self._vtk_lock)

        # Mouse interaction state
        self.last_mouse_pos = None
        self.vtk_display.installEventFilter(self)

        # ブレンダー風カメラ操作: 0 キーで復元するカスタム視点
        # マウスドラッグ/ズームでカメラが動くたびに自動スナップショットする
        self._saved_camera_state = None
        # ドラッグ中にカメラが動いたことを検知するフラグ
        self._camera_dirty_during_drag = False

        # メッシュ選択機能
        self.picker = vtk.vtkCellPicker()
        self.picker.SetTolerance(0.005)
        self.selected_actor = None
        self.selected_link_name = None
        self.original_color = None
        self.robot_model = None  # URDFRobotModelへの参照
        self.is_dragging_joint = False
        self.drag_start_pos = None
        self.drag_start_angle = 0.0
        self.drag_joint_name_pair = None
        self.drag_start_angle_pair = 0.0

        # ホイール回転セッション状態: マウスドラッグと同等の
        # Pair / LinkGroup / step_snap / Valkey Live / Undo を実現する
        self._wheel_active = False
        self._wheel_joint_name = None
        self._wheel_start_angle = 0.0
        self._wheel_pair_name = None
        self._wheel_start_angle_pair = 0.0
        self._wheel_group_preset_idx = -1
        self._wheel_group_base_angles = {}
        self._wheel_stop_timer = QtCore.QTimer(self)
        self._wheel_stop_timer.setSingleShot(True)
        self._wheel_stop_timer.setInterval(180)
        self._wheel_stop_timer.timeout.connect(self._on_wheel_stopped)

        # ハイライト点滅用タイマー
        self.highlight_timer = QtCore.QTimer()
        self.highlight_timer.timeout.connect(self._toggle_highlight)
        self.highlight_visible = True

        # ボタンのレイアウト (横並び)
        button_layout = QtWidgets.QHBoxLayout()
        button_layout.setContentsMargins(0, 0, 0, 0)
        button_layout.setSpacing(4)

        # Homeボタン
        self.home_button = QtWidgets.QPushButton("Home")
        button_layout.addWidget(self.home_button)

        # Zeroボタン
        self.zero_button = QtWidgets.QPushButton("Zero")
        button_layout.addWidget(self.zero_button)

        # L<->R swap button
        self.lr_swap_button = QtWidgets.QPushButton("L↔R")
        self.lr_swap_button.setFixedWidth(42)
        self.lr_swap_button.setToolTip("Swap left and right joint angles")
        button_layout.addWidget(self.lr_swap_button)

        # Reframeボタン
        self.reset_button = QtWidgets.QPushButton("Reframe")
        button_layout.addWidget(self.reset_button)

        # Partial Home menu (Upper / Lower body)
        self.body_home_menu_button = QtWidgets.QToolButton()
        self.body_home_menu_button.setText("\u2630")
        self.body_home_menu_button.setToolTip("UpperBody-Home / LowerBody-Home")
        self.body_home_menu_button.setPopupMode(QtWidgets.QToolButton.InstantPopup)
        self.body_home_menu_button.setFixedWidth(28)
        self.body_home_menu_button.setStyleSheet("QToolButton { color: black; }")
        body_home_menu = QtWidgets.QMenu(self.body_home_menu_button)
        body_home_menu.setStyleSheet("QMenu { color: black; background-color: white; }")
        self.body_home_upper_action = body_home_menu.addAction("UpperBody-Home")
        self.body_home_lower_action = body_home_menu.addAction("LoweBody-Home")
        self.body_home_menu_button.setMenu(body_home_menu)
        button_layout.addWidget(self.body_home_menu_button)

        button_layout.addStretch()

        # FPS label (right-aligned)
        self.fps_label = QtWidgets.QLabel("FPS: 100")
        self.fps_label.setStyleSheet("color: black;")
        button_layout.addWidget(self.fps_label)

        # Right-top overlay panel: Step / Pair / Opp / Valkey
        _step_s = load_app_settings()
        # Step 設定は JointEditor 側 (joint_sliders_step_*) を単一の真実とする。
        # 3D ビューのチェックは JointEditor と双方向同期し、deg 値は
        # joint_editor.step_snapping_deg を参照する。

        self._overlay_panel = QtWidgets.QFrame(self)
        self._overlay_panel.setObjectName("vtkOverlay")
        self._overlay_panel.setStyleSheet("""
            QFrame#vtkOverlay {
                background-color: rgba(255,255,255,220);
                border: 1px solid #bbbbbb;
                border-radius: 4px;
            }
            QLabel { background-color: transparent; color: #555555;
                     font-size: 10px; font-weight: bold; }
            QCheckBox { background-color: transparent; color: #222222;
                        font-size: 11px; }
            QCheckBox:disabled { color: #aaaaaa; }
            QComboBox { background-color: #f5f5f5; color: #222222;
                        font-size: 11px; border: 1px solid #cccccc;
                        border-radius: 3px; padding: 1px 4px; }
        """)
        _ov_layout = QtWidgets.QVBoxLayout(self._overlay_panel)
        _ov_layout.setContentsMargins(6, 4, 6, 4)
        _ov_layout.setSpacing(4)

        self.step_snap_check = QtWidgets.QCheckBox("Step")
        # 初期値は JointEditor と同じキー (joint_sliders_step_snapping) を参照。
        # main() 側で joint_editor.step_snapping_checkbox と双方向同期する。
        self.step_snap_check.setChecked(bool(_step_s.get("joint_sliders_step_snapping", False)))
        _ov_layout.addWidget(self.step_snap_check)

        self.pair_check = QtWidgets.QCheckBox("Pair")
        self.pair_check.setChecked(bool(_step_s.get("vtk_drag_pair_enabled", False)))
        self.pair_check.stateChanged.connect(self._save_step_settings)
        _ov_layout.addWidget(self.pair_check)

        self.opp_check = QtWidgets.QCheckBox("Opp")
        self.opp_check.setChecked(bool(_step_s.get("vtk_drag_opp_enabled", False)))
        self.opp_check.setEnabled(self.pair_check.isChecked())
        self.opp_check.stateChanged.connect(self._save_step_settings)
        # Pair のサブ項目であることを示すため半文字分インデント
        self.opp_check.setStyleSheet("QCheckBox { margin-left: 8px; }")
        _ov_layout.addWidget(self.opp_check)

        self.group_preset_combo = FlatComboButton()
        self.group_preset_combo.addItem("Individual")
        self.group_preset_combo.setMaximumWidth(130)
        _ov_layout.addWidget(self.group_preset_combo)

        self.valkey_check = QtWidgets.QCheckBox("Valkey")
        self.valkey_check.setChecked(bool(_step_s.get("valkey_enabled", False)))
        self.valkey_check.stateChanged.connect(self._save_step_settings)
        self.valkey_check.toggled.connect(lambda v: self.valkey_toggled.emit(v))
        _ov_layout.addWidget(self.valkey_check)

        # Live: スライダー/3Dビューでの角度変更をリアルタイムに Valkey へ送る
        # JointEditor 側の Live と同期。Valkey が OFF のときは操作不可（グレーアウト）。
        self.live_check = QtWidgets.QCheckBox("Live")
        self.live_check.setChecked(bool(_step_s.get("valkey_live_enabled", False)))
        self.live_check.setEnabled(self.valkey_check.isChecked())
        self.live_check.stateChanged.connect(self._save_step_settings)
        self.live_check.toggled.connect(lambda v: self.live_toggled.emit(v))
        # Valkey のサブ項目であることを示すため半文字分インデント
        self.live_check.setStyleSheet("QCheckBox { margin-left: 8px; }")
        _ov_layout.addWidget(self.live_check)

        # Valkey OFF になったら Live もクリアして disabled にする
        def _on_valkey_toggled_for_live(v):
            self.live_check.setEnabled(bool(v))
            if not v and self.live_check.isChecked():
                # blockSignals せずに toggle → live_toggled が発火 → JointEditor 側にも伝播
                self.live_check.setChecked(False)
        self.valkey_check.toggled.connect(_on_valkey_toggled_for_live)

        self.pair_check.toggled.connect(lambda v: self.opp_check.setEnabled(v))

        self._overlay_panel.adjustSize()
        self._overlay_panel.raise_()

        # 背景色設定 (Color A: ベース色, Color B: グラデーション用)
        self.bg_color_a = list(VTK_BG_COLOR_A)
        self.bg_color_b = list(VTK_BG_COLOR_B)
        self.bg_gradient_type = VTK_BG_GRADIENT_TYPE  # "none", "vertical", "horizontal", "radial"

        # BG/Light slider values (sliders are now in Settings dialog)
        self.bg_slider_value = VTK_BG_SLIDER_DEFAULT
        self.light_slider_value = 70  # デフォルト70%

        layout.addLayout(button_layout, stretch=0)

        self.setup_camera()
        self.coordinate_axes_actor = self.create_coordinate_axes()
        self.renderer.AddActor(self.coordinate_axes_actor)

        # ライティングの設定
        self.lights = []

        light1 = vtk.vtkLight()
        light1.SetPosition(0.5, 0.5, 1.0)
        light1.SetIntensity(0.7)
        light1.SetLightTypeToSceneLight()
        self.lights.append(light1)

        light2 = vtk.vtkLight()
        light2.SetPosition(-1.0, -0.5, 0.2)
        light2.SetIntensity(0.7)
        light2.SetLightTypeToSceneLight()
        self.lights.append(light2)

        light3 = vtk.vtkLight()
        light3.SetPosition(0.3, -1.0, 0.2)
        light3.SetIntensity(0.7)
        light3.SetLightTypeToSceneLight()
        self.lights.append(light3)

        light4 = vtk.vtkLight()
        light4.SetPosition(1.0, 0.0, 0.3)
        light4.SetIntensity(0.3)
        light4.SetLightTypeToSceneLight()
        self.lights.append(light4)

        self.renderer.SetAmbient(0.7, 0.7, 0.7)
        self.renderer.LightFollowCameraOff()
        for light in self.lights:
            self.renderer.AddLight(light)

        # 初期の背景色を設定（グラデーション対応）
        self.update_background()

    def showEvent(self, event):
        """ウィジェット表示時の処理"""
        super(STLViewerWidget, self).showEvent(event)
        # Update render window size and trigger first render
        QtCore.QTimer.singleShot(100, self.safe_render)

    def resizeEvent(self, event):
        """Handle window resize."""
        super(STLViewerWidget, self).resizeEvent(event)
        size = self.vtk_display.size()
        if size.width() > 0 and size.height() > 0:
            self.render_window.SetSize(size.width(), size.height())
            self.safe_render()
        if hasattr(self, '_overlay_panel'):
            self._reposition_overlay()

    def _reposition_overlay(self):
        """右上オーバーレイパネルを vtk_display の右上に配置"""
        MARGIN = 8
        panel = self._overlay_panel
        panel.adjustSize()
        pw = panel.width()
        ph = panel.height()
        vx = self.vtk_display.x()
        vy = self.vtk_display.y()
        vw = self.vtk_display.width()
        panel.setGeometry(vx + vw - pw - MARGIN, vy + MARGIN, pw, ph)
        panel.raise_()

    def _get_mouse_pos(self, event):
        """Get mouse position from event (Qt5/Qt6 compatible)."""
        if hasattr(event, 'position'):
            return event.position().toPoint()
        return event.pos()

    def eventFilter(self, obj, event):
        """Handle mouse events for camera control and mesh selection."""
        if obj == self.vtk_display:
            if event.type() == QtCore.QEvent.MouseButtonPress:
                mouse_pos = self._get_mouse_pos(event)
                self.last_mouse_pos = mouse_pos
                # 0-9 カメラショートカットを受け取れるようフォーカスを掴む
                self.setFocus(QtCore.Qt.MouseFocusReason)

                if event.button() == QtCore.Qt.LeftButton:
                    # Shift+左ドラッグは「中ドラッグ相当のパン」。メッシュピックも
                    # 関節ドラッグ開始もせず、以降の MouseMove でパン処理を行う。
                    if event.modifiers() & QtCore.Qt.ShiftModifier:
                        pass
                    else:
                        # 左クリック: メッシュをピック
                        actor = self.pick_actor_at(mouse_pos.x(), mouse_pos.y())
                        if actor:
                            self.select_mesh(actor)
                            # 選択したメッシュでドラッグ開始
                            self.start_joint_drag(mouse_pos)
                        else:
                            self.deselect_mesh()
                return True

            elif event.type() == QtCore.QEvent.MouseMove and self.last_mouse_pos:
                mouse_pos = self._get_mouse_pos(event)
                delta = mouse_pos - self.last_mouse_pos

                if event.buttons() & QtCore.Qt.LeftButton:
                    if self.is_dragging_joint:
                        # 関節ドラッグ中
                        self.update_joint_drag(mouse_pos)
                    elif event.modifiers() & QtCore.Qt.ShiftModifier:
                        # Shift+左ドラッグ = スクリーン平面上でのパン。
                        # カメラの向きに追従して「ドラッグ方向にモデルが動く」
                        # 感覚にするため、world XY 固定ではなく **カメラの
                        # screen_right / view_up** をベースに動かす。
                        camera = self.renderer.GetActiveCamera()
                        scale = camera.GetParallelScale()
                        pos = np.array(camera.GetPosition(), dtype=float)
                        focal = np.array(camera.GetFocalPoint(), dtype=float)
                        view_up = np.array(camera.GetViewUp(), dtype=float)
                        view_dir = focal - pos
                        _vd_n = np.linalg.norm(view_dir)
                        if _vd_n > 1e-9:
                            view_dir /= _vd_n
                        screen_right = np.cross(view_dir, view_up)
                        _sr_n = np.linalg.norm(screen_right)
                        if _sr_n > 1e-9:
                            screen_right /= _sr_n
                        _up_n = np.linalg.norm(view_up)
                        if _up_n > 1e-9:
                            view_up_n = view_up / _up_n
                        else:
                            view_up_n = view_up
                        # ドラッグ方向 (右=+x, 下=+y) と同じ方向にモデルが動く
                        # ように、カメラは逆方向にオフセットする。感度は
                        # 従来 (中ドラッグ) と同じ scale * 0.002。
                        factor = scale * 0.002
                        world_shift = (
                            -screen_right * (delta.x() * factor)
                            + view_up_n * (delta.y() * factor)
                        )
                        new_focal = focal + world_shift
                        new_pos = pos + world_shift
                        camera.SetFocalPoint(*new_focal)
                        camera.SetPosition(*new_pos)
                        self.renderer.ResetCameraClippingRange()
                        self.safe_render()
                        self._camera_dirty_during_drag = True
                    else:
                        # カメラ回転
                        camera = self.renderer.GetActiveCamera()
                        camera.Azimuth(-delta.x() * 0.5)
                        camera.Elevation(delta.y() * 0.5)
                        camera.OrthogonalizeViewUp()
                        self.renderer.ResetCameraClippingRange()
                        self.safe_render()
                        self._camera_dirty_during_drag = True
                elif event.buttons() & QtCore.Qt.RightButton:
                    # Zoom
                    camera = self.renderer.GetActiveCamera()
                    scale = camera.GetParallelScale()
                    scale *= 1.0 + delta.y() * 0.01
                    camera.SetParallelScale(max(0.01, scale))
                    self.renderer.ResetCameraClippingRange()
                    self.safe_render()
                    self._camera_dirty_during_drag = True
                elif event.buttons() & QtCore.Qt.MiddleButton:
                    # Pan
                    camera = self.renderer.GetActiveCamera()
                    scale = camera.GetParallelScale()
                    camera.SetFocalPoint(
                        camera.GetFocalPoint()[0] - delta.x() * scale * 0.002,
                        camera.GetFocalPoint()[1] + delta.y() * scale * 0.002,
                        camera.GetFocalPoint()[2]
                    )
                    camera.SetPosition(
                        camera.GetPosition()[0] - delta.x() * scale * 0.002,
                        camera.GetPosition()[1] + delta.y() * scale * 0.002,
                        camera.GetPosition()[2]
                    )
                    self.renderer.ResetCameraClippingRange()
                    self.safe_render()
                    self._camera_dirty_during_drag = True

                self.last_mouse_pos = mouse_pos
                return True

            elif event.type() == QtCore.QEvent.MouseButtonRelease:
                if event.button() == QtCore.Qt.LeftButton:
                    self.end_joint_drag()
                # ドラッグ中にカメラが動いていれば 0 キー用にスナップショット
                if self._camera_dirty_during_drag:
                    self._snapshot_camera_state()
                    self._camera_dirty_during_drag = False
                self.last_mouse_pos = None
                return True

            elif event.type() == QtCore.QEvent.MouseButtonDblClick:
                if event.button() == QtCore.Qt.LeftButton:
                    actor = self.pick_actor_at(event.pos().x(), event.pos().y())
                    if actor and self.robot_model:
                        # グローバル座標とローカル座標を取得
                        global_pos = event.globalPosition().toPoint()
                        local_pos = event.pos()
                        self._open_joint_dialog(actor, global_pos, local_pos)
                return True

            elif event.type() == QtCore.QEvent.Wheel:
                delta = event.angleDelta().y()

                # メッシュ選択中はホイールで関節回転
                if self.selected_link_name and self.robot_model:
                    joint_name, joint = self.find_joint_for_link(self.selected_link_name)
                    if joint_name:
                        self._apply_wheel_rotation(joint_name, joint, delta)
                        return True

                # 非選択時はカメラズーム
                camera = self.renderer.GetActiveCamera()
                scale = camera.GetParallelScale()
                scale *= 1.0 - delta * 0.001
                camera.SetParallelScale(max(0.01, scale))
                self.renderer.ResetCameraClippingRange()
                self.safe_render()
                # ホイールズーム後もカメラ状態をスナップショット
                self._snapshot_camera_state()
                return True

            elif event.type() == QtCore.QEvent.KeyPress:
                # ブレンダー風カメラビューショートカット（テンキー/メイン両対応）
                if self._handle_camera_view_key(event):
                    return True

        return super(STLViewerWidget, self).eventFilter(obj, event)

    # ---------------------------------------------------------------
    # Blender-style camera view shortcuts (numpad/main row 0..9)
    # ---------------------------------------------------------------
    def keyPressEvent(self, event):
        """ウィジェット自身がフォーカスを持っているときに 0-9 を受ける。
        eventFilter は QLabel(vtk_display) にフォーカスがある場合の受け皿。"""
        if self._handle_camera_view_key(event):
            event.accept()
            return
        super(STLViewerWidget, self).keyPressEvent(event)

    def _snapshot_camera_state(self):
        """カメラの現在の位置/焦点/view_up/parallel_scale をスナップショット。
        0キー押下で復元される。"""
        try:
            cam = self.renderer.GetActiveCamera()
            self._saved_camera_state = {
                "position":       tuple(cam.GetPosition()),
                "focal_point":    tuple(cam.GetFocalPoint()),
                "view_up":        tuple(cam.GetViewUp()),
                "parallel_scale": float(cam.GetParallelScale()),
            }
        except Exception as _e:
            print(f"[Camera] snapshot failed: {_e}")

    def _restore_camera_state(self):
        """スナップショットされたカメラ状態を復元。無ければ何もしない。"""
        s = self._saved_camera_state
        if not s:
            print("[Camera] No saved camera state to restore")
            return
        try:
            cam = self.renderer.GetActiveCamera()
            cam.SetPosition(*s["position"])
            cam.SetFocalPoint(*s["focal_point"])
            cam.SetViewUp(*s["view_up"])
            cam.SetParallelScale(s["parallel_scale"])
            cam.OrthogonalizeViewUp()
            self.renderer.ResetCameraClippingRange()
            self.safe_render()
        except Exception as _e:
            print(f"[Camera] restore failed: {_e}")

    def _apply_standard_view(self, azimuth_deg, elevation_deg):
        """現在の焦点点と距離を保ちながら方位/仰角のみを設定する。"""
        try:
            cam = self.renderer.GetActiveCamera()
            focal = cam.GetFocalPoint()
            pos = cam.GetPosition()
            dx = pos[0] - focal[0]
            dy = pos[1] - focal[1]
            dz = pos[2] - focal[2]
            distance = math.sqrt(dx * dx + dy * dy + dz * dz)
            if distance < 1e-6:
                distance = 1.0
            az = math.radians(azimuth_deg)
            el = math.radians(elevation_deg)
            cam.SetPosition(
                focal[0] + distance * math.cos(el) * math.cos(az),
                focal[1] + distance * math.cos(el) * math.sin(az),
                focal[2] + distance * math.sin(el),
            )
            cam.SetViewUp(0, 0, 1)
            cam.OrthogonalizeViewUp()
            self.renderer.ResetCameraClippingRange()
            self.safe_render()
        except Exception as _e:
            print(f"[Camera] standard view failed: {_e}")

    def _rotate_camera_delta(self, azimuth_delta_deg, elevation_delta_deg):
        """現在のカメラを Azimuth / Elevation で微小回転する。"""
        try:
            cam = self.renderer.GetActiveCamera()
            if azimuth_delta_deg:
                cam.Azimuth(float(azimuth_delta_deg))
            if elevation_delta_deg:
                cam.Elevation(float(elevation_delta_deg))
            cam.OrthogonalizeViewUp()
            self.renderer.ResetCameraClippingRange()
            self.safe_render()
        except Exception as _e:
            print(f"[Camera] rotate failed: {_e}")

    def _flip_camera_through_focal(self):
        """焦点点を対称中心としてカメラ位置を面対称反転する（9キー）。"""
        try:
            cam = self.renderer.GetActiveCamera()
            focal = cam.GetFocalPoint()
            pos = cam.GetPosition()
            cam.SetPosition(
                2 * focal[0] - pos[0],
                2 * focal[1] - pos[1],
                2 * focal[2] - pos[2],
            )
            # ViewUp は世界上向きを維持
            cam.SetViewUp(0, 0, 1)
            cam.OrthogonalizeViewUp()
            self.renderer.ResetCameraClippingRange()
            self.safe_render()
        except Exception as _e:
            print(f"[Camera] flip failed: {_e}")

    def _handle_camera_view_key(self, event) -> bool:
        """0-9 (テンキー/メイン両方) を受けてブレンダー風カメラ操作を実行。
        処理したら True を返す。"""
        # Qt では numpad の 0..9 も Key_0..Key_9 と同じ値。KeypadModifier で区別
        # 可能だがここでは両方受ける。
        key = event.key()
        # Digit キー以外はスルー
        digit_map = {
            QtCore.Qt.Key_0: 0, QtCore.Qt.Key_1: 1, QtCore.Qt.Key_2: 2,
            QtCore.Qt.Key_3: 3, QtCore.Qt.Key_4: 4, QtCore.Qt.Key_5: 5,
            QtCore.Qt.Key_6: 6, QtCore.Qt.Key_7: 7, QtCore.Qt.Key_8: 8,
            QtCore.Qt.Key_9: 9,
        }
        if key not in digit_map:
            return False
        # 修飾キー付きは無視（Ctrl+1 / Cmd+1 等は他のショートカットに任せる）。
        # KeypadModifier や NumLock 相当のフラグは許容する。
        # PySide6 の KeyboardModifier は int() 不可なので QFlags のまま bitwise で判定。
        mods = event.modifiers()
        blocking_mods = (QtCore.Qt.ControlModifier | QtCore.Qt.MetaModifier |
                         QtCore.Qt.AltModifier | QtCore.Qt.ShiftModifier)
        if bool(mods & blocking_mods):
            return False

        n = digit_map[key]
        if n == 0:
            self._restore_camera_state()
        elif n == 1:
            self._apply_standard_view(0, 0)          # Front (+X から見る)
        elif n == 3:
            self._apply_standard_view(90, 0)         # Right (+Y から見る)
        elif n == 7:
            self._apply_standard_view(0, 90)         # Top (+Z から見下ろす)
        elif n == 2:
            self._rotate_camera_delta(0, +15)        # elevation +15
        elif n == 8:
            self._rotate_camera_delta(0, -15)        # elevation -15
        elif n == 4:
            self._rotate_camera_delta(-15, 0)        # azimuth -15
        elif n == 6:
            self._rotate_camera_delta(+15, 0)        # azimuth +15
        elif n == 9:
            self._flip_camera_through_focal()
        elif n == 5:
            return False  # 5 は未割り当て
        return True

    def safe_render(self):
        """オフスクリーンレンダリングでQLabelを更新"""
        try:
            if not self._initialized:
                return
            # OffscreenRenderer takes _vtk_lock itself, only around the actual VTK
            # render+readback call — see its render_lock docstring.
            self.offscreen_renderer.update_display(self.vtk_display)
        except Exception as e:
            print(f"[VTK] Render error: {e}")

    def force_render(self):
        """強制レンダリング（アクター追加後に使用）"""
        try:
            if self._initialized:
                self.offscreen_renderer.update_display(self.vtk_display)
        except Exception as e:
            print(f"[VTK] force_render error: {e}")

    def set_robot_model(self, robot_model):
        """URDFRobotModelへの参照を設定"""
        self.robot_model = robot_model

    def pick_actor_at(self, x, y):
        """指定位置のアクターをピック"""
        # QLabelの座標をVTK座標に変換（Y軸反転）
        height = self.vtk_display.height()
        vtk_y = height - y
        self.picker.Pick(x, vtk_y, 0, self.renderer)
        return self.picker.GetActor()

    def find_link_name_for_actor(self, actor):
        """アクターに対応するリンク名を検索"""
        if not self.robot_model:
            return None
        for link_name, actors_list in self.robot_model.link_actors.items():
            if isinstance(actors_list, list):
                if actor in actors_list:
                    return link_name
            elif actors_list == actor:
                return link_name
        return None

    def find_joint_for_link(self, link_name):
        """リンク名に対応する関節を検索"""
        if not self.robot_model:
            return None
        for joint_name, joint in self.robot_model.joints.items():
            if joint.child_link == link_name:
                return joint_name, joint
        return None, None

    def _open_joint_dialog(self, actor, mouse_pos=None, local_pos=None):
        """メッシュに対応するジョイント編集ダイアログを開く"""
        link_name = self.find_link_name_for_actor(actor)
        if not link_name:
            return

        joint_name, joint_info = self.find_joint_for_link(link_name)
        if not joint_name or not joint_info:
            print(f"[VTK] No joint found for link: {link_name}")
            return

        # 現在の角度を取得（既に度数）
        current_deg = self.robot_model.get_joint_angle(joint_name)

        # ダイアログを作成
        dialog = SingleJointDialog(joint_name, joint_info, current_deg, self)
        dialog.angle_changed.connect(self._on_dialog_angle_changed)

        # マウス位置に応じてダイアログを配置
        OFFSET = 60  # マウスからのオフセット（px）
        if mouse_pos and local_pos:
            dialog.show()  # サイズを取得するために一度表示
            dialog_size = dialog.size()

            # 3Dビューの中心を基準に左右を判断
            view_center_x = self.vtk_display.width() / 2
            is_left_side = local_pos.x() < view_center_x

            if is_left_side:
                # 左側: モーダルの右下がマウス位置、さらに左にオフセット
                x = mouse_pos.x() - dialog_size.width() - OFFSET
                y = mouse_pos.y() - dialog_size.height()
            else:
                # 右側: モーダルの左下がマウス位置、さらに右にオフセット
                x = mouse_pos.x() + OFFSET
                y = mouse_pos.y() - dialog_size.height()
            dialog.move(x, y)
        else:
            dialog.show()

        # ダイアログをインスタンス変数に保持（参照を維持）
        if not hasattr(self, '_joint_dialogs'):
            self._joint_dialogs = []
        self._joint_dialogs.append(dialog)
        dialog.finished.connect(lambda: self._joint_dialogs.remove(dialog) if dialog in self._joint_dialogs else None)

    def _on_dialog_angle_changed(self, joint_name, angle_deg):
        """ダイアログからの角度変更を処理"""
        if not self.robot_model:
            return

        angles = self.robot_model.get_current_angles()
        angles[joint_name] = angle_deg  # 既に度数
        self.robot_model.apply_joint_angles(angles)
        self.safe_render()

        # JointEditorPanelに通知（angles_changedシグナルを発行）
        self.joint_angle_changed.emit(joint_name, angle_deg)

    def select_mesh(self, actor):
        """メッシュを選択状態にする"""
        # 以前の選択を解除
        self.deselect_mesh()

        if actor is None:
            return

        # リンク名を検索
        link_name = self.find_link_name_for_actor(actor)
        if link_name is None:
            return

        # 選択状態を保存
        self.selected_actor = actor
        self.selected_link_name = link_name
        self.original_color = actor.GetProperty().GetColor()

        # ハイライト点滅開始
        self.highlight_visible = True
        actor.GetProperty().SetColor(*MESH_HIGHLIGHT_COLOR)
        self.highlight_timer.start(MESH_HIGHLIGHT_BLINK_INTERVAL)
        self.safe_render()

        print(f"[VTK] Selected link: {link_name}")

    def deselect_mesh(self):
        """メッシュの選択を解除"""
        self.highlight_timer.stop()
        if self.selected_actor and self.original_color:
            self.selected_actor.GetProperty().SetColor(*self.original_color)
            self.safe_render()
        self.selected_actor = None
        self.selected_link_name = None
        self.original_color = None

    def _toggle_highlight(self):
        """ハイライトの点滅を切り替え"""
        if not self.selected_actor:
            return
        self.highlight_visible = not self.highlight_visible
        if self.highlight_visible:
            self.selected_actor.GetProperty().SetColor(*MESH_HIGHLIGHT_COLOR)
        else:
            if self.original_color:
                self.selected_actor.GetProperty().SetColor(*self.original_color)
        self.safe_render()

    def _get_joint_world_axis(self, joint):
        """関節の回転軸をワールド座標系で返す。"""
        cached = getattr(self.robot_model, '_link_world_transforms', None)
        parent_world_T = cached.get(joint.parent_link, np.eye(4)) if cached else np.eye(4)
        jt_R = _rpy_to_rotation_matrix(*joint.origin_rpy)
        local_axis = np.array(joint.axis, dtype=float)
        axis_world = parent_world_T[:3, :3] @ jt_R @ local_axis
        mag = np.linalg.norm(axis_world)
        return axis_world / mag if mag > 1e-6 else np.array([1.0, 0.0, 0.0])

    def _compute_drag_screen_tangent(self, joint):
        """カメラ視点での「正回転方向」スクリーンタンジェントを返す (screen_tan_x, screen_tan_y)。"""
        axis_world = self._get_joint_world_axis(joint)

        camera = self.renderer.GetActiveCamera()
        cam_pos = np.array(camera.GetPosition())
        focal = np.array(camera.GetFocalPoint())
        view_up = np.array(camera.GetViewUp())

        forward = focal - cam_pos
        fwd_mag = np.linalg.norm(forward)
        if fwd_mag > 1e-6:
            forward /= fwd_mag
        right = np.cross(forward, view_up)
        r_mag = np.linalg.norm(right)
        if r_mag > 1e-6:
            right /= r_mag
        up = np.cross(right, forward)
        u_mag = np.linalg.norm(up)
        if u_mag > 1e-6:
            up /= u_mag

        # Axis projected onto screen (right=+screenX, up=+screenUp=-screenY)
        proj_x = float(np.dot(axis_world, right))
        proj_y = float(np.dot(axis_world, up))

        # Screen tangent for positive rotation (right-hand rule):
        # screen_tan_x = proj_y, screen_tan_y = proj_x  (screen Y is downward)
        screen_tan_x = proj_y
        screen_tan_y = proj_x
        mag = math.sqrt(screen_tan_x ** 2 + screen_tan_y ** 2)
        if mag > 0.05:
            return (screen_tan_x / mag, screen_tan_y / mag)
        # Axis nearly perpendicular to screen: fall back to horizontal drag
        return (1.0, 0.0)

    def start_joint_drag(self, pos):
        """関節ドラッグを開始"""
        if not self.selected_link_name or not self.robot_model:
            return False

        joint_name, joint = self.find_joint_for_link(self.selected_link_name)
        if joint_name is None:
            print(f"[VTK] No joint found for link: {self.selected_link_name}")
            return False

        self.is_dragging_joint = True
        self.drag_start_pos = pos
        self.drag_start_angle = self.robot_model.get_joint_angle(joint_name)
        self._drag_screen_tangent = self._compute_drag_screen_tangent(joint)
        if self.joint_editor and self.joint_editor._joint_is_rev(joint_name):
            tx, ty = self._drag_screen_tangent
            self._drag_screen_tangent = (-tx, -ty)
        self.drag_joint_name_pair = None
        self.drag_start_angle_pair = 0.0
        if self.pair_check.isChecked():
            pair_name = self._get_pair_joint_name(joint_name)
            if pair_name and pair_name in self.robot_model.get_current_angles():
                self.drag_joint_name_pair = pair_name
                self.drag_start_angle_pair = self.robot_model.get_joint_angle(pair_name)
                print(f"[VTK] Pair joint: {pair_name}, start angle: {self.drag_start_angle_pair}")
        # Record group preset base angles at drag start
        self._drag_group_preset_idx = -1
        self._drag_group_base_angles = {}
        je = self.joint_editor
        if je and je.current_group_preset_index >= 0:
            self._drag_group_preset_idx = je.current_group_preset_index
            self._drag_group_base_angles = dict(self.robot_model.get_current_angles())
        print(f"[VTK] Start dragging joint: {joint_name}, angle: {self.drag_start_angle}")
        return True

    def update_joint_drag(self, pos):
        """関節ドラッグを更新"""
        if not self.is_dragging_joint or not self.selected_link_name:
            return

        joint_name, joint = self.find_joint_for_link(self.selected_link_name)
        if joint_name is None:
            return

        drag_x = pos.x() - self.drag_start_pos.x()
        drag_y = pos.y() - self.drag_start_pos.y()

        # 関節軸のスクリーン投影に基づいた正回転方向へのドラッグ量を計算
        tan_x, tan_y = self._drag_screen_tangent
        delta = drag_x * tan_x + drag_y * tan_y
        delta_angle = delta * MESH_DRAG_SENSITIVITY

        # 新しい角度を計算（リミット適用）
        new_angle = self.drag_start_angle + delta_angle
        step = self._current_step_deg()
        if step > 0:
            new_angle = round(new_angle / step) * step
        new_angle = max(joint.limit_lower, min(joint.limit_upper, new_angle))

        # 角度を更新
        angles = self.robot_model.get_current_angles()
        angles[joint_name] = new_angle
        if self.drag_joint_name_pair:
            effective_delta = new_angle - self.drag_start_angle
            if self._pair_should_negate(joint_name, self.drag_joint_name_pair):
                effective_delta = -effective_delta
            pair_joint = self.robot_model.joints.get(self.drag_joint_name_pair)
            pair_angle = self.drag_start_angle_pair + effective_delta
            if pair_joint:
                pair_angle = max(pair_joint.limit_lower, min(pair_joint.limit_upper, pair_angle))
            angles[self.drag_joint_name_pair] = pair_angle

        # Apply link group preset: move other enabled members proportionally
        je = self.joint_editor
        if (je and self._drag_group_preset_idx >= 0 and
                0 <= self._drag_group_preset_idx < len(je.joint_group_presets)):
            preset = je.joint_group_presets[self._drag_group_preset_idx]
            members = preset.get("members", {})
            dragged_member = members.get(joint_name, {})
            if dragged_member.get("enabled", False):
                dragged_scale = float(dragged_member.get("scale", 1.0))
                if abs(dragged_scale) > 1e-9:
                    master_delta = (new_angle - self.drag_start_angle) / dragged_scale
                    for jname, member in members.items():
                        if jname == joint_name:
                            continue
                        if not member.get("enabled", False):
                            continue
                        scale_j = float(member.get("scale", 1.0))
                        base_j = self._drag_group_base_angles.get(jname, 0.0)
                        joint_j = self.robot_model.joints.get(jname)
                        new_j = base_j + master_delta * scale_j
                        if joint_j:
                            new_j = max(joint_j.limit_lower, min(joint_j.limit_upper, new_j))
                        angles[jname] = new_j

        self.robot_model.apply_joint_angles(angles)
        self.safe_render()

    def _start_wheel_session(self, joint_name):
        """マウスホイール回転セッションを開始。drag_start_* と同じ意味の基準を記録する。
        既にセッション中で joint が変わった場合は、直前セッションを end してから開始。"""
        if self._wheel_active and self._wheel_joint_name == joint_name:
            return
        if self._wheel_active:
            # 別関節に切り替わったら、直前分をまず確定させる
            self._on_wheel_stopped()

        self._wheel_active = True
        self._wheel_joint_name = joint_name
        self._wheel_start_angle = self.robot_model.get_joint_angle(joint_name)

        # Pair
        self._wheel_pair_name = None
        self._wheel_start_angle_pair = 0.0
        if self.pair_check.isChecked():
            pair_name = self._get_pair_joint_name(joint_name)
            if pair_name and pair_name in self.robot_model.get_current_angles():
                self._wheel_pair_name = pair_name
                self._wheel_start_angle_pair = self.robot_model.get_joint_angle(pair_name)

        # Link group
        self._wheel_group_preset_idx = -1
        self._wheel_group_base_angles = {}
        je = self.joint_editor
        if je and je.current_group_preset_index >= 0:
            self._wheel_group_preset_idx = je.current_group_preset_index
            self._wheel_group_base_angles = dict(self.robot_model.get_current_angles())

    def _apply_wheel_rotation(self, joint_name, joint, delta):
        """ホイール1ノッチ分の関節回転を適用する。
        マウスドラッグ (update_joint_drag) と同じく Pair / LinkGroup / step_snap を尊重し、
        ノッチ毎に wheel_angles_updated を発火して Live 送信/スライダー反映へ繋げる。
        180ms 停止で _on_wheel_stopped が呼ばれ、joint_drag_ended 相当を発火する。"""
        self._start_wheel_session(joint_name)

        current_angle = self.robot_model.get_joint_angle(joint_name)
        step = self._current_step_deg()
        if step > 0:
            # Snap ON: 1ノッチあたり ±1 グリッド分進める。加算量方式 (delta*sens
            # を足して round) だと、1ノッチの delta_angle (通例 120*0.1=12°)
            # が step/2 未満のとき round が元グリッドに戻して動かなくなる。
            # (例: step=24 で threshold=12° / banker's rounding、step=90/180 で全く動かない)
            # 方向のみを使い、現在角度から見て次のグリッド点へ移す。
            if delta > 0:
                new_angle = math.floor(current_angle / step + 1.0) * step
            else:
                new_angle = math.ceil(current_angle / step - 1.0) * step
        else:
            new_angle = current_angle + delta * MESH_WHEEL_SENSITIVITY
        new_angle = max(joint.limit_lower, min(joint.limit_upper, new_angle))

        angles = self.robot_model.get_current_angles()
        angles[joint_name] = new_angle

        # Pair 対応: drag と同じ effective_delta 計算
        if self._wheel_pair_name:
            effective_delta = new_angle - self._wheel_start_angle
            if self._pair_should_negate(joint_name, self._wheel_pair_name):
                effective_delta = -effective_delta
            pair_joint = self.robot_model.joints.get(self._wheel_pair_name)
            pair_angle = self._wheel_start_angle_pair + effective_delta
            if pair_joint:
                pair_angle = max(pair_joint.limit_lower,
                                 min(pair_joint.limit_upper, pair_angle))
            angles[self._wheel_pair_name] = pair_angle

        # Link group 同期: drag と同じ比例配分
        je = self.joint_editor
        if (je and self._wheel_group_preset_idx >= 0 and
                0 <= self._wheel_group_preset_idx < len(je.joint_group_presets)):
            preset = je.joint_group_presets[self._wheel_group_preset_idx]
            members = preset.get("members", {})
            dragged_member = members.get(joint_name, {})
            if dragged_member.get("enabled", False):
                dragged_scale = float(dragged_member.get("scale", 1.0))
                if abs(dragged_scale) > 1e-9:
                    master_delta = (new_angle - self._wheel_start_angle) / dragged_scale
                    for jname, member in members.items():
                        if jname == joint_name:
                            continue
                        if not member.get("enabled", False):
                            continue
                        scale_j = float(member.get("scale", 1.0))
                        base_j = self._wheel_group_base_angles.get(jname, 0.0)
                        joint_j = self.robot_model.joints.get(jname)
                        new_j = base_j + master_delta * scale_j
                        if joint_j:
                            new_j = max(joint_j.limit_lower,
                                        min(joint_j.limit_upper, new_j))
                        angles[jname] = new_j

        self.robot_model.apply_joint_angles(angles)
        self.safe_render()

        # ノッチ毎に JointEditor スライダー反映 + Valkey Live 送信を要求
        self.wheel_angles_updated.emit(dict(angles))

        # 停止検出タイマーをリスタート
        self._wheel_stop_timer.start()

    def _on_wheel_stopped(self):
        """ホイールが 180ms 止まったら、drag 終了と同じ後処理を行う:
        push_undo + JointEditor 反映 + node 保存 + Live ON なら Valkey 一発書き込み。"""
        if not self._wheel_active:
            return
        self._wheel_active = False
        self._wheel_joint_name = None
        self._wheel_pair_name = None
        self._wheel_group_preset_idx = -1
        self._wheel_group_base_angles = {}
        # ドラッグ終了と同じシグナルを流用。on_joint_drag_ended 側で
        # push_undo / _save_to_node / Valkey 書き込み(Live ON時) が実行される。
        self.joint_drag_ended.emit()

    def _get_pair_joint_name(self, joint_name):
        """l_xxx ↔ r_xxx のペア関節名を返す。対応なければ None。"""
        if joint_name.startswith("l_"):
            return "r_" + joint_name[2:]
        if joint_name.startswith("r_"):
            return "l_" + joint_name[2:]
        return None

    def _pair_should_negate(self, primary_name: str, pair_name: str) -> bool:
        """ペアドラッグで FK delta を反転すべきか。
        JOINT_TO_MERIDIM の符号積でミラーに必要な反転を判定。
        Pair モード: ミラー（物理空間で左右対称）。
        Opp モード: ミラーの逆（同方向に動く）。
        """
        j = JOINT_TO_MERIDIM.get(primary_name)
        p = JOINT_TO_MERIDIM.get(pair_name)
        if j is not None and p is not None:
            mirror_needs_negate = j[1] * p[1] < 0
        else:
            # Fallback: R側のロール/ヨー軸は MJCF 上で物理的に逆方向
            pn = pair_name.lower()
            mirror_needs_negate = pn.startswith("r_") and (pn.endswith("_xr") or pn.endswith("_zy"))

        if getattr(self, 'opp_check', None) and self.opp_check.isChecked():
            return not mirror_needs_negate  # Opp = 同方向（ミラーの逆）
        return mirror_needs_negate  # Pair = ミラー

    def _current_step_deg(self) -> float:
        """3Dビュー drag/wheel で使う角度スナップ幅を返す。0 ならスナップなし。
        JointEditor の Step 設定 (step_snapping_enabled / step_snapping_deg) を
        単一の真実として参照する。3D 側 step_snap_check は表示上のミラー。"""
        je = self.joint_editor
        if je is None or not getattr(je, "step_snapping_enabled", False):
            return 0.0
        deg = float(getattr(je, "step_snapping_deg", 0.0))
        return deg if deg > 0.0 else 0.0

    def _save_step_settings(self):
        # Step 系は JointEditor 側 (_on_step_snapping_*) が
        # joint_sliders_step_* を保存するため、ここでは扱わない。
        s = load_app_settings()
        s["vtk_drag_pair_enabled"] = self.pair_check.isChecked()
        s["vtk_drag_opp_enabled"] = self.opp_check.isChecked()
        s["valkey_enabled"] = self.valkey_check.isChecked()
        s["valkey_live_enabled"] = self.live_check.isChecked()
        save_app_settings(s)

    def end_joint_drag(self):
        """関節ドラッグを終了"""
        if self.is_dragging_joint:
            joint_name, _ = self.find_joint_for_link(self.selected_link_name)
            if joint_name:
                angle = self.robot_model.get_joint_angle(joint_name)
                dbg("[TRIGGER]", f"3D view joint drag ended",
                    joint=joint_name, angle=angle)
            # シグナルを発行してノードへの保存をトリガー
            self.joint_drag_ended.emit()
        self.is_dragging_joint = False
        self.drag_start_pos = None

    def reset_camera(self):
        """カメラビューをリセットし、すべてのSTLモデルをビューに収める"""
        num_actors = self.renderer.GetActors().GetNumberOfItems()
        print(f"[VTK] reset_camera called, actors: {num_actors}, initialized: {self._initialized}")

        if num_actors == 0:
            print("[VTK] No actors, setting up default camera")
            self.setup_camera()
            return

        # すべてのアクターの合計バウンディングボックスを計算
        bounds = [float('inf'), float('-inf'), 
                float('inf'), float('-inf'), 
                float('inf'), float('-inf')]
        
        actors = self.renderer.GetActors()
        actors.InitTraversal()
        actor = actors.GetNextActor()
        while actor:
            actor_bounds = actor.GetBounds()
            # X軸の最小値と最大値
            bounds[0] = min(bounds[0], actor_bounds[0])
            bounds[1] = max(bounds[1], actor_bounds[1])
            # Y軸の最小値と最大値
            bounds[2] = min(bounds[2], actor_bounds[2])
            bounds[3] = max(bounds[3], actor_bounds[3])
            # Z軸の最小値と最大値
            bounds[4] = min(bounds[4], actor_bounds[4])
            bounds[5] = max(bounds[5], actor_bounds[5])
            actor = actors.GetNextActor()

        # バウンディングボックスの中心を計算
        center = [(bounds[1] + bounds[0]) / 2,
                (bounds[3] + bounds[2]) / 2,
                (bounds[5] + bounds[4]) / 2]

        # バウンディングボックスの対角線の長さを計算
        diagonal = ((bounds[1] - bounds[0]) ** 2 +
                (bounds[3] - bounds[2]) ** 2 +
                (bounds[5] - bounds[4]) ** 2) ** 0.5

        camera = self.renderer.GetActiveCamera()
        camera.ParallelProjectionOn()
        
        # カメラの位置を設定（バウンディングボックスの対角線の2倍の距離）
        distance = diagonal
        camera.SetPosition(center[0] + distance, center[1], center[2])
        camera.SetFocalPoint(center[0], center[1], center[2])
        camera.SetViewUp(0, 0, 1)
        
        # パラレルスケールを設定してビューに収める
        camera.SetParallelScale(diagonal * 0.5)

        # クリッピング範囲を更新
        self.renderer.ResetCameraClippingRange()
        self.safe_render()

        print("Camera reset complete - All STL models fitted to view")
        # Notify listeners with the computed fit parameters
        focal_pt = camera.GetFocalPoint()
        self.camera_fitted.emit(focal_pt[0], focal_pt[1], focal_pt[2], float(distance))

    def reset_view_to_fit(self):
        """すべてのSTLモデルが見えるようにビューをリセットして調整"""
        self.reset_camera()
        self.safe_render()

    def create_coordinate_axes(self):
        """座標軸の作成（線と独立したテキスト）"""
        base_assembly = vtk.vtkAssembly()
        length = 0.1
        text_offset = 0.02
        
        # ラインの作成部分は変更なし
        for i, (color, _) in enumerate([
            ((1,0,0), "X"),
            ((0,1,0), "Y"),
            ((0,0,1), "Z")
        ]):
            line = vtk.vtkLineSource()
            line.SetPoint1(0, 0, 0)
            end_point = [0, 0, 0]
            end_point[i] = length
            line.SetPoint2(*end_point)
            
            mapper = vtk.vtkPolyDataMapper()
            mapper.SetInputConnection(line.GetOutputPort())
            
            actor = vtk.vtkActor()
            actor.SetMapper(mapper)
            actor.GetProperty().SetColor(*color)
            actor.GetProperty().SetLineWidth(2)
            
            base_assembly.AddPart(actor)

        # テキスト部分をvtkBillboardTextActor3Dに変更
        for i, (color, label) in enumerate([
            ((1,0,0), "X"),
            ((0,1,0), "Y"),
            ((0,0,1), "Z")
        ]):
            text_position = [0, 0, 0]
            text_position[i] = length + text_offset
            
            text_actor = vtk.vtkBillboardTextActor3D()  # vtkTextActor3Dから変更
            text_actor.SetInput(label)
            text_actor.SetPosition(*text_position)
            text_actor.GetTextProperty().SetColor(*color)
            text_actor.GetTextProperty().SetFontSize(12)
            text_actor.GetTextProperty().SetJustificationToCentered()
            text_actor.GetTextProperty().SetVerticalJustificationToCentered()
            text_actor.SetScale(0.02)  # 単一の値でスケールを設定
            
            self.renderer.AddActor(text_actor)
            if not hasattr(self, 'text_actors'):
                self.text_actors = []
            self.text_actors.append(text_actor)
        
        return base_assembly

    def update_coordinate_axes(self, position):
        """座標軸とテキストの位置を更新"""
        # ラインの位置を更新
        transform = vtk.vtkTransform()
        transform.Identity()
        transform.Translate(position[0], position[1], position[2])
        self.coordinate_axes_actor.SetUserTransform(transform)
        
        # テキストの位置を更新
        if hasattr(self, 'text_actors'):
            for text_actor in self.text_actors:
                original_pos = list(text_actor.GetPosition())
                text_actor.SetPosition(
                    original_pos[0] + position[0],
                    original_pos[1] + position[1],
                    original_pos[2] + position[2]
                )
        
        self.safe_render()

    def update_stl_transform(self, node, point_xyz):
        """STLの位置を更新"""
        # base_linkの場合は処理をスキップ
        if isinstance(node, BaseLinkNode):
            return

        if node in self.stl_actors and node in self.transforms:
            print(f"Updating transform for node {node.name()} to position {point_xyz}")
            transform = self.transforms[node]
            transform.Identity()
            transform.Translate(point_xyz[0], point_xyz[1], point_xyz[2])
            
            self.stl_actors[node].SetUserTransform(transform)

            # base_linkに接続された最初のノードの場合、座標軸も更新
            if hasattr(node, 'graph'):
                base_node = node.graph.get_node_by_name('base_link')
                if base_node:
                    for port in base_node.output_ports():
                        for connected_port in port.connected_ports():
                            if connected_port.node() == node:
                                self.base_connected_node = node
                                self.update_coordinate_axes(point_xyz)
                                break

            self.safe_render()
        else:
            # base_link以外のノードの場合のみ警告を表示
            if not isinstance(node, BaseLinkNode):
                print(f"Warning: No STL actor or transform found for node {node.name()}")

    def reset_stl_transform(self, node):
        """STLの位置をリセット"""
        # base_linkの場合は処理をスキップ
        if isinstance(node, BaseLinkNode):
            return

        if node in self.transforms:
            print(f"Resetting transform for node {node.name()}")
            transform = self.transforms[node]
            transform.Identity()
            
            self.stl_actors[node].SetUserTransform(transform)
            
            # 座標軸のリセット（必要な場合）
            if node == self.base_connected_node:
                self.update_coordinate_axes([0, 0, 0])
                self.base_connected_node = None
            
            self.safe_render()
        else:
            # base_link以外のノードの場合のみ警告を表示
            if not isinstance(node, BaseLinkNode):
                print(f"Warning: No transform found for node {node.name()}")

    def load_stl_for_node(self, node):
        """ノード用のSTLファイルを読み込む（色の適用を含む）"""
        # base_linkの場合は処理をスキップ
        if isinstance(node, BaseLinkNode):
            return

        if node.stl_file:
            print(f"Loading STL for node: {node.name()}, file: {node.stl_file}")
            reader = vtk.vtkSTLReader()
            reader.SetFileName(node.stl_file)

            mapper = vtk.vtkPolyDataMapper()
            mapper.SetInputConnection(reader.GetOutputPort())

            actor = vtk.vtkActor()
            actor.SetMapper(mapper)

            transform = vtk.vtkTransform()
            transform.Identity()
            actor.SetUserTransform(transform)

            # 既存のアクターを削除
            if node in self.stl_actors:
                self.renderer.RemoveActor(self.stl_actors[node])

            self.stl_actors[node] = actor
            self.transforms[node] = transform
            self.renderer.AddActor(actor)

            # ノードの色情報を適用
            self.apply_color_to_node(node)

            self.reset_camera()
            self.safe_render()
            print(f"STL file loaded and rendered: {node.stl_file}")

    def apply_color_to_node(self, node):
        """ノードのSTLモデルに色を適用"""
        if node in self.stl_actors:
            # デフォルトの色を設定（色情報がない場合）
            if not hasattr(node, 'node_color') or node.node_color is None:
                node.node_color = [1.0, 1.0, 1.0]  # 白色をデフォルトに

            # 色の適用
            actor = self.stl_actors[node]
            actor.GetProperty().SetColor(*node.node_color)
            print(f"Applied color to node {node.name()}: RGB({node.node_color[0]:.3f}, {node.node_color[1]:.3f}, {node.node_color[2]:.3f})")
            self.safe_render()

    def remove_stl_for_node(self, node):
        """ノードのSTLを削除"""
        if node in self.stl_actors:
            self.renderer.RemoveActor(self.stl_actors[node])
            del self.stl_actors[node]
            if node in self.transforms:
                del self.transforms[node]
                
            # 座標軸のリセット（必要な場合）
            if node == self.base_connected_node:
                self.update_coordinate_axes([0, 0, 0])
                self.base_connected_node = None
                
            self.safe_render()
            print(f"Removed STL for node: {node.name()}")

    def setup_camera(self):
        """カメラの初期設定"""
        camera = self.renderer.GetActiveCamera()
        camera.ParallelProjectionOn()
        camera.SetPosition(1, 0, 0)
        camera.SetFocalPoint(0, 0, 0)
        camera.SetViewUp(0, 0, 1)

    def cleanup(self):
        """STLビューアのリソースをクリーンアップ"""
        # cleanup()はaboutToQuit経由の明示呼び出しと__del__の両方から呼ばれ得る。
        # VTKオブジェクトの解放後に二重で解放処理を行うとネイティブクラッシュ
        # (セグメンテーション違反)を起こすため、一度だけ実行するようにガードする。
        if getattr(self, '_cleanup_done', False):
            return
        self._cleanup_done = True

        # ハイライト点滅タイマーを止める。動いたままだと、削除済み/解放済みの
        # selected_actorに対して_toggle_highlight()がsafe_render()経由で
        # ネイティブクラッシュを起こすことがある。
        if hasattr(self, 'highlight_timer') and self.highlight_timer:
            self.highlight_timer.stop()
        self.selected_actor = None

        # VTKオブジェクトの解放
        if hasattr(self, 'renderer'):
            if self.renderer:
                # アクターの削除
                # GetActors() はrenderer内部のライブなコレクションを返すため、
                # 走査中にRemoveActor()で直接変更するとイテレータが破壊され
                # ネイティブクラッシュ(アクセス違反)を起こす。先にPythonリストへ
                # スナップショットしてから削除する。
                for actor in list(self.renderer.GetActors()):
                    self.renderer.RemoveActor(actor)
                
                # テキストアクターの削除
                for actor in self.text_actors:
                    self.renderer.RemoveActor(actor)
                self.text_actors.clear()

        # インタラクターの終了
        if hasattr(self, 'iren'):
            if self.iren:
                self.iren.TerminateApp()

        # レンダーウィンドウのクリーンアップ
        if hasattr(self, 'vtkWidget'):
            if self.vtkWidget:
                self.vtkWidget.close()

        # 参照の解放
        self.stl_actors.clear()
        self.transforms.clear()

        # 以降、遅延実行されたsafe_render/force_render(QTimer.singleShotなど)が
        # 解放済みのレンダーウィンドウに触れないようにする。
        self._initialized = False

    def __del__(self):
        """デストラクタでクリーンアップを実行"""
        self.cleanup()

    def update_rotation_axis(self, node, axis_id):
        """ノードの回転軸を更新"""
        try:
            print(f"Updating rotation axis for node {node.name()} to axis {axis_id}")
            
            if node in self.stl_actors and node in self.transforms:
                transform = self.transforms[node]
                actor = self.stl_actors[node]
                
                # 現在の位置を保持
                current_position = list(actor.GetPosition())
                
                # 変換をリセット
                transform.Identity()
                
                # 位置を再設定
                transform.Translate(*current_position)
                
                # 新しい回転軸に基づいて回転を設定
                # 必要に応じてここに回転の処理を追加
                
                # 変換を適用
                actor.SetUserTransform(transform)
                
                # ビューを更新
                self.safe_render()
                print(f"Successfully updated rotation axis for node {node.name()}")
            else:
                print(f"No STL actor or transform found for node {node.name()}")
                
        except Exception as e:
            print(f"Error updating rotation axis: {str(e)}")
            traceback.print_exc()

    def update_background(self, value=None):
        """背景色をスライダーの値に基づいて白をミキシング + グラデーション適用"""
        if value is None:
            value = self.bg_slider_value

        # 0〜100の値を0〜1の範囲に変換（右に行くほど白をミキシング）
        t = value / 100.0
        white = [1.0, 1.0, 1.0]

        # Color A に白をミキシング
        r1 = self.bg_color_a[0] * (1 - t) + white[0] * t
        g1 = self.bg_color_a[1] * (1 - t) + white[1] * t
        b1 = self.bg_color_a[2] * (1 - t) + white[2] * t

        # Color B に白をミキシング
        r2 = self.bg_color_b[0] * (1 - t) + white[0] * t
        g2 = self.bg_color_b[1] * (1 - t) + white[1] * t
        b2 = self.bg_color_b[2] * (1 - t) + white[2] * t

        # グラデーションタイプに応じて設定
        if self.bg_gradient_type == "vertical":
            # 上下グラデーション（上=Color B, 下=Color A）
            self.renderer.SetGradientBackground(True)
            self.renderer.SetBackground(r1, g1, b1)   # 下
            self.renderer.SetBackground2(r2, g2, b2)  # 上
        else:
            # 単色（Color Aのみ使用）
            self.renderer.SetGradientBackground(False)
            self.renderer.SetBackground(r1, g1, b1)

        self.safe_render()

    def set_bg_colors(self, color_a, color_b):
        """背景色A/Bを設定してスライダーの現在値で再描画"""
        self.bg_color_a = list(color_a)
        self.bg_color_b = list(color_b)
        self.update_background()

    def set_bg_gradient_type(self, gradient_type):
        """グラデーションタイプを設定して再描画"""
        self.bg_gradient_type = gradient_type
        self.update_background()

    def set_bg_slider_value(self, value):
        """BG slider value from Settings dialog"""
        self.bg_slider_value = value
        self.update_background(value)

    def set_light_slider_value(self, value):
        """Light slider value from Settings dialog"""
        self.light_slider_value = value
        self.update_light_intensity(value)

    def update_light_intensity(self, value):
        """ライトの強度をスライダーの値に基づいて更新"""
        # 0〜100の値を0〜1の範囲に変換
        intensity = value / 100.0
        for light in self.lights:
            light.SetIntensity(intensity)
        # アンビエントライトも調整
        self.renderer.SetAmbient(intensity, intensity, intensity)
        self.safe_render()

    def open_urdf_loader_website(self):
        """URDF Loadersのウェブサイトを開く"""
        url = QtCore.QUrl(
            "https://gkjohnson.github.io/urdf-loaders/javascript/example/bundle/")
        QtGui.QDesktopServices.openUrl(url)


# ---------------------------------------------------------------
# Context menu style (Copy / Duplicate / Delete / Divide)
# ---------------------------------------------------------------
_CONTEXT_MENU_STYLESHEET = """
QMenu {
    background-color: #2b2b2b;
    color: #e6e6e6;
    border: 1px solid #555555;
    padding: 4px 0px;
}
QMenu::item {
    background-color: transparent;
    padding: 6px 24px 6px 20px;
    margin: 1px 4px;
    border-radius: 3px;
}
QMenu::item:selected {
    background-color: #3d7fd6;
    color: #ffffff;
}
QMenu::item:pressed {
    background-color: #1f5aa8;
    color: #ffffff;
}
QMenu::item:disabled {
    color: #7a7a7a;
}
QMenu::separator {
    height: 1px;
    background: #555555;
    margin: 4px 8px;
}
"""


class DivideDialog(QtWidgets.QDialog):
    """Divide 実行時に分割数とイージングを問い合わせるモーダル。"""

    def __init__(self, initial_easing_option, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Divide Pose Transition")
        self.setModal(True)
        # 初期表示のイージングを覚えておき、accept 時に変更判定に使う
        self._initial_easing_option = str(initial_easing_option or EASING_OPTIONS[0])

        form = QtWidgets.QFormLayout(self)

        self._split_spin = QtWidgets.QSpinBox()
        self._split_spin.setRange(1, 99)
        self._split_spin.setValue(1)
        # ホイールでも増減できるように focus policy を強めに
        self._split_spin.setFocusPolicy(QtCore.Qt.WheelFocus)
        self._split_spin.setKeyboardTracking(True)
        form.addRow("Split count:", self._split_spin)

        self._easing_combo = QtWidgets.QComboBox()
        self._easing_combo.addItems(EASING_OPTIONS)
        try:
            _init_idx = easing_index(self._initial_easing_option)
            self._easing_combo.setCurrentIndex(_init_idx)
        except Exception:
            self._easing_combo.setCurrentIndex(0)
        form.addRow("Easing (target):", self._easing_combo)

        btns = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel
        )
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)
        form.addRow(btns)

    def get_split_count(self) -> int:
        return int(self._split_spin.value())

    def get_easing_option(self) -> str:
        return str(self._easing_combo.currentText())

    def easing_user_changed(self) -> bool:
        """ユーザーがイージングプルダウンを変更したかどうか。"""
        try:
            init_idx = easing_index(self._initial_easing_option)
        except Exception:
            init_idx = 0
        return int(self._easing_combo.currentIndex()) != int(init_idx)


class CustomNodeGraph(NodeGraph):
    node_long_pressed = QtCore.Signal(object)  # emits the node after 1-second hold

    def __init__(self, stl_viewer):
        # カスタムビューアを作成
        custom_viewer = CustomViewer()

        # カスタムビューアを使用してNodeGraphを初期化
        super(CustomNodeGraph, self).__init__(viewer=custom_viewer)

        # ノードグラフの背景色とグリッド色を設定
        self.set_background_color(*NODE_GRAPH_BG_COLOR)
        self.set_grid_color(*NODE_GRAPH_GRID_COLOR)

        # Viewerにgraphへの参照を設定（キーボードイベント処理用）
        custom_viewer.set_graph(self)

        self.stl_viewer = stl_viewer
        self.robot_name = "robot_x"
        self.project_dir = None
        self.meshes_dir = None
        self.last_save_dir = None

        # Undo/redo hook — set by build_motion_editor
        self._undo_push_fn = None

        # ポート接続/切断のシグナルを接続
        self.port_connected.connect(self.on_port_connected)
        self.port_disconnected.connect(self.on_port_disconnected)

        # ノードタイプの登録
        try:
            # BaseLinkNodeの登録
            self.register_node(BaseLinkNode)
            print(f"Registered node type: {BaseLinkNode.NODE_NAME}")

            # FooNodeの登録
            self.register_node(FooNode)
            print(f"Registered node type: {FooNode.NODE_NAME}")

            # PoseNodeの登録
            self.register_node(PoseNode)
            print(f"Registered node type: {PoseNode.NODE_NAME}")

            self.register_node(DefineNode)
            print(f"Registered node type: {DefineNode.NODE_NAME}")
            self.register_node(WaitNode)
            print(f"Registered node type: {WaitNode.NODE_NAME}")
            self.register_node(BranchingNode)
            print(f"Registered node type: {BranchingNode.NODE_NAME}")
            self.register_node(CommandNode)
            print(f"Registered node type: {CommandNode.NODE_NAME}")
            self.register_node(MixNode)
            print(f"Registered node type: {MixNode.NODE_NAME}")
            self.register_node(JumpNode)
            print(f"Registered node type: {JumpNode.NODE_NAME}")
            self.register_node(MemoNode)
            print(f"Registered node type: {MemoNode.NODE_NAME}")

        except Exception as e:
            print(f"Error registering node types: {str(e)}")
            import traceback
            traceback.print_exc()

        self.motion_action_state = None

        # 他の初期化コード...
        self._cleanup_handlers = []
        self._cached_positions = {}
        self._selection_cache = set()

        # 選択関連の変数を初期化
        self._selection_start = None
        self._is_selecting = False

        # ビューの設定
        self._view = self.widget

        # ラバーバンドの作成
        self._rubber_band = QtWidgets.QRubberBand(
            QtWidgets.QRubberBand.Shape.Rectangle,
            self._view
        )

        # オリジナルのイベントハンドラを保存
        self._original_handlers = {
            'press': self._view.mousePressEvent,
            'move': self._view.mouseMoveEvent,
            'release': self._view.mouseReleaseEvent
        }

        # 新しいイベントハンドラを設定
        self._view.mousePressEvent = self.custom_mouse_press
        self._view.mouseMoveEvent = self.custom_mouse_move
        self._view.mouseReleaseEvent = self.custom_mouse_release

        # インスペクタウィンドウの初期化
        self.inspector_window = InspectorWindow(stl_viewer=self.stl_viewer)

        # Pose Branching 用 Formula 定義（Pose Inspector の Forms で編集）
        _default_formula_body = "\n".join(["(Not Available Now)"] * 5)
        self.motion_formulas = {
            "Form1:foo": _default_formula_body,
            "Form2:bar": _default_formula_body,
        }
        # User Value 設定用（マイコン向けの編集メモ。モーションJSON非保存）
        self.user_value_session = default_user_value_session()

        # Long-press detection (1 second hold on a node)
        self._long_press_timer = QtCore.QTimer()
        self._long_press_timer.setSingleShot(True)
        self._long_press_timer.setInterval(1000)
        self._long_press_timer.timeout.connect(self._on_long_press_timeout)
        self._long_press_node = None
        self._long_press_start = None

    def _node_at_view_pos(self, widget_pos):
        """Return the node model under widget-space position, or None."""
        try:
            view = self.viewer()
            if view is None:
                return None
            # Use global coordinates for reliable cross-widget mapping
            global_pos = self._view.mapToGlobal(widget_pos)
            viewport_local = view.viewport().mapFromGlobal(global_pos)
            scene_pos = view.mapToScene(viewport_local)
            for item in view.scene().items(scene_pos):
                # Walk up item hierarchy to find a NodeItem (handles child text/port items)
                candidate = item
                while candidate is not None:
                    if isinstance(candidate, NodeItem):
                        node = getattr(candidate, 'node', None)
                        if isinstance(node, BaseNode):
                            return node
                    candidate = candidate.parentItem()
        except Exception as e:
            print(f"[LongPress] _node_at_view_pos error: {e}")
        return None

    def _node_from_scene_pos(self, scene_pos):
        """シーン座標でノードモデルを返す (CustomViewer から呼ばれる)。

        NodeItem には model への逆参照がないため、
        scene.items() で NodeItem を集め、graph.all_nodes() の .view と突き合わせる。
        """
        try:
            view = self.viewer()
            if view is None:
                return None
            # Collect NodeItem instances at this scene position
            hit_node_items: set = set()
            for item in view.scene().items(scene_pos):
                candidate = item
                while candidate is not None:
                    if isinstance(candidate, NodeItem):
                        hit_node_items.add(candidate)
                        break
                    candidate = candidate.parentItem()
            if not hit_node_items:
                return None
            # Match against graph nodes via node.view (model → NodeItem)
            for node in self.all_nodes():
                if getattr(node, 'view', None) in hit_node_items:
                    return node
        except Exception as e:
            print(f"[LongPress] _node_from_scene_pos error: {e}")
            import traceback; traceback.print_exc()
        return None

    def _start_long_press(self, scene_pos, viewport_pos):
        """CustomViewer.mousePressEvent から呼ばれる。ノード上ならタイマー起動。"""
        self._long_press_timer.stop()
        self._long_press_node = None
        self._long_press_start = viewport_pos
        node = self._node_from_scene_pos(scene_pos)
        if node is not None:
            self._long_press_node = node
            self._long_press_timer.start()
            print(f"[LongPress] Armed for node: {node.name()}")

    def _cancel_long_press_if_dragged(self, current_viewport_pos):
        """ドラッグ距離が閾値を超えたら長押しタイマーをキャンセル。"""
        if self._long_press_timer.isActive() and self._long_press_start is not None:
            delta = current_viewport_pos - self._long_press_start
            if delta.manhattanLength() > 5:
                self._long_press_timer.stop()
                self._long_press_node = None

    def _cancel_long_press(self):
        """CustomViewer.mouseReleaseEvent から呼ばれる。"""
        self._long_press_timer.stop()
        self._long_press_node = None
        self._long_press_start = None

    def _on_long_press_timeout(self):
        """1秒長押し確定: node_long_pressed を emit する。"""
        node = self._long_press_node
        self._long_press_node = None
        if node is not None:
            self.node_long_pressed.emit(node)

    # ---------------------------------------------------------------
    # Context menu (right-click / macOS Ctrl+left-click)
    # ---------------------------------------------------------------
    def _show_context_menu(self, scene_pos, global_pos):
        """CustomViewer から呼ばれる。カーソル位置にコンテキストメニューを表示する。

        - クリック位置にノードがあり、それが未選択なら単体選択に切り替える
        - 選択なしなら何もしない
        - Divide は「エッジで結ばれた PoseNode 2つ」のときだけ表示
        """
        try:
            clicked_node = self._node_from_scene_pos(scene_pos)
        except Exception:
            clicked_node = None

        if clicked_node is not None:
            try:
                if not clicked_node.selected():
                    for _n in self.selected_nodes():
                        _n.set_selected(False)
                    clicked_node.set_selected(True)
            except Exception as _e:
                print(f"[ContextMenu] selection sync failed: {_e}")

        selected = list(self.selected_nodes())
        if not selected:
            return

        # _copy_nodes / _duplicate_nodes / _collect_graph_name_set 等は
        # CustomViewer 側に実装されているのでビュー経由で呼ぶ
        viewer = self.viewer()

        menu = QtWidgets.QMenu(self._view)
        menu.setStyleSheet(_CONTEXT_MENU_STYLESHEET)
        act_copy = menu.addAction("Copy")
        act_dup = menu.addAction("Duplicate")
        act_del = menu.addAction("Delete")

        divide_pair = self._detect_divide_pair(selected)
        act_divide = None
        if divide_pair is not None:
            menu.addSeparator()
            act_divide = menu.addAction("Divide")

        chosen = menu.exec_(global_pos)
        if chosen is None:
            return

        try:
            if chosen == act_copy:
                if viewer and hasattr(viewer, '_copy_nodes'):
                    viewer._copy_nodes(selected)
            elif chosen == act_dup:
                if self._undo_push_fn:
                    self._undo_push_fn()
                if viewer and hasattr(viewer, '_duplicate_nodes'):
                    # 右下（+x, +y）にオフセットして複製
                    viewer._duplicate_nodes(selected, offset_x=80, offset_y=80)
            elif chosen == act_del:
                deletable = [n for n in selected if not isinstance(n, BaseLinkNode)]
                if not deletable:
                    print("[ContextMenu] No deletable nodes in selection")
                    return
                if self._undo_push_fn:
                    self._undo_push_fn()
                for _n in deletable:
                    self.remove_node(_n)
                print(f"[ContextMenu] Deleted {len(deletable)} node(s)")
            elif chosen == act_divide and divide_pair is not None:
                src, dst = divide_pair
                self._show_divide_dialog(src, dst)
        except Exception as _e:
            print(f"[ContextMenu] action failed: {_e}")
            import traceback
            traceback.print_exc()

    def _detect_divide_pair(self, selected):
        """選択が「エッジで結ばれた PoseNode 2つ」なら (元, 先) を返す。それ以外は None。"""
        if len(selected) != 2:
            return None
        a, b = selected
        if not (isinstance(a, PoseNode) and isinstance(b, PoseNode)):
            return None

        def _connected(src, dst):
            try:
                for out_port in src.output_ports():
                    for cp in out_port.connected_ports():
                        if cp.node() is dst:
                            return True
            except Exception:
                pass
            return False

        if _connected(a, b):
            return (a, b)
        if _connected(b, a):
            return (b, a)
        return None

    # ---------------------------------------------------------------
    # Divide: modal + execution
    # ---------------------------------------------------------------
    def _representative_easing_option(self, node):
        """ノードの joint_easings から最多の easing option 文字列を返す。"""
        easings = getattr(node, 'joint_easings', None) or {}
        if not easings:
            return EASING_OPTIONS[0]
        counts = {}
        for v in easings.values():
            try:
                key = easing_option(v)
            except Exception:
                key = EASING_OPTIONS[0]
            counts[key] = counts.get(key, 0) + 1
        # 最多を選ぶ（同数なら EASING_OPTIONS の並び順で先勝ち）
        best = None
        for opt in EASING_OPTIONS:
            if opt in counts:
                if best is None or counts[opt] > counts[best]:
                    best = opt
        return best or EASING_OPTIONS[0]

    def _show_divide_dialog(self, src, dst):
        """Divide モーダルを開き、OK なら _execute_divide を呼ぶ。"""
        initial = self._representative_easing_option(dst)
        dlg = DivideDialog(initial, parent=self._view.window() if self._view else None)
        if dlg.exec_() != QtWidgets.QDialog.Accepted:
            return
        n_new = dlg.get_split_count()
        chosen_easing = dlg.get_easing_option() if dlg.easing_user_changed() else None

        if self._undo_push_fn:
            self._undo_push_fn()
        self._execute_divide(src, dst, n_new, chosen_easing)

    def _execute_divide(self, src, dst, n_new, chosen_easing):
        """src → dst の間に n_new 枚の PoseNode を挿入する。

        - スライダー値: dst.joint_easings を各ジョイントで用い normalized time で補間
          （chosen_easing が指定されていれば代わりにそれを一律使用）
        - フレーム: dst.frames を (n_new+1) 個の区間に等分割。端数は dst に加算
          （合計は変わらない）
        - 配置: src → dst の直線上を等分割
        - 生成ノードの joint_easings: chosen_easing 指定時はそれで一律、無指定なら linear
        - 遷移先の joint_easings: chosen_easing 指定時のみ一律上書き
        """
        try:
            self._execute_divide_impl(src, dst, n_new, chosen_easing)
        except Exception as _e:
            print(f"[Divide] execution failed: {_e}")
            import traceback
            traceback.print_exc()

    def _execute_divide_impl(self, src, dst, n_new, chosen_easing):
        if not isinstance(src, PoseNode) or not isinstance(dst, PoseNode):
            print("[Divide] src/dst must be PoseNode")
            return
        n_new = int(max(1, n_new))
        print(f"[Divide] start: src={src.name()} dst={dst.name()} "
              f"n_new={n_new} chosen_easing={chosen_easing}")

        # フレーム分割
        total_frames = int(getattr(dst, 'frames', get_default_hz_fps()))
        segs = n_new + 1
        per_frames = total_frames // segs
        remainder = total_frames - per_frames * segs
        if per_frames <= 0:
            # 0 フレームでは意味がないので警告して中断
            QtWidgets.QMessageBox.warning(
                self._view.window() if self._view else None,
                "Divide",
                f"分割数 {n_new} が遷移先フレーム数 {total_frames} に対して大きすぎます。\n"
                f"各区間が 1 フレーム以上になる分割数を指定してください。"
            )
            return

        # 遷移元/遷移先位置
        _sp = src.pos()
        _dp = dst.pos()
        sx, sy = (_sp[0], _sp[1]) if isinstance(_sp, (list, tuple)) else (_sp.x(), _sp.y())
        dx, dy = (_dp[0], _dp[1]) if isinstance(_dp, (list, tuple)) else (_dp.x(), _dp.y())

        # 補間用: ジョイントごとの easing 解決
        dst_easings = dict(getattr(dst, 'joint_easings', {}))
        src_angles = dict(getattr(src, 'angles_deg', {}))
        dst_angles = dict(getattr(dst, 'angles_deg', {}))

        # 既存の src → dst エッジを削除
        try:
            for out_port in src.output_ports():
                for cp in list(out_port.connected_ports()):
                    if cp.node() is dst:
                        out_port.disconnect_from(cp)
        except Exception as _e:
            print(f"[Divide] disconnect src→dst failed: {_e}")

        # 名前生成のヘルパは CustomViewer 側に実装されているので viewer 経由で呼ぶ
        viewer = self.viewer()
        if viewer and hasattr(viewer, '_collect_graph_name_set'):
            taken_names = viewer._collect_graph_name_set()
        else:
            taken_names = set()
        src_pose_name = getattr(src, 'pose_name', src.name())

        # 新規ノード生成
        new_nodes = []
        for i in range(1, n_new + 1):
            t = float(i) / float(segs)  # 0 < t < 1
            pos_x = sx + (dx - sx) * t
            pos_y = sy + (dy - sy) * t

            new_node = self.create_node('motion.nodes.PoseNode')
            if new_node is None:
                print(f"[Divide] create_node failed at i={i}")
                continue
            new_node.set_pos(pos_x, pos_y)
            new_node.set_color(*NODE_COLOR_DEFAULT)
            # CustomViewer._next_unique_numbered_name は @staticmethod
            new_node.pose_name = CustomViewer._next_unique_numbered_name(src_pose_name, taken_names)
            new_node.set_name(new_node.pose_name)
            new_node.duration = getattr(src, 'duration', 1.0)
            new_node.frames = per_frames

            # スライダー値 = start + easing(t) * (end - start)
            # ここで使う easing は「補間の形状を決める入力」。
            # chosen_easing 指定時は一律その形、無指定時はジョイント毎の遷移先 easing を尊重。
            new_angles = {}
            for jname, dst_val in dst_angles.items():
                start_val = src_angles.get(jname, dst_val)
                if chosen_easing is not None:
                    joint_ease = chosen_easing
                else:
                    joint_ease = dst_easings.get(jname, EASING_OPTIONS[0])
                try:
                    eased_t = easing_value(joint_ease, t)
                except Exception:
                    eased_t = t
                new_angles[jname] = start_val + (dst_val - start_val) * eased_t
            new_node.angles_deg = new_angles

            # 分割によってイージングは "消費" され、区間ごとの遷移は直線化されるため
            # 生成ノードの easing は常に全ジョイント linear。
            new_node.joint_easings = {j: EASING_OPTIONS[0] for j in dst_angles.keys()}

            new_nodes.append(new_node)

        # 接続を張り直す: src → new1 → ... → newN → dst
        chain = [src] + new_nodes + [dst]
        for i in range(len(chain) - 1):
            a, b = chain[i], chain[i + 1]
            try:
                out_ports = a.output_ports()
                in_ports = b.input_ports()
                if out_ports and in_ports:
                    out_ports[0].connect_to(in_ports[0])
            except Exception as _e:
                print(f"[Divide] connect {a.name()}→{b.name()} failed: {_e}")

        # 遷移先のフレーム更新（端数を吸収）
        dst.frames = per_frames + remainder
        if remainder != 0:
            print(f"[Divide] frames not divisible: total={total_frames}, "
                  f"segs={segs}, per={per_frames}, remainder={remainder} → 遷移先に加算")

        # 遷移先の easing を確定
        # 分割によりイージングは "消費" され、遷移先までの区間も直線化されるため、
        # 遷移先の全ジョイントは常に linear にリセットする（遷移元だけが元の設定を保持）。
        desired_dst_easings = {j: EASING_OPTIONS[0] for j in dst_angles.keys()}
        dst.joint_easings = desired_dst_easings
        # dst.angles_deg は変更しないが、選択変更時のポーラーで書き戻される可能性があるので
        # スナップショットを取っておく
        desired_dst_angles = dict(dst_angles)

        # 生成ノードを選択状態にする
        try:
            for _n in list(self.selected_nodes()):
                _n.set_selected(False)
            for _n in new_nodes:
                _n.set_selected(True)
        except Exception:
            pass

        # 200ms周期のセレクション監視ポーラーが on_node_selection_changed → _save_to_node を
        # 呼び、ジョイントエディタ表示中のジョイント easings（欠損は linear 既定で埋められる）
        # を current_pose_node(=dst になり得る) に書き戻して dst.joint_easings が汚染される。
        # ポーラーが発火した後（>200ms）に狙って dst の joint_easings / angles_deg を復元する。
        def _restore_dst_state(_node=dst,
                                _easings=desired_dst_easings,
                                _angles=desired_dst_angles):
            try:
                _node.joint_easings = dict(_easings)
                _node.angles_deg = dict(_angles)
            except Exception as _err:
                print(f"[Divide] restore dst state failed: {_err}")

        # 複数回スケジュールしてタイミングブレを吸収
        QtCore.QTimer.singleShot(260, _restore_dst_state)
        QtCore.QTimer.singleShot(500, _restore_dst_state)

        print(f"[Divide] Inserted {len(new_nodes)} node(s) between "
              f"{src.name()} and {dst.name()} (per={per_frames}F, dst={dst.frames}F)")

    def custom_mouse_press(self, event):
        """カスタムマウスプレスイベントハンドラ"""
        try:
            # 左ボタンの処理
            if event.button() == QtCore.Qt.MouseButton.LeftButton:
                self._selection_start = event.position().toPoint()
                self._is_selecting = True

                # ラバーバンドの設定
                if self._rubber_band:
                    rect = QtCore.QRect(self._selection_start, QtCore.QSize())
                    self._rubber_band.setGeometry(rect)
                    self._rubber_band.show()

                # Ctrlキーが押されていない場合は選択をクリア
                if not event.modifiers() & QtCore.Qt.KeyboardModifier.ControlModifier:
                    for node in self.selected_nodes():
                        node.set_selected(False)

            # オリジナルのイベントハンドラを呼び出し
            self._original_handlers['press'](event)

        except Exception as e:
            print(f"Error in mouse press: {str(e)}")

    def custom_mouse_move(self, event):
        """カスタムマウス移動イベントハンドラ"""
        try:
            if self._is_selecting and self._selection_start:
                current_pos = event.position().toPoint()
                rect = QtCore.QRect(self._selection_start,
                                    current_pos).normalized()
                if self._rubber_band:
                    self._rubber_band.setGeometry(rect)

            # オリジナルのイベントハンドラを呼び出し
            self._original_handlers['move'](event)

        except Exception as e:
            print(f"Error in mouse move: {str(e)}")

    def _widget_rect_to_scene_rect(self, rect):
        """ラバーバンドの矩形（widget座標）をシーン座標に変換。_view は NodeGraphWidget のため mapToScene は viewer で行う"""
        view = self.viewer()
        if view is self._view:
            return view.mapToScene(rect).boundingRect()
        tl = view.viewport().mapFrom(self._view, rect.topLeft())
        br = view.viewport().mapFrom(self._view, rect.bottomRight())
        viewport_rect = QtCore.QRect(tl, br).normalized()
        return view.mapToScene(viewport_rect).boundingRect()

    def custom_mouse_release(self, event):
        """カスタムマウスリリースイベントハンドラ"""
        try:
            if event.button() == QtCore.Qt.MouseButton.LeftButton and self._is_selecting:
                if self._rubber_band and self._selection_start:
                    # 選択範囲の処理
                    rect = self._rubber_band.geometry()
                    scene_rect = self._widget_rect_to_scene_rect(rect)

                    # 範囲内のノードを選択
                    for node in self.all_nodes():
                        node_pos = node.pos()
                        if isinstance(node_pos, (list, tuple)):
                            node_point = QtCore.QPointF(
                                node_pos[0], node_pos[1])
                        else:
                            node_point = node_pos

                        if scene_rect.contains(node_point):
                            node.set_selected(True)

                    # ラバーバンドを隠す
                    self._rubber_band.hide()

                # 選択状態をリセット
                self._selection_start = None
                self._is_selecting = False

            # オリジナルのイベントハンドラを呼び出し
            self._original_handlers['release'](event)

        except Exception as e:
            print(f"Error in mouse release: {str(e)}")

    def cleanup(self):
        """リソースのクリーンアップ"""
        # cleanup()はaboutToQuit経由の明示呼び出しと__del__の両方から呼ばれ得る。
        # 2回目以降は既に解放済みのノード/インスペクタウィンドウ(C++側は削除済み)に
        # 触れてネイティブクラッシュを起こし得るため、一度だけ実行するようにガードする。
        if getattr(self, '_cleanup_done', False):
            return
        self._cleanup_done = True

        try:
            print("Starting cleanup process...")
            
            # イベントハンドラの復元
            if hasattr(self, '_view') and self._view:
                if hasattr(self, '_original_handlers'):
                    self._view.mousePressEvent = self._original_handlers['press']
                    self._view.mouseMoveEvent = self._original_handlers['move']
                    self._view.mouseReleaseEvent = self._original_handlers['release']
                    print("Restored original event handlers")

            # ラバーバンドのクリーンアップ
            try:
                if hasattr(self, '_rubber_band') and self._rubber_band and not self._rubber_band.isHidden():
                    self._rubber_band.hide()
                    self._rubber_band.setParent(None)
                    self._rubber_band.deleteLater()
                    self._rubber_band = None
                    print("Cleaned up rubber band")
            except Exception as e:
                print(f"Warning: Rubber band cleanup - {str(e)}")
                
            # ノードのクリーンアップ
            for node in self.all_nodes():
                try:
                    # STLデータのクリーンアップ
                    if self.stl_viewer:
                        self.stl_viewer.remove_stl_for_node(node)
                    # ノードの削除
                    self.remove_node(node)
                except Exception as e:
                    print(f"Error cleaning up node: {str(e)}")

            # インスペクタウィンドウのクリーンアップ
            if hasattr(self, 'inspector_window') and self.inspector_window:
                try:
                    self.inspector_window.close()
                    self.inspector_window.deleteLater()
                    self.inspector_window = None
                    print("Cleaned up inspector window")
                except Exception as e:
                    print(f"Error cleaning up inspector window: {str(e)}")

            # キャッシュのクリア
            try:
                self._cached_positions.clear()
                self._selection_cache.clear()
                if hasattr(self, '_cleanup_handlers'):
                    self._cleanup_handlers.clear()
                print("Cleared caches")
            except Exception as e:
                print(f"Error clearing caches: {str(e)}")

            print("Cleanup process completed")

        except Exception as e:
            print(f"Error during cleanup: {str(e)}")

    def __del__(self):
        """デストラクタでクリーンアップを実行"""
        self.cleanup()

    def remove_node(self, node):
        """ノード削除時のメモリリーク対策"""
        # キャッシュからノード関連データを削除
        if node in self._cached_positions:
            del self._cached_positions[node]
        self._selection_cache.discard(node)

        # Remove pipes from scene before disconnecting ports
        self._remove_pipes_for_node(node)

        # ポート接続の解除
        for port in node.input_ports():
            for connected_port in list(port.connected_ports()):
                self.disconnect_ports(port, connected_port)

        for port in node.output_ports():
            for connected_port in list(port.connected_ports()):
                self.disconnect_ports(port, connected_port)

        # STLデータのクリーンアップ
        if self.stl_viewer:
            self.stl_viewer.remove_stl_for_node(node)

        super(CustomNodeGraph, self).remove_node(node)

    def _remove_pipes_for_node(self, node):
        """Remove all pipes connected to a node from the scene."""
        try:
            viewer = self.viewer()
            if not viewer:
                return
            scene = viewer.scene()
            if not scene:
                return

            # Get the node's view item for comparison
            node_view = node.view if hasattr(node, 'view') else None

            # Find and remove pipes connected to this node
            items_to_remove = []
            for item in scene.items():
                if isinstance(item, (CustomPipe, PipeItem)):
                    if isinstance(item, CustomLivePipe):
                        continue
                    # Check if pipe is connected to this node
                    input_port = getattr(item, 'input_port', None)
                    output_port = getattr(item, 'output_port', None)

                    # Check by port's node reference
                    should_remove = False
                    if input_port:
                        port_node = getattr(input_port, 'node', None)
                        if port_node is node_view or port_node is node:
                            should_remove = True
                    if output_port:
                        port_node = getattr(output_port, 'node', None)
                        if port_node is node_view or port_node is node:
                            should_remove = True

                    if should_remove:
                        items_to_remove.append(item)

            for item in items_to_remove:
                scene.removeItem(item)

            if items_to_remove:
                scene.update()
        except Exception as e:
            print(f"[remove_node] Error removing pipes: {e}")

    def optimize_node_positions(self):
        """ノード位置の計算を最適化"""
        # 位置計算のキャッシュを活用
        for node in self.all_nodes():
            if node not in self._cached_positions:
                pos = self.calculate_node_position(node)
                self._cached_positions[node] = pos
            node.set_pos(*self._cached_positions[node])

    def setup_custom_view(self):
        """ビューのイベントハンドラをカスタマイズ"""
        # オリジナルのイベントハンドラを保存
        self._view.mousePressEvent_original = self._view.mousePressEvent
        self._view.mouseMoveEvent_original = self._view.mouseMoveEvent
        self._view.mouseReleaseEvent_original = self._view.mouseReleaseEvent
        
        # 新しいイベントハンドラを設定
        self._view.mousePressEvent = lambda event: self._view_mouse_press(event)
        self._view.mouseMoveEvent = lambda event: self._view_mouse_move(event)
        self._view.mouseReleaseEvent = lambda event: self._view_mouse_release(event)

    def eventFilter(self, obj, event):
        """イベントフィルターでマウスイベントを処理"""
        if obj is self._view:
            if event.type() == QtCore.QEvent.Type.MouseButtonPress:
                return self._handle_mouse_press(event)
            elif event.type() == QtCore.QEvent.Type.MouseMove:
                return self._handle_mouse_move(event)
            elif event.type() == QtCore.QEvent.Type.MouseButtonRelease:
                return self._handle_mouse_release(event)
        
        return super(CustomNodeGraph, self).eventFilter(obj, event)

    def _handle_mouse_press(self, event):
        """マウスプレスイベントの処理"""
        if event.button() == QtCore.Qt.MouseButton.LeftButton:
            self._selection_start = event.position().toPoint()
            self._is_selecting = True

            # 選択範囲の設定
            if self._rubber_band:
                rect = QtCore.QRect(self._selection_start, QtCore.QSize())
                self._rubber_band.setGeometry(rect)
                self._rubber_band.show()

            # Ctrlキーが押されていない場合は既存の選択をクリア
            if not event.modifiers() & QtCore.Qt.KeyboardModifier.ControlModifier:
                for node in self.selected_nodes():
                    node.set_selected(False)

        return False  # イベントを伝播させる

    def _handle_mouse_move(self, event):
        """マウス移動イベントの処理"""
        if self._is_selecting and self._selection_start is not None and self._rubber_band:
            current_pos = event.position().toPoint()
            rect = QtCore.QRect(self._selection_start,
                                current_pos).normalized()
            self._rubber_band.setGeometry(rect)

        return False  # イベントを伝播させる

    def _handle_mouse_release(self, event):
        """マウスリリースイベントの処理"""
        if (event.button() == QtCore.Qt.MouseButton.LeftButton and
                self._is_selecting and self._rubber_band):
            try:
                # 選択範囲の取得
                rect = self._rubber_band.geometry()
                scene_rect = self._widget_rect_to_scene_rect(rect)

                # 範囲内のノードを選択
                for node in self.all_nodes():
                    node_pos = node.pos()
                    if isinstance(node_pos, (list, tuple)):
                        node_point = QtCore.QPointF(node_pos[0], node_pos[1])
                    else:
                        node_point = node_pos

                    if scene_rect.contains(node_point):
                        node.set_selected(True)

                # ラバーバンドを隠す
                self._rubber_band.hide()

            except Exception as e:
                print(f"Error in mouse release: {str(e)}")
            finally:
                # 状態をリセット
                self._selection_start = None
                self._is_selecting = False

        return False  # イベントを伝播させる

    def mousePressEvent(self, event):
        if event.button() == QtCore.Qt.LeftButton:
            # ビューの座標系でマウス位置を取得
            view = self.scene().views()[0]
            self._selection_start = view.mapFromGlobal(event.globalPos())
            
            # Ctrlキーが押されていない場合は既存の選択をクリア
            if not event.modifiers() & QtCore.Qt.ControlModifier:
                for node in self.selected_nodes():
                    node.set_selected(False)
            
            # ラバーバンドの開始位置を設定
            self._rubber_band.setGeometry(QtCore.QRect(self._selection_start, QtCore.QSize()))
            self._rubber_band.show()
        
        super(CustomNodeGraph, self).mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._selection_start is not None:
            # ビューの座標系で現在位置を取得
            view = self.scene().views()[0]
            current_pos = view.mapFromGlobal(event.globalPos())
            
            # ラバーバンドの領域を更新
            rect = QtCore.QRect(self._selection_start, current_pos).normalized()
            self._rubber_band.setGeometry(rect)
        
        super(CustomNodeGraph, self).mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if event.button() == QtCore.Qt.LeftButton and self._selection_start is not None:
            # ビューの座標系でラバーバンドの領域を取得
            view = self.scene().views()[0]
            rubber_band_rect = self._rubber_band.geometry()
            scene_rect = view.mapToScene(rubber_band_rect).boundingRect()
            
            # 範囲内のノードを選択
            for node in self.all_nodes():
                node_center = QtCore.QPointF(node.pos()[0], node.pos()[1])
                if scene_rect.contains(node_center):
                    node.set_selected(True)
            
            # ラバーバンドをクリア
            self._rubber_band.hide()
            self._selection_start = None
        
        super(CustomNodeGraph, self).mouseReleaseEvent(event)

    def _view_mouse_press(self, event):
        """ビューのマウスプレスイベント"""
        if event.button() == QtCore.Qt.MouseButton.LeftButton:
            self._selection_start = event.position().toPoint()
            self._is_selecting = True

            # 選択範囲の設定
            if self._rubber_band:
                rect = QtCore.QRect(self._selection_start, QtCore.QSize())
                self._rubber_band.setGeometry(rect)
                self._rubber_band.show()

            # Ctrlキーが押されていない場合は既存の選択をクリア
            if not event.modifiers() & QtCore.Qt.KeyboardModifier.ControlModifier:
                for node in self.selected_nodes():
                    node.set_selected(False)

        # 元のイベントハンドラを呼び出し
        if hasattr(self._view, 'mousePressEvent_original'):
            self._view.mousePressEvent_original(event)

    def _view_mouse_move(self, event):
        """ビューのマウス移動イベント"""
        if self._is_selecting and self._selection_start is not None and self._rubber_band:
            current_pos = event.position().toPoint()
            rect = QtCore.QRect(self._selection_start,
                                current_pos).normalized()
            self._rubber_band.setGeometry(rect)

        # 元のイベントハンドラを呼び出し
        if hasattr(self._view, 'mouseMoveEvent_original'):
            self._view.mouseMoveEvent_original(event)

    def _view_mouse_release(self, event):
        """ビューのマウスリリースイベント"""
        if (event.button() == QtCore.Qt.MouseButton.LeftButton and
                self._is_selecting and self._rubber_band):
            try:
                # 選択範囲の取得
                rect = self._rubber_band.geometry()
                scene_rect = self._widget_rect_to_scene_rect(rect)

                # 範囲内のノードを選択
                for node in self.all_nodes():
                    node_pos = node.pos()
                    if isinstance(node_pos, (list, tuple)):
                        node_point = QtCore.QPointF(node_pos[0], node_pos[1])
                    else:
                        node_point = node_pos

                    if scene_rect.contains(node_point):
                        node.set_selected(True)

                # ラバーバンドを隠す
                self._rubber_band.hide()

            except Exception as e:
                print(f"Error in mouse release: {str(e)}")
            finally:
                # 状態をリセット
                self._selection_start = None
                self._is_selecting = False

        # 元のイベントハンドラを呼び出し
        if hasattr(self._view, 'mouseReleaseEvent_original'):
            self._view.mouseReleaseEvent_original(event)

    def create_base_link(self):
        """初期のbase_linkノードを作成"""
        try:
            node_type = f"{BaseLinkNode.__identifier__}.{BaseLinkNode.NODE_NAME}"
            base_node = self.create_node(node_type)
            base_node.set_name('start')
            base_node.set_pos(self.snap_to_grid(50), self.snap_to_grid(50))
            # ライトグレーに設定（背景より少し濃い）
            base_node.set_color(*NODE_COLOR_DEFAULT)
            print("Base Link node created successfully")
            return base_node
        except Exception as e:
            print(f"Error creating base link node: {str(e)}")
            import traceback
            traceback.print_exc()
            raise

    def register_nodes(self, node_classes):
        """複数のノードクラスを一度に登録"""
        for node_class in node_classes:
            self.register_node(node_class)
            print(f"Registered node type: {node_class.__identifier__}")

    def on_port_connected(self, input_port, output_port):
        """ポートが接続された時の処理"""
        if self._undo_push_fn:
            self._undo_push_fn()
        print(f"**Connecting port: {output_port.name()}")
        
        # 接続情報の出力
        parent_node = output_port.node()
        child_node = input_port.node()
        print(f"Parent node: {parent_node.name()}, Child node: {child_node.name()}")
        
        try:
            # 全ノードの位置を再計算
            print("Recalculating all node positions after connection...")
            self.recalculate_all_positions()
            
        except Exception as e:
            print(f"Error in port connection: {str(e)}")
            print(f"Detailed connection information:")
            print(f"  Output port: {output_port.name()} from {parent_node.name()}")
            print(f"  Input port: {input_port.name()} from {child_node.name()}")
            traceback.print_exc()

    def on_port_disconnected(self, input_port, output_port):
        """ポートが切断された時の処理"""
        if self._undo_push_fn:
            self._undo_push_fn()
        child_node = input_port.node()  # 入力ポートを持つノードが子
        parent_node = output_port.node()  # 出力ポートを持つノードが親
        
        print(f"\nDisconnecting ports:")
        print(f"Parent node: {parent_node.name()}, Child node: {child_node.name()}")
        
        try:
            # 子ノードの位置をリセット
            if hasattr(child_node, 'current_transform'):
                del child_node.current_transform
            
            # STLの位置をリセット
            self.stl_viewer.reset_stl_transform(child_node)
            print(f"Reset position for node: {child_node.name()}")

            # 全ノードの位置を再計算
            print("Recalculating all node positions after disconnection...")
            self.recalculate_all_positions()

        except Exception as e:
            print(f"Error in port disconnection: {str(e)}")
            traceback.print_exc()

    def update_robot_name(self, text):
        """ロボット名を更新するメソッド"""
        self.robot_name = text
        print(f"Robot name updated to: {text}")

        # 必要に応じて追加の処理
        # 例：ウィンドウタイトルの更新
        if hasattr(self, 'widget') and self.widget:
            if self.widget.window():
                title = f"LegacyMotionEditor v{_LME_VERSION} {text}"
                self.widget.window().setWindowTitle(title)

    def get_node_by_name(self, name):
        for node in self.all_nodes():
            if node.name() == name:
                return node
        return None

    def show_inspector(self, node, screen_pos=None):
        """
        ノードのインスペクタウィンドウを表示
        """
        try:
            # 既存のインスペクタウィンドウをクリーンアップ
            if hasattr(self, 'inspector_window') and self.inspector_window is not None:
                try:
                    self.inspector_window.close()
                    self.inspector_window.deleteLater()
                except:
                    pass
                self.inspector_window = None

            # 新しいインスペクタウィンドウを作成
            self.inspector_window = InspectorWindow(stl_viewer=self.stl_viewer)
            
            # ウィンドウサイズを取得
            inspector_size = self.inspector_window.sizeHint()

            if self.widget and self.widget.window():
                # 保存された位置があればそれを使用し、なければデフォルト位置を計算
                if hasattr(self, 'last_inspector_position') and self.last_inspector_position:
                    x = self.last_inspector_position.x()
                    y = self.last_inspector_position.y()
                    
                    # スクリーンの情報を取得して位置を検証
                    screen = QtWidgets.QApplication.primaryScreen()
                    screen_geo = screen.availableGeometry()
                    
                    # 画面外にはみ出していないか確認
                    if x < screen_geo.x() or x + inspector_size.width() > screen_geo.right() or \
                    y < screen_geo.y() or y + inspector_size.height() > screen_geo.bottom():
                        # 画面外の場合はデフォルト位置を使用
                        main_geo = self.widget.window().geometry()
                        x = main_geo.x() + (main_geo.width() - inspector_size.width()) // 2
                        y = main_geo.y() + 50
                else:
                    # デフォルトの位置を計算
                    main_geo = self.widget.window().geometry()
                    x = main_geo.x() + (main_geo.width() - inspector_size.width()) // 2
                    y = main_geo.y() + 50

                # ウィンドウの初期設定と表示
                self.inspector_window.setWindowTitle(f"Node Inspector - {node.name()}")
                self.inspector_window.current_node = node
                self.inspector_window.graph = self
                self.inspector_window.update_info(node)
                
                self.inspector_window.move(x, y)
                self.inspector_window.show()
                self.inspector_window.raise_()
                self.inspector_window.activateWindow()

                print(f"Inspector window displayed for node: {node.name()}")

        except Exception as e:
            print(f"Error showing inspector: {str(e)}")
            traceback.print_exc()

    def open_joint_sliders_for_pose(self, node, screen_pos=None):
        """Pose ダブルクリックで Joint Sliders を開き、該当 Pose を編集対象にする。"""
        je = getattr(self, "joint_editor", None)
        if not je:
            return
        try:
            je.show_for_pose_node(node, screen_pos)
            je.show()
            je.raise_()
            je.activateWindow()
        except Exception as e:
            print(f"Error opening Joint Sliders for pose: {e}")
            traceback.print_exc()

    def show_define_editor(self, node, screen_pos=None):
        """DefineNode の編集モーダルを表示"""
        try:
            # User Value モーダルと同じく QMainWindow を親にする（graph 内 widget を親にすると見た目が崩れる）
            _dlg_parent = self.widget.window()
            dlg = AddDefineShellDialog(self, node, parent=_dlg_parent)
            if screen_pos:
                dlg.move(screen_pos)
            dlg.exec()
        except Exception as e:
            print(f"Error showing define editor: {e}")
            traceback.print_exc()

    def show_wait_editor(self, node, screen_pos=None):
        """WaitNode の編集モーダルを表示"""
        try:
            _dlg_parent = self.widget.window()
            fps = getattr(self, "_wait_editor_fps", 100)
            dlg = WaitEditDialog(self, node, fps=fps, parent=_dlg_parent)
            if screen_pos:
                dlg.move(screen_pos)
            dlg.exec()
        except Exception as e:
            print(f"Error showing wait editor: {e}")
            traceback.print_exc()

    def show_memo_editor(self, node, screen_pos=None):
        """MemoNode のテキスト編集モーダルを表示"""
        try:
            _dlg_parent = self.widget.window()
            dlg = MemoEditDialog(node, parent=_dlg_parent)
            if screen_pos:
                dlg.move(screen_pos)
            dlg.exec()
        except Exception as e:
            print(f"Error showing memo editor: {e}")
            traceback.print_exc()

    def show_branching_editor(self, node, screen_pos=None):
        """BranchingNode の分岐編集モーダル（Pose の Branching 相当）"""
        try:
            _dlg_parent = self.widget.window()
            dlg = BranchingShellDialog(self, node, parent=_dlg_parent)
            if screen_pos:
                dlg.move(screen_pos)
            dlg.exec()
        except Exception as e:
            print(f"Error showing branching editor: {e}")
            traceback.print_exc()

    def show_jump_editor(self, node, screen_pos=None):
        """JumpNode のジャンプ先アクション設定"""
        try:
            _dlg_parent = self.widget.window()
            dlg = JumpEditDialog(self, node, parent=_dlg_parent)
            if screen_pos:
                dlg.move(screen_pos)
            dlg.exec()
        except Exception as e:
            print(f"Error showing jump editor: {e}")
            traceback.print_exc()

    def show_command_editor(self, node, screen_pos=None):
        """CommandNode の編集ウィンドウを表示"""
        ce = getattr(self, "command_editor", None)
        if not ce:
            return
        try:
            ce.set_command_node(node)
            ce.show()
            ce.raise_()
            ce.activateWindow()
            if screen_pos:
                ce.move(screen_pos)
        except Exception as e:
            print(f"Error showing command editor: {e}")
            traceback.print_exc()

    def show_mix_editor(self, node, screen_pos=None):
        """MixNode の編集ウィンドウを表示"""
        me = getattr(self, "mix_editor", None)
        if not me:
            return
        try:
            me.set_mix_node(node)
            me.show()
            me.raise_()
            me.activateWindow()
            if screen_pos:
                me.move(screen_pos)
        except Exception as e:
            print(f"Error showing mix editor: {e}")
            traceback.print_exc()

    def remove_node(self, node):
        self.stl_viewer.remove_stl_for_node(node)
        # NodeGraphQt の Port.connected_ports() は接続相手のポート名が dict に
        # 見つからないと KeyError で例外を投げ、削除が中断してしまう。
        # 空名 (add_output('')) ポートや古いプロジェクトの整合性ずれで発生する。
        # ここで壊れた接続エントリを事前に除去してから super に委譲する。
        try:
            self._sanitize_node_connections(node)
        except Exception:
            pass
        super(CustomNodeGraph, self).remove_node(node)

    def _sanitize_node_connections(self, node):
        """削除対象ノードの接続情報から、相手側に存在しないポート名を除去する。"""
        try:
            all_ports = list(node.input_ports()) + list(node.output_ports())
        except Exception:
            return
        for p in all_ports:
            try:
                is_in = p.type_() == "in"
            except Exception:
                is_in = None
            try:
                conn_map = p.model.connected_ports
            except Exception:
                continue
            for other_id in list(conn_map.keys()):
                other = self.get_node_by_id(other_id)
                if other is None:
                    del conn_map[other_id]
                    continue
                try:
                    valid_names = set((other.outputs() if is_in else other.inputs()).keys())
                except Exception:
                    valid_names = None
                if valid_names is None:
                    continue
                kept = [n for n in conn_map[other_id] if n in valid_names]
                if not kept:
                    del conn_map[other_id]
                else:
                    conn_map[other_id] = kept

    def snap_to_grid(self, value):
        """値をグリッドにスナップ"""
        return round(value / NODE_GRAPH_GRID_SNAP_SIZE) * NODE_GRAPH_GRID_SNAP_SIZE

    def create_node(self, node_type, name=None, pos=None, skip_auto_position=False):
        new_node = super(CustomNodeGraph, self).create_node(node_type, name)

        if pos is None:
            pos = QPointF(0, 0)
        elif isinstance(pos, (tuple, list)):
            pos = QPointF(*pos)

        print(f"Initial position for new node: {pos}")  # デバッグ情報

        if skip_auto_position:
            # 自動配置をスキップして、指定された位置をそのまま使用
            adjusted_pos = pos
            print(f"Skipping auto-position, using specified position: {adjusted_pos}")
        else:
            adjusted_pos = self.find_non_overlapping_position(pos)
            print(f"Adjusted position for new node: {adjusted_pos}")  # デバッグ情報

        # グリッドにスナップ
        snapped_x = self.snap_to_grid(adjusted_pos.x())
        snapped_y = self.snap_to_grid(adjusted_pos.y())
        new_node.set_pos(snapped_x, snapped_y)

        # ポート位置を再配置（矢印が正しく表示されるように）
        if hasattr(new_node, 'view') and new_node.view is not None:
            view = new_node.view
            # ジオメトリ変更を通知してから再描画
            view.prepareGeometryChange()
            view._set_base_size()
            view._ports_aligned_after_move = True
            view._align_ports(0.0)
            # 各ポートのジオメトリ変更も通知し、接続パイプも再描画
            for port in list(view.inputs) + list(view.outputs):
                port.prepareGeometryChange()
                port.update()
                # 接続されているパイプがあれば再描画
                port.redraw_connected_pipes()
            view.update()
            # シーン全体を更新
            if view.scene():
                view.scene().update()

        return new_node

    def find_non_overlapping_position(self, pos, offset_x=50, offset_y=30, items_per_row=16):
        all_nodes = self.all_nodes()
        current_node_count = len(all_nodes)
        
        # 現在の行を計算
        row = current_node_count // items_per_row
        
        # 行内での位置を計算
        position_in_row = current_node_count % items_per_row
        
        # 基準となるX座標を計算（各行の開始X座標）
        base_x = pos.x()
        
        # 基準となるY座標を計算
        # 新しい行は前の行の開始位置から200ポイント下
        base_y = pos.y() + (row * 200)
        
        # 現在のノードのX,Y座標を計算
        new_x = base_x + (position_in_row * offset_x)
        new_y = base_y + (position_in_row * offset_y)
        
        new_pos = QPointF(new_x, new_y)
        
        print(f"Positioning node {current_node_count + 1}")
        print(f"Row: {row + 1}, Position in row: {position_in_row + 1}")
        print(f"Position: ({new_pos.x()}, {new_pos.y()})")
        
        # オーバーラップチェックと位置の微調整
        iteration = 0
        while any(self.nodes_overlap(new_pos, node.pos()) for node in all_nodes):
            new_pos += QPointF(5, 5)  # 微小なオフセットで調整
            iteration += 1
            if iteration > 10:
                break
        
        return new_pos

    def nodes_overlap(self, pos1, pos2, threshold=5):
        pos1 = self.ensure_qpointf(pos1)
        pos2 = self.ensure_qpointf(pos2)
        overlap = (abs(pos1.x() - pos2.x()) < threshold and
                abs(pos1.y() - pos2.y()) < threshold)
        # デバッグ出力を条件付きに
        if overlap:
            print(f"Overlap detected: pos1={pos1}, pos2={pos2}")
        return overlap

    def ensure_qpointf(self, pos):
        if isinstance(pos, QPointF):
            return pos
        elif isinstance(pos, (tuple, list)):
            return QPointF(*pos)
        else:
            print(f"Warning: Unsupported position type: {type(pos)}")  # デバッグ情報
            return QPointF(0, 0)  # デフォルト値を返す

    def _save_node_data(self, node, project_dir):
        """ノードデータの保存"""
        print(f"\nStarting _save_node_data for node: {node.name()}")
        node_elem = ET.Element("node")
        
        try:
            # 基本情報
            print(f"  Saving basic info for node: {node.name()}")
            ET.SubElement(node_elem, "id").text = hex(id(node))
            ET.SubElement(node_elem, "name").text = node.name()
            ET.SubElement(node_elem, "type").text = node.__class__.__name__

            # output_count の保存
            if hasattr(node, 'output_count'):
                ET.SubElement(node_elem, "output_count").text = str(node.output_count)
                print(f"  Saved output_count: {node.output_count}")

            # STLファイル情報
            if hasattr(node, 'stl_file') and node.stl_file:
                print(f"  Processing STL file for node {node.name()}: {node.stl_file}")
                stl_elem = ET.SubElement(node_elem, "stl_file")
                
                try:
                    stl_path = os.path.abspath(node.stl_file)
                    print(f"    Absolute STL path: {stl_path}")

                    if self.meshes_dir and stl_path.startswith(self.meshes_dir):
                        rel_path = os.path.relpath(stl_path, self.meshes_dir)
                        stl_elem.set('base_dir', 'meshes')
                        stl_elem.text = os.path.join('meshes', rel_path)
                        print(f"    Using meshes relative path: {rel_path}")
                    else:
                        rel_path = os.path.relpath(stl_path, project_dir)
                        stl_elem.set('base_dir', 'project')
                        stl_elem.text = rel_path
                        print(f"    Using project relative path: {rel_path}")

                except Exception as e:
                    print(f"    Error processing STL file: {str(e)}")
                    stl_elem.set('error', str(e))

            # 位置情報
            pos = node.pos()
            pos_elem = ET.SubElement(node_elem, "position")
            if isinstance(pos, (list, tuple)):
                ET.SubElement(pos_elem, "x").text = str(pos[0])
                ET.SubElement(pos_elem, "y").text = str(pos[1])
            else:
                ET.SubElement(pos_elem, "x").text = str(pos.x())
                ET.SubElement(pos_elem, "y").text = str(pos.y())

            # 物理プロパティ
            if hasattr(node, 'volume_value'):
                ET.SubElement(node_elem, "volume").text = str(node.volume_value)
                print(f"  Saved volume: {node.volume_value}")

            if hasattr(node, 'mass_value'):
                ET.SubElement(node_elem, "mass").text = str(node.mass_value)
                print(f"  Saved mass: {node.mass_value}")

            # 慣性テンソル
            if hasattr(node, 'inertia'):
                inertia_elem = ET.SubElement(node_elem, "inertia")
                for key, value in node.inertia.items():
                    inertia_elem.set(key, str(value))
                print("  Saved inertia tensor")

            # 色情報
            if hasattr(node, 'node_color'):
                color_elem = ET.SubElement(node_elem, "color")
                color_elem.text = ' '.join(map(str, node.node_color))
                print(f"  Saved color: {node.node_color}")

            # 回転軸
            if hasattr(node, 'rotation_axis'):
                ET.SubElement(node_elem, "rotation_axis").text = str(node.rotation_axis)
                print(f"  Saved rotation axis: {node.rotation_axis}")

            # Massless Decoration
            if hasattr(node, 'massless_decoration'):
                ET.SubElement(node_elem, "massless_decoration").text = str(node.massless_decoration)
                print(f"  Saved massless_decoration: {node.massless_decoration}")

            # ポイントデータ
            if hasattr(node, 'points'):
                points_elem = ET.SubElement(node_elem, "points")
                for i, point in enumerate(node.points):
                    point_elem = ET.SubElement(points_elem, "point")
                    point_elem.set('index', str(i))
                    ET.SubElement(point_elem, "name").text = point['name']
                    ET.SubElement(point_elem, "type").text = point['type']
                    ET.SubElement(point_elem, "xyz").text = ' '.join(map(str, point['xyz']))
                print(f"  Saved {len(node.points)} points")

            # 累積座標
            if hasattr(node, 'cumulative_coords'):
                coords_elem = ET.SubElement(node_elem, "cumulative_coords")
                for coord in node.cumulative_coords:
                    coord_elem = ET.SubElement(coords_elem, "coord")
                    ET.SubElement(coord_elem, "point_index").text = str(coord['point_index'])
                    ET.SubElement(coord_elem, "xyz").text = ' '.join(map(str, coord['xyz']))
                print(f"  Saved cumulative coordinates")

            print(f"  Completed saving node data for: {node.name()}")
            return node_elem

        except Exception as e:
            print(f"ERROR in _save_node_data for node {node.name()}: {str(e)}")
            traceback.print_exc()
            raise

    def save_project(self, file_path=None):
        """プロジェクトの保存（循環参照対策版）"""
        print("\n=== Starting Project Save ===")
        try:
            # STLビューアの状態を一時的にバックアップ
            stl_viewer_state = None
            if hasattr(self, 'stl_viewer'):
                print("Backing up STL viewer state...")
                stl_viewer_state = {
                    'actors': dict(self.stl_viewer.stl_actors),
                    'transforms': dict(self.stl_viewer.transforms)
                }
                # STLビューアの参照を一時的にクリア
                self.stl_viewer.stl_actors.clear()
                self.stl_viewer.transforms.clear()

            # ファイルパスの取得
            if not file_path:
                default_filename = f"urdf_pj_{datetime.datetime.now().strftime('%Y%m%d%H%M')}.xml"
                default_dir = self.last_save_dir or self.meshes_dir or os.getcwd()
                file_path, _ = QtWidgets.QFileDialog.getSaveFileName(
                    None,
                    "Save Project",
                    os.path.join(default_dir, default_filename),
                    "XML Files (*.xml)"
                )
                if not file_path:
                    print("Save cancelled by user")
                    return False

            self.project_dir = os.path.dirname(os.path.abspath(file_path))
            self.last_save_dir = self.project_dir
            print(f"Project will be saved to: {file_path}")

            # XMLツリーの作成
            print("Creating XML structure...")
            root = ET.Element("project")
            
            # ロボット名の保存
            robot_name_elem = ET.SubElement(root, "robot_name")
            robot_name_elem.text = self.robot_name
            print(f"Saving robot name: {self.robot_name}")
            
            if self.meshes_dir:
                try:
                    meshes_rel_path = os.path.relpath(self.meshes_dir, self.project_dir)
                    ET.SubElement(root, "meshes_directory").text = meshes_rel_path
                    print(f"Added meshes directory reference: {meshes_rel_path}")
                except ValueError:
                    ET.SubElement(root, "meshes_directory").text = self.meshes_dir
                    print(f"Added absolute meshes path: {self.meshes_dir}")

            # ノード情報の保存
            print("\nSaving nodes...")
            nodes_elem = ET.SubElement(root, "nodes")
            total_nodes = len(self.all_nodes())
            
            for i, node in enumerate(self.all_nodes(), 1):
                print(f"Processing node {i}/{total_nodes}: {node.name()}")
                # 一時的にSTLビューアの参照を削除
                stl_viewer_backup = node.stl_viewer if hasattr(node, 'stl_viewer') else None
                if hasattr(node, 'stl_viewer'):
                    delattr(node, 'stl_viewer')
                
                node_elem = self._save_node_data(node, self.project_dir)
                nodes_elem.append(node_elem)
                
                # STLビューアの参照を復元
                if stl_viewer_backup is not None:
                    node.stl_viewer = stl_viewer_backup

            # 接続情報の保存
            print("\nSaving connections...")
            connections = ET.SubElement(root, "connections")
            connection_count = 0
            
            for node in self.all_nodes():
                for port in node.output_ports():
                    for connected_port in port.connected_ports():
                        conn = ET.SubElement(connections, "connection")
                        ET.SubElement(conn, "from_node").text = node.name()
                        ET.SubElement(conn, "from_port").text = port.name()
                        ET.SubElement(conn, "to_node").text = connected_port.node().name()
                        ET.SubElement(conn, "to_port").text = connected_port.name()
                        connection_count += 1
                        print(f"Added connection: {node.name()}.{port.name()} -> "
                            f"{connected_port.node().name()}.{connected_port.name()}")

            print(f"Total connections saved: {connection_count}")

            # ファイルの保存
            print("\nWriting to file...")
            tree = ET.ElementTree(root)
            tree.write(file_path, encoding='utf-8', xml_declaration=True)

            # STLビューアの状態を復元
            if stl_viewer_state and hasattr(self, 'stl_viewer'):
                print("Restoring STL viewer state...")
                self.stl_viewer.stl_actors = stl_viewer_state['actors']
                self.stl_viewer.transforms = stl_viewer_state['transforms']
                self.stl_viewer.safe_render()

            print(f"\nProject successfully saved to: {file_path}")
            
            QtWidgets.QMessageBox.information(
                None,
                "Save Complete",
                f"Project saved successfully to:\n{file_path}"
            )

            return True

        except Exception as e:
            error_msg = f"Error saving project: {str(e)}"
            print(f"\nERROR: {error_msg}")
            print("Traceback:")
            traceback.print_exc()
            
            # エラー時もSTLビューアの状態を復元
            if 'stl_viewer_state' in locals() and stl_viewer_state and hasattr(self, 'stl_viewer'):
                print("Restoring STL viewer state after error...")
                self.stl_viewer.stl_actors = stl_viewer_state['actors']
                self.stl_viewer.transforms = stl_viewer_state['transforms']
                self.stl_viewer.safe_render()
            
            QtWidgets.QMessageBox.critical(
                None,
                "Save Error",
                error_msg
            )
            return False

    def clear_graph(self):
        for node in self.all_nodes():
            self.remove_node(node)
        # Clean up any orphaned pipe items from the scene
        self._clear_orphaned_pipes()

    def _clear_orphaned_pipes(self):
        """Remove any orphaned pipe items from the scene after node deletion."""
        try:
            viewer = self.viewer()
            if not viewer:
                return
            scene = viewer.scene()
            if not scene:
                return

            # Get all valid node items currently in the graph
            valid_node_items = set()
            for node in self.all_nodes():
                if hasattr(node, 'view') and node.view:
                    valid_node_items.add(node.view)

            # Find and remove orphaned pipes
            items_to_remove = []
            for item in scene.items():
                if isinstance(item, (CustomPipe, PipeItem)):
                    # Skip LivePipe (used for drawing new connections)
                    if isinstance(item, CustomLivePipe):
                        continue

                    # Check if pipe has valid connections
                    input_port = getattr(item, 'input_port', None)
                    output_port = getattr(item, 'output_port', None)

                    # If either port is None, it's orphaned
                    if not input_port or not output_port:
                        items_to_remove.append(item)
                        continue

                    # Check if port's node is still valid
                    input_node = getattr(input_port, 'node', None)
                    output_node = getattr(output_port, 'node', None)

                    if not input_node or not output_node:
                        items_to_remove.append(item)
                        continue

                    # Check if node is still in the scene
                    input_node_scene = getattr(input_node, 'scene', lambda: None)()
                    output_node_scene = getattr(output_node, 'scene', lambda: None)()

                    if input_node_scene is None or output_node_scene is None:
                        items_to_remove.append(item)
                        continue

                    # Check if node is still in valid nodes
                    if input_node not in valid_node_items and output_node not in valid_node_items:
                        items_to_remove.append(item)

            for item in items_to_remove:
                scene.removeItem(item)
                print(f"[clear_graph] Removed orphaned pipe: {item}")

            # Force scene update
            scene.update()
        except Exception as e:
            print(f"[clear_graph] Error clearing orphaned pipes: {e}")

    def connect_ports(self, from_port, to_port):
        """指定された2つのポートを接続"""
        if from_port and to_port:
            try:
                # 利用可能なメソッドを探して接続を試みる
                if hasattr(self, 'connect_nodes'):
                    connection = self.connect_nodes(
                        from_port.node(), from_port.name(),
                        to_port.node(), to_port.name())
                elif hasattr(self, 'add_edge'):
                    connection = self.add_edge(
                        from_port.node().id, from_port.name(),
                        to_port.node().id, to_port.name())
                elif hasattr(from_port, 'connect_to'):
                    connection = from_port.connect_to(to_port)
                else:
                    raise AttributeError("No suitable connection method found")

                if connection:
                    print(
                        f"Connected {from_port.node().name()}.{from_port.name()} to {to_port.node().name()}.{to_port.name()}")
                    return True
                else:
                    print("Failed to connect ports: Connection not established")
                    return False
            except Exception as e:
                print(f"Error connecting ports: {str(e)}")
                return False
        else:
            print("Failed to connect ports: Invalid port(s)")
            return False

    def calculate_cumulative_coordinates(self, node):
        """ノードの累積座標を計算（ルートからのパスを考慮）"""
        if isinstance(node, BaseLinkNode):
            return [0, 0, 0]  # base_linkは原点

        # 親ノードとの接続情報を取得
        input_port = node.input_ports()[0]  # 最初の入力ポート
        if not input_port.connected_ports():
            return [0, 0, 0]  # 接続されていない場合は原点

        parent_port = input_port.connected_ports()[0]
        parent_node = parent_port.node()
        
        # 親ノードの累積座標を再帰的に計算
        parent_coords = self.calculate_cumulative_coordinates(parent_node)
        
        # 接続されているポートのインデックスを取得
        port_name = parent_port.name()
        if '_' in port_name:
            port_index = int(port_name.split('_')[1]) - 1
        else:
            port_index = 0
            
        # 親ノードのポイント座標を取得
        if 0 <= port_index < len(parent_node.points):
            point_xyz = parent_node.points[port_index]['xyz']
            
            # 累積座標の計算
            return [
                parent_coords[0] + point_xyz[0],
                parent_coords[1] + point_xyz[1],
                parent_coords[2] + point_xyz[2]
            ]
        return parent_coords

    def add_node_below_selected(self):
        """最後に選択したノードの下に新しいノードを追加"""
        # 選択されたノードを取得
        selected_nodes = self.selected_nodes()

        if selected_nodes:
            # 最後に選択されたノードの位置を取得
            last_selected = selected_nodes[-1]
            print(f"\n=== Add Node Debug ===")
            print(f"Selected node: {last_selected.name()} at pos {last_selected.view.pos()}")

            # 選択されているノードの親を探す
            parent_node = None
            for input_port in last_selected.input_ports():
                if input_port.name() == 'parent':
                    connected_ports = input_port.connected_ports()
                    if connected_ports:
                        parent_node = connected_ports[0].node()
                        print(f"Found parent: {parent_node.name()} at pos {parent_node.view.pos()}")
                        break

            # 親ノードがある場合は、親ノードを基準に配置する
            if parent_node:
                target_node = parent_node
                print(f"Using parent as target: {target_node.name()}")
            else:
                # 親がない場合は、選択されているノード自体を親として使用
                target_node = last_selected
                print(f"No parent found, using selected node as target: {target_node.name()}")

            target_pos = target_node.view.pos()

            # target_node の child portに接続されている子ノードがあるかチェック
            child_port = None
            for output_port in target_node.output_ports():
                if output_port.name() == 'child':
                    child_port = output_port
                    break

            # 既に子ノードがある場合は、その横に配置
            if child_port and child_port.connected_ports():
                # 最初の子ノードのy座標を使用（親から70px下）
                first_child = child_port.connected_ports()[0].node()
                first_child_pos = first_child.view.pos()
                print(f"Target has existing children. First child: {first_child.name()} at {first_child_pos}")

                # 全ての子ノードの中で最も右にあるノードを見つける
                max_x = first_child_pos.x()
                for connected_port in child_port.connected_ports():
                    child_node = connected_port.node()
                    child_x = child_node.view.pos().x()
                    if child_x > max_x:
                        max_x = child_x

                # 最も右の子ノードからさらに150px右に配置
                new_pos = QPointF(max_x + 150, first_child_pos.y())
                print(f"Placing new node at: ({new_pos.x()}, {new_pos.y()}) - to the right of siblings")
            else:
                # 子ノードがない場合は、親ノードの真下（70ピクセル下）に配置
                new_pos = QPointF(target_pos.x(), target_pos.y() + 70)
                print(f"Target has no children. Placing new node at: ({new_pos.x()}, {new_pos.y()}) - 70px below target")
        else:
            # 選択されていない場合は(0, 0)に配置
            new_pos = QPointF(0, 0)
            target_node = None
            print("No node selected, placing at (0, 0)")

        # 新しいノードを作成（自動配置をスキップして、手動で計算した位置を使用）
        new_node = self.create_node(
            'insilico.nodes.FooNode',
            name=f'Node_{len(self.all_nodes())}',
            pos=new_pos,
            skip_auto_position=True
        )
        print(f"Created new node: {new_node.name()} at {new_node.view.pos()}")

        # 新しいノードを選択状態にする
        self.clear_selection()
        new_node.set_selected(True)
        print(f"Set selection to new node: {new_node.name()}")

        print("=== End Debug ===\n")

        return new_node

    def recalculate_all_positions(self):
        """すべてのノードの位置を再計算"""
        print("Starting position recalculation for all nodes...")
        
        try:
            # base_linkノードを探す
            base_node = None
            for node in self.all_nodes():
                if isinstance(node, BaseLinkNode):
                    base_node = node
                    break
            
            if not base_node:
                print("Error: Base link node not found")
                return
            
            # 再帰的に位置を更新
            visited_nodes = set()
            print(f"Starting from base node: {base_node.name()}")
            self._recalculate_node_positions(base_node, [0, 0, 0], visited_nodes)
            
            # STLビューアの更新
            if hasattr(self, 'stl_viewer'):
                self.stl_viewer.safe_render()
            
            print("Position recalculation completed")

        except Exception as e:
            print(f"Error during position recalculation: {str(e)}")
            traceback.print_exc()

    def _recalculate_node_positions(self, node, parent_coords, visited):
        """再帰的にノードの位置を計算"""
        if node in visited:
            return
        visited.add(node)
        
        print(f"\nProcessing node: {node.name()}")
        print(f"Parent coordinates: {parent_coords}")
        
        try:
            # 出力ポートを処理
            for port_idx, output_port in enumerate(node.output_ports()):
                for connected_port in output_port.connected_ports():
                    child_node = connected_port.node()
                    
                    # ポイントデータの確認
                    if hasattr(node, 'points') and port_idx < len(node.points):
                        point_data = node.points[port_idx]
                        point_xyz = point_data['xyz']
                        
                        # 新しい位置を計算
                        new_position = [
                            parent_coords[0] + point_xyz[0],
                            parent_coords[1] + point_xyz[1],
                            parent_coords[2] + point_xyz[2]
                        ]
                        
                        print(f"Child node: {child_node.name()}")
                        print(f"Point data: {point_xyz}")
                        print(f"Calculated position: {new_position}")
                        
                        # STL位置を更新
                        self.stl_viewer.update_stl_transform(child_node, new_position)
                        
                        # 子ノードの累積座標を更新
                        if hasattr(child_node, 'cumulative_coords'):
                            for coord in child_node.cumulative_coords:
                                coord['xyz'] = new_position.copy()
                        
                        # 再帰的に子ノードを処理
                        self._recalculate_node_positions(child_node, new_position, visited)
                    else:
                        print(f"Warning: No point data found for port {port_idx} in node {node.name()}")

        except Exception as e:
            print(f"Error processing node {node.name()}: {str(e)}")
            traceback.print_exc()

    def disconnect_ports(self, from_port, to_port):
        """ポートの接続を解除"""
        try:
            print(f"Disconnecting ports: {from_port.node().name()}.{from_port.name()} -> {to_port.node().name()}.{to_port.name()}")
            
            # 接続を解除する前に位置情報をリセット
            child_node = to_port.node()
            if child_node:
                self.stl_viewer.reset_stl_transform(child_node)
            
            # 利用可能なメソッドを探して接続解除を試みる
            if hasattr(self, 'disconnect_nodes'):
                success = self.disconnect_nodes(
                    from_port.node(), from_port.name(),
                    to_port.node(), to_port.name())
            elif hasattr(from_port, 'disconnect_from'):
                success = from_port.disconnect_from(to_port)
            else:
                success = False
                print("No suitable disconnection method found")
                
            if success:
                print("Ports disconnected successfully")
                # on_port_disconnectedイベントを呼び出す
                self.on_port_disconnected(to_port, from_port)
                return True
            else:
                print("Failed to disconnect ports")
                return False
                
        except Exception as e:
            print(f"Error disconnecting ports: {str(e)}")
            return False


# =============================================================================
# モーションエディタ用クラス群
# =============================================================================

class PoseNode(BaseNode):
    """ポーズを保持するノード"""
    __identifier__ = 'motion.nodes'
    NODE_NAME = 'PoseNode'
    __view__ = CustomNodeItem

    def __init__(self):
        super(PoseNode, self).__init__(CustomNodeItem)
        # 入力ポート（非表示、複数接続許可）
        input_port = self.add_input('', color=NODE_POSE_INPUT_PORT_COLOR, multi_input=True)
        if input_port:
            try:
                self.view.inputs[0].setVisible(False)
            except Exception:
                pass

        # 出力ポート（default）
        self.output_count = 0
        self._add_pose_output('default', 0)

        # ポーズデータ
        self.pose_name = "pose"
        _df = get_default_hz_fps()
        self.frames = _df  # 1秒あたりの更新回数（Hz）
        self.duration = 1.0  # 区間の名目時間（秒）
        self.angles_deg = {}
        self.joint_easings = {}
        self.branching_enabled = False
        self.branch_outputs_swapped = False
        self.branch_if_left = "UserVal_0"
        self.branch_if_op = "=="
        self.branch_if_right = "UserVal_1"
        self.branch_if_uv_enabled = True
        self.branch_if_formula_enabled = False
        self.branch_if_formula = "Form1:foo"
        self.branch_if_pad_enabled = False
        self.branch_if_pad_button = "L1"
        self.branch_if_pad_analog_enabled = False
        self.branch_if_pad_analog_axis = "Lx"
        self.branch_if_pad_analog_op = ">="
        self.branch_if_pad_analog_threshold = 0
        self.out_port_labels = ["default"]
        self.out_port_priorities = [0]

        # ダブルクリックで詳細編集
        self._original_double_click = self.view.mouseDoubleClickEvent
        self.view.mouseDoubleClickEvent = self._on_double_click

        # Left badge shows the frame count, formatted as "[N]f".
        try:
            self.view._compute_left_badge = lambda n=self: f"{int(getattr(n, 'frames', 0))}f"
        except Exception:
            pass

        # ポーズノード用の色を適用（ビュー初期化後に実行）
        QtCore.QTimer.singleShot(20, self._apply_node_colors)

    def _apply_node_colors(self):
        """ポーズノードの色設定を適用"""
        v = getattr(self, 'view', None)
        if not v:
            return
        # タイトル文字色（通常時 + ハイライト時）
        if hasattr(v, 'set_title_color'):
            v.set_title_color(*NODE_POSE_TITLE_COLOR, highlight_color=NODE_POSE_TITLE_HIGHLIGHT_COLOR)
        # タイトル背景色
        if hasattr(v, '_title_bg_color'):
            v._title_bg_color = QtGui.QColor(*NODE_POSE_TITLE_BG_COLOR)
        # パネル背景色
        self.set_color(*NODE_POSE_PANEL_BG_COLOR)
        # ハイライト時/通常時の色をviewに保存
        if hasattr(v, 'set_normal_colors'):
            v.set_normal_colors(
                panel_bg=NODE_POSE_PANEL_BG_COLOR,
                input_port=NODE_POSE_INPUT_PORT_COLOR,
                input_port_border=NODE_POSE_INPUT_PORT_BORDER_COLOR,
                output_port=NODE_POSE_OUTPUT_PORT_COLOR,
                output_port_border=NODE_POSE_OUTPUT_PORT_BORDER_COLOR
            )
        if hasattr(v, 'set_highlight_colors'):
            v.set_highlight_colors(
                panel_bg=NODE_POSE_PANEL_BG_HIGHLIGHT_COLOR,
                input_port=NODE_POSE_INPUT_PORT_HIGHLIGHT_COLOR,
                input_port_border=NODE_POSE_INPUT_PORT_HIGHLIGHT_BORDER_COLOR,
                output_port=NODE_POSE_OUTPUT_PORT_HIGHLIGHT_COLOR,
                output_port_border=NODE_POSE_OUTPUT_PORT_HIGHLIGHT_BORDER_COLOR
            )
        self._apply_pose_output_colors()
        # 入力ポート色
        for port in self.input_ports():
            port.color = NODE_POSE_INPUT_PORT_COLOR
            port.border_color = NODE_POSE_INPUT_PORT_BORDER_COLOR

    def _apply_pose_output_colors(self):
        """Branching ON時は青/赤ポートにする。swap時は色を入れ替え"""
        output_colors = []
        if getattr(self, 'branching_enabled', False):
            swapped = getattr(self, 'branch_outputs_swapped', False)
            if swapped:
                # Swapped: port 0 = to (red/right), port 1 = otherwise (blue/center)
                output_colors = [
                    (NODE_POSE_BRANCH_TO_PORT_COLOR, NODE_POSE_BRANCH_TO_PORT_BORDER_COLOR),
                    (NODE_POSE_BRANCH_OTHERWISE_PORT_COLOR, NODE_POSE_BRANCH_OTHERWISE_PORT_BORDER_COLOR),
                ]
            else:
                # Normal: port 0 = otherwise (blue/center), port 1 = to (red/right)
                output_colors = [
                    (NODE_POSE_BRANCH_OTHERWISE_PORT_COLOR, NODE_POSE_BRANCH_OTHERWISE_PORT_BORDER_COLOR),
                    (NODE_POSE_BRANCH_TO_PORT_COLOR, NODE_POSE_BRANCH_TO_PORT_BORDER_COLOR),
                ]

        for i, port in enumerate(self.output_ports()):
            if i < len(output_colors):
                port.color, port.border_color = output_colors[i]
            else:
                port.color = NODE_POSE_OUTPUT_PORT_COLOR
                port.border_color = NODE_POSE_OUTPUT_PORT_BORDER_COLOR

        view_outputs = getattr(getattr(self, 'view', None), 'outputs', [])
        view = getattr(self, 'view', None)
        if view is not None:
            if output_colors:
                view._branching_output_colors = list(output_colors)
            elif hasattr(view, '_branching_output_colors'):
                delattr(view, '_branching_output_colors')
        for i, port_view in enumerate(view_outputs):
            if i < len(output_colors):
                port_view.color, port_view.border_color = output_colors[i]
            else:
                port_view.color = NODE_POSE_OUTPUT_PORT_COLOR
                port_view.border_color = NODE_POSE_OUTPUT_PORT_BORDER_COLOR
            port_view.update()

    def _lock_output_row_height(self):
        """OutPortを横並びで増やすため、追加前のノード高さを保持する"""
        view = getattr(self, 'view', None)
        if view is not None and not hasattr(view, '_fixed_output_row_height'):
            view._fixed_output_row_height = view._height

    def enable_branching_output(self):
        """BranchingをONにしてElse用の出力ポートを用意する"""
        self.branching_enabled = True
        self._lock_output_row_height()
        if self.output_count < 2:
            self._add_pose_output("else", 10)
        self._sync_branching_port_labels()
        self._apply_pose_output_colors()
        try:
            QtCore.QTimer.singleShot(10, self._do_position_outputs)
        except Exception:
            pass

    def _sync_branching_port_labels(self):
        """BranchingのThen/Else割り当てを内部ラベルへ反映する"""
        if not getattr(self, 'branching_enabled', False) or len(self.out_port_labels) < 2:
            return
        labels = ("then", "else") if getattr(self, 'branch_outputs_swapped', False) else ("else", "then")
        self.out_port_labels[0] = labels[0]
        self.out_port_labels[1] = labels[1]

    def _add_pose_output(self, label="default", priority=0):
        """出力ポートを追加"""
        if self.output_count < 8:
            self.output_count += 1
            port_name = f'out_{self.output_count}'
            super(PoseNode, self).add_output(port_name, display_name=False)
            if not hasattr(self, 'out_port_labels'):
                self.out_port_labels = []
                self.out_port_priorities = []
            self.out_port_labels.append(label)
            self.out_port_priorities.append(priority)
            try:
                QtCore.QTimer.singleShot(10, self._do_position_outputs)
            except Exception:
                pass
            return port_name
        return None

    def _do_position_outputs(self):
        """出力ポートを下部中央に配置"""
        try:
            if not self.view:
                return
            outputs = self.view.outputs
            if not outputs:
                return

            # キャッシュを一時的に無効化して再描画を強制
            old_cache_mode = self.view.cacheMode()
            self.view.setCacheMode(QtWidgets.QGraphicsItem.NoCache)
            self.view.prepareGeometryChange()

            node_width = self.view._width
            node_height = self.view._height
            spacing = 8
            first_port_width = outputs[0].boundingRect().width()
            total_width = (first_port_width * len(outputs)) + (spacing * (len(outputs) - 1))
            start_x = (node_width - total_width) / 2
            for i, port in enumerate(outputs):
                port.setVisible(True)
                port_width = port.boundingRect().width()
                port_height = port.boundingRect().height()
                port_x = start_x + (i * (port_width + spacing))
                port_y = node_height - port_height - 5
                port.prepareGeometryChange()
                port.setPos(port_x, port_y)
                port.update()
                if port in self.view._output_items:
                    self.view._output_items[port].setVisible(False)
            self._apply_pose_output_colors()

            # ノード全体を更新
            self.view.update()
            if self.view.scene():
                self.view.scene().update()

            # キャッシュモードを元に戻す
            self.view.setCacheMode(old_cache_mode)
        except Exception as e:
            print(f"[PoseNode] Error positioning ports: {e}")

    def _on_double_click(self, event):
        """ダブルクリックで Joint Sliders を開く"""
        if hasattr(self, 'graph') and hasattr(self.graph, 'open_joint_sliders_for_pose'):
            try:
                graph_view = self.graph.viewer()
                scene_pos = event.scenePos()
                view_pos = graph_view.mapFromScene(scene_pos)
                screen_pos = graph_view.mapToGlobal(view_pos)
                self.graph.open_joint_sliders_for_pose(self, screen_pos)
            except Exception as e:
                print(f"[PoseNode] Error on double click: {e}")
                traceback.print_exc()


class DefineNode(BaseNode):
    """User Value 定義用ノード（グラフ上のパネル）。"""

    __identifier__ = "motion.nodes"
    NODE_NAME = "DefineNode"
    __view__ = CustomNodeItem

    def __init__(self):
        super(DefineNode, self).__init__(CustomNodeItem)
        input_port = self.add_input("", color=NODE_POSE_INPUT_PORT_COLOR, multi_input=True)
        if input_port:
            try:
                self.view.inputs[0].setVisible(False)
            except Exception:
                pass
        self.output_count = 0
        self._add_define_output("default", 0)

        self.define_uv_index = 0
        self.define_memo = ""
        self.define_kind = "literal"
        self.define_literal = 0
        self.define_register_name = ""

        self.out_port_labels = ["default"]
        self.out_port_priorities = [0]

        self._original_double_click = self.view.mouseDoubleClickEvent
        self.view.mouseDoubleClickEvent = self._on_double_click

        QtCore.QTimer.singleShot(20, self._apply_define_colors)

    def _add_define_output(self, label="default", priority=0):
        if self.output_count < 8:
            self.output_count += 1
            port_name = f"out_{self.output_count}"
            super(DefineNode, self).add_output(port_name, display_name=False)
            if not hasattr(self, "out_port_labels"):
                self.out_port_labels = []
                self.out_port_priorities = []
            self.out_port_labels.append(label)
            self.out_port_priorities.append(priority)
            try:
                QtCore.QTimer.singleShot(10, self._do_position_outputs)
            except Exception:
                pass
            return port_name
        return None

    def _do_position_outputs(self):
        try:
            if not self.view:
                return
            outputs = self.view.outputs
            if not outputs:
                return
            node_width = self.view._width
            spacing = 8
            first_port_width = outputs[0].boundingRect().width()
            total_width = (first_port_width * len(outputs)) + (
                spacing * (len(outputs) - 1)
            )
            start_x = (node_width - total_width) / 2
            for i, port in enumerate(outputs):
                port_width = port.boundingRect().width()
                port_height = port.boundingRect().height()
                port_x = start_x + (i * (port_width + spacing))
                port_y = self.view._height - port_height - 5
                port.setPos(port_x, port_y)
                port.update()
                if port in self.view._output_items:
                    self.view._output_items[port].setVisible(False)
        except Exception as e:
            print(f"[DefineNode] Error positioning ports: {e}")

    def _apply_define_colors(self):
        v = getattr(self, "view", None)
        if not v:
            return
        if hasattr(v, "set_title_color"):
            v.set_title_color(
                *NODE_DEFINE_TITLE_COLOR,
                highlight_color=NODE_DEFINE_TITLE_HIGHLIGHT_COLOR,
            )
        if hasattr(v, "_title_bg_color"):
            v._title_bg_color = QtGui.QColor(*NODE_DEFINE_TITLE_BG_COLOR)
        self.set_color(*NODE_DEFINE_PANEL_BG_COLOR)
        if hasattr(v, "set_normal_colors"):
            v.set_normal_colors(
                panel_bg=NODE_DEFINE_PANEL_BG_COLOR,
                input_port=NODE_POSE_INPUT_PORT_COLOR,
                input_port_border=NODE_POSE_INPUT_PORT_BORDER_COLOR,
                output_port=NODE_POSE_OUTPUT_PORT_COLOR,
                output_port_border=NODE_POSE_OUTPUT_PORT_BORDER_COLOR,
            )
        if hasattr(v, "set_highlight_colors"):
            v.set_highlight_colors(
                panel_bg=NODE_DEFINE_PANEL_BG_HIGHLIGHT_COLOR,
                input_port=NODE_POSE_INPUT_PORT_HIGHLIGHT_COLOR,
                input_port_border=NODE_POSE_INPUT_PORT_HIGHLIGHT_BORDER_COLOR,
                output_port=NODE_POSE_OUTPUT_PORT_HIGHLIGHT_COLOR,
                output_port_border=NODE_POSE_OUTPUT_PORT_HIGHLIGHT_BORDER_COLOR,
            )
        for port in self.input_ports():
            port.color = NODE_POSE_INPUT_PORT_COLOR
            port.border_color = NODE_POSE_INPUT_PORT_BORDER_COLOR
        for port in self.output_ports():
            port.color = NODE_POSE_OUTPUT_PORT_COLOR
            port.border_color = NODE_POSE_OUTPUT_PORT_BORDER_COLOR

    def _on_double_click(self, event):
        if hasattr(self, "graph") and hasattr(self.graph, "show_define_editor"):
            try:
                graph_view = self.graph.viewer()
                scene_pos = event.scenePos()
                view_pos = graph_view.mapFromScene(scene_pos)
                screen_pos = graph_view.mapToGlobal(view_pos)
                self.graph.show_define_editor(self, screen_pos)
            except Exception as e:
                print(f"[DefineNode] Error on double click: {e}")
                traceback.print_exc()


class WaitNode(BaseNode):
    """Wait ノード: 関節データを持たず指定フレーム数だけ停止する。"""

    __identifier__ = "motion.nodes"
    NODE_NAME = "WaitNode"
    __view__ = CustomNodeItem

    def __init__(self):
        super(WaitNode, self).__init__(CustomNodeItem)
        input_port = self.add_input("", color=NODE_POSE_INPUT_PORT_COLOR, multi_input=True)
        if input_port:
            try:
                self.view.inputs[0].setVisible(False)
            except Exception:
                pass
        self.output_count = 0
        self._add_wait_output("default", 0)

        self.wait_name = "wait"
        self.frames = 0
        self.duration = 0.0
        self.out_port_labels = ["default"]
        self.out_port_priorities = [0]

        self._original_double_click = self.view.mouseDoubleClickEvent
        self.view.mouseDoubleClickEvent = self._on_double_click

        QtCore.QTimer.singleShot(20, self._apply_wait_colors)

    def _add_wait_output(self, label="default", priority=0):
        self.output_count += 1
        port_name = f"out_{self.output_count}"
        super(WaitNode, self).add_output(port_name, display_name=False)
        if not hasattr(self, "out_port_labels"):
            self.out_port_labels = []
            self.out_port_priorities = []
        self.out_port_labels.append(label)
        self.out_port_priorities.append(priority)
        return port_name

    def _apply_wait_colors(self):
        v = getattr(self, "view", None)
        if not v:
            return
        if hasattr(v, "set_title_color"):
            v.set_title_color(
                *NODE_WAIT_TITLE_COLOR,
                highlight_color=NODE_WAIT_TITLE_HIGHLIGHT_COLOR,
            )
        if hasattr(v, "_title_bg_color"):
            v._title_bg_color = QtGui.QColor(*NODE_WAIT_TITLE_BG_COLOR)
        self.set_color(*NODE_WAIT_PANEL_BG_COLOR)
        if hasattr(v, "set_normal_colors"):
            v.set_normal_colors(
                panel_bg=NODE_WAIT_PANEL_BG_COLOR,
                input_port=NODE_POSE_INPUT_PORT_COLOR,
                input_port_border=NODE_POSE_INPUT_PORT_BORDER_COLOR,
                output_port=NODE_POSE_OUTPUT_PORT_COLOR,
                output_port_border=NODE_POSE_OUTPUT_PORT_BORDER_COLOR,
            )
        if hasattr(v, "set_highlight_colors"):
            v.set_highlight_colors(
                panel_bg=NODE_WAIT_PANEL_BG_HIGHLIGHT_COLOR,
                input_port=NODE_POSE_INPUT_PORT_HIGHLIGHT_COLOR,
                input_port_border=NODE_POSE_INPUT_PORT_HIGHLIGHT_BORDER_COLOR,
                output_port=NODE_POSE_OUTPUT_PORT_HIGHLIGHT_COLOR,
                output_port_border=NODE_POSE_OUTPUT_PORT_HIGHLIGHT_BORDER_COLOR,
            )
        for port in self.input_ports():
            port.color = NODE_POSE_INPUT_PORT_COLOR
            port.border_color = NODE_POSE_INPUT_PORT_BORDER_COLOR
        for port in self.output_ports():
            port.color = NODE_POSE_OUTPUT_PORT_COLOR
            port.border_color = NODE_POSE_OUTPUT_PORT_BORDER_COLOR

    def refresh_body_text(self):
        """ノード本体の表示テキストを更新"""
        try:
            fps = getattr(self, "_display_fps", 100)
            dur = self.frames / max(1, float(fps))
            lines = [self.wait_name, f"{self.frames} f / {dur:.2f} s"]
            body = "\n".join(lines)
            if hasattr(self.view, "set_body_text"):
                self.view.set_body_text(body)
        except Exception:
            pass

    def _on_double_click(self, event):
        if hasattr(self, "graph") and hasattr(self.graph, "show_wait_editor"):
            try:
                graph_view = self.graph.viewer()
                scene_pos = event.scenePos()
                view_pos = graph_view.mapFromScene(scene_pos)
                screen_pos = graph_view.mapToGlobal(view_pos)
                self.graph.show_wait_editor(self, screen_pos)
            except Exception as e:
                print(f"[WaitNode] Error on double click: {e}")
                traceback.print_exc()


class BranchingNode(BaseNode):
    """分岐用パネル（ダブルクリックで BranchingShellDialog）。"""

    __identifier__ = "motion.nodes"
    NODE_NAME = "BranchingNode"
    __view__ = CustomNodeItem

    def __init__(self):
        super(BranchingNode, self).__init__(CustomNodeItem)
        # 入力ポート：Define/Pose 同様に描画は非表示。接続アンカーはパネル幾何中心（_hidden_input_at_panel_center）
        self.view._hidden_input_at_panel_center = True
        self.add_input("", color=NODE_POSE_INPUT_PORT_COLOR, multi_input=True)
        try:
            self.view.inputs[0].setVisible(False)
        except Exception:
            pass
        self.output_count = 0
        self._add_branch_output("default", 0)

        self.branching_enabled = False
        self.branch_outputs_swapped = False
        self.branch_if_left = "UserVal_0"
        self.branch_if_op = "=="
        self.branch_if_right = "UserVal_1"
        self.branch_if_uv_enabled = True
        self.branch_if_formula_enabled = False
        self.branch_if_formula = "Form1:foo"
        self.branch_if_pad_enabled = False
        self.branch_if_pad_button = "L1"
        self.branch_if_pad_analog_enabled = False
        self.branch_if_pad_analog_axis = "Lx"
        self.branch_if_pad_analog_op = ">="
        self.branch_if_pad_analog_threshold = 0
        self.out_port_labels = ["default"]
        self.out_port_priorities = [0]

        self._original_double_click = self.view.mouseDoubleClickEvent
        self.view.mouseDoubleClickEvent = self._on_double_click

        # Left badge shows the PAD button name if IF-PAD is on,
        # or the analog axis name if IF-Analog is on.
        try:
            self.view._compute_left_badge = self._compute_left_badge_text
        except Exception:
            pass

        QtCore.QTimer.singleShot(20, self._apply_branching_node_colors)

    def _compute_left_badge_text(self):
        """Return the left-badge text. Priority: PAD button > Analog axis > None."""
        if getattr(self, "branch_if_pad_enabled", False):
            btn = getattr(self, "branch_if_pad_button", "")
            if btn:
                return str(btn)
        if getattr(self, "branch_if_pad_analog_enabled", False):
            axis = getattr(self, "branch_if_pad_analog_axis", "")
            if axis:
                return str(axis)
        return None

    def _add_branch_output(self, label="default", priority=0):
        if self.output_count < 8:
            self.output_count += 1
            port_name = f"out_{self.output_count}"
            super(BranchingNode, self).add_output(port_name, display_name=False)
            if not hasattr(self, "out_port_labels"):
                self.out_port_labels = []
                self.out_port_priorities = []
            self.out_port_labels.append(label)
            self.out_port_priorities.append(priority)
            try:
                QtCore.QTimer.singleShot(10, self._do_position_outputs)
            except Exception:
                pass
            return port_name
        return None

    def _lock_output_row_height(self):
        view = getattr(self, "view", None)
        if view is not None and not hasattr(view, "_fixed_output_row_height"):
            view._fixed_output_row_height = view._height

    def enable_branching_output(self):
        self.branching_enabled = True
        self._lock_output_row_height()
        if self.output_count < 2:
            self._add_branch_output("else", 10)
        self._sync_branching_port_labels()
        self._apply_branch_output_colors()
        try:
            QtCore.QTimer.singleShot(10, self._do_position_outputs)
        except Exception:
            pass

    def _sync_branching_port_labels(self):
        if not getattr(self, "branching_enabled", False) or len(self.out_port_labels) < 2:
            return
        labels = (
            ("then", "else")
            if getattr(self, "branch_outputs_swapped", False)
            else ("else", "then")
        )
        self.out_port_labels[0] = labels[0]
        self.out_port_labels[1] = labels[1]

    def _do_position_outputs(self):
        try:
            if not self.view:
                return
            outputs = self.view.outputs
            if not outputs:
                return
            node_width = self.view._width
            spacing = 8
            # Port 0 is always at left/center, Port 1 is always at right
            # Colors swap when branch_outputs_swapped is True, positions stay fixed
            _branch_port_gap = 4
            if (
                len(outputs) == 2
                and getattr(self, "branching_enabled", False)
            ):
                w0 = outputs[0].boundingRect().width()
                h0 = outputs[0].boundingRect().height()
                cx = node_width / 2.0
                port_y = self.view._height - h0 - 5
                # Port 0 (otherwise) at center, Port 1 (to) at right edge
                x_center = cx - w0 / 2.0
                x_right = node_width - w0 - 5
                outputs[0].setPos(x_center, port_y)
                outputs[1].setPos(x_right, port_y)
                for port in outputs:
                    port.update()
                    if port in self.view._output_items:
                        self.view._output_items[port].setVisible(False)
                return
            first_port_width = outputs[0].boundingRect().width()
            total_width = (first_port_width * len(outputs)) + (
                spacing * (len(outputs) - 1)
            )
            start_x = (node_width - total_width) / 2
            for i, port in enumerate(outputs):
                port_width = port.boundingRect().width()
                port_height = port.boundingRect().height()
                port_x = start_x + (i * (port_width + spacing))
                port_y = self.view._height - port_height - 5
                port.setPos(port_x, port_y)
                port.update()
                if port in self.view._output_items:
                    self.view._output_items[port].setVisible(False)
        except Exception as e:
            print(f"[BranchingNode] Error positioning ports: {e}")

    def _apply_branch_output_colors(self):
        output_colors = []
        if getattr(self, "branching_enabled", False):
            # Check if outputs are swapped
            swapped = getattr(self, "branch_outputs_swapped", False)
            if swapped:
                # Swapped: port 0 = to (red/center), port 1 = otherwise (blue/right)
                output_colors = [
                    (NODE_POSE_BRANCH_TO_PORT_COLOR, NODE_POSE_BRANCH_TO_PORT_BORDER_COLOR),
                    (NODE_POSE_BRANCH_OTHERWISE_PORT_COLOR, NODE_POSE_BRANCH_OTHERWISE_PORT_BORDER_COLOR),
                ]
            else:
                # Normal: port 0 = otherwise (blue/center), port 1 = to (red/right)
                output_colors = [
                    (NODE_POSE_BRANCH_OTHERWISE_PORT_COLOR, NODE_POSE_BRANCH_OTHERWISE_PORT_BORDER_COLOR),
                    (NODE_POSE_BRANCH_TO_PORT_COLOR, NODE_POSE_BRANCH_TO_PORT_BORDER_COLOR),
                ]
        for i, port in enumerate(self.output_ports()):
            if i < len(output_colors):
                port.color, port.border_color = output_colors[i]
            else:
                port.color = NODE_POSE_OUTPUT_PORT_COLOR
                port.border_color = NODE_POSE_OUTPUT_PORT_BORDER_COLOR
        view_outputs = getattr(getattr(self, "view", None), "outputs", [])
        view = getattr(self, "view", None)
        if view is not None:
            if output_colors:
                view._branching_output_colors = list(output_colors)
            elif hasattr(view, "_branching_output_colors"):
                delattr(view, "_branching_output_colors")
        for i, port_view in enumerate(view_outputs):
            if i < len(output_colors):
                port_view.color, port_view.border_color = output_colors[i]
            else:
                port_view.color = NODE_POSE_OUTPUT_PORT_COLOR
                port_view.border_color = NODE_POSE_OUTPUT_PORT_BORDER_COLOR
            port_view.update()

    def _apply_branching_node_colors(self):
        v = getattr(self, "view", None)
        if not v:
            return
        if hasattr(v, "set_title_color"):
            v.set_title_color(
                *NODE_BRANCH_TITLE_COLOR,
                highlight_color=NODE_BRANCH_TITLE_HIGHLIGHT_COLOR,
            )
        if hasattr(v, "_title_bg_color"):
            v._title_bg_color = QtGui.QColor(*NODE_BRANCH_TITLE_BG_COLOR)
        self.set_color(*NODE_BRANCH_PANEL_BG_COLOR)
        if hasattr(v, "set_normal_colors"):
            v.set_normal_colors(
                panel_bg=NODE_BRANCH_PANEL_BG_COLOR,
                input_port=NODE_POSE_INPUT_PORT_COLOR,
                input_port_border=NODE_POSE_INPUT_PORT_BORDER_COLOR,
                output_port=NODE_POSE_OUTPUT_PORT_COLOR,
                output_port_border=NODE_POSE_OUTPUT_PORT_BORDER_COLOR,
            )
        if hasattr(v, "set_highlight_colors"):
            v.set_highlight_colors(
                panel_bg=NODE_BRANCH_PANEL_BG_HIGHLIGHT_COLOR,
                input_port=NODE_POSE_INPUT_PORT_HIGHLIGHT_COLOR,
                input_port_border=NODE_POSE_INPUT_PORT_HIGHLIGHT_BORDER_COLOR,
                output_port=NODE_POSE_OUTPUT_PORT_HIGHLIGHT_COLOR,
                output_port_border=NODE_POSE_OUTPUT_PORT_BORDER_COLOR,
            )
        for port in self.input_ports():
            port.color = NODE_POSE_INPUT_PORT_COLOR
            port.border_color = NODE_POSE_INPUT_PORT_BORDER_COLOR
        self._apply_branch_output_colors()

    def _on_double_click(self, event):
        if hasattr(self, "graph") and hasattr(self.graph, "show_branching_editor"):
            try:
                graph_view = self.graph.viewer()
                scene_pos = event.scenePos()
                view_pos = graph_view.mapFromScene(scene_pos)
                screen_pos = graph_view.mapToGlobal(view_pos)
                self.graph.show_branching_editor(self, screen_pos)
            except Exception as e:
                print(f"[BranchingNode] Error on double click: {e}")
                traceback.print_exc()


class CommandNode(BaseNode):
    """Commandノードを保持するノード（サーボコマンド送信用）"""
    __identifier__ = 'motion.nodes'
    NODE_NAME = 'CommandNode'
    __view__ = CustomNodeItem

    def __init__(self):
        super(CommandNode, self).__init__(CustomNodeItem)
        # 入力ポート（非表示、複数接続許可）
        input_port = self.add_input('', color=NODE_COMMAND_INPUT_PORT_COLOR, multi_input=True)
        if input_port:
            try:
                self.view.inputs[0].setVisible(False)
            except Exception:
                pass

        # 出力ポート（default）
        self.output_count = 0
        self._add_command_output('default', 0)

        # Commandデータ
        self.command_name = "command"
        self.frames = 1  # デフォルトframes = 1
        self.duration = 0.01  # 区間の名目時間（秒）
        # command_settings: {joint_name: {command_type: int, value: int}}
        self.command_settings = {}
        self.out_port_labels = ["default"]
        self.out_port_priorities = [0]

        # ダブルクリックで詳細編集
        self._original_double_click = self.view.mouseDoubleClickEvent
        self.view.mouseDoubleClickEvent = self._on_double_click

        # Commandノード用の色を適用（ビュー初期化後に実行）
        QtCore.QTimer.singleShot(20, self._apply_node_colors)

    def _apply_node_colors(self):
        """Commandノードの色設定を適用"""
        v = getattr(self, 'view', None)
        if not v:
            return
        # タイトル文字色（通常時 + ハイライト時）
        if hasattr(v, 'set_title_color'):
            v.set_title_color(*NODE_COMMAND_TITLE_COLOR, highlight_color=NODE_COMMAND_TITLE_HIGHLIGHT_COLOR)
        # タイトル背景色
        if hasattr(v, '_title_bg_color'):
            v._title_bg_color = QtGui.QColor(*NODE_COMMAND_TITLE_BG_COLOR)
        # パネル背景色
        self.set_color(*NODE_COMMAND_PANEL_BG_COLOR)
        # ハイライト時/通常時の色をviewに保存
        if hasattr(v, 'set_normal_colors'):
            v.set_normal_colors(
                panel_bg=NODE_COMMAND_PANEL_BG_COLOR,
                input_port=NODE_COMMAND_INPUT_PORT_COLOR,
                input_port_border=NODE_COMMAND_INPUT_PORT_BORDER_COLOR,
                output_port=NODE_COMMAND_OUTPUT_PORT_COLOR,
                output_port_border=NODE_COMMAND_OUTPUT_PORT_BORDER_COLOR
            )
        if hasattr(v, 'set_highlight_colors'):
            v.set_highlight_colors(
                panel_bg=NODE_COMMAND_PANEL_BG_HIGHLIGHT_COLOR,
                input_port=NODE_COMMAND_INPUT_PORT_HIGHLIGHT_COLOR,
                input_port_border=NODE_COMMAND_INPUT_PORT_HIGHLIGHT_BORDER_COLOR,
                output_port=NODE_COMMAND_OUTPUT_PORT_HIGHLIGHT_COLOR,
                output_port_border=NODE_COMMAND_OUTPUT_PORT_HIGHLIGHT_BORDER_COLOR
            )
        self._apply_command_output_colors()
        # 入力ポート色
        for port in self.input_ports():
            port.color = NODE_COMMAND_INPUT_PORT_COLOR
            port.border_color = NODE_COMMAND_INPUT_PORT_BORDER_COLOR

    def _apply_command_output_colors(self):
        """出力ポートの色を適用"""
        for port in self.output_ports():
            port.color = NODE_COMMAND_OUTPUT_PORT_COLOR
            port.border_color = NODE_COMMAND_OUTPUT_PORT_BORDER_COLOR

        view_outputs = getattr(getattr(self, 'view', None), 'outputs', [])
        for port_view in view_outputs:
            port_view.color = NODE_COMMAND_OUTPUT_PORT_COLOR
            port_view.border_color = NODE_COMMAND_OUTPUT_PORT_BORDER_COLOR
            port_view.update()

    def _add_command_output(self, label="default", priority=0):
        """出力ポートを追加"""
        if self.output_count < 8:
            self.output_count += 1
            port_name = f'out_{self.output_count}'
            super(CommandNode, self).add_output(port_name, display_name=False)
            if not hasattr(self, 'out_port_labels'):
                self.out_port_labels = []
                self.out_port_priorities = []
            self.out_port_labels.append(label)
            self.out_port_priorities.append(priority)
            try:
                QtCore.QTimer.singleShot(10, self._do_position_outputs)
            except Exception:
                pass
            return port_name
        return None

    def _do_position_outputs(self):
        """出力ポートを下部中央に配置"""
        try:
            if not self.view:
                return
            outputs = self.view.outputs
            if not outputs:
                return
            node_width = self.view._width
            spacing = 8
            first_port_width = outputs[0].boundingRect().width()
            total_width = (first_port_width * len(outputs)) + (spacing * (len(outputs) - 1))
            start_x = (node_width - total_width) / 2
            for i, port in enumerate(outputs):
                port.setVisible(True)
                port_width = port.boundingRect().width()
                port_height = port.boundingRect().height()
                port_x = start_x + (i * (port_width + spacing))
                port_y = self.view._height - port_height - 5
                port.setPos(port_x, port_y)
                port.update()
                if port in self.view._output_items:
                    self.view._output_items[port].setVisible(False)
            self._apply_command_output_colors()
        except Exception as e:
            print(f"[CommandNode] Error positioning ports: {e}")

    def _on_double_click(self, event):
        """ダブルクリックで Command Editor を開く"""
        if hasattr(self, 'graph') and hasattr(self.graph, 'show_command_editor'):
            try:
                graph_view = self.graph.viewer()
                scene_pos = event.scenePos()
                view_pos = graph_view.mapFromScene(scene_pos)
                screen_pos = graph_view.mapToGlobal(view_pos)
                self.graph.show_command_editor(self, screen_pos)
            except Exception as e:
                print(f"[CommandNode] Error on double click: {e}")
                traceback.print_exc()


class MixNode(BaseNode):
    """Mixノードを保持するノード（PoseNodeベース）"""
    __identifier__ = 'motion.nodes'
    NODE_NAME = 'MixNode'
    __view__ = CustomNodeItem

    def __init__(self):
        super(MixNode, self).__init__(CustomNodeItem)
        # 入力ポート（非表示、複数接続許可）
        input_port = self.add_input('', color=NODE_MIX_INPUT_PORT_COLOR, multi_input=True)
        if input_port:
            try:
                self.view.inputs[0].setVisible(False)
            except Exception:
                pass

        # 出力ポート（default）
        self.output_count = 0
        self._add_mix_output('default', 0)

        # Mixデータ
        self.mix_name = "mix"
        self.frames = 1  # デフォルトframes = 1
        self.duration = 0.01  # 区間の名目時間（秒）
        self.angles_deg = {}
        self.joint_easings = {}
        # mix_settings: {joint_name: {enabled: bool, input_source: str, gain: float}}
        self.mix_settings = {}
        self.out_port_labels = ["default"]
        self.out_port_priorities = [0]

        # ダブルクリックで詳細編集
        self._original_double_click = self.view.mouseDoubleClickEvent
        self.view.mouseDoubleClickEvent = self._on_double_click

        # Mixノード用の色を適用（ビュー初期化後に実行）
        QtCore.QTimer.singleShot(20, self._apply_node_colors)

    def _apply_node_colors(self):
        """Mixノードの色設定を適用"""
        v = getattr(self, 'view', None)
        if not v:
            return
        # タイトル文字色（通常時 + ハイライト時）
        if hasattr(v, 'set_title_color'):
            v.set_title_color(*NODE_MIX_TITLE_COLOR, highlight_color=NODE_MIX_TITLE_HIGHLIGHT_COLOR)
        # タイトル背景色
        if hasattr(v, '_title_bg_color'):
            v._title_bg_color = QtGui.QColor(*NODE_MIX_TITLE_BG_COLOR)
        # パネル背景色
        self.set_color(*NODE_MIX_PANEL_BG_COLOR)
        # ハイライト時/通常時の色をviewに保存
        if hasattr(v, 'set_normal_colors'):
            v.set_normal_colors(
                panel_bg=NODE_MIX_PANEL_BG_COLOR,
                input_port=NODE_MIX_INPUT_PORT_COLOR,
                input_port_border=NODE_MIX_INPUT_PORT_BORDER_COLOR,
                output_port=NODE_MIX_OUTPUT_PORT_COLOR,
                output_port_border=NODE_MIX_OUTPUT_PORT_BORDER_COLOR
            )
        if hasattr(v, 'set_highlight_colors'):
            v.set_highlight_colors(
                panel_bg=NODE_MIX_PANEL_BG_HIGHLIGHT_COLOR,
                input_port=NODE_MIX_INPUT_PORT_HIGHLIGHT_COLOR,
                input_port_border=NODE_MIX_INPUT_PORT_HIGHLIGHT_BORDER_COLOR,
                output_port=NODE_MIX_OUTPUT_PORT_HIGHLIGHT_COLOR,
                output_port_border=NODE_MIX_OUTPUT_PORT_HIGHLIGHT_BORDER_COLOR
            )
        self._apply_mix_output_colors()
        # 入力ポート色
        for port in self.input_ports():
            port.color = NODE_MIX_INPUT_PORT_COLOR
            port.border_color = NODE_MIX_INPUT_PORT_BORDER_COLOR

    def _apply_mix_output_colors(self):
        """出力ポートの色を適用"""
        for port in self.output_ports():
            port.color = NODE_MIX_OUTPUT_PORT_COLOR
            port.border_color = NODE_MIX_OUTPUT_PORT_BORDER_COLOR

        view_outputs = getattr(getattr(self, 'view', None), 'outputs', [])
        for port_view in view_outputs:
            port_view.color = NODE_MIX_OUTPUT_PORT_COLOR
            port_view.border_color = NODE_MIX_OUTPUT_PORT_BORDER_COLOR
            port_view.update()

    def _add_mix_output(self, label="default", priority=0):
        """出力ポートを追加"""
        if self.output_count < 8:
            self.output_count += 1
            port_name = f'out_{self.output_count}'
            super(MixNode, self).add_output(port_name, display_name=False)
            if not hasattr(self, 'out_port_labels'):
                self.out_port_labels = []
                self.out_port_priorities = []
            self.out_port_labels.append(label)
            self.out_port_priorities.append(priority)
            try:
                QtCore.QTimer.singleShot(10, self._do_position_outputs)
            except Exception:
                pass
            return port_name
        return None

    def _do_position_outputs(self):
        """出力ポートを下部中央に配置"""
        try:
            if not self.view:
                return
            outputs = self.view.outputs
            if not outputs:
                return
            node_width = self.view._width
            spacing = 8
            first_port_width = outputs[0].boundingRect().width()
            total_width = (first_port_width * len(outputs)) + (spacing * (len(outputs) - 1))
            start_x = (node_width - total_width) / 2
            for i, port in enumerate(outputs):
                port.setVisible(True)
                port_width = port.boundingRect().width()
                port_height = port.boundingRect().height()
                port_x = start_x + (i * (port_width + spacing))
                port_y = self.view._height - port_height - 5
                port.setPos(port_x, port_y)
                port.update()
                if port in self.view._output_items:
                    self.view._output_items[port].setVisible(False)
            self._apply_mix_output_colors()
        except Exception as e:
            print(f"[MixNode] Error positioning ports: {e}")

    def _on_double_click(self, event):
        """ダブルクリックで Mix Editor を開く"""
        if hasattr(self, 'graph') and hasattr(self.graph, 'show_mix_editor'):
            try:
                graph_view = self.graph.viewer()
                scene_pos = event.scenePos()
                view_pos = graph_view.mapFromScene(scene_pos)
                screen_pos = graph_view.mapToGlobal(view_pos)
                self.graph.show_mix_editor(self, screen_pos)
            except Exception as e:
                print(f"[MixNode] Error on double click: {e}")
                traceback.print_exc()


class JumpNode(BaseNode):
    """別アクション番号を指定するジャンプノード（再生時はパススルー）。"""

    __identifier__ = "motion.nodes"
    NODE_NAME = "JumpNode"
    __view__ = CustomNodeItem

    def __init__(self):
        super(JumpNode, self).__init__(CustomNodeItem)
        self.view._hidden_input_at_panel_center = True
        self.add_input("", color=NODE_POSE_INPUT_PORT_COLOR, multi_input=True)
        try:
            self.view.inputs[0].setVisible(False)
        except Exception:
            pass
        self.output_count = 0
        self._add_jump_output("default", 0)

        self.jump_target_action_index = 0
        self.jump_type = "action"
        self.jump_target_function = ""
        self.set_name("Jump to")

        self._original_double_click = self.view.mouseDoubleClickEvent
        self.view.mouseDoubleClickEvent = self._on_double_click

        QtCore.QTimer.singleShot(20, self._apply_jump_node_colors)

    def _add_jump_output(self, label="default", priority=0):
        if self.output_count < 8:
            self.output_count += 1
            port_name = f"out_{self.output_count}"
            super(JumpNode, self).add_output(port_name, display_name=False)
            if not hasattr(self, "out_port_labels"):
                self.out_port_labels = []
                self.out_port_priorities = []
            self.out_port_labels.append(label)
            self.out_port_priorities.append(priority)
            try:
                QtCore.QTimer.singleShot(10, self._do_position_outputs)
            except Exception:
                pass
            return port_name
        return None

    def _do_position_outputs(self):
        try:
            if not self.view:
                return
            outputs = self.view.outputs
            if not outputs:
                return
            node_width = self.view._width
            is_function = getattr(self, "jump_type", "action") == "function"
            spacing = 8
            first_port_width = outputs[0].boundingRect().width()
            total_width = (first_port_width * len(outputs)) + (
                spacing * (len(outputs) - 1)
            )
            start_x = (node_width - total_width) / 2
            for i, port in enumerate(outputs):
                port_width = port.boundingRect().width()
                port_height = port.boundingRect().height()
                port_x = start_x + (i * (port_width + spacing))
                port_y = self.view._height - port_height - 5
                port.setPos(port_x, port_y)
                # function JumpNode の最初のポートは「戻り先」として表示する
                show = is_function and i == 0
                port.setVisible(show)
                port.update()
                if port in self.view._output_items:
                    self.view._output_items[port].setVisible(False)  # ラベルは常に非表示
        except Exception as e:
            print(f"[JumpNode] Error positioning ports: {e}")

    def refresh_body_text(self):
        v = getattr(self, "view", None)
        if v is not None:
            if getattr(self, "jump_type", "action") == "function":
                fn = getattr(self, "jump_target_function", "") or "?"
                v._body_text = f"func: {fn}"
            else:
                n = int(getattr(self, "jump_target_action_index", 0))
                title = ""
                mas = getattr(self, "graph", None) and getattr(self.graph, "motion_action_state", None)
                if mas:
                    items = mas.get("items", [])
                    if 0 <= n < len(items):
                        title = (items[n].get("title") or "").strip()
                v._body_text = f"Action_{n + 1}:{title}" if title else f"Action_{n + 1}"
            v.update()
        # jump_type が変わった時にポート表示を更新する
        self._do_position_outputs()

    def _resize_for_body(self):
        v = getattr(self, "view", None)
        if v is None:
            return
        if getattr(v, "_height", 0) < 102:
            v._height = 102
            if hasattr(v, "_draw_node_horizontal"):
                v._draw_node_horizontal()

    def _apply_jump_node_colors(self):
        v = getattr(self, "view", None)
        if not v:
            return
        if hasattr(v, "set_title_color"):
            v.set_title_color(
                *NODE_JUMP_TITLE_COLOR,
                highlight_color=NODE_JUMP_TITLE_HIGHLIGHT_COLOR,
            )
        if hasattr(v, "_title_bg_color"):
            v._title_bg_color = QtGui.QColor(*NODE_JUMP_TITLE_BG_COLOR)
        self.set_color(*NODE_JUMP_PANEL_BG_COLOR)
        if hasattr(v, "set_normal_colors"):
            v.set_normal_colors(
                panel_bg=NODE_JUMP_PANEL_BG_COLOR,
                input_port=NODE_POSE_INPUT_PORT_COLOR,
                input_port_border=NODE_POSE_INPUT_PORT_BORDER_COLOR,
                output_port=NODE_POSE_OUTPUT_PORT_COLOR,
                output_port_border=NODE_POSE_OUTPUT_PORT_BORDER_COLOR,
            )
        if hasattr(v, "set_highlight_colors"):
            v.set_highlight_colors(
                panel_bg=NODE_JUMP_PANEL_BG_HIGHLIGHT_COLOR,
                input_port=NODE_POSE_INPUT_PORT_HIGHLIGHT_COLOR,
                input_port_border=NODE_POSE_INPUT_PORT_HIGHLIGHT_BORDER_COLOR,
                output_port=NODE_POSE_OUTPUT_PORT_HIGHLIGHT_COLOR,
                output_port_border=NODE_POSE_OUTPUT_PORT_HIGHLIGHT_BORDER_COLOR,
            )
        for port in self.input_ports():
            port.color = NODE_POSE_INPUT_PORT_COLOR
            port.border_color = NODE_POSE_INPUT_PORT_BORDER_COLOR
        is_function = getattr(self, "jump_type", "action") == "function"
        for i, port in enumerate(self.output_ports()):
            if is_function and i == 0:
                # 戻り先ポートは緑で表示
                port.color = (80, 200, 120, 255)
                port.border_color = (60, 160, 90, 255)
            else:
                port.color = NODE_POSE_OUTPUT_PORT_COLOR
                port.border_color = NODE_POSE_OUTPUT_PORT_BORDER_COLOR
        self._resize_for_body()
        self.refresh_body_text()
        try:
            QtCore.QTimer.singleShot(10, self._do_position_outputs)
        except Exception:
            pass

    def _on_double_click(self, event):
        if hasattr(self, "graph") and hasattr(self.graph, "show_jump_editor"):
            try:
                graph_view = self.graph.viewer()
                scene_pos = event.scenePos()
                view_pos = graph_view.mapFromScene(scene_pos)
                screen_pos = graph_view.mapToGlobal(view_pos)
                self.graph.show_jump_editor(self, screen_pos)
            except Exception as e:
                print(f"[JumpNode] Error on double click: {e}")
                traceback.print_exc()


# ---------------------------------------------------------------------------
# MemoNode — 純粋な注釈用ノード。Export では無視される。
# 98% 白のタイトル/パネル。入力ポート無し、出力ポートは透明で見えないが
# 実接続可能 (Export 時は無視)。ドット位置には memo_text のプレビュー (単一行
# 末尾 ... クリップ) を描画。ダブルクリックで MemoEditDialog。
# ---------------------------------------------------------------------------
NODE_MEMO_TITLE_COLOR = (70, 70, 70)              # dark gray on white
NODE_MEMO_TITLE_HIGHLIGHT_COLOR = (30, 60, 130)   # blue on selection
# NodeGraphQt 親クラスの _paint_horizontal がタイトル領域に QColor(0,0,0,80)
# を塗るので、それを完全に隠すには不透明な白を上塗りする必要がある。
NODE_MEMO_TITLE_BG_COLOR = (250, 250, 250, 255)   # opaque 98% white
NODE_MEMO_TITLE_BG_HIGHLIGHT_COLOR = (225, 235, 255, 255)
NODE_MEMO_PANEL_BG_COLOR = (250, 250, 250)        # 98% white
NODE_MEMO_PANEL_BG_HIGHLIGHT_COLOR = (240, 245, 255)
# ポートの色: パネル色と揃えて視覚上見えないようにする
NODE_MEMO_OUTPUT_PORT_COLOR = NODE_MEMO_PANEL_BG_COLOR
NODE_MEMO_OUTPUT_PORT_BORDER_COLOR = NODE_MEMO_PANEL_BG_COLOR
# Memo 由来のパイプ (矢印) の色: グレー
NODE_MEMO_PIPE_COLOR = (140, 140, 140)


class MemoNode(BaseNode):
    """注釈用のメモノード。Export では無視される。"""

    __identifier__ = "motion.nodes"
    NODE_NAME = "MemoNode"
    __view__ = CustomNodeItem

    def __init__(self):
        super(MemoNode, self).__init__(CustomNodeItem)
        # 入力ポート無し（add_input を呼ばない）
        self.output_count = 0
        self._add_memo_output()

        self.memo_text = ""
        self.modal_width = 420
        self.modal_height = 320
        self.set_name("Memo")

        # ポート配置をタイトル右端に切替えるフラグ (CustomNodeItem._align_ports_horizontal で参照)
        self.view._memo_port_at_title_right = True
        # 本文プレビューを左寄せ描画 (CustomNodeItem.paint で参照)
        self.view._body_align_left = True

        self._original_double_click = self.view.mouseDoubleClickEvent
        self.view.mouseDoubleClickEvent = self._on_double_click

        QtCore.QTimer.singleShot(20, self._apply_memo_colors)
        # レイアウトが完全に整ってから再配置
        QtCore.QTimer.singleShot(30, self._reposition_memo_port)

    def _reposition_memo_port(self):
        """ポートをタイトル右端に強制再配置。他のイベントで下部へ戻された場合の
        リカバリ用。_align_ports_horizontal 内の分岐で通常時も同じ位置に配置される。"""
        v = getattr(self, "view", None)
        if v is None:
            return
        try:
            v._memo_port_at_title_right = True
            if hasattr(v, "_align_ports"):
                v._align_ports(0.0)
            v.update()
        except Exception as _e:
            print(f"[MemoNode] reposition port failed: {_e}")

    def _add_memo_output(self):
        self.output_count += 1
        port_name = f"out_{self.output_count}"
        super(MemoNode, self).add_output(port_name, display_name=False)
        return port_name

    def _apply_memo_colors(self):
        v = getattr(self, "view", None)
        if not v:
            return
        if hasattr(v, "set_title_color"):
            v.set_title_color(
                *NODE_MEMO_TITLE_COLOR,
                highlight_color=NODE_MEMO_TITLE_HIGHLIGHT_COLOR,
            )
        if hasattr(v, "_title_bg_color"):
            v._title_bg_color = QtGui.QColor(*NODE_MEMO_TITLE_BG_COLOR)
        self.set_color(*NODE_MEMO_PANEL_BG_COLOR)
        if hasattr(v, "set_normal_colors"):
            v.set_normal_colors(
                panel_bg=NODE_MEMO_PANEL_BG_COLOR,
                input_port=NODE_MEMO_OUTPUT_PORT_COLOR,
                input_port_border=NODE_MEMO_OUTPUT_PORT_BORDER_COLOR,
                output_port=NODE_MEMO_OUTPUT_PORT_COLOR,
                output_port_border=NODE_MEMO_OUTPUT_PORT_BORDER_COLOR,
            )
        if hasattr(v, "set_highlight_colors"):
            v.set_highlight_colors(
                panel_bg=NODE_MEMO_PANEL_BG_HIGHLIGHT_COLOR,
                input_port=NODE_MEMO_OUTPUT_PORT_COLOR,
                input_port_border=NODE_MEMO_OUTPUT_PORT_BORDER_COLOR,
                output_port=NODE_MEMO_OUTPUT_PORT_COLOR,
                output_port_border=NODE_MEMO_OUTPUT_PORT_BORDER_COLOR,
            )
        for port in self.output_ports():
            port.color = NODE_MEMO_OUTPUT_PORT_COLOR
            port.border_color = NODE_MEMO_OUTPUT_PORT_BORDER_COLOR
        # タイトル背景も透明にする（title_bg_rgb は 4-tuple 対応）
        v._normal_title_bg = NODE_MEMO_TITLE_BG_COLOR
        v._highlight_title_bg = NODE_MEMO_TITLE_BG_HIGHLIGHT_COLOR
        self.refresh_body_text()

    # 東アジア幅ベースの重み: full-width / wide 文字は 2、それ以外は 1。
    # 合計 24 単位まで表示し、超えたら文頭 + '...' に切る。
    # これで全角 12 文字 / 半角 24 文字を上限として、文頭が確実に見える。
    _MEMO_PREVIEW_MAX_WEIGHT = 24

    @staticmethod
    def _clip_text_east_asian(text: str, max_weight: int) -> str:
        weight = 0
        out = []
        for ch in text:
            w = 2 if unicodedata.east_asian_width(ch) in ("F", "W") else 1
            if weight + w > max_weight:
                return "".join(out) + "..."
            weight += w
            out.append(ch)
        return "".join(out)

    def refresh_body_text(self):
        """メモ内容の単一行プレビュー。改行はスペースに置換。
        東アジア幅で 24 単位 (全角 12 / 半角 24) 相当までを左寄せで見せ、
        超えた分は '...' で終了。文頭が必ず読める。"""
        v = getattr(self, "view", None)
        if v is None:
            return
        text = (self.memo_text or "")
        text = text.replace("\r\n", " ").replace("\n", " ").replace("\r", " ").strip()
        if not text:
            preview = "(empty)"
        else:
            preview = MemoNode._clip_text_east_asian(text, self._MEMO_PREVIEW_MAX_WEIGHT)
        v._body_text = preview
        v.update()

    def _on_double_click(self, event):
        if hasattr(self, "graph") and hasattr(self.graph, "show_memo_editor"):
            try:
                self.graph.show_memo_editor(self)
            except Exception as e:
                print(f"[MemoNode] Error on double click: {e}")
                traceback.print_exc()


class MemoEditDialog(QtWidgets.QDialog):
    """MemoNode の本文を編集するリサイズ可能なテキストモーダル。
    閉じたときに memo_text と modal サイズをノードへ保存する。"""

    def __init__(self, memo_node, parent=None):
        super(MemoEditDialog, self).__init__(parent)
        self.memo_node = memo_node
        self.setWindowTitle("Memo")
        # ノードごとに保存されているモーダルサイズを復元
        w = int(getattr(memo_node, "modal_width", 420) or 420)
        h = int(getattr(memo_node, "modal_height", 320) or 320)
        self.resize(w, h)
        self.setMinimumSize(240, 140)
        self.setSizeGripEnabled(True)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)
        self._text = QtWidgets.QTextEdit(self)
        self._text.setPlainText(getattr(memo_node, "memo_text", "") or "")
        self._text.setAcceptRichText(False)
        self._text.setLineWrapMode(QtWidgets.QTextEdit.WidgetWidth)
        layout.addWidget(self._text, 1)

        btn_row = QtWidgets.QHBoxLayout()
        btn_row.addStretch()
        close_btn = QtWidgets.QPushButton("Close")
        close_btn.clicked.connect(self.accept)
        btn_row.addWidget(close_btn)
        layout.addLayout(btn_row)

    def accept(self):
        self._save()
        super(MemoEditDialog, self).accept()

    def closeEvent(self, event):
        self._save()
        super(MemoEditDialog, self).closeEvent(event)

    def _save(self):
        try:
            self.memo_node.memo_text = self._text.toPlainText()
            self.memo_node.modal_width = int(self.width())
            self.memo_node.modal_height = int(self.height())
            if hasattr(self.memo_node, "refresh_body_text"):
                self.memo_node.refresh_body_text()
        except Exception as _e:
            print(f"[MemoEditDialog] save failed: {_e}")


JOINT_ROW_MIME_TYPE = "application/x-meridian-joint-row"


class JointRowWidget(QtWidgets.QWidget):
    """JointEditor内でドラッグ移動できるジョイント行"""

    def __init__(self, joint_name, joint_editor, parent=None):
        super(JointRowWidget, self).__init__(parent)
        self.joint_name = joint_name
        self.joint_editor = joint_editor
        self._drag_start_pos = None
        self._drop_indicator = None
        self.setAcceptDrops(True)
        self.setCursor(QtCore.Qt.OpenHandCursor)

    def _event_pos(self, event):
        if hasattr(event, "position"):
            return event.position().toPoint()
        return event.pos()

    def mousePressEvent(self, event):
        if event.button() == QtCore.Qt.LeftButton:
            self._drag_start_pos = self._event_pos(event)
            self.setCursor(QtCore.Qt.ClosedHandCursor)
        super(JointRowWidget, self).mousePressEvent(event)

    def mouseReleaseEvent(self, event):
        self.setCursor(QtCore.Qt.OpenHandCursor)
        super(JointRowWidget, self).mouseReleaseEvent(event)

    def mouseMoveEvent(self, event):
        if not (event.buttons() & QtCore.Qt.LeftButton) or self._drag_start_pos is None:
            super(JointRowWidget, self).mouseMoveEvent(event)
            return
        if (self._event_pos(event) - self._drag_start_pos).manhattanLength() < QtWidgets.QApplication.startDragDistance():
            super(JointRowWidget, self).mouseMoveEvent(event)
            return

        drag = QtGui.QDrag(self)
        mime = QtCore.QMimeData()
        mime.setData(JOINT_ROW_MIME_TYPE, self.joint_name.encode("utf-8"))
        drag.setMimeData(mime)
        drag.exec_(QtCore.Qt.MoveAction) if hasattr(drag, "exec_") else drag.exec(QtCore.Qt.MoveAction)
        self.setCursor(QtCore.Qt.OpenHandCursor)

    def dragEnterEvent(self, event):
        if event.mimeData().hasFormat(JOINT_ROW_MIME_TYPE):
            event.acceptProposedAction()
        else:
            super(JointRowWidget, self).dragEnterEvent(event)

    def dragMoveEvent(self, event):
        if event.mimeData().hasFormat(JOINT_ROW_MIME_TYPE):
            source = bytes(event.mimeData().data(JOINT_ROW_MIME_TYPE)).decode("utf-8")
            if source == self.joint_name:
                self.clear_drop_indicator()
            else:
                pos = self._event_pos(event)
                self.set_drop_indicator("bottom" if pos.y() >= (self.height() / 2) else "top")
                self.joint_editor.clear_column_drop_indicators()
            event.acceptProposedAction()
        else:
            super(JointRowWidget, self).dragMoveEvent(event)

    def dragLeaveEvent(self, event):
        self.clear_drop_indicator()
        super(JointRowWidget, self).dragLeaveEvent(event)

    def dropEvent(self, event):
        if not event.mimeData().hasFormat(JOINT_ROW_MIME_TYPE):
            super(JointRowWidget, self).dropEvent(event)
            return
        source = bytes(event.mimeData().data(JOINT_ROW_MIME_TYPE)).decode("utf-8")
        if source == self.joint_name:
            self.clear_drop_indicator()
            event.acceptProposedAction()
            return
        pos = self._event_pos(event)
        insert_after = pos.y() >= (self.height() / 2)
        self.joint_editor.move_joint_row(
            source,
            target_group=self.joint_editor.joint_display_groups.get(self.joint_name, self.joint_editor._joint_group(self.joint_name)),
            target_jname=self.joint_name,
            insert_after=insert_after,
        )
        self.joint_editor.clear_drop_indicators()
        event.acceptProposedAction()

    def set_drop_indicator(self, position):
        if self._drop_indicator != position:
            self._drop_indicator = position
            self.update()

    def clear_drop_indicator(self):
        if self._drop_indicator is not None:
            self._drop_indicator = None
            self.update()

    def paintEvent(self, event):
        super(JointRowWidget, self).paintEvent(event)
        if self._drop_indicator not in ("top", "bottom"):
            return
        painter = QtGui.QPainter(self)
        pen = QtGui.QPen(QtGui.QColor(*MINT_GREEN_COLOR), 3)
        painter.setPen(pen)
        y = 1 if self._drop_indicator == "top" else self.height() - 2
        painter.drawLine(0, y, self.width(), y)


class JointColumnGroupBox(QtWidgets.QGroupBox):
    """ジョイント行を列末尾へドロップできるグループボックス"""

    def __init__(self, group_key, title, joint_editor, parent=None):
        super(JointColumnGroupBox, self).__init__(title, parent)
        self.group_key = group_key
        self.joint_editor = joint_editor
        self._drop_line_y = None
        self._drop_target_jname = None
        self._drop_insert_after = False
        self.setAcceptDrops(True)
        # Set title color to dark gray
        self.setStyleSheet("QGroupBox::title { color: #555555; }")

    def _event_pos(self, event):
        if hasattr(event, "position"):
            return event.position().toPoint()
        return event.pos()

    def _row_widgets(self):
        layout = self.joint_editor.column_layouts.get(self.group_key)
        if not layout:
            return []
        rows = []
        for i in range(layout.count()):
            widget = layout.itemAt(i).widget()
            if isinstance(widget, JointRowWidget):
                rows.append(widget)
        return rows

    def _update_drop_line_from_pos(self, pos):
        rows = self._row_widgets()
        if not rows:
            self._drop_target_jname = None
            self._drop_insert_after = False
            self._drop_line_y = max(24, self.height() - 8)
            self.update()
            return

        nearest = None
        for row in rows:
            top = row.mapTo(self, QtCore.QPoint(0, 0)).y()
            bottom = top + row.height()
            for y, insert_after in ((top, False), (bottom, True)):
                distance = abs(pos.y() - y)
                if nearest is None or distance < nearest[0]:
                    nearest = (distance, row, y, insert_after)

        _distance, row, y, insert_after = nearest
        self._drop_target_jname = row.joint_name
        self._drop_insert_after = insert_after
        self._drop_line_y = y
        self.update()

    def dragEnterEvent(self, event):
        if event.mimeData().hasFormat(JOINT_ROW_MIME_TYPE):
            self.joint_editor.clear_row_drop_indicators()
            self._update_drop_line_from_pos(self._event_pos(event))
            event.acceptProposedAction()
        else:
            super(JointColumnGroupBox, self).dragEnterEvent(event)

    def dragMoveEvent(self, event):
        if event.mimeData().hasFormat(JOINT_ROW_MIME_TYPE):
            self.joint_editor.clear_row_drop_indicators()
            self._update_drop_line_from_pos(self._event_pos(event))
            event.acceptProposedAction()
        else:
            super(JointColumnGroupBox, self).dragMoveEvent(event)

    def dragLeaveEvent(self, event):
        self.set_drop_highlight(False)
        super(JointColumnGroupBox, self).dragLeaveEvent(event)

    def dropEvent(self, event):
        if not event.mimeData().hasFormat(JOINT_ROW_MIME_TYPE):
            super(JointColumnGroupBox, self).dropEvent(event)
            return
        source = bytes(event.mimeData().data(JOINT_ROW_MIME_TYPE)).decode("utf-8")
        self._update_drop_line_from_pos(self._event_pos(event))
        self.joint_editor.move_joint_row(
            source,
            target_group=self.group_key,
            target_jname=self._drop_target_jname,
            insert_after=self._drop_insert_after,
        )
        self.joint_editor.clear_drop_indicators()
        event.acceptProposedAction()

    def set_drop_highlight(self, highlighted):
        if highlighted:
            self._drop_line_y = max(24, self.height() - 8)
        else:
            self._drop_line_y = None
            self._drop_target_jname = None
            self._drop_insert_after = False
        self.update()

    def paintEvent(self, event):
        super(JointColumnGroupBox, self).paintEvent(event)
        if self._drop_line_y is None:
            return
        painter = QtGui.QPainter(self)
        pen = QtGui.QPen(QtGui.QColor(*MINT_GREEN_COLOR), 3)
        painter.setPen(pen)
        margin = 8
        y = max(18, min(self.height() - 4, self._drop_line_y))
        painter.drawLine(margin, y, self.width() - margin, y)


def _lr_counterpart_joint(jname: str) -> "str | None":
    """Return the L↔R counterpart joint name, or None for non-paired joints.

    Handles both canonical MJCF names (l_*/r_* prefix) and legacy URDF
    link-chain names (contains _l_ or _r_ infix).
    """
    if jname.startswith("l_"):
        # Also replace any remaining _l_ segments (e.g. l_leg_upper_to_l_leg_lower)
        return "r_" + jname[2:].replace("_l_", "_r_")
    if jname.startswith("r_"):
        return "l_" + jname[2:].replace("_r_", "_l_")
    if "_l_" in jname:
        return jname.replace("_l_", "_r_")
    if "_r_" in jname:
        return jname.replace("_r_", "_l_")
    return None



class JointEditorPanel(QtWidgets.QWidget):
    """関節角度をSlider+SpinBoxで編集するパネル"""

    angles_changed = QtCore.Signal(dict)
    # Live チェックボックス切替シグナル（3Dビュー側と同期する）
    live_toggled = QtCore.Signal(bool)

    def __init__(self, parent=None):
        super(JointEditorPanel, self).__init__(parent)
        self.robot_model = None
        self.sliders = {}
        self.spinboxes = {}
        self.easing_combos = {}
        self.joint_rows = {}
        self.joint_name_labels = {}
        self.joint_display_groups = {}
        self.joint_display_order = []
        self.column_layouts = {}
        self.column_widgets = {}
        self.current_pose_node = None
        self.joint_settings = {}
        self.joint_group_presets = []
        self.current_group_preset_index = -1
        self.group_master_base_angles = {}
        self._updating_group_controls = False
        self.graph = None
        self._updating = False
        self._undo_push_fn = None  # set from build_motion_editor
        settings = load_app_settings()
        self.always_on_top = settings.get("joint_sliders_always_on_top", True)
        self.show_c_joints = settings.get("show_c_joints", True)
        self.step_snapping_enabled = settings.get("joint_sliders_step_snapping", False)
        self.step_snapping_deg = float(settings.get("joint_sliders_step_deg", 1.0))
        self.pair_enabled = settings.get("joint_sliders_pair_enabled", False)
        self._setup_ui()
        self._refresh_pose_meta_row()
        self._apply_always_on_top(restore_visible=False)

    def _setup_ui(self):
        self.main_layout = QtWidgets.QVBoxLayout(self)
        self.main_layout.setContentsMargins(4, 4, 4, 4)

        # 1行目: Name, Duration, Frames, preset buttons, Easing, L↔R
        self.pose_meta_widget = QtWidgets.QWidget()
        pose_meta_layout = QtWidgets.QHBoxLayout(self.pose_meta_widget)
        pose_meta_layout.setContentsMargins(0, 4, 0, 6)
        pose_meta_layout.setSpacing(8)

        # Name
        pose_meta_layout.addWidget(create_label("Name:"))
        self.pose_name_edit = QtWidgets.QLineEdit()
        self.pose_name_edit.setFixedWidth(120)
        self.pose_name_edit.editingFinished.connect(self._apply_pose_name_from_ui)
        pose_meta_layout.addWidget(self.pose_name_edit)

        # Duration
        duration_container = QtWidgets.QHBoxLayout()
        duration_container.setContentsMargins(0, 0, 0, 0)
        duration_container.setSpacing(0)
        duration_container.addWidget(create_label("Duration:"))
        self.pose_duration_label = QtWidgets.QLabel("—")
        self.pose_duration_label.setFixedWidth(70)
        self.pose_duration_label.setAlignment(
            QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter
        )
        self.pose_duration_label.setToolTip(
            "Duration = Frames / Simulation Step Rate (FPS)"
        )
        self.pose_duration_label.setStyleSheet(
            "color: black; padding: 2px 0px;"
        )
        duration_container.addWidget(self.pose_duration_label)
        pose_meta_layout.addLayout(duration_container)

        # Frames
        pose_meta_layout.addWidget(create_label("Frames:"))
        self.pose_frames_spin = QtWidgets.QSpinBox()
        self.pose_frames_spin.setRange(1, 9999)
        self.pose_frames_spin.setValue(get_default_hz_fps())
        self.pose_frames_spin.setFixedWidth(60)
        self.pose_frames_spin.setToolTip(
            "Update rate for this pose segment in Hz. Sets the playback timer interval."
        )
        self.pose_frames_spin.valueChanged.connect(self._apply_pose_frames_from_ui)
        self.pose_frames_spin.editingFinished.connect(self._apply_pose_frames_from_ui)
        pose_meta_layout.addWidget(self.pose_frames_spin)

        # Frame preset buttons (configurable via Settings)
        self.frame_preset_buttons = []
        for preset_val in get_frame_presets():
            btn = QtWidgets.QPushButton(str(preset_val))
            btn.setFixedWidth(36)
            btn.clicked.connect(lambda checked, b=btn: self._set_frames_preset(int(b.text())))
            pose_meta_layout.addWidget(btn)
            self.frame_preset_buttons.append(btn)

        # Easing (same size as Line-L combo)
        easing_label = create_label("Easing:")
        pose_meta_layout.addWidget(easing_label)
        self.bulk_easing_combo = QtWidgets.QComboBox()
        self.bulk_easing_combo.addItems(EASING_OPTIONS)
        self.bulk_easing_combo.setCurrentIndex(0)
        # "color: black;" alone only fixes the closed-box text; on Windows the
        # popup list (QAbstractItemView) ignores the palette and needs its own
        # rule or its item text renders white-on-white when the dropdown opens.
        self.bulk_easing_combo.setStyleSheet(
            "QComboBox { color: black; } QComboBox QAbstractItemView { color: black; }"
        )
        self.bulk_easing_combo.currentIndexChanged.connect(self._on_bulk_easing_changed)
        pose_meta_layout.addWidget(self.bulk_easing_combo)

        # L<->R swap button
        self._lr_swap_btn = QtWidgets.QPushButton("L↔R")
        self._lr_swap_btn.setFixedWidth(42)
        self._lr_swap_btn.setToolTip("Swap left and right joint angles")
        self._lr_swap_btn.clicked.connect(self._on_lr_swap)
        pose_meta_layout.addWidget(self._lr_swap_btn)

        # Live: スライダー/3Dビューでの角度変更をリアルタイムに Valkey へ送る
        # 3Dビュー側の Live と同期。Valkey が OFF のときは外部から disabled にされる。
        _live_initial = bool(load_app_settings().get("valkey_live_enabled", False))
        self.live_check = QtWidgets.QCheckBox("Live")
        self.live_check.setChecked(_live_initial)
        self.live_check.setEnabled(False)  # 起動直後は Valkey 状態を見て親から有効化
        self.live_check.setToolTip(
            "Realtime: send angle changes to Valkey while dragging sliders / 3D view"
        )
        self.live_check.toggled.connect(lambda v: self.live_toggled.emit(v))
        pose_meta_layout.addWidget(self.live_check)

        pose_meta_layout.addStretch()
        self.main_layout.addWidget(self.pose_meta_widget)

        # 2行目: Stay on Top, Show Line-C, Step, etc. (dark gray text)
        options_layout = QtWidgets.QHBoxLayout()
        options_layout.setContentsMargins(0, 0, 0, 0)
        options_layout.setSpacing(12)

        # UbuntuではQCheckBox::indicatorの枠線がパレット任せになり背景に埋もれて
        # 見えなくなる（Macはネイティブ描画で枠線が常に見える）ため、枠線だけでなく
        # 背景をグレーにして周囲(白)とのコントラストで箱の存在が分かるようにする。
        dark_gray_style = (
            "QCheckBox { color: #555555; } "
            "QCheckBox::indicator { width: 13px; height: 13px; "
            "border: 1px solid #888888; border-radius: 2px; background-color: #c0c0c0; } "
            "QCheckBox::indicator:checked { background-color: #4a90d9; border: 1px solid #2f6fb5; }"
        )

        self.always_on_top_checkbox = QtWidgets.QCheckBox("Stay on Top")
        self.always_on_top_checkbox.setStyleSheet(dark_gray_style)
        self.always_on_top_checkbox.setChecked(self.always_on_top)
        self.always_on_top_checkbox.toggled.connect(self._on_always_on_top_toggled)
        options_layout.addWidget(self.always_on_top_checkbox)

        self.show_c_checkbox = QtWidgets.QCheckBox("Show Line-C")
        self.show_c_checkbox.setStyleSheet(dark_gray_style)
        self.show_c_checkbox.setChecked(self.show_c_joints)
        self.show_c_checkbox.toggled.connect(self._on_show_c_toggled)
        options_layout.addWidget(self.show_c_checkbox)

        self.step_snapping_checkbox = QtWidgets.QCheckBox("Step")
        self.step_snapping_checkbox.setStyleSheet(dark_gray_style)
        self.step_snapping_checkbox.setChecked(self.step_snapping_enabled)
        self.step_snapping_checkbox.toggled.connect(self._on_step_snapping_toggled)
        options_layout.addWidget(self.step_snapping_checkbox)

        self.step_snapping_input = QtWidgets.QDoubleSpinBox()
        self.step_snapping_input.setRange(0.1, 360.0)
        self.step_snapping_input.setDecimals(1)
        self.step_snapping_input.setSingleStep(0.1)
        self.step_snapping_input.setValue(max(0.1, self.step_snapping_deg))
        self.step_snapping_input.setFixedWidth(70)
        self.step_snapping_input.valueChanged.connect(self._on_step_snapping_value_changed)
        options_layout.addWidget(self.step_snapping_input)

        step_unit_label = create_label("deg")
        step_unit_label.setStyleSheet(dark_gray_style)
        options_layout.addWidget(step_unit_label)

        self.group_preset_combo = QtWidgets.QComboBox()
        self.group_preset_combo.setStyleSheet(_MAIN_WINDOW_COMBO_TEXT_STYLE)
        self.group_preset_combo.currentIndexChanged.connect(self._on_group_preset_changed)
        options_layout.addWidget(self.group_preset_combo)
        self.group_master_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.group_master_slider.setRange(-1800, 1800)
        self.group_master_slider.setValue(0)
        self.group_master_slider.setFixedWidth(160)
        self.group_master_slider.valueChanged.connect(self._on_group_master_changed)
        self.group_master_slider.sliderPressed.connect(self._on_group_master_pressed)
        options_layout.addWidget(self.group_master_slider)
        self.group_master_value_label = create_label("0.0 deg")
        self.group_master_value_label.setFixedWidth(60)
        options_layout.addWidget(self.group_master_value_label)
        self.group_master_center_button = QtWidgets.QPushButton("Center")
        self.group_master_center_button.clicked.connect(self._center_group_master)
        options_layout.addWidget(self.group_master_center_button)

        self.zero_button = QtWidgets.QPushButton("zero")
        self.zero_button.clicked.connect(self._on_zero_button_clicked)
        options_layout.addWidget(self.zero_button)

        self.home_button = QtWidgets.QPushButton("Home")
        self.home_button.clicked.connect(self._on_home_button_clicked)
        options_layout.addWidget(self.home_button)

        self.pair_checkbox = QtWidgets.QCheckBox("Pair")
        self.pair_checkbox.setStyleSheet(dark_gray_style)
        self.pair_checkbox.setChecked(self.pair_enabled)
        self.pair_checkbox.toggled.connect(self._on_pair_toggled)
        options_layout.addWidget(self.pair_checkbox)

        options_layout.addStretch()
        self.main_layout.addLayout(options_layout)
        self._refresh_group_preset_controls()

        self.scroll_area = QtWidgets.QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)
        self.scroll_content = QtWidgets.QWidget()
        self.scroll_layout = QtWidgets.QHBoxLayout(self.scroll_content)
        self.scroll_layout.setContentsMargins(0, 0, 0, 0)
        self.scroll_layout.setSpacing(6)
        self.scroll_area.setWidget(self.scroll_content)
        self.main_layout.addWidget(self.scroll_area)

    def refresh_pose_duration_from_frames(self):
        """再生 FPS 変更などでラベルだけ更新（Duration は node に保存された秒のまま）。"""
        self._update_pose_duration_label()

    def _update_pose_duration_label(self):
        node = self.current_pose_node
        if not node:
            self.pose_duration_label.setText("—")
            return
        frames = getattr(node, "frames", get_default_hz_fps())
        fps = get_default_hz_fps()
        duration_sec = frames / fps if fps > 0 else 0
        self.pose_duration_label.setText(f"{duration_sec:.2f} sec")

    def _on_group_master_pressed(self):
        """Called when group master slider is pressed — push undo before drag."""
        if self._undo_push_fn:
            self._undo_push_fn()

    def _sync_pose_frames_only(self):
        """Frames変更時にDuration表示を更新（Duration = Frames / FPS）。"""
        node = self.current_pose_node
        if not node:
            self.pose_duration_label.setText("—")
            return
        node.frames = int(self.pose_frames_spin.value())
        self._update_pose_duration_label()
        # Redraw the left badge "[N]f" right away.
        try:
            if getattr(node, "view", None) is not None:
                node.view.update()
        except Exception:
            pass

    def _set_frames_preset(self, value):
        """Framesプリセットボタンから値を設定"""
        if self._undo_push_fn:
            self._undo_push_fn()
        self.pose_frames_spin.setValue(value)
        self._sync_pose_frames_only()

    def update_frame_presets(self, presets):
        """Update frame preset button labels from settings"""
        if not hasattr(self, 'frame_preset_buttons'):
            return
        for i, btn in enumerate(self.frame_preset_buttons):
            if i < len(presets):
                btn.setText(str(presets[i]))

    def _refresh_pose_meta_row(self):
        node = self.current_pose_node
        self.pose_name_edit.blockSignals(True)
        self.pose_frames_spin.blockSignals(True)
        if node:
            self.pose_name_edit.setEnabled(True)
            self.pose_frames_spin.setEnabled(True)
            self.pose_name_edit.setText(node.pose_name)
            self.pose_frames_spin.setValue(
                getattr(node, "frames", get_default_hz_fps()))
            self._update_pose_duration_label()
        else:
            self.pose_name_edit.setEnabled(False)
            self.pose_frames_spin.setEnabled(False)
            self.pose_name_edit.clear()
            self.pose_frames_spin.setValue(get_default_hz_fps())
            self.pose_duration_label.setText("—")
        self.pose_name_edit.blockSignals(False)
        self.pose_frames_spin.blockSignals(False)

    def _apply_pose_name_from_ui(self):
        if self.current_pose_node:
            if self._undo_push_fn:
                self._undo_push_fn()
            text = self.pose_name_edit.text()
            self.current_pose_node.pose_name = text
            self.current_pose_node.set_name(text)

    def _apply_pose_frames_from_ui(self, *args):
        if self._undo_push_fn:
            self._undo_push_fn()
        self._sync_pose_frames_only()

    def show_for_pose_node(self, node, screen_pos=None):
        """Pose ダブルクリック等: 先に開いている Pose へ保存してから切り替え。"""
        _ = screen_pos
        self._save_to_node()
        self.set_current_pose_node(node)
        if node and self.robot_model:
            self.set_angles(node.angles_deg)
            self._update_3dview()

    def _on_always_on_top_toggled(self, checked):
        self.always_on_top = checked
        settings = load_app_settings()
        settings["joint_sliders_always_on_top"] = checked
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
                self.activateWindow()

    def _on_show_c_toggled(self, checked):
        self.show_c_joints = checked
        settings = load_app_settings()
        settings["show_c_joints"] = checked
        save_app_settings(settings)
        self._apply_c_visibility()

    def _on_step_snapping_toggled(self, checked):
        self.step_snapping_enabled = checked
        settings = load_app_settings()
        settings["joint_sliders_step_snapping"] = checked
        save_app_settings(settings)

    def _on_step_snapping_value_changed(self, value):
        self.step_snapping_deg = max(0.1, float(value))
        settings = load_app_settings()
        settings["joint_sliders_step_deg"] = self.step_snapping_deg
        save_app_settings(settings)

    def _snap_slider_value(self, slider, val):
        if not self.step_snapping_enabled:
            return val
        step = max(0.1, self.step_snapping_deg)
        snapped_deg = round((val / 10.0) / step) * step
        snapped_val = int(round(snapped_deg * 10))
        return max(slider.minimum(), min(slider.maximum(), snapped_val))

    def _on_bulk_easing_changed(self, index):
        if self._updating:
            return
        if self._undo_push_fn:
            self._undo_push_fn()
        self._updating = True
        for combo in self.easing_combos.values():
            combo.setCurrentIndex(index)
        self._updating = False
        self._save_to_node()

    def _on_joint_easing_changed(self, _index):
        if self._updating:
            return
        if self._undo_push_fn:
            self._undo_push_fn()
        self._sync_bulk_easing_combo()
        self._save_to_node()

    def _sync_bulk_easing_combo(self):
        if not self.easing_combos:
            self.bulk_easing_combo.setCurrentIndex(0)
            return
        indices = {combo.currentIndex() for combo in self.easing_combos.values()}
        self._updating = True
        self.bulk_easing_combo.setCurrentIndex(indices.pop() if len(indices) == 1 else -1)
        self._updating = False

    def _default_joint_setting(self, jname):
        default_rev = jname.startswith("r_") and jname.endswith("_xr")
        return {
            "display_name": jname,
            "rev": default_rev,
            "max_speed_rad_s": DEFAULT_JOINT_SPEED,
            "speed_preset_name": "",
        }

    def _default_joint_group_preset(self):
        return {
            "name": "Group1",
            "master": 0.0,
            "members": {
                jname: {"enabled": False, "scale": 1.0}
                for jname in self.joint_display_order
            },
        }

    def _ensure_joint_group_presets(self):
        valid_joints = set(self.spinboxes.keys())
        for preset in self.joint_group_presets:
            members = preset.setdefault("members", {})
            for jname in valid_joints:
                current = members.get(jname, {})
                members[jname] = {
                    "enabled": bool(current.get("enabled", False)),
                    "scale": float(current.get("scale", 1.0)),
                }
            for jname in list(members.keys()):
                if jname not in valid_joints:
                    del members[jname]
            preset.setdefault("name", "Group1")
            preset["master"] = float(preset.get("master", 0.0))
        if self.joint_group_presets:
            self.current_group_preset_index = min(
                max(-1, self.current_group_preset_index),
                len(self.joint_group_presets) - 1
            )
        else:
            self.current_group_preset_index = -1

    def get_joint_group_presets(self):
        self._ensure_joint_group_presets()
        return json.loads(json.dumps(self.joint_group_presets))

    def set_joint_group_presets(self, presets, current_index=-1):
        self.joint_group_presets = presets if isinstance(presets, list) else []
        self.current_group_preset_index = current_index if isinstance(current_index, int) else -1
        self._ensure_joint_group_presets()
        self._refresh_group_preset_controls()
        self._select_group_preset(self.current_group_preset_index, reset_master=True)

    def _refresh_group_preset_controls(self):
        if not hasattr(self, "group_preset_combo"):
            return
        self._updating_group_controls = True
        self.group_preset_combo.clear()
        self.group_preset_combo.addItem("Individual")
        for idx, preset in enumerate(self.joint_group_presets):
            name = preset.get("name", "") or f"Preset {idx + 1}"
            self.group_preset_combo.addItem(name)
        combo_index = self.current_group_preset_index + 1
        combo_index = max(0, min(combo_index, self.group_preset_combo.count() - 1))
        self.group_preset_combo.setCurrentIndex(combo_index)
        self._updating_group_controls = False

    def _group_preset(self, index=None):
        idx = self.current_group_preset_index if index is None else index
        if idx < 0 or idx >= len(self.joint_group_presets):
            return None
        return self.joint_group_presets[idx]

    def _on_group_preset_changed(self, combo_index):
        if self._updating_group_controls:
            return
        if self._undo_push_fn:
            self._undo_push_fn()
        self._select_group_preset(combo_index - 1, reset_master=True)

    def _select_group_preset(self, preset_index, reset_master=True):
        self.current_group_preset_index = preset_index
        self._ensure_joint_group_presets()
        self.group_master_base_angles = self.get_angles()
        self._updating_group_controls = True
        if reset_master:
            self.group_master_slider.setValue(0)
        self.group_master_value_label.setText(f"{self.group_master_slider.value() / 10.0:.1f} deg")
        self._updating_group_controls = False
        self._apply_group_visibility()

    def _on_group_master_changed(self, value):
        snapped = self._snap_slider_value(self.group_master_slider, value)
        if snapped != value:
            self._updating_group_controls = True
            self.group_master_slider.setValue(snapped)
            self._updating_group_controls = False
            value = snapped
        self.group_master_value_label.setText(f"{value / 10.0:.1f} deg")
        if not self._updating_group_controls:
            self._apply_group_master()

    def _apply_group_master(self):
        preset = self._group_preset()
        if not preset:
            return
        master = self.group_master_slider.value() / 10.0
        angles = dict(self.group_master_base_angles)
        for jname, member in preset.get("members", {}).items():
            if member.get("enabled", False) and jname in self.spinboxes:
                base = self.group_master_base_angles.get(jname, self.spinboxes[jname].value())
                angles[jname] = base + master * float(member.get("scale", 1.0))
        self.set_angles(angles)
        self._update_3dview()

    def _center_group_master(self):
        self.group_master_base_angles = self.get_angles()
        self._updating_group_controls = True
        self.group_master_slider.setValue(0)
        self.group_master_value_label.setText("0.0 deg")
        self._updating_group_controls = False

    def _on_zero_button_clicked(self):
        """Set all sliders to 0 degrees."""
        if self._undo_push_fn:
            self._undo_push_fn()
        zero_angles = {jname: 0.0 for jname in self.spinboxes}
        self.set_angles(zero_angles)
        self._update_3dview()
        self._center_group_master()

    def _on_home_button_clicked(self):
        """Move sliders to Home position."""
        if self._undo_push_fn:
            self._undo_push_fn()
        home_angles = getattr(self, "home_position_angles", None)
        if home_angles:
            self.set_angles(home_angles)
            self._update_3dview()
            self._center_group_master()
        else:
            # No home position set, default to zero
            self._on_zero_button_clicked()

    def _resolve_home_target_angles(self):
        home_angles = getattr(self, "home_position_angles", None)
        if home_angles:
            return dict(home_angles)
        return {jname: 0.0 for jname in self.spinboxes}

    def apply_partial_body_home(self, part: str):
        """Apply Home only to upper or lower body joints (part: 'upper' | 'lower')."""
        if not self.robot_model:
            print("[JointEditor] No robot model — partial Home skipped")
            return
        upper, lower = classify_upper_lower_body_joints(self.robot_model)
        target = upper if part == "upper" else lower
        if not target:
            print(f"[JointEditor] No joints classified for partial Home ({part})")
            return
        if self._undo_push_fn:
            self._undo_push_fn()
        home_angles = self._resolve_home_target_angles()
        current = self.get_angles()
        merged = dict(current)
        for jname in target:
            if jname in merged:
                merged[jname] = float(home_angles.get(jname, 0.0))
        self.set_angles(merged)
        self._update_3dview()
        self._center_group_master()

    def set_home_position(self, angles):
        """Set the Home position angles."""
        self.home_position_angles = dict(angles) if angles else {}

    def _apply_group_visibility(self):
        """グループプリセット非対象の関節を薄く表示していたが、デフォルト preset ですべて
        enabled=False だと全スライダーがグレーに見えて操作不能に見えるため、常に通常表示。"""
        for row_widget in self.joint_rows.values():
            row_widget.setGraphicsEffect(None)

    def _ensure_joint_settings(self):
        if not self.robot_model:
            return
        for jname in self.robot_model.joint_order:
            current = self.joint_settings.get(jname, {})
            if "rev" in current:
                rev = bool(current["rev"])
            elif "dir" in current:
                rev = current["dir"] == "CCW"
            else:
                rev = jname.startswith("r_") and jname.endswith("_xr")
            self.joint_settings[jname] = {
                "display_name": current.get("display_name", jname),
                "rev": rev,
                "max_speed_rad_s": float(current.get("max_speed_rad_s", DEFAULT_JOINT_SPEED)),
                "speed_preset_name": str(current.get("speed_preset_name", "") or ""),
            }

    def _apply_joint_settings_to_ui(self):
        for jname, label in self.joint_name_labels.items():
            setting = self.joint_settings.get(jname, self._default_joint_setting(jname))
            display_name = setting.get("display_name", jname)
            rev = bool(setting.get("rev", jname.startswith("r_") and jname.endswith("_xr")))
            max_speed_rads = float(setting.get("max_speed_rad_s", DEFAULT_JOINT_SPEED))
            max_speed_degs = math.degrees(max_speed_rads)
            label.setText(display_name)
            label.setToolTip(
                f"Original: {jname}\nRev: {rev}\nMax speed: {max_speed_degs:.2f} deg/s ({max_speed_rads:.4f} rad/s)")

    def get_joint_settings(self):
        self._ensure_joint_settings()
        return {
            jname: {
                "display_name": setting.get("display_name", jname),
                "rev": bool(setting.get("rev", jname.startswith("r_") and jname.endswith("_xr"))),
                "max_speed_rad_s": float(setting.get("max_speed_rad_s", DEFAULT_JOINT_SPEED)),
                "speed_preset_name": str(setting.get("speed_preset_name", "") or ""),
            }
            for jname, setting in self.joint_settings.items()
        }

    def set_joint_settings(self, settings):
        self.joint_settings = {}
        if self.robot_model:
            for jname in self.robot_model.joint_order:
                current = settings.get(jname, {}) if isinstance(settings, dict) else {}
                if "rev" in current:
                    rev = bool(current["rev"])
                elif "dir" in current:
                    rev = current["dir"] == "CCW"
                else:
                    rev = jname.startswith("r_") and jname.endswith("_xr")
                self.joint_settings[jname] = {
                    "display_name": current.get("display_name", jname),
                    "rev": rev,
                    "max_speed_rad_s": float(current.get("max_speed_rad_s", DEFAULT_JOINT_SPEED)),
                    "speed_preset_name": str(current.get("speed_preset_name", "") or ""),
                }
                self._apply_joint_limits_to_controls(jname)
        self._apply_joint_settings_to_ui()
        if self.spinboxes:
            self._update_3dview()

    def _joint_is_rev(self, jname):
        """関節が Rev（逆転）かどうか。Rev=True の時、スライダー増加 → モデル角は減少。"""
        setting = self.joint_settings.get(jname, {})
        if "rev" in setting:
            return bool(setting["rev"])
        if "dir" in setting:
            return setting["dir"] == "CCW"
        return jname.startswith("r_") and jname.endswith("_xr")

    def get_angles_for_3d(self, angles=None):
        """UI display angles → FK model angles (negate Rev joints)."""
        source = self.get_angles() if angles is None else angles
        return {jname: -v if self._joint_is_rev(jname) else v for jname, v in source.items()}

    def fk_to_ui_angles(self, fk_angles):
        """FK model angles → UI display angles (inverse of get_angles_for_3d)."""
        return {jname: -v if self._joint_is_rev(jname) else v for jname, v in fk_angles.items()}

    def _joint_display_limits(self, jname):
        if not self.robot_model or jname not in self.robot_model.joints:
            return -180.0, 180.0
        jt = self.robot_model.joints[jname]
        if self._joint_is_rev(jname):
            return -jt.limit_upper, -jt.limit_lower
        return jt.limit_lower, jt.limit_upper

    def _apply_joint_limits_to_controls(self, jname):
        if jname not in self.spinboxes or jname not in self.sliders:
            return
        lower, upper = self._joint_display_limits(jname)
        centered_limit = max(abs(lower), abs(upper))
        spin = self.spinboxes[jname]
        slider = self.sliders[jname]
        self._updating = True
        spin.setRange(lower, upper)
        slider.setMinimum(int(round(-centered_limit * 10)))
        slider.setMaximum(int(round(centered_limit * 10)))
        clamped = max(lower, min(upper, spin.value()))
        spin.setValue(clamped)
        slider.setValue(int(round(clamped * 10)))
        self._updating = False

    def get_joint_layout(self):
        self._rebuild_joint_display_order()
        return {
            "order": list(self.joint_display_order),
            "groups": dict(self.joint_display_groups),
        }

    def set_joint_layout(self, layout_data):
        if not self.robot_model or not isinstance(layout_data, dict):
            return
        order = layout_data.get("order", [])
        groups = layout_data.get("groups", {})
        if not isinstance(order, list) or not isinstance(groups, dict):
            return

        valid_joints = set(self.robot_model.joint_order)
        self.joint_display_order = [j for j in order if j in valid_joints]
        self.joint_display_order += [
            j for j in self.robot_model.joint_order
            if j not in self.joint_display_order
        ]
        self.joint_display_groups = {
            jname: groups.get(jname, self._joint_group(jname))
            for jname in self.robot_model.joint_order
        }
        for jname, group in list(self.joint_display_groups.items()):
            if group not in self.column_layouts:
                self.joint_display_groups[jname] = self._joint_group(jname)
        self._rebuild_joint_rows_from_layout()

    def _joint_dialog_parent(self):
        """User Value / Define と同様、メインウィンドウを親にする。"""
        g = getattr(self, "graph", None)
        w = getattr(g, "widget", None) if g else None
        top = w.window() if w else None
        return top if top is not None else self.window()

    def _show_joint_settings_dialog(self):
        self._ensure_joint_settings()
        dialog = JointSettingsDialog(self, self._joint_dialog_parent())
        dialog.exec()

    def _show_joint_group_dialog(self):
        self._ensure_joint_group_presets()
        dialog = JointGroupDialog(self, self._joint_dialog_parent())
        dialog.exec()

    def _apply_c_visibility(self):
        c_widget = self.column_widgets.get("C")
        if c_widget:
            c_widget.setVisible(self.show_c_joints)

    def _joint_group(self, joint_name):
        prefix = joint_name.strip().upper()[:1]
        if prefix == "L":
            return "L"
        if prefix == "R":
            return "R"
        return "C"

    def get_ordered_joint_names(self):
        order = [j for j in self.joint_display_order if j in self.spinboxes]
        order += [j for j in self.spinboxes.keys() if j not in order]
        return order

    def _create_joint_column(self, key, title):
        group_box = JointColumnGroupBox(key, title, self)
        layout = QtWidgets.QVBoxLayout(group_box)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(2)
        self.column_widgets[key] = group_box
        self.column_layouts[key] = layout
        self.scroll_layout.addWidget(group_box, stretch=1)
        return group_box, layout

    def build_from_robot(self, robot_model):
        """ロボットモデルからスライダーを生成"""
        self.robot_model = robot_model
        previous_groups = dict(self.joint_display_groups)
        previous_order = list(self.joint_display_order)
        # 既存をクリア
        self.sliders.clear()
        self.spinboxes.clear()
        self.easing_combos.clear()
        self.joint_rows.clear()
        self.joint_name_labels.clear()
        self.joint_display_groups.clear()
        self.joint_display_order.clear()
        self.column_layouts.clear()
        self.column_widgets.clear()
        while self.scroll_layout.count():
            item = self.scroll_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        for key, title in (("L", "Line-L"), ("R", "Line-R"), ("C", "Line-C")):
            self._create_joint_column(key, title)

        ordered_joints = [j for j in previous_order if j in robot_model.joint_order]
        ordered_joints += [j for j in robot_model.joint_order if j not in ordered_joints]
        # Passive (closed-loop 従属) joint は slider を持たせない。値は
        # solve_passive_angles で active から自動計算される。
        _passive = getattr(robot_model, 'passive_joints', set()) or set()
        if _passive:
            ordered_joints = [j for j in ordered_joints if j not in _passive]

        for jname in ordered_joints:
            jt = robot_model.joints[jname]
            row_widget = JointRowWidget(jname, self)
            row_layout = QtWidgets.QHBoxLayout(row_widget)
            row_layout.setContentsMargins(0, 0, 0, 0)
            row_layout.setSpacing(4)

            label = create_label(jname)
            label.setFixedWidth(95)
            label.setToolTip(f"{jt.limit_lower:.1f} ~ {jt.limit_upper:.1f} deg")
            row_layout.addWidget(label)

            slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
            lower, upper = self._joint_display_limits(jname)
            centered_limit = max(abs(lower), abs(upper))
            slider.setMinimum(int(round(-centered_limit * 10)))
            slider.setMaximum(int(round(centered_limit * 10)))
            slider.setValue(0)
            row_layout.addWidget(slider)

            spin = ArithmeticDoubleSpinBox()
            spin.setRange(lower, upper)
            spin.setDecimals(1)
            spin.setSingleStep(1.0)
            spin.setValue(0.0)
            # 数字2.5文字分だけ幅を広げ、桁数の多い値（例: -120.0）でも
            # 上下ボタンに文字が隠れないようにする。
            _digit_w = QtGui.QFontMetrics(spin.font()).horizontalAdvance("0")
            spin.setFixedWidth(62 + round(_digit_w * 2.5))
            row_layout.addWidget(spin)

            easing_combo = QtWidgets.QComboBox()
            easing_combo.addItems(EASING_OPTIONS)
            easing_combo.setCurrentIndex(0)
            easing_combo.setFixedWidth(100)
            # "QComboBox { color: black; }" fixes the closed-box display text
            # (without it, the current-selection text renders white-on-white
            # on Windows — confirmed visually, not just a theoretical risk).
            _popup_style = (
                "QComboBox { color: black; } "
                "QComboBox QAbstractItemView { color: black; }"
            )
            if sys.platform == "win32":
                # Windows: in the open popup, "color: black" alone only
                # reliably paints the hovered/selected row (that path goes
                # through the style's highlight-drawing code, which honours
                # it); resting rows fall back to native rendering where the
                # text stays white-on-white. Explicitly styling the
                # background too makes Qt fully own the item painting so the
                # text colour actually applies to every row, not just the
                # hovered one.
                _popup_style = (
                    "QComboBox { color: black; } "
                    "QComboBox QAbstractItemView { color: black; background: white; }"
                )
            easing_combo.setStyleSheet(_popup_style)
            row_layout.addWidget(easing_combo)

            self.sliders[jname] = slider
            self.spinboxes[jname] = spin
            self.easing_combos[jname] = easing_combo
            self.joint_rows[jname] = row_widget
            self.joint_name_labels[jname] = label

            # スライダー変更時は3Dビュー更新のみ（リアルタイム）
            slider.valueChanged.connect(
                lambda val, jn=jname: self._on_slider_changed(jn, val))
            # スライダーリリース時にノードへ保存
            slider.sliderReleased.connect(
                lambda jn=jname: self._on_slider_released(jn))

            # スピンボックス変更時は3Dビュー更新のみ（リアルタイム）
            spin.valueChanged.connect(
                lambda val, jn=jname: self._on_spin_changed(jn, val))
            # スピンボックス確定時にノードへ保存
            spin.editingFinished.connect(
                lambda jn=jname: self._on_spin_finished(jn))
            easing_combo.currentIndexChanged.connect(self._on_joint_easing_changed)

            group = previous_groups.get(jname, self._joint_group(jname))
            if group not in self.column_layouts:
                group = self._joint_group(jname)
            self.joint_display_groups[jname] = group
            self.joint_display_order.append(jname)
            self.column_layouts[group].addWidget(row_widget)

        for layout in self.column_layouts.values():
            layout.addStretch()
        self._ensure_joint_settings()
        self._apply_joint_settings_to_ui()
        self._ensure_joint_group_presets()
        self._refresh_group_preset_controls()
        self._select_group_preset(self.current_group_preset_index, reset_master=True)
        self.set_joint_easings({})
        self._apply_c_visibility()

    def _layout_insert_index(self, layout, target_jname=None, insert_after=False):
        if target_jname and target_jname in self.joint_rows:
            idx = layout.indexOf(self.joint_rows[target_jname])
            if idx >= 0:
                return idx + (1 if insert_after else 0)
        stretch_idx = max(0, layout.count() - 1)
        return stretch_idx

    def _rebuild_joint_display_order(self):
        order = []
        for group in ("L", "R", "C"):
            layout = self.column_layouts.get(group)
            if not layout:
                continue
            for i in range(layout.count()):
                widget = layout.itemAt(i).widget()
                if isinstance(widget, JointRowWidget):
                    order.append(widget.joint_name)
                    self.joint_display_groups[widget.joint_name] = group
        self.joint_display_order = order

    def clear_row_drop_indicators(self):
        for row_widget in self.joint_rows.values():
            if hasattr(row_widget, "clear_drop_indicator"):
                row_widget.clear_drop_indicator()

    def clear_column_drop_indicators(self):
        for column_widget in self.column_widgets.values():
            if hasattr(column_widget, "set_drop_highlight"):
                column_widget.set_drop_highlight(False)

    def clear_drop_indicators(self):
        self.clear_row_drop_indicators()
        self.clear_column_drop_indicators()

    def move_joint_row(self, source_jname, target_group, target_jname=None, insert_after=False):
        row_widget = self.joint_rows.get(source_jname)
        target_layout = self.column_layouts.get(target_group)
        if not row_widget or not target_layout:
            return
        if self._undo_push_fn:
            self._undo_push_fn()

        current_group = self.joint_display_groups.get(source_jname)
        current_layout = self.column_layouts.get(current_group)
        if current_layout:
            current_layout.removeWidget(row_widget)

        insert_index = self._layout_insert_index(target_layout, target_jname, insert_after)
        target_layout.insertWidget(insert_index, row_widget)
        row_widget.show()
        self.joint_display_groups[source_jname] = target_group
        self._rebuild_joint_display_order()
        self._apply_c_visibility()

    def _rebuild_joint_rows_from_layout(self):
        if not self.robot_model:
            return
        for layout in self.column_layouts.values():
            for jname, row_widget in self.joint_rows.items():
                if layout.indexOf(row_widget) >= 0:
                    layout.removeWidget(row_widget)
        for jname in self.joint_display_order:
            row_widget = self.joint_rows.get(jname)
            if not row_widget:
                continue
            group = self.joint_display_groups.get(jname, self._joint_group(jname))
            layout = self.column_layouts.get(group)
            if layout:
                layout.insertWidget(self._layout_insert_index(layout), row_widget)
                row_widget.show()
        self._rebuild_joint_display_order()
        self._apply_c_visibility()

    def _pair_needs_mirror(self, primary_name: str, pair_name: str) -> bool:
        """Return True if pair joint UI value should be negated for left-right mirror.

        Uses JOINT_TO_MERIDIM sign product: L joints have sign +1.0, axis-inverted R
        joints (roll/yaw) have sign -1.0, so their product < 0 → needs mirror negate.
        Pitch joints share the same sign on both sides → no negate.
        """
        j = JOINT_TO_MERIDIM.get(primary_name)
        p = JOINT_TO_MERIDIM.get(pair_name)
        if j is not None and p is not None:
            return j[1] * p[1] < 0
        # Fallback for joints not in JOINT_TO_MERIDIM
        pn = pair_name.lower()
        return pn.startswith("r_") and (pn.endswith("_xr") or pn.endswith("_zy"))

    def _on_slider_changed(self, jname, val):
        if self._updating:
            return
        self._updating = True
        slider = self.sliders[jname]
        snapped_val = self._snap_slider_value(slider, val)
        if snapped_val != val:
            slider.setValue(snapped_val)
        deg = snapped_val / 10.0
        self.spinboxes[jname].setValue(deg)
        actual_val = int(round(self.spinboxes[jname].value() * 10))
        if actual_val != snapped_val:
            slider.setValue(actual_val)
        if self.pair_enabled:
            pair_name = self._get_pair_joint_name(jname)
            if pair_name and pair_name in self.sliders:
                primary_deg = self.spinboxes[jname].value()
                pair_deg = -primary_deg if self._pair_needs_mirror(jname, pair_name) else primary_deg
                self.sliders[pair_name].setValue(int(round(pair_deg * 10)))
                self.spinboxes[pair_name].setValue(pair_deg)
        self._updating = False
        self._update_3dview()

    def _on_spin_changed(self, jname, val):
        if self._updating:
            return
        self._updating = True
        self.sliders[jname].setValue(int(round(float(val) * 10)))
        if self.pair_enabled:
            pair_name = self._get_pair_joint_name(jname)
            if pair_name and pair_name in self.spinboxes:
                pair_val = -val if self._pair_needs_mirror(jname, pair_name) else val
                self.spinboxes[pair_name].setValue(pair_val)
                self.sliders[pair_name].setValue(int(round(float(pair_val) * 10)))
        self._updating = False
        self._update_3dview()

    def _get_pair_joint_name(self, joint_name):
        """l_xxx ↔ r_xxx のペア関節名を返す。対応なければ None。"""
        if joint_name.startswith("l_"):
            return "r_" + joint_name[2:]
        if joint_name.startswith("r_"):
            return "l_" + joint_name[2:]
        return None

    def _on_pair_toggled(self, checked):
        self.pair_enabled = checked
        s = load_app_settings()
        s["joint_sliders_pair_enabled"] = checked
        save_app_settings(s)

    def _update_3dview(self):
        """3Dビューを更新（リアルタイム）"""
        angles = self.get_angles_for_3d()
        self.angles_changed.emit(angles)

    def _on_slider_released(self, jname):
        """スライダーリリース時"""
        dbg("[TRIGGER]", f"Slider released", joint=jname)
        if self._undo_push_fn:
            self._undo_push_fn()
        self._save_to_node()

    def _on_spin_finished(self, jname):
        """スピンボックス確定時"""
        dbg("[TRIGGER]", f"SpinBox editingFinished", joint=jname)
        if self._undo_push_fn:
            self._undo_push_fn()
        self._save_to_node()

    def _save_to_node(self):
        """選択中のPoseNodeに角度を保存（リリース/確定時）"""
        if self.current_pose_node is not None:
            angles = self.get_angles()
            node_id = id(self.current_pose_node)
            node_name = self.current_pose_node.pose_name
            dbg("[SAVE]", f"Saving angles to node",
                node_id=node_id, name=node_name, angles=angles)
            self.current_pose_node.angles_deg = dict(angles)
            self.current_pose_node.joint_easings = self.get_joint_easings()
            dbg("[SAVE]", f"After save, node.angles_deg",
                node_id=node_id, angles_deg=self.current_pose_node.angles_deg)
        else:
            dbg("[SAVE]", "current_pose_node is None, skip save")

    def get_angles(self):
        result = {}
        for jname, spin in self.spinboxes.items():
            result[jname] = spin.value()
        return result

    def get_joint_easings(self):
        return {
            jname: easing_option(combo.currentText())
            for jname, combo in self.easing_combos.items()
        }

    def set_joint_easings(self, joint_easings):
        self._updating = True
        for jname, combo in self.easing_combos.items():
            combo.setCurrentIndex(easing_index(
                joint_easings.get(jname, EASING_OPTIONS[0]) if isinstance(joint_easings, dict) else EASING_OPTIONS[0]
            ))
        self._updating = False
        self._sync_bulk_easing_combo()

    def set_angles(self, angles_deg):
        """anglesをUIに反映（シグナルを出さない）"""
        dbg("[LOAD]", f"set_angles called", angles_deg=angles_deg)
        self._updating = True
        for jname, deg in angles_deg.items():
            if jname in self.spinboxes:
                self.spinboxes[jname].setValue(deg)
                self.sliders[jname].setValue(int(deg * 10))
        self._updating = False

    def _update_slider(self, joint_name, angle_deg):
        """単一ジョイントのスライダーを更新（シグナルを出さない）"""
        if joint_name not in self.spinboxes:
            return
        self._updating = True
        self.spinboxes[joint_name].setValue(angle_deg)
        self.sliders[joint_name].setValue(int(angle_deg * 10))
        self._updating = False

    def _on_lr_swap(self):
        """Mirror L/R joint angles in FK space, independent of Rev settings."""
        if self._undo_push_fn:
            self._undo_push_fn()

        # Operate in FK space so mirror is correct regardless of per-joint Rev settings.
        # UI-space swap only works when l_rev != r_rev for mirror-negate joints,
        # but joints like hipjoint_zy/xr have both sides Rev=False.
        fk_angles = self.get_angles_for_3d()
        new_fk = dict(fk_angles)
        paired: set = set()

        for jname in list(fk_angles.keys()):
            if jname in paired:
                continue
            counterpart = _lr_counterpart_joint(jname)
            if not counterpart or counterpart not in fk_angles or counterpart in paired:
                continue

            j = JOINT_TO_MERIDIM.get(jname)
            p = JOINT_TO_MERIDIM.get(counterpart)
            if j is not None and p is not None:
                mirror_negate = j[1] * p[1] < 0
            else:
                cn = counterpart.lower()
                mirror_negate = cn.startswith("r_") and (cn.endswith("_xr") or cn.endswith("_zy"))

            if mirror_negate:
                new_fk[jname] = -fk_angles[counterpart]
                new_fk[counterpart] = -fk_angles[jname]
            else:
                new_fk[jname] = fk_angles[counterpart]
                new_fk[counterpart] = fk_angles[jname]

            paired.add(jname)
            paired.add(counterpart)

        # Negate unpaired center (c_) joints in FK space
        for jname in list(new_fk.keys()):
            if jname in paired:
                continue
            jn = jname.lower()
            if jn.startswith("c_") or jn.split("_")[0] == "c":
                new_fk[jname] = -new_fk[jname]

        self.set_angles(self.fk_to_ui_angles(new_fk))
        self._save_to_node()
        self._update_3dview()

    def set_current_pose_node(self, node):
        old_id = id(self.current_pose_node) if self.current_pose_node else None
        new_id = id(node) if node else None
        new_name = node.pose_name if node else None
        dbg("[NODE]", f"set_current_pose_node", old_id=old_id, new_id=new_id, name=new_name)
        self.current_pose_node = node
        self.set_joint_easings(getattr(node, 'joint_easings', {}) if node else {})
        self._refresh_pose_meta_row()



# branch/jump ノードは1tickごとに次ノードへ進む(t=1.0がほぼ即座に成立する
# instant-transition)ため、Branch/Jumpを高速ループさせる歩行のようなアクション
# では _get_next_node()/_start_segment() のデバッグprintが毎tick(最大LOOP_HZ)
# 発行されうる。コンソールへの書き込みが遅い環境(確認済み: 一部のUbuntu環境)
# では、このprint自体のI/O待ちが tick 処理時間を支配し、要求tick間隔(10ms等)を
# 大きく超えて実質のtickレートが頭打ちになり、再生全体がスローモーションに
# 見える原因になっていた。Pose間の通常再生はセグメントが長く遷移頻度が低いため
# 影響が目立たない。既定で無効にし、必要な時だけTrueにして調査する。
_PLAYBACK_VERBOSE_LOG = False


def _playback_log(msg):
    if _PLAYBACK_VERBOSE_LOG:
        print(msg)


class PlaybackController(QtCore.QObject):
    """モーション再生を管理"""

    pose_changed = QtCore.Signal(dict)
    playback_finished = QtCore.Signal()
    node_incomplete = QtCore.Signal(object)  # Emitted when node didn't reach target angles
    node_highlight = QtCore.Signal(object)

    def __init__(self, parent=None):
        super(PlaybackController, self).__init__(parent)
        self.timer = QtCore.QTimer()
        self.timer.setTimerType(QtCore.Qt.PreciseTimer)
        self.timer.timeout.connect(self._tick)
        self.fps = 100
        self.interpolation = EASING_OPTIONS[0]
        self.branch_mode = "default-first"

        self.is_playing = False
        self.is_paused = False

        self.segments = []
        self.current_segment_idx = 0
        self.segment_start_time = 0  # セグメント開始時刻(ms)
        self.prev_angles = {}
        self.next_angles = {}
        self.next_easings = {}
        self.segment_duration_ms = 0  # セグメント継続時間(ms)

        self.graph = None
        self.robot_model = None
        self._manual_dialog_parent = None

        # 再生中の「実効」関節角（サーボ最高速度でクランプ後）と tick 用の実時間
        self._speed_limited_angles = {}
        self._last_tick_time_ms = None

        # Pose 区間の 3D 更新レート [ms]（Frames Hz から決定。非 Pose では ~33ms）
        self._playback_tick_ms = 33

        # 実時間計測用
        self.elapsed_timer = QtCore.QElapsedTimer()

        # JumpNode cross-action playback state
        self._jump_original_action_idx = None  # Original action index when playback started
        self._jump_callback = None  # Callback for switching actions: callback(target_idx, save_current) -> BaseLinkNode
        self._virtual_action_idx = None  # Current action index when playing virtual nodes

        # Single-node mode: stop after the first segment (used by long-press)
        self._single_node_mode = False

        # Action-only mode: stop at JumpNode instead of crossing action boundaries
        self._action_only_mode = False

        # Function JumpNodes executed this playback (avoid double-call for non-computed)
        self._function_jumps_executed = set()

        # Computed motion (ProjectCode IK via LMEMotionRuntime)
        self.motion_runtime = None
        self._computed_motion_active = False
        self._ik_executing = False
        self.home_position_angles = {}

    def _collect_computed_func_names(self, action_data) -> list:
        if not isinstance(action_data, dict):
            return []
        names = []
        for n in (action_data.get("nodes", []) or []):
            if isinstance(n, dict) and n.get("jump_type") == "function":
                fn = n.get("jump_target_function", "") or ""
                if fn:
                    names.append(fn)
        return names

    def _halt_computed_motion_runtime(self) -> None:
        rt = self.motion_runtime
        cmdr = getattr(rt, "_commander", None) if rt else None
        if cmdr is None:
            return
        ctrl = getattr(cmdr, "controller", None) or getattr(cmdr, "walk", None)
        fn = getattr(ctrl, "stop", None) or getattr(ctrl, "stop_walk", None) if ctrl is not None else None
        if callable(fn):
            try:
                fn()
            except Exception:
                pass

    def _set_ik_gate_for_target(self, target_node) -> None:
        """Enable/disable every-tick IK based on the next graph node."""
        fn = ""
        if (isinstance(target_node, (JumpNode, VirtualJumpNode))
                and getattr(target_node, "jump_type", "") == "function"):
            fn = getattr(target_node, "jump_target_function", "") or ""
        if fn and fn in self._computed_func_names:
            self._ik_executing = True
            return
        ctrl = getattr(getattr(self.motion_runtime, "_commander", None), "controller", None)
        if ctrl is None:
            ctrl = getattr(getattr(self.motion_runtime, "_commander", None), "walk", None)
        mid = isinstance(target_node, (BranchingNode, VirtualBranchingNode))
        gathering = bool(ctrl is not None and getattr(ctrl, "busy", False))
        if mid and gathering:
            self._ik_executing = True
            return
        if not mid:
            self._halt_computed_motion_runtime()
        self._ik_executing = False

    def set_home_position_angles(self, angles):
        self.home_position_angles = dict(angles) if angles else {}

    def _action_is_computed_motion(self) -> bool:
        """True if the current action contains any function JumpNode."""
        action_data, _ = self._action_snapshot(self._effective_action_index())
        if not isinstance(action_data, dict):
            return False
        return any(
            isinstance(n, dict) and n.get("jump_type") == "function"
            for n in action_data.get("nodes", []) or []
        )

    def _effective_action_index(self):
        if self._virtual_action_idx is not None:
            return self._virtual_action_idx
        mas = getattr(self.graph, "motion_action_state", None) if self.graph else None
        if mas:
            return mas.get("current")
        return None

    def _action_snapshot(self, action_idx):
        mas = getattr(self.graph, "motion_action_state", None) if self.graph else None
        if not mas or action_idx is None:
            return None, ""
        items = mas.get("items", [])
        if action_idx < 0 or action_idx >= len(items):
            return None, ""
        entry = items[action_idx]
        return entry.get("data") or {}, str(entry.get("title") or "")

    def set_jump_callback(self, callback):
        """Set callback for cross-action jumping.

        callback signature: callback(target_action_idx, save_current=True) -> BaseLinkNode or None
        - target_action_idx: index of action to jump to
        - save_current: if True, save current action state before switching
        - returns: StartNode (BaseLinkNode) of target action, or None if failed
        """
        self._jump_callback = callback

    def play_from_start(self, graph, robot_model):
        """StartNode (BaseLinkNode) から再生開始"""
        # BaseLinkNodeを探す
        start_node = None
        for node in graph.all_nodes():
            if isinstance(node, BaseLinkNode):
                start_node = node
                break

        if not start_node:
            print("[Playback] No StartNode (BaseLinkNode) found")
            self.playback_finished.emit()
            return

        self.play(start_node, graph, robot_model)

    def play(self, start_node, graph, robot_model, action_only=False):
        """再生開始"""
        # Single source of truth for _action_only_mode: every play() call (直接呼び出し
        # も play_action_only 経由も) explicitly states whether it's action-only, so a
        # stale True from a previous ▶︎. run can never leak into a later ▶︎_ run.
        self._action_only_mode = action_only
        self.graph = graph
        self.robot_model = robot_model
        self.segments = []
        self.current_segment_idx = 0
        self._visited_nodes = set()  # Track visited nodes for cycle detection
        self._function_jumps_executed = set()

        self._computed_func_names = []
        self._ik_executing = False
        if self._action_is_computed_motion():
            from LegacyMotionEditor_Utils import LMEMotionRuntime
            self.motion_runtime = LMEMotionRuntime()
            project_code = getattr(graph, "project_code", "") or ""
            self.motion_runtime.reset(project_code)
            self._computed_motion_active = True
            action_data, _ = self._action_snapshot(self._effective_action_index())
            self._computed_func_names = self._collect_computed_func_names(action_data)
            _playback_log(f"[Playback] Computed motion runtime initialised, funcs={self._computed_func_names}")
        else:
            self.motion_runtime = None
            self._computed_motion_active = False

        # グラフを辿ってセグメントリストを構築
        path = self._build_path(start_node)
        if len(path) < 2:
            print("[Playback] Need at least 2 nodes to play")
            self.playback_finished.emit()
            return

        for i in range(len(path) - 1):
            self.segments.append((path[i], path[i + 1]))
            self._visited_nodes.add(id(path[i]))
        if path:
            self._visited_nodes.add(id(path[-1]))

        # タイマー開始
        self.elapsed_timer.start()
        self._start_segment(0)
        self.is_playing = True
        self.is_paused = False
        self._restart_playback_timer()
        _playback_log(f"[Playback] Started with {len(self.segments)} segments")

    def _restart_playback_timer(self):
        ms = int(getattr(self, "_playback_tick_ms", 33))
        self.timer.start(max(1, ms))

    def pause(self):
        if self.is_playing and not self.is_paused:
            self.timer.stop()
            self.is_paused = True
        elif self.is_paused:
            self._resync_playback_tick_clock()
            self._restart_playback_timer()
            self.is_paused = False

    def stop(self):
        self.timer.stop()
        self.is_playing = False
        self.is_paused = False
        self.segments = []
        self._speed_limited_angles = {}
        self._last_tick_time_ms = None
        self._playback_tick_ms = 33

        # With virtual nodes, we don't switch the UI, so no need to "return"
        # Just log which action we were virtually playing
        if self._jump_original_action_idx is not None:
            _playback_log(f"[Playback] Finished (started from Action_{self._jump_original_action_idx + 1})")
            if self._virtual_action_idx is not None:
                _playback_log(f"[Playback] Was playing Action_{self._virtual_action_idx + 1} virtually")

        # Reset jump state
        self._jump_original_action_idx = None
        self._virtual_action_idx = None
        self._single_node_mode = False
        self._action_only_mode = False
        self._function_jumps_executed = set()

        # Clear computed motion runtime
        self.motion_runtime = None
        self._computed_motion_active = False
        self._ik_executing = False

        self.playback_finished.emit()
        _playback_log("[Playback] Stopped")

    def play_action_only(self, start_node, graph, robot_model):
        """Action-only mode: play from start_node, stop at any JumpNode."""
        self.play(start_node, graph, robot_model, action_only=True)

    def play_single_pose(self, target_node, graph, robot_model):
        """Long-press: interpolate from current robot position to target_node's pose."""
        if not isinstance(target_node, PoseNode):
            return

        self.stop()
        self.graph = graph
        self.robot_model = robot_model
        self._single_node_mode = True
        self._visited_nodes = set()

        # prev = current robot angles converted to UI space
        if robot_model:
            self.prev_angles = self._ui_angles_from_robot_phys(
                dict(robot_model.get_current_angles()))
        else:
            self.prev_angles = dict(target_node.angles_deg)

        self.next_angles = dict(target_node.angles_deg)
        self.next_easings = dict(getattr(target_node, 'joint_easings', {}))

        # Duration from node.frames / playback fps
        frames = max(1, int(getattr(target_node, 'frames', 1)))
        fps = max(1, float(self.fps))
        self.segment_duration_ms = max(1, int(round(frames / fps * 1000.0)))
        self._playback_tick_ms = 16  # ~60 Hz display refresh

        # segments list: dummy pair so _tick can unpack current_next
        self.segments = [(target_node, target_node)]
        self.current_segment_idx = 0

        # Speed-limited start state
        self._speed_limited_angles = dict(self.prev_angles)
        for k in self.next_angles:
            self._speed_limited_angles.setdefault(k, self.prev_angles.get(k, 0.0))

        self.elapsed_timer.start()
        self.segment_start_time = self.elapsed_timer.elapsed()
        self._last_tick_time_ms = self.elapsed_timer.elapsed()

        self.is_playing = True
        self.is_paused = False
        self._restart_playback_timer()
        self.node_highlight.emit(target_node)
        _playback_log(
            f"[Playback] Single-node play: {target_node.name()}, "
            f"frames={frames}, duration_ms={self.segment_duration_ms}")

    def _build_path(self, start_node):
        """start_nodeからグラフを辿ってノードのリストを構築

        BranchingNodeに到達したらパス構築を停止し、
        再生時に動的に条件評価して次のノードを決定する。
        Supports both real and virtual nodes.
        """
        path = [start_node]
        visited = {id(start_node)}
        current = start_node
        _playback_log(f"[Playback] Building path from: {start_node.name()}")
        while True:
            # BranchingNodeに到達したら、ここで一旦停止
            # 再生時に_tickで動的に次のノードを決定する
            if isinstance(current, (BranchingNode, VirtualBranchingNode)):
                branching_en = getattr(current, "branching_enabled", False)
                _playback_log(f"[Playback] At BranchingNode: {current.name()}, branching_enabled={branching_en}")
                if branching_en:
                    _playback_log(f"[Playback] Stopping at BranchingNode for dynamic evaluation: {current.name()}")
                    break

            next_node = self._get_next_node_simple(current)
            if next_node is None:
                _playback_log(f"[Playback] No next node from: {current.name()}")
                break
            if id(next_node) in visited:
                _playback_log(f"[Playback] Already visited: {next_node.name()}")
                break
            _playback_log(f"[Playback] Adding to path: {next_node.name()}")
            path.append(next_node)
            visited.add(id(next_node))
            current = next_node

            # Action-only mode (▶︎.): stop once a JumpNode is reached instead of
            # following whatever is wired to its output (which may lead further
            # through the graph, masking the intended stop point).
            if self._action_only_mode and isinstance(current, (JumpNode, VirtualJumpNode)):
                _playback_log(f"[Playback] Stopping at JumpNode (action-only mode): {current.name()}")
                break
        _playback_log(f"[Playback] Built path with {len(path)} nodes: {[n.name() for n in path]}")
        return path

    def _get_next_node_simple(self, node):
        """パス構築用：条件評価せずに最初の接続先を返す。Supports virtual nodes."""
        # Check if this is a virtual node
        is_virtual = isinstance(node, (
            VirtualBaseLinkNode, VirtualPoseNode, VirtualDefineNode, VirtualWaitNode,
            VirtualBranchingNode, VirtualJumpNode, VirtualMixNode
        ))
        if is_virtual:
            out_ports = node.output_ports()
            if out_ports:
                connected = out_ports[0].connected_ports()
                if connected:
                    return connected[0].node()
            return None

        # Real node: use _sorted_output_connections
        connected_nodes = _sorted_output_connections(node)
        if connected_nodes:
            return connected_nodes[0]
        return None

    def _resolve_branch_value(self, name):
        """UserVal_0, UserVal_1, Pad_*などの名前から値を解決する"""
        if not name:
            return 0

        # Check Pad registers first (exact match for dynamic values)
        if name in PAD_REGISTER_VALUES:
            return PAD_REGISTER_VALUES[name]

        # Check Pad register aliases (case-insensitive, for dynamic values)
        name_lower = name.lower()
        if name_lower in PAD_REGISTER_ALIASES:
            internal_name = PAD_REGISTER_ALIASES[name_lower]
            return PAD_REGISTER_VALUES.get(internal_name, 0)

        # Check fixed PS3 button bit values (for branch condition right-hand side)
        if name_lower in PAD_BUTTON_BIT_VALUES:
            return PAD_BUTTON_BIT_VALUES[name_lower]

        # Try to parse UserVal_n format
        if name.upper().startswith("USERVAL_"):
            try:
                idx = int(name[8:])  # UserVal_0 -> index 0
                if 0 <= idx < USER_VALUE_SESSION_COUNT:
                    uv_session = getattr(self.graph, "user_value_session", None)
                    if uv_session and isinstance(uv_session, list) and idx < len(uv_session):
                        slot = uv_session[idx]
                        if isinstance(slot, dict):
                            if slot.get("kind") == "literal":
                                return int(slot.get("value", 0))
                            elif slot.get("kind") == "register":
                                # Recursive resolve for register reference
                                return self._resolve_branch_value(slot.get("name", ""))
                return 0
            except (ValueError, TypeError):
                pass
        # Try to parse as direct integer
        try:
            return int(name)
        except (ValueError, TypeError):
            return 0

    def _evaluate_branch_condition(self, node):
        """BranchingNodeの条件を評価してTrue/Falseを返す"""
        if not getattr(node, "branching_enabled", False):
            return True  # Default to first branch if not enabled

        # PAD mode: IF PAD || [button] → True when the button IS pressed
        if getattr(node, "branch_if_pad_enabled", False):
            btn = getattr(node, "branch_if_pad_button", "L1")
            pad_key = PAD_IF_BUTTON_TO_PAD_KEY.get(btn)
            val = PAD_REGISTER_VALUES.get(pad_key, 0) if pad_key else 0
            is_pressed = bool(val)
            return is_pressed

        # PAD analog mode: IF PAD [axis] [>=|<=] [threshold]
        if getattr(node, "branch_if_pad_analog_enabled", False):
            axis = getattr(node, "branch_if_pad_analog_axis", "Lx")
            op = getattr(node, "branch_if_pad_analog_op", ">=")
            pad_key = PAD_IF_ANALOG_AXIS_TO_PAD_KEY.get(axis, "Pad_Lx")
            val = PAD_REGISTER_VALUES.get(pad_key, 0)
            threshold = int(getattr(node, "branch_if_pad_analog_threshold", 0))
            result = (val >= threshold) if op == ">=" else (val <= threshold)
            return result

        if not getattr(node, "branch_if_uv_enabled", True):
            return False

        left_name = getattr(node, "branch_if_left", "UserVal_0")
        op = getattr(node, "branch_if_op", "==")
        right_name = getattr(node, "branch_if_right", "UserVal_1")

        # Debug: Print Pad register values only if Pad_* or pad_* is used in condition
        def _is_pad_register(n):
            nl = n.lower()
            return n.startswith("Pad_") or nl in PAD_REGISTER_ALIASES or nl in PAD_BUTTON_BIT_VALUES
        if _is_pad_register(left_name) or _is_pad_register(right_name):
            all_zero = all(v == 0 for v in PAD_REGISTER_VALUES.values())
            if all_zero:
                _playback_log("[Branch] WARNING: All Pad values are 0 — open Pad Monitor and enable 'Use PC Pad'.")

        left_val = self._resolve_branch_value(left_name)
        right_val = self._resolve_branch_value(right_name)

        try:
            if op == "==":
                result = left_val == right_val
            elif op == "!=":
                result = left_val != right_val
            elif op == ">":
                result = left_val > right_val
            elif op == ">=":
                result = left_val >= right_val
            elif op == "<":
                result = left_val < right_val
            elif op == "<=":
                result = left_val <= right_val
            elif op == "and":
                result = bool(left_val) and bool(right_val)
            elif op == "or":
                result = bool(left_val) or bool(right_val)
            else:
                result = True
        except Exception as e:
            print(f"[Branch] Evaluation error: {e}")
            result = True

        return result

    def _get_jump_successor_node(self, node):
        """Follow the wired output of a JumpNode (real or virtual)."""
        if isinstance(node, VirtualJumpNode):
            out_ports = node.output_ports()
            if out_ports:
                connected = out_ports[0].connected_ports()
                if connected:
                    return connected[0].node()
            return None
        connected_nodes = _sorted_output_connections(node)
        return connected_nodes[0] if connected_nodes else None

    def _jump_to_action(self, target_action_idx, jump_type="action"):
        """Cross-action jump helper. Returns StartNode of target action or None."""
        _playback_log(f"[Playback] JumpNode: jumping to Action_{target_action_idx + 1} (type={jump_type})")

        # Determine current effective action index (virtual takes precedence)
        effective_current_idx = self._virtual_action_idx
        if effective_current_idx is None:
            mas = getattr(self.graph, "motion_action_state", None)
            if mas:
                effective_current_idx = mas.get("current", 0)

        # Same-action loop: jump to start of current action
        if effective_current_idx is not None and target_action_idx == effective_current_idx:
            _playback_log(f"[Playback] JumpNode: same-action loop for Action_{target_action_idx + 1}")
            self._visited_nodes.clear()
            if self._virtual_action_idx is None:
                for n in self.graph.all_nodes():
                    if isinstance(n, BaseLinkNode):
                        _playback_log("[Playback] JumpNode: returning to real graph StartNode")
                        return n
                _playback_log("[Playback] JumpNode: no BaseLinkNode found in real graph")
            elif self._jump_callback:
                start_node = self._jump_callback(target_action_idx, save_current=False)
                if start_node:
                    _playback_log("[Playback] JumpNode: rebuilt virtual graph for same-action loop")
                    return start_node
            return None

        if self._jump_callback:
            if self._jump_original_action_idx is None:
                mas = getattr(self.graph, "motion_action_state", None)
                if mas:
                    self._jump_original_action_idx = mas.get("current", 0)
                    _playback_log(f"[Playback] JumpNode: saved original action index {self._jump_original_action_idx}")

            start_node = self._jump_callback(target_action_idx, save_current=True)
            if start_node:
                _playback_log(f"[Playback] JumpNode: got StartNode from Action_{target_action_idx + 1}")
                self._visited_nodes.clear()
                # Update computed motion state for the new action
                new_data, _ = self._action_snapshot(target_action_idx)
                if isinstance(new_data, dict):
                    new_func_names = self._collect_computed_func_names(new_data)
                    if new_func_names:
                        self._computed_func_names = new_func_names
                        self._computed_motion_active = True
                        self._ik_executing = False  # BranchingNode in new action re-evaluates
                        if self.motion_runtime is None:
                            from LegacyMotionEditor_Utils import LMEMotionRuntime
                            project_code = getattr(self.graph, "project_code", "") or ""
                            self.motion_runtime = LMEMotionRuntime()
                            self.motion_runtime.reset(project_code)
                        _playback_log(f"[Playback] Computed motion continues in new action: {new_func_names}")
                    else:
                        self._computed_func_names = []
                        self._computed_motion_active = False
                        self._ik_executing = False
                        _playback_log("[Playback] Computed motion OFF (new action has no function JumpNodes)")
                return start_node
            _playback_log(f"[Playback] JumpNode: failed to get StartNode from Action_{target_action_idx + 1}")
        else:
            _playback_log("[Playback] JumpNode: no jump callback set, cannot jump to other action")
        return None

    def _get_next_node(self, node):
        """次のノードを取得 (supports both real and virtual nodes)"""
        if isinstance(node, (JumpNode, VirtualJumpNode)):
            jump_type = getattr(node, "jump_type", "action")
            if jump_type == "function":
                # IK is driven every tick by _tick(); just follow the wired successor.
                return self._get_jump_successor_node(node)

            target_action_idx = getattr(node, "jump_target_action_index", 0)
            return self._jump_to_action(target_action_idx, jump_type)

        is_branching = isinstance(node, (BranchingNode, VirtualBranchingNode))
        if is_branching and getattr(node, "branching_enabled", False):
            condition_result = self._evaluate_branch_condition(node)
            swapped = getattr(node, "branch_outputs_swapped", False)
            out_ports = node.output_ports()

            if len(out_ports) >= 2:
                if swapped:
                    target_port_idx = 0 if condition_result else 1
                else:
                    target_port_idx = 1 if condition_result else 0

                connected = out_ports[target_port_idx].connected_ports()
                if connected:
                    target_node = connected[0].node()
                    is_valid_target = isinstance(target_node, (
                        PoseNode, DefineNode, WaitNode, BranchingNode, JumpNode, MixNode,
                        VirtualPoseNode, VirtualDefineNode, VirtualWaitNode, VirtualBranchingNode, VirtualJumpNode, VirtualMixNode
                    ))
                    if is_valid_target:
                        branch_name = "To (red)" if condition_result else "Otherwise (blue)"
                        if swapped:
                            branch_name = "Otherwise (blue)" if condition_result else "To (red)"
                        _playback_log(f"[Playback] _get_next_node (Branch -> {branch_name}): {node.name()} -> {target_node.name()}")
                        self._set_ik_gate_for_target(target_node)
                        return target_node

            # Fallback: BranchingNode has no valid target from selected port
            _playback_log(f"[Playback] _get_next_node (Branch): {node.name()} has no valid target from selected port")
            return None

        # Check if this is a virtual node - use virtual port connections
        is_virtual = isinstance(node, (
            VirtualBaseLinkNode, VirtualPoseNode, VirtualDefineNode, VirtualWaitNode,
            VirtualBranchingNode, VirtualJumpNode, VirtualMixNode
        ))
        if is_virtual:
            out_ports = node.output_ports()
            if out_ports:
                connected = out_ports[0].connected_ports()
                if connected:
                    target_node = connected[0].node()
                    if target_node:
                        _playback_log(f"[Playback] _get_next_node (virtual): {node.name()} -> {target_node.name()}")
                        return target_node
            _playback_log(f"[Playback] _get_next_node (virtual): {node.name()} has no connections")
            return None

        # Default behavior: use _sorted_output_connections (only for real non-BranchingNode)
        connected_nodes = _sorted_output_connections(node)
        if connected_nodes:
            target = connected_nodes[0]
            if isinstance(node, PoseNode) and len(connected_nodes) > 0:
                _playback_log(f"[Playback] {node.name()} has connections to: {[n.name() for n in connected_nodes]}")
            _playback_log(f"[Playback] _get_next_node: {node.name()} -> {target.name()}")
            return target
        _playback_log(f"[Playback] _get_next_node: {node.name()} has no valid connections")
        return None

    def _resync_playback_tick_clock(self):
        """一時停止・分岐ダイアログ後など、elapsed との dt が跳ばないよう同期する。"""
        self._last_tick_time_ms = self.elapsed_timer.elapsed()

    def _ui_angles_from_robot_phys(self, phys_angles):
        """ロボット内部角（FKに渡す度数・Dir適用後）を Joint Sliders / Pose と同じ UI 座標へ変換する。"""
        if not phys_angles:
            return {}
        je = getattr(self.graph, "joint_editor", None) if self.graph else None
        if not je:
            return dict(phys_angles)
        return je.fk_to_ui_angles({jname: float(v) for jname, v in phys_angles.items()})

    def _max_speed_deg_per_sec(self, angle_key):
        """Joint max angular velocity in deg/s. Converts from internal rad/s storage.
        angle_key matches Pose angles_deg (usually URDF joint name; resolves display_name for legacy data).
        """
        je = getattr(self.graph, "joint_editor", None) if self.graph else None
        if not je or not getattr(je, "joint_settings", None):
            return max(1e-9, math.degrees(float(DEFAULT_JOINT_SPEED)))
        st = je.joint_settings.get(angle_key)
        if not st:
            ak = str(angle_key)
            for _jname, setting in je.joint_settings.items():
                disp = str(setting.get("display_name", "") or "")
                if disp == ak:
                    st = setting
                    break
            else:
                st = {}
        vmax_rads = float(st.get("max_speed_rad_s", DEFAULT_JOINT_SPEED))
        return max(1e-9, math.degrees(vmax_rads))

    def _apply_joint_speed_limits(self, ideal_angles, dt_sec):
        """ideal_angles・出力ともに UI 座標 [度]。dt_sec あたりに v_max [度/秒] を超えないよう追従する。"""
        out = {}
        for jname, ideal in ideal_angles.items():
            cur = self._speed_limited_angles.get(jname, ideal)
            v_max = self._max_speed_deg_per_sec(jname)
            delta = ideal - cur
            max_step = v_max * dt_sec
            if abs(delta) <= max_step:
                cur = ideal
            else:
                cur = cur + math.copysign(max_step, delta)
            self._speed_limited_angles[jname] = cur
            out[jname] = cur
        return out

    def _start_segment(self, idx):
        """セグメントの再生を開始"""
        if idx >= len(self.segments):
            self.stop()
            self.playback_finished.emit()
            return
        self.current_segment_idx = idx
        self.segment_start_time = self.elapsed_timer.elapsed()

        prev_node, next_node = self.segments[idx]

        # Easing interpolation uses node target values as fixed endpoints, not actual servo
        # position. The speed limiter (_speed_limited_angles) tracks actual position
        # separately and catches up from wherever the servo physically is.
        # Exception: BaseLinkNode (start marker) uses actual position as the origin.
        # BranchingNode/DefineNode/JumpNode etc. hold no pose — continue from actual position.
        if isinstance(prev_node, (PoseNode, VirtualPoseNode)):
            self.prev_angles = dict(prev_node.angles_deg)
        elif isinstance(prev_node, (BaseLinkNode, VirtualBaseLinkNode)):
            if self.robot_model:
                self.prev_angles = self._ui_angles_from_robot_phys(
                    dict(self.robot_model.get_current_angles()))
            else:
                self.prev_angles = dict(self._speed_limited_angles) if self._speed_limited_angles else {}
        elif isinstance(prev_node, (BranchingNode, VirtualBranchingNode,
                                    DefineNode, VirtualDefineNode,
                                    WaitNode, VirtualWaitNode,
                                    JumpNode, VirtualJumpNode,
                                    MixNode, VirtualMixNode,
                                    CommandNode, VirtualCommandNode)):
            self.prev_angles = dict(self._speed_limited_angles) if self._speed_limited_angles else {}
        else:
            self.prev_angles = {}

        # 次のノードの角度を取得
        if isinstance(next_node, (PoseNode, VirtualPoseNode)):
            self.next_angles = dict(next_node.angles_deg)
            self.next_easings = dict(getattr(next_node, 'joint_easings', {}))
        elif isinstance(next_node, (DefineNode, VirtualDefineNode)):
            self.next_angles = dict(self.prev_angles)
            self.next_easings = {}
        elif isinstance(next_node, (WaitNode, VirtualWaitNode)):
            self.next_angles = dict(self.prev_angles)
            self.next_easings = {}
        elif isinstance(next_node, (BranchingNode, VirtualBranchingNode)):
            self.next_angles = dict(self.prev_angles)
            self.next_easings = {}
        elif isinstance(next_node, (JumpNode, VirtualJumpNode)):
            self.next_angles = dict(self.prev_angles)
            self.next_easings = {}
        elif isinstance(next_node, (BaseLinkNode, VirtualBaseLinkNode)):
            # Start node from cross-action jump - instant transition
            self.next_angles = dict(self.prev_angles)
            self.next_easings = {}
        elif isinstance(next_node, (MixNode, VirtualMixNode)):
            # MixNode: maintains previous angles (mixing is applied at runtime)
            self.next_angles = dict(self.prev_angles)
            self.next_easings = {}
        elif isinstance(next_node, (CommandNode, VirtualCommandNode)):
            # CommandNode: maintains previous angles (command is sent at runtime)
            self.next_angles = dict(self.prev_angles)
            self.next_easings = {}
        else:
            self.next_angles = {}
            self.next_easings = {}

        # Segment duration from node.frames (frame count) and playback FPS.
        # Duration (seconds) = frames / fps
        # Tick rate: match playback FPS so Valkey writes align with animation frames.
        # When FPS > 60, tick faster than display (e.g. 100fps → 10ms tick = 100Hz writes).
        # When FPS ≤ 60, cap at 16ms so we don't waste CPU on redundant display updates.
        DISPLAY_REFRESH_MS = 16  # ~60 Hz display cap
        _fps = max(1, float(self.fps))
        _fps_tick_ms = max(1, int(round(1000.0 / _fps)))
        _tick_ms = min(DISPLAY_REFRESH_MS, _fps_tick_ms)
        # Computed-motion CALL nodes typically advance by 1/LOOP_HZ per tick.
        # Playback fps stays at LOOP_HZ so function jumps run at control rate,
        # not DISPLAY_REFRESH_MS (~60Hz).
        _ik_tick_ms = _fps_tick_ms
        if isinstance(next_node, (PoseNode, VirtualPoseNode)):
            frames = max(1, int(getattr(next_node, 'frames', 1)))
            fps = max(1, float(self.fps))
            dur_sec = frames / fps
            self.segment_duration_ms = max(1, int(round(dur_sec * 1000.0)))
            self._playback_tick_ms = _tick_ms
        elif isinstance(next_node, (DefineNode, VirtualDefineNode)):
            self.segment_duration_ms = 1  # instant transition
            self._playback_tick_ms = DISPLAY_REFRESH_MS
        elif isinstance(next_node, (WaitNode, VirtualWaitNode)):
            frames = int(getattr(next_node, 'frames', 0))
            if frames <= 0:
                self.segment_duration_ms = 1  # 0 frames → instant pass-through
            else:
                fps = max(1, float(self.fps))
                dur_sec = frames / fps
                self.segment_duration_ms = max(1, int(round(dur_sec * 1000.0)))
            self._playback_tick_ms = DISPLAY_REFRESH_MS
        elif isinstance(next_node, (BranchingNode, VirtualBranchingNode)):
            self.segment_duration_ms = 1  # instant transition
            # During computed walk the branch↔IK loop must keep LOOP_HZ cadence.
            if self._computed_motion_active:
                self._playback_tick_ms = _ik_tick_ms
            else:
                self._playback_tick_ms = DISPLAY_REFRESH_MS
        elif isinstance(next_node, (JumpNode, VirtualJumpNode)):
            _jump_type = getattr(next_node, 'jump_type', 'action')
            if _jump_type == 'function':
                self.segment_duration_ms = 1
                self._playback_tick_ms = _ik_tick_ms
            else:
                self.segment_duration_ms = 1
                self._playback_tick_ms = (
                    _ik_tick_ms if self._computed_motion_active else DISPLAY_REFRESH_MS
                )
        elif isinstance(next_node, (BaseLinkNode, VirtualBaseLinkNode)):
            self.segment_duration_ms = 1  # instant transition for cross-action start
            self._playback_tick_ms = DISPLAY_REFRESH_MS
        elif isinstance(next_node, (MixNode, VirtualMixNode)):
            # MixNode has frames for duration
            frames = max(1, int(getattr(next_node, 'frames', 1)))
            fps = max(1, float(self.fps))
            dur_sec = frames / fps
            self.segment_duration_ms = max(1, int(round(dur_sec * 1000.0)))
            self._playback_tick_ms = _tick_ms
        elif isinstance(next_node, (CommandNode, VirtualCommandNode)):
            # CommandNode has frames for duration
            frames = max(1, int(getattr(next_node, 'frames', 1)))
            fps = max(1, float(self.fps))
            dur_sec = frames / fps
            self.segment_duration_ms = max(1, int(round(dur_sec * 1000.0)))
            self._playback_tick_ms = _tick_ms
        else:
            self.segment_duration_ms = 1
            self._playback_tick_ms = DISPLAY_REFRESH_MS

        # Keep _speed_limited_angles at actual position — do NOT reset to prev_angles.
        # Only initialize entries for joints not yet tracked.
        for k in set(list(self.prev_angles.keys()) + list(self.next_angles.keys())):
            self._speed_limited_angles.setdefault(k, self.prev_angles.get(k, 0.0))
        self._last_tick_time_ms = self.elapsed_timer.elapsed()

        self.node_highlight.emit(next_node)
        if self.is_playing and not self.is_paused:
            self._restart_playback_timer()
        _playback_log(
            f"[Playback] Segment {idx}: {prev_node.name()} -> {next_node.name()}, "
            f"duration_ms={self.segment_duration_ms}, tick_ms={self._playback_tick_ms}"
        )

    def _tick(self):
        """タイマーティック（実時間ベース）"""
        current_time = self.elapsed_timer.elapsed()
        elapsed_in_segment = current_time - self.segment_start_time

        if self._last_tick_time_ms is None:
            self._last_tick_time_ms = current_time
        dt_ms = current_time - self._last_tick_time_ms
        dt_sec = max(1e-4, min(dt_ms / 1000.0, 0.5))
        self._last_tick_time_ms = current_time

        # 進捗率を計算（0.0 ~ 1.0）
        t = min(1.0, elapsed_in_segment / max(1, self.segment_duration_ms))

        _, current_next = self.segments[self.current_segment_idx]

        force_advance = False

        # Computed-motion state, needed below regardless of which branch runs it.
        _computed_active_now = (
            self._computed_motion_active and self._ik_executing
            and self.motion_runtime is not None and self._computed_func_names
        )

        # Computed motion IK runs on a background thread (_computed_ik_worker in
        # main()) at LOOP_HZ so it is not delayed by GUI-thread rendering.
        # _tick() only interpolates / emits pose_changed for pose-to-pose playback.
        if not _computed_active_now:
            # 角度を補間（理想値）
            interp_angles = {}
            all_joints = set(list(self.prev_angles.keys()) + list(self.next_angles.keys()))
            for jname in all_joints:
                a = self.prev_angles.get(jname, 0.0)
                b = self.next_angles.get(jname, 0.0)
                t_interp = easing_value(self.next_easings.get(jname, self.interpolation), t)
                interp_angles[jname] = a + (b - a) * t_interp

            # Always clamp to joint max_speed. pose_changed → Valkey → MuJoCoStudio.
            limited_angles = self._apply_joint_speed_limits(interp_angles, dt_sec)
            # WaitNode holds position; skip Valkey write to avoid redundant updates.
            if not isinstance(current_next, (WaitNode, VirtualWaitNode)):
                self.pose_changed.emit(limited_angles)
        else:
            # Angle output for this tick is owned by the background IK worker
            # instead (applies directly to the model + Valkey, and keeps
            # self._speed_limited_angles updated under stl_viewer._vtk_lock so the
            # handoff below, when this segment eventually ends, still sees an
            # up-to-date position). Only used here as a fallback for the
            # is_incomplete check just below, which doesn't apply to non-Pose
            # nodes like the Branch/Jump pair a walk loop cycles through anyway.
            limited_angles = self._speed_limited_angles

        if t >= 1.0 or force_advance:
            # Check if target angles were reached (threshold: 1 degree)
            _, current_next = self.segments[self.current_segment_idx]
            is_incomplete = False
            ANGLE_THRESHOLD = 1.0  # degrees
            for jname, target in self.next_angles.items():
                actual = limited_angles.get(jname, target)
                if abs(actual - target) > ANGLE_THRESHOLD:
                    is_incomplete = True
                    break
            if is_incomplete and isinstance(current_next, (PoseNode, VirtualPoseNode)):
                self.node_incomplete.emit(current_next)

            # 次のセグメントへ
            next_idx = self.current_segment_idx + 1
            if next_idx >= len(self.segments):
                # Single-node mode: stop after the one segment (long-press play)
                if self._single_node_mode:
                    self.stop()
                    return
                # 最後のセグメントの次のノードから更に辿れるか確認
                _, last_next = self.segments[self.current_segment_idx]

                # Action-only mode (▶︎.): stop once we've arrived at a JumpNode,
                # instead of resolving/following its jump target (which may cross
                # into another Action). Covers JumpNodes reached via dynamic
                # BranchingNode evaluation, in addition to the _build_path stop.
                if self._action_only_mode and isinstance(last_next, (JumpNode, VirtualJumpNode)):
                    _playback_log(f"[Playback] Stopping at JumpNode (action-only mode): {last_next.name()}")
                    self.stop()
                    return

                further = self._get_next_node(last_next)
                # Check for both real and virtual node types (including BaseLinkNode for cross-action jumps)
                is_valid_further = further and isinstance(
                    further, (
                        PoseNode, DefineNode, WaitNode, BranchingNode, JumpNode, BaseLinkNode, MixNode, CommandNode,
                        VirtualPoseNode, VirtualDefineNode, VirtualWaitNode, VirtualBranchingNode, VirtualJumpNode, VirtualBaseLinkNode, VirtualMixNode, VirtualCommandNode
                    )
                )
                if is_valid_further:
                    # Check for cycles - allow loops by resetting visited nodes
                    if id(further) in self._visited_nodes:
                        _playback_log(f"[Playback] Loop detected at: {further.name()}, continuing loop")
                        # Reset visited nodes to allow loop, but keep the target
                        self._visited_nodes.clear()
                        self._visited_nodes.add(id(further))
                    else:
                        self._visited_nodes.add(id(further))
                    self.segments.append((last_next, further))
                    self._start_segment(next_idx)
                else:
                    self.stop()
                    self.playback_finished.emit()
            else:
                self._start_segment(next_idx)

class LMEValkeyClient:
    """Write playback pose as Meridim90 to Valkey; read FB from Valkey."""

    def __init__(self):
        self._enabled = False
        self._host = VALKEY_DEFAULT_HOST
        self._port = VALKEY_DEFAULT_PORT
        self._write_key = VALKEY_DEFAULT_WRITE_KEY
        self._read_key  = VALKEY_DEFAULT_READ_KEY
        self._write_client = None
        self._read_client  = None
        self._fb_thread: threading.Thread | None = None
        self._fb_running = False
        self._fb_status = "---"
        self._fb_callback = None  # callable(status_str) to update UI
        self._last_mrd: list | None = None  # 前回送信パケット（部分更新マージ用）
        self._lme_seq: int = 0  # Valkey slot 88: recv デバッグ用シーケンス
        self._autostart_running = False  # Windows: Docker auto-start in progress

    # ── public API ──────────────────────────────────────────────────────────

    def set_fb_callback(self, cb):
        self._fb_callback = cb

    def update_config(self, cfg: dict) -> None:
        self._enabled   = bool(cfg.get("enabled", False))
        self._host      = cfg.get("host",      VALKEY_DEFAULT_HOST)
        self._port      = int(cfg.get("port",  VALKEY_DEFAULT_PORT))
        self._write_key = cfg.get("write_key", VALKEY_DEFAULT_WRITE_KEY)
        self._read_key  = cfg.get("read_key",  VALKEY_DEFAULT_READ_KEY)
        if not self._enabled:
            self._last_mrd = None
        self._reconnect()

    _write_debug_printed = False
    _write_call_count = 0

    def write_angles(self, angles_deg: dict) -> None:
        LMEValkeyClient._write_call_count += 1
        n = LMEValkeyClient._write_call_count
        if not self._enabled or not _VALKEY_OK:
            if n <= 3:
                print(f"[LMEValkey] write_angles called but skipped: enabled={self._enabled}, valkey_ok={_VALKEY_OK}")
            return
        if self._write_client is None:
            if n <= 3:
                print(f"[LMEValkey] write_angles: _write_client is None")
            return
        # 毎回ゼロ初期化すると、angles_deg に無い関節が 0° 指令になって
        # PhysicalOn 側でプルプル震え（ゼロリセット）の原因になる。
        # 前回パケットをベースに、今回更新された関節だけ上書きする。
        if self._last_mrd is None:
            self._last_mrd = [0.0] * MERIDIM_SIZE
        mrd = list(self._last_mrd)
        # Pre-set SERVO_ON for ALL mapped joints so stale TORQUE_OFF never
        # causes half-the-joints-dead twitching between frames.
        for _, (idx, _mul) in JOINT_TO_MERIDIM.items():
            if 0 < idx < MERIDIM_SIZE:
                mrd[idx - 1] = 1.0  # SERVO_ON = int16(1), Meridim ×100 wire format
        for jname, deg in angles_deg.items():
            entry = JOINT_TO_MERIDIM.get(jname)
            if entry is None:
                continue
            idx, mul = entry
            if 0 <= idx < MERIDIM_SIZE:
                # Meridim ×100 wire format: degrees × 100 (e.g. -32.2° → -3220.0)
                mrd[idx] = round(float(deg) / mul * 100.0, 2)
                if idx > 0:
                    mrd[idx - 1] = 1.0
        # Commander が同じ Valkey キーへ再書き込みするため、LME 発パケットを識別する。
        mrd[LME_PACKET_MARKER_SLOT] = LME_PACKET_MARKER_VALUE
        mrd[88] = float(self._lme_seq)
        self._lme_seq = (self._lme_seq + 1) % 1000000
        self._last_mrd = mrd
        if not LMEValkeyClient._write_debug_printed:
            matched = [k for k in angles_deg if k in JOINT_TO_MERIDIM]
            unmatched = [k for k in angles_deg if k not in JOINT_TO_MERIDIM]
            print(f"[LMEValkey] joint mapping: {len(matched)}/{len(angles_deg)} matched")
            if unmatched:
                print(f"[LMEValkey] unmatched joints: {unmatched}")
            LMEValkeyClient._write_debug_printed = True
        try:
            mapping = {str(i): str(v) for i, v in enumerate(mrd)}
            self._write_client.hset(self._write_key, mapping=mapping)
        except Exception as e:
            print(f"[LMEValkey] write error: {e}")

    def request_reset(self) -> None:
        """Pulse Meridim[0]=RESET so MuJoCoStudio respawns (same as R key)."""
        if not self._enabled or not _VALKEY_OK or self._write_client is None:
            return
        mrd = list(self._last_mrd) if self._last_mrd is not None else [0.0] * MERIDIM_SIZE
        mrd[0] = float(MASTER_CMD_RESET)
        try:
            mapping = {str(i): str(v) for i, v in enumerate(mrd)}
            self._write_client.hset(self._write_key, mapping=mapping)
        except Exception as e:
            print(f"[LMEValkey] reset write error: {e}")

    def read_fb(self) -> list | None:
        if not _VALKEY_OK or self._read_client is None:
            return None
        try:
            raw = self._read_client.hgetall(self._read_key)
            if not raw:
                return None
            return [float(raw.get(str(i), 0.0)) for i in range(MERIDIM_SIZE)]
        except Exception:
            return None

    def get_fb_status(self) -> str:
        return self._fb_status

    def start_fb_reader(self, interval_ms: int = 200) -> None:
        if self._fb_running:
            return
        self._fb_running = True
        self._fb_thread = threading.Thread(
            target=self._fb_loop, args=(interval_ms / 1000.0,), daemon=True
        )
        self._fb_thread.start()

    def stop_fb_reader(self) -> None:
        self._fb_running = False

    # ── internals ──────────────────────────────────────────────────────────

    def _reconnect(self) -> None:
        self._close_clients()
        if not self._enabled or not _VALKEY_OK:
            self._fb_status = "Disabled"
            self._notify_fb("Disabled")
            return
        try:
            self._write_client = _valkey_lib.Valkey(
                host=self._host, port=self._port,
                decode_responses=True,
                socket_connect_timeout=1.0, socket_timeout=1.0,
            )
            self._write_client.ping()
            self._read_client = _valkey_lib.Valkey(
                host=self._host, port=self._port,
                decode_responses=True,
                socket_connect_timeout=1.0, socket_timeout=1.0,
            )
            # Initialize write key with zeros if absent
            if not self._write_client.exists(self._write_key):
                self._write_client.hset(
                    self._write_key,
                    mapping={str(i): "0.0" for i in range(MERIDIM_SIZE)},
                )
            status = f"Connected  ({self._host}:{self._port})"
            self._fb_status = status
            self._notify_fb(status)
            print(f"[LMEValkey] {status}")
        except Exception as e:
            self._write_client = None
            self._read_client  = None
            status = f"Error: {e}"
            self._fb_status = status
            self._notify_fb(status)
            print(f"[LMEValkey] {status}")
            self._maybe_autostart_docker()

    def _maybe_autostart_docker(self) -> None:
        """Best-effort bring up Valkey in the background, then retry connecting.

        - Windows: launches Docker Desktop and starts the physicalon-valkey
          container.
        - macOS / Linux: spawns a native valkey-server (or redis-server)
          detached via subprocess so it outlives LME.
        """
        if self._autostart_running:
            return
        self._autostart_running = True
        host, port = self._host, self._port

        def _run():
            try:
                if ensure_valkey_running(host, port):
                    # Only reconnect if config hasn't changed/disabled meanwhile.
                    if self._enabled and self._host == host and self._port == port:
                        self._reconnect()
            finally:
                self._autostart_running = False

        threading.Thread(target=_run, daemon=True).start()

    def _close_clients(self) -> None:
        for c in (self._write_client, self._read_client):
            if c is not None:
                try:
                    c.close()
                except Exception:
                    pass
        self._write_client = None
        self._read_client  = None

    def _fb_loop(self, interval_sec: float) -> None:
        while self._fb_running:
            fb = self.read_fb()
            if fb is not None:
                non_zero = sum(1 for v in fb if v != 0.0)
                status = f"FB: {non_zero} joints active"
            else:
                status = "FB: no data"
            self._fb_status = status
            self._notify_fb(status)
            import time as _time
            _time.sleep(interval_sec)

    def _notify_fb(self, text: str) -> None:
        cb = self._fb_callback
        if cb is not None:
            try:
                cb(text)
            except Exception:
                pass


# Global Valkey client (initialized on first config apply)
lme_valkey = LMEValkeyClient()

try:
    from RobotLabelBridge import NameConverter as _RLBNameConverter
    _RLB_OK = True
except ImportError:
    _RLBNameConverter = None
    _RLB_OK = False


def build_joint_name_map(robot_model) -> dict:
    """Build {old_name: canonical_name} via RobotLabelBridge for a loaded model."""
    if not _RLB_OK or robot_model is None:
        return {}
    try:
        nc = _RLBNameConverter()
        joint_items = []
        for jname in robot_model.joint_order:
            jt = robot_model.joints.get(jname)
            item = {"name": jname}
            if jt is not None:
                item["axis"] = list(jt.axis) if jt.axis else None
                item["parent"] = jt.parent_link or None
                item["child"] = jt.child_link or None
                item["origin_rpy"] = list(jt.origin_rpy) if jt.origin_rpy else None
            joint_items.append(item)
        result = nc.convert_model(joints=joint_items)
        return {src: tgt for src, tgt in result.resolved_joint_map.items() if src != tgt}
    except Exception as e:
        print(f"[Normalize] build_joint_name_map error: {e}")
        return {}


def apply_joint_name_map(graph, robot_model, name_map: dict) -> int:
    """Rename joint names in all graph nodes and robot_model. Returns count of renames."""
    if not name_map:
        return 0
    renamed = 0
    for node in graph.all_nodes():
        if not hasattr(node, "angles_deg"):
            continue
        new_angles = {}
        for jname, angle in node.angles_deg.items():
            new_name = name_map.get(jname, jname)
            new_angles[new_name] = angle
            if new_name != jname:
                renamed += 1
        node.angles_deg = new_angles
    if robot_model is not None:
        robot_model.joint_order = [name_map.get(j, j) for j in robot_model.joint_order]
        old_joints = dict(robot_model.joints)
        robot_model.joints.clear()
        for jname, jobj in old_joints.items():
            new_name = name_map.get(jname, jname)
            jobj.name = new_name
            robot_model.joints[new_name] = jobj
        robot_model.current_angles = {
            name_map.get(k, k): v for k, v in robot_model.current_angles.items()
        }
        # child_map: {link_name: [joint_name, ...]} — update joint name values
        robot_model.child_map = {
            link: [name_map.get(j, j) for j in jlist]
            for link, jlist in robot_model.child_map.items()
        }
        # parent_map: {child_link: joint_name} — update joint name values
        robot_model.parent_map = {
            link: name_map.get(j, j)
            for link, j in robot_model.parent_map.items()
        }
    return renamed


def build_motion_data_dict(urdf_path, robot_model, graph, playback_ctrl, joint_editor=None,
                           include_user_value=False):
    """現在のグラフ・ジョイント・再生設定を辞書にまとめる（ファイル保存・アクション切替用）。"""
    pose_nodes = [n for n in graph.all_nodes() if isinstance(n, PoseNode)]
    define_nodes = [n for n in graph.all_nodes() if isinstance(n, DefineNode)]
    wait_nodes = [n for n in graph.all_nodes() if isinstance(n, WaitNode)]
    branch_nodes = [n for n in graph.all_nodes() if isinstance(n, BranchingNode)]
    mix_nodes = [n for n in graph.all_nodes() if isinstance(n, MixNode)]
    command_nodes = [n for n in graph.all_nodes() if isinstance(n, CommandNode)]
    jump_nodes = [n for n in graph.all_nodes() if isinstance(n, JumpNode)]
    memo_nodes = [n for n in graph.all_nodes() if isinstance(n, MemoNode)]
    node_id_map = {}
    for i, node in enumerate(pose_nodes):
        node_id_map[id(node)] = f"pose_{i}"
    for i, node in enumerate(define_nodes):
        node_id_map[id(node)] = f"define_{i}"
    for i, node in enumerate(wait_nodes):
        node_id_map[id(node)] = f"wait_{i}"
    for i, node in enumerate(branch_nodes):
        node_id_map[id(node)] = f"branch_{i}"
    for i, node in enumerate(mix_nodes):
        node_id_map[id(node)] = f"mix_{i}"
    for i, node in enumerate(command_nodes):
        node_id_map[id(node)] = f"command_{i}"
    for i, node in enumerate(jump_nodes):
        node_id_map[id(node)] = f"jump_{i}"
    for i, node in enumerate(memo_nodes):
        node_id_map[id(node)] = f"memo_{i}"

    nodes_data = []
    for node in pose_nodes:
        nid = node_id_map[id(node)]
        nodes_data.append({
            "id": nid,
            "node_type": "pose",
            "name": node.pose_name,
            "duration": node.duration,
            "frames": int(getattr(node, "frames", get_default_hz_fps())),
            "angles_deg": dict(node.angles_deg),
            "joint_easings": dict(getattr(node, 'joint_easings', {})),
            "pos_x": node.pos()[0] if isinstance(node.pos(), (list, tuple)) else node.pos().x() if hasattr(node.pos(), 'x') else 0,
            "pos_y": node.pos()[1] if isinstance(node.pos(), (list, tuple)) else node.pos().y() if hasattr(node.pos(), 'y') else 0,
            "branching_enabled": getattr(node, 'branching_enabled', False),
            "branch_outputs_swapped": getattr(node, 'branch_outputs_swapped', False),
            "branch_if_left": getattr(node, 'branch_if_left', "UserVal_0"),
            "branch_if_op": normalize_branch_if_op_stored(
                getattr(node, 'branch_if_op', "==")
            ),
            "branch_if_right": getattr(node, 'branch_if_right', "UserVal_1"),
            "branch_if_uv_enabled": getattr(node, 'branch_if_uv_enabled', True),
            "branch_if_formula_enabled": getattr(node, 'branch_if_formula_enabled', False),
            "branch_if_formula": getattr(node, 'branch_if_formula', "Form1:foo"),
            "branch_if_pad_enabled": getattr(node, 'branch_if_pad_enabled', False),
            "branch_if_pad_button": getattr(node, 'branch_if_pad_button', "L1"),
            "branch_if_pad_analog_enabled": getattr(node, 'branch_if_pad_analog_enabled', False),
            "branch_if_pad_analog_axis": getattr(node, 'branch_if_pad_analog_axis', "Lx"),
            "branch_if_pad_analog_op": getattr(node, 'branch_if_pad_analog_op', ">="),
            "branch_if_pad_analog_threshold": int(getattr(node, 'branch_if_pad_analog_threshold', 0)),
            "out_port_labels": list(node.out_port_labels),
            "out_port_priorities": list(node.out_port_priorities),
        })
    for node in define_nodes:
        nid = node_id_map[id(node)]
        nodes_data.append({
            "id": nid,
            "node_type": "define",
            "name": node.name(),
            "pos_x": node.pos()[0] if isinstance(node.pos(), (list, tuple)) else node.pos().x() if hasattr(node.pos(), 'x') else 0,
            "pos_y": node.pos()[1] if isinstance(node.pos(), (list, tuple)) else node.pos().y() if hasattr(node.pos(), 'y') else 0,
            "define_uv_index": getattr(node, "define_uv_index", 0),
            "define_memo": getattr(node, "define_memo", ""),
            "define_kind": getattr(node, "define_kind", "literal"),
            "define_literal": getattr(node, "define_literal", 0),
            "define_register_name": getattr(node, "define_register_name", ""),
        })
    for node in wait_nodes:
        nid = node_id_map[id(node)]
        nodes_data.append({
            "id": nid,
            "node_type": "wait",
            "name": getattr(node, "wait_name", node.name()),
            "frames": int(getattr(node, "frames", 0)),
            "duration": float(getattr(node, "duration", 0.0)),
            "pos_x": node.pos()[0] if isinstance(node.pos(), (list, tuple)) else node.pos().x() if hasattr(node.pos(), 'x') else 0,
            "pos_y": node.pos()[1] if isinstance(node.pos(), (list, tuple)) else node.pos().y() if hasattr(node.pos(), 'y') else 0,
            "out_port_labels": list(getattr(node, "out_port_labels", ["default"])),
            "out_port_priorities": list(getattr(node, "out_port_priorities", [0])),
        })
    for node in branch_nodes:
        nid = node_id_map[id(node)]
        nodes_data.append({
            "id": nid,
            "node_type": "branch",
            "name": node.name(),
            "pos_x": node.pos()[0] if isinstance(node.pos(), (list, tuple)) else node.pos().x() if hasattr(node.pos(), "x") else 0,
            "pos_y": node.pos()[1] if isinstance(node.pos(), (list, tuple)) else node.pos().y() if hasattr(node.pos(), "y") else 0,
            "branching_enabled": getattr(node, "branching_enabled", False),
            "branch_outputs_swapped": getattr(node, "branch_outputs_swapped", False),
            "branch_if_left": getattr(node, "branch_if_left", "UserVal_0"),
            "branch_if_op": normalize_branch_if_op_stored(
                getattr(node, "branch_if_op", "==")
            ),
            "branch_if_right": getattr(node, "branch_if_right", "UserVal_1"),
            "branch_if_uv_enabled": getattr(node, "branch_if_uv_enabled", True),
            "branch_if_formula_enabled": getattr(
                node, "branch_if_formula_enabled", False
            ),
            "branch_if_formula": getattr(node, "branch_if_formula", "Form1:foo"),
            "branch_if_pad_enabled": getattr(node, "branch_if_pad_enabled", False),
            "branch_if_pad_button": getattr(node, "branch_if_pad_button", "L1"),
            "branch_if_pad_analog_enabled": getattr(node, "branch_if_pad_analog_enabled", False),
            "branch_if_pad_analog_axis": getattr(node, "branch_if_pad_analog_axis", "Lx"),
            "branch_if_pad_analog_op": getattr(node, "branch_if_pad_analog_op", ">="),
            "branch_if_pad_analog_threshold": int(getattr(node, "branch_if_pad_analog_threshold", 0)),
            "out_port_labels": list(node.out_port_labels),
            "out_port_priorities": list(node.out_port_priorities),
        })
    for node in jump_nodes:
        nid = node_id_map[id(node)]
        nodes_data.append({
            "id": nid,
            "node_type": "jump",
            "name": node.name(),
            "jump_target_action_index": int(
                getattr(node, "jump_target_action_index", 0)
            ),
            "jump_type": getattr(node, "jump_type", "action"),
            "jump_target_function": getattr(node, "jump_target_function", ""),
            "pos_x": node.pos()[0] if isinstance(node.pos(), (list, tuple)) else node.pos().x() if hasattr(node.pos(), "x") else 0,
            "pos_y": node.pos()[1] if isinstance(node.pos(), (list, tuple)) else node.pos().y() if hasattr(node.pos(), "y") else 0,
            "out_port_labels": list(node.out_port_labels),
            "out_port_priorities": list(node.out_port_priorities),
        })
    for node in mix_nodes:
        nid = node_id_map[id(node)]
        nodes_data.append({
            "id": nid,
            "node_type": "mix",
            "name": getattr(node, "mix_name", "mix"),
            "frames": int(getattr(node, "frames", 1)),
            "mix_settings": dict(getattr(node, "mix_settings", {})),
            "pos_x": node.pos()[0] if isinstance(node.pos(), (list, tuple)) else node.pos().x() if hasattr(node.pos(), "x") else 0,
            "pos_y": node.pos()[1] if isinstance(node.pos(), (list, tuple)) else node.pos().y() if hasattr(node.pos(), "y") else 0,
            "out_port_labels": list(getattr(node, "out_port_labels", ["default"])),
            "out_port_priorities": list(getattr(node, "out_port_priorities", [0])),
        })
    for node in command_nodes:
        nid = node_id_map[id(node)]
        nodes_data.append({
            "id": nid,
            "node_type": "command",
            "name": getattr(node, "command_name", "command"),
            "frames": int(getattr(node, "frames", 1)),
            "command_settings": dict(getattr(node, "command_settings", {})),
            "pos_x": node.pos()[0] if isinstance(node.pos(), (list, tuple)) else node.pos().x() if hasattr(node.pos(), "x") else 0,
            "pos_y": node.pos()[1] if isinstance(node.pos(), (list, tuple)) else node.pos().y() if hasattr(node.pos(), "y") else 0,
            "out_port_labels": list(getattr(node, "out_port_labels", ["default"])),
            "out_port_priorities": list(getattr(node, "out_port_priorities", [0])),
        })
    for node in memo_nodes:
        nid = node_id_map[id(node)]
        nodes_data.append({
            "id": nid,
            "node_type": "memo",
            "name": node.name(),
            "memo_text": getattr(node, "memo_text", "") or "",
            "modal_width": int(getattr(node, "modal_width", 420) or 420),
            "modal_height": int(getattr(node, "modal_height", 320) or 320),
            "pos_x": node.pos()[0] if isinstance(node.pos(), (list, tuple)) else node.pos().x() if hasattr(node.pos(), "x") else 0,
            "pos_y": node.pos()[1] if isinstance(node.pos(), (list, tuple)) else node.pos().y() if hasattr(node.pos(), "y") else 0,
        })

    edges_data = []

    def _get_valid_connections_for_port(node, port):
        """Get valid connections for a port.
        For non-branching nodes, only the LAST connection per port is returned.
        For BranchingNode, all connections are returned.
        """
        valid_connections = []
        for connected_port in port.connected_ports():
            target = connected_port.node()
            if target is None:
                continue
            # Check if connected_port is an input port
            target_input_ports = target.input_ports()
            if connected_port not in target_input_ports:
                continue
            valid_connections.append(connected_port)

        # For non-branching nodes, only use the LAST connection (newest)
        branching_enabled = getattr(node, "branching_enabled", False)
        is_branching_type = isinstance(node, BranchingNode) or branching_enabled
        if not is_branching_type and len(valid_connections) > 1:
            print(f"[Save] {node.name()} has {len(valid_connections)} connections, using last one")
            valid_connections = [valid_connections[-1]]

        return valid_connections

    # Capture edges from PoseNode, DefineNode, WaitNode, BranchingNode, MixNode, CommandNode, JumpNode
    for node in graph.all_nodes():
        if not isinstance(node, (PoseNode, DefineNode, WaitNode, BranchingNode, MixNode, CommandNode, JumpNode)):
            continue
        src_id = node_id_map.get(id(node))
        if not src_id:
            continue
        for port_idx, port in enumerate(node.output_ports()):
            valid_connections = _get_valid_connections_for_port(node, port)
            for connected_port in valid_connections:
                target = connected_port.node()
                if isinstance(
                    target, (PoseNode, DefineNode, WaitNode, BranchingNode, MixNode, CommandNode, JumpNode)
                ) and id(target) in node_id_map:
                    label = node.out_port_labels[port_idx] if port_idx < len(node.out_port_labels) else ""
                    priority = node.out_port_priorities[port_idx] if port_idx < len(node.out_port_priorities) else 0
                    edges_data.append({
                        "from": src_id,
                        "to": node_id_map[id(target)],
                        "from_port": port_idx,
                        "label": label,
                        "priority": priority
                    })

    # Capture edges from BaseLinkNode (Start node) - only last connection
    for node in graph.all_nodes():
        if not isinstance(node, BaseLinkNode):
            continue
        for port_idx, port in enumerate(node.output_ports()):
            valid_connections = _get_valid_connections_for_port(node, port)
            for connected_port in valid_connections:
                target = connected_port.node()
                if isinstance(
                    target, (PoseNode, DefineNode, WaitNode, BranchingNode, MixNode, CommandNode, JumpNode)
                ) and id(target) in node_id_map:
                    edges_data.append({
                        "from": "start",
                        "to": node_id_map[id(target)],
                        "label": "",
                        "priority": 0
                    })

    selected = graph.selected_nodes()
    start_id = ""
    for s in selected:
        if isinstance(s, PoseNode) and id(s) in node_id_map:
            start_id = node_id_map[id(s)]
            break
    if not start_id and pose_nodes:
        start_id = node_id_map[id(pose_nodes[0])]

    data = {
        "version": 1,
        "urdf_path": urdf_path,
        "joint_order": list(robot_model.joint_order) if robot_model else [],
        "joint_settings": joint_editor.get_joint_settings() if joint_editor else {},
        "joint_layout": joint_editor.get_joint_layout() if joint_editor else {},
        "joint_group_presets": joint_editor.get_joint_group_presets() if joint_editor else [],
        "current_group_preset_index": joint_editor.current_group_preset_index if joint_editor else -1,
        "motion_formulas": dict(getattr(graph, "motion_formulas", {})),
        "nodes": nodes_data,
        "edges": edges_data,
        "playback": {
            "start_node_id": start_id,
            "interpolation": playback_ctrl.interpolation if playback_ctrl else EASING_OPTIONS[0],
            "fps": playback_ctrl.fps if playback_ctrl else 100,
            "branch_mode": playback_ctrl.branch_mode if playback_ctrl else "default-first"
        }
    }
    if include_user_value:
        data["user_value_session"] = normalize_user_value_session(
            getattr(graph, "user_value_session", None)
        )
    return data


def _write_action_data_to_xml(action_elem, data):
    """Helper: Write single action data to XML element"""
    # Motion Formulas
    formulas_elem = ET.SubElement(action_elem, "MotionFormulas")
    motion_formulas_data = data.get("motion_formulas", {})
    if isinstance(motion_formulas_data, dict):
        for key, value in motion_formulas_data.items():
            f_elem = ET.SubElement(formulas_elem, "Formula")
            f_elem.set("key", key)
            f_elem.text = str(value)

    # Nodes
    nodes_elem = ET.SubElement(action_elem, "Nodes")
    for node_data in data.get("nodes", []):
        node_elem = ET.SubElement(nodes_elem, "Node")
        node_elem.set("id", node_data.get("id", ""))
        node_elem.set("type", node_data.get("node_type", ""))
        node_elem.set("name", node_data.get("name", ""))
        node_elem.set("pos_x", str(node_data.get("pos_x", 0)))
        node_elem.set("pos_y", str(node_data.get("pos_y", 0)))

        if node_data.get("node_type") == "pose":
            node_elem.set("duration", str(node_data.get("duration", 1.0)))
            node_elem.set("frames", str(node_data.get("frames", 1)))
            node_elem.set("branching_enabled", str(node_data.get("branching_enabled", False)))
            node_elem.set("branch_outputs_swapped", str(node_data.get("branch_outputs_swapped", False)))
            node_elem.set("branch_if_left", node_data.get("branch_if_left", "UserVal_0"))
            node_elem.set("branch_if_op", node_data.get("branch_if_op", "=="))
            node_elem.set("branch_if_right", node_data.get("branch_if_right", "UserVal_1"))
            node_elem.set("branch_if_uv_enabled", str(node_data.get("branch_if_uv_enabled", True)))
            node_elem.set("branch_if_formula_enabled", str(node_data.get("branch_if_formula_enabled", False)))
            node_elem.set("branch_if_formula", node_data.get("branch_if_formula", "Form1:foo"))
            node_elem.set("branch_if_pad_enabled", str(node_data.get("branch_if_pad_enabled", False)))
            node_elem.set("branch_if_pad_button", node_data.get("branch_if_pad_button", "L1"))
            node_elem.set("branch_if_pad_analog_enabled", str(node_data.get("branch_if_pad_analog_enabled", False)))
            node_elem.set("branch_if_pad_analog_axis", node_data.get("branch_if_pad_analog_axis", "Lx"))
            node_elem.set("branch_if_pad_analog_op", node_data.get("branch_if_pad_analog_op", ">="))
            node_elem.set("branch_if_pad_analog_threshold", str(int(node_data.get("branch_if_pad_analog_threshold", 0))))

            # Angles
            angles_elem = ET.SubElement(node_elem, "Angles")
            angles_data = node_data.get("angles_deg", {})
            if isinstance(angles_data, dict):
                for jname, angle in angles_data.items():
                    a_elem = ET.SubElement(angles_elem, "Angle")
                    a_elem.set("joint", jname)
                    a_elem.set("value", str(angle))

            # Easings
            easings_elem = ET.SubElement(node_elem, "Easings")
            easings_data = node_data.get("joint_easings", {})
            if isinstance(easings_data, dict):
                for jname, easing in easings_data.items():
                    e_elem = ET.SubElement(easings_elem, "Easing")
                    e_elem.set("joint", jname)
                    e_elem.set("value", easing)

            # Output ports
            ports_elem = ET.SubElement(node_elem, "OutputPorts")
            labels = node_data.get("out_port_labels", [])
            priorities = node_data.get("out_port_priorities", [])
            for i, label in enumerate(labels):
                p_elem = ET.SubElement(ports_elem, "Port")
                p_elem.set("label", label)
                p_elem.set("priority", str(priorities[i] if i < len(priorities) else 0))

        elif node_data.get("node_type") == "define":
            node_elem.set("define_uv_index", str(node_data.get("define_uv_index", 0)))
            node_elem.set("define_memo", node_data.get("define_memo", ""))
            node_elem.set("define_kind", node_data.get("define_kind", "literal"))
            node_elem.set("define_literal", str(node_data.get("define_literal", 0)))
            node_elem.set("define_register_name", node_data.get("define_register_name", ""))

        elif node_data.get("node_type") == "wait":
            node_elem.set("frames", str(node_data.get("frames", 0)))
            node_elem.set("duration", str(node_data.get("duration", 0.0)))
            ports_elem = ET.SubElement(node_elem, "OutputPorts")
            labels = node_data.get("out_port_labels", ["default"])
            priorities = node_data.get("out_port_priorities", [0])
            for i, label in enumerate(labels):
                p_elem = ET.SubElement(ports_elem, "Port")
                p_elem.set("label", label)
                p_elem.set("priority", str(priorities[i] if i < len(priorities) else 0))

        elif node_data.get("node_type") == "branch":
            node_elem.set("branching_enabled", str(node_data.get("branching_enabled", False)))
            node_elem.set("branch_outputs_swapped", str(node_data.get("branch_outputs_swapped", False)))
            node_elem.set("branch_if_left", node_data.get("branch_if_left", "UserVal_0"))
            node_elem.set("branch_if_op", node_data.get("branch_if_op", "=="))
            node_elem.set("branch_if_right", node_data.get("branch_if_right", "UserVal_1"))
            node_elem.set("branch_if_uv_enabled", str(node_data.get("branch_if_uv_enabled", True)))
            node_elem.set("branch_if_formula_enabled", str(node_data.get("branch_if_formula_enabled", False)))
            node_elem.set("branch_if_formula", node_data.get("branch_if_formula", "Form1:foo"))
            node_elem.set("branch_if_pad_enabled", str(node_data.get("branch_if_pad_enabled", False)))
            node_elem.set("branch_if_pad_button", node_data.get("branch_if_pad_button", "L1"))
            node_elem.set("branch_if_pad_analog_enabled", str(node_data.get("branch_if_pad_analog_enabled", False)))
            node_elem.set("branch_if_pad_analog_axis", node_data.get("branch_if_pad_analog_axis", "Lx"))
            node_elem.set("branch_if_pad_analog_op", node_data.get("branch_if_pad_analog_op", ">="))
            node_elem.set("branch_if_pad_analog_threshold", str(int(node_data.get("branch_if_pad_analog_threshold", 0))))
            ports_elem = ET.SubElement(node_elem, "OutputPorts")
            labels = node_data.get("out_port_labels", [])
            priorities = node_data.get("out_port_priorities", [])
            for i, label in enumerate(labels):
                p_elem = ET.SubElement(ports_elem, "Port")
                p_elem.set("label", label)
                p_elem.set("priority", str(priorities[i] if i < len(priorities) else 0))

        elif node_data.get("node_type") == "jump":
            node_elem.set("jump_target_action_index", str(node_data.get("jump_target_action_index", 0)))
            node_elem.set("jump_type", node_data.get("jump_type", "action"))
            node_elem.set("jump_target_function", node_data.get("jump_target_function", ""))
            ports_elem = ET.SubElement(node_elem, "OutputPorts")
            labels = node_data.get("out_port_labels", [])
            priorities = node_data.get("out_port_priorities", [])
            for i, label in enumerate(labels):
                p_elem = ET.SubElement(ports_elem, "Port")
                p_elem.set("label", label)
                p_elem.set("priority", str(priorities[i] if i < len(priorities) else 0))

        elif node_data.get("node_type") == "mix":
            node_elem.set("frames", str(node_data.get("frames", 1)))
            # Save mix_settings as JSON string
            mix_settings = node_data.get("mix_settings", {})
            node_elem.set("mix_settings_json", json.dumps(mix_settings))
            ports_elem = ET.SubElement(node_elem, "OutputPorts")
            labels = node_data.get("out_port_labels", ["default"])
            priorities = node_data.get("out_port_priorities", [0])
            for i, label in enumerate(labels):
                p_elem = ET.SubElement(ports_elem, "Port")
                p_elem.set("label", label)
                p_elem.set("priority", str(priorities[i] if i < len(priorities) else 0))

        elif node_data.get("node_type") == "memo":
            # 本文は特殊文字を含む可能性があるため CDATA として子要素に保持
            node_elem.set("modal_width", str(int(node_data.get("modal_width", 420) or 420)))
            node_elem.set("modal_height", str(int(node_data.get("modal_height", 320) or 320)))
            memo_text = node_data.get("memo_text", "") or ""
            text_elem = ET.SubElement(node_elem, "MemoText")
            text_elem.text = memo_text

    # Edges (connections)
    edges_elem = ET.SubElement(action_elem, "Edges")
    for edge_data in data.get("edges", []):
        if isinstance(edge_data, dict):
            edge_elem = ET.SubElement(edges_elem, "Edge")
            edge_elem.set("from", edge_data.get("from", ""))
            edge_elem.set("to", edge_data.get("to", ""))
            edge_elem.set("from_port", str(edge_data.get("from_port", 0)))
            edge_elem.set("label", edge_data.get("label", ""))
            edge_elem.set("priority", str(edge_data.get("priority", 0)))

    # Playback settings
    playback_data = data.get("playback", {})
    if isinstance(playback_data, dict):
        playback_elem = ET.SubElement(action_elem, "Playback")
        playback_elem.set("start_node_id", playback_data.get("start_node_id", ""))
        playback_elem.set("interpolation", playback_data.get("interpolation", "linear"))
        playback_elem.set("fps", str(playback_data.get("fps", 100)))
        playback_elem.set("branch_mode", playback_data.get("branch_mode", "default-first"))


def save_project_xml(filepath, urdf_path, robot_model, graph, playback_ctrl, joint_editor=None,
                     motion_action_state=None, capture_current_func=None, model_type='',
                     view_settings=None, home_position=None):
    """プロジェクトをXML形式で保存（全Actionを含む）"""
    try:
        # Capture current action data first
        if capture_current_func and motion_action_state:
            current_idx = motion_action_state.get("current", 0)
            items = motion_action_state.get("items", [])
            if 0 <= current_idx < len(items):
                items[current_idx]["data"] = capture_current_func()

        # Get current data for global settings
        current_data = build_motion_data_dict(
            urdf_path, robot_model, graph, playback_ctrl, joint_editor,
            include_user_value=True)

        root = ET.Element("StopMotionProject")
        root.set("version", "2")  # Version 2 for multi-action support

        # URDF Path (relative to LME package / save file; never store machine-absolute)
        urdf_elem = ET.SubElement(root, "URDFPath")
        urdf_elem.text = path_for_project_save(current_data.get("urdf_path", ""), filepath)

        # Model Type (urdf or mjcf)
        model_type_elem = ET.SubElement(root, "ModelType")
        model_type_elem.text = model_type if model_type else "urdf"

        # Robot Name
        robot_name_elem = ET.SubElement(root, "RobotName")
        robot_name_elem.text = getattr(graph, "robot_name", "") or ""

        # Joint Order
        joint_order_elem = ET.SubElement(root, "JointOrder")
        for jname in current_data.get("joint_order", []):
            j_elem = ET.SubElement(joint_order_elem, "Joint")
            j_elem.set("name", jname)

        # Joint Settings (global)
        joint_settings_elem = ET.SubElement(root, "JointSettings")
        joint_settings_data = current_data.get("joint_settings", {})
        if isinstance(joint_settings_data, dict):
            for jname, settings in joint_settings_data.items():
                js_elem = ET.SubElement(joint_settings_elem, "Joint")
                js_elem.set("name", jname)
                if isinstance(settings, dict):
                    for key, value in settings.items():
                        js_elem.set(key, str(value))

        # Joint Speed Presets (servo model presets)
        from LegacyMotionEditor_Utils import get_joint_speed_presets
        speed_presets_elem = ET.SubElement(root, "JointSpeedPresets")
        for preset_name, speed_rads in get_joint_speed_presets():
            preset_elem = ET.SubElement(speed_presets_elem, "Preset")
            preset_elem.set("name", preset_name)
            preset_elem.set("speed_rad_s", str(speed_rads))

        # Joint Layout (global) - contains "order" and "groups"
        joint_layout_elem = ET.SubElement(root, "JointLayout")
        joint_layout_data = current_data.get("joint_layout", {})
        if isinstance(joint_layout_data, dict):
            # Save order as comma-separated string
            order_list = joint_layout_data.get("order", [])
            if isinstance(order_list, list) and order_list:
                order_elem = ET.SubElement(joint_layout_elem, "Order")
                order_elem.text = ",".join(order_list)
            # Save groups (joint_name -> group_key like "L", "R", "C")
            groups_dict = joint_layout_data.get("groups", {})
            if isinstance(groups_dict, dict):
                for jname, group_key in groups_dict.items():
                    jl_elem = ET.SubElement(joint_layout_elem, "Joint")
                    jl_elem.set("name", jname)
                    jl_elem.set("group", str(group_key))

        # Joint Group Presets (global)
        presets_elem = ET.SubElement(root, "JointGroupPresets")
        presets_elem.set("currentIndex", str(current_data.get("current_group_preset_index", -1)))
        joint_group_presets = current_data.get("joint_group_presets", [])
        if isinstance(joint_group_presets, list):
            for preset in joint_group_presets:
                if isinstance(preset, dict):
                    preset_elem = ET.SubElement(presets_elem, "Preset")
                    preset_elem.set("name", preset.get("name", ""))
                    members = preset.get("members", {})
                    if isinstance(members, dict):
                        for jname, member in members.items():
                            if isinstance(member, dict):
                                m_elem = ET.SubElement(preset_elem, "Member")
                                m_elem.set("name", str(jname))
                                m_elem.set("enabled", "1" if member.get("enabled", False) else "0")
                                m_elem.set("scale", str(float(member.get("scale", 1.0))))

        # View Settings (3D view background, light, etc.)
        if view_settings and isinstance(view_settings, dict):
            view_elem = ET.SubElement(root, "ViewSettings")
            # Background colors (RGB float lists)
            bg_color_a = view_settings.get("bg_color_a", [])
            bg_color_b = view_settings.get("bg_color_b", [])
            if bg_color_a:
                view_elem.set("bg_color_a", ",".join(str(c) for c in bg_color_a))
            if bg_color_b:
                view_elem.set("bg_color_b", ",".join(str(c) for c in bg_color_b))
            # Gradient type
            view_elem.set("bg_gradient_type", str(view_settings.get("bg_gradient_type", "vertical")))
            # Slider values
            view_elem.set("bg_slider_value", str(view_settings.get("bg_slider_value", 50)))
            view_elem.set("light_slider_value", str(view_settings.get("light_slider_value", 70)))

        # Home Position (joint angles for Home button)
        if home_position and isinstance(home_position, dict):
            home_elem = ET.SubElement(root, "HomePosition")
            for jname, angle in home_position.items():
                hj_elem = ET.SubElement(home_elem, "Joint")
                hj_elem.set("name", str(jname))
                hj_elem.set("angle", str(angle))

        # Camera Presets (A-E: name, azimuth, elevation, distance, focal_x/y/z)
        cam_presets_data = view_settings.get("camera_presets", {}) if view_settings else {}
        if cam_presets_data:
            cp_elem = ET.SubElement(root, "CameraPresets")
            for cam_name, preset in cam_presets_data.items():
                ce = ET.SubElement(cp_elem, "Camera")
                ce.set("name",      str(cam_name))
                ce.set("label",     str(preset.get("name",      "")))
                ce.set("azimuth",   str(preset.get("azimuth",   0)))
                ce.set("elevation", str(preset.get("elevation", 0)))
                ce.set("distance",  str(preset.get("distance",  1.0)))
                ce.set("focal_x",   str(preset.get("focal_x",  0.0)))
                ce.set("focal_y",   str(preset.get("focal_y",  0.0)))
                ce.set("focal_z",   str(preset.get("focal_z",  0.0)))

        # Project-wide Python code (user logic + walk params in ProjectCode)
        project_code = getattr(graph, "project_code", "") or ""
        graph.project_code = project_code
        if project_code:
            code_elem = ET.SubElement(root, "ProjectCode")
            code_elem.text = project_code

        # UserValueSession (root-level, shared across all actions)
        uv_session = getattr(graph, "user_value_session", None) or []
        if isinstance(uv_session, list):
            uv_root_elem = ET.SubElement(root, "UserValueSession")
            for i, slot in enumerate(uv_session):
                if not isinstance(slot, dict):
                    continue
                kind = slot.get("kind", "literal")
                if kind == "register":
                    name_val = str(slot.get("name", "")).strip()
                    if not name_val:
                        continue
                    s_elem = ET.SubElement(uv_root_elem, "Slot")
                    s_elem.set("index", str(i))
                    s_elem.set("kind", "register")
                    s_elem.set("name", name_val)
                else:
                    val = slot.get("value", 0)
                    if val == 0:
                        continue
                    s_elem = ET.SubElement(uv_root_elem, "Slot")
                    s_elem.set("index", str(i))
                    s_elem.set("kind", "literal")
                    s_elem.set("value", str(int(val)))

        # Actions (multiple)
        actions_elem = ET.SubElement(root, "Actions")
        if motion_action_state:
            current_idx = motion_action_state.get("current", 0)
            actions_elem.set("currentIndex", str(current_idx))
            items = motion_action_state.get("items", [])
            for i, item in enumerate(items):
                action_elem = ET.SubElement(actions_elem, "Action")
                action_elem.set("index", str(i))
                action_elem.set("title", item.get("title", ""))
                action_data = item.get("data")
                if action_data and isinstance(action_data, dict):
                    _write_action_data_to_xml(action_elem, action_data)
                elif i == current_idx:
                    # Current action - use current_data
                    _write_action_data_to_xml(action_elem, current_data)
                else:
                    # Non-current action with no data: write empty-but-valid nodes/edges
                    # so it loads cleanly (avoids missing <Nodes> element on reload)
                    _write_action_data_to_xml(action_elem, {"nodes": [], "edges": []})
        else:
            # No motion_action_state - save current as single action
            actions_elem.set("currentIndex", "0")
            action_elem = ET.SubElement(actions_elem, "Action")
            action_elem.set("index", "0")
            action_elem.set("title", "")
            _write_action_data_to_xml(action_elem, current_data)

        # Write to file with pretty formatting
        tree = ET.ElementTree(root)
        ET.indent(tree, space="  ")
        tree.write(filepath, encoding="utf-8", xml_declaration=True)

        print(f"[ProjectXML] Saved {len(motion_action_state.get('items', []) if motion_action_state else 1)} actions to {filepath}")
        return True
    except Exception as e:
        print(f"[ProjectXML] Error saving: {e}")
        traceback.print_exc()
        return False


def _parse_action_data_from_xml(action_elem, global_data):
    """Helper: Parse single action data from XML element"""
    data = copy.deepcopy(global_data)  # Start with global settings

    # Motion Formulas (action-specific)
    motion_formulas = {}
    formulas_elem = action_elem.find("MotionFormulas")
    if formulas_elem is not None:
        for f_elem in formulas_elem.findall("Formula"):
            key = f_elem.get("key", "")
            motion_formulas[key] = f_elem.text or ""
    data["motion_formulas"] = motion_formulas

    # Nodes
    nodes_data = []
    nodes_elem = action_elem.find("Nodes")
    if nodes_elem is not None:
        for node_elem in nodes_elem.findall("Node"):
            node_type = node_elem.get("type", "")
            node_data = {
                "id": node_elem.get("id", ""),
                "node_type": node_type,
                "name": node_elem.get("name", ""),
                "pos_x": float(node_elem.get("pos_x", 0)),
                "pos_y": float(node_elem.get("pos_y", 0)),
            }

            if node_type == "pose":
                node_data["duration"] = float(node_elem.get("duration", 1.0))
                node_data["frames"] = int(node_elem.get("frames", 1))
                node_data["branching_enabled"] = node_elem.get("branching_enabled", "False").lower() == "true"
                node_data["branch_outputs_swapped"] = node_elem.get("branch_outputs_swapped", "False").lower() == "true"
                node_data["branch_if_left"] = node_elem.get("branch_if_left", "UserVal_0")
                node_data["branch_if_op"] = node_elem.get("branch_if_op", "==")
                node_data["branch_if_right"] = node_elem.get("branch_if_right", "UserVal_1")
                node_data["branch_if_uv_enabled"] = node_elem.get("branch_if_uv_enabled", "True").lower() == "true"
                node_data["branch_if_formula_enabled"] = node_elem.get("branch_if_formula_enabled", "False").lower() == "true"
                node_data["branch_if_formula"] = node_elem.get("branch_if_formula", "Form1:foo")
                node_data["branch_if_pad_enabled"] = node_elem.get("branch_if_pad_enabled", "False").lower() == "true"
                node_data["branch_if_pad_button"] = node_elem.get("branch_if_pad_button", "L1")
                node_data["branch_if_pad_analog_enabled"] = node_elem.get("branch_if_pad_analog_enabled", "False").lower() == "true"
                node_data["branch_if_pad_analog_axis"] = node_elem.get("branch_if_pad_analog_axis", "Lx")
                node_data["branch_if_pad_analog_op"] = node_elem.get("branch_if_pad_analog_op", ">=")
                node_data["branch_if_pad_analog_threshold"] = int(node_elem.get("branch_if_pad_analog_threshold", 0))

                angles = {}
                angles_elem = node_elem.find("Angles")
                if angles_elem is not None:
                    for a_elem in angles_elem.findall("Angle"):
                        angles[a_elem.get("joint", "")] = float(a_elem.get("value", 0))
                node_data["angles_deg"] = angles

                easings = {}
                easings_elem = node_elem.find("Easings")
                if easings_elem is not None:
                    for e_elem in easings_elem.findall("Easing"):
                        easings[e_elem.get("joint", "")] = e_elem.get("value", "linear")
                node_data["joint_easings"] = easings

                labels, priorities = [], []
                ports_elem = node_elem.find("OutputPorts")
                if ports_elem is not None:
                    for p_elem in ports_elem.findall("Port"):
                        labels.append(p_elem.get("label", ""))
                        priorities.append(int(p_elem.get("priority", 0)))
                node_data["out_port_labels"] = labels if labels else ["default"]
                node_data["out_port_priorities"] = priorities if priorities else [0]

            elif node_type == "define":
                node_data["define_uv_index"] = int(node_elem.get("define_uv_index", 0))
                node_data["define_memo"] = node_elem.get("define_memo", "")
                node_data["define_kind"] = node_elem.get("define_kind", "literal")
                node_data["define_literal"] = int(node_elem.get("define_literal", 0))
                node_data["define_register_name"] = node_elem.get("define_register_name", "")

            elif node_type == "wait":
                node_data["frames"] = int(node_elem.get("frames", 0))
                node_data["duration"] = float(node_elem.get("duration", 0.0))
                labels, priorities = [], []
                ports_elem = node_elem.find("OutputPorts")
                if ports_elem is not None:
                    for p_elem in ports_elem.findall("Port"):
                        labels.append(p_elem.get("label", ""))
                        priorities.append(int(p_elem.get("priority", 0)))
                node_data["out_port_labels"] = labels if labels else ["default"]
                node_data["out_port_priorities"] = priorities if priorities else [0]

            elif node_type == "branch":
                node_data["branching_enabled"] = node_elem.get("branching_enabled", "False").lower() == "true"
                node_data["branch_outputs_swapped"] = node_elem.get("branch_outputs_swapped", "False").lower() == "true"
                node_data["branch_if_left"] = node_elem.get("branch_if_left", "UserVal_0")
                node_data["branch_if_op"] = node_elem.get("branch_if_op", "==")
                node_data["branch_if_right"] = node_elem.get("branch_if_right", "UserVal_1")
                node_data["branch_if_uv_enabled"] = node_elem.get("branch_if_uv_enabled", "True").lower() == "true"
                node_data["branch_if_formula_enabled"] = node_elem.get("branch_if_formula_enabled", "False").lower() == "true"
                node_data["branch_if_formula"] = node_elem.get("branch_if_formula", "Form1:foo")
                node_data["branch_if_pad_enabled"] = node_elem.get("branch_if_pad_enabled", "False").lower() == "true"
                node_data["branch_if_pad_button"] = node_elem.get("branch_if_pad_button", "L1")
                node_data["branch_if_pad_analog_enabled"] = node_elem.get("branch_if_pad_analog_enabled", "False").lower() == "true"
                node_data["branch_if_pad_analog_axis"] = node_elem.get("branch_if_pad_analog_axis", "Lx")
                node_data["branch_if_pad_analog_op"] = node_elem.get("branch_if_pad_analog_op", ">=")
                node_data["branch_if_pad_analog_threshold"] = int(node_elem.get("branch_if_pad_analog_threshold", 0))
                labels, priorities = [], []
                ports_elem = node_elem.find("OutputPorts")
                if ports_elem is not None:
                    for p_elem in ports_elem.findall("Port"):
                        labels.append(p_elem.get("label", ""))
                        priorities.append(int(p_elem.get("priority", 0)))
                node_data["out_port_labels"] = labels if labels else ["default"]
                node_data["out_port_priorities"] = priorities if priorities else [0]

            elif node_type == "jump":
                node_data["jump_target_action_index"] = int(node_elem.get("jump_target_action_index", 0))
                node_data["jump_type"] = node_elem.get("jump_type", "action")
                node_data["jump_target_function"] = node_elem.get("jump_target_function", "")
                labels, priorities = [], []
                ports_elem = node_elem.find("OutputPorts")
                if ports_elem is not None:
                    for p_elem in ports_elem.findall("Port"):
                        labels.append(p_elem.get("label", ""))
                        priorities.append(int(p_elem.get("priority", 0)))
                node_data["out_port_labels"] = labels if labels else ["default"]
                node_data["out_port_priorities"] = priorities if priorities else [0]

            elif node_type == "mix":
                node_data["frames"] = int(node_elem.get("frames", 1))
                # Load mix_settings from JSON string
                mix_settings_json = node_elem.get("mix_settings_json", "{}")
                try:
                    node_data["mix_settings"] = json.loads(mix_settings_json)
                except (json.JSONDecodeError, TypeError):
                    node_data["mix_settings"] = {}
                labels, priorities = [], []
                ports_elem = node_elem.find("OutputPorts")
                if ports_elem is not None:
                    for p_elem in ports_elem.findall("Port"):
                        labels.append(p_elem.get("label", ""))
                        priorities.append(int(p_elem.get("priority", 0)))
                node_data["out_port_labels"] = labels if labels else ["default"]
                node_data["out_port_priorities"] = priorities if priorities else [0]

            elif node_type == "memo":
                try:
                    node_data["modal_width"] = int(node_elem.get("modal_width", 420))
                except (TypeError, ValueError):
                    node_data["modal_width"] = 420
                try:
                    node_data["modal_height"] = int(node_elem.get("modal_height", 320))
                except (TypeError, ValueError):
                    node_data["modal_height"] = 320
                text_elem = node_elem.find("MemoText")
                node_data["memo_text"] = (text_elem.text or "") if text_elem is not None else ""

            nodes_data.append(node_data)
    data["nodes"] = nodes_data

    # Edges
    edges_data = []
    edges_elem = action_elem.find("Edges")
    if edges_elem is not None:
        for edge_elem in edges_elem.findall("Edge"):
            fp_str = edge_elem.get("from_port", "")
            edge_entry = {
                "from": edge_elem.get("from", ""),
                "to": edge_elem.get("to", ""),
                "label": edge_elem.get("label", ""),
                "priority": int(edge_elem.get("priority", 0)),
            }
            if fp_str != "":
                try:
                    edge_entry["from_port"] = int(fp_str)
                except ValueError:
                    pass
            edges_data.append(edge_entry)
    data["edges"] = edges_data

    # Playback
    playback_elem = action_elem.find("Playback")
    if playback_elem is not None:
        data["playback"] = {
            "start_node_id": playback_elem.get("start_node_id", ""),
            "interpolation": playback_elem.get("interpolation", "linear"),
            "fps": int(playback_elem.get("fps", 100)),
            "branch_mode": playback_elem.get("branch_mode", "default-first"),
        }
    else:
        data["playback"] = {}

    return data


def load_project_xml(filepath, graph, stl_viewer, joint_editor, playback_ctrl,
                     motion_state=None, parent_window=None, motion_action_state=None):
    """XMLファイルからプロジェクトを読み込み（複数Action対応）"""
    try:
        # Remove existing robot model before loading new project
        if motion_state and motion_state.get("robot_model"):
            print("[LoadProject] Removing existing robot model...")
            motion_state["robot_model"].remove_actors()
            motion_state["robot_model"] = None
            if stl_viewer:
                stl_viewer.safe_render()

        tree = ET.parse(filepath)
        root = tree.getroot()

        version = int(root.get("version", 1))

        # Parse global settings
        global_data = {"version": version}

        # URDF Path (resolve relative against this XML, then LME package dir)
        urdf_elem = root.find("URDFPath")
        global_data["urdf_path"] = resolve_project_path(
            urdf_elem.text if urdf_elem is not None and urdf_elem.text else "",
            filepath,
        )

        # Model Type (urdf or mjcf)
        model_type_elem = root.find("ModelType")
        global_data["model_type"] = model_type_elem.text if model_type_elem is not None and model_type_elem.text else "urdf"

        # Robot Name
        robot_name_elem = root.find("RobotName")
        global_data["robot_name"] = robot_name_elem.text if robot_name_elem is not None and robot_name_elem.text else ""

        # Joint Order
        joint_order = []
        joint_order_elem = root.find("JointOrder")
        if joint_order_elem is not None:
            for j_elem in joint_order_elem.findall("Joint"):
                joint_order.append(j_elem.get("name", ""))
        global_data["joint_order"] = joint_order

        # Joint Settings
        joint_settings = {}
        js_elem = root.find("JointSettings")
        if js_elem is not None:
            for j_elem in js_elem.findall("Joint"):
                jname = j_elem.get("name", "")
                settings = {}
                for key, value in j_elem.attrib.items():
                    if key != "name":
                        try:
                            if value.lower() in ("true", "false"):
                                settings[key] = value.lower() == "true"
                            elif "." in value:
                                settings[key] = float(value)
                            else:
                                settings[key] = int(value)
                        except ValueError:
                            settings[key] = value
                joint_settings[jname] = settings
        global_data["joint_settings"] = joint_settings

        # Joint Speed Presets (servo model presets)
        joint_speed_presets = []
        jsp_elem = root.find("JointSpeedPresets")
        if jsp_elem is not None:
            for preset_elem in jsp_elem.findall("Preset"):
                preset_name = preset_elem.get("name", "")
                try:
                    speed_rads = float(preset_elem.get("speed_rad_s", "0"))
                except ValueError:
                    speed_rads = 0.0
                if preset_name:
                    joint_speed_presets.append((preset_name, speed_rads))
        global_data["joint_speed_presets"] = joint_speed_presets

        # Joint Layout - contains "order" and "groups"
        joint_layout = {"order": [], "groups": {}}
        jl_elem = root.find("JointLayout")
        if jl_elem is not None:
            # Read order from Order element
            order_elem = jl_elem.find("Order")
            if order_elem is not None and order_elem.text:
                joint_layout["order"] = [j.strip() for j in order_elem.text.split(",") if j.strip()]
            # Read groups from Joint elements
            for j_elem in jl_elem.findall("Joint"):
                jname = j_elem.get("name", "")
                group_key = j_elem.get("group", "")
                if jname and group_key:
                    joint_layout["groups"][jname] = group_key
        global_data["joint_layout"] = joint_layout

        # Joint Group Presets
        presets = []
        presets_elem = root.find("JointGroupPresets")
        if presets_elem is not None:
            global_data["current_group_preset_index"] = int(presets_elem.get("currentIndex", -1))
            for preset_elem in presets_elem.findall("Preset"):
                preset = {"name": preset_elem.get("name", ""), "members": {}}
                for m_elem in preset_elem.findall("Member"):
                    jname = m_elem.get("name", "")
                    if jname:
                        try:
                            scale = float(m_elem.get("scale", "1.0"))
                        except ValueError:
                            scale = 1.0
                        preset["members"][jname] = {
                            "enabled": m_elem.get("enabled", "0") == "1",
                            "scale": scale,
                        }
                presets.append(preset)
        else:
            global_data["current_group_preset_index"] = -1
        global_data["joint_group_presets"] = presets

        # View Settings (3D view background, light, etc.)
        view_settings = {}
        view_elem = root.find("ViewSettings")
        if view_elem is not None:
            # Background colors
            bg_color_a_str = view_elem.get("bg_color_a", "")
            bg_color_b_str = view_elem.get("bg_color_b", "")
            if bg_color_a_str:
                try:
                    view_settings["bg_color_a"] = [float(c) for c in bg_color_a_str.split(",")]
                except ValueError:
                    pass
            if bg_color_b_str:
                try:
                    view_settings["bg_color_b"] = [float(c) for c in bg_color_b_str.split(",")]
                except ValueError:
                    pass
            # Gradient type
            view_settings["bg_gradient_type"] = view_elem.get("bg_gradient_type", "vertical")
            # Slider values
            try:
                view_settings["bg_slider_value"] = int(view_elem.get("bg_slider_value", 50))
            except ValueError:
                view_settings["bg_slider_value"] = 50
            try:
                view_settings["light_slider_value"] = int(view_elem.get("light_slider_value", 70))
            except ValueError:
                view_settings["light_slider_value"] = 70
        global_data["view_settings"] = view_settings

        # Home Position
        home_position = {}
        home_elem = root.find("HomePosition")
        if home_elem is not None:
            for hj_elem in home_elem.findall("Joint"):
                jname = hj_elem.get("name", "")
                try:
                    angle = float(hj_elem.get("angle", 0))
                    if jname:
                        home_position[jname] = angle
                except ValueError:
                    pass
        global_data["home_position"] = home_position

        # Camera Presets (A-E)
        loaded_cam_presets = {}
        cp_elem = root.find("CameraPresets")
        if cp_elem is not None:
            for ce in cp_elem.findall("Camera"):
                cname = ce.get("name", "")
                if not cname:
                    continue
                try:
                    loaded_cam_presets[cname] = {
                        "name":      ce.get("label",     ""),
                        "azimuth":   int(float(ce.get("azimuth",   0))),
                        "elevation": int(float(ce.get("elevation", 0))),
                        "distance":  float(ce.get("distance",  1.0)),
                        "focal_x":   float(ce.get("focal_x",  0.0)),
                        "focal_y":   float(ce.get("focal_y",  0.0)),
                        "focal_z":   float(ce.get("focal_z",  0.0)),
                    }
                except (ValueError, TypeError):
                    pass
        view_settings["camera_presets"] = loaded_cam_presets

        # Apply loaded joint speed presets (servo model presets)
        if global_data.get("joint_speed_presets"):
            from LegacyMotionEditor_Utils import save_joint_speed_presets
            save_joint_speed_presets(global_data["joint_speed_presets"])
            print(f"[ProjectXML] Applied {len(global_data['joint_speed_presets'])} servo model presets")

        # Project-wide Python code (user logic + walk params in ProjectCode)
        code_elem = root.find("ProjectCode")
        graph.project_code = (code_elem.text or "").strip() if code_elem is not None else ""

        # UserValueSession (root-level, shared across all actions)
        uv_elem = root.find("UserValueSession")
        if uv_elem is not None:
            from LegacyMotionEditor_Utils import default_user_value_session, USER_VALUE_SESSION_COUNT
            uv_list = default_user_value_session()
            for slot_elem in uv_elem.findall("Slot"):
                try:
                    idx = int(slot_elem.get("index", -1))
                except ValueError:
                    continue
                if not (0 <= idx < USER_VALUE_SESSION_COUNT):
                    continue
                kind = slot_elem.get("kind", "literal")
                if kind == "register":
                    uv_list[idx] = {"kind": "register", "name": slot_elem.get("name", "")}
                else:
                    try:
                        val = int(slot_elem.get("value", 0))
                    except ValueError:
                        val = 0
                    uv_list[idx] = {"kind": "literal", "value": val}
            global_data["user_value_session"] = uv_list

        # Check for multi-action format (version 2)
        actions_elem = root.find("Actions")
        if actions_elem is not None and version >= 2:
            # Multi-action format
            current_action_idx = int(actions_elem.get("currentIndex", 0))
            action_items = []

            for action_elem in actions_elem.findall("Action"):
                title = action_elem.get("title", "")
                action_data = _parse_action_data_from_xml(action_elem, global_data)
                action_items.append({
                    "title": title,
                    "data": action_data,
                })

            if not action_items:
                # No actions found, create empty one
                action_items.append({"title": "", "data": global_data})

            # Update motion_action_state if provided
            if motion_action_state is not None:
                motion_action_state["items"] = action_items
                motion_action_state["current"] = min(current_action_idx, len(action_items) - 1)

            # Load the current action
            current_data = action_items[motion_action_state["current"] if motion_action_state else 0]["data"]
            rm, urdf_path, ok = load_motion_data(
                current_data, graph, stl_viewer, joint_editor, playback_ctrl,
                motion_state=motion_state, skip_urdf=False, parent_window=parent_window
            )

            if ok:
                print(f"[ProjectXML] Loaded {len(action_items)} actions from {filepath}")
                robot_name = global_data.get("robot_name", "")
                if robot_name:
                    graph.robot_name = robot_name
                    if hasattr(graph, "name_input") and graph.name_input:
                        graph.name_input.setText(robot_name)
            return rm, urdf_path, ok, len(action_items), global_data.get("view_settings", {}), global_data.get("home_position", {})

    except Exception as e:
        print(f"[ProjectXML] Error loading: {e}")
        traceback.print_exc()
        return None, "", False, 0, {}


def _csv_escape(value):
    text = str(value)
    if any(ch in text for ch in [",", '"', "\n", "\r"]):
        return '"' + text.replace('"', '""') + '"'
    return text


def _sorted_output_connections(node):
    """Get sorted list of connected nodes from output ports.

    For PoseNode/DefineNode without branching, only the LAST connection per port is used
    (newest connection takes precedence). For BranchingNode, all connections are returned.
    Only connections to INPUT ports are valid (output -> input direction).
    """
    connections = []
    out_ports = node.output_ports()
    priorities = getattr(node, "out_port_priorities", None)
    branching_enabled = getattr(node, "branching_enabled", False)
    is_branching_type = isinstance(node, BranchingNode) or branching_enabled

    for port_idx, port in enumerate(out_ports):
        priority = 0
        if priorities and port_idx < len(priorities):
            priority = priorities[port_idx]

        # Filter to only INPUT ports (valid output->input connections)
        valid_connections = []
        for connected_port in port.connected_ports():
            target = connected_port.node()
            if target is None:
                continue
            # A valid connection: our OUTPUT port -> target's INPUT port
            target_input_ports = target.input_ports()
            if connected_port in target_input_ports:
                valid_connections.append(connected_port)

        # For non-branching nodes, only use the LAST (newest) connection per port
        if not is_branching_type and len(valid_connections) > 1:
            print(f"[Warning] {node.name()} port[{port_idx}] has {len(valid_connections)} connections, using last one")
            valid_connections = [valid_connections[-1]]

        for conn_idx, connected_port in enumerate(valid_connections):
            target = connected_port.node()
            if isinstance(target, (PoseNode, DefineNode, WaitNode, BranchingNode, MixNode, CommandNode, JumpNode)):
                connections.append((priority, port_idx, conn_idx, target))

    # Sort by priority, port_idx, connection_idx
    connections.sort(key=lambda item: (item[0], item[1], item[2]))
    result = [target for _priority, _port_idx, _conn_idx, target in connections]
    return result


def cleanup_orphaned_connections(graph):
    """Remove extra connections for non-branching nodes.

    For non-branching nodes (PoseNode, DefineNode without branching, BaseLinkNode),
    only the LAST connection per port is kept. Extra connections are removed.
    For BranchingNode, all connections are kept.
    Should be called before playback and before saving projects.
    """
    if not graph:
        return 0

    removed_count = 0
    node_types = (PoseNode, DefineNode, WaitNode, BranchingNode, MixNode, CommandNode, JumpNode, BaseLinkNode)

    for node in graph.all_nodes():
        if not isinstance(node, node_types):
            continue

        # Check if this is a branching-type node
        branching_enabled = getattr(node, "branching_enabled", False)
        is_branching_type = isinstance(node, BranchingNode) or branching_enabled

        for port in node.output_ports():
            # Get all valid connections (to input ports)
            valid_connections = []
            invalid_connections = []

            for connected_port in port.connected_ports():
                target = connected_port.node()
                if target is None:
                    invalid_connections.append(connected_port)
                    continue

                target_input_ports = target.input_ports()
                if connected_port not in target_input_ports:
                    invalid_connections.append(connected_port)
                    continue

                valid_connections.append(connected_port)

            # Remove invalid connections (null targets or non-input ports)
            for orphaned_port in invalid_connections:
                try:
                    target_name = orphaned_port.node().name() if orphaned_port.node() else "unknown"
                    print(f"[Cleanup] Removing invalid connection: {node.name()} -> {target_name}")
                    if hasattr(port, 'disconnect_from'):
                        port.disconnect_from(orphaned_port)
                        removed_count += 1
                except Exception as e:
                    print(f"[Cleanup] Error removing connection: {e}")

            # For non-branching nodes, keep only the LAST connection
            if not is_branching_type and len(valid_connections) > 1:
                # Keep the last one, remove the rest
                connections_to_remove = valid_connections[:-1]
                for orphaned_port in connections_to_remove:
                    try:
                        target_name = orphaned_port.node().name() if orphaned_port.node() else "unknown"
                        print(f"[Cleanup] Removing extra connection: {node.name()} -> {target_name}")
                        if hasattr(port, 'disconnect_from'):
                            port.disconnect_from(orphaned_port)
                            removed_count += 1
                    except Exception as e:
                        print(f"[Cleanup] Error removing connection: {e}")

    if removed_count > 0:
        print(f"[Cleanup] Removed {removed_count} extra connection(s)")

    return removed_count


def collect_motion_export_nodes(graph):
    """Startから分岐を含めて到達順にPoseNodeを集める"""
    start_node = None
    for node in graph.all_nodes():
        if isinstance(node, BaseLinkNode):
            start_node = node
            break
    if start_node is None:
        return []

    result = []
    visited = set()
    stack = list(reversed(_sorted_output_connections(start_node)))
    while stack:
        node = stack.pop()
        node_id = id(node)
        if node_id in visited:
            continue
        visited.add(node_id)
        result.append(node)
        next_nodes = _sorted_output_connections(node)
        stack.extend(reversed(next_nodes))
    return result


def build_motion_export_csv(graph, robot_model):
    """Poseごとの保存角度をCSVテキストへ変換する"""
    nodes = collect_motion_export_nodes(graph)
    joint_order = list(robot_model.joint_order) if robot_model else []
    lines = []
    for node in nodes:
        if isinstance(node, (DefineNode, WaitNode, BranchingNode, JumpNode, MemoNode)):
            continue
        row = [
            _csv_escape(getattr(node, "pose_name", node.name())),
            str(getattr(node, "frames", 1)),
        ]
        angles = getattr(node, "angles_deg", {})
        for jname in joint_order:
            row.append(_csv_escape(f"{jname}={angles.get(jname, 0.0):.3f}"))
        lines.append(",".join(row))
    return "\n".join(lines)


def load_motion_data(data, graph, stl_viewer, joint_editor, playback_ctrl,
                     motion_state=None, skip_urdf=False, parent_window=None):
    """辞書データからモーションを復元。skip_urdf=True のときは現在のロボットを維持（アクション切替用）。"""
    try:
        urdf_path = resolve_project_path(data.get("urdf_path", ""))

        robot_model = None
        if skip_urdf:
            # Action switching: reuse existing robot model and joint settings
            # JointSliders settings are shared across all actions
            if motion_state:
                robot_model = motion_state.get("robot_model")
                urdf_path = motion_state.get("urdf_path", "") or urdf_path
            # Note: joint_settings, joint_layout, joint_group_presets are NOT applied
            # because they are shared globally across all actions
        else:
            # Get model type (urdf or mjcf)
            model_type = data.get("model_type", "urdf")

            # Load model based on type
            if urdf_path and os.path.exists(urdf_path):
                # Remove existing actors before creating new ones
                if motion_state and motion_state.get('robot_model'):
                    motion_state['robot_model'].remove_actors()

                if model_type == 'mjcf':
                    # MJCF load
                    from LegacyMotionEditor_Importer import MJCFParser
                    mjcf_parser = MJCFParser()
                    working_dir = os.path.dirname(urdf_path)
                    mjcf_data = mjcf_parser.parse_mjcf(urdf_path, working_dir=working_dir)
                    robot_model = build_robot_model_from_mjcf(urdf_path, mjcf_data)
                else:
                    # URDF/Xacro/SDF/SRDF load - parse_urdf_file handles all
                    from LegacyMotionEditor_Importer import parse_urdf_file
                    result = parse_urdf_file(urdf_path, parent_widget=parent_window)
                    if result:
                        _, _, parsed_data = result
                        robot_model = build_robot_model_from_urdf(urdf_path, parsed_data)
                    else:
                        raise RuntimeError(f"Failed to parse model file: {urdf_path}")

                robot_model.build_vtk_actors(stl_viewer.renderer)
                # Track immediately so re-entry can clean up orphaned actors
                if motion_state is not None:
                    motion_state['robot_model'] = robot_model
                robot_model.model_type = model_type
                joint_editor.build_from_robot(robot_model)
                joint_editor.set_joint_settings(data.get("joint_settings", {}))
                joint_editor.set_joint_layout(data.get("joint_layout", {}))
                joint_editor.set_joint_group_presets(
                    data.get("joint_group_presets", []),
                    data.get("current_group_preset_index", -1),
                )
                # Update motion_state with model_type
                if motion_state is not None:
                    motion_state['model_type'] = model_type
                stl_viewer.reset_camera()
                stl_viewer.safe_render()
            else:
                if urdf_path:
                    print(f"[MotionJSON] Model file not found: {urdf_path}")
                # Ask user to select file - use unified file dialog
                fp, _ = QtWidgets.QFileDialog.getOpenFileName(
                    parent_window,
                    "Select Robot Model File",
                    "",
                    "All Model Files (*.urdf *.xacro *.sdf *.xml *.zip);;"
                    "URDF Files (*.urdf);;"
                    "Xacro Files (*.xacro);;"
                    "SDF Files (*.sdf);;"
                    "MJCF Files (*.xml *.zip);;"
                    "All Files (*)",
                )
                if fp:
                    # Refuse a derived MJCF (_play / _psclon / _pN) as the
                    # source. If a derived file is loaded again, we make
                    # double-derived files (like _play_play.xml) and rewrite
                    # the mesh paths twice, and the scene stops loading.
                    _fp_stem = os.path.splitext(os.path.basename(fp))[0]
                    _DERIVED = ("_play", "_psclon", "_p1", "_p2", "_p3", "_p4", "_p5", "_p6")
                    _bad_sfx = next(
                        (_s for _s in _DERIVED if _fp_stem.endswith(_s)), None)
                    if _bad_sfx:
                        QtWidgets.QMessageBox.warning(
                            parent_window,
                            "Cannot open a derived file",
                            f"Selected file:\n  {fp}\n\n"
                            f"The suffix '{_bad_sfx}' means this MJCF was "
                            f"made by LME or PhysicalOn from an original "
                            f"file. Please open the original MJCF instead.")
                        return
                    # Remove existing actors before creating new ones
                    if motion_state and motion_state.get('robot_model'):
                        motion_state['robot_model'].remove_actors()

                    # Auto-detect and parse the selected file
                    from LegacyMotionEditor_Importer import parse_model_file
                    result = parse_model_file(fp, parent_widget=parent_window)
                    if result:
                        fp, _, parsed_data, detected_type = result
                        if detected_type == 'mjcf':
                            robot_model = build_robot_model_from_mjcf(fp, parsed_data)
                        else:
                            robot_model = build_robot_model_from_urdf(fp, parsed_data)
                        model_type = detected_type
                    else:
                        raise RuntimeError(f"Failed to parse model file: {fp}")

                    robot_model.build_vtk_actors(stl_viewer.renderer)
                    # Track immediately so re-entry can clean up orphaned actors
                    if motion_state is not None:
                        motion_state['robot_model'] = robot_model
                    robot_model.model_type = model_type
                    joint_editor.build_from_robot(robot_model)
                    joint_editor.set_joint_settings(data.get("joint_settings", {}))
                    joint_editor.set_joint_layout(data.get("joint_layout", {}))
                    joint_editor.set_joint_group_presets(
                        data.get("joint_group_presets", []),
                        data.get("current_group_preset_index", -1),
                    )
                    urdf_path = fp
                    # Update motion_state with model_type
                    if motion_state is not None:
                        motion_state['model_type'] = model_type
                    stl_viewer.reset_camera()
                    stl_viewer.safe_render()

        # 既存の Pose / Define / Wait / Branch / Jump / Mix / Command / Memo ノードを削除
        for n in list(graph.all_nodes()):
            if isinstance(n, (PoseNode, DefineNode, WaitNode, BranchingNode, JumpNode, MixNode, CommandNode, MemoNode)):
                graph.remove_node(n)

        # Clean up orphaned pipes after node deletion
        if hasattr(graph, '_clear_orphaned_pipes'):
            graph._clear_orphaned_pipes()

        # motion_formulas and user_value_session are shared across actions
        if not skip_urdf:
            if "motion_formulas" in data:
                mf = data["motion_formulas"]
                if isinstance(mf, dict):
                    graph.motion_formulas = {str(k): str(v) for k, v in mf.items()}

            if data.get("user_value_session") is not None:
                graph.user_value_session = normalize_user_value_session(
                    data["user_value_session"]
                )
            else:
                graph.user_value_session = default_user_value_session()

        # ノードの復元
        node_map = {}
        for nd in data.get("nodes", []):
            nid = nd["id"]
            ntype = nd.get("node_type", "pose")
            if ntype == "define":
                node = graph.create_node(
                    "motion.nodes.DefineNode",
                    name=nd.get("name", "define"),
                    pos=QtCore.QPointF(nd.get("pos_x", 0), nd.get("pos_y", 0)),
                    skip_auto_position=True,
                )
                node.set_name(nd.get("name", "define"))
                node.define_uv_index = int(nd.get("define_uv_index", 0))
                node.define_memo = nd.get("define_memo", "") or ""
                node.define_kind = nd.get("define_kind", "literal")
                try:
                    node.define_literal = int(nd.get("define_literal", 0))
                except (TypeError, ValueError):
                    node.define_literal = 0
                node.define_register_name = nd.get("define_register_name", "") or ""
                node_map[nid] = node
                continue
            if ntype == "wait":
                node = graph.create_node(
                    "motion.nodes.WaitNode",
                    name=nd.get("name", "wait"),
                    pos=QtCore.QPointF(nd.get("pos_x", 0), nd.get("pos_y", 0)),
                    skip_auto_position=True,
                )
                node.wait_name = nd.get("name", "wait")
                node.set_name(node.wait_name)
                node.frames = int(nd.get("frames", 0))
                node.duration = float(nd.get("duration", 0.0))
                labels = nd.get("out_port_labels", ["default"])
                priorities = nd.get("out_port_priorities", [0])
                node.out_port_labels = list(labels)
                node.out_port_priorities = list(priorities)
                node_map[nid] = node
                continue
            if ntype == "branch":
                node = graph.create_node(
                    "motion.nodes.BranchingNode",
                    name=nd.get("name", "branch"),
                    pos=QtCore.QPointF(nd.get("pos_x", 0), nd.get("pos_y", 0)),
                    skip_auto_position=True,
                )
                node.set_name(nd.get("name", "branch"))
                node.branching_enabled = nd.get("branching_enabled", False)
                node.branch_outputs_swapped = nd.get("branch_outputs_swapped", False)
                node.branch_if_left = nd.get("branch_if_left", "UserVal_0")
                node.branch_if_op = normalize_branch_if_op_stored(
                    nd.get("branch_if_op", "==")
                )
                node.branch_if_right = nd.get("branch_if_right", "UserVal_1")
                node.branch_if_uv_enabled = nd.get("branch_if_uv_enabled", True)
                node.branch_if_formula_enabled = nd.get(
                    "branch_if_formula_enabled", False
                )
                node.branch_if_formula = nd.get("branch_if_formula", "Form1:foo")
                node.branch_if_pad_enabled = nd.get("branch_if_pad_enabled", False)
                node.branch_if_pad_button = nd.get("branch_if_pad_button", "L1")
                node.branch_if_pad_analog_enabled = nd.get("branch_if_pad_analog_enabled", False)
                node.branch_if_pad_analog_axis = nd.get("branch_if_pad_analog_axis", "Lx")
                node.branch_if_pad_analog_op = nd.get("branch_if_pad_analog_op", ">=")
                node.branch_if_pad_analog_threshold = int(nd.get("branch_if_pad_analog_threshold", 0))
                labels = nd.get("out_port_labels", ["default"])
                priorities = nd.get("out_port_priorities", [0])
                if node.branching_enabled:
                    node._lock_output_row_height()
                while node.output_count < len(labels):
                    node._add_branch_output(
                        labels[node.output_count]
                        if node.output_count < len(labels)
                        else "branch",
                        priorities[node.output_count]
                        if node.output_count < len(priorities)
                        else 10,
                    )
                node.out_port_labels = list(labels)
                node.out_port_priorities = list(priorities)
                node._sync_branching_port_labels()
                node._apply_branch_output_colors()
                QtCore.QTimer.singleShot(30, node._apply_branching_node_colors)
                node_map[nid] = node
                continue
            if ntype == "jump":
                node = graph.create_node(
                    "motion.nodes.JumpNode",
                    name=nd.get("name", "Jump to"),
                    pos=QtCore.QPointF(nd.get("pos_x", 0), nd.get("pos_y", 0)),
                    skip_auto_position=True,
                )
                node.set_name(nd.get("name", "Jump to"))
                try:
                    node.jump_target_action_index = int(
                        nd.get("jump_target_action_index", 0)
                    )
                except (TypeError, ValueError):
                    node.jump_target_action_index = 0
                node.jump_type = nd.get("jump_type", "action")
                node.jump_target_function = nd.get("jump_target_function", "")
                labels = nd.get("out_port_labels", ["default"])
                priorities = nd.get("out_port_priorities", [0])
                while node.output_count < len(labels):
                    node._add_jump_output(
                        labels[node.output_count]
                        if node.output_count < len(labels)
                        else "default",
                        priorities[node.output_count]
                        if node.output_count < len(priorities)
                        else 0,
                    )
                node.out_port_labels = list(labels)
                node.out_port_priorities = list(priorities)
                node.refresh_body_text()
                QtCore.QTimer.singleShot(20, node._apply_jump_node_colors)
                node_map[nid] = node
                continue
            if ntype == "mix":
                node = graph.create_node(
                    "motion.nodes.MixNode",
                    name=nd.get("name", "mix"),
                    pos=QtCore.QPointF(nd.get("pos_x", 0), nd.get("pos_y", 0)),
                    skip_auto_position=True,
                )
                node.mix_name = nd.get("name", "mix")
                node.set_name(node.mix_name)
                node.frames = int(nd.get("frames", 1))
                node.mix_settings = dict(nd.get("mix_settings", {}))
                labels = nd.get("out_port_labels", ["default"])
                priorities = nd.get("out_port_priorities", [0])
                node.out_port_labels = list(labels)
                node.out_port_priorities = list(priorities)
                node_map[nid] = node
                continue
            if ntype == "command":
                node = graph.create_node(
                    "motion.nodes.CommandNode",
                    name=nd.get("name", "command"),
                    pos=QtCore.QPointF(nd.get("pos_x", 0), nd.get("pos_y", 0)),
                    skip_auto_position=True,
                )
                node.command_name = nd.get("name", "command")
                node.set_name(node.command_name)
                node.frames = int(nd.get("frames", 1))
                node.command_settings = dict(nd.get("command_settings", {}))
                labels = nd.get("out_port_labels", ["default"])
                priorities = nd.get("out_port_priorities", [0])
                node.out_port_labels = list(labels)
                node.out_port_priorities = list(priorities)
                node_map[nid] = node
                continue
            if ntype == "memo":
                node = graph.create_node(
                    "motion.nodes.MemoNode",
                    name=nd.get("name", "Memo"),
                    pos=QtCore.QPointF(nd.get("pos_x", 0), nd.get("pos_y", 0)),
                    skip_auto_position=True,
                )
                node.set_name(nd.get("name", "Memo"))
                node.memo_text = nd.get("memo_text", "") or ""
                try:
                    node.modal_width = int(nd.get("modal_width", 420) or 420)
                except (TypeError, ValueError):
                    node.modal_width = 420
                try:
                    node.modal_height = int(nd.get("modal_height", 320) or 320)
                except (TypeError, ValueError):
                    node.modal_height = 320
                node.refresh_body_text()
                node_map[nid] = node
                continue
            node = graph.create_node(
                'motion.nodes.PoseNode',
                name=nd.get("name", "pose"),
                pos=QtCore.QPointF(nd.get("pos_x", 0), nd.get("pos_y", 0)),
                skip_auto_position=True
            )
            node.pose_name = nd.get("name", "pose")
            node.set_name(node.pose_name)
            node.duration = nd.get("duration", 1.0)
            node.frames = int(nd.get("frames", get_default_hz_fps()))
            node.angles_deg = nd.get("angles_deg", {})
            node.joint_easings = nd.get("joint_easings", {})
            node.branching_enabled = nd.get("branching_enabled", False)
            node.branch_outputs_swapped = nd.get("branch_outputs_swapped", False)
            node.branch_if_left = nd.get("branch_if_left", "UserVal_0")
            node.branch_if_op = normalize_branch_if_op_stored(
                nd.get("branch_if_op", "==")
            )
            node.branch_if_right = nd.get("branch_if_right", "UserVal_1")
            node.branch_if_uv_enabled = nd.get("branch_if_uv_enabled", True)
            node.branch_if_formula_enabled = nd.get("branch_if_formula_enabled", False)
            node.branch_if_formula = nd.get("branch_if_formula", "Form1:foo")
            node.branch_if_pad_enabled = nd.get("branch_if_pad_enabled", False)
            node.branch_if_pad_button = nd.get("branch_if_pad_button", "L1")
            node.branch_if_pad_analog_enabled = nd.get("branch_if_pad_analog_enabled", False)
            node.branch_if_pad_analog_axis = nd.get("branch_if_pad_analog_axis", "Lx")
            node.branch_if_pad_analog_op = nd.get("branch_if_pad_analog_op", ">=")
            node.branch_if_pad_analog_threshold = int(nd.get("branch_if_pad_analog_threshold", 0))
            labels = nd.get("out_port_labels", ["default"])
            priorities = nd.get("out_port_priorities", [0])
            if node.branching_enabled:
                node._lock_output_row_height()
            # ポート数を合わせる
            while node.output_count < len(labels):
                node._add_pose_output(
                    labels[node.output_count] if node.output_count < len(labels) else "branch",
                    priorities[node.output_count] if node.output_count < len(priorities) else 10
                )
            node.out_port_labels = list(labels)
            node.out_port_priorities = list(priorities)
            node._sync_branching_port_labels()
            node._apply_pose_output_colors()
            node_map[nid] = node

        # Find BaseLinkNode (Start node) for start edges
        base_link_node = None
        for n in graph.all_nodes():
            if isinstance(n, BaseLinkNode):
                base_link_node = n
                # Clear existing connections from Start node
                for port in n.output_ports():
                    for cp in list(port.connected_ports()):
                        try:
                            graph.disconnect_ports(port, cp)
                        except Exception:
                            pass
                break

        # ポートを最終位置に確定してからエッジを接続する。
        # _add_*_output() がスケジュールする QTimer(10ms) より先に同期実行することで、
        # establish_connection() 時点のポート座標が正しくなり矢印X座標ずれを防ぐ。
        for _node in node_map.values():
            for _method in ('_do_position_outputs', '_do_position_output_port'):
                _fn = getattr(_node, _method, None)
                if callable(_fn):
                    try:
                        _fn()
                    except Exception:
                        pass
        if base_link_node:
            _fn = getattr(base_link_node, '_do_position_output_port', None)
            if callable(_fn):
                try:
                    _fn()
                except Exception:
                    pass

        # エッジの復元
        for edge in data.get("edges", []):
            from_id = edge["from"]
            to_node = node_map.get(edge["to"])

            # Handle "start" source (BaseLinkNode)
            if from_id == "start":
                if base_link_node and to_node:
                    from_ports = base_link_node.output_ports()
                    to_ports = to_node.input_ports()
                    if from_ports and to_ports:
                        try:
                            from_port_obj = from_ports[0]
                            to_port_obj = to_ports[0]
                            if hasattr(from_port_obj, 'connect_to'):
                                from_port_obj.connect_to(to_port_obj)
                            else:
                                graph.connect_ports(from_port_obj, to_port_obj)
                        except Exception as e:
                            print(f"[MotionJSON] Error connecting from start: {e}")
                continue

            from_node = node_map.get(from_id)
            if from_node and to_node:
                # 対応する出力ポートを探す
                from_ports = from_node.output_ports()
                to_ports = to_node.input_ports()
                if from_ports and to_ports:
                    # labelに一致するポートを探す
                    target_port = from_ports[0]
                    for i, fp in enumerate(from_ports):
                        if i < len(from_node.out_port_labels) and from_node.out_port_labels[i] == edge.get("label", ""):
                            target_port = fp
                            break
                    try:
                        from_port_obj = target_port
                        to_port_obj = to_ports[0]
                        if hasattr(from_port_obj, 'connect_to'):
                            from_port_obj.connect_to(to_port_obj)
                        else:
                            graph.connect_ports(from_port_obj, to_port_obj)
                    except Exception as e:
                        print(f"[MotionJSON] Error connecting: {e}")

        # Playback設定 (shared across actions)
        pb = data.get("playback", {})
        if playback_ctrl and not skip_urdf:
            playback_ctrl.interpolation = pb.get("interpolation", EASING_OPTIONS[0])
            playback_ctrl.fps = pb.get("fps", 100)
            playback_ctrl.branch_mode = pb.get("branch_mode", "default-first")

        # 開始ノードの選択
        start_id = pb.get("start_node_id", "")
        if start_id in node_map:
            sn = node_map[start_id]
            sn.set_selected(True)
            if robot_model and isinstance(sn, PoseNode):
                joint_editor.set_current_pose_node(sn)
                joint_editor.set_angles(sn.angles_deg)
                # モーション JSON ロード直後の初期ポーズも「瞬間移動」なので
                # 閉ループ IK は warm-start を捨てて canonical zero から解く。
                robot_model.apply_joint_angles(
                    joint_editor.get_angles_for_3d(sn.angles_deg),
                    snap=True)
                stl_viewer.safe_render()

        print(f"[MotionJSON] Loaded {len(node_map)} nodes, {len(data.get('edges', []))} edges")
        return robot_model, urdf_path, True

    except Exception as e:
        print(f"[MotionJSON] Error loading: {e}")
        traceback.print_exc()
        QtWidgets.QMessageBox.critical(
            parent_window,
            "Load Error",
            f"Failed to load motion JSON:\n{e}",
        )
        return None, "", False


def load_motion_json(filepath, graph, stl_viewer, joint_editor, playback_ctrl,
                     parent_window=None):
    """JSONファイルからモーションを復元"""
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict) and data.get("urdf_path"):
            data["urdf_path"] = resolve_project_path(data.get("urdf_path", ""), filepath)
        return load_motion_data(
            data,
            graph,
            stl_viewer,
            joint_editor,
            playback_ctrl,
            motion_state=None,
            skip_urdf=False,
            parent_window=parent_window,
        )
    except Exception as e:
        print(f"[MotionJSON] Error reading file: {e}")
        traceback.print_exc()
        QtWidgets.QMessageBox.critical(
            parent_window,
            "Load Error",
            f"Failed to read motion JSON:\n{e}",
        )
        return None, "", False


def delete_selected_node(graph):
    selected_nodes = graph.selected_nodes()
    if selected_nodes:
        for node in selected_nodes:
            # BaseLinkNodeは削除不可
            if isinstance(node, BaseLinkNode):
                print("Cannot delete Base Link node")
                continue
            graph.remove_node(node)
        print(f"Deleted {len(selected_nodes)} node(s)")
    else:
        print("No node selected for deletion")

def cleanup_and_exit(_also_quit_app=False):
    """アプリケーションのクリーンアップと終了。

    app.aboutToQuit 経由で呼ばれる時点で、アプリは既にquit処理の最中にある
    （このシグナル自体がQApplication.quit()呼び出しの結果として発火する）。
    その状態でさらにQApplication.instance().quit()を呼ぶと、Qtのquit処理へ
    再入(reentrant)することになり、X11(Ubuntu)のプラットフォームプラグインでは
    これがセグメンテーション違反を起こすことを確認した（macOSでは未確認）。
    そのため既定では quit() を呼ばない。起動時エラー時（アプリがまだ
    quit処理に入っていない）の直接呼び出しでのみ _also_quit_app=True で
    明示的にquit()させる。"""
    # Session save must happen BEFORE graph.cleanup() destroys nodes.
    if _session_cb["save"] is not None:
        try:
            _session_cb["save"]()
            _session_cb["save"] = None  # prevent double-save on re-entry
        except Exception as _se:
            print(f"[Session] Save in cleanup failed: {_se}")
    _run_companion_shutdown()
    try:
        # グラフのクリーンアップ
        if 'graph' in globals():
            try:
                graph.cleanup()
            except Exception as e:
                print(f"Error cleaning up graph: {str(e)}")

        # STLビューアのクリーンアップ
        if 'stl_viewer' in globals():
            try:
                stl_viewer.cleanup()
            except Exception as e:
                print(f"Error cleaning up STL viewer: {str(e)}")

        # その他のリソースのクリーンアップ
        for window in QtWidgets.QApplication.topLevelWidgets():
            try:
                window.close()
            except Exception as e:
                print(f"Error closing window: {str(e)}")

    except Exception as e:
        print(f"Error during cleanup: {str(e)}")
    finally:
        lme_valkey.stop_fb_reader()
        # アプリケーションの終了（起動時エラーからの直接呼び出し時のみ）
        if _also_quit_app and QtWidgets.QApplication.instance():
            QtWidgets.QApplication.instance().quit()

def signal_handler(_signum, _frame):
    """Ctrl+Cシグナルのハンドラ"""
    print("\nCtrl+C detected, closing application...")
    if _session_cb["save"] is not None:
        try:
            _session_cb["save"]()
            _session_cb["save"] = None
        except Exception as _e:
            print(f"[Session] Save on signal failed: {_e}")
    _run_companion_shutdown()
    try:
        # アプリケーションのクリーンアップと終了
        if QtWidgets.QApplication.instance():
            # 全てのウィンドウを閉じる
            for window in QtWidgets.QApplication.topLevelWidgets():
                try:
                    window.close()
                except:
                    pass

            # アプリケーションの終了
            QtWidgets.QApplication.instance().quit()
    except Exception as e:
        print(f"Error during application shutdown: {str(e)}")
    finally:
        # 強制終了
        sys.exit(0)

def place_window_top_left(window, width=None, height=None):
    """Place window at the top-left of the available screen (below menu bar)."""
    screen = QtWidgets.QApplication.primaryScreen()
    ag = screen.availableGeometry() if screen else QtCore.QRect(0, 0, 1280, 800)
    w = int(width) if width else window.width()
    h = int(height) if height else window.height()
    window.setGeometry(ag.x(), ag.y(), w, h)


if __name__ == '__main__':
    try:
        # デバッグログ初期化
        DebugLogger.init()

        # macOSでPythonアプリをGUIアプリとして認識させる
        if sys.platform == 'darwin':
            try:
                from Foundation import NSBundle
                bundle = NSBundle.mainBundle()
                info = bundle.localizedInfoDictionary() or bundle.infoDictionary()
                if info:
                    info['LSBackgroundOnly'] = '0'
            except ImportError:
                try:
                    import ctypes, ctypes.util
                    objc = ctypes.cdll.LoadLibrary(ctypes.util.find_library('objc'))
                    objc.objc_getClass.restype = ctypes.c_void_p
                    objc.sel_registerName.restype = ctypes.c_void_p
                    objc.objc_msgSend.restype = ctypes.c_void_p
                    objc.objc_msgSend.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
                    NSApp = objc.objc_msgSend(
                        objc.objc_getClass(b'NSApplication'),
                        objc.sel_registerName(b'sharedApplication'))
                    objc.objc_msgSend.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int]
                    objc.objc_msgSend(
                        NSApp,
                        objc.sel_registerName(b'setActivationPolicy:'),
                        0)  # NSApplicationActivationPolicyRegular = 0
                except Exception:
                    pass

        # Ctrl+C / terminate handlers (SIGINT / SIGTERM / SIGBREAK on Windows)
        install_signal_handlers(signal_handler)

        # Load Valkey settings and apply to client at startup
        _vk_init = load_app_settings()
        lme_valkey.update_config({
            "enabled":   _vk_init.get("valkey_enabled",   False),
            "host":      _vk_init.get("valkey_host",      VALKEY_DEFAULT_HOST),
            "port":      int(_vk_init.get("valkey_port",  VALKEY_DEFAULT_PORT)),
            "write_key": _vk_init.get("valkey_write_key", VALKEY_DEFAULT_WRITE_KEY),
            "read_key":  _vk_init.get("valkey_read_key",  VALKEY_DEFAULT_READ_KEY),
        })
        lme_valkey.start_fb_reader()

        app = QtWidgets.QApplication(sys.argv)
        import logging
        logging.basicConfig(level=logging.WARNING)
        apply_dark_theme(app)

        # アプリケーション終了時のクリーンアップ設定
        app.aboutToQuit.connect(cleanup_and_exit)

        def _save_session_on_quit():
            g = main_window.geometry()
            s = load_app_settings()
            s["window_geometry"] = {"x": g.x(), "y": g.y(), "w": g.width(), "h": g.height()}
            save_app_settings(s)
        app.aboutToQuit.connect(_save_session_on_quit)
        
        timer = QtCore.QTimer()
        timer.start(500)
        timer.timeout.connect(lambda: None)

        # メインウィンドウの作成
        main_window = QtWidgets.QMainWindow()
        main_window.setWindowTitle(f"LegacyMotionEditor v{_LME_VERSION}")
        _startup_s = load_app_settings()
        _wg = _startup_s.get("window_geometry") or {}
        _w = int(_wg.get("w") or 1200)
        _h = int(_wg.get("h") or 650)
        place_window_top_left(main_window, _w, _h)

        # セントラルウィジェットの設定
        central_widget = QtWidgets.QWidget()
        outer_layout = QtWidgets.QHBoxLayout(central_widget)
        outer_layout.setContentsMargins(0, 0, 0, 0)
        outer_layout.setSpacing(0)

        # 左セクション（action_bar + left_panel + graph）
        left_section = QtWidgets.QWidget()
        left_section_layout = QtWidgets.QVBoxLayout(left_section)
        left_section_layout.setContentsMargins(0, 0, 0, 0)
        left_section_layout.setSpacing(0)

        action_bar = QtWidgets.QWidget()
        action_bar_layout = QtWidgets.QHBoxLayout(action_bar)
        action_bar_layout.setContentsMargins(6, 4, 6, 4)

        action_combo = QtWidgets.QComboBox()
        action_combo.setStyleSheet(_MAIN_WINDOW_COMBO_TEXT_STYLE)
        action_combo.setMinimumWidth(220)
        action_combo.setSizePolicy(
            QtWidgets.QSizePolicy.Expanding,
            QtWidgets.QSizePolicy.Fixed,
        )
        action_rename_btn = QtWidgets.QPushButton("Rename")
        action_add_btn = QtWidgets.QPushButton("Add")
        action_dup_btn = QtWidgets.QPushButton("Duplicate")
        action_reorder_btn = QtWidgets.QPushButton("Reorder")
        action_del_btn = QtWidgets.QPushButton("Delete")
        for _b in (
            action_rename_btn,
            action_add_btn,
            action_dup_btn,
            action_reorder_btn,
            action_del_btn,
        ):
            _b.setFixedWidth(88)
        action_bar_layout.addWidget(action_combo, 1)
        action_bar_layout.addWidget(action_add_btn)
        action_bar_layout.addWidget(action_dup_btn)
        action_bar_layout.addWidget(action_rename_btn)
        action_bar_layout.addWidget(action_reorder_btn)
        action_bar_layout.addWidget(action_del_btn)


        left_section_layout.addWidget(action_bar)

        # 左コンテンツ（left_panel + graph）を入れるウィジェット
        left_content = QtWidgets.QWidget()
        main_layout = QtWidgets.QHBoxLayout(left_content)
        main_layout.setContentsMargins(0, 0, 0, 5)
        main_layout.setSpacing(0)
        left_section_layout.addWidget(left_content, 1)

        # STLビューアとグラフの設定（先に作成）
        stl_viewer = STLViewerWidget(central_widget)
        graph = CustomNodeGraph(stl_viewer)
        graph.setup_custom_view()

        # base_linkノードの作成
        base_node = graph.create_base_link()

        # --- モーションエディタ用の状態変数 ---
        motion_state = {
            'robot_model': None,
            'urdf_path': '',
            'model_type': '',  # 'urdf' or 'mjcf'
        }
        motion_action_state = {
            "current": 0,
            "items": [{"title": "", "data": None}],
        }
        graph.motion_action_state = motion_action_state
        graph.project_code = ""

        # --- PlaybackController ---
        playback_ctrl = PlaybackController()
        graph.playback_ctrl = playback_ctrl

        # 左パネルの設定
        left_panel = QtWidgets.QWidget()
        left_panel.setFixedWidth(LEFT_PANEL_WIDTH)
        left_layout = QtWidgets.QVBoxLayout(left_panel)

        # Import Model ボタン（Name: フィールドの上）
        _import_model_btn = QtWidgets.QPushButton("Import Model")
        _import_model_btn.setFixedWidth(120)
        left_layout.addWidget(_import_model_btn)

        # 名前入力フィールドの設定
        name_label = create_label("Name:")
        left_layout.addWidget(name_label)
        name_input = QtWidgets.QLineEdit("robot_x")
        name_input.setFixedWidth(120)
        name_input.setStyleSheet("QLineEdit { padding-left: 3px; padding-top: 0px; padding-bottom: 0px; }")
        left_layout.addWidget(name_input)

        # 名前入力フィールドとグラフを接続（graphが定義された後に接続）
        name_input.textChanged.connect(graph.update_robot_name)

        # ボタンの作成と設定
        buttons = {
            "--spacer1--": None,
            "Add Pose": None,
            "Add Branching": None,
            "Add Jump": None,
            "Add Define": None,
            "Delete Node": None,
            "Code": None,
            "--spacer2--": None,
            "Save Project": None,
            "Load Project": None,
            "Import Model": _import_model_btn,
            "Export Motion": None,
            "Export Cartridge": None,
            "--spacer3--": None,
            "Sliders": None,
            "Value List": None,
            "Pad": None,
            "MuJoCo Studio": None,
            "Config": None,
        }

        _BUTTON_DISPLAY = {}  # ボタン表示名はキー名をそのまま使用
        # Define/Command/Mix をまとめた QComboBox のキー
        _ADD_SUB_KEY = "Add Define"
        _ADD_SUB_ITEMS = ["Wait", "Define", "Command", "Mix", "Memo"]

        use_pc_pad_checkbox = [None]  # Padボタン横のチェックボックス
        for button_text in buttons.keys():

            if button_text.startswith("--spacer"):
                spacer = QtWidgets.QWidget()
                spacer.setFixedHeight(1)
                left_layout.addWidget(spacer)
            elif button_text == "Import Model":
                pass  # already placed above Name: field
            elif button_text == "Pad":
                # Padボタンは横にチェックボックスを配置
                pad_row = QtWidgets.QHBoxLayout()
                pad_row.setContentsMargins(0, 0, 0, 0)
                pad_row.setSpacing(4)
                display = _BUTTON_DISPLAY.get(button_text, button_text)
                button = QtWidgets.QPushButton(display)
                button.setFixedWidth(100)
                pad_row.addWidget(button)
                use_pc_pad_checkbox[0] = QtWidgets.QCheckBox()
                use_pc_pad_checkbox[0].setToolTip("Use PC Pad")
                pad_row.addWidget(use_pc_pad_checkbox[0])
                pad_row.addStretch()
                left_layout.addLayout(pad_row)
                buttons[button_text] = button
            elif button_text == _ADD_SUB_KEY:
                # QPushButton + QMenu でセンタリングと外観を他ボタンと統一
                combo = QtWidgets.QPushButton("Add Other")
                combo.setFixedWidth(120)
                combo.setStyleSheet("QPushButton::menu-indicator { width: 0px; }")
                _sub_menu = QtWidgets.QMenu(combo)
                _sub_menu.setStyleSheet("QMenu { color: black; } QMenu::item { color: black; }")
                for _item in _ADD_SUB_ITEMS:
                    _sub_menu.addAction(_item)
                combo.setMenu(_sub_menu)
                left_layout.addWidget(combo)
                buttons[button_text] = combo
            else:
                display = _BUTTON_DISPLAY.get(button_text, button_text)
                button = QtWidgets.QPushButton(display)
                button.setFixedWidth(120)
                left_layout.addWidget(button)
                buttons[button_text] = button

        left_layout.addStretch()

        # ボタンのコネクション設定（既存）
        buttons["Delete Node"].clicked.connect(
            lambda: delete_selected_node(graph))

        # Code editor window (single instance, reused)
        _code_editor_window = [None]

        def _open_code_editor():
            settings = load_app_settings()
            mode = settings.get("code_editor_mode", "internal")
            if mode == "external":
                _open_external_code_editor(settings)
            else:
                _open_internal_code_editor()

        def _open_internal_code_editor():
            if not _CODE_EDITOR_OK:
                QtWidgets.QMessageBox.warning(
                    None, "Code Editor",
                    "LegacyMotionEditor_CodeEditor.py が見つかりません。"
                )
                return
            win = _code_editor_window[0]
            if win is None:
                win = CodeEditorWindow()
                win.code_saved.connect(_on_code_saved)
                _code_editor_window[0] = win
            project_code = getattr(graph, "project_code", "") or ""
            if not project_code.strip():
                project_code = CODE_DEFAULT_TEMPLATE
            win.set_code(project_code)
            win.show()
            win.raise_()
            win.activateWindow()

        def _on_code_saved(code):
            graph.project_code = code
            print(
                f"[Code] Project code saved ({len(code)} chars total, "
                f"{len(get_function_names(code))} functions)"
            )

        def _open_external_code_editor(settings):
            import tempfile
            app_path = settings.get("code_editor_path", "")
            if not app_path:
                QtWidgets.QMessageBox.warning(
                    None, "Code Editor",
                    "外部エディタのパスが設定されていません。\n"
                    "Config → Code Editor で設定してください。\n"
                    "(例: macOS .app / Windows Code.exe / Linux の code)"
                )
                return
            # Write project code to temp file, then open with external editor
            code = getattr(graph, "project_code", "") or ""
            tmp = tempfile.NamedTemporaryFile(
                suffix=".py", prefix="lme_code_", delete=False, mode="w", encoding="utf-8"
            )
            tmp.write(code)
            tmp_path = tmp.name
            tmp.close()
            try:
                launch_external_editor(app_path, tmp_path)
                QtWidgets.QMessageBox.information(
                    None, "Code Editor",
                    f"外部エディタで開きました:\n{tmp_path}\n\n"
                    "編集後、ファイルを保存してから\n「Load Code from File」で読み込んでください。"
                )
            except Exception as e:
                QtWidgets.QMessageBox.critical(None, "Code Editor Error", str(e))

        buttons["Code"].clicked.connect(_open_code_editor)




        # --- Joint Editor Panel (独立ウィンドウとして表示) ---
        joint_editor = JointEditorPanel()
        joint_editor.graph = graph
        graph.joint_editor = joint_editor
        stl_viewer.joint_editor = joint_editor
        graph.motion_state = motion_state

        def sync_playback_home_angles():
            playback_ctrl.set_home_position_angles(
                getattr(joint_editor, "home_position_angles", None) or {}
            )

        _orig_set_home_position = joint_editor.set_home_position

        def _set_home_position_and_sync(angles):
            _orig_set_home_position(angles)
            sync_playback_home_angles()

        joint_editor.set_home_position = _set_home_position_and_sync
        sync_playback_home_angles()

        # Sync overlay group preset combo with joint_editor
        def _sync_overlay_group_combo():
            combo = stl_viewer.group_preset_combo
            combo.blockSignals(True)
            combo.clear()
            combo.addItem("Individual")
            for p in joint_editor.joint_group_presets:
                combo.addItem(p.get("name", "") or "Preset")
            idx = max(0, joint_editor.current_group_preset_index + 1)
            idx = min(idx, combo.count() - 1)
            combo.setCurrentIndex(idx)
            combo.blockSignals(False)

        def _on_overlay_group_combo_changed(combo_idx):
            joint_editor._select_group_preset(combo_idx - 1, reset_master=True)
            je_combo = joint_editor.group_preset_combo
            je_combo.blockSignals(True)
            je_combo.setCurrentIndex(combo_idx)
            je_combo.blockSignals(False)

        stl_viewer.group_preset_combo.currentIndexChanged.connect(_on_overlay_group_combo_changed)

        # Patch _select_group_preset to keep overlay combo in sync.
        # This is the single entry point for changing the active preset, so syncing
        # here avoids double-firing issues that arose from a second currentIndexChanged
        # connection on the JE combo.
        _orig_select_preset = joint_editor._select_group_preset
        def _patched_select_preset(preset_index, reset_master=True):
            _orig_select_preset(preset_index, reset_master=reset_master)
            combo = stl_viewer.group_preset_combo
            combo.blockSignals(True)
            new_idx = max(0, min(preset_index + 1, combo.count() - 1))
            combo.setCurrentIndex(new_idx)
            combo.blockSignals(False)
        joint_editor._select_group_preset = _patched_select_preset

        # Patch _refresh_group_preset_controls to also update overlay
        _orig_refresh_group = joint_editor._refresh_group_preset_controls
        def _patched_refresh_group():
            _orig_refresh_group()
            _sync_overlay_group_combo()
        joint_editor._refresh_group_preset_controls = _patched_refresh_group

        joint_editor.setWindowTitle("Joint Sliders")
        joint_editor.setMinimumWidth(JOINT_EDITOR_WIDTH)
        joint_editor.setMinimumHeight(400)

        # Slidersボタンの接続
        def show_sliders_window():
            joint_editor.show()
            joint_editor.raise_()
            joint_editor.activateWindow()
        buttons["Sliders"].clicked.connect(show_sliders_window)

        # --- Command Editor Panel (独立ウィンドウとして表示) ---
        command_editor = CommandEditorPanel()
        command_editor.graph = graph
        graph.command_editor = command_editor
        command_editor.setWindowTitle("Command Editor")
        command_editor.setMinimumWidth(600)
        command_editor.setMinimumHeight(400)

        # --- Mix Editor Panel (独立ウィンドウとして表示) ---
        mix_editor = MixEditorPanel()
        mix_editor.graph = graph
        graph.mix_editor = mix_editor
        mix_editor.setWindowTitle("Mix Editor")
        mix_editor.setMinimumWidth(800)
        mix_editor.setMinimumHeight(400)

        def show_value_list():
            dlg = ValueListDialog(graph, parent=main_window)
            dlg.exec()

        buttons["Value List"].clicked.connect(show_value_list)

        # Padボタンの接続
        pad_dialog = [None]
        pad_signal_connected = [False]

        def _sync_checkbox_to_dialog(checked):
            """外側のチェックボックス → Padモーダルに同期."""
            if pad_dialog[0] is not None:
                pad_dialog[0].use_pc_pad_checkbox.blockSignals(True)
                pad_dialog[0].use_pc_pad_checkbox.setChecked(checked)
                pad_dialog[0].use_pc_pad_checkbox.blockSignals(False)
                pad_dialog[0]._on_use_pc_pad_toggled(checked)

        def _sync_dialog_to_checkbox(checked):
            """Padモーダル → 外側のチェックボックスに同期."""
            if use_pc_pad_checkbox[0] is not None:
                use_pc_pad_checkbox[0].blockSignals(True)
                use_pc_pad_checkbox[0].setChecked(checked)
                use_pc_pad_checkbox[0].blockSignals(False)

        if use_pc_pad_checkbox[0] is not None:
            use_pc_pad_checkbox[0].toggled.connect(_sync_checkbox_to_dialog)

        def show_pad_window():
            # Reuse existing instance (don't create new one if exists)
            if pad_dialog[0] is None:
                # parent=None: child dialogs raise the whole LME app on macOS
                # and bury MuJoCoStudio (separate process) behind the main window.
                pad_dialog[0] = PadMonitorDialog(None)
                # シグナル接続
                pad_dialog[0].use_pc_pad_changed.connect(_sync_dialog_to_checkbox)
                pad_signal_connected[0] = True
                # 初期状態を同期
                if use_pc_pad_checkbox[0] is not None:
                    _sync_dialog_to_checkbox(pad_dialog[0].use_pc_pad_checkbox.isChecked())
                _wire_pad_playback_buttons()
                pad_dialog[0].set_mujoco_running_checker(_mujoco_studio_is_running)
                pad_dialog[0].open_mujoco_requested.connect(launch_mujoco_studio)
                pad_dialog[0].respawn_requested.connect(lme_valkey.request_reset)
            # Sync play-button color if already playing when PAD opens
            if hasattr(pad_dialog[0], "set_play_active"):
                pad_dialog[0].set_play_active(
                    bool(playback_ctrl.is_playing and not playback_ctrl.is_paused)
                )
            pad_dialog[0].show()
            pad_dialog[0].raise_()
        buttons["Pad"].clicked.connect(show_pad_window)

        _pad_playback_handlers = [None]  # (play_fn, stop_fn, home_fn, zero_fn) set after handlers are defined

        def _wire_pad_playback_buttons():
            """Connect PAD Home / Zero / ▶︎_ / ■ to the main window handlers."""
            dlg = pad_dialog[0]
            handlers = _pad_playback_handlers[0]
            if dlg is None or handlers is None or getattr(dlg, "_playback_wired", False):
                return
            play_fn, stop_fn, home_fn, zero_fn = handlers
            dlg.play_requested.connect(play_fn)
            dlg.stop_requested.connect(stop_fn)
            dlg.home_requested.connect(home_fn)
            dlg.zero_requested.connect(zero_fn)
            dlg._playback_wired = True

        def on_normalize_joints_from_config():
            """Config ボタンから関節名の正規化を実行"""
            if not _RLB_OK:
                QtWidgets.QMessageBox.warning(
                    main_window, "Normalize",
                    "RobotLabelBridge is not available (import failed)."
                )
                return
            rm = motion_state.get('robot_model')
            if rm is None:
                QtWidgets.QMessageBox.warning(
                    main_window, "Normalize",
                    "No robot model loaded. Load a model first."
                )
                return
            name_map = build_joint_name_map(rm)
            if not name_map:
                QtWidgets.QMessageBox.information(
                    main_window, "Normalize",
                    "All joint names are already canonical. No changes needed."
                )
                return
            detail = "\n".join(f"  {src} → {tgt}" for src, tgt in sorted(name_map.items()))
            reply = QtWidgets.QMessageBox.question(
                main_window, "Normalize Joint Names",
                f"Rename the following joints in the graph?\n\n{detail}",
                QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
                QtWidgets.QMessageBox.Yes,
            )
            if reply != QtWidgets.QMessageBox.Yes:
                return
            count = apply_joint_name_map(graph, rm, name_map)
            joint_editor.build_from_robot(rm)
            QtWidgets.QMessageBox.information(
                main_window, "Normalize",
                f"Done. {len(name_map)} joint names renamed ({count} occurrences)."
            )
            print(f"[Normalize] {len(name_map)} joints renamed, {count} occurrences")

        # Clear handlers (used by SettingsDialog signals)
        def _on_clear_all():
            """Clear models + nodes and reset to initial state."""
            if motion_state['robot_model'] is not None:
                motion_state['robot_model'].remove_actors()
            motion_state['robot_model'] = None
            motion_state['urdf_path'] = ''
            motion_state['model_type'] = ''
            stl_viewer.set_robot_model(None)
            graph.clear_graph()
            graph.robot_name = "robot_x"
            graph.project_dir = None
            graph.meshes_dir = None
            graph.create_base_link()
            stl_viewer.safe_render()

        def _on_clear_models():
            """Clear imported robot model (3D meshes) without removing nodes."""
            if motion_state['robot_model'] is not None:
                motion_state['robot_model'].remove_actors()
            motion_state['robot_model'] = None
            motion_state['urdf_path'] = ''
            motion_state['model_type'] = ''
            stl_viewer.set_robot_model(None)
            graph.meshes_dir = None
            stl_viewer.safe_render()

        def _on_clear_nodes():
            """Clear all graph nodes (and their STL actors) then recreate base_link."""
            graph.clear_graph()
            graph.create_base_link()
            stl_viewer.safe_render()

        # Settingsボタンの接続
        settings_dialog = [None]  # リストで参照を保持
        def show_settings_window():
            if settings_dialog[0] is None or not settings_dialog[0].isVisible():
                settings_dialog[0] = SettingsDialog(
                    stl_viewer.bg_color_a,
                    stl_viewer.bg_color_b,
                    stl_viewer.bg_gradient_type,
                    stl_viewer.bg_slider_value,
                    stl_viewer.light_slider_value,
                    main_window
                )
                settings_dialog[0].bg_color_changed.connect(stl_viewer.set_bg_colors)
                settings_dialog[0].bg_gradient_changed.connect(stl_viewer.set_bg_gradient_type)
                settings_dialog[0].bg_slider_changed.connect(stl_viewer.set_bg_slider_value)
                settings_dialog[0].light_slider_changed.connect(stl_viewer.set_light_slider_value)
                settings_dialog[0].motion_defaults_changed.connect(
                    on_motion_defaults_from_settings
                )
                settings_dialog[0].joint_settings_requested.connect(
                    joint_editor._show_joint_settings_dialog
                )
                settings_dialog[0].sliders_settings_requested.connect(
                    show_sliders_window
                )
                settings_dialog[0].link_group_settings_requested.connect(
                    joint_editor._show_joint_group_dialog
                )
                settings_dialog[0].frame_presets_changed.connect(
                    joint_editor.update_frame_presets
                )
                settings_dialog[0].set_home_position_requested.connect(
                    lambda: joint_editor.set_home_position(joint_editor.get_angles())
                )
                settings_dialog[0].valkey_changed.connect(lme_valkey.update_config)
                lme_valkey.set_fb_callback(
                    lambda txt: settings_dialog[0].set_valkey_status(txt)
                    if settings_dialog[0] and settings_dialog[0].isVisible() else None
                )
                settings_dialog[0].normalize_joints_requested.connect(
                    on_normalize_joints_from_config
                )
                settings_dialog[0].clear_all_requested.connect(
                    _on_clear_all
                )
                settings_dialog[0].clear_models_requested.connect(
                    _on_clear_models
                )
                settings_dialog[0].clear_nodes_requested.connect(
                    _on_clear_nodes
                )
                settings_dialog[0].undo_limit_changed.connect(
                    undo_stack.set_max_size
                )
            settings_dialog[0].show()
            settings_dialog[0].raise_()
            settings_dialog[0].activateWindow()
        buttons["Config"].clicked.connect(show_settings_window)

        def launch_mujoco_studio():
            import subprocess
            if _mujoco_studio_is_running():
                # 既に起動している場合はウィンドウを最前面へ
                _bring_mujoco_studio_to_front()
                return
            studio_path = os.path.join(_LEGACY_EDITOR_DIR, "LegacyMotionEditor_MuJoCoStudio.py")
            s = load_app_settings()
            model = (motion_state.get("urdf_path") or s.get("last_model_path") or "").strip()
            if model and not os.path.isfile(model):
                model = ""
            # Do not open URDF / xacro / SDF files in MuJoCoStudio.
            # MuJoCoStudio can only open MJCF (.xml). Tell the user to
            # use urdf_kitchen, which can change URDF into MJCF.
            # https://github.com/Ninagawa123/LegacyMotionEditor/issues/1
            if model:
                _ext = os.path.splitext(model)[1].lower()
                if _ext in (".urdf", ".xacro", ".sdf"):
                    _btn_open = QtWidgets.QMessageBox.StandardButton.Ok
                    _mb = QtWidgets.QMessageBox(main_window)
                    _mb.setIcon(QtWidgets.QMessageBox.Icon.Information)
                    _mb.setWindowTitle("MuJoCoStudio requires MJCF")
                    _mb.setTextFormat(QtCore.Qt.TextFormat.RichText)
                    _mb.setText(
                        "<p>MuJoCoStudio can only open <b>MJCF (.xml)</b> files.<br>"
                        f"Your file is a <b>{_ext.upper().lstrip('.')}</b> file:</p>"
                        f"<pre>{model}</pre>"
                        "<p>You can change URDF to MJCF with <b>URDF_kitchen</b>:<br>"
                        '<a href="https://github.com/Ninagawa123/URDF_kitchen">'
                        'https://github.com/Ninagawa123/URDF_kitchen</a></p>'
                    )
                    _mb.setStandardButtons(_btn_open)
                    _mb.setTextInteractionFlags(QtCore.Qt.TextInteractionFlag.TextBrowserInteraction)
                    _mb.exec()
                    return
            cmd = [
                sys.executable, studio_path,
                "--valkey-host", str(getattr(lme_valkey, "_host", None) or s.get("valkey_host", VALKEY_DEFAULT_HOST)),
                "--valkey-port", str(int(getattr(lme_valkey, "_port", 0) or s.get("valkey_port", VALKEY_DEFAULT_PORT))),
                "--receive-key", str(getattr(lme_valkey, "_write_key", None) or s.get("valkey_write_key", VALKEY_DEFAULT_WRITE_KEY)),
                "--publish-key", str(getattr(lme_valkey, "_read_key", None) or s.get("valkey_read_key", VALKEY_DEFAULT_READ_KEY)),
            ]
            if model:
                cmd.extend(["--model", model])
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
            _mujoco_studio_procs.append(proc)
            dlg = pad_dialog[0]
            if dlg is not None:
                dlg._refresh_open_mujoco_btn()
        buttons["MuJoCo Studio"].clicked.connect(launch_mujoco_studio)

        export_motion_dialog = [None]
        def show_export_motion_window():
            text = build_motion_export_csv(graph, motion_state['robot_model'])
            export_motion_dialog[0] = ExportMotionDialog(text, main_window)
            export_motion_dialog[0].show()
            export_motion_dialog[0].raise_()
            export_motion_dialog[0].activateWindow()
        buttons["Export Motion"].clicked.connect(show_export_motion_window)

        # --- 右パネル: 3Dビュー + Playback Controls ---
        right_panel = QtWidgets.QWidget()
        right_panel.setMinimumWidth(0)
        right_layout = QtWidgets.QVBoxLayout(right_panel)
        right_layout.setContentsMargins(0, 6, 6, 0)  # left, top, right, bottom
        right_layout.setSpacing(0)
        right_layout.addWidget(stl_viewer, stretch=1)

        # 再生コントロール
        playback_bar = QtWidgets.QWidget()
        pb_layout = QtWidgets.QVBoxLayout(playback_bar)
        pb_layout.setContentsMargins(4, 0, 4, 2)
        pb_layout.setSpacing(2)

        # カメラプリセット用データ (A〜E、設定ファイルから読み込み)
        _DEFAULT_CAM_PRESETS = {
            "A": {"name": "Front", "azimuth": 0,   "elevation": 0,  "distance": 1.0, "focal_x": 0.0, "focal_y": 0.0, "focal_z": 0.0},
            "B": {"name": "Side",  "azimuth": 90,  "elevation": 0,  "distance": 1.0, "focal_x": 0.0, "focal_y": 0.0, "focal_z": 0.0},
            "C": {"name": "Top",   "azimuth": 0,   "elevation": 90, "distance": 1.0, "focal_x": 0.0, "focal_y": 0.0, "focal_z": 0.0},
            "D": {"name": "",      "azimuth": 180, "elevation": 0,  "distance": 1.0, "focal_x": 0.0, "focal_y": 0.0, "focal_z": 0.0},
            "E": {"name": "",      "azimuth": 0,   "elevation": 0,  "distance": 1.0, "focal_x": 0.0, "focal_y": 0.0, "focal_z": 0.0},
        }
        _saved_cam = load_app_settings().get("camera_presets", {})
        camera_presets = {}
        for _k, _dv in _DEFAULT_CAM_PRESETS.items():
            _sv = _saved_cam.get(_k, {})
            camera_presets[_k] = {fld: _sv.get(fld, _dv[fld]) for fld in _dv}
        cam_id_to_name = {1: "A", 2: "B", 3: "C", 4: "D", 5: "E"}

        def _save_camera_presets():
            s = load_app_settings()
            s["camera_presets"] = {k: dict(v) for k, v in camera_presets.items()}
            save_app_settings(s)

        # 1行目: Cam: A B C D E [≡]
        cam_row = QtWidgets.QWidget()
        cam_layout = QtWidgets.QHBoxLayout(cam_row)
        cam_layout.setContentsMargins(0, 0, 0, 0)
        cam_layout.setSpacing(3)

        black_text_style = "color: black;"

        cam_label = create_label("Cam:")
        cam_label.setStyleSheet(black_text_style)
        cam_layout.addWidget(cam_label)

        cam_group = QtWidgets.QButtonGroup(cam_row)
        for _cam_i, _cam_name in enumerate(["A", "B", "C", "D", "E"], 1):
            _r = QtWidgets.QRadioButton(_cam_name)
            _r.setStyleSheet(black_text_style)
            if _cam_i == 1:
                _r.setChecked(True)
            cam_group.addButton(_r, _cam_i)
            cam_layout.addWidget(_r)

        cam_settings_btn = QtWidgets.QPushButton("≡")
        cam_settings_btn.setFixedWidth(28)
        cam_settings_btn.setToolTip("Camera preset settings")
        cam_layout.addWidget(cam_settings_btn)

        cam_layout.addStretch()
        pb_layout.addWidget(cam_row)

        def apply_camera_preset(preset):
            """プリセット (azimuth, elevation, distance, focal_x/y/z) をカメラに適用"""
            import math
            azimuth   = preset["azimuth"]
            elevation = preset["elevation"]
            distance  = max(0.01, float(preset.get("distance", 1.0)))
            fx = float(preset.get("focal_x", 0.0))
            fy = float(preset.get("focal_y", 0.0))
            fz = float(preset.get("focal_z", 0.0))
            camera = stl_viewer.renderer.GetActiveCamera()
            camera.ParallelProjectionOn()
            camera.SetFocalPoint(fx, fy, fz)
            az_rad = math.radians(azimuth)
            el_rad = math.radians(elevation)
            camera.SetPosition(
                fx + distance * math.cos(el_rad) * math.cos(az_rad),
                fy + distance * math.cos(el_rad) * math.sin(az_rad),
                fz + distance * math.sin(el_rad),
            )
            camera.SetViewUp(0, 0, 1)
            camera.OrthogonalizeViewUp()
            # Orthographic zoom: parallel scale = half the visible height
            camera.SetParallelScale(distance * 0.5)
            stl_viewer.renderer.ResetCameraClippingRange()
            stl_viewer.safe_render()

        def on_camera_selected(btn_id):
            cam_name = cam_id_to_name[btn_id]
            apply_camera_preset(camera_presets[cam_name])

        def on_reframe():
            """Reframeボタン: モデル全体を画面に収めて正面(A)を向ける。設定は変更しない。"""
            import math
            renderer = stl_viewer.renderer
            actors = renderer.GetActors()
            if actors.GetNumberOfItems() == 0:
                return
            bounds = [float('inf'), float('-inf'),
                      float('inf'), float('-inf'),
                      float('inf'), float('-inf')]
            actors.InitTraversal()
            actor = actors.GetNextActor()
            while actor:
                ab = actor.GetBounds()
                bounds[0] = min(bounds[0], ab[0]); bounds[1] = max(bounds[1], ab[1])
                bounds[2] = min(bounds[2], ab[2]); bounds[3] = max(bounds[3], ab[3])
                bounds[4] = min(bounds[4], ab[4]); bounds[5] = max(bounds[5], ab[5])
                actor = actors.GetNextActor()
            fx = (bounds[0] + bounds[1]) / 2
            fy = (bounds[2] + bounds[3]) / 2
            fz = (bounds[4] + bounds[5]) / 2
            diagonal = math.sqrt((bounds[1]-bounds[0])**2 + (bounds[3]-bounds[2])**2 + (bounds[5]-bounds[4])**2)
            distance = max(0.01, diagonal)
            preset_a = camera_presets["A"]
            az_rad = math.radians(preset_a["azimuth"])
            el_rad = math.radians(preset_a["elevation"])
            camera = renderer.GetActiveCamera()
            camera.ParallelProjectionOn()
            camera.SetFocalPoint(fx, fy, fz)
            camera.SetPosition(
                fx + distance * math.cos(el_rad) * math.cos(az_rad),
                fy + distance * math.cos(el_rad) * math.sin(az_rad),
                fz + distance * math.sin(el_rad),
            )
            camera.SetViewUp(0, 0, 1)
            camera.OrthogonalizeViewUp()
            camera.SetParallelScale(distance * 0.5)
            renderer.ResetCameraClippingRange()
            stl_viewer.safe_render()

        def open_camera_presets_dialog():
            """≡ボタン: カメラプリセット設定モーダルを開く"""
            dialog = QtWidgets.QDialog(main_window)
            dialog.setWindowTitle("Camera Presets")
            dialog.setModal(True)

            dlg_layout = QtWidgets.QVBoxLayout(dialog)
            dlg_layout.setSpacing(4)
            dlg_layout.setContentsMargins(10, 8, 10, 8)

            # ヘッダ行
            hdr = QtWidgets.QWidget()
            hdr_layout = QtWidgets.QHBoxLayout(hdr)
            hdr_layout.setContentsMargins(0, 0, 0, 0)
            hdr_layout.setSpacing(4)
            for _txt, _w in [("", 16), ("Name", 82), ("H", 74), ("V", 68),
                              ("Dist", 68), ("X", 68), ("Y", 68), ("Z", 68)]:
                _lbl = QtWidgets.QLabel(_txt)
                _lbl.setFixedWidth(_w)
                _lbl.setStyleSheet("color: #666; font-size: 10px;")
                _lbl.setAlignment(QtCore.Qt.AlignCenter)
                hdr_layout.addWidget(_lbl)
            hdr_layout.addStretch()
            dlg_layout.addWidget(hdr)

            def _make_dspin(lo, hi, step, val, w=68):
                sp = QtWidgets.QDoubleSpinBox()
                sp.setRange(lo, hi)
                sp.setSingleStep(step)
                sp.setDecimals(2)
                sp.setFixedWidth(w)
                sp.setValue(val)
                return sp

            _widgets = {}
            for cam_name in ["A", "B", "C", "D", "E"]:
                p = camera_presets[cam_name]
                row_w = QtWidgets.QWidget()
                row_layout = QtWidgets.QHBoxLayout(row_w)
                row_layout.setContentsMargins(0, 0, 0, 0)
                row_layout.setSpacing(4)

                lbl = QtWidgets.QLabel(cam_name)
                lbl.setFixedWidth(16)
                lbl.setStyleSheet("font-weight: bold;")
                row_layout.addWidget(lbl)

                name_edit = QtWidgets.QLineEdit()
                name_edit.setPlaceholderText("(optional)")
                name_edit.setFixedWidth(82)
                name_edit.setText(p["name"])
                row_layout.addWidget(name_edit)

                az_spin = QtWidgets.QSpinBox()
                az_spin.setRange(-180, 180)
                az_spin.setSingleStep(15)
                az_spin.setSuffix("°")
                az_spin.setFixedWidth(74)
                az_spin.setValue(p["azimuth"])
                row_layout.addWidget(az_spin)

                el_spin = QtWidgets.QSpinBox()
                el_spin.setRange(-90, 90)
                el_spin.setSingleStep(15)
                el_spin.setSuffix("°")
                el_spin.setFixedWidth(68)
                el_spin.setValue(p["elevation"])
                row_layout.addWidget(el_spin)

                dist_spin = _make_dspin(0.05, 20.0, 0.05, p.get("distance", 1.0))
                row_layout.addWidget(dist_spin)
                fx_spin = _make_dspin(-5.0, 5.0, 0.05, p.get("focal_x", 0.0))
                row_layout.addWidget(fx_spin)
                fy_spin = _make_dspin(-5.0, 5.0, 0.05, p.get("focal_y", 0.0))
                row_layout.addWidget(fy_spin)
                fz_spin = _make_dspin(-5.0, 5.0, 0.05, p.get("focal_z", 0.0))
                row_layout.addWidget(fz_spin)

                row_layout.addStretch()
                dlg_layout.addWidget(row_w)
                _widgets[cam_name] = {
                    "name_edit": name_edit, "az_spin": az_spin, "el_spin": el_spin,
                    "dist_spin": dist_spin, "fx_spin": fx_spin,
                    "fy_spin": fy_spin, "fz_spin": fz_spin,
                }

            def _on_preset_changed(cam_name):
                w = _widgets[cam_name]
                p = camera_presets[cam_name]
                p["name"]      = w["name_edit"].text()
                p["azimuth"]   = w["az_spin"].value()
                p["elevation"] = w["el_spin"].value()
                p["distance"]  = w["dist_spin"].value()
                p["focal_x"]   = w["fx_spin"].value()
                p["focal_y"]   = w["fy_spin"].value()
                p["focal_z"]   = w["fz_spin"].value()
                _save_camera_presets()
                if cam_id_to_name.get(cam_group.checkedId()) == cam_name:
                    apply_camera_preset(p)

            for cam_name, w in _widgets.items():
                _cn = cam_name
                w["name_edit"].textChanged.connect(lambda _, cn=_cn: _on_preset_changed(cn))
                w["az_spin"].valueChanged.connect(lambda _, cn=_cn: _on_preset_changed(cn))
                w["el_spin"].valueChanged.connect(lambda _, cn=_cn: _on_preset_changed(cn))
                w["dist_spin"].valueChanged.connect(lambda _, cn=_cn: _on_preset_changed(cn))
                w["fx_spin"].valueChanged.connect(lambda _, cn=_cn: _on_preset_changed(cn))
                w["fy_spin"].valueChanged.connect(lambda _, cn=_cn: _on_preset_changed(cn))
                w["fz_spin"].valueChanged.connect(lambda _, cn=_cn: _on_preset_changed(cn))

            close_btn = QtWidgets.QPushButton("Close")
            close_btn.clicked.connect(dialog.accept)
            dlg_layout.addWidget(close_btn)
            dialog.exec_()

        def _on_camera_fitted(focal_x, focal_y, focal_z, distance):
            """モデルロード後の fit-to-frame 値で全プリセットの focal/distance を更新"""
            for p in camera_presets.values():
                p["focal_x"]  = focal_x
                p["focal_y"]  = focal_y
                p["focal_z"]  = focal_z
                p["distance"] = distance
            _save_camera_presets()

        cam_group.idClicked.connect(on_camera_selected)
        cam_settings_btn.clicked.connect(open_camera_presets_dialog)
        stl_viewer.reset_button.clicked.connect(on_reframe)
        stl_viewer.camera_fitted.connect(_on_camera_fitted)

        # Zero/Homeボタンの接続（LME 3D + Valkey → MuJoCoStudio）
        def _send_current_pose_to_studio():
            lme_valkey.write_angles(joint_editor.get_angles_for_3d())

        def on_zero_pose():
            """モデルをZeroポーズに設定して Studio へ送る"""
            joint_editor._on_zero_button_clicked()
            _send_current_pose_to_studio()

        def on_home_pose():
            """モデルをHomeポーズに設定して Studio へ送る"""
            joint_editor._on_home_button_clicked()
            _send_current_pose_to_studio()

        stl_viewer.zero_button.clicked.connect(on_zero_pose)
        stl_viewer.home_button.clicked.connect(on_home_pose)
        stl_viewer.lr_swap_button.clicked.connect(joint_editor._on_lr_swap)

        def on_upper_body_home():
            joint_editor.apply_partial_body_home("upper")

        def on_lower_body_home():
            joint_editor.apply_partial_body_home("lower")

        stl_viewer.body_home_upper_action.triggered.connect(on_upper_body_home)
        stl_viewer.body_home_lower_action.triggered.connect(on_lower_body_home)

        # 2行目: ● Play: | |◀︎ | ▶︎. | ▶︎_ | ■ | Valkey
        btn_row = QtWidgets.QWidget()
        btn_layout = QtWidgets.QHBoxLayout(btn_row)
        btn_layout.setContentsMargins(0, 0, 0, 0)
        play_label = QtWidgets.QLabel("● Play:")
        play_label.setStyleSheet("color: black; font-weight: bold;")
        rewind_btn = QtWidgets.QPushButton("|◀︎")
        play_action_btn = QtWidgets.QPushButton("▶︎.")
        play_full_btn = QtWidgets.QPushButton("▶︎_")
        stop_btn = QtWidgets.QPushButton("■")
        btn_layout.addWidget(play_label)
        btn_layout.addWidget(rewind_btn)
        btn_layout.addWidget(play_action_btn)
        btn_layout.addWidget(play_full_btn)
        btn_layout.addWidget(stop_btn)
        btn_layout.addStretch()
        pb_layout.addWidget(btn_row)

        def on_valkey_toggled(checked):
            s = load_app_settings()
            s["valkey_enabled"] = checked
            save_app_settings(s)
            lme_valkey.update_config({
                "enabled":   checked,
                "host":      s.get("valkey_host",      VALKEY_DEFAULT_HOST),
                "port":      int(s.get("valkey_port",  VALKEY_DEFAULT_PORT)),
                "write_key": s.get("valkey_write_key", VALKEY_DEFAULT_WRITE_KEY),
                "read_key":  s.get("valkey_read_key",  VALKEY_DEFAULT_READ_KEY),
            })

        # Valkey は 3D ビュー右上オーバーレイの valkey_check で制御
        stl_viewer.valkey_toggled.connect(on_valkey_toggled)
        # 起動時に現在のチェック状態で初期化
        on_valkey_toggled(stl_viewer.valkey_check.isChecked())

        # --- Live: スライダー/3D で角度を動かしたときに Valkey へリアルタイム送信 ---
        # 3Dビュー(stl_viewer.live_check) と JointEditor(joint_editor.live_check) の
        # 2つのチェックボックスが存在し、on/off が連動する。
        # 送信は 30ms のシングルショットタイマーでスロットル（最新値のみ送る）。
        live_state = {
            "enabled": bool(stl_viewer.live_check.isChecked()
                            and stl_viewer.valkey_check.isChecked()),
            "pending": None,
            "syncing": False,  # 相互 sync 中の無限ループ防止
        }
        live_flush_timer = QtCore.QTimer()
        live_flush_timer.setSingleShot(True)
        live_flush_timer.setInterval(30)

        def _live_flush():
            angles = live_state["pending"]
            live_state["pending"] = None
            if angles is None:
                return
            if not live_state["enabled"]:
                return
            try:
                lme_valkey.write_angles(angles)
            except Exception as _e:
                print(f"[Live] write_angles failed: {_e}")

        live_flush_timer.timeout.connect(_live_flush)

        def live_schedule_write(angles):
            if not live_state["enabled"]:
                return
            live_state["pending"] = dict(angles) if angles else {}
            if not live_flush_timer.isActive():
                live_flush_timer.start()

        # JointEditor の Live 初期状態を stl_viewer に合わせる（起動時のみ）
        joint_editor.live_check.setEnabled(stl_viewer.valkey_check.isChecked())
        joint_editor.live_check.setChecked(stl_viewer.live_check.isChecked())

        def _live_send_now():
            """Live ON になった瞬間、その時点の関節値を1回 Valkey へ即時送信する。
            スロットルタイマーを迂回して即時 write。"""
            if not live_state["enabled"]:
                return
            try:
                angles = joint_editor.get_angles_for_3d()
                lme_valkey.write_angles(angles)
                print(f"[Live] Sent initial snapshot ({len(angles)} joints) on Live ON")
            except Exception as _e:
                print(f"[Live] initial snapshot send failed: {_e}")

        def _on_live_toggled_from_stl(v):
            if live_state["syncing"]:
                return
            live_state["syncing"] = True
            joint_editor.live_check.setChecked(bool(v))
            live_state["syncing"] = False
            live_state["enabled"] = bool(v) and stl_viewer.valkey_check.isChecked()
            if live_state["enabled"]:
                _live_send_now()

        def _on_live_toggled_from_editor(v):
            if live_state["syncing"]:
                return
            live_state["syncing"] = True
            stl_viewer.live_check.setChecked(bool(v))
            live_state["syncing"] = False
            live_state["enabled"] = bool(v) and stl_viewer.valkey_check.isChecked()
            if live_state["enabled"]:
                _live_send_now()

        stl_viewer.live_toggled.connect(_on_live_toggled_from_stl)
        joint_editor.live_toggled.connect(_on_live_toggled_from_editor)

        def _on_valkey_toggled_for_live_sync(v):
            # Valkey OFF になったら JointEditor 側の Live も強制 OFF + disabled
            joint_editor.live_check.setEnabled(bool(v))
            if not v and joint_editor.live_check.isChecked():
                # stl_viewer 側は stl_viewer._on_valkey_toggled_for_live で既にクリア
                live_state["syncing"] = True
                joint_editor.live_check.setChecked(False)
                live_state["syncing"] = False
            live_state["enabled"] = (
                stl_viewer.live_check.isChecked() and bool(v)
            )
        stl_viewer.valkey_toggled.connect(_on_valkey_toggled_for_live_sync)
        # 起動時反映
        _on_valkey_toggled_for_live_sync(stl_viewer.valkey_check.isChecked())

        # --- Step 設定の双方向同期 ---
        # JointEditor 側 (step_snapping_checkbox / step_snapping_input) を単一の
        # 真実とし、3D ビュー側 (stl_viewer.step_snap_check) は ON/OFF のミラー。
        # deg 値は STLViewerWidget._current_step_deg() が JointEditor から直接
        # 読み出すので同期不要。
        _step_sync_guard = {"active": False}

        def _sync_step_from_stl(checked):
            if _step_sync_guard["active"]:
                return
            _step_sync_guard["active"] = True
            joint_editor.step_snapping_checkbox.setChecked(bool(checked))
            _step_sync_guard["active"] = False

        def _sync_step_from_editor(checked):
            if _step_sync_guard["active"]:
                return
            _step_sync_guard["active"] = True
            stl_viewer.step_snap_check.setChecked(bool(checked))
            _step_sync_guard["active"] = False

        stl_viewer.step_snap_check.toggled.connect(_sync_step_from_stl)
        joint_editor.step_snapping_checkbox.toggled.connect(_sync_step_from_editor)
        # 起動時: JointEditor 側 (joint_sliders_step_snapping) を真実として揃える
        _sync_step_from_editor(joint_editor.step_snapping_checkbox.isChecked())

        # 3D ビュー「Step」チェックボックス右クリックで STEP 角度エディタを開く。
        # JointEditor の step_snapping_input を介した保存/反映と同じ経路。
        _step_value_dialog = {"ref": None}

        def _open_step_value_dialog(local_pos):
            existing = _step_value_dialog["ref"]
            if existing is not None:
                try:
                    existing.close()
                except Exception:
                    pass
            dlg = StepValueDialog(
                float(joint_editor.step_snapping_deg),
                parent=stl_viewer,
            )

            def _apply(v):
                # JointEditor 側の spin に流すことで、_on_step_snapping_value_changed が
                # 発火 → step_snapping_deg 更新と設定保存が行われる。
                joint_editor.step_snapping_input.setValue(float(v))

            dlg.value_changed.connect(_apply)
            _step_value_dialog["ref"] = dlg
            # チェックボックス上のクリック位置を画面座標に変換して表示
            global_pos = stl_viewer.step_snap_check.mapToGlobal(local_pos)
            dlg.adjustSize()
            dlg.move(global_pos)
            dlg.show()

        stl_viewer.step_snap_check.setContextMenuPolicy(
            QtCore.Qt.CustomContextMenu
        )
        stl_viewer.step_snap_check.customContextMenuRequested.connect(
            _open_step_value_dialog
        )

        right_layout.addWidget(playback_bar)

        # Initialize FPS from settings
        _initial_fps = get_default_hz_fps()
        playback_ctrl.fps = _initial_fps
        stl_viewer.fps_label.setText(f"FPS: {_initial_fps}")

        def on_motion_defaults_from_settings(hz_fps, _unused):
            """設定で Hz(FPS) を保存したときの追随。"""
            val = max(1, min(1000, int(hz_fps)))
            stl_viewer.fps_label.setText(f"FPS: {val}")
            playback_ctrl.fps = val
            if joint_editor.isVisible():
                joint_editor.refresh_pose_duration_from_frames()

        # --- モーションエディタ用コールバック ---
        def on_load_model():
            """統一モデルインポート: URDF/Xacro/SDF/MJCF を自動判定して読み込み"""
            try:
                print("[Motion] on_load_model called")

                if select_and_parse_model is None:
                    raise RuntimeError("LegacyMotionEditor_Importer.py could not be imported.")

                parsed = select_and_parse_model(main_window)
                if not parsed:
                    print("[Motion] Model load cancelled")
                    return

                model_path, _working_dir, model_data, model_type = parsed
                print(f"[Motion] Selected file: {model_path} (type: {model_type})")

                # 既存のロボットモデルを削除
                if motion_state['robot_model']:
                    print("[Motion] Removing existing robot model")
                    motion_state['robot_model'].remove_actors()

                # モデルタイプに応じてビルド
                if model_type == 'mjcf':
                    print("[Motion] Building robot model from MJCF...")
                    rm = build_robot_model_from_mjcf(model_path, model_data)
                    robot_name = model_data.get('model_name', '')
                else:
                    # urdf, sdf, xacro -> all use URDF builder
                    print(f"[Motion] Building robot model from {model_type.upper()}...")
                    rm = build_robot_model_from_urdf(model_path, model_data)
                    robot_name = model_data.get('robot_name', '')

                print(f"[Motion] Parsed: {len(rm.links)} links, {len(rm.joints)} joints")

                # URDF/SDF/Xacro 読み込み時: 正規化を提案
                if model_type != 'mjcf' and _RLB_OK:
                    reply = QtWidgets.QMessageBox.question(
                        main_window,
                        "Joint Name Normalization",
                        "Normalize joint names using RobotLabelBridge?\n\n"
                        "Renames joints to canonical form (e.g. l_shoulder_yp)\n"
                        "for Valkey / Meridim compatibility.",
                        QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
                        QtWidgets.QMessageBox.Yes,
                    )
                    if reply == QtWidgets.QMessageBox.Yes:
                        name_map = build_joint_name_map(rm)
                        count = apply_joint_name_map(graph, rm, name_map)
                        print(f"[Normalize] {len(name_map)} joints renamed, {count} occurrences updated")

                print("[Motion] Building VTK actors...")
                rm.build_vtk_actors(stl_viewer.renderer)
                print(f"[Motion] Actors built: {len(rm.link_actors)}")

                print("[Motion] Applying joint angles...")
                rm.apply_joint_angles(rm.get_default_angles())

                motion_state['robot_model'] = rm
                motion_state['urdf_path'] = model_path
                motion_state['model_type'] = model_type
                rm.model_type = model_type
                _s = load_app_settings()
                _s["last_model_path"] = model_path
                _s["last_model_type"] = model_type
                save_app_settings(_s)
                stl_viewer.set_robot_model(rm)
                joint_editor.build_from_robot(rm)

                # Update robot name in UI
                if robot_name:
                    graph.robot_name = robot_name
                    if hasattr(graph, 'name_input') and graph.name_input:
                        graph.name_input.setText(robot_name)
                    print(f"[Motion] Robot name set to: {robot_name}")

                print("[Motion] Resetting camera...")
                stl_viewer.reset_camera()

                print("[Motion] Rendering...")
                stl_viewer.safe_render()

                def delayed_render():
                    print("[Motion] Delayed force_render triggered")
                    stl_viewer.force_render()
                    stl_viewer.reset_camera()
                QtCore.QTimer.singleShot(300, delayed_render)

                print(f"[Motion] {model_type.upper()} loaded successfully: {model_path}")
            except Exception as e:
                print(f"[Motion] Error loading model: {e}")
                traceback.print_exc()
                QtWidgets.QMessageBox.critical(
                    main_window, "Model Load Error", str(e))

        def on_add_pose_node():
            """PoseNodeを追加"""
            push_undo()
            try:
                rm = motion_state['robot_model']
                pos = QtCore.QPointF(0, 0)
                selected_nodes = graph.selected_nodes()
                source_node = None
                if selected_nodes:
                    selected_node = selected_nodes[0]
                    source_node = selected_node
                    selected_pos = selected_node.pos()
                    if isinstance(selected_pos, (list, tuple)):
                        base_x, base_y = selected_pos[0], selected_pos[1]
                    elif hasattr(selected_pos, 'x') and hasattr(selected_pos, 'y'):
                        base_x, base_y = selected_pos.x(), selected_pos.y()
                    else:
                        base_x, base_y = 0, 0

                    view = getattr(selected_node, 'view', None)
                    node_width = getattr(view, '_width', 160)
                    node_height = getattr(view, '_height', 90)
                    pos = QtCore.QPointF(base_x + get_node_offset_x(), base_y + get_node_offset_y())

                new_node = graph.create_node(
                    'motion.nodes.PoseNode',
                    name=f'pose_{len([n for n in graph.all_nodes() if isinstance(n, PoseNode)])}',
                    pos=pos,
                    skip_auto_position=True
                )
                new_node.set_pos(pos.x(), pos.y())
                new_node.set_color(*NODE_COLOR_DEFAULT)  # ノードの色を設定
                if rm:
                    new_node.angles_deg = joint_editor.get_angles()
                    new_node.joint_easings = joint_editor.get_joint_easings()
                else:
                    new_node.angles_deg = {}
                    new_node.joint_easings = {}
                new_node.pose_name = new_node.name()

                # Auto-connect from selected node to new node
                if source_node is not None:
                    out_ports = source_node.output_ports()
                    in_ports = new_node.input_ports()
                    if out_ports and in_ports:
                        # Find first available (unconnected) output port
                        from_port = None
                        for port in out_ports:
                            if not port.connected_ports():
                                from_port = port
                                break
                        # For BaseLinkNode (start), only connect if no existing connections
                        # to prevent multiple connections causing playback issues
                        if isinstance(source_node, BaseLinkNode):
                            has_existing = any(p.connected_ports() for p in out_ports)
                            if has_existing:
                                from_port = None  # Skip auto-connect
                        # For other nodes, if all ports are connected, use the first one
                        elif from_port is None:
                            from_port = out_ports[0]

                        if from_port is not None:
                            to_port = in_ports[0]
                            try:
                                if hasattr(from_port, 'connect_to'):
                                    from_port.connect_to(to_port)
                                else:
                                    graph.connect_ports(from_port, to_port)
                            except Exception as conn_e:
                                print(f"[Motion] Auto-connect failed: {conn_e}")

                print(f"[Motion] Added PoseNode: {new_node.name()}")
            except Exception as e:
                print(f"[Motion] Error adding PoseNode: {e}")
                traceback.print_exc()

        def on_add_define_node():
            """DefineNode を追加（ダブルクリックで編集モーダル）"""
            push_undo()
            try:
                pos = QtCore.QPointF(0, 0)
                selected_nodes = graph.selected_nodes()
                if selected_nodes:
                    selected_node = selected_nodes[0]
                    selected_pos = selected_node.pos()
                    if isinstance(selected_pos, (list, tuple)):
                        base_x, base_y = selected_pos[0], selected_pos[1]
                    elif hasattr(selected_pos, 'x') and hasattr(selected_pos, 'y'):
                        base_x, base_y = selected_pos.x(), selected_pos.y()
                    else:
                        base_x, base_y = 0, 0
                    view = getattr(selected_node, 'view', None)
                    node_width = getattr(view, '_width', 160)
                    node_height = getattr(view, '_height', 90)
                    pos = QtCore.QPointF(base_x + get_node_offset_x(), base_y + get_node_offset_y())

                idx = len([n for n in graph.all_nodes() if isinstance(n, DefineNode)])
                new_node = graph.create_node(
                    'motion.nodes.DefineNode',
                    name=f'define_{idx}',
                    pos=pos,
                    skip_auto_position=True
                )
                new_node.set_pos(pos.x(), pos.y())
                new_node.set_color(*NODE_DEFINE_PANEL_BG_COLOR)
                print(f"[Motion] Added DefineNode: {new_node.name()}")
            except Exception as e:
                print(f"[Motion] Error adding DefineNode: {e}")
                traceback.print_exc()

        def on_add_wait_node():
            """WaitNode を追加"""
            push_undo()
            try:
                pos = QtCore.QPointF(0, 0)
                selected_nodes = graph.selected_nodes()
                if selected_nodes:
                    selected_node = selected_nodes[0]
                    selected_pos = selected_node.pos()
                    if isinstance(selected_pos, (list, tuple)):
                        base_x, base_y = selected_pos[0], selected_pos[1]
                    elif hasattr(selected_pos, 'x') and hasattr(selected_pos, 'y'):
                        base_x, base_y = selected_pos.x(), selected_pos.y()
                    else:
                        base_x, base_y = 0, 0
                    pos = QtCore.QPointF(base_x + get_node_offset_x(), base_y + get_node_offset_y())

                idx = len([n for n in graph.all_nodes() if isinstance(n, WaitNode)])
                new_node = graph.create_node(
                    'motion.nodes.WaitNode',
                    name=f'wait_{idx}',
                    pos=pos,
                    skip_auto_position=True
                )
                new_node.wait_name = f'wait_{idx}'
                new_node.set_name(new_node.wait_name)
                new_node.set_pos(pos.x(), pos.y())
                new_node.set_color(*NODE_WAIT_PANEL_BG_COLOR)
                # Pass fps so dialog and body text compute duration correctly
                fps = getattr(playback_ctrl, 'fps', 100) if playback_ctrl else 100
                new_node._display_fps = fps
                graph._wait_editor_fps = fps
                print(f"[Motion] Added WaitNode: {new_node.name()}")
            except Exception as e:
                print(f"[Motion] Error adding WaitNode: {e}")
                traceback.print_exc()

        def on_joint_angles_changed(angles):
            """Joint Editor値変更時。angles は active joint (slider を持つもの) の
            FK 空間角度。rm.apply_joint_angles が内部で passive を自動マージし 3D
            を更新する。Valkey/Live 送信は active のみ (実機は passive モーターなし)。
            """
            rm = motion_state['robot_model']
            if rm:
                rm.apply_joint_angles(angles)
                stl_viewer.safe_render()
            # Live ON なら 30ms スロットルで Valkey へ書き込む (active のみ)
            live_schedule_write(angles)

        def on_single_joint_changed(joint_name, angle_deg):
            """ダイアログからの単一ジョイント角度変更時"""
            if playback_ui_state.get("locked", False):
                return
            # angle_deg は FK 空間 (robot_model 基準)。UI 空間へ変換してからスライダーへ渡す。
            ui_angle = joint_editor.fk_to_ui_angles({joint_name: angle_deg})[joint_name]
            joint_editor._update_slider(joint_name, ui_angle)
            # 現在のノードに保存
            joint_editor._save_to_node()
            # Live ON なら現在の全関節を Valkey へ 30ms スロットルで送信
            live_schedule_write(joint_editor.get_angles_for_3d())

        def on_node_selection_changed():
            """ノード選択変更時"""
            if playback_ui_state.get("locked", False):
                return
            dbg("[NODE]", "on_node_selection_changed called",
                current_node=id(joint_editor.current_pose_node) if joint_editor.current_pose_node else None)

            # まず現在のノードに保存してから切り替え
            joint_editor._save_to_node()

            selected = graph.selected_nodes()
            dbg("[NODE]", f"Selected nodes count: {len(selected)}")

            for node in selected:
                node_type = type(node).__name__
                dbg("[NODE]", f"Checking node", node_type=node_type, node_id=id(node))
                if isinstance(node, PoseNode):
                    dbg("[NODE]", f"PoseNode selected",
                        node_id=id(node), name=node.pose_name,
                        angles_deg=node.angles_deg)
                    joint_editor.set_current_pose_node(node)
                    joint_editor.set_angles(node.angles_deg)
                    rm = motion_state['robot_model']
                    if rm:
                        # snap=True: 前ポーズからの瞬間移動なので閉ループ IK の
                        # warm-start (前回解) を破棄し、canonical zero から
                        # 解き直す (逆ブランチ跳躍による破綻を防ぐ)
                        rm.apply_joint_angles(
                            joint_editor.get_angles_for_3d(node.angles_deg),
                            snap=True)
                        stl_viewer.safe_render()
                    return
            dbg("[NODE]", "No PoseNode selected, setting current_pose_node to None")
            joint_editor.set_current_pose_node(None)

        def on_node_long_pressed(node):
            """ノード長押し: そのノードへ1コマ分補間再生 (Valkey にも送信)。"""
            if not isinstance(node, PoseNode):
                return
            rm = motion_state['robot_model']
            if not rm:
                return
            if playback_ui_state.get("locked", False):
                return
            playback_ui_state["locked"] = True
            playback_ui_state["restore_node"] = node  # 再生後も同じポーズを維持
            playback_restore_timer.stop()
            playback_ctrl.play_single_pose(node, graph, rm)

        graph.node_long_pressed.connect(on_node_long_pressed)

        playback_ui_state = {
            "locked": False,
            "restore_node": None,
        }
        playback_restore_timer = QtCore.QTimer()
        playback_restore_timer.setSingleShot(True)

        def _restore_pose_after_playback():
            playback_ui_state["locked"] = False
            node = playback_ui_state.get("restore_node")
            rm = motion_state['robot_model']
            if rm and isinstance(node, PoseNode):
                rm.apply_joint_angles(joint_editor.get_angles_for_3d(node.angles_deg))
                stl_viewer.safe_render()

        playback_restore_timer.timeout.connect(_restore_pose_after_playback)

        def _schedule_playback_restore():
            playback_ui_state["locked"] = True
            playback_restore_timer.start(500)

        def _set_play_label_playing(playing: bool):
            """● Play: ラベルを再生中は青、停止/待機は黒に切替。PAD ▶︎_ も同期。"""
            if playing:
                play_label.setStyleSheet("color: #2196F3; font-weight: bold;")
            else:
                play_label.setStyleSheet("color: black; font-weight: bold;")
            dlg = pad_dialog[0]
            if dlg is not None and hasattr(dlg, "set_play_active"):
                dlg.set_play_active(playing)

        def _find_start_node_in_graph():
            for node in graph.all_nodes():
                if isinstance(node, BaseLinkNode):
                    return node
            return None

        def _focus_start_node_in_graph():
            """現在のグラフの StartNode (BaseLinkNode) を選択・フォーカスする。
            起動復元・プロジェクトロード・Action 切替後に呼ぶことで
            Play ボタンが即座に機能するようにする。"""
            start_node = _find_start_node_in_graph()
            if start_node is None:
                return
            graph.clear_selection()
            start_node.set_selected(True)
            try:
                pos = start_node.pos()
                if isinstance(pos, (list, tuple)):
                    graph.viewer().centerOn(QtCore.QPointF(pos[0], pos[1]))
                elif hasattr(pos, 'x'):
                    graph.viewer().centerOn(pos)
            except Exception:
                pass

        def _find_selected_node():
            for node in graph.selected_nodes():
                if isinstance(node, (PoseNode, DefineNode, WaitNode, BranchingNode, JumpNode,
                                     BaseLinkNode, MixNode, CommandNode)):
                    return node
            return None

        def on_rewind():
            """|◀︎: Start を選択・フォーカス、ロボットをプリポジション（再生なし）"""
            try:
                start_node = _find_start_node_in_graph()
                if not start_node:
                    QtWidgets.QMessageBox.warning(
                        main_window, "Warning", "No StartNode found.")
                    return
                # Focus the start node in the graph view
                graph.clear_selection()
                start_node.set_selected(True)
                try:
                    pos = start_node.pos()
                    if isinstance(pos, (list, tuple)):
                        graph.viewer().centerOn(QtCore.QPointF(pos[0], pos[1]))
                    elif hasattr(pos, 'x'):
                        graph.viewer().centerOn(pos)
                except Exception:
                    pass
                on_node_highlight(start_node)
                # Pre-position robot to first reachable PoseNode from start
                rm = motion_state['robot_model']
                if not rm:
                    return
                connected = _sorted_output_connections(start_node)
                first_pose = None
                for n in connected:
                    if isinstance(n, PoseNode):
                        first_pose = n
                        break
                if first_pose is None and connected:
                    _cur = connected[0]
                    _seen = {id(start_node)}
                    for _ in range(20):
                        if id(_cur) in _seen:
                            break
                        _seen.add(id(_cur))
                        if isinstance(_cur, PoseNode):
                            first_pose = _cur
                            break
                        _nxt = _sorted_output_connections(_cur)
                        if not _nxt:
                            break
                        _cur = _nxt[0]
                if first_pose is not None:
                    angles = dict(first_pose.angles_deg)
                else:
                    angles = {jname: 0.0 for jname in rm.joints.keys()
                              if rm.joints[jname].joint_type != 'fixed'}
                fk_angles = joint_editor.get_angles_for_3d(angles)
                rm.apply_joint_angles(fk_angles)
                stl_viewer.safe_render()
                lme_valkey.write_angles(fk_angles)
            except Exception as e:
                print(f"[Motion] Rewind error: {e}")
                traceback.print_exc()

        def on_play_action_only():
            """▶︎.: 選択ノード（未選択時はStart）から再生、JumpNodeで停止"""
            try:
                cleanup_orphaned_connections(graph)
                rm = motion_state['robot_model']
                if not rm:
                    QtWidgets.QMessageBox.warning(
                        main_window, "Warning", "No URDF loaded.")
                    return
                start_node = _find_selected_node()
                selected_pose = start_node if isinstance(start_node, PoseNode) else None
                if not start_node:
                    start_node = _find_start_node_in_graph()
                if not start_node:
                    QtWidgets.QMessageBox.warning(
                        main_window, "Warning", "No StartNode found.")
                    return
                playback_ui_state["restore_node"] = selected_pose
                playback_ui_state["locked"] = True
                playback_restore_timer.stop()
                on_node_highlight(start_node)
                _set_play_label_playing(True)
                playback_ctrl.play_action_only(start_node, graph, rm)
            except Exception as e:
                playback_ui_state["locked"] = False
                playback_restore_timer.stop()
                _set_play_label_playing(False)
                print(f"[Motion] Play action only error: {e}")
                traceback.print_exc()

        # Boot Playback 用の内部状態:
        # ▶︎_ で Boot Action へ内部的に切替えたとき、combo 表示は元の Action の
        # ままにしておき、再生終了後に元の Action へ黙って戻す。
        _boot_playback_state = {"return_idx": None}

        def _restore_action_after_boot_playback():
            """▶︎_ で Boot に切替えて再生した場合、再生終了後に元の Action へ戻す。"""
            ret_idx = _boot_playback_state.get("return_idx")
            if ret_idx is None:
                return
            _boot_playback_state["return_idx"] = None
            try:
                # 現在（= Boot）の graph 状態をスナップショットして保存
                cur = motion_action_state.get("current", 0)
                items = motion_action_state.get("items", [])
                if 0 <= cur < len(items):
                    try:
                        items[cur]["data"] = capture_motion_action_snapshot()
                    except Exception as _snap_e:
                        print(f"[Motion] Snapshot Boot after playback failed: {_snap_e}")
                # 元の Action に戻す
                if 0 <= ret_idx < len(items):
                    motion_action_state["current"] = ret_idx
                    data = items[ret_idx].get("data")
                    if data is not None:
                        apply_motion_action_data(data)
                # combo は元々触っていないので index はそのまま (return_idx)。
                # 念のため一致させておく
                try:
                    action_combo.blockSignals(True)
                    action_combo.setCurrentIndex(ret_idx)
                    action_combo.blockSignals(False)
                    action_combo.setEnabled(True)
                except Exception:
                    pass
            except Exception as _e:
                print(f"[Motion] Restore action after Boot playback failed: {_e}")
                try:
                    action_combo.setEnabled(True)
                except Exception:
                    pass

        playback_ctrl.playback_finished.connect(_restore_action_after_boot_playback)

        def on_play_full():
            """▶︎_: Boot Action の Start から再生する。

            - motion_action_state.items のうち title に 'boot' を含む Action
              (大文字小文字無視) を Boot と識別。
            - 現在の Action が Boot でなければ、現状をスナップショットして
              内部的に Boot の内容へ切替える。ただし combo 表示は元のまま
              保ち、再生終了後 (_restore_action_after_boot_playback) で
              自動的に元の Action へ戻す。ユーザから見ると「開いているページは
              変わらないが、実行は Boot から」の体験になる。
            - Boot Action が見つからない場合は現在 Action の StartNode から再生。
            - 選択ノードは無視して常に StartNode 起点で再生。
            """
            try:
                cleanup_orphaned_connections(graph)
                rm = motion_state['robot_model']
                if not rm:
                    QtWidgets.QMessageBox.warning(
                        main_window, "Warning", "No URDF loaded.")
                    return

                # Boot Action を探す（title に 'boot' を含む、大文字小文字無視）
                items = motion_action_state.get("items", []) if motion_action_state else []
                boot_idx = next(
                    (i for i, it in enumerate(items)
                     if "boot" in (it.get("title") or "").lower()),
                    None,
                )
                cur_idx = motion_action_state.get("current", 0) if motion_action_state else 0

                # 現在の Action が Boot でなければ、内部的に Boot へ切替
                if boot_idx is not None and boot_idx != cur_idx:
                    try:
                        motion_action_state["items"][cur_idx]["data"] = (
                            capture_motion_action_snapshot()
                        )
                    except Exception as _e:
                        print(f"[Motion] Failed to snapshot current action: {_e}")
                    motion_action_state["current"] = boot_idx
                    entry = motion_action_state["items"][boot_idx]
                    data = entry.get("data")
                    if data is not None:
                        apply_motion_action_data(data)
                    # combo 表示は cur_idx のまま維持（自動ジャンプさせない）。
                    # 内部 state と combo 表示が乖離するので、再生終了まで combo
                    # 操作を封じる。playback_finished で復元＆再有効化する。
                    _boot_playback_state["return_idx"] = cur_idx
                    try:
                        action_combo.setEnabled(False)
                    except Exception:
                        pass

                start_node = _find_start_node_in_graph()
                if not start_node:
                    # 起動に失敗したので、Boot 切替を巻き戻しておく
                    if _boot_playback_state.get("return_idx") is not None:
                        _restore_action_after_boot_playback()
                    QtWidgets.QMessageBox.warning(
                        main_window, "Warning", "No StartNode found.")
                    return
                playback_ui_state["restore_node"] = None
                playback_ui_state["locked"] = True
                playback_restore_timer.stop()
                on_node_highlight(start_node)
                _set_play_label_playing(True)
                playback_ctrl.play(start_node, graph, rm)
            except Exception as e:
                playback_ui_state["locked"] = False
                playback_restore_timer.stop()
                _set_play_label_playing(False)
                # 例外時も Boot 切替を巻き戻す
                if _boot_playback_state.get("return_idx") is not None:
                    try:
                        _restore_action_after_boot_playback()
                    except Exception:
                        pass
                print(f"[Motion] Play full error: {e}")
                traceback.print_exc()

        def on_start_over():
            """最初から再生（0.5秒後に開始）"""
            try:
                # Clean up orphaned connections before playback
                cleanup_orphaned_connections(graph)

                rm = motion_state['robot_model']
                if not rm:
                    QtWidgets.QMessageBox.warning(
                        main_window, "Warning", "No URDF loaded.")
                    return
                # Find BaseLinkNode (StartNode)
                start_node = None
                for node in graph.all_nodes():
                    if isinstance(node, BaseLinkNode):
                        start_node = node
                        break
                if not start_node:
                    QtWidgets.QMessageBox.warning(
                        main_window, "Warning", "No StartNode found.")
                    return
                selected_pose = None
                for node in graph.selected_nodes():
                    if isinstance(node, PoseNode):
                        selected_pose = node
                        break
                playback_ui_state["restore_node"] = selected_pose
                playback_ui_state["locked"] = True
                playback_restore_timer.stop()

                # Clear previous incomplete markers
                clear_incomplete_markers()

                # Auto-cleanup: If Start has multiple connections, keep only the last one
                out_ports = start_node.output_ports()
                if out_ports:
                    all_connected = []
                    for port in out_ports:
                        for cp in port.connected_ports():
                            target = cp.node()
                            if isinstance(target, (PoseNode, DefineNode, WaitNode, BranchingNode, MixNode, CommandNode, JumpNode)):
                                all_connected.append((port, cp, target))

                    if len(all_connected) > 1:
                        print(f"[Start Over] Found {len(all_connected)} connections from start, cleaning up...")
                        # Keep only the last connection, remove others
                        for out_port, in_port, target in all_connected[:-1]:
                            print(f"[Start Over] Removing old connection to: {target.name()}")
                            try:
                                graph.disconnect_ports(out_port, in_port)
                            except Exception as disc_e:
                                print(f"[Start Over] Disconnect error: {disc_e}")

                # Find the connected node from start_node
                connected_nodes = _sorted_output_connections(start_node)
                print(f"[Start Over] Connected nodes from start: {[n.name() for n in connected_nodes]}")
                first_pose_node = None
                for node in connected_nodes:
                    if isinstance(node, PoseNode):
                        first_pose_node = node
                        break

                # If no direct PoseNode, traverse through intermediate nodes
                # (e.g. Start → BranchingNode → BranchingNode → PoseNode)
                if first_pose_node is None and connected_nodes:
                    _cur = connected_nodes[0]
                    _seen = {id(start_node)}
                    for _ in range(20):
                        if id(_cur) in _seen:
                            break
                        _seen.add(id(_cur))
                        if isinstance(_cur, PoseNode):
                            first_pose_node = _cur
                            print(f"[Start Over] Found PoseNode via traversal: {_cur.name()}")
                            break
                        _nxt = _sorted_output_connections(_cur)
                        if not _nxt:
                            break
                        _cur = _nxt[0]

                # Instantly jump 3D model to start position
                if first_pose_node is not None:
                    initial_angles = dict(first_pose_node.angles_deg)
                    print(f"[Start Over] Using angles from: {first_pose_node.name()}")
                else:
                    # Fallback: all angles = 0
                    initial_angles = {jname: 0.0 for jname in rm.joints.keys()
                                      if rm.joints[jname].joint_type != 'fixed'}
                    print("[Start Over] No reachable PoseNode found, using zero angles")
                rm.apply_joint_angles(joint_editor.get_angles_for_3d(initial_angles))
                stl_viewer.safe_render()
                # Pre-send start pose to Valkey so the physical robot moves there
                # during the 500ms delay, preventing an abrupt jump on playback start.
                lme_valkey.write_angles(joint_editor.get_angles_for_3d(initial_angles))

                # Highlight the start node immediately
                on_node_highlight(start_node)
                # Start playback after 0.5 second delay
                QtCore.QTimer.singleShot(500, lambda: playback_ctrl.play(start_node, graph, rm))
            except Exception as e:
                playback_ui_state["locked"] = False
                playback_restore_timer.stop()
                print(f"[Motion] Start Over error: {e}")
                traceback.print_exc()

        # Computed-motion (walking) ticks at up to LOOP_HZ (~100Hz) so Valkey/servo
        # writes stay precise, but the offscreen VTK->QLabel render (OffscreenRenderer)
        # is comparatively expensive per call — it reads the framebuffer back, reshapes
        # it in numpy, converts to QImage/QPixmap, and scales it. Calling IK
        # synchronously inside PlaybackController._tick() still occasionally makes a
        # tick take long enough that the control loop falls behind wall-clock time.
        # A same-thread QTimer for rendering is not enough: Qt's event loop is
        # single-threaded and the render still blocks whatever else wants to run.
        # Tried moving the render itself to a background thread with the GL context
        # handed over via MakeCurrent() under a lock — VTK's OpenGL2 backend does not
        # tolerate that (vtkOpenGLVertexArrayObject errors, corrupted rendering), so
        # reverted. Moving the *IK computation* to a background thread instead avoids
        # that: apply_joint_angles() only mutates plain vtkTransform/vtkMatrix4x4
        # objects (CPU-side data, no GL calls), so it's safe to call off the GUI
        # thread as long as it can't run at the same time as a render reading that
        # same data — guarded here by stl_viewer._vtk_lock, with the GL context never
        # leaving the GUI thread. Same split on macOS and Ubuntu.
        def on_playback_pose(angles):
            """再生中の姿勢更新（Poseベース再生。計算モーションは IK ワーカー側）。
            fk_angles は active joint のみ。rm.apply_joint_angles が passive を
            自動マージ。Valkey 送信は active のみ (on_joint_angles_changed と同方針)。"""
            rm = motion_state['robot_model']
            fk_angles = joint_editor.get_angles_for_3d(angles)
            if rm:
                with stl_viewer._vtk_lock:
                    rm.apply_joint_angles(fk_angles)
            lme_valkey.write_angles(fk_angles)

        def _render_during_playback():
            if playback_ctrl.is_playing:
                stl_viewer.safe_render()

        playback_render_qtimer = QtCore.QTimer()
        playback_render_qtimer.timeout.connect(_render_during_playback)
        playback_render_qtimer.start(50)  # ~20 Hz visual refresh

        def _computed_ik_worker():
            """Runs computed-motion (walking) IK on its own precise loop,
            independent of the GUI thread, so it isn't slowed down by
            rendering or any other GUI-thread work. Applies angles directly
            to the model + Valkey. PlaybackController._tick() skips its own
            computed-motion handling while this worker is active so the two
            paths don't fight over the same state."""
            from LegacyMotionEditor_Utils import PAD_REGISTER_VALUES
            while True:
                active = (
                    playback_ctrl.is_playing
                    and playback_ctrl._computed_motion_active
                    and playback_ctrl._ik_executing
                    and playback_ctrl.motion_runtime is not None
                    and playback_ctrl._computed_func_names
                )
                if not active:
                    time.sleep(0.02)
                    continue

                t0 = playback_ctrl.elapsed_timer.elapsed()
                project_code = getattr(playback_ctrl.graph, "project_code", "") or ""
                try:
                    loop_hz = float(
                        (getattr(playback_ctrl.motion_runtime, "_ns", None) or {})
                        .get("LOOP_HZ", 100) or 100)
                except Exception:
                    loop_hz = 100.0

                angles = None
                for fn in list(playback_ctrl._computed_func_names):
                    if playback_ctrl.motion_runtime.call_function(
                            fn, project_code, PAD_REGISTER_VALUES):
                        angles = playback_ctrl.motion_runtime.get_angles_dict()

                if angles:
                    rm = motion_state['robot_model']
                    with stl_viewer._vtk_lock:
                        playback_ctrl.next_angles.update(angles)
                        playback_ctrl.prev_angles.update(angles)
                        limited = playback_ctrl._apply_joint_speed_limits(
                            angles, 1.0 / loop_hz)
                        fk_limited = joint_editor.get_angles_for_3d(limited)
                        if rm:
                            rm.apply_joint_angles(fk_limited)
                    lme_valkey.write_angles(fk_limited)

                elapsed_ms = playback_ctrl.elapsed_timer.elapsed() - t0
                remaining_sec = max(0.0, (1000.0 / loop_hz - elapsed_ms) / 1000.0)
                time.sleep(remaining_sec)

        threading.Thread(target=_computed_ik_worker, daemon=True).start()

        def on_joint_drag_ended():
            """3Dビューでの関節ドラッグ終了時"""
            if playback_ui_state.get("locked", False):
                return
            dbg("[TRIGGER]", "on_joint_drag_ended signal received")
            rm = motion_state['robot_model']
            if rm:
                # robot_model の角度は FK 空間 (URDF 軸基準)。
                # joint_editor / node は UI 空間 (Rev 関節は符号が逆) なので変換が必要。
                fk_angles = rm.get_current_angles()
                dbg("[TRIGGER]", f"Updating JointEditor from robot_model", angles=fk_angles)
                push_undo()  # save before-state before committing 3D drag
                joint_editor.set_angles(joint_editor.fk_to_ui_angles(fk_angles))
                # ノードに保存
                joint_editor._save_to_node()
                # Live ON のときだけ Valkey へ送信（Live OFF なら書き込まない）
                if live_state["enabled"]:
                    lme_valkey.write_angles(fk_angles)
            else:
                dbg("[TRIGGER]", "robot_model is None, skip")

        def on_wheel_angles_updated(fk_angles):
            """3Dビューのマウスホイール中: ノッチ毎の途中反映。
            drag と違いホイールは離散イベントなので、Undo/保存は停止検出後
            (joint_drag_ended) に集約し、ここでは UI スライダー同期と Live 送信のみ行う。"""
            if playback_ui_state.get("locked", False):
                return
            if not fk_angles:
                return
            # JointEditor スライダーを silent 反映（angles_changed は発火しない）
            joint_editor.set_angles(joint_editor.fk_to_ui_angles(fk_angles))
            # Live ON なら 30ms スロットルで Valkey へ書き込み
            live_schedule_write(fk_angles)

        def on_stop():
            was_active = (
                playback_ctrl.is_playing or
                playback_ctrl.is_paused or
                playback_ui_state.get("locked", False)
            )
            playback_ctrl.stop()
            _set_play_label_playing(False)
            if was_active:
                _schedule_playback_restore()

        # --- シグナル接続 ---
        buttons["Import Model"].clicked.connect(on_load_model)
        buttons["Add Pose"].clicked.connect(on_add_pose_node)
        def _on_add_sub_action(action):
            text = action.text()
            if text == "Wait":
                on_add_wait_node()
            elif text == "Define":
                on_add_define_node()
            elif text == "Command":
                on_add_command()
            elif text == "Mix":
                on_add_mix()
            elif text == "Memo":
                on_add_memo()
        buttons["Add Define"].menu().triggered.connect(_on_add_sub_action)

        def on_add_branching():
            """BranchingNode を追加（ダブルクリックで BranchingShellDialog）"""
            push_undo()
            try:
                pos = QtCore.QPointF(0, 0)
                selected_nodes = graph.selected_nodes()
                if selected_nodes:
                    selected_node = selected_nodes[0]
                    selected_pos = selected_node.pos()
                    if isinstance(selected_pos, (list, tuple)):
                        base_x, base_y = selected_pos[0], selected_pos[1]
                    elif hasattr(selected_pos, "x") and hasattr(
                        selected_pos, "y"
                    ):
                        base_x, base_y = selected_pos.x(), selected_pos.y()
                    else:
                        base_x, base_y = 0, 0
                    view = getattr(selected_node, "view", None)
                    node_width = getattr(view, "_width", 160)
                    node_height = getattr(view, "_height", 90)
                    pos = QtCore.QPointF(
                        base_x + get_node_offset_x(), base_y + get_node_offset_y()
                    )

                idx = len(
                    [n for n in graph.all_nodes() if isinstance(n, BranchingNode)]
                )
                new_node = graph.create_node(
                    "motion.nodes.BranchingNode",
                    name=f"branch_{idx}",
                    pos=pos,
                    skip_auto_position=True,
                )
                new_node.set_pos(pos.x(), pos.y())
                new_node.set_color(*NODE_BRANCH_PANEL_BG_COLOR)
                new_node.branching_enabled = True
                new_node.enable_branching_output()
                new_node._sync_branching_port_labels()
                print(f"[Motion] Added BranchingNode: {new_node.name()}")
            except Exception as e:
                print(f"[Motion] Error adding BranchingNode: {e}")
                traceback.print_exc()

        buttons["Add Branching"].clicked.connect(on_add_branching)

        def on_add_command():
            """CommandNode を追加（ダブルクリックで編集）"""
            push_undo()
            try:
                pos = QtCore.QPointF(0, 0)
                selected_nodes = graph.selected_nodes()
                if selected_nodes:
                    selected_node = selected_nodes[0]
                    selected_pos = selected_node.pos()
                    if isinstance(selected_pos, (list, tuple)):
                        base_x, base_y = selected_pos[0], selected_pos[1]
                    elif hasattr(selected_pos, "x") and hasattr(selected_pos, "y"):
                        base_x, base_y = selected_pos.x(), selected_pos.y()
                    else:
                        base_x, base_y = 0, 0
                    view = getattr(selected_node, "view", None)
                    node_width = getattr(view, "_width", 160)
                    node_height = getattr(view, "_height", 90)
                    pos = QtCore.QPointF(
                        base_x + get_node_offset_x(), base_y + get_node_offset_y()
                    )

                idx = len([n for n in graph.all_nodes() if isinstance(n, CommandNode)])
                new_node = graph.create_node(
                    "motion.nodes.CommandNode",
                    name=f"cmd_{idx}",
                    pos=pos,
                    skip_auto_position=True,
                )
                new_node.set_pos(pos.x(), pos.y())
                new_node.command_name = f"cmd_{idx}"
                new_node.set_name(f"cmd_{idx}")
                print(f"[Motion] Added CommandNode: {new_node.name()}")
            except Exception as e:
                print(f"[Motion] Error adding CommandNode: {e}")
                traceback.print_exc()

        def on_add_mix():
            """MixNode を追加（ダブルクリックで編集）"""
            push_undo()
            try:
                pos = QtCore.QPointF(0, 0)
                selected_nodes = graph.selected_nodes()
                if selected_nodes:
                    selected_node = selected_nodes[0]
                    selected_pos = selected_node.pos()
                    if isinstance(selected_pos, (list, tuple)):
                        base_x, base_y = selected_pos[0], selected_pos[1]
                    elif hasattr(selected_pos, "x") and hasattr(selected_pos, "y"):
                        base_x, base_y = selected_pos.x(), selected_pos.y()
                    else:
                        base_x, base_y = 0, 0
                    view = getattr(selected_node, "view", None)
                    node_width = getattr(view, "_width", 160)
                    node_height = getattr(view, "_height", 90)
                    pos = QtCore.QPointF(
                        base_x + get_node_offset_x(), base_y + get_node_offset_y()
                    )

                idx = len([n for n in graph.all_nodes() if isinstance(n, MixNode)])
                new_node = graph.create_node(
                    "motion.nodes.MixNode",
                    name=f"mix_{idx}",
                    pos=pos,
                    skip_auto_position=True,
                )
                new_node.set_pos(pos.x(), pos.y())
                new_node.mix_name = f"mix_{idx}"
                new_node.set_name(f"mix_{idx}")
                print(f"[Motion] Added MixNode: {new_node.name()}")
            except Exception as e:
                print(f"[Motion] Error adding MixNode: {e}")
                traceback.print_exc()

        def on_add_memo():
            """MemoNode を追加。純粋な注釈ノードで、Export では無視される。"""
            push_undo()
            try:
                pos = QtCore.QPointF(0, 0)
                selected_nodes = graph.selected_nodes()
                if selected_nodes:
                    selected_node = selected_nodes[0]
                    selected_pos = selected_node.pos()
                    if isinstance(selected_pos, (list, tuple)):
                        base_x, base_y = selected_pos[0], selected_pos[1]
                    elif hasattr(selected_pos, "x") and hasattr(selected_pos, "y"):
                        base_x, base_y = selected_pos.x(), selected_pos.y()
                    else:
                        base_x, base_y = 0, 0
                    pos = QtCore.QPointF(
                        base_x + get_node_offset_x(), base_y + get_node_offset_y()
                    )
                idx = len([n for n in graph.all_nodes() if isinstance(n, MemoNode)])
                new_node = graph.create_node(
                    "motion.nodes.MemoNode",
                    name=f"memo_{idx}",
                    pos=pos,
                    skip_auto_position=True,
                )
                new_node.set_pos(pos.x(), pos.y())
                new_node.set_name("Memo")
                new_node.refresh_body_text()
                print(f"[Motion] Added MemoNode: {new_node.name()}")
            except Exception as e:
                print(f"[Motion] Error adding MemoNode: {e}")
                traceback.print_exc()

        def on_add_jump():
            """JumpNode を追加（ダブルクリックでジャンプ先アクション設定）"""
            push_undo()
            try:
                pos = QtCore.QPointF(0, 0)
                selected_nodes = graph.selected_nodes()
                if selected_nodes:
                    selected_node = selected_nodes[0]
                    selected_pos = selected_node.pos()
                    if isinstance(selected_pos, (list, tuple)):
                        base_x, base_y = selected_pos[0], selected_pos[1]
                    elif hasattr(selected_pos, "x") and hasattr(
                        selected_pos, "y"
                    ):
                        base_x, base_y = selected_pos.x(), selected_pos.y()
                    else:
                        base_x, base_y = 0, 0
                    view = getattr(selected_node, "view", None)
                    node_width = getattr(view, "_width", 160)
                    node_height = getattr(view, "_height", 90)
                    pos = QtCore.QPointF(
                        base_x + get_node_offset_x(), base_y + get_node_offset_y()
                    )

                idx = len(
                    [n for n in graph.all_nodes() if isinstance(n, JumpNode)]
                )
                new_node = graph.create_node(
                    "motion.nodes.JumpNode",
                    name=f"jump_{idx}",
                    pos=pos,
                    skip_auto_position=True,
                )
                new_node.set_pos(pos.x(), pos.y())
                new_node.set_name("Jump to")
                new_node.jump_target_action_index = min(
                    int(motion_action_state["current"]),
                    max(0, len(motion_action_state["items"]) - 1),
                )
                new_node.refresh_body_text()
                QtCore.QTimer.singleShot(15, new_node._apply_jump_node_colors)
                print(f"[Motion] Added JumpNode -> Action_{new_node.jump_target_action_index + 1}")
            except Exception as e:
                print(f"[Motion] Error adding JumpNode: {e}")
                traceback.print_exc()

        buttons["Add Jump"].clicked.connect(on_add_jump)
        joint_editor.angles_changed.connect(on_joint_angles_changed)
        stl_viewer.joint_drag_ended.connect(on_joint_drag_ended)
        stl_viewer.wheel_angles_updated.connect(on_wheel_angles_updated)
        stl_viewer.joint_angle_changed.connect(on_single_joint_changed)
        rewind_btn.clicked.connect(on_rewind)
        play_action_btn.clicked.connect(on_play_action_only)
        play_full_btn.clicked.connect(on_play_full)
        stop_btn.clicked.connect(on_stop)
        _pad_playback_handlers[0] = (on_play_full, on_stop, on_home_pose, on_zero_pose)
        _wire_pad_playback_buttons()
        playback_ctrl.pose_changed.connect(on_playback_pose)
        playback_ctrl.playback_finished.connect(_schedule_playback_restore)
        playback_ctrl.playback_finished.connect(lambda: _set_play_label_playing(False))

        # Playback node highlight control
        _highlighted_node = [None]      # logical node currently playing (real or virtual)
        _highlighted_real_view = [None]  # actual view object showing the highlight border

        def _find_graph_node_for_virtual(virtual_node):
            """Return the real graph node whose name matches the given virtual node, or None."""
            if not isinstance(virtual_node, (
                VirtualBaseLinkNode, VirtualPoseNode, VirtualDefineNode, VirtualWaitNode,
                VirtualBranchingNode, VirtualJumpNode, VirtualMixNode, VirtualCommandNode
            )):
                return None
            v_name = virtual_node.name()
            for real_node in graph.all_nodes():
                if callable(getattr(real_node, 'name', None)) and real_node.name() == v_name:
                    return real_node
            return None

        def clear_playback_highlight():
            """Clear playback highlight from all nodes (keeps incomplete markers)"""
            if _highlighted_real_view[0] is not None:
                _highlighted_real_view[0]._is_playing = False
                _highlighted_real_view[0].update()
                _highlighted_real_view[0] = None
            for node in graph.all_nodes():
                view = getattr(node, 'view', None)
                if view and getattr(view, '_is_playing', False):
                    view._is_playing = False
                    view.update()
            _highlighted_node[0] = None

        def clear_incomplete_markers():
            """Clear incomplete markers from all nodes (called on Start Over)"""
            for node in graph.all_nodes():
                view = getattr(node, 'view', None)
                if view:
                    if getattr(view, '_is_incomplete', False):
                        view._is_incomplete = False
                        view.update()

        def on_node_highlight(node):
            """Highlight the currently playing node (real or virtual)."""
            # Clear previous UI highlight
            if _highlighted_real_view[0] is not None:
                _highlighted_real_view[0]._is_playing = False
                _highlighted_real_view[0].update()
                _highlighted_real_view[0] = None

            _highlighted_node[0] = node

            if node is None:
                return

            # Find the view: real node has .view; virtual node needs name-based lookup.
            # Only look up a real node when the virtual node belongs to the currently
            # displayed action — otherwise cross-action virtual nodes (e.g. routing hub)
            # incorrectly match same-named nodes (branch_0, Jump to …) in the displayed graph.
            view = getattr(node, 'view', None)
            if view is None:
                virtual_action = getattr(playback_ctrl, '_virtual_action_idx', None)
                current_action = motion_action_state.get("current", 0)
                if virtual_action is None or virtual_action == current_action:
                    real_node = _find_graph_node_for_virtual(node)
                    view = getattr(real_node, 'view', None) if real_node else None

            if view is not None:
                view._is_playing = True
                view.update()
                _highlighted_real_view[0] = view

        def on_node_incomplete(node):
            """Mark node as incomplete (didn't reach target angles)"""
            if node is not None:
                view = getattr(node, 'view', None)
                if view:
                    view._is_incomplete = True
                    view.update()

        playback_ctrl.node_highlight.connect(on_node_highlight)
        playback_ctrl.node_incomplete.connect(on_node_incomplete)
        playback_ctrl.playback_finished.connect(clear_playback_highlight)

        def motion_action_combo_label(i, title):
            t = (title or "").strip()
            if t:
                return f"Action_{i + 1}:{t}"
            return f"Action_{i + 1}:"

        def capture_motion_action_snapshot():
            return copy.deepcopy(
                build_motion_data_dict(
                    motion_state["urdf_path"],
                    motion_state["robot_model"],
                    graph,
                    playback_ctrl,
                    joint_editor,
                    include_user_value=True,
                )
            )

        # ---- Undo / Redo system ----
        _undo_limit = load_app_settings().get("undo_limit", 100)
        undo_stack = LMEUndoStack(max_size=_undo_limit)
        _undo_guard = [False]

        def _capture_undo_snapshot():
            """Full snapshot: all motion-action data + current action index + node positions."""
            items_snap = []
            for entry in motion_action_state["items"]:
                items_snap.append(copy.deepcopy(entry))
            return {
                "items": items_snap,
                "current": motion_action_state["current"],
            }

        def push_undo():
            """Push a before-state snapshot onto the undo stack (guard against double-push)."""
            if _undo_guard[0]:
                return
            _undo_guard[0] = True
            snap = _capture_undo_snapshot()
            undo_stack.push(snap)
            QtCore.QTimer.singleShot(0, lambda: _undo_guard.__setitem__(0, False))

        def push_undo_param():
            """Push undo for slider/spin parameter edits.
            Synchronous guard reset avoids a timer dead-zone that would swallow
            a rapid subsequent node operation into the same undo step."""
            if _undo_guard[0]:
                return
            _undo_guard[0] = True
            snap = _capture_undo_snapshot()
            undo_stack.push(snap)
            _undo_guard[0] = False  # synchronous – no dead-zone between ops

        def _restore_undo_snapshot(snap):
            """Restore a full snapshot produced by _capture_undo_snapshot."""
            if snap is None:
                return
            _undo_guard[0] = True
            try:
                motion_action_state["items"] = copy.deepcopy(snap["items"])
                target_idx = max(0, min(snap["current"], len(motion_action_state["items"]) - 1))
                motion_action_state["current"] = target_idx
                refresh_motion_action_combo(select_index=target_idx)
                if motion_action_state["items"]:
                    apply_motion_action_data(motion_action_state["items"][target_idx]["data"])
            finally:
                # Timer-based reset: post-restore queued signals (node reconnect etc.)
                # fire before the guard clears, preventing spurious undo entries.
                QtCore.QTimer.singleShot(0, lambda: _undo_guard.__setitem__(0, False))

        def perform_undo():
            if not undo_stack.can_undo():
                return
            current_snap = _capture_undo_snapshot()
            prev_snap = undo_stack.do_undo(current_snap)
            _restore_undo_snapshot(prev_snap)

        def perform_redo():
            if not undo_stack.can_redo():
                return
            current_snap = _capture_undo_snapshot()
            next_snap = undo_stack.do_redo(current_snap)
            _restore_undo_snapshot(next_snap)

        # Wire undo hooks into subsystems
        joint_editor._undo_push_fn = push_undo_param  # slider/spin: synchronous guard
        graph._undo_push_fn = push_undo
        viewer = graph.viewer()
        viewer._undo_push_fn = push_undo

        def _push_undo_raw(snap):
            """Raw push for NodeGraphQt viewer — guards against double-push from restore."""
            if _undo_guard[0]:
                return
            undo_stack.push(snap)
            _undo_guard[0] = True
            QtCore.QTimer.singleShot(0, lambda: _undo_guard.__setitem__(0, False))

        viewer._undo_push_raw_fn = _push_undo_raw
        viewer._capture_snap_fn = _capture_undo_snapshot

        # Keyboard shortcuts (main window — fallback when main window has focus)
        undo_shortcut = QtWidgets.QShortcut(QtGui.QKeySequence("Ctrl+Z"), main_window)
        undo_shortcut.activated.connect(perform_undo)
        redo_shortcut_y = QtWidgets.QShortcut(QtGui.QKeySequence("Ctrl+Y"), main_window)
        redo_shortcut_y.activated.connect(perform_redo)
        redo_shortcut_shift_z = QtWidgets.QShortcut(QtGui.QKeySequence("Ctrl+Shift+Z"), main_window)
        redo_shortcut_shift_z.activated.connect(perform_redo)

        # Application-level event filter — fires Undo/Redo from any window
        # (Config dialog, Joint Sliders, modals, etc.)
        # Text-editing widgets (QLineEdit etc.) are skipped so their native Ctrl+Z works,
        # EXCEPT when they are inside the Joint Sliders panel — there, LME Undo takes priority.
        _TEXT_WIDGET_TYPES = (
            QtWidgets.QLineEdit, QtWidgets.QTextEdit,
            QtWidgets.QPlainTextEdit, QtWidgets.QAbstractSpinBox,
        )

        class _UndoAppFilter(QtCore.QObject):
            def eventFilter(self, obj, event):
                if event.type() != QtCore.QEvent.KeyPress:
                    return False
                mods = event.modifiers()
                key  = event.key()
                if not primary_mod_held(mods):
                    return False
                fw = QtWidgets.QApplication.focusWidget()
                if isinstance(fw, _TEXT_WIDGET_TYPES):
                    # Inside Joint Sliders: override Ctrl+Z / Cmd+Z → LME Undo
                    if fw is None or not joint_editor.isAncestorOf(fw):
                        return False  # elsewhere: let the text widget handle it
                shift = bool(mods & QtCore.Qt.ShiftModifier)
                if key == QtCore.Qt.Key_Z and not shift:
                    perform_undo()
                    return True
                if key == QtCore.Qt.Key_Y or (key == QtCore.Qt.Key_Z and shift):
                    perform_redo()
                    return True
                return False

        _undo_app_filter = _UndoAppFilter(main_window)
        QtWidgets.QApplication.instance().installEventFilter(_undo_app_filter)
        # ---- End Undo / Redo system ----

        # ---- Blender-style camera view shortcuts (global 0-9) ----
        # 3D ビュー内でクリックしていなくても 0-9 でカメラビュー切替できるように
        # アプリレベルでキー入力を捕まえて stl_viewer.  _handle_camera_view_key に流す。
        # テキスト編集中の widget (QLineEdit / QSpinBox 等) にフォーカスがある場合は
        # 数字入力を優先し、こちらは何もしない。
        _CAMERA_DIGIT_KEYS = (
            QtCore.Qt.Key_0, QtCore.Qt.Key_1, QtCore.Qt.Key_2, QtCore.Qt.Key_3,
            QtCore.Qt.Key_4, QtCore.Qt.Key_5, QtCore.Qt.Key_6, QtCore.Qt.Key_7,
            QtCore.Qt.Key_8, QtCore.Qt.Key_9,
        )

        class _CameraKeyAppFilter(QtCore.QObject):
            def eventFilter(self, obj, event):
                if event.type() != QtCore.QEvent.KeyPress:
                    return False
                if event.key() not in _CAMERA_DIGIT_KEYS:
                    return False
                mods = event.modifiers()
                # 修飾キー付きは触らない（Ctrl+1 等のショートカット/入力に譲る）
                if bool(mods & (QtCore.Qt.ControlModifier | QtCore.Qt.MetaModifier |
                                QtCore.Qt.AltModifier | QtCore.Qt.ShiftModifier)):
                    return False
                # テキスト編集中は数字入力を優先
                fw = QtWidgets.QApplication.focusWidget()
                if isinstance(fw, _TEXT_WIDGET_TYPES):
                    return False
                # STL ビューアが未初期化なら何もしない
                if 'stl_viewer' not in globals() or stl_viewer is None:
                    return False
                try:
                    if stl_viewer._handle_camera_view_key(event):
                        return True
                except Exception as _e:
                    print(f"[Camera] app filter error: {_e}")
                return False

        _camera_key_filter = _CameraKeyAppFilter(main_window)
        QtWidgets.QApplication.instance().installEventFilter(_camera_key_filter)

        def make_empty_motion_action_data():
            rm = motion_state["robot_model"]
            return {
                "version": 1,
                "urdf_path": motion_state.get("urdf_path") or "",
                "joint_order": list(rm.joint_order) if rm else [],
                "joint_settings": joint_editor.get_joint_settings(),
                "joint_layout": joint_editor.get_joint_layout(),
                "joint_group_presets": joint_editor.get_joint_group_presets(),
                "current_group_preset_index": (
                    joint_editor.current_group_preset_index
                ),
                "motion_formulas": {},
                "nodes": [],
                "edges": [],
                "playback": {
                    "start_node_id": "",
                    "interpolation": playback_ctrl.interpolation,
                    "fps": playback_ctrl.fps,
                    "branch_mode": playback_ctrl.branch_mode,
                },
                "user_value_session": default_user_value_session(),
            }

        def sync_playback_bar_widgets():
            stl_viewer.fps_label.setText(f"FPS: {max(1, min(1000, int(playback_ctrl.fps)))}")

        def refresh_motion_action_combo(select_index=None):
            sel = (
                motion_action_state["current"]
                if select_index is None
                else select_index
            )
            action_combo.blockSignals(True)
            action_combo.clear()
            for i, entry in enumerate(motion_action_state["items"]):
                action_combo.addItem(
                    motion_action_combo_label(i, entry.get("title", "")),
                    i,
                )
            if 0 <= sel < action_combo.count():
                action_combo.setCurrentIndex(sel)
            action_combo.blockSignals(False)
            for _jn in graph.all_nodes():
                if type(_jn).__name__ == "JumpNode" and hasattr(_jn, "refresh_body_text"):
                    _jn.refresh_body_text()

        def apply_motion_action_data(data):
            _rm, _up, ok = load_motion_data(
                copy.deepcopy(data),
                graph,
                stl_viewer,
                joint_editor,
                playback_ctrl,
                motion_state=motion_state,
                skip_urdf=True,
                parent_window=main_window,
            )
            if not ok:
                return
            rm = motion_state.get("robot_model")
            if rm:
                stl_viewer.set_robot_model(rm)
            sync_playback_bar_widgets()

        def playback_jump_callback(target_action_idx, save_current=True):
            """Callback for PlaybackController to jump to another action.

            Uses virtual nodes to continue playback without switching the graph UI.

            Args:
                target_action_idx: Index of action to jump to
                save_current: If True, save current action state before switching

            Returns:
                VirtualBaseLinkNode (StartNode) of target action, or None if failed
            """
            items = motion_action_state.get("items", [])
            if target_action_idx < 0 or target_action_idx >= len(items):
                print(f"[JumpCallback] Invalid target action index: {target_action_idx}")
                return None

            current_idx = motion_action_state.get("current", 0)

            # Save current action state if requested (but don't switch UI)
            if save_current and current_idx != target_action_idx:
                items[current_idx]["data"] = capture_motion_action_snapshot()
                print(f"[JumpCallback] Saved current action {current_idx}")

            # Get target action data
            target_entry = items[target_action_idx]
            target_data = target_entry.get("data")
            if target_data is None:
                # Jump to current action (same index) or data not yet saved — capture live
                if target_action_idx == current_idx:
                    target_data = capture_motion_action_snapshot()
                    items[target_action_idx]["data"] = copy.deepcopy(target_data)
                    print(f"[JumpCallback] Captured live snapshot for self-jump to Action_{target_action_idx + 1}")
                else:
                    print(f"[JumpCallback] Target action {target_action_idx} has no data")
                    return None

            # Build virtual graph from action data (no UI change)
            virtual_start = build_virtual_graph_from_action_data(target_data)
            if virtual_start:
                print(f"[JumpCallback] Built virtual graph for Action_{target_action_idx + 1}")
                # Store which action we're virtually playing (for nested jumps)
                playback_ctrl._virtual_action_idx = target_action_idx
                return virtual_start

            print(f"[JumpCallback] Failed to build virtual graph for action {target_action_idx}")
            return None

        # Connect jump callback to playback controller
        playback_ctrl.set_jump_callback(playback_jump_callback)

        def on_motion_action_combo_changed(new_index):
            if new_index < 0:
                return
            old = motion_action_state["current"]
            if new_index == old:
                return
            motion_action_state["items"][old]["data"] = (
                capture_motion_action_snapshot()
            )
            motion_action_state["current"] = new_index
            entry = motion_action_state["items"][new_index]
            data = entry.get("data")
            if data is None:
                return
            apply_motion_action_data(data)
            # Re-highlight current playing node in the newly loaded graph
            if playback_ctrl.is_playing and _highlighted_node[0] is not None:
                on_node_highlight(_highlighted_node[0])
            else:
                # 再生中でなければ start を自動選択
                QtCore.QTimer.singleShot(100, _focus_start_node_in_graph)

        def on_motion_action_rename():
            i = motion_action_state["current"]
            dlg = QtWidgets.QDialog(main_window)
            dlg.setWindowTitle("Rename Action")
            v = QtWidgets.QVBoxLayout(dlg)
            v.addWidget(QtWidgets.QLabel(motion_action_combo_label(
                i, motion_action_state["items"][i].get("title", ""),
            )))
            ed = QtWidgets.QLineEdit(
                motion_action_state["items"][i].get("title", "")
            )
            v.addWidget(ed)
            bb = QtWidgets.QDialogButtonBox(
                QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel
            )
            bb.accepted.connect(dlg.accept)
            bb.rejected.connect(dlg.reject)
            v.addWidget(bb)
            if dlg.exec() != QtWidgets.QDialog.Accepted:
                return
            motion_action_state["items"][i]["title"] = ed.text().strip()
            refresh_motion_action_combo(i)

        def on_motion_action_add():
            push_undo()
            i_old = motion_action_state["current"]
            motion_action_state["items"][i_old]["data"] = (
                capture_motion_action_snapshot()
            )
            n = len(motion_action_state["items"]) + 1
            dlg = QtWidgets.QDialog(main_window)
            dlg.setWindowTitle("Add Action")
            v = QtWidgets.QVBoxLayout(dlg)
            v.addWidget(QtWidgets.QLabel(f"Action_{n}:"))
            ed = QtWidgets.QLineEdit()
            ed.setPlaceholderText("Title (e.g. Walk1)")
            v.addWidget(ed)
            bb = QtWidgets.QDialogButtonBox(
                QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel
            )
            bb.accepted.connect(dlg.accept)
            bb.rejected.connect(dlg.reject)
            v.addWidget(bb)
            if dlg.exec() != QtWidgets.QDialog.Accepted:
                return
            title = ed.text().strip()
            new_data = make_empty_motion_action_data()
            motion_action_state["items"].append(
                {"title": title, "data": new_data}
            )
            motion_action_state["current"] = len(motion_action_state["items"]) - 1
            refresh_motion_action_combo(motion_action_state["current"])
            apply_motion_action_data(new_data)

        def on_motion_action_duplicate():
            push_undo()
            i_old = motion_action_state["current"]
            motion_action_state["items"][i_old]["data"] = (
                capture_motion_action_snapshot()
            )
            snap = copy.deepcopy(motion_action_state["items"][i_old]["data"])
            title_base = motion_action_state["items"][i_old].get("title") or ""
            new_title = (f"{title_base} copy").strip() or "copy"
            motion_action_state["items"].append(
                {"title": new_title, "data": snap}
            )
            motion_action_state["current"] = len(motion_action_state["items"]) - 1
            refresh_motion_action_combo(motion_action_state["current"])
            apply_motion_action_data(snap)

        def on_motion_action_delete():
            push_undo()
            if len(motion_action_state["items"]) <= 1:
                QtWidgets.QMessageBox.warning(
                    main_window,
                    "Actions",
                    "At least one action is required.",
                )
                return
            i = motion_action_state["current"]
            motion_action_state["items"].pop(i)
            new_i = min(i, len(motion_action_state["items"]) - 1)
            motion_action_state["current"] = new_i
            refresh_motion_action_combo(new_i)
            entry = motion_action_state["items"][new_i]
            data = entry.get("data")
            if data is not None:
                apply_motion_action_data(data)

        def on_motion_action_reorder():
            """Actionリストをドラッグ＆ドロップで並び替えるダイアログ"""
            items = motion_action_state["items"]
            if len(items) <= 1:
                return

            # 現在のグラフ状態を保存してから並び替え
            i_cur = motion_action_state["current"]
            items[i_cur]["data"] = capture_motion_action_snapshot()

            dlg = QtWidgets.QDialog(main_window)
            dlg.setWindowTitle("Reorder Actions")
            dlg.setMinimumWidth(340)
            dlg.setMinimumHeight(300)
            v = QtWidgets.QVBoxLayout(dlg)
            v.addWidget(QtWidgets.QLabel("Drag to reorder, then click OK to apply:"))

            lw = QtWidgets.QListWidget()
            lw.setDragDropMode(QtWidgets.QAbstractItemView.DragDropMode.InternalMove)
            lw.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.SingleSelection)
            lw.setAlternatingRowColors(True)
            lw.setSpacing(2)
            lw.setStyleSheet(
                "QListWidget { background: #ffffff; color: #141414; alternate-background-color: #e8e8e8; }"
                "QListWidget::item { color: #141414; }"
                "QListWidget::item:selected { background: #2a82da; color: #ffffff; }"
            )

            for i, entry in enumerate(items):
                label = motion_action_combo_label(i, entry.get("title", ""))
                lw_item = QtWidgets.QListWidgetItem(f"≡  {label}")
                lw_item.setData(QtCore.Qt.ItemDataRole.UserRole, i)
                lw.addItem(lw_item)

            v.addWidget(lw)

            bb = QtWidgets.QDialogButtonBox(
                QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel
            )
            bb.accepted.connect(dlg.accept)
            bb.rejected.connect(dlg.reject)
            v.addWidget(bb)

            if dlg.exec() != QtWidgets.QDialog.Accepted:
                return

            # 新しい順序（各行の元のインデックスを取得）
            new_order = [
                lw.item(row).data(QtCore.Qt.ItemDataRole.UserRole)
                for row in range(lw.count())
            ]

            # 変化がなければ終了
            if new_order == list(range(len(items))):
                return

            push_undo()

            # old_index → new_index のマッピング
            old_to_new = {orig: new_pos for new_pos, orig in enumerate(new_order)}

            # 現在開いているActionが新順序で何番目か
            new_current = old_to_new[i_cur]

            # items を並び替え
            motion_action_state["items"] = [items[orig] for orig in new_order]
            motion_action_state["current"] = new_current

            # 全 Action の保存済みデータ内の JumpNode インデックスを更新
            for entry in motion_action_state["items"]:
                data = entry.get("data")
                if not isinstance(data, dict):
                    continue
                for nd in data.get("nodes", []):
                    if nd.get("node_type") == "jump" and "jump_target_action_index" in nd:
                        old_idx = nd["jump_target_action_index"]
                        nd["jump_target_action_index"] = old_to_new.get(old_idx, old_idx)

            # 現在グラフ上にある JumpNode も更新
            for node in graph.all_nodes():
                if type(node).__name__ == "JumpNode" and hasattr(node, "jump_target_action_index"):
                    old_idx = node.jump_target_action_index
                    node.jump_target_action_index = old_to_new.get(old_idx, old_idx)
                    if hasattr(node, "refresh_body_text"):
                        node.refresh_body_text()

            refresh_motion_action_combo(new_current)

        def on_save_project():
            """プロジェクトXML保存"""
            try:
                # Clean up orphaned connections before saving
                cleanup_orphaned_connections(graph)

                # Generate default filename: LME_ModelName_YYYYMMDD_HHMM.xml
                from datetime import datetime
                now = datetime.now()
                date_str = now.strftime("%Y%m%d_%H%M")
                model_name = "Unknown"
                rm = motion_state.get('robot_model')
                if rm and hasattr(rm, 'robot_name') and rm.robot_name:
                    model_name = rm.robot_name
                elif motion_state.get('urdf_path'):
                    # Extract name from URDF path
                    base = os.path.basename(motion_state['urdf_path'])
                    model_name = os.path.splitext(base)[0]
                # Sanitize model name (remove invalid characters)
                model_name = "".join(c for c in model_name if c.isalnum() or c in ('_', '-'))
                default_filename = f"LME_{model_name}_{date_str}.xml"

                # ベースディレクトリ: 前回保存パス > URDF dir > cwd
                _sv = load_app_settings()
                _last_proj = _sv.get("last_project_path", "")
                if _last_proj and os.path.isdir(os.path.dirname(_last_proj)):
                    base_dir = os.path.dirname(_last_proj)
                elif motion_state.get('urdf_path'):
                    base_dir = os.path.dirname(motion_state['urdf_path'])
                else:
                    base_dir = os.getcwd()
                default_path = os.path.join(base_dir, default_filename)

                fp, _ = QtWidgets.QFileDialog.getSaveFileName(
                    main_window, "Save Project", default_path, "XML Files (*.xml)")
                if fp:
                    if not fp.lower().endswith('.xml'):
                        fp += '.xml'
                    # Collect view settings from stl_viewer
                    current_view_settings = {
                        "bg_color_a": list(stl_viewer.bg_color_a),
                        "bg_color_b": list(stl_viewer.bg_color_b),
                        "bg_gradient_type": stl_viewer.bg_gradient_type,
                        "bg_slider_value": stl_viewer.bg_slider_value,
                        "light_slider_value": stl_viewer.light_slider_value,
                        "camera_presets": {k: dict(v) for k, v in camera_presets.items()},
                    }
                    # Get home position from joint_editor
                    current_home_position = getattr(joint_editor, 'home_position_angles', {})
                    ok = save_project_xml(
                        fp, motion_state['urdf_path'],
                        motion_state['robot_model'], graph, playback_ctrl, joint_editor,
                        motion_action_state=motion_action_state,
                        capture_current_func=capture_motion_action_snapshot,
                        model_type=motion_state.get('model_type', ''),
                        view_settings=current_view_settings,
                        home_position=current_home_position)
                    if ok:
                        # 保存パスを記録（次回 Save のデフォルトディレクトリに使用）
                        _sv2 = load_app_settings()
                        _sv2["last_project_path"] = fp
                        save_app_settings(_sv2)
                        action_count = len(motion_action_state.get("items", []))
                        QtWidgets.QMessageBox.information(
                            main_window, "Saved", f"Project saved to:\n{fp}\n({action_count} actions)")
            except Exception as e:
                print(f"[Project] Save error: {e}")
                traceback.print_exc()
                QtWidgets.QMessageBox.critical(
                    main_window, "Save Error", str(e))

        def on_load_project():
            """プロジェクトXML読込"""
            try:
                fp, _ = QtWidgets.QFileDialog.getOpenFileName(
                    main_window, "Load Project", "", "XML Files (*.xml);;JSON Files (*.json)")
                if not fp:
                    return
                # XML or JSON based on extension
                action_count = 1
                loaded_view_settings = {}
                loaded_home_position = {}
                if fp.lower().endswith('.xml'):
                    result = load_project_xml(
                        fp, graph, stl_viewer, joint_editor, playback_ctrl,
                        motion_state=motion_state,
                        parent_window=main_window,
                        motion_action_state=motion_action_state,
                    )
                    if len(result) == 6:
                        rm, urdf_path, ok, action_count, loaded_view_settings, loaded_home_position = result
                    elif len(result) == 5:
                        rm, urdf_path, ok, action_count, loaded_view_settings = result
                    elif len(result) == 4:
                        rm, urdf_path, ok, action_count = result
                    else:
                        rm, urdf_path, ok = result
                else:
                    # Fallback to JSON for backwards compatibility
                    rm, urdf_path, ok = load_motion_json(
                        fp, graph, stl_viewer, joint_editor, playback_ctrl,
                        parent_window=main_window,
                    )
                if not ok:
                    return
                if rm is not None:
                    motion_state['robot_model'] = rm
                    motion_state['urdf_path'] = urdf_path
                    stl_viewer.set_robot_model(rm)
                elif urdf_path:
                    motion_state['urdf_path'] = urdf_path
                # Refresh action combo to show loaded actions
                refresh_motion_action_combo(motion_action_state["current"])
                sync_playback_bar_widgets()
                # Apply loaded view settings
                if loaded_view_settings:
                    if "bg_color_a" in loaded_view_settings and "bg_color_b" in loaded_view_settings:
                        stl_viewer.set_bg_colors(
                            loaded_view_settings["bg_color_a"],
                            loaded_view_settings["bg_color_b"]
                        )
                    if "bg_gradient_type" in loaded_view_settings:
                        stl_viewer.set_bg_gradient_type(loaded_view_settings["bg_gradient_type"])
                    if "bg_slider_value" in loaded_view_settings:
                        stl_viewer.set_bg_slider_value(loaded_view_settings["bg_slider_value"])
                    if "light_slider_value" in loaded_view_settings:
                        stl_viewer.set_light_slider_value(loaded_view_settings["light_slider_value"])
                # Apply loaded home position
                if loaded_home_position:
                    joint_editor.set_home_position(loaded_home_position)
                # Apply loaded camera presets
                if loaded_view_settings and "camera_presets" in loaded_view_settings:
                    for k, v in loaded_view_settings["camera_presets"].items():
                        if k in camera_presets:
                            camera_presets[k].update(v)
                    _save_camera_presets()
                # ノードの色適用 (singleShot 20ms) が完了した後に start を選択
                QtCore.QTimer.singleShot(200, _focus_start_node_in_graph)
                # Clean up any orphaned connections after loading
                removed = cleanup_orphaned_connections(graph)
                cleanup_msg = f"\n(Cleaned up {removed} orphaned connections)" if removed > 0 else ""
                _s = load_app_settings()
                _s["last_xml_path"] = fp
                _s["last_project_path"] = fp
                save_app_settings(_s)
                QtWidgets.QMessageBox.information(
                    main_window, "Loaded", f"Project loaded from:\n{fp}\n({action_count} actions){cleanup_msg}")
            except Exception as e:
                print(f"[Project] Load error: {e}")
                traceback.print_exc()
                QtWidgets.QMessageBox.critical(
                    main_window, "Load Error", str(e))

        buttons["Save Project"].clicked.connect(on_save_project)
        buttons["Load Project"].clicked.connect(on_load_project)


        # ---- Export Cartridge: write a PhysicalOn-compatible Logic Cartridge -----
        class _BootBasePicker(QtWidgets.QDialog):
            """Dialog for selecting which actions are Boot and Base."""
            def __init__(self, items, parent=None):
                super().__init__(parent)
                self.setWindowTitle("Boot / Base Action Selection")
                titles = [it.get("title") or f"Action_{i + 1}" for i, it in enumerate(items)]

                self._boot_combo = QtWidgets.QComboBox()
                self._base_combo = QtWidgets.QComboBox()
                for combo in (self._boot_combo, self._base_combo):
                    combo.setStyleSheet(_MAIN_WINDOW_COMBO_TEXT_STYLE)
                    combo.addItem("(none)", -1)
                    for i, t in enumerate(titles):
                        combo.addItem(t, i)

                # Pre-select based on name matching.
                boot_default = next(
                    (i for i, t in enumerate(titles) if "boot" in t.lower()), -1
                )
                base_default = next(
                    (i for i, t in enumerate(titles) if "base" in t.lower()), -1
                )
                self._boot_combo.setCurrentIndex(boot_default + 1)
                self._base_combo.setCurrentIndex(base_default + 1)

                form = QtWidgets.QFormLayout()
                form.addRow("Boot action  (plays once at startup):", self._boot_combo)
                form.addRow("Base action  (idle loop / branching origin):", self._base_combo)

                note = QtWidgets.QLabel(
                    "A jump to Base is appended automatically at the end of Boot.\n"
                    "A self-loop (JUMP 0) is appended automatically at the end of Base.\n"
                    "Use CMP nodes inside Base to branch on button input."
                )
                note.setStyleSheet("color: gray; font-size: 11px;")

                btn_box = QtWidgets.QDialogButtonBox(
                    QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
                btn_box.accepted.connect(self._on_accept)
                btn_box.rejected.connect(self.reject)

                layout = QtWidgets.QVBoxLayout(self)
                layout.addLayout(form)
                layout.addWidget(note)
                layout.addWidget(btn_box)
                self.setMinimumWidth(480)

            def _on_accept(self):
                b = self._boot_combo.currentData()
                a = self._base_combo.currentData()
                boot = b if b is not None and b >= 0 else None
                base = a if a is not None and a >= 0 else None
                if boot is not None and base is not None and boot == base:
                    QtWidgets.QMessageBox.warning(
                        self, "Invalid Selection",
                        "Boot and Base cannot be the same action.\n"
                        "Please select a different action for each."
                    )
                    return
                self.accept()

            def boot_idx(self):
                d = self._boot_combo.currentData()
                return d if d is not None and d >= 0 else None

            def base_idx(self):
                d = self._base_combo.currentData()
                return d if d is not None and d >= 0 else None

        def on_export_cartridge():
            """Export motion_action_state to a Logic Cartridge (.py) file."""
            try:
                from LegacyMotionEditor_Utils import export_cartridge

                cur_idx = motion_action_state.get("current", 0)
                items = motion_action_state.get("items", [])
                if not items:
                    QtWidgets.QMessageBox.warning(
                        main_window, "Export Cartridge",
                        "No actions to export.")
                    return

                # Always refresh the live action snapshot before export.
                items[cur_idx]["data"] = capture_motion_action_snapshot()

                stale = [
                    f"Action_{i + 1}"
                    for i, it in enumerate(items)
                    if i != cur_idx and it.get("data") is None
                ]
                if stale:
                    ret = QtWidgets.QMessageBox.warning(
                        main_window, "Export Cartridge",
                        "These actions have never been saved and will export empty:\n"
                        + ", ".join(stale)
                        + "\n\nSwitch to each action once before export, or continue anyway?",
                        QtWidgets.QMessageBox.Ok | QtWidgets.QMessageBox.Cancel,
                    )
                    if ret != QtWidgets.QMessageBox.Ok:
                        return

                # CommandNode / MixNode は V1 カートリッジエクスポータで未対応。
                # そのままエクスポートするとこれらのノードは "スキップ" され、
                # LME 上の再生挙動と実機出力が食い違うことになる。
                # ダイアログで明確に警告し、ユーザーに続行/中断を選ばせる。
                unsupported: list[tuple[str, str, str]] = []  # (action_title, node_type, node_label)
                counts = {"command": 0, "mix": 0}
                for i, it in enumerate(items):
                    data = it.get("data") or {}
                    for node in data.get("nodes", []):
                        nt = node.get("node_type")
                        if nt in ("command", "mix"):
                            title = it.get("title") or f"Action_{i + 1}"
                            label = node.get("name", node.get("id", ""))
                            unsupported.append((title, nt, label))
                            counts[nt] += 1
                if unsupported:
                    preview_lines = [
                        f"  • [{t}]  {nt.capitalize()}Node: {lbl}"
                        for (t, nt, lbl) in unsupported[:10]
                    ]
                    preview = "\n".join(preview_lines)
                    more = (f"\n  ... (+{len(unsupported) - 10} more)"
                            if len(unsupported) > 10 else "")
                    header = "⚠ 未対応ノードが含まれています / Unsupported nodes present\n"
                    summary_line = (
                        f"CommandNode: {counts['command']} 個   "
                        f"MixNode: {counts['mix']} 個"
                    )
                    body = (
                        f"{header}\n"
                        f"{summary_line}\n\n"
                        "V1 Logic Cartridge exporter は CommandNode / MixNode を\n"
                        "まだサポートしていません。これらのノードはスキップされ、\n"
                        "書き出された Cartridge の動作は LME 上の再生と食い違い\n"
                        "ます (機能が抜けた状態で出力されます)。\n\n"
                        "The V1 exporter does not support CommandNode / MixNode.\n"
                        "They will be skipped in the output, so the Cartridge\n"
                        "behavior will differ from LME playback.\n\n"
                        "続行しますか？ / Continue anyway?\n\n"
                        f"{preview}{more}"
                    )
                    # critical アイコン + 明示的な "Skip & Export" / "Cancel" ボタン
                    box = QtWidgets.QMessageBox(main_window)
                    box.setIcon(QtWidgets.QMessageBox.Critical)
                    box.setWindowTitle("Unsupported Node Types (V1)")
                    box.setText(body)
                    _btn_skip = box.addButton(
                        "Skip and export anyway",
                        QtWidgets.QMessageBox.AcceptRole,
                    )
                    _btn_cancel = box.addButton(
                        "Cancel export", QtWidgets.QMessageBox.RejectRole
                    )
                    box.setDefaultButton(_btn_cancel)
                    box.exec()
                    if box.clickedButton() is not _btn_skip:
                        return

                # Boot / Base selection dialog.
                picker = _BootBasePicker(items, parent=main_window)
                if picker.exec() != QtWidgets.QDialog.Accepted:
                    return
                boot_action_idx = picker.boot_idx()
                base_action_idx = picker.base_idx()

                from datetime import datetime as _dt
                model_name = "robot"
                rm = motion_state.get("robot_model")
                if rm and hasattr(rm, "robot_name") and rm.robot_name:
                    model_name = rm.robot_name
                elif motion_state.get("urdf_path"):
                    base = os.path.basename(motion_state["urdf_path"])
                    model_name = os.path.splitext(base)[0]
                model_name = "".join(c for c in model_name if c.isalnum() or c in ("_", "-")) or "robot"

                # Determine default save directory (last used → URDF dir → cwd).
                _ex_settings = load_app_settings()
                _last_export_dir = _ex_settings.get("last_cartridge_export_dir", "")
                if _last_export_dir and os.path.isdir(_last_export_dir):
                    base_dir = _last_export_dir
                elif motion_state.get("urdf_path"):
                    base_dir = os.path.dirname(motion_state["urdf_path"])
                else:
                    base_dir = os.getcwd()
                default_path = os.path.join(base_dir, f"Logic_cartridge_{model_name}.py")

                fp, _sel = QtWidgets.QFileDialog.getSaveFileName(
                    main_window, "Export Logic Cartridge",
                    default_path, "Python Files (*.py)")
                if not fp:
                    return
                if not fp.lower().endswith(".py"):
                    fp += ".py"

                # Remember the save directory for next time.
                _sv = load_app_settings()
                _sv["last_cartridge_export_dir"] = os.path.dirname(os.path.abspath(fp))
                save_app_settings(_sv)

                loop_hz = int(getattr(playback_ctrl, "fps", 100) or 100)
                _rm = motion_state.get("robot_model")
                _jo = getattr(_rm, "joint_order", None) if _rm else None
                joints_param = tuple(_jo) if _jo else None
                # MJCF-loaded models expose the set of joints backed by an
                # <actuator>. Passing it lets export_cartridge drop passive
                # linkage joints (parallelogram followers, coincident-anchored
                # ankles, etc.) so PhysicalOn's _resolve_player_joint_map does
                # not fail with "expected one actuator, found 0" and disable
                # the whole player. URDF loads leave this set empty → no-op.
                _aj = getattr(_rm, "actuator_joints", None) if _rm else None
                actuator_joints_param = set(_aj) if _aj else None
                # Servo per-joint speed presets — embedded as JOINT_MAX_SPEED_DEG_S
                # in the cartridge so MotionPlayer.advance clamps at the same
                # rate LME playback does. Without this, physics servo cannot
                # follow short-duration POS commanded at impossible rates.
                _je = getattr(graph, "joint_editor", None)
                joint_settings_param = (
                    dict(_je.joint_settings)
                    if _je and getattr(_je, "joint_settings", None)
                    else None
                )

                result = export_cartridge(
                    motion_action_state,
                    fp,
                    robot_name=model_name,
                    source_project=motion_state.get("urdf_path", ""),
                    loop_hz=loop_hz,
                    boot_action_idx=boot_action_idx,
                    base_action_idx=base_action_idx,
                    project_code=getattr(graph, "project_code", "") or "",
                    joints=joints_param,
                    joint_settings=joint_settings_param,
                    actuator_joints=actuator_joints_param,
                )

                # Report.
                lc = ", ".join(str(n) for n in result.line_counts)
                boot_label = (
                    f"#{result.boot_action_idx + 1}" if result.boot_action_idx is not None else "(none)"
                )
                base_label = (
                    f"#{result.base_action_idx + 1}" if result.base_action_idx is not None else "(none)"
                )
                summary = (
                    f"Exported to:\n{result.save_path}\n\n"
                    f"Actions: {result.action_count}   Lines: [{lc}]\n"
                    f"Boot: {boot_label}   Base: {base_label}"
                )
                if result.warnings:
                    preview = "\n".join(
                        f"  • action {w.action_index} {w.node_id}: {w.message}"
                        for w in result.warnings[:15]
                    )
                    more = (
                        f"\n  ... (+{len(result.warnings) - 15} more)"
                        if len(result.warnings) > 15 else ""
                    )
                    summary += f"\n\nWarnings ({len(result.warnings)}):\n{preview}{more}"
                QtWidgets.QMessageBox.information(main_window, "Cartridge Exported", summary)
                print(f"[ExportCartridge] Saved {result.action_count} actions to {result.save_path}")

            except Exception as e:
                print(f"[ExportCartridge] Error: {e}")
                traceback.print_exc()
                QtWidgets.QMessageBox.critical(
                    main_window, "Export Cartridge Error", str(e))

        buttons["Export Cartridge"].clicked.connect(on_export_cartridge)

        action_combo.currentIndexChanged.connect(on_motion_action_combo_changed)
        action_rename_btn.clicked.connect(on_motion_action_rename)
        action_add_btn.clicked.connect(on_motion_action_add)
        action_dup_btn.clicked.connect(on_motion_action_duplicate)
        action_reorder_btn.clicked.connect(on_motion_action_reorder)
        action_del_btn.clicked.connect(on_motion_action_delete)
        refresh_motion_action_combo()

        # ノード選択変更の検出（ポーリング方式）
        _prev_selected = [set()]
        def _check_selection():
            cur = set(id(n) for n in graph.selected_nodes())
            if cur != _prev_selected[0]:
                _prev_selected[0] = cur
                on_node_selection_changed()
        sel_timer = QtCore.QTimer()
        sel_timer.timeout.connect(_check_selection)
        sel_timer.start(200)

        # 左コンテンツにleft_panelとgraph.widgetを追加
        graph.widget.setMinimumWidth(0)
        main_layout.addWidget(left_panel)
        main_layout.addWidget(graph.widget, 1)

        # スプリッターの設定（left_section と right_panel）
        splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        splitter.addWidget(left_section)
        splitter.addWidget(right_panel)
        splitter.setSizes([LEFT_PANEL_WIDTH + SPLITTER_NODE_GRAPH_WIDTH, SPLITTER_3DVIEW_WIDTH])
        # 各パネルの折りたたみ設定
        splitter.setCollapsible(0, True)
        splitter.setCollapsible(1, True)

        # メインレイアウトにスプリッターを追加
        outer_layout.addWidget(splitter)

        # メインウィンドウにセントラルウィジェットを設定
        main_window.setCentralWidget(central_widget)

        # グラフに名前入力フィールドを関連付け
        graph.name_input = name_input

        # ウィンドウを表示（位置は起動時に左上へ設定済み）
        main_window.show()
        main_window.raise_()
        main_window.activateWindow()

        # 起動時にStartノードを選択
        graph.clear_selection()
        base_node.set_selected(True)

        # macOSでPythonアプリをフォアグラウンドに持ってくる
        try:
            if sys.platform == 'darwin':
                import subprocess
                subprocess.Popen([
                    'osascript', '-e',
                    'tell application "System Events" to set frontmost '
                    'of the first process whose unix id is '
                    + str(os.getpid()) + ' to true'
                ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except Exception:
            pass

        print("Application started. Double-click on a node to open the inspector.")
        print("Click 'Add Node' button to add new nodes.")
        print("Select a node and click 'Delete Node' to remove it.")
        print("Use 'Save' and 'Load' buttons to save and load your project.")
        print("Use 'Import URDF' to load a robot model for motion editing.")
        print("Press Ctrl+C in the terminal to close all windows and exit.")

        # Session auto-save: captures complete state to save/_lme_session.xml on exit.
        def _do_save_session():
            try:
                cleanup_orphaned_connections(graph)
                current_view_settings = {
                    "bg_color_a": list(stl_viewer.bg_color_a),
                    "bg_color_b": list(stl_viewer.bg_color_b),
                    "bg_gradient_type": stl_viewer.bg_gradient_type,
                    "bg_slider_value": stl_viewer.bg_slider_value,
                    "light_slider_value": stl_viewer.light_slider_value,
                    "camera_presets": {k: dict(v) for k, v in camera_presets.items()},
                }
                current_home_position = getattr(joint_editor, 'home_position_angles', {})
                session_path = ensure_session_save_dir()
                ok = save_project_xml(
                    session_path,
                    motion_state.get('urdf_path') or "",
                    motion_state.get('robot_model'),
                    graph, playback_ctrl, joint_editor,
                    motion_action_state=motion_action_state,
                    capture_current_func=capture_motion_action_snapshot,
                    model_type=motion_state.get('model_type', ''),
                    view_settings=current_view_settings,
                    home_position=current_home_position,
                )
                if ok:
                    print(f"[Session] Saved to {session_path}")
                else:
                    print("[Session] Save returned False")
            except Exception as _se:
                print(f"[Session] Save error: {_se}")

        _session_cb["save"] = _do_save_session

        def _close_companion_windows():
            for w in (
                pad_dialog[0],
                _code_editor_window[0],
                settings_dialog[0],
                export_motion_dialog[0],
                command_editor,
                mix_editor,
                joint_editor,
            ):
                if w is None:
                    continue
                try:
                    if hasattr(w, "shutdown"):
                        w.shutdown()
                    else:
                        w.close()
                except Exception:
                    pass
            app_inst = QtWidgets.QApplication.instance()
            if app_inst is None:
                return
            for w in list(app_inst.topLevelWidgets()):
                if w is main_window:
                    continue
                try:
                    if hasattr(w, "shutdown"):
                        w.shutdown()
                    else:
                        w.close()
                except Exception:
                    pass

        _lme_quit_closers.clear()
        _lme_quit_closers.append(_close_companion_windows)

        # ×ボタン / Alt+F4 用 closeEvent オーバーライド
        def _main_window_close_event(event):
            if _session_cb["save"] is not None:
                try:
                    _session_cb["save"]()
                    _session_cb["save"] = None
                except Exception as _ce:
                    print(f"[Session] Save on close failed: {_ce}")
            _run_companion_shutdown()
            event.accept()
            app_inst = QtWidgets.QApplication.instance()
            if app_inst is not None:
                QtCore.QTimer.singleShot(0, app_inst.quit)

        main_window.closeEvent = _main_window_close_event

        # 起動時に前回セッションを自動復元
        # 優先順位: 1) save/_lme_session.xml  2) last_xml_path  3) last_model_path / DEBUG
        def _try_restore_xml(xml_path, label):
            """Load a project XML and apply results. Returns True on success."""
            try:
                print(f"[Auto] Restoring {label}: {xml_path}")
                result = load_project_xml(
                    xml_path, graph, stl_viewer, joint_editor, playback_ctrl,
                    motion_state=motion_state, parent_window=main_window,
                    motion_action_state=motion_action_state,
                )
                if not result or len(result) < 3:
                    print(f"[Auto] {label}: unexpected result")
                    return False
                rm, urdf_path_r, ok = result[0], result[1], result[2]
                loaded_view = result[4] if len(result) >= 5 else {}
                loaded_home = result[5] if len(result) >= 6 else {}
                if ok:
                    if rm:
                        motion_state['robot_model'] = rm
                        motion_state['urdf_path'] = urdf_path_r
                        stl_viewer.set_robot_model(rm)
                    if isinstance(loaded_view, dict):
                        if "bg_color_a" in loaded_view and "bg_color_b" in loaded_view:
                            stl_viewer.set_bg_colors(loaded_view["bg_color_a"], loaded_view["bg_color_b"])
                        if "bg_gradient_type" in loaded_view:
                            stl_viewer.set_bg_gradient_type(loaded_view["bg_gradient_type"])
                        if "bg_slider_value" in loaded_view:
                            stl_viewer.bg_slider_value = loaded_view["bg_slider_value"]
                        if "light_slider_value" in loaded_view:
                            stl_viewer.light_slider_value = loaded_view["light_slider_value"]
                        if "camera_presets" in loaded_view:
                            for k, v in loaded_view["camera_presets"].items():
                                if k in camera_presets:
                                    camera_presets[k].update(v)
                            _save_camera_presets()
                    if isinstance(loaded_home, dict) and loaded_home:
                        joint_editor.set_home_position(loaded_home)
                    refresh_motion_action_combo(motion_action_state["current"])
                    sync_playback_bar_widgets()
                    # ノードの色適用 (singleShot 20ms) が完了した後に start を選択
                    QtCore.QTimer.singleShot(200, _focus_start_node_in_graph)
                    print(f"[Auto] {label} restored: {xml_path}")
                    return True
                print(f"[Auto] {label} restore failed (ok=False): {xml_path}")
                return False
            except Exception as e:
                print(f"[Auto] {label} restore error: {e}")
                traceback.print_exc()
                return False

        def auto_restore_last_session():
            _rs = load_app_settings()
            last_xml  = _rs.get("last_xml_path", "")
            last_model = _rs.get("last_model_path", "")
            last_mtype = _rs.get("last_model_type", "urdf")

            # 1) セッションファイルを最優先で復元（last_xml_pathは更新しない）
            _session_xml = resolve_session_file_for_load()
            if _session_xml:
                if _try_restore_xml(_session_xml, "session"):
                    return

            # 2) 前回手動保存プロジェクト
            if last_xml and os.path.exists(last_xml):
                if _try_restore_xml(last_xml, "project"):
                    return
                print(f"[Auto] Project restore failed (ok=False): {last_xml}")

            restore_path = last_model if (last_model and os.path.exists(last_model)) else (
                DEBUG_URDF_PATH if (DEBUG_AUTO_LOAD_URDF and os.path.exists(DEBUG_URDF_PATH)) else ""
            )
            if not restore_path:
                return
            try:
                print(f"[Auto] Restoring model: {restore_path}")
                if motion_state.get('robot_model'):
                    motion_state['robot_model'].remove_actors()

                _pf = parse_model_file if parse_model_file is not None else None
                if _pf is None:
                    print("[Auto] parse_model_file unavailable, skipping model restore")
                    return
                parsed = _pf(restore_path)
                if not parsed:
                    print(f"[Auto] Failed to parse model: {restore_path}")
                    return
                model_path, _wdir, model_data, model_type = parsed

                if model_type == 'mjcf':
                    rm = build_robot_model_from_mjcf(model_path, model_data)
                    robot_name = model_data.get('model_name', '') or model_data.get('robot_name', '')
                else:
                    rm = build_robot_model_from_urdf(model_path, model_data)
                    robot_name = model_data.get('robot_name', '')
                robot_name = robot_name or os.path.splitext(os.path.basename(restore_path))[0]

                rm.build_vtk_actors(stl_viewer.renderer)
                rm.apply_joint_angles(rm.get_default_angles())
                motion_state['robot_model'] = rm
                motion_state['urdf_path'] = model_path
                motion_state['model_type'] = model_type
                rm.model_type = model_type
                stl_viewer.set_robot_model(rm)
                joint_editor.build_from_robot(rm)
                if robot_name:
                    graph.robot_name = robot_name
                    if hasattr(graph, 'name_input') and graph.name_input:
                        graph.name_input.setText(robot_name)
                stl_viewer.reset_camera()
                stl_viewer.safe_render()
                print(f"[Auto] Model restored: {restore_path} (type={model_type})")
            except Exception as e:
                print(f"[Auto] Model restore error: {e}")
                traceback.print_exc()

        QtCore.QTimer.singleShot(500, auto_restore_last_session)

        # タイマーの設定（シグナル処理のため）
        timer = QtCore.QTimer()
        timer.start(500)
        timer.timeout.connect(lambda: None)

        # アプリケーションの実行
        sys.exit(app.exec() if hasattr(app, 'exec') else app.exec_())

    except Exception as e:
        print(f"An error occurred: {str(e)}")
        print("Traceback:")
        print(traceback.format_exc())
        cleanup_and_exit(_also_quit_app=True)
        sys.exit(1)
