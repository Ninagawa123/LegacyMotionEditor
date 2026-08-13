#!/usr/bin/env python3
"""
File Name: LegacyMotionEditor_MuJoCoStudio.py
Description: Lightweight Valkey streaming preview (MuJoCoStudio) for LegacyMotionEditor.

Author      : Izumi Ninagawa
License     : MIT License
Copyright (c) 2026 Izumi Ninagawa

LME companion (not PhysicalOn). Features:
  - Testgrid stage only (scene XML and PNG generation bundled in-script)
  - P1 robot via --model (LME launch) or saved settings
  - Valkey receive → Meridim → MuJoCo ctrl; publish IMU back
  - Based on merimujoco by holypong https://github.com/holypong/merimujoco
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import struct
import subprocess
import sys
import threading
import time
import traceback
import xml.etree.ElementTree as ET
import zlib
from multiprocessing import shared_memory
from typing import Optional

import numpy as np

# ---------------------------------------------------------------------------
# Paths / constants
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from LegacyMotionEditor_Utils import (  # noqa: E402
    JOINT_TO_MERIDIM,
    VALKEY_DEFAULT_HOST,
    VALKEY_DEFAULT_PORT,
    VALKEY_DEFAULT_READ_KEY,
    VALKEY_DEFAULT_WRITE_KEY,
    ensure_valkey_container_running,
    path_for_project_save,
    resolve_project_path,
    valkey_available,
)

WINDOW_TITLE = "LegacyMotionEditor MuJoCoStudio"
SAVE_DIR = os.path.join(SCRIPT_DIR, "save")
SETTINGS_PATH = os.path.join(SAVE_DIR, "LegacyMotionEditor_MuJoCoStudio_settings.json")
ACTIVE_SCENE_REL = "log/_active_scene.xml"

# Embedded stage scene (replaces stage/scene_with_testgrid.xml).
# Texture path "testgrid_mjcf/testgrid_grid.png" is relative to log/ where
# the active scene XML is written, matching ensure_testgrid()'s output dir.
_SCENE_XML = """\
<mujoco model="lme_testgrid">
  <compiler angle="radian" autolimits="true" balanceinertia="true"/>
  <option solver="Newton" cone="elliptic" integrator="implicitfast" impratio="10"/>
  <default>
    <default class="stage_floor">
      <geom friction="0.08 0.003 0.0001"/>
    </default>
  </default>
  <statistic center="0 0 0.300000" extent="0.800000"/>
  <visual>
    <headlight diffuse="0.60 0.60 0.60" ambient="0.30 0.30 0.30" specular="0.05 0.05 0.05"/>
    <rgba haze="0.10 0.10 0.10 1"/>
    <global azimuth="-130" elevation="-20" offwidth="1280" offheight="720"/>
  </visual>
  <asset>
    <texture type="skybox" builtin="gradient"
             rgb1="0.68 0.68 0.68" rgb2="0.84 0.84 0.84"
             width="512" height="3072"/>
    <texture type="2d" name="testgrid_tex" file="testgrid_mjcf/testgrid_grid.png"/>
    <material name="testgrid_mat_flat" texture="testgrid_tex"
              texrepeat="15 15" texuniform="false" reflectance="0"/>
  </asset>
  <worldbody>
    <light pos="0 0 6" dir="0 0 -1" directional="true"
           diffuse="0.65 0.65 0.65" specular="0.10 0.10 0.10"/>
    <light pos=" 4  0 3" dir="-1  0 -1" directional="false"
           diffuse="0.25 0.25 0.25" specular="0.04 0.04 0.04" cutoff="70"/>
    <light pos="-4  0 3" dir=" 1  0 -1" directional="false"
           diffuse="0.25 0.25 0.25" specular="0.04 0.04 0.04" cutoff="70"/>
    <light pos=" 0  4 3" dir=" 0 -1 -1" directional="false"
           diffuse="0.25 0.25 0.25" specular="0.04 0.04 0.04" cutoff="70"/>
    <light pos=" 0 -4 3" dir=" 0  1 -1" directional="false"
           diffuse="0.25 0.25 0.25" specular="0.04 0.04 0.04" cutoff="70"/>
    <geom name="testgrid_floor_flat" class="stage_floor" type="box" size="15 15 0.002"
          pos="0 0 -0.001" material="testgrid_mat_flat"
          group="2" contype="1" conaffinity="1"
          condim="3"
          solref="0.005 1" solimp="0.9 0.95 0.001 0.5 2"/>
  </worldbody>
</mujoco>"""
LOG_DIR = os.path.join(SCRIPT_DIR, "log")

MSG_SIZE = 90
MASTER_CMD_RESET = 5556
MERIDIM_CMD_TORQUE_OFF = 0
MERIDIM_CMD_SERVO_ON = 1

SENSOR_STRIDE = 15
# shm meta: 0-3 reserved; 4=reset, 6=stop
META_RESET_REQUESTED = 4
META_STOP_REQUESTED = 6
META_SIZE = 7

WIDTH_DEFAULT, HEIGHT_DEFAULT = 1280, 720
WIDTH_MIN, HEIGHT_MIN = 640, 360
CTRL_HZ_DEFAULT = 100

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("LME_MuJoCoStudio")


# ===========================================================================
# Motor subprocess (Valkey ↔ shared memory) — early exit when --shm-ctrl
# ===========================================================================
class _PlayerLink:
    def __init__(self, host, port, read_key, write_key, joint_lookup):
        self.read_key = read_key
        self.write_key = write_key
        self.joint_lookup = joint_lookup
        self.mdata = [0.0] * MSG_SIZE
        try:
            import valkey
            self.client = valkey.Valkey(
                host=host, port=port, decode_responses=True,
                socket_connect_timeout=0.5, socket_timeout=0.5)
        except Exception as e:
            logger.warning("[studio] Valkey init failed: %s", e)
            self.client = None

    def get_data(self):
        if self.client is None:
            return None
        try:
            raw = self.client.hgetall(self.read_key)
            if not raw or len(raw) < MSG_SIZE:
                return None
            return [float(raw[str(i)]) for i in range(MSG_SIZE)]
        except Exception:
            return None

    def set_data(self, data):
        if self.client is None:
            return
        try:
            self.client.hset(
                self.write_key,
                mapping={str(i): str(float(v)) for i, v in enumerate(data)})
        except Exception:
            pass


def _player_receive_shared(link, ctrl_arr, n_ctrl) -> bool:
    rcv = link.get_data()
    if not rcv or len(rcv) != MSG_SIZE:
        return False
    reset_req = (rcv[0] == MASTER_CMD_RESET)
    for ctrl_idx, meridis_idx, meridis_mul in link.joint_lookup:
        if ctrl_idx < 0 or ctrl_idx >= n_ctrl:
            continue
        command_rad = np.radians(float(rcv[meridis_idx]) / 100.0 * meridis_mul)
        cmd = int(round(float(rcv[meridis_idx - 1])))
        low = cmd & 0xFF
        if low == MERIDIM_CMD_TORQUE_OFF:
            continue
        if low == MERIDIM_CMD_SERVO_ON:
            ctrl_arr[ctrl_idx] = round(command_rad, 4)
    for i in range(len(link.mdata)):
        link.mdata[i] = float(rcv[i]) if i < len(rcv) else 0.0
    return reset_req


def _player_send_shared(link, sensor_arr) -> None:
    """Publish IMU (accel / gyro / RPY). Joint angles are not echoed."""
    R = sensor_arr[0:9].reshape(3, 3)
    wx, wy, wz = sensor_arr[9:12]
    ax, ay, az = sensor_arr[12:15]
    yaw = math.atan2(float(R[1, 0]), float(R[0, 0]))
    pitch = math.asin(max(-1.0, min(1.0, -float(R[2, 0]))))
    roll = math.atan2(float(R[2, 1]), float(R[2, 2]))
    link.mdata[2] = round(float(ax) * 100.0, 4)
    link.mdata[3] = round(float(ay) * 100.0, 4)
    link.mdata[4] = round(float(az) * 100.0, 4)
    link.mdata[5] = round(math.degrees(float(wx)) * 100.0, 4)
    link.mdata[6] = round(math.degrees(float(wy)) * 100.0, 4)
    link.mdata[7] = round(math.degrees(float(wz)) * 100.0, 4)
    link.mdata[12] = round(math.degrees(roll) * 100.0, 4)
    link.mdata[13] = round(math.degrees(pitch) * 100.0, 4)
    link.mdata[14] = round(math.degrees(yaw) * 100.0, 4)
    link.set_data([round(v, 6) for v in link.mdata])


def _run_motor_control_subprocess(
        shm_ctrl_name, ctrl_size, shm_sensor_name, shm_meta_name,
        valkey_host, valkey_port, receive_key, publish_key, joint_map,
        ctrl_hz=100) -> None:
    shm_ctrl = shared_memory.SharedMemory(name=shm_ctrl_name)
    shm_sensor = shared_memory.SharedMemory(name=shm_sensor_name)
    shm_meta = shared_memory.SharedMemory(name=shm_meta_name)
    try:
        from multiprocessing import resource_tracker as _rt
        for shm in (shm_ctrl, shm_sensor, shm_meta):
            try:
                _rt.unregister(shm._name, "shared_memory")
            except Exception:
                pass
    except Exception:
        pass
    ctrl_arr = np.ndarray((ctrl_size,), dtype=np.float64, buffer=shm_ctrl.buf)
    sensor_arr = np.ndarray((SENSOR_STRIDE,), dtype=np.float64, buffer=shm_sensor.buf)
    meta_arr = np.ndarray((META_SIZE,), dtype=np.float64, buffer=shm_meta.buf)

    link = _PlayerLink(
        valkey_host, valkey_port, receive_key, publish_key, joint_map or [])

    interval = 1.0 / max(1, ctrl_hz)
    deadline = time.perf_counter() + interval
    logger.info("Motor control process started")
    try:
        while meta_arr[META_STOP_REQUESTED] == 0.0:
            if _player_receive_shared(link, ctrl_arr, ctrl_size):
                meta_arr[META_RESET_REQUESTED] = 1.0
            _player_send_shared(link, sensor_arr)
            now = time.perf_counter()
            rem = deadline - now
            if rem > 0:
                time.sleep(rem)
            deadline += interval
            if deadline < time.perf_counter() - interval:
                deadline = time.perf_counter() + interval
    except KeyboardInterrupt:
        pass
    finally:
        for shm in (shm_ctrl, shm_sensor, shm_meta):
            try:
                shm.close()
            except Exception:
                pass
        logger.info("Motor control process exiting")


def _parse_motor_args():
    p = argparse.ArgumentParser()
    p.add_argument("--shm-ctrl", required=True)
    p.add_argument("--ctrl-size", type=int, required=True)
    p.add_argument("--shm-sensor", required=True)
    p.add_argument("--shm-meta", required=True)
    p.add_argument("--valkey-host", required=True)
    p.add_argument("--valkey-port", type=int, required=True)
    p.add_argument("--receive-key", required=True)
    p.add_argument("--publish-key", required=True)
    p.add_argument("--joint-map-json", default="[]")
    p.add_argument("--ctrl-hz", type=int, default=100)
    return p.parse_args()


if "--shm-ctrl" in sys.argv:
    logging.basicConfig(
        level=logging.WARNING,
        format="%(asctime)s [%(levelname)s] [motor] %(message)s",
        datefmt="%H:%M:%S",
        force=True,
    )
    a = _parse_motor_args()
    _run_motor_control_subprocess(
        a.shm_ctrl, a.ctrl_size, a.shm_sensor, a.shm_meta,
        a.valkey_host, a.valkey_port, a.receive_key, a.publish_key,
        json.loads(a.joint_map_json),
        a.ctrl_hz)
    sys.exit(0)


# ===========================================================================
# Main app imports (heavy) — after motor early-exit
# ===========================================================================
import mujoco  # noqa: E402
import pygame  # noqa: E402


# ===========================================================================
# Settings
# ===========================================================================
def default_settings() -> dict:
    return {
        "model_path": "",
        "valkey": {
            "host": VALKEY_DEFAULT_HOST,
            "port": VALKEY_DEFAULT_PORT,
            # Studio READs what LME WRITEs; PUBLISH is IMU back to LME/PhysicalOn
            "receive_key": VALKEY_DEFAULT_WRITE_KEY,  # merikey_psclon_sub
            "publish_key": VALKEY_DEFAULT_READ_KEY,   # merikey_psclon_pub
        },
        "ctrl_hz": CTRL_HZ_DEFAULT,
        "camera": {
            "azimuth": -130.0,
            "elevation": -20.0,
            "distance": 2.0,
            "lookat": [0.0, 0.0, 0.3],
        },
        "window": {
            "width": WIDTH_DEFAULT,
            "height": HEIGHT_DEFAULT,
        },
    }


def parse_viewer_args(argv=None):
    p = argparse.ArgumentParser(description="LME MuJoCoStudio preview")
    p.add_argument("--model", default="", help="Robot MJCF path")
    p.add_argument("--valkey-host", default="")
    p.add_argument("--valkey-port", type=int, default=0)
    p.add_argument("--receive-key", default="")
    p.add_argument("--publish-key", default="")
    return p.parse_args(argv)


def apply_cli_overrides(cfg: dict, args) -> dict:
    if args is None:
        return cfg
    model = str(getattr(args, "model", "") or "").strip()
    if model:
        cfg["model_path"] = resolve_project_path(model, SETTINGS_PATH)
    vk = cfg["valkey"]
    host = str(getattr(args, "valkey_host", "") or "").strip()
    if host:
        vk["host"] = host
    port = int(getattr(args, "valkey_port", 0) or 0)
    if port:
        vk["port"] = port
    recv = str(getattr(args, "receive_key", "") or "").strip()
    if recv:
        vk["receive_key"] = recv
    pub = str(getattr(args, "publish_key", "") or "").strip()
    if pub:
        vk["publish_key"] = pub
    return cfg


def load_settings() -> dict:
    cfg = default_settings()
    if not os.path.isfile(SETTINGS_PATH):
        return cfg
    try:
        with open(SETTINGS_PATH, "r", encoding="utf-8") as f:
            raw = json.load(f)
        if isinstance(raw, dict):
            cfg["model_path"] = resolve_project_path(
                str(raw.get("model_path", "") or ""), SETTINGS_PATH)
            vk = raw.get("valkey") or {}
            cfg["valkey"].update({k: vk[k] for k in cfg["valkey"] if k in vk})
            if "ctrl_hz" in raw:
                cfg["ctrl_hz"] = int(raw["ctrl_hz"])
            cam = raw.get("camera") or {}
            cfg["camera"].update({k: cam[k] for k in cfg["camera"] if k in cam})
            win = raw.get("window") or {}
            cfg["window"].update({k: win[k] for k in cfg["window"] if k in win})
    except Exception as e:
        logger.warning("settings load failed: %s", e)
    return cfg


def save_settings(cfg: dict) -> None:
    try:
        os.makedirs(SAVE_DIR, exist_ok=True)
        out = json.loads(json.dumps(cfg))
        out["model_path"] = path_for_project_save(
            out.get("model_path", ""), SETTINGS_PATH)
        with open(SETTINGS_PATH, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2, ensure_ascii=False)
    except Exception as e:
        logger.warning("settings save failed: %s", e)


# ===========================================================================
# Testgrid texture / scene helpers
# ===========================================================================
def _save_png_rgb(arr: np.ndarray, path: str) -> None:
    h, w = arr.shape[:2]
    raw = b""
    for y in range(h):
        raw += b"\x00" + arr[y].tobytes()

    def chunk(tag, data):
        return (struct.pack(">I", len(data)) + tag + data
                + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF))

    ihdr = struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)
    with open(path, "wb") as f:
        f.write(b"\x89PNG\r\n\x1a\n")
        f.write(chunk(b"IHDR", ihdr))
        f.write(chunk(b"IDAT", zlib.compress(raw, 6)))
        f.write(chunk(b"IEND", b""))


def ensure_testgrid(script_dir: str) -> None:
    tg = os.path.join(script_dir, "log", "testgrid_mjcf")
    os.makedirs(tg, exist_ok=True)
    png = os.path.join(tg, "testgrid_grid.png")
    if os.path.isfile(png):
        return
    size = 1000
    arr = np.full((size, size, 3), 145, dtype=np.uint8)
    for i in range(0, size, 10):
        arr[i, :] = (112, 112, 112)
        arr[:, i] = (112, 112, 112)
    for i in range(0, size, 100):
        arr[max(0, i - 1):i + 2, :] = (82, 82, 82)
        arr[:, max(0, i - 1):i + 2] = (82, 82, 82)
    arr[0:3, :] = arr[size - 3:, :] = (50, 50, 50)
    arr[:, 0:3] = arr[:, size - 3:] = (50, 50, 50)
    _save_png_rgb(arr, png)
    logger.info("testgrid texture generated: %s", png)


def _meshdir_rel_to_main(src_path: str, main_xml_dir: str) -> str:
    """meshdir relative to the MAIN MJCF (include parent), not the included file.

    MuJoCo resolves meshdir + mesh/@file from the top-level XML directory.
    Active scene lives in log/, robot play XML lives next to the source model.
    """
    src_dir = os.path.dirname(os.path.abspath(src_path))
    rel = os.path.relpath(src_dir, os.path.abspath(main_xml_dir)).replace("\\", "/")
    return "" if rel == "." else rel


def _rel_asset_paths(root, src_path: str) -> None:
    """Rewrite mesh/texture file attrs to paths relative to the source MJCF dir."""
    src_dir = os.path.dirname(os.path.abspath(src_path))

    def resolve(raw: str):
        if not raw or raw.startswith("builtin:"):
            return None
        if os.path.isabs(raw):
            return raw if os.path.isfile(raw) else None
        cand = os.path.normpath(os.path.join(src_dir, raw))
        return cand if os.path.isfile(cand) else None

    def fix(elem, attr):
        full = resolve(elem.get(attr) or "")
        if full:
            elem.set(attr, os.path.relpath(full, src_dir).replace("\\", "/"))

    for mesh in root.findall(".//mesh"):
        fix(mesh, "file")
    for tex in root.findall(".//texture"):
        fix(tex, "file")


def _compute_spawn_z(src_path: str, clearance: float = 0.010) -> float:
    try:
        tmp_m = mujoco.MjModel.from_xml_path(src_path)
        tmp_d = mujoco.MjData(tmp_m)
        mujoco.mj_fwdPosition(tmp_m, tmp_d)
        base_z = 0.0
        for j in range(tmp_m.njnt):
            if tmp_m.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE:
                base_z = float(tmp_d.xpos[tmp_m.jnt_bodyid[j], 2])
                break
        min_gz = float("inf")
        for i in range(tmp_m.ngeom):
            gtype = tmp_m.geom_type[i]
            if gtype == mujoco.mjtGeom.mjGEOM_PLANE:
                continue
            is_col = not (tmp_m.geom_contype[i] == 0 and tmp_m.geom_conaffinity[i] == 0)
            if not is_col and gtype != mujoco.mjtGeom.mjGEOM_MESH:
                continue
            gz = float(tmp_d.geom_xpos[i, 2])
            sz = tmp_m.geom_size[i]
            xmat = tmp_d.geom_xmat[i].reshape(3, 3)
            if gtype == mujoco.mjtGeom.mjGEOM_SPHERE:
                bot = gz - float(sz[0])
            elif gtype == mujoco.mjtGeom.mjGEOM_BOX:
                bot = gz - (abs(float(sz[0]) * xmat[0, 2])
                            + abs(float(sz[1]) * xmat[1, 2])
                            + abs(float(sz[2]) * xmat[2, 2]))
            elif gtype == mujoco.mjtGeom.mjGEOM_MESH:
                mid = int(tmp_m.geom_dataid[i])
                vadr = int(tmp_m.mesh_vertadr[mid])
                vnum = int(tmp_m.mesh_vertnum[mid])
                if vnum <= 0:
                    continue
                verts = tmp_m.mesh_vert[vadr:vadr + vnum]
                bot = float(gz + (verts @ xmat[:, 2]).min())
            else:
                bot = gz - float(sz[0]) if sz[0] > 0 else gz
            min_gz = min(min_gz, bot)
        if math.isinf(min_gz):
            return 0.27
        return max(clearance, base_z + clearance - min_gz)
    except Exception as e:
        logger.warning("spawn_z failed: %s", e)
        return 0.27


def generate_play_mjcf(src_path: str, out_path: str,
                     pos=(0.0, 0.0, 0.27),
                     quat=(1.0, 0.0, 0.0, 0.0),
                     main_xml_dir: str | None = None) -> bool:
    try:
        tree = ET.parse(src_path)
        root = tree.getroot()
    except ET.ParseError:
        try:
            with open(src_path, "rb") as f:
                root = ET.fromstring(f.read().decode("utf-8", errors="replace"))
            tree = ET.ElementTree(root)
        except Exception as e:
            logger.error("MJCF parse failed: %s", e)
            return False

    for tag in ("compiler", "option"):
        for el in list(root.findall(tag)):
            root.remove(el)
    # meshdir is relative to the MAIN scene XML (log/_active_scene.xml).
    main_xml_dir = os.path.abspath(main_xml_dir or os.path.dirname(out_path))
    meshdir_rel = _meshdir_rel_to_main(src_path, main_xml_dir)
    comp = ET.Element("compiler")
    comp.set("angle", "radian")
    comp.set("autolimits", "true")
    if meshdir_rel:
        comp.set("meshdir", meshdir_rel)
    root.insert(0, comp)

    default_el = root.find("default")
    if default_el is not None:
        for child in list(default_el.iter()):
            if child.tag in ("motor", "position", "velocity", "general", "intvelocity"):
                for attr in ("ctrlrange", "ctrllimited"):
                    child.attrib.pop(attr, None)

    worldbody = root.find("worldbody")
    if worldbody is not None:
        root_body = worldbody.find("body")
        if root_body is not None:
            root_body.set("pos", f"{pos[0]} {pos[1]} {pos[2]}")
            root_body.set("quat", f"{quat[0]} {quat[1]} {quat[2]} {quat[3]}")

    actuator_el = root.find("actuator")
    if actuator_el is not None:
        for act in actuator_el:
            for attr in ("ctrlrange", "ctrllimited"):
                act.attrib.pop(attr, None)

    _rel_asset_paths(root, src_path)
    root.set("model", f"{os.path.splitext(os.path.basename(src_path))[0]}_play")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    try:
        ET.indent(root, space="  ")
    except AttributeError:
        pass
    ET.ElementTree(root).write(out_path, encoding="unicode", xml_declaration=False)
    logger.info("Generated %s", out_path)
    return True


def write_active_scene(script_dir: str, p1_rel: str = "") -> str:
    out = os.path.join(script_dir, ACTIVE_SCENE_REL.replace("/", os.sep))
    os.makedirs(os.path.dirname(out), exist_ok=True)
    root = ET.fromstring(_SCENE_XML)
    # Drop any existing robot includes
    for inc in list(root.findall("include")):
        root.remove(inc)
    if p1_rel and os.path.isfile(os.path.join(script_dir, p1_rel)):
        abs_p1 = os.path.join(script_dir, p1_rel)
        scene_dir = os.path.dirname(out)
        p1_dir = os.path.dirname(os.path.abspath(abs_p1))
        meshdir_rel = os.path.relpath(p1_dir, scene_dir).replace("\\", "/")
        compiler = root.find("compiler")
        if compiler is not None and meshdir_rel and meshdir_rel != ".":
            compiler.set("meshdir", meshdir_rel)
        inc = ET.Element("include")
        # Path must be relative to the generated active scene (log/).
        inc.set("file", os.path.relpath(abs_p1, scene_dir).replace("\\", "/"))
        # Insert after compiler
        idx = 1
        for i, child in enumerate(list(root)):
            if child.tag == "compiler":
                idx = i + 1
                break
        root.insert(idx, inc)
    # Texture path "testgrid_mjcf/testgrid_grid.png" in _SCENE_XML is already
    # relative to log/ (where ensure_testgrid writes the PNG), so no rewrite needed.
    ET.ElementTree(root).write(out, encoding="unicode", xml_declaration=False)
    return out


def resolve_joint_map(model: "mujoco.MjModel") -> list:
    """Map actuators → Meridim using JOINT_TO_MERIDIM."""
    resolved = []
    for aid in range(model.nu):
        jid = int(model.actuator_trnid[aid, 0])
        if jid < 0:
            continue
        jname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, jid) or ""
        bare = jname[3:] if jname.startswith("p1_") else jname
        entry = JOINT_TO_MERIDIM.get(bare)
        if entry is None:
            continue
        meridim_idx, mul = int(entry[0]), float(entry[1])
        resolved.append([aid, meridim_idx, mul])
    logger.info("joint map: %d actuators matched", len(resolved))
    return resolved


# ===========================================================================
# App runtime
# ===========================================================================
class StudioApp:
    def __init__(self, args=None):
        self.cfg = apply_cli_overrides(load_settings(), args)
        self.model: Optional[mujoco.MjModel] = None
        self.data: Optional[mujoco.MjData] = None
        self.cam = mujoco.MjvCamera()
        self.renderer: Optional[mujoco.Renderer] = None

        self.motor_proc: Optional[subprocess.Popen] = None
        self.shm_ctrl = self.shm_sensor = self.shm_meta = None
        self.ctrl_arr = None
        self.sensor_arr = None
        self.meta_arr = None
        self.joint_map: list = []
        self.valkey_running = False
        self._valkey_autostart_done = threading.Event()
        self._valkey_autostart_ok = False
        self._valkey_autostart_handled = False

        self._status = ""
        self._font_sm = None

        self._button_down = False
        self._last_mouse = (0, 0)
        self._orbit_btn = 1  # left
        self.win_w = WIDTH_DEFAULT
        self.win_h = HEIGHT_DEFAULT

    def _ensure_renderer(self) -> None:
        """(Re)create MuJoCo Renderer for the current window size."""
        if self.model is None:
            return
        if self.renderer is not None:
            try:
                self.renderer.close()
            except Exception:
                pass
            self.renderer = None
        w = max(WIDTH_MIN, int(self.win_w))
        h = max(HEIGHT_MIN, int(self.win_h))
        self.renderer = mujoco.Renderer(self.model, height=h, width=w)

    # ----- model / scene -----
    def prepare_scene(self) -> str:
        ensure_testgrid(SCRIPT_DIR)
        src = resolve_project_path(
            (self.cfg.get("model_path") or "").strip(), SETTINGS_PATH)
        p1_rel = ""
        if src and os.path.isfile(src):
            src_stem = os.path.splitext(os.path.basename(src))[0]
            out = os.path.join(os.path.dirname(os.path.abspath(src)),
                               f"{src_stem}_play.xml")
            z = _compute_spawn_z(src)
            scene_dir = os.path.join(SCRIPT_DIR, os.path.dirname(ACTIVE_SCENE_REL))
            if generate_play_mjcf(src, out, pos=(0.0, 0.0, z),
                                main_xml_dir=scene_dir):
                p1_rel = os.path.relpath(out, SCRIPT_DIR).replace("\\", "/")
        return write_active_scene(SCRIPT_DIR, p1_rel)

    def load_model(self, restore_cam: bool = True) -> None:
        scene = self.prepare_scene()
        self.model = mujoco.MjModel.from_xml_path(scene)
        self.data = mujoco.MjData(self.model)
        self.model.opt.timestep = 0.002
        mujoco.mjv_defaultCamera(self.cam)
        self.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        if restore_cam:
            cam_cfg = self.cfg.get("camera") or {}
            self.cam.azimuth = float(cam_cfg.get("azimuth", -130.0))
            self.cam.elevation = float(cam_cfg.get("elevation", -20.0))
            self.cam.distance = float(cam_cfg.get("distance", 2.0))
            lookat = cam_cfg.get("lookat", [0.0, 0.0, 0.3])
            self.cam.lookat[:] = lookat
        else:
            self.cam.distance = 2.0
            self.cam.azimuth = -130
            self.cam.elevation = -20
            self.cam.lookat[:] = [0, 0, 0.3]
        self._ensure_renderer()
        self.joint_map = resolve_joint_map(self.model)
        logger.info("Loaded scene nu=%d nq=%d", self.model.nu, self.model.nq)

    def respawn_model(self) -> None:
        """Reset robot pose/velocities to the initial spawn (R key)."""
        if self.model is None or self.data is None:
            return
        mujoco.mj_resetData(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)
        if self.ctrl_arr is not None:
            n = min(len(self.ctrl_arr), self.model.nu)
            self.ctrl_arr[:n] = np.asarray(self.data.ctrl[:n])
        self._status = "Respawned"
        logger.info("Model respawned (R)")

    # ----- Valkey / motor -----
    def start_valkey(self) -> None:
        if self.model is None or self.data is None:
            return
        self.stop_valkey()
        n_ctrl = int(self.model.nu)
        if n_ctrl <= 0:
            self._status = "No actuators (pass --model MJCF)"
            return
        vk = self.cfg["valkey"]
        self.shm_ctrl = shared_memory.SharedMemory(create=True, size=n_ctrl * 8)
        self.shm_sensor = shared_memory.SharedMemory(create=True, size=SENSOR_STRIDE * 8)
        self.shm_meta = shared_memory.SharedMemory(create=True, size=META_SIZE * 8)
        self.ctrl_arr = np.ndarray((n_ctrl,), dtype=np.float64, buffer=self.shm_ctrl.buf)
        self.sensor_arr = np.ndarray((SENSOR_STRIDE,), dtype=np.float64, buffer=self.shm_sensor.buf)
        self.meta_arr = np.ndarray((META_SIZE,), dtype=np.float64, buffer=self.shm_meta.buf)
        self.ctrl_arr[:] = np.asarray(self.data.ctrl[:n_ctrl])
        self.sensor_arr[:] = 0.0
        self.meta_arr[:] = 0.0

        cmd = [
            sys.executable, os.path.abspath(__file__),
            "--shm-ctrl", self.shm_ctrl.name,
            "--ctrl-size", str(n_ctrl),
            "--shm-sensor", self.shm_sensor.name,
            "--shm-meta", self.shm_meta.name,
            "--valkey-host", str(vk["host"]),
            "--valkey-port", str(int(vk["port"])),
            "--receive-key", str(vk["receive_key"]),
            "--publish-key", str(vk["publish_key"]),
            "--joint-map-json", json.dumps(self.joint_map),
            "--ctrl-hz", str(int(self.cfg.get("ctrl_hz", CTRL_HZ_DEFAULT))),
        ]
        self.motor_proc = subprocess.Popen(cmd)
        self.valkey_running = True
        self._status = f"Valkey ON  recv={vk['receive_key']}"
        logger.info("Valkey motor started pid=%s", self.motor_proc.pid)

    def stop_valkey(self) -> None:
        if self.meta_arr is not None:
            try:
                self.meta_arr[META_STOP_REQUESTED] = 1.0
            except Exception:
                pass
        if self.motor_proc is not None:
            try:
                self.motor_proc.wait(timeout=1.5)
            except Exception:
                try:
                    self.motor_proc.kill()
                except Exception:
                    pass
            self.motor_proc = None
        for shm in (self.shm_ctrl, self.shm_sensor, self.shm_meta):
            if shm is None:
                continue
            try:
                shm.close()
                shm.unlink()
            except Exception:
                pass
        self.shm_ctrl = self.shm_sensor = self.shm_meta = None
        self.ctrl_arr = self.sensor_arr = self.meta_arr = None
        if self.valkey_running:
            self._status = "Valkey OFF"
        self.valkey_running = False

    def sync_ctrl(self) -> None:
        if self.ctrl_arr is None or self.data is None:
            return
        n = min(len(self.ctrl_arr), self.model.nu)
        self.data.ctrl[:n] = self.ctrl_arr[:n]

    def sync_sensors(self) -> None:
        if self.sensor_arr is None or self.model is None or self.data is None:
            return
        # Freejoint body orientation as crude IMU
        body_id = -1
        for j in range(self.model.njnt):
            if self.model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE:
                body_id = int(self.model.jnt_bodyid[j])
                break
        if body_id < 0:
            return
        R = self.data.xmat[body_id].reshape(3, 3)
        self.sensor_arr[0:9] = R.ravel()
        # gyro / accel approx from qvel if freejoint
        self.sensor_arr[9:12] = 0.0
        self.sensor_arr[12:15] = [0.0, 0.0, 9.81]

    # ----- main loop -----
    def run(self) -> None:
        os.makedirs(LOG_DIR, exist_ok=True)
        win_cfg = self.cfg.get("window") or {}
        self.win_w = max(WIDTH_MIN, int(win_cfg.get("width", WIDTH_DEFAULT)))
        self.win_h = max(HEIGHT_MIN, int(win_cfg.get("height", HEIGHT_DEFAULT)))
        pygame.display.init()
        pygame.font.init()
        try:
            pygame.event.init()
        except Exception:
            pass
        # Do not pygame.init() — that also starts joystick HIDAPI, which on macOS
        # can steal DualShock from Apple GameController in the LME process.
        pygame.display.set_caption(WINDOW_TITLE)
        info = pygame.display.Info()
        sw = int(getattr(info, "current_w", 0) or 0)
        if sw <= 0:
            try:
                import tkinter as _tk
                _r = _tk.Tk()
                _r.withdraw()
                sw = int(_r.winfo_screenwidth())
                _r.destroy()
            except Exception:
                sw = 1920
        if sys.platform == "darwin":
            top = 25  # room for the macOS menu bar
        elif os.name == "nt":
            # On Windows, SDL_VIDEO_WINDOW_POS places the outer window (title
            # bar included) at this y-coordinate. y=0 pushes the title bar
            # above the visible screen, leaving the window undecorated and
            # unmovable/unresizable. Offset by the actual caption+border
            # height so the title bar stays on-screen.
            try:
                import ctypes
                user32 = ctypes.windll.user32
                top = user32.GetSystemMetrics(4) + user32.GetSystemMetrics(33) * 2  # SM_CYCAPTION, SM_CYSIZEFRAME
            except Exception:
                top = 32
        else:
            top = 0
        os.environ["SDL_VIDEO_WINDOW_POS"] = f"{max(0, sw - self.win_w)},{top}"
        os.environ.pop("SDL_VIDEO_CENTERED", None)
        screen = pygame.display.set_mode(
            (self.win_w, self.win_h), pygame.RESIZABLE)
        self._font_sm = pygame.font.SysFont("Arial", 16)
        clock = pygame.time.Clock()

        self.load_model()
        vk = self.cfg.get("valkey") or {}
        vk_host = vk.get("host", VALKEY_DEFAULT_HOST)
        vk_port = int(vk.get("port", VALKEY_DEFAULT_PORT) or VALKEY_DEFAULT_PORT)
        if valkey_available(vk_host, vk_port):
            self.start_valkey()
            logger.info("Valkey available — started on launch")
        elif os.name == "nt":
            # Windows only: try to auto-start Docker Desktop / the
            # physicalon-valkey container off the main thread so the window
            # keeps rendering while we wait (can take up to a minute). On
            # Mac/Ubuntu, Valkey is expected to already be running, so this
            # matches the original (no auto-start) behavior there.
            logger.warning("Valkey server not reachable — attempting auto-start (Windows)")

            def _autostart():
                ok = ensure_valkey_container_running(vk_host, vk_port)
                self._valkey_autostart_ok = ok
                self._valkey_autostart_done.set()

            threading.Thread(target=_autostart, daemon=True).start()
        else:
            logger.warning("Valkey server not reachable")

        running = True
        try:
            while running:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        running = False
                    elif event.type == pygame.VIDEORESIZE:
                        self.win_w = max(WIDTH_MIN, int(event.w))
                        self.win_h = max(HEIGHT_MIN, int(event.h))
                        screen = pygame.display.set_mode(
                            (self.win_w, self.win_h), pygame.RESIZABLE)
                        self._ensure_renderer()
                    elif event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_r:
                            self.respawn_model()
                        elif event.key == pygame.K_ESCAPE:
                            running = False
                    elif event.type == pygame.MOUSEBUTTONDOWN:
                        self._button_down = True
                        self._orbit_btn = event.button
                        self._last_mouse = event.pos
                    elif event.type == pygame.MOUSEBUTTONUP:
                        self._button_down = False
                    elif event.type == pygame.MOUSEMOTION and self._button_down:
                        dx = event.pos[0] - self._last_mouse[0]
                        dy = event.pos[1] - self._last_mouse[1]
                        self._last_mouse = event.pos
                        shift = pygame.key.get_mods() & pygame.KMOD_SHIFT
                        if self._orbit_btn == 1 and not shift:
                            self.cam.azimuth -= dx * 0.3
                            self.cam.elevation = max(
                                -89, min(89, self.cam.elevation - dy * 0.3))
                        else:
                            # Right-drag or Shift+left-drag: pan (lookat translation)
                            self.cam.lookat[0] -= dx * 0.002 * self.cam.distance
                            self.cam.lookat[1] += dy * 0.002 * self.cam.distance
                    elif event.type == pygame.MOUSEWHEEL:
                        self.cam.distance = max(
                            0.2, self.cam.distance * (0.9 if event.y > 0 else 1.1))

                # Pick up the result of the background Docker/Valkey auto-start
                if (self._valkey_autostart_done.is_set()
                        and not self._valkey_autostart_handled):
                    self._valkey_autostart_handled = True
                    if self._valkey_autostart_ok and not self.valkey_running:
                        self.start_valkey()
                        logger.info("Valkey available — started after Docker auto-start")
                    elif not self._valkey_autostart_ok:
                        self._status = "Valkey OFF (auto-start failed)"
                        logger.warning("Valkey auto-start via Docker failed")

                # Physics
                if self.model is not None and self.data is not None:
                    self.sync_ctrl()
                    steps = max(1, int(round((1.0 / 60.0) / float(self.model.opt.timestep))))
                    for _ in range(steps):
                        mujoco.mj_step(self.model, self.data)
                    self.sync_sensors()
                    if self.meta_arr is not None and self.meta_arr[META_RESET_REQUESTED]:
                        self.meta_arr[META_RESET_REQUESTED] = 0.0
                        self.respawn_model()

                # Render MuJoCo → pygame surface
                assert self.renderer is not None and self.data is not None
                self.renderer.update_scene(self.data, self.cam)
                pixels = self.renderer.render()  # H x W x 3
                frame = pygame.surfarray.make_surface(
                    np.transpose(pixels, (1, 0, 2)))
                if frame.get_size() != (self.win_w, self.win_h):
                    frame = pygame.transform.smoothscale(
                        frame, (self.win_w, self.win_h))
                screen.blit(frame, (0, 0))

                # Status HUD (top-left)
                tip_col = (55, 60, 70)
                screen.blit(
                    self._font_sm.render("Press R to Respawn", True, tip_col),
                    (12, 8))
                status = self._status or (
                    "Valkey ON" if self.valkey_running else "Valkey OFF")
                screen.blit(
                    self._font_sm.render(status, True, (45, 50, 60)),
                    (12, 28))
                model_lbl = os.path.basename(self.cfg.get("model_path") or "(none)")
                screen.blit(
                    self._font_sm.render(f"model: {model_lbl}", True, (55, 60, 70)),
                    (12, 48))

                pygame.display.flip()
                clock.tick(60)
        except KeyboardInterrupt:
            logger.info("Interrupted — shutting down")
        finally:
            # Save camera position and window size for next session
            try:
                self.cfg["camera"] = {
                    "azimuth": float(self.cam.azimuth),
                    "elevation": float(self.cam.elevation),
                    "distance": float(self.cam.distance),
                    "lookat": [float(v) for v in self.cam.lookat],
                }
                self.cfg["window"] = {
                    "width": self.win_w,
                    "height": self.win_h,
                }
                save_settings(self.cfg)
                logger.info("Camera and window state saved")
            except Exception as e:
                logger.warning("Failed to save camera/window state: %s", e)
            self.stop_valkey()
            if self.renderer is not None:
                try:
                    self.renderer.close()
                except Exception:
                    pass
            pygame.quit()


def main():
    try:
        StudioApp(parse_viewer_args()).run()
    except KeyboardInterrupt:
        pass
    except Exception:
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
