# LegacyMotionEditor

[English](README.md) | [日本語](README_jp.md)

<img src="./doc/LegacyMotionEditoralpha.png" width=800>

**LegacyMotionEditor** (hereafter **LME**) is a **node-graph motion editor** for robots.  
It loads URDF / MJCF models and lets you assemble poses into a timeline using a node graph.  
It supports intuitive pose creation with a 3D preview, and comes bundled with MuJoCo Studio for real-time verification on the MuJoCo physics simulator. Gamepad control is also supported for free robot operation.

**Version:** 0.0.2  
**Author:** Izumi Ninagawa  
**License:** MIT — Copyright (c) 2026 Izumi Ninagawa (see [`LICENSE`](LICENSE))  
Third-party packages keep their own licenses (PySide6: LGPL, pygame-ce: LGPL-2.1).  
Partially based on [merimujoco](https://github.com/holypong/merimujoco/blob/main/README.md).

---
<img src="./doc/img1.png" width=600>

## File Structure

| File | Role |
|---|---|
| `LegacyMotionEditor.py` | Main editor UI |
| `LegacyMotionEditor_Utils.py` | Shared helpers, pad monitor, walk runtime |
| `LegacyMotionEditor_MuJoCoStudio.py` | Lightweight Valkey → MuJoCo preview |
| `LegacyMotionEditor_CodeEditor.py` | ProjectCode inline editor |
| `LegacyMotionEditor_Importer.py` | URDF / MJCF importer |
| `RobotLabelBridge.py` | Joint / link name canonicalization |
| `requirements.txt` | Python dependencies |

---

## Requirements

- **Python 3.10+**

| Group | Packages |
|---|---|
| Core editor | `numpy`, `Qt.py`, `PySide6`, `vtk`, `NodeGraphQt`, `trimesh`, `pycollada` |
| Optional: Valkey streaming | `valkey` |
| Optional: Pad / MuJoCo Studio | `pygame-ce`, `mujoco` |

---

## Installation

[`uv`](https://docs.astral.sh/uv/) is recommended.

```bash
# Install uv (once)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create venv and install dependencies
cd LegacyMotionEditor
uv venv --python 3.11
source .venv/bin/activate          # Windows: .venv\Scripts\activate
uv pip install -r requirements.txt
export QT_PREFERRED_BINDING=PySide6   # Windows: set QT_PREFERRED_BINDING=PySide6
```

### pip alternative

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export QT_PREFERRED_BINDING=PySide6
```

### Installing Valkey

Using Valkey enables motion playback on the MuJoCo physics simulator.

> Official documentation: https://valkey.io/docs/installation/

```bash
# macOS (Homebrew)
brew install valkey
valkey-server          # start (default port 6379)

# Ubuntu / Debian
sudo apt install valkey-server
sudo systemctl start valkey

# Windows (Docker recommended)
docker run -d -p 6379:6379 valkey/valkey

# Python client
pip install valkey
```

LME connects to `127.0.0.1:6379` by default. To change this, open MuJoCo Studio Settings (press `I`).

### Run

```bash
python LegacyMotionEditor.py
```

---

## Getting Started

1. Click **Load Project** in the left menu and open `LME_sample_save.xml` from the `save/` directory.
2. Select an action from the **Action** dropdown at the top.
3. Double-click any Pose node and confirm you can adjust joints with the sliders.
4. Confirm you can also manipulate joints directly in the 3D view.
5. Check **Valkey** in the top-right of the 3D view — joint data begins streaming to the in-memory database.
6. Press the **▶︎_** button below the 3D view to play back the full motion with action transitions.
7. Click **MuJoCo Studio** in the left menu to open MuJoCo Studio.
8. Click the **Pad** button to open the GamePad window.
9. Press the D-pad down button on the pad — the robot in MuJoCo will respond and move.

---

## Features

### Actions

Select an action from the dropdown in the top-left of the screen.  
An action is one unit of motion that bundles multiple poses.  
Actions are connected via Jump nodes.  
The action that runs at startup is named **Boot**; the looping entry point is named **Base**.

### Node Graph

The pose timeline within an action is represented as nodes.  
Drag from a node's port to draw a connection line, then connect it to the next node.  
Playback follows connection order.

| Node type | Description |
|---|---|
| **Pose** | Single-frame joint angle snapshot |
| **Define** | Variable definition (stores shared poses or constants by name) |
| **Branching** | Left/right branch based on UserVal / Pad values |
| **Command** | Insert a playback control command |
| **Mix** | Blend correction values into joint angles |
| **Jump** | Jump to any action or function |
| **Code (ProjectCode)** | Custom Python snippet (Walk IK, etc.) |

- Double-click a node to rename or edit details
- Right-click a node to delete / duplicate / manage connections

---

### 3D View

- Displays STL / MJCF models in real time
- Drag joints directly in the view to manipulate the pose
- **Home** — set pose to the home position (configured in Config)
- **Zero** — reset all joints to zero degrees
- **L↔R** — swap left and right joint angles
- **Reframe** — reset the camera view
- **≡** — set only upper body / lower body to the home position

---

### Joint Sliders

- Double-clicking a node opens the Joint Sliders floating window
- Set any joint angle via slider or numeric input
- **Step** — move joints in fixed angle increments
- **Pair** mode — change left and right symmetrically at the same time
- **Opp** mode — change left and right in opposite directions at the same time
- **Group** preset — switch joint groups (upper body / lower body, etc.)
- **Easing** — set interpolation type per joint or all at once

---

### Playback & Walk Controller

- **|◀︎** return to action start / **▶︎.** play action / **▶︎_** play with action transitions / **■** stop
- 3D view and Valkey stream update in real time during playback
- Implement walking motion by writing Walk IK (`walk_ik_step`) inside **ProjectCode**
- Gamepad input controls branching, speed, and stop at runtime

---

### Project Save / Load

- **Save Project** — save the node graph, joint data, and robot name to XML
- **Load Project** — fully restore from a saved XML (robot name, node positions, and connections included)
- **Export Motion** — export motion data to another format (not yet implemented)
- **Export Cartridge** — write a logic cartridge for PhysicalOn
- Session state is auto-saved on exit (`save/_lme_session.xml`)

---

### Valkey Streaming

- Streams joint angles in real time via [Valkey](https://valkey.io/) (Redis-compatible) during editing and playback
- MuJoCo Studio and physical robots (`PhysicalOn`) subscribe as receivers
- Toggle streaming with the **Valkey** checkbox in the top-right of the 3D view

---

### Pad Monitor (Gamepad)

- Click **Pad** to open the gamepad monitor (requires `pygame-ce`)
- Button / stick input is available as `Pad_*` / `UserVal_*` conditions in Branching nodes
- Toggle continuous PC pad monitoring with the checkbox

<img src="./doc/img2.png" width=400>

---

### MuJoCo Studio

- Click **MuJoCo Studio** to launch `LegacyMotionEditor_MuJoCoStudio.py` as a separate process
- Displays the Valkey angle stream in real time on the MuJoCo physics simulator
- Built-in lightweight scene with test grid

<img src="./doc/img3.png" width=800>

---

### Config

- Click **Config** to open the settings dialog
- Configure undo history limit, debug log output, Valkey connection, and more

---

### Shortcuts

- `Ctrl+Z` / `Ctrl+Y` / `Ctrl+Shift+Z` — Undo / Redo
- `Ctrl+C` / `Ctrl+V` / `Ctrl+D` — Copy / Paste / Duplicate nodes
- `Del` — Delete selected nodes

---

## RobotLabelBridge

`RobotLabelBridge.py` is a module that maps robot joint / link names to **canonical short names** (e.g. `l_knee_yp`).  
It can be used from LME to batch-rename joints in a loaded model, and can also run as a standalone tool.  
(Full documentation will be available in a separate repository.)
