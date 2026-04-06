<div align="center">

```
██╗  ██╗██╗███╗   ██╗███████╗████████╗██╗ ██████╗ ██████╗ ██████╗ ███████╗
██║ ██╔╝██║████╗  ██║██╔════╝╚══██╔══╝██║██╔════╝██╔═══██╗██╔══██╗██╔════╝
█████╔╝ ██║██╔██╗ ██║█████╗     ██║   ██║██║     ██║   ██║██████╔╝█████╗  
██╔═██╗ ██║██║╚██╗██║██╔══╝     ██║   ██║██║     ██║   ██║██╔══██╗██╔══╝  
██║  ██╗██║██║ ╚████║███████╗   ██║   ██║╚██████╗╚██████╔╝██║  ██║███████╗
╚═╝  ╚═╝╚═╝╚═╝  ╚═══╝╚══════╝   ╚═╝   ╚═╝ ╚═════╝ ╚═════╝ ╚═╝  ╚═╝╚══════╝
```

**your hands are the interface. your face is the data. the screen obeys.**

![Python](https://img.shields.io/badge/Python-3.8+-1a1a2e?style=for-the-badge&logo=python&logoColor=00ffcc)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-1a1a2e?style=for-the-badge&logo=opencv&logoColor=00ffcc)
![MediaPipe](https://img.shields.io/badge/MediaPipe-Latest-1a1a2e?style=for-the-badge&logo=google&logoColor=00ffcc)
![License](https://img.shields.io/badge/License-Experimental-1a1a2e?style=for-the-badge&logoColor=ff0055)

</div>

---

## ⚡ what is this

**KinetiCore** is an experimental computer vision sandbox that collapses the distance between physical intent and digital execution. No controllers. No peripherals. No latency theater.

Just your bare hands talking directly to the machine — and the machine actually listening.

Built on **MediaPipe** + **OpenCV**, it runs real-time hand landmark tracking, biometric face mapping, and gesture-driven spatial interactions at frame speed. This isn't a demo. It's a proof of something.

---

## 🧬 modules

Each script is a standalone experiment. Run them independently, break them, combine them.

| File | What it does |
|---|---|
| `gestures.py` | Core gesture engine — detects pointing, rock, OK, call-me, open palm in real-time. Overlays labeled bounding boxes on both hands and face simultaneously. |
| `devil.py` | Triggers eye-crop captures with a 🤘 gesture. Freezes and overlays the eye images back onto the live feed. OK clears the stack. Face mesh renders as a green constellation across your skin. |
| `interactive_molecule.py` | Finger-driven 3D molecule manipulation. Move it, rotate it, own it. |
| `index_cubes.py` | Your index finger becomes a spatial cursor controlling floating cubes in the frame. |
| `half_face.py` | Splits and mirrors facial geometry in real-time. Symmetry is a lie — this proves it. |
| `mirror_screen.py` | Reflects and warps the webcam feed through landmark anchors. You, distorted. |
| `dithering_filter.py` | Applies a raw dithering shader to the live camera feed. Turns your face into a cyberpunk printout. |
| `simple_monkey.py` | Entry-point chaos. A minimal gesture loop to test your setup before going deeper. |

---

## 🔧 setup

```bash
# clone the repo
git clone https://github.com/mavroul1s/KinetiCore.git
cd KinetiCore

# install dependencies
pip install opencv-python mediapipe numpy

# run any module
python gestures.py
python devil.py
python interactive_molecule.py
# etc.
```

> Requires a working webcam. Everything runs locally — no cloud, no API calls, no surveillance theater.

---

## 🖐 gesture reference

| Gesture | Trigger | Action |
|---|---|---|
| ☝️ Pointing | Index up, all others down | `execute` label |
| 🤙 Call Me | Thumb + pinky up | `call me` label |
| 👌 OK | Thumb + index pinched (dist < 0.05) | `ok` / clear state |
| 🤘 Rock | Index + pinky up, middle + ring down | `yeah` / capture eye |

---

## 🧠 architecture

```
webcam feed
    │
    ▼
OpenCV frame capture (flipped, BGR→RGB)
    │
    ├──▶ MediaPipe Hands ──▶ 21 landmark skeleton ──▶ gesture classifier
    │
    └──▶ MediaPipe Face Detection / FaceMesh ──▶ biometric overlay
                                                      │
                                                      ▼
                                              rendered output frame
```

Latency is kept near-zero by processing each frame inline with no async overhead. What you see is what's happening.

---

## 🔭 what's next

- [ ] Gesture-to-keystroke mapping (control your OS with your hands)
- [ ] Persistent gesture sequences (macros via motion)
- [ ] Depth estimation via landmark z-axis
- [ ] Multi-hand interaction events
- [ ] WebSocket output stream for external integrations

---

## ⚠️ disclaimer

This is raw, experimental code. It was built to explore ideas, not to be production-safe. Expect rough edges, undocumented behavior, and scripts that do strange things to your webcam feed. That's the point.

---

<div align="center">

*bridging physical intent and execution.*

**`[ KinetiCore — the body as input device ]`**

</div>
