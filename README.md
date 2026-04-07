```
██╗  ██╗██╗███╗   ██╗███████╗████████╗██╗ ██████╗ ██████╗ ██████╗ ███████╗
██║ ██╔╝██║████╗  ██║██╔════╝╚══██╔══╝██║██╔════╝██╔═══██╗██╔══██╗██╔════╝
█████╔╝ ██║██╔██╗ ██║█████╗     ██║   ██║██║     ██║   ██║██████╔╝█████╗
██╔═██╗ ██║██║╚██╗██║██╔══╝     ██║   ██║██║     ██║   ██║██╔══██╗██╔══╝
██║  ██╗██║██║ ╚████║███████╗   ██║   ██║╚██████╗╚██████╔╝██║  ██║███████╗
╚═╝  ╚═╝╚═╝╚═╝  ╚═══╝╚══════╝   ╚═╝   ╚═╝ ╚═════╝ ╚═════╝ ╚═╝  ╚═╝╚══════╝
```

> *An experimental sandbox for low-latency computer vision — bare-hand spatial manipulation and biometric mapping. No controllers. Just you.*

---

## What is this?

KinetiCore is a collection of real-time computer vision experiments built with **Python**, **OpenCV**, and **MediaPipe**. Each script is a standalone demo that uses your webcam to track your hands and face — then does something interesting with that data.

Everything runs locally. No internet. No API. Just your camera and your hands.

---

## Setup

```bash
git clone https://github.com/mavroul1s/KinetiCore.git
cd KinetiCore
pip install opencv-python mediapipe numpy
```

Then run any script directly:

```bash
python gestures.py
python devil.py
# etc.
```

---

## The Scripts

### `gestures.py` — Gesture + Face Detection

A real-time webcam tool tracking faces and hand gestures. It identifies specific signs (pointing, OK, rock, call me) and overlays custom text labels.

---

### `devil.py` — Eye Snapper

Tracks face and hand landmarks. Using a "rock" gesture captures and overlays eye crops, while an "OK" gesture clears them.

---

### `index_cubes.py` — Finger Painter

A real-time webcam tool tracking face and hand landmarks. It renders facial meshes in green and hand skeletons in red on a separate black canvas. Pointing your index finger lets you draw random colorful blocks, while a "rock" gesture clears the canvas. The live feed and the drawing canvas are shown in separate windows.

---

### `half_face.py` — Split Mesh

A real-time webcam tool for hand and face tracking. It applies custom visual styles to the detected landmarks. Hand points and connections are drawn completely in black. For the face, it dynamically uses the nose tip to bisect the mesh, rendering the facial tesselation and landmarks exclusively on the left half of the user's face.

---

### `interactive_molecule.py` — Hand-Driven Molecule

An interactive 3D molecule viewer using webcam hand tracking. Rotate the methane molecule with an open hand, or pinch to grab and stretch atoms.
---

### `mirror_screen.py` — Mirror Mode

Reflects and warps the webcam feed using facial landmark anchors. The effect distorts and mirrors your image in real-time as your face moves.

---

### `dithering_filter.py` — Dither Cam

A desktop image dithering tool featuring Floyd-Steinberg, JJN, and Atkinson algorithms. It offers a dark GUI, live previews, and threshold control.
---

### `simple_monkey.py` — Starter Demo

A webcam tracker for faces and hands. It toggles between two displayed images based on whether the index finger enters the face bounding box.
---

## Stack

- [MediaPipe](https://mediapipe.dev/) — hand + face landmark detection
- [OpenCV](https://opencv.org/) — frame capture, rendering, display
- [NumPy](https://numpy.org/) — canvas and matrix operations

---

*Experimental. Expect rough edges. That's the point.*
