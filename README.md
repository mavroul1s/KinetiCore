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
Gemini said
A real-time webcam tool tracking faces and hand gestures. It identifies specific signs (pointing, OK, rock, call me) and overlays custom text labels.

---

### `devil.py` — Eye Snapper
Tracks face and hand landmarks. Using a "rock" gesture captures and overlays eye crops, while an "OK" gesture clears them.

---

### `index_cubes.py` — Finger Painter
Opens two windows side by side: *"me?"* (your raw camera feed) and *"you?"* (a black canvas). Point ☝️ your index finger and colored cubes spawn at your fingertip, painting the canvas as you move. Flash 🤘 rock to clear everything and start over. The canvas persists between frames — it only resets when you tell it to.

---

### `half_face.py` — Split Mesh
Renders the full face mesh tesselation but **only on the left half of your face**, using your nose as the dividing line. The right side stays clean. Your hands are tracked too, drawn as a black skeleton overlay. A simple but striking visual — half human, half wireframe.

---

### `interactive_molecule.py` — Hand-Driven Molecule
A 3D molecule visualization you control with your hands. Move your hand around and the molecule follows, rotates, and responds to your position in space.

---

### `mirror_screen.py` — Mirror Mode
Reflects and warps the webcam feed using facial landmark anchors. The effect distorts and mirrors your image in real-time as your face moves.

---

### `dithering_filter.py` — Dither Cam
A desktop image dithering tool featuring Floyd-Steinberg, JJN, and Atkinson algorithms. It offers a dark GUI, live previews, and threshold control.
---

### `simple_monkey.py` — Starter Demo
The entry point. A minimal gesture loop — good for testing that your camera and MediaPipe are working before running the heavier scripts.

---

## Stack

- [MediaPipe](https://mediapipe.dev/) — hand + face landmark detection
- [OpenCV](https://opencv.org/) — frame capture, rendering, display
- [NumPy](https://numpy.org/) — canvas and matrix operations

---

*Experimental. Expect rough edges. That's the point.*
