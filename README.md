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
The core demo. Opens your webcam and reads your hand in real-time. It recognizes **pointing**, **rock** 🤘, **OK** 👌, and **call me** 🤙, and displays a labeled box next to your hand whenever a gesture is detected. At the same time, it draws a bounding box around your face. Two trackers running simultaneously, no lag.

---

### `devil.py` — Eye Snapper
Flash a 🤘 rock sign and it **photographs your eye** — crops it from the live feed and stamps it back onto the screen, right where your eye is. Do it again and again and the eyes stack up. Flash **OK** 👌 to wipe them all. Your face is covered in a green dot constellation (478 face mesh landmarks) the whole time.

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
Applies a live dithering shader to your camera feed. Turns the video into a raw, pixelated, high-contrast render — like looking at yourself through a broken CRT.

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
