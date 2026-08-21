# 🧩 Hand-Tracked Jigsaw Puzzle Game

A gesture-controlled jigsaw puzzle game built with OpenCV and MediaPipe. No mouse needed — just your hand!

## Demo
> Pinch your fingers to grab a piece, drag it to the correct position, and watch it snap into place!

## Tech Stack
- **OpenCV** — game window + image rendering
- **MediaPipe** — real-time hand tracking
- **NumPy** — pixel math for piece slicing
- **Pygame** — background music

## Features
-  Pinch gesture to grab and drag pieces
-  Snap logic using Pythagorean distance
-  Background music
-  Live webcam feed in corner
-  Completion message on solving
-  4 difficulty levels (2x2 to 5x5)

## Setup

### Requirements
- Python 3.11
- Mac/Linux

### Install dependencies
pip install opencv-python mediapipe pygame numpy

### Run
python main.py

## Controls
- Pinch thumb + index finger — grab piece
- Move hand while pinching — drag piece
- Open fingers — release piece
- Press N — next level (after completing)
- Press Q — quit

## Project Structure
jigsaw_game/
├── main.py       # Main game loop
├── slicer.py     # Image slicing logic
├── images/       # Puzzle images (level1-4.jpg)
└── sounds/       # Background music

Ragini Rai
