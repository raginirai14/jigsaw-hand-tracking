import cv2
import numpy as np
import random
import os
import urllib.request
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import pygame
from slicer import slice_image

# Init music
pygame.mixer.init()
music_path = os.path.expanduser("~/jigsaw_game/sounds/background.mp3")
if os.path.exists(music_path):
    pygame.mixer.music.load(music_path)
    pygame.mixer.music.set_volume(0.5)
    pygame.mixer.music.play(-1)

CANVAS_SIZE = 600
SNAP_THRESHOLD = 200
SMOOTH = 0.35

LEVELS = [
    ("images/level1.jpg", 2),
    ("images/level2.jpg", 3),
    ("images/level3.jpg", 4),
]

model_path = os.path.expanduser("~/jigsaw_game/hand_landmarker.task")
if not os.path.exists(model_path):
    print("Downloading model...")
    urllib.request.urlretrieve(
        "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
        model_path
    )

base_options = python.BaseOptions(model_asset_path=model_path)
options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=1)
detector = vision.HandLandmarker.create_from_options(options)
def load_level(idx):
    img_path, grid = LEVELS[idx]
    pieces, piece_size = slice_image(img_path, grid)
    positions = []
    for i, p in enumerate(pieces):
        row = i // grid
        col = i % grid
        # Spread pieces evenly so none go off screen
        p["current"] = [
            col * (CANVAS_SIZE // grid),
            row * (CANVAS_SIZE // grid)
        ]
        p["snapped"] = False
    random.shuffle(pieces)
    # Now scramble positions slightly
    for p in pieces:
        p["current"] = [
            random.randint(10, CANVAS_SIZE - piece_size - 10),
            random.randint(10, CANVAS_SIZE - piece_size - 10)
        ]
    return pieces, piece_size
def draw_canvas(pieces, piece_size, dragging, mx, my, pinching, completed):
    canvas = np.ones((CANVAS_SIZE, CANVAS_SIZE, 3), dtype=np.uint8) * 40

    # Grid lines
    grid_size = CANVAS_SIZE // piece_size
    for i in range(grid_size + 1):
        cv2.line(canvas, (i * piece_size, 0), (i * piece_size, CANVAS_SIZE), (80, 80, 80), 1)
        cv2.line(canvas, (0, i * piece_size), (CANVAS_SIZE, i * piece_size), (80, 80, 80), 1)

    # Draw pieces — dragged piece always on top
    draw_order = [i for i in range(len(pieces)) if i != dragging]
    if dragging is not None:
        draw_order.append(dragging)

    for i in draw_order:
        p = pieces[i]
        x, y = p["current"]
        x = max(0, min(x, CANVAS_SIZE - piece_size))
        y = max(0, min(y, CANVAS_SIZE - piece_size))

        piece_img = p["img"].copy()

        if p["snapped"]:
            cv2.rectangle(piece_img, (0, 0), (piece_size-1, piece_size-1), (0, 255, 0), 3)
        elif i == dragging:
            cv2.rectangle(piece_img, (0, 0), (piece_size-1, piece_size-1), (0, 255, 255), 4)

        canvas[y:y+piece_size, x:x+piece_size] = piece_img

    # Cursor
    color = (0, 255, 0) if pinching else (0, 255, 255)
    cv2.circle(canvas, (mx, my), 10, color, -1)
    cv2.circle(canvas, (mx, my), 16, color, 2)

    # Pinch label
    if pinching:
        cv2.putText(canvas, "PINCHING", (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # Completion overlay
    if completed:
        overlay = canvas.copy()
        cv2.rectangle(overlay, (50, 200), (550, 400), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, canvas, 0.3, 0, canvas)
        cv2.putText(canvas, "PUZZLE COMPLETE!", (65, 275),
            cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3)
        cv2.putText(canvas, "Amazing! All pieces placed!", (130, 320),
            cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
        cv2.putText(canvas, "Press N  next level    Q  quit", (100, 365),
            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (200, 200, 200), 1)

    return canvas

def get_pinch(landmarks, w, h):
    thumb = landmarks[4]
    index = landmarks[8]
    tx, ty = int(thumb.x * w), int(thumb.y * h)
    ix, iy = int(index.x * w), int(index.y * h)
    cx, cy = (tx + ix) // 2, (ty + iy) // 2
    dist = ((tx - ix)**2 + (ty - iy)**2) ** 0.5
    return cx, cy, dist < 50

def check_snap(pieces, piece_size):
    for p in pieces:
        if p["snapped"]:
            continue
        cx, cy = p["current"]
        gx, gy = p["correct"]
        dist = ((cx - gx)**2 + (cy - gy)**2) ** 0.5
        print(f"dist from correct: {dist:.1f}")
        if dist < SNAP_THRESHOLD:
            p["current"] = list(p["correct"])
            p["snapped"] = True
            print("✅ SNAPPED!")

def all_snapped(pieces):
    return all(p["snapped"] for p in pieces)

# Init
cap = cv2.VideoCapture(0)
cam_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
cam_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

level_idx = 0
pieces, piece_size = load_level(level_idx)
dragging = None
offset = [0, 0]
completed = False
smooth_x, smooth_y = CANVAS_SIZE // 2, CANVAS_SIZE // 2
pinching = False
mx, my = CANVAS_SIZE // 2, CANVAS_SIZE // 2

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = detector.detect(mp_image)

    pinching = False

    if result.hand_landmarks:
        landmarks = result.hand_landmarks[0]
        cx, cy, pinching = get_pinch(landmarks, cam_w, cam_h)

        raw_x = int(cx / cam_w * CANVAS_SIZE)
        raw_y = int(cy / cam_h * CANVAS_SIZE)

        smooth_x = int(smooth_x * (1 - SMOOTH) + raw_x * SMOOTH)
        smooth_y = int(smooth_y * (1 - SMOOTH) + raw_y * SMOOTH)
        mx, my = smooth_x, smooth_y

        if not completed:
            if pinching:
                if dragging is None:
                    for i, p in enumerate(pieces):
                        px, py = p["current"]
                        if not p["snapped"] and px <= mx <= px + piece_size and py <= my <= py + piece_size:
                            dragging = i
                            offset = [mx - px, my - py]
                            break
                else:
                    pieces[dragging]["current"] = [mx - offset[0], my - offset[1]]
            else:
                if dragging is not None:
                    check_snap(pieces, piece_size)
                    dragging = None

    if not completed and all_snapped(pieces):
        completed = True
        print("🎉 LEVEL COMPLETE!")

    canvas = draw_canvas(pieces, piece_size, dragging, mx, my, pinching, completed)

    # Camera in corner
    small_cam = cv2.resize(frame, (200, 150))
    canvas[0:150, 400:600] = small_cam

    cv2.imshow("Jigsaw Puzzle", canvas)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('n') and completed:
        level_idx = (level_idx + 1) % len(LEVELS)
        pieces, piece_size = load_level(level_idx)
        dragging = None
        completed = False

cap.release()
cv2.destroyAllWindows()
pygame.mixer.music.stop()