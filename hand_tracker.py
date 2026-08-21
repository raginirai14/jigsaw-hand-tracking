import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np

# Download the hand landmarker model
import urllib.request
import os

model_path = os.path.expanduser("~/jigsaw_game/hand_landmarker.task")
if not os.path.exists(model_path):
    print("Downloading hand landmarker model...")
    urllib.request.urlretrieve(
        "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
        model_path
    )
    print("✅ Model downloaded!")

base_options = python.BaseOptions(model_asset_path=model_path)
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=1
)
detector = vision.HandLandmarker.create_from_options(options)

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = detector.detect(mp_image)

    if result.hand_landmarks:
        h, w, _ = frame.shape
        landmarks = result.hand_landmarks[0]

        thumb = landmarks[4]
        index = landmarks[8]

        tx, ty = int(thumb.x * w), int(thumb.y * h)
        ix, iy = int(index.x * w), int(index.y * h)

        cv2.circle(frame, (tx, ty), 10, (0, 255, 0), -1)
        cv2.circle(frame, (ix, iy), 10, (0, 0, 255), -1)

        dist = ((tx - ix)**2 + (ty - iy)**2) ** 0.5
        if dist < 40:
            cv2.putText(frame, "PINCH!", (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    cv2.imshow("Hand Tracker", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()