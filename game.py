import cv2
import numpy as np
import random
from slicer import slice_image

WINDOW = "Jigsaw Puzzle"
CANVAS_SIZE = 600

def shuffle_pieces(pieces, piece_size):
    for p in pieces:
        p["current"] = [
            random.randint(0, CANVAS_SIZE - piece_size),
            random.randint(0, CANVAS_SIZE - piece_size)
        ]
    return pieces

def draw_canvas(pieces, piece_size):
    canvas = np.ones((CANVAS_SIZE, CANVAS_SIZE, 3), dtype=np.uint8) * 40
    for p in pieces:
        x, y = p["current"]
        canvas[y:y+piece_size, x:x+piece_size] = p["img"]
    return canvas

pieces, piece_size = slice_image("images/level1.jpg", 2)
pieces = shuffle_pieces(pieces, piece_size)

while True:
    canvas = draw_canvas(pieces, piece_size)
    cv2.imshow(WINDOW, canvas)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()