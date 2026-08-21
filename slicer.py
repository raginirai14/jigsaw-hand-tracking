import cv2
import numpy as np

def slice_image(image_path, grid_size):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (600, 600))
    piece_size = 600 // grid_size
    pieces = []

    for row in range(grid_size):
        for col in range(grid_size):
            x = col * piece_size
            y = row * piece_size
            piece = img[y:y+piece_size, x:x+piece_size]
            correct_pos = (x, y)
            pieces.append({"img": piece, "correct": correct_pos, "current": [x, y]})

    return pieces, piece_size

