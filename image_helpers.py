import itertools
import cv2
import numpy as np

BoundingBox = tuple[float, ...]


def get_anchors(bounding_box: BoundingBox):
    x_center, y_center, width, height = [int(x) for x in bounding_box]

    left = int(x_center - (width / 2))
    top = int(y_center - (height / 2))
    right = left + width
    bottom = top + height

    return top, right, bottom, left


def empty_mask(shape: tuple[int, int]):
    return np.zeros(shape, dtype=np.uint8)


def single_hist_distance(h1: cv2.Mat, h2: cv2.Mat, method=cv2.HISTCMP_HELLINGER):
    return cv2.compareHist(h1, h2, method=method)


def is_grass(pixel: tuple[int, int, int], sensitivity=20):
    """Check if a pixel HUE matches grass hue"""
    rgb = np.array([[[0, 255, 0]]], np.uint8)
    hsv = cv2.cvtColor(rgb, cv2.COLOR_BGR2HSV)
    min_color = np.array([hsv[0][0][0] - sensitivity, 100, 70])
    max_color = np.array([hsv[0][0][0] + sensitivity, 255, 255])
    _pixel = cv2.cvtColor(np.array([[pixel]], np.uint8), cv2.COLOR_BGR2HSV)
    return min_color[0] <= _pixel[0][0][0] <= max_color[0]


def no_grass_mask(frame: np.ndarray, bounding_box: BoundingBox):
    mask = empty_mask(frame.shape[:2])
    top, right, bottom, left = get_anchors(bounding_box)
    mask[top:bottom, left:right] = 255
    vertical = range(top, bottom)
    horizontal = range(left, right)
    for y, x in itertools.product(vertical, horizontal):
        if is_grass(frame[y, x]):
            mask[y, x] = 0
    return mask


def draw_bounding_box(img: cv2.Mat, bounding_box: BoundingBox, color=(255, 0, 0)):
    center_x = int(bounding_box[0])
    center_y = int(bounding_box[1])
    width = int(bounding_box[2] / 2)
    height = int(bounding_box[3] / 2)
    top_left = (center_x - width, center_y - height)
    bottom_right = (center_x + width, center_y + height)
    return cv2.rectangle(img, top_left, bottom_right, color, 3)
