import cv2
import numpy as np
import random
import copy
from math import pi

# use cv2.floodFill to extract accessible area
def extract_access_area(read_path, kernel_size, seed_point, save_path):
    img = cv2.imread(read_path)
    h, w = img.shape[:2]
    mask = np.zeros((h + 2, w + 2, 1), np.uint8)
    blurred_img = cv2.GaussianBlur(img, kernel_size, 0)
    gray_img = cv2.cvtColor(blurred_img, cv2.COLOR_RGB2GRAY)
    # Binary
    _, binary_img = cv2.threshold(gray_img, 200, 255, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(binary_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    result = np.zeros((h, w, 3), np.uint8)
    cv2.drawContours(result, contours, -1, (0, 255, 0), 1)
    cv2.floodFill(result, mask, seed_point, (0, 120, 0), (0, 0, 0), (30, 30, 30), cv2.FLOODFILL_FIXED_RANGE)
    cv2.imwrite(save_path, result)
    return
