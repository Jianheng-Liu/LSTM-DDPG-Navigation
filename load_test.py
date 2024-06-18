import cv2
import numpy as np


img = cv2.imread('maps/env.png')
print(type(img))
print(img.shape)
h, w = img.shape[:2]
mask = np.zeros((h+2, w+2, 1), np.uint8)
print(img[23][111])
blurred_img = cv2.GaussianBlur(img, (5, 5), 0)
gray_img = cv2.cvtColor(blurred_img, cv2.COLOR_RGB2GRAY)
# Binary
_, binary_img = cv2.threshold(gray_img, 200, 255, cv2.THRESH_BINARY)
contours, hierarchy = cv2.findContours(binary_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
result = np.zeros((h, w, 3), np.uint8)
cv2.drawContours(result, contours, -1, (0, 255, 0), 1)
cv2.floodFill(result, mask, (125, 100), (0, 120, 0), (0, 0, 0), (30, 30, 30), cv2.FLOODFILL_FIXED_RANGE)
cv2.imwrite('maps/environments/env.png', result)
cv2.imshow("Map", img)
cv2.waitKey()
cv2.destroyAllWindows()
cv2.imshow("Map", result)
cv2.waitKey()
cv2.destroyAllWindows()

