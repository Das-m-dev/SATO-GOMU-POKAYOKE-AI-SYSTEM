import cv2, json
from pathlib import Path

img = cv2.imread("data_raw/ng/tear/NG_1.jpg")
scale = 0.25
small = cv2.resize(img, None, fx=scale, fy=scale)

x, y, w, h = cv2.selectROI("Drag a box around the corner/lip area, press ENTER", sma)