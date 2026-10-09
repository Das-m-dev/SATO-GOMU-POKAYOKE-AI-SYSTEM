import cv2, json
from pathlib import Path

img = cv2.imread("data_raw/ng/tear/NG_1.jpg")
scale = 0.25
small = cv2.resize(img, None, fx=scale, fy=scale)

x, y, w, h = cv2.selectROI("Drag a box around the corner/lip area, press ENTER", small)
cv2.destroyAllWindows()

roi = {"x": int(x / scale), "y": int(y / scale),
       "w": int(w / scale), "h": int(h / scale)}
Path("configs").mkdir(exist_ok=True)
json.dump(roi, open("configs/roi.json", "w"), indent=2)
print("Saved ROI (full-resolution pixels):", roi)