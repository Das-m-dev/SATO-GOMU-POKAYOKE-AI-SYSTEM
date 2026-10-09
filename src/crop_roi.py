import json, cv2
from pathlib import Path

cfg = json.load(open("configs/roi.json"))
x, y, w, h = cfg["x"], cfg["w"], cfg["h"]

for p in Path("data_raw").rglob("*.jpg"):
    img = cv2.imread(str(p))
    crop = img[y:y + h, x:x + w]
    dst = Path("data_roi") / p.relative_to("data_raw")
    dst.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(dst), crop)
    print("saved", dst, crop.shape)