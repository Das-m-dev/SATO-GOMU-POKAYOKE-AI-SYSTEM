from pathlib import Path
import cv2
import pandas as pd

EXTS = {".jpg", ".jpeg", ".png", ".bmp"}
rows = []

for p in Path("data_raw").rglob("*"):
    if p.suffix.lower() not in EXTS:
        continue
    img = cv2.imread(str(p)) #load the image as a numpy array of numbers
    if img is None:
        print("Unreadable:", p)
        continue
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) #convert to one brightness channel
    parts = p.relative_to("data_raw").parts
    rows.append({
        "path": str(p),
        "label": parts[0],
        "defect": parts[1] if parts[0] == "ng" and len(parts) > 2 else "", #ok or ng
        "h": img.shape[0],
        "w": img.shape[1],
        "brightness": float(gray.mean()), #avg pixel value (0 dark to 225 bright)
        "sharpness": float(cv2.Laplacian(gray, cv2.CV_64F).var()), #blur measure (low = blurry)
    })

df = pd.DataFrame(rows)
Path("outputs").mkdir(exist_ok=True)
df.to_csv("outputs/audit.csv", index=False)

print("\nCounts by label/defect:")
print(df.groupby(["label", "defect"]).size())
print("\nImage sizes:")
print(df.groupby(["h", "w"]).size())
print("\nBrightness / sharpness by label:")
print(df.groupby("label")[["brightness", "sharpness"]].agg(["mean", "std"]).round(1))