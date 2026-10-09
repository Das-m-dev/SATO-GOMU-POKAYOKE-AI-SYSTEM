import sys
from pathlib import Path

import cv2
import numpy as np
from anomalib.engine import Engine

from config import OUTPUT_DIR, ROOT
from model_factory import build_model

target = sys.argv[1]                       # an image file or a folder
ckpt = max(OUTPUT_DIR.rglob("*.ckpt"), key=lambda p: p.stat().st_mtime)
print("Using checkpoint:", ckpt)

engine = Engine(default_root_dir=str(OUTPUT_DIR))
batches = engine.predict(model=build_model(), data_path=target, ckpt_path=str(ckpt))

out_dir = ROOT / "outputs" / "predictions"
out_dir.mkdir(parents=True, exist_ok=True)

for batch in batches:
    for i, path in enumerate(batch.image_path):
        score = float(batch.pred_score[i])
        label = "NG" if bool(batch.pred_label[i]) else "OK"
        print(f"{Path(path).name:35s} score={score:.3f}  ->  {label}")

        # Heatmap overlay on the original image
        amap = batch.anomaly_map[i].squeeze().cpu().numpy()
        amap = (amap - amap.min()) / (amap.max() - amap.min() + 1e-8)
        heat = cv2.applyColorMap((amap * 255).astype(np.uint8), cv2.COLORMAP_JET)
        img = cv2.imread(str(path))
        heat = cv2.resize(heat, (img.shape[1], img.shape[0]))
        overlay = cv2.addWeighted(img, 0.6, heat, 0.4, 0)
        cv2.imwrite(str(out_dir / f"{Path(path).stem}_{label}.jpg"), overlay)