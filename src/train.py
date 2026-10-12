from anomalib.data import Folder
from anomalib.engine import Engine
from config import DATASET_DIR, OUTPUT_DIR, IMAGE_SIZE
from model_factory import build_model

# Default PatchCore preprocessing center-crops the image, which could cut off
# the corner where defects are. Resize only, no crop.
pre = PreProcessor(transform=Compose([
    Resize(IMAGE_SIZE, antialias=True),
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
]))

datamodule = Folder(
    name="seal",
    root=DATASET_DIR,
    normal_dir="train/good",
    abnormal_dir="test/defect",
    normal_test_dir="test/good",
    train_batch_size=4,
    eval_batch_size=4,
    num_workers=0,  # 0 avoids multiprocessing issues on Windows
)

model = build_model()


engine = Engine(default_root_dir=str(OUTPUT_DIR))
engine.fit(model=model, datamodule=datamodule)
print(engine.test(model=model, datamodule=datamodule))
