from torchvision.transforms.v2 import Compose, Resize, Normalize
from anomalib.data import Folder
from anomalib.engine import Engine
from anomalib.models import Patchcore
from anomalib.pre_processing import PreProcessor
from config import DATASET_DIR, OUTPUT_DIR, IMAGE_SIZE

#Default PatchCore preprocessing center-crops the image, which could cut off
# the corner where defects are. Resize o
pre = PreProcessor(transform=Compose([
    Resize(IMAGE_SIZE, antialias=True),
    Normalize(mean=[0.484, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
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

model = Patchcore(
    backbone="wide_resnet50_2",
    layers=("layer2", "layer3"),
    coreset_sampling_ratio=0.1,
    pre_processor=pre,
)

engine = Engine(default_root_dir=str(OUTPUT_DIR))
engine.fit(model=model, datamodule=datamodule)
print(engine.test(model=model, datamodule=datamodule))
