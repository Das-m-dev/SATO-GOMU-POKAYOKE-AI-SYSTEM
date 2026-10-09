from torchvision.transforms.v2 import Compose, Resize, Normalize
from anomalib.models import Patchcore
from anomalib.pre_processing import PreProcessor
from config import IMAGE_SIZE


def build_model():
    pre = PreProcessor(transform=Compose([
        Resize(IMAGE_SIZE, antialias=True),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]))
    return Patchcore(
        backbone="wide_resnet50_2",
        layers=("layer2", "layer3"),
        coreset_sampling_ratio=0.1,
        pre_processor=pre,
    )