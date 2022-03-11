from .deep_visions import DeepVisionsYatav, DeepVisionsTest, DeepVisionsJeju
from .data_loader import VideoDataset

__all__ = ["init_dataset", "VideoDataset"]

__factory = {
    "dv_yatav": DeepVisionsYatav,
    "dv_test": DeepVisionsTest,
    "dv_jeju": DeepVisionsJeju,
}


def init_dataset(name, *args, **kwargs):
    if name not in __factory.keys():
        raise KeyError(f"Unknown datasets: {name}...")
    return __factory[name](*args, **kwargs)
