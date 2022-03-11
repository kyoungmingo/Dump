import re
from pathlib import Path

import numpy as np


class DeepVisionsBaseDataset:
    def __init__(self, root, dataset_dir, verbose=True, *args, **kwargs):
        self.dataset_dir = dataset_dir
        self.dataset_dir = Path(root) / self.dataset_dir

        self.train_dir = self.dataset_dir / "train"
        self.test_dir = self.dataset_dir / "test"
        self.labels = self.dataset_dir / "labels.txt"

        self._check_before_run()

        self.classes = np.genfromtxt(
            self.labels, delimiter=None, dtype=None, encoding="UTF-8"
        ).tolist()
        self.class_idx = {cls_type: idx for idx, cls_type in enumerate(self.classes)}

        self.num_classes = len(self.classes)

        self.train = self._process_dir(self.train_dir)
        self.test = self._process_dir(self.test_dir)

        self.num_train_vids, self.num_test_vids = len(self.train), len(self.test)

        if verbose:
            self.print_dataset_statistics(
                self.num_train_vids, self.num_test_vids, self.num_classes
            )

    def _check_before_run(self):
        if not self.dataset_dir.is_dir():
            raise RuntimeError(
                f"Dataset is not available, Please check whethter {self.dataset_dir} does exist."
            )
        if not self.train_dir.is_dir():
            raise RuntimeError(
                f"Train list is not available. Please check whether '{self.train_dir}' does exist."
            )
        if not self.test_dir.is_dir():
            raise RuntimeError(
                f"Test list is not available. Please check whether '{self.test_dir}' does exist."
            )

        if not self.labels.is_file():
            raise RuntimeError(
                f"Label file is not available. Please check whether '{self.labels}' does exist."
            )

    def _process_dir(self, data_dir: Path):
        pattern = re.compile(r"([\w]+)/([-_~\w]+).mp4")
        vid_list = data_dir.glob("**/*.mp4")

        datasets = []
        for vid_path in vid_list:
            str_vid_path = str(vid_path)
            cls_type, _ = pattern.search(str_vid_path).groups()
            datasets.append((vid_path, self.class_idx[cls_type]))

        return datasets

    def print_dataset_statistics(self, num_train_vids, num_test_vids, num_classes):
        print("Dataset statistics:")
        print("  ------------------------------------------")
        print(f"  Total number of classses: {num_classes:4d}")
        print(f"  Data type: 'train' | # of videos: {num_train_vids:8d}")
        print(f"  Data type:  'test' | # of videos: {num_test_vids:8d}")
        print("  ------------------------------------------")
