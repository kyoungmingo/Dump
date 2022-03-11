from random import randint

import numpy as np
import cv2

import torch
from torch.utils.data import Dataset

from .box_detector_torch import box_ext

class VideoDataset(Dataset):
    def __init__(
        self,
        dataset,
        num_sample_frames,
        random_shift=False,
        shift_ratio=3,
        transform=None,
        box_crop=False,
    ):
        self.dataset = dataset

        self.num_sample_frames = num_sample_frames
        self.random_shift = random_shift
        self.shift_ratio = shift_ratio
        if self.shift_ratio < 2:
            print("Random temporal shift ratio must be lager and equal than '2'...")
            print("Shift ratio is changed to be 2.")
            self.shift_ratio = 2

        self.transform = transform
        self.box_crop_check = box_crop
        if self.box_crop_check:
            self.box_detector = box_ext()

    def _get_sample_from_vid(self, vid_path):
        cap = cv2.VideoCapture(vid_path)

        vid_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if vid_length == 0:
            vid_length = 60
        sample_step = (vid_length - 1) // (self.num_sample_frames - 1)

        if self.random_shift:
            offset_range = sample_step // self.shift_ratio

            sample_indices = []
            for idx in range(self.num_sample_frames):
                sample_index = sample_step * idx + randint(-offset_range, offset_range)
                sample_index = min(sample_index, vid_length - 1)
                sample_index = max(sample_index, 0)

                sample_indices.append(sample_index)
        else:
            sample_indices = [
                sample_step * idx for idx in range(self.num_sample_frames)
            ]
        sample_indices += [sample_indices[-1] for _ in range(8 - len(sample_indices))]

        frame_list = []
        for idx in range(vid_length):
            _, frame = cap.read()

            if frame is not None:
                if idx in sample_indices:
                    frame_list.append(np.expand_dims(frame, axis=0))
            else:
                frame_list.append(frame_list[-1])

        return np.concatenate(frame_list, axis=0)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        # try:
        vid_path, label = self.dataset[index]

        vid = self._get_sample_from_vid(str(vid_path))
        if self.box_crop_check:
            vid = self.box_detector.box_detect_crop(vid)
        vid = torch.from_numpy(vid)

        if self.transform is not None:
            vid = self.transform(vid)

        return vid, label
        # except Exception as e:
        #     print(e)
        #     with open("error_video_list.txt", "a") as f:
        #         f.write(str(vid_path) + "\n")

        #     return torch.tensor(np.zeros((3, 16, 128, 128))), 100
