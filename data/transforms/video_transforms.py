import PIL
import random
import numpy as np
import torch
from torch.nn.functional import interpolate

import torchvision
from torchvision.transforms import RandomResizedCrop
from torchvision.transforms._transforms_video import (
    ToTensorVideo,
    NormalizeVideo,
    CenterCropVideo,
    RandomCropVideo,
    RandomHorizontalFlipVideo,
)


__all__ = [
    "ToTensorVideo",
    "ResizeVideo",
    "NormalizeVideo",
    "CenterCropVideo",
    "RandomCropVideo",
    "RandomResizedCropVideo",
    "RandomHorizontalFlipVideo",
    "ColorJitter",
]


def _is_tensor_video_clip(clip):
    if not torch.is_tensor(clip):
        raise TypeError("clip should be Tesnor. Got %s" % type(clip))

    if not clip.ndimension() == 4:
        raise ValueError("clip should be 4D. Got %dD" % clip.dim())

    return True


class ResizeVideo:
    def __init__(self, size, interpolation_mode="bilinear"):
        if isinstance(size, tuple):
            assert len(size) == 2, "size should be tuple (height, width)"
        self.size = tuple(size)
        self.interpolation_mode = interpolation_mode

    def __call__(self, clip):
        return interpolate(
            clip, size=self.size, mode=self.interpolation_mode, align_corners=False
        )


class RandomResizedCropVideo(RandomResizedCrop):
    def __init__(
        self,
        size,
        scale=(0.08, 1.0),
        ratio=(3.0 / 4.0, 4.0 / 3.0),
        interpolation_mode="bilinear",
    ):
        if isinstance(size, tuple):
            assert len(size) == 2, "size should be tuple (height, width)"
            self.size = size
        else:
            self.size = (size, size)

        self.interpolation_mode = interpolation_mode
        self.scale = scale
        self.ratio = ratio

    def __call__(self, clip):
        """
        Args:
            clip (torch.tensor): Video clip to be cropped. Size is (C, T, H, W)
        Returns:
            torch.tensor: randomly cropped/resized video clip.
                size is (C, T, H, W)
        """
        i, j, h, w = self.get_params(clip, self.scale, self.ratio)
        return self.resized_crop(clip, i, j, h, w, self.size, self.interpolation_mode)

    def __repr__(self):
        return (
            self.__class__.__name__
            + "(size={0}, interpolation_mode={1}, scale={2}, ratio={3})".format(
                self.size, self.interpolation_mode, self.scale, self.ratio
            )
        )

    def resized_crop(self, clip, i, j, h, w, size, interpolation_mode="bilinear"):
        """
        Do spatial cropping and resizing to the video clip
        Args:
            clip (torch.tensor): Video clip to be cropped. Size is (C, T, H, W)
            i (int): i in (i,j) i.e coordinates of the upper left corner.
            j (int): j in (i,j) i.e coordinates of the upper left corner.
            h (int): Height of the cropped region.
            w (int): Width of the cropped region.
            size (tuple(int, int)): height and width of resized clip
        Returns:
            clip (torch.tensor): Resized and cropped clip. Size is (C, T, H, W)
        """
        assert _is_tensor_video_clip(clip), "clip should be a 4D torch.tensor"
        clip = self.crop(clip, i, j, h, w)
        clip = self.resize(clip, size, interpolation_mode)
        return clip

    @staticmethod
    def resize(clip, target_size, interpolation_mode):
        assert len(target_size) == 2, "target size should be tuple (height, width)"
        return torch.nn.functional.interpolate(
            clip, size=target_size, mode=interpolation_mode, align_corners=False,
        )

    @staticmethod
    def crop(clip, i, j, h, w):
        """
        Args:
            clip (torch.tensor): Video clip to be cropped. Size is (C, T, H, W)
        """
        assert len(clip.size()) == 4, "clip should be a 4D tensor"
        return clip[..., i : i + h, j : j + w]


class ColorJitter:
    """Randomly change the brightness, contrast and saturation and hue of the clip
    Args:
    brightness (float): How much to jitter brightness. brightness_factor
    is chosen uniformly from [max(0, 1 - brightness), 1 + brightness].
    contrast (float): How much to jitter contrast. contrast_factor
    is chosen uniformly from [max(0, 1 - contrast), 1 + contrast].
    saturation (float): How much to jitter saturation. saturation_factor
    is chosen uniformly from [max(0, 1 - saturation), 1 + saturation].
    hue(float): How much to jitter hue. hue_factor is chosen uniformly from
    [-hue, hue]. Should be >=0 and <= 0.5.
    """

    def __init__(self, brightness=0, contrast=0, saturation=0, hue=0):
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue

    def get_params(self, brightness, contrast, saturation, hue):
        if brightness > 0:
            brightness_factor = random.uniform(
                max(0, 1 - brightness), 1 + brightness)
        else:
            brightness_factor = None

        if contrast > 0:
            contrast_factor = random.uniform(
                max(0, 1 - contrast), 1 + contrast)
        else:
            contrast_factor = None

        if saturation > 0:
            saturation_factor = random.uniform(
                max(0, 1 - saturation), 1 + saturation)
        else:
            saturation_factor = None

        if hue > 0:
            hue_factor = random.uniform(-hue, hue)
        else:
            hue_factor = None
        return brightness_factor, contrast_factor, saturation_factor, hue_factor

    def __call__(self, clip):
        """
        Args:
        clip (list): list of PIL.Image
        Returns:
        list PIL.Image : list of transformed PIL.Image
        """
        clips = []
        clip = np.array(clip)
        for img in clip:
            clips.append(PIL.Image.fromarray(img))

        brightness, contrast, saturation, hue = self.get_params(
            self.brightness, self.contrast, self.saturation, self.hue)

        # Create img transform function sequence
        img_transforms = []
        if brightness is not None:
            img_transforms.append(lambda img: torchvision.transforms.functional.adjust_brightness(img, brightness))
        if saturation is not None:
            img_transforms.append(lambda img: torchvision.transforms.functional.adjust_saturation(img, saturation))
        if hue is not None:
            img_transforms.append(lambda img: torchvision.transforms.functional.adjust_hue(img, hue))
        if contrast is not None:
            img_transforms.append(lambda img: torchvision.transforms.functional.adjust_contrast(img, contrast))
        random.shuffle(img_transforms)

        # Apply to all images
        jittered_clip = []
        for img in clips:
            for func in img_transforms:
                jittered_img = func(img)
            jittered_clip.append(np.array(jittered_img))

        return torch.from_numpy(np.array(jittered_clip))