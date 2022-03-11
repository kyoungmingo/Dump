from torchvision import transforms as T

from .video_transforms import (
    ToTensorVideo,
    ResizeVideo,
    NormalizeVideo,
    CenterCropVideo,
    RandomCropVideo,
    RandomResizedCropVideo,
    RandomHorizontalFlipVideo,
    ColorJitter,
)


def convert_size_to_tuple(size):
    if len(size) == 1:
        size = (size[0], size[0])
    else:
        assert len(size) == 2, "Size must be in shape of (height, width)..."

    return tuple(size)


def build_transforms(args, is_train=True):
    vid_size = convert_size_to_tuple(args.vid_size)
    if args.random_crop:
        crop_size = (int(args.crop_scale * value) for value in vid_size)
    else:
        crop_size = vid_size

    trfms = []

    if is_train:
        # Color Jitter
        if args.color_jitter:
            trfms.append(ColorJitter(0.15, 0, 0.1, 0.02))
        trfms.append(ToTensorVideo())
        # Random Crop
        if args.random_crop:
            trfms.append(ResizeVideo(crop_size))
            if args.random_crop_scales is not None:
                trfms.append(RandomResizedCropVideo(vid_size, args.random_crop_scales))
            else:
                trfms.append(RandomCropVideo(vid_size))
        else:
            trfms.append(ResizeVideo(vid_size))

        # Random Horizontal Flip
        if args.random_horizontal_flip > 0.0:
            trfms.append(RandomHorizontalFlipVideo(p=args.random_horizontal_flip))
    else:
        trfms.append(ToTensorVideo())
        # Only Resize for test
        trfms.append(ResizeVideo(vid_size))

    # Normalize
    trfms.append(NormalizeVideo(args.norm_mean, args.norm_std))

    return T.Compose(trfms)
