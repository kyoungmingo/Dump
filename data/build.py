from torch.utils.data import DataLoader

from .datasets import init_dataset, VideoDataset
from .transforms import build_transforms
from .collate_batch import collate_fn


def make_data_loader(args):
    print(f"Initializing data {args.dataset_name}")

    dataset = init_dataset(name=args.dataset_name, root=args.dataset_root)
    num_classes = dataset.num_classes

    if args.train:
        train_loader = DataLoader(
            VideoDataset(
                dataset.train,
                args.num_sample_frames,
                random_shift=args.random_shift,
                shift_ratio=args.shift_ratio,
                transform=build_transforms(args, is_train=True),
            ),
            batch_size=args.train_batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            collate_fn=collate_fn,
            drop_last=True,
            pin_memory=True,
        )
    else:
        train_loader = None

    val_loader = DataLoader(
        VideoDataset(
            dataset.test,
            args.num_sample_frames,
            random_shift=False,
            transform=build_transforms(args, is_train=False),
            box_crop=args.box_crop,
        ),
        batch_size=args.test_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        drop_last=False,
        pin_memory=True,
    )

    # ["dumping", "bent", "good", "walking", "nothing"]
    return train_loader, val_loader, num_classes, dataset.classes
