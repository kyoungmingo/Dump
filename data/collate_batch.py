import torch


def collate_fn(batch):
    imgs, labels = zip(*batch)
    labels = torch.tensor(labels, dtype=torch.int64)
    return torch.stack(imgs, dim=0), labels
