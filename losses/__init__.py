from torch import nn

from .softmax import CrossEntropyLabelSmooth


def make_loss(args, num_classes):
    if args.label_smooth:
        xent_criterion = CrossEntropyLabelSmooth(num_classes=num_classes)
    else:
        xent_criterion = nn.CrossEntropyLoss()

    def loss_func(score, target):
        losses = {}
        losses["X-entropy"] = xent_criterion(score, target)

        return losses

    return loss_func
