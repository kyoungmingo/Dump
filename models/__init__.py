from .r2plus1d import R2Plus1D

__all__ = ["R2Plus1D"]


def build_model(args, num_classes):
    if args.model_name == "r2plus1d_18":
        model = R2Plus1D(
            num_classes,
            backbone_name="resnet18",
            backbone_pretrained=args.backbone_pretrained,
        )
    else:
        raise NotImplementedError(
            f"Specified model name '{args.model_name}' is not implemented yet!"
        )

    return model
