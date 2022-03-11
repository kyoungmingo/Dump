import torch
from torch import nn
from torchvision.models.video.resnet import r2plus1d_18

from utils.print_colors import print_warning


class R2Plus1D(nn.Module):
    def __init__(
        self,
        num_classes,
        backbone_name,
        backbone_pretrained="./weights/r2plus1d_18-91a641e6.pth",
    ):
        super().__init__()
        if backbone_name == "resnet18":
            model = r2plus1d_18(pretrained=False, progress=False)
            model.fc = nn.Linear(model.fc.in_features, num_classes)

            model._initialize_weights()
        else:
            raise NotImplementedError(
                f"The specified backbone `{backbone_name}` is not implemented yet!"
            )
        self.model = model
        self.model_state_dict = model.state_dict()

        if backbone_pretrained != "":
            self.load_param(backbone_pretrained, except_name="fc", strict=True)

    def forward(self, vid):
        return self.model(vid)

    def load_param(self, weight_path, except_name="fc", strict=True):
        state_dict = torch.load(weight_path)
        for i in state_dict:
            if strict is False and i not in self.model_state_dict:
                print_warning(f"{i} is skipped since it is not implemented in model...")
                continue
            if except_name in i:
                print_warning(f"{i} is skipped since you specified in 'except_name'...")
                continue
            self.model_state_dict[i].copy_(state_dict[i])
