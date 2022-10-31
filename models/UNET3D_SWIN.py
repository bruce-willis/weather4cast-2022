from monai.networks.nets import SwinUNETR
import torch
from torch import nn

net = SwinUNETR(img_size=(32,256,256), in_channels=11, out_channels=1, depths=(2,4,2,2))


class SwinWeather(nn.Module):
    def __init__(self, **kwargs) -> None:
        super().__init__()
        self.input_shape = (32, 252, 252)
        self.target_shape = (4, 256, 256)

        self.model = SwinUNETR(img_size=(32, 256, 256), in_channels=11, out_channels=1) 

    def forward(self, x):
        x = torch.nn.functional.interpolate(x, size=self.target_shape)
        x = torch.repeat_interleave(x, 8, dim=2)
        x = self.model(x)
        x = torch.nn.functional.interpolate(x, size=self.input_shape)
        return x

if __name__ == "__main__":
    model = SwinWeather()
    B, C, T, H, W = 8, 11, 4, 252, 252
    weather_input = torch.rand(B, C, T, H, W)
    weather_output = model(weather_input)
    assert weather_output.shape == (B, 1, 32, H, W)
    