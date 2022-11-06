import torch
from monai.networks.nets import SwinUNETR
from torch import nn

from models.baseline_UNET3D import DownConv


class SwinWeather(nn.Module):

    def __init__(self,
                 in_channels=11,
                 out_channels=1,
                 swin_input_shape=(32, 256, 256),
                 prediction_shape=(32, 252, 252),
                 channel_conv_params=None,
                 interpolate_mode='nearest',
                 use_checkpoint=False,
                 **kwargs) -> None:
        super().__init__()
        self.swin_input_shape = swin_input_shape
        self.prediction_shape = prediction_shape
        self.interpolate_mode = interpolate_mode

        if channel_conv_params is not None:
            self.channel_conv = DownConv(in_channels=4, out_channels=swin_input_shape[0], **channel_conv_params)
        else:
            self.channel_conv = None

        self.model = SwinUNETR(img_size=swin_input_shape,
                               in_channels=in_channels,
                               out_channels=out_channels,
                               use_checkpoint=use_checkpoint)

    def forward(self, x):
        if x.size(2) != self.swin_input_shape[0]:
            if self.channel_conv is None:
                repeat_num = self.swin_input_shape[0] // x.size(2)
                x = torch.repeat_interleave(x, repeat_num, dim=2)
            else:
                x = x.transpose(1, 2)
                x, _ = self.channel_conv(x)
                x = x.transpose(1, 2)
        x = torch.nn.functional.interpolate(x, size=self.swin_input_shape, mode=self.interpolate_mode)
        x = self.model(x)
        x = torch.nn.functional.interpolate(x, size=self.prediction_shape, mode=self.interpolate_mode)
        return x


if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    channel_conv_params = {'dropout_rate': 0.5, 'pooling': False, 'normalization': 'instance', 'activation': 'rrelu'}
    model = SwinWeather(swin_input_shape=(32, 256, 256),
                        channel_conv_params=channel_conv_params,
                        interpolate_mode='trilinear').to(device)
    B, C, T, H, W = 2, 11, 4, 252, 252
    weather_input = torch.rand(B, C, T, H, W, device=device)
    with torch.no_grad():
        weather_output = model(weather_input)
    assert weather_output.shape == (B, 1, 32, H, W)
