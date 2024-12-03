import torch
import torch.nn as nn

from .layers import ConvBlock, UpBlock, DownBlock, Flatten, Unflatten, FCBlock

class UnconditionalUNet(nn.Module):
    def __init__(
        self,
        in_channels: int,
        num_hiddens: int,
    ):
        super().__init__()

        self.input_conv = ConvBlock(in_channels, num_hiddens)

        self.down_blocks = nn.ModuleList()
        self.down_blocks.append(DownBlock(num_hiddens, num_hiddens))
        self.down_blocks.append(DownBlock(num_hiddens, num_hiddens * 2))

        self.flatten = Flatten()

        self.unflatten = Unflatten(num_hiddens * 2)

        self.up_blocks = nn.ModuleList()
        self.up_blocks.append(UpBlock(num_hiddens * 4, num_hiddens))
        self.up_blocks.append(UpBlock(num_hiddens * 2, num_hiddens))

        self.out_conv1 = ConvBlock(num_hiddens * 2, num_hiddens)

        self.out_conv2 = nn.Conv2d(num_hiddens, in_channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.shape[-2:] == (28, 28), "Expect input shape to be (28, 28)."

        x = self.input_conv(x)
        first_x = x

        down_outputs = []
        for down_block in self.down_blocks:
            x = down_block(x)
            down_outputs.append(x)
        
        x = self.flatten(x)

        x = self.unflatten(x)

        for i, up_block in enumerate(self.up_blocks):
            x = torch.cat([x, down_outputs[-i - 1]], dim=1)
            x = up_block(x)
        
        x = torch.cat([x, first_x], dim=1)
        x = self.out_conv1(x)

        x = self.out_conv2(x)
        return x

class TimeConditionalUNet(nn.Module):
    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        num_hiddens: int,
    ):
        super().__init__()
        
        self.input_conv = ConvBlock(in_channels, num_hiddens)

        self.down_blocks = nn.ModuleList()
        self.down_blocks.append(DownBlock(num_hiddens, num_hiddens))
        self.down_blocks.append(DownBlock(num_hiddens, num_hiddens * 2))

        self.flatten = Flatten()

        self.unflatten = Unflatten(num_hiddens * 2)

        self.up_blocks = nn.ModuleList()
        self.up_blocks.append(UpBlock(num_hiddens * 4, num_hiddens))
        self.up_blocks.append(UpBlock(num_hiddens * 2, num_hiddens))

        self.out_conv1 = ConvBlock(num_hiddens * 2, num_hiddens)

        self.out_conv2 = nn.Conv2d(num_hiddens, in_channels, kernel_size=3, padding=1)

        self.fc_lower = FCBlock(1, num_hiddens * 2)
        self.fc_higher = FCBlock(1, num_hiddens)

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x: (N, C, H, W) input tensor.
            t: (N,) normalized time tensor.

        Returns:
            (N, C, H, W) output tensor.
        """
        assert x.shape[-2:] == (28, 28), "Expect input shape to be (28, 28)."

        t_low = self.fc_lower(t)
        t_high = self.fc_higher(t)
        t_low = t_low.unsqueeze(-1).unsqueeze(-1)
        t_high = t_high.unsqueeze(-1).unsqueeze(-1)

        x = self.input_conv(x)
        first_x = x

        down_outputs = []
        for down_block in self.down_blocks:
            x = down_block(x)
            down_outputs.append(x)
        
        x = self.flatten(x)

        x = self.unflatten(x) + t_low

        for i, up_block in enumerate(self.up_blocks):
            x = torch.cat([x, down_outputs[-i - 1]], dim=1)
            x = up_block(x)
            if i == 0:
                x += t_high
        
        x = torch.cat([x, first_x], dim=1)
        x = self.out_conv1(x)

        x = self.out_conv2(x)
        return x
