import torch

import unittest
from diffuse.models import Conv, ConvBlock, UpConv, UpBlock, DownConv, DownBlock, Flatten, Unflatten

class TestLayers(unittest.TestCase):
    def test_conv(self):
        x = torch.randn(1, 1, 28, 28)
        conv = Conv(1, 16)
        x = conv(x)
        assert x.shape == (1, 16, 28, 28)

    def test_down_conv(self):
        x = torch.randn(1, 16, 28, 28)
        down_conv = DownConv(16, 32)
        x = down_conv(x)
        assert x.shape == (1, 32, 14, 14)

    def test_up_conv(self):
        x = torch.randn(1, 32, 14, 14)
        up_conv = UpConv(32, 16)
        x = up_conv(x)
        assert x.shape == (1, 16, 28, 28)

    def test_flatten(self):
        x = torch.randn(1, 1, 7, 7)
        flatten = Flatten()
        x = flatten(x)
        assert x.shape == (1, 1, 1, 1)
    
    def test_unflatten(self):
        x = torch.randn(1, 1, 1, 1)
        unflatten = Unflatten(1)
        x = unflatten(x)
        assert x.shape == (1, 1, 7, 7)

    def test_conv_block(self):
        x = torch.randn(1, 1, 7, 7)
        conv_block = ConvBlock(1, 16)
        x = conv_block(x)
        assert x.shape == (1, 16, 7, 7)

    def test_down_block(self):
        x = torch.randn(1, 16, 7, 7)
        down_block = DownBlock(16, 32)
        x = down_block(x)
        assert x.shape == (1, 32, 4, 4)

    def test_up_block(self):
        x = torch.randn(1, 32, 4, 4)
        up_block = UpBlock(32, 16)
        x = up_block(x)
        assert x.shape == (1, 16, 8, 8)
