import torch

import unittest
from diffuse.models import UnconditionalUNet

class TestUnconditionalUnet(unittest.TestCase):
    def test_unconditional_unet(self):
        unet = UnconditionalUNet(1, 16)
        x = torch.randn(1, 1, 28, 28)
        y = unet(x)
        assert y.shape == x.shape
