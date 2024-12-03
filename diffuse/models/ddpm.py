import torch
from torch import nn

import matplotlib.pyplot as plt

from .unet import TimeConditionalUNet

class DDPM(nn.Module):
    def __init__(
        self,
        unet: TimeConditionalUNet,
        betas: tuple[float, float] = (1e-4, 0.02),
        num_ts: int = 300,
        p_uncond: float = 0.1,
    ):
        super().__init__()
        self.unet = unet
        self.betas = betas
        self.num_ts = num_ts
        self.p_uncond = p_uncond
        self.ddpm_schedule = ddpm_schedule(betas[0], betas[1], num_ts)

        for k, v in ddpm_schedule(betas[0], betas[1], num_ts).items():
            self.register_buffer(k, v, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (N, C, H, W) input tensor.

        Returns:
            (,) diffusion loss.
        """
        return ddpm_forward(
            self.unet, self.ddpm_schedule, x, self.num_ts
        )

    @torch.inference_mode()
    def sample(
        self,
        img_wh: tuple[int, int],
        seed: int = 0,
    ):
        return ddpm_sample(
            self.unet, self.ddpm_schedule, img_wh, self.num_ts, seed
        )

def ddpm_schedule(beta1: float, beta2: float, num_ts: int) -> dict:
    """Constants for DDPM training and sampling.

    Arguments:
        beta1: float, starting beta value.
        beta2: float, ending beta value.
        num_ts: int, number of timesteps.

    Returns:
        dict with keys:
            betas: linear schedule of betas from beta1 to beta2.
            alphas: 1 - betas.
            alpha_bars: cumulative product of alphas.
    """
    assert beta1 < beta2 < 1.0, "Expect beta1 < beta2 < 1.0."

    betas = torch.linspace(beta1, beta2, num_ts)
    alphas = 1.0 - betas
    alpha_bars = torch.cumprod(alphas, dim=0)

    return {
        "d_betas": betas,
        "d_alphas": alphas,
        "d_alpha_bars": alpha_bars,
    }

def ddpm_forward(
    unet: TimeConditionalUNet,
    ddpm_schedule: dict,
    x_0: torch.Tensor,
    num_ts: int,
    display=False
) -> torch.Tensor:
    """Algorithm 1 of the DDPM paper.

    Args:
        unet: TimeConditionalUNet
        ddpm_schedule: dict
        x_0: (N, C, H, W) input tensor.
        num_ts: int, number of timesteps.
    Returns:
        (,) diffusion loss.
    """
    unet.train()

    if display:
        plt.imshow(x_0[0].squeeze().detach().cpu(), cmap="gray")
        plt.show()

    alpha_bars = ddpm_schedule["d_alpha_bars"]

    t = torch.randint(0, num_ts, (x_0.shape[0],))

    alpha_bars_t = alpha_bars[t].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).to(x_0.device)
    t = (t / (num_ts)).unsqueeze(-1).to(x_0.device)
    
    epsilon = torch.randn_like(x_0)
    x_t = torch.sqrt(alpha_bars_t) * x_0 + torch.sqrt(1.0 - alpha_bars_t) * epsilon
    if display:
        plt.imshow(x_t[0].squeeze().detach().cpu(), cmap="gray")
        plt.show()
    epsilon_hat = unet(x_t, t)
    loss = nn.MSELoss()(epsilon, epsilon_hat)

    return loss

@torch.inference_mode()
def ddpm_sample(
    unet: TimeConditionalUNet,
    ddpm_schedule: dict,
    img_wh: tuple[int, int],
    num_ts: int,
    seed: int = 0,
) -> torch.Tensor:
    """Algorithm 2 of the DDPM paper with classifier-free guidance.

    Args:
        unet: TimeConditionalUNet
        ddpm_schedule: dict
        img_wh: (H, W) output image width and height.
        num_ts: int, number of timesteps.
        seed: int, random seed.

    Returns:
        (N, C, H, W) final sample.
    """
    unet.eval()

    betas = ddpm_schedule["d_betas"]
    alphas = ddpm_schedule["d_alphas"]
    alpha_bars = ddpm_schedule["d_alpha_bars"]

    x_t = torch.randn(1, 1, img_wh[0], img_wh[1])
    for t in range(num_ts - 1, 0, -1):
        z = torch.randn(1, 1, img_wh[0], img_wh[1]) # if t > 1 else torch.zeros(1, 1, img_wh[0], img_wh[1])

        input_t = torch.tensor(t / (num_ts)).unsqueeze(-1).unsqueeze(-1).to(x_t.device)

        unet = unet.to(x_t.device)

        x_0_hat = (1 / torch.sqrt(alpha_bars[t])) * (x_t - torch.sqrt(1.0 - alpha_bars[t]) * unet(x_t, input_t))
        x_t = (torch.sqrt(alpha_bars[t - 1]) * betas[t])/(1 - alpha_bars[t]) * x_0_hat  \
                + (torch.sqrt(alphas[t]) * (1 - alpha_bars[t - 1]))/(1 - alpha_bars[t]) * x_t \
                + torch.sqrt(betas[t]) * z

    return x_t
