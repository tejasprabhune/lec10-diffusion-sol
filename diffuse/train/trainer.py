from __future__ import annotations
from enum import Enum
from dataclasses import dataclass

import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt

import torch
from torch import Tensor
from torch.nn import MSELoss
from torch.optim import AdamW
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor

from ..models import UnconditionalUNet, TimeConditionalUNet, DDPM
from ..data import NoisyMNIST


class Task(Enum):
    UNCON_UNET = 0
    TIME_UNET = 1
    CLASS_UNET = 2

    def from_str(arg: str) -> Task:
        if arg == "unconditional_unet":
            return Task.UNCON_UNET
        elif arg == "time_unet":
            return Task.TIME_UNET
        else:
            return Task.CLASS_UNET

class Data(Enum):
    MNIST = 0
    NOISY_MNIST = 1

    def from_str(arg: str) -> Data:
        if arg == "mnist":
            return Data.MNIST
        else:
            return Data.NOISY_MNIST

    def make_dataset(self, train: bool = True) -> Dataset:
        if self == Data.MNIST:
            return MNIST(root="data", download=True, transform=ToTensor(), train=train)
        else:
            return NoisyMNIST(noise_level=0.5, train=train)

@dataclass
class TrainingConfig:
    task: Task
    data: Data
    n_epochs: int
    batch_size: int = 256
    lr: float = 1e-3
    gamma: float = 0.9
    d: int = 64
    device: int | str = 0

    def from_args(args: argparse.Namespace) -> TrainingConfig:
        return TrainingConfig(Task.from_str(args.task), Data.from_str(args.data), args.n_epochs)

    def train(self):
        if self.task == Task.UNCON_UNET:
            train_unconditional_unet(self)
        elif self.task == Task.TIME_UNET:
            train_time_unet(self)
        else:
            raise NotImplementedError("Class-conditioned U-Net training not implemented")
        
def plot_losses(losses: list[float], title: str) -> plt.Figure:
    plt.plot(losses)
    plt.yscale("log")
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.title(title)
    plt.show()

    return plt.gcf()

def train_unconditional_unet(t: TrainingConfig):
    model: UnconditionalUNet = UnconditionalUNet(1, t.d).to(t.device)
    data: NoisyMNIST = t.data.make_dataset(train=True)
    dataloader: DataLoader = DataLoader(data, batch_size=t.batch_size, shuffle=True)
    test_data: NoisyMNIST = t.data.make_dataset(train=False)
    test_dataloader: DataLoader = DataLoader(test_data, batch_size=3, shuffle=True)
    optimizer: AdamW = AdamW(model.parameters(), lr=1e-4)
    loss_fn: MSELoss = MSELoss()
    train_losses = []

    def train_step(x: Tensor, y: Tensor) -> Tensor:
        x = x.to(t.device).to(torch.float32)
        y = y.to(t.device).to(torch.float32)
        optimizer.zero_grad()
        y_pred = model(x)
        loss = loss_fn(y_pred, y)
        loss.backward()
        optimizer.step()
        return loss

    def plot_pred(test_x: Tensor, test_y: Tensor) -> plt.Figure:
        test_y_pred = model(test_x)

        fig, axs = plt.subplots(3, 3, figsize=(6, 6))

        for i in range(3):
            axs[i, 0].imshow(test_y[i].cpu().squeeze(), cmap="gray")
            axs[i, 0].axis("off")
            axs[i, 0].set_title("Ground Truth")

            axs[i, 1].imshow(test_x[i].cpu().squeeze(), cmap="gray")
            axs[i, 1].axis("off")
            axs[i, 1].set_title("Input (sigma = 0.5)")

            axs[i, 2].imshow(test_y_pred[i].detach().cpu().squeeze(), cmap="gray")
            axs[i, 2].axis("off")
            axs[i, 2].set_title("Prediction")

        plt.show()

        return fig
    
    for epoch in range(t.n_epochs):
        train_tqdm = tqdm(dataloader, desc=f"Epoch {epoch}", dynamic_ncols=True)
        for x, y in train_tqdm:
            loss = train_step(x, y)
            train_losses.append(loss.item())
            train_tqdm.set_postfix(loss=loss.item())
        print(f"Epoch {epoch}, loss: {loss.item()}")

    test_x, test_y = next(iter(test_dataloader))
    test_x = test_x.to(t.device)
    test_y = test_y.to(t.device)
    fig = plot_pred(test_x, test_y)
    fig.savefig("out/unconditional_final.png")

    fig = plot_losses(train_losses, "Unconditional U-Net Losses")
    fig.savefig("out/unconditional_losses.png")
    
    torch.save(model.state_dict(), "out/unconditional_unet.pth")

def train_time_unet(t: TrainingConfig):
    model: TimeConditionalUNet = TimeConditionalUNet(1, 1, t.d).to(t.device)
    ddpm: DDPM = DDPM(unet=model).to(t.device)

    data: MNIST = t.data.make_dataset(train=True)
    dataloader: DataLoader = DataLoader(data, batch_size=t.batch_size, shuffle=True)
    optimizer: AdamW = AdamW(ddpm.parameters(), lr=1e-3)
    scheduler: ExponentialLR = ExponentialLR(optimizer, gamma=t.gamma)
    train_losses = []
    
    def train_step(x: Tensor) -> Tensor:
        x = x.to(t.device).float()
        optimizer.zero_grad()
        loss = ddpm(x)
        loss.backward()
        optimizer.step()
        return loss

    def plot_pred() -> plt.Figure:
        fig, axs = plt.subplots(4, 10, figsize=(9, 3))

        plot_tqdm = tqdm(range(40), desc="Plotting samples")
        for i in range(4):
            for j in range(10):
                x = ddpm.sample((28, 28))
                axs[i, j].imshow(x.squeeze().detach().cpu(), cmap="gray")
                axs[i, j].axis("off")
                plot_tqdm.update(1)
        plot_tqdm.close()

        plt.show()

        return fig

    ddpm.train()
    ddpm = ddpm.to(t.device)
    model = model.to(t.device)
    for epoch in range(5):
        train_tqdm = tqdm(dataloader, desc=f"Epoch {epoch}")
        for x, _ in train_tqdm:
            x = x.to(t.device)
            loss = train_step(x)
            train_losses.append(loss.item())
            train_tqdm.set_postfix(loss=loss.item())
        scheduler.step()

    fig = plot_pred()
    fig.savefig("out/time_conditional_5_epochs.png")

    ddpm.train()
    ddpm = ddpm.to(t.device)
    model = model.to(t.device)
    for epoch in range(5, t.n_epochs):
        train_tqdm = tqdm(dataloader, desc=f"Epoch {epoch}")
        for x, _ in train_tqdm:
            x = x.to(t.device)
            loss = train_step(x)
            train_losses.append(loss.item())
            train_tqdm.set_postfix(loss=loss.item())
        scheduler.step()

    fig = plot_pred()
    fig.savefig("out/time_conditional_final.png")

    fig = plot_losses(train_losses, "Time-conditional U-Net Losses")
    fig.savefig("out/time_conditional_losses.png")
    
    torch.save(model.state_dict(), "out/time_conditional_unet.pth")
