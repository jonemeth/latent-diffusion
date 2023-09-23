from dataclasses import dataclass
from typing import Optional
import lightning.pytorch as pl
import numpy as np
import torch.nn
from tqdm import tqdm

from unet import UNet, UNetConfig


def get_timestep_embedding(timesteps: torch.Tensor, embedding_dim: int) -> torch.Tensor:
    """
    From : https://github.com/hojonathanho/diffusion/blob/master/diffusion_tf/nn.py
    Original comments:
    From Fairseq.
    Build sinusoidal embeddings.
    This matches the implementation in tensor2tensor, but differs slightly
    from the description in Section 3.5 of "Attention Is All You Need".
    """
    assert embedding_dim % 2 == 0
    assert len(timesteps.shape) == 1  # and timesteps.dtype == tf.int32

    half_dim = embedding_dim // 2
    emb = np.log(10000.0) / float(half_dim - 1)
    emb = torch.exp(torch.arange(0, half_dim, dtype=torch.float32) * -emb)

    # emb = tf.range(num_embeddings, dtype=tf.float32)[:, None] * emb[None, :]
    emb = timesteps.type(torch.float32)[:, None] * emb[None, :]
    emb = torch.concat([torch.sin(emb), torch.cos(emb)], dim=1)

    # if embedding_dim % 2 == 1:  # zero pad
    #     # emb = tf.concat([emb, tf.zeros([num_embeddings, 1])], axis=1)
    #     emb = torch.tf.pad(emb, [[0, 0], [0, 1]])

    return emb


@dataclass
class DiffusionConfig:
    unet_config: UNetConfig
    time_steps: int
    learning_rate: float
    weight_decay: float
    qkv_weight_decay: float


class Diffusion(pl.LightningModule):
    def __init__(self, cfg: DiffusionConfig):
        super().__init__()

        self.save_hyperparameters()

        self.cfg = cfg

        self.network = UNet(self.cfg.unet_config)

        beta_list = [1e-4 + (0.02 - 1e-4) * (t / (self.cfg.time_steps - 1))
                     for t in range(0, self.cfg.time_steps)]
        alpha_list = [1.0 - b for b in beta_list]
        alpha_hat_list = [float(np.prod(alpha_list[0:t]))
                          for t in range(1, self.cfg.time_steps + 1)]

        self.beta_list = torch.nn.Parameter(
            torch.tensor([0.0] + beta_list), requires_grad=False)
        self.alpha_list = torch.nn.Parameter(
            torch.tensor([0.0] + alpha_list), requires_grad=False)
        self.alpha_hat_list = torch.nn.Parameter(
            torch.tensor([0.0] + alpha_hat_list), requires_grad=False)

        self.time_embeddings = torch.nn.Parameter(
            get_timestep_embedding(torch.arange(
                1, self.cfg.time_steps + 1), self.cfg.unet_config.channels),
            requires_grad=False)

    def get_loss(self, batch):
        x0, y = batch

        times = torch.randint(1, self.cfg.time_steps,
                              size=[x0.shape[0]], dtype=torch.int64,
                              device=self.device, requires_grad=False)

        alpha_hat = self.alpha_hat_list[times].view([-1, 1, 1, 1])

        eps = torch.randn_like(x0, requires_grad=False)

        xt = torch.sqrt(alpha_hat) * x0 + torch.sqrt(1.0 - alpha_hat) * eps

        eps_theta = self.network(xt, self.time_embeddings[times - 1, :], y)
        loss = ((eps - eps_theta) ** 2).sum([1, 2, 3])
        loss = loss.mean()

        return loss

    def training_step(self, *args, **kwargs) -> pl.utilities.types.STEP_OUTPUT:
        batch, _ = args
        loss = self.get_loss(batch)

        self.log("loss", loss, logger=True, on_step=True)

        return loss

    def configure_optimizers(self):
        params_high_decay, params_low_decay = self.network.separate_parameters()
        print(f"separate_parameters - high: {len(params_high_decay)}, low: {len(params_low_decay)}")
        optim_groups = [
            {"params": params_high_decay, "weight_decay": self.cfg.qkv_weight_decay},
            {"params": params_low_decay, "weight_decay": self.cfg.weight_decay},
        ]

        optimizer = torch.optim.AdamW(
            optim_groups, lr=self.cfg.learning_rate, betas=(0.9, 0.999))
        return optimizer

    def sample(self, n: int, y: Optional[torch.Tensor] = None):
        size = (n, self.cfg.unet_config.input_channels,
                self.cfg.unet_config.input_size, self.cfg.unet_config.input_size)
        if y:
            y = y.to(self.device)

        x = torch.randn(size=size, dtype=torch.float32, device=self.device)

        print("Generating samples:")
        for t in tqdm(range(self.cfg.time_steps, 0, -1)):
            times = torch.tensor([t] * n, dtype=torch.int64, device=self.device)

            eps_theta = self.network(
                x, self.time_embeddings[times - 1, :], y).detach()

            z = torch.randn_like(x) if t > 1 else 0.0

            a = self.alpha_list[times]
            a_hat = self.alpha_hat_list[times]
            sigma = torch.sqrt(self.beta_list[times])

            x = (1.0 / torch.sqrt(a))[:, None, None, None] * \
                (x - ((1.0 - a) / torch.sqrt(1.0 - a_hat))[:, None, None, None] * eps_theta) + \
                sigma[:, None, None, None] * z

        return x
