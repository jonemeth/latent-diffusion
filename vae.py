from dataclasses import dataclass
from typing import List, Tuple
import torch
from torch import nn

import lightning.pytorch as pl
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import StochasticWeightAveraging
from config import load_config

from image_managers import IImageManager
from nn import BasicBlock, ResBlock, Downsample, Upsample


LOG_2PI = 1.83736998048

def log_normal_pdf(x: torch.Tensor, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    return -0.5 * ((x - mu) ** 2 * torch.exp(-logvar) + logvar + LOG_2PI)

def log_standard_normal_pdf(x: torch.Tensor) -> torch.Tensor:
    return -0.5 * (x ** 2 + LOG_2PI)


@dataclass
class VAEConfig:
    num_latents: int
    beta: float
    stage_depths: List[int]
    channels: int
    channel_mults: List[int]
    learning_rate: float
    weight_decay: float
    x_logvar_learning_rate: float


class VAE(pl.LightningModule):
    def __init__(self, cfg: VAEConfig):
        super().__init__()
        self.save_hyperparameters()

        self.cfg = cfg

        num_stages = len(self.cfg.stage_depths)

        encoder_layers = [torch.nn.Conv2d(
            3, self.cfg.channels, kernel_size=3, padding="same")]

        for i, (depth, cm) in enumerate(zip(self.cfg.stage_depths, self.cfg.channel_mults)):
            if i > 0:
                encoder_layers.append(Downsample(
                    self.cfg.channel_mults[i-1]*self.cfg.channels, cm*self.cfg.channels))
            for _ in range(depth):
                encoder_layers.append(ResBlock(
                    cm*self.cfg.channels, cm*self.cfg.channels, cm*self.cfg.channels,
                    False, False))
        encoder_layers.append(BasicBlock(
            self.cfg.channel_mults[-1]*self.cfg.channels, 2*self.cfg.num_latents))

        self.encoder = nn.Sequential(*encoder_layers)

        decoder_layers = [torch.nn.Conv2d(
            self.cfg.num_latents, self.cfg.channel_mults[-1]*self.cfg.channels,
            kernel_size=3, padding="same")]

        for i, (depth, cm) in reversed(list(enumerate(zip(self.cfg.stage_depths,
                                                          self.cfg.channel_mults)))):
            if i < num_stages-1:
                decoder_layers.append(
                    Upsample(self.cfg.channel_mults[i+1]*self.cfg.channels, cm*self.cfg.channels))
            for _ in range(depth):
                decoder_layers.append(
                    ResBlock(cm*self.cfg.channels, cm*self.cfg.channels, cm*self.cfg.channels,
                             False, False))

        decoder_layers.append(BasicBlock(self.cfg.channels, 3))

        self.decoder = nn.Sequential(*decoder_layers)

        self.x_logvar = nn.Parameter(torch.tensor([-2.0]))


    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        mu, logvar = torch.chunk(self.encoder(x), 2, dim=1)
        return mu, logvar

    @staticmethod
    def reparameterization(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        eps = torch.randn(size=mu.shape, device=mu.get_device())
        return mu + eps * (0.5*logvar).exp()


    def decode(self, z: torch.Tensor) -> torch.Tensor:
        mu = self.decoder(z)
        return mu

    def get_metrics(self, batch):
        x, _ = batch

        z_mu, z_logvar = self.encode(x)
        z = self.reparameterization(z_mu, z_logvar)

        x_mu = self.decode(z)

        log_px_z = log_normal_pdf(x, x_mu, self.x_logvar)
        log_qz_x = log_normal_pdf(z, z_mu, z_logvar)
        log_pz = log_standard_normal_pdf(z)

        elbo = log_px_z.sum(dim=[1, 2, 3]) + \
            self.cfg.beta * (log_pz.sum(dim=[1, 2, 3]) -
                         log_qz_x.sum(dim=[1, 2, 3]))

        loss = -elbo.mean()

        metrics = {
            'loss': loss,
            'px_mse': ((x-x_mu)**2).mean(),
            'px_prob': log_px_z.mean(),
            'dim_kl': (log_qz_x-log_pz).mean()
        }

        return metrics

    def training_step(self, *args, **kwargs) -> pl.utilities.types.STEP_OUTPUT:
        batch, _ = args
        metrics = self.get_metrics(batch)

        for k, v in metrics.items():
            self.log(k, v, logger=True, on_step=True)

        self.log("x_std", (self.x_logvar/2.0).exp(), logger=True, on_step=True)

        return metrics['loss']

    def configure_optimizers(self):
        optim_groups = [
            {"params": list(self.encoder.parameters())+list(self.decoder.parameters()),
             "lr": self.cfg.learning_rate, "weight_decay": self.cfg.weight_decay},
            {"params": [self.x_logvar],
                "lr": self.cfg.x_logvar_learning_rate, "weight_decay": 0.0},
        ]

        optimizer = torch.optim.AdamW(optim_groups, betas=(0.9, 0.999))
        return optimizer


class Callback(pl.callbacks.Callback):
    def __init__(self, image_manager: IImageManager, logger: WandbLogger):
        super().__init__()
        self.image_manager = image_manager
        self.logger = logger

        self.epoch_ix = 0
        self.imgs, _ = next(iter(self.image_manager.create_train_loader(64)))
        self.logger.log_image(
            "imgs", [self.image_manager.plot_image_grid(self.imgs.cpu())])

    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: VAE):
        if 0 != trainer.global_rank:
            return

        self.epoch_ix += 1

        if 0 != self.epoch_ix % 5:
            return

        pl_module.eval()

        with torch.no_grad():
            self.imgs = self.imgs.to(pl_module.device)
            z_mu, z_logvar = pl_module.encode(self.imgs)
            z_sample = pl_module.reparameterization(z_mu, z_logvar)

            recs1 = pl_module.decode(z_mu).cpu()
            recs2 = pl_module.decode(z_sample).cpu()

        self.logger.log_image(f"{self.epoch_ix:04d}_recs1",
                              [self.image_manager.plot_image_grid(recs1)])
        self.logger.log_image(f"{self.epoch_ix:04d}_recs2",
                              [self.image_manager.plot_image_grid(recs2)])

        pl_module.train()


def main():
    exp_id = "cifar10-vae-temp"

    config = load_config("configs/cifar10.yaml",
                         ["image_manager", "vae_config", "vae"])


    logger = WandbLogger(project="LatentDiffusion",
                         name=exp_id,
                         version=exp_id,
                         log_model=True)

    image_manager = config["image_manager"]
    train_loader = image_manager.create_train_loader(
        config["vae"]["batch_size"])

    model = config["vae"]["model"]

    trainer = pl.Trainer(max_epochs=config["vae"]["max_epochs"],
                         log_every_n_steps=10,
                         gradient_clip_val=1.0,
                         logger=logger,
                         callbacks=[Callback(image_manager, logger),
                                    StochasticWeightAveraging(config["vae"]["swa_lrs"])])

    trainer.fit(model=model, train_dataloaders=train_loader)


if __name__ == "__main__":
    main()
