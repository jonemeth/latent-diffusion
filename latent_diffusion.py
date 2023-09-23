from pathlib import Path
from typing import Any, Optional

import torch
from lightning.pytorch.utilities.types import STEP_OUTPUT
import lightning.pytorch as pl
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import StochasticWeightAveraging
from pytorch_lightning.utilities import rank_zero_only

from config import load_config


from image_managers import IImageManager
from vae import VAE, VAEConfig
from diffusion import Diffusion, DiffusionConfig




class LatentDiffusion(pl.LightningModule):
    def __init__(self, vae_config: VAEConfig, diffusion_config: DiffusionConfig):
        super().__init__()

        self.save_hyperparameters()

        self.vae = VAE(vae_config)
        self.diffusion = Diffusion(diffusion_config)

        for param in self.vae.parameters():
            param.requires_grad = False

    def load_vae_weights(self, logger: WandbLogger, reference: str):
        artifact = logger.use_artifact(reference)
        artifact_dir = artifact.download()

        state_dict = torch.load(
            Path(artifact_dir) / "model.ckpt")["state_dict"]

        self.vae.load_state_dict(state_dict)


    def sample(self, n: int, y: Optional[torch.Tensor] = None):
        z = self.diffusion.sample(n, y)
        return self.vae.decode(z)

    def training_step(self, *args: Any, **kwargs: Any) -> STEP_OUTPUT:
        batch, _ = args
        x, y = batch

        if not self.diffusion.network.cfg.conditional:
            y = None

        with torch.no_grad():
            z_mu, z_logvar = self.vae.encode(x)
            z = self.vae.reparameterization(z_mu, z_logvar).detach()

        loss = self.diffusion.get_loss((z, y))

        self.log("loss", loss, logger=True, on_step=True)

        return loss

    def configure_optimizers(self):
        return self.diffusion.configure_optimizers()


class Callback(pl.callbacks.Callback):
    def __init__(self, image_manager: IImageManager, logger: Optional[WandbLogger] = None):
        super().__init__()
        self.image_manager = image_manager
        self.logger = logger

        self.epoch_ix = 0
        self.n = 100

        if 10 == image_manager.num_classes():
            y = torch.tensor([i//10 for i in range(100)])
            self.y = torch.nn.functional.one_hot(y,
                                                 num_classes=10).type(torch.float32)
        else:
            _, y = next(iter(self.image_manager.create_train_loader(self.n)))
            self.y = y.type(torch.float32)


    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: LatentDiffusion):
        self.epoch_ix += 1

        if 0 != trainer.global_rank:
            return

        if 0 != self.epoch_ix % 20:
            return

        pl_module.eval()

        y = self.y if pl_module.diffusion.network.cfg.conditional else None

        with torch.no_grad():
            imgs = pl_module.sample(self.n, y).cpu()

        if self.logger:
            self.logger.log_image(f"{self.epoch_ix:04d}_imgs",
                                [self.image_manager.plot_image_grid(imgs)])

        pl_module.train()


def main():
    vae_checkpoint_reference = "freezingarrow/StableDiffusion/model-cifar10-vae-v28:v0"
    exp_id = "cifar10-vae-v28-ldm-v05-uncond"

    logger = WandbLogger(project="StableDiffusion",
                         name=exp_id,
                         version=exp_id,
                         log_model=True)

    config = load_config("configs/cifar10.yaml", ["image_manager", "latent_diffusion"])

    model: LatentDiffusion = config["latent_diffusion"]["model"]

    if 0 == rank_zero_only.rank:
        model.load_vae_weights(logger, vae_checkpoint_reference)


    image_manager = config["image_manager"]
    train_loader = image_manager.create_train_loader(
        config["latent_diffusion"]["batch_size"])


    trainer = pl.Trainer(max_epochs=config["latent_diffusion"]["max_epochs"],
                         log_every_n_steps=10,
                         gradient_clip_val=1.0,
                         logger=logger,
                         callbacks=[Callback(image_manager, logger),
                                    StochasticWeightAveraging(
                                        config["latent_diffusion"]["swa_lrs"]
                                        )])

    trainer.fit(model=model, train_dataloaders=train_loader)


if __name__ == "__main__":
    main()
