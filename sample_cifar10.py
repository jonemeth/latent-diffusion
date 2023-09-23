from pathlib import Path
import torch
import matplotlib.image
import wandb
from config import load_config
from image_managers import Cifar10Manager
from latent_diffusion import

from torchmetrics.image.inception import InceptionScore

def main():
    # config = load_config("configs/cifar10.yaml", ["image_manager", "vae", "latent_diffusion"])
    # model = config["latent_diffusion"]["model"]
    # model.set_vae(config["vae"]["model"])

    checkpoint_reference = "freezingarrow/StableDiffusion/model-cifar10-vae-v17-ldm-v03:v0"

    api = wandb.Api()
    artifact = api.artifact(checkpoint_reference)
    artifact_dir = artifact.download()

    model = LatentDiffusion.load_from_checkpoint(Path(artifact_dir) / "model.ckpt")
    model.to("cuda")

    # labels = torch.tensor([i//10 for i in range(100)])
    # labels = torch.nn.functional.one_hot(torch.tensor(labels), num_classes=10).type(torch.float32)
    # with torch.no_grad():
    #     images = model.sample(100, labels).cpu()

    # grid = Cifar10Manager().plot_image_grid(images)

    # matplotlib.image.imsave('grid.png', grid)


if __name__ == "__main__":
    main()
