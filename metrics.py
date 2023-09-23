from pathlib import Path

from torchmetrics.image.inception import InceptionScore
from torchmetrics.image.fid import FrechetInceptionDistance
import wandb
from image_managers import Cifar10Manager

from latent_diffusion import LatentDiffusion
from vae import VAE


def get_artifact_path(checkpoint_reference: str):
    api = wandb.Api()
    artifact = api.artifact(checkpoint_reference)
    artifact_dir = artifact.download()
    return Path(artifact_dir) / "model.ckpt"


def main():
    image_manager = Cifar10Manager("../data")

    checkpoint_reference = "freezingarrow/StableDiffusion/model-cifar10-vae-v28-ldm-v05-uncond:v0"
    model = LatentDiffusion.load_from_checkpoint( get_artifact_path(checkpoint_reference) )
    # vae_checkpoint_reference = "freezingarrow/StableDiffusion/model-cifar10-vae-v26:v0"
    # model = VAE.load_from_checkpoint(get_artifact_path(vae_checkpoint_reference))

    model.to("cuda")
    model.eval()

    inception = InceptionScore(normalize=True)
    fid = FrechetInceptionDistance(normalize=True)

    loader = image_manager.create_train_loader(batch_size=100)
    loader_iter = iter(loader)

    for i in range(10):
        print(i, flush=True)

        real, _ = next(loader_iter)

        # z, _ = model.encode(real.to("cuda"))
        # fake = model.decode(z).cpu()
        fake = model.sample(100).cpu()

        real = image_manager.denorm(real)
        fid.update(real, real=True)


        fake = image_manager.denorm(fake)
        inception.update(fake)
        fid.update(fake, real=False)


    print(inception.compute())
    print(fid.compute())

    # labels = torch.tensor([i//10 for i in range(100)])
    # labels = torch.nn.functional.one_hot(torch.tensor(labels), num_classes=10).type(torch.float32)
    # with torch.no_grad():
    #     images = model.sample(100, labels).cpu()

    # grid = Cifar10Manager().plot_image_grid(images)

    # matplotlib.image.imsave('grid.png', grid)


if __name__ == "__main__":
    main()
