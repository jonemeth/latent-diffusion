"""
main.py
"""

import lightning.pytorch as pl
import matplotlib.pyplot as plt
import torch.nn
from lightning.pytorch.loggers import CSVLogger
from lightning.pytorch.callbacks import StochasticWeightAveraging
from celeba import get_loader, color_std, color_mean
from diffusion import Diffusion


BATCH_SIZE = 48
IMAGE_SIZE = 64
TIME_STEPS = 1000

# torch.autograd.set_detect_anomaly(True)


class Callback(pl.callbacks.Callback):
    def __init__(self, loader):
        super().__init__()

        self.epoch_ix = 0

        self.n = 64

        ys = []
        for _, y in loader:
            ys.append(y)

            if sum(y.shape[0] for y in ys) >= self.n:
                break

        self.y = torch.cat(ys, dim=0)[0:self.n].type(torch.float32)

    def on_train_epoch_end(self, trainer, pl_module):
        pl_module.eval()

        if 0 != trainer.global_rank:
            return
        self.epoch_ix += 1

        if 0 != self.epoch_ix % 5:
            return

        with torch.no_grad():
            predictions = pl_module.sample(self.n, self.y).detach().cpu()

        predictions *= torch.tensor(color_std)[None, :, None, None]
        predictions += torch.tensor(color_mean)[None, :, None, None]
        predictions = torch.clamp(predictions, 0.0, 1.0)

        predictions = predictions.permute(0, 2, 3, 1)

        fig = plt.figure(figsize=(8, 8), dpi=120)

        for i in range(predictions.shape[0]):
            plt.subplot(8, 8, i + 1)
            plt.imshow(predictions[i, :, :, :])
            plt.axis('off')

        plt.savefig(f"image_at_epoch_{self.epoch_ix:04d}.png", dpi=120)
        plt.close(fig)

        pl_module.train()


def main():
    train_loader = get_loader(IMAGE_SIZE, BATCH_SIZE)
    model = Diffusion(IMAGE_SIZE, TIME_STEPS)

    trainer = pl.Trainer(max_epochs=300,
                         log_every_n_steps=10,
                         logger=CSVLogger("logs", name="my_exp_name", flush_logs_every_n_steps=1),
                         callbacks=[Callback(train_loader),
                                    StochasticWeightAveraging(swa_lrs=1e-5)])
    trainer.fit(model=model, train_dataloaders=train_loader)


if __name__ == "__main__":
    main()
