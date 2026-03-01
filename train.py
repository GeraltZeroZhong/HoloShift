import os
import sys

import hydra
import pytorch_lightning as pl
from omegaconf import DictConfig
from pytorch_lightning.callbacks import ModelCheckpoint, RichProgressBar
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning import seed_everything

sys.path.append(os.path.join(os.path.dirname(__file__), "src"))


@hydra.main(version_base="1.3", config_path="configs", config_name="train")
def main(cfg: DictConfig):
    seed_everything(cfg.seed, workers=True)
    root = hydra.utils.get_original_cwd()

    if not os.path.isabs(cfg.data.data_dir):
        cfg.data.data_dir = os.path.join(root, cfg.data.data_dir)

    datamodule = hydra.utils.instantiate(cfg.data)
    model = hydra.utils.instantiate(cfg.model)

    ckpt_dir = os.path.join(root, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    callbacks = [
        ModelCheckpoint(
            dirpath=ckpt_dir,
            filename="best-{epoch:02d}-{val_loss:.4f}",
            monitor="val/loss",
            mode="min",
            save_top_k=1,
            save_last=True,
        ),
        RichProgressBar(),
    ]
    logger = CSVLogger(save_dir=os.path.join(root, "logs"), name="holoshift")

    trainer = hydra.utils.instantiate(cfg.trainer, callbacks=callbacks, logger=logger, enable_checkpointing=True)
    trainer.fit(model=model, datamodule=datamodule)


if __name__ == "__main__":
    main()
