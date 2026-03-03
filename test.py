import os
import sys

import hydra
from omegaconf import DictConfig
from pytorch_lightning import seed_everything

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))


@hydra.main(version_base="1.3", config_path="configs", config_name="train")
def main(cfg: DictConfig):
    """Run evaluation using the exact same data pipeline setup as train.py."""
    seed_everything(cfg.seed, workers=True)
    root = hydra.utils.get_original_cwd()

    if not os.path.isabs(cfg.data.data_dir):
        cfg.data.data_dir = os.path.join(root, cfg.data.data_dir)

    datamodule = hydra.utils.instantiate(cfg.data)
    model = hydra.utils.instantiate(cfg.model)

    trainer = hydra.utils.instantiate(
        cfg.trainer,
        logger=False,
        enable_checkpointing=False,
    )

    ckpt_path = cfg.get("ckpt_path", None)
    trainer.test(model=model, datamodule=datamodule, ckpt_path=ckpt_path)


if __name__ == "__main__":
    main()
