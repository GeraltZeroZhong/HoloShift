from typing import Optional

import pytorch_lightning as pl
from torch_geometric.loader import DataLoader

from .dataset import EvoPointDataset


class EvoPointDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir: str = "./data/processed_graphs",
        batch_size: int = 4,
        num_workers: int = 0,
        calib_batch_size: int = 1,
        split_seed: int = 42,
        split_ranges: dict | None = None,
        fallback_num_features: int = 144,
    ):
        super().__init__()
        if split_ranges is None:
            split_ranges = {
                "train": [0.0, 0.7],
                "val": [0.7, 0.8],
                "calib": [0.8, 0.9],
                "test": [0.9, 1.0],
                "all": [0.0, 1.0],
            }
        self.save_hyperparameters()

    def setup(self, stage: Optional[str] = None):
        if stage in ("fit", None):
            self.train_set = EvoPointDataset(
                self.hparams.data_dir,
                split="train",
                split_seed=self.hparams.split_seed,
                split_ranges=self.hparams.split_ranges,
                fallback_num_features=self.hparams.fallback_num_features,
            )
            self.val_set = EvoPointDataset(
                self.hparams.data_dir,
                split="val",
                split_seed=self.hparams.split_seed,
                split_ranges=self.hparams.split_ranges,
                fallback_num_features=self.hparams.fallback_num_features,
            )
            self.calib_set = EvoPointDataset(
                self.hparams.data_dir,
                split="calib",
                split_seed=self.hparams.split_seed,
                split_ranges=self.hparams.split_ranges,
                fallback_num_features=self.hparams.fallback_num_features,
            )
        if stage in ("test", None):
            self.test_set = EvoPointDataset(
                self.hparams.data_dir,
                split="test",
                split_seed=self.hparams.split_seed,
                split_ranges=self.hparams.split_ranges,
                fallback_num_features=self.hparams.fallback_num_features,
            )

    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=self.hparams.batch_size, shuffle=True, num_workers=self.hparams.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_set, batch_size=self.hparams.batch_size, shuffle=False, num_workers=self.hparams.num_workers)

    def calib_dataloader(self):
        return DataLoader(
            self.calib_set,
            batch_size=self.hparams.calib_batch_size,
            shuffle=False,
            num_workers=self.hparams.num_workers,
        )

    def test_dataloader(self):
        return DataLoader(self.test_set, batch_size=self.hparams.batch_size, shuffle=False, num_workers=self.hparams.num_workers)
