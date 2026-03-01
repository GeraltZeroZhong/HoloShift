from typing import Optional

import pytorch_lightning as pl
from torch_geometric.loader import DataLoader

from .dataset import EvoPointDataset


class EvoPointDataModule(pl.LightningDataModule):
    def __init__(self, data_dir: str = "./data/processed_graphs", batch_size: int = 4, num_workers: int = 0):
        super().__init__()
        self.save_hyperparameters()

    def setup(self, stage: Optional[str] = None):
        if stage in ("fit", None):
            self.train_set = EvoPointDataset(self.hparams.data_dir, split="train")
            self.val_set = EvoPointDataset(self.hparams.data_dir, split="val")
            self.calib_set = EvoPointDataset(self.hparams.data_dir, split="calib")
        if stage in ("test", None):
            self.test_set = EvoPointDataset(self.hparams.data_dir, split="test")

    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=self.hparams.batch_size, shuffle=True, num_workers=self.hparams.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_set, batch_size=self.hparams.batch_size, shuffle=False, num_workers=self.hparams.num_workers)

    def calib_dataloader(self):
        return DataLoader(self.calib_set, batch_size=1, shuffle=False, num_workers=self.hparams.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_set, batch_size=self.hparams.batch_size, shuffle=False, num_workers=self.hparams.num_workers)
