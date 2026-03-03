import glob
import os
import random
from typing import List

import torch
from torch_geometric.data import Data, InMemoryDataset


class EvoPointDataset(InMemoryDataset):
    SPLITS = {
        "train": (0.0, 0.7),
        "val": (0.7, 0.8),
        "calib": (0.8, 0.9),
        "test": (0.9, 1.0),
        "all": (0.0, 1.0),
    }

    def __init__(self, root: str, split: str = "train"):
        if split not in self.SPLITS:
            raise ValueError(f"Unknown split: {split}")
        self.split = split
        super().__init__(root)
        if not os.path.exists(self.processed_paths[0]):
            self.process()
        self.data, self.slices = torch.load(self.processed_paths[0], weights_only=False)

    @property
    def processed_file_names(self) -> List[str]:
        return [f"graph_cache_{self.split}.pt"]

    def process(self):
        raw_files = sorted(glob.glob(os.path.join(self.root, "*.pt")))
        random.seed(42)
        random.shuffle(raw_files)

        lo, hi = self.SPLITS[self.split]
        n = len(raw_files)
        files = raw_files[int(n * lo): int(n * hi)]

        if self.split == "val":
            print(f"[EvoPointDataset] val split selected {len(files)} files from {n} raw files")

        data_list = []
        skipped_missing = 0
        for f in files:
            d = torch.load(f, weights_only=False)
            if "x" not in d or "pos" not in d or "y_delta" not in d:
                skipped_missing += 1
                continue

            pos = d["pos"].float()
            y_delta = d["y_delta"].float()

            # Normalize coordinates to the origin to remove large-coordinate magnitude effects.
            pos = pos - pos.mean(dim=0, keepdim=True)

            data_list.append(
                Data(
                    x=d["x"].float(),
                    pos=pos,
                    y=y_delta,
                    edge_index=d.get("edge_index", None),
                    edge_attr=d.get("edge_attr", None),
                    pair_id=d.get("pair_id", os.path.splitext(os.path.basename(f))[0]),
                    residue_ids=d.get("residue_ids", None),
                )
            )

        if self.split == "val":
            print(
                "[EvoPointDataset] val split kept "
                f"{len(data_list)}/{len(files)} files "
                f"(missing={skipped_missing})"
            )

        if not data_list:
            data_list = [
                Data(
                    x=torch.zeros((1, 130), dtype=torch.float32),
                    pos=torch.zeros((1, 3), dtype=torch.float32),
                    y=torch.zeros((1, 3), dtype=torch.float32),
                    edge_index=torch.zeros((2, 0), dtype=torch.long),
                    edge_attr=torch.zeros((0, 2), dtype=torch.float32),
                )
            ]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
