import lightning as pl
import os
from torch.utils.data import DataLoader
from typing import Optional
from datasets.multiviewx_dataset import MultiviewX
from datasets.sampler import RandomPairSampler
from datasets.wildtrack_dataset import Wildtrack
from datasets.pedestrian_dataset import PedestrianDataset


class PedestrianDataModule(pl.LightningDataModule):
    def __init__(
            self,
            data_dir: str = "../data/MultiviewX",
            batch_size: int = 1,
            num_workers: int = 4,
            resolution=None,
            bounds=None,
            load_depth=False,
            use_grid_mask=False,
            history_frames=0,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.resolution = resolution
        self.bounds = bounds
        self.load_depth = load_depth
        self.dataset = os.path.basename(self.data_dir)
        self.data_predict = None
        self.data_test = None
        self.data_val = None
        self.data_train = None
        self.useGridMask = use_grid_mask
        self.history_frames = history_frames

    def setup(self, stage: Optional[str] = None):
        if 'wildtrack' in self.dataset.lower():
            base = Wildtrack(self.data_dir)
        elif 'multiviewx' in self.dataset.lower():
            base = MultiviewX(self.data_dir)
        else:
            raise ValueError(f'Unknown dataset name {self.dataset}')

        if stage == 'fit':
            self.data_train = PedestrianDataset(
                base,
                is_train=True,
                resolution=self.resolution,
                bounds=self.bounds,
                use_grid_mask=self.useGridMask,
                history_frames=self.history_frames,
            )
        if stage == 'fit' or stage == 'validate':
            self.data_val = PedestrianDataset(
                base,
                is_train=False,
                resolution=self.resolution,
                bounds=self.bounds,
            )
        if stage == 'test':
            self.data_test = PedestrianDataset(
                base,
                is_train=False,
                resolution=self.resolution,
                bounds=self.bounds
            )
        if stage == 'predict':
            self.data_predict = PedestrianDataset(
                base,
                is_train=False,
                resolution=self.resolution,
                bounds=self.bounds,
            )

    def train_dataloader(self):
        return DataLoader(
            self.data_train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            sampler=RandomPairSampler(self.data_train)
        )

    def val_dataloader(self):
        return DataLoader(
            self.data_val,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.data_test,
            batch_size=1,
            num_workers=self.num_workers
        )

    def predict_dataloader(self):
        return DataLoader(
            self.data_predict,
            batch_size=1,
            num_workers=self.num_workers
        )
