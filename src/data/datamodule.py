import lightning as L
from torch_geometric.loader import DataLoader
from src.data.dataset import AirDataset


class AirDataModule(L.LightningDataModule):
    def __init__(self, processed_dir, split_dir, pos_scale=10., current_frame_idx=18,
                 future_frames=90, min_future_frames=30, radius=2., train_batch_size=4,
                 val_batch_size=4, num_workers=4):
        super().__init__()
        self.dataset_config = dict(processed_dir=processed_dir, split_dir=split_dir, pos_scale=pos_scale,
                                   current_frame_idx=current_frame_idx, future_frames=future_frames,
                                   min_future_frames=min_future_frames, radius=radius)
        self.train_batch_size, self.val_batch_size, self.num_workers = train_batch_size, val_batch_size, num_workers

    def setup(self, stage=None):
        if stage in (None, 'fit'):
            self.train_dataset = AirDataset(**self.dataset_config, split='train')
            self.val_dataset = AirDataset(**self.dataset_config, split='val')
        if stage == 'validate':
            self.val_dataset = AirDataset(**self.dataset_config, split='val')
        if stage == 'test':
            raise ValueError('This dataset has no test split; use merged val')

    def _loader(self, dataset, batch_size, shuffle=False):
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                          num_workers=self.num_workers, persistent_workers=self.num_workers > 0,
                          pin_memory=False)

    def train_dataloader(self):
        return self._loader(self.train_dataset, self.train_batch_size, True)

    def val_dataloader(self):
        return self._loader(self.val_dataset, self.val_batch_size)

    def test_dataloader(self):
        raise ValueError('No test split is defined')
