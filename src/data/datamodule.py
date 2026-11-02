import lightning as L
from torch_geometric.loader import DataLoader
from src.data.dataset import AirDataset


class AirDataModule(L.LightningDataModule):
    def __init__(self, processed_dir, split_dir, pos_scale=10., current_frame_idx=18,
                 future_frames=90, min_future_frames=30, radius=2., train_batch_size=4,
                 val_batch_size=4, num_workers=4, val_windows_path=None):
        super().__init__()
        self.dataset_config = dict(processed_dir=processed_dir, split_dir=split_dir, pos_scale=pos_scale,
                                   current_frame_idx=current_frame_idx, future_frames=future_frames,
                                   min_future_frames=min_future_frames, radius=radius)
        self.train_batch_size, self.val_batch_size, self.num_workers = train_batch_size, val_batch_size, num_workers
        self.val_windows_path=val_windows_path

    def setup(self, stage=None):
        if stage in (None, 'fit'):
            self.train_dataset = AirDataset(**self.dataset_config, split='train')
            self.val_dataset = self._validation_dataset()
        if stage == 'validate':
            self.val_dataset = self._validation_dataset()
        if stage == 'test':
            raise ValueError('This dataset has no test split; use merged val')

    def _validation_dataset(self):
        dataset=AirDataset(**self.dataset_config,split='val')
        if self.val_windows_path:
            import csv
            with open(self.val_windows_path) as stream:rows=list(csv.DictReader(stream))
            existing={(r['scenario'],r['start_frame'],r['n_targets']) for r in dataset.windows}
            if not rows or any((r['scenario'],r['start_frame'],r['n_targets']) not in existing for r in rows):
                raise ValueError('Validation override must be a nonempty subset of existing windows')
            dataset.windows=rows
        return dataset

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
