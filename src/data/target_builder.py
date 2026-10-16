"""Contiguous history/future target selection, shared with preprocessing."""


class AirTargetBuilder:
    def __init__(self, current_frame_idx=18, min_future_frames=30):
        self.current_frame_idx = current_frame_idx
        self.min_future_frames = min_future_frames

    def __call__(self, data):
        valid = data['agent']['valid_mask']
        stop = self.current_frame_idx + self.min_future_frames + 1
        data['agent']['train_mask'] = valid[:, :stop].all(dim=-1)
        return data
