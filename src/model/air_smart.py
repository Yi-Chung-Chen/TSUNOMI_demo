# Not a contribution
# Changes made by NVIDIA CORPORATION & AFFILIATES enabling <CAT-K> or otherwise documented as
# NVIDIA-proprietary are not a contribution and subject to the following terms and conditions:
# SPDX-FileCopyrightText: Copyright (c) <year> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Aircraft SMART Lightning module: teacher-forced behavior cloning."""
import math
import lightning as L
import torch
from src.tokens.token_processor import AirTokenProcessor
from src.model.decoder import AirDecoder
from src.metrics.cross_entropy import token_cross_entropy
from src.common.torch_geometry import rotate
from src.common.units import to_metres


class AirSMART(L.LightningModule):
    def __init__(self, motion_dict, hidden_dim=64, num_heads=4, num_layers=4, lr=5e-4,
                 label_smoothing=.1, dropout=.1, time_span=18, current_frame_idx=18,
                 num_future_steps=90, num_types=32, num_freq_bands=16,
                 open_loop_modes=6, weight_decay=.01, lr_min_ratio=.01):
        super().__init__()
        self.save_hyperparameters()
        self.token_processor = AirTokenProcessor(motion_dict)
        shift = self.token_processor.shift
        if current_frame_idx % shift or num_future_steps % shift:
            raise ValueError('History and future frames must be divisible by shift')
        self.current_node = current_frame_idx // shift
        token_dim = self.token_processor.cluster_centers[0].numel() + self.token_processor.dz[0].numel()
        self.decoder = AirDecoder(K=self.token_processor.K, token_dim=token_dim, hidden_dim=hidden_dim,
                                  num_heads=num_heads, num_layers=num_layers, dropout=dropout,
                                  shift=shift, time_span=time_span, num_types=num_types, num_freq_bands=num_freq_bands)

    @property
    def token_features(self):
        p = self.token_processor
        return torch.cat([p.cluster_centers.flatten(1), p.dz], -1)

    def forward(self, batch):
        tokenized = self.token_processor.tokenize_agent(batch)
        return self.decoder(tokenized, self.token_features), tokenized

    def training_step(self, batch, batch_idx):
        logits, tokens = self(batch)
        result = token_cross_entropy(logits, tokens, self.current_node, self.hparams.label_smoothing)
        for name in ['loss', 'unsmoothed_ce', 'accuracy']:
            self.log(f'train/{name}', result[name], on_step=False, on_epoch=True, batch_size=max(1, result['count']))
        return result['loss']

    @torch.no_grad()
    def open_loop_minade(self, logits, tokens, batch):
        """Top-rank token modes with GT-conditioned anchors (teacher-forced)."""
        c, p = self.current_node, self.token_processor
        future_tokens = self.hparams.num_future_steps // p.shift
        ids = logits[:, c:c + future_tokens].topk(min(self.hparams.open_loop_modes, p.K), -1).indices
        traj = p.traj[ids]
        origin, head = tokens['gt_pos'][:, c:c + future_tokens], tokens['gt_heading'][:, c:c + future_tokens]
        xy = rotate(traj[..., 1:, :2], head[:, :, None, None]) + origin[:, :, None, None, :2]
        z = p.dz[ids][..., 1:] + origin[:, :, None, None, 2]
        position = torch.cat([xy, z[..., None]], -1).permute(0, 2, 1, 3, 4).flatten(2, 3)
        prediction = to_metres(position, p.meta['pos_scale'])
        a, now = batch['agent'], self.hparams.current_frame_idx
        raw = to_metres(a['position'][:, now + 1:now + self.hparams.num_future_steps + 1], p.meta['pos_scale'])
        eligible = a['valid_mask'][:, now:now + self.hparams.num_future_steps + 1].all(-1) & a['train_mask']
        if not eligible.any():
            return None, 0
        errors = (prediction[eligible] - raw[eligible, None]).norm(dim=-1).mean(-1).min(-1).values
        return errors.mean(), int(eligible.sum())

    def validation_step(self, batch, batch_idx):
        logits, tokens = self(batch)
        result = token_cross_entropy(logits, tokens, self.current_node, self.hparams.label_smoothing)
        self.log('val/loss', result['loss'], batch_size=max(1, result['count']), sync_dist=True)
        self.log('val/next_token_accuracy', result['accuracy'], batch_size=max(1, result['count']), sync_dist=True)
        ade, n = self.open_loop_minade(logits, tokens, batch)
        if n:
            self.log('val/open_loop_minADE_30s_m', ade, batch_size=n, sync_dist=True)
        self.log('val/open_loop_agents_30s', float(n), reduce_fx='sum', batch_size=1, sync_dist=True)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
        epochs = max(1, self.trainer.max_epochs)
        def schedule(epoch):
            ratio = .5 * (1 + math.cos(math.pi * epoch / epochs))
            return self.hparams.lr_min_ratio + (1 - self.hparams.lr_min_ratio) * ratio
        return dict(optimizer=optimizer, lr_scheduler=dict(scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer, schedule), interval='epoch'))
