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

import os
import pickle
"""3D adaptation of SMART token_processor._match_agent_token.

No map, perspective, actions or StarCraft state. GT matching chains from the
previous reconstructed state. Every frame in a token must be valid.
"""
import pickle
import torch
from torch import nn
from src.common.torch_geometry import rotate, contours, wrap_angle, causal_headings


class AirTokenProcessor(nn.Module):
    def __init__(self, motion_dict, num_k=1, temp=1., match_batch_size=128):
        super().__init__()
        with open(motion_dict, 'rb') as stream:
            d = pickle.load(stream)
        self.meta = d['meta']
        self.shift, self.K = self.meta['shift'], self.meta['K']
        self.num_k, self.temp, self.match_batch_size = num_k, temp, match_batch_size
        for name in ['traj', 'dz', 'cluster_centers']:
            self.register_buffer(name, torch.from_numpy(d[name]).float())

    def distance(self, target_pos, target_heading, previous_pos, previous_heading, endpoint_idx=-1):
        local = rotate(target_pos[:, :2] - previous_pos[:, :2], -previous_heading)
        target = contours(local, wrap_angle(target_heading - previous_heading), self.meta['radius'])
        diff = target[:, None] - self.cluster_centers[None, :, endpoint_idx]
        dist = diff.square().sum(-1).sqrt().mean(-1)
        delta_z = target_pos[:, 2] - previous_pos[:, 2]
        return dist + self.meta['lambda_z'] * (delta_z[:, None] - self.dz[None, :, endpoint_idx]).abs()

    def match(self, pos, heading, previous_pos, previous_heading, endpoint_idx=-1, sample=False):
        ids = []
        for start in range(0, len(pos), self.match_batch_size):
            sl = slice(start, start + self.match_batch_size)
            dist = self.distance(pos[sl], heading[sl], previous_pos[sl], previous_heading[sl], endpoint_idx)
            if sample and self.num_k > 1:
                values, indices = dist.topk(min(self.num_k, self.K), largest=False)
                choice = torch.multinomial(torch.softmax(-values / self.temp, -1), 1)
                ids.append(indices.gather(1, choice).squeeze(-1))
            else:
                ids.append(dist.argmin(-1))
        return torch.cat(ids) if ids else torch.empty(0, device=pos.device, dtype=torch.long)

    def decode(self, ids, previous_pos, previous_heading):
        traj = self.traj[ids]
        xy = rotate(traj[..., :2], previous_heading[:, None]) + previous_pos[:, None, :2]
        z = self.dz[ids] + previous_pos[:, None, 2]
        heading = wrap_angle(traj[..., 2] + previous_heading[:, None])
        return torch.cat([xy, z[..., None]], -1), heading

    @torch.no_grad()
    def reconstruct(self, position, heading, valid, on_ground, start_frame=0,
                    horizon_frames=None, one_shot=False, sample=False, heading_prepared=False):
        # Keep matching/rollout geometry in float64; model embeddings use float32.
        position, heading = position.double(), heading.double()
        if not heading_prepared and self.meta['heading_source'] == 'chord':
            heading = causal_headings(position, heading, valid, on_ground, self.shift, self.meta['pos_scale'])
        horizon_frames = position.shape[1] - start_frame - 1 if horizon_frames is None else horizon_frames
        previous_pos = position[:, start_frame].clone()
        previous_heading = heading[:, start_frame].clone()
        ids_list, positions, headings, token_valid, dense = [], [previous_pos], [previous_heading], [valid[:, start_frame]], []
        for offset in range(0, horizon_frames, self.shift):
            length = min(self.shift, horizon_frames - offset)
            begin, end = start_frame + offset, start_frame + offset + length
            ok = valid[:, begin:end + 1].all(-1)
            if one_shot:
                previous_pos, previous_heading = position[:, begin], heading[:, begin]
            ids = self.match(position[:, end], heading[:, end], previous_pos, previous_heading, length, sample)
            decoded, head = self.decode(ids, previous_pos, previous_heading)
            dense.append(decoded[:, 1:length + 1])
            previous_pos = torch.where(ok[:, None], decoded[:, length], position[:, end])
            previous_heading = torch.where(ok, head[:, length], heading[:, end])
            ids_list.append(torch.where(ok, ids, torch.zeros_like(ids)))
            positions.append(previous_pos)
            headings.append(previous_heading)
            token_valid.append(ok)
        zeros = torch.zeros((len(position), 1), dtype=torch.long, device=position.device)
        return dict(idx=torch.cat([zeros, torch.stack(ids_list, 1)], 1),
                    position=torch.stack(positions, 1), heading=torch.stack(headings, 1),
                    valid_mask=torch.stack(token_valid, 1), dense_position=torch.cat(dense, 1))

    @torch.no_grad()
    def tokenize_agent(self, data):
        a = data['agent']
        p, h, v, g = a['position'].double(), a['heading'].double(), a['valid_mask'], a['on_ground']
        if self.meta['heading_source'] == 'chord':
            h = causal_headings(p, h, v, g, self.shift, self.meta['pos_scale'])
        gt = self.reconstruct(p, h, v, g, heading_prepared=True)
        sampled = gt if self.num_k == 1 else self.reconstruct(p, h, v, g, sample=True, heading_prepared=True)
        steps = torch.arange(0, p.shape[1], self.shift, device=p.device)
        raw = p[:, steps]
        return dict(gt_idx=gt['idx'], gt_pos=gt['position'], gt_heading=gt['heading'],
                    gt_z=gt['position'][..., 2], gt_z_raw=raw[..., 2],
                    gt_dz=torch.cat([torch.zeros_like(raw[:, :1, 2]), gt['position'][:, 1:, 2] - gt['position'][:, :-1, 2]], 1),
                    gt_pos_raw=raw, gt_heading_raw=h[:, steps],
                    gt_dense=gt['dense_position'], valid_mask=gt['valid_mask'],
                    sampled_idx=sampled['idx'], sampled_pos=sampled['position'],
                    sampled_heading=sampled['heading'], sampled_z=sampled['position'][..., 2],
                    on_ground=g[:, steps], type=a['type'], train_mask=a['train_mask'],
                    batch=a['batch'] if 'batch' in a else torch.zeros(len(p), device=p.device, dtype=torch.long))

    def forward(self, data):
        return {}, self.tokenize_agent(data)
