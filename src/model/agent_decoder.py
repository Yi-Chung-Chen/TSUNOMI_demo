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

"""SMART temporal/agent attention with 3D geometry and optional runway anchors.

Adapted from src/smart/modules/agent_decoder.py; shared attention, Fourier
embeddings, MLP and initialization are copied with their licenses preserved.
"""
import torch
from torch import nn
from src.common.attention_layer import AttentionLayer
from src.common.fourier_embedding import FourierEmbedding, MLPEmbedding
from src.common.mlp_layer import MLPLayer
from src.common.torch_geometry import rotate, wrap_angle
from src.common.units import FRAME_HZ


class AirAgentDecoder(nn.Module):
    def __init__(self, K, token_dim, hidden_dim=64, num_heads=4, num_layers=4,
                 num_freq_bands=16, dropout=.1, shift=6, time_span=18, num_types=32,
                 runway_anchors=False, runway_anchor_path=None, pos_scale=10.):
        super().__init__()
        if hidden_dim % num_heads:
            raise ValueError('hidden_dim must be divisible by num_heads')
        self.shift, self.time_span = shift, time_span
        self.hidden_dim = hidden_dim
        self.type_emb = nn.Embedding(num_types, hidden_dim)
        self.token_emb = MLPEmbedding(input_dim=token_dim, hidden_dim=hidden_dim)
        self.x_emb = FourierEmbedding(input_dim=5, hidden_dim=hidden_dim, num_freq_bands=num_freq_bands)
        self.temporal_emb = FourierEmbedding(input_dim=6, hidden_dim=hidden_dim, num_freq_bands=num_freq_bands)
        self.spatial_emb = FourierEmbedding(input_dim=6, hidden_dim=hidden_dim, num_freq_bands=num_freq_bands)
        self.fusion = MLPEmbedding(input_dim=hidden_dim * 2, hidden_dim=hidden_dim)
        kwargs = dict(hidden_dim=hidden_dim, num_heads=num_heads, head_dim=hidden_dim // num_heads,
                      dropout=dropout, bipartite=False, has_pos_emb=True)
        self.temporal_layers = nn.ModuleList([AttentionLayer(**kwargs) for _ in range(num_layers)])
        self.spatial_layers = nn.ModuleList([AttentionLayer(**kwargs) for _ in range(num_layers)])
        self.next_token_head = MLPLayer(hidden_dim, hidden_dim, K)
        self.runway_anchors = runway_anchors
        if runway_anchors:
            import csv
            from pathlib import Path
            path = Path(runway_anchor_path) if runway_anchor_path else Path(__file__).resolve().parents[2]/'assets/ord_runway_ends.csv'
            with path.open() as stream:
                rows = list(csv.DictReader(stream))
            if len(rows)!=16 or [int(r['anchor_id']) for r in rows]!=list(range(16)):
                raise ValueError('Expected fixed16 runway ends in anchor_id order')
            self.register_buffer('anchor_position',torch.tensor([[float(r[k])/pos_scale for k in ['x_m','y_m','z_m']] for r in rows],dtype=torch.float32))
            self.register_buffer('anchor_heading',torch.tensor([float(r['inbound_heading_rad']) for r in rows],dtype=torch.float32))
            self.anchor_emb = nn.Embedding(16,hidden_dim)
            self.anchor_pos_emb = FourierEmbedding(input_dim=6,hidden_dim=hidden_dim,num_freq_bands=num_freq_bands)
            self.anchor_layers = nn.ModuleList([AttentionLayer(hidden_dim=hidden_dim,num_heads=num_heads,
                head_dim=hidden_dim//num_heads,dropout=dropout,bipartite=True,has_pos_emb=True) for _ in range(num_layers)])

    def build_edges(self, valid, batch):
        n, steps = valid.shape
        device = valid.device
        src_t, dst_t = torch.meshgrid(torch.arange(steps, device=device), torch.arange(steps, device=device), indexing='ij')
        keep = (src_t < dst_t) & ((dst_t - src_t) * self.shift <= self.time_span)
        src_t, dst_t = src_t[keep], dst_t[keep]
        agents = torch.arange(n, device=device)[:, None]
        src = agents * steps + src_t
        dst = agents * steps + dst_t
        mask = valid[:, src_t] & valid[:, dst_t]
        temporal = torch.stack([src[mask], dst[mask]])
        same = (batch[:, None] == batch[None]) & ~torch.eye(n, dtype=torch.bool, device=device)
        a, b = same.nonzero(as_tuple=True)
        times = torch.arange(steps, device=device)[None]
        src, dst = a[:, None] * steps + times, b[:, None] * steps + times
        mask = valid[a] & valid[b]
        spatial = torch.stack([src[mask], dst[mask]])
        return temporal, spatial

    def relative_features(self, position, heading, edge_index, steps, temporal):
        src, dst = edge_index
        delta = position[src] - position[dst]
        local = rotate(delta[:, :2], -heading[dst])
        distance = delta[:, :2].norm(dim=-1)
        angle = wrap_angle(heading[src] - heading[dst])
        if temporal:
            last = ((src % steps) - (dst % steps)).float() * self.shift / FRAME_HZ
        else:
            last = torch.atan2(local[:, 1], local[:, 0])
        return torch.cat([local, delta[:, 2:3], distance[:, None], angle[:, None], last[:, None]], -1)

    def anchor_edges(self,position,heading,valid):
        """Shared16 sources; pose-dependent features recomputed on EVERY forward."""
        n,steps=valid.shape
        nodes=valid.reshape(-1).nonzero(as_tuple=True)[0]
        edge=torch.stack([torch.arange(16,device=position.device).repeat(len(nodes)),nodes.repeat_interleave(16)])
        pos=torch.cat([self.anchor_position.to(position.dtype),position.reshape(-1,3)])
        heads=torch.cat([self.anchor_heading.to(heading.dtype),heading.reshape(-1)])
        geometry_edge=edge.clone();geometry_edge[1]+=16
        features=self.relative_features(pos,heads,geometry_edge,steps,False)
        return edge,features

    def forward(self, tokenized, token_features, sampled=False):
        prefix = 'sampled' if sampled else 'gt'
        ids, pos, heading = (tokenized[f'{prefix}_idx'], tokenized[f'{prefix}_pos'].float(),
                             tokenized[f'{prefix}_heading'].float())
        valid = tokenized['valid_mask']
        n, steps = ids.shape
        motion = torch.cat([torch.zeros_like(pos[:, :1]), pos[:, 1:] - pos[:, :-1]], 1)
        feature = torch.stack([motion[..., :2].norm(dim=-1),
                               wrap_angle(torch.atan2(motion[..., 1], motion[..., 0]) - heading),
                               pos[..., 2], tokenized['on_ground'].float(), motion[..., 2]], -1)
        categorical = self.type_emb(tokenized['type'])[:, None].expand(-1, steps, -1).reshape(-1, self.hidden_dim)
        state = self.x_emb(feature.reshape(-1, 5), [categorical]).reshape(n, steps, -1)
        token = self.token_emb(token_features.float())[ids]
        x = self.fusion(torch.cat([state, token], -1)).reshape(n * steps, -1)
        edge_t, edge_a = self.build_edges(valid, tokenized['batch'])
        flat_pos, flat_head = pos.reshape(-1, 3), heading.reshape(-1)
        r_t = self.temporal_emb(self.relative_features(flat_pos, flat_head, edge_t, steps, True))
        r_a = self.spatial_emb(self.relative_features(flat_pos, flat_head, edge_a, steps, False))
        if self.runway_anchors:
            edge_anchor,features_anchor=self.anchor_edges(pos,heading,valid)
            r_anchor=self.anchor_pos_emb(features_anchor)
            anchor_x=self.anchor_emb.weight
        for layer_index,(temporal, spatial) in enumerate(zip(self.temporal_layers, self.spatial_layers)):
            x = temporal(x, r_t, edge_t)
            x = spatial(x, r_a, edge_a)
            if self.runway_anchors:
                x=self.anchor_layers[layer_index]((anchor_x,x),r_anchor,edge_anchor)
        return self.next_token_head(x.reshape(n, steps, -1))
