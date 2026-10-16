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

"""BC-only joint scene inference using observed history and top-k probability."""
import torch
from src.rollout.sampling import sample_next_token
from src.common.units import to_metres


def history_state(model, tokens, n_rollouts):
    current = model.current_node
    keys = ['gt_idx', 'gt_pos', 'gt_heading', 'valid_mask', 'on_ground']
    state = {key: tokens[key][:, :current + 1].clone() for key in keys}
    state['valid_mask'][:, -1] = True  # Dataset selects agents present at current.
    for key in ['type', 'train_mask']:
        state[key] = tokens[key]
    batch = tokens['batch']
    graphs = int(batch.max()) + 1
    for key, value in state.items():
        state[key] = value.repeat((n_rollouts,) + (1,) * (value.ndim - 1))
    state['batch'] = torch.cat([batch + r * graphs for r in range(n_rollouts)])
    return state


@torch.no_grad()
def rollout(model, batch, n_rollouts=32, num_k=5, temp=1., seed=817):
    # Truncate before tokenization, so future GT is never even matched.
    history = batch.clone()
    for key in ['position', 'heading', 'valid_mask', 'on_ground', 'alt_fallback']:
        history['agent'][key] = history['agent'][key][:, :model.hparams.current_frame_idx + 1]
    tokens = model.token_processor.tokenize_agent(history)
    p = model.token_processor
    state = history_state(model, tokens, n_rollouts)
    generator = torch.Generator(device=state['gt_pos'].device).manual_seed(seed)
    positions, headings = [], []
    n = len(tokens['type'])
    for step in range(model.hparams.num_future_steps // p.shift):
        logits = model.decoder(state, model.token_features)[:, -1]
        ids = sample_next_token(p, logits, num_k=num_k, temp=temp, generator=generator)
        decoded, decoded_head = p.decode(ids, state['gt_pos'][:, -1], state['gt_heading'][:, -1])
        positions.append(decoded[:, 1:])
        headings.append(decoded_head[:, 1:])
        state['gt_idx'] = torch.cat([state['gt_idx'], ids[:, None]], 1)
        state['gt_pos'] = torch.cat([state['gt_pos'], decoded[:, -1:]], 1)
        state['gt_heading'] = torch.cat([state['gt_heading'], decoded_head[:, -1:]], 1)
        state['valid_mask'] = torch.cat([state['valid_mask'], torch.ones_like(state['valid_mask'][:, :1])], 1)
        state['on_ground'] = torch.cat([state['on_ground'], state['on_ground'][:, -1:]], 1)
    dense = to_metres(torch.cat(positions, 1), p.meta['pos_scale'])
    return dict(position=dense.reshape(n_rollouts, n, -1, 3), heading=torch.cat(headings, 1).reshape(n_rollouts, n, -1))
