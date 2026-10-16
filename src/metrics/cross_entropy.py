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

"""Masked token CE with smoothing and a separate unsmoothed diagnostic."""
import torch
import torch.nn.functional as F


def future_token_mask(tokenized, current_node):
    valid = tokenized['valid_mask']
    contiguous = valid[:, current_node:].long().cumprod(-1).bool()
    result = torch.zeros_like(valid[:, 1:])
    result[:, current_node:] = contiguous[:, 1:]
    return result & valid[:, :-1] & tokenized['train_mask'][:, None]


def token_cross_entropy(logits, tokenized, current_node, label_smoothing=.1):
    mask = future_token_mask(tokenized, current_node)
    prediction = logits[:, :-1][mask]
    target = tokenized['gt_idx'][:, 1:][mask]
    if not len(target):
        zero = logits.sum() * 0.
        return dict(loss=zero, unsmoothed_ce=zero, accuracy=zero.detach(), count=0)
    loss = F.cross_entropy(prediction, target, label_smoothing=label_smoothing)
    with torch.no_grad():
        ce = F.cross_entropy(prediction, target)
        accuracy = (prediction.argmax(-1) == target).float().mean()
    return dict(loss=loss, unsmoothed_ce=ce, accuracy=accuracy, count=len(target))
