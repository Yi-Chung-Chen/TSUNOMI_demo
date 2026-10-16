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

"""SMART top-k probability sampling for BC-only inference."""
import torch


@torch.no_grad()
def sample_next_token(processor, logits, num_k=5, temp=1., generator=None):
    if temp <= 0 or num_k < 1:
        raise ValueError('Sampling temperature and K must be positive')
    values, candidates = logits.topk(min(num_k, processor.K), dim=-1, sorted=True)
    picked = torch.multinomial(torch.softmax(values / temp, dim=-1), 1, generator=generator)
    return candidates.gather(1, picked).squeeze(1)
