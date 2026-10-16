"""Torch equivalents of dictionary geometry (same conventions and units)."""
import torch
from src.common.units import to_metres, FRAME_HZ


def wrap_angle(value):
    return torch.atan2(torch.sin(value), torch.cos(value))


def rotate(xy, heading):
    c, s = heading.cos(), heading.sin()
    return torch.stack([xy[..., 0] * c - xy[..., 1] * s, xy[..., 0] * s + xy[..., 1] * c], -1)


def contours(position, heading, radius=2.):
    offsets = heading.new_tensor([0., torch.pi / 2, torch.pi, 3 * torch.pi / 2])
    angle = heading[..., None] + offsets
    return position[..., None, :2] + radius * torch.stack([angle.cos(), angle.sin()], -1)


def causal_headings(position, heading, valid, on_ground, shift, pos_scale):
    result = heading.clone()
    delta = position[:, shift:, :2] - position[:, :-shift, :2]
    speed = to_metres(delta, pos_scale).norm(dim=-1) * FRAME_HZ / shift
    eligible = valid.unfold(1, shift + 1, 1).all(-1) & ~on_ground[:, shift:] & (speed > 10.)
    result[:, shift:] = torch.where(eligible, torch.atan2(delta[..., 1], delta[..., 0]), result[:, shift:])
    return result
