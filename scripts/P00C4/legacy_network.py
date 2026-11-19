"""Symmetric two-pressure decoder with exact fracture-wall values and optional local branch."""

import numpy as np
import torch
from torch import nn

R = 0.047
C = 0.016 / (8.314462618 * 300)
RSTD = 0.016 * 101000 / (8.314462618 * 273.5)
RB = 1228.5
GF = 0.98 * C / 153360


def storage(pm, pf):
    return 0.07 * C * pm + RB * RSTD * 0.016 * pm / (2450000 + pm), 0.02 * C * pf, GF * (pm - pf)


def capacity(pm):
    return 0.07 * C + RB * RSTD * 0.016 * 2450000 / (2450000 + pm) ** 2


def dense(n, m, width, depth):
    layers = [nn.Linear(n, width), nn.SiLU()]
    for _ in range(depth - 1):
        layers.extend([nn.Linear(width, width), nn.SiLU()])
    layers.append(nn.Linear(width, m))
    return nn.Sequential(*layers)


class FieldModel(nn.Module):
    def __init__(self, local=True):
        super().__init__()
        self.local_enabled = local
        self.condition = dense(2, 16, 32, 2)
        self.far = dense(28, 2, 128, 4)
        if local:
            self.local = dense(26, 2, 96, 3)

    def p(self, d, k, xy, t):
        x, y = xy[:, 0].abs(), xy[:, 1].abs()
        r1 = torch.sqrt((x - d / 2) ** 2 + y * y)
        r2 = torch.sqrt((x + d / 2) ** 2 + y * y)
        dist1 = (r1 - R).clamp_min(0)
        dist2 = (r2 - R).clamp_min(0)
        # Snap only floating-point geometric roundoff. The band is registered separately from the physical radius.
        dist1 = torch.where(dist1 <= 1e-6, torch.zeros_like(dist1), dist1)
        dist2 = torch.where(dist2 <= 1e-6, torch.zeros_like(dist2), dist2)
        ell1 = torch.log1p(dist1 / R)
        ell2 = torch.log1p(dist2 / R)
        phi = ell1 * ell2 / (ell1 + ell2 + 1e-8)
        tm = torch.as_tensor(t, device=xy.device, dtype=xy.dtype)
        tm = tm.expand(len(xy)) if tm.ndim == 0 else tm
        lt = torch.log1p(tm / 3600) / 8
        day = tm / 86400
        const = lambda v: torch.full_like(x, float(v))
        cond = self.condition(
            torch.stack([const((d - 4.5) / 1.5), const(np.log(k) / np.log(2))], 1)
        )
        base = torch.stack(
            [
                x / 8,
                y / 6,
                lt,
                torch.sqrt(day / 90),
                ell1 / 6,
                ell2 / 6,
                dist1.clamp_max(3) / 3,
                dist2.clamp_max(3) / 3,
                (x - d / 2) / r1,
                y / r1,
                torch.exp(-dist1 / (0.05 + 0.12 * torch.sqrt(day * k))),
                torch.log1p(day * k) / 6,
            ],
            1,
        )
        raw = self.far(torch.cat([cond, base], 1))
        if self.local_enabled:
            localin = torch.stack(
                [
                    ell1 / 6,
                    ell2 / 6,
                    lt,
                    torch.sqrt(day / 90),
                    (x - d / 2) / r1,
                    y / r1,
                    torch.log1p(day * k) / 6,
                    dist1.clamp_max(1),
                    x / 8,
                    y / 6,
                ],
                1,
            )
            loc = self.local(torch.cat([cond, localin], 1))
            blend = 1 / (1 + (dist1 / 0.3) ** 4)
            raw = raw + blend[:, None] * loc
        pm = 100000 + 500000 * torch.sigmoid(raw[:, 0])
        # Positive distance factor is zero on either circular wall. No matrix-wall constraint is imposed.
        f = -torch.expm1(-phi * nn.functional.softplus(raw[:, 1]))
        pf = torch.sqrt(100000**2 + (600000**2 - 100000**2) * f)
        return torch.stack([pm, pf], 1)
