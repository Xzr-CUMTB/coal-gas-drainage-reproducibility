"""Hole-centred polar POD on a symmetry half-domain. No samples inside holes."""

from base import *

NA, NR = 97, 81


def ray(d, a):
    co = np.cos(a)
    si = np.sin(a)
    return np.minimum(
        np.where(co >= 0, 8 - d / 2, d / 2) / np.maximum(abs(co), 1e-12),
        6 / np.maximum(abs(si), 1e-12),
    )


def nodes(d):
    a = np.linspace(0, np.pi, NA)
    u = np.linspace(0, 1, NR)
    rad = R * np.exp(u[:, None] * np.log(ray(d, a) / R))
    return np.stack([d / 2 + rad * np.cos(a), rad * np.sin(a)], axis=-1).reshape(-1, 2)


def interpolate(a, d, xy):
    x = abs(xy[:, 0]) - d / 2
    y = abs(xy[:, 1])
    ang = np.arctan2(y, x)
    rr = np.hypot(x, y)
    u = np.clip(np.log(np.maximum(rr, R) / R) / np.log(ray(d, ang) / R), 0, 1) * (NR - 1)
    v = ang / np.pi * (NA - 1)
    i = np.minimum(u.astype(int), NR - 2)
    j = np.minimum(v.astype(int), NA - 2)
    u = (u - i)[:, None]
    v = (v - j)[:, None]
    return (a[i, j] * (1 - v) + a[i, j + 1] * v) * (1 - u) + (
        a[i + 1, j] * (1 - v) + a[i + 1, j + 1] * v
    ) * u


def param(d, k):
    return [(d - 4.5) / 1.5, np.log2(k)]


def reconstruct(d, k):
    with np.load(D / "pod.npz") as z:
        return (
            (np.exp(-float(z["gamma"]) * ((z["X"] - param(d, k)) ** 2).sum(1)) @ z["coef"]) @ z["B"]
            + z["mu"]
        ).reshape(10, NR, NA, 2)
