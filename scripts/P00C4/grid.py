"""Shared positive quadrature, rectangular probes, wall arcs and near-hole diagnostics."""

import numpy as np
from common import DATA

TIMES = np.array(
    [
        0.0,
        0.01,
        3600,
        21600,
        86400,
        3 * 86400,
        7 * 86400,
        10 * 86400,
        20 * 86400,
        30 * 86400,
        60 * 86400,
        90 * 86400,
    ]
)
MAIN = np.array([3600, 86400, 864000, 2592000, 7776000])
R = 0.047


def quadrature(d):
    # Gauss abscissae in x; for each x exclude the exact circular cavity before y quadrature.
    centers = (-d / 2, d / 2)
    xs = list(np.linspace(-8, 8, 81))
    ys = list(np.linspace(-6, 6, 61)) + [-5, 5]
    for a in centers:
        for off in [
            -0.25,
            -0.141,
            -0.094,
            -R,
            -0.035,
            -0.02,
            0,
            0.02,
            0.035,
            R,
            0.094,
            0.141,
            0.25,
        ]:
            xs.append(a + off)
    ys += list(np.array([-0.25, -0.141, -0.094, -R, -0.02, 0, 0.02, R, 0.094, 0.141, 0.25]))
    xs = np.unique(np.round(xs, 9))
    ys = np.unique(np.round(ys, 9))
    g, w = np.polynomial.legendre.leggauss(3)
    blocks = []
    for xl, xr in zip(xs[:-1], xs[1:]):
        for xx, wx in zip((xl + xr) / 2 + (xr - xl) * g / 2, w * (xr - xl) / 2):
            dy = min(abs(xx - a) for a in centers)
            cut = np.sqrt(max(0, R * R - dy * dy))
            edges = ys if not cut else np.unique(np.r_[ys, -cut, cut])
            lo, hi = edges[:-1], edges[1:]
            ok = np.abs((lo + hi) / 2) >= cut
            lo, hi = lo[ok, None], hi[ok, None]
            y = ((lo + hi) / 2 + (hi - lo) * g / 2).ravel()
            weight = (wx * (hi - lo) * w / 2).ravel()
            blocks.append(np.column_stack([np.full(len(y), xx), y, weight]))
    return np.concatenate(blocks)


def build(d, root=None):
    directory = (DATA if root is None else root) / f"grid_D{d:.9f}"
    directory.mkdir(parents=True, exist_ok=True)
    if (directory / "geometry.npz").exists():
        with np.load(directory / "geometry.npz") as g:
            x, y = np.meshgrid(g["cart_x"], g["cart_y"])
            mask = g["mask"]
            coords = np.concatenate(
                [g["quad_xy"], np.c_[x[mask], y[mask]], g["near_xy"], g["line_xy"]]
            )
        if not (directory / "coordinate_input.csv").exists():
            np.savetxt(directory / "coordinate_input.csv", coords, delimiter=",", fmt="%.15g")
        return directory, len(coords)
    q = quadrature(d)
    ctr = np.where(q[:, 0] < 0, -d / 2, d / 2)
    delta = q[:, :2] - np.c_[ctr, np.zeros(len(q))]
    rad = np.linalg.norm(delta, axis=1)
    ii = rad < 0.04702
    q[ii, :2] = np.c_[ctr[ii], np.zeros(ii.sum())] + delta[ii] / rad[ii, None] * 0.04702
    x = np.linspace(-8.25, 8.25, 133)
    y = np.linspace(-6.25, 6.25, 101)
    X, Y = np.meshgrid(x, y)
    xy = np.column_stack([X.ravel(), Y.ravel()])
    dist = lambda xy: np.minimum(
        np.hypot(xy[:, 0] - d / 2, xy[:, 1]), np.hypot(xy[:, 0] + d / 2, xy[:, 1])
    )
    valid = (abs(xy[:, 0]) <= 8) & (abs(xy[:, 1]) <= 6) & (dist(xy) >= R)
    boundary = np.where(~valid, -1, np.where((abs(xy[:, 0]) == 8) | (abs(xy[:, 1]) == 6), 1, 0))
    for j, ctr in enumerate([-d / 2, d / 2]):
        boundary[np.hypot(xy[:, 0] - ctr, xy[:, 1]) < R] = 2 + j
    theta = (np.arange(96) + 0.5) * 2 * np.pi / 96
    circles = []
    walls = []
    for j, ctr in enumerate([-d / 2, d / 2]):
        cs = np.cos(theta)
        sn = np.sin(theta)
        walls.append(
            np.column_stack(
                [
                    ctr + R * cs,
                    R * sn,
                    -cs,
                    -sn,
                    np.full(96, 2 * np.pi * R / 96),
                    np.full(96, j + 1),
                ]
            )
        )
        # Offset 10 micrometres clears roundoff in geometric evaluation next to a curved wall.
        for distance in [0.00001, 0.001, 0.004, 0.012, 0.035, 0.065, 0.094]:
            circles.append(np.column_stack([ctr + (R + distance) * cs, (R + distance) * sn]))
    probes = np.concatenate(circles)
    xx = np.linspace(-8, 8, 801)
    line = np.column_stack([xx, np.zeros_like(xx)])
    line = line[dist(line) >= R]
    coords = np.concatenate([q[:, :2], xy[valid], probes, line])
    np.savez_compressed(
        directory / "geometry.npz",
        quad_xy=q[:, :2],
        volume_weight=q[:, 2],
        near_quad=dist(q[:, :2]) <= R + 0.094,
        cart_x=x,
        cart_y=y,
        mask=valid.reshape(X.shape),
        cart_boundary=boundary.reshape(X.shape),
        near_xy=probes,
        line_xy=line,
        wall=np.concatenate(walls),
        geom_volume=192 - 2 * np.pi * R**2,
        critical=abs(q[:, 1]) >= 5,
        t_s=TIMES,
    )
    np.savetxt(directory / "coordinate_input.csv", coords, delimiter=",", fmt="%.15g")
    return directory, len(coords)
