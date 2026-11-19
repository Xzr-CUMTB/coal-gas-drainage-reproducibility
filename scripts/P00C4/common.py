"""Portable arrays, frozen model loading and physical formulas; no native process tools."""

import csv, hashlib, json, time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data/P00C4"
REPORT = ROOT / "reports/P00C4"
CHECK = ROOT / "checkpoints/P00C4"
R = 0.047
C = 0.016 / (8.314462618 * 300)
RSTD = 0.016 * 101000 / (8.314462618 * 273.5)
GF = 0.98 * C / 153360
MU = 1.03e-5
ENG = np.array([3600, 21600, 86400, 259200, 604800, 864000, 1728000, 2592000, 5184000, 7776000.0])
MAIN = np.array([3600, 86400, 864000, 2592000, 7776000.0])


def sha(p):
    h = hashlib.sha256()
    with Path(p).open("rb") as f:
        for b in iter(lambda: f.read(2**20), b""):
            h.update(b)
    return h.hexdigest()


def save(p, v):
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(
        json.dumps(
            v,
            ensure_ascii=False,
            indent=2,
            default=lambda x: x.item() if hasattr(x, "item") else x.tolist(),
        ),
        encoding="utf8",
    )
    tmp.replace(p)


def read(p):
    with Path(p).open(encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def rows(p, rs):
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    rs = list(rs)
    with p.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, list(dict.fromkeys(k for r in rs for k in r)))
        w.writeheader()
        w.writerows(rs)


def native_csv(p):
    text = Path(p).read_text(encoding="utf-8-sig")
    header = next(
        x
        for x in text.splitlines()
        if x.startswith("% ")
        and "," in x
        and ("t_s" in x or "case_index" in x or "order," in x or "triangles," in x)
    )
    return np.genfromtxt(
        p, delimiter=",", comments="%", names=[s.strip() for s in header[2:].split(",")]
    )


def schedule():
    b = np.unique(np.r_[0, np.geomspace(1e-5, 7776000, 245), [0, 0.01], ENG])
    return np.r_[(b[:-1, None] + np.diff(b)[:, None] * np.arange(8) / 8).ravel(), b[-1]]


def integral(t, q):
    q = np.asarray(q)
    dd = np.diff(t)
    return (
        np.r_[0, np.cumsum((q[1:] + q[:-1]) * 0.5 * dd)]
        if q.ndim == 1
        else np.vstack(
            [np.zeros((1, q.shape[1])), np.cumsum((q[1:] + q[:-1]) * 0.5 * dd[:, None], axis=0)]
        )
    )


def derived(pm, pf):
    return (
        0.07 * C * pm + 1228.5 * RSTD * 0.016 * pm / (2450000 + pm),
        0.02 * C * pf,
        GF * (pm - pf),
    )


def initial(volume):
    return volume * sum(derived(600000.0, 600000.0)[:2])


def load_model(kind, seed=2026, device="cpu", dtype=None):
    if kind == "KRR":
        with np.load(CHECK / "baselines/krr.npz") as z:
            return {k: z[k] for k in z.files}
    import torch

    if kind == "A1":
        from legacy_network import FieldModel

        model = FieldModel(local=False)
        path = CHECK / f"baselines/{kind}_{seed}.pt"
    else:
        from network import BoundaryModel

        model = BoundaryModel()
        path = CHECK / (f"baselines/A1_{seed}.pt" if kind == "B0" else f"B1_{seed}.pt")
    state = torch.load(path, map_location="cpu", weights_only=True)
    model.load_state_dict(state["state_dict"])
    return model.to(device=device, dtype=dtype or torch.float32).eval()


def flow(model, d, k, krr=None):
    t = schedule()
    if isinstance(model, dict):
        z = np.array([(d - 4.5) / 1.5, np.log(k) / np.log(2)])
        ker = np.exp(-float(model["gamma"]) * ((model["x"] - z) ** 2).sum(1))
        q = 1e-5 * np.exp((ker @ model["coef"] + model["mean"]).reshape(-1, 2))
    else:
        import torch

        with torch.no_grad():
            q = (
                model.q(
                    d,
                    k,
                    torch.as_tensor(
                        t,
                        device=next(model.parameters()).device,
                        dtype=next(model.parameters()).dtype,
                    ),
                )
                .cpu()
                .numpy()
            )
    return t, q, integral(t, q)


def pressure(model, d, k, xy, t):
    import torch

    par = next(model.parameters())
    with torch.no_grad():
        return np.concatenate(
            [
                model.p(
                    d, k, torch.as_tensor(xy[i : i + 32768], dtype=par.dtype, device=par.device), t
                )
                .cpu()
                .numpy()
                for i in range(0, len(xy), 32768)
            ]
        )


def weighted95(x, w):
    o = np.argsort(x)
    return float(x[o[min(len(o) - 1, np.searchsorted(np.cumsum(w[o]), 0.95 * w.sum()))]])


def outer_points(level=512):
    nx = level
    ny = 3 * level // 4
    x = -8 + (np.arange(nx) + 0.5) * 16 / nx
    y = -6 + (np.arange(ny) + 0.5) * 12 / ny
    xy = np.concatenate(
        [
            np.c_[x, np.full(nx, -6)],
            np.c_[x, np.full(nx, 6)],
            np.c_[np.full(ny, -8), y],
            np.c_[np.full(ny, 8), y],
        ]
    )
    normal = np.concatenate(
        [
            np.tile([0, -1], (nx, 1)),
            np.tile([0, 1], (nx, 1)),
            np.tile([-1, 0], (ny, 1)),
            np.tile([1, 0], (ny, 1)),
        ]
    )
    w = np.r_[np.full(2 * nx, 16 / nx), np.full(2 * ny, 12 / ny)]
    face = np.r_[np.zeros(nx, int), np.ones(nx, int), np.full(ny, 2, int), np.full(ny, 3, int)]
    return xy, normal, w, face


def boundary(model, d, k, t, level=512, qtotal=None, points=False):
    import torch

    xy, n, w, face = outer_points(level)
    par = next(model.parameters())
    xx = torch.tensor(xy, dtype=par.dtype, device=par.device, requires_grad=True)
    with torch.enable_grad():
        pf = model.p(d, k, xx, float(t))[:, 1]
        grad = torch.autograd.grad(pf.sum(), xx)[0].detach().cpu().numpy().astype(float)
    p = pf.detach().cpu().numpy().astype(float)
    gn = (grad * n).sum(1)
    flux = -C * p * (k * 2.6e-17) / MU * gn
    r = dict(
        t_s=t,
        level=level,
        outer_net_kg_s=float(w @ flux),
        outer_abs_kg_s=float(w @ abs(flux)),
        normal_gradient_max_Pa_m=float(abs(gn).max()),
    )
    if qtotal is not None:
        r["outer_abs_over_Q"] = r["outer_abs_kg_s"] / max(qtotal, 1e-12)
    for j, name in enumerate(("bottom", "top", "left", "right")):
        a = face == j
        r[name + "_net_kg_s"] = float(w[a] @ flux[a])
        r[name + "_abs_kg_s"] = float(w[a] @ abs(flux[a]))
    if points:
        return r, dict(xy=xy, normal=n, w=w, face=face, gradient=gn, flux=flux, pf=p)
    return r
