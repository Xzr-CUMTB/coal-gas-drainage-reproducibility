"""Same portable output payload as proxy, using measured native CSV exports."""

import time
from pathlib import Path
import numpy as np
from common import *
from grid import build


def finalize(raw, s, out, task="F+", geometry="cached"):
    start = time.perf_counter()
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    b = native_csv(raw / "balance.csv")
    t = b["t_s"]
    q = np.c_[b["q1_kg_s"], b["q2_kg_s"]]
    e = integral(t, q)
    rows(
        out / "flow.csv",
        [
            dict(t_s=ti, q1_kg_s=qi[0], q2_kg_s=qi[1], E1_kg=ei[0], E2_kg=ei[1])
            for ti, qi, ei in zip(t, q, e)
        ],
    )
    if task == "Q":
        return time.perf_counter() - start
    folder, _ = build(float(s["d_m"]))
    g = dict(np.load(folder / "geometry.npz"))
    np.savez_compressed(out / "geometry.npz", **g)
    lengths = [len(g["quad_xy"]), int(g["mask"].sum()), len(g["near_xy"]), len(g["line_xy"])]
    fields = {a: [] for a in ("quad", "cart", "near", "line")}
    ds = []
    w = g["volume_weight"]
    M0 = initial(float(g["geom_volume"]))
    wall = native_csv(raw / "wall.csv") if task == "F+" else None
    if task == "F+":
        kt, kq, ke = flow(load_model("KRR"), float(s["d_m"]), float(s["k_mult"]))
    for ti in ENG:
        a = np.loadtxt(raw / f"field_{round(ti)}.csv", delimiter=",", comments="%")
        parts = np.split(a[:, 2:], np.cumsum(lengths)[:-1])
        assert np.isfinite(a).all()
        for name, p in zip(("quad", "cart", "near", "line"), parts):
            sm, sf, ga = derived(p[:, 0], p[:, 1])
            fields[name].append(np.c_[p[:, :2], sm, sf, ga].astype(np.float32))
        p = parts[0][:, :2]
        sm, sf, ga = derived(p[:, 0], p[:, 1])
        mass = float(w @ (sm + sf))
        ei = float(np.interp(ti, t, e.sum(1)))
        r = dict(
            t_s=ti,
            matrix_kg=float(w @ sm),
            fracture_kg=float(w @ sf),
            Gamma_kg_s=float(w @ ga),
            M0_kg=M0,
            E_native_kg=ei,
            mass_fraction=abs(mass + ei - M0) / M0,
            coverage_sampled_m3=float(w[(p[:, 0] <= 300000) & (p[:, 1] <= 300000)].sum()),
            min_pressure_Pa=float(p.min()),
            max_pressure_Pa=float(p.max()),
            finite=True,
        )
        if task == "F+":
            xy, n, ww, face = outer_points(512)
            b = np.loadtxt(raw / f"outer_{round(ti)}.csv", delimiter=",", comments="%")
            assert np.isfinite(b).all()
            pf = b[:, 2]
            gn = (b[:, 3:5] * n).sum(1)
            flux = -C * pf * float(s["k_m2"]) / MU * gn
            fields["outer_" + str(round(ti))] = np.c_[pf, gn, flux]
            r.update(
                outer_net_kg_s=float(ww @ flux),
                outer_abs_kg_s=float(ww @ abs(flux)),
                outer_abs_over_Q=float(ww @ abs(flux))
                / max(float(np.interp(ti, kt, kq.sum(1))), 1e-12),
                wall_pf_max_Pa=float(
                    max(
                        np.interp(ti, wall["t_s"], wall["wall1_Pa"]),
                        np.interp(ti, wall["t_s"], wall["wall2_Pa"]),
                    )
                ),
            )
            for j, name in enumerate(("bottom", "top", "left", "right")):
                ii = face == j
                r[name + "_net_kg_s"] = float(ww[ii] @ flux[ii])
                r[name + "_abs_kg_s"] = float(ww[ii] @ abs(flux[ii]))
        ds.append(r)
    np.savez_compressed(
        out / "fields.npz", t_s=ENG, **{a: np.asarray(b) for a, b in fields.items()}
    )
    rows(out / "diagnostics.csv", ds)
    return time.perf_counter() - start
