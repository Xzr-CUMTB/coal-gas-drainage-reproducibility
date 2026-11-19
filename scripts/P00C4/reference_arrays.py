"""Portable native-export conversion and independent physical checks."""

import json, time
import numpy as np
from common import *
from numeric_checks import flow_check, storage_check


def convert(run, s, g, folder, quality="plus"):
    start = time.perf_counter()
    folder.mkdir(parents=True, exist_ok=True)
    flow_raw = native_csv(run / "balance.csv")
    t = np.atleast_1d(flow_raw["t_s"])
    q = np.column_stack([flow_raw["q1_kg_s"], flow_raw["q2_kg_s"]])
    assert len(t) == 2041 and np.allclose(t, schedule())
    ref = dict(
        flow_t=t,
        flow_q=q,
        flow_mm=flow_raw["mm_kg"],
        flow_mf=flow_raw["mf_kg"],
        flow_area=flow_raw["area_m2"],
        t_s=ENG,
    )
    fields = {a: [] for a in ("quad", "cart", "near", "line")}
    ds = []
    outer_array = []
    lengths = [len(g["quad_xy"]), int(g["mask"].sum()), len(g["near_xy"]), len(g["line_xy"])]
    w = g["volume_weight"]
    M0 = initial(float(g["geom_volume"]))
    E = integral(t, q).sum(1)
    krr = load_model("KRR")
    kt, kq, ke = flow(krr, float(s["d_m"]), float(s["k_mult"]))
    wall = native_csv(run / "wall.csv") if quality == "plus" else None
    for ti in ENG:
        a = np.loadtxt(run / f"field_{round(ti)}.csv", delimiter=",", comments="%")
        assert len(a) == sum(lengths) and np.isfinite(a).all(), ("invalid fields", str(run), ti)
        parts = np.split(a[:, 2:], np.cumsum(lengths)[:-1])
        for name, x in zip(fields, parts):
            fields[name].append(x)
        p = parts[0][:, :2]
        sm, sf, ga = derived(p[:, 0], p[:, 1])
        assert np.allclose(np.column_stack([sm, sf, ga]), parts[0][:, 2:], rtol=2e-10, atol=1e-12)
        mass = float(w @ (sm + sf))
        r = dict(
            t_s=ti,
            matrix_kg=float(w @ sm),
            fracture_kg=float(w @ sf),
            Gamma_kg_s=float(w @ ga),
            M0_kg=M0,
            E_native_kg=float(np.interp(ti, t, E)),
            mass_fraction=abs(mass + np.interp(ti, t, E) - M0) / M0,
            coverage_sampled_m3=float(w[(p[:, 0] <= 300000) & (p[:, 1] <= 300000)].sum()),
            finite=True,
            min_pressure_Pa=float(p.min()),
            max_pressure_Pa=float(p.max()),
        )
        if quality == "plus":
            xy, n, ww, face = outer_points(512)
            b = np.loadtxt(run / f"outer_{round(ti)}.csv", delimiter=",", comments="%")
            assert np.isfinite(b).all() and len(b) == len(xy)
            pf = b[:, 2]
            gn = (b[:, 3:5] * n).sum(1)
            flux = -C * pf * float(s["k_m2"]) / MU * gn
            outer_array.append(np.column_stack([pf, gn, flux]))
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
    ref.update({a: np.asarray(v)[:, :, :2].astype(np.float32) for a, v in fields.items()})
    np.savez_compressed(folder / "reference.npz", **ref)
    np.savez_compressed(
        folder / "fields.npz",
        t_s=ENG,
        **{a: np.asarray(v, dtype=np.float32) for a, v in fields.items()},
        outer=np.asarray(outer_array),
    )
    np.savez_compressed(folder / "geometry.npz", **g)
    rows(folder / "diagnostics.csv", ds)
    rows(
        folder / "flow.csv",
        [
            dict(t_s=ti, q1_kg_s=qi[0], q2_kg_s=qi[1], E1_kg=ei[0], E2_kg=ei[1])
            for ti, qi, ei in zip(t, q, integral(t, q))
        ],
    )
    r = flow_check(ref)
    storage = storage_check(g, ref)
    r["storage_max_relative"] = max(x["relative_error"] for x in storage)
    r["outer_max_ratio"] = max((x.get("outer_abs_over_Q", 0) for x in ds))
    r["wall_max_Pa"] = max((x.get("wall_pf_max_Pa", 0) for x in ds))
    r["Fplus_numerical_pass"] = r["outer_max_ratio"] <= 0.01 and r["wall_max_Pa"] <= 1
    rows(folder / "storage_check.csv", storage)
    save(folder / "QC.json", r)
    return dict(post_python_s=time.perf_counter() - start, QC=r)
