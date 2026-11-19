"""Metrics rebuilt from reference pressure channels and frozen network outputs."""

import numpy as np
from common import ENG, MAIN, integral, flow, pressure, derived, initial, weighted95, boundary


def qscore(t, q, qr):
    e = integral(t, q)
    er = integral(t, qr)
    ii = t >= 3600
    dt = np.diff(t[ii])
    trap = lambda v: np.sum((v[1:] + v[:-1]) * 0.5 * dt)
    l1 = trap(abs(q[ii] - qr[ii]).sum(1)) / trap(qr[ii].sum(1))
    rs = []
    for ti in MAIN:
        j = int(np.argmin(abs(t - ti)))
        r = dict(
            t_s=ti,
            E_pred_kg=float(e[j].sum()),
            E_ref_kg=float(er[j].sum()),
            E_error_fraction=float(abs(e[j].sum() - er[j].sum()) / max(er[j].sum(), 0.01)),
        )
        for c in (0, 1):
            r.update(
                {
                    f"E{c+1}_pred_kg": e[j, c],
                    f"E{c+1}_ref_kg": er[j, c],
                    f"E{c+1}_error_fraction": abs(e[j, c] - er[j, c]) / max(er[j, c], 0.01),
                }
            )
        rs.append(r)
    return (
        dict(
            Q_pass=bool(l1 <= 0.03 and max(r["E_error_fraction"] for r in rs) <= 0.01),
            Q_L1_fraction=float(l1),
            E_max_fraction=max(r["E_error_fraction"] for r in rs),
            Q_negative_count=int((q < 0).sum()),
            E_nonmonotone_count=int((np.diff(e, axis=0) < 0).sum()),
            Q_asymmetry_max_fraction=float(
                np.max(abs(q[:, 0] - q[:, 1]) / np.maximum(q.sum(1) / 2, 1e-15))
            ),
        ),
        rs,
        e,
        er,
    )


def evaluate(model, d, k, g, ref, q, kind, seed, scene, outer=True):
    t = ref["flow_t"]
    qr = ref["flow_q"]
    summary, qs, e, er = qscore(t, q, qr)
    if kind == "KRR":
        return (
            dict(
                scene_id=scene,
                kind=kind,
                seed=seed,
                **summary,
                F_pass="NOT_APPLICABLE",
                F_plus_pass="NOT_APPLICABLE",
            ),
            qs,
            [],
        )
    w = g["volume_weight"]
    n = g["near_quad"]
    M0 = initial(float(g["geom_volume"]))
    fs = []
    for j, ti in enumerate(ENG):
        p = pressure(model, d, k, g["quad_xy"], float(ti)).astype(float)
        pn = pressure(model, d, k, g["near_xy"], float(ti)).astype(float)
        pw = pressure(model, d, k, g["wall"][:, :2], float(ti)).astype(float)
        delta = p - ref["quad"][j].astype(float)
        rms = np.sqrt(w @ (delta**2) / w.sum())
        pn95 = np.quantile(abs(pn - ref["near"][j].astype(float)), 0.95, axis=0)
        sm, sf, ga = derived(p[:, 0], p[:, 1])
        mass = float(w @ (sm + sf))
        E = float(np.interp(ti, t, e.sum(1)))
        Eref = float(np.interp(ti, t, er.sum(1)))
        res = mass + E - M0
        r = dict(
            t_s=ti,
            rms_pm_Pa=rms[0],
            rms_pf_Pa=rms[1],
            probe_near95_pm_Pa=pn95[0],
            probe_near95_pf_Pa=pn95[1],
            volume_near95_pm_Pa=weighted95(abs(delta[n, 0]), w[n]),
            volume_near95_pf_Pa=weighted95(abs(delta[n, 1]), w[n]),
            near_max_Pa=float(max(abs(delta[n]).max(), abs(pn - ref["near"][j]).max())),
            matrix_kg=float(w @ sm),
            fracture_kg=float(w @ sf),
            Gamma_total_kg_s=float(w @ ga),
            M0_kg=M0,
            E_pred_kg=E,
            E_ref_kg=Eref,
            mass_residual_kg=res,
            mass_fraction=abs(res) / M0,
            mass_fraction_Eref=abs(res) / max(Eref, 0.01),
            wall_pf_max_Pa=float(abs(pw[:, 1] - 100000).max()),
            min_pressure_Pa=float(p.min()),
            max_pressure_Pa=float(p.max()),
            finite=bool(np.isfinite(p).all()),
            coverage_sampled_m3=float(w[(p[:, 0] <= 300000) & (p[:, 1] <= 300000)].sum()),
        )
        if outer:
            r.update(boundary(model, d, k, ti, 512, float(np.interp(ti, t, q.sum(1)))))
        fs.append(r)
    f = summary["Q_pass"] and all(
        max(r["rms_pm_Pa"], r["rms_pf_Pa"]) <= 5000
        and max(
            r["probe_near95_pm_Pa"],
            r["probe_near95_pf_Pa"],
            r["volume_near95_pm_Pa"],
            r["volume_near95_pf_Pa"],
        )
        <= 10000
        and r["mass_fraction"] <= 0.01
        and r["finite"]
        for r in fs
    )
    fp = f and outer and all(r["wall_pf_max_Pa"] <= 1 and r["outer_abs_over_Q"] <= 0.01 for r in fs)
    summary.update(
        scene_id=scene,
        kind=kind,
        seed=seed,
        F_pass=bool(f),
        F_plus_pass=bool(fp) if outer else "NOT_TESTED",
        RMS_max_Pa=max(max(r["rms_pm_Pa"], r["rms_pf_Pa"]) for r in fs),
        probe_near95_max_Pa=max(max(r["probe_near95_pm_Pa"], r["probe_near95_pf_Pa"]) for r in fs),
        volume_near95_max_Pa=max(
            max(r["volume_near95_pm_Pa"], r["volume_near95_pf_Pa"]) for r in fs
        ),
        mass_max_fraction=max(r["mass_fraction"] for r in fs),
        wall_max_Pa=max(r["wall_pf_max_Pa"] for r in fs),
        outer_ratio_max=max((r.get("outer_abs_over_Q", 0) for r in fs)) if outer else "NA",
    )
    return summary, qs, fs
