from pathlib import Path
import sys, os, json, time

ROOT = Path(__file__).absolute().parents[2]
C4 = ROOT / "scripts/P00C4"
sys.path.insert(0, str(C4))
from common import (
    np,
    sha,
    read,
    rows,
    save,
    flow,
    pressure,
    load_model,
    derived,
    initial,
    integral,
    ENG,
    MAIN,
    boundary,
    outer_points,
    schedule,
    weighted95,
    C,
    MU,
    R,
    native_csv,
)

D = ROOT / "data/P01A"
REP = ROOT / "reports/P01A"
NATIVE = ROOT / "05_models/P01A"
SRC = ROOT / "scripts/P01A"


def get(p):
    return json.loads(Path(p).read_text(encoding="utf-8-sig"))


def rel(p):
    return Path(p).relative_to(ROOT).as_posix()


def member(p):
    return dict(path=rel(p), bytes=p.stat().st_size, sha256=sha(p))


def lock(p, x):
    if p.exists():
        raise FileExistsError(p)
    save(p, json.loads(json.dumps(x, default=lambda a: a.tolist())))


def scene_list():
    return read(D / "design.csv")


def geometry(d):
    from grid import build

    p, _ = build(float(d), D / "grids")
    return p


def t20(t, e):
    hit = np.flatnonzero(e >= 20)
    if not len(hit):
        return dict(status="NOT_REACHED", days=None)
    j = int(hit[0])
    assert j > 0 and e[j] > e[j - 1]
    return dict(
        status="REACHED",
        days=(t[j - 1] + (20 - e[j - 1]) * (t[j] - t[j - 1]) / (e[j] - e[j - 1])) / 86400,
        lo_s=t[j - 1],
        hi_s=t[j],
        lo_kg=e[j - 1],
        hi_kg=e[j],
        negative_increments=int(np.sum(np.diff(e) < 0)),
    )


def zone(g, p):
    xy = g["quad_xy"]
    ii = (abs(xy[:, 0]) <= 0.5) & (abs(xy[:, 1]) <= 1)
    w = g["volume_weight"][ii]
    return w @ p[ii].astype(float) / w.sum()


def score_arrays(g, f, ref, t, q, bounds):
    from scoring import qscore

    summary, qr, e, er = qscore(t, q, ref["flow_q"])
    idx = t >= 3600
    for j in range(2):
        summary[f"E{j+1}_max"] = max(x[f"E{j+1}_error_fraction"] for x in qr)
        summary[f"QL1_{j+1}"] = np.trapz(abs(q[idx, j] - ref["flow_q"][idx, j]), t[idx]) / np.trapz(
            ref["flow_q"][idx, j], t[idx]
        )
    rr = []
    w = g["volume_weight"]
    near = g["near_quad"]
    m0 = initial(float(g["geom_volume"]))
    for i, ti in enumerate(ENG):
        p = f["quad"][i, :, :2].astype(float)
        d = p - ref["quad"][i]
        dn = abs(f["near"][i, :, :2].astype(float) - ref["near"][i])
        rz = zone(g, ref["quad"][i])
        pz = zone(g, p)
        sm, sf, ga = derived(p[:, 0], p[:, 1])
        et = np.interp(ti, t, e.sum(1))
        r = dict(
            t_s=ti,
            wall=float(bounds[i]["wall_pf_max_Pa"]),
            outer=float(bounds[i]["outer_abs_over_Q"]),
            M0=m0,
            Mm=w @ sm,
            Mf=w @ sf,
            Gamma=w @ ga,
            mass=abs(w @ (sm + sf) + et - m0) / m0,
            coverage_m3=w[(p[:, 0] <= 3e5) & (p[:, 1] <= 3e5)].sum(),
            ref_coverage_m3=w[(ref["quad"][i] <= 3e5).all(1)].sum(),
        )
        for j, tag in enumerate(("pm", "pf")):
            r["rms_" + tag] = np.sqrt(w @ (d[:, j] ** 2) / w.sum())
            r["ring_" + tag] = np.quantile(dn[:, j], 0.95)
            r["near_" + tag] = weighted95(abs(d[near, j]), w[near])
            r["zone_" + tag] = pz[j]
            r["ref_zone_" + tag] = rz[j]
        rr.append(r)
    for key, cols in dict(
        rms=["rms_pm", "rms_pf"],
        near=["near_pm", "near_pf", "ring_pm", "ring_pf"],
        mass=["mass"],
        wall=["wall"],
        outer=["outer"],
    ).items():
        summary[key] = max(r[c] for r in rr for c in cols)
    summary["F_pass"] = bool(
        summary["Q_pass"]
        and summary["rms"] <= 5000
        and summary["near"] <= 10000
        and summary["mass"] <= 0.01
        and np.isfinite(f["quad"]).all()
        and np.isfinite(f["near"]).all()
    )
    summary["Fplus"] = bool(summary["F_pass"] and summary["wall"] <= 1 and summary["outer"] <= 0.01)
    return summary, rr, qr
