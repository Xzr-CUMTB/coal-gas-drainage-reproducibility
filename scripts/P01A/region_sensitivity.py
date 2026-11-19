"""Complement the sealed rectangle-centre rule with boundary-node half weights."""

from base import *


def main():
    rr = []
    for s in scene_list():
        sid = s["scene_id"]
        g = dict(np.load(geometry(s["d_m"]) / "geometry.npz"))
        xy = g["quad_xy"]
        ii = (abs(xy[:, 0]) <= 0.5) & (abs(xy[:, 1]) <= 1)
        w = g["volume_weight"][ii].copy()
        edges = np.isclose(abs(xy[ii, 0]), 0.5, atol=1e-12, rtol=0)
        w[edges] *= 0.5
        assert np.isclose(w.sum(), 2.0, atol=1e-8) and min(w) > 0
        with np.load(D / "references" / (sid + "_coarse.npz")) as z:
            ref = z["quad"]
        with np.load(D / "frozen" / sid / "B1_2026.npz") as z:
            pred = z["quad"]
        for j, t in enumerate(ENG):
            a, b = [w @ values[j, ii].astype(float) / w.sum() for values in (ref, pred)]
            ap, bp = [zone(g, q[j]) for q in (ref, pred)]
            for l, c in enumerate(("pm", "pf")):
                rr.append(
                    dict(
                        sid=sid,
                        t_s=t,
                        field=c,
                        exact_area_m3=w.sum(),
                        frozen_area_m3=g["volume_weight"][ii].sum(),
                        ref_half_Pa=a[l],
                        pred_half_Pa=b[l],
                        paired_error_half_Pa=b[l] - a[l],
                        ref_change_Pa=a[l] - ap[l],
                        pred_change_Pa=b[l] - bp[l],
                        difference_of_errors_Pa=(b[l] - a[l]) - (bp[l] - ap[l]),
                    )
                )
    rows(REP / "REGION_WEIGHT_SENSITIVITY.csv", rr)


if __name__ == "__main__":
    main()
