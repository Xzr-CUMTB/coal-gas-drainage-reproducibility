from base import *
import torch


def main():
    model = load_model("B1", 2026, "cuda", torch.float64)
    torch.set_num_threads(4)
    rr = []
    regions = []
    for s in scene_list():
        sid = s["scene_id"]
        d = float(s["d_m"])
        k = float(s["k_mult"])
        g = dict(np.load(geometry(d) / "geometry.npz"))
        xy = g["quad_xy"]
        w = g["volume_weight"]
        ii = (abs(xy[:, 0]) <= 0.5) & (abs(xy[:, 1]) <= 1)
        regions.append(
            dict(
                sid=sid,
                n=len(w[ii]),
                weight_sum_m3=w[ii].sum(),
                analytical_volume_m3=2.0,
                min_weight_m3=w[ii].min(),
                max_weight_m3=w[ii].max(),
                min_center_distance_m=np.min(np.hypot(abs(xy[ii, 0]) - d / 2, xy[ii, 1])),
                radius_m=R,
            )
        )
        t, q, e = flow(load_model("KRR"), d, k)
        for ti in ENG:
            dm = df = ga = 0.0
            for j in range(0, len(w), 8192):
                wx = w[j : j + 8192]
                xx = torch.as_tensor(xy[j : j + 8192], dtype=torch.float64, device="cuda")
                tt = torch.full(
                    (len(xx),), float(ti), dtype=torch.float64, device="cuda", requires_grad=True
                )
                pm, pf = model.p(d, k, xx, tt).T
                sm, sf, gg = derived(pm, pf)
                dsm = torch.autograd.grad(sm.sum(), tt, retain_graph=True)[0].detach().cpu().numpy()
                dsf = torch.autograd.grad(sf.sum(), tt)[0].detach().cpu().numpy()
                dm += wx @ dsm
                df += wx @ dsf
                ga += wx @ gg.detach().cpu().numpy()
            total = np.interp(ti, t, q.sum(1))
            rr.append(
                dict(
                    sid=sid,
                    t_s=ti,
                    dMm_dt_kg_s=dm,
                    dMf_dt_kg_s=df,
                    Gamma_kg_s=ga,
                    Q_kg_s=total,
                    Rm_kg_s=dm + ga,
                    Rf_kg_s=df - ga + total,
                    total_abs_Q=abs(dm + df + total) / total,
                )
            )
    rows(REP / "PREDICTED_INSTANTANEOUS.csv", rr)
    rows(REP / "REGION_WEIGHTS.csv", regions)


if __name__ == "__main__":
    main()
