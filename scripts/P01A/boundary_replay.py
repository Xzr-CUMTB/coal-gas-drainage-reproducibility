"""Direct audit of wall pressure and absolute outer normal flux; uses NumPy only."""

from base import *


def verify(dest):
    results = []
    for s in scene_list():
        sid = s["scene_id"]
        folder = D / "frozen" / sid
        with np.load(D / "boundaries" / (sid + ".npz")) as z:
            arr = {k: z[k] for k in z.files}
        with np.load(folder / "B1_2026.npz") as z:
            wall = abs(z["wall"][:, :, 1].astype(float) - 1e5).max(1)
        orig = read(folder / "B1_2026.csv")
        with np.load(folder / "flow.npz") as z:
            qr = np.interp(ENG, z["t"], z["q"].sum(1))
        for name, key in [("native", "native_pf_grad"), ("B1", "proxy_pf_grad")]:
            p = arr[key][:, :, 0]
            dn = arr[key][:, :, 1]
            flux = -C * float(s["k_m2"]) / MU * p * dn
            ratio = abs(flux) @ arr["arc_weights_m"] / qr
            wp = wall if name == "B1" else arr["native_wall_max_Pa"].max(1)
            for i, ti in enumerate(ENG):
                assert wp[i] <= 1 and ratio[i] <= 0.01
                if name == "B1":
                    assert (
                        np.isclose(ratio[i], float(orig[i]["outer_abs_over_Q"]), atol=1e-12)
                        and abs(wp[i] - float(orig[i]["wall_pf_max_Pa"])) <= 0.125
                    )
                results.append(
                    dict(
                        sid=sid,
                        t_s=ti,
                        method=name,
                        wall_Pa=wp[i],
                        absolute_fluxe_ratio=ratio[i],
                        net_kg_s=flux[i] @ arr["arc_weights_m"],
                    )
                )
    rows(dest / "BOUNDARY_ARRAY_REPLAY.csv", results)
    return len(results)
