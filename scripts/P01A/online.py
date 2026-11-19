"""Both sides write C4 full online payload and the same engineering endpoints."""

from base import *


def endpoints(folder):
    f = np.load(folder / "fields.npz")
    g = dict(np.load(folder / "geometry.npz"))
    flowtab = read(folder / "flow.csv")
    t = np.array([float(x["t_s"]) for x in flowtab])
    q = np.array([[float(x["q1_kg_s"]), float(x["q2_kg_s"])] for x in flowtab])
    e = integral(t, q)
    a = f["quad"]
    save(
        folder / "engineering.json",
        dict(
            T20=t20(t, e.sum(1)),
            E90=e[-1].tolist(),
            center_mean_Pa=[zone(g, p[:, :2]).tolist() for p in a],
            times_s=ENG.tolist(),
        ),
    )


def proxy(s, out, device="cpu"):
    import torch
    from predict import execute

    torch.set_num_threads(4)
    execute(float(s["d_m"]), float(s["k_mult"]), "F", "B1", 2026, device, out, "fresh", "plus")
    endpoints(out)


def native(s, run, gridfolder):
    import online_native

    online_native.build = lambda d: (gridfolder, 0)
    online_native.finalize(run, s, run / "payload")
    endpoints(run / "payload")


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--d", type=float, required=True)
    p.add_argument("--k", type=float, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    a = p.parse_args()
    if not (3 <= a.d <= 6 and 0.5 <= a.k <= 2):
        p.error("d in [3,6]; k in [0.5,2]")
    if a.out.exists():
        p.error("new output directory required")
    proxy(dict(d_m=a.d, k_mult=a.k), a.out, a.device)
