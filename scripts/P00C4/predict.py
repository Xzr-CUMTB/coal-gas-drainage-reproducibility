"""Portable complete Q/F payload; native process code is deliberately separate."""

import argparse, json, tempfile, time, uuid
from pathlib import Path
import numpy as np
from common import *
from grid import build


def sync(model):
    import torch

    if not isinstance(model, dict) and next(model.parameters()).is_cuda:
        torch.cuda.synchronize()


def execute(
    d, k, task, kind, seed, device, out, geometry_mode="cached", quality="plus", models=None
):
    import torch

    started = time.perf_counter()
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    marks = {}
    t0 = time.perf_counter()
    if models is None:
        models = (load_model("KRR"), None if task == "Q" else load_model(kind, seed, device))
    krr, model = models
    sync(model if model is not None else krr)
    marks["load_s"] = time.perf_counter() - t0
    geom = None
    tmp = None
    t0 = time.perf_counter()
    if task == "F":
        if geometry_mode == "fresh":
            tmp = tempfile.TemporaryDirectory(prefix="c4geometry_")
            folder, _ = build(d, Path(tmp.name))
        else:
            folder, _ = build(d)
        with np.load(folder / "geometry.npz") as z:
            geom = {a: z[a] for a in z.files}
        # Carry the actual shared query geometry in every full-field output.
        np.savez_compressed(out / "geometry.npz", **geom)
    marks["input_s"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    t, q, e = flow(krr, d, k)
    sync(krr)
    marks["Q_kernel_s"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    rows(
        out / "flow.csv",
        [
            dict(t_s=ti, q1_kg_s=qi[0], q2_kg_s=qi[1], E1_kg=ei[0], E2_kg=ei[1])
            for ti, qi, ei in zip(t, q, e)
        ],
    )
    marks["Q_write_s"] = time.perf_counter() - t0
    fields = {}
    diagnostics = []
    kernel = 0.0
    derive_time = 0.0
    physics_time = 0.0
    write_time = 0.0
    if task == "F":
        w = geom["volume_weight"]
        M0 = initial(float(geom["geom_volume"]))
        X, Y = np.meshgrid(geom["cart_x"], geom["cart_y"])
        mask = geom["mask"]
        sets = dict(
            quad=geom["quad_xy"],
            cart=np.c_[X[mask], Y[mask]],
            near=geom["near_xy"],
            line=geom["line_xy"],
        )
        for ti in ENG:
            points = {}
            for name, xy in sets.items():
                t0 = time.perf_counter()
                p = pressure(model, d, k, xy, float(ti))
                sync(model)
                kernel += time.perf_counter() - t0
                t0 = time.perf_counter()
                sm, sf, ga = derived(p[:, 0].astype(float), p[:, 1].astype(float))
                points[name] = np.column_stack([p, sm, sf, ga]).astype(np.float32)
                derive_time += time.perf_counter() - t0
            t0 = time.perf_counter()
            p = points["quad"][:, :2].astype(float)
            sm, sf, ga = derived(p[:, 0], p[:, 1])
            mass = float(w @ (sm + sf))
            E = float(np.interp(ti, t, e.sum(1)))
            wp = pressure(model, d, k, geom["wall"][:, :2], float(ti))
            sync(model)
            r = dict(
                t_s=ti,
                matrix_kg=float(w @ sm),
                fracture_kg=float(w @ sf),
                Gamma_kg_s=float(w @ ga),
                M0_kg=M0,
                E_KRR_kg=E,
                mass_fraction=abs(mass + E - M0) / M0,
                coverage_sampled_m3=float(w[(p[:, 0] <= 300000) & (p[:, 1] <= 300000)].sum()),
                wall_pf_max_Pa=float(abs(wp[:, 1] - 100000).max()),
                min_pressure_Pa=float(p.min()),
                max_pressure_Pa=float(p.max()),
                finite=bool(all(np.isfinite(x).all() for x in points.values())),
            )
            if quality == "plus":
                br, bp = boundary(
                    model, d, k, float(ti), 512, float(np.interp(ti, t, q.sum(1))), True
                )
                r.update(br)
                fields["outer_" + str(round(ti))] = np.column_stack(
                    [bp["pf"], bp["gradient"], bp["flux"]]
                )
            physics_time += time.perf_counter() - t0
            diagnostics.append(r)
            for name, p in points.items():
                fields.setdefault(name, []).append(p)
        t0 = time.perf_counter()
        np.savez_compressed(
            out / "fields.npz", t_s=ENG, **{a: np.asarray(b) for a, b in fields.items()}
        )
        rows(out / "diagnostics.csv", diagnostics)
        write_time += time.perf_counter() - t0
    marks.update(
        field_kernel_s=kernel,
        derive_integral_s=derive_time,
        physics_and_integral_s=physics_time,
        field_write_s=write_time,
    )
    if tmp is not None:
        tmp.cleanup()
    qualification = {"scope": "homogeneous symmetric dual-well benchmark"}
    result = dict(
        d_m=d,
        k_mult=k,
        task=task,
        quality=quality,
        model="KRR" if task == "Q" else kind,
        seed="NA" if task == "Q" else seed,
        device=device,
        geometry_mode=geometry_mode,
        components=marks,
        qualification=qualification,
        online_s=time.perf_counter() - started,
    )
    save(out / "metadata.json", result)
    result["online_s"] = time.perf_counter() - started
    # End-to-end caller clock also includes this final metadata write and process startup.
    return result
