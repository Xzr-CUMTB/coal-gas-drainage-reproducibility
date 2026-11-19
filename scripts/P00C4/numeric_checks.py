"""Portable dense-flow, storage and mesh checks from pressure-only reference arrays."""

import json
import numpy as np
from common import *


def flow_check(ref):
    t = ref["flow_t"]
    q = ref["flow_q"]
    e = integral(t, q)
    mass = ref["flow_mm"] + ref["flow_mf"]
    M0 = initial(float(ref["flow_area"][0]))
    Mfe = mass[0]
    ix = np.array([np.argmin(abs(t - z)) for z in MAIN])
    primary = e[ix].sum(1)
    ratios = []
    for c in (0, 1):
        e2 = integral(t[::2], q[::2, c])
        coarse = np.interp(MAIN, t[::2], e2)
        ratios.extend(abs(coarse - e[ix, c]) / np.maximum(abs(e[ix, c]), 1e-4))
    r = dict(
        uniform=float(np.max(abs(mass + e.sum(1) - M0)) / M0),
        discrete=float(np.max(abs(mass + e.sum(1) - Mfe)) / Mfe),
        main_uniform=float(np.max(abs(mass[ix] + e[ix].sum(1) - M0) / np.maximum(primary, 1e-6))),
        main_discrete=float(np.max(abs(mass[ix] + e[ix].sum(1) - Mfe) / np.maximum(primary, 1e-6))),
        asym=float(np.max(abs(e[ix, 0] - e[ix, 1]) / np.maximum(primary / 2, 1e-15))),
        quadrature=float(max(ratios)),
        min_Q=float(q.min()),
        n_times=len(t),
    )
    r["passed"] = bool(
        max(r["uniform"], r["discrete"]) <= 0.001
        and max(r["main_uniform"], r["main_discrete"]) <= 0.01
        and r["asym"] <= 0.005
        and r["quadrature"] <= 1e-4
        and r["min_Q"] >= 0
    )
    return r


def storage_check(g, ref):
    w = g["volume_weight"]
    rs = []
    for j, t in enumerate(ref["t_s"]):
        p = ref["quad"][j].astype(float)
        sm, sf, ga = derived(p[:, 0], p[:, 1])
        mass = float(w @ (sm + sf))
        i = int(np.argmin(abs(ref["flow_t"] - t)))
        native = float(ref["flow_mm"][i] + ref["flow_mf"][i])
        r = dict(
            t_s=t, independent_kg=mass, native_kg=native, relative_error=abs(mass / native - 1)
        )
        assert np.isfinite(p).all() and r["relative_error"] <= 0.001
        rs.append(r)
    return rs


def mesh_check(g, coarse, fine, scene, first):
    w = g["volume_weight"]
    near = g["near_quad"]
    rs = []
    ec = integral(coarse["flow_t"], coarse["flow_q"]).sum(1)
    ef = integral(fine["flow_t"], fine["flow_q"]).sum(1)
    for j, t in enumerate(fine["t_s"]):
        i = int(np.argmin(abs(coarse["t_s"] - t)))
        dc = coarse["quad"][i].astype(float) - fine["quad"][j].astype(float)
        rms = np.sqrt(w @ (dc**2) / w.sum())
        nmax = max(
            abs(dc[near]).max(),
            abs(coarse["near"][i].astype(float) - fine["near"][j].astype(float)).max(),
        )
        ediff = abs(
            float(np.interp(t, coarse["flow_t"], ec)) / float(np.interp(t, fine["flow_t"], ef)) - 1
        )
        pc = coarse["quad"][i]
        pf = fine["quad"][j]
        cov = lambda a: w[(a[:, 0] <= 300000) & (a[:, 1] <= 300000)].sum() / w.sum()
        cd = abs(cov(pc) - cov(pf))
        rs.append(
            dict(
                scene_id=scene,
                first=first,
                second="fine",
                t_s=t,
                rms_pm_Pa=rms[0],
                rms_pf_Pa=rms[1],
                near_max_Pa=nmax,
                E_relative=ediff,
                sampled_coverage_fraction_difference=cd,
                passed=bool(max(rms) <= 5000 and nmax <= 5000 and ediff <= 0.01 and cd <= 0.01),
            )
        )
    return rs
