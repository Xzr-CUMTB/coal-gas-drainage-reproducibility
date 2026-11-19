"""Score lossless pressure arrays and measured clocks. Requires only NumPy for replay."""

from base import *
import itertools, shutil
from numeric_checks import flow_check, storage_check, mesh_check


def load_arrays(path, keys=None):
    with np.load(path) as z:
        return {k: z[k] for k in (keys or z.files)}


def gridfile(s):
    return D / "grids" / ("grid_D%.9f" % float(s["d_m"])) / "geometry.npz"


def refpath(s, mesh):
    return D / "references" / (s + "_" + mesh + ".npz")


def recompute(dest, all_models=True, negative=False):
    dest.mkdir(exist_ok=True, parents=True)
    results = []
    fields = []
    flows = []
    engine = []
    balance = []
    meshrows = []
    ranks = []
    opt = []
    derived_qc = []
    delta = dict(E90=[], T20=[])
    nativeqc = read(REP / "NATIVE_QC.csv")
    old_qc = {}
    for r in nativeqc:
        old_qc.setdefault((r["sid"], r["mesh"], r["repeat"]), []).append(r)
    for s in scene_list():
        sid = s["scene_id"]
        g = load_arrays(gridfile(s))
        ref = load_arrays(refpath(sid, "coarse"))
        flowf = load_arrays(D / "frozen" / sid / "flow.npz")
        t = flowf["t"]
        q = flowf["q"]
        assert np.allclose(t, ref["flow_t"], rtol=1e-12, atol=1e-9)
        for m in ["coarse", "medium", "fine"] if sid in ("P00", "P11") else ["coarse"]:
            rx = ref if m == "coarse" else load_arrays(refpath(sid, m))
            fr = flow_check(rx)
            sr = storage_check(g, rx)
            qo = old_qc[(sid, m, "1")]
            good = (
                fr["passed"]
                and max(x["relative_error"] for x in sr) <= 0.001
                and all(
                    float(x["min_pressure_Pa"]) > 0
                    and x["finite"] == "True"
                    and float(x["wall_pf_max_Pa"]) <= 1
                    and float(x["outer_abs_over_Q"]) <= 0.01
                    and x["pass_all"] == "True"
                    for x in qo
                )
            )
            derived_qc.append(dict(scene=sid, mesh=m, accepted=good, **fr))
            if m == "fine":
                for level in ("medium", "coarse"):
                    rc = ref if level == "coarse" else load_arrays(refpath(sid, level))
                    meshrows += mesh_check(g, rc, rx, sid, level)
                rc_e = integral(ref["flow_t"], ref["flow_q"]).sum(1)
                rf_e = integral(rx["flow_t"], rx["flow_q"]).sum(1)
                delta["E90"].append(abs(rc_e[-1] - rf_e[-1]))
                delta["T20"].append(abs(t20(t, rc_e)["days"] - t20(t, rf_e)["days"]))
        folder = D / "frozen" / sid
        models = [p.stem for p in sorted(folder.glob("*.csv"))] if all_models else ["B1_2026"]
        for tag in models:
            f = load_arrays(folder / (tag + ".npz"), ["quad", "near"])
            bd = read(folder / (tag + ".csv"))
            su, rr, eq = score_arrays(g, f, ref, t, q, bd)
            results.append(dict(sid=sid, model=tag, **su))
            fields += [dict(sid=sid, model=tag, **r) for r in rr]
            if tag == "B1_2026":
                flows += [dict(sid=sid, **r) for r in eq]
                if negative and sid == "P00":
                    f["quad"] = f["quad"].copy()
                    f["quad"][0, :, 0] += 50000.0
                    bad, _, _ = score_arrays(g, f, ref, t, q, bd)
                    assert bad["rms"] > 5000 and not bad["Fplus"] and bad["Q_pass"]
                    save(
                        dest / "NEGATIVE_CONTROL.json",
                        dict(
                            added_matrix_Pa=50000,
                            original_Fplus=su["Fplus"],
                            perturbed_rms_Pa=bad["rms"],
                            perturbed_Fplus=bad["Fplus"],
                            Q=bad["Q_pass"],
                        ),
                    )
        e = integral(t, q).sum(1)
        er = integral(t, ref["flow_q"]).sum(1)
        assert np.allclose(e, flowf["e"].sum(1))
        a = t20(t, e)
        b = t20(t, er)
        dt = None if a["days"] is None or b["days"] is None else abs(a["days"] - b["days"])
        engine.append(
            dict(
                sid=sid,
                d=float(s["d_m"]),
                k=float(s["k_mult"]),
                E90=e[-1],
                ref_E90=er[-1],
                T20=a["days"],
                ref_T20=b["days"],
                abs_days=dt,
                abs_hours=dt * 24 if dt is not None else None,
                rel=dt / b["days"] if dt is not None else None,
                **{"pred_" + k: v for k, v in a.items()},
                **{"ref_" + k: v for k, v in b.items()}
            )
        )
        # Native dense derivative and ten-snapshot interval mass balance are distinct diagnostics.
        mass = ref["flow_mm"] + ref["flow_mf"]
        derivative = np.gradient(mass, t)
        qsum = ref["flow_q"].sum(1)
        for ti in np.r_[0, ENG]:
            idx = np.argmin(abs(t - ti))
            balance.append(
                dict(
                    sid=sid,
                    t_s=t[idx],
                    actual_Q=qsum[idx],
                    actual_M=mass[idx],
                    M0uniform=initial(g["geom_volume"]),
                    E=er[idx],
                    native_dMdt=derivative[idx],
                    abs_res_Q=abs(derivative[idx] + qsum[idx]) / max(qsum[idx], 1e-12),
                    method="dense nonuniform numerical derivative; first time one-sided",
                )
            )
    valid_mesh = all(r["passed"] for r in meshrows if r["first"] == "coarse") and all(
        r["accepted"] for r in derived_qc if r["mesh"] == "fine"
    )
    for r in results:
        r["reference_qualified"] = valid_mesh and next(
            x["accepted"] for x in derived_qc if x["scene"] == r["sid"] and x["mesh"] == "coarse"
        )
        r["eligible_Fplus"] = bool(r["Fplus"] and r["reference_qualified"])
    choices = read(REP / "PRE_REFERENCE_CHOICES.csv")
    for k, grp in itertools.groupby(engine, key=lambda r: r["k"]):
        group = list(grp)
        for target in ("E90", "T20"):
            band = 2 * max(delta[target])
            best = (min if target == "T20" else max)(group, key=lambda r: r["ref_" + target])
            chosen = next(r for r in choices if float(r["k"]) == k and r["target"] == target)[
                "scene"
            ]
            actual = next(r for r in group if r["sid"] == chosen)
            opt.append(
                dict(
                    k=k,
                    target=target,
                    pred_choice=chosen,
                    ref_best=best["sid"],
                    objective_gap=abs(actual["ref_" + target] - best["ref_" + target]),
                    tie_band=band,
                )
            )
            for a, b in itertools.combinations(group, 2):
                x = b["ref_" + target] - a["ref_" + target]
                y = b[target] - a[target]
                ranks.append(
                    dict(
                        k=k,
                        target=target,
                        a=a["sid"],
                        b=b["sid"],
                        ref_diff=x,
                        pred_diff=y,
                        band=band,
                        resolvable=abs(x) > band,
                        pred_resolvable=abs(y) > band,
                        correct=(np.sign(x) == np.sign(y)) if abs(x) > band else "TIE",
                    )
                )
    for name, rr in [
        ("SCORES", results),
        ("FIELD_SCORES", fields),
        ("FLOW_SCORES", flows),
        ("ENGINEERING", engine),
        ("NATIVE_INSTANTANEOUS", balance),
        ("MESH_REPLAY", meshrows),
        ("NATIVE_FLOW_REPLAY", derived_qc),
        ("PAIRWISE_RANKING", ranks),
        ("CANDIDATE_GAP", opt),
    ]:
        rows(dest / (name + ".csv"), rr)
    return results
