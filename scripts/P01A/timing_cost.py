from base import *
import math, statistics


def calculate(dest):
    native = [r for r in read(REP / "COMSOL_SOLVE_LEDGER.csv") if r["status"] == "DONE"]
    proxy = read(REP / "PROXY_TIMING.csv")
    scores = [r for r in read(dest / "SCORES.csv") if r["model"] == "B1_2026"]
    table = []
    for sid in ("P00", "P11"):
        n = [float(r["online_s"]) for r in native if r["sid"] == sid and r["mesh"] == "coarse"]
        p = [float(r["online_s"]) for r in proxy if r["sid"] == sid]
        assert len(n) == len(p) == 3
        ok = next(r for r in scores if r["sid"] == sid)["eligible_Fplus"] == "True" and all(
            r["ref_qc"] == "True" for r in native if r["sid"] == sid and r["mesh"] == "coarse"
        )
        a, b = np.median(n), np.median(p)
        table.append(
            dict(
                sid=sid,
                n=3,
                native_median_s=a,
                native_min=min(n),
                native_max=max(n),
                proxy_median_s=b,
                proxy_min=min(p),
                proxy_max=max(p),
                raw_ratio=a / b,
                raw_time_reduction_pct=100 * (1 - b / a),
                qualified=ok,
                effective_speed=a / b if ok else "NA",
            )
        )
    rows(dest / "MATCHED_LATENCY.csv", table)
    a = sum(float(r["online_s"]) for r in native if r["rep"] == "1" and r["mesh"] == "coarse")
    b = sum(float(r["online_s"]) for r in proxy if r["rep"] == "1")
    gather = get(REP / "BATCH_AGGREGATION.json")
    choices = set(r["scene"] for r in read(REP / "PRE_REFERENCE_CHOICES.csv"))
    verify = sum(
        float(r["online_s"])
        for r in native
        if r["rep"] == "1" and r["mesh"] == "coarse" and r["sid"] in choices
    )
    rr = [
        dict(stage="P01A", account="batch", item="12 native fresh F+ first calls", seconds=a),
        dict(stage="P01A", account="batch", item="12 proxy fresh F+ first calls", seconds=b),
        dict(
            stage="P01A",
            account="batch",
            item="native candidate endpoint aggregation",
            seconds=gather["native_s"],
        ),
        dict(
            stage="P01A",
            account="batch",
            item="proxy candidate endpoint aggregation",
            seconds=gather["proxy_s"],
        ),
        dict(
            stage="P01A",
            account="conditional screening strategy",
            item="preselected independent numerical checks; reused measured first calls, subset of native batch",
            seconds=verify,
        ),
    ]
    if (REP / "OFFLINE_BASE.csv").exists():
        rr += read(REP / "OFFLINE_BASE.csv")
    off = sum(float(r["seconds"]) for r in rr if r["account"] == "reproducible lower bound")
    scenarios = []
    for m in table:
        for count in (12, 100, 1000):
            for rate in (0, 0.05):
                saved = m["native_median_s"] - m["proxy_median_s"] - rate * m["native_median_s"]
                scenarios.append(
                    dict(
                        sid=m["sid"],
                        request_count=count,
                        assumed_verify_fraction=rate,
                        offline_floor_s=off,
                        COMSOL_extrapolated_s=count * m["native_median_s"],
                        proxy_lifecycle_lower_s=off
                        + count * (m["proxy_median_s"] + rate * m["native_median_s"]),
                        break_even_floor=(
                            math.ceil(off / saved) if m["qualified"] and saved > 0 else "NA"
                        ),
                        eligible=m["qualified"],
                        scenario="extrapolation at case median, excludes unknown cost",
                    )
                )
    rows(dest / "TIMING_AND_COST_LEDGER.csv", rr)
    rows(dest / "COST_SCENARIOS.csv", scenarios)
    save(
        dest / "BATCH_COST.json",
        dict(
            native_12_s=a,
            proxy_12_s=b,
            aggregation=gather,
            verification_subset=sorted(choices),
            verification_s=verify,
            screen_plus_check_s=b + gather["proxy_s"] + verify,
            offline_known_floor_s=off,
            lifecycle_batch_s_lower=off + b + gather["proxy_s"] + verify,
            native_batch_total_s=a + gather["native_s"],
        ),
    )
    return table
