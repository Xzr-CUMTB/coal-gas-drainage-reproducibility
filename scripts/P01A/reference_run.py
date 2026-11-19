"""Single reusable COMSOL class, frozen physics, one budget reservation per cold process."""

from base import *
import argparse, re, shutil, subprocess
from grid import build
from production import parameters, extra, SIZE
from online import native

BIN = None
BUILD = NATIVE / "compiled"


def make_source():
    src = (C4 / "Layout.java.in").read_text()
    init = (
        'm=ModelUtil.create("Layout");m.param().loadFile("'
        + (BUILD / "active.txt").as_posix()
        + '");int ordinal=(int)Math.round(m.param().evaluate("pjob"));out="'
        + NATIVE.as_posix()
        + '/job_"+String.format(java.util.Locale.ROOT,"%02d",ordinal)+"/"; h=m.param().evaluate("ph");hw=m.param().evaluate("phw");hi=m.param().evaluate("phi");'
    )
    src = src.replace('m=ModelUtil.create("Layout");', init)
    ext = (
        extra(Path("P01A_QUERY"), Path("P01A_OUTER"), True)
        .replace('"P01A_QUERY"', 'out+"query.csv"')
        .replace('"P01A_OUTER"', 'out+"outer.csv"')
    )
    # Only archive time (after all online exports) gets removed from the paired clock.
    ext = ext.replace(
        "java.util.Set<Integer> keep",
        'if(m.param().evaluate("parch")>0.5){ java.util.Set<Integer> keep',
    ).replace('m.save(out+"review.mph");', 'm.save(out+"review.mph");}')
    vals = dict(
        OUT=".",
        CASE="homogeneous dual well",
        BULK=1,
        WELL=1,
        STRIP=1,
        CAP=86400,
        TOL="3e-5",
        PRE="false",
        PARAMS=parameters(4, 2.6e-17)
        .replace('"4[m]"', '"pd[m]"')
        .replace('"2.6e-17[m^2]"', '"pk[m^2]"'),
        TIMELIST=" ".join(f"{t:.16g}" for t in schedule()),
        EXTRA=ext,
    )
    for k, v in vals.items():
        src = src.replace("__" + k + "__", str(v))
    assert '"d","pd[m]"' in src and '"perm","pk[m^2]"' in src and src.count(".runAll()") == 1
    return src.replace("P00C4 homogeneous", "P01A homogeneous")


def compile_once():
    BUILD.mkdir(exist_ok=True)
    p = BUILD / "Job.java"
    src = make_source()
    if (BUILD / "compile.json").exists():
        old = get(BUILD / "compile.json")
        assert (
            p.read_text(encoding="utf8") == src and sha(p.with_suffix(".class")) == old["class_sha"]
        )
        return
    p.write_text(src, encoding="utf8")
    start = time.perf_counter()
    with (BUILD / "compile.log").open("wb") as f:
        subprocess.run(
            [str(BIN / "comsolcompile.exe"), str(p)],
            cwd=BUILD,
            stdout=f,
            stderr=subprocess.STDOUT,
            check=True,
        )
    save(
        BUILD / "compile.json",
        dict(
            wall_s=time.perf_counter() - start,
            source_sha=sha(p),
            class_sha=sha(p.with_suffix(".class")),
        ),
    )


def marks(log):
    pairs = dict((k, int(v)) for k, v in re.findall(rb"C4_TIME (\w+) (\d+)", log))
    return {k.decode(): v / 1e9 for k, v in pairs.items()}


def worker(run):
    rec = get(run / "run.json")
    s = rec["scene"]
    d = float(s["d_m"])
    begin = time.perf_counter()
    gp, _ = build(d, run / "query_build")
    shutil.copyfile(gp / "coordinate_input.csv", run / "query.csv")
    np.savetxt(run / "outer.csv", outer_points()[0], delimiter=",", fmt="%.15g")
    query_s = time.perf_counter() - begin
    tick = time.perf_counter()
    with (run / "console.log").open("wb") as f:
        proc = subprocess.Popen(
            [
                str(BIN / "comsolbatch.exe"),
                "-np",
                "4",
                "-inputfile",
                str(BUILD / "Job.class"),
                "-batchlog",
                str(run / "batch.log"),
            ],
            cwd=run,
            stdout=f,
            stderr=subprocess.STDOUT,
        )
        save(run / "native_pid.json", dict(pid=proc.pid, started=time.time()))
        rc = proc.wait(timeout=1200)
    batch_s = time.perf_counter() - tick
    log = (run / "console.log").read_bytes()
    assert rc == 0 and b"C4_DONE" in log, "native unsuccessful; check console"
    m = marks(log)
    tick = time.perf_counter()
    native(s, run, gp)
    payload_s = time.perf_counter() - tick
    save(
        run / "components.json",
        dict(
            query_s=query_s,
            batch_s=batch_s,
            archive_s=m["complete"] - m["Fplus_exported"],
            startup_exit_s=batch_s - (m["complete"] - m["entered"]),
            geometry_physics_s=m["physics"] - m["entered"],
            mesh_s=m["mesh"] - m["physics"],
            setup_s=m["solve_start"] - m["mesh"],
            solve_s=m["solved"] - m["solve_start"],
            native_exports_s=m["Fplus_exported"] - m["solved"],
            payload_s=payload_s,
        ),
    )


def qc(s, run):
    from reference_arrays import convert

    g = dict(np.load(run / "payload/geometry.npz"))
    target = run / "audit"
    meta = convert(run, s, g, target)
    ref = dict(np.load(target / "reference.npz"))
    q = meta["QC"]
    assert np.all(np.diff(ref["flow_t"]) > 0) and np.max(abs(ref["flow_t"] - schedule())) < 1e-6
    assert all(
        np.isfinite(ref[k]).all() and np.min(ref[k]) > 0 for k in ["quad", "near", "cart", "line"]
    )
    assert np.all(g["volume_weight"] > 0)
    # Actual coordinate order, not just the count.
    expected = np.loadtxt(run / "query.csv", delimiter=",")
    for ti in ENG:
        data = np.loadtxt(
            run / f"field_{round(ti)}.csv", delimiter=",", comments="%", usecols=(0, 1)
        )
        assert np.max(abs(data - expected)) < 1e-7
    save(
        run / "qc_receipt.json",
        dict(
            **meta,
            positive_P=True,
            coordinate_order_verified=True,
            pass_all=bool(q["passed"] and q["Fplus_numerical_pass"]),
        ),
    )


def run_reference_cases(cases, jobs):
    for ordinal, (case_id, mesh, repeat) in enumerate(jobs, start=1):
        case = cases[case_id]
        folder = NATIVE / f"job_{ordinal:02d}"
        folder.mkdir()
        bulk, well, strip = SIZE[mesh]
        parameters = {
            "pjob": ordinal,
            "ph": bulk,
            "phw": well,
            "phi": strip,
            "pd": case["d_m"],
            "pk": case["k_m2"],
            "parch": int(repeat == 1),
        }
        config = "".join(f"{key} {value}\n" for key, value in parameters.items())
        (folder / "parameters.txt").write_text(config, encoding="ascii")
        (BUILD / "active.txt").write_text(config, encoding="ascii")
        record = dict(scene=case, sid=case_id, mesh=mesh, rep=repeat, status="RUNNING")
        save(folder / "run.json", record)
        command = [
            sys.executable,
            str(Path(__file__)),
            "--comsol-bin",
            str(BIN),
            "--out",
            str(NATIVE),
            "--worker",
            str(folder),
        ]
        start = time.perf_counter()
        try:
            with (folder / "worker.log").open("wb") as log:
                subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
            record["wall_s"] = time.perf_counter() - start
            record.update(get(folder / "components.json"))
            record["online_s"] = record["wall_s"] - record["archive_s"]
            qc(case, folder)
            record["ref_qc"] = get(folder / "qc_receipt.json")["pass_all"]
            record["status"] = "DONE"
        except Exception as error:
            record.update(status="FAILED", error=str(error))
            raise
        finally:
            save(folder / "run.json", record)
        print(f"{case_id} {mesh} repeat {repeat}: {record['status']}", flush=True)


def main():
    global BIN, NATIVE, BUILD
    import os

    cases = {case["scene_id"]: case for case in scene_list()}
    parser = argparse.ArgumentParser(description="Run the dual-borehole COMSOL reference model.")
    parser.add_argument("--comsol-bin", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--case", choices=cases)
    mode.add_argument("--series", action="store_true")
    mode.add_argument("--compile-only", action="store_true")
    mode.add_argument("--worker", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--mesh", choices=SIZE, default="coarse")
    args = parser.parse_args()
    if os.name != "nt":
        parser.error("The COMSOL batch launcher requires Windows.")
    BIN = args.comsol_bin.resolve()
    NATIVE = args.out.resolve()
    BUILD = NATIVE / "compiled"
    for name in ("comsolcompile.exe", "comsolbatch.exe"):
        if not (BIN / name).is_file():
            parser.error(f"{BIN / name} does not exist")
    if args.worker:
        worker(args.worker.resolve())
        return
    NATIVE.mkdir(parents=True, exist_ok=False)
    compile_once()
    if args.compile_only:
        return
    if args.series:
        representatives = ("P00", "P11")
        jobs = [(case, mesh, 1) for case in representatives for mesh in SIZE]
        jobs += [(case, "coarse", 1) for case in cases if case not in representatives]
        jobs += [(case, "coarse", rep) for case in representatives for rep in (2, 3)]
    else:
        jobs = [(args.case, args.mesh, 1)]
    run_reference_cases(cases, jobs)


if __name__ == "__main__":
    main()
