"""Reproduce the reported validation and timing tables from archived arrays."""

import argparse
import math
from pathlib import Path

import numpy as np

from assessment import recompute
from base import REP, ROOT, get, read, save, sha
from boundary_replay import verify
from timing_cost import calculate

TABLES = (
    "SCORES",
    "FIELD_SCORES",
    "FLOW_SCORES",
    "ENGINEERING",
    "PAIRWISE_RANKING",
    "CANDIDATE_GAP",
    "NATIVE_FLOW_REPLAY",
    "MESH_REPLAY",
    "NATIVE_INSTANTANEOUS",
    "MATCHED_LATENCY",
    "COST_SCENARIOS",
    "BOUNDARY_ARRAY_REPLAY",
)


def verify_inputs():
    for name in ("inputs.sha256", "weights.sha256"):
        for line in (ROOT / "data" / name).read_text(encoding="utf-8").splitlines():
            digest, path = line.split("  ", 1)
            source = ROOT / path
            if not source.is_file():
                raise FileNotFoundError(
                    f"Missing {path}. Extract the release input archive into the repository root."
                )
            if sha(source) != digest:
                raise ValueError(f"Checksum mismatch for {path}")


def equal(actual, expected):
    if isinstance(expected, dict):
        return (
            isinstance(actual, dict)
            and actual.keys() == expected.keys()
            and all(equal(actual[k], value) for k, value in expected.items())
        )
    if actual == expected:
        return True
    try:
        return math.isclose(float(actual), float(expected), rel_tol=1e-9, abs_tol=1e-7)
    except (ValueError, TypeError):
        return False


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    verify_inputs()
    args.out.mkdir(parents=True, exist_ok=False)
    if not hasattr(np, "trapz"):
        np.trapz = np.trapezoid
    recompute(args.out, all_models=True, negative=True)
    calculate(args.out)
    verify(args.out)
    counts = {}
    for name in TABLES:
        actual = read(args.out / f"{name}.csv")
        expected = read(REP / f"{name}.csv")
        if len(actual) != len(expected):
            raise AssertionError(f"Row count differs in {name}")
        for index, (a, b) in enumerate(zip(actual, expected), start=1):
            if not equal(a, b):
                raise AssertionError(f"Table mismatch in {name}, data row {index}")
        counts[name] = len(actual)
    if not equal(get(args.out / "BATCH_COST.json"), get(REP / "BATCH_COST.json")):
        raise AssertionError("Batch cost mismatch")
    save(args.out / "verification.json", {"status": "PASS", "rows": counts})
    print(f"Verified {sum(counts.values())} rows across {len(counts)} tables.")


if __name__ == "__main__":
    main()
