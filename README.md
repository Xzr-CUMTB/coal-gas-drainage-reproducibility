# Dual-borehole coal gas drainage benchmark

Experimental code for the homogeneous two-dimensional dual-borehole benchmark. The B1 neural surrogate predicts matrix and fracture pressures. Kernel ridge regression (KRR) predicts the two borehole flow rates. The benchmark evaluates 12 held-out cases and includes A1, B0, B1 and POD-KRR comparisons.

## Reproduce the reported tables

Use Python 3.9 or later and NumPy.

```sh
python -m pip install -r requirements.txt
```

Use the `benchmark-inputs.zip` archive supplied with the review materials. Run from the repository root.

```sh
python -m zipfile -e benchmark-inputs.zip .
python scripts/P01A/reproduce.py --out outputs/reproduction
```

The evaluation checks input hashes, recalculates pressure and flow errors, mass balance, boundary fluxes, mesh comparisons, engineering endpoints and timing summaries. It compares the recalculated values with the archived tables. The output directory must be new. This workflow uses the saved predictions and COMSOL fields. The original matched timings were measured with COMSOL 6.3 on an AMD Ryzen 7 9700X and with PyTorch 2.7.1 on an NVIDIA GeForce RTX 5070 Ti.

## Predict a new case

Place the frozen B1 and KRR weights supplied with the review materials under `checkpoints/P00C4/`, then install [PyTorch](https://pytorch.org/get-started/locally/). The example uses a 4.25 m borehole spacing and a permeability multiplier of 1.1.

```sh
python scripts/P01A/online.py --d 4.25 --k 1.1 --device cpu --out outputs/example
```

Use `--device cuda` on a CUDA-enabled system. The supported input range is 3–6 m for spacing and 0.5–2 for the multiplier. The reference permeability is 2.6e-17 m². Outputs include `flow.csv` (kg/s and cumulative kg), `fields.npz` (pressure in Pa), and `engineering.json` (including the first time the combined extracted mass reaches 20 kg). Predictions use the supplied fitted parameters.

## Rerun a COMSOL reference case

The batch runner uses COMSOL Multiphysics 6.3 with the Darcy's Law interface on Windows. Supply the directory containing `comsolcompile.exe` and `comsolbatch.exe`.

```powershell
python scripts/P01A/reference_run.py --comsol-bin "C:/path/to/COMSOL/Multiphysics/bin/win64" --case P00 --mesh coarse --out outputs/comsol-p00
```

Use `--series` in place of `--case P00 --mesh coarse` to reproduce the full 20-run design, including mesh comparisons and cold-process repeats. New COMSOL exports, logs and timings go to the requested output directory.

## Benchmark inputs

| File or directory | Contents |
| --- | --- |
| `data/P01A/design.csv` | Case IDs, borehole spacing and permeability |
| `data/P01A/grids/` | Sampling coordinates, region masks and integration weights |
| `data/P01A/references/` | COMSOL pressure fields, borehole flow and stored gas mass |
| `data/P01A/frozen/` | Predictions used for the reported comparisons |
| `data/P01A/boundaries/` | Boundary pressures and normal gradients |
| `reports/P01A/` | Protocol settings, archived numerical tables and timing inputs |
| `checkpoints/P00C4/` | Model parameters used by the benchmark |

The repository contains source, input settings and checksum manifests. The numerical archive and weights are separate review materials. The data describe a synthetic cross-section of unit thickness. Matrix and fracture pressures are absolute pressures. The 20 kg endpoint refers to the sum of the extracted gas from both boreholes.
