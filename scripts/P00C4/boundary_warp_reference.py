"""P00C4 optional mathematical prototype: rectangular outer-Neumann coordinate map.

This is a proposed output parameterization, NOT a trained/validated gas surrogate.
Use only for the current homogeneous/isotropic rectangular domain with circular holes
well inside the identity region. Keep labels, quadrature weights and geometry in the
physical coordinates. Apply the map to every spatial feature of the fracture decoder,
including its circular-distance prefactor; do not apply it to reference data.
"""

from __future__ import annotations
import torch


def flatten_axis(x: torch.Tensor, half_width: float, band_width: float = 1.0) -> torch.Tensor:
    if not 0.0 < band_width < half_width:
        raise ValueError("Require 0 < band_width < half_width.")
    a = half_width - band_width
    z = x.abs()
    s = ((z - a) / band_width).clamp(0.0, 1.0)
    # H(0)=0, H'(0)=1, H''(0)=0; H(1)=1, H'(1)=H''(1)=0.
    h = s + 4 * s**3 - 7 * s**4 + 3 * s**5
    outer = x.sign() * (a + band_width * h)
    return torch.where(z <= a, x, outer)


def boundary_coordinates(xy: torch.Tensor) -> torch.Tensor:
    if xy.ndim != 2 or xy.shape[1] != 2:
        raise ValueError("Expected [N,2] physical coordinates.")
    return torch.stack((flatten_axis(xy[:, 0], 8.0), flatten_axis(xy[:, 1], 6.0)), dim=1)


def structural_checks() -> dict:
    checks = {}
    for L in (8.0, 6.0):
        x = torch.tensor([-L, -L + 1, 0, L - 1, L], dtype=torch.float64, requires_grad=True)
        y = flatten_axis(x, L)
        dy = torch.autograd.grad(y.sum(), x, create_graph=True)[0]
        dd = torch.autograd.grad(dy.sum(), x)[0]
        assert torch.allclose(y, x, atol=1e-12, rtol=0)
        assert dy[[0, -1]].abs().max().item() < 1e-12
        assert torch.allclose(dy[1:4], torch.ones(3, dtype=dy.dtype), atol=1e-12, rtol=0)
        assert dd[[0, -1]].abs().max().item() < 1e-12
        checks[str(L)] = {
            "boundary_derivative_max": float(dy[[0, -1]].abs().max().detach()),
            "boundary_second_derivative_max": float(dd[[0, -1]].abs().max().detach()),
        }
    theta = torch.linspace(0, 2 * torch.pi, 1024, dtype=torch.float64)
    for d in (3.0, 4.5, 6.0):
        for cx in (-d / 2, d / 2):
            points = torch.stack((cx + 0.047 * torch.cos(theta), 0.047 * torch.sin(theta)), dim=1)
            assert torch.equal(points, boundary_coordinates(points))
    checks["circular_walls_unchanged"] = True
    checks["status"] = "STRUCTURAL_UNIT_TEST_ONLY_NO_TRAINING_OR_ACCURACY_CLAIM"
    return checks


if __name__ == "__main__":
    import json

    print(json.dumps(structural_checks(), indent=2))
