"""Single new structure: A1 evaluated through a consistent C2 rectangular boundary map."""

from legacy_network import FieldModel
from boundary_warp_reference import boundary_coordinates


class BoundaryModel(FieldModel):
    def __init__(self):
        super().__init__(local=False)

    def p(self, d, k, xy, t):
        # Parent receives mapped coordinates for EVERY spatial feature and the hole distance factor.
        # Data labels and physical quadrature weights are unchanged.
        return super().p(d, k, boundary_coordinates(xy), t)
