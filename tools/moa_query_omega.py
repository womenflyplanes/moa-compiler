"""Restricted MoA Omega lifting for query-disjoint output ownership.

An executable legality certificate over the finite query index domain.
Not a ten-field machine-resource planner or timing predictor.
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class QueryOmega:
    query_extent: int
    machine_shape: tuple
    owner_intervals: tuple

    def verify(self):
        n = self.query_extent
        shape = self.machine_shape
        if type(n) is not int or n < 1:
            return False
        if not shape or any(type(x) is not int or x < 1 for x in shape):
            return False
        p = 1
        for extent in shape:
            p *= extent
        if len(self.owner_intervals) != p:
            return False
        step = (n + p - 1) // p
        for owner, interval in enumerate(self.owner_intervals):
            if interval != (min(owner * step, n), min((owner + 1) * step, n)):
                return False
        return True

    def psi_owner(self, query_index):
        """Return gamma-addressed owner of a valid query index."""
        if not self.verify() or type(query_index) is not int:
            raise ValueError("invalid lifted mapping or query index")
        if not 0 <= query_index < self.query_extent:
            raise IndexError(query_index)
        for owner, (start, stop) in enumerate(self.owner_intervals):
            if start <= query_index < stop:
                return owner
        raise AssertionError("coverage invariant violated")


def lift_queries(n, machine_shape):
    if type(n) is not int or n < 1:
        raise ValueError("positive query extent required")
    shape = tuple(machine_shape)
    if not shape or any(type(x) is not int or x < 1 for x in shape):
        raise ValueError("positive machine shape required")
    p = 1
    for extent in shape:
        p *= extent
    step = (n + p - 1) // p
    intervals = tuple((min(i * step, n), min((i + 1) * step, n))
                      for i in range(p))
    result = QueryOmega(n, shape, intervals)
    if not result.verify():
        raise AssertionError("MoA Omega query partition certificate failed")
    return result
