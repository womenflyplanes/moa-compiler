"""Minimal, testable MoA index -> DNF -> sequential ONF prototype.

DNF is a shape/index-domain specification independent of storage layout.
ONF is an explicit mapping to a flat storage offset. This is NOT a general
psi-reduction engine or a proof of correctness for arbitrary MoA programs.
"""
from dataclasses import dataclass
from itertools import product
from math import prod
from typing import Tuple

Shape = Tuple[int, ...]
Index = Tuple[int, ...]


def validate_shape(shape: Shape) -> Shape:
    shape = tuple(shape)
    if any(not isinstance(d, int) or isinstance(d, bool) or d < 0 for d in shape):
        raise ValueError('shape extents must be nonnegative integers')
    return shape


def iota(shape: Shape):
    """All complete indices in lexicographic order; scalar shape yields ()."""
    shape = validate_shape(shape)
    return tuple(product(*(range(d) for d in shape)))


def gamma(index: Index, shape: Shape, layout: str = 'row-major') -> int:
    """Map complete valid index to flat offset for a chosen layout."""
    shape = validate_shape(shape)
    index = tuple(index)
    if len(index) != len(shape):
        raise ValueError('gamma requires a complete index')
    if any(not isinstance(i, int) or isinstance(i, bool) or i < 0 or i >= d
           for i, d in zip(index, shape)):
        raise IndexError('index outside shape')
    if layout not in ('row-major', 'column-major'):
        raise ValueError('unknown layout')
    order = range(len(shape)) if layout == 'row-major' else range(len(shape)-1, -1, -1)
    offset = 0
    for axis in order:
        offset = offset * shape[axis] + index[axis]
    return offset


@dataclass(frozen=True)
class DNFSelect:
    """Machine-independent psi selection (complete or prefix index)."""
    shape: Shape
    prefix: Index

    def __post_init__(self):
        shape = validate_shape(self.shape)
        prefix = tuple(self.prefix)
        if len(prefix) > len(shape):
            raise ValueError('index longer than array shape')
        for i, d in zip(prefix, shape):
            if not isinstance(i, int) or isinstance(i, bool) or i < 0 or i >= d:
                raise IndexError('invalid psi prefix index')
        object.__setattr__(self, 'shape', shape)
        object.__setattr__(self, 'prefix', prefix)

    @property
    def result_shape(self) -> Shape:
        return self.shape[len(self.prefix):]

    def full_indices(self):
        return tuple(self.prefix + suffix for suffix in iota(self.result_shape))


@dataclass(frozen=True)
class SequentialONF:
    """Address-bearing realization of a DNF selection."""
    dnf: DNFSelect
    layout: str = 'row-major'

    def addresses(self):
        return tuple(gamma(idx, self.dnf.shape, self.layout)
                     for idx in self.dnf.full_indices())

    def evaluate(self, storage):
        if len(storage) != prod(self.dnf.shape):
            raise ValueError('storage size disagrees with shape')
        values = tuple(storage[a] for a in self.addresses())
        return values[0] if not self.dnf.result_shape else values


def psi(prefix: Index, shape: Shape, storage, layout='row-major'):
    """Reference selection via explicit logical indices and physical layout."""
    return SequentialONF(DNFSelect(shape, prefix), layout).evaluate(storage)


def pack(values_by_index, shape: Shape, layout='row-major'):
    """Construct flat storage from a mapping keyed by logical complete indices."""
    shape = validate_shape(shape)
    if set(values_by_index) != set(iota(shape)):
        raise ValueError('logical index set disagrees with shape')
    result = [None] * prod(shape)
    for idx, value in values_by_index.items():
        result[gamma(idx, shape, layout)] = value
    return result
