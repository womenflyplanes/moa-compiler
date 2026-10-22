"""Universal MoA IR v2: shape-aware psi selection and layout-free normal forms.

A deliberately bounded prototype: concrete shaped arrays, constant index tuples,
partial psi, and one-dimensional partition lifting. DNF is not a general rewrite
system here. Physical row-major addressing is an explicit, separate lowering.
"""
from dataclasses import dataclass
from itertools import product
from math import prod
from typing import Tuple, Union

Shape = Tuple[int, ...]
Index = Tuple[int, ...]


def check_shape(shape: Shape) -> Shape:
    if not isinstance(shape, tuple) or any(type(d) is not int or d < 0 for d in shape):
        raise ValueError('shape must be a tuple of nonnegative integers')
    return shape


def iota(shape: Shape):
    check_shape(shape)
    return tuple(product(*(range(d) for d in shape)))


def tau(shape: Shape) -> int:
    return prod(check_shape(shape))


def check_index(index: Index, shape: Shape, *, partial=True) -> Index:
    check_shape(shape)
    if not isinstance(index, tuple) or len(index) > len(shape) or (not partial and len(index) != len(shape)):
        raise IndexError((index, shape))
    if any(type(i) is not int or i < 0 or i >= d for i, d in zip(index, shape)):
        raise IndexError((index, shape))
    return index


@dataclass(frozen=True)
class Array:
    name: str
    shape: Shape

    def __post_init__(self):
        check_shape(self.shape)
        if not isinstance(self.name, str) or not self.name:
            raise ValueError('array name required')


@dataclass(frozen=True)
class Psi:
    array: Array
    index: Index

    def __post_init__(self):
        check_index(self.index, self.array.shape)

    @property
    def shape(self) -> Shape:
        return self.array.shape[len(self.index):]


@dataclass(frozen=True)
class SelectionDNF:
    """Layout-independent normal form of a concrete prefix psi selection.

    The selection has residual shape, and each residual index maps to an
    original array index. No addresses or memory order are chosen.
    """
    source: Array
    prefix: Index
    shape: Shape
    source_indices: Tuple[Index, ...]

    def __post_init__(self):
        check_index(self.prefix, self.source.shape)
        if self.shape != self.source.shape[len(self.prefix):]:
            raise ValueError('residual shape mismatch')
        expected = tuple(self.prefix + suffix for suffix in iota(self.shape))
        if self.source_indices != expected:
            raise ValueError('source index domain mismatch')


def psi_to_dnf(node: Psi) -> SelectionDNF:
    """Reduce a concrete prefix psi selection to an explicit index-domain DNF."""
    return SelectionDNF(node.array, node.index, node.shape,
                        tuple(node.index + j for j in iota(node.shape)))


def compose_psi(first: Psi, suffix: Index) -> Psi:
    """(i psi A), then j psi, equals (i concatenated j) psi A."""
    check_index(suffix, first.shape)
    return Psi(first.array, first.index + suffix)


def gamma_row_major(index: Index, shape: Shape) -> int:
    """Explicit ONF layout mapping, not part of psi or DNF semantics."""
    check_index(index, shape, partial=False)
    address = 0
    for i, d in zip(index, shape):
        address = address * d + i
    return address


def lower_dnf_row_major(dnf: SelectionDNF):
    """Map a DNF selection to scalar offsets under a chosen row-major layout."""
    return tuple(gamma_row_major(i, dnf.source.shape) for i in dnf.source_indices)


@dataclass(frozen=True)
class Partition1D:
    source: Array
    tile: int

    def __post_init__(self):
        if len(self.source.shape) != 1:
            raise ValueError('Partition1D requires rank-one array')
        if type(self.tile) is not int or self.tile <= 0:
            raise ValueError('tile must be a positive integer')

    @property
    def shape(self) -> Shape:
        n = self.source.shape[0]
        return ((n + self.tile - 1) // self.tile, self.tile)

    def original_index(self, lifted: Index) -> Index:
        check_index(lifted, self.shape, partial=False)
        p, t = lifted
        i = p * self.tile + t
        if i >= self.source.shape[0]:
            raise IndexError('masked tail element')
        return (i,)

    def valid_indices(self):
        return tuple(j for j in iota(self.shape) if j[0] * self.tile + j[1] < self.source.shape[0])


@dataclass(frozen=True)
class LiftedPsi:
    partition: Partition1D
    index: Index

    def __post_init__(self):
        self.partition.original_index(self.index)


def lift_psi(node: Psi, tile: int) -> LiftedPsi:
    if len(node.array.shape) != 1 or len(node.index) != 1:
        raise ValueError('rank-one scalar psi required')
    part = Partition1D(node.array, tile)
    return LiftedPsi(part, divmod(node.index[0], tile))


def verify_lift(source: Array, tile: int) -> bool:
    part = Partition1D(source, tile)
    originals = tuple(part.original_index(i) for i in part.valid_indices())
    if len(set(originals)) != len(originals) or set(originals) != set(iota(source.shape)):
        return False
    return all(part.original_index(lift_psi(Psi(source, i), tile).index) == i
               for i in iota(source.shape))


Node = Union[Array, Psi, SelectionDNF, Partition1D, LiftedPsi]
