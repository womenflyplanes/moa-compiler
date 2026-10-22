"""Universal MoA IR v1: shape/index semantics and exact 1-D partition lifting.

Pure Python, standard library only. This is NOT a general DNF rewrite engine.
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


def gamma(index: Index, shape: Shape) -> int:
    check_shape(shape)
    if len(index) != len(shape) or any(type(i) is not int or i < 0 or i >= d for i, d in zip(index, shape)):
        raise IndexError((index, shape))
    address = 0
    for i, d in zip(index, shape):
        address = address * d + i
    return address


@dataclass(frozen=True)
class Array:
    name: str
    shape: Shape

    def __post_init__(self):
        check_shape(self.shape)
        if not self.name:
            raise ValueError('array name required')


@dataclass(frozen=True)
class Psi:
    array: Array
    index: Index

    def __post_init__(self):
        gamma(self.index, self.array.shape)


@dataclass(frozen=True)
class Partition1D:
    """A lifted domain of shape (ceil(N/T), T), with tail validity mask."""
    source: Array
    tile: int

    def __post_init__(self):
        if len(self.source.shape) != 1:
            raise ValueError('Partition1D requires rank-one source')
        if type(self.tile) is not int or self.tile <= 0:
            raise ValueError('tile must be positive integer')

    @property
    def shape(self) -> Shape:
        n = self.source.shape[0]
        return ((n + self.tile - 1) // self.tile, self.tile)

    def original_index(self, lifted: Index) -> Index:
        gamma(lifted, self.shape)
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


Node = Union[Array, Psi, Partition1D, LiftedPsi]


def psi_to_address(node: Union[Psi, LiftedPsi]):
    """Lower an indexed selection to (array name, row-major scalar address)."""
    if isinstance(node, Psi):
        return node.array.name, gamma(node.index, node.array.shape)
    if isinstance(node, LiftedPsi):
        source = node.partition.source
        return source.name, gamma(node.partition.original_index(node.index), source.shape)
    raise TypeError(type(node))


def lift_psi(node: Psi, tile: int) -> LiftedPsi:
    """Rewrite rank-one ψ selection into a shape-lifted ψ selection."""
    if len(node.array.shape) != 1:
        raise ValueError('lift_psi v1 supports rank-one arrays only')
    p, t = divmod(node.index[0], tile) if type(tile) is int and tile > 0 else (None, None)
    partition = Partition1D(node.array, tile)
    return LiftedPsi(partition, (p, t))


def verify_lift(source: Array, tile: int):
    """Exhaustively verify coverage, uniqueness, and address equivalence."""
    part = Partition1D(source, tile)
    originals = [part.original_index(j) for j in part.valid_indices()]
    if len(originals) != len(set(originals)) or set(originals) != set(iota(source.shape)):
        return False
    return all(psi_to_address(Psi(source, i)) == psi_to_address(lift_psi(Psi(source, i), tile))
               for i in iota(source.shape))


def trace_lift(source: Array, tile: int):
    part = Partition1D(source, tile)
    return {
        'ir': 'universal-moa-ir-v1', 'source': {'name': source.name, 'shape': source.shape},
        'transform': 'rank-one partition/lift', 'tile': tile,
        'lifted_shape': part.shape, 'valid_domain': part.valid_indices(),
        'padding_slots': prod(part.shape) - prod(source.shape),
        'gamma_layout': 'row-major', 'verified': verify_lift(source, tile),
        'limitations': ['no general DNF reduction', 'no dependency analysis', 'no parallel schedule']
    }
