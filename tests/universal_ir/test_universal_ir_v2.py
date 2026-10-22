#!/usr/bin/env python3
"""Concrete semantic checks for Universal MoA IR v2 (stdlib only)."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'research' / 'universal_ir'))
from moa_universal_ir_v2 import (Array, Psi, Partition1D, check_index,
    compose_psi, gamma_row_major, iota, lift_psi, lower_dnf_row_major,
    psi_to_dnf, tau, verify_lift)


def rejects(call, error):
    try:
        call()
    except error:
        return
    raise AssertionError(f'expected {error.__name__}')


def main():
    cases = 0
    for shape in ((), (0,), (1,), (3,), (2, 3), (2, 0, 4), (2, 3, 4)):
        x = Array('X', shape)
        assert len(iota(shape)) == tau(shape)
        for prefix_len in range(len(shape) + 1):
            prefixes = iota(shape[:prefix_len])
            for prefix in prefixes:
                p = Psi(x, prefix)
                d = psi_to_dnf(p)
                assert d.shape == shape[prefix_len:]
                assert d.source_indices == tuple(prefix + s for s in iota(d.shape))
                offsets = lower_dnf_row_major(d)
                assert offsets == tuple(gamma_row_major(i, shape) for i in d.source_indices)
                for suffix_len in range(len(d.shape) + 1):
                    for suffix in iota(d.shape[:suffix_len]):
                        composed = compose_psi(p, suffix)
                        assert composed.index == prefix + suffix
                        assert composed.shape == shape[len(prefix) + len(suffix):]
                cases += 1
        print(f'PASS shape={shape} psi/DNF/ONF cases={cases}')
    for n in (0, 1, 2, 7, 8, 9, 16, 17):
        x = Array('X', (n,))
        for tile in (1, 2, 4, 8, 16):
            assert verify_lift(x, tile)
            p = Partition1D(x, tile)
            assert len(p.valid_indices()) == n
            for i in range(n):
                assert p.original_index(lift_psi(Psi(x, (i,)), tile).index) == (i,)
            print(f'PASS lift N={n} tile={tile} lifted_shape={p.shape}')
    rejects(lambda: Psi(Array('X', (2, 3)), (2,)), IndexError)
    rejects(lambda: Psi(Array('X', (2, 3)), (0, 0, 0)), IndexError)
    rejects(lambda: Psi(Array('X', (2, 3)), (-1,)), IndexError)
    rejects(lambda: Psi(Array('X', (2, 3)), (True,)), IndexError)
    rejects(lambda: gamma_row_major((1,), (2, 3)), IndexError)
    rejects(lambda: Partition1D(Array('X', (2, 3)), 2), ValueError)
    rejects(lambda: Partition1D(Array('X', (3,)), 0), ValueError)
    rejects(lambda: Partition1D(Array('X', (3,)), 2).original_index((1, 1)), IndexError)
    print(f'PASS {cases} concrete psi/DNF cases, 40 lifts, 8 rejection checks')


if __name__ == '__main__':
    main()
