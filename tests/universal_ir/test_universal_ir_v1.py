import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from research.universal_ir.moa_universal_ir_v1 import (Array, Psi, Partition1D,
    LiftedPsi, gamma, iota, lift_psi, psi_to_address, trace_lift, verify_lift)


def rejects(f, error):
    try:
        f()
    except error:
        return
    raise AssertionError(f'expected {error.__name__}')


def run():
    for n in (0, 1, 2, 7, 8, 9, 16, 17):
        for tile in (1, 2, 4, 8, 16):
            a = Array('X', (n,))
            assert verify_lift(a, tile)
            part = Partition1D(a, tile)
            assert len(part.valid_indices()) == n
            for i in range(n):
                before = Psi(a, (i,))
                after = lift_psi(before, tile)
                assert psi_to_address(before) == psi_to_address(after) == ('X', i)
            print(f'PASS N={n} tile={tile} lifted_shape={part.shape} valid={n}')
    assert iota(()) == ((),)
    assert iota((2, 0, 3)) == ()
    assert gamma((1, 2), (2, 3)) == 5
    rejects(lambda: gamma((2,), (2,)), IndexError)
    rejects(lambda: Partition1D(Array('X', (3,)), 0), ValueError)
    rejects(lambda: Partition1D(Array('X', (2, 3)), 2), ValueError)
    rejects(lambda: LiftedPsi(Partition1D(Array('X', (5,)), 4), (1, 3)), IndexError)
    print('PASS invalid-index, masked-tail, rank and tile rejection')
    print(json.dumps(trace_lift(Array('X', (9,)), 4), indent=2))


if __name__ == '__main__':
    run()
