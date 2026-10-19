"""Independent logical-index checks for the initial DNF -> ONF representation."""
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MODULE = ROOT / 'research/attention_rc1/moa_index_dnf_onf_v1.py'
spec = importlib.util.spec_from_file_location('moa_index_dnf_onf_v1', MODULE)
import sys
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)


def test_shape(shape):
    logical = {idx: (idx, sum(idx)) for idx in module.iota(shape)}
    for layout in ('row-major', 'column-major'):
        flat = module.pack(logical, shape, layout)
        # (iota(rho A)) psi A == A, elementwise in logical index order
        reconstructed = tuple(module.psi(idx, shape, flat, layout)
                              for idx in module.iota(shape))
        expected = tuple(logical[idx] for idx in module.iota(shape))
        assert reconstructed == expected, (shape, layout)
        for depth in range(len(shape) + 1):
            for full_idx in module.iota(shape):
                prefix = full_idx[:depth]
                dnf = module.DNFSelect(shape, prefix)
                onf = module.SequentialONF(dnf, layout)
                selected = onf.evaluate(flat)
                expected_values = tuple(logical[idx] for idx in dnf.full_indices())
                expected_selected = (expected_values[0] if not dnf.result_shape
                                     else expected_values)
                assert selected == expected_selected, (shape, layout, prefix)
    print('PASS shape=' + str(shape) + ' layouts=row/column prefix=all')


def main():
    for shape in ((), (5,), (2, 3), (2, 3, 4), (2, 0, 3)):
        test_shape(shape)
    for shape, idx in (((2, 3), (2,)), ((2, 3), (-1,)), ((2, 3), (0, 3))):
        try:
            module.DNFSelect(shape, idx)
        except IndexError:
            pass
        else:
            raise AssertionError('invalid index accepted')
    print('PASS invalid-index rejection')


if __name__ == '__main__':
    main()
