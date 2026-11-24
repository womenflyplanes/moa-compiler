"""Untimed finite checks against the recovered hand C and reference arrays."""
import argparse
import ctypes
import math
from pathlib import Path

import numpy as np


def call(path, name, *arrays, scale=None):
    function = getattr(ctypes.CDLL(str(path.resolve())), name)
    pointer = np.ctypeslib.ndpointer(dtype=np.float64, flags="C_CONTIGUOUS")
    function.argtypes = [pointer] * len(arrays)
    args = list(arrays)
    if scale is not None:
        function.argtypes = [pointer] * len(arrays) + [ctypes.c_double]
        args.append(scale)
    function.restype = None
    function(*args)


def check(label, observed, expected):
    np.testing.assert_allclose(observed, expected, atol=1e-12, rtol=1e-12)
    print(f"{label}: max abs {np.max(np.abs(observed - expected)):.3g}")


def probabilities(scores):
    weights = np.exp(scores - scores.max(axis=-1, keepdims=True))
    return weights / weights.sum(axis=-1, keepdims=True)


def verify(generated, hand, hand_so, block_data):
    rng = np.random.default_rng(20261117)
    q, k, v, u = rng.uniform(-1, 1, size=(4, 2, 4, 8))
    scale = 1 / math.sqrt(8)
    a = probabilities((q @ k.swapaxes(-2, -1)) * scale)
    g = u @ v.swapaxes(-2, -1)
    gs = scale * a * (g - (a*g).sum(axis=-1, keepdims=True))
    expected = (a.swapaxes(-2, -1) @ u, gs @ k, gs.swapaxes(-2, -1) @ q)
    paper2 = {
        "backward": ((q, k, v, a, u), expected),
        "fused": ((q, k, v, u), (a @ v, *expected)),
    }
    hand_results = {}
    for name, (inputs, refs) in paper2.items():
        for form in ("dnf", "onf"):
            observed = tuple(np.empty_like(ref) for ref in refs)
            call(hand_so, f"moa_{name}_{form}", *inputs, *observed, scale=scale)
            for index, (actual, ref) in enumerate(zip(observed, refs)):
                check(f"hand {name} {form}[{index}]", actual, ref)
            if form == "dnf":
                hand_results[name] = observed

    data = hand / "verification/data"

    def read(name):
        return np.fromfile(data / (name + ".bin"), dtype=np.float64)

    for backend in ("openmp", "openacc"):
        for name, (inputs, _) in paper2.items():
            refs = hand_results[name]
            observed = tuple(np.empty_like(ref) for ref in refs)
            call(generated / name / backend / "kernel.so", name, *inputs, *observed)
            for index, (actual, ref) in enumerate(zip(observed, refs)):
                check(f"{backend} {name}[{index}]", actual, ref)
        for marker in ("c", "g"):
            inputs = [np.fromfile(block_data / f"fb_{x}.bin", dtype=np.float64)
                      for x in ("X", "gamma1", "gamma2", "Wq", "Wk", "Wv", "Wg", "Wu", "Wd", "GY")]
            names = ("Y", "GX", "Ggamma1", "Ggamma2", "GWq", "GWk", "GWv", "GWg", "GWu", "GWd")
            refs = [np.fromfile(block_data / f"fb_{marker}_{x}.bin", dtype=np.float64) for x in names]
            observed = [np.empty_like(ref) for ref in refs]
            call(generated / "block" / backend / "kernel.so", "block", *inputs, *observed)
            for name_out, actual, ref in zip(names, observed, refs):
                check(f"{backend} block vs hand {marker} {name_out}", actual, ref)
        qd, kd, vd = rng.uniform(-1, 1, size=(4, 8)), k, v
        out = np.empty((4, 8))
        ref = np.stack([probabilities(kd[h//2] @ qd[h] * scale) @ vd[h//2] for h in range(4)])
        call(generated / "decode" / backend / "kernel.so", "decode", qd, kd, vd, out)
        check(f"{backend} decode GQA", out, ref)
        for variant, h, gcount in (("mqa", 4, 1), ("single_decode", 1, 1)):
            qt, kt, vt = (np.ascontiguousarray(x[:limit]) for x, limit in
                          ((qd, h), (kd, gcount), (vd, gcount)))
            observed = np.empty((h, 8))
            ref = np.stack([probabilities(kt[0] @ qi * scale) @ vt[0] for qi in qt])
            call(generated / variant / backend / "kernel.so", "decode", qt, kt, vt, observed)
            check(f"{backend} {variant}", observed, ref)
        key = rng.normal(size=(2, 8))
        val = rng.normal(size=(2, 8))
        kc, vc = (np.pad(x, ((0, 0), (0, 1), (0, 0))) for x in (kd, vd))
        call(generated / "append" / backend / "kernel.so", "append", key, val, kc, vc)
        np.testing.assert_array_equal(kc, np.concatenate((kd, key[:,None]), axis=1))
        np.testing.assert_array_equal(vc, np.concatenate((vd, val[:,None]), axis=1))
        print(f"{backend} append: exact, including unchanged prefix")
        for name, inputs, outputs in (
            ("rmsnorm", ("X", "GZ", "gamma"), ("Z", "GX_norm", "Ggamma")),
            ("ffn", ("X", "GF", "Wg", "Wu", "Wd"), ("F", "GX_ffn", "GWg", "GWu", "GWd")),
        ):
            refs = [read("ref_" + output) for output in outputs]
            observed = [np.empty_like(ref) for ref in refs]
            call(generated / name / backend / "kernel.so", name, *(read(x) for x in inputs), *observed)
            for name_out, actual, ref in zip(outputs, observed, refs):
                check(f"{backend} {name} {name_out}", actual, ref)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("generated", type=Path)
    parser.add_argument("hand", type=Path)
    parser.add_argument("hand_so", type=Path)
    parser.add_argument("block_data", type=Path)
    args = parser.parse_args()
    verify(args.generated, args.hand, args.hand_so, args.block_data)
