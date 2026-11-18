"""Operation and storage projection. Timing estimates are explicitly uncalibrated scenarios."""
import argparse
import ast
import csv
import json
from collections import Counter
from math import prod
from pathlib import Path

from compile import BLOCK_WORKSPACE, KERNELS, phases


def work(nodes, extents, instances=1):
    tally = Counter()
    for node in nodes:
        if isinstance(node, ast.For):
            bound = node.iter.args[0].id
            tally.update(work(node.body, extents, instances * extents[bound]))
            continue
        value = getattr(node, "value", None)
        if value is None:
            continue
        for expr in ast.walk(value):
            # Address arithmetic belongs to γ, outside the scalar expression count.
            if isinstance(expr, ast.Subscript):
                for index in ast.walk(expr.slice):
                    if isinstance(index, ast.BinOp):
                        tally[type(index.op).__name__] -= instances
            elif isinstance(expr, ast.Call):
                tally[expr.func.id] += instances
            elif isinstance(expr, ast.BinOp):
                tally[type(expr.op).__name__] += instances
        if isinstance(node, ast.AugAssign):
            tally[type(node.op).__name__] += instances
    return tally


def project(name, extents):
    shapes, inputs, outputs, spec = KERNELS[name]
    storage = {"rmsnorm": ("inverse",), "ffn": ("Hcache", "GU", "GV"), "block": BLOCK_WORKSPACE}
    scratch = {"backward": ("g",), "fused": ("g", "a"), "decode": ("a",), "ffn": ("hidden",),
               "block": ("g", "hidden")}
    size = lambda a: 8 * prod(extents[e] for e in shapes[a])
    ops = work(phases(spec), extents)
    for key in ("FloorDiv", "Mod"):
        ops.pop(key, None)
    return {
        "kernel": name,
        **extents,
        "floating_multiply": ops["Mult"],
        "floating_add": ops["Add"],
        "floating_subtract": ops["Sub"],
        "floating_divide": ops["Div"],
        "sqrt": ops["sqrt"], "exp": ops["exp"], "fmax": ops["fmax"],
        "first_phase_logical_owners": extents[phases(spec)[0].iter.args[0].id],
        "max_phase_logical_owners": max(extents[x.iter.args[0].id] for x in phases(spec)
                                         if isinstance(x, ast.For) and x.iter.func.id == "owners"),
        "input_bytes": sum(map(size, inputs)),
        "output_bytes": sum(map(size, outputs)),
        "append_cache_bytes_each_direction": sum(map(size, ("K", "V"))) if name == "append" else 0,
        "shared_workspace_bytes": sum(map(size, storage.get(name, ()))),
        "max_private_scratch_per_owner_bytes": max(map(size, scratch.get(name, ())), default=0)
        * (2 if name == "fused" else 1),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("anchors", type=Path)
    parser.add_argument("out", type=Path)
    args = parser.parse_args()
    anchors = json.loads(args.anchors.read_text())
    rows = []
    for n in (64, 128, 256, 512, 1024, 2048, 4096, 8192):
        extents = dict(B=2, N=n, D=64, E=64, F=256, H=8, G=2, T=2*n, C=n+1)
        for name in KERNELS:
            row = project(name, extents)
            # Paper I A100 model at Ω⟨256⟩ uses x = BN²D for forward.
            # Use the two leading forward contractions (2x) as a common work unit.
            a100 = anchors["A100"]["256"]
            row["a100_paper1_equal_multiply_rate_scenario_ms"] = (
                a100["intercept_ms"] + a100["slope"] * row["floating_multiply"] / 2
            )
            # Transfer the frozen affine law as a sensitivity calculation
            # only when the phase owner counts can populate the CPU team.
            owner_counts = [extents[x.iter.args[0].id] for x in phases(KERNELS[name][3])
                            if isinstance(x, ast.For) and x.iter.func.id == "owners"]
            for threads, ua in anchors["UA"].items():
                key = f"ua_{threads}_thread_paper1_equal_multiply_rate_scenario_ms"
                if int(threads) > min(owner_counts) or name == "append":
                    row[key] = None
                    continue
                (n0, t0, m0), (n1, t1, m1) = ua
                beta = (t1-t0)/(n1*n1-n0*n0)
                slope = (t1-t0)/(m1-m0)
                alpha = t0 - beta*n0*n0
                row[key] = 1000 * (alpha + slope * row["floating_multiply"])
            if name == "append":
                row["a100_paper1_equal_multiply_rate_scenario_ms"] = None
            rows.append(row)
    with args.out.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
