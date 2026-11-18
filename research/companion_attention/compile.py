"""Restricted ψ-indexed companion-kernel compiler.

A subscript in a phase specification denotes a tuple ψ selection. Indices
are rendered as γ addresses only in the C emitter. Loop annotations store
the selected Ω schedule. The JSON trace exposes the expressions and schedule.
"""
import argparse
import ast
import json
from pathlib import Path

from block import full_block


def phases(spec):
    """Parse a small statement language. Uppercase extents are fixed at compile time."""
    return ast.parse(spec).body


class Compiler:
    def __init__(self, shapes, extents, backend):
        self.shapes = shapes
        self.extents = extents
        self.backend = backend

    def expression(self, node):
        if isinstance(node, ast.Name):
            return str(self.extents.get(node.id, node.id))
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return str(node.value)
        if isinstance(node, ast.Subscript):
            assert isinstance(node.value, ast.Name)
            name = node.value.id
            shape = self.shapes[name]
            indices = node.slice.elts if isinstance(node.slice, ast.Tuple) else [node.slice]
            assert len(indices) == len(shape), (name, indices, shape)
            offset = self.expression(indices[0])
            for extent, index in zip(shape[1:], indices[1:]):
                offset = f"({offset} * {self.extents[extent]} + {self.expression(index)})"
            return f"{name}[{offset}]"
        if isinstance(node, ast.BinOp):
            ops = {ast.Add: "+", ast.Sub: "-", ast.Mult: "*", ast.Div: "/", ast.FloorDiv: "/", ast.Mod: "%"}
            return f"({self.expression(node.left)} {ops[type(node.op)]} {self.expression(node.right)})"
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
            return f"(-{self.expression(node.operand)})"
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            name = node.func.id
            assert name in {"sqrt", "exp", "fmax"}
            return f"{name}(" + ", ".join(self.expression(arg) for arg in node.args) + ")"
        raise ValueError(ast.dump(node))

    def statements(self, nodes, depth=1):
        lines = []
        pad = "    " * depth
        for node in nodes:
            if isinstance(node, ast.For):
                var = node.target.id
                mode = node.iter.func.id
                limit, = node.iter.args
                assert mode in {"serial", "owners"}
                if mode == "owners":
                    pragma = "omp parallel for schedule(static)" if self.backend == "openmp" else "acc parallel loop gang"
                    lines.append(pad + "#pragma " + pragma)
                elif self.backend == "openacc":
                    lines.append(pad + "#pragma acc loop seq")
                lines.append(pad + f"for (int {var} = 0; {var} < {self.expression(limit)}; ++{var}) {{")
                lines.extend(self.statements(node.body, depth + 1))
                lines.append(pad + "}")
            elif isinstance(node, ast.AnnAssign):
                name = node.target.id
                if isinstance(node.annotation, ast.Subscript):
                    extent = self.expression(node.annotation.slice)
                    value = f"double {name}[{extent}];"
                else:
                    ctype = "int" if node.annotation.id == "index" else "double"
                    value = f"{ctype} {name} = {self.expression(node.value)};"
                lines.append(pad + value)
            elif isinstance(node, ast.Assign):
                target, = node.targets
                lines.append(pad + f"{self.expression(target)} = {self.expression(node.value)};")
            elif isinstance(node, ast.AugAssign):
                op = {ast.Add: "+=", ast.Mult: "*=", ast.Div: "/="}[type(node.op)]
                lines.append(pad + f"{self.expression(node.target)} {op} {self.expression(node.value)};")
            else:
                raise ValueError(ast.dump(node))
        return lines

    def emit(self, name, inputs, outputs, spec, workspace=(), inout=()):
        args = [f"const double *restrict {a}" for a in inputs]
        args += [f"double *restrict {a}" for a in (*outputs, *inout)]
        from math import prod

        def size(array):
            return prod(self.extents[k] for k in self.shapes[array])

        if any(size(a) > 2**31 - 1 for a in self.shapes):
            raise ValueError("address exceeds the supported C int range")
        lines = ["#include <math.h>", "#include <stdlib.h>", "", f"void {name}(" + ", ".join(args) + ") {"]
        for a in workspace:
            lines.append(f"    double *{a} = malloc({size(a)} * sizeof(double));")
            lines.append(f"    if (!{a}) abort();")
        if self.backend == "openacc":
            region = []
            for arrays, clause in ((inputs, "copyin"), (outputs, "copyout"),
                                   (workspace, "create"), (inout, "copy")):
                sections = (f"{a}[0:{size(a)}]" for a in arrays)
                if arrays:
                    region.append(clause + "(" + ", ".join(sections) + ")")
            lines += ["    #pragma acc data " + " ".join(region), "    {"]
        lines.extend(self.statements(phases(spec), 2 if self.backend == "openacc" else 1))
        if self.backend == "openacc":
            lines.append("    }")
        lines.extend(f"    free({a});" for a in workspace)
        return "\n".join(lines + ["}", ""])


TRAINING = {
    **dict.fromkeys(("Q", "K", "GQ", "GK"), ("B", "N", "D")),
    **dict.fromkeys(("V", "U", "GV", "O"), ("B", "N", "E")),
    "A": ("B", "N", "N"), "g": ("N",), "a": ("N",),
}
BACKWARD = """\
for b in owners(B):
    scale: scalar = 1.0 / sqrt(D)
    g: scratch[N]
    for k in serial(N):
        for v in serial(E):
            GV[b,k,v] = 0.0
            for i in serial(N):
                GV[b,k,v] += A[b,i,k] * U[b,i,v]
        for j in serial(D):
            GQ[b,k,j] = 0.0
            GK[b,k,j] = 0.0
    for i in serial(N):
        for k in serial(N):
            g[k] = 0.0
            for v in serial(E):
                g[k] += U[b,i,v] * V[b,k,v]
        correction: scalar = 0.0
        for k in serial(N):
            correction += A[b,i,k] * g[k]
        for k in serial(N):
            gs: scalar = scale * A[b,i,k] * (g[k] - correction)
            for j in serial(D):
                GQ[b,i,j] += gs * K[b,k,j]
                GK[b,k,j] += gs * Q[b,i,j]
"""

FUSED = """\
for b in owners(B):
    scale: scalar = 1.0 / sqrt(D)
    g: scratch[N]
    a: scratch[N]
    for k in serial(N):
        for v in serial(E):
            O[b,k,v] = 0.0
            GV[b,k,v] = 0.0
        for j in serial(D):
            GQ[b,k,j] = 0.0
            GK[b,k,j] = 0.0
    for i in serial(N):
        m: scalar = -INFINITY
        for k in serial(N):
            a[k] = 0.0
            for j in serial(D):
                a[k] += Q[b,i,j] * K[b,k,j]
            a[k] *= scale
            m = fmax(m, a[k])
        z: scalar = 0.0
        for k in serial(N):
            a[k] = exp(a[k] - m)
            z += a[k]
        for k in serial(N):
            a[k] /= z
        for k in serial(N):
            g[k] = 0.0
            for v in serial(E):
                value: scalar = V[b,k,v]
                upstream: scalar = U[b,i,v]
                g[k] += upstream * value
                O[b,i,v] += a[k] * value
                GV[b,k,v] += a[k] * upstream
        correction: scalar = 0.0
        for k in serial(N):
            correction += a[k] * g[k]
        for k in serial(N):
            gs: scalar = scale * a[k] * (g[k] - correction)
            for j in serial(D):
                GQ[b,i,j] += gs * K[b,k,j]
                GK[b,k,j] += gs * Q[b,i,j]
"""

DECODE = """\
for h in owners(H):
    group: index = h // (H // G)
    scale: scalar = 1.0 / sqrt(D)
    a: scratch[N]
    m: scalar = -INFINITY
    for k in serial(N):
        a[k] = 0.0
        for j in serial(D):
            a[k] += Q[h,j] * K[group,k,j]
        a[k] *= scale
        m = fmax(m, a[k])
    z: scalar = 0.0
    for k in serial(N):
        a[k] = exp(a[k] - m)
        z += a[k]
    for v in serial(E):
        O[h,v] = 0.0
        for k in serial(N):
            O[h,v] += (a[k] / z) * V[group,k,v]
"""

RMS = """\
for b in owners(B):
    for i in serial(N):
        squares: scalar = 0.0
        product: scalar = 0.0
        for j in serial(D):
            squares += X[b,i,j] * X[b,i,j]
            product += U[b,i,j] * W[j] * X[b,i,j]
        inv: scalar = 1.0 / sqrt(squares / D + 1e-6)
        inverse[b,i] = inv
        for j in serial(D):
            Z[b,i,j] = W[j] * X[b,i,j] * inv
            GX[b,i,j] = (U[b,i,j] * W[j] - X[b,i,j] * inv * inv * product / D) * inv
for j in owners(D):
    GW[j] = 0.0
    for b in serial(B):
        for i in serial(N):
            GW[j] += U[b,i,j] * X[b,i,j] * inverse[b,i]
"""

APPEND = """\
for h in owners(G):
    for j in serial(D):
        K[h,N,j] = key[h,j]
    for v in serial(E):
        V[h,N,v] = value[h,v]
"""

FFN = """\
for t in owners(T):
    hidden: scratch[F]
    for f in serial(F):
        gate: scalar = 0.0
        up: scalar = 0.0
        gh: scalar = 0.0
        for d in serial(D):
            gate += X[t,d] * Wg[d,f]
            up += X[t,d] * Wu[d,f]
            gh += U[t,d] * Wd[f,d]
        sig: scalar = 1.0 / (1.0 + exp(-gate))
        silu: scalar = gate * sig
        hidden[f] = silu * up
        Hcache[t,f] = hidden[f]
        GU[t,f] = gh * up * sig * (1.0 + gate * (1.0 - sig))
        GV[t,f] = gh * silu
    for d in serial(D):
        Y[t,d] = 0.0
        GX[t,d] = 0.0
        for f in serial(F):
            Y[t,d] += hidden[f] * Wd[f,d]
            GX[t,d] += GU[t,f] * Wg[d,f] + GV[t,f] * Wu[d,f]
for d in owners(D):
    for f in serial(F):
        GWg[d,f] = 0.0
        GWu[d,f] = 0.0
        GWd[f,d] = 0.0
        for t in serial(T):
            GWg[d,f] += X[t,d] * GU[t,f]
            GWu[d,f] += X[t,d] * GV[t,f]
            GWd[f,d] += Hcache[t,f] * U[t,d]
"""

KERNELS = {
    "backward": (TRAINING, ("Q", "K", "V", "A", "U"), ("GV", "GQ", "GK"), BACKWARD),
    "fused": (TRAINING, ("Q", "K", "V", "U"), ("O", "GV", "GQ", "GK"), FUSED),
    "decode": ({"Q": ("H", "D"), "K": ("G", "N", "D"), "V": ("G", "N", "E"),
                "O": ("H", "E"), "a": ("N",)},
               ("Q", "K", "V"), ("O",), DECODE),
    "rmsnorm": ({**dict.fromkeys(("X", "U", "Z", "GX"), ("B", "N", "D")),
                 "W": ("D",), "GW": ("D",), "inverse": ("B", "N")},
                ("X", "U", "W"), ("Z", "GX", "GW"), RMS),
    "append": ({"K": ("G", "C", "D"), "V": ("G", "C", "E"), "key": ("G", "D"),
                "value": ("G", "E")}, ("key", "value"), (), APPEND),
    "ffn": ({**dict.fromkeys(("X", "U", "Y", "GX"), ("T", "D")),
             **dict.fromkeys(("Wg", "Wu", "GWg", "GWu"), ("D", "F")),
             **dict.fromkeys(("Wd", "GWd"), ("F", "D")),
             **dict.fromkeys(("Hcache", "GU", "GV"), ("T", "F")),
             "hidden": ("F",)},
            ("X", "U", "Wg", "Wu", "Wd"), ("Y", "GX", "GWg", "GWu", "GWd"), FFN),
}


KERNELS["block"], BLOCK_WORKSPACE = full_block(TRAINING, KERNELS["ffn"][0], BACKWARD, FFN)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("kernel", choices=KERNELS)
    parser.add_argument("backend", choices=("openmp", "openacc"))
    parser.add_argument("destination", type=Path)
    for flag, default in {"B": 2, "N": 4, "D": 8, "E": 8, "H": 4, "G": 2, "F": 16}.items():
        parser.add_argument("--" + flag.lower(), type=int, default=default)
    args = parser.parse_args()
    extents = {key: getattr(args, key.lower()) for key in ("B", "N", "D", "E", "H", "G", "F")}
    if min(extents.values()) < 1 or extents["H"] % extents["G"]:
        parser.error("Extents must be positive and G must divide H")
    if args.kernel == "block" and extents["E"] != extents["D"]:
        parser.error("The recovered full-block contract requires E = D")
    extents.update(T=extents["B"] * extents["N"], C=extents["N"] + 1)
    shapes, inputs, outputs, spec = KERNELS[args.kernel]
    workspace = {"rmsnorm": ("inverse",), "ffn": ("Hcache", "GU", "GV"), "block": BLOCK_WORKSPACE}.get(args.kernel, ())
    inout = ("K", "V") if args.kernel == "append" else ()
    compiler = Compiler(shapes, extents, args.backend)
    args.destination.mkdir(parents=True, exist_ok=True)
    trace = {
        "shapes": shapes, "extents": extents, "input": inputs, "observable": (*outputs, *inout), "workspace": workspace,
        "psi_phase_tree": ast.dump(ast.parse(spec), indent=2),
        "gamma": "row-major Horner linearization over each declared tuple",
        "omega": [
            {"index": node.target.id, "shape": [extents[node.iter.args[0].id]]}
            for node in phases(spec) if isinstance(node, ast.For) and node.iter.func.id == "owners"
        ],
        "backend": args.backend,
    }
    (args.destination / "trace.json").write_text(json.dumps(trace, indent=2) + "\n")
    (args.destination / "kernel.c").write_text(compiler.emit(args.kernel, inputs, outputs, spec, workspace, inout))


if __name__ == "__main__":
    main()
