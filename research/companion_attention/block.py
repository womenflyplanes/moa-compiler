"""Compose the Paper IV pre-norm block from the existing phase specifications."""
import ast


def substitute(source, **bindings):
    """Bind phase-array names structurally before the common γ lowering."""
    class Bind(ast.NodeTransformer):
        def visit_Name(self, node):
            return ast.copy_location(ast.Name(id=bindings.get(node.id, node.id), ctx=node.ctx), node)
    return ast.unparse(Bind().visit(ast.parse(source)))


NORM = """\
for t in owners(T):
    ss: scalar = 0.0
    for d in serial(D):
        ss += X[t,d] * X[t,d]
    inv[t] = 1.0 / sqrt(ss / D + 1e-6)
    for d in serial(D):
        Z[t,d] = scale[d] * X[t,d] * inv[t]
"""

NORM_ADJOINT = """\
for t in owners(T):
    corr: scalar = 0.0
    for d in serial(D):
        corr += U[t,d] * scale[d] * X[t,d]
    for d in serial(D):
        result[t,d] = residual[t,d] + inv[t] * (U[t,d] * scale[d] - X[t,d] * inv[t] * inv[t] * corr / D)
for d in owners(D):
    Gscale[d] = 0.0
    for t in serial(T):
        Gscale[d] += U[t,d] * X[t,d] * inv[t]
"""

PROJECTION = """\
for t in owners(T):
    for o in serial(D):
        z: scalar = 0.0
        for d in serial(D):
            z += Z1[t,d] * W[d,o]
        array[t // N,t % N,o] = z
"""

ATTENTION = """\
for t in owners(T):
    b: index = t // N
    i: index = t % N
    scale: scalar = 1.0 / sqrt(D)
    m: scalar = -INFINITY
    for k in serial(N):
        score: scalar = 0.0
        for d in serial(D):
            score += Q[b,i,d] * K[b,k,d]
        A[b,i,k] = score * scale
        m = fmax(m, A[b,i,k])
    sumexp: scalar = 0.0
    for k in serial(N):
        A[b,i,k] = exp(A[b,i,k] - m)
        sumexp += A[b,i,k]
    for k in serial(N):
        A[b,i,k] /= sumexp
    for d in serial(D):
        aout: scalar = 0.0
        for k in serial(N):
            aout += A[b,i,k] * V[b,k,d]
        R[t,d] = X[t,d] + aout
"""

PROJECTION_ADJOINT = """\
for t in owners(T):
    for d in serial(D):
        z: scalar = 0.0
        for o in serial(D):
            z += GQ[t // N,t % N,o] * Wq[d,o] + GK[t // N,t % N,o] * Wk[d,o] + GV[t // N,t % N,o] * Wv[d,o]
        GZ1[t,d] = z
for d in owners(D):
    for o in serial(D):
        GWq[d,o] = 0.0
        GWk[d,o] = 0.0
        GWv[d,o] = 0.0
        for t in serial(T):
            GWq[d,o] += Z1[t,d] * GQ[t // N,t % N,o]
            GWk[d,o] += Z1[t,d] * GK[t // N,t % N,o]
            GWv[d,o] += Z1[t,d] * GV[t // N,t % N,o]
"""


def full_block(training, ffn_shapes, backward, ffn):
    inputs = ("X", "scale1", "scale2", "Wq", "Wk", "Wv", "Wg", "Wu", "Wd", "GY")
    outputs = ("Y", "GX", "Gscale1", "Gscale2", "GWq", "GWk", "GWv", "GWg", "GWu", "GWd")
    shapes = {**training, **ffn_shapes}
    shapes.update(dict.fromkeys(("R", "Z1", "Z2", "GZ1", "GZ2", "GY", "FF", "GR"), ("T", "D")))
    shapes.update(dict.fromkeys(("scale1", "scale2", "Gscale1", "Gscale2"), ("D",)))
    shapes.update(dict.fromkeys(("inv1", "inv2"), ("T",)))
    shapes.update(dict.fromkeys(("Wq", "Wk", "Wv", "GWq", "GWk", "GWv"), ("D", "D")))
    shapes["FFgv"] = ("T", "F")
    parts = [substitute(NORM, Z="Z1", scale="scale1", inv="inv1")]
    parts.extend(substitute(PROJECTION, W="W" + key.lower(), array=key) for key in ("Q", "K", "V"))
    parts.extend([
        ATTENTION,
        substitute(NORM, X="R", Z="Z2", scale="scale2", inv="inv2"),
        substitute(ffn, X="Z2", U="GY", Y="FF", GX="GZ2", GV="FFgv"),
        "for t in owners(T):\n    for d in serial(D):\n        Y[t,d] = R[t,d] + FF[t,d]",
        substitute(NORM_ADJOINT, X="R", U="GZ2", scale="scale2", inv="inv2",
                   result="GR", residual="GY", Gscale="Gscale2"),
        "for b in owners(B):\n    for i in serial(N):\n        for d in serial(D):\n            U[b,i,d] = GR[b*N+i,d]",
        backward,
        PROJECTION_ADJOINT,
        substitute(NORM_ADJOINT, U="GZ1", scale="scale1", inv="inv1",
                   result="GX", residual="GR", Gscale="Gscale1"),
    ])
    shapes["GV"] = ("B", "N", "E")
    shapes["U"] = ("B", "N", "E")
    workspace = ("R", "Z1", "Z2", "GZ1", "GZ2", "FF", "GR", "inv1", "inv2", "Q", "K", "V",
                 "A", "U", "GQ", "GK", "GV", "Hcache", "GU", "FFgv")
    return (shapes, inputs, outputs, "\n".join(parts)), workspace
