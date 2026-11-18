"""Exact-real canonicalization with explicit finite reduction binders."""
import sympy as s
from .recover import Reduce, Maximum, Bounds


def algebra(expr):
    return s.factor(expr)


def canonical(expr, bindings=None, depth=0):
    bindings = {} if bindings is None else bindings
    if expr.func in (Reduce, Maximum):
        body,index,extent = expr.args
        variable = s.Symbol(f"bound_{depth}", integer=True)
        body = canonical(body, {**bindings,index:variable},depth+1)
        extent = canonical(extent,bindings,depth)
        if expr.func == Maximum:
            return Maximum(body,variable,extent)
        if body == 0:
            return s.S.Zero
        if variable not in body.free_symbols:
            return extent*body
        independent,dependent=body.as_independent(variable,as_Add=False)
        return independent*Reduce(dependent,variable,extent)
    if not expr.args:
        return bindings.get(expr,expr)
    if expr.func == s.floor and expr.args[0] == Bounds["H"]/Bounds["G"]:
        # Shape premise G divides H.
        return Bounds["H"]/Bounds["G"]
    args=[canonical(x,bindings,depth) for x in expr.args]
    value=expr.func(*args)
    return algebra(value) if value.func in (s.Add,s.Mul,s.Pow) else value


def free_symbols(rank):
    return tuple(s.Symbol(f"free_{i}",integer=True,nonnegative=True) for i in range(rank))


def normalize(expr):
    result=expr
    for _ in range(4):
        next_value=canonical(result)
        if next_value==result:
            return next_value
        result=next_value
    return result
