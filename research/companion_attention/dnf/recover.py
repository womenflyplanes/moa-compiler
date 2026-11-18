"""Recover exact-real selected-component expressions from restricted compiler phases.

Loops are visited symbolically. An additive update to a target whose tuple
does not contain the loop index becomes a bounded reduction on loop exit.
A scalar fmax recurrence becomes a bounded maximum. Array temporaries are
substituted at each ψ selection. Unsupported recurrences fail explicitly.
"""
import ast
import itertools
from dataclasses import dataclass

import sympy as s

Fresh = itertools.count()
Bounds = {name: s.Symbol(name, integer=True, positive=True)
          for name in ("B", "N", "D", "E", "F", "H", "G", "T", "C")}
Psi = s.Function("psi")
Reduce = s.Function("sum_over")
Maximum = s.Function("max_over")


def select(name, *indices):
    return Psi(s.Symbol(name), *indices)


@dataclass
class Rule:
    indices: tuple
    value: object


def reduce_sum(expr, index, extent):
    """Normalize linearity and extract factors independent of the bound index."""
    expr = s.factor(expr)
    if expr == 0:
        return s.S.Zero
    if index not in expr.free_symbols:
        return extent * expr
    independent, dependent = expr.as_independent(index, as_Add=False)
    return independent * Reduce(dependent, index, extent)


class Recover:
    def __init__(self, shapes, inputs):
        self.shapes = shapes
        self.inputs = set(inputs)
        self.scalar = dict(Bounds)
        self.scalar["INFINITY"] = s.oo
        self.array = {}
        self.events = []
        self.domains = {}

    def read(self, name, indices):
        if name not in self.array:
            if name not in self.inputs:
                raise ValueError(f"Read before definition: {name}{indices}")
            return select(name, *indices)
        rule = self.array[name]
        if len(rule.indices) != len(indices):
            raise ValueError(f"Rank mismatch reading {name}")
        bindings = {}
        for formal, actual in zip(rule.indices, indices):
            if isinstance(formal, s.Symbol):
                if formal in bindings and bindings[formal] != actual:
                    raise ValueError(f"Repeated tuple index for {name}")
                bindings[formal] = actual
            elif formal != actual:
                raise ValueError(f"Unsupported partial array write: {name}")
        return rule.value.xreplace(bindings)

    def expr(self, node):
        if isinstance(node, ast.Constant):
            return s.Rational(str(node.value))
        if isinstance(node, ast.Name):
            return self.scalar[node.id]
        if isinstance(node, ast.Subscript):
            seq = node.slice.elts if isinstance(node.slice, ast.Tuple) else [node.slice]
            return self.read(node.value.id, tuple(self.expr(x) for x in seq))
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
            return -self.expr(node.operand)
        if isinstance(node, ast.BinOp):
            left, right = self.expr(node.left), self.expr(node.right)
            if isinstance(node.op, ast.Add):
                return left + right
            if isinstance(node.op, ast.Sub):
                return left - right
            if isinstance(node.op, ast.Mult):
                return left * right
            if isinstance(node.op, ast.Div):
                return left / right
            if isinstance(node.op, ast.FloorDiv):
                return s.floor(left / right)
            if isinstance(node.op, ast.Mod):
                return s.Mod(left, right)
        if isinstance(node, ast.Call):
            return {"sqrt": s.sqrt, "exp": s.exp, "fmax": s.Max}[node.func.id](
                *(self.expr(x) for x in node.args))
        raise ValueError(ast.dump(node))

    def target(self, node):
        if isinstance(node, ast.Name):
            return node.id, None
        assert isinstance(node, ast.Subscript)
        seq = node.slice.elts if isinstance(node.slice, ast.Tuple) else [node.slice]
        return node.value.id, tuple(self.expr(x) for x in seq)

    def store(self, name, indices, value, mode):
        if indices is None:
            self.scalar[name] = value
        else:
            self.array[name] = Rule(indices, value)
        self.events.append((name, indices, mode))

    def statements(self, nodes):
        for node in nodes:
            if isinstance(node, ast.For):
                self.loop(node)
            elif isinstance(node, ast.AnnAssign):
                if isinstance(node.annotation, ast.Subscript):
                    continue
                self.store(node.target.id, None, self.expr(node.value), "assign")
            elif isinstance(node, ast.Assign):
                name, indices = self.target(node.targets[0])
                mode = "assign"
                # Recognize the complete max recurrence, including its target.
                if (isinstance(node.value, ast.Call) and node.value.func.id == "fmax"
                        and isinstance(node.targets[0], ast.Name)
                        and isinstance(node.value.args[0], ast.Name)
                        and node.value.args[0].id == name):
                    mode = "max"
                self.store(name, indices, self.expr(node.value), mode)
            elif isinstance(node, ast.AugAssign):
                name, indices = self.target(node.target)
                old = self.scalar[name] if indices is None else self.read(name, indices)
                value = self.expr(node.value)
                if isinstance(node.op, ast.Add):
                    new, mode = old + value, "add"
                elif isinstance(node.op, ast.Mult):
                    new, mode = old * value, "multiply"
                elif isinstance(node.op, ast.Div):
                    new, mode = old / value, "divide"
                else:
                    raise ValueError(ast.dump(node))
                self.store(name, indices, new, mode)
            else:
                raise ValueError(ast.dump(node))

    def loop(self, node):
        index = s.Dummy("i" + str(next(Fresh)), integer=True)
        extent = self.expr(node.iter.args[0])
        self.domains[index] = extent
        name = node.target.id
        previous_index = self.scalar.get(name)
        arrays_before, scalars_before = dict(self.array), dict(self.scalar)
        self.scalar[name] = index
        start = len(self.events)
        self.statements(node.body)
        changes = self.events[start:]
        for changed in dict.fromkeys(e[0] for e in changes):
            entries = [e for e in changes if e[0] == changed]
            _, indices, last_mode = entries[-1]
            if indices is not None and any(index in x.free_symbols for x in indices):
                continue
            value = self.scalar[changed] if indices is None else self.array[changed].value
            if all(e[2] == "add" for e in entries):
                if indices is None:
                    before = scalars_before[changed]
                elif changed in arrays_before:
                    after = self.array[changed]
                    self.array[changed] = arrays_before[changed]
                    before = self.read(changed, after.indices)
                    self.array[changed] = after
                else:
                    raise ValueError(f"Reduction without initialization: {changed}")
                if index in before.free_symbols:
                    raise ValueError(f"Loop-dependent initial reduction: {changed}")
                result = before + reduce_sum(s.factor(value-before), index, extent)
            elif all(e[2] == "max" for e in entries):
                before = scalars_before[changed]
                if before != -s.oo:
                    raise ValueError("Maximum requires negative-infinity identity")
                candidate = [arg for arg in value.args if arg != -s.oo]
                result = Maximum(s.Max(*candidate), index, extent)
            else:
                # Repeated simple writes independent of the loop index are not
                # interpreted as reductions. Locals are consumed within scope.
                continue
            if indices is None:
                self.scalar[changed] = result
            else:
                self.array[changed] = Rule(indices, result)
            # Propagate a single reduction update to its enclosing loop.
            self.events = [e for n, e in enumerate(self.events)
                           if n < start or e[0] != changed]
            self.events.append((changed, indices, "add" if last_mode == "add" else "max"))
        if previous_index is None:
            self.scalar.pop(name, None)
        else:
            self.scalar[name] = previous_index

    def run(self, spec):
        self.statements(ast.parse(spec).body)
        return self
