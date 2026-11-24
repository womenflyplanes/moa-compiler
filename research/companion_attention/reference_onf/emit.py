"""Restricted structured ONF emitter with explicit storage and lift directives.

Recipes contain tuple selections and operational schedule decisions. Address
lowering is shared. Hand reference C is read only by the separate comparator.
"""
import ast


class Emitter:
    def __init__(self, arrays, precision="double"):
        self.arrays = arrays
        self.precision = precision

    def expression(self, node):
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Constant):
            return repr(node.value) if not isinstance(node.value, str) else node.value
        if isinstance(node, ast.Subscript):
            name = node.value.id
            indices = node.slice.elts if isinstance(node.slice, ast.Tuple) else [node.slice]
            if len(indices) == 1:
                return f"{name}[{self.expression(indices[0])}]"
            assert len(indices) <= len(self.arrays[name]), (name, len(indices))
            trailing = self.arrays[name][-len(indices)+1:]
            address = self.expression(indices[0])
            for extent, index in zip(trailing, indices[1:]):
                address = f"({address} * {extent} + {self.expression(index)})"
            return f"{name}[{address}]"
        if isinstance(node, ast.BinOp):
            op = {ast.Add:"+",ast.Sub:"-",ast.Mult:"*",ast.Div:"/",ast.Mod:"%",ast.FloorDiv:"/"}[type(node.op)]
            return f"({self.expression(node.left)} {op} {self.expression(node.right)})"
        if isinstance(node, ast.UnaryOp):
            assert isinstance(node.op,ast.USub)
            return "-" + self.expression(node.operand)
        if isinstance(node, ast.Compare):
            assert len(node.ops) == 1
            op={ast.Gt:">",ast.Lt:"<",ast.Eq:"=="}[type(node.ops[0])]
            return f"{self.expression(node.left)} {op} {self.expression(node.comparators[0])}"
        if isinstance(node, ast.Call):
            if node.func.id == "float_cast":
                return "(float) " + self.expression(node.args[0])
            if node.func.id == "f32":
                return str(node.args[0].value) + "f"
            return node.func.id + "(" + ", ".join(self.expression(x) for x in node.args) + ")"
        raise ValueError(ast.dump(node))

    def statements(self, nodes, depth=1):
        result=[]
        pad="    "*depth
        for node in nodes:
            if isinstance(node,ast.With):
                call=node.items[0].context_expr
                if call.func.id=="scope":
                    result.append(pad+"{")
                    result.extend(self.statements(node.body,depth+1))
                    result.append(pad+"}")
                else:
                    assert call.func.id=="directive"
                    result.append(pad+"#pragma "+call.args[0].value)
                    result.extend(self.statements(node.body,depth))
            elif isinstance(node,ast.For):
                assert node.iter.func.id=="serial" and len(node.iter.args)==1
                bound=self.expression(node.iter.args[0])
                name=node.target.id
                result.append(pad+f"for (int {name} = 0; {name} < {bound}; {name}++) {{")
                result.extend(self.statements(node.body,depth+1))
                result.append(pad+"}")
            elif isinstance(node,ast.Return):
                result.append(pad+"return "+self.expression(node.value)+";")
            elif isinstance(node,ast.If):
                result.append(pad+"if ("+self.expression(node.test)+") {")
                result.extend(self.statements(node.body,depth+1))
                result.append(pad+"}")
                assert not node.orelse
            elif isinstance(node,ast.AnnAssign):
                name=node.target.id
                annotation=node.annotation
                if isinstance(annotation,ast.Subscript):
                    size=self.expression(annotation.slice)
                    if annotation.value.id=="heapwide":
                        text=f"{self.precision} *{name} = malloc(sizeof({self.precision}) * (size_t)B*N*N);"
                    elif annotation.value.id=="heap_reverse":
                        text=f"{self.precision} *{name} = malloc({size} * sizeof({self.precision}));"
                    elif annotation.value.id=="heap":
                        text=f"{self.precision} *{name} = malloc(sizeof({self.precision}) * {size});"
                    elif annotation.value.id=="scratch":
                        text=f"{self.precision} {name}[{size}];"
                    else:
                        raise ValueError(ast.dump(annotation))
                else:
                    ctype={"scalar":self.precision,"const_scalar":"const "+self.precision,"index":"int"}[annotation.id]
                    text=f"{ctype} {name}"
                    if node.value is not None:
                        text+=" = "+self.expression(node.value)
                    text+=";"
                result.append(pad+text)
            elif isinstance(node,ast.Assign):
                result.append(pad+self.expression(node.targets[0])+" = "+self.expression(node.value)+";")
            elif isinstance(node,ast.AugAssign):
                op={ast.Add:"+=",ast.Sub:"-=",ast.Mult:"*=",ast.Div:"/="}[type(node.op)]
                result.append(pad+self.expression(node.target)+" "+op+" "+self.expression(node.value)+";")
            elif isinstance(node,ast.Expr):
                result.append(pad+self.expression(node.value)+";")
            else:
                raise ValueError(ast.dump(node))
        return result

    def body(self, spec):
        return "\n".join(self.statements(ast.parse(spec).body))+"\n"
