"""Independent RMSNorm and gated-FFN selected-component schedule recipes."""
SHAPES={name:('NT','D') for name in ('X','GZ','Z','GX','GF','F')}
SHAPES.update({name:('D','DFF') for name in ('Wg','Wu','GWg','GWu')})
SHAPES.update({'U':('NT','DFF'),'Wd':('DFF','D'),'GWd':('DFF','D')})

def rmsnorm(backend):
    assert backend=='openmp'
    return '''with directive("omp parallel for"):
    for t in serial(NT):
        sumsq: scalar = 0.0
        for d in serial(D):
            x: scalar = X[t,d]
            sumsq += x * x
        r[t] = sqrt(sumsq / D + EPS)
with directive("omp parallel for collapse(2)"):
    for t in serial(NT):
        for d in serial(D):
            Z[t,d] = gamma[d] * (X[t,d] / r[t])
with directive("omp parallel for"):
    for t in serial(NT):
        correction: scalar = 0.0
        for d in serial(D):
            x_hat_d: scalar = X[t,d] / r[t]
            correction += GZ[t,d] * gamma[d] * x_hat_d
        for e in serial(D):
            x_hat_e: scalar = X[t,e] / r[t]
            GX[t,e] = (GZ[t,e] * gamma[e] - (x_hat_e / D) * correction) / r[t]
with directive("omp parallel for reduction(+:Ggamma[0:D])"):
    for t in serial(NT):
        for d in serial(D):
            Ggamma[d] += GZ[t,d] * (X[t,d] / r[t])
'''

def ffn(backend):
    assert backend=='openmp'
    return '''with directive("omp parallel for"):
    for t in serial(NT):
        Vrow: heap[DFF]
        Hrow: heap[DFF]
        for f in serial(DFF):
            u: scalar = 0.0
            v: scalar = 0.0
            for d in serial(D):
                u += X[t,d] * Wg[d,f]
                v += X[t,d] * Wu[d,f]
            U[t,f] = u
            Vrow[f] = v
            Hrow[f] = silu(u) * v
        for d in serial(D):
            fo: scalar = 0.0
            for f in serial(DFF):
                fo += Hrow[f] * Wd[f,d]
            F[t,d] = fo
        free(Vrow)
        free(Hrow)
with directive("omp parallel for reduction(+:GWg[0:D*DFF], GWu[0:D*DFF], GWd[0:DFF*D])"):
    for t in serial(NT):
        Vrow: heap[DFF]
        Hrow: heap[DFF]
        GHrow: heap[DFF]
        GUrow: heap[DFF]
        GVrow: heap[DFF]
        for f in serial(DFF):
            v: scalar = 0.0
            for d in serial(D):
                v += X[t,d] * Wu[d,f]
            Vrow[f] = v
            Hrow[f] = silu(U[t,f]) * v
        for f in serial(DFF):
            for d in serial(D):
                GWd[f,d] += Hrow[f] * GF[t,d]
        for f in serial(DFF):
            gh: scalar = 0.0
            for d in serial(D):
                gh += GF[t,d] * Wd[f,d]
            GHrow[f] = gh
            GVrow[f] = gh * silu(U[t,f])
            GUrow[f] = gh * Vrow[f] * silu_grad(U[t,f])
        for d in serial(D):
            for f in serial(DFF):
                GWg[d,f] += X[t,d] * GUrow[f]
                GWu[d,f] += X[t,d] * GVrow[f]
        for d in serial(D):
            gx: scalar = 0.0
            for f in serial(DFF):
                gx += GUrow[f] * Wg[d,f] + GVrow[f] * Wu[d,f]
            GX[t,d] = gx
        free(Vrow)
        free(Hrow)
        free(GHrow)
        free(GUrow)
        free(GVrow)
'''

def chain_gpu(backend):
    """Compose forward norm with the gated-FFN recipe and fused norm adjoint."""
    import ast
    assert backend=='openacc'
    norm_ast=ast.parse(rmsnorm('openmp')).body[:2]
    ffn_ast=ast.parse(ffn('openmp')).body
    class GPU(ast.NodeTransformer):
        def visit_With(self,node):
            self.generic_visit(node)
            arg=node.items[0].context_expr.args[0]
            arg.value='acc parallel loop collapse(2)' if 'collapse(2)' in arg.value else 'acc parallel loop'
            return node
        def visit_AnnAssign(self,node):
            self.generic_visit(node)
            if isinstance(node.annotation,ast.Subscript):node.annotation.value.id='scratch'
            return node
        def visit_Expr(self,node):
            if isinstance(node.value,ast.Call) and node.value.func.id=='free':return None
            return self.generic_visit(node)
        def visit_Name(self,node):
            node.id={'X':'Z','GX':'GZ_from_ffn','gx':'gz'}.get(node.id,node.id)
            return node
        def visit_Subscript(self,node):
            self.generic_visit(node)
            if isinstance(node.value,ast.Name) and node.value.id=='GZ_from_ffn':node.slice=node.slice.elts[-1]
            return node
        def visit_AugAssign(self,node):
            self.generic_visit(node)
            if isinstance(node.target,ast.Subscript) and node.target.value.id in ('GWg','GWu','GWd'):
                return ast.With(items=[ast.withitem(context_expr=ast.Call(func=ast.Name(id='directive',ctx=ast.Load()),args=[ast.Constant('acc atomic update')],keywords=[]))],body=[node])
            return node
    norm_ast=GPU().visit(ast.Module(body=norm_ast,type_ignores=[])).body
    # Restore the norm's input and scale; only its launch directives change.
    norm_spec=ast.unparse(ast.fix_missing_locations(ast.Module(body=norm_ast,type_ignores=[])))
    # The norm forward recipe uses X; GPU's FFN-input substitution applied above.
    norm_spec=norm_spec.replace('Z[t, d] / r[t]','X[t, d] / r[t]').replace('x: scalar = Z[t, d]','x: scalar = X[t, d]').replace('gamma[d]','gam[d]')
    ff=GPU().visit(ast.Module(body=ffn_ast,type_ignores=[])).body
    last=ff[-1].body[0].body
    last.insert(5,ast.parse('GZ_from_ffn: scratch[D]').body[0])
    # Allocation order is part of the profile: the local chain gradient follows rows.
    tail=ast.parse('''with scope():
    correction: scalar = 0.0
    for d in serial(D):
        x_hat_d: scalar = X[t,d] / r[t]
        correction += GZ_from_ffn[d] * gam[d] * x_hat_d
    for e in serial(D):
        x_hat_e: scalar = X[t,e] / r[t]
        GX_norm[t,e] = (GZ_from_ffn[e] * gam[e] - (x_hat_e / D) * correction) / r[t]
        with directive("acc atomic update"):
            Ggamma[e] += GZ_from_ffn[e] * x_hat_e
for d in serial(D):
    GX_ffn[t,d] = GZ_from_ffn[d]
''').body
    last.extend(tail)
    return norm_spec+'\n'+ast.unparse(ast.fix_missing_locations(ast.Module(body=ff,type_ignores=[])))+'\n'

SHAPES.update({'GX_norm':('NT','D'),'GX_ffn':('NT','D')})
