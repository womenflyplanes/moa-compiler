"""Additional published variants, independently constructed from component recipes."""
import ast
import attention,block_parts,full_block
SHAPES=dict(full_block.SHAPES)
SHAPES.update(attention.SHAPES)
SHAPES.update({'r':('B','N'),'GZ':('B','N','D'),'Z':('B','N','D')})

def render(tree):return ast.unparse(ast.fix_missing_locations(tree))+'\n'

def rms_verify(backend):
    assert backend=='openmp'
    return '''with directive("omp parallel for collapse(2)"):
    for b in serial(B):
        for n in serial(N):
            sumsq: scalar = 0.0
            for d in serial(D):
                x: scalar = X[b,n,d]
                sumsq += x * x
            r[b,n] = sqrt(sumsq / D + EPS)
with directive("omp parallel for collapse(3)"):
    for b in serial(B):
        for n in serial(N):
            for d in serial(D):
                x: scalar = X[b,n,d]
                Z[b,n,d] = gamma[d] * (x / r[b,n])
with directive("omp parallel for collapse(2)"):
    for b in serial(B):
        for n in serial(N):
            rr: scalar = r[b,n]
            correction: scalar = 0.0
            for d in serial(D):
                x_hat_d: scalar = X[b,n,d] / rr
                correction += GZ[b,n,d] * gamma[d] * x_hat_d
            for e in serial(D):
                x_hat_e: scalar = X[b,n,e] / rr
                GX[b,n,e] = (GZ[b,n,e] * gamma[e] - (x_hat_e / D) * correction) / rr
with directive("omp parallel for collapse(2) reduction(+:Ggamma[0:D])"):
    for b in serial(B):
        for n in serial(N):
            rr: scalar = r[b,n]
            for d in serial(D):
                x_hat_d: scalar = X[b,n,d] / rr
                Ggamma[d] += GZ[b,n,d] * x_hat_d
'''

def ffn_verify(backend):
    tree=ast.parse(block_parts.ffn(backend))
    class Stack(ast.NodeTransformer):
        def visit_AnnAssign(self,node):
            self.generic_visit(node)
            if isinstance(node.annotation,ast.Subscript):node.annotation.value.id='scratch'
            return node
        def visit_Expr(self,node):
            if isinstance(node.value,ast.Call) and node.value.func.id=='free':return None
            return self.generic_visit(node)
        def visit_Name(self,node):
            if node.id=='fo':node.id='f_out'
            return node
    tree=Stack().visit(tree)
    body=tree.body[1].body[0].body
    # The verification variant separates GH from the GU/GV component pass.
    ghloop=next(n for n in body if isinstance(n,ast.For) and any(isinstance(x,ast.AnnAssign) and x.target.id=='gh' for x in n.body))
    tail=ghloop.body[-2:];ghloop.body=ghloop.body[:-2]
    tail=ast.parse('''for f in serial(DFF):
    GVrow[f] = GHrow[f] * silu(U[t,f])
    GUrow[f] = GHrow[f] * Vrow[f] * silu_grad(U[t,f])
''').body[0]
    body.insert(body.index(ghloop)+1,tail)
    return render(tree)

def block_verify(backend):
    text=full_block.block(backend)
    if backend=='openacc':
        return text.replace('scratch[MAX_N]','scratch[N]').replace('scratch[512]','scratch[DFF]').replace('with directive("acc parallel loop"):\n    for b in serial(B):\n        for ir in serial(N):','with directive("acc parallel loop collapse(2)"):\n    for b in serial(B):\n        for ir in serial(N):',1)
    # The fixed CPU verification uses stack row scratch instead of heap rows.
    tree=ast.parse(text)
    class Stack(ast.NodeTransformer):
        def visit_AnnAssign(self,node):
            self.generic_visit(node)
            if isinstance(node.annotation,ast.Subscript) and node.annotation.value.id=='heap':node.annotation.value.id='scratch'
            return node
        def visit_Expr(self,node):
            if isinstance(node.value,ast.Call) and node.value.func.id=='free':return None
            return self.generic_visit(node)
    return render(Stack().visit(tree))

def chain_bench(backend):
    tree=ast.parse(block_parts.chain_gpu(backend))
    class Adapt(ast.NodeTransformer):
        def visit_Name(self,node):
            if node.id=='GZ_from_ffn':node.id='GZf'
            return node
        def visit_AnnAssign(self,node):
            self.generic_visit(node)
            if isinstance(node.annotation,ast.Subscript):node.annotation.slice=ast.Constant(512)
            return node
    tree=Adapt().visit(tree)
    body=tree.body[-1].body[0].body
    # Move the separate GX_ffn store into the local-gradient loop.
    last=body.pop();gxloop=next(n for n in body if isinstance(n,ast.For) and any(isinstance(x,ast.AnnAssign) and x.target.id=='gz' for x in n.body))
    gxloop.body.append(ast.parse('GX_ffn[t,d] = gz').body[0])
    scope=body.pop();assert isinstance(scope,ast.With)
    body.extend(scope.body)
    return render(tree)

def fused_atomic(backend):
    assert backend=='openacc'
    tail='''with directive("acc parallel loop collapse(2)"):
    for b in serial(B):
        for ir in serial(N):
            for j in serial(D):
                GQ[b,ir,j] = 0.0
            for ic in serial(N):
                gs: scalar = GS[b,ir,ic]
                for j in serial(D):
                    GQ[b,ir,j] += gs * K[b,ic,j]
                    with directive("acc atomic update"):
                        GK[b,ic,j] += gs * Q[b,ir,j]
'''
    return attention.forward(backend,'AROW')+attention.value_adjoint(backend,'AROW')+attention.score_adjoint(backend,'AROW')+tail

def block_serial(backend):
    assert backend=='serial'
    tree=ast.parse(full_block.block('openmp'))
    class SerialStack(ast.NodeTransformer):
        def visit_With(self,node):
            self.generic_visit(node)
            return node.body if node.items[0].context_expr.func.id=='directive' else node
        def visit_AnnAssign(self,node):
            self.generic_visit(node)
            if isinstance(node.annotation,ast.Subscript) and node.annotation.value.id=='heap':node.annotation.value.id='scratch'
            return node
        def visit_Expr(self,node):
            if isinstance(node.value,ast.Call) and node.value.func.id=='free':return None
            return self.generic_visit(node)
    base=SerialStack().visit(tree).body
    attention_phase=base[2]
    row=attention_phase.body[0].body
    for node in ast.walk(attention_phase):
        if isinstance(node,ast.AnnAssign) and node.target.id=='m':node.value=ast.parse('-DBL_MAX',mode='eval').body
        if isinstance(node,ast.Assign) and isinstance(node.targets[0],ast.Subscript) and node.targets[0].value.id=='R1':
            node.targets[0].value.id='AttnOut';node.value=ast.Name(id='o',ctx=ast.Load())
    ff_forward=base[4];ff_out=ff_forward.body.pop()
    ff_out_phase=ast.parse('for t in serial(NT):\n    pass').body[0];ff_out_phase.body=[ff_out]
    ff_backward=base[5]
    ff_backward.body.insert(0,ast.parse('GHrow: scratch[DFF]').body[0])
    for node in ast.walk(ff_backward):
        if isinstance(node,ast.For) and node.target.id=='f' and any(isinstance(n,ast.AnnAssign) and n.target.id=='gh' for n in node.body):
            pos=next(i for i,n in enumerate(node.body) if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Subscript) and n.targets[0].value.id=='GVrow')
            node.body.insert(pos,ast.parse('GHrow[f] = gh').body[0])
    # Remove the old outside declaration; the fixed verifier declares gz in d.
    ff_backward.body=[n for n in ff_backward.body if not (isinstance(n,ast.AnnAssign) and n.target.id=='gz2')]
    ff_backward.body.pop()  # old GR1=GY feature loop
    class Gz(ast.NodeTransformer):
        def visit_Name(self,node):
            if node.id=='gz2':node.id='gz'
            return node
        def visit_Assign(self,node):
            self.generic_visit(node)
            if len(node.targets)==1 and isinstance(node.targets[0],ast.Name) and node.targets[0].id=='gz' and isinstance(node.value,ast.Constant) and node.value.value==0.0:
                return ast.AnnAssign(target=node.targets[0],annotation=ast.Name(id='scalar',ctx=ast.Load()),value=node.value,simple=1)
            return node
    ff_backward=Gz().visit(ff_backward)
    seed_gx=ast.parse('for i in serial(NT*D):\n    GX[i] += GR1[i]').body[0]
    phases=base[:3]+ast.parse('for i in serial(NT*D):\n    R1[i] = X[i] + AttnOut[i]').body+[
        base[3],ff_forward,ff_out_phase]+ast.parse('for i in serial(NT*D):\n    GR1[i] += GY[i]').body+[
        ff_backward,base[6],seed_gx,*base[8:]]
    return render(ast.Module(body=phases,type_ignores=[]))
SHAPES['GX']=('B','N','D')
SHAPES['AttnOut']=('B','N','D')
