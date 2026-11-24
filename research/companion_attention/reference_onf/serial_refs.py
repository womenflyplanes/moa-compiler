"""Published benchmark serial-reference implementations (including their lifetimes)."""
import ast
import attention
SHAPES=attention.SHAPES

def strip_directives(spec):
    class Strip(ast.NodeTransformer):
        def visit_With(self,node):
            self.generic_visit(node);return node.body
    return ast.unparse(ast.fix_missing_locations(Strip().visit(ast.parse(spec))))+'\n'

def forward_ref(backend,output=True):
    assert backend=='serial'
    tree=ast.parse(strip_directives(attention.forward('openmp')))
    row=tree.body[0].body[0].body
    row.pop(0);row.pop() # allocation/free move outside all rows
    if not output:row.pop()
    return 'arow: heap[N]\n'+ast.unparse(ast.fix_missing_locations(tree))+'\nfree(arow)\n'
def forward_a(backend):return forward_ref(backend,False)

def backward_ref(backend,release=True):
    body=strip_directives(attention.value_adjoint('openmp'))+'ga_row: heap[N]\n'+'''for b in serial(B):
    for ir in serial(N):
        for j in serial(D):
            GQ[b,ir,j] = 0.0
    for ir in serial(N):
        for ic in serial(N):
            g: scalar = 0.0
            for id in serial(D):
                g += dO[b,ir,id] * V[b,ic,id]
            ga_row[ic] = g
        corr: scalar = 0.0
        for ic in serial(N):
            corr += A[b,ir,ic] * ga_row[ic]
        for ic in serial(N):
            gs: scalar = scale * A[b,ir,ic] * (ga_row[ic] - corr)
            for j in serial(D):
                GQ[b,ir,j] += gs * K[b,ic,j]
                GK[b,ic,j] += gs * Q[b,ir,j]
'''
    return body+('free(ga_row)\n' if release else '')
def backward_leak(backend):return backward_ref(backend,False)
def backward_grouped_init(backend):return backward_ref(backend)

def decode_ref(backend):
    return 's: heap[n]\ne: heap[n]\n'+strip_directives(attention.decode('openacc'))+'free(s)\nfree(e)\n'
def decode_float_ref(backend):return strip_directives(attention.decode('openmp'))
