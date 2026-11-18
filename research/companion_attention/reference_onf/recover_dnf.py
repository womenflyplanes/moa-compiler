"""Independent symbolic recovery of actual reference-mode recipe ASTs.

The execution engine consumes statements and bounded loops; it does not replace
an algorithm by an expected formula. Helpers expand from separately matched
implementations. Parallel directives affect ownership, not exact-real values.
"""
import ast,sys
from pathlib import Path
import sympy as s
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from dnf.recover import Recover,Bounds,Rule,select,Maximum,Psi
from dnf.canonical import normalize,free_symbols

class ReferenceRecover(Recover):
    def __init__(self,shapes,inputs):
        super().__init__(shapes,inputs)
        self.scalar.update({'NT':Bounds['T'],'DFF':Bounds['F'],'n':Bounds['N'],
                            'dk':Bounds['D'],'dv':Bounds['E'],
                            'EPS':s.Rational(1,1000000),'scale':1/s.sqrt(Bounds['D']),
                            'DBL_MAX':s.Integer(2)**1024-s.Integer(2)**971})
    def affine_pair(self,expr):
        variables=list(expr.atoms(s.Dummy))
        if len(variables)!=2:return None
        for large,small in (variables,variables[::-1]):
            radix=s.diff(expr,large)
            if (s.diff(expr,small)==1 and s.expand(expr-large*radix-small)==0
                and self.domains.get(small)==radix and radix.is_positive):
                return large,small,radix
        return None
    def read(self,name,indices):
        if name not in self.array:return super().read(name,indices)
        rule=self.array[name]
        if len(rule.indices)!=len(indices):raise ValueError('Rank mismatch '+name)
        bindings={}
        for formal,actual in zip(rule.indices,indices):
            if isinstance(formal,s.Symbol):
                if formal in bindings and bindings[formal]!=actual:raise ValueError('Repeated index')
                bindings[formal]=actual
            elif formal==actual:continue
            else:
                pair=self.affine_pair(formal)
                if pair is None:raise ValueError('Unsupported affine write '+name+' '+str(formal))
                large,small,radix=pair
                direct=self.affine_pair(actual)
                if direct is not None and direct[2]==radix:
                    bindings[large],bindings[small]=direct[:2]
                else:
                    bindings[large]=s.floor(actual/radix)
                    bindings[small]=actual-radix*s.floor(actual/radix)
        return rule.value.xreplace(bindings)
    def expr(self,node):
        if isinstance(node,ast.Call):
            if node.func.id=='float_cast':return self.expr(node.args[0])
            if node.func.id=='f32':return s.Rational(str(node.args[0].value))
            if node.func.id in ('expf','sqrtf'):
                return {'expf':s.exp,'sqrtf':s.sqrt}[node.func.id](self.expr(node.args[0]))
            if node.func.id=='silu':
                x=self.expr(node.args[0]);return x/(1+s.exp(-x))
            if node.func.id=='silu_grad':
                x=self.expr(node.args[0]);sigmoid=1/(1+s.exp(-x));return sigmoid*(1+x*(1-sigmoid))
        return super().expr(node)
    def statements(self,nodes):
        for node in nodes:
            if isinstance(node,ast.Return):
                self.store("__return__",None,self.expr(node.value),"assign")
            elif isinstance(node,ast.With):
                self.statements(node.body)
            elif isinstance(node,ast.AnnAssign) and node.value is None:
                continue
            elif isinstance(node,ast.Expr) and isinstance(node.value,ast.Call):
                name=node.value.func.id
                if name=='free':continue
                if name!='softmax_row':raise ValueError(ast.dump(node))
                array=node.value.args[0].id
                # Expand the checked helper's actual three loops, including its
                # finite maximum sentinel. The common shift cancels exactly.
                helper=f'''helper_m: scalar = -DBL_MAX
for helper_i in serial(N):
    if {array}[helper_i] > helper_m:
        helper_m = {array}[helper_i]
helper_s: scalar = 0.0
for helper_i in serial(N):
    {array}[helper_i] = exp({array}[helper_i] - helper_m)
    helper_s += {array}[helper_i]
for helper_i in serial(N):
    {array}[helper_i] /= helper_s
'''
                self.statements(ast.parse(helper).body)
            elif isinstance(node,ast.If):
                if node.orelse or len(node.body)!=1:raise ValueError(ast.dump(node))
                test=node.test;assignment=node.body[0]
                if (not isinstance(test,ast.Compare) or len(test.ops)!=1 or not isinstance(test.ops[0],ast.Gt)
                    or not isinstance(assignment,ast.Assign) or len(assignment.targets)!=1
                    or ast.unparse(test.comparators[0])!=ast.unparse(assignment.targets[0])
                    or ast.dump(test.left)!=ast.dump(assignment.value)):
                    raise ValueError('Unsupported conditional recurrence')
                name,indices=self.target(assignment.targets[0]);assert indices is None
                old=self.scalar[name]
                self.store(name,None,s.Max(old,self.expr(assignment.value)),'max')
            else:super().statements([node])
    def loop(self,node):
        # Base recovery requires -infinity for a maximum recurrence. Any finite
        # sentinel produces a common finite softmax shift, so retain it in the
        # bound maximum rather than silently changing the initial value.
        import dnf.recover as base
        index=s.Dummy('reference_i'+str(next(base.Fresh)),integer=True)
        extent=self.expr(node.iter.args[0]);self.domains[index]=extent
        name=node.target.id;previous=self.scalar.get(name)
        before_arrays,before_scalars=dict(self.array),dict(self.scalar)
        self.scalar[name]=index;start=len(self.events);self.statements(node.body)
        changes=self.events[start:]
        for changed in dict.fromkeys(e[0] for e in changes):
            entries=[e for e in changes if e[0]==changed];_,indices,last_mode=entries[-1]
            if indices is not None and any(index in x.free_symbols for x in indices):continue
            value=self.scalar[changed] if indices is None else self.array[changed].value
            if all(e[2]=='add' for e in entries):
                if indices is None:before=before_scalars[changed]
                elif changed in before_arrays:
                    after=self.array[changed];self.array[changed]=before_arrays[changed]
                    before=self.read(changed,after.indices);self.array[changed]=after
                else:raise ValueError('Uninitialized reduction '+changed)
                if index in before.free_symbols:raise ValueError('Loop-dependent entry '+changed)
                result=before+base.reduce_sum(s.factor(value-before),index,extent)
            elif all(e[2]=='max' for e in entries):
                result=Maximum(value,index,extent)
            else:continue
            if indices is None:self.scalar[changed]=result
            else:self.array[changed]=Rule(indices,result)
            self.events=[e for n,e in enumerate(self.events) if n<start or e[0]!=changed]
            self.events.append((changed,indices,'add' if last_mode=='add' else 'max'))
        if previous is None:self.scalar.pop(name,None)
        else:self.scalar[name]=previous

    def zero(self,name,rank):
        self.array[name]=Rule(tuple(s.Dummy('zero',integer=True) for _ in range(rank)),s.S.Zero)

def observed(rule):
    free=free_symbols(len(rule.indices))
    if len(set(rule.indices))!=len(rule.indices) or not all(isinstance(x,s.Symbol) for x in rule.indices):
        raise ValueError('Non-coordinate output tuple '+str(rule.indices))
    return free,rule.value.xreplace(dict(zip(rule.indices,free)))

def aliases(expr,names):
    return expr.replace(lambda n:n.func==Psi,lambda n:select(names.get(str(n.args[0]),str(n.args[0])),*n.args[1:]))


def observed_affine(recovered,name):
    rule=recovered.array[name];free=free_symbols(len(rule.indices))
    return free,recovered.read(name,free)

def collapse_tile_sums(expr,tile):
    """Exact finite-sum reindexing under N=NT*tile, tile>0, tile divides N."""
    from dnf.recover import Reduce
    N=Bounds['N']
    def transform(node):
        if node.func!=Reduce or s.simplify(node.args[2]*tile-N)!=0:return node
        body,outer,extent=node.args
        factors=s.Mul.make_args(body)
        candidates=[x for x in factors if x.func==Reduce and x.args[2]==tile]
        if len(candidates)!=1:return node
        inner=candidates[0];term=body.xreplace({inner:inner.args[0]});local=inner.args[1]
        k=s.Dummy('tile_flat',integer=True)
        term=term.xreplace({outer:s.floor(k/tile),local:k-tile*s.floor(k/tile)})
        return Reduce(term,k,N)
    previous=None
    while expr!=previous:
        previous=expr;expr=expr.replace(lambda n:n.func==Reduce,transform)
    return expr
