"""Parametric owner-index lifting proof plus exhaustive finite regressions.

Certifies work ownership. Physical slot binding is an external machine premise.
"""
import ast,json,math,itertools,re
from pathlib import Path
from check import PROFILES
from emit import Emitter
from compare import parse,normalize
from pycparser import c_ast,c_generator
GEN=c_generator.CGenerator()
import attention,block_parts,full_block

FIELDS=('U_d','W_d','M_d','t_d','nmax_d','Mcap_d','P_d','X_d','C_d','R_d')

def coordinate(ordinal,shape):
    values=[]
    for n in reversed(shape):values.append(ordinal%n);ordinal//=n
    assert ordinal==0
    return tuple(reversed(values))
def gamma(coord,shape):
    if len(coord)!=len(shape) or any(not 0<=v<n for v,n in zip(coord,shape)):raise ValueError('invalid machine coordinate')
    result=0
    for v,n in zip(coord,shape):result=result*n+v
    return result

def certificate(L,shape):
    if L<=0 or not shape or any(type(n)!=int or n<=0 for n in shape):raise ValueError('positive extents required')
    P=math.prod(shape);seen=[]
    for r in range(P):
        c=coordinate(r,shape);assert gamma(c,shape)==r
        begin,end=L*r//P,L*(r+1)//P
        for j in range(end-begin):
            i=begin+j;inverse=(P*(i+1)+L-1)//L-1
            assert inverse==r and i-L*inverse//P==j
            seen.append(i)
    assert seen==list(range(L))
    return {'work_count':L,'rho_machine':shape,'tau_work_preserved':len(seen)==L,'rectangular_reshape':L%P==0,
            'rectangular_shape':[*shape,L//P] if L%P==0 else None}

def schedules(spec):
    result=[]
    for node in ast.parse(spec).body:
        if not isinstance(node,ast.With):continue
        call=node.items[0].context_expr
        if call.func.id!='directive':continue
        pragma=call.args[0].value
        if not ('parallel for' in pragma or 'parallel loop' in pragma):continue
        collapsed=re.search(r'collapse\((\d+)\)',pragma);rank=int(collapsed.group(1)) if collapsed else 1
        loops=[];current=node.body
        for _ in range(rank):
            if len(current)!=1 or not isinstance(current[0],ast.For):raise ValueError('Nonperfect collapsed nest')
            loop=current[0];loops.append({'index':loop.target.id,'extent':ast.unparse(loop.iter.args[0])});current=loop.body
        result.append({'directive':pragma,'owner_axes':loops,'logical_owner_count':' * '.join(x['extent'] for x in loops)})
    return result

def c_domains(body):
    root=parse(body);items=root.block_items;phases=[]
    for pos,node in enumerate(items):
        if not isinstance(node,c_ast.Pragma) or not ('parallel for' in node.string or 'parallel loop' in node.string):continue
        m=re.search(r'collapse\((\d+)\)',node.string);rank=int(m.group(1)) if m else 1
        loop=items[pos+1];axes=[]
        for depth in range(rank):
            if not isinstance(loop,c_ast.For):raise ValueError('Invalid collapsed owner nest')
            axes.append({'index':loop.init.decls[0].name,'extent':GEN.visit(loop.cond.right)})
            if depth<rank-1:loop=loop.stmt.block_items[0] if isinstance(loop.stmt,c_ast.Compound) else loop.stmt
        phases.append({'directive':re.sub(r'\s+','',node.string),'owner_axes':axes})
    return phases

def run(out,programs):
    cases=[certificate(L,shape) for L in range(1,38) for shape in ((1,),(2,),(8,),(2,3),(2,4,2),(64,),(108,))]
    profiles=[]
    for name,_,_,recipe,backend in PROFILES:
        module=full_block if recipe=='block' else block_parts if recipe in ('rmsnorm','ffn','chain_gpu') else attention
        spec=getattr(module,recipe)(backend)
        hand=(programs/(name+'.hand.c')).read_text();compiled=(programs/(name+'.compiled.c')).read_text()
        hand_domains,compiled_domains=c_domains(hand),c_domains(compiled)
        assert normalize(parse(hand))==normalize(parse(compiled)) and hand_domains==compiled_domains
        profiles.append({'profile':name,'phases':schedules(spec),'hand_owner_domains':hand_domains,'compiled_owner_domains':compiled_domains,'domains_and_operational_trees_equal':True,
          'fixed_machine_correspondence':'For every common admissible (machine shape, slot binding, runtime schedule choice), applying the same index-lifting relation to these equal operational domains yields equal lifted ONFs. No equality of independent nondeterministic runtime choices is asserted.'})
    out.mkdir(parents=True,exist_ok=True)
    value={'proof':{'premises':'L>0, machine shape has positive integer extents; same selected owner domain and same slot binding',
      'forward':'i = floor(L*r/P)+j; 0<=j<floor(L*(r+1)/P)-floor(L*r/P)',
      'inverse':'r = ceil(P*(i+1)/L)-1; j=i-floor(L*r/P)',
      'coverage':'for 0<=i<L the inverse gives 0<=r<P and a unique legal j',
      'preservation':'sum_r (floor(L*(r+1)/P)-floor(L*r/P)) = L',
      'rectangular_case':'if P divides L, rho(lifted work)=rho(machine) concatenated with <L/P>, tau unchanged',
      'nondivisible_case':'segmented legal-index relation; no assertion of an unpadded rectangular reshape',
      'scope':'algorithmic work-index bijection; reductions may have multiple output contributors; no measured physical binding'},
      'machine_property_array_order':FIELDS,'physical_property_values':{x:None for x in FIELDS},
      'finite_regressions':cases,'selected_profiles':profiles,
      'physical_slot_binding':'not available; symbolic coordinates must be bound to recorded CPU affinity or GPU launch geometry before a physical placement claim'}
    (out/'omega_lift_certificates.json').write_text(json.dumps(value,indent=2))
    print('PASS',len(cases),'lift regressions;',len(profiles),'profile domains extracted')
if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('out',type=Path);p.add_argument('--programs',type=Path,required=True);a=p.parse_args();run(a.out,a.programs)
