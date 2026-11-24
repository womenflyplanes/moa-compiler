"""Recover and compare all selected recipe DNFs; write a closed block/chain DAG."""
import ast,json,hashlib,sys
from pathlib import Path
from recover_dnf import *
import attention,block_parts,full_block,reference_equations as equations
from dnf import hand

ROWS=[];GRAPHS={}
def views(expr):
    token_arrays={'X','Q','K','V','R1','GR1','GQ','GK','GV','GX','Z1','Z2','GZ1','GZ2','GY','AttnOut'}
    def replace(node):
        name=str(node.args[0]);ix=node.args[1:]
        if name in token_arrays:
            if len(ix)==3:return select(name,ix[0]*Bounds['N']+ix[1],ix[2])
            if len(ix)==1:return select(name,s.floor(ix[0]/Bounds['D']),s.Mod(ix[0],Bounds['D']))
        return node
    return expr.replace(lambda n:n.func==Psi,replace)

def validate_shape(recovered,output,rule,shapes):
    shape=shapes.get(output)
    if shape is None:
        if output in ('Ggamma','Ggamma1','Ggamma2'):shape=('D',)
        elif output in ('r','r1','r2'):shape=('T',)
        elif output=='out':shape=('E',)
        else:raise AssertionError('Unknown output shape '+output)
    aliases={'NT':'T','DFF':'F','n':'N','dk':'D','dv':'E'}
    expected=tuple(Bounds[aliases.get(x,x)] for x in shape)
    if output=='r' and len(expected)==2 and len(rule.indices)==1:expected=(Bounds['T'],)
    if len(expected)==3 and len(rule.indices)==2:expected=(Bounds['T'],expected[-1])
    if len(expected)>=2 and len(rule.indices)==1 and output in ('GX','GR1','R1','AttnOut','GY','X'):expected=(Bounds['T']*Bounds['D'],)
    if len(rule.indices)!=len(expected):raise AssertionError(('Output rank',output,rule.indices,expected))
    for index,extent in zip(rule.indices,expected):
        actual_extent=recovered.domains.get(index)
        if actual_extent is None and hasattr(recovered,'affine_pair'):
            pair=recovered.affine_pair(index)
            if pair is not None:actual_extent=s.simplify(recovered.domains[pair[0]]*pair[2])
        if actual_extent!=extent:raise AssertionError(('Output domain',output,index,recovered.domains.get(index),extent))

def check(label,actual,expected,free,profile,dependencies=None):
    left,right=normalize(views(actual)),normalize(views(expected))
    equal=left==right
    row={'component':label,'profile':profile,'equal':equal,'free_tuple':[str(x) for x in free],
         'recovered_dnf':str(left),'hand_dnf':str(right),
         'recovered_sha256':hashlib.sha256(s.srepr(left).encode()).hexdigest(),
         'hand_sha256':hashlib.sha256(s.srepr(right).encode()).hexdigest()}
    if dependencies is not None:row['dependencies']=dependencies
    ROWS.append(row);print(label,equal,flush=True)
    if not equal:
        Path('/tmp/recipe_dnf_failure.json').write_text(json.dumps(row,indent=2))
        raise AssertionError(label)


def standalone():
    for recipe in ('forward','backward','fused','decode','rmsnorm','ffn'):
        for backend in ('openmp','openacc') if recipe in ('forward','backward','fused','decode') else ('openmp',):
            profile=recipe+'.'+backend
            module=attention if recipe in ('forward','backward','fused','decode') else block_parts
            inputs={'forward':('Q','K','V'),'backward':('Q','K','V','dO','A'),'fused':('Q','K','V','dO'),'decode':('q','K','V'),'rmsnorm':('X','gamma','GZ'),'ffn':('X','Wg','Wu','Wd','GF')}[recipe]
            r=ReferenceRecover(module.SHAPES if recipe!='decode' else {'K':('N','D'),'V':('N','E'),'out':('E',)},inputs)
            for name,rank in [('Ggamma',1),('GWg',2),('GWu',2),('GWd',2)]:r.zero(name,rank)
            r.run(getattr(module,recipe)(backend))
            outputs={'forward':['Out','A'],'backward':['GV','GQ','GK'],'fused':['Out','GV','GQ','GK'],'decode':['out'],'rmsnorm':['Z','GX','Ggamma'],'ffn':['F','GX','GWg','GWu','GWd']}[recipe]
            for out in outputs:
                validate_shape(r,out,r.array[out],r.shapes)
                f,x=observed(r.array[out])
                if recipe in ('forward','backward','fused'):
                    x=aliases(x,{'dO':'U'})
                    if out=='A':
                        b,i,k=f
                        score=lambda key:hand.sum_at(Bounds['D'],lambda j:hand.select('Q',b,i,j)*hand.select('K',b,key,j))/s.sqrt(Bounds['D'])
                        expected=hand.probability(score,k)
                    else:
                        expected=hand.training('backward' if recipe=='backward' else 'fused',f)['O' if out=='Out' else out].xreplace({Bounds['E']:Bounds['D']})
                elif recipe=='decode':expected=hand.single_decode(f[0])
                elif recipe=='rmsnorm':
                    x=aliases(x,{'gamma':'W','GZ':'U'});expected=hand.norm(f if len(f)==2 else (*free_symbols(1),f[0]),token=True)['GW' if out=='Ggamma' else out]
                else:
                    x=aliases(x,{'GF':'U'});expected=hand.ffn(f,weight=out.startswith('GW'))['Y' if out=='F' else out]
                check(profile+'.'+out,x,expected,f,profile)


def composed(kind,backend):
    module=full_block if kind=='block' else block_parts
    spec=module.block(backend) if kind=='block' else module.chain_gpu(backend)
    inputs=set(('X','gamma1','gamma2','Wq','Wk','Wv','Wg','Wu','Wd','GY') if kind=='block' else ('X','gam','GZ','Wg','Wu','Wd','GF'))
    available=set(inputs);producers={x:'input:'+x for x in inputs};nodes=[]
    phases=[x for x in ast.parse(spec).body if isinstance(x,ast.With)]
    zeroes={5:{'GWg':2,'GWu':2,'GWd':2},6:{'Ggamma2':1},8:{'GQ':3,'GK':3},9:{'GWq':2,'GWk':2,'GWv':2},10:{'Ggamma1':1}} if kind=='block' else {3:{'GWg':2,'GWu':2,'GWd':2,'Ggamma':1}}
    for phase,node in enumerate(phases):
        r=ReferenceRecover(module.SHAPES,available)
        for name,rank in zeroes.get(phase,{}).items():r.zero(name,rank)
        r.statements([node])
        locals={x.target.id for x in ast.walk(node) if isinstance(x,ast.AnnAssign) and isinstance(x.annotation,ast.Subscript)}
        exports={name for name,indices,_ in r.events if indices is not None}-locals
        for output in sorted(exports):
            validate_shape(r,output,r.array[output],r.shapes)
            f,actual=observed(r.array[output]);expected=getattr(equations,kind)(phase,output,f)
            references={str(n.args[0]) for n in actual.atoms(s.Function) if n.func==Psi}
            dependencies={name:producers[name] for name in sorted(references)}
            check(f'{kind}.{backend}.phase{phase}.{output}',actual,expected,f,kind+'.'+backend,dependencies)
            nodes.append({'id':f'phase{phase}.{output}','array':output,'dependencies':dependencies,'equal':True})
        producers.update({output:f'phase{phase}.{output}' for output in exports});available.update(exports)
    roots=('Y','GX','Ggamma1','Ggamma2','GWq','GWk','GWv','GWg','GWu','GWd') if kind=='block' else ('Z','F','GX_norm','Ggamma','GX_ffn','GWg','GWu','GWd')
    GRAPHS[kind+'.'+backend]={'inputs':sorted(inputs),'nodes':nodes,'roots':{name:producers[name] for name in roots}}


def run(out):
    standalone()
    composed('block','openmp');composed('block','openacc');composed('chain','openacc')
    out.mkdir(parents=True,exist_ok=True)
    (out/'reference_recipe_dnfs.json').write_text(json.dumps({'arithmetic':'exact real','inputs':'actual structured reference recipes','premises':['positive integer extents','D=E for training','NT=B*N','epsilon=1/1000000','finite scores and softmax shifts','zero entry for additive gradient arrays','valid tuple views','helpers matched independently to hand C'], 'results':ROWS,'composition_graphs':GRAPHS},indent=2))
    print('PASS',len(ROWS),'recovered recipe DNF comparisons',flush=True)
if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('out',type=Path);a=p.parse_args();run(a.out)
