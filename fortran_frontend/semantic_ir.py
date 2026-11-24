"""Fail-closed semantic subset for the four published Fortran examples.

This is a bounded source adapter, not a general Fortran parser. It admits the
checked loop/declaration skeletons, typed scalar parameters and arithmetic ASTs.
Iteration state is part of the DNF contract, not an interchangeable schedule.
"""
import ast
import copy
import hashlib
import json
import math
import numpy as np
import re
from pathlib import Path

SCHEMA = 'moa.fortran.subset.v1'
# Filled from the reviewed source skeletons; executable statements cannot be ignored.
SKELETONS = {'32aca1654e575db025cc0c55ffafb4da55dc0ab25ecbd3693f949a059dfea613': 'island', '3230b4f28da2128a4e331a069d09a73ed043e90499d9a74ef7228629326f7fee': 'rbgs', 'cd22480c8d8355f53abc3d7352cf8039c2febb784c828ee5c3e36aa6428ca5f7': 'matmul', 'ad99387dce6f4c41506cdd2542e866bad6c7f830cfcac419c120c8bc3700213d': 'jacobi'}


def clean(text):
    return '\n'.join(line.split('!')[0].strip().lower() for line in text.splitlines()
                     if line.split('!')[0].strip())


def skeleton(text):
    s = clean(text)
    s = re.sub(r'^(end\s+)?program\s+\w+', lambda m: (m[1] or '')+'program PROGRAM', s, flags=re.M)
    s = re.sub(r'^(\s*(?:u_new\(i,j\)|u\(i,j\)|c\(i,j\))\s*=)(.+)$', lambda m:m[1]+'EXPR' if ('u(' in m[2] or 'a(' in m[2]) else m[0],s,flags=re.M)
    s = '\n'.join(re.sub(r'(?<![\w])\d+(?:\.\d*)?(?:[ed][+-]?\d+)?', 'NUM', line) if 'parameter' in line else line for line in s.splitlines())
    return re.sub(r'\s+', '', s)


def parameter_values(src, family):
    result = {}
    for line in src.splitlines():
        if 'parameter' not in line: continue
        is_real=line.lstrip().startswith('real')
        cast=np.float32 if is_real else int
        for name, value in re.findall(r'(\w+)\s*=\s*([^,]+)', line.split('::',1)[1]):
            tree = ast.parse(value.strip(), mode='eval')
            def ev(n):
                if isinstance(n,ast.Constant) and type(n.value) in (int,float): return cast(n.value)
                if isinstance(n,ast.Name) and n.id in result: return cast(result[n.id])
                if isinstance(n,ast.BinOp) and isinstance(n.op,(ast.Add,ast.Sub,ast.Mult,ast.Div)):
                    a,b=ev(n.left),ev(n.right)
                    return cast({ast.Add:lambda:a+b,ast.Sub:lambda:a-b,ast.Mult:lambda:a*b,ast.Div:lambda:a/b}[type(n.op)]())
                raise ValueError('Unsupported scalar parameter expression')
            result[name]=float(ev(tree.body)) if is_real else int(ev(tree.body))
    names=('n',) if family=='matmul' else ('nx','ny','niter')
    for name in names:
        v=result.get(name)
        if type(v) not in (int,float) or not math.isfinite(v) or int(v)!=v or v<1:
            raise ValueError('Invalid positive integer extent/iteration '+name)
        result[name]=int(v)
    # The public fixture drivers print component 32; this subset keeps that contract.
    if min(result[n] for n in (('n',) if family=='matmul' else ('nx','ny')))<32:
        raise ValueError('Fixture driver requires each extent >=32')
    if family!='matmul' and (not math.isfinite(result.get('h',float('nan'))) or result['h']<=0):
        raise ValueError('Expected positive finite h')
    if family=='island':
        for a,n in [('isx','nx'),('isy','ny')]:
            v=result.get(a)
            if not isinstance(v,(int,float)) or int(v)!=v or v<=0 or result[n]%v:
                raise ValueError('Island extent must divide its array extent')
            result[a]=int(v)
    return result


def expression_node(node, family):
    if isinstance(node,ast.Constant) and type(node.value) is float and math.isfinite(node.value):
        return {'op':'literal','value':repr(node.value)}
    if isinstance(node,ast.Name) and node.id=='h' and family!='matmul':
        return {'op':'scalar','name':'h'}
    if isinstance(node,ast.UnaryOp) and isinstance(node.op,(ast.USub,ast.UAdd)):
        return {'op':'neg' if isinstance(node.op,ast.USub) else 'pos','arg':expression_node(node.operand,family)}
    if isinstance(node,ast.BinOp) and type(node.op) in (ast.Add,ast.Sub,ast.Mult,ast.Div):
        return {'op':{ast.Add:'add',ast.Sub:'sub',ast.Mult:'mul',ast.Div:'div'}[type(node.op)],
                'left':expression_node(node.left,family),'right':expression_node(node.right,family)}
    if isinstance(node,ast.Call) and isinstance(node.func,ast.Name) and not node.keywords:
        name=node.func.id
        args=[ast.unparse(x).replace(' ','') for x in node.args]
        allowed={'a':[['i','k']],'b':[['k','j']],'c':[['i','j']]} if family=='matmul' else {'u':[['i-1','j'],['i+1','j'],['i','j-1'],['i','j+1']],'f':[['i','j']]}
        if name in allowed and args in allowed[name]:
            return {'op':'select','array':name,'indices':args}
    raise ValueError('Unsupported expression; cannot silently approximate source semantics')


def selections(n):
    if n['op']=='select': return [(n['array'],n['indices'])]
    return sum((selections(v) for v in n.values() if isinstance(v,dict)),[])


def render(n, language='fortran', island=False, mpi=False):
    op=n['op']
    if op=='literal': return str(n['value'])
    if op=='scalar': return n['name']
    if op=='select':
        a=n['array'];idx=n['indices']
        if island and a=='u':
            return {('i-1','j'):'ul',('i+1','j'):'ur',('i','j-1'):'ud',('i','j+1'):'uu'}[tuple(idx)]
        if mpi and a=='f' and island: return 'f(i,j_local)'
        if language=='matlab' and a in ('u','f'):
            # Existing MATLAB loops already shift the base index to 2..N+1.
            return a+'('+','.join(idx)+')'
        return a+'('+','.join(idx)+')'
    if op in ('neg','pos'): return '('+('-' if op=='neg' else '+')+render(n['arg'],language,island,mpi)+')'
    return '('+render(n['left'],language,island,mpi)+{'add':'+','sub':'-','mul':'*','div':'/'}[op]+render(n['right'],language,island,mpi)+')'


def frontend(path):
    src=clean(Path(path).read_text())
    digest=hashlib.sha256(skeleton(src).encode()).hexdigest()
    family=SKELETONS.get(digest)
    if family is None: raise ValueError('Unsupported Fortran statement/loop/declaration skeleton')
    target='c' if family=='matmul' else 'u_new' if family=='jacobi' else 'u'
    matches=re.findall(r'^'+target+r'\(i,j\)\s*=\s*(.+)$',src,re.M)
    # Matmul resets C before the reduction; only the recurrent assignment is the update.
    rhs=next((s for s in matches if 'c(i,j)' in s),None) if family=='matmul' else matches[-1] if matches else None
    if rhs is None: raise ValueError('Missing numerical update')
    expr=expression_node(ast.parse(rhs,mode='eval').body,family)
    sels=selections(expr)
    if family=='matmul':
        if expr['op']!='add' or expr['left']!={'op':'select','array':'c','indices':['i','j']} or selections(expr['right'])!=[('a',['i','k']),('b',['k','j'])]:
            raise ValueError('Matmul must accumulate a product term once per k')
    else:
        expected=[('u',['i-1','j']),('u',['i+1','j']),('u',['i','j-1']),('u',['i','j+1'])]
        if [s for s in sels if s[0]=='u']!=expected: raise ValueError('Expected four axial neighbours in source order')
    params=parameter_values(src,family)
    state={'jacobi':'previous_sweep','island':'lexicographic_islands','rbgs':'red_then_black','matmul':'ordered_k_reduction'}[family]
    data={'schema':SCHEMA,'source':str(path),'source_sha256':hashlib.sha256(Path(path).read_bytes()).hexdigest(),
          'family':family,'parameters':params,'update':expr,'state_semantics':state,
          'entry_contract':{'precision':'real32','initial_u':0,'forcing':1,'boundaries':0} if family!='matmul' else {'precision':'real32','initial_a':1,'initial_b':1,'initial_c':0},
          'layout':'column_major','index_origin':0 if family!='matmul' else 1}
    data['dnf']=dnf_text(data)
    validate(data)
    return data


def dnf_text(data):
    # Explicit versioned state accompanies the scalar ψ expression.
    def moa(n):
        if n['op']=='select': return '⟨'+','.join(n['indices'])+'⟩ ψ '+n['array']
        if n['op'] in ('literal','scalar'):return str(n.get('value',n.get('name')))
        if n['op'] in ('neg','pos'):return ('-' if n['op']=='neg' else '+')+'('+moa(n['arg'])+')'
        return '('+moa(n['left'])+{'add':'+','sub':'-','mul':'*','div':'/'}[n['op']]+moa(n['right'])+')'
    return '⟨i,j⟩ ψ '+('c' if data['family']=='matmul' else 'u_next')+' = '+moa(data['update'])+'; state='+data['state_semantics']


def validate(data):
    if not isinstance(data,dict) or data.get('schema')!=SCHEMA:raise ValueError('Backend requires structured semantic IR, not an unvalidated DNF string')
    if data.get('family') not in ('jacobi','island','rbgs','matmul'): raise ValueError('Unknown algorithm family')
    if data.get('layout')!='column_major':raise ValueError('Unsupported layout')
    states={'jacobi':('previous_sweep',),'island':('lexicographic_islands','snapshot_islands'),'rbgs':('red_then_black',),'matmul':('ordered_k_reduction',)}
    if data.get('state_semantics') not in states[data['family']]:raise ValueError('Unsupported state semantics')
    family=data['family'];params=data.get('parameters',{})
    required=('n',) if family=='matmul' else ('nx','ny','niter','h')
    if any(k not in params for k in required):raise ValueError('Missing shape parameters')
    allowed=set(required)|({'isx','isy','nix','niy'} if family=='island' else set())
    if set(params)!=allowed:raise ValueError('Unsupported or missing parameters')
    for k,v in params.items():
        if type(v) not in (int,float) or not math.isfinite(v) or v<=0 or (k!='h' and int(v)!=v):raise ValueError('Invalid parameter '+k)
    if min(params[k] for k in (('n',) if family=='matmul' else ('nx','ny')))<32:raise ValueError('Fixture drivers require extents >=32')
    if family=='island' and (params['nx']%params['isx'] or params['ny']%params['isy'] or params['nix']!=params['nx']/params['isx'] or params['niy']!=params['ny']/params['isy']):raise ValueError('Invalid island tiling')
    entry={'precision':'real32','initial_u':0,'forcing':1,'boundaries':0} if family!='matmul' else {'precision':'real32','initial_a':1,'initial_b':1,'initial_c':0}
    if data.get('entry_contract')!=entry:raise ValueError('Unsupported entry contract')
    if data.get('index_origin')!=(1 if family=='matmul' else 0):raise ValueError('Unsupported index origin')

    if data['dnf']!=dnf_text(data):raise ValueError('DNF display and semantic expression disagree')
    # Reparse the emitted scalar expression to validate node/identifier vocabulary.
    checked=expression_node(ast.parse(render(data['update']),mode='eval').body,data['family'])
    if checked!=data['update']:raise ValueError('Invalid arithmetic AST')
    return data


def snapshot_islands(data):
    validate(data)
    if data['family']!='island':raise ValueError('Snapshot transform requires island source')
    new=copy.deepcopy(data);new['state_semantics']='snapshot_islands';new['dnf']=dnf_text(new)
    new['transformation_note']='Explicit change of finite-sweep algorithm; not equivalent to lexicographic island sweeps'
    return new
