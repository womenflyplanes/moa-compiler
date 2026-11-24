"""Explicit DNF→ONF→logical Ω lifting for the bounded semantic source adapter.

Stages are executable checked records. Schedule templates are still trusted;
these certificates do not certify a physical GPU or a runtime's thread placement.
"""
from copy import deepcopy
import hashlib
import json
from math import prod

TARGETS=('serial','matlab','openmp','openacc','mpi')


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def dnf(data):
    from semantic_ir import validate
    validate(data)
    p=data['parameters'];family=data['family'];state=data['state_semantics']
    body=deepcopy(data['update'])
    if family=='matmul':
        # Eliminate the zero-initialized reduction accumulator from denotation.
        if body['op']!='add' or body['left']!={'op':'select','array':'c','indices':['i','j']}:
            raise ValueError('Matmul ONF must have its declared zero-initialized accumulator')
        expr={'op':'finite_sum','binder':'k','lower':1,'upper':p['n'],'body':body['right']}
        domain={'i':[1,p['n']],'j':[1,p['n']]};shapes={x:[p['n'],p['n']] for x in ('a','b','c')}
    else:
        expr=body;domain={'i':[1,p['nx']],'j':[1,p['ny']]};shapes={x:[p['nx']+2,p['ny']+2] for x in ('u','f')}
        # Explicit array-version semantics prevent illegal interchange of solvers.
        def versions(n):
            if n['op']=='select' and n['array']=='u':
                n['version']={
                    'previous_sweep':{'kind':'previous_sweep'},
                    'red_then_black':{'kind':'previous_color_phase','phase_order':[0,1]},
                    'lexicographic_islands':{'kind':'latest_predecessor','order':['iy','ix','j','i']},
                    'snapshot_islands':{'kind':'conditional','inside_current_island':'latest_inner_predecessor','outside_current_island':'previous_sweep'},
                }[state]
            for child in n.values():
                if isinstance(child,dict) and 'op' in child:versions(child)
        versions(expr)
    return {'schema':'moa.dnf.v1','family':family,'result_index':['i','j'],
            'domain_inclusive':domain,'array_shapes':shapes,'entry_contract':deepcopy(data['entry_contract']),
            'parameters':deepcopy(p),'expression':expr,'state_semantics':state,
            'sweeps':p.get('niter',1),'scope':'versioned recurrence for iterative solver; final component finite sum for matmul'}


def onf(data,target):
    if target not in TARGETS:raise ValueError('Unsupported target paradigm')
    normal=dnf(data);family=data['family'];state=data['state_semantics'];p=data['parameters']
    if family=='island':
        if target in ('openmp','openacc','mpi') and state!='snapshot_islands':raise ValueError('Parallel island mapping cannot change DNF state reads')
        if target in ('serial','matlab') and state!='lexicographic_islands':raise ValueError('This template does not implement serial snapshot islands')
    phases=[]
    if family=='matmul':
        phases=[{'operation':'map_reduce','domain':['j','i'],'parallel_axes':['j','i'],
                 'sequential_reduction':{'axis':'k','initial':'0.0','order':'increasing'},'write':'c(i,j)'}]
    elif family=='jacobi':
        phases=[{'operation':'map','domain':['j','i'],'parallel_axes':['j','i'],'reads':'u_previous','writes':'u_new','boundary':'zero, retained'},
                {'operation':'commit_sweep','after':'map_complete','implementation':'copy/swap equal logical state'}]
    elif family=='rbgs':
        phases=[{'operation':'color_map','colors':[0,1],'barrier_between_colors':True,'domain':['j','i'],
                 'parallel_axes':['j','i'],'membership':'(i+j) mod 2 = color','writes':'u_current_phase'}]
    else:
        if state=='snapshot_islands':phases.append({'operation':'snapshot','from':'u','to':'u_old','barrier_after':True})
        phases.append({'operation':'island_sweep','domain':['iy','ix'],'parallel_axes':['iy','ix'] if state=='snapshot_islands' else [],
                       'inner_sequential_axes':['j','i'],'read_versions':state,'island_shape':[p['isx'],p['isy']]})
    origin=1 if family=='matmul' else 0
    layouts={name:{'shape':shape,'origin':[origin,origin],'gamma':'(i-lower_i) + shape[0]*(j-lower_j)','order':'column_major'} for name,shape in normal['array_shapes'].items()}
    return {'schema':'moa.onf.v1','dnf_sha256':digest(normal),'dnf':normal,'target':target,
            'phases':phases,'layouts':layouts,'precision':'real32','arithmetic_tree':deepcopy(data['update']),
            'obligations':['shape and index validity','entry-state validity','ordered arithmetic preserved','state-version reads preserved','write independence for parallel domain','barriers precede dependent phases'],
            'trusted_basis':'reviewed schedule template and semantic scalar lowering; not a proof-assistant theorem'}


def lift(data,target,owners=None):
    operational=onf(data,target);p=data['parameters'];family=data['family'];parallel=target in ('openmp','openacc','mpi')
    if owners is not None and (type(owners) is not int or owners<1):raise ValueError('Owner count must be positive integer')
    if not parallel and owners not in (None,1):raise ValueError('Serial/MATLAB mapping has one logical owner')
    mapping={'schema':'moa.lifted_onf.v1','onf_sha256':digest(operational),'onf':operational,
             'target':target,'owners':owners,'physical_execution_certified':False}
    if target=='mpi':
        axis='i' if family=='matmul' else 'j';extent=p['n'] if family=='matmul' else p['ny']
        other=p['n'] if family=='matmul' else p['nx']
        if owners and owners>extent:raise ValueError('More MPI ranks than owned rows')
        if owners and family=='island' and (extent%owners or (extent//owners)%p['isy']):raise ValueError('Island slab boundaries must align with islands')
        mapping.update({'ownership_axis':axis,'ownership_rule':'start(r)=r*(L//P)+min(r,L%P); count(r)=L//P+[r<L%P]',
                        'communication':[{'kind':'broadcast','buffer':'b','payload_scalars':p['n']*p['n'],'frequency':'once before kernel'}] if family=='matmul' else [{'kind':'nearest_slab_halo','width':1,'payload_scalars':other,'frequency':'before each color phase' if family=='rbgs' else 'before each sweep','buffer':'u_old' if family=='island' else 'u'}],
                        'binding':'MPI rank equals logical slab owner','halo_storage':'extra communicated storage; not a τ-preserving reshape of owned values',
                        'local_storage':{'origin':[1,1],'shape':['count(rank)','n'],'gamma':'(i_local-1)+count(rank)*(j-1)'} if family=='matmul' else {'origin':[0,0],'shape':['nx+2','count(rank)+2'],'gamma':'i+(nx+2)*j_local'},
                        'diagnostic_collectives':'printed sample reductions are driver output, distinct from kernel halo/input communication'})
        if owners:
            intervals=[]
            for rank in range(owners):
                start=rank*(extent//owners)+min(rank,extent%owners);count=extent//owners+(rank<extent%owners)
                intervals.append({'owner':rank,'start_zero_based':start,'count':int(count),'owned_scalars':int(count*other)})
            assert intervals[0]['start_zero_based']==0
            assert all(a['start_zero_based']+a['count']==b['start_zero_based'] for a,b in zip(intervals,intervals[1:]))
            assert sum(x['owned_scalars'] for x in intervals)==extent*other
            mapping.update({'intervals':intervals,'tau_owned_before':extent*other,'tau_owned_after':sum(x['owned_scalars'] for x in intervals),'ownership_cover_and_disjoint':True})
    elif parallel:
        mapping.update({'ownership_domain':'independent islands' if family=='island' else 'independent (j,i) points within each phase',
                        'binding':'runtime-assigned; logical work domain only, no physical placement assertion',
                        'communication':'shared-address-space phase barriers; OpenACC data region around sweeps',
                        'workspace':'u_old snapshot for island; u_new for Jacobi; in-place phased u for RBGS; independent c per output for matmul'})
        sizes=[p['n'],p['n']] if family=='matmul' else [p['nx'],p['ny']]
        mapping.update({'tau_work_before':prod(sizes),'tau_work_after':prod(sizes),'bijection':'(i,j) ↔ i + extent_i*j on zero-based work indices'})
    else:mapping.update({'binding':'one owner','communication':[]})
    return mapping


def stages(data,target,owners=None):
    return {'dnf':dnf(data),'onf':onf(data,target),'lifted_onf':lift(data,target,owners)}
