"""Negative and mutation checks for the bounded source→DNF→code path."""
import copy
import json
from pathlib import Path
import pytest
from semantic_ir import frontend, snapshot_islands
import poisson_backend as backend

BASE=Path(__file__).parent/'examples'


def input_ir(tmp_path, stem='poisson_jacobi', transform=lambda x:x):
    path=tmp_path/(stem+'.f90');path.write_text(transform((BASE/(stem+'.f90')).read_text()))
    return frontend(path)


def executable(text):
    return '\n'.join(s for s in text.splitlines() if not s.lstrip().startswith(('!','%')))


def test_case_insensitive(tmp_path):
    low=input_ir(tmp_path);up=input_ir(tmp_path,transform=str.upper)
    assert low['update']==up['update'] and low['parameters']==up['parameters']


@pytest.mark.parametrize('edit',[lambda x:x.replace('0.25 *','0.125 *'),lambda x:x.replace('+ h*h*f(i,j)','')])
def test_source_mutation_changes_dnf_and_code(tmp_path,edit):
    a=input_ir(tmp_path);b=input_ir(tmp_path,transform=edit)
    assert a['dnf']!=b['dnf']
    for prefix in ('fortran','openmp','gpu','mpi','matlab'):
        fn=getattr(backend,'gen_'+prefix+'_jacobi')
        assert executable(fn(a))!=executable(fn(b))


@pytest.mark.parametrize('source',['program empty\nend program','subroutine x\nend subroutine', (BASE/'poisson_jacobi.f90').read_text().replace('u = u_new','u = 2*u_new'),(BASE/'matmul.f90').read_text().replace('A = 1.0','A = 2.0'),(BASE/'matmul.f90').read_text().replace('C(i,j) = 0.0','C(i,j) = 2.0')])
def test_unknown_source_rejected(tmp_path,source):
    p=tmp_path/'bad.f90';p.write_text(source)
    with pytest.raises(ValueError):frontend(p)


@pytest.mark.parametrize('prefix',['fortran','openmp','gpu','mpi','matlab'])
def test_untrusted_string_rejected(prefix):
    with pytest.raises(ValueError):getattr(backend,'gen_'+prefix+'_jacobi')('arbitrary DNF')


@pytest.mark.parametrize('prefix',['openmp','gpu','mpi'])
def test_state_change_explicit(tmp_path,prefix):
    d=input_ir(tmp_path,'poisson_island_gauss');fn=getattr(backend,'gen_'+prefix+'_island')
    with pytest.raises(ValueError):fn(d)
    snap=snapshot_islands(d)
    assert snap['dnf']!=d['dnf'];assert 'snapshot_islands' in fn(snap)


@pytest.mark.parametrize('defect',['state','display','layout','shape','entry'])
def test_invalid_ir_rejected(tmp_path,defect):
    d=input_ir(tmp_path)
    if defect=='state':d['state_semantics']='anything'
    elif defect=='display':d['dnf']='zero'
    elif defect=='layout':d['layout']='row_major'
    elif defect=='shape':d['parameters']['nx']=0
    elif defect=='entry':d['entry_contract']['forcing']=2
    with pytest.raises(ValueError):backend.gen_fortran_jacobi(d)


def test_all_generators_lower_real_ir(tmp_path):
    for stem,fam in [('matmul','matmul'),('poisson_jacobi','jacobi'),('poisson_rbgs','rbgs'),('poisson_island_gauss','island')]:
        a=input_ir(tmp_path,stem)
        b=copy.deepcopy(a)
        # Numerical arithmetic mutation, not a change to display text alone.
        from semantic_ir import dnf_text
        b['update']={'op':'mul','left':{'op':'literal','value':'0.5'},'right':b['update']}
        if fam=='matmul':
            b['update']=copy.deepcopy(a['update']);b['update']['right']={'op':'mul','left':{'op':'literal','value':'0.5'},'right':a['update']['right']}
        b['dnf']=dnf_text(b)
        for prefix in ('fortran','openmp','gpu','mpi','matlab'):
            x,y=(snapshot_islands(a),snapshot_islands(b)) if fam=='island' and prefix in ('openmp','gpu','mpi') else (a,b)
            fn=getattr(backend,'gen_'+prefix+'_'+fam)
            assert executable(fn(x))!=executable(fn(y))


def test_dnf_eliminates_matmul_accumulator(tmp_path):
    from moa_stages import dnf
    d=dnf(input_ir(tmp_path,'matmul'))
    assert d['expression']['op']=='finite_sum'
    assert '"array": "c"' not in json.dumps(d['expression'])


def test_island_dnf_includes_state_dependencies(tmp_path):
    from moa_stages import dnf
    source=input_ir(tmp_path,'poisson_island_gauss')
    a=dnf(source);b=dnf(snapshot_islands(source))
    assert a['expression']!=b['expression']
    assert a['state_semantics']!=b['state_semantics']


@pytest.mark.parametrize('paradigm',['serial','matlab','openmp','openacc','mpi'])
def test_stages_bind_ir(tmp_path,paradigm):
    from moa_stages import stages,digest
    s=stages(input_ir(tmp_path),paradigm)
    assert s['onf']['dnf_sha256']==digest(s['dnf'])
    assert s['lifted_onf']['onf_sha256']==digest(s['onf'])
    assert s['lifted_onf']['physical_execution_certified'] is False


@pytest.mark.parametrize('stem',['matmul','poisson_jacobi','poisson_rbgs'])
def test_mpi_lifting_preserves_owned_tau(tmp_path,stem):
    from moa_stages import lift
    d=input_ir(tmp_path,stem)
    for owners in (1,2,3,5,7,64):
        plan=lift(d,'mpi',owners)
        assert plan['tau_owned_before']==plan['tau_owned_after']
        assert plan['ownership_cover_and_disjoint']


def test_illegal_island_lift_rejected(tmp_path):
    from moa_stages import lift
    d=snapshot_islands(input_ir(tmp_path,'poisson_island_gauss'))
    for p in (1,2,4):assert lift(d,'mpi',p)['ownership_cover_and_disjoint']
    for p in (3,5,8,65):
        with pytest.raises(ValueError):lift(d,'mpi',p)


def test_compile_exports_bound_stages(tmp_path):
    from compile_moa import compile_source
    import hashlib
    out=tmp_path/'compiled'
    c=compile_source(BASE/'matmul.f90','mpi',out,owners=3)
    for name,want in c['file_sha256'].items():assert hashlib.sha256((out/name).read_bytes()).hexdigest()==want
    assert json.loads((out/'03_lifted_onf.json').read_text())['tau_owned_after']==64*64
    assert 'nprocs /= 3' in (out/'generated.f90').read_text()


def test_compile_invalid_mapping_writes_nothing(tmp_path):
    from compile_moa import compile_source
    out=tmp_path/'bad'
    with pytest.raises(ValueError):compile_source(BASE/'poisson_island_gauss.f90','mpi',out,owners=3,snapshot=True)
    assert not out.exists()
