"""Independent finite audit of Gate 10 Ω ownership and MPI collective metadata."""
import json, re
from pathlib import Path
from math import prod
from moa_mpi import compile_mpi

CASES=[(4,4,8,8,(2,3)),(1,3,5,7,(2,3)),(7,5,3,2,(3,2)),(9,2,4,3,(2,2,2)),(3,6,2,1,(1,)),(11,7,8,4,(3,3))]

def parse_c_array(src, name):
    match=re.search(r'const (?:int|size_t) '+name+r'\[\] = \{([^}]*)\};',src)
    if not match: raise AssertionError('missing generated C array '+name)
    return [int(x.strip()) for x in match.group(1).split(',')]

def audit(folder,n,m,d,dv,shape):
    p=prod(shape)
    src=(folder/'generated_attention_mpi.c').read_text()
    cert=json.loads((folder/'mpi_ownership_certificate.json').read_text())
    begin=parse_c_array(src,'owner_begin');end=parse_c_array(src,'owner_end')
    counts=parse_c_array(src,'counts');displs=parse_c_array(src,'displs')
    assert len(begin)==len(end)==len(counts)==len(displs)==p
    # Independent ψ-index enumeration: compute rank from integer division of
    # each query component, without calling omega_lift/derive_collective.
    expected_owners=[[] for _ in range(p)]
    for i in range(n):
        rank=((i+1)*p-1)//n
        expected_owners[rank].append(i)
    expected_begin=[(n*r)//p for r in range(p)]
    expected_end=[(n*(r+1))//p for r in range(p)]
    assert begin==expected_begin and end==expected_end
    assert expected_owners==[list(range(begin[r],end[r])) for r in range(p)]
    coverage=[0]*(n*dv)
    for r in range(p):
        for i in range(begin[r],end[r]):
            for v in range(dv): coverage[i*dv+v]+=1
    assert all(x==1 for x in coverage), 'output component missing or duplicated'
    expected_counts=[len(owners)*dv for owners in expected_owners]
    expected_displs=[];cursor=0
    for count in expected_counts:
        expected_displs.append(cursor);cursor+=count
    assert counts==expected_counts and displs==expected_displs
    assert cursor==n*dv
    collective=cert['collective']
    assert collective['counts']==counts and collective['displacements']==displs
    assert collective['output_components']==n*dv
    assert collective['per_rank_receive_components']==n*dv
    assert collective['aggregate_logical_receive_components']==p*n*dv
    assert collective['aggregate_nonlocal_required_components']==(p-1)*n*dv
    assert cert['rho_machine']==list(shape) and cert['tau_machine']==p
    assert cert['rho_output']==[n,dv] and cert['tau_output']==n*dv
    assert [c['query_begin'] for c in cert['components']]==begin
    assert [c['query_end'] for c in cert['components']]==end
    assert [c['output_components'] for c in cert['components']]==counts
    return {'rho_machine':list(shape),'rho_output':[n,dv], 'mpi_ranks':p,
            'counts':counts,'displacements':displs,'empty_owners':sum(a==b for a,b in zip(begin,end)),
            'tau_output':n*dv,'aggregate_logical_receive_components':p*n*dv,
            'aggregate_nonlocal_required_components':(p-1)*n*dv,
            'aggregate_nonlocal_required_bytes':(p-1)*n*dv*8,
            'result':'PASS'}

def main():
    root=Path(__file__).resolve().parent
    build=root/'gate11_audit_build';build.mkdir(exist_ok=True)
    results=[]
    for idx,(n,m,d,dv,shape) in enumerate(CASES):
        folder=build/f'case{idx}';folder.mkdir(exist_ok=True)
        compile_mpi(folder,n,m,d,dv,shape)
        result=audit(folder,n,m,d,dv,shape)
        results.append(result)
        print(f'Gate11 case{idx} PASS: rho_machine={shape} ranks={result["mpi_ranks"]} counts={result["counts"]}')
    # Mutation test: audit must reject tampered collective displacement.
    folder=build/'case0';source=folder/'generated_attention_mpi.c'
    original=source.read_text()
    try:
        source.write_text(original.replace('const int displs[] = {','const int displs[] = {1,',1))
        try: audit(folder,*CASES[0])
        except (AssertionError,ValueError): pass
        else: raise AssertionError('audit failed to reject mutated displacement')
    finally: source.write_text(original)
    report={'gate':'11','scope':'static finite Ω ownership and logical MPI collective audit; not network traffic or performance',
            'cases':results,'mutation_test':'PASS: altered C displacements rejected',
            'gate10_real_mpi_evidence':'User-reported six cases passed on dgx08, 25 trials each; not rerun by this script'}
    (root/'GATE11_AUDIT_RESULTS.json').write_text(json.dumps(report,indent=2)+'\n')
    print('Gate11 PASS: 6 static audits and 1 mutation rejection')
if __name__=='__main__':main()
