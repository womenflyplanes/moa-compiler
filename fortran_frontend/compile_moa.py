#!/usr/bin/env python3
"""Compile a supported source contract and export explicit MoA stage records."""
import argparse
import hashlib
import json
from pathlib import Path
from semantic_ir import frontend, snapshot_islands
from moa_stages import stages, digest
import poisson_backend as backend


def compile_source(source,target,out,owners=None,snapshot=False):
    data=frontend(source)
    if snapshot:data=snapshot_islands(data)
    record=stages(data,target,owners)
    if target=='mpi' and owners is None:raise ValueError('MPI compilation requires an explicit logical rank count')
    prefix={'serial':'fortran','matlab':'matlab','openmp':'openmp','openacc':'gpu','mpi':'mpi'}[target]
    code=getattr(backend,'gen_'+prefix+'_'+data['family'])(data)
    if target=='mpi':
        code=code.replace('  call MPI_Comm_size(MPI_COMM_WORLD, nprocs, ierr)',
                          '  call MPI_Comm_size(MPI_COMM_WORLD, nprocs, ierr)\n'+
                          f"  if (nprocs /= {owners}) then\n    if (rank==0) print *, 'MPI rank count differs from lifted ONF contract'\n    call MPI_Abort(MPI_COMM_WORLD, 3, ierr)\n  end if")
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    files={name:json.dumps(record[name],indent=2,ensure_ascii=False)+'\n' for name in ('dnf','onf','lifted_onf')}
    for index,name in enumerate(files,1):(out/f'{index:02d}_{name}.json').write_text(files[name])
    filename='generated.m' if target=='matlab' else 'generated.f90'
    (out/filename).write_text(code)
    certificate={'schema':'moa.compilation_certificate.v1','source_sha256':data['source_sha256'],
                 'source_contract':data,'target':target,'owners':owners,'physical_execution_certified':False,
                 'stage_sha256':{k:digest(v) for k,v in record.items()},
                 'file_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [out/f'{i:02d}_{n}.json' for i,n in enumerate(files,1)]+[out/filename]},
                 'lowering_modules_sha256':{name:hashlib.sha256((Path(__file__).parent/name).read_bytes()).hexdigest() for name in ('semantic_ir.py','moa_stages.py','lower_semantic_ir.py','poisson_backend.py','compile_moa.py')},
                 'proof_boundary':'bounded source adapter, structured versioned DNF, checked schedule selection and logical ownership; trusted schedule templates; separate numerical and hardware validation required'}
    (out/'certificate.json').write_text(json.dumps(certificate,indent=2,ensure_ascii=False)+'\n')
    return certificate


def main():
    p=argparse.ArgumentParser();p.add_argument('source');p.add_argument('--target',choices=('serial','matlab','openmp','openacc','mpi'),required=True)
    p.add_argument('--owners',type=int);p.add_argument('--snapshot-islands',action='store_true');p.add_argument('--out',required=True);args=p.parse_args()
    c=compile_source(args.source,args.target,args.out,args.owners,args.snapshot_islands)
    print('Generated',args.out,'target='+c['target'],'physical execution: not certified')

if __name__=='__main__':main()
