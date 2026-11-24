"""Untimed full-field backend checks. OpenACC uses explicit host fallback.

This harness adds diagnostic output to generated programs without changing their
numerical updates. It covers uneven MPI slabs and nonsquare stencil shapes.
"""
import copy
import json
import os
from pathlib import Path
import subprocess
import tempfile
import numpy as np
from semantic_ir import frontend, snapshot_islands
import poisson_backend as be


def value(node,u,f,h,i,j,a=None,b=None,c=None,k=None):
    op=node['op']
    if op=='literal':return np.float32(node['value'])
    if op=='scalar':return h
    if op=='select':
        arrays={'u':u,'f':f,'a':a,'b':b,'c':c};inds=[]
        for text in node['indices']:
            if text=='k':inds.append(k)
            elif text.startswith('i'):inds.append(i+(-1 if '-1' in text else 1 if '+1' in text else 0))
            else:inds.append(j+(-1 if '-1' in text else 1 if '+1' in text else 0))
        return arrays[node['array']][tuple(inds)]
    if op in ('pos','neg'):
        x=value(node['arg'],u,f,h,i,j,a,b,c,k);return x if op=='pos' else -x
    x=value(node['left'],u,f,h,i,j,a,b,c,k);y=value(node['right'],u,f,h,i,j,a,b,c,k)
    return {'add':lambda:x+y,'sub':lambda:x-y,'mul':lambda:x*y,'div':lambda:x/y}[op]()


def reference(d):
    p=d['parameters'];expr=d['update']
    if d['family']=='matmul':
        n=p['n'];a=np.ones((n,n),dtype=np.float32);b=a.copy();c=np.zeros_like(a)
        for j in range(n):
            for i in range(n):
                for k in range(n):c[i,j]=value(expr,None,None,None,i,j,a,b,c,k)
        return c
    nx,ny=p['nx'],p['ny'];u=np.zeros((nx+2,ny+2),dtype=np.float32);f=np.ones_like(u);h=np.float32(p['h'])
    for it in range(p['niter']):
        if d['family']=='jacobi':
            target=u.copy()
            for j in range(1,ny+1):
                for i in range(1,nx+1):target[i,j]=value(expr,u,f,h,i,j)
            u=target
        elif d['family']=='rbgs':
            for color in (0,1):
                for j in range(1,ny+1):
                    for i in range(1,nx+1):
                        if (i+j)%2==color:u[i,j]=value(expr,u,f,h,i,j)
        else:
            old=u.copy()
            for j0 in range(1,ny+1,p['isy']):
                for i0 in range(1,nx+1,p['isx']):
                    for j in range(j0,j0+p['isy']):
                        for i in range(i0,i0+p['isx']):
                            if d['state_semantics']=='snapshot_islands':
                                # A separate island-local view fixes all external reads at old state.
                                view=old.copy();view[i0:i0+p['isx'],j0:j0+p['isy']]=u[i0:i0+p['isx'],j0:j0+p['isy']]
                            else:view=u
                            u[i,j]=value(expr,view,f,h,i,j)
    return u[1:-1,1:-1]


def diagnostic(source,d,mode):
    p=d['parameters'];mat=d['family']=='matmul';mpi=mode=='mpi'
    declarations='  real, allocatable :: audit_full(:,:)\n  integer :: audit_i, audit_j\n'
    source=source.replace('  implicit none','  implicit none\n'+declarations,1)
    nx=str(p['n'] if mat else p['nx']);ny=str(p['n'] if mat else p['ny'])
    body=f'  allocate(audit_full({nx},{ny}))\n  audit_full=0.0\n'
    if mpi:
        body+= '  audit_full(i0_global:i0_global+n_local-1,:)=C\n' if mat else '  audit_full(:,j0_global:j1_global)=u(1:nx,1:ny_local)\n'
        body+=f'  call MPI_Allreduce(MPI_IN_PLACE,audit_full,{nx}*{ny},MPI_REAL,MPI_SUM,MPI_COMM_WORLD,ierr)\n  if(rank==0) then\n'
    else:body+= '  audit_full=C\n' if mat else '  audit_full=u(1:nx,1:ny)\n'
    body+=f"  open(91,file='field.txt',status='replace')\n  do audit_j=1,{ny}\n    do audit_i=1,{nx}\n      write(91,'(ES24.16)') audit_full(audit_i,audit_j)\n    end do\n  end do\n  close(91)\n"
    if mpi:body+='  end if\n'
    if mpi:source=source.replace('  call MPI_Finalize(ierr)',body+'  call MPI_Finalize(ierr)')
    else:
        import re
        source=re.sub(r'(?m)^end program',body+'end program',source)
    return source


def main():
    import argparse
    pa=argparse.ArgumentParser();pa.add_argument('--out',default='audit_results/full_fields.json');args=pa.parse_args()
    records=[];env=dict(os.environ,ACC_DEVICE_TYPE='host',OMPI_MCA_rmaps_base_oversubscribe='1',UCX_VFS_ENABLE='n')
    with tempfile.TemporaryDirectory() as td:
        td=Path(td)
        for stem,family in [('poisson_jacobi','jacobi'),('poisson_rbgs','rbgs'),('poisson_island_gauss','island'),('matmul','matmul'),('poisson_jacobi','jacobi_half'),('poisson_jacobi','jacobi_no_source')]:
            case=family
            family='jacobi' if family.startswith('jacobi') else family
            text=(Path(__file__).parent/'examples'/f'{stem}.f90').read_text()
            if family=='matmul':text=text.replace('n = 64','n = 33')
            else:text=text.replace('nx = 64','nx = 40').replace('ny = 64','ny = 48').replace('niter = 100','niter = 7').replace('isx = 16','isx = 8').replace('isy = 16','isy = 8')
            if case=='jacobi_half':text=text.replace('0.25 *','0.125 *')
            if case=='jacobi_no_source':text=text.replace('+ h*h*f(i,j)','')
            inp=td/'input.f90';inp.write_text(text);d=frontend(inp)
            configs=[('serial',1),('openmp',1),('openmp',4),('gpu',1)]+[('mpi',n) for n in ([1,2,3,6] if family=='island' else [1,2,3,5])]
            for mode,count in configs:
                contract=snapshot_islands(d) if family=='island' and mode!='serial' else d
                prefix={'serial':'fortran','openmp':'openmp','gpu':'gpu','mpi':'mpi'}[mode]
                source=getattr(be,'gen_'+prefix+'_'+family)(contract)
                src=td/'code.f90';src.write_text(diagnostic(source,contract,mode));exe=td/'kernel'
                flags=['-fopenmp'] if mode=='openmp' else ['-fopenacc','-foffload=disable'] if mode=='gpu' else []
                cmd=[os.environ.get('MPIFC','mpifort') if mode=='mpi' else os.environ.get('FC','gfortran'),'-O2','-fcheck=all','-ffp-contract=off']+flags+[str(src),'-o',str(exe)]
                p=subprocess.run(cmd,capture_output=True,text=True,env=env)
                if p.returncode:raise RuntimeError(p.stderr)
                run=(['mpirun','--oversubscribe','-np',str(count)] if mode=='mpi' else [])+[str(exe)]
                env['OMP_NUM_THREADS']=str(count)
                p=subprocess.run(run,cwd=td,capture_output=True,text=True,env=env,timeout=40)
                if p.returncode:raise RuntimeError(p.stdout+p.stderr)
                expected=reference(contract);got=np.loadtxt(td/'field.txt').reshape(expected.shape,order='F')
                err=float(np.max(np.abs(got-expected)));assert np.allclose(got,expected,rtol=3e-6,atol=1e-9),(family,mode,count,err)
                records.append({'case':case,'family':family,'mode':mode,'threads_or_ranks':count,'state':contract['state_semantics'],'shape':list(expected.shape),'max_abs_error':err,'status':'PASS'})
                print(records[-1],flush=True)
    Path(args.out).parent.mkdir(parents=True,exist_ok=True);Path(args.out).write_text(json.dumps(records,indent=2)+'\n')

if __name__=='__main__':main()
