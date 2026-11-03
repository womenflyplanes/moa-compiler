"""Gate 07: typed MoA Ω schedule -> MPI C (replicated Q/K/V, disjoint O)."""
from pathlib import Path
from moa_typed_ir import ScheduledAttention,verify_schedule
from moa_machine import Machine,Work,resource_certificate
from moa_mpi_collective import derive_collective

def emit_mpi(ir):
    if not isinstance(ir,ScheduledAttention): raise ValueError('MPI needs typed Ω schedule')
    verify_schedule(ir)
    s=ir.schedule
    collective=derive_collective(ir)
    intervals=s.intervals
    counts=collective.counts
    displs=collective.displacements
    code=['/* MoA ψ DNF -> stable summary -> γ ONF -> Ω MPI ownership. */',
          '/* Contract: each rank holds identical complete read-only Q,K,V. */',
          '/* All ranks receive complete O via MPI_Allgatherv; no network model inferred. */',
          '#include <mpi.h>','#include <math.h>','#include <stddef.h>','#include <stdlib.h>',
          'static inline size_t gamma2(size_t i,size_t j,size_t w){return i*w+j;}',
          'int generated_attention_mpi(const double *Q,const double *K,const double *V,',
          ' double *O,size_t n,size_t m,size_t d,size_t dv,MPI_Comm comm){',
          ' int rank=0,p=0;',
          ' if(MPI_Comm_rank(comm,&rank)!=MPI_SUCCESS)return 1;',
          ' if(MPI_Comm_size(comm,&p)!=MPI_SUCCESS)return 2;',
          f' if(p!={len(intervals)} || n!={s.query_extent} || m!={s.key_extent} || d!={s.score_extent} || dv!={s.value_extent})return 3;',
          ' const int counts[] = {'+','.join(map(str,counts))+'};',
          ' const int displs[] = {'+','.join(map(str,displs))+'};',
          ' const size_t owner_begin[] = {'+','.join(str(b) for b,e in intervals)+'};',
          ' const size_t owner_end[] = {'+','.join(str(e) for b,e in intervals)+'};',
          ' const double scale=1.0/sqrt((double)d);',
          ' double *local=(double*)malloc((size_t)(counts[rank]?counts[rank]:1)*sizeof(double));',
          ' if(!local)return 4;',
          ' for(size_t i=owner_begin[rank];i<owner_end[rank];i++){',
          '  double *u=(double*)calloc(dv,sizeof(double));if(!u){free(local);return 5;}',
          '  double a=-INFINITY,z=0.0;',
          '  for(size_t k=0;k<m;k++){',
          '   double score=0.0;',
          '   for(size_t j=0;j<d;j++)score+=Q[gamma2(i,j,d)]*K[gamma2(k,j,d)];',
          '   score*=scale;',
          '   double next=fmax(a,score);',
          '   double old=(z==0.0)?0.0:exp(a-next),fresh=exp(score-next);',
          '   z=old*z+fresh;',
          '   for(size_t v=0;v<dv;v++)u[v]=old*u[v]+fresh*V[gamma2(k,v,dv)];',
          '   a=next;',
          '  }',
          '  for(size_t v=0;v<dv;v++)local[(i-owner_begin[rank])*dv+v]=u[v]/z;',
          '  free(u);',
          ' }',
          ' int status=MPI_Allgatherv(local,counts[rank],MPI_DOUBLE,O,counts,displs,MPI_DOUBLE,comm);',
          ' free(local);',
          ' return status==MPI_SUCCESS?0:6;',
          '}']
    return '\n'.join(code)+'\n'

def compile_mpi(folder,n,m,d,dv,machine_shape):
    from moa_typed_ir import specification,stable_online_fold,gamma_lower,schedule_omega
    from moa_machine import Work,Machine
    import json
    ir=schedule_omega(gamma_lower(stable_online_fold(specification())),Work(n,m,d,dv),Machine(tuple(machine_shape)))
    folder=Path(folder)
    out=folder/'generated_attention_mpi.c';out.write_text(emit_mpi(ir))
    cert=resource_certificate(Work(n,m,d,dv),Machine(tuple(machine_shape)))
    cert['mpi_contract']='replicated Q,K,V on each rank; Allgatherv complete O on each rank'
    from dataclasses import asdict
    cert['collective']=asdict(derive_collective(ir))
    cert['collective_payload_components_total']=n*dv
    cert['network_bytes_prediction']='not yet derived; collective algorithm and topology unspecified'
    (folder/'mpi_ownership_certificate.json').write_text(json.dumps(cert,indent=2)+'\n')
    return out
if __name__=='__main__':compile_mpi(Path(__file__).parent,4,4,8,8,(2,3))
