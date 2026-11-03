"""Ω ownership-driven OpenMP generation for fixed finite machine shape and query extent."""
from moa_machine import Work,Machine,omega_lift,verify_lift,resource_certificate
from moa_typed_ir import specification,stable_online_fold,gamma_lower,emit_c
from pathlib import Path
import json

def emit_owned_openmp(n,m,d,dv,machine_shape):
    """Typed Ω schedule → γ addressed C, without editing emitted source text."""
    from moa_typed_ir import schedule_omega, verify_schedule
    work=Work(n,m,d,dv); machine=Machine(tuple(machine_shape))
    lowered=gamma_lower(stable_online_fold(specification()))
    scheduled=schedule_omega(lowered,work,machine)
    verify_schedule(scheduled)
    return emit_c(scheduled),resource_certificate(work,machine)

def compile_owned(folder,n,m,d,dv,machine_shape):
    folder=Path(folder); code,cert=emit_owned_openmp(n,m,d,dv,machine_shape)
    (folder/'generated_attention_owned.c').write_text(code)
    (folder/'owned_execution_certificate.json').write_text(json.dumps(cert,indent=2)+'\n')
    return cert

if __name__=='__main__':
    compile_owned(Path(__file__).parent,4,4,8,8,(2,3))
