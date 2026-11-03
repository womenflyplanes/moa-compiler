from moa_typed_ir import *
from moa_machine import Work,Machine
from moa_mpi import emit_mpi
count=0
base=gamma_lower(stable_online_fold(specification()))
for n in range(1,13):
 for shape in ((1,),(2,),(2,3),(3,2,2),(17,)):
  s=schedule_omega(base,Work(n,4,8,8),Machine(shape))
  code=emit_mpi(s)
  assert 'MPI_Allgatherv' in code and 'MPI_Comm_size' in code
  assert f'if(p!={Machine(shape).tau}' in code
  assert 'owner_begin' in code and 'gamma2' in code
  count+=1
try:emit_mpi(base)
except ValueError:pass
else:raise AssertionError('unscheduled IR accepted')
print(f'Gate07 MPI typed IR: {count} mappings PASS')
