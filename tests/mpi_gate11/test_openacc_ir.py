from dataclasses import replace
from moa_typed_ir import *
from moa_machine import Work,Machine
count=0
base=gamma_lower(stable_online_fold(specification()))
for n in range(1,13):
 for shape in ((1,),(2,),(2,3),(3,2,2),(17,)):
  ir=schedule_omega(base,Work(n,4,8,8),Machine(shape))
  code=emit_openacc(ir)
  assert '#pragma acc parallel loop gang' in code
  assert '#pragma acc data copyin' in code
  assert f'double u[8]' in code
  assert 'owner_begin' in code
  count+=1
try:emit_openacc(base)
except ValueError:pass
else:raise AssertionError('unscheduled IR accepted')
print(f'Gate06 OpenACC typed IR: {count} mappings PASS')
