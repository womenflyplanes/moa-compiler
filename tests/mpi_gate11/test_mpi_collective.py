from moa_typed_ir import specification,stable_online_fold,gamma_lower,schedule_omega
from moa_machine import Work,Machine
from moa_mpi import emit_mpi
from moa_mpi_collective import derive_collective,INT_MAX
base=gamma_lower(stable_online_fold(specification()))
tests=0
for n in range(1,21):
 for dv in (1,2,7,16):
  for shape in ((1,),(2,),(2,3),(3,2,2),(17,)):
   ir=schedule_omega(base,Work(n,4,8,dv),Machine(shape))
   c=derive_collective(ir)
   p=Machine(shape).tau
   assert c.output_components==n*dv
   assert c.aggregate_logical_receive_components==p*n*dv
   assert c.aggregate_nonlocal_required_components==(p-1)*n*dv
   assert sum(c.counts)==n*dv
   assert all(c.displacements[r]==sum(c.counts[:r]) for r in range(p))
   code=emit_mpi(ir)
   assert 'const int counts[] = {'+','.join(map(str,c.counts))+'};' in code
   assert 'const int displs[] = {'+','.join(map(str,c.displacements))+'};' in code
   tests+=1
# Avoid materializing billions of indices merely to test MPI integer bounds.
from unittest.mock import patch
from dataclasses import replace
small=schedule_omega(base,Work(1,1,1,1),Machine((1,)))
large=replace(small,schedule=replace(small.schedule,query_extent=INT_MAX,value_extent=2,intervals=((0,INT_MAX),)))
try:
 with patch('moa_mpi_collective.verify_schedule',return_value=True):derive_collective(large)
except ValueError:pass
else:raise AssertionError('overflow accepted')
print(f'Gate08 MPI collective metadata: {tests} configurations PASS')
