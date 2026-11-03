from moa_typed_ir import *
from moa_machine import Work,Machine
from dataclasses import replace
base=gamma_lower(stable_online_fold(specification()))
count=0
for n in range(1,13):
 for shape in ((1,),(2,),(2,3),(3,2,2),(17,)):
  ir=schedule_omega(base,Work(n,4,8,8),Machine(shape))
  assert verify_schedule(ir)
  code=emit_c(ir)
  assert 'owner_begin' in code and 'owner_end' in code
  count+=1
  bad=replace(ir,schedule=replace(ir.schedule,intervals=((0,n),)*len(ir.schedule.intervals)))
  if len(ir.schedule.intervals)>1:
   try: verify_schedule(bad)
   except ValueError: pass
   else: raise AssertionError('invalid ownership accepted')
print(f'Gate05 typed Ω scheduling: {count} checks PASS')
