from dataclasses import replace
from moa_machine import *
checked=0
for n in range(1,14):
 for shape in [(1,),(2,),(3,),(2,2),(2,3),(3,2),(2,2,2),(3,1,2)]:
  for m,d,dv in [(1,1,1),(3,4,2),(5,7,3)]:
   w=Work(n,m,d,dv); machine=Machine(shape)
   cert=resource_certificate(w,machine)
   owners=omega_lift(w,machine)
   assert verify_lift(w,machine,owners)
   assert cert['tau_output']==sum(x['output_components'] for x in cert['components'])
   assert cert['total_score_multiplications']==sum(x['score_multiplications'] for x in cert['components'])
   assert all(gamma(o.coordinate,shape)==i for i,o in enumerate(owners))
   checked+=1
w=Work(7,5,4,3); machine=Machine((2,3)); owners=omega_lift(w,machine)
for bad in [owners[:-1],(replace(owners[0],end=owners[0].end+1),)+owners[1:],
            (replace(owners[0],coordinate=(1,1)),)+owners[1:]]:
 try:verify_lift(w,machine,bad)
 except ValueError:pass
 else:raise AssertionError('invalid Ω lifting accepted')
for shape in [(),(2,0),(2,-1),(2,2.0)]:
 try:Machine(shape)
 except ValueError:pass
 else:raise AssertionError('invalid machine shape accepted')
print(f'PASS Ω machine lifting and exact logical resource certificates: {checked} shapes, 7 rejection cases')
