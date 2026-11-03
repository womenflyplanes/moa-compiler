from dataclasses import replace
from moa_typed_ir import *
a=specification()
assert stable_online_fold(a)
assert verify(gamma_lower(stable_online_fold(a)))
for bad in [replace(a,stable=False),replace(a,key_index='i'),replace(a,values=Psi(V,('i','v')))]:
 try: stable_online_fold(bad)
 except ValueError: pass
 else: raise AssertionError('bad dependency accepted')
l=gamma_lower(stable_online_fold(a))
try: verify(replace(l,ownership='overlap'))
except ValueError: pass
else: raise AssertionError('bad ownership accepted')
try: verify(replace(l,addresses=l.addresses[:-1]))
except ValueError: pass
else: raise AssertionError('bad address accepted')
print('PASS typed MoA rules: valid derivation + five rejection tests')
