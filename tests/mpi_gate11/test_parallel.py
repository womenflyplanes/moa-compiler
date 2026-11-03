from moa_parallel import emit_owned_openmp
from moa_machine import Machine,Work,verify_lift,omega_lift
from pathlib import Path
import subprocess,tempfile
shapes=[(1,),(2,),(2,3),(3,2,2),(7,),(11,)]
with tempfile.TemporaryDirectory() as td:
 for shape in shapes:
  code,cert=emit_owned_openmp(4,4,8,8,shape)
  assert 'owner_begin' in code and cert['tau_machine']==__import__('math').prod(shape)
  Path(td,'generated.c').write_text(code)
  subprocess.run(['cc','-O1','-fopenmp','-c',str(Path(td,'generated.c')),'-o',str(Path(td,'generated.o'))],check=True)
 for n in range(1,12):
  for p in range(1,10):
   shape=(p,)
   code,cert=emit_owned_openmp(n,4,8,8,shape)
   assert sum(x['output_components'] for x in cert['components'])==n*8
print('Gate04 Ω-owned OpenMP emission: 99 shape/extent checks and 6 compilation checks PASS')
