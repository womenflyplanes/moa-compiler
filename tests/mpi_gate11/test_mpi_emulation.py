"""Compile and execute emitted MPI C with a *sequential fake-MPI harness*.
This tests generated rank-specific code and collective metadata, not MPI runtime.
"""
from pathlib import Path
import subprocess, tempfile
from moa_mpi import compile_mpi
ROOT=Path(__file__).parent
cases=[(4,4,8,8,(2,3)),(1,3,5,7,(2,3)),(7,5,3,2,(3,2)),(9,2,4,3,(2,2,2)),(3,6,2,1,(1,)),(11,7,8,4,(3,3))]
for n,m,d,dv,shape in cases:
    p=1
    for x in shape:p*=x
    with tempfile.TemporaryDirectory() as temp:
        target=Path(temp)
        c=compile_mpi(target,n,m,d,dv,shape)
        binary=target/'run'
        cmd=['cc','-std=c11','-O1','-Wall','-Wextra','-Werror',f'-DN={n}',f'-DM={m}',f'-DD={d}',f'-DDV={dv}',f'-DP={p}','-I',str(ROOT/'mpi_syntax_stub'),str(c),str(ROOT/'test_mpi_emulated.c'),'-lm','-o',str(binary)]
        subprocess.run(cmd,check=True)
        subprocess.run([str(binary)],check=True)
print('Gate 09: 6 generated MPI variants, 150 emulated multi-rank trials PASS (not MPI runtime)')
