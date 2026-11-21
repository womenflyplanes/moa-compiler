import subprocess, re
cases = [
    ("JACOBI-MPI","moa_compiler/generated/poisson_jacobi_mpi.f90","/tmp/bench_jacobi_mpi"),
    ("ISLAND-MPI","moa_compiler/generated/poisson_island_gauss_mpi.f90","/tmp/bench_island_mpi"),
    ("RBGS-MPI","moa_compiler/generated/poisson_rbgs_mpi.f90","/tmp/bench_rbgs_mpi"),
    ("MATMUL-MPI","moa_compiler/generated/matmul_mpi.f90","/tmp/bench_matmul_mpi"),
]
print("program,ranks,time_s")
for name, src, bin_path in cases:
    subprocess.run(["mpifort","-O2","-o",bin_path,src], check=True, capture_output=True)
    for ranks in (1,2,4):
        times = []
        for _ in range(3):
            out = subprocess.run(["mpirun","-np",str(ranks),bin_path], capture_output=True, text=True, check=True).stdout
            m = re.search(r"time=\s*([0-9.E+-]+)", out)
            times.append(float(m.group(1)))
        print(f"{name},{ranks},{min(times):.6e}")
