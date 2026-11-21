import pathlib, re
GEN = pathlib.Path("moa_compiler/generated")
OUT = pathlib.Path("/tmp/delta_bench")
OUT.mkdir(exist_ok=True)
def scaled(src_path, nx, niter=100):
    t = src_path.read_text()
    t = re.sub(r"nx = 64", f"nx = {nx}", t)
    t = re.sub(r"ny = 64", f"ny = {nx}", t)
    t = re.sub(r"niter = 100", f"niter = {niter}", t)
    t = re.sub(r"h = 1\.0 / 63\.0", f"h = 1.0 / {nx-1}.0", t)
    t = re.sub(r"n = 64", f"n = {nx}", t)
    return t
for nx in (256, 1024, 2048):
    for stem, backend in [
        ("poisson_jacobi", "out"),
        ("poisson_jacobi", "openmp"),
        ("poisson_jacobi", "gpu"),
        ("poisson_jacobi", "mpi"),
        ("poisson_island_gauss", "out"),
        ("poisson_island_gauss", "openmp"),
        ("poisson_island_gauss", "gpu"),
        ("poisson_island_gauss", "mpi"),
        ("poisson_rbgs", "out"),
        ("poisson_rbgs", "openmp"),
        ("poisson_rbgs", "gpu"),
        ("poisson_rbgs", "mpi"),
    ]:
        src = GEN / f"{stem}_{backend}.f90" if backend != "out" else GEN / f"{stem}_out.f90"
        if not src.exists():
            print(f"SKIP {src} not found")
            continue
        out_f = OUT / f"{stem}_{backend}_nx{nx}.f90"
        out_f.write_text(scaled(src, nx))
        print(f"Wrote {out_f}")
print("Done. Copy /tmp/delta_bench to Delta, compile there.")
