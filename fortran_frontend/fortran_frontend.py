import re, pathlib, json

def parse_fortran(path):
    text = pathlib.Path(path).read_text()
    lines = []
    for ln in text.splitlines():
        ln = ln.split("!")[0].strip()
        if ln:
            lines.append(ln)
    src = "\n".join(lines)
    shape = {}
    for m in re.finditer(r"real\s*::\s*(\w+)\s*\(([^)]+)\)", src):
        shape[m.group(1)] = m.group(2)
    neighbours = []
    for m in re.finditer(r"(\w+)\(\s*i\s*([+-]\s*\d+)?\s*,\s*j\s*([+-]\s*\d+)?\s*\)", src):
        arr, di, dj = m.group(1), m.group(2), m.group(3)
        if arr in ("u", "u_new"):
            di = int(di.replace(" ", "")) if di else 0
            dj = int(dj.replace(" ", "")) if dj else 0
            if (di, dj) != (0, 0):
                neighbours.append((di, dj))
    neighbours = list(dict.fromkeys(neighbours))
    is_island = ("isx" in src and "isy" in src) or ("i0" in src and "j0" in src and "ix" in src)
    is_rbgs = (not is_island) and ("mod(i+j" in src or "mod(i + j" in src)
    is_matmul = "A(i,k)" in src.replace(" ", "") and "B(k,j)" in src.replace(" ", "")
    is_jacobi = (not is_island) and (not is_rbgs) and (not is_matmul) and len(neighbours) >= 4
    island = {}
    m1 = re.search(r"isx\s*=\s*(\d+)", src)
    m2 = re.search(r"isy\s*=\s*(\d+)", src)
    if m1 and m2:
        island = {"isx": int(m1.group(1)), "isy": int(m2.group(1))}
    return {"source": str(path), "shape": shape, "neighbours": neighbours,
            "is_island": is_island, "island": island, "is_rbgs": is_rbgs,
            "is_matmul": is_matmul, "is_jacobi": is_jacobi}

def psi_reduction(parsed):
    if parsed["is_matmul"]:
        dnf = "C psi (i,j) = sum_k ( (i,k) psi A ) * ( (k,j) psi B ) # matmul DNF"
        onf = "ONF matmul: loop j, loop i, loop k, gamma_col, F=d*k*n, M=d*k"
        return dnf, onf
    n = parsed["neighbours"]
    n_str = ",".join([f"({di},{dj})" for di, dj in n])
    dnf = (f"u_new psi ivec = 0.25 * sum_{{ (di,dj) in {{{n_str}}} }} "
           f"(ivec+(di,dj)) psi u + h^2 * (ivec psi f) "
           f"# Poisson stencil DNF, rav + gamma_col")
    if parsed.get("is_island"):
        onf = ("ONF Island Gauss: same DNF, "
               f"parallel over islands (ix,iy), island size {parsed.get('island', {})}, "
               "sequential Gauss-Seidel inside island (i,j), "
               "in-place, one array, halo at island boundaries, gamma_col")
    elif parsed["is_rbgs"]:
        onf = ("ONF RBGS: same DNF, split by parity: "
               "phase color=0: ivec where mod(i+j,2)==0, "
               "phase color=1: ivec where mod(i+j,2)==1, "
               "in-place, no second array, gamma_col")
    else:
        onf = ("ONF Jacobi: same DNF, one phase, all ivec in parallel, "
               "two arrays u / u_new, variable swap only, gamma_col")
    return dnf, onf

def frontend_to_dnf(path):
    parsed = parse_fortran(path)
    dnf, onf = psi_reduction(parsed)
    return {"source": parsed["source"], "shape": parsed["shape"],
            "neighbours": parsed["neighbours"],
            "is_jacobi": parsed["is_jacobi"],
            "is_island": parsed.get("is_island", False),
            "island": parsed.get("island", {}),
            "is_rbgs": parsed["is_rbgs"], "is_matmul": parsed["is_matmul"],
            "dnf": dnf, "onf": onf,
            "gamma": "gamma_col(ivec, rho) = ivec[0] + ivec[1]*rho[0]  # gamma_col=7",
            "rav": "rav A = A(:) in column-major order",
            "psi": "ivec psi A = (rav A)[gamma(ivec, rho A)]"}

if __name__ == "__main__":
    import sys
    for f in sys.argv[1:]:
        out = frontend_to_dnf(f)
        print(f"\n=== {f} ===")
        print(f"DNF: {out['dnf']}")
        print(f"ONF: {out['onf']}")
        print(f"PSI: {out['psi']}")
        print(f"GAMMA: {out['gamma']}")
        print(f"Neighbours: {out['neighbours']}")
        p = pathlib.Path(f)
        (p.with_suffix(".dnf.json")).write_text(json.dumps(out, indent=2))
        print(f"Wrote {p.with_suffix('.dnf.json')}")
