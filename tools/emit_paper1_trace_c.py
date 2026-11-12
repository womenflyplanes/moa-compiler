"""Bounded Paper I typed-DNF -> gamma/ONF -> C compiler (output-only).

Consumes the saved trace; rejects unsupported expressions rather than
silently substituting an attention kernel. Not a general psi normalizer.
"""
import argparse
import hashlib
import json
from pathlib import Path


class ContractError(ValueError):
    pass


def require(ok, why):
    if not ok:
        raise ContractError(why)


def check_access(node, name, shape, indices):
    require(node == {"array": {"name": name, "shape": shape}, "indices": indices},
            "unsupported psi access: " + name)


def address(name, shape, indices):
    require(len(shape) == len(indices) == 2, "expected rank-two access")
    require(all(s in ("n", "m", "d", "dv") for s in shape), "unknown extent")
    require(all(i in ("i", "j", "k", "v") for i in indices), "unknown index")
    return f"{name}[((size_t){indices[0]})*(size_t){shape[1]}+(size_t){indices[1]}]"


def emit(trace):
    require(trace.get("scope") == "restricted forward-attention typed rewrite; not general MoA compilation",
            "unsupported scope")
    dnf = trace["dnf"]
    score = dnf["score"]
    require(score["factor"] == "1/sqrt(d)" and dnf["stable"] is True, "unsupported reduction")
    require(dnf["key_index"] == "k" and dnf["value_index"] == "v", "unsupported indices")
    require(score["term"]["index"] == "j", "unsupported score reduction index")
    product = score["term"]["term"]
    require(set(product) == {"lhs", "rhs"}, "unsupported score product")
    q, k, v = product["lhs"], product["rhs"], dnf["values"]
    check_access(q, "Q", ["n", "d"], ["i", "j"])
    check_access(k, "K", ["m", "d"], ["k", "j"])
    check_access(v, "V", ["m", "dv"], ["k", "v"])
    summary = trace["stable_summary"]
    require(summary["score"] == score and summary["values"] == v, "DNF/summary disagreement")
    require(summary["state"] == ["a", "z", "u"] and summary["key_index"] == "k"
            and summary["value_index"] == "v", "unsupported summary")
    onf = trace["onf"]
    require(onf["summary"] == summary and onf["ownership"] == "query-disjoint",
            "ONF/summary disagreement")
    specs = [
        ("Q", ["n", "d"], ["i", "j"], "d"),
        ("K", ["m", "d"], ["k", "j"], "d"),
        ("V", ["m", "dv"], ["k", "v"], "dv"),
        ("O", ["n", "dv"], ["i", "v"], "dv"),
    ]
    require(len(onf["addresses"]) == len(specs), "wrong ONF address count")
    for node, (name, shape, idx, width) in zip(onf["addresses"], specs):
        require(node == {"array": {"name": name, "shape": shape},
                         "indices": idx, "width": width}, "ONF address mismatch")
    qa = address(q["array"]["name"], q["array"]["shape"], q["indices"])
    ka = address(k["array"]["name"], k["array"]["shape"], k["indices"])
    va = address(v["array"]["name"], v["array"]["shape"], v["indices"])
    oa = address("O", ["n", "dv"], ["i", "v"])
    return f"""/* Generated from validated MoA DNF and gamma/ONF; output-only. */
#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
int moa_trace_attention(const double *Q, const double *K, const double *V,
                        double *O, size_t n, size_t m, size_t d, size_t dv) {{
  if (!Q || !K || !V || !O || !n || !m || !d || !dv) return -1;
  if (n > SIZE_MAX/d || m > SIZE_MAX/d || m > SIZE_MAX/dv || n > SIZE_MAX/dv) return -2;
  if (m > SIZE_MAX/sizeof(double) || dv > SIZE_MAX/sizeof(double)) return -2;
  for (size_t i=0; i<n; ++i) {{
    double *u=(double*)calloc(dv,sizeof(double));
    if (!u) return -3;
    double a=-INFINITY, z=0.0;
    for (size_t k=0; k<m; ++k) {{
      double score=0.0;
      for (size_t j=0; j<d; ++j) score += {qa} * {ka};
      score /= sqrt((double)d);
      double anew=fmax(a,score);
      double old_weight=(a==-INFINITY)?0.0:exp(a-anew);
      double new_weight=exp(score-anew);
      z=z*old_weight+new_weight;
      for (size_t v=0; v<dv; ++v) u[v]=u[v]*old_weight+new_weight*{va};
      a=anew;
    }}
    for (size_t v=0; v<dv; ++v) {oa}=u[v]/z;
    free(u);
  }}
  return 0;
}}
"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("trace", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    raw = args.trace.read_bytes()
    source = emit(json.loads(raw))
    args.output.write_text(source)
    print(json.dumps({"trace_sha256": hashlib.sha256(raw).hexdigest(),
                      "c_sha256": hashlib.sha256(source.encode()).hexdigest(),
                      "output": str(args.output)}, sort_keys=True))


if __name__ == "__main__":
    main()
