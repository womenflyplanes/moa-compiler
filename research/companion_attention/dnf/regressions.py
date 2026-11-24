"""Adversarial checks for symbolic-recovery and DNF comparisons."""
import ast
import json
from pathlib import Path

from compile import KERNELS
from dnf import hand
from dnf.canonical import free_symbols, normalize
from dnf.contracts import validate
from dnf.recover import Bounds, Recover, Reduce, select
import sympy as s


def expected(name,output,free):
    if name in ("backward","fused"):
        return hand.training(name,free)[output]
    if name=="decode":
        return hand.decode(free)[output]
    if name=="rmsnorm":
        return hand.norm(free)[output]
    return hand.ffn(free,weight=output.startswith("GW"))[output]


def equal_output(name,output,spec):
    shapes,inputs,_,_=KERNELS[name]
    recovered=Recover(shapes,inputs).run(spec)
    rule=recovered.array[output]
    free=free_symbols(len(rule.indices))
    actual=rule.value.xreplace(dict(zip(rule.indices,free)))
    return normalize(actual)==normalize(expected(name,output,free))


def main(destination):
    cases=[
        ("score_scale","fused","O","scale: scalar = 1.0 / sqrt(D)","scale: scalar = 2.0 / sqrt(D)"),
        ("gradient_sign","backward","GQ","g[k] - correction","g[k] + correction"),
        ("gradient_bound","backward","GQ","for v in serial(E):\n                g[k]","for v in serial(D):\n                g[k]"),
        ("gradient_operand","backward","GV","A[b,i,k] * U[b,i,v]","A[b,k,i] * U[b,i,v]"),
        ("decode_group","decode","O","h // (H // G)","h // G"),
        ("decode_normalizer","decode","O","a[k] / z","a[k] / (z + 1.0)"),
        ("norm_epsilon","rmsnorm","GX","1e-6","1e-5"),
        ("ffn_derivative","ffn","GX","gate * (1.0 - sig)","gate * (1.0 + sig)"),
        ("ffn_weight_selection","ffn","GWu","X[t,d] * GV[t,f]","X[t,d] * GU[t,f]"),
    ]
    results=[]
    for label,name,output,old,new in cases:
        source=KERNELS[name][3]
        assert old in source
        altered=source.replace(old,new)
        rejected=not equal_output(name,output,altered)
        assert rejected,label
        results.append({"check":label,"expected_mismatch_detected":rejected})
    # Renaming bound indices must preserve the DNF.
    class Rename(ast.NodeTransformer):
        def visit_Name(self,node):
            aliases={x:"renamed_"+x for x in ("b","i","k","j","v")}
            return ast.copy_location(ast.Name(id=aliases.get(node.id,node.id),ctx=node.ctx),node)
    source=ast.unparse(Rename().visit(ast.parse(KERNELS["backward"][3])))
    for output in ("GV","GQ","GK"):
        assert equal_output("backward",output,source)
    results.append({"check":"capture_avoiding_bound_renaming","equal":True})
    # Free domains are part of the output contract even when the scalar
    # expression itself is unchanged.
    from dnf.check import validate_domain
    shapes,inputs,_,source=KERNELS["backward"]
    wrong_domain=Recover(shapes,inputs).run(source.replace("owners(B)","owners(D)"))
    try:
        validate_domain(wrong_domain,wrong_domain.array["GV"],shapes["GV"])
    except AssertionError:
        results.append({"check":"free_output_domain","expected_mismatch_detected":True})
    else:
        raise AssertionError("Free output domain mutation accepted")
    # Bound extents and nested shadowed binders must be retained.
    i,j=s.symbols("i j",integer=True)
    a=Reduce(select("X",i),i,Bounds["N"])
    b=Reduce(select("X",j),j,Bounds["D"])
    assert normalize(a)!=normalize(b)
    results.append({"check":"different_reduction_extents","expected_mismatch_detected":True})
    shape,inp,out,spec=KERNELS["backward"]
    modified={**shape,"V":("B","N","D")}
    try:
        validate("backward",(modified,inp,out,spec))
    except AssertionError:
        results.append({"check":"input_shape_mutation","expected_mismatch_detected":True})
    else:
        raise AssertionError("Shape mismatch accepted")
    # Verify block residual and a projection-adjoint defect against independent
    # phase equations. Keep source phase boundaries as emitted.
    for label,phase,output,old,new in (
        ("block_residual",8,"Y","R[t, d] + FF[t, d]","FF[t, d]"),
        ("projection_adjoint",13,"GZ1","GK[","GQ["),
    ):
        nodes=ast.parse(KERNELS["block"][3]).body
        original=ast.unparse(nodes[phase])
        assert old in original,(label,original)
        shapes,inputs,_,_=KERNELS["block"]
        recovered=Recover(shapes,set(shapes)).run(original.replace(old,new))
        rule=recovered.array[output]
        free=free_symbols(len(rule.indices))
        actual=rule.value.xreplace(dict(zip(rule.indices,free)))
        assert normalize(actual)!=normalize(hand.block_phase(phase,output,free))
        results.append({"check":label,"expected_mismatch_detected":True})
    destination.mkdir(parents=True,exist_ok=True)
    (destination/"adversarial_checks.json").write_text(json.dumps(results,indent=2)+"\n")
    print(f"PASS {len(results)} adversarial and renaming checks")


if __name__=="__main__":
    import argparse
    p=argparse.ArgumentParser()
    p.add_argument("destination",type=Path)
    main(p.parse_args().destination)
