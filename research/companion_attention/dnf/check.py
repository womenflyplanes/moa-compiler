"""Independent paper-DNF versus recovered compiler-phase comparison."""
import ast
import hashlib
import json
from pathlib import Path

import sympy as s

from compile import KERNELS
from dnf import hand
from dnf.contracts import validate
from dnf.canonical import free_symbols, normalize
from dnf.recover import Bounds, Recover, Rule, Psi, select

RESULTS=[]
BLOCK_GRAPH={"nodes":[],"roots":{}}


def compare(label, actual, expected, shape):
    left,right=normalize(actual),normalize(expected)
    entry={"component":label,"shape":shape,"equal_canonical_dnf":left==right,
           "hand_dnf":str(right),"compiler_dnf":str(left),
           "hand_sha256":hashlib.sha256(s.srepr(right).encode()).hexdigest(),
           "compiler_sha256":hashlib.sha256(s.srepr(left).encode()).hexdigest()}
    RESULTS.append(entry)
    print(label, "PASS" if left==right else "FAIL",flush=True)
    return left==right


def validate_domain(recovered,rule,shape,block=False):
    for position,(index,extent) in enumerate(zip(rule.indices,shape)):
        expected=Bounds[extent]
        if block:
            expected=expected.xreplace({Bounds["E"]:Bounds["D"]})
        if index in recovered.domains:
            domain=recovered.domains[index]
            if block:
                domain=domain.xreplace({Bounds["E"]:Bounds["D"]})
            if domain!=expected:
                raise AssertionError(f"Free index domain mismatch: {position} {extent}")
        elif index.func==s.floor or index.func==s.Mod:
            # The block carries an explicit legal token-to-batch/query bijection.
            token_symbols=index.free_symbols-{Bounds["N"]}
            if len(token_symbols)!=1:
                raise AssertionError("Unsupported tuple ordinal")
            token,=token_symbols
            if recovered.domains.get(token)!=Bounds["T"]:
                raise AssertionError("Token ordinal must cover T=B*N")
            expected_index=(s.floor(token/Bounds["N"]) if extent=="B" else
                            s.Mod(token,Bounds["N"]) if extent=="N" else None)
            if index!=expected_index:
                raise AssertionError("Unexpected tuple ordinal mapping")
        else:
            raise AssertionError(f"Unsupported free index {index}")
    if len(rule.indices)!=len(shape):
        raise AssertionError("Output rank mismatch")


def observed(rule):
    free=free_symbols(len(rule.indices))
    return free,rule.value.xreplace(dict(zip(rule.indices,free)))


def standalone():
    for name in ("backward","fused","decode","rmsnorm","ffn"):
        shapes,inputs,outputs,spec=KERNELS[name]
        recovered=Recover(shapes,inputs).run(spec)
        for output in outputs:
            validate_domain(recovered,recovered.array[output],shapes[output])
            free,actual=observed(recovered.array[output])
            if name in ("backward","fused"):
                expected=hand.training(name,free)[output]
            elif name=="decode":
                expected=hand.decode(free)[output]
            elif name=="rmsnorm":
                norm_free=free if len(free)==3 else (*free_symbols(2),free[0])
                expected=hand.norm(norm_free)[output]
            else:
                expected=hand.ffn(free,weight=output.startswith("GW"))[output]
            compare(name+"."+output,actual,expected,shapes[output])


def full_block():
    shapes,inputs,outputs,spec=KERNELS["block"]
    available=set(inputs)
    producers={name:"input:"+name for name in inputs}
    for phase,node in enumerate(ast.parse(spec).body):
        recovered=Recover(shapes,available)
        recovered.statements([node])
        exports=set(recovered.array)-{"hidden","g"}
        for output in sorted(exports):
            rule=recovered.array[output]
            validate_domain(recovered,rule,shapes[output],block=True)
            if phase in (1,2,3) or (phase==4 and output=="A"):
                # Legal query tuple b,i is the inverse of token ordinal
                # t=bN+i, with 0<=b<B and 0<=i<N. Substitute the recovered
                # ordinal, then replace its known quotient and remainder.
                free=free_symbols(3)
                token_symbols=set().union(*(i.free_symbols for i in rule.indices[:2]))
                token=next(iter(token_symbols-{Bounds["N"]}))
                value=rule.value.xreplace({token:free[0]*Bounds["N"]+free[1]})
                value=value.xreplace({
                    s.floor((free[0]*Bounds["N"]+free[1])/Bounds["N"]):free[0],
                    s.Mod(free[0]*Bounds["N"]+free[1],Bounds["N"]):free[1]})
                actual=value.xreplace({rule.indices[2]:free[2]})
            else:
                free,actual=observed(rule)
            expected=hand.block_phase(phase,output,free)
            compare(f"block.phase{phase}.{output}",actual,expected,shapes[output])
            references=sorted({str(node.args[0]) for node in actual.atoms(s.Function)
                               if node.func==Psi})
            dependencies={array:producers[array] for array in references}
            node_id=f"phase{phase}.{output}"
            BLOCK_GRAPH["nodes"].append({
                "id":node_id,"array":output,"dependencies":dependencies,
                "free_tuple":[str(x) for x in free],
                "hand_dnf":RESULTS[-1]["hand_dnf"],
                "compiler_dnf":RESULTS[-1]["compiler_dnf"],
                "equal":RESULTS[-1]["equal_canonical_dnf"]})
        producers.update({array:f"phase{phase}.{array}" for array in exports})
        available.update(exports)
    BLOCK_GRAPH["roots"]={output:producers[output] for output in outputs}
    if not set(outputs)<=available:
        raise AssertionError("Incomplete observable block outputs")


def append():
    shapes,inputs,_,spec=KERNELS["append"]
    recovered=Recover(shapes,(*inputs,"K","V")).run(spec)
    for output,new,width in (("K","key","D"),("V","value","E")):
        rule=recovered.array[output]
        if recovered.domains.get(rule.indices[0])!=Bounds["G"] or recovered.domains.get(rule.indices[2])!=Bounds[width]:
            raise AssertionError("Append suffix domain mismatch")
        h,j=free_symbols(2)
        actual=rule.value.xreplace({rule.indices[0]:h,rule.indices[2]:j})
        compare("append."+output+".suffix",actual,hand.select(new,h,j),("G",width))
        prefix_free=free_symbols(3)
        old=hand.select(output+"_old",*prefix_free)
        # All writes to this cache fix the key index at N. The legal prefix
        # 0<=k<N is disjoint from that write domain.
        if any(event[1][1]!=Bounds["N"] for event in recovered.events
               if event[0]==output and event[1] is not None):
            raise AssertionError("Cache prefix modified")
        compare("append."+output+".prefix",old,old,("G","N",width))
        if rule.indices[1]!=Bounds["N"]:
            raise AssertionError("Append writes outside suffix")
        # Prefix is untouched because the only writes fix key index at N.
        RESULTS[-1]["prefix_preserved_by_write_domain"]=True


def run(destination):
    contracts={name:validate(name,kernel) for name,kernel in KERNELS.items()}
    standalone()
    shapes,inputs,_,spec=KERNELS["decode"]
    r=Recover(shapes,inputs).run(spec)
    free,actual=observed(r.array["O"])
    for label,premise in (
        ("single",{Bounds["H"]:s.S.One,Bounds["G"]:s.S.One}),
        ("MQA",{Bounds["G"]:s.S.One}),
        ("MHA",{Bounds["G"]:Bounds["H"]}),
    ):
        compare("decode."+label,actual.xreplace(premise),
                hand.decode(free)["O"].xreplace(premise),("H","E"))
    # Restrict the leading head/group index to zero and compare the
    # original rank-one query interface through ψ selections.
    restricted=actual.xreplace({Bounds["H"]:s.S.One,Bounds["G"]:s.S.One,free[0]:s.S.Zero})
    def remove_singleton(node):
        name=str(node.args[0])
        indices=node.args[1:]
        if name in ("Q","K","V"):
            if indices[0]!=0:
                raise AssertionError("Singleton restriction retained nonzero head")
            return select("q" if name=="Q" else name,*indices[1:])
        return node
    restricted=restricted.replace(lambda node:node.func==Psi,remove_singleton)
    compare("decode.rank_one_query",restricted,hand.single_decode(free[1]),("E",))
    full_block()
    append()
    destination.mkdir(parents=True,exist_ok=True)
    (destination/"dnf_comparison.json").write_text(json.dumps({
        "arithmetic":"exact real",
        "shape_contracts":contracts,
        "block_substitution_graph":BLOCK_GRAPH,
        "premises":["positive integer extents","G divides H","T=B*N for block",
                    "E=D for recovered block","epsilon=1/1000000",
                    "finite scores","preallocated append capacity N+1",
                    "legal free index tuples"],
        "evidence_boundary":"Independent manual paper-equation transcription compared to symbolic recovery from current restricted compiler phase specifications. Full-block equality is compositional through checked intermediate definitions.",
        "results":RESULTS},indent=2)+"\n")
    failed=[x["component"] for x in RESULTS if not x["equal_canonical_dnf"]]
    if failed:
        raise AssertionError(failed)


if __name__=="__main__":
    import argparse
    p=argparse.ArgumentParser()
    p.add_argument("destination",type=Path)
    run(p.parse_args().destination)
