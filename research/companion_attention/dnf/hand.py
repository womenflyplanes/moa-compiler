"""Independent selected-component transcription of paper equations.

Paper II sections 3.1–3.5, Algorithms 1–2.
Paper III selected decode, head grouping and catenation.
Paper IV sections 3–4, 6.1–6.2 and section 9 composition.

This module does not read or import compiler phase specifications.
"""
import sympy as s
from .recover import Bounds, Reduce, select

B, N, D, E, F, H, G, T, C = (Bounds[n] for n in ("B","N","D","E","F","H","G","T","C"))


def sum_at(extent, term):
    index = s.Dummy("hand_index", integer=True)
    return Reduce(term(index), index, extent)


def probability(score, k):
    return s.exp(score(k)) / sum_at(N, lambda r: s.exp(score(r)))


def training(name, free):
    b, row, feature = free
    scale = 1/s.sqrt(D)
    score = lambda i,k: scale*sum_at(D, lambda h: select("Q", b,i,h)*select("K",b,k,h))
    weight = (lambda i,k: select("A",b,i,k)) if name == "backward" else (
        lambda i,k: probability(lambda r: score(i,r),k))
    g = lambda i,k: sum_at(E,lambda v: select("U",b,i,v)*select("V",b,k,v))
    correction = lambda i: sum_at(N,lambda r: weight(i,r)*g(i,r))
    softmax_adjoint = lambda i,k: weight(i,k)*(g(i,k)-correction(i))*scale
    return {
        "O": sum_at(N,lambda k: weight(row,k)*select("V",b,k,feature)),
        "GV": sum_at(N,lambda i: weight(i,row)*select("U",b,i,feature)),
        "GQ": sum_at(N,lambda k: softmax_adjoint(row,k)*select("K",b,k,feature)),
        "GK": sum_at(N,lambda i: softmax_adjoint(i,row)*select("Q",b,i,feature)),
    }


def decode(free):
    h,v = free
    group = s.floor(h/(H/G))  # H/G is integral under the shape contract.
    score = lambda k: sum_at(D,lambda j: select("Q",h,j)*select("K",group,k,j))/s.sqrt(D)
    return {"O": sum_at(N,lambda k: probability(score,k)*select("V",group,k,v))}


def norm(free, token=False):
    prefix = free[:-1]
    d = free[-1]
    epsilon = s.Rational(1,1000000)
    r = s.sqrt(sum_at(D,lambda j: select("X",*prefix,j)**2)/D+epsilon)
    normalized = lambda j: select("X",*prefix,j)/r
    correction = sum_at(D,lambda j: select("U",*prefix,j)*select("W",j)*normalized(j))
    result = {
        "Z": select("W",d)*normalized(d),
        "GX": (select("U",*prefix,d)*select("W",d)-normalized(d)*correction/D)/r,
    }
    if token:
        result["GW"] = sum_at(T, lambda t: select("U",t,d)*select("X",t,d)/s.sqrt(
            sum_at(D,lambda j: select("X",t,j)**2)/D+epsilon))
    else:
        result["GW"] = sum_at(B,lambda b: sum_at(N, lambda i:
            select("U",b,i,d)*select("X",b,i,d)/s.sqrt(
                sum_at(D,lambda j: select("X",b,i,j)**2)/D+epsilon)))
    return result


def ffn(free, weight=False):
    t,d = free
    gate = lambda token,f: sum_at(D,lambda j: select("X",token,j)*select("Wg",j,f))
    up = lambda token,f: sum_at(D,lambda j: select("X",token,j)*select("Wu",j,f))
    sigmoid = lambda a: 1/(1+s.exp(-a))
    silu = lambda a: a*sigmoid(a)
    hidden = lambda token,f: silu(gate(token,f))*up(token,f)
    gh = lambda token,f: sum_at(D,lambda j: select("U",token,j)*select("Wd",f,j))
    gu = lambda token,f: gh(token,f)*up(token,f)*sigmoid(gate(token,f))*(
        1+gate(token,f)*(1-sigmoid(gate(token,f))))
    gv = lambda token,f: gh(token,f)*silu(gate(token,f))
    if weight:
        # free tuple supplied in the order of each weight array.
        return {
            "GWg": sum_at(T,lambda token: select("X",token,t)*gu(token,d)),
            "GWu": sum_at(T,lambda token: select("X",token,t)*gv(token,d)),
            "GWd": sum_at(T,lambda token: hidden(token,t)*select("U",token,d)),
        }
    return {
        "Y": sum_at(F,lambda f: hidden(t,f)*select("Wd",f,d)),
        "GX": sum_at(F,lambda f: gu(t,f)*select("Wg",d,f)+gv(t,f)*select("Wu",d,f)),
        "Hcache": hidden(t,d), "GU": gu(t,d), "GV": gv(t,d),
    }


def block_phase(phase, output, free):
    """Paper IV component equations with named substitution points.

    Every component is checked before substituting it in the block DAG.
    This preserves the complete DNF composition without expanding repeated
    subexpressions exponentially.
    """
    eps=s.Rational(1,1000000)
    if phase in (0,5):
        token=free[0]
        x="X" if phase==0 else "R"
        scale="scale1" if phase==0 else "scale2"
        inv=1/s.sqrt(sum_at(D,lambda j: select(x,token,j)**2)/D+eps)
        return inv if output.startswith("inv") else select(scale,free[1])*select(x,*free)*inv
    if phase in (1,2,3):
        b,i,d=free
        ztoken=b*N+i
        weight={1:"Wq",2:"Wk",3:"Wv"}[phase]
        return sum_at(D,lambda j: select("Z1",ztoken,j)*select(weight,j,d))
    if phase==4:
        if output=="A":
            b,i,k=free
            score=lambda r: sum_at(D,lambda j: select("Q",b,i,j)*select("K",b,r,j))/s.sqrt(D)
            return probability(score,k)
        token,d=free
        b,i=s.floor(token/N),s.Mod(token,N)
        score=lambda r: sum_at(D,lambda j: select("Q",b,i,j)*select("K",b,r,j))/s.sqrt(D)
        return select("X",token,d)+sum_at(N,lambda k: probability(score,k)*select("V",b,k,d))
    if phase==6:
        aliases={"X":"Z2","U":"GY","Y":"FF","GX":"GZ2","GV":"FFgv"}
        source_name={v:k for k,v in aliases.items()}.get(output,output)
        return rename_arrays(ffn(free)[source_name],aliases)
    if phase==7:
        d,f=free
        if output=="GWd":
            f,d=free
            return sum_at(T,lambda token:select("Hcache",token,f)*select("GY",token,d))
        gu="GU" if output=="GWg" else "FFgv"
        return sum_at(T,lambda token:select("Z2",token,d)*select(gu,token,f))
    if phase==8:
        return select("R",*free)+select("FF",*free)
    if phase in (9,15):
        token,d=free
        x,u,w,inv,residual = (("R","GZ2","scale2","inv2","GY") if phase==9 else
                              ("X","GZ1","scale1","inv1","GR"))
        # Direct paper formula in x-hat and r, followed by r=1/inv.
        r=1/select(inv,token)
        xhat=lambda j:select(x,token,j)/r
        correction=sum_at(D,lambda j:select(u,token,j)*select(w,j)*xhat(j))
        return select(residual,token,d)+(select(u,token,d)*select(w,d)-xhat(d)*correction/D)/r
    if phase in (10,16):
        d,=free
        x,u,inv=(("R","GZ2","inv2") if phase==10 else ("X","GZ1","inv1"))
        return sum_at(T,lambda token:select(u,token,d)*select(x,token,d)*select(inv,token))
    if phase==11:
        b,i,d=free
        return select("GR",b*N+i,d)
    if phase==12:
        return training("backward",free)[output]
    if phase==13:
        token,d=free
        b,i=s.floor(token/N),s.Mod(token,N)
        return sum_at(D,lambda o: sum(
            select(grad,b,i,o)*select(weight,d,o)
            for grad,weight in (("GQ","Wq"),("GK","Wk"),("GV","Wv"))))
    if phase==14:
        d,o=free
        grad={"GWq":"GQ","GWk":"GK","GWv":"GV"}[output]
        return sum_at(T,lambda token:select("Z1",token,d)*select(grad,s.floor(token/N),s.Mod(token,N),o))
    raise ValueError((phase,output))


def rename_arrays(expr, aliases):
    from .recover import Psi
    return expr.replace(lambda node: node.func==Psi,
                        lambda node:select(aliases.get(str(node.args[0]),str(node.args[0])),*node.args[1:]))


def single_decode(v):
    """Original rank-one current-query DNF, with no singleton query axis."""
    score=lambda k:sum_at(D,lambda j:select("q",j)*select("K",k,j))/s.sqrt(D)
    return sum_at(N,lambda k:probability(score,k)*select("V",k,v))
