"""Independent phase equations for the complete hand block and norm-FFN chain.

No recipe or emitter imports. Named intermediate cuts are verified at their
producers and connected by the recorded substitution DAG.
"""
import sympy as s
from dnf.hand import sum_at,probability
from dnf.recover import Bounds,select
N,D,F,T=(Bounds[x] for x in ('N','D','F','T'))
EPS=s.Rational(1,1000000)

def silu(x):return x/(1+s.exp(-x))
def silu_grad(x):
    q=1/(1+s.exp(-x));return q*(1+x*(1-q))
def norm_r(x,t):return s.sqrt(sum_at(D,lambda j:select(x,t,j)**2)/D+EPS)
def norm_adjoint(x,r,gz,gamma,t,d,upstream=None):
    rr=select(r,t);xh=lambda j:select(x,t,j)/rr
    grad=(upstream if upstream is not None else lambda j:select(gz,t,j))
    corr=sum_at(D,lambda j:grad(j)*select(gamma,j)*xh(j))
    return (grad(d)*select(gamma,d)-xh(d)*corr/D)/rr

def ffn_adjoint(x,upstream,t,d,u='U',v=None):
    gate=lambda f:select(u,t,f)
    up=lambda f: select(v,t,f) if v else sum_at(D,lambda j:select(x,t,j)*select('Wu',j,f))
    gh=lambda f:sum_at(D,lambda j:select(upstream,t,j)*select('Wd',f,j))
    gu=lambda f:gh(f)*up(f)*silu_grad(gate(f))
    gv=lambda f:gh(f)*silu(gate(f))
    gx=sum_at(F,lambda f:gu(f)*select('Wg',d,f)+gv(f)*select('Wu',d,f))
    return gx,gu,gv

def block(phase,output,free):
    if phase in (0,3):
        t=free[0];x,r,z,gamma=('X','r1','Z1','gamma1') if phase==0 else ('R1','r2','Z2','gamma2')
        rr=norm_r(x,t)
        return rr if output==r else select(gamma,free[1])*select(x,*free)/rr
    if phase==1:
        t,o=free;weight={'Q':'Wq','K':'Wk','V':'Wv'}[output]
        return sum_at(D,lambda j:select('Z1',t,j)*select(weight,j,o))
    if phase==2:
        b,i,d=free
        score=lambda k:sum_at(D,lambda j:select('Q',b,i,j)*select('K',b,k,j))/s.sqrt(D)
        a=lambda k:probability(score,k)
        if output=='A':return a(d)
        return select('X',b,i,d)+sum_at(N,lambda k:a(k)*select('V',b,k,d))
    if phase==4:
        t,d=free
        u=lambda f:sum_at(D,lambda j:select('Z2',t,j)*select('Wg',j,f))
        v=lambda f:sum_at(D,lambda j:select('Z2',t,j)*select('Wu',j,f))
        h=lambda f:silu(u(f))*v(f)
        return {'U':lambda:u(d),'Vg':lambda:v(d),'H':lambda:h(d),
                'Y':lambda:select('R1',t,d)+sum_at(F,lambda f:h(f)*select('Wd',f,d))}[output]()
    if phase==5:
        if output=='GR1':return select('GY',*free)
        if output=='GZ2':return ffn_adjoint('Z2','GY',*free,v='Vg')[0]
        a,b=free
        if output=='GWd':return sum_at(T,lambda t:select('H',t,a)*select('GY',t,b))
        which=1 if output=='GWg' else 2
        return sum_at(T,lambda t:select('Z2',t,a)*ffn_adjoint('Z2','GY',t,a,v='Vg')[which](b))
    if phase in (6,10):
        x,r,gz,gamma,gx,gg=('R1','r2','GZ2','gamma2','GR1','Ggamma2') if phase==6 else ('X','r1','GZ1','gamma1','GX','Ggamma1')
        if output==gg:
            d,=free
            return sum_at(T,lambda t:select(gz,t,d)*select(x,t,d)/select(r,t))
        t,d=free
        return select(gx,t,d)+norm_adjoint(x,r,gz,gamma,t,d)
    if phase==7:
        i,=free
        return select('GR1',s.floor(i/D),s.Mod(i,D))
    if phase==8:
        b,i,d=free
        weights=lambda row,key:select('A',b,row,key)
        g=lambda row,key:sum_at(D,lambda j:select('GR1',b,row,j)*select('V',b,key,j))
        gs=lambda row,key:weights(row,key)*(g(row,key)-sum_at(N,lambda k:weights(row,k)*g(row,k)))/s.sqrt(D)
        return {'GV':lambda:sum_at(N,lambda row:weights(row,i)*select('GR1',b,row,d)),
                'GQ':lambda:sum_at(N,lambda key:gs(i,key)*select('K',b,key,d)),
                'GK':lambda:sum_at(N,lambda row:gs(row,i)*select('Q',b,row,d))}[output]()
    if phase==9:
        if output=='GZ1':
            t,d=free
            return sum_at(D,lambda o:sum(select(grad,t,o)*select(weight,d,o) for grad,weight in [('GQ','Wq'),('GK','Wk'),('GV','Wv')]))
        d,o=free;grad={'GWq':'GQ','GWk':'GK','GWv':'GV'}[output]
        return sum_at(T,lambda t:select('Z1',t,d)*select(grad,t,o))
    raise ValueError((phase,output))

def chain(phase,output,free):
    if phase==0:return norm_r('X',free[0])
    if phase==1:
        t,d=free;return select('gam',d)*select('X',t,d)/select('r',t)
    if phase==2:
        t,d=free;u=lambda f:sum_at(D,lambda j:select('Z',t,j)*select('Wg',j,f));v=lambda f:sum_at(D,lambda j:select('Z',t,j)*select('Wu',j,f))
        return u(d) if output=='U' else sum_at(F,lambda f:silu(u(f))*v(f)*select('Wd',f,d))
    if phase==3:
        if output in ('GWg','GWu','GWd'):
            a,b=free
            if output=='GWd':return sum_at(T,lambda t:silu(select('U',t,a))*sum_at(D,lambda j:select('Z',t,j)*select('Wu',j,a))*select('GF',t,b))
            which=1 if output=='GWg' else 2
            return sum_at(T,lambda t:select('Z',t,a)*ffn_adjoint('Z','GF',t,a)[which](b))
        if output=='Ggamma':
            d,=free
            return sum_at(T,lambda t:ffn_adjoint('Z','GF',t,d)[0]*select('X',t,d)/select('r',t))
        t,d=free
        if output=='GX_ffn':return ffn_adjoint('Z','GF',t,d)[0]
        if output=='GX_norm':return norm_adjoint('X','r',None,'gam',t,d,lambda j:ffn_adjoint('Z','GF',t,j)[0])
    raise ValueError((phase,output))

def serial_block(phase,output,free):
    if phase in (0,1,4,5,8,9,11,12,13):
        mapped={0:0,1:1,4:3,5:4,8:5,9:6,11:8,12:9,13:10}[phase]
        return block(mapped,output,free)
    if phase==2:
        value=block(2,'A' if output=='A' else 'R1',free)
        return value if output=='A' else value-select('X',*free)
    if phase==3:
        i,=free;t=s.floor(i/D);d=s.Mod(i,D)
        return select('X',t,d)+select('AttnOut',t,d)
    if phase==6:
        t,d=free
        return select('R1',t,d)+sum_at(F,lambda f:select('H',t,f)*select('Wd',f,d))
    if phase in (7,10):
        i,=free;t=s.floor(i/D);d=s.Mod(i,D)
        return select('GY' if phase==7 else 'GR1',t,d)
    raise ValueError((phase,output))
