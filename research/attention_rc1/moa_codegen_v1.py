#!/usr/bin/env python3
"""Reproducible AST-driven C codegen experiment; not a full MoA compiler.
Uses general quotient/remainder index lifting for query/key loops, emits C for
an attention AST whose operation semantics are explicitly supported here.
"""
from dataclasses import dataclass,asdict
from pathlib import Path
from math import prod
import hashlib,json,subprocess,tempfile,ctypes,random,math,platform

@dataclass(frozen=True)
class Loop:
    name:str
    extent:int

@dataclass(frozen=True)
class Lift:
    name:str
    extent:int
    tile:int
    outer:str
    inner:str
    @property
    def outer_extent(self):return (self.extent+self.tile-1)//self.tile
    def indices(self):return [a*self.tile+b for a in range(self.outer_extent) for b in range(self.tile) if a*self.tile+b<self.extent]

def lift_loop(loop,tile):
    if tile<=0 or loop.extent<=0:raise ValueError('positive extents required')
    x=Lift(loop.name,loop.extent,tile,loop.name+'b',loop.name+'l')
    assert x.indices()==list(range(loop.extent))
    return x

def emit_c(query,key):
    # The two lift objects, not fixed numeric tile constants, determine loop structure.
    def loops(x,body,indent='  '):
        return (f'{indent}for(int {x.outer}=0;{x.outer}<{x.outer_extent};++{x.outer}){{\n'
                f'{indent} for(int {x.inner}=0;{x.inner}<{x.tile};++{x.inner}){{\n'
                f'{indent}  int {x.name}={x.outer}*{x.tile}+{x.inner};\n'
                f'{indent}  if({x.name}>= {x.extent})continue;\n'+body+
                f'{indent} }}\n{indent}}}\n')
    key_body='''      double sum=0.0;
      for(int d=0;d<D;++d)sum+=Q[(b*N+i)*D+d]*K[(b*N+j)*D+d];
      s[j]=sum/sqrt((double)D);
'''
    query_body=('''    double *s=(double*)malloc((size_t)N*sizeof(double));
    if(!s)return -2;
'''+loops(key,key_body,'    ')+'''    double mx=-INFINITY;
    for(int j=0;j<N;++j)if(s[j]>mx)mx=s[j];
    double den=0.0;
    for(int j=0;j<N;++j){s[j]=exp(s[j]-mx);den+=s[j];}
    for(int j=0;j<N;++j)P[(b*N+i)*N+j]=s[j]/den;
    for(int v=0;v<Dv;++v){
      double acc=0.0;
      for(int j=0;j<N;++j)acc+=P[(b*N+i)*N+j]*V[(b*N+j)*Dv+v];
      O[(b*N+i)*Dv+v]=acc;
    }
    free(s);
''')
    return ('#include <math.h>\n#include <stdlib.h>\n'
            'int moa_attention_generated(const double*Q,const double*K,const double*V,double*P,double*O,int B,int N,int D,int Dv){\n'
            f' if(B<1||N!={query.extent}||N!={key.extent}||D<1||Dv<1)return -1;\n'
            ' for(int b=0;b<B;++b){\n'+loops(query,query_body,'  ')+ ' }\n return 0;\n}\n')

def reference(Q,K,V,B,N,D,Dv):
    P=[0.0]*(B*N*N);O=[0.0]*(B*N*Dv)
    for b in range(B):
        for i in range(N):
            s=[sum(Q[(b*N+i)*D+d]*K[(b*N+j)*D+d] for d in range(D))/math.sqrt(D) for j in range(N)]
            mx=max(s);e=[math.exp(x-mx) for x in s];den=sum(e)
            for j in range(N):P[(b*N+i)*N+j]=e[j]/den
            for v in range(Dv):O[(b*N+i)*Dv+v]=sum(P[(b*N+i)*N+j]*V[(b*N+j)*Dv+v] for j in range(N))
    return P,O

def run():
    cases=[(2,4,8,8,2,2),(2,5,3,2,2,3),(1,7,4,5,3,4),(2,9,8,3,4,5)]
    root=Path(__file__).resolve().parent
    records=[];rng=random.Random(20261016)
    with tempfile.TemporaryDirectory() as temp:
        for B,N,D,Dv,Tq,Tk in cases:
            q=lift_loop(Loop('i',N),Tq);k=lift_loop(Loop('j',N),Tk)
            source=emit_c(q,k);digest=hashlib.sha256(source.encode()).hexdigest()
            c=Path(temp)/'generated.c';so=Path(temp)/f'generated_{N}_{Tq}_{Tk}.so';c.write_text(source)
            subprocess.run(['cc','-O2','-fPIC','-shared',str(c),'-o',str(so),'-lm'],check=True)
            fn=ctypes.CDLL(str(so)).moa_attention_generated
            ptr=ctypes.POINTER(ctypes.c_double)
            fn.argtypes=[ptr]*5+[ctypes.c_int]*4
            Q=[rng.uniform(-1,1) for _ in range(B*N*D)]
            K=[rng.uniform(-1,1) for _ in range(B*N*D)]
            V=[rng.uniform(-1,1) for _ in range(B*N*Dv)]
            arr=lambda x:(ctypes.c_double*len(x))(*x)
            a,b,cval=arr(Q),arr(K),arr(V)
            p=(ctypes.c_double*(B*N*N))();o=(ctypes.c_double*(B*N*Dv))()
            rc=fn(a,b,cval,p,o,B,N,D,Dv)
            pr,orr=reference(Q,K,V,B,N,D,Dv)
            ep=max(abs(x-y) for x,y in zip(p,pr));eo=max(abs(x-y) for x,y in zip(o,orr))
            assert rc==0 and ep<1e-12 and eo<1e-12
            records.append({'shape':[B,N,D,Dv],'tiles':[Tq,Tk],'index_checks':{'query':len(q.indices()),'key':len(k.indices())},'max_abs_P':ep,'max_abs_O':eo,'generated_source_sha256':digest,'pass':True})
            print(f'PASS B={B} N={N} D={D} Dv={Dv} tiles={Tq},{Tk} P={ep:.3e} O={eo:.3e}')
            if N==5:(root/'moa_generated_attention_example.c').write_text(source)
    report={'status':'PASS','scope':'AST-driven loop lifting and C emission for supported forward attention operation; not general DNF-to-ONF proof, not 15D machine model, not performance prediction','compiler':'cc -O2 -fPIC -shared -lm','python':platform.python_version(),'tests':records}
    (root/'moa_codegen_validation_v1.json').write_text(json.dumps(report,indent=2)+'\n')
    print('TRACE moa_codegen_validation_v1.json')
if __name__=='__main__':run()
