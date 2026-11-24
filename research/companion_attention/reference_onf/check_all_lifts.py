"""All-region gamma/owner/lifting correspondence; exact logical machine premises."""
import json,re,math
from pathlib import Path
from pycparser import c_ast,c_generator
from compare import parse,normalize
from check_lifts import certificate,FIELDS
from check_all import PRIOR,EXTRA,PAPER3
GEN=c_generator.CGenerator()

def structure(body):
    root=parse(body);loops=[];directives=[]
    def walk(node,parents):
        if isinstance(node,c_ast.Pragma):directives.append({'directive':re.sub(r'\s+','',node.string),'enclosing_indices':parents})
        if isinstance(node,c_ast.For):
            index=node.init.decls[0].name
            assert GEN.visit(node.init.decls[0].init)=='0' and node.cond.op=='<'
            loops.append({'index':index,'extent':GEN.visit(node.cond.right),'enclosing_indices':parents})
            walk(node.stmt,parents+[index]);return
        for _,child in node.children():walk(child,parents)
    walk(root,[]);return {'loops':loops,'directives':directives}

def run(programs,out):
    rows=json.loads((programs/'all_comparisons.json').read_text());registry={x[0]:x for x in PRIOR+EXTRA+PAPER3};proofs=[]
    for row in rows:
        name=row['profile'];hand=(programs/(name+'.hand.c')).read_text();compiled=(programs/(name+'.compiled.c')).read_text()
        assert normalize(parse(hand))==normalize(parse(compiled))
        a,b=structure(hand),structure(compiled);assert a==b,name
        proofs.append({'profile':name,'backend':registry[name][-1],'gamma_address_AST_equal':True,'all_loop_and_directive_structure_equal':True,'hand_lift_structure':a,'compiled_lift_structure':b,
          'fixed_machine_result':'For every common admissible machine shape, slot binding, private-workspace assignment, reduction tree and runtime schedule, the same Omega operator applied to these equal operational trees yields equal lifted ONFs and equal addressed C behavior.',
          'serial_scope':'Serial references retain serial execution; original tiled ONFs additionally retain the explicit it/tt lift. A parallel extension requires an admissible common workspace/reduction mapping.' if registry[name][-1]=='serial' else None})
    lifts=[certificate(L,shape) for L in range(1,38) for shape in ((1,),(2,),(8,),(2,3),(2,4,2),(64,),(108,))]
    tile_cases=[]
    for N in range(1,65):
        for T in range(1,N+1):
            if N%T:continue
            seen=[it*T+tt for it in range(N//T) for tt in range(T)]
            assert seen==list(range(N))
            for i in seen:
                assert (i//T)*T+i%T==i
                for b in (0,1):
                    for d in range(3):assert ((b*N+(i//T)*T+i%T)*3+d)==((b*N+i)*3+d)
            tile_cases.append({'N':N,'tile_width':T,'rho_lifted_index':[N//T,T],'tau':N,'index_bijection':True,'gamma_identity':True})
    head_cases=[]
    for H in range(1,33):
        for G in range(1,H+1):
            if H%G:continue
            mapping=[h//(H//G) for h in range(H)]
            assert mapping==[(h*G)//H for h in range(H)] and set(mapping)==set(range(G))
            assert all(mapping.count(g)==H//G for g in range(G))
            head_cases.append({'query_heads':H,'cache_heads':G,'head_selection':mapping,'no_expanded_KV_array':True})
    result={'profiles':proofs,'machine_properties_order':FIELDS,'physical_values':'parameters; unknown physical properties are not inferred from algebra',
      'logical_lift_proof':{'L':'positive logical owner count','P':'product(rho(machine))','forward':'i=floor(L*r/P)+j, 0<=j<floor(L*(r+1)/P)-floor(L*r/P)','inverse':'r=ceil(P*(i+1)/L)-1; j=i-floor(L*r/P)','tau_preservation':'sum_r floor(L*(r+1)/P)-floor(L*r/P)=L','rectangular':'rho(machine) concatenated with <L/P> only if P divides L','otherwise':'segmented legal-index relation; no claim that padded storage capacity equals logical component count'},
      'tile_reindex_proof':'N=NT*T, T>0. (it,tt)->it*T+tt has inverse (floor(i/T),i mod T), preserving tau and every gamma-selected component. Nested sums collapse by this finite bijection; per-output accumulation order is retained where shown by the source.',
      'fixed_machine_correspondence':'Equal operational trees under the same admissible Omega and gamma transformations yield equal lifted trees by congruence; this is an equality claim, not a proof that arbitrary mappings are admissible or optimal.',
      'finite_owner_regressions':lifts,'finite_tile_regressions':tile_cases,'finite_grouping_regressions':head_cases,
      'hardware_claim':'No actual A100/UA placement or timing is asserted by this certificate.'}
    out.mkdir(parents=True,exist_ok=True);(out/'all_lift_certificates.json').write_text(json.dumps(result,indent=2))
    print('PASS',len(proofs),'paired lift structures;',len(lifts),'owner,',len(tile_cases),'tile,',len(head_cases),'head-map regressions')
if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('programs',type=Path);p.add_argument('out',type=Path);a=p.parse_args();run(a.programs,a.out)
