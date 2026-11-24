"""Focused fresh-device accumulator contract audit, independent of host fallback."""
import json,re
from pathlib import Path
from pycparser import c_ast
from compare import parse
import check_all  # registers all five archival variants
from data import DATA,ACCUMULATORS,corrected_directive

def writes(body):
    events=[]
    class Walk(c_ast.NodeVisitor):
        def visit_Assignment(self,node):
            if isinstance(node.lvalue,c_ast.ArrayRef):events.append((node.lvalue.name.name,node.op))
            self.generic_visit(node)
    Walk().visit(parse(body));return events

def check(root,out):
    rows=[]
    for profile,expected in ACCUMULATORS.items():
        events=writes((root/(profile+'.compiled.c')).read_text());first={}
        for name,op in events:first.setdefault(name,op)
        clauses={name:clause for clause,entries in DATA[profile] for name,_ in entries}
        bad=sorted(name for name,op in first.items() if op!='=' and clauses.get(name) in ('create','copyout'))
        assert set(bad)==expected,(profile,bad,expected)
        correction=corrected_directive(profile)
        copy_names=set(re.findall(r'([A-Za-z_]\w*)\[0:',correction.split(' copy(')[1]))
        assert copy_names==expected
        rows.append({'profile':profile,'archival_uninitialized_additive_targets':bad,
          'corrected_initialized_targets':sorted(copy_names),'fresh_device_contract_pass':True,
          'premises':['host buffers initialized to zero','fresh absent device allocations','matched full-domain computational phases'],
          'scope':'focused static audit of first additive writes; not general definite-assignment analysis',
          'actual_device_execution':'unavailable in this workspace'})
    out.mkdir(parents=True,exist_ok=True);(out/'device_initialization.json').write_text(json.dumps(rows,indent=2))
    print('PASS all corrected initialization contracts; archival defects confirmed')
if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('programs',type=Path);p.add_argument('out',type=Path);a=p.parse_args();check(a.programs,a.out)
