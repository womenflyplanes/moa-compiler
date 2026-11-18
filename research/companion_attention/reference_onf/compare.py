"""Independent ordered-C-AST comparison. No source is used by the emitter."""
import re, subprocess
from decimal import Decimal
import sympy as sp
from pycparser import c_parser, c_generator, c_ast

GEN=c_generator.CGenerator()

def clean(text):
    return re.sub(r'//[^\n]*|/\*.*?\*/','',text,flags=re.S).replace('\\\n',' ')

def function(text,name):
    text=clean(text)
    match=re.search(r'\b'+name+r'\s*\([^;{}]*\)\s*\{',text)
    if not match: raise ValueError(name)
    start=match.end(); depth=1
    for i in range(start,len(text)):
        depth+=(text[i]=='{')-(text[i]=='}')
        if depth==0:return text[start:i]
    raise ValueError('unbalanced function')

def timed(text):
    text=clean(text)
    start=re.search(r'double t0\s*=.*?;',text).end()
    stop=re.search(r'double t1\s*=',text[start:]).start()+start
    return text[start:stop]

def compute_region(text):
    """Separate outer data contract from its ordered computational body."""
    text=clean(text).strip()
    if text.startswith('#pragma acc data'):
        pos=text.index('{')
        return text[:pos].strip(),text[pos+1:text.rfind('}')]
    return None,text

def parse(body):
    macros='#define QKV(b,i,d) ((b)*N*D+(i)*D+(d))\n#define AA(b,i,j) ((b)*N*N+(i)*N+(j))\n'
    unit='typedef unsigned long size_t;\nvoid kernel(void) {\n'+body+'\n}'
    result=subprocess.run(['cpp','-P','-'],input=macros+unit,text=True,capture_output=True,check=True).stdout
    return c_parser.CParser().parse(result).ext[-1].body

def integer(node):
    if isinstance(node,c_ast.ID):return sp.Symbol(node.name)
    if isinstance(node,c_ast.Constant) and node.type=='int':return sp.Integer(node.value)
    if isinstance(node,c_ast.Cast):return integer(node.expr)
    if isinstance(node,c_ast.BinaryOp) and node.op in ('+','-','*'):
        left,right=integer(node.left),integer(node.right)
        return {'+':lambda:left+right,'-':lambda:left-right,'*':lambda:left*right}[node.op]()
    raise ValueError(GEN.visit(node))

def normalize(node):
    if node is None:return None
    if isinstance(node,c_ast.Compound):
        items=[normalize(x) for x in node.block_items or []]
        return ['sequence',items]
    if isinstance(node,c_ast.For):
        body=normalize(node.stmt)
        if not isinstance(node.stmt,c_ast.Compound):body=['sequence',[body]]
        return ['for',normalize(node.init),normalize(node.cond),normalize(node.next),body]
    if isinstance(node,c_ast.If):
        def wrap(x):
            value=normalize(x)
            return value if isinstance(x,c_ast.Compound) or x is None else ['sequence',[value]]
        return ['if',normalize(node.cond),wrap(node.iftrue),wrap(node.iffalse)]
    if isinstance(node,c_ast.ArrayRef):
        return ['select',normalize(node.name),str(sp.expand(integer(node.subscript))) ]
    if isinstance(node,c_ast.Constant) and node.type in ("double","float"):
        value=Decimal(node.value.rstrip("fFlL"))
        sign,digits,exponent=value.as_tuple()
        digits=list(digits)
        while len(digits)>1 and digits[-1]==0:
            digits.pop(); exponent+=1
        return ["Constant",node.type,[sign,digits,exponent]]
    if isinstance(node,c_ast.Pragma):return ['pragma',re.sub(r'\s+','',node.string)]
    # Everything else retains ordered scalar operations, types, names and literals.
    return [type(node).__name__,{name:getattr(node,name) for name in node.attr_names if name!='align'},
            [[name,normalize(child)] for name,child in node.children()]]

def compare(hand,compiled):
    return normalize(parse(hand))==normalize(parse(compiled))
