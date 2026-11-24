"""Syntax-check all fresh exports and verify their file bindings."""
import hashlib,json,subprocess,sys
from pathlib import Path
for case in sorted(Path(sys.argv[1]).iterdir()):
    if not case.is_dir():continue
    c=json.loads((case/'certificate.json').read_text())
    for name,sha in c['files_sha256'].items():assert hashlib.sha256((case/name).read_bytes()).hexdigest()==sha
    flags=['-fopenacc','-foffload=disable'] if c['backend']=='openacc' else ['-fopenmp']
    subprocess.run(['gcc','-fsyntax-only',*flags,str(case/'generated.c')],check=True,capture_output=True)
    print('PASS',case.name)
