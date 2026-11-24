#!/usr/bin/env python3
"""Unified entry point; preserve distinct checked frontend/contract scopes."""
import argparse
import subprocess
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent


def main():
    pa=argparse.ArgumentParser(description=__doc__);sub=pa.add_subparsers(dest='command',required=True)
    sub.add_parser('status',help='Print completed and remaining work')
    p=sub.add_parser('attention',help='Emit one of the 44 completed correspondence cases')
    p.add_argument('--case',required=True);p.add_argument('--out',required=True);p.add_argument('--archival-entry',action='store_true')
    p=sub.add_parser('fortran',help='Compile the checked Fortran subset to explicit stages')
    p.add_argument('source');p.add_argument('--target',choices=('serial','matlab','openmp','openacc','mpi'),required=True);p.add_argument('--owners',type=int);p.add_argument('--snapshot-islands',action='store_true');p.add_argument('--out',required=True)
    p=sub.add_parser('validate',help='Replay host/compiler and attention correspondence checks')
    p.add_argument('--with-fortran',action='store_true');p.add_argument('--out',default='validation_results/unified')
    a=pa.parse_args()
    if a.command=='status':print((ROOT/'RESUME_HERE.md').read_text());return
    if a.command=='attention':
        sys.path.insert(0,str(ROOT/'tools'));from export_attention_case import compile_case
        r=compile_case(a.case,a.out,a.archival_entry);print(r['profile'],r['proof_status']);return
    if a.command=='fortran':
        cmd=[sys.executable,str(ROOT/'fortran_frontend/compile_moa.py'),a.source,'--target',a.target,'--out',a.out]
        if a.owners is not None:cmd+=['--owners',str(a.owners)]
        if a.snapshot_islands:cmd+=['--snapshot-islands']
    else:
        cmd=[sys.executable,str(ROOT/'scripts/validate_release.py'),'--out',a.out]
        if a.with_fortran:cmd+=['--with-fortran']
    subprocess.run(cmd,cwd=ROOT,check=True)

if __name__=='__main__':main()
