"""Read-only replay of preserved Paper I source/runner bindings; no job submission."""
import argparse,inspect,subprocess,sys,tarfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser();p.add_argument('--out',required=True);a=p.parse_args();out=Path(a.out).resolve();dest=out/'extracted';dest.mkdir(parents=True,exist_ok=True)
with tarfile.open(ROOT/'evidence/historical/paper1_job_137615.tar.gz') as t:
    for m in t.getmembers():
        target=(dest/m.name).resolve()
        if target!=dest and dest not in target.parents:raise ValueError('Unsafe archive path')
        if not(m.isfile() or m.isdir()):raise ValueError('Nonregular archive member')
    if 'filter' in inspect.signature(t.extractall).parameters:t.extractall(dest,filter='data')
    else:t.extractall(dest)
subprocess.run([sys.executable,str(ROOT/'research/companion_attention/reference_onf/audit_bindings.py'),str(dest/'MoA_PaperI_Matched_Validation_20261110'),str(ROOT/'evidence/attention_hand/original_hand'),str(out/'evidence')],check=True)
