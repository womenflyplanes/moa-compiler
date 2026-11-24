"""Bounded Fortran→structured MoA semantic IR adapter; unsupported input fails."""
import json
from pathlib import Path
from semantic_ir import frontend as frontend_to_dnf

if __name__ == '__main__':
    import sys
    for name in sys.argv[1:]:
        data=frontend_to_dnf(name)
        target=Path(name).with_suffix('.dnf.json')
        target.write_text(json.dumps(data,indent=2,ensure_ascii=False)+'\n')
        print(data['dnf']); print('Wrote',target)
