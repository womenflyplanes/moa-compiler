#!/usr/bin/env python3
"""Compare saved symbolic MoA stage JSON; structural identity only, not a proof."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path

STAGES = ("01_psi_dnf.json", "02_stable_materialized_onf.json",
          "03_gamma_onf.json", "04_omega.json")

def audit(directory):
    cases = sorted(p for p in Path(directory).iterdir() if p.is_dir()
                   and (p / STAGES[0]).is_file())
    if not cases:
        raise ValueError("no cases with DNF records")
    stages = {}
    missing = []
    for stage in STAGES:
        groups = defaultdict(list)
        for case in cases:
            path = case / stage
            if not path.is_file():
                missing.append(str(path))
                continue
            obj = json.loads(path.read_text(encoding="utf-8"))
            normalized = json.dumps(obj, sort_keys=True, separators=(",", ":"),
                                    ensure_ascii=False)
            groups[hashlib.sha256(normalized.encode("utf-8")).hexdigest()].append(case.name)
        stages[stage] = [{"sha256": digest, "cases": names}
                         for digest, names in sorted(groups.items())]
    return {"case_count": len(cases), "stages": stages, "missing": missing,
            "scope": "symbolic serialized structural identity; not semantic equivalence"}

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("validation_dir", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = audit(args.validation_dir)
    if args.output:
        args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"cases": result["case_count"],
                      "unique_by_stage": {k: len(v) for k, v in result["stages"].items()},
                      "missing": result["missing"]}, indent=2))
    return 1 if result["missing"] else 0

if __name__ == "__main__":
    raise SystemExit(main())
