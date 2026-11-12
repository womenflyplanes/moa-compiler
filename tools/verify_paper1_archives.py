#!/usr/bin/env python3
"""Run bounded Paper I CPU checks against two immutable external evidence ZIPs.

This script does not submit hardware jobs, modify source archives, or claim GPU validation.
Usage:
 python tools/verify_paper1_archives.py --consolidated PATH --matched PATH --output /tmp/moa-verification
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import zipfile

ARCHIVES = {
    "consolidated": ("296b2767e77749a909039c64c4909ef7761a644c4942d42680d49f5cb4a78659", "MoA_PaperI_Consolidated_20261108"),
    "matched": ("099b9ca79f65e51a636884bde0efa75d0fc5f985db75b0e1c6c7565a203fa151", "MoA_PaperI_Matched_Validation_20261110"),
}

def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

def extract_checked(path, dest, expected_root):
    with zipfile.ZipFile(path) as archive:
        if archive.testzip() is not None:
            raise ValueError(f"Corrupt ZIP member in {path}")
        for info in archive.infolist():
            name = info.filename
            parts = Path(name).parts
            if not parts or parts[0] != expected_root or any(part in ("..", ".") for part in parts) or name.startswith("/"):
                raise ValueError(f"Unsafe or unexpected ZIP member: {name}")
            mode = (info.external_attr >> 16) & 0o170000
            if mode == 0o120000:
                raise ValueError(f"Symlink ZIP member rejected: {name}")
        archive.extractall(dest)
    root = dest / expected_root
    if not root.is_dir():
        raise ValueError(f"Missing archive root: {root}")
    return root

def run_case(label, cmd, cwd, output, env):
    result = subprocess.run(cmd, cwd=cwd, env=env, capture_output=True, text=True, timeout=180)
    (output / f"{label}.stdout.txt").write_text(result.stdout, encoding="utf-8")
    (output / f"{label}.stderr.txt").write_text(result.stderr, encoding="utf-8")
    return {"name": label, "command": cmd, "exit_code": result.returncode, "stdout": result.stdout.strip(), "stderr": result.stderr.strip()}

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--consolidated", type=Path, required=True)
    parser.add_argument("--matched", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        parser.error("Output directory must be absent or empty; frozen evidence is never overwritten")
    output.mkdir(parents=True, exist_ok=True)
    paths = {"consolidated": args.consolidated.resolve(), "matched": args.matched.resolve()}
    record = {"environment": {"python": sys.version, "platform": platform.platform()}, "archive_sha256": {}, "tests": [], "scope": "finite local CPU and protocol tests; no GPU validation, hardware prediction, or general proof"}
    roots = {}
    for key, path in paths.items():
        expected_sha, expected_root = ARCHIVES[key]
        actual = sha256(path)
        record["archive_sha256"][key] = actual
        if actual != expected_sha:
            raise ValueError(f"{key} SHA256 mismatch: {actual} != {expected_sha}")
        roots[key] = extract_checked(path, output / "extracted", expected_root)
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = "2"
    cases = [
        ("cpu_compiler", [sys.executable, "test_compiler.py"], roots["consolidated"] / "latest_compiler" / "compiler"),
        ("full_contract", [sys.executable, "test_full_contract.py"], roots["matched"]),
        ("protocol_guards", [sys.executable, "test_protocol.py"], roots["matched"]),
    ]
    try:
        for label, cmd, cwd in cases:
            result = run_case(label, cmd, cwd, output, env)
            record["tests"].append(result)
            print(f"{label}: {'PASS' if result['exit_code'] == 0 else 'FAIL'}")
            if result["exit_code"] != 0:
                break
    finally:
        (output / "verification.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    return 0 if len(record["tests"]) == len(cases) and all(t["exit_code"] == 0 for t in record["tests"]) else 1

if __name__ == "__main__":
    sys.exit(main())
