"""Create a deterministic append-only research snapshot manifest from a clean checkout.

This is a manifest generator, not a backup of files outside the checkout.
Use 'git archive' or a versioned ZIP to preserve bytes alongside the manifest.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess


def git(root, *args):
    return subprocess.check_output(["git", "-C", str(root), *args], text=True).strip()


def create_manifest(root):
    root = root.resolve()
    status = git(root, "status", "--porcelain")
    if status:
        raise RuntimeError("Refusing to freeze a dirty checkout: " + status[:200])
    head = git(root, "rev-parse", "HEAD")
    paths = subprocess.check_output(
        ["git", "-C", str(root), "ls-files", "-z"]
    ).split(b"\0")
    files = []
    for raw in paths:
        if not raw:
            continue
        path = raw.decode("utf-8")
        source = root / path
        if source.is_symlink():
            raise RuntimeError("Symlink requires explicit archival policy: " + path)
        data = source.read_bytes()
        files.append({"path": path, "bytes": len(data),
                      "sha256": hashlib.sha256(data).hexdigest()})
    return {
        "format": "moa-research-snapshot-v1",
        "git_commit": head,
        "tracked_file_count": len(files),
        "files": files,
        "external_evidence": [
            {"name": "MoA_PaperI_Consolidated_20261108.zip",
             "sha256": "296b2767e77749a909039c64c4909ef7761a644c4942d42680d49f5cb4a78659",
             "stored_in_repository": False},
            {"name": "MoA_PaperI_Matched_Validation_20261110.zip",
             "sha256": "099b9ca79f65e51a636884bde0efa75d0fc5f985db75b0e1c6c7565a203fa151",
             "stored_in_repository": False}
        ]
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path("."))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = create_manifest(args.repo)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise FileExistsError("Append-only: choose a new manifest path")
    args.output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(f"Frozen {manifest['tracked_file_count']} tracked files at {manifest['git_commit']}")


if __name__ == "__main__":
    main()
