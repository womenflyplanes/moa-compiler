"""Regression checks for the append-only SHA256 manifest freeze."""
import importlib.util
import json
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("freeze", ROOT / "tools/freeze_research_snapshot.py")
freeze = importlib.util.module_from_spec(spec)
spec.loader.exec_module(freeze)


class FreezeTests(unittest.TestCase):
    def test_freeze_rejects_dirty_and_preserves_file_hash(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            def git(*args):
                subprocess.run(["git", "-C", d, *args], check=True, capture_output=True)
            git("init", "-q")
            git("config", "user.name", "Snapshot Test")
            git("config", "user.email", "test@example.invalid")
            (root/"proof.tex").write_bytes(b"MoA psi rho gamma Omega\n")
            git("add", "proof.tex")
            git("commit", "-qm", "baseline")
            snapshot = freeze.create_manifest(root)
            self.assertEqual(snapshot["tracked_file_count"], 1)
            self.assertEqual(snapshot["files"][0]["path"], "proof.tex")
            self.assertEqual(snapshot["files"][0]["bytes"], len(b"MoA psi rho gamma Omega\n"))
            (root/"proof.tex").write_text("changed")
            with self.assertRaises(RuntimeError):
                freeze.create_manifest(root)

    def test_external_archives_are_explicitly_not_in_repo(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            subprocess.run(["git", "init", "-q", d], check=True, capture_output=True)
            subprocess.run(["git", "-C", d, "config", "user.name", "Snapshot Test"], check=True)
            subprocess.run(["git", "-C", d, "config", "user.email", "test@example.invalid"], check=True)
            subprocess.run(["git", "-C", d, "commit", "--allow-empty", "-qm", "initial"], check=True)
            snapshot = freeze.create_manifest(root)
            self.assertTrue(all(x["stored_in_repository"] is False
                                for x in snapshot["external_evidence"]))


if __name__ == "__main__":
    unittest.main()
