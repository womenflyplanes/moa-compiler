"""Unit tests for the archive verifier's input-safety and integrity gates.

These tests use synthetic ZIPs; they do not assert Paper I numerical correctness.
Run: python -m unittest discover -s tests -p 'test_archive_verifier.py'
"""
import importlib.util
from pathlib import Path
import tempfile
import unittest
import zipfile

MODULE_PATH = Path(__file__).resolve().parents[1] / "tools" / "verify_paper1_archives.py"
SPEC = importlib.util.spec_from_file_location("paper1_archive_verifier", MODULE_PATH)
verifier = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(verifier)


class ArchiveVerifierTests(unittest.TestCase):
    def make_zip(self, directory, name):
        path = directory / "fixture.zip"
        with zipfile.ZipFile(path, "w") as archive:
            archive.writestr(name, "fixture")
        return path

    def test_digest_is_stable(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "fixture"
            path.write_bytes(b"abc")
            self.assertEqual(verifier.sha256(path),
                             "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad")

    def test_valid_archive_root(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            archive = self.make_zip(root, "Expected/file.txt")
            extracted = verifier.extract_checked(archive, root / "out", "Expected")
            self.assertEqual((extracted / "file.txt").read_text(), "fixture")

    def test_rejects_traversal_and_unexpected_roots(self):
        for name in ("Other/file.txt", "Expected/../escape.txt",
                     "Expected/..\\escape.txt", "Expected\\file.txt"):
            with self.subTest(name=name), tempfile.TemporaryDirectory() as td:
                root = Path(td)
                archive = self.make_zip(root, name)
                with self.assertRaises(ValueError):
                    verifier.extract_checked(archive, root / "out", "Expected")

    def test_rejects_symlinks(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            archive = root / "fixture.zip"
            with zipfile.ZipFile(archive, "w") as z:
                entry = zipfile.ZipInfo("Expected/link")
                entry.create_system = 3
                entry.external_attr = (0o120777 << 16)
                z.writestr(entry, "target")
            with self.assertRaises(ValueError):
                verifier.extract_checked(archive, root / "out", "Expected")


if __name__ == "__main__":
    unittest.main()
