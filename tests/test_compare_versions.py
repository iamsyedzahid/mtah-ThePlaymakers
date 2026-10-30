"""R5 version comparison. Synthetic data only; the "teammate" is a folder made here."""
import shutil
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import compare_versions as cv  # noqa: E402


def _rows(other):
    return {row["role"]: row for row in cv.compare(other, seeds=(101,), n_frames=12_000)}


def _checkout(tmp_path, **files):
    (tmp_path / "mtah").mkdir()
    for name, text in files.items():
        (tmp_path / "mtah" / f"{name}.py").write_text(text, encoding="utf-8")
    return tmp_path


def test_an_identical_copy_ties_exactly(tmp_path):
    (tmp_path / "mtah").mkdir()
    shutil.copy(ROOT / "mtah" / "evidence.py", tmp_path / "mtah" / "evidence.py")
    row = _rows(tmp_path)["R2 evidence"]
    assert row["diff"] == 0.0 and row["note"].startswith("tie")


def test_a_swap_changes_only_through_their_file(tmp_path):
    other = _checkout(tmp_path, evidence=(
        "import numpy as np\n"
        "def evidence(tracking):\n"
        "    return np.zeros(tracking.n_frames)\n"
    ))
    rows = _rows(other)
    assert rows["R2 evidence"]["diff"] != 0.0
    assert rows["R3 touches"]["note"] == "they have no such file"
    assert rows["all their roles"]["spell"] == rows["R2 evidence"]["spell"]


def test_their_broken_code_is_reported_not_fatal(tmp_path):
    other = _checkout(tmp_path, touches="raise RuntimeError('nope')\n")
    rows = _rows(other)
    assert "failed: RuntimeError" in rows["R3 touches"]["note"]
    assert rows["ours (baseline)"]["spell"] > 0


def test_loading_a_swap_leaves_our_modules_alone(tmp_path):
    import mtah.evidence as before

    other = _checkout(tmp_path, evidence="def evidence(tracking):\n    return None\n")
    cv.load_modules(other, ("evidence",))
    assert sys.modules["mtah.evidence"] is before


def test_a_folder_without_mtah_is_refused(tmp_path):
    with pytest.raises(SystemExit):
        cv.compare(Path(tmp_path), seeds=(101,), n_frames=1_000)
