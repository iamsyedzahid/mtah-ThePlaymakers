"""run.py --real, exercised on synthetic data only. Never reads real match data or ground truth."""
import hashlib
import re

import numpy as np

import run
from mtah.config import SECOND_HALF_D_BLOCK
from mtah.interfaces import A, D, H, X
from mtah.synthetic import make_synthetic

T = 3_000


def _fake_inputs():
    syn = make_synthetic(T, seed=0)
    return syn.tracking, list(syn.set_piece_intervals)  # synthetic: no published block to add


def test_published_block_is_added_once():
    windows = [(10, 20), (30, 40)]
    assert run.with_published_blocks(windows) == windows + [SECOND_HALF_D_BLOCK]
    assert windows == [(10, 20), (30, 40)]  # input untouched


def test_real_run_writes_a_valid_file_and_returns_its_hash(tmp_path, capsys):
    out = tmp_path / "The Playmakers.csv"
    digest = run.main_real(out, load=_fake_inputs, n_expected=T, halves=((0, T - 1),))
    assert digest == hashlib.sha256(out.read_bytes()).hexdigest()
    lines = out.read_text(encoding="utf-8").splitlines()
    assert lines[0] == "frame,label" and len(lines) == T + 1
    assert [int(line.split(",")[0]) for line in lines[1:]] == list(range(T))
    assert {line.split(",")[1] for line in lines[1:]} <= {H, A, D, X}
    assert f"sha256 {digest}" in capsys.readouterr().out


def test_real_run_prints_totals_only(tmp_path, capsys):
    run.main_real(tmp_path / "x.csv", load=_fake_inputs, n_expected=T, halves=((0, T - 1),))
    printed = capsys.readouterr().out
    assert not re.search(r"^\s*\d+,[HADX]\s*$", printed, re.MULTILINE)  # no frame,label rows
    assert "frame,label" not in printed
    assert len(printed.splitlines()) < 15


def test_real_run_is_byte_identical(tmp_path):
    a = run.main_real(tmp_path / "a.csv", load=_fake_inputs, n_expected=T, halves=((0, T - 1),))
    b = run.main_real(tmp_path / "b.csv", load=_fake_inputs, n_expected=T, halves=((0, T - 1),))
    assert a == b and (tmp_path / "a.csv").read_bytes() == (tmp_path / "b.csv").read_bytes()


def test_wrong_frame_count_is_refused(tmp_path):
    try:
        run.main_real(tmp_path / "x.csv", load=_fake_inputs, n_expected=T + 1)
    except ValueError as err:
        assert "expected" in str(err)
    else:
        raise AssertionError("a recording of the wrong length must be refused")
    assert not (tmp_path / "x.csv").exists()


def test_summary_counts_possessions_like_the_scorer():
    labels = np.array([X, H, H, X, H, D, A, A, X, A, H, D, D], dtype="<U1")
    s = run.summary(labels, halves=((0, labels.size - 1),))
    assert s["possessions"] == 3  # H (with an X inside), A (with an X inside), H
    assert s["graded_frames"] == labels.size
    assert abs(sum(s[f"share_{k}"] for k in "HADX") - 1.0) < 1e-12
