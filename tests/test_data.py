import json
import pickle
from pathlib import Path

import numpy as np
import pytest

from deepdta.config import Config
from deepdta.data import AffinityDataset, build_vocab, encode, kd_to_pkd, load_davis


def test_pkd_conversion():
    assert kd_to_pkd(10000) == pytest.approx(5.0)
    assert kd_to_pkd(1) == pytest.approx(9.0)
    assert kd_to_pkd(0.1) > kd_to_pkd(100)


def test_vocab_reserves_zero_and_is_deterministic():
    v = build_vocab(["CBA", "AB"])
    assert v == {"A": 1, "B": 2, "C": 3}
    assert 0 not in v.values()


def test_encode_pads_truncates_and_handles_unknown():
    v = {"A": 1, "B": 2}
    assert encode("AB", v, 5) == [1, 2, 0, 0, 0]
    assert encode("ABABAB", v, 3) == [1, 2, 1]
    assert encode("AZ", v, 3) == [1, 0, 0]
    assert len(encode("", v, 4)) == 4


@pytest.fixture
def fake_root(tmp_path):
    (tmp_path / "folds").mkdir()
    json.dump({"d1": "CCO", "d2": "CC=O", "d3": "C#N"}, open(tmp_path / "ligands_can.txt", "w"))
    json.dump({"p1": "MKV", "p2": "AAA", "p3": "KKK", "p4": "VVV"}, open(tmp_path / "proteins.txt", "w"))
    Y = np.full((3, 4), 100.0)
    Y[1, 2] = np.nan
    Y[0, 0] = 10000.0
    pickle.dump(Y, open(tmp_path / "Y", "wb"))
    json.dump([[2, 3], [4, 5], [6, 7], [8, 9], [10]], open(tmp_path / "folds" / "train_fold_setting1.txt", "w"))
    json.dump([0, 1], open(tmp_path / "folds" / "test_fold_setting1.txt", "w"))
    return tmp_path


def test_pairs_skip_nan_and_map_to_drug_protein(fake_root):
    d = load_davis(Config(data_root=str(fake_root), protein_max_len=5, smiles_max_len=6))
    assert len(d.labels) == 11
    assert (1, 2) not in set(zip(d.rows, d.cols))
    assert (d.rows[0], d.cols[0]) == (0, 0) and d.labels[0] == pytest.approx(5.0)


def test_split_is_disjoint_and_covers_all_pairs(fake_root):
    d = load_davis(Config(data_root=str(fake_root), val_fold=1))
    parts = [set(d.train_idx), set(d.val_idx), set(d.test_idx)]
    assert sum(len(p) for p in parts) == len(set().union(*parts)) == 11
    assert set(d.val_idx) == {4, 5}


def test_dataset_item_shapes_and_dtypes(fake_root):
    cfg = Config(data_root=str(fake_root), protein_max_len=5, smiles_max_len=6)
    d = load_davis(cfg)
    drug, protein, y = AffinityDataset(d, d.train_idx)[0]
    assert drug.shape == (6,) and protein.shape == (5,) and y.shape == ()
    assert drug.dtype.is_floating_point is False and y.dtype.is_floating_point


REAL = Path("data/davis")
real_data = pytest.mark.skipif(not (REAL / "Y").exists(), reason="Davis data not downloaded")


@real_data
def test_real_davis_counts_match_paper():
    d = load_davis(Config())
    assert len(d.drug_ids) == 68 and len(d.protein_ids) == 442
    assert len(d.labels) == 30056
    assert d.drug_enc.shape == (68, 85) and d.protein_enc.shape == (442, 1200)


@real_data
def test_real_davis_labels_and_splits():
    d = load_davis(Config())
    assert 4.9 < d.labels.min() and d.labels.max() < 11.0
    parts = [set(d.train_idx), set(d.val_idx), set(d.test_idx)]
    assert sum(len(p) for p in parts) == len(set().union(*parts)) == len(d.labels)
