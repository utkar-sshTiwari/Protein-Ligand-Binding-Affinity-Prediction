import json
import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from .config import Config


def load_raw(root):
    root = Path(root)
    with open(root / "ligands_can.txt") as f:
        ligands = json.load(f)
    with open(root / "proteins.txt") as f:
        proteins = json.load(f)
    with open(root / "Y", "rb") as f:
        Y = pickle.load(f, encoding="latin1")
    return ligands, proteins, np.asarray(Y, dtype=float)


def kd_to_pkd(kd_nm):
    return -np.log10(np.asarray(kd_nm, dtype=float) / 1e9)


def build_vocab(strings):
    chars = sorted(set("".join(strings)))
    return {ch: i + 1 for i, ch in enumerate(chars)}


def encode(seq, vocab, max_len):
    ids = [vocab.get(ch, 0) for ch in seq[:max_len]]
    ids += [0] * (max_len - len(ids))
    return ids


@dataclass
class DavisData:
    drug_enc: np.ndarray
    protein_enc: np.ndarray
    rows: np.ndarray
    cols: np.ndarray
    labels: np.ndarray
    smiles_vocab: dict
    protein_vocab: dict
    drug_ids: list
    protein_ids: list
    train_idx: list
    val_idx: list
    test_idx: list


def load_davis(cfg: Config) -> DavisData:
    root = Path(cfg.data_root)
    ligands, proteins, Y = load_raw(root)

    drug_ids, protein_ids = list(ligands.keys()), list(proteins.keys())
    smiles_vocab = build_vocab(ligands.values())
    protein_vocab = build_vocab(proteins.values())

    drug_enc = np.array([encode(ligands[d], smiles_vocab, cfg.smiles_max_len) for d in drug_ids])
    protein_enc = np.array([encode(proteins[p], protein_vocab, cfg.protein_max_len) for p in protein_ids])

    pkd = kd_to_pkd(Y)
    rows, cols = np.where(~np.isnan(pkd))
    labels = pkd[rows, cols]

    with open(root / "folds" / "train_fold_setting1.txt") as f:
        train_folds = json.load(f)
    with open(root / "folds" / "test_fold_setting1.txt") as f:
        test_idx = json.load(f)

    val_idx = list(train_folds[cfg.val_fold])
    train_idx = [i for k, fold in enumerate(train_folds) if k != cfg.val_fold for i in fold]

    return DavisData(drug_enc, protein_enc, rows, cols, labels, smiles_vocab, protein_vocab,
                     drug_ids, protein_ids, train_idx, val_idx, list(test_idx))


class AffinityDataset(Dataset):
    def __init__(self, data: DavisData, idx):
        self.idx = list(idx)
        self.rows, self.cols, self.labels = data.rows, data.cols, data.labels
        self.drug = torch.from_numpy(data.drug_enc).long()
        self.protein = torch.from_numpy(data.protein_enc).long()

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, i):
        k = self.idx[i]
        return (self.drug[self.rows[k]],
                self.protein[self.cols[k]],
                torch.tensor(self.labels[k], dtype=torch.float))
