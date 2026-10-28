import argparse
import json
from pathlib import Path

import torch

from deepdta.config import Config
from deepdta.data import encode
from deepdta.model import DeepDTA


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="runs/deepdta_davis/best.pt")
    ap.add_argument("--data-root", default="data/davis")
    ap.add_argument("--smiles")
    ap.add_argument("--protein-id")
    ap.add_argument("--protein-seq")
    ap.add_argument("--list-proteins", action="store_true")
    a = ap.parse_args()

    proteins = json.load(open(Path(a.data_root) / "proteins.txt"))
    if a.list_proteins:
        print("\n".join(proteins)); return
    seq = a.protein_seq or proteins[a.protein_id]

    ckpt = torch.load(a.ckpt, map_location="cpu")
    cfg = Config(**{**ckpt["config"], "data_root": a.data_root})
    model = DeepDTA(len(ckpt["smiles_vocab"]), len(ckpt["protein_vocab"]), cfg)
    model.load_state_dict(ckpt["model_state"]); model.eval()

    d = torch.tensor([encode(a.smiles, ckpt["smiles_vocab"], cfg.smiles_max_len)])
    p = torch.tensor([encode(seq, ckpt["protein_vocab"], cfg.protein_max_len)])
    with torch.no_grad():
        pkd = model(d, p).item()
    print(f"predicted pKd = {pkd:.2f}  (about Kd = {10 ** (9 - pkd):.1f} nM; higher pKd = stronger binding)")


if __name__ == "__main__":
    main()
