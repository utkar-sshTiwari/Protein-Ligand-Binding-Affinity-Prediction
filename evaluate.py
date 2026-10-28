import argparse
import json

import torch
from torch.utils.data import DataLoader

from deepdta.config import Config
from deepdta.data import AffinityDataset, load_davis
from deepdta.engine import evaluate
from deepdta.model import DeepDTA
from deepdta.report import plot_scatter


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="runs/deepdta_davis/best.pt")
    ap.add_argument("--data-root", default="data/davis")
    a = ap.parse_args()

    ckpt = torch.load(a.ckpt, map_location="cpu")
    cfg = Config(**{**ckpt["config"], "data_root": a.data_root})
    data = load_davis(cfg)
    model = DeepDTA(len(ckpt["smiles_vocab"]), len(ckpt["protein_vocab"]), cfg)
    model.load_state_dict(ckpt["model_state"])

    loader = DataLoader(AffinityDataset(data, data.test_idx), batch_size=cfg.batch_size)
    m, preds, trues = evaluate(model, loader, torch.device("cpu"))
    print(json.dumps(m, indent=2))
    plot_scatter(trues, preds, m, a.ckpt.replace("best.pt", "test_scatter.png"))


if __name__ == "__main__":
    main()
