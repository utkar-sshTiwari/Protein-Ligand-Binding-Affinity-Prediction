import argparse
import csv
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from deepdta.config import Config
from deepdta.data import AffinityDataset, load_davis
from deepdta.engine import evaluate, train_one_epoch
from deepdta.model import DeepDTA
from deepdta.report import plot_history, plot_label_distribution, plot_scatter


def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="data/davis")
    ap.add_argument("--out-dir", default="runs/deepdta_davis")
    ap.add_argument("--epochs", type=int, default=Config.epochs)
    ap.add_argument("--batch-size", type=int, default=Config.batch_size)
    ap.add_argument("--lr", type=float, default=Config.lr)
    ap.add_argument("--seed", type=int, default=Config.seed)
    ap.add_argument("--subset", type=int, default=0, help="use only N pairs (smoke test)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()

    cfg = Config(data_root=a.data_root, epochs=a.epochs, batch_size=a.batch_size, lr=a.lr, seed=a.seed)
    set_seed(cfg.seed)
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    device = torch.device(a.device)

    data = load_davis(cfg)
    if a.subset:
        data.train_idx = data.train_idx[:a.subset]
        data.val_idx = data.val_idx[:max(a.subset // 4, 1)]
        data.test_idx = data.test_idx[:max(a.subset // 4, 1)]
    print(f"pairs: {len(data.labels)} | train {len(data.train_idx)} val {len(data.val_idx)} test {len(data.test_idx)}")
    plot_label_distribution(data.labels, out / "label_distribution.png")

    mk = lambda idx, shuffle: DataLoader(AffinityDataset(data, idx), batch_size=cfg.batch_size, shuffle=shuffle)
    train_loader, val_loader, test_loader = mk(data.train_idx, True), mk(data.val_idx, False), mk(data.test_idx, False)

    model = DeepDTA(len(data.smiles_vocab), len(data.protein_vocab), cfg).to(device)
    print(f"model parameters: {sum(p.numel() for p in model.parameters()):,} | device: {device}")
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    loss_fn = nn.MSELoss()

    writer = SummaryWriter(str(out / "tensorboard"))
    try:
        d, p, _ = next(iter(val_loader))
        writer.add_graph(model, (d.to(device), p.to(device)))
    except Exception as e:
        print("skipping add_graph:", e)

    best_val, history = float("inf"), []
    for epoch in range(1, cfg.epochs + 1):
        train_mse = train_one_epoch(model, train_loader, optimizer, loss_fn, device)
        val_m, _, _ = evaluate(model, val_loader, device)
        history.append({"epoch": epoch, "train_mse": train_mse, "val_mse": val_m["mse"], "val_ci": val_m["ci"]})
        writer.add_scalar("MSE/train", train_mse, epoch)
        writer.add_scalar("MSE/val", val_m["mse"], epoch)
        writer.add_scalar("CI/val", val_m["ci"], epoch)
        print(f"epoch {epoch:3d} | train MSE {train_mse:.4f} | val MSE {val_m['mse']:.4f} | val CI {val_m['ci']:.4f}")
        if val_m["mse"] < best_val:
            best_val = val_m["mse"]
            torch.save({"model_state": model.state_dict(), "config": cfg.to_dict(),
                        "smiles_vocab": data.smiles_vocab, "protein_vocab": data.protein_vocab,
                        "epoch": epoch}, out / "best.pt")
    writer.close()

    with open(out / "history.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=history[0].keys()); w.writeheader(); w.writerows(history)
    plot_history(history, out / "training_curves.png")

    ckpt = torch.load(out / "best.pt", map_location=device)
    model.load_state_dict(ckpt["model_state"])
    test_m, preds, trues = evaluate(model, test_loader, device)
    test_m["best_epoch"] = ckpt["epoch"]
    print("TEST:", json.dumps(test_m, indent=2))
    with open(out / "test_results.json", "w") as f:
        json.dump(test_m, f, indent=2)
    plot_scatter(trues, preds, test_m, out / "test_scatter.png")
    print(f"results saved in {out}/")


if __name__ == "__main__":
    main()
