import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_history(history, path):
    epochs = [h["epoch"] for h in history]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(epochs, [h["train_mse"] for h in history], label="train MSE")
    ax.plot(epochs, [h["val_mse"] for h in history], label="validation MSE")
    ax.set_xlabel("epoch"); ax.set_ylabel("MSE"); ax.set_title("Training curves"); ax.legend()
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def plot_scatter(trues, preds, metrics, path):
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(trues, preds, s=4, alpha=0.3)
    lo, hi = float(min(trues.min(), preds.min())), float(max(trues.max(), preds.max()))
    ax.plot([lo, hi], [lo, hi], "r--", lw=1)
    ax.set_xlabel("true pKd"); ax.set_ylabel("predicted pKd")
    ax.set_title(f"Test set | CI {metrics['ci']:.3f}  MSE {metrics['mse']:.3f}")
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def plot_label_distribution(labels, path):
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(labels, bins=50)
    ax.set_xlabel("pKd"); ax.set_ylabel("count"); ax.set_title("Davis label distribution")
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)
