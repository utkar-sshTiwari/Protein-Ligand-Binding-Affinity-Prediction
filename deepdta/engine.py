import torch

from .metrics import regression_metrics


def train_one_epoch(model, loader, optimizer, loss_fn, device):
    model.train()
    total, n = 0.0, 0
    for drug, protein, y in loader:
        drug, protein, y = drug.to(device), protein.to(device), y.to(device)
        optimizer.zero_grad()
        loss = loss_fn(model(drug, protein), y)
        loss.backward()
        optimizer.step()
        total += loss.item() * len(y)
        n += len(y)
    return total / n


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    preds, trues = [], []
    for drug, protein, y in loader:
        preds.append(model(drug.to(device), protein.to(device)).cpu())
        trues.append(y)
    return torch.cat(preds).numpy(), torch.cat(trues).numpy()


def evaluate(model, loader, device):
    preds, trues = predict(model, loader, device)
    return regression_metrics(trues, preds), preds, trues
