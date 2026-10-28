import torch
import torch.nn as nn

from .config import Config


class CNNBranch(nn.Module):
    def __init__(self, vocab_size, kernel, embed_dim=128, filters=32):
        super().__init__()
        self.embed = nn.Embedding(vocab_size + 1, embed_dim, padding_idx=0)
        self.convs = nn.Sequential(
            nn.Conv1d(embed_dim, filters, kernel), nn.ReLU(),
            nn.Conv1d(filters, filters * 2, kernel), nn.ReLU(),
            nn.Conv1d(filters * 2, filters * 3, kernel), nn.ReLU(),
        )
        self.out_dim = filters * 3

    def forward(self, x):
        x = self.embed(x).transpose(1, 2)
        x = self.convs(x)
        return x.max(dim=2).values


class DeepDTA(nn.Module):
    def __init__(self, smiles_vocab_size, protein_vocab_size, cfg: Config):
        super().__init__()
        self.drug_branch = CNNBranch(smiles_vocab_size, cfg.smiles_kernel, cfg.embed_dim, cfg.filters)
        self.protein_branch = CNNBranch(protein_vocab_size, cfg.protein_kernel, cfg.embed_dim, cfg.filters)
        joint = self.drug_branch.out_dim + self.protein_branch.out_dim
        self.head = nn.Sequential(
            nn.Linear(joint, 1024), nn.ReLU(), nn.Dropout(cfg.dropout),
            nn.Linear(1024, 1024), nn.ReLU(), nn.Dropout(cfg.dropout),
            nn.Linear(1024, 512), nn.ReLU(),
            nn.Linear(512, 1),
        )

    def forward(self, drug, protein):
        z = torch.cat([self.drug_branch(drug), self.protein_branch(protein)], dim=1)
        return self.head(z).squeeze(1)
