from dataclasses import dataclass, asdict


@dataclass
class Config:
    data_root: str = "data/davis"
    protein_max_len: int = 1200
    smiles_max_len: int = 85
    val_fold: int = 0
    embed_dim: int = 128
    filters: int = 32
    smiles_kernel: int = 4
    protein_kernel: int = 8
    dropout: float = 0.1
    batch_size: int = 256
    lr: float = 1e-3
    epochs: int = 30
    seed: int = 42

    def to_dict(self):
        return asdict(self)
