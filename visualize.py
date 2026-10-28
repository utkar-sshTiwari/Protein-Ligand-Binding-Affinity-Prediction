import shutil
from pathlib import Path

import torch
from torchviz import make_dot

from deepdta.config import Config
from deepdta.model import DeepDTA

cfg = Config()
model = DeepDTA(smiles_vocab_size=30, protein_vocab_size=25, cfg=cfg)
drug = torch.randint(1, 30, (2, cfg.smiles_max_len))
protein = torch.randint(1, 25, (2, cfg.protein_max_len))
graph = make_dot(model(drug, protein), params=dict(model.named_parameters()))

Path("runs").mkdir(exist_ok=True)
Path("runs/model_graph.gv").write_text(graph.source)
print("wrote runs/model_graph.gv")
if shutil.which("dot"):
    graph.render("runs/model_graph", format="png", cleanup=True)
    print("wrote runs/model_graph.png")
