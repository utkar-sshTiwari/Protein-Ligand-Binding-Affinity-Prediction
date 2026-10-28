import torch
import torch.nn as nn

from deepdta.config import Config
from deepdta.model import DeepDTA

CFG = Config(protein_max_len=100, smiles_max_len=30)


def make_model():
    return DeepDTA(smiles_vocab_size=20, protein_vocab_size=22, cfg=CFG)


def batch(n=8):
    torch.manual_seed(0)
    return (torch.randint(1, 21, (n, CFG.smiles_max_len)),
            torch.randint(1, 23, (n, CFG.protein_max_len)))


def test_forward_output_shape():
    d, p = batch(8)
    assert make_model()(d, p).shape == (8,)


def test_padding_embedding_is_zero_and_stays_zero():
    m = make_model()
    assert torch.all(m.drug_branch.embed.weight[0] == 0)
    d, p = batch(4)
    m(d, p).sum().backward()
    assert torch.all(m.drug_branch.embed.weight.grad[0] == 0)


def test_all_parameters_receive_gradients():
    m = make_model()
    d, p = batch(4)
    nn.MSELoss()(m(d, p), torch.ones(4)).backward()
    missing = [n for n, q in m.named_parameters() if q.grad is None]
    assert not missing, missing


def test_eval_mode_is_deterministic():
    m = make_model().eval()
    d, p = batch(4)
    assert torch.equal(m(d, p), m(d, p))


def test_can_overfit_a_tiny_batch():
    torch.manual_seed(1)
    m = make_model()
    d, p = batch(8)
    y = torch.linspace(5, 9, 8)
    opt = torch.optim.Adam(m.parameters(), lr=1e-3)
    first = None
    for _ in range(150):
        opt.zero_grad()
        loss = nn.MSELoss()(m(d, p), y)
        loss.backward(); opt.step()
        first = first if first is not None else loss.item()
    assert loss.item() < 0.1 * first
