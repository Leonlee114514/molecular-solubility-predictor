"""Graph Neural Network for molecular solubility prediction.
Lightweight 3-layer GIN with global mean pooling — pure PyTorch, no PyG/DGL needed.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from rdkit import Chem

# ── Atom featurizer ──

_ATOMIC_NUM_SET = {1, 6, 7, 8, 9, 15, 16, 17, 35, 53}  # H, C, N, O, F, P, S, Cl, Br, I

def _atomic_num_onehot(atomic_num):
    """One-hot for top 10 common elements + 'other' = 11 dims."""
    elements = [1, 6, 7, 8, 9, 15, 16, 17, 35, 53]
    vec = [0] * 11
    try:
        vec[elements.index(atomic_num)] = 1
    except ValueError:
        vec[10] = 1
    return vec

def _formal_charge_onehot(charge):
    """One-hot: -2, -1, 0, +1, +2 → 5 dims."""
    idx = charge + 2
    if idx < 0:
        idx = 0
    if idx > 4:
        idx = 4
    vec = [0] * 5
    vec[idx] = 1
    return vec

def _hybridization_onehot(hyb):
    """sp, sp2, sp3, sp3d, other → 5 dims."""
    mapping = {
        Chem.rdchem.HybridizationType.SP: 0,
        Chem.rdchem.HybridizationType.SP2: 1,
        Chem.rdchem.HybridizationType.SP3: 2,
        Chem.rdchem.HybridizationType.SP3D: 3,
    }
    vec = [0] * 5
    vec[mapping.get(hyb, 4)] = 1
    return vec

ATOM_FEATURE_DIM = 37  # 11 + 6 + 5 + 5 + 1 + 5 + 3 + 1


class MoleculeGraphEncoder:
    """Convert an RDKit Mol to graph tensors (node features, edge index, batch index)."""

    def mol_to_graph(self, mol):
        n_atoms = mol.GetNumAtoms()
        if n_atoms == 0:
            return None

        # ── Node features ──
        x = torch.zeros((n_atoms, ATOM_FEATURE_DIM), dtype=torch.float32)
        for atom in mol.GetAtoms():
            idx = atom.GetIdx()
            feats = []
            feats.extend(_atomic_num_onehot(atom.GetAtomicNum()))         # 11
            degree = min(atom.GetDegree(), 5)
            dh = [0] * 6
            dh[degree] = 1
            feats.extend(dh)                                                # 6
            feats.extend(_formal_charge_onehot(atom.GetFormalCharge()))     # 5
            feats.extend(_hybridization_onehot(atom.GetHybridization()))    # 5
            feats.append(int(atom.GetIsAromatic()))                         # 1
            nh = min(atom.GetTotalNumHs(), 4)
            hh = [0] * 5
            hh[nh] = 1
            feats.extend(hh)                                                # 5
            chiral = 0
            if atom.HasProp("_CIPCode"):
                c = atom.GetProp("_CIPCode")
                if c == "R":
                    chiral = 1
                elif c == "S":
                    chiral = 2
            ch = [0] * 3
            ch[chiral] = 1
            feats.extend(ch)                                                # 3
            feats.append(int(atom.IsInRing()))                              # 1
            x[idx] = torch.tensor(feats, dtype=torch.float32)

        # ── Edge index ──
        edges = []
        for bond in mol.GetBonds():
            u, v = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            edges.append((u, v))
            edges.append((v, u))
        if not edges:
            # Single-atom case
            edge_index = torch.zeros((2, 0), dtype=torch.long)
        else:
            edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

        return {"x": x, "edge_index": edge_index}


# ── Collation for mini-batches ──

def collate_graphs(graphs):
    """Merge a list of single-molecule graph dicts into one disconnected graph.
    Returns (x, edge_index, batch) ready for SolubilityGNN.forward().
    """
    xs = []
    eis = []
    batches = []
    offset = 0
    for i, g in enumerate(graphs):
        if g is None:
            continue
        xs.append(g["x"])
        if g["edge_index"].numel() > 0:
            eis.append(g["edge_index"] + offset)
        batches.append(torch.full((g["x"].size(0),), i, dtype=torch.long))
        offset += g["x"].size(0)
    if not xs:
        return None
    x = torch.cat(xs, dim=0)
    edge_index = torch.cat(eis, dim=1) if eis else torch.zeros((2, 0), dtype=torch.long)
    batch = torch.cat(batches, dim=0)
    return {"x": x, "edge_index": edge_index, "batch": batch}


# ── Model layers ──

class GINBlock(nn.Module):
    """Graph Isomorphism Network convolution block."""

    def __init__(self, dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
            nn.ReLU(),
            nn.Linear(dim, dim),
        )
        self.eps = nn.Parameter(torch.zeros(1))

    def forward(self, x, edge_index):
        # Message passing: sum neighbor features
        row, col = edge_index
        out = torch.zeros_like(x)
        out = out.index_add(0, row, x[col])
        out = (1 + self.eps) * x + out
        return self.mlp(out) + x  # residual


def _global_mean_pool(x, batch):
    """Average node embeddings per graph."""
    num_graphs = batch.max().item() + 1
    out = x.new_zeros((num_graphs, x.size(1)))
    out = out.index_add_(0, batch, x)
    counts = batch.bincount(minlength=num_graphs).unsqueeze(1).float()
    return out / counts.clamp(min=1)


class SolubilityGNN(nn.Module):
    """3-layer GIN → global mean pool → MLP head → scalar logS."""

    def __init__(self, atom_dim=ATOM_FEATURE_DIM, hidden_dim=128, num_layers=3):
        super().__init__()
        self.atom_embed = nn.Linear(atom_dim, hidden_dim)
        self.convs = nn.ModuleList([GINBlock(hidden_dim) for _ in range(num_layers)])
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, data):
        """data: dict with keys 'x', 'edge_index' (single mol), or
        'x', 'edge_index', 'batch' (mini-batch collated).
        """
        x = data["x"]
        edge_index = data["edge_index"]
        batch = data.get("batch", torch.zeros(x.size(0), dtype=torch.long, device=x.device))

        x = F.relu(self.atom_embed(x))
        for conv in self.convs:
            x = F.relu(conv(x, edge_index))
        x = _global_mean_pool(x, batch)
        return self.head(x).squeeze(-1)


# ── Save / Load ──

def save_gnn_model(model, path):
    torch.save(model.state_dict(), path)

def load_gnn_model(path, device="cpu"):
    model = SolubilityGNN()
    model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
    model.to(device)
    model.eval()
    return model


def transfer_backbone(model, pretrained_dict):
    """Load pre-trained backbone weights, skipping incompatible head layers."""
    model_dict = model.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items()
                       if k in model_dict and v.shape == model_dict[k].shape
                       and not k.startswith("head.")}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    n_loaded = len(pretrained_dict)
    n_total = len([k for k in model_dict if not k.startswith("head.")])
    return n_loaded, n_total
