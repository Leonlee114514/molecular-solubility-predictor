"""
GNNExplainer for DisSolve — edge & node-feature attribution for SolubilityGNN.

Learns a soft edge mask that identifies which bonds (and which atom features)
are most influential for a given prediction. Uses the same forward logic as
SolubilityGNN but injects edge weights during message passing.

Reference: "GNNExplainer: Generating Explanations for Graph Neural Networks"
           (Ying et al., NeurIPS 2019)
"""

import torch
import torch.nn.functional as F
import numpy as np
from rdkit import Chem


# ── Internal forward with edge weights (replicates SolubilityGNN.forward) ──

def _global_mean_pool(x, batch):
    """Average node embeddings per graph (same as gnn_model)."""
    num_graphs = batch.max().item() + 1
    out = x.new_zeros((num_graphs, x.size(1)))
    out = out.index_add_(0, batch, x)
    counts = batch.bincount(minlength=num_graphs).unsqueeze(1).float()
    return out / counts.clamp(min=1)


def _forward_with_edge_weights(model, x, edge_index, edge_weights=None):
    """Run SolubilityGNN forward pass with optional edge weights.

    Args:
        model: A SolubilityGNN instance.
        x: Node feature tensor (n_nodes, 37).
        edge_index: Edge index (2, n_edges).
        edge_weights: Optional per-edge weight (n_edges,) to scale messages.
                      Higher weight = more influence for that edge.

    Returns:
        Scalar logS prediction.
    """
    batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
    x = F.relu(model.atom_embed(x))
    for conv in model.convs:
        row, col = edge_index
        out = torch.zeros_like(x)
        if edge_weights is not None:
            # Scale each neighbor's contribution by its edge weight
            out = out.index_add(0, row, x[col] * edge_weights.unsqueeze(1))
        else:
            out = out.index_add(0, row, x[col])
        out = (1 + conv.eps) * x + out
        x = F.relu(conv.mlp(out) + x)  # residual connection
    x = _global_mean_pool(x, batch)
    return model.head(x).squeeze(-1)


# ── GNNExplainer ──

class GNNExplainer:
    """Edge & feature attribution for a trained SolubilityGNN.

    Usage:
        explainer = GNNExplainer(model, lr=0.01, epochs=300)
        result = explainer.explain(x, edge_index, edge_index_bonds)

    Result contains:
        - bond_importance: dict mapping (bond_idx) -> importance (0~1)
        - feature_importance: dict mapping feature_dim -> importance (0~1)
        - edge_mask_raw: per-edge learned mask values
        - elapsed: optimization time in seconds
    """

    # Default regularisation coefficients
    EDGE_ENTROPY_WEIGHT = 2.0
    EDGE_SIZE_WEIGHT = 0.05
    FEAT_ENTROPY_WEIGHT = 0.5
    FEAT_SIZE_WEIGHT = 0.01

    def __init__(self, model, lr=0.01, epochs=300, device="cpu"):
        self.model = model.eval().to(device)
        self.lr = lr
        self.epochs = epochs
        self.device = device
        # Freeze model parameters
        for p in self.model.parameters():
            p.requires_grad = False

    def explain(self, x, edge_index, num_edges=None,
                edge_entropy_weight=None, edge_size_weight=None,
                feat_entropy_weight=None, feat_size_weight=None,
                return_raw=False):
        """Run GNNExplainer and return bond-level importance.

        Args:
            x: Node features (n_nodes, 37) tensor on CPU.
            edge_index: Edge index (2, E) tensor on CPU.
            num_edges: Number of unique molecular bonds (for bond mapping).
                       If None, assumes every 2 rows = 1 bond (bidirectional).
            edge_entropy_weight: Regularisation strength (default: 2.0).
            edge_size_weight: Sparsity strength (default: 0.05).
            return_raw: If True, also return raw edge_mask and feature_mask.

        Returns:
            dict with keys:
                - bond_importance: list of (bond_idx, importance_0_to_1, atom_i, atom_j)
                - feature_importance: list of (dim_idx, importance_0_to_1)
                - elapsed: float seconds
                - edge_mask_raw (optional): full tensor
                - feature_mask_raw (optional): full tensor
        """
        import time
        t0 = time.time()

        # Move to device
        x = x.to(self.device)
        edge_index = edge_index.to(self.device)
        num_edges = edge_index.size(1)

        # ── Get original prediction ──
        with torch.no_grad():
            original_pred = _forward_with_edge_weights(self.model, x, edge_index)

        # ── Learnable masks ──
        edge_logits = torch.nn.Parameter(
            torch.zeros(num_edges, device=self.device)
        )
        # Feature mask: one weight per atom feature dimension (37)
        feat_logits = torch.nn.Parameter(
            torch.zeros(x.size(1), device=self.device)
        )

        optimizer = torch.optim.Adam([edge_logits, feat_logits], lr=self.lr)

        ew = edge_entropy_weight if edge_entropy_weight is not None else self.EDGE_ENTROPY_WEIGHT
        es = edge_size_weight if edge_size_weight is not None else self.EDGE_SIZE_WEIGHT
        fw = feat_entropy_weight if feat_entropy_weight is not None else self.FEAT_ENTROPY_WEIGHT
        fs = feat_size_weight if feat_size_weight is not None else self.FEAT_SIZE_WEIGHT

        best_loss = float("inf")
        best_edge_mask = None
        best_feat_mask = None

        for epoch in range(self.epochs):
            optimizer.zero_grad()

            edge_mask = torch.sigmoid(edge_logits)
            feat_mask = torch.sigmoid(feat_logits)

            # Apply feature mask: scale node features
            x_masked = x * feat_mask.unsqueeze(0)

            # Forward with edge mask
            pred = _forward_with_edge_weights(self.model, x_masked, edge_index, edge_mask)

            # ── Prediction loss (MSE) ──
            pred_loss = F.mse_loss(pred, original_pred.detach())

            # ── Regularisation ──
            # Edge entropy: encourage binary (0 or 1) edge masks
            edge_entropy = -(
                edge_mask * torch.log(edge_mask + 1e-8) +
                (1 - edge_mask) * torch.log(1 - edge_mask + 1e-8)
            ).mean()

            # Edge size: encourage sparsity
            edge_size = edge_mask.sum() / num_edges

            # Feature entropy
            feat_entropy = -(
                feat_mask * torch.log(feat_mask + 1e-8) +
                (1 - feat_mask) * torch.log(1 - feat_mask + 1e-8)
            ).mean()

            # Feature size
            feat_size = feat_mask.sum() / feat_mask.numel()

            loss = pred_loss + ew * edge_entropy + es * edge_size \
                            + fw * feat_entropy + fs * feat_size

            loss.backward()
            optimizer.step()

            if loss.item() < best_loss:
                best_loss = loss.item()
                best_edge_mask = torch.sigmoid(edge_logits).detach().cpu()
                best_feat_mask = torch.sigmoid(feat_logits).detach().cpu()

        t1 = time.time()
        elapsed = t1 - t0

        # ── Map bidirectional edges back to bonds ──
        # edge_index has (u,v) and (v,u) for each bond.
        # Average the two directions to get one importance per bond.
        edge_index_cpu = edge_index.cpu()
        bond_map = {}
        for i in range(edge_index_cpu.size(1)):
            u, v = int(edge_index_cpu[0, i]), int(edge_index_cpu[1, i])
            # Use (min, max) as bond key so (u,v) and (v,u) map to same key
            key = (min(u, v), max(u, v))
            if key not in bond_map:
                bond_map[key] = []
            bond_map[key].append(i)

        bond_importance = []
        for (a, b), eidxs in bond_map.items():
            imp = float(best_edge_mask[eidxs].mean())
            bond_importance.append((int(a), int(b), imp))

        # ── Feature importance ──
        feat_imp = [float(v) for v in best_feat_mask]

        result = {
            "bond_importance": bond_importance,       # [(atom_i, atom_j, importance), ...]
            "feature_importance": feat_imp,            # [imp_dim0, imp_dim1, ..., imp_dim36]
            "elapsed": elapsed,
        }
        if return_raw:
            result["edge_mask_raw"] = best_edge_mask
            result["feature_mask_raw"] = best_feat_mask

        return result

    @staticmethod
    def bond_importance_to_smarts_weights(mol, bond_importance, threshold=0.1):
        """Convert bond importance list to per-bond weight dict for RDKit highlighting.

        Args:
            mol: RDKit Mol object.
            bond_importance: List of (atom_i, atom_j, importance) tuples.
            threshold: Minimum importance to include.

        Returns:
            Dict mapping RDKit bond indices to importance weights.
            Bonds below threshold get weight 0.
        """
        bond_weights = {}
        for a, b, imp in bond_importance:
            if imp >= threshold:
                bond = mol.GetBondBetweenAtoms(a, b)
                if bond is not None:
                    bond_weights[bond.GetIdx()] = imp
        return bond_weights

    @staticmethod
    def get_top_bonds(bond_importance, top_k=5):
        """Return the top-K most important bonds.

        Returns:
            List of (atom_i, atom_j, importance) sorted descending by importance.
        """
        sorted_bonds = sorted(bond_importance, key=lambda x: x[2], reverse=True)
        return sorted_bonds[:top_k]

    @staticmethod
    def get_atom_importance_from_bonds(mol, bond_importance, threshold=0.05):
        """Aggregate bond importance to atom-level scores.

        Each atom's importance = sum of incident bond importances.

        Returns:
            Dict mapping atom_idx -> importance_score.
        """
        atom_imp = {}
        for a, b, imp in bond_importance:
            if imp < threshold:
                continue
            atom_imp[a] = atom_imp.get(a, 0.0) + imp
            atom_imp[b] = atom_imp.get(b, 0.0) + imp
        # Normalise
        max_val = max(atom_imp.values()) if atom_imp else 1.0
        if max_val > 0:
            for k in atom_imp:
                atom_imp[k] /= max_val
        return atom_imp
