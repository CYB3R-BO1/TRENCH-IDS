"""Label-free Task Signature Extraction for TCTRL.

Extracts a label-free task signature from incoming task data using
a frozen encoder. All normalization statistics are computed ONLY
on meta-train episodes to prevent data leakage.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import json
import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader

from trench_ids.constants import FLOW_RELATIONS
from trench_ids.cl.inference import load_checkpoint
from trench_ids.cl.train import load_split, sample_graphs
from trench_ids.model.rhgnn import RelationSpecificHeteroGNN
from trench_ids.vocab import load_vocab


@dataclass
class SignatureStats:
    """Normalization statistics computed from meta-train episodes."""
    # Embedding statistics
    emb_mean: torch.Tensor          # [64]
    emb_std: torch.Tensor           # [64]
    # Scalar feature statistics (min-max)
    rel_freq_min: torch.Tensor      # [5]
    rel_freq_max: torch.Tensor      # [5]
    node_count_min: torch.Tensor    # [4]
    node_count_max: torch.Tensor    # [4]
    # Covariance eigenspectrum (B1 only)
    eigval_min: Optional[torch.Tensor] = None
    eigval_max: Optional[torch.Tensor] = None
    
    def to_dict(self) -> dict:
        """Convert to JSON-serializable dict."""
        return {
            'emb_mean': self.emb_mean.tolist(),
            'emb_std': self.emb_std.tolist(),
            'rel_freq_min': self.rel_freq_min.tolist(),
            'rel_freq_max': self.rel_freq_max.tolist(),
            'node_count_min': self.node_count_min.tolist(),
            'node_count_max': self.node_count_max.tolist(),
            'eigval_min': self.eigval_min.tolist() if self.eigval_min is not None else None,
            'eigval_max': self.eigval_max.tolist() if self.eigval_max is not None else None,
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'SignatureStats':
        """Create from dict."""
        return cls(
            emb_mean=torch.tensor(data['emb_mean']),
            emb_std=torch.tensor(data['emb_std']),
            rel_freq_min=torch.tensor(data['rel_freq_min']),
            rel_freq_max=torch.tensor(data['rel_freq_max']),
            node_count_min=torch.tensor(data['node_count_min']),
            node_count_max=torch.tensor(data['node_count_max']),
            eigval_min=torch.tensor(data['eigval_min']) if data.get('eigval_min') else None,
            eigval_max=torch.tensor(data['eigval_max']) if data.get('eigval_max') else None,
        )


class TaskSignatureExtractor:
    """Extracts label-free task signatures from incoming task data.
    
    Uses a frozen encoder to extract embeddings, then computes
    distribution statistics. All normalization is fitted on meta-train
    episodes only to prevent data leakage.
    """
    
    def __init__(
        self,
        frozen_encoder: RelationSpecificHeteroGNN,
        device: torch.device,
        batch_size: int = 32,
        max_graphs: int = 200,
    ):
        self.frozen_encoder = frozen_encoder
        self.device = device
        self.batch_size = batch_size
        self.max_graphs = max_graphs
        self.frozen_encoder.eval()
        
        # Will be set by fit()
        self.stats: Optional[SignatureStats] = None
    
    def extract_signature(
        self,
        graphs: list,
        include_covariance: bool = False,
    ) -> torch.Tensor:
        """Extract raw (unnormalized) signature from graphs.
        
        Args:
            graphs: List of HeteroData graphs from target task
            include_covariance: Whether to include covariance eigenspectrum (B1)
        
        Returns:
            Raw signature tensor (137 dims for B0, 164 for B1)
        """
        loader = DataLoader(graphs, batch_size=self.batch_size, shuffle=False)
        
        all_embeddings = []
        relation_counts = []
        node_counts = []
        
        with torch.no_grad():
            for batch in loader:
                batch = batch.to(self.device)
                output = self.frozen_encoder(batch)
                
                # Flow embeddings
                flow_emb = output.fused["flow"]
                all_embeddings.append(flow_emb.cpu())
                
                # Relation frequencies
                rel_counts = {}
                for edge_type, edge_index in batch.edge_index_dict.items():
                    rel = edge_type[1]
                    if rel in FLOW_RELATIONS:
                        rel_counts[rel] = rel_counts.get(rel, 0) + edge_index.size(1)
                relation_counts.append(torch.tensor([rel_counts.get(r, 0) for r in FLOW_RELATIONS]))
                
                # Node counts
                node_counts.append(torch.tensor([
                    batch["host"].num_nodes,
                    batch["protocol"].num_nodes,
                    batch["service"].num_nodes,
                    batch["port"].num_nodes,
                ]))
        
        # Aggregate embeddings
        all_emb = torch.cat(all_embeddings, dim=0)  # [N, 64]
        mean_emb = all_emb.mean(0)
        std_emb = all_emb.std(0)
        
        # Relation frequencies (normalized)
        rel_counts = torch.stack(relation_counts).sum(0).float()
        rel_freq = rel_counts / (rel_counts.sum() + 1e-8) if rel_counts.sum() > 0 else rel_counts
        
        # Node counts
        node_counts = torch.stack([c.float() for c in node_counts]).mean(0)
        
        # B0 signature
        signature = torch.cat([mean_emb, std_emb, rel_freq, node_counts])
        
        # B1: add covariance eigenspectrum
        if include_covariance:
            # Centered covariance
            centered = all_emb - mean_emb
            cov = (centered.T @ centered) / (all_emb.size(0) - 1)
            eigvals = torch.linalg.eigvalsh(cov).flatten()  # ascending
            eigvals = eigvals[-16:]  # largest 16
            signature = torch.cat([signature, eigvals])
        
        return signature
    
    def fit(self, meta_train_episodes: List[dict]) -> SignatureStats:
        """Compute normalization statistics from meta-train episodes only.
        
        Args:
            meta_train_episodes: List of meta-train episode dicts
        
        Returns:
            SignatureStats with normalization parameters
        """
        all_signatures = []
        
        for episode in meta_train_episodes:
            # Extract signature from episode's target support graphs
            graphs = episode['target_support_graphs']  # List of HeteroData
            sig = self.extract_signature(graphs, include_covariance=False)
            all_signatures.append(sig)
        
        all_sigs = torch.stack(all_signatures)  # [N, 137]
        
        # Embedding stats (mean/std)
        emb_mean = all_sigs[:, :64].mean(0)
        emb_std = all_sigs[:, :64].std(0)
        emb_std = torch.clamp(emb_std, min=1e-6)
        
        # Relation frequency stats (min-max)
        rel_freq = all_sigs[:, 128:133]
        rel_freq_min = rel_freq.min(0).values
        rel_freq_max = rel_freq.max(0).values
        rel_freq_max = torch.maximum(rel_freq_max, rel_freq_min + 1e-6)
        
        # Node count stats (min-max)
        node_counts = all_sigs[:, 133:137]
        node_min = node_counts.min(0).values
        node_max = node_counts.max(0).values
        node_max = torch.maximum(node_max, node_min + 1e-6)
        
        self.stats = SignatureStats(
            emb_mean=emb_mean,
            emb_std=emb_std,
            rel_freq_min=rel_freq_min,
            rel_freq_max=rel_freq_max,
            node_count_min=node_min,
            node_count_max=node_max,
        )
        
        return self.stats
    
    def normalize(self, signature: torch.Tensor) -> torch.Tensor:
        """Apply meta-train normalization to a signature.
        
        Args:
            signature: Raw signature [137] or [164]
        
        Returns:
            Normalized signature (same shape)
        """
        if self.stats is None:
            raise RuntimeError("Must call fit() before normalize()")
        
        s = self.stats
        normalized = signature.clone()
        
        # Normalize embeddings (z-score)
        normalized[:64] = (normalized[:64] - s.emb_mean) / s.emb_std
        normalized[64:128] = (normalized[64:128] - s.emb_mean) / s.emb_std
        
        # Normalize relation frequencies (min-max)
        rel = normalized[128:133]
        normalized[128:133] = (rel - s.rel_freq_min) / (s.rel_freq_max - s.rel_freq_min + 1e-8)
        
        # Normalize node counts (min-max)
        nc = normalized[133:137]
        normalized[133:137] = (nc - s.node_count_min) / (s.node_count_max - s.node_count_min + 1e-8)
        
        # Covariance eigenspectrum (if B1)
        if signature.size(0) > 137 and s.eigval_min is not None:
            eig = normalized[137:]
            normalized[137:] = (eig - s.eigval_min) / (s.eigval_max - s.eigval_min + 1e-8)
        
        return normalized
    
    def save_stats(self, path: Path):
        """Save normalization stats to file."""
        if self.stats is None:
            raise RuntimeError("No stats to save")
        path.write_text(json.dumps(self.stats.to_dict(), indent=2))
    
    def load_stats(self, path: Path):
        """Load normalization stats from file."""
        data = json.loads(path.read_text())
        self.stats = SignatureStats.from_dict(data)


def extract_task_signature(
    graphs: list,
    frozen_encoder: RelationSpecificHeteroGNN,
    device: torch.device,
    stats: Optional[SignatureStats] = None,
    include_covariance: bool = False,
    batch_size: int = 32,
    max_graphs: int = 200,
) -> torch.Tensor:
    """Convenience function to extract and normalize a task signature.
    
    Args:
        graphs: Target task graphs
        frozen_encoder: Frozen backbone encoder
        device: Device
        stats: Normalization stats (if None, returns raw signature)
        include_covariance: Whether to include covariance eigenspectrum
        batch_size: Batch size
        max_graphs: Max graphs to use
    
    Returns:
        Normalized (if stats provided) or raw signature tensor
    """
    extractor = TaskSignatureExtractor(frozen_encoder, device, batch_size, max_graphs)
    sig = extractor.extract_signature(graphs, include_covariance=include_covariance)
    
    if stats is not None:
        # Apply normalization
        extractor.stats = stats
        sig = extractor.normalize(sig)
    
    return sig


if __name__ == "__main__":
    # Quick test
    print("TaskSignature module loaded successfully")