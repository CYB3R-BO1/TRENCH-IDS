"""Episode Generator for TCTRL Meta-Learning.

Generates stratified, flow-disjoint meta-learning episodes with full
manifest recording for reproducibility.
"""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import torch
from torch_geometric.data import HeteroData
from torch_geometric.loader import DataLoader

from trench_ids.cl.task_signature import TaskSignatureExtractor
from trench_ids.cl.train import load_split
from trench_ids.labels import attack_classes_for_task, canonical_classes
from trench_ids.model.rhgnn import RelationSpecificHeteroGNN


@dataclass
class EpisodeManifest:
    """Manifest for a single meta-learning episode."""
    episode_id: str
    split: str  # 'train', 'val', 'test'
    source_task: int
    target_task: int
    source_dataset: str
    target_dataset: str
    source_flow_ids_hash: str
    target_support_ids_hash: str
    target_query_ids_hash: str
    seed: int
    source_flow_ids: List[str] = field(default_factory=list)
    target_support_ids: List[str] = field(default_factory=list)
    target_query_ids: List[str] = field(default_factory=list)
    target_query_ids: List[str] = field(default_factory=list)


class EpisodeGenerator:
    """Generates stratified, flow-disjoint meta-learning episodes.
    
    Episodes are constructed from the training corpus with explicit
    flow_id separation to prevent data leakage. Episodes are stratified
    by (source_class, target_class, source_dataset, target_dataset) pairs.
    
    Each episode consists of:
    - Source support: flows from source class
    - Target support: flows from target class (adaptation)
    - Target query: flows from target class (evaluation)
    
    All flow_ids are disjoint across source/target_support/target_query.
    """
    
    def __init__(
        self,
        graphs_dir: Path,
        seed: int = 42,
        max_support_per_class: int = 64,
        max_query_per_class: int = 64,
    ):
        self.graphs_dir = graphs_dir
        self.seed = seed
        self.max_support_per_class = max_support_per_class
        self.max_query_per_class = max_query_per_class
        self.rng = random.Random(seed)
        
        # Cache of class -> flow_ids
        self._class_to_flow_ids: Dict[str, List[str]] = {}
        self._flow_id_to_graph: Dict[str, HeteroData] = {}
        self._class_to_dataset: Dict[str, str] = {}
        self._initialized = False
    
    def initialize(self, excluded_flow_ids: Optional[Set[str]] = None):
        """Load and cache flow_ids per class from the training corpus.
        
        Args:
            excluded_flow_ids: Set of flow_ids to exclude (e.g., T3/T4 eval flows for B0)
        """
        if self._initialized:
            return
        
        self._class_to_flow_ids = {}
        self._flow_id_to_graph = {}
        self._class_to_dataset = {}
        excluded = excluded_flow_ids or set()
        
        # Load graphs from all task train splits
        for task in range(1, 7):
            task_graphs = load_split(self.graphs_dir, task, "train")
            # Get attack classes for this task (excluding Benign)
            task_attack_classes = attack_classes_for_task(task)
            
            for graph in task_graphs:
                flow_id = self._get_flow_id(graph)
                if flow_id in excluded:
                    continue
                
                # Extract dataset from flow_id prefix (e.g., "ToN-", "CSE-", "BoT-")
                dataset = flow_id.split('-')[0] if '-' in flow_id else "unknown"
                
                # Graph contains Benign + task's attack classes
                # Assign flow_id to each attack class in this task
                for class_name in task_attack_classes:
                    if class_name not in self._class_to_flow_ids:
                        self._class_to_flow_ids[class_name] = []
                    self._class_to_flow_ids[class_name].append(flow_id)
                    self._flow_id_to_graph[flow_id] = graph
                    self._class_to_dataset[flow_id] = dataset
        
        # Verify we have enough flows per class
        for cls, ids in self._class_to_flow_ids.items():
            if len(ids) < self.max_support_per_class + self.max_query_per_class:
                print(f"Warning: Class {cls} has only {len(ids)} flows, "
                      f"need {self.max_support_per_class + self.max_query_per_class}")
        
        self._initialized = True
        print(f"Initialized EpisodeGenerator with {len(self._class_to_flow_ids)} classes, "
              f"{sum(len(v) for v in self._class_to_flow_ids.values())} total flows")
    
    def _get_flow_id(self, graph: HeteroData) -> str:
        """Extract flow_id from graph."""
        # Try to get from graph attributes
        if hasattr(graph, 'flow_id'):
            return graph.flow_id
        # Try from flow node attributes
        if 'flow_id' in graph['flow']:
            return graph['flow'].flow_id[0].item() if hasattr(graph['flow'].flow_id[0], 'item') else str(graph['flow'].flow_id[0])
        # Fallback: generate from graph hash
        return hashlib.md5(str(graph).encode()).hexdigest()[:16]
    
    def _hash_ids(self, ids: List[str]) -> str:
        """Create deterministic hash of flow IDs for manifest."""
        sorted_ids = sorted(ids)
        return hashlib.sha256(','.join(sorted_ids).encode()).hexdigest()[:16]
    
    def generate_episodes(
        self,
        n_train: int = 400,
        n_val: int = 100,
        n_test: int = 0,
        min_flows_per_episode: int = 100,
    ) -> Tuple[List[Dict], List[Dict], List[Dict]]:
        """Generate stratified, flow-disjoint episodes.
        
        Args:
            n_train: Number of meta-train episodes
            n_val: Number of meta-val episodes
            n_test: Number of meta-test episodes (usually 0 for B0)
            min_flows_per_episode: Minimum total flows per episode
        
        Returns:
            (train_episodes, val_episodes, test_episodes)
            Each is a list of episode dicts with graphs and manifest
        """
        if not self._initialized:
            self.initialize()
        
        # Use TASK-LEVEL episodes (source task -> target task)
        all_tasks = list(range(1, 7))
        
        # Exclude the B0 evaluation task transition from meta-training
        # For B0 (T3->T4), we exclude T3 and T4 from meta-training
        excluded_tasks = {3, 4}  # T3 and T4 excluded for B0
        
        valid_task_pairs = []
        for src_task in range(1, 7):
            for tgt_task in range(1, 7):
                if src_task != tgt_task and src_task not in {3, 4} and tgt_task not in {3, 4}:
                    valid_task_pairs.append((src_task, tgt_task))
        
        if not valid_task_pairs:
            raise ValueError("No valid source-target task pairs")
        
        # Get the primary dataset for each task (from flow_id prefixes)
        task_to_dataset = {}
        for task in range(1, 7):
            if task in {3, 4}:  # Excluded for B0
                continue
            # Get dataset from first available flow_id of this task's classes
            for class_name in attack_classes_for_task(task):
                ids = self._class_to_flow_ids.get(class_name, [])
                if ids:
                    dataset = ids[0].split('-')[0] if '-' in ids[0] else 'unknown'
                    task_to_dataset[task] = dataset
                    break
        
        # Create task pairs with their actual datasets
        all_pairs = []
        for src_task, tgt_task in valid_task_pairs:
            src_ds = task_to_dataset.get(src_task)
            tgt_ds = task_to_dataset.get(tgt_task)
            if src_ds and tgt_ds:
                all_pairs.append((src_task, tgt_task, src_ds, tgt_ds))
        
        if not all_pairs:
            raise ValueError("No valid source-target task pairs with datasets")
        
        # Generate episodes
        episodes = {'train': [], 'val': [], 'test': []}
        split_counts = {'train': n_train, 'val': n_val, 'test': n_test}
        
        for split, count in split_counts.items():
            for i in range(count):
                # Select pair (cycle through for balance)
                pair_idx = (i * 17) % len(all_pairs)  # Prime for dispersion
                src_task, tgt_task, src_ds, tgt_ds = all_pairs[pair_idx]
                
                # Get all attack classes for source and target tasks
                src_classes = attack_classes_for_task(src_task)
                tgt_classes = attack_classes_for_task(tgt_task)
                
                # Get all flow_ids for source task (from all its attack classes) - filter by dataset
                src_ids = []
                for cls in src_classes:
                    src_ids.extend([fid for fid in self._class_to_flow_ids.get(cls, []) 
                                   if self._class_to_dataset[fid] == src_ds])
                
                # Get all flow_ids for target task - use ALL datasets for target (adapt to full target distribution)
                # Deduplicate flow_ids since multiple classes can share the same graph
                tgt_ids_set = set()
                for cls in tgt_classes:
                    tgt_ids_set.update(self._class_to_flow_ids.get(cls, []))
                tgt_ids = list(tgt_ids_set)
                
                if len(src_ids) < min_flows_per_episode or len(tgt_ids) < 2 * min_flows_per_episode:
                    # Fallback: use any available
                    src_ids = []
                    for cls in src_classes:
                        src_ids.extend(self._class_to_flow_ids.get(cls, []))
                    tgt_ids = []
                    for cls in tgt_classes:
                        tgt_ids.extend(self._class_to_flow_ids.get(cls, []))
                
                if len(src_ids) < min_flows_per_episode or len(tgt_ids) < 2 * min_flows_per_episode:
                    raise ValueError(f"Not enough flows for task {src_task}->{tgt_task}: "
                                   f"src={len(src_ids)}, tgt={len(tgt_ids)}")
                
                # Sample disjoint sets (per-episode disjointness)
                self.rng.shuffle(src_ids)
                self.rng.shuffle(tgt_ids)
                
                # Ensure source and target IDs are disjoint
                src_set = set(src_ids)
                tgt_ids = [fid for fid in tgt_ids if fid not in src_set]
                
                # Check we have enough for support + query
                required_tgt = self.max_support_per_class + self.max_query_per_class
                if len(src_ids) < self.max_support_per_class or len(tgt_ids) < self.max_support_per_class + self.max_query_per_class:
                    raise ValueError(f"Not enough disjoint flows for task {src_task}->{tgt_task}: "
                                   f"src={len(src_ids)}, tgt={len(tgt_ids)}")
                
                # Slice source and target
                src_ids = src_ids[:self.max_support_per_class]
                tgt_sup_ids = tgt_ids[:self.max_support_per_class]
                tgt_qry_ids = tgt_ids[self.max_support_per_class:self.max_support_per_class + self.max_query_per_class]
                
                # Verify disjointness (per-episode)
                assert set(src_ids).isdisjoint(tgt_sup_ids), "Source and target support overlap!"
                assert set(src_ids).isdisjoint(tgt_qry_ids), "Source and target query overlap!"
                assert set(tgt_sup_ids).isdisjoint(tgt_qry_ids), "Target support and query overlap!"
                
                # Load graphs
                src_graphs = [self._flow_id_to_graph[fid] for fid in src_ids]
                tgt_sup_graphs = [self._flow_id_to_graph[fid] for fid in tgt_sup_ids]
                tgt_qry_graphs = [self._flow_id_to_graph[fid] for fid in tgt_qry_ids]
                
                # Create manifest
                manifest = EpisodeManifest(
                    episode_id=f"{split}_{i:04d}",
                    split=split,
                    source_task=src_task,
                    target_task=tgt_task,
                    source_dataset=src_ds,
                    target_dataset=tgt_ds,
                    source_flow_ids_hash=self._hash_ids(src_ids),
                    target_support_ids_hash=self._hash_ids(tgt_sup_ids),
                    target_query_ids_hash=self._hash_ids(tgt_qry_ids),
                    seed=self.seed + i,
                    source_flow_ids=src_ids,
                    target_support_ids=tgt_sup_ids,
                    target_query_ids=tgt_qry_ids,
                )
                
                episodes[split].append({
                    'manifest': manifest,
                    'source_support': src_graphs,
                    'target_support': tgt_sup_graphs,
                    'target_query': tgt_qry_graphs,
                })
        
        print(f"Generated {len(episodes['train'])} train, {len(episodes['val'])} val, {len(episodes['test'])} test episodes")
        return episodes['train'], episodes['val'], episodes['test']
    
    def save_manifests(self, episodes: Dict[str, List[Dict]], output_dir: Path):
        """Save episode manifests to JSON."""
        output_dir.mkdir(parents=True, exist_ok=True)
        for split, episodes_list in episodes.items():
            manifest_path = output_dir / f"episodes_{split}_manifest.json"
            data = [e['manifest'].__dict__ for e in episodes_list]
            manifest_path.write_text(json.dumps(data, indent=2))
    
    def load_episodes(self, episodes_data: List[Dict], checkpoint_dir: Optional[Path] = None) -> List[Dict]:
        """Load episode graphs from manifest data."""
        episodes = []
        for e in episodes_data:
            # episodes_data contains the manifest dicts directly (saved by save_manifests)
            manifest = e if isinstance(e, EpisodeManifest) else EpisodeManifest(**e)
            src_graphs = [self._flow_id_to_graph[fid] for fid in manifest.source_flow_ids]
            tgt_sup_graphs = [self._flow_id_to_graph[fid] for fid in manifest.target_support_ids]
            tgt_qry_graphs = [self._flow_id_to_graph[fid] for fid in manifest.target_query_ids]

            episode = {
                'manifest': manifest,
                'source_support': src_graphs,
                'target_support': tgt_sup_graphs,
                'target_query': tgt_qry_graphs,
            }
            if checkpoint_dir is not None:
                episode['source_checkpoint'] = str(checkpoint_dir / f"checkpoint_task_{manifest.source_task}.pt")
            episodes.append(episode)
        return episodes


def generate_b0_meta_data(
    graphs_dir: Path,
    t3_t4_eval_flow_ids: Set[str],
    n_meta_train: int = 400,
    n_meta_val: int = 100,
    seed: int = 42,
    output_dir: Path = Path("runs/meta_transfer/B0_poc/meta_episodes"),
) -> Tuple[List[Dict], List[Dict]]:
    """Generate B0 meta-training and meta-validation episodes.
    
    Args:
        graphs_dir: Path to graphs directory
        t3_t4_eval_flow_ids: Set of flow_ids to exclude (T3/T4 evaluation)
        n_meta_train: Number of meta-train episodes
        n_meta_val: Number of meta-val episodes
        seed: Random seed
        output_dir: Output directory for episodes
    
    Returns:
        (meta_train_episodes, meta_val_episodes)
    """
    generator = EpisodeGenerator(graphs_dir, seed=seed)
    generator.initialize(excluded_flow_ids=t3_t4_eval_flow_ids)
    
    train_eps, val_eps, _ = generator.generate_episodes(
        n_train=n_meta_train,
        n_val=n_meta_val,
        n_test=0,
    )
    
    output_dir.mkdir(parents=True, exist_ok=True)
    generator.save_manifests({'train': train_eps, 'val': val_eps, 'test': []}, output_dir)
    
    return train_eps, val_eps


def load_meta_episodes(
    episodes_dir: Path,
    generator: EpisodeGenerator,
    split: str,
    checkpoint_dir: Optional[Path] = None,
) -> List[Dict]:
    """Load meta episodes from saved manifests."""
    manifest_path = episodes_dir / f"episodes_{split}_manifest.json"
    data = json.loads(manifest_path.read_text())
    return generator.load_episodes(data, checkpoint_dir=checkpoint_dir)


if __name__ == "__main__":
    print("EpisodeGenerator module loaded successfully")