"""Dry-run verification: inspect relation-specific parameter mappings for the 3-layer model."""

from __future__ import annotations

from pathlib import Path

import torch

from trench_ids.model.rhgnn import RelationSpecificHeteroGNN
from trench_ids.vocab import load_vocab


FLOW_RELATIONS = ["originates", "terminated_by", "targeted_by", "protocol_of", "service_of"]


def get_relation_parameter_mapping(model: RelationSpecificHeteroGNN) -> dict[str, dict]:
    """Extract parameter mapping for each FLOW relation across all layers."""
    mapping = {}
    
    for relation in FLOW_RELATIONS:
        mapping[relation] = {
            "layers": [],
            "total_params": 0,
            "total_tensors": 0,
        }
    
    for name, param in model.named_parameters():
        # Parse relation-specific parameters
        # Pattern: layers.{i}.conv.{rel_lins|combine_lins}.{src}__{relation}__{dst}
        if "layers." in name and ".conv." in name:
            parts = name.split(".")
            if len(parts) >= 5:
                layer_idx = int(parts[1])
                param_type = parts[3]  # rel_lins or combine_lins
                edge_key = parts[4]    # src__relation__dst
                
                edge_parts = edge_key.split("__")
                if len(edge_parts) == 3:
                    src, relation, dst = edge_parts
                    if relation in FLOW_RELATIONS and dst == "flow":
                        if relation not in mapping:
                            mapping[relation] = {"layers": [], "total_params": 0, "total_tensors": 0}
                        
                        # Ensure layer entry exists
                        while len(mapping[relation]["layers"]) <= layer_idx:
                            mapping[relation]["layers"].append({
                                "rel_lins": {"weight": None, "bias": None},
                                "combine_lins": {"weight": None, "bias": None},
                            })
                        
                        layer_dict = mapping[relation]["layers"][layer_idx]
                        if param_type in layer_dict:
                            if "weight" in name:
                                layer_dict[param_type]["weight"] = (name, param.shape, param.numel())
                            else:
                                layer_dict[param_type]["bias"] = (name, param.shape, param.numel())
                        
                        mapping[relation]["total_params"] += param.numel()
                        mapping[relation]["total_tensors"] += 1
    
    return mapping


def print_mapping(mapping: dict[str, dict]) -> None:
    """Print formatted relation parameter mapping."""
    print("=" * 80)
    print("RELATION-SPECIFIC PARAMETER MAPPING (3-layer model)")
    print("=" * 80)
    
    for relation in FLOW_RELATIONS:
        info = mapping.get(relation, {"layers": [], "total_params": 0, "total_tensors": 0})
        print(f"\n{'='*80}")
        print(f"Relation: {relation}")
        print(f"Total tensors: {info['total_tensors']}, Total parameters: {info['total_params']:,}")
        print(f"{'='*80}")
        
        for layer_idx, layer in enumerate(info["layers"]):
            print(f"\n  Layer {layer_idx}:")
            for param_type in ["rel_lins", "combine_lins"]:
                params = layer[param_type]
                if params["weight"] or params["bias"]:
                    print(f"    {param_type}:")
                    if params["weight"]:
                        name, shape, count = params["weight"]
                        print(f"      weight: {name}")
                        print(f"        shape: {shape}, params: {count:,}")
                    if params["bias"]:
                        name, shape, count = params["bias"]
                        print(f"      bias:   {name}")
                        print(f"        shape: {shape}, params: {count:,}")
    
    # Summary table
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    print(f"{'Relation':<20} {'Layers':<8} {'Tensors':<10} {'Parameters':<15} {'Params/Layer':<15}")
    print("-" * 80)
    for relation in FLOW_RELATIONS:
        info = mapping.get(relation, {"layers": [], "total_params": 0, "total_tensors": 0})
        num_layers = len([l for l in info["layers"] if any(l["rel_lins"].values()) or any(l["combine_lins"].values())])
        params_per_layer = info["total_params"] // num_layers if num_layers else 0
        print(f"{relation:<20} {num_layers:<8} {info['total_tensors']:<10} {info['total_params']:<15,} {params_per_layer:<15,}")
    
    # Check structural identity
    print(f"\n{'='*80}")
    print("STRUCTURAL IDENTITY CHECK")
    print(f"{'='*80}")
    first_rel = FLOW_RELATIONS[0]
    first_info = mapping[first_rel]
    all_identical = True
    for relation in FLOW_RELATIONS[1:]:
        info = mapping[relation]
        if (info["total_params"] != first_info["total_params"] or 
            info["total_tensors"] != first_info["total_tensors"] or
            len(info["layers"]) != len(first_info["layers"])):
            all_identical = False
            print(f"  {relation} DIFFERS from {first_rel}: params={info['total_params']}, tensors={info['total_tensors']}, layers={len(info['layers'])}")
    
    if all_identical:
        print(f"  All 5 FLOW relations have IDENTICAL parameter structure:")
        print(f"    {first_info['total_tensors']} tensors, {first_info['total_params']:,} params, {len(first_info['layers'])} layers")
        print(f"    -> Can use shared structured control distribution")


def main():
    graphs_dir = Path("data/graphs")
    vocab = load_vocab(graphs_dir / "vocab.json")
    sample_graph = torch.load(graphs_dir / "task_1_train.pt", weights_only=False)[0]

    model = RelationSpecificHeteroGNN.from_graph(
        sample_graph,
        hidden_dim=64,
        protocol_vocab_size=len(vocab["PROTOCOL"]),
        service_vocab_size=len(vocab["L7_PROTO"]),
        num_layers=3,
        attn_dim=128,
        port_tail_buckets=32,
        fusion="attention",
        use_residual=True,
    )

    mapping = get_relation_parameter_mapping(model)
    print_mapping(mapping)


if __name__ == "__main__":
    main()