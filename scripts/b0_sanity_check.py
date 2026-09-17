"""B0 POC Pre-Training Sanity Check.

Run this BEFORE the full 20-epoch training to verify:

  1. Identity path   -- adapter at zero-init produces identical logits to baseline
  2. Normalization   -- emb_std not collapsed (>0.01) with 400 meta-train episodes
  3. Oracle sanity   -- oracle reaches >=90% acc on T4 test after 20 supervised epochs
  4. FOMAML grads    -- at least one adapter grad is non-zero after one episode

Usage:
    uv run python scripts/b0_sanity_check.py \
        --checkpoint-dir runs/gnn_3layer_residual_s42 \
        --graphs-dir data/graphs \
        --episodes-dir runs/meta_transfer/B0_poc/meta_episodes \
        --device cuda
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from trench_ids.cl.device import resolve_device
from trench_ids.cl.episode_generator import EpisodeGenerator, load_meta_episodes
from trench_ids.cl.fomaml import run_fomaml_episode
from trench_ids.cl.inference import load_checkpoint
from trench_ids.cl.task_signature import extract_task_signature
from trench_ids.cl.train import load_split
from trench_ids.model.transfer_adapter import (
    TransferAdapter, LightweightClassifierHead, fuse_adapted
)

PASS = "\033[92m  PASS\033[0m"
FAIL = "\033[91m  FAIL\033[0m"


def _get_last_layer_outputs(model, batch):
    model.eval()
    with torch.no_grad():
        x_dict = model.encoders(batch)
        edge_index_dict = batch.edge_index_dict
        prev_fused = x_dict
        for i, layer in enumerate(model.layers):
            out = layer(prev_fused, edge_index_dict)
            if i == len(model.layers) - 2:
                residual = out.fused["flow"]
            prev_fused = out.fused
        relation_embeds = out.relations["flow"]
        fusion_module = model.layers[-1].fusion["flow"]
    return relation_embeds, fusion_module, residual


def check_identity_path(model, adapter, graphs_dir, device, batch_size=32):
    print("\n[1] Identity path check ...")
    test_graphs = load_split(graphs_dir, 4, "test")[:50]
    head = LightweightClassifierHead(64, 11).to(device)
    dummy_sig = torch.zeros(137, device=device)

    mismatches = 0
    max_diff = 0.0
    loader = DataLoader(test_graphs, batch_size=batch_size, shuffle=False)

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            baseline_out = model(batch)
            baseline_fused = baseline_out.fused["flow"]
            rel_emb, fus_mod, residual = _get_last_layer_outputs(model, batch)
            adapted = adapter(rel_emb, dummy_sig)
            adapter_fused = fuse_adapted(adapted, fusion_module=fus_mod, residual=residual)
            diff = (baseline_fused - adapter_fused).abs().max().item()
            max_diff = max(max_diff, diff)
            base_logits = head(baseline_fused)
            adapt_logits = head(adapter_fused)
            mismatches += (base_logits.argmax(-1) != adapt_logits.argmax(-1)).sum().item()

    ok = mismatches == 0 and max_diff < 1e-3
    tag = PASS if ok else FAIL
    print(f"  max |fused diff|      = {max_diff:.2e}  (threshold: 1e-3)")
    print(f"  prediction mismatches = {mismatches}  (threshold: 0)")
    print(f"{tag}")
    return ok


def check_normalization(model, episodes_dir, generator, device, batch_size=32):
    print("\n[2] Normalization health check ...")
    episodes = load_meta_episodes(episodes_dir, generator, "train")
    n = len(episodes)
    print(f"  Loaded {n} meta-train episodes")

    all_sigs = []
    for idx, episode in enumerate(episodes):
        if idx % 100 == 0:
            print(f"    extracting signature {idx+1}/{n} ...")
        sig = extract_task_signature(
            episode["target_support"], model, device,
            stats=None, include_covariance=False, batch_size=batch_size
        )
        all_sigs.append(sig)

    sigs = torch.stack(all_sigs)
    emb_std = sigs[:, :64].std(0, unbiased=False)
    min_std = emb_std.min().item()
    mean_std = emb_std.mean().item()

    ok = min_std > 0.01
    tag = PASS if ok else FAIL
    print(f"  emb_std.min()  = {min_std:.4f}  (threshold: > 0.01)")
    print(f"  emb_std.mean() = {mean_std:.4f}")
    print(f"{tag}")
    return ok, sigs


def check_oracle(model, adapter, device, graphs_dir, batch_size=32, n_epochs=20):
    print("\n[3] Oracle sanity check ...")
    t4_support = load_split(graphs_dir, 4, "train")
    t4_test = load_split(graphs_dir, 4, "test")
    head = LightweightClassifierHead(64, 11).to(device)
    dummy_sig = torch.zeros(137, device=device)
    oracle_opt = torch.optim.Adam(
        list(adapter.parameters()) + list(head.parameters()), lr=1e-3
    )
    support_loader = DataLoader(t4_support, batch_size=batch_size, shuffle=True)

    model.eval()
    for epoch in range(n_epochs):
        adapter.train(); head.train()
        for batch in support_loader:
            batch = batch.to(device)
            oracle_opt.zero_grad()
            with torch.no_grad():
                rel_emb, fus_mod, residual = _get_last_layer_outputs(model, batch)
            adapted = adapter(rel_emb, dummy_sig)
            fused = fuse_adapted(adapted, fusion_module=fus_mod, residual=residual)
            logits = head(fused)
            loss = F.cross_entropy(logits, batch["flow"].y)
            loss.backward()
            oracle_opt.step()

        if (epoch + 1) % 5 == 0:
            adapter.eval(); head.eval()
            correct = total = 0
            with torch.no_grad():
                for batch in DataLoader(t4_test, batch_size=batch_size):
                    batch = batch.to(device)
                    rel_emb, fus_mod, residual = _get_last_layer_outputs(model, batch)
                    adapted = adapter(rel_emb, dummy_sig)
                    fused = fuse_adapted(adapted, fusion_module=fus_mod, residual=residual)
                    pred = head(fused).argmax(-1)
                    correct += (pred == batch["flow"].y).sum().item()
                    total += batch["flow"].y.numel()
            acc = correct / total if total > 0 else 0.0
            print(f"    epoch {epoch+1}: acc={acc:.3f}")

    adapter.eval(); head.eval()
    correct = total = 0
    with torch.no_grad():
        for batch in DataLoader(t4_test, batch_size=batch_size):
            batch = batch.to(device)
            rel_emb, fus_mod, residual = _get_last_layer_outputs(model, batch)
            adapted = adapter(rel_emb, dummy_sig)
            fused = fuse_adapted(adapted, fusion_module=fus_mod, residual=residual)
            pred = head(fused).argmax(-1)
            correct += (pred == batch["flow"].y).sum().item()
            total += batch["flow"].y.numel()

    final_acc = correct / total if total > 0 else 0.0
    ok = final_acc >= 0.90
    tag = PASS if ok else FAIL
    print(f"  Oracle final_acc = {final_acc:.4f}  (threshold: >= 0.90)")
    print(f"{tag}")
    return ok, final_acc


def check_fomaml_gradients(model, checkpoint_dir, generator, episodes_dir, device):
    print("\n[4] FOMAML gradient flow check ...")
    episodes = load_meta_episodes(
        episodes_dir, generator, "train", checkpoint_dir=checkpoint_dir
    )
    if not episodes:
        print(f"  {FAIL}  (no episodes loaded)")
        return False

    episode = episodes[0]
    adapter = TransferAdapter(hidden_dim=64, adapter_dim=32).to(device)
    head = LightweightClassifierHead(64, 11).to(device)
    dummy_sig = torch.zeros(137, device=device)

    meta_loss, meta_grads = run_fomaml_episode(
        model,
        adapter,
        head,
        episode["source_checkpoint"],
        episode["target_support"][:50],
        episode["target_query"][:50],
        dummy_sig,
        k_steps=1,
        alpha=0.01,
        batch_size=16,
        device=device,
    )

    non_zero_grads = sum(
        1 for g in meta_grads.values()
        if g is not None and g.abs().max().item() > 1e-10
    )
    total_grads = len(meta_grads)

    ok = non_zero_grads > 0
    tag = PASS if ok else FAIL
    print(f"  meta_loss             = {meta_loss:.6f}")
    print(f"  non-zero grad tensors = {non_zero_grads}/{total_grads}")
    if not ok:
        print("  *** All grads zero -- computation graph is broken. Check fomaml.py ***")
    print(f"{tag}")
    return ok


def main():
    parser = argparse.ArgumentParser(description="B0 POC pre-training sanity checks")
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("runs/gnn_3layer_residual_s42"))
    parser.add_argument("--graphs-dir", type=Path, default=Path("data/graphs"))
    parser.add_argument("--episodes-dir", type=Path, default=Path("runs/meta_transfer/B0_poc/meta_episodes"))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--skip-oracle", action="store_true", help="Skip the slow oracle check")
    args = parser.parse_args()

    device = resolve_device(args.device)
    print(f"Device: {device}", flush=True)

    checkpoint_path = args.checkpoint_dir / "checkpoint_task_3.pt"
    print(f"Loading checkpoint: {checkpoint_path}", flush=True)
    model, base_classifier, _ = load_checkpoint(checkpoint_path, args.graphs_dir, device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    adapter = TransferAdapter(hidden_dim=64, adapter_dim=32).to(device)

    print("\nInitializing EpisodeGenerator ...")
    generator = EpisodeGenerator(args.graphs_dir)
    generator.initialize()

    results = {}

    results["identity_path"] = check_identity_path(
        model, adapter, args.graphs_dir, device, args.batch_size
    )

    results["fomaml_gradients"] = check_fomaml_gradients(
        model, args.checkpoint_dir, generator, args.episodes_dir, device
    )

    norm_ok, all_sigs = check_normalization(
        model, args.episodes_dir, generator, device, args.batch_size
    )
    results["normalization"] = norm_ok
    emb_std = all_sigs[:, :64].std(0, unbiased=False)
    rel_std = all_sigs[:, 128:133].std(0, unbiased=False)
    print(f"  [debug] emb_std range: [{emb_std.min():.4f}, {emb_std.max():.4f}]")
    print(f"  [debug] rel_freq_std range: [{rel_std.min():.4f}, {rel_std.max():.4f}]")

    if not args.skip_oracle:
        oracle_adapter = TransferAdapter(hidden_dim=64, adapter_dim=32).to(device)
        oracle_ok, oracle_acc = check_oracle(
            model, oracle_adapter, device, args.graphs_dir, args.batch_size
        )
        results["oracle"] = oracle_ok
    else:
        print("\n[3] Oracle check SKIPPED (--skip-oracle)")
        results["oracle"] = None

    print("="*60, flush=True)
    print("SANITY CHECK SUMMARY", flush=True)
    print("="*60, flush=True)
    all_pass = True
    for name, ok in results.items():
        if ok is None:
            status = "  SKIP"
        elif ok:
            status = "  PASS"
        else:
            status = "  FAIL"
            all_pass = False
        print(f"  {name:<25s}: {status}", flush=True)
    print("="*60, flush=True)

    if all_pass:
        print("\nAll checks passed -- safe to run full 20-epoch B0 training.")
        print("  uv run python -m trench_ids.cl.train_meta train --task 3 --meta-epochs 20 --device cuda")
    else:
        print("\nOne or more checks FAILED -- fix before running full B0.")
        sys.exit(1)


if __name__ == "__main__":
    main()
