#!/usr/bin/env python3
"""Train a basal-melt emulator.

Trains one architecture, one seed, on the cavity graphs built by
``02_build_graphs.py``, and writes a checkpoint plus a metrics record.

Held-out skill is reported against two baselines, because R2 against a raw melt
field is flattering on its own: melt varies over two orders of magnitude between
the cold and warm cavities, so predicting the per-shelf mean already scores well.
The baselines are the global mean and a per-shelf mean, and the emulator has to
beat both to have learned anything spatial.

Examples
--------
    python scripts/03_train.py --arch gat --seed 0 --epochs 5      # smoke test
    python scripts/03_train.py --arch gat --seed 0 --split shelf
    python scripts/03_train.py --arch egcn --split scenario --test-scenario SMITH_bi646
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from aisgnn.config import (                                        # noqa: E402
    EDGE_FEATURES,
    GRAPH_DIR,
    NODE_FEATURES,
    TRAIN_DIR,
    ensure_dirs,
)
from aisgnn.data.dataset import (                                  # noqa: E402
    index_graphs,
    load_batch,
    make_split,
    stack_features,
)
from aisgnn.models.architectures import ModelConfig, build_model   # noqa: E402


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #

def metrics(pred: np.ndarray, truth: np.ndarray,
            shelf_ids: np.ndarray | None = None) -> dict:
    """Skill of ``pred`` against ``truth``, with baselines for context."""
    err = pred - truth
    rmse = float(np.sqrt(np.mean(err ** 2)))
    mae = float(np.mean(np.abs(err)))
    bias = float(np.mean(err))
    ss_res = float(np.sum(err ** 2))
    ss_tot = float(np.sum((truth - truth.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    out = {"rmse": rmse, "mae": mae, "bias": bias, "r2": r2,
           "rmse_baseline_mean": float(np.sqrt(np.mean((truth - truth.mean()) ** 2)))}

    if shelf_ids is not None:
        # Per-shelf mean: the score to beat before claiming spatial skill.
        per_shelf = np.empty_like(truth)
        for s in np.unique(shelf_ids):
            m = shelf_ids == s
            per_shelf[m] = truth[m].mean()
        out["rmse_baseline_shelfmean"] = float(
            np.sqrt(np.mean((truth - per_shelf) ** 2)))
        out["skill_vs_shelfmean"] = float(
            1.0 - rmse / out["rmse_baseline_shelfmean"]
            if out["rmse_baseline_shelfmean"] > 0 else np.nan)
    return out


@torch.no_grad()
def evaluate(model, graphs, shelf_ids) -> dict:
    model.eval()
    preds, truths = [], []
    for data in graphs:
        preds.append(model.denormalise(model(data)).cpu().numpy())
        truths.append(data.y.cpu().numpy())
    return metrics(np.concatenate(preds), np.concatenate(truths), shelf_ids)


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #

def train(model, train_graphs, val_graphs, val_ids, epochs: int, lr: float,
          weight_decay: float, patience: int, device: str,
          clip: float = 1.0) -> dict:
    """Train with early stopping on validation RMSE."""
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=8)
    loss_fn = torch.nn.HuberLoss(delta=1.0)   # robust to the extreme melt tail

    best = {"rmse": float("inf")}
    best_state = None
    history = []
    since_improved = 0
    order = np.arange(len(train_graphs))

    for epoch in range(1, epochs + 1):
        model.train()
        np.random.shuffle(order)
        total = 0.0
        for i in order:
            data = train_graphs[i]
            opt.zero_grad(set_to_none=True)
            pred = model(data)
            target = (data.y - model.target_mean) / model.target_scale
            loss = loss_fn(pred, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            opt.step()
            total += float(loss)

        val = evaluate(model, val_graphs, val_ids)
        sched.step(val["rmse"])
        history.append({"epoch": epoch, "train_loss": total / len(order), **val})

        if val["rmse"] < best["rmse"] - 1e-6:
            best = val
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            since_improved = 0
        else:
            since_improved += 1

        if epoch % 10 == 0 or epoch == 1:
            log(f"  epoch {epoch:4d}  loss {total / len(order):.4f}  "
                f"val RMSE {val['rmse']:.4f}  R2 {val['r2']:.3f}")

        if since_improved >= patience:
            log(f"  early stop at epoch {epoch} (no improvement for {patience})")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    return {"best_val": best, "history": history}


def shelf_id_array(graphs) -> np.ndarray:
    """Per-node shelf label, for the per-shelf baseline."""
    ids = []
    for i, data in enumerate(graphs):
        ids.append(np.full(data.y.shape[0], i))
    return np.concatenate(ids) if ids else np.empty(0, int)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--arch", default="gat", choices=["mlp", "gcn", "gat", "egcn"])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--split", default="shelf",
                   choices=["shelf", "year", "scenario", "random"])
    p.add_argument("--test-scenario", default=None)
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--patience", type=int, default=40)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-5)
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--layers", type=int, default=4)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--graphs", type=Path, default=GRAPH_DIR)
    p.add_argument("--outdir", type=Path, default=None)
    p.add_argument("--max-graphs", type=int, default=None,
                   help="cap the number of graphs, for smoke tests")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    ensure_dirs()
    outdir = args.outdir or TRAIN_DIR
    outdir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    records = index_graphs(args.graphs)
    if not records:
        raise SystemExit(f"no graphs in {args.graphs}; run scripts/02_build_graphs.py")
    if args.max_graphs:
        records = records[:args.max_graphs]

    split = make_split(records, mode=args.split, seed=args.seed,
                       test_scenario=args.test_scenario)
    log(split.summary())
    if not split.train or not split.val or not split.test:
        raise SystemExit("split produced an empty partition; too few graphs")

    log(f"loading graphs onto {args.device}")
    train_graphs = load_batch(split.train, args.device)
    val_graphs = load_batch(split.val, args.device)
    test_graphs = load_batch(split.test, args.device)

    cfg = ModelConfig(in_channels=len(NODE_FEATURES),
                      edge_channels=len(EDGE_FEATURES),
                      hidden=args.hidden, layers=args.layers, heads=args.heads,
                      dropout=args.dropout)
    model = build_model(args.arch, cfg).to(args.device)

    x_train, y_train = stack_features(split.train)
    model.standardiser.fit(torch.as_tensor(x_train, dtype=torch.float32,
                                           device=args.device))
    model.fit_target(torch.as_tensor(y_train, dtype=torch.float32,
                                     device=args.device))
    log(f"{args.arch}: {model.n_parameters()} parameters, "
        f"{len(train_graphs)} training graphs, "
        f"{sum(int(g.y.shape[0]) for g in train_graphs)} nodes")

    t0 = time.time()
    result = train(model, train_graphs, val_graphs, shelf_id_array(val_graphs),
                   epochs=args.epochs, lr=args.lr, weight_decay=args.weight_decay,
                   patience=args.patience, device=args.device)
    elapsed = time.time() - t0

    test = evaluate(model, test_graphs, shelf_id_array(test_graphs))
    log(f"test RMSE {test['rmse']:.4f} m/yr  R2 {test['r2']:.3f}  "
        f"vs shelf-mean baseline {test.get('rmse_baseline_shelfmean', float('nan')):.4f}")

    tag = f"{args.arch}_{args.split}_seed{args.seed}"
    torch.save({"arch": args.arch, "cfg": vars(cfg), "seed": args.seed,
                "state_dict": model.state_dict()}, outdir / f"{tag}.pt")
    (outdir / f"{tag}.json").write_text(json.dumps({
        "arch": args.arch, "seed": args.seed, "split": args.split,
        "test_scenario": args.test_scenario,
        "n_parameters": model.n_parameters(),
        "n_train": len(split.train), "n_val": len(split.val), "n_test": len(split.test),
        "split_note": split.note,
        "epochs_run": len(result["history"]), "seconds": elapsed,
        "val": result["best_val"], "test": test,
        "history": result["history"],
    }, indent=2))

    log(f"wrote {outdir / tag}.pt and .json ({elapsed:.0f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
