#!/usr/bin/env python3
"""Run the emulator analyses on the trained ensembles.

Covers, in order:

* **skill** -- collate every training run into a per-architecture, per-split
  table, against the per-shelf-mean baseline;
* **connectivity** -- the distance over which an upstream perturbation changes
  predicted melt, per shelf and per scenario, with the MLP baseline included as
  the zero-coupling control;
* **controls** -- interventional sensitivity of melt to each input, and how it
  changes between scenarios;
* **response** -- open-loop forcing sweeps with threshold detection, and
  closed-loop sweeps that can show hysteresis;
* **ice** -- the reduced flowline driven by an abrupt versus a gradual melt
  history.

Results are written to ``runs/analysis`` as JSON for the figure scripts.

Examples
--------
    python scripts/04_analyse.py --skill-only
    python scripts/04_analyse.py --shelves Filchner-Ronne Ross
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from aisgnn.config import (                                        # noqa: E402
    EDGE_FEATURES,
    GRAPH_DIR,
    NODE_FEATURES,
    RUN_DIR,
    TRAIN_DIR,
    ensure_dirs,
)
from aisgnn.coupling.flowline import FlowlineConfig, compare_histories  # noqa: E402
from aisgnn.data.dataset import index_graphs                       # noqa: E402
from aisgnn.dynsys.sweeps import (                                 # noqa: E402
    closed_loop_sweep,
    detect_threshold,
    phase_space,
    sweep,
)
from aisgnn.interpret.attention import sensitivity_length_scale    # noqa: E402
from aisgnn.interpret.intervention import control_shift, intervene  # noqa: E402
from aisgnn.models.architectures import ModelConfig, build_model   # noqa: E402


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load_model(path: Path, device: str):
    blob = torch.load(path, map_location=device, weights_only=False)
    cfg = ModelConfig(**blob["cfg"])
    model = build_model(blob["arch"], cfg).to(device)
    model.load_state_dict(blob["state_dict"])
    model.eval()
    return model, blob["arch"]


# --------------------------------------------------------------------------- #
# Skill
# --------------------------------------------------------------------------- #

def collate_skill(train_dir: Path) -> dict:
    """Summarise every completed training run."""
    runs = []
    for path in sorted(train_dir.glob("*.json")):
        try:
            runs.append(json.loads(path.read_text()))
        except json.JSONDecodeError:
            log(f"  unreadable: {path.name}")

    grouped: dict[tuple[str, str], list] = defaultdict(list)
    for r in runs:
        grouped[(r["arch"], r["split"])].append(r)

    summary = {}
    for (arch, split), group in sorted(grouped.items()):
        rmse = np.array([g["test"]["rmse"] for g in group])
        r2 = np.array([g["test"]["r2"] for g in group])
        base = np.array([g["test"].get("rmse_baseline_shelfmean", np.nan)
                         for g in group])
        summary[f"{arch}_{split}"] = {
            "arch": arch, "split": split, "n_seeds": len(group),
            "rmse_mean": float(rmse.mean()), "rmse_std": float(rmse.std(ddof=1))
            if len(group) > 1 else 0.0,
            "r2_mean": float(r2.mean()),
            "r2_std": float(r2.std(ddof=1)) if len(group) > 1 else 0.0,
            "baseline_rmse": float(np.nanmean(base)),
            "beats_baseline": bool(rmse.mean() < np.nanmean(base)),
            "epochs_mean": float(np.mean([g["epochs_run"] for g in group])),
            "seconds_mean": float(np.mean([g["seconds"] for g in group])),
        }
    return {"n_runs": len(runs), "by_arch_split": summary}


# --------------------------------------------------------------------------- #
# Analyses that need a model and a graph
# --------------------------------------------------------------------------- #

def pick_graphs(shelves: list[str], scenarios: tuple[str, ...],
                graph_dir: Path, device: str) -> dict:
    """One representative graph per (shelf, simulation), the latest year."""
    records = index_graphs(graph_dir)
    chosen: dict[tuple[str, str], object] = {}
    for r in records:
        if shelves and r.shelf not in shelves:
            continue
        if scenarios and r.simulation not in scenarios:
            continue
        key = (r.shelf, r.simulation)
        if key not in chosen or r.year > chosen[key].year:
            chosen[key] = r
    return {k: v.load().to_pyg().to(device) for k, v in chosen.items()}


def run_connectivity(models: dict, graphs: dict, n_sources: int) -> list[dict]:
    """Length scale of upstream influence, per architecture, shelf and scenario."""
    out = []
    for arch, model in models.items():
        for (shelf, sim), data in graphs.items():
            try:
                ls = sensitivity_length_scale(model, data, n_sources=n_sources)
            except (RuntimeError, ValueError) as exc:
                log(f"  {arch} {shelf} {sim}: {exc}")
                continue
            out.append({
                "arch": arch, "shelf": shelf, "simulation": sim,
                "length_scale_km": (float(ls.length_scale / 1e3)
                                    if np.isfinite(ls.length_scale) else None),
                "upper_bound_km": (float(ls.upper_bound / 1e3)
                                   if np.isfinite(ls.upper_bound) else None),
                "resolved": bool(ls.resolved),
                "n_bins_used": int(ls.n_bins_used),
                "r_squared": float(ls.r_squared) if np.isfinite(ls.r_squared) else None,
                "max_distance_km": float(ls.max_distance / 1e3),
                "distances_km": (ls.distances / 1e3).tolist(),
                "influence": np.nan_to_num(ls.influence, nan=0.0).tolist(),
                "note": ls.note,
            })
            if ls.resolved:
                log(f"  {arch:5s} {shelf:16s} {sim:12s} L = {ls.length_scale / 1e3:7.1f} km "
                    f"(R2 {ls.r_squared:.2f}, {ls.n_bins_used} bins)")
            else:
                log(f"  {arch:5s} {shelf:16s} {sim:12s} L unresolved, "
                    f"< {ls.upper_bound / 1e3:.1f} km -- {ls.note}")
    return out


def run_controls(models: dict, graphs: dict) -> dict:
    """Interventional sensitivity per feature, and its change between scenarios."""
    per_arch = {}
    for arch, model in models.items():
        by_key = {}
        for (shelf, sim), data in graphs.items():
            res = intervene(model, data)
            by_key[f"{shelf}|{sim}"] = {
                k: {"sensitivity": v["sensitivity"], "baseline": v["baseline_melt"]}
                for k, v in res.items()}
        per_arch[arch] = by_key

        # Scenario shift, averaged over shelves present in both runs.
        shifts = defaultdict(list)
        shelves = {k.split("|")[0] for k in by_key}
        for shelf in shelves:
            a = by_key.get(f"{shelf}|SMITH_bf663")
            b = by_key.get(f"{shelf}|SMITH_bi646")
            if a and b:
                for feature, delta in control_shift(
                        {k: {"sensitivity": v["sensitivity"]} for k, v in a.items()},
                        {k: {"sensitivity": v["sensitivity"]} for k, v in b.items()}
                ).items():
                    shifts[feature].append(delta)
        per_arch[f"{arch}_scenario_shift"] = {
            k: float(np.nanmean(v)) for k, v in shifts.items()}
    return per_arch


def run_response(models: dict, graphs: dict, offsets: np.ndarray,
                 closed_loop_arch: str | None = None) -> list[dict]:
    """Open-loop sweeps with threshold detection, plus a closed-loop sweep.

    The closed-loop sweep relaxes to a fixed point at every forcing step and so
    costs a few hundred forward passes per curve; it is run for one architecture
    only, chosen as the best-performing graph model.
    """
    out = []
    for arch, model in models.items():
        for (shelf, sim), data in graphs.items():
            res = sweep(model, data, "thermal_driving", offsets,
                        collect_embeddings=True)
            thr = detect_threshold(res)

            record = {
                "arch": arch, "shelf": shelf, "simulation": sim,
                "offsets": res.values.tolist(), "melt": res.melt.tolist(),
                "gradient": res.gradient.tolist(),
                "threshold_location": thr.location, "max_gradient": thr.max_gradient,
                "sharpness": thr.sharpness, "is_abrupt": bool(thr.is_abrupt),
                "melt_before": thr.melt_before, "melt_after": thr.melt_after,
                "mode": res.mode, "note": res.note,
            }

            if res.embeddings is not None and res.embeddings.ndim == 2:
                try:
                    scores, ratio = phase_space(res.embeddings)
                    record["phase_space"] = scores.tolist()
                    record["explained_variance"] = ratio.tolist()
                except ValueError:
                    pass

            if closed_loop_arch is not None and arch != closed_loop_arch:
                out.append(record)
                log(f"  {arch:5s} {shelf:16s} {sim:12s} sharpness {thr.sharpness:6.2f} "
                    f"abrupt={thr.is_abrupt}")
                continue

            try:
                loop = closed_loop_sweep(model, data, "thermal_driving",
                                         offsets[::4])
                record["closed_loop"] = {
                    "offsets": loop.values.tolist(),
                    "forward": loop.forward.tolist(),
                    "reverse": loop.reverse.tolist(),
                    "width": loop.width if loop.converged else None,
                    "loop_area": loop.loop_area if loop.converged else None,
                    "converged": loop.converged, "n_failed": loop.n_failed,
                    "loop_gain": loop.loop_gain, "note": loop.note,
                }
            except (RuntimeError, ValueError) as exc:
                log(f"  closed loop failed for {arch} {shelf}: {exc}")

            out.append(record)
            log(f"  {arch:5s} {shelf:16s} {sim:12s} sharpness {thr.sharpness:6.2f} "
                f"abrupt={thr.is_abrupt}"
                + (f"  loop width {record['closed_loop']['width']:.3f} m/yr"
                   if record.get("closed_loop", {}).get("converged")
                   else "  closed loop not converged"
                   if "closed_loop" in record else ""))
    return out


def run_ice(before: float, after: float) -> dict:
    """Grounding-line response to an abrupt versus a gradual melt increase."""
    out = {}
    for label, slope, b0 in (("prograde", -1.0e-3, -400.0),
                             ("retrograde", 1.0e-3, -900.0)):
        cfg = FlowlineConfig(bed_slope=slope, bed_depth_at_origin=b0)
        runs = compare_histories(cfg, before=before, after=after,
                                 t_shift=100.0, years=400.0)
        out[label] = {
            "retrograde": cfg.retrograde,
            **{k: {"time": v.time[::8].tolist(),
                   "position_km": (v.position[::8] / 1e3).tolist(),
                   "sea_level_mm": v.sea_level[::8].tolist(),
                   "total_retreat_km": v.total_retreat / 1e3,
                   "final_sea_level_mm": float(v.sea_level[-1]),
                   "collapsed": v.collapsed}
               for k, v in runs.items()},
        }
        log(f"  {label:11s} gradual {runs['gradual'].total_retreat / 1e3:6.1f} km, "
            f"abrupt {runs['abrupt'].total_retreat / 1e3:6.1f} km")
    return out


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--train-dir", type=Path, default=TRAIN_DIR)
    p.add_argument("--graphs", type=Path, default=GRAPH_DIR)
    p.add_argument("--outdir", type=Path, default=None)
    p.add_argument("--split", default="shelf",
                   help="which trained split to analyse")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--shelves", nargs="*",
                   default=["Filchner-Ronne", "Ross", "Pine Island", "Getz", "Amery"])
    p.add_argument("--n-sources", type=int, default=24)
    p.add_argument("--skill-only", action="store_true")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    ensure_dirs()
    outdir = args.outdir or (RUN_DIR / "analysis")
    outdir.mkdir(parents=True, exist_ok=True)

    log("collating training skill")
    skill = collate_skill(args.train_dir)
    (outdir / "skill.json").write_text(json.dumps(skill, indent=2))
    log(f"  {skill['n_runs']} runs, {len(skill['by_arch_split'])} arch/split groups")
    for key, s in sorted(skill["by_arch_split"].items()):
        log(f"    {key:18s} RMSE {s['rmse_mean']:.4f}+-{s['rmse_std']:.4f}  "
            f"R2 {s['r2_mean']:+.3f}  baseline {s['baseline_rmse']:.4f}  "
            f"beats={s['beats_baseline']}")

    if args.skill_only:
        return 0

    models = {}
    for arch in ("mlp", "gcn", "gat", "egcn"):
        path = args.train_dir / f"{arch}_{args.split}_seed{args.seed}.pt"
        if path.is_file():
            models[arch], _ = load_model(path, args.device)
        else:
            log(f"  no checkpoint for {arch} ({path.name})")
    if not models:
        raise SystemExit("no trained checkpoints found")
    log(f"loaded {len(models)} models: {sorted(models)}")

    graphs = pick_graphs(args.shelves, ("SMITH_bf663", "SMITH_bi646"),
                         args.graphs, args.device)
    log(f"analysing {len(graphs)} (shelf, scenario) graphs")

    log("connectivity length scales")
    connectivity = run_connectivity(models, graphs, args.n_sources)
    (outdir / "connectivity.json").write_text(json.dumps(connectivity, indent=2))

    log("dominant controls")
    controls = run_controls(models, graphs)
    (outdir / "controls.json").write_text(json.dumps(controls, indent=2))

    log("forcing response")
    offsets = np.linspace(-1.0, 4.0, 81)
    closed_arch = next((a for a in ("gcn", "gat", "egcn") if a in models), None)
    log(f"  closed-loop sweeps for {closed_arch}")
    response = run_response(models, graphs, offsets, closed_loop_arch=closed_arch)
    (outdir / "response.json").write_text(json.dumps(response, indent=2))

    log("ice-sheet response")
    ice = run_ice(before=1.0, after=12.0)
    (outdir / "ice.json").write_text(json.dumps(ice, indent=2))

    log(f"analysis written to {outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
