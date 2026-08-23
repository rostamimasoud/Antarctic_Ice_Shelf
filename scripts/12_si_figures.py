#!/usr/bin/env python3
"""Compute and draw the supplementary figures S2-S6.

Three of the five need data that the main analysis does not produce, so they are
computed here and cached to ``runs/si`` before plotting:

S2  continuation in CDW temperature for every cavity        (from the box-model run)
S3  two-parameter bistability map in (sigma, T_CDW)         (computed here)
S4  sensitivity of hysteresis width to each fitted parameter (computed here)
S5  emulator training curves and per-seed skill             (from the training runs)
S6  early-warning indicator sensitivity to window and bandwidth (computed here)

Examples
--------
    python scripts/12_si_figures.py --only 2 5        # the ones needing no compute
    python scripts/12_si_figures.py --grid 12         # coarser bistability map
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib                                                  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt                                    # noqa: E402
from matplotlib.lines import Line2D                                # noqa: E402
from matplotlib.patches import Patch                               # noqa: E402

from aisgnn.boxmodel.calibrate import OBSERVATIONS, calibrate      # noqa: E402
from aisgnn.boxmodel.cavity import CAVITIES, BoxParams, CavityBoxModel  # noqa: E402
from aisgnn.boxmodel.continuation import (                         # noqa: E402
    bifurcation_diagram,
    bistability_map,
)
from aisgnn.config import (                                        # noqa: E402
    BOXMODEL_DIR,
    FIGURE_DIR,
    RUN_DIR,
    TRAIN_DIR,
    ensure_dirs,
)
from aisgnn.dynsys.ews import indicator_sensitivity                # noqa: E402
from aisgnn.viz import style as st                                 # noqa: E402

#: Cavities shown in the two-parameter map and the sensitivity panel.
MAP_CAVITIES = ("Filchner-Ronne", "Ross", "Amery", "Pine Island")
SENSITIVITY_CAVITY = "Ross"
FITTED = ("c_ovt", "c_dsw", "c_exp", "lambda_atm", "eps_out")


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load(path: Path):
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None


def unavailable(ax, message: str) -> None:
    ax.text(0.5, 0.5, message, ha="center", va="center", fontsize=6.5,
            color=st.INK["muted"], transform=ax.transAxes, wrap=True)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


# --------------------------------------------------------------------------- #
# Computation
# --------------------------------------------------------------------------- #

def compute_bistability(params: dict, n_grid: int, outdir: Path) -> dict:
    """Map which regimes are attractors over the (sigma, T_CDW) plane."""
    out = {}
    for cavity in MAP_CAVITIES:
        obs = OBSERVATIONS[cavity]
        base = BoxParams(CAVITIES[cavity], T_cdw=obs.T_cdw, sigma=obs.sigma, **params)
        sigma = np.linspace(0.5, 25.0, n_grid)
        tcdw = np.linspace(-1.0, 2.5, n_grid)
        t0 = time.time()
        res = bistability_map(base, "sigma", sigma, "T_cdw", tcdw)
        out[cavity] = {
            "sigma": res["x"].tolist(), "T_cdw": res["y"].tolist(),
            "cold": res["cold"].astype(int).tolist(),
            "warm": res["warm"].astype(int).tolist(),
            "bistable": res["bistable"].astype(int).tolist(),
            "present": {"sigma": obs.sigma, "T_cdw": obs.T_cdw},
        }
        frac = float(res["bistable"].mean())
        log(f"  {cavity:16s} {n_grid}x{n_grid} grid, bistable over "
            f"{100 * frac:.0f}% of the plane ({time.time() - t0:.0f}s)")
    (outdir / "bistability_map.json").write_text(json.dumps(out))
    return out


def compute_parameter_sensitivity(params: dict, outdir: Path) -> dict:
    """Hysteresis width in sigma as each fitted parameter is varied."""
    cavity = SENSITIVITY_CAVITY
    obs = OBSERVATIONS[cavity]
    factors = (0.5, 0.707, 1.0, 1.414, 2.0)
    out: dict[str, dict] = {"cavity": cavity, "factors": list(factors),
                            "width": {}}

    for name in FITTED:
        widths = []
        for f in factors:
            trial = dict(params)
            trial[name] = params[name] * f
            if name == "eps_out":
                trial[name] = min(trial[name], 1.0)
            base = BoxParams(CAVITIES[cavity], T_cdw=obs.T_cdw,
                             sigma=obs.sigma, **trial)
            try:
                _, _, hyst = bifurcation_diagram(
                    base, "sigma", p_start=obs.sigma, p_min=0.0, p_max=45.0,
                    regime=obs.regime)
                widths.append(hyst.width if hyst else None)
            except (RuntimeError, np.linalg.LinAlgError):
                widths.append(None)
        out["width"][name] = widths
        shown = ", ".join("--" if w is None else f"{w:.1f}" for w in widths)
        log(f"  {name:12s} widths at x{factors}: {shown}")

    (outdir / "parameter_sensitivity.json").write_text(json.dumps(out))
    return out


def compute_ews_sensitivity(params: dict, outdir: Path) -> dict:
    """Variance-trend Kendall tau over a grid of windows and bandwidths.

    The trajectory ramps the sea-ice formation rate down towards the cold->warm
    fold with additive noise, which is the situation the indicators are meant to
    warn about.
    """
    cavity = "Ross"
    obs = OBSERVATIONS[cavity]
    base = BoxParams(CAVITIES[cavity], T_cdw=obs.T_cdw, sigma=obs.sigma, **params)

    sig = load(BOXMODEL_DIR / "continuation_sigma.json") or {}
    folds = [f["p"] for f in sig.get(cavity, {}).get("folds", [])
             if f["direction"] == "cold_to_warm" and f["p"] < obs.sigma]
    target = max(folds) if folds else 2.0

    model = CavityBoxModel(base)
    x0 = model.equilibrium(obs.regime)
    years = 4000.0

    def forcing(t: float) -> dict:
        frac = min(max(t / years, 0.0), 1.0)
        return {"sigma": obs.sigma + frac * (target * 1.02 - obs.sigma)}

    t, X = model.integrate(x0, years=years, n_out=3000, forcing=forcing,
                           noise=0.05, seed=0)
    series = X[:, 0]              # cavity temperature

    windows = (150, 250, 400, 600)
    bandwidths = (50, 100, 200, 400)
    grid = indicator_sensitivity(series, windows, bandwidths)

    out = {"cavity": cavity, "sigma_start": obs.sigma, "sigma_target": target,
           "windows": list(windows), "bandwidths": list(bandwidths),
           "tau": {f"{w}|{b}": (None if not np.isfinite(v) else float(v))
                   for (w, b), v in grid.items()},
           "time": t[::10].tolist(), "series": series[::10].tolist()}
    finite = [v for v in grid.values() if np.isfinite(v)]
    log(f"  {cavity}: tau over {len(grid)} filter choices, "
        f"range {min(finite):+.2f} to {max(finite):+.2f}")
    (outdir / "ews_sensitivity.json").write_text(json.dumps(out))
    return out


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #

def fig_s2(cont: dict, path: Path) -> list[str]:
    """Continuation in CDW temperature for every cavity."""
    st.use_style()
    names = [c for c, r in (cont or {}).items() if r.get("ok")]
    if not names:
        fig, ax = plt.subplots(figsize=(st.WIDTH_DOUBLE, 3.0))
        unavailable(ax, "no CDW-temperature continuation found")
        return st.save(fig, path)

    ncol = 4
    nrow = int(np.ceil(len(names) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(st.WIDTH_DOUBLE, 2.0 * nrow),
                             sharex=True)
    axes = np.atleast_1d(axes).ravel()

    for ax, cavity in zip(axes, names):
        rec = cont[cavity]
        p = np.asarray(rec["p"])
        melt = np.asarray(rec["melt"])
        stable = np.asarray(rec["stable"], bool)
        edges = np.flatnonzero(np.diff(stable.astype(int)) != 0) + 1
        for seg in np.split(np.arange(p.size), edges):
            if seg.size < 2:
                continue
            is_stable = bool(stable[seg[0]])
            warm = melt[seg].mean() > 0.5 * (melt.max() + melt.min())
            ax.plot(p[seg], melt[seg],
                    color=(st.REGIME["warm"] if warm else st.REGIME["cold"])
                    if is_stable else st.REGIME["unstable"],
                    ls="-" if is_stable else (0, (3, 2)),
                    lw=1.2 if is_stable else 0.9)
        for f in rec["folds"]:
            ax.plot([f["p"]], [f["melt"]], "o", ms=3.0,
                    color=st.INK["primary"], mec=st.INK["surface"], mew=0.5)
        ax.axvline(rec["present_day"], color=st.INK["muted"], lw=0.6, ls=(0, (1, 2)))
        ax.text(0.04, 0.93, cavity, transform=ax.transAxes, fontsize=5.8,
                ha="left", va="top", color=st.INK["secondary"])
        st.soften_grid(ax)

    for ax in axes[len(names):]:
        ax.set_visible(False)
    for ax in axes[-ncol:]:
        if ax.get_visible():
            ax.set_xlabel("$T_{\\mathrm{CDW}}$ ($^\\circ$C)")
    for i in range(0, len(names), ncol):
        axes[i].set_ylabel("Melt (m yr$^{-1}$)")

    fig.tight_layout()
    return st.save(fig, path)


def fig_s3(data: dict, path: Path) -> list[str]:
    """Two-parameter bistability map."""
    st.use_style()
    names = list(data or {})
    if not names:
        fig, ax = plt.subplots(figsize=(st.WIDTH_DOUBLE, 3.0))
        unavailable(ax, "no bistability map computed")
        return st.save(fig, path)

    fig, axes = plt.subplots(1, len(names),
                             figsize=(st.WIDTH_DOUBLE, 2.2), sharey=True)
    axes = np.atleast_1d(axes)

    for ax, cavity in zip(axes, names):
        rec = data[cavity]
        sigma = np.asarray(rec["sigma"])
        tcdw = np.asarray(rec["T_cdw"])
        cold = np.asarray(rec["cold"], bool)
        warm = np.asarray(rec["warm"], bool)

        # 0 neither, 1 cold only, 2 warm only, 3 bistable.
        code = np.zeros_like(cold, dtype=int)
        code[cold & ~warm] = 1
        code[warm & ~cold] = 2
        code[cold & warm] = 3
        # The bistable region is the headline category here, so it needs a fill
        # that reads as an area and as a legend swatch; the pale tint used for a
        # background band in the main figures is too light for both.
        bistable_fill = "#C4BCAE"
        cmap = matplotlib.colors.ListedColormap(
            ["#FFFFFF", st.REGIME["cold"], st.REGIME["warm"], bistable_fill])
        ax.pcolormesh(sigma, tcdw, code, cmap=cmap, vmin=0, vmax=3,
                      shading="auto", rasterized=True)
        ax.plot([rec["present"]["sigma"]], [rec["present"]["T_cdw"]],
                marker="*", ms=7, color=st.INK["primary"],
                mec=st.INK["surface"], mew=0.6)
        ax.text(0.04, 0.94, cavity, transform=ax.transAxes, fontsize=5.8,
                ha="left", va="top", color=st.INK["primary"])
        ax.set_xlabel("$\\Sigma$ (m yr$^{-1}$)")
    axes[0].set_ylabel("$T_{\\mathrm{CDW}}$ ($^\\circ$C)")

    fig.legend(handles=[
        Patch(facecolor=st.REGIME["cold"], label="Cold state only"),
        Patch(facecolor=st.REGIME["warm"], label="Warm state only"),
        Patch(facecolor="#C4BCAE", edgecolor=st.INK["secondary"], lw=0.4,
              label="Bistable: both states available"),
        Line2D([], [], marker="*", ls="", color=st.INK["primary"], ms=6,
               label="Present-day forcing"),
    ], fontsize=6.0, loc="lower center", ncol=4, frameon=False,
        bbox_to_anchor=(0.5, -0.04))

    fig.tight_layout(rect=(0, 0.08, 1, 1))
    return st.save(fig, path)


def fig_s4(data: dict, path: Path) -> list[str]:
    """Hysteresis width against each fitted parameter."""
    st.use_style()
    fig, ax = plt.subplots(figsize=(st.WIDTH_ONE_HALF, 2.6))
    if not data or not data.get("width"):
        unavailable(ax, "no parameter sensitivity computed")
        return st.save(fig, path)

    factors = np.asarray(data["factors"], float)
    for colour, (name, widths) in zip(st.CATEGORICAL, data["width"].items()):
        vals = [np.nan if w is None else w for w in widths]
        ax.plot(factors, vals, marker="o", ms=3, lw=1.2, color=colour, label=name)

    ax.axvline(1.0, color=st.INK["muted"], lw=0.7, ls=(0, (1, 2)))
    ax.set_xscale("log")
    ax.set_xticks(factors)
    ax.set_xticklabels([f"{f:g}" for f in factors])
    ax.set_xlabel("Multiple of the calibrated value")
    ax.set_ylabel("Hysteresis width $\\Delta\\Sigma$ (m yr$^{-1}$)")
    ax.legend(fontsize=5.6)
    st.soften_grid(ax)
    fig.tight_layout()
    return st.save(fig, path)


def fig_s5(skill: dict, train_dir: Path, path: Path) -> list[str]:
    """Per-seed skill and training curves."""
    st.use_style()
    fig, axes = plt.subplots(1, 2, figsize=(st.WIDTH_DOUBLE, 2.6))

    runs = [load(p) for p in sorted(train_dir.glob("*.json"))]
    runs = [r for r in runs if r]

    ax = axes[0]
    if not runs:
        unavailable(ax, "no training runs found")
    else:
        archs = sorted({r["arch"] for r in runs})
        splits = sorted({r["split"] for r in runs})
        colours = dict(zip(archs, st.categorical(len(archs))))
        for i, split in enumerate(splits):
            for k, arch in enumerate(archs):
                vals = [r["test"]["rmse"] for r in runs
                        if r["arch"] == arch and r["split"] == split]
                if vals:
                    xs = np.full(len(vals), i + (k - (len(archs) - 1) / 2) * 0.18)
                    ax.plot(xs, vals, "o", ms=3.0, color=colours[arch],
                            alpha=0.85, label=arch if i == 0 else None)
        ax.set_xticks(range(len(splits)))
        ax.set_xticklabels(splits)
        ax.set_ylabel("Test RMSE (m yr$^{-1}$)")
        ax.set_xlabel("Held-out dimension")

        ax.legend(fontsize=5.8)
        st.soften_grid(ax)
    st.panel_label(ax, "a")

    ax = axes[1]
    curves = [r for r in runs if r.get("history") and r["split"] == "shelf"]
    if not curves:
        unavailable(ax, "no training histories found")
    else:
        archs = sorted({r["arch"] for r in curves})
        colours = dict(zip(archs, st.categorical(len(archs))))
        for arch in archs:
            for r in [c for c in curves if c["arch"] == arch][:2]:
                hist = r["history"]
                ax.plot([h["epoch"] for h in hist], [h["rmse"] for h in hist],
                        color=colours[arch], lw=0.9, alpha=0.8,
                        label=arch if r is [c for c in curves
                                            if c["arch"] == arch][0] else None)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Validation RMSE (m yr$^{-1}$)")

        handles = [Line2D([], [], color=colours[a], lw=1.2, label=a) for a in archs]
        ax.legend(handles=handles, fontsize=5.8)
        st.soften_grid(ax)
    st.panel_label(ax, "b")

    fig.tight_layout()
    return st.save(fig, path)


def fig_s6(data: dict, path: Path) -> list[str]:
    """Indicator-trend sensitivity to window and bandwidth."""
    st.use_style()
    fig, axes = plt.subplots(1, 2, figsize=(st.WIDTH_DOUBLE, 2.5))

    if not data:
        for ax in axes:
            unavailable(ax, "no early-warning sensitivity computed")
        return st.save(fig, path)

    ax = axes[0]
    ax.plot(data["time"], data["series"], color=st.REGIME["cold"], lw=0.6)
    ax.set_xlabel("Time (yr)")
    ax.set_ylabel("Cavity temperature ($^\\circ$C)")
    st.soften_grid(ax)
    st.panel_label(ax, "a")

    ax = axes[1]
    windows = data["windows"]
    bandwidths = data["bandwidths"]
    grid = np.full((len(bandwidths), len(windows)), np.nan)
    for j, b in enumerate(bandwidths):
        for i, w in enumerate(windows):
            v = data["tau"].get(f"{w}|{b}")
            if v is not None:
                grid[j, i] = v

    lim = np.nanmax(np.abs(grid)) if np.isfinite(grid).any() else 1.0
    im = ax.imshow(grid, cmap=st.DIVERGING, vmin=-lim, vmax=lim,
                   origin="lower", aspect="auto")
    ax.set_xticks(range(len(windows)))
    ax.set_xticklabels(windows)
    ax.set_yticks(range(len(bandwidths)))
    ax.set_yticklabels(bandwidths)
    ax.set_xlabel("Rolling window (samples)")
    ax.set_ylabel("Detrending bandwidth (samples)")

    for j in range(grid.shape[0]):
        for i in range(grid.shape[1]):
            if np.isfinite(grid[j, i]):
                ax.text(i, j, f"{grid[j, i]:+.2f}", ha="center", va="center",
                        fontsize=5.2, color=st.INK["primary"])
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    st.panel_label(ax, "b")

    fig.tight_layout()
    return st.save(fig, path)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--boxmodel", type=Path, default=None)
    p.add_argument("--train-dir", type=Path, default=TRAIN_DIR)
    p.add_argument("--outdir", type=Path, default=None)
    p.add_argument("--cache", type=Path, default=None)
    p.add_argument("--grid", type=int, default=14,
                   help="bistability map resolution per axis")
    p.add_argument("--only", nargs="*", type=int, default=None)
    p.add_argument("--recompute", action="store_true")
    args = p.parse_args()

    ensure_dirs()
    bdir = args.boxmodel or BOXMODEL_DIR
    outdir = args.outdir or FIGURE_DIR
    cache = args.cache or (RUN_DIR / "si")
    cache.mkdir(parents=True, exist_ok=True)
    outdir.mkdir(parents=True, exist_ok=True)

    wanted = set(args.only) if args.only else {2, 3, 4, 5, 6}
    written: list[str] = []

    params = None
    if wanted & {3, 4, 6}:
        need = [(3, "bistability_map.json"), (4, "parameter_sensitivity.json"),
                (6, "ews_sensitivity.json")]
        if args.recompute or any(n in wanted and not (cache / f).is_file()
                                 for n, f in need):
            log("calibrating the box model")
            params = calibrate().params

    if 2 in wanted:
        written += fig_s2(load(bdir / "continuation_T_cdw.json"),
                          outdir / "figS02_continuation_tcdw")

    if 3 in wanted:
        path = cache / "bistability_map.json"
        data = None if args.recompute else load(path)
        if data is None:
            log(f"computing bistability maps on a {args.grid}x{args.grid} grid")
            data = compute_bistability(params, args.grid, cache)
        written += fig_s3(data, outdir / "figS03_bistability_map")

    if 4 in wanted:
        path = cache / "parameter_sensitivity.json"
        data = None if args.recompute else load(path)
        if data is None:
            log("computing hysteresis-width sensitivity")
            data = compute_parameter_sensitivity(params, cache)
        written += fig_s4(data, outdir / "figS04_parameter_sensitivity")

    if 5 in wanted:
        written += fig_s5(load((RUN_DIR / "analysis") / "skill.json"),
                          args.train_dir, outdir / "figS05_emulator_detail")

    if 6 in wanted:
        path = cache / "ews_sensitivity.json"
        data = None if args.recompute else load(path)
        if data is None:
            log("computing early-warning indicator sensitivity")
            data = compute_ews_sensitivity(params, cache)
        written += fig_s6(data, outdir / "figS06_ews_sensitivity")

    for w in written:
        print(f"wrote {w}")
    print(f"{len(written)} files in {outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
