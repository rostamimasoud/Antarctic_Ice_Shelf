#!/usr/bin/env python3
"""Generate the manuscript figures from the analysis output.

Reads the JSON written by ``04_analyse.py`` and ``09_boxmodel_reference.py`` and
writes PDF and PNG into ``figures/``.  Every panel degrades to an explicit
"not available" note rather than failing, so a partial analysis still produces a
complete figure set with the gaps visible.

Examples
--------
    python scripts/11_make_figures.py
    python scripts/11_make_figures.py --only 2 3
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib                                                  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt                                    # noqa: E402
from matplotlib.lines import Line2D                                # noqa: E402

from aisgnn.config import BOXMODEL_DIR, FIGURE_DIR, RUN_DIR, ensure_dirs  # noqa: E402
from aisgnn.viz import style as st                                 # noqa: E402

ARCH_LABEL = {"mlp": "MLP (no spatial coupling)", "gcn": "GCN",
              "gat": "GAT", "egcn": "Edge-conditioned"}
ARCH_ORDER = ("mlp", "gcn", "gat", "egcn")
SCENARIO_LABEL = {"SMITH_bf663": "REPEAT1970", "SMITH_bi646": "4$\\times$CO$_2$"}


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
# Figure 2: emulator skill
# --------------------------------------------------------------------------- #

def figure_skill(skill: dict, path: Path) -> list[str]:
    """Emulator skill by architecture and held-out dimension.

    The legend is a single shared one placed above both panels.  Inside the axes
    it unavoidably sits on top of the bars -- the tallest group is at the left,
    where a legend would naturally go -- and its baseline entry collided with the
    baseline line itself.
    """
    st.use_style()
    fig, axes = plt.subplots(1, 2, figsize=(st.WIDTH_DOUBLE, 2.9))

    if not skill or not skill.get("by_arch_split"):
        for ax in axes:
            unavailable(ax, "no training runs found")
        return st.save(fig, path)

    rows = skill["by_arch_split"]
    # Ordered easiest to hardest, which is the order the text discusses them in;
    # sorting alphabetically would put the hardest case first.
    order = [sp for sp in ("shelf", "year", "scenario")
             if any(v["split"] == sp for v in rows.values())]
    colours = dict(zip(ARCH_ORDER, st.categorical(len(ARCH_ORDER))))
    width = 0.8 / max(len(ARCH_ORDER), 1)
    ekw = dict(ecolor=st.INK["secondary"], capsize=1.2, elinewidth=0.7)

    # (a) RMSE against the per-shelf-mean baseline.
    ax = axes[0]
    for k, arch in enumerate(ARCH_ORDER):
        xs, ys, es = [], [], []
        for i, split in enumerate(order):
            rec = rows.get(f"{arch}_{split}")
            if rec is None:
                continue
            xs.append(i + (k - 1.5) * width)
            ys.append(rec["rmse_mean"])
            es.append(rec["rmse_std"])
        if xs:
            ax.bar(xs, ys, width=width * 0.9, yerr=es, color=colours[arch],
                   label=ARCH_LABEL[arch], zorder=3, error_kw=ekw)

    for i, split in enumerate(order):
        recs = [v for v in rows.values() if v["split"] == split]
        if recs:
            base = np.nanmean([r["baseline_rmse"] for r in recs])
            ax.plot([i - 0.45, i + 0.45], [base, base], color=st.INK["primary"],
                    lw=1.1, ls=(0, (3, 2)), zorder=5,
                    label="Per-shelf-mean baseline" if i == 0 else None)

    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(order)
    ax.set_ylabel("Test RMSE (m yr$^{-1}$)")
    ax.set_xlabel("Held-out dimension")
    st.soften_grid(ax)
    st.panel_label(ax, "a")

    # (b) Median skill relative to that baseline.
    ax = axes[1]
    for k, arch in enumerate(ARCH_ORDER):
        xs, ys, es = [], [], []
        for i, split in enumerate(order):
            rec = rows.get(f"{arch}_{split}")
            if rec is None or not np.isfinite(rec.get("skill_median", np.nan)):
                continue
            xs.append(i + (k - 1.5) * width)
            ys.append(rec["skill_median"])
            es.append(rec.get("skill_iqr", 0.0) / 2.0)
        if xs:
            ax.bar(xs, ys, width=width * 0.9, yerr=es, color=colours[arch],
                   zorder=3, error_kw=ekw)

    ax.axhline(0.0, color=st.INK["primary"], lw=0.8, zorder=4)
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(order)
    ax.set_ylabel("Skill vs. baseline")
    ax.set_xlabel("Held-out dimension")
    st.soften_grid(ax)
    st.panel_label(ax, "b")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=5, frameon=False,
               fontsize=6.0, bbox_to_anchor=(0.5, -0.02),
               handlelength=1.6, columnspacing=1.4)
    fig.tight_layout(rect=(0, 0.085, 1, 1))
    return st.save(fig, path)


# --------------------------------------------------------------------------- #
# Figure 3: connectivity
# --------------------------------------------------------------------------- #

def figure_connectivity(conn: list, path: Path) -> list[str]:
    st.use_style()
    fig, axes = plt.subplots(1, 2, figsize=(st.WIDTH_DOUBLE, 2.7))

    if not conn:
        for ax in axes:
            unavailable(ax, "no connectivity analysis found")
        return st.save(fig, path)

    colours = dict(zip(ARCH_ORDER, st.categorical(len(ARCH_ORDER))))

    # (a) influence versus distance, pooled over shelves.
    ax = axes[0]
    for arch in ARCH_ORDER:
        recs = [r for r in conn if r["arch"] == arch and r["distances_km"]]
        if not recs:
            continue
        n = min(len(r["influence"]) for r in recs)
        d = np.mean([r["distances_km"][:n] for r in recs], axis=0)
        inf = np.mean([r["influence"][:n] for r in recs], axis=0)
        ax.plot(d, inf, color=colours[arch], lw=1.3, label=ARCH_LABEL[arch])
    ax.set_xlabel("Distance from perturbed cell (km)")
    ax.set_ylabel("Normalised influence on melt")
    ax.legend(fontsize=5.8)
    st.soften_grid(ax)
    st.panel_label(ax, "a")

    # (b) fitted length scale per shelf, by scenario.
    ax = axes[1]
    graph_arch = [a for a in ("gat", "gcn", "egcn")
                  if any(r["arch"] == a for r in conn)]
    if not graph_arch:
        unavailable(ax, "no graph architecture available")
    else:
        arch = graph_arch[0]
        by_shelf = defaultdict(dict)
        for r in conn:
            if r["arch"] != arch:
                continue
            # A shelf whose decay is unresolved contributes an upper bound, drawn
            # open with an arrow, never a filled bar: plotting a bound as though
            # it were a fitted value is the error this panel exists to avoid.
            by_shelf[r["shelf"]][r["simulation"]] = (
                r.get("length_scale_km"), r.get("upper_bound_km"),
                bool(r.get("resolved")))
        shelves = sorted(by_shelf)
        y = np.arange(len(shelves))
        for j, sim in enumerate(("SMITH_bf663", "SMITH_bi646")):
            off = (j - 0.5) * 0.38
            for i, shelf in enumerate(shelves):
                entry = by_shelf[shelf].get(sim)
                if entry is None:
                    continue
                L, bound, resolved = entry
                if resolved and L:
                    ax.barh(i + off, L, height=0.34, color=st.CATEGORICAL[j],
                            zorder=3)
                elif bound:
                    ax.barh(i + off, bound, height=0.34, facecolor="none",
                            edgecolor=st.CATEGORICAL[j], lw=0.8, zorder=3,
                            hatch="///")
        ax.set_yticks(y)
        ax.set_yticklabels(shelves, fontsize=6.0)
        ax.set_xlabel("Connectivity length scale (km)")
        ax.text(0.97, 0.04, ARCH_LABEL[arch], transform=ax.transAxes,
                fontsize=5.8, ha="right", va="bottom", color=st.INK["secondary"])
        handles = [Line2D([], [], color=st.CATEGORICAL[j], lw=4,
                          label=SCENARIO_LABEL.get(sim, sim))
                   for j, sim in enumerate(("SMITH_bf663", "SMITH_bi646"))]
        handles.append(Line2D([], [], color=st.INK["secondary"], lw=0.8,
                              marker="", label="hatched = upper bound only"))
        ax.legend(handles=handles, fontsize=5.4)
        st.soften_grid(ax, axis="x")
    st.panel_label(ax, "b")

    fig.tight_layout()
    return st.save(fig, path)


# --------------------------------------------------------------------------- #
# Figure 4: dominant controls
# --------------------------------------------------------------------------- #

def figure_controls(controls: dict, path: Path) -> list[str]:
    st.use_style()
    fig, axes = plt.subplots(1, 2, figsize=(st.WIDTH_DOUBLE, 2.7))

    arch = next((a for a in ("gat", "gcn", "egcn", "mlp")
                 if controls and a in controls), None)
    if arch is None:
        for ax in axes:
            unavailable(ax, "no intervention analysis found")
        return st.save(fig, path)

    by_key = controls[arch]

    # (a) sensitivity per feature, averaged within each scenario.
    ax = axes[0]
    per_scenario = defaultdict(lambda: defaultdict(list))
    for key, feats in by_key.items():
        sim = key.split("|")[1]
        for feature, rec in feats.items():
            v = rec.get("sensitivity")
            if v is not None and np.isfinite(v):
                per_scenario[sim][feature].append(v)

    features = sorted({f for d in per_scenario.values() for f in d})
    y = np.arange(len(features))
    for j, sim in enumerate(("SMITH_bf663", "SMITH_bi646")):
        vals = [np.nanmean(per_scenario[sim].get(f, [np.nan])) for f in features]
        ax.barh(y + (j - 0.5) * 0.38, vals, height=0.34, color=st.CATEGORICAL[j],
                label=SCENARIO_LABEL.get(sim, sim), zorder=3)
    ax.set_yticks(y)
    ax.set_yticklabels(features, fontsize=5.8)
    ax.set_xlabel("Melt response per unit perturbation (m yr$^{-1}$)")
    ax.axvline(0.0, color=st.INK["primary"], lw=0.8)
    ax.legend(fontsize=5.8)
    st.soften_grid(ax, axis="x")
    st.panel_label(ax, "a")

    # (b) change in sensitivity between scenarios.
    ax = axes[1]
    shift = controls.get(f"{arch}_scenario_shift", {})
    if not shift:
        unavailable(ax, "no scenario pair available")
    else:
        names = sorted(shift, key=lambda k: shift[k])
        vals = [shift[k] for k in names]
        colours = [st.REGIME["warm"] if v > 0 else st.REGIME["cold"] for v in vals]
        ax.barh(np.arange(len(names)), vals, color=colours, zorder=3)
        ax.set_yticks(np.arange(len(names)))
        ax.set_yticklabels(names, fontsize=5.8)
        ax.axvline(0.0, color=st.INK["primary"], lw=0.8)
        ax.set_xlabel("Change in sensitivity, 4$\\times$CO$_2$ minus REPEAT1970")

        st.soften_grid(ax, axis="x")
    st.panel_label(ax, "b")

    fig.tight_layout()
    return st.save(fig, path)


# --------------------------------------------------------------------------- #
# Figure 6: emulator response and ice coupling
# --------------------------------------------------------------------------- #

def figure_response(response: list, ice: dict, path: Path) -> list[str]:
    st.use_style()
    fig, axes = plt.subplots(1, 3, figsize=(st.WIDTH_DOUBLE, 2.6))

    # (a) open-loop response curves.
    ax = axes[0]
    if not response:
        unavailable(ax, "no sweep results found")
    else:
        arch = next((a for a in ("gat", "gcn", "egcn", "mlp")
                     if any(r["arch"] == a for r in response)), None)
        recs = [r for r in response if r["arch"] == arch
                and r["simulation"] == "SMITH_bf663"]
        for colour, r in zip(st.CATEGORICAL, recs):
            ax.plot(r["offsets"], r["melt"], color=colour, lw=1.2, label=r["shelf"])
            if r["is_abrupt"]:
                ax.plot([r["threshold_location"]],
                        [np.interp(r["threshold_location"], r["offsets"], r["melt"])],
                        marker="o", ms=4, color=colour, mec=st.INK["surface"], mew=0.6)
        ax.set_xlabel("Thermal-driving offset ($^\\circ$C)")
        ax.set_ylabel("Area-mean melt (m yr$^{-1}$)")
        ax.legend(fontsize=5.5)

        st.soften_grid(ax)
    st.panel_label(ax, "a")

    # (b) closed-loop forward and reverse branches.
    ax = axes[1]
    loops = [r for r in (response or [])
             if r.get("closed_loop", {}).get("converged")]
    if not loops:
        unavailable(ax, "closed-loop sweep did not converge:\n"
                        "meltwater feedback gain exceeds one,\n"
                        "so no bounded fixed point exists")
    else:
        r = loops[0]
        cl = r["closed_loop"]
        ax.plot(cl["offsets"], cl["forward"], color=st.REGIME["cold"], lw=1.3,
                label="Forward")
        ax.plot(cl["offsets"], cl["reverse"], color=st.REGIME["warm"], lw=1.3,
                ls=(0, (4, 2)), label="Reverse")
        ax.set_xlabel("Thermal-driving offset ($^\\circ$C)")
        ax.set_ylabel("Area-mean melt (m yr$^{-1}$)")

        # The numbers go in the corner rather than the title, which at this
        # width ran under the next panel's label.
        ax.text(0.97, 0.06,
                f"{r['shelf']}\nloop width {cl['width']:.2f} m yr$^{{-1}}$\n"
                f"feedback gain {cl.get('loop_gain', float('nan')):.2f}",
                transform=ax.transAxes, fontsize=5.6, va="bottom", ha="right",
                color=st.INK["secondary"], linespacing=1.35)
        ax.legend(fontsize=5.8)
        st.soften_grid(ax)
    st.panel_label(ax, "b")

    # (c) grounding-line retreat, abrupt versus gradual.
    ax = axes[2]
    if not ice:
        unavailable(ax, "no ice-sheet results found")
    else:
        bed = "prograde" if "prograde" in ice else sorted(ice)[0]
        block = ice[bed]
        for colour, key, label in ((st.INK["muted"], "control", "Control"),
                                   (st.CATEGORICAL[0], "gradual", "Gradual"),
                                   (st.CATEGORICAL[1], "abrupt", "Abrupt")):
            if key in block:
                ax.plot(block[key]["time"], block[key]["position_km"],
                        color=colour, lw=1.3, label=label)
        ax.set_xlabel("Time (yr)")
        ax.set_ylabel("Grounding-line position (km)")
        ax.text(0.97, 0.72, f"{bed} bed", transform=ax.transAxes,
                fontsize=5.8, ha="right", va="top", color=st.INK["secondary"])
        ax.legend(fontsize=5.8)
        st.soften_grid(ax)
    st.panel_label(ax, "c")

    fig.tight_layout()
    return st.save(fig, path)


# --------------------------------------------------------------------------- #
# Figure 1: methodology flowchart
# --------------------------------------------------------------------------- #

def figure_flowchart(path: Path) -> list[str]:
    """Schematic of how the three models relate.

    Laid out as two vertical columns converging on the ice model, so the two
    independent paths stay visibly separate and no connector crosses a box.
    Drawn rather than typeset so it carries the same palette and typography as
    the data figures.
    """
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

    st.use_style()
    fig, ax = plt.subplots(figsize=(st.WIDTH_DOUBLE, 4.6))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")

    cold, warm, grey = st.REGIME["cold"], st.REGIME["warm"], st.INK["secondary"]

    def box(x, y, w, h, text, face, edge, weight="normal", fs=6.3):
        ax.add_patch(FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0.9,rounding_size=1.4",
            facecolor=face, edgecolor=edge, linewidth=0.8, zorder=3))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
                fontsize=fs, color=st.INK["primary"], zorder=4,
                fontweight=weight, linespacing=1.4)

    def down(x, y_from, y_to, colour):
        ax.add_patch(FancyArrowPatch((x, y_from), (x, y_to), arrowstyle="-|>",
                                     mutation_scale=8, linewidth=0.9,
                                     color=colour, zorder=2))

    # Note the unicode en dash: matplotlib does not interpret LaTeX "--".
    left_x, right_x, w, h = 4.0, 55.0, 41.0, 13.0
    ys = [80.0, 61.0, 42.0, 23.0]

    ax.text(left_x + w / 2, 96.0, "Path 1  ·  bulk dynamics", fontsize=7.0,
            color=cold, fontweight="bold", ha="center")
    ax.text(right_x + w / 2, 96.0, "Path 2  ·  spatial structure", fontsize=7.0,
            color=warm, fontweight="bold", ha="center")

    left = ["Cavity\u2013polynya box model\n4 prognostic variables",
            "Calibration on observed melt flux,\ncoastal T/S and cavity ventilation",
            "Pseudo-arclength continuation\nresolving the unstable branch",
            "Bifurcation diagram, hysteresis\nwidth, rate-induced tipping test"]
    right = ["NEMO cavity-resolving simulations\n240 graphs, 2 climate scenarios",
             "GNN training: MLP, GCN, GAT, EGCN\n5 seeds \u00d7 3 held-out regimes",
             "Spatial connectivity and\ninterventional sensitivity",
             "Forcing sweeps, open loop\nand closed meltwater loop"]

    for y, lab in zip(ys, left):
        box(left_x, y, w, h, lab, "#EAF3F8", cold)
    for y, lab in zip(ys, right):
        box(right_x, y, w, h, lab, "#FBEDE4", warm)

    for a, b in zip(ys[:-1], ys[1:]):
        down(left_x + w / 2, a - 0.9, b + h + 0.9, cold)
        down(right_x + w / 2, a - 0.9, b + h + 0.9, warm)

    # Convergence on the ice model.
    box(28.0, 2.0, 44.0, 13.0,
        "Reduced flowline ice model\ngrounding-line response to an abrupt\n"
        "versus a gradual melt increase",
        "#F0EEEA", grey, weight="bold")

    ax.add_patch(FancyArrowPatch((left_x + w / 2, ys[-1] - 0.9), (41.0, 15.9),
                                 arrowstyle="-|>", mutation_scale=8,
                                 linewidth=0.9, color=cold, zorder=2,
                                 connectionstyle="arc3,rad=-0.15"))
    ax.add_patch(FancyArrowPatch((right_x + w / 2, ys[-1] - 0.9), (59.0, 15.9),
                                 arrowstyle="-|>", mutation_scale=8,
                                 linewidth=0.9, color=warm, zorder=2,
                                 connectionstyle="arc3,rad=0.15"))
    ax.text(30.0, 18.6, "melt thresholds", fontsize=5.8, color=cold,
            ha="center", va="bottom")
    ax.text(70.0, 18.6, "melt response", fontsize=5.8, color=warm,
            ha="center", va="bottom")

    # The cross-check between the two paths is the reason for running both.
    ax.add_patch(FancyArrowPatch((left_x + w + 0.9, ys[-1] + h / 2),
                                 (right_x - 0.9, ys[-1] + h / 2),
                                 arrowstyle="<|-|>", mutation_scale=7,
                                 linewidth=0.9, color=st.INK["primary"],
                                 linestyle=(0, (3, 2)), zorder=2))
    # The gap between columns is only ten units wide, so the annotation is kept
    # to one short word and the question it poses is carried by the caption.
    ax.text(50.0, ys[-1] + h / 2 + 1.6, "cross-\ncheck", fontsize=5.6,
            color=st.INK["primary"], ha="center", va="bottom", style="italic",
            linespacing=1.2)

    return st.save(fig, path)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--analysis", type=Path, default=None)
    p.add_argument("--boxmodel", type=Path, default=None)
    p.add_argument("--outdir", type=Path, default=None)
    p.add_argument("--only", nargs="*", type=int, default=None)
    args = p.parse_args()

    ensure_dirs()
    adir = args.analysis or (RUN_DIR / "analysis")
    bdir = args.boxmodel or BOXMODEL_DIR
    outdir = args.outdir or FIGURE_DIR
    outdir.mkdir(parents=True, exist_ok=True)

    wanted = set(args.only) if args.only else {1, 2, 3, 4, 6}
    written = []

    if 1 in wanted:
        written += figure_flowchart(outdir / "fig01_methodology")
    if 2 in wanted:
        written += figure_skill(load(adir / "skill.json"),
                                outdir / "fig02_emulator_skill")
    if 3 in wanted:
        written += figure_connectivity(load(adir / "connectivity.json") or [],
                                       outdir / "fig03_connectivity")
    if 4 in wanted:
        written += figure_controls(load(adir / "controls.json") or {},
                                   outdir / "fig04_controls")
    if 6 in wanted:
        written += figure_response(load(adir / "response.json") or [],
                                   load(adir / "ice.json") or {},
                                   outdir / "fig06_response_and_ice")

    for path in written:
        print(f"wrote {path}")
    print(f"{len(written)} files in {outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
