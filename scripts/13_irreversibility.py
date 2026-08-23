#!/usr/bin/env python3
"""Irreversibility figures, drawn from the numerical continuation output.

Every curve here is read from ``runs/boxmodel/continuation_sigma.json``, which is
the same file the manuscript's bistable-window table is computed from.  Nothing
is idealised: the Pine Island S-curve is 764 continued points, of which 180 lie
on the unstable branch that no forward integration can reach.

The four messages, one per panel of the combined figure:

a  the forward and reverse paths do not coincide, which is what irreversibility
   means operationally;
b  how much forcing reduction would be needed to undo a transition, per cavity;
c  where present-day forcing sits relative to each window;
d  how large the melt jump is when a cavity tips.

A note on direction, because it is easy to state backwards.  A cold cavity tips
when sea-ice formation *falls* through the lower fold: less brine gives less
dense shelf water, so warm water is no longer excluded.  The tipping direction on
these axes is therefore right to left.

Examples
--------
    python scripts/13_irreversibility.py
    python scripts/13_irreversibility.py --only 5 --cavity Ross
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib                                                  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt                                    # noqa: E402
from matplotlib.lines import Line2D                                # noqa: E402
from matplotlib.patches import FancyArrowPatch, Patch              # noqa: E402

from aisgnn.config import BOXMODEL_DIR, FIGURE_DIR, ensure_dirs    # noqa: E402
from aisgnn.viz import style as st                                 # noqa: E402

#: Requested palette; verified colourblind-safe (deutan dE 23.1, tritan 34.1).
COLD = "#2C7BB6"
WARM = "#D7191C"
UNSTABLE = "#A0A0A0"
WINDOW = "#E4E0D8"
PRESENT = "#1A1A1A"

REF_WIDTH = 10.0        # m/yr, reference line in panel b
REF_JUMP = 5.0          # m/yr, reference line in panel d


def load_continuation(path: Path) -> dict:
    return json.loads(path.read_text())


def bistable(cont: dict) -> dict:
    """Cavities for which the continuation resolved both folds."""
    return {c: r for c, r in cont.items()
            if r.get("ok") and "hysteresis" in r}


# --------------------------------------------------------------------------- #
# Panel a: the hysteresis loop for one cavity
# --------------------------------------------------------------------------- #

def panel_hysteresis(ax, rec: dict, cavity: str, annotate: bool = True) -> None:
    """Draw the continued S-curve, separating stable branches from the unstable one."""
    p = np.asarray(rec["p"])
    m = np.asarray(rec["melt"])
    stable = np.asarray(rec["stable"], bool)
    hyst = rec["hysteresis"]
    lo, hi = hyst["p_reverse"], hyst["p_forward"]

    ax.axvspan(lo, hi, color=WINDOW, zorder=0, lw=0)

    # Split into contiguous runs so the dashed unstable segment is not bridged.
    mid = 0.5 * (m.max() + m.min())
    edges = np.flatnonzero(np.diff(stable.astype(int)) != 0) + 1
    for seg in np.split(np.arange(p.size), edges):
        if seg.size < 2:
            continue
        is_stable = bool(stable[seg[0]])
        warm = m[seg].mean() > mid
        ax.plot(p[seg], m[seg],
                color=(WARM if warm else COLD) if is_stable else UNSTABLE,
                ls="-" if is_stable else (0, (4, 2)),
                lw=1.6 if is_stable else 1.1,
                zorder=3 if is_stable else 2)

    # Folds, and the jumps a slowly forced system makes when it reaches them.
    for fold in rec["folds"]:
        ax.plot([fold["p"]], [fold["melt"]], "o", ms=5.0, color=PRESENT,
                mec="white", mew=0.8, zorder=6)

    cold_end = min((f for f in rec["folds"] if f["direction"] == "cold_to_warm"),
                   key=lambda f: f["p"], default=None)
    warm_end = max((f for f in rec["folds"] if f["direction"] == "warm_to_cold"),
                   key=lambda f: f["p"], default=None)

    if annotate and cold_end is not None:
        ax.add_patch(FancyArrowPatch(
            (cold_end["p"], cold_end["melt"]), (cold_end["p"], m.max() * 0.94),
            arrowstyle="-|>", mutation_scale=10, lw=1.3, color=WARM, zorder=5))
        ax.text(lo + 0.22 * (hi - lo), m.max() * 0.70,
                f"tips: melt jumps\n{hyst['melt_jump']:.1f} m yr$^{{-1}}$",
                fontsize=5.8, color=WARM, ha="left", va="center")

    if annotate and warm_end is not None:
        ax.add_patch(FancyArrowPatch(
            (warm_end["p"], warm_end["melt"]), (warm_end["p"], m.min() * 1.06),
            arrowstyle="-|>", mutation_scale=10, lw=1.3, color=COLD, zorder=5))

    ax.axvline(rec["present_day"], color=PRESENT, lw=0.9, ls=(0, (1, 2)), zorder=4)
    ax.text(rec["present_day"] + 0.25 * (hi - lo) * 0.06, m.max() * 1.06,
            "present day", fontsize=5.8, color=PRESENT, ha="left", va="bottom")

    if annotate:
        y_arrow = m.min() + 0.16 * (m.max() - m.min())
        ax.add_patch(FancyArrowPatch((lo, y_arrow), (hi, y_arrow),
                                     arrowstyle="<|-|>", mutation_scale=8,
                                     lw=1.0, color=PRESENT, zorder=5))
        ax.text(0.5 * (lo + hi), y_arrow + 0.035 * (m.max() - m.min()),
                f"$\\Delta\\Sigma$ = {hyst['width']:.1f} m yr$^{{-1}}$",
                fontsize=6.0, color=PRESENT, ha="center", va="bottom")

    ax.set_xlim(0, min(p.max(), hi * 1.45))
    ax.set_ylim(0, m.max() * 1.12)
    ax.set_xlabel("Sea-ice formation rate $\\Sigma$ (m yr$^{-1}$)")
    ax.set_ylabel("Basal melt rate (m yr$^{-1}$)")
    ax.text(0.97, 0.06, cavity, transform=ax.transAxes, fontsize=6.4,
            ha="right", va="bottom", color=st.INK["secondary"])
    st.soften_grid(ax)


def legend_hysteresis(ax, fontsize: float = 5.8) -> None:
    ax.legend(handles=[
        Line2D([], [], color=COLD, lw=1.6, label="Cold branch (stable)"),
        Line2D([], [], color=WARM, lw=1.6, label="Warm branch (stable)"),
        Line2D([], [], color=UNSTABLE, lw=1.1, ls=(0, (4, 2)),
               label="Unstable branch"),
        Line2D([], [], color=PRESENT, marker="o", ls="", ms=4,
               label="Saddle-node fold"),
        Patch(facecolor=WINDOW, label="Bistable window"),
    ], loc="upper right", fontsize=fontsize, framealpha=0.92, frameon=True,
        edgecolor="none", facecolor="white")


# --------------------------------------------------------------------------- #
# Panels b and d: ranked bar charts
# --------------------------------------------------------------------------- #

def panel_bars(ax, values: dict, xlabel: str, colour: str, reference: float,
               ref_label: str, xmax: float | None = None) -> None:
    """Horizontal bars ranked by magnitude, with a reference line."""
    order = sorted(values, key=lambda k: values[k])
    y = np.arange(len(order))
    vals = [values[k] for k in order]

    ax.barh(y, vals, height=0.66, color=colour, zorder=3)
    for yi, v in zip(y, vals):
        ax.text(v + (xmax or max(vals)) * 0.015, yi, f"{v:.1f}",
                va="center", ha="left", fontsize=5.8, color=st.INK["primary"],
                zorder=6,
                bbox=dict(facecolor="white", edgecolor="none", pad=0.6))

    ax.axvline(reference, color=PRESENT, lw=0.9, ls=(0, (4, 2)), zorder=5)
    ax.text(reference, len(order) - 0.35, ref_label, fontsize=5.6,
            color=PRESENT, ha="center", va="bottom")

    ax.set_yticks(y)
    ax.set_yticklabels(order, fontsize=6.0)
    ax.set_xlabel(xlabel)
    ax.set_xlim(0, xmax or max(vals) * 1.22)
    st.soften_grid(ax, axis="x")


# --------------------------------------------------------------------------- #
# Panel c: present-day position within each window
# --------------------------------------------------------------------------- #

def panel_position(ax, records: dict) -> None:
    """Each cavity's bistable window as a segment, with present-day forcing marked."""
    order = sorted(records, key=lambda c: records[c]["hysteresis"]["p_forward"])
    y = np.arange(len(order))

    for i, cavity in enumerate(order):
        rec = records[cavity]
        lo = rec["hysteresis"]["p_reverse"]
        hi = rec["hysteresis"]["p_forward"]
        now = rec["present_day"]

        ax.plot([lo, hi], [i, i], color=WINDOW, lw=6.0, solid_capstyle="butt",
                zorder=2)
        ax.plot([lo, hi], [i, i], color=st.INK["secondary"], lw=0.7, zorder=3)
        for edge, colour in ((lo, WARM), (hi, COLD)):
            ax.plot([edge, edge], [i - 0.26, i + 0.26], color=colour, lw=1.4,
                    zorder=4)
        ax.plot([now], [i], marker="*", ms=8, color=PRESENT, mec="white",
                mew=0.6, zorder=6)

    ax.set_yticks(y)
    ax.set_yticklabels(order, fontsize=6.0)
    ax.set_xlabel("Sea-ice formation rate $\\Sigma$ (m yr$^{-1}$)")
    ax.set_xlim(0, 32)
    ax.set_ylim(-0.7, len(order) - 0.3)
    st.soften_grid(ax, axis="x")

    ax.legend(handles=[
        Patch(facecolor=WINDOW, edgecolor=st.INK["secondary"], lw=0.7,
              label="Bistable window"),
        Line2D([], [], color=WARM, lw=1.4, label="$\\Sigma_{\\mathrm{rev}}$: cold state lost"),
        Line2D([], [], color=COLD, lw=1.4, label="$\\Sigma_{\\mathrm{fwd}}$: warm state lost"),
        Line2D([], [], color=PRESENT, marker="*", ls="", ms=7,
               label="Present-day forcing"),
    ], loc="lower right", fontsize=5.4, framealpha=0.9, frameon=True,
        edgecolor="none", facecolor="white")


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #

def figure_loop(cont: dict, cavity: str, path: Path) -> list[str]:
    st.use_style()
    fig, ax = plt.subplots(figsize=(st.WIDTH_ONE_HALF, 3.3))
    panel_hysteresis(ax, cont[cavity], cavity)
    legend_hysteresis(ax)
    fig.tight_layout()
    return st.save(fig, path)


def figure_widths(records: dict, path: Path) -> list[str]:
    st.use_style()
    fig, ax = plt.subplots(figsize=(st.WIDTH_ONE_HALF, 3.0))
    panel_bars(ax, {c: r["hysteresis"]["width"] for c, r in records.items()},
               "Hysteresis width $\\Delta\\Sigma$ (m yr$^{-1}$)", COLD,
               REF_WIDTH, f"{REF_WIDTH:.0f}", xmax=32)
    fig.tight_layout()
    return st.save(fig, path)


def figure_position(records: dict, path: Path) -> list[str]:
    st.use_style()
    fig, ax = plt.subplots(figsize=(st.WIDTH_ONE_HALF, 3.0))
    panel_position(ax, records)
    fig.tight_layout()
    return st.save(fig, path)


def figure_jump(records: dict, path: Path) -> list[str]:
    st.use_style()
    fig, ax = plt.subplots(figsize=(st.WIDTH_ONE_HALF, 3.0))
    panel_bars(ax, {c: r["hysteresis"]["melt_jump"] for c, r in records.items()},
               "Melt discontinuity $\\Delta m$ (m yr$^{-1}$)", WARM,
               REF_JUMP, f"{REF_JUMP:.0f}", xmax=12.5)
    fig.tight_layout()
    return st.save(fig, path)


def figure_combined(cont: dict, records: dict, cavity: str,
                    path: Path) -> list[str]:
    st.use_style()
    fig, axes = plt.subplots(2, 2, figsize=(st.WIDTH_DOUBLE, 5.6))

    panel_hysteresis(axes[0, 0], cont[cavity], cavity)
    legend_hysteresis(axes[0, 0], fontsize=5.2)
    st.panel_label(axes[0, 0], "a", dx=-0.16)

    panel_bars(axes[0, 1], {c: r["hysteresis"]["width"] for c, r in records.items()},
               "Hysteresis width $\\Delta\\Sigma$ (m yr$^{-1}$)", COLD,
               REF_WIDTH, f"{REF_WIDTH:.0f}", xmax=32)
    st.panel_label(axes[0, 1], "b", dx=-0.22)

    panel_position(axes[1, 0], records)
    st.panel_label(axes[1, 0], "c", dx=-0.22)

    panel_bars(axes[1, 1], {c: r["hysteresis"]["melt_jump"] for c, r in records.items()},
               "Melt discontinuity $\\Delta m$ (m yr$^{-1}$)", WARM,
               REF_JUMP, f"{REF_JUMP:.0f}", xmax=12.5)
    st.panel_label(axes[1, 1], "d", dx=-0.22)

    fig.tight_layout()
    return st.save(fig, path)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--boxmodel", type=Path, default=None)
    p.add_argument("--outdir", type=Path, default=None)
    p.add_argument("--cavity", default="Pine Island",
                   help="cavity shown in the hysteresis-loop panel")
    p.add_argument("--only", nargs="*", type=int, default=None)
    args = p.parse_args()

    ensure_dirs()
    bdir = args.boxmodel or BOXMODEL_DIR
    outdir = args.outdir or FIGURE_DIR
    outdir.mkdir(parents=True, exist_ok=True)

    cont = load_continuation(bdir / "continuation_sigma.json")
    records = bistable(cont)
    if args.cavity not in records:
        raise SystemExit(f"{args.cavity} has no resolved bistable window; "
                         f"available: {sorted(records)}")
    print(f"{len(records)} cavities with a resolved bistable window; "
          f"loop panel shows {args.cavity} "
          f"({len(cont[args.cavity]['p'])} continued points, "
          f"{sum(1 for s in cont[args.cavity]['stable'] if not s)} unstable)")

    wanted = set(args.only) if args.only else {1, 2, 3, 4, 5}
    written: list[str] = []
    slug = args.cavity.replace(" ", "_")

    if 1 in wanted:
        written += figure_loop(cont, args.cavity, outdir / f"fig07_hysteresis_{slug}")
    if 2 in wanted:
        written += figure_widths(records, outdir / "fig08_hysteresis_widths")
    if 3 in wanted:
        written += figure_position(records, outdir / "fig09_present_position")
    if 4 in wanted:
        written += figure_jump(records, outdir / "fig10_melt_jump")
    if 5 in wanted:
        written += figure_combined(cont, records, args.cavity,
                                   outdir / "fig11_irreversibility")

    for w in written:
        print(f"wrote {w}")
    print(f"{len(written)} files in {outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
