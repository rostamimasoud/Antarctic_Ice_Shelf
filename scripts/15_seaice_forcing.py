#!/usr/bin/env python3
"""Observed coastal sea-ice production, and what it implies for the windows.

The sea-ice formation rate is the control parameter of the box model, and until
now it was set from order-of-magnitude values compiled from the regional
literature.  This script replaces that with an observational estimate and asks
whether the bifurcation result survives.

For each cavity the coastal polynya is taken to be the open-ocean cells within
``--radius`` of that cavity's ice front whose mean sea-ice production exceeds
\\SI{0.5}{\\metre} per \\SI{30}{\\day}, the threshold Nakata et al. use to
separate active polynyas from the surrounding pack.  The polynya-mean production
is converted to an annual rate over the eight-month freezing season, which
carries \\SI{96}{\\percent} of the total, and compared with the value the model
was given.

Data.  The sea-ice production field is from Nakata et al. (2021); the ice-shelf
masks are from Burgard et al. (2022).  Both are used here in the 5 km
stereographic form redistributed by Saddier et al. (2026) under CC-BY-4.0, whose
Figure 1 shows the same production field.  This figure is not that one: the map
is context for panels b and c, which test our own forcing and our own bistable
windows against the observations.

Examples
--------
    python scripts/15_seaice_forcing.py --data local_runs/data/external
    python scripts/15_seaice_forcing.py --radius 100e3 --radius 200e3
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
from matplotlib.colors import ListedColormap, Normalize            # noqa: E402
from matplotlib.lines import Line2D                                # noqa: E402
from matplotlib.patches import Patch                               # noqa: E402

from aisgnn.config import BOXMODEL_DIR, FIGURE_DIR, RUN_DIR, ensure_dirs  # noqa: E402
from aisgnn.viz import style as st                                 # noqa: E402

#: Threshold separating an active coastal polynya from the surrounding pack,
#: in metres of ice per 30 days (Nakata et al. 2021).
POLYNYA_THRESHOLD = 0.5

#: Months of the freezing season, March to October, carrying 96% of production.
FREEZING_MONTHS = 8.0

#: Forcing the model was given, from Supplementary Table S4 (m of ice per year).
ASSUMED = {
    "Filchner-Ronne": 10.0, "Ross": 12.0, "Amery": 8.0, "Fimbul": 5.0,
    "Larsen C": 5.0, "Riiser-Larsen": 5.0, "Shackleton": 6.0, "Totten": 5.0,
    "Getz": 6.0, "Pine Island": 4.0, "Thwaites": 4.0,
}

ORDER = ("Filchner-Ronne", "Ross", "Amery", "Fimbul", "Larsen C",
         "Riiser-Larsen", "Shackleton", "Totten", "Getz", "Pine Island",
         "Thwaites")

COLD = "#2C7BB6"
WARM = "#D7191C"
WINDOW = "#E4E0D8"
INK = "#1A1A1A"


def log(msg: str) -> None:
    print(msg, flush=True)


def polynya_stats(data: Path, radius: float) -> dict:
    """Polynya area and mean production for every cavity, from the 5 km fields."""
    import xarray as xr
    from scipy.spatial import cKDTree

    masks = xr.open_dataset(
        data / "nemo_5km_isf_masks_and_info_and_distance_new_oneFRIS.nc")
    prod_ds = xr.open_dataset(data / "ice_prod_5km_Nakata_mean.nc").isel(time=0)
    area_ds = xr.open_dataset(data / "gridarea.nc")

    # The production and area fields are on a wider window of the same 5 km
    # stereographic grid, so selecting on the mask coordinates aligns them
    # exactly rather than by interpolation.
    prod = prod_ds["prod"].sel(x=masks.x, y=masks.y).values
    area = area_ds["cell_area"].sel(x=masks.x, y=masks.y).values
    ids = masks.ISF_mask.values

    names = {int(n): str(v)
             for n, v in zip(masks.Nisf.values, masks.isf_name.values)}
    X, Y = np.meshgrid(masks.x.values, masks.y.values)
    ocean = np.abs(ids - 1.0) < 1e-6

    ocean_pts = np.column_stack([X[ocean], Y[ocean]])
    tree = cKDTree(ocean_pts)

    out = {"radius": radius, "cavities": {}, "grid": {}}
    for cavity in ORDER:
        key = next((i for i, v in names.items() if v == cavity), None)
        if key is None:
            log(f"  {cavity}: absent from the mask, skipped")
            continue
        shelf = np.abs(ids - key) < 1e-6
        if not shelf.any():
            log(f"  {cavity}: no cells, skipped")
            continue

        # Ocean cells near this shelf's front.  Assigning by distance to the
        # shelf rather than by a drawn box keeps the choice to one number.
        near = set()
        for hit in tree.query_ball_point(np.column_stack([X[shelf], Y[shelf]]),
                                         r=radius):
            near.update(hit)
        sel = np.zeros(ocean_pts.shape[0], bool)
        sel[np.fromiter(near, int, len(near))] = True

        pr, ar = prod[ocean][sel], area[ocean][sel]
        active = pr > POLYNYA_THRESHOLD
        if not active.any():
            log(f"  {cavity}: no cells above threshold within {radius/1e3:.0f} km")
            continue

        xi = float((pr[active] * ar[active]).sum() / ar[active].sum())
        out["cavities"][cavity] = {
            "polynya_area": float(ar[active].sum()),
            "xi_m_per_30d": xi,
            "sigma_obs": xi * FREEZING_MONTHS,
            "sigma_assumed": ASSUMED[cavity],
            "n_cells": int(active.sum()),
        }

    out["grid"] = {"prod": prod, "ids": ids, "x": masks.x.values,
                   "y": masks.y.values, "names": names}
    return out


def figure(stats: dict, cont: dict, path: Path) -> list[str]:
    """Map of production, our forcing against it, and the windows it implies."""
    st.use_style()
    fig = plt.figure(figsize=(st.WIDTH_DOUBLE, 4.6))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.0, 1.02], wspace=0.34,
                          hspace=0.55)

    g = stats["grid"]
    rows = stats["cavities"]

    # ------------------------------------------------------------------ #
    # a: the production field
    # ------------------------------------------------------------------ #
    ax = fig.add_subplot(gs[:, 0])
    prod = np.where(np.abs(g["ids"] - 1.0) < 1e-6, g["prod"], np.nan)
    grounded = np.abs(g["ids"]) < 1e-6

    # Two segments about the polynya threshold.  The split is the polynya
    # definition, not decoration: without it the pack swamps the coastal signal.
    below = plt.get_cmap("Blues_r")(np.linspace(0.35, 0.92, 128))
    above = plt.get_cmap("magma_r")(np.linspace(0.08, 0.92, 128))
    cmap = ListedColormap(np.vstack([below, above]))
    cmap.set_bad(alpha=0.0)

    def warp(v):
        """Map [0, thr] and [thr, vmax] onto the two halves of the ramp."""
        v = np.asarray(v, float)
        hi = np.nanmax(g["prod"])
        lo_part = 0.5 * np.clip(v, 0, POLYNYA_THRESHOLD) / POLYNYA_THRESHOLD
        hi_part = 0.5 * np.clip(v - POLYNYA_THRESHOLD, 0, None) / (hi - POLYNYA_THRESHOLD)
        return lo_part + hi_part

    ax.imshow(np.where(grounded, 1.0, np.nan),
              extent=[g["x"][0], g["x"][-1], g["y"][0], g["y"][-1]],
              origin="lower", cmap=ListedColormap(["#E8E8E6"]), zorder=1)
    im = ax.imshow(warp(prod),
                   extent=[g["x"][0], g["x"][-1], g["y"][0], g["y"][-1]],
                   origin="lower", cmap=cmap, vmin=0, vmax=1, zorder=2)

    # Labels are nudged by hand where cavities sit close together on the map;
    # a leader line keeps the association explicit once a label has moved.
    NUDGE = {
        "Thwaites": (-0.12, -0.10), "Pine Island": (-0.13, -0.05),
        "Getz": (0.03, 0.12), "Larsen C": (-0.04, 0.09),
        "Riiser-Larsen": (-0.03, -0.12), "Fimbul": (0.11, -0.10),
        "Shackleton": (0.11, 0.05), "Totten": (0.06, 0.11),
        "Amery": (-0.13, -0.07), "Ross": (0.0, 0.02),
        "Filchner-Ronne": (-0.09, -0.02),
    }
    span = g["x"][-1] - g["x"][0]
    for cavity in rows:
        key = next(i for i, v in g["names"].items() if v == cavity)
        shelf = np.abs(g["ids"] - key) < 1e-6
        ax.contour(g["x"], g["y"], shelf.astype(float), levels=[0.5],
                   colors=[INK], linewidths=0.6, zorder=4)
        yy, xx = np.nonzero(shelf)
        cx = g["x"][int(xx.mean())]
        cy = g["y"][int(yy.mean())]
        dx, dy = NUDGE.get(cavity, (0.0, 0.0))
        ax.annotate(cavity, xy=(cx, cy),
                    xytext=(cx + dx * span, cy + dy * span),
                    fontsize=4.8, color=INK, ha="center", va="center",
                    zorder=6,
                    arrowprops=(dict(arrowstyle="-", lw=0.4,
                                     color=st.INK["muted"])
                                if (dx or dy) else None),
                    bbox=dict(facecolor="white", alpha=0.78, pad=0.7,
                              edgecolor="none"))

    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_aspect("equal")

    cb = fig.colorbar(im, ax=ax, orientation="horizontal", fraction=0.042,
                      pad=0.03, shrink=0.86)
    ticks = [0.0, 0.25, POLYNYA_THRESHOLD, 1.5, 2.5, 3.5]
    cb.set_ticks([warp(t) for t in ticks])
    cb.set_ticklabels([f"{t:g}" for t in ticks])
    cb.set_label("Sea-ice production $\\xi$ (m per 30 d)", fontsize=6,
                 labelpad=2)
    cb.ax.tick_params(labelsize=5.5, pad=1.5)
    # The rule marks the polynya threshold, where the ramp changes hue.
    cb.ax.axvline(warp(POLYNYA_THRESHOLD), color=INK, lw=0.8)
    st.panel_label(ax, "a", dx=0.01, dy=1.02)

    # ------------------------------------------------------------------ #
    # b: derived forcing against the value the model was given
    # ------------------------------------------------------------------ #
    ax = fig.add_subplot(gs[0, 1])
    order = [c for c in ORDER if c in rows]
    y = np.arange(len(order))
    obs = np.array([rows[c]["sigma_obs"] for c in order])
    ass = np.array([rows[c]["sigma_assumed"] for c in order])

    for i, c in enumerate(order):
        ax.plot([ass[i], obs[i]], [i, i], color=st.INK["muted"], lw=0.8, zorder=2)
    ax.plot(ass, y, "o", ms=3.4, color=st.INK["secondary"], zorder=4,
            mec="white", mew=0.4)
    ax.plot(obs, y, "o", ms=3.4, color=COLD, zorder=4, mec="white", mew=0.4)

    ax.set_yticks(y); ax.set_yticklabels(order, fontsize=5.6)
    ax.set_xlabel("Sea-ice formation rate $\\Sigma$ (m yr$^{-1}$)")
    ax.set_xlim(0, max(obs.max(), ass.max()) * 1.12)
    ax.legend(handles=[
        Line2D([], [], color=st.INK["secondary"], marker="o", ls="", ms=3.4,
               label="Prescribed (Table S4)"),
        Line2D([], [], color=COLD, marker="o", ls="", ms=3.4,
               label="Derived from observations"),
    ], fontsize=5.2, loc="upper center", frameon=False, ncol=2,
        bbox_to_anchor=(0.5, -0.30))
    st.soften_grid(ax, axis="x")
    st.panel_label(ax, "b", dx=-0.30)

    # ------------------------------------------------------------------ #
    # c: the derived forcing placed in each window
    # ------------------------------------------------------------------ #
    ax = fig.add_subplot(gs[1, 1])
    win = [c for c in order
           if cont.get(c, {}).get("ok") and "hysteresis" in cont.get(c, {})]
    y = np.arange(len(win))
    inside = 0
    for i, c in enumerate(win):
        h = cont[c]["hysteresis"]
        lo, hi_ = h["p_reverse"], h["p_forward"]
        s = rows[c]["sigma_obs"]
        inside += lo < s < hi_
        ax.plot([lo, hi_], [i, i], color=WINDOW, lw=5.5, solid_capstyle="butt",
                zorder=2)
        ax.plot([lo, hi_], [i, i], color=st.INK["secondary"], lw=0.6, zorder=3)
        for edge, col in ((lo, WARM), (hi_, COLD)):
            ax.plot([edge, edge], [i - 0.24, i + 0.24], color=col, lw=1.2,
                    zorder=4)
        ax.plot([s], [i], marker="*", ms=7, color=INK, mec="white", mew=0.5,
                zorder=6)

    ax.set_yticks(y); ax.set_yticklabels(win, fontsize=5.6)
    ax.set_xlabel("Sea-ice formation rate $\\Sigma$ (m yr$^{-1}$)")
    ax.set_xlim(0, 32)
    ax.set_ylim(-0.7, len(win) - 0.3)
    ax.legend(handles=[
        Patch(facecolor=WINDOW, edgecolor=st.INK["secondary"], lw=0.6,
              label="Bistable window"),
        Line2D([], [], color=INK, marker="*", ls="", ms=6,
               label="Derived forcing"),
        Line2D([], [], color=WARM, lw=1.2,
               label="$\\Sigma_{\\mathrm{rev}}$"),
        Line2D([], [], color=COLD, lw=1.2,
               label="$\\Sigma_{\\mathrm{fwd}}$"),
    ], fontsize=5.2, loc="upper center", frameon=False, ncol=4,
        bbox_to_anchor=(0.5, -0.30))
    st.soften_grid(ax, axis="x")
    st.panel_label(ax, "c", dx=-0.30)

    log(f"{inside}/{len(win)} cavities inside the window with derived forcing")
    return st.save(fig, path)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", type=Path, required=True,
                   help="directory holding the Nakata and Burgard 5 km fields")
    p.add_argument("--boxmodel", type=Path, default=None)
    p.add_argument("--outdir", type=Path, default=None)
    p.add_argument("--radius", type=float, action="append", default=None,
                   help="polynya search radius from the ice front, m "
                        "(repeat for a sensitivity test; first is plotted)")
    args = p.parse_args()

    ensure_dirs()
    radii = args.radius or [150e3]
    outdir = args.outdir or FIGURE_DIR
    outdir.mkdir(parents=True, exist_ok=True)
    cont = json.loads(((args.boxmodel or BOXMODEL_DIR)
                       / "continuation_sigma.json").read_text())

    every = {}
    for r in radii:
        log(f"assigning polynya cells within {r/1e3:.0f} km of the ice front")
        every[r] = polynya_stats(args.data, r)

    main_stats = every[radii[0]]
    written = figure(main_stats, cont, outdir / "figS07_seaice_forcing")

    print()
    print(f"{'cavity':16s} {'Ap(1e9 m2)':>11s} {'xi(m/30d)':>10s} "
          f"{'Sig_obs':>8s} {'Sig_used':>9s} {'ratio':>6s}")
    for c, v in main_stats["cavities"].items():
        print(f"{c:16s} {v['polynya_area']/1e9:11.1f} {v['xi_m_per_30d']:10.2f} "
              f"{v['sigma_obs']:8.1f} {v['sigma_assumed']:9.1f} "
              f"{v['sigma_obs']/v['sigma_assumed']:6.2f}")

    if len(radii) > 1:
        print()
        print("sensitivity of the derived Sigma to the search radius (m/yr):")
        hdr = "  ".join(f"{r/1e3:6.0f}km" for r in radii)
        print(f"{'cavity':16s} {hdr}")
        for c in main_stats["cavities"]:
            vals = "  ".join(
                f"{every[r]['cavities'].get(c, {}).get('sigma_obs', float('nan')):8.1f}"
                for r in radii)
            print(f"{c:16s} {vals}")

    out = {str(int(r)): {c: v for c, v in every[r]["cavities"].items()}
           for r in radii}
    dest = (RUN_DIR / "analysis") / "seaice_forcing.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, indent=2))

    for w in written:
        print(f"wrote {w}")
    print(f"wrote {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
