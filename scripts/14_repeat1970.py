#!/usr/bin/env python3
"""Compare each cavity's 1970s state with its present-day state.

The question is whether the past fifty years have moved cavities towards the
saddle-node they can actually reach.  That fold is not the same one for every
cavity, and getting it wrong inverts the answer:

* a cavity on the **cold** branch is lost when sea-ice formation *falls* through
  the lower fold, so its distance to danger is ``sigma - sigma_rev``;
* a cavity already on the **warm** branch cannot cross that fold at all.  The
  only fold it can reach is the upper one, and reaching it means *recovery*, so
  its distance is ``sigma_fwd - sigma``.

Sea-ice formation has fallen at every cavity since 1970.  Cold cavities
therefore move towards their tipping fold and warm cavities move away from their
recovery fold: both are a deterioration, but they are different statements and
the manuscript has to make them separately.

The attribution splits the movement between the change in sea-ice formation and
the change in CDW temperature.  The latter enters through the fold itself: the
saddle-node locus is a curve in the (sigma, T_cdw) plane, so warming the deep
water displaces the fold even though continuing in T_cdw alone produces no fold
at all.  ``d sigma_fold / d T_cdw`` is obtained by re-running the continuation at
perturbed T_cdw rather than assumed.

Examples
--------
    python scripts/14_repeat1970.py
    python scripts/14_repeat1970.py --dtcdw 0.3 --no-attribution
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
from matplotlib.patches import Patch                               # noqa: E402

from aisgnn.boxmodel.cavity import CAVITIES, BoxParams, CavityBoxModel  # noqa: E402
from aisgnn.boxmodel.continuation import bifurcation_diagram       # noqa: E402
from aisgnn.config import BOXMODEL_DIR, FIGURE_DIR, RUN_DIR, ensure_dirs  # noqa: E402
from aisgnn.viz import style as st                                 # noqa: E402

#: Sea-ice formation rate in the 1970s, m of ice per year.
#:
#: These are *supplied* values, not an output of this study and not present in
#: any table of the manuscript.  They are carried through the analysis as given
#: so the arithmetic is reproducible, and every result below is linear in the
#: difference from the present-day column, so a revised 1970s estimate rescales
#: the answer without changing its sign.
SIGMA_1970: dict[str, float] = {
    "Ross": 14.5,
    "Getz": 7.2,
    "Pine Island": 4.8,
    "Riiser-Larsen": 6.0,
    "Fimbul": 6.0,
    "Shackleton": 7.0,
    "Amery": 9.0,
    "Larsen C": 6.0,
    "Totten": 6.0,
}

#: Present-day forcing, from Supplementary Table S4 of the manuscript.
PRESENT: dict[str, tuple[float, float]] = {
    # cavity: (T_cdw degC, sigma m/yr)
    "Filchner-Ronne": (0.5, 10.0),
    "Ross": (0.3, 12.0),
    "Amery": (-0.3, 8.0),
    "Fimbul": (0.0, 5.0),
    "Larsen C": (-0.5, 5.0),
    "Riiser-Larsen": (0.0, 5.0),
    "Shackleton": (0.3, 6.0),
    "Totten": (0.5, 5.0),
    "Getz": (0.9, 6.0),
    "Pine Island": (1.2, 4.0),
    "Thwaites": (1.1, 4.0),
}

#: Observed present-day regime, from the same table.
REGIME: dict[str, str] = {
    "Filchner-Ronne": "cold", "Ross": "cold", "Amery": "cold", "Fimbul": "cold",
    "Larsen C": "cold", "Riiser-Larsen": "cold", "Shackleton": "warm",
    "Totten": "warm", "Getz": "warm", "Pine Island": "warm", "Thwaites": "warm",
}

COLD = "#2C7BB6"
WARM = "#D7191C"
WINDOW = "#E4E0D8"
INK = "#1A1A1A"


def log(msg: str) -> None:
    print(msg, flush=True)


def load(path: Path):
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None


def calibrated(bdir: Path) -> dict:
    """Fitted process parameters from the calibration run."""
    cal = load(bdir / "calibration.json")
    if not cal or "params" not in cal:
        raise SystemExit(f"no calibration in {bdir}; run scripts/03 first")
    return cal["params"]


def model_for(cavity: str, params: dict, sigma: float,
              t_cdw: float | None = None) -> CavityBoxModel:
    t = PRESENT[cavity][0] if t_cdw is None else t_cdw
    return CavityBoxModel(BoxParams(CAVITIES[cavity], T_cdw=t, sigma=sigma,
                                    **params))


# --------------------------------------------------------------------------- #
# 1 and 2: position in the window, and the state at each end
# --------------------------------------------------------------------------- #

def state_at(cavity: str, params: dict, sigma: float) -> dict | None:
    """Equilibrium melt and DSW fraction at one forcing value.

    The cavity is held in its observed regime.  If that regime has ceased to be
    an attractor at this forcing the function returns ``None`` rather than
    silently reporting the other branch, which would read as a smooth change
    when it is in fact a transition.
    """
    m = model_for(cavity, params, sigma)
    try:
        x = m.equilibrium(REGIME[cavity])
    except RuntimeError:
        return None
    d = m.diagnostics(x)
    return {"melt": float(d.melt_rate), "chi": float(d.chi_dsw),
            "thermal_driving": float(d.thermal_driving),
            "melt_flux": float(d.melt_flux)}


def positions(cont: dict, params: dict) -> list[dict]:
    """Distance to the reachable fold in 1970 and today, per cavity."""
    rows = []
    for cavity, s1970 in SIGMA_1970.items():
        rec = cont.get(cavity)
        if not (rec and rec.get("ok") and "hysteresis" in rec):
            log(f"  {cavity}: no resolved window, skipped")
            continue
        h = rec["hysteresis"]
        rev, fwd, width = h["p_reverse"], h["p_forward"], h["width"]
        now = PRESENT[cavity][1]
        regime = REGIME[cavity]

        # The fold this cavity can actually reach, and what reaching it means.
        if regime == "cold":
            fold, target = rev, "cold-to-warm (tipping)"
            d1970, dnow = s1970 - rev, now - rev
        else:
            fold, target = fwd, "warm-to-cold (recovery)"
            d1970, dnow = fwd - s1970, fwd - now

        rows.append({
            "cavity": cavity, "regime": regime, "target": target,
            "sigma_1970": s1970, "sigma_present": now,
            "sigma_rev": rev, "sigma_fwd": fwd, "width": width,
            "fold": fold,
            "d_1970": d1970, "d_present": dnow,
            "closer_by": d1970 - dnow,
            "pct_of_width": 100.0 * (d1970 - dnow) / width,
            "state_1970": state_at(cavity, params, s1970),
            "state_present": state_at(cavity, params, now),
        })
    return rows


# --------------------------------------------------------------------------- #
# 4: how much of the movement is sigma and how much is CDW warming
# --------------------------------------------------------------------------- #

def fold_sensitivity(cavity: str, params: dict, delta: float = 0.25,
                     ds: float = 0.02) -> dict | None:
    """``d sigma_fold / d T_cdw`` by central difference on the continuation.

    Both folds are tracked.  The difference stays two-sided throughout: a
    one-sided estimate would be reported to the same precision while being a
    different quantity.  Where the widest perturbation destroys the window --
    Totten's is only \\SI{6.2}{\\metre\\per\\year} across, and Riiser-Larsen's
    upper fold leaves the continued range -- the step is narrowed rather than
    made one-sided, and the step actually used is recorded so the table can
    report it.
    """
    t0 = PRESENT[cavity][0]
    for d in [delta] + [x for x in (0.15, 0.10, 0.05) if x < delta]:
        out = {}
        for sign in (-1, +1):
            base = BoxParams(CAVITIES[cavity], T_cdw=t0 + sign * d,
                             sigma=PRESENT[cavity][1], **params)
            try:
                _, _, hyst = bifurcation_diagram(base, "sigma",
                                                 PRESENT[cavity][1], 0.05, 45.0,
                                                 regime=REGIME[cavity], ds=ds)
            except RuntimeError:
                hyst = None
            if hyst is None:
                break
            out[sign] = hyst
        if len(out) == 2:
            return {
                "d_rev_dT": (out[+1].p_reverse - out[-1].p_reverse) / (2.0 * d),
                "d_fwd_dT": (out[+1].p_forward - out[-1].p_forward) / (2.0 * d),
                "delta": d,
            }
    return None


def attribute(rows: list[dict], params: dict, dt_cdw: float,
              delta: float) -> list[dict]:
    """Split the movement between the sigma change and the CDW warming.

    ``D`` is the distance to the reachable fold.  For a cold cavity
    ``D = sigma - sigma_rev(T_cdw)`` so

        dD = d(sigma) - (d sigma_rev / d T_cdw) * d(T_cdw),

    and for a warm cavity ``D = sigma_fwd(T_cdw) - sigma`` so the two terms swap
    sign.  A negative contribution shortens the distance, i.e. moves the cavity
    towards the fold.
    """
    out = []
    for r in rows:
        sens = fold_sensitivity(r["cavity"], params, delta=delta)
        if sens is None:
            log(f"  {r['cavity']}: fold sensitivity did not resolve, skipped")
            out.append({**r, "sens": None})
            continue
        d_sigma = r["sigma_present"] - r["sigma_1970"]
        if r["regime"] == "cold":
            from_sigma = d_sigma
            from_tcdw = -sens["d_rev_dT"] * dt_cdw
        else:
            from_sigma = -d_sigma
            from_tcdw = sens["d_fwd_dT"] * dt_cdw
        out.append({**r, "sens": sens, "d_sigma": d_sigma,
                    "from_sigma": from_sigma, "from_tcdw": from_tcdw,
                    "total": from_sigma + from_tcdw,
                    "pct_sigma_of_width": 100.0 * from_sigma / r["width"],
                    "pct_tcdw_of_width": 100.0 * from_tcdw / r["width"],
                    "pct_total_of_width": 100.0 * (from_sigma + from_tcdw)
                    / r["width"]})
    return out


# --------------------------------------------------------------------------- #
# 3: the trajectory figure
# --------------------------------------------------------------------------- #

def figure_trajectories(rows: list[dict], path: Path) -> list[str]:
    """One panel per cavity: the window, and the 1970-to-present move inside it."""
    st.use_style()
    fig, axes = plt.subplots(3, 3, figsize=(st.WIDTH_DOUBLE, 5.4), sharey=True)

    for ax, r in zip(axes.ravel(), rows):
        lo, hi = r["sigma_rev"], r["sigma_fwd"]
        span = hi - lo
        ax.axvspan(lo, hi, color=WINDOW, lw=0, zorder=0)
        ax.plot([lo, lo], [0.25, 0.75], color=WARM, lw=1.5, zorder=4)
        ax.plot([hi, hi], [0.25, 0.75], color=COLD, lw=1.5, zorder=4)
        # Mark the fold this cavity can actually reach; the other one is
        # unreachable from the branch it occupies and is drawn only for context.
        ax.plot([r["fold"]], [0.20], marker="^", ms=4.5, color=INK,
                mec="white", mew=0.5, zorder=8, clip_on=False)

        # The move itself, drawn as an arrow from 1970 to the present.
        ax.annotate("", xy=(r["sigma_present"], 0.5),
                    xytext=(r["sigma_1970"], 0.5),
                    arrowprops=dict(arrowstyle="-|>", lw=1.2, color=INK,
                                    mutation_scale=9, shrinkA=0, shrinkB=0),
                    zorder=6)
        ax.plot([r["sigma_1970"]], [0.5], "o", ms=4.0, color=COLD,
                mec="white", mew=0.6, zorder=7)
        ax.plot([r["sigma_present"]], [0.5], "o", ms=4.0, color=WARM,
                mec="white", mew=0.6, zorder=7)

        # Say which way it moved in words.  A signed percentage alone reads as
        # "less bad" for the warm cavities, when in fact they have moved further
        # from the only fold that would return them to the cold state.
        moved = (f"{r['pct_of_width']:.0f}% of $\\Delta\\Sigma$ closer to tipping"
                 if r["regime"] == "cold" else
                 f"{abs(r['pct_of_width']):.0f}% of $\\Delta\\Sigma$ "
                 f"further from recovery")
        ax.annotate(moved, xy=(0.5, 0.82), xycoords="axes fraction",
                    ha="center", fontsize=5.4, color=INK)
        ax.text(0.5, 0.06, f"{r['cavity']} ({r['regime']})",
                transform=ax.transAxes, ha="center", va="bottom",
                fontsize=6.2, color=st.INK["secondary"])

        pad = 0.10 * span
        ax.set_xlim(lo - pad, hi + pad)
        ax.set_ylim(0, 1)
        ax.set_yticks([])
        ax.spines["left"].set_visible(False)
        st.soften_grid(ax, axis="x")

    for ax in axes[-1]:
        ax.set_xlabel("Sea-ice formation rate $\\Sigma$ (m yr$^{-1}$)")

    fig.legend(handles=[
        Patch(facecolor=WINDOW, label="Bistable window"),
        Line2D([], [], color=WARM, lw=1.5,
               label="$\\Sigma_{\\mathrm{rev}}$: cold state lost"),
        Line2D([], [], color=COLD, lw=1.5,
               label="$\\Sigma_{\\mathrm{fwd}}$: warm state lost"),
        Line2D([], [], color=COLD, marker="o", ls="", ms=4, label="1970s"),
        Line2D([], [], color=WARM, marker="o", ls="", ms=4, label="Present day"),
        Line2D([], [], color=INK, marker="^", ls="", ms=4,
               label="Reachable fold"),
    ], fontsize=5.8, loc="lower center", ncol=6, frameon=False,
        bbox_to_anchor=(0.5, 0.0))

    fig.tight_layout(rect=(0, 0.055, 1, 1))
    return st.save(fig, path)


# --------------------------------------------------------------------------- #
# LaTeX
# --------------------------------------------------------------------------- #

def tex_positions(rows: list[dict]) -> str:
    body = "\n".join(
        f"{r['cavity']:15s} & {r['regime']:4s} & {r['sigma_1970']:5.1f} & "
        f"{r['sigma_present']:5.1f} & {r['fold']:5.2f} & {r['d_1970']:5.2f} & "
        f"{r['d_present']:5.2f} & {r['closer_by']:5.2f} & "
        f"{r['pct_of_width']:5.1f} \\\\"
        for r in rows)
    return f"""\\begin{{table}}[htbp]
\\centering
\\caption{{\\textbf{{Movement towards the reachable fold since the 1970s.}}
$\\Sigma_{{\\mathrm{{fold}}}}$ is the saddle-node the cavity can reach from the
branch it occupies: the lower fold $\\Sigma_{{\\mathrm{{rev}}}}$ for a cold
cavity, whose cold state is lost when sea-ice formation falls through it, and
the upper fold $\\Sigma_{{\\mathrm{{fwd}}}}$ for a warm cavity, which it would
have to cross upwards to recover. $D$ is the distance to that fold. Sea-ice
formation has fallen at every cavity, so cold cavities have moved towards
tipping and warm cavities away from recovery; the final column expresses the
movement as a percentage of the hysteresis width $\\Delta\\Sigma$. The 1970s
forcing is prescribed rather than derived here (Methods).}}
\\label{{tab:since1970}}
\\small
\\begin{{tabular}}{{llrrrrrrr}}
\\toprule
Cavity & Regime & $\\Sigma_{{1970}}$ & $\\Sigma_{{\\mathrm{{now}}}}$ &
$\\Sigma_{{\\mathrm{{fold}}}}$ & $D_{{1970}}$ & $D_{{\\mathrm{{now}}}}$ &
$\\Delta D$ & \\% of $\\Delta\\Sigma$ \\\\
 & & \\multicolumn{{6}}{{c}}{{(\\si{{\\metre\\per\\year}})}} & \\\\
\\midrule
{body}
\\bottomrule
\\end{{tabular}}
\\end{{table}}
"""


def tex_states(rows: list[dict]) -> str:
    def cell(s, key, fmt):
        return "--" if s is None else format(s[key], fmt)

    body = "\n".join(
        f"{r['cavity']:15s} & {r['regime']:4s} & "
        f"{cell(r['state_1970'], 'melt', '6.3f')} & "
        f"{cell(r['state_present'], 'melt', '6.3f')} & "
        f"{cell(r['state_1970'], 'chi', '6.4f')} & "
        f"{cell(r['state_present'], 'chi', '6.4f')} \\\\"
        for r in rows)
    return f"""\\begin{{table}}[htbp]
\\centering
\\caption{{\\textbf{{Box-model equilibrium in the 1970s and today.}}
Each cavity is held on the branch it is observed to occupy and re-equilibrated
at the two forcing values; $\\chi$ is the dense-shelf-water fraction of the
inflow, near one for a cold cavity and near zero for a warm one. The changes are
small because both forcings lie well inside the bistable window, far from either
fold: the model responds to a falling $\\Sigma$ gradually until the fold is
reached, and only then abruptly. This is the point of the bifurcation analysis
and the reason the equilibrium melt change alone understates the risk.}}
\\label{{tab:state1970}}
\\small
\\begin{{tabular}}{{llrrrr}}
\\toprule
& & \\multicolumn{{2}}{{c}}{{Melt rate (\\si{{\\metre\\per\\year}})}} &
\\multicolumn{{2}}{{c}}{{DSW fraction $\\chi$}} \\\\
\\cmidrule(lr){{3-4}} \\cmidrule(lr){{5-6}}
Cavity & Regime & 1970s & Present & 1970s & Present \\\\
\\midrule
{body}
\\bottomrule
\\end{{tabular}}
\\end{{table}}
"""


def tex_attribution(rows: list[dict], dt_cdw: float) -> str:
    def num(r, k, fmt):
        return "--" if r.get("sens") is None else format(r[k], fmt)

    body = "\n".join(
        f"{r['cavity']:15s} & {r['regime']:4s} & "
        f"{num(r, 'from_sigma', '6.2f')} & {num(r, 'from_tcdw', '6.2f')} & "
        f"{num(r, 'total', '6.2f')} & {num(r, 'pct_total_of_width', '5.1f')} & "
        f"{'--' if r.get('sens') is None else format(r['sens']['delta'], '4.2f')} \\\\"
        for r in rows)
    return f"""\\begin{{table}}[htbp]
\\centering
\\caption{{\\textbf{{Attribution of the movement towards the reachable fold.}}
The distance to the fold changes both because the forcing moves and because the
fold itself moves: the saddle-node locus is a curve in the
$(\\Sigma, T_{{\\mathrm{{CDW}}}})$ plane, so warming the deep water displaces it
even though continuing in $T_{{\\mathrm{{CDW}}}}$ alone produces no fold
(Supplementary Fig.~S1). The sensitivity
$\\partial\\Sigma_{{\\mathrm{{fold}}}}/\\partial T_{{\\mathrm{{CDW}}}}$ is
obtained by re-running the continuation at $T_{{\\mathrm{{CDW}}}} \\pm \\delta$
rather than assumed; $\\delta$ is the widest perturbation for which the window
resolves at both signs and is listed in the final column. Negative entries
shorten the distance to the fold. The $T_{{\\mathrm{{CDW}}}}$ column is
evaluated at an assumed warming of \\SI{{{dt_cdw:.2f}}}{{\\celsius}} since the
1970s and scales linearly with it; this study does not constrain that number, so
the column shows the relative size of the two pathways rather than an
attribution result.}}
\\label{{tab:attribution1970}}
\\small
\\begin{{tabular}}{{llrrrrr}}
\\toprule
Cavity & Regime & From $\\Delta\\Sigma$ & From $\\Delta T_{{\\mathrm{{CDW}}}}$ &
Total & \\% of $\\Delta\\Sigma$ & $\\delta$ (\\si{{\\celsius}}) \\\\
 & & \\multicolumn{{3}}{{c}}{{(\\si{{\\metre\\per\\year}})}} & & \\\\
\\midrule
{body}
\\bottomrule
\\end{{tabular}}
\\end{{table}}
"""


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--boxmodel", type=Path, default=None)
    p.add_argument("--outdir", type=Path, default=None)
    p.add_argument("--dtcdw", type=float, default=0.2,
                   help="assumed CDW warming since the 1970s, degC")
    p.add_argument("--delta", type=float, default=0.25,
                   help="half-width of the T_cdw perturbation for the sensitivity")
    p.add_argument("--no-attribution", action="store_true",
                   help="skip the continuation reruns, which take a few minutes")
    args = p.parse_args()

    ensure_dirs()
    bdir = args.boxmodel or BOXMODEL_DIR
    outdir = args.outdir or FIGURE_DIR
    outdir.mkdir(parents=True, exist_ok=True)

    params = calibrated(bdir)
    cont = load(bdir / "continuation_sigma.json")
    if cont is None:
        raise SystemExit(f"no continuation output in {bdir}")

    log("equilibrating each cavity at the 1970s and present-day forcing")
    rows = positions(cont, params)
    rows.sort(key=lambda r: -r["pct_of_width"])

    if not args.no_attribution:
        log(f"re-running continuation at T_cdw +/- {args.delta} degC "
            f"({len(rows)} cavities, a few minutes)")
        rows = attribute(rows, params, args.dtcdw, args.delta)

    written = figure_trajectories(rows, outdir / "figS07_since1970")

    tex = (tex_positions(rows) + "\n" + tex_states(rows)
           + ("\n" + tex_attribution(rows, args.dtcdw)
              if not args.no_attribution else ""))
    tex_path = (RUN_DIR / "analysis") / "since1970_tables.tex"
    tex_path.parent.mkdir(parents=True, exist_ok=True)
    tex_path.write_text(tex)

    json_path = (RUN_DIR / "analysis") / "since1970.json"
    json_path.write_text(json.dumps(
        {"dt_cdw_assumed": args.dtcdw, "rows": rows}, indent=2, default=float))

    print()
    print(f"{'cavity':15s} {'reg':5s} {'S1970':>6s} {'Snow':>6s} {'fold':>6s} "
          f"{'D1970':>6s} {'Dnow':>6s} {'dD':>6s} {'%dS':>6s}")
    for r in rows:
        print(f"{r['cavity']:15s} {r['regime']:5s} {r['sigma_1970']:6.1f} "
              f"{r['sigma_present']:6.1f} {r['fold']:6.2f} {r['d_1970']:6.2f} "
              f"{r['d_present']:6.2f} {r['closer_by']:6.2f} "
              f"{r['pct_of_width']:6.1f}")

    for w in written:
        print(f"wrote {w}")
    print(f"wrote {tex_path}")
    print(f"wrote {json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
