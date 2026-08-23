#!/usr/bin/env python3
"""Box-model reference: calibration, bifurcation structure and rate-induced tipping.

Produces the independent ground truth against which the emulator's reconstructed
bifurcation diagrams are later checked:

1. calibrate the shared process parameters on four well-observed cavities;
2. continue the steady states of every cavity in CDW temperature and in sea-ice
   formation rate, resolving the unstable branch and locating both saddle-nodes;
3. quantify hysteresis width and the melt jump at each fold;
4. bisect on the forcing rate to find where rate-induced tipping sets in;
5. write results to ``runs/boxmodel`` and draw the summary figure.

Examples
--------
    python scripts/09_boxmodel_reference.py                     # everything
    python scripts/09_boxmodel_reference.py --cavities Filchner-Ronne Ross
    python scripts/09_boxmodel_reference.py --skip-rtipping     # faster
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from aisgnn.boxmodel.calibrate import (                      # noqa: E402
    CALIBRATION_SET,
    OBSERVATIONS,
    calibrate,
    report,
)
from aisgnn.boxmodel.cavity import CAVITIES, BoxParams, CavityBoxModel  # noqa: E402
from aisgnn.boxmodel.continuation import bifurcation_diagram  # noqa: E402
from aisgnn.boxmodel.rtipping import critical_rate            # noqa: E402
from aisgnn.config import BOXMODEL_DIR, FIGURE_DIR, ensure_dirs  # noqa: E402

#: Continuation ranges per control parameter: (start, min, max).
RANGES = {
    "T_cdw": (-1.0, 3.5),
    "sigma": (0.0, 45.0),
}


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# --------------------------------------------------------------------------- #
# Continuation over all cavities
# --------------------------------------------------------------------------- #

def run_continuation(cavities: list[str], params: dict[str, float],
                     parameter: str) -> dict[str, dict]:
    """Continue every cavity in one control parameter and collect the folds."""
    p_min, p_max = RANGES[parameter]
    out: dict[str, dict] = {}

    for cav in cavities:
        obs = OBSERVATIONS[cav]
        base = BoxParams(CAVITIES[cav], T_cdw=obs.T_cdw, sigma=obs.sigma, **params)
        p_start = obs.T_cdw if parameter == "T_cdw" else obs.sigma

        try:
            branch, folds, hyst = bifurcation_diagram(
                base, parameter, p_start=p_start, p_min=p_min, p_max=p_max,
                regime=obs.regime)
        except (RuntimeError, np.linalg.LinAlgError) as exc:
            log(f"  {cav:16s} {parameter:10s} continuation failed: {exc}")
            out[cav] = {"ok": False, "error": str(exc)}
            continue

        record = {
            "ok": True,
            "parameter": parameter,
            "p": branch.p.tolist(),
            "melt": branch.melt.tolist(),
            "melt_flux": branch.melt_flux.tolist(),
            "chi": branch.chi.tolist(),
            "q_total": branch.q_total.tolist(),
            "stable": branch.stable.tolist(),
            "eig_max": branch.eig_max.tolist(),
            "folds": [{"p": f.p, "melt": f.melt, "direction": f.direction,
                       "refined": f.refined, "residual": f.residual} for f in folds],
            "n_unstable": int((~branch.stable).sum()),
            "present_day": p_start,
        }
        if hyst is not None:
            record["hysteresis"] = {
                "p_forward": hyst.p_forward, "p_reverse": hyst.p_reverse,
                "width": hyst.width, "melt_jump": hyst.melt_jump,
            }

        out[cav] = record
        n_f = len(folds)
        width = f"{hyst.width:.3f}" if hyst else "monostable"
        log(f"  {cav:16s} {parameter:10s} {len(branch):5d} pts  "
            f"{record['n_unstable']:4d} unstable  {n_f} folds  width={width}")

    return out


# --------------------------------------------------------------------------- #
# Rate-induced tipping
# --------------------------------------------------------------------------- #

def run_rtipping(cavities: list[str], params: dict[str, float],
                 continuation: dict[str, dict],
                 parameter: str = "sigma") -> dict[str, dict]:
    """Locate the critical forcing rate for cavities that have a B-tipping fold.

    The ramp is applied to the sea-ice formation rate, because that is the
    parameter that actually carries the saddle-node: continuing in CDW
    temperature produces no fold for the cold cavities, so a temperature ramp has
    no quasi-static threshold to undershoot and rate-induced tipping is not
    defined for it.

    A cold cavity tips when sea-ice formation *falls* through the lower fold, so
    the ramp runs downward and the target is placed at a fraction of the way from
    the present-day value towards the threshold.  Any tipping that occurs is then
    unambiguously rate-induced: the forcing never reaches the bifurcation.
    """
    out: dict[str, dict] = {}

    for cav in cavities:
        rec = continuation.get(cav, {})
        if not rec.get("ok") or OBSERVATIONS[cav].regime != "cold":
            continue

        folds = [f for f in rec["folds"] if f["direction"] == "cold_to_warm"]
        if not folds:
            continue

        p0 = rec["present_day"]
        # The relevant threshold is the cold->warm fold the present state would
        # reach first; for a downward ramp that is the nearest one below p0.
        below = [f["p"] for f in folds if f["p"] < p0]
        p_bif = max(below) if below else min(f["p"] for f in folds)
        if p_bif == p0:
            continue

        obs = OBSERVATIONS[cav]
        base = BoxParams(CAVITIES[cav], T_cdw=obs.T_cdw, sigma=obs.sigma, **params)

        results = []
        for frac in (0.7, 0.8, 0.9, 0.95):
            target = p0 + frac * (p_bif - p0)
            try:
                res = critical_rate(base, parameter, p0, target,
                                    tau_bounds=(5.0, 5000.0), p_bifurcation=p_bif)
            except (RuntimeError, ValueError) as exc:
                log(f"  {cav:16s} target={target:6.2f}: {exc}")
                res = None

            if res is None:
                results.append({"fraction": frac, "p_target": target,
                                "tau_critical": None})
                continue

            results.append({
                "fraction": frac, "p_target": target,
                "tau_critical": res.tau_critical,
                "rate_critical": res.rate_critical,
                "threshold_reduction": res.threshold_reduction,
            })
            log(f"  {cav:16s} target={target:6.2f} m/yr "
                f"({100 * frac:.0f}% of the way to the B-threshold {p_bif:.2f})  "
                f"tips if ramped faster than {res.tau_critical:7.1f} yr")

        if any(r.get("tau_critical") for r in results):
            out[cav] = {"parameter": parameter, "p_present": p0,
                        "p_bifurcation": p_bif, "targets": results}
        else:
            log(f"  {cav:16s} no rate-induced tipping within the rate bracket")

    return out


# --------------------------------------------------------------------------- #
# Figure
# --------------------------------------------------------------------------- #

def make_figure(cont_T: dict, cont_sigma: dict, rtip: dict, path: Path) -> list[str]:
    """Four-panel summary of the box-model bifurcation structure."""
    import matplotlib.pyplot as plt

    from aisgnn.viz import style as st

    st.use_style()
    fig = plt.figure(figsize=(st.WIDTH_DOUBLE, 4.8))
    gs = fig.add_gridspec(2, 2, hspace=0.42, wspace=0.28)
    axes = [fig.add_subplot(gs[i, j]) for i in (0, 1) for j in (0, 1)]

    def draw_branch(ax, rec, xlabel):
        """Stable branches solid and coloured by regime; unstable branch dashed."""
        p = np.asarray(rec["p"])
        melt = np.asarray(rec["melt"])
        stable = np.asarray(rec["stable"], bool)

        hy = rec.get("hysteresis")
        if hy:
            ax.axvspan(hy["p_reverse"], hy["p_forward"], color=st.REGIME["bistable"],
                       zorder=0, lw=0)

        # Split into runs so the dashed unstable segment is not bridged.
        edges = np.flatnonzero(np.diff(stable.astype(int)) != 0) + 1
        for seg in np.split(np.arange(p.size), edges):
            if seg.size < 2:
                continue
            is_stable = bool(stable[seg[0]])
            warm = melt[seg].mean() > 0.5 * (melt.max() + melt.min())
            colour = st.REGIME["warm"] if warm else st.REGIME["cold"]
            ax.plot(p[seg], melt[seg],
                    color=colour if is_stable else st.REGIME["unstable"],
                    ls="-" if is_stable else (0, (3, 2)),
                    lw=1.4 if is_stable else 1.0, zorder=3 if is_stable else 2)

        for f in rec["folds"]:
            ax.plot([f["p"]], [f["melt"]], marker="o", ms=4.0,
                    color=st.INK["primary"], mec=st.INK["surface"], mew=0.6, zorder=5)

        ax.axvline(rec["present_day"], color=st.INK["muted"], lw=0.7,
                   ls=(0, (1, 2)), zorder=1)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Basal melt rate (m yr$^{-1}$)")
        st.soften_grid(ax)

    # (a), (b): the two control parameters for Filchner-Ronne.
    cav = "Filchner-Ronne"
    for ax, cont, xlabel, letter in (
        (axes[0], cont_T, "CDW temperature $T_{\\mathrm{CDW}}$ ($^\\circ$C)", "a"),
        (axes[1], cont_sigma, "Sea-ice formation rate $\\Sigma$ (m yr$^{-1}$)", "b"),
    ):
        rec = cont.get(cav)
        if rec and rec.get("ok"):
            draw_branch(ax, rec, xlabel)
            ax.text(0.04, 0.93, cav, transform=ax.transAxes,
                    fontsize=6.0, ha="left", va="top",
                    color=st.INK["secondary"])
        st.panel_label(ax, letter)

    # Legend by proxy: identity is never colour-alone, so it is spelled out.
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    axes[0].legend(handles=[
        Line2D([], [], color=st.REGIME["cold"], lw=1.4, label="Cold regime (stable)"),
        Line2D([], [], color=st.REGIME["warm"], lw=1.4, label="Warm regime (stable)"),
        Line2D([], [], color=st.REGIME["unstable"], lw=1.0, ls=(0, (3, 2)),
               label="Unstable branch"),
        Patch(facecolor=st.REGIME["bistable"], label="Bistable window"),
    ], loc="upper left", fontsize=6.0)

    # (c): hysteresis width in CDW temperature across cavities.
    ax = axes[2]
    rows = [(c, r["hysteresis"]["width"], r["hysteresis"]["melt_jump"])
            for c, r in cont_T.items()
            if r.get("ok") and "hysteresis" in r]
    rows.sort(key=lambda t: t[1])
    if rows:
        names = [r[0] for r in rows]
        widths = [r[1] for r in rows]
        y = np.arange(len(rows))
        ax.barh(y, widths, height=0.6, color=st.REGIME["cold"], zorder=3)
        ax.set_yticks(y)
        ax.set_yticklabels(names, fontsize=6.0)
        ax.set_xlabel("Hysteresis width $\\Delta T_{\\mathrm{hyst}}$ ($^\\circ$C)")
        for yi, w in zip(y, widths):
            ax.text(w + 0.02, yi, f"{w:.2f}", va="center", fontsize=5.5,
                    color=st.INK["secondary"])
        st.soften_grid(ax, axis="x")
    else:
        ax.text(0.5, 0.5, "no bistable cavities found", ha="center", va="center",
                transform=ax.transAxes, color=st.INK["muted"], fontsize=6.5)
        ax.set_axis_off()
    st.panel_label(ax, "c")

    # (d): rate-induced tipping boundary.
    ax = axes[3]
    plotted = 0
    for colour, (cav_name, rec) in zip(st.CATEGORICAL, sorted(rtip.items())):
        pts = [(t["p_target"], t["tau_critical"]) for t in rec["targets"]
               if t.get("tau_critical")]
        if not pts:
            continue
        xs, ys = zip(*pts)
        ax.plot(xs, ys, marker="o", ms=3.0, color=colour, lw=1.2, label=cav_name)
        ax.axvline(rec["p_bifurcation"], color=colour, lw=0.7, ls=(0, (1, 2)))
        plotted += 1

    if plotted:
        ax.set_yscale("log")
        ax.set_xlabel("Target sea-ice formation rate $\\Sigma$ (m yr$^{-1}$)")
        ax.set_ylabel("Critical ramp duration (yr)")
        ax.legend(loc="best", fontsize=6.0)
        st.soften_grid(ax)
    else:
        ax.text(0.5, 0.5, "no rate-induced tipping detected", ha="center",
                va="center", transform=ax.transAxes, color=st.INK["muted"],
                fontsize=6.5)
        ax.set_axis_off()
    st.panel_label(ax, "d")

    return st.save(fig, path)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cavities", nargs="*", default=None,
                   help="cavities to analyse (default: all with observations)")
    p.add_argument("--skip-rtipping", action="store_true",
                   help="skip the rate-induced tipping bisection")
    p.add_argument("--no-figure", action="store_true")
    p.add_argument("--outdir", type=Path, default=None)
    args = p.parse_args()

    ensure_dirs()
    outdir = args.outdir or BOXMODEL_DIR
    outdir.mkdir(parents=True, exist_ok=True)

    cavities = args.cavities or list(OBSERVATIONS)
    unknown = [c for c in cavities if c not in OBSERVATIONS]
    if unknown:
        raise SystemExit(f"unknown cavities: {unknown}; known: {sorted(OBSERVATIONS)}")

    log(f"calibrating on {', '.join(CALIBRATION_SET)}")
    cal = calibrate()
    print(report(cal))
    (outdir / "calibration.json").write_text(json.dumps(
        {"params": cal.params, "cost": cal.cost, "success": cal.success,
         "fitted": cal.fitted, "predicted": cal.predicted}, indent=2))

    log("continuation in CDW temperature")
    cont_T = run_continuation(cavities, cal.params, "T_cdw")

    log("continuation in sea-ice formation rate")
    cont_sigma = run_continuation(cavities, cal.params, "sigma")

    rtip: dict[str, dict] = {}
    if not args.skip_rtipping:
        log("rate-induced tipping")
        rtip = run_rtipping(cavities, cal.params, cont_sigma, parameter="sigma")

    (outdir / "continuation_T_cdw.json").write_text(json.dumps(cont_T))
    (outdir / "continuation_sigma.json").write_text(json.dumps(cont_sigma))
    (outdir / "rtipping.json").write_text(json.dumps(rtip, indent=2))

    n_bistable = sum(1 for r in cont_T.values() if r.get("ok") and "hysteresis" in r)
    log(f"bistable in T_cdw: {n_bistable} of {len(cavities)} cavities")

    if not args.no_figure:
        paths = make_figure(cont_T, cont_sigma, rtip, FIGURE_DIR / "fig05_boxmodel_bifurcation")
        log(f"figure written: {', '.join(paths)}")

    log(f"results in {outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
