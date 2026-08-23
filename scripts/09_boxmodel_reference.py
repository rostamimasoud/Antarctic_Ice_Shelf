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
from aisgnn.boxmodel.rtipping import critical_rate, run_ramp  # noqa: E402
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

def compute_ramp_trajectories(params: dict, cavity: str, cont_sigma: dict,
                              durations=(5.0, 25.0, 100.0, 500.0)) -> dict:
    """Ramp sea-ice formation towards the fold at several rates.

    Two sets of runs.  The first ramps to a target \SI{5}{\percent} short of the
    quasi-static threshold: if any of these tipped, the tipping would be
    rate-induced, because the forcing never reaches the bifurcation.  The second
    is a positive control that ramps \SI{5}{\percent} past the threshold and must
    tip, which is what shows that a null result from the first set reflects the
    system rather than a detector that never fires.
    """
    rec = cont_sigma.get(cavity, {})
    folds = [f["p"] for f in rec.get("folds", [])
             if f["direction"] == "cold_to_warm" and f["p"] < rec.get("present_day", 0)]
    if not folds:
        return {}

    p_bif = max(folds)
    p0 = rec["present_day"]
    obs = OBSERVATIONS[cavity]
    base = BoxParams(CAVITIES[cavity], T_cdw=obs.T_cdw, sigma=obs.sigma, **params)

    out = {"cavity": cavity, "p_present": p0, "p_bifurcation": p_bif,
           "safe": [], "control": None}

    for tau in durations:
        target = p0 + 0.95 * (p_bif - p0)
        res = run_ramp(base, "sigma", p0, target, tau)
        out["safe"].append({"tau": tau, "target": target, "tipped": res.tipped,
                            "t": res.t[::4].tolist(),
                            "melt": res.melt[::4].tolist()})
        log(f"  tau={tau:6.1f} yr to sigma={target:5.2f} (95% of the way): "
            f"tipped={res.tipped}")

    target = p0 + 1.05 * (p_bif - p0)
    res = run_ramp(base, "sigma", p0, target, durations[0])
    out["control"] = {"tau": durations[0], "target": target, "tipped": res.tipped,
                      "t": res.t[::4].tolist(), "melt": res.melt[::4].tolist()}
    log(f"  positive control to sigma={target:5.2f} (105%): tipped={res.tipped}")
    return out


def make_figure(cont_T: dict, cont_sigma: dict, ramps: dict, path: Path,
                cavity: str = "Ross") -> list[str]:
    """Dynamical-systems characterisation of one cavity, in four panels.

    Panels c and d previously showed hysteresis widths taken from the CDW
    continuation and a rate-induced tipping summary.  Both were empty by
    construction: no cavity is bistable in CDW temperature, so the first had
    nothing to plot, and no cavity exhibits rate-induced tipping, so the second
    had nothing either.  They are replaced by the two diagnostics that do carry
    information: the leading eigenvalue along the branch, which is the formal
    signature of the saddle-node, and the ramp experiment itself, which shows the
    null result together with the positive control that validates it.
    """
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    from aisgnn.viz import style as st

    st.use_style()
    fig, axes = plt.subplots(2, 2, figsize=(st.WIDTH_DOUBLE, 4.9))

    def branches(ax, rec, xlabel, shade=True):
        p = np.asarray(rec["p"])
        m = np.asarray(rec["melt"])
        stable = np.asarray(rec["stable"], bool)
        hy = rec.get("hysteresis")
        if shade and hy:
            ax.axvspan(hy["p_reverse"], hy["p_forward"],
                       color=st.REGIME["bistable"], zorder=0, lw=0)
        mid = 0.5 * (m.max() + m.min())
        edges = np.flatnonzero(np.diff(stable.astype(int)) != 0) + 1
        for seg in np.split(np.arange(p.size), edges):
            if seg.size < 2:
                continue
            ok = bool(stable[seg[0]])
            warm = m[seg].mean() > mid
            ax.plot(p[seg], m[seg],
                    color=(st.REGIME["warm"] if warm else st.REGIME["cold"])
                    if ok else st.REGIME["unstable"],
                    ls="-" if ok else (0, (3, 2)), lw=1.4 if ok else 1.0,
                    zorder=3 if ok else 2)
        for f in rec["folds"]:
            ax.plot([f["p"]], [f["melt"]], "o", ms=4.0, color=st.INK["primary"],
                    mec="white", mew=0.6, zorder=5)
        ax.axvline(rec["present_day"], color=st.INK["muted"], lw=0.7,
                   ls=(0, (1, 2)), zorder=1)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Melt rate (m yr$^{-1}$)")
        st.soften_grid(ax)

    # (a) CDW temperature: monotone, no fold anywhere in range.
    ax = axes[0, 0]
    rec_T = cont_T.get(cavity)
    if rec_T and rec_T.get("ok"):
        branches(ax, rec_T, "CDW temperature $T_{\\mathrm{CDW}}$ ($^\\circ$C)",
                 shade=False)
        ax.text(0.96, 0.08, "no fold in range", transform=ax.transAxes,
                fontsize=5.8, ha="right", va="bottom", color=st.INK["secondary"])
    else:
        unavailable(ax, "no CDW continuation")
    st.panel_label(ax, "a")

    # (b) Sea-ice formation: the parameter that carries the bifurcation.
    ax = axes[0, 1]
    rec_s = cont_sigma.get(cavity)
    if rec_s and rec_s.get("ok"):
        branches(ax, rec_s, "Sea-ice formation rate $\\Sigma$ (m yr$^{-1}$)")
        ax.legend(handles=[
            Line2D([], [], color=st.REGIME["cold"], lw=1.4, label="Cold, stable"),
            Line2D([], [], color=st.REGIME["warm"], lw=1.4, label="Warm, stable"),
            Line2D([], [], color=st.REGIME["unstable"], lw=1.0, ls=(0, (3, 2)),
                   label="Unstable"),
            Patch(facecolor=st.REGIME["bistable"], label="Bistable window"),
        ], loc="upper right", fontsize=5.2, frameon=True, edgecolor="none",
            facecolor="white", framealpha=0.9)
    else:
        unavailable(ax, "no continuation in sigma")
    st.panel_label(ax, "b")

    # (c) Leading eigenvalue: zero exactly at the folds.
    ax = axes[1, 0]
    if rec_s and rec_s.get("ok") and rec_s.get("eig_max"):
        p = np.asarray(rec_s["p"])
        lam = np.asarray(rec_s["eig_max"])
        stable = np.asarray(rec_s["stable"], bool)
        scale = 1e9                      # eigenvalues are O(1e-8) per second
        ax.axhline(0.0, color=st.INK["primary"], lw=0.8, zorder=4)
        ax.plot(p[stable], lam[stable] * scale, ".", ms=1.6,
                color=st.REGIME["cold"], zorder=3)
        ax.plot(p[~stable], lam[~stable] * scale, ".", ms=1.6,
                color=st.REGIME["unstable"], zorder=3)
        for f in rec_s["folds"]:
            ax.axvline(f["p"], color=st.INK["muted"], lw=0.7, ls=(0, (2, 2)),
                       zorder=1)
        ax.set_xlabel("Sea-ice formation rate $\\Sigma$ (m yr$^{-1}$)")
        ax.set_ylabel("Leading eigenvalue\n$\\mathrm{Re}(\\lambda)$ ($10^{-9}$ s$^{-1}$)")
        ax.legend(handles=[
            Line2D([], [], color=st.REGIME["cold"], marker=".", ls="", ms=5,
                   label="Stable, $\\mathrm{Re}(\\lambda)<0$"),
            Line2D([], [], color=st.REGIME["unstable"], marker=".", ls="", ms=5,
                   label="Unstable, $\\mathrm{Re}(\\lambda)>0$"),
        ], loc="upper right", fontsize=5.2, frameon=False)
        st.soften_grid(ax)
    else:
        unavailable(ax, "no eigenvalues stored")
    st.panel_label(ax, "c")

    # (d) Ramp experiment: the null result with its positive control.
    ax = axes[1, 1]
    if ramps and ramps.get("safe"):
        # The safe ramps differ only in rate, which is a magnitude, so they take
        # a sequential ramp; the control is a different category and keeps the
        # warm identity colour.  Using the categorical set for both put the
        # 25-year ramp in the same vermillion as the control.
        import matplotlib.pyplot as _plt
        shades = _plt.get_cmap("aisgnn_seq")(np.linspace(0.35, 0.95,
                                                         len(ramps["safe"])))
        for colour, run in zip(shades, ramps["safe"]):
            ax.plot(run["t"], run["melt"], color=colour, lw=1.1,
                    label=f"$\\tau$ = {run['tau']:.0f} yr")
        ctrl = ramps.get("control")
        if ctrl:
            ax.plot(ctrl["t"], ctrl["melt"], color=st.REGIME["warm"], lw=1.5,
                    ls=(0, (4, 2)), zorder=5,
                    label="past threshold (control)")
        ax.set_xscale("symlog", linthresh=10)
        ax.set_xlabel("Time (yr)")
        ax.set_ylabel("Melt rate (m yr$^{-1}$)")
        ax.legend(loc="upper left", fontsize=5.2, frameon=False, ncol=2)
        st.soften_grid(ax)
    else:
        unavailable(ax, "no ramp experiments")
    st.panel_label(ax, "d")

    fig.tight_layout()
    return st.save(fig, path)


def unavailable(ax, message: str) -> None:
    ax.text(0.5, 0.5, message, ha="center", va="center", fontsize=6.5,
            transform=ax.transAxes)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


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
    p.add_argument("--figure-cavity", default="Ross",
                   help="cavity shown in the bifurcation figure")
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

    log("ramp trajectories for the rate-tipping panel")
    ramps = compute_ramp_trajectories(cal.params, args.figure_cavity, cont_sigma)
    (outdir / "ramps.json").write_text(json.dumps(ramps))

    if not args.no_figure:
        paths = make_figure(cont_T, cont_sigma, ramps,
                            FIGURE_DIR / "fig05_boxmodel_bifurcation",
                            cavity=args.figure_cavity)
        log(f"figure written: {', '.join(paths)}")

    log(f"results in {outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
