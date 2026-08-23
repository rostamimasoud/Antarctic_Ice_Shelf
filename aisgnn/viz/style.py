"""Figure styling for the manuscript.

The categorical palette below was checked with a colour-vision-deficiency
validator: in this order every adjacent pair clears a deutan/tritan separation
of dE >= 8 (OKLab x100) and a normal-vision separation of dE >= 20.  The order is
therefore fixed -- assign hues by position and never cycle or reshuffle them,
because reordering reintroduces the pairs that fail.

Three of the hues sit below 3:1 contrast against a white surface, so every chart
that uses them must carry a legend or direct labels; identity is never conveyed
by colour alone.  That is the house rule here in any case.
"""

from __future__ import annotations

from collections.abc import Sequence

import matplotlib as mpl
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# --------------------------------------------------------------------------- #
# Palettes
# --------------------------------------------------------------------------- #

#: Validated categorical order.  Do not permute.
CATEGORICAL: tuple[str, ...] = (
    "#0072B2",   # blue
    "#D55E00",   # vermillion
    "#009E73",   # bluish green
    "#E69F00",   # orange
    "#CC79A7",   # reddish purple
    "#56B4E9",   # sky blue
)

#: Circulation regimes.  These are identities, not magnitudes.
REGIME = {
    "cold": "#0072B2",
    "warm": "#D55E00",
    "unstable": "#8C8C8C",
    "bistable": "#E8E4DC",
}

#: Model / data provenance.
SOURCE = {
    "box_model": "#0072B2",
    "emulator": "#D55E00",
    "nemo": "#009E73",
    "roms_utas": "#E69F00",
    "mitgcm": "#CC79A7",
    "observations": "#3A3A3A",
}

#: Ink tokens.  Text never wears a series colour.
INK = {"primary": "#1A1A1A", "secondary": "#4A4A4A", "muted": "#7A7A7A",
       "grid": "#DCDCDC", "surface": "#FFFFFF"}

#: Sequential ramp for magnitude: a single hue, light to dark.
SEQUENTIAL = LinearSegmentedColormap.from_list(
    "aisgnn_seq", ["#EAF3F8", "#B8D8EA", "#7FB8D8", "#3D8FBF", "#0072B2", "#004E7A"])

#: Diverging ramp for signed anomalies: two hues about a neutral grey midpoint.
DIVERGING = LinearSegmentedColormap.from_list(
    "aisgnn_div", ["#00485F", "#0072B2", "#8FBED5", "#EFEFEF",
                   "#E9A87C", "#D55E00", "#8A3D00"])

#: Melt-rate ramp: perceptually ordered, light for refreezing through to dark for
#: intense melt.  Registered so it can be referenced by name.
MELT = LinearSegmentedColormap.from_list(
    "aisgnn_melt", ["#F7F7F5", "#CFE3EC", "#8FC0D6", "#E8C07A", "#D55E00", "#7A2E00"])

def _register(cmap) -> None:
    """Register a colormap across matplotlib versions, ignoring re-registration."""
    try:
        mpl.colormaps.register(cmap)                       # matplotlib >= 3.6
        return
    except AttributeError:
        pass
    except ValueError:
        return                                             # already registered
    try:
        mpl.cm.register_cmap(name=cmap.name, cmap=cmap)    # matplotlib < 3.6
    except (AttributeError, ValueError):
        pass


for _cm in (SEQUENTIAL, DIVERGING, MELT):
    _register(_cm)


# --------------------------------------------------------------------------- #
# Nature-style rcParams
# --------------------------------------------------------------------------- #

#: Nature column widths in inches.
WIDTH_SINGLE = 3.50    # 89 mm
WIDTH_ONE_HALF = 4.72  # 120 mm
WIDTH_DOUBLE = 7.20    # 183 mm

RC = {
    "figure.dpi": 150,
    # Nature: 300 dpi halftone, 600 combination, 1000-1200 line art.
    # PDF output is vector so unaffected; this governs the PNG copies.
    "savefig.dpi": 600,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
    "savefig.facecolor": INK["surface"],
    "figure.facecolor": INK["surface"],
    "axes.facecolor": INK["surface"],

    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 7,
    "axes.labelsize": 7,
    "axes.titlesize": 8,
    "xtick.labelsize": 6.5,
    "ytick.labelsize": 6.5,
    "legend.fontsize": 6.5,

    # Recessive frame: only the axes the reader needs.
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.6,
    "axes.edgecolor": INK["secondary"],
    "axes.labelcolor": INK["primary"],
    "axes.titlelocation": "left",
    "axes.titleweight": "bold",
    "axes.titlepad": 4.0,

    "xtick.color": INK["secondary"],
    "ytick.color": INK["secondary"],
    "xtick.labelcolor": INK["primary"],
    "ytick.labelcolor": INK["primary"],
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.major.size": 2.5,
    "ytick.major.size": 2.5,
    "xtick.direction": "out",
    "ytick.direction": "out",

    "grid.color": INK["grid"],
    "grid.linewidth": 0.4,
    "axes.grid": False,

    # Thin marks.
    "lines.linewidth": 1.2,
    "lines.markersize": 3.0,
    "lines.markeredgewidth": 0.0,
    "patch.linewidth": 0.5,

    "legend.frameon": False,
    "legend.handlelength": 1.4,
    "legend.handletextpad": 0.5,
    "legend.columnspacing": 1.0,
    "legend.labelspacing": 0.3,

    "axes.prop_cycle": mpl.cycler(color=list(CATEGORICAL)),
    "errorbar.capsize": 1.5,
    "pdf.fonttype": 42,      # embed as TrueType so text stays editable
    "ps.fonttype": 42,
}


def use_style() -> None:
    """Apply the manuscript rcParams globally.

    Keys unknown to the installed matplotlib are skipped rather than raising, so
    the same figures build on an older local install and on the cluster.
    """
    known = {k: v for k, v in RC.items() if k in mpl.rcParams}
    mpl.rcParams.update(known)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def figure(width: float = WIDTH_DOUBLE, height: float | None = None,
           nrows: int = 1, ncols: int = 1, **kwargs):
    """Create a styled figure sized to a Nature column width."""
    use_style()
    height = height if height is not None else width * 0.62
    return plt.subplots(nrows, ncols, figsize=(width, height), **kwargs)


def panel_label(ax, letter: str, dx: float = -0.14, dy: float = 1.06) -> None:
    """Place a bold panel letter in the axes' upper left."""
    ax.text(dx, dy, letter, transform=ax.transAxes, fontsize=8.5,
            fontweight="bold", va="top", ha="left", color=INK["primary"])


def soften_grid(ax, axis: str = "y") -> None:
    """Add a recessive grid behind the data."""
    ax.grid(True, axis=axis, color=INK["grid"], linewidth=0.4, zorder=0)
    ax.set_axisbelow(True)


def direct_label(ax, x: float, y: float, text: str, color: str,
                 dx: float = 3.0, **kwargs) -> None:
    """Label a series at its end.

    The marker beside the text carries the identity; the text itself stays in
    ink, per the house rule that text never wears a series colour.
    """
    ax.annotate(text, xy=(x, y), xytext=(dx, 0), textcoords="offset points",
                va="center", ha="left", fontsize=6.5, color=INK["primary"], **kwargs)
    ax.plot([x], [y], marker="o", markersize=3.0, color=color, zorder=5,
            markeredgecolor=INK["surface"], markeredgewidth=0.5)


def categorical(n: int) -> list[str]:
    """First ``n`` categorical hues, in the validated order.

    Raises
    ------
    ValueError
        If more than six are requested.  A seventh generated hue would not have
        been validated; group the tail into 'Other' or use small multiples.
    """
    if n > len(CATEGORICAL):
        raise ValueError(
            f"{n} categorical hues requested but only {len(CATEGORICAL)} are validated; "
            f"group the remainder or switch to small multiples")
    return list(CATEGORICAL[:n])


def symmetric_norm(values: Sequence[float] | np.ndarray, quantile: float = 0.99):
    """A diverging normaliser centred on zero, robust to outliers."""
    v = np.abs(np.asarray(values, float))
    v = v[np.isfinite(v)]
    lim = float(np.quantile(v, quantile)) if v.size else 1.0
    return mpl.colors.Normalize(vmin=-lim, vmax=lim)


def save(fig, path, formats: Sequence[str] = ("pdf", "png")) -> list[str]:
    """Save a figure in several formats, returning the paths written."""
    from pathlib import Path

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    written = []
    for fmt in formats:
        out = path.with_suffix(f".{fmt}")
        fig.savefig(out, format=fmt)
        written.append(str(out))
    return written
