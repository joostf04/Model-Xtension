import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

_METHODS = ["IPAD", "IPAD-disc", "IPAD-roll", "IPAD-oracle", "OCMT", "OCMT-dw"]
_METRICS = ["fdr_local", "fdr_avg", "power_local", "r2_oos", "avg_discoveries"]


def load_results(path: str = "results/raw.parquet") -> pd.DataFrame:
    return pd.read_parquet(path)


def summarize(results: pd.DataFrame) -> pd.DataFrame:
    group_cols = ["tau", "overlap", "snr", "delta", "method"]
    agg = {
        f"{m}_{stat}": pd.NamedAgg(column=m, aggfunc=stat)
        for m in _METRICS if m in results.columns
        for stat in ("mean", "std")
    }
    return results.groupby(group_cols).agg(**agg).reset_index()


def table_fdr_power(
    summary: pd.DataFrame,
    null: str = "local",
    delta: float = 0.99,
    snr: float = 4,
) -> pd.DataFrame:
    """Pivot table: rows = (tau, overlap), cols = (method × {fdr, power}).

    null: 'local' or 'avg'.  Returns a DataFrame ready for .to_latex().
    """
    sub = summary[(summary["delta"] == delta) & (summary["snr"] == snr)]
    rows = []
    for (tau, overlap), grp in sub.groupby(["tau", "overlap"]):
        row: dict = {"tau": tau, "overlap": overlap}
        for method in _METHODS:
            m = grp[grp["method"] == method]
            if len(m) == 1:
                row[f"{method}_fdr"]   = round(m[f"fdr_{null}_mean"].item(), 3)
                row[f"{method}_power"] = round(m["power_local_mean"].item(), 3)
        rows.append(row)
    return pd.DataFrame(rows)


def _facet_plot(
    summary: pd.DataFrame, metric: str, delta: float, ylabel: str
) -> matplotlib.figure.Figure:
    taus = sorted(summary["tau"].unique())
    snrs = sorted(summary["snr"].unique())
    sub = summary[summary["delta"] == delta]

    fig, axes = plt.subplots(
        len(taus), len(snrs),
        figsize=(4.5 * len(snrs), 3.5 * len(taus)),
        sharey=True, sharex=True,
    )
    axes = np.array(axes).reshape(len(taus), len(snrs))

    for i, tau in enumerate(taus):
        for j, snr in enumerate(snrs):
            ax = axes[i, j]
            df = sub[(sub["tau"] == tau) & (sub["snr"] == snr)]
            for method in _METHODS:
                mdf = df[df["method"] == method].sort_values("overlap")
                ax.plot(mdf["overlap"], mdf[metric], marker="o", label=method, lw=1.5)
            if "fdr" in metric:
                ax.axhline(0.20, color="k", ls="--", lw=0.8, label="q = 0.20")
            ax.set_title(f"τ = {tau},  SNR = {snr}", fontsize=9)
            ax.set_xlabel("Overlap fraction")
            ax.set_ylabel(ylabel)

    handles, labels = axes[0, -1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(labels),
               fontsize=8, bbox_to_anchor=(0.5, -0.04))
    fig.tight_layout()
    return fig


def plot_fdr_by_overlap(
    summary: pd.DataFrame, null: str = "local", delta: float = 0.99
) -> matplotlib.figure.Figure:
    return _facet_plot(summary, f"fdr_{null}_mean", delta, f"FDR ({null})")


def plot_power_by_overlap(
    summary: pd.DataFrame, delta: float = 0.99
) -> matplotlib.figure.Figure:
    return _facet_plot(summary, "power_local_mean", delta, "Power (local)")


def plot_delta_sensitivity(
    summary: pd.DataFrame, tau: float = 0.7, overlap: float = 0.5
) -> matplotlib.figure.Figure:
    sub = summary[(summary["tau"] == tau) & (summary["overlap"] == overlap)]
    snrs = sorted(sub["snr"].unique())

    fig, axes = plt.subplots(1, len(snrs), figsize=(5 * len(snrs), 4))
    axes = np.array(axes).flatten()

    for ax, snr in zip(axes, snrs):
        df = sub[(sub["snr"] == snr) & (sub["method"] == "IPAD-disc")].sort_values("delta")
        ax2 = ax.twinx()
        ax.plot(df["delta"], df["fdr_local_mean"], "b-o", label="FDR (local)")
        ax2.plot(df["delta"], df["power_local_mean"], "r--s", label="Power (local)")
        ax.axhline(0.20, color="b", ls=":", lw=0.8)
        ax.set_xlabel("δ")
        ax.set_ylabel("FDR (local)", color="b")
        ax2.set_ylabel("Power (local)", color="r")
        ax.set_title(f"Discounted IPAD  |  τ={tau}, overlap={overlap}, SNR={snr}")

    fig.tight_layout()
    return fig
