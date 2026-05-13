import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.patches as mpatches
import seaborn as sns

from config import PROJECT_DIR
from config import HUE_COL, PALETTE, EDGE_COLOR, GRID_COLOR, STEPS_EXTRA_BINS, PLANNING_BINS
"""
Usage:
    Call plot_steps_planning(df) with a DataFrame that contains
    'Steps_extra', 'plannings', and 'subject' columns.
    Rows with Steps_extra > 5 are grouped together
"""

# ── Helpers ────────────────────────────────────────────────────────────────
def _bin_steps_extra(series: pd.Series) -> pd.Series:
    """Map raw Steps_extra values to labelled bins."""
    labels = list(STEPS_EXTRA_BINS.keys())
    out    = pd.Series(pd.NA, index=series.index, dtype="object")
    for label, pred in STEPS_EXTRA_BINS.items():
        out[pred(series)] = label
    return pd.Categorical(out, categories=labels, ordered=True)


def _bin_plannings(series: pd.Series) -> pd.Series:
    """Map raw plannings values to labelled bins."""
    labels = list(PLANNING_BINS.keys())
    out    = pd.Series(pd.NA, index=series.index, dtype="object")
    for label, (lo, hi) in PLANNING_BINS.items():
        out[(series >= lo) & (series <= hi)] = label
    return pd.Categorical(out, categories=labels, ordered=True)


def _subject_rel_freq_binned(df: pd.DataFrame,
                              binned_col: str,
                              all_labels: list,
                              hue_col: str) -> pd.DataFrame:
    """
    Compute within-subject relative frequency for an already-binned column.
    Returns long-form DataFrame: [binned_col, hue_col, 'proportion'].
    """
    records = []
    for subj, grp in df.groupby(hue_col):
        counts = (grp[binned_col]
                  .value_counts(normalize=True)
                  .reindex(all_labels, fill_value=0))
        for val, prop in counts.items():
            records.append({binned_col: val, hue_col: subj, "proportion": prop})
    return pd.DataFrame(records)


def _format_ax(ax, xlabel, is_first, ylabel="Relative Frequency"):
    """Apply shared axis cosmetics."""
    ax.set_xlabel(xlabel, fontsize=10.5, labelpad=6, fontweight="semibold")
    ax.set_ylabel(ylabel if is_first else "", fontsize=9.5)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.2f}"))
    ax.tick_params(axis="both", labelsize=8.5, length=3)
    ax.yaxis.grid(True, color=GRID_COLOR, linewidth=0.7, linestyle="--", alpha=0.8)
    ax.set_axisbelow(True)
    sns.despine(ax=ax)


# ── Main plotting function ─────────────────────────────────────────────────
def plot_steps_planning(df: pd.DataFrame, outlier_thr = 5, exclude_outliers = False) -> None:
    """
    Three-panel figure:
      Left   — relative frequency of Steps_extra (binned, hue = subject)
      Middle — relative frequency of plannings   (binned, hue = subject)
      Right  — mean plannings ± 95 % CI by Steps_extra ≤5 (hue = subject)

    Rows with Steps_extra > 5 appear in the left panel but are excluded
    from the middle and right panels.

    Parameters
    ----------
    df : DataFrame with columns 'steps_extra', 'plannings', 'subject'.
    """
    subjects = sorted(df[HUE_COL].unique())

    # ── Bin raw columns ───────────────────────────────────────────────────
    df = df.copy()
    df["steps_extra_bin"] = _bin_steps_extra(df["steps_extra"])
    df["plannings_bin"]   = _bin_plannings(df["plannings"])

    steps_labels    = list(STEPS_EXTRA_BINS.keys())
    planning_labels = list(PLANNING_BINS.keys())

    # ── Style ─────────────────────────────────────────────────────────────
    sns.set_theme(style="ticks", context="paper", font_scale=1.15)

    fig, axes = plt.subplots(
        nrows=1, ncols=3,
        figsize=(14, 4.5),
        constrained_layout=True,
    )
    #fig.suptitle(
    #    "Steps Extra & Planning Fixations Analysis",
    #    fontsize=15, fontweight="bold", y=1.02,
    #)

    # ── LEFT: relative frequency of Steps_extra ───────────────────────────
    ax = axes[0]
    plot_df = _subject_rel_freq_binned(df, "steps_extra_bin", steps_labels, HUE_COL)
    sns.barplot(
        data=plot_df,
        x="steps_extra_bin", y="proportion",
        hue=HUE_COL, palette=PALETTE,
        edgecolor=EDGE_COLOR, linewidth=0.6, alpha=0.88,
        order=steps_labels, legend=False,
        ax=ax,
    )
    _format_ax(ax, "Extra Steps", is_first=True)

    # ── MIDDLE: relative frequency of plannings ───────────────────────────
    # Subset without >4 (for middle & right panels)
    ax = axes[1]
    if exclude_outliers:
        df_filtered = df[df["steps_extra"] <= outlier_thr].copy()
        
        plot_df2 = _subject_rel_freq_binned(
            df_filtered, "plannings_bin", planning_labels, HUE_COL
        )
    else:
        plot_df2 = _subject_rel_freq_binned(
            df, "plannings_bin", planning_labels, HUE_COL
        )

    sns.barplot(
        data=plot_df2,
        x="plannings_bin", y="proportion",
        hue=HUE_COL, palette=PALETTE,
        edgecolor=EDGE_COLOR, linewidth=0.6, alpha=0.88,
        order=planning_labels, legend=False,
        ax=ax,
    )
    ax.set_xticks(range(len(planning_labels)))
    ax.set_xticklabels(planning_labels, rotation=45, ha="right", fontsize=8)
    _format_ax(ax, "Planning Fixations", is_first=False)

    # ── RIGHT: mean plannings ± 95 % CI by Steps_extra group ─────────────
    ax = axes[2]
    # Work on filtered data (Steps_extra ≤ 5); use raw plannings on y
    steps_labels_filtered = [l for l in steps_labels if l != ">4"]
    if exclude_outliers:
        df_filtered = df[df["steps_extra"] <= outlier_thr].copy()
        sns.barplot(
            data=df_filtered,
            x="steps_extra_bin", y="plannings",
            hue=HUE_COL, palette=PALETTE,
            edgecolor=EDGE_COLOR, linewidth=0.6, alpha=0.88,
            order=steps_labels_filtered, #_filtered,
            errorbar=("ci", 95),
            capsize=0.08,
            err_kws={"linewidth": 1.2, "color": EDGE_COLOR},
            legend=False,
            ax=ax,
        )
    else:
        sns.barplot(
            data=df,
            x="steps_extra_bin", 
            y="plannings",
            hue=HUE_COL, palette=PALETTE,
            edgecolor=EDGE_COLOR, linewidth=0.6, alpha=0.88,
            order=steps_labels, #_filtered,
            errorbar=("ci", 95),
            capsize=0.08,
            err_kws={"linewidth": 1.2, "color": EDGE_COLOR},
            legend=False,
            ax=ax,
        )
        
    # Override y-axis formatter for raw count scale
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.0f}"))
    ax.set_xlabel("Extra Steps", fontsize=10.5, labelpad=6, fontweight="semibold")
    ax.set_ylabel("Planning Fixations", fontsize=10.5, fontweight="semibold")
    ax.tick_params(axis="both", labelsize=8.5, length=3)
    ax.yaxis.grid(True, color=GRID_COLOR, linewidth=0.7, linestyle="--", alpha=0.8)
    ax.set_axisbelow(True)
    sns.despine(ax=ax)

    # ── Shared legend ─────────────────────────────────────────────────────
    legend_handles = [
        mpatches.Patch(facecolor=PALETTE[s], edgecolor=EDGE_COLOR,
                       linewidth=0.6, alpha=0.88, label=s)
        for s in subjects
    ]
    fig.legend(
        handles=legend_handles,
        title="Subject", title_fontsize=10, fontsize=9.5,
        loc="upper right", bbox_to_anchor=(1.0, 1.01),
        frameon=True, framealpha=0.9, edgecolor="#cccccc",
    )

    # ── Save ──────────────────────────────────────────────────────────────
    out_path = f"{PROJECT_DIR}/figures_new/fig1A.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"Saved → {out_path}")
    plt.show()

# Fig1A
#df = pd.read_pickle(f'{PROJECT_DIR}/behaviordata_v4.pkl')
#df['subject'].replace({"bart": "B", "london": "L"}, inplace=True)
#print(df['subject'].value_counts())

#plot_steps_planning(df)

# Fig1B 