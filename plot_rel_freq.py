"""
Relative Frequency Plots for Primate Maze Behavioral Data
----------------------------------------------------------
Plots the relative frequency (proportion) of each discrete feature
in a single row of n subplots, with hue by subject (two primates).

Usage:
    Configure FEATURES and HUE_COL, supply your real df, then call
    plot_relative_freq(df, FEATURES).
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.patches as mpatches
import seaborn as sns

# ── Environment variables ──────────────────────────────────────────────────
# Dict of  column_name -> display_label  for every feature to be plotted.
FEATURES = {
    "shortest_paths":          "# Shortest\nPaths",
    "shortest_path_length":   "Shortest Path\nLength",
    "steps_teleporter_saved": "Steps Saved\nby Teleporter",
    "plannings":              "Planning\nFixations",
    "steps_extra":            "Extra\nSteps",
    "teleporter_used":        "Teleporter\nUsed",
}

HUE_COL    = "subject"
EDGE_COLOR = "#1a1a2e"
GRID_COLOR = "#e0e0e0"
PALETTE    = {'B': "#3A86FF", 'L': "#FF6B6B"}


# ── Helper ─────────────────────────────────────────────────────────────────
def _subject_rel_freq(df: pd.DataFrame, col: str, hue_col: str) -> pd.DataFrame:
    """
    Compute within-subject relative frequency for a discrete column.
    Returns a long-form DataFrame with columns [col, hue_col, 'proportion'].
    Each subject's proportions sum to 1 independently.
    """
    all_vals = sorted(df[col].dropna().unique())
    records  = []
    for subj, grp in df.groupby(hue_col):
        counts = grp[col].value_counts(normalize=True).reindex(all_vals, fill_value=0)
        for val, prop in counts.items():
            records.append({col: val, hue_col: subj, "proportion": prop})
    return pd.DataFrame(records)


# ── Main plotting method ───────────────────────────────────────────────────
def plot_relative_freq(df: pd.DataFrame, list_features: list) -> None:
    """
    Plot relative frequency distributions for each feature in `features`,
    split by subject (hue).

    Parameters
    ----------
    df       : DataFrame containing the data; must include a `HUE_COL` column.
    features : list of column names for features to plot.
    """
    features = {col: FEATURES[col] for col in list_features}
    df['subject'] = df['subject'].apply(lambda x: x[0].upper())
    subjects = sorted(df[HUE_COL].unique())
    n        = len(features)

    sns.set_theme(style="ticks", context="paper", font_scale=1.2)

    fig, axes = plt.subplots(
        nrows=1, ncols=n,
        figsize=(3.5 * n, 5.0),
        constrained_layout=True,
    )
    # Ensure axes is always iterable even for n == 1
    if n == 1:
        axes = [axes]

    #fig.suptitle(
    #    "Relative Frequency of Behavioral Features by Subject",
    #    fontsize=20, fontweight="bold", y=1.1,
    #)

    for ax, (col, label) in zip(axes, features.items()):
        plot_df     = _subject_rel_freq(df, col, HUE_COL)
        unique_vals = sorted(df[col].dropna().unique())

        sns.barplot(
            data=plot_df,
            x=col,
            y="proportion",
            hue=HUE_COL,
            palette=PALETTE,
            edgecolor=EDGE_COLOR,
            linewidth=0.6,
            alpha=0.88,
            legend=False,
            ax=ax,
        )

        # Rotate x-tick labels when there are many unique values
        if len(unique_vals) > 6:
            ax.set_xticks(range(len(unique_vals)))
            ax.set_xticklabels(unique_vals, rotation=45, ha="right", fontsize=8)

        # ── Axis formatting ───────────────────────────────────────────────
        ax.set_xlabel(label, fontsize=11, labelpad=6, fontweight="semibold")
        ax.set_ylabel("Relative Frequency" if ax is axes[0] else "", fontsize=9.5)

        # Real numbers 0–1, rounded to 2 decimal places
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.2f}"))
        ax.tick_params(axis="both", labelsize=8.5, length=3)

        ax.yaxis.grid(True, color=GRID_COLOR, linewidth=0.7, linestyle="--", alpha=0.8)
        ax.set_axisbelow(True)
        sns.despine(ax=ax)

    # ── Shared figure legend ──────────────────────────────────────────────
    legend_handles = [
        mpatches.Patch(facecolor=PALETTE[s], edgecolor=EDGE_COLOR,
                       linewidth=0.6, alpha=0.88, label=s)
        for s in subjects
    ]
    fig.legend(
        handles=legend_handles,
        title="Subject",
        title_fontsize=10,
        fontsize=9.5,
        loc="upper right",
        bbox_to_anchor=(1.0, 1.01),
        frameon=True,
        framealpha=0.9,
        edgecolor="#cccccc",
    )

    fig.savefig(
        "relative_freq_features.png",
        dpi=180,
        bbox_inches="tight",
        facecolor="white",
    )
    print("Saved → relative_freq_features.png")
    plt.show()


# ── Entry point ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Synthetic data — replace with your real DataFrame
    df = pd.read_pickle('behaviordata_v2.pkl')
    #features = ['shortest_paths', 'shortest_path_length', 'steps_teleporter_saved', 'teleporter_used']
    features = ['steps_extra', 'plannings']

    #plot_relative_freq(df, features)
    
    fig, axs = plt.subplots(1, 1, figsize = (5, 5))
    df['steps_extra'] = df['steps_extra'].clip(upper=8)
    df['subject'] = df['subject'].apply(lambda x: x[0].upper())
    sns.barplot(data = df, 
                x = 'steps_extra', 
                y = 'plannings', 
                hue = HUE_COL,
                palette=PALETTE,
                edgecolor=EDGE_COLOR,
                linewidth=0.6,
                alpha=0.88,
                legend=True,
                ax=axs)
    handles, labels = axs.get_legend_handles_labels()
    axs.legend(handles, ['B (r = 0.28, p < 0.0001)', 'L (r = 0.39, p < 0.0001)'], title="Subject", fontsize=9.5, title_fontsize=10)
    axs.set_xticklabels(list(map(str, range(8))) + ['\u22658'])
    plt.show()