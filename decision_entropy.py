import pandas as pd
import numpy as np
from scipy.stats import entropy as scipy_entropy
from collections import Counter
from figures_new.config import PROJECT_DIR


# ── helpers ──────────────────────────────────────────────────────────────────

def nodes_to_list(node):
    """
    Convert the `nodes` value to a hashable tuple.
    """
    return tuple(int(x) for x in node.split(','))                   # unrecognised → will be dropped

def shannon_entropy(counts, base=2):
    """Shannon entropy H from a Counter or list of counts."""
    vals = np.array(list(counts.values()), dtype=float)
    probs = vals / vals.sum()
    return scipy_entropy(probs, base=base)


def max_entropy(n_paths, base=2):
    """Upper-bound entropy when all n_paths are equally likely."""
    if n_paths <= 1:
        return 0.0
    return np.log(n_paths) / np.log(base)


# ── main function ─────────────────────────────────────────────────────────────

def compute_decision_entropy(
    df_path: str = f"{PROJECT_DIR}/behaviordata_v4.pkl",
    optimal_only: bool = True,
    min_trials: int = 5,
) -> pd.DataFrame:
    """
    Compute per-(start, target) path-choice entropy.

    Parameters
    ----------
    df : DataFrame with columns
        start, target, nodes, shortest_paths, Shortest_path_length
    optimal_only : bool
        If True, restrict analysis to trials where the agent took an
        optimal path (path length == Shortest_path_length).
    min_trials : int
        Drop (start, target) pairs with fewer than this many qualifying
        trials — too few observations make entropy unreliable.

    Returns
    -------
    summary : DataFrame, one row per (start, target), with columns
        n_trials          – qualifying trials counted
        n_unique_paths    – distinct paths observed
        n_shortest_paths  – theoretical # of optimal paths (from data)
        entropy_bits      – Shannon entropy of path-choice distribution
        max_entropy_bits  – log2(n_shortest_paths), the theoretical max
        normalised_entropy – entropy / max_entropy  (0 = fully biased,
                             1 = perfectly uniform across optimal paths)
        path_distribution – dict mapping path → probability
    """

    df = pd.read_pickle(df_path).query("steps_extra == 0")  # focus on trials with no extra steps
    df["nodes"] = df["nodes"].apply(nodes_to_list)
    df = df.dropna(subset=["nodes"])

    records = []

    for (start, target), grp in df.groupby(["start", "target"]):
        if len(grp) < min_trials:
            continue

        path_counts = Counter(grp["nodes"])
        total       = sum(path_counts.values())
        n_sp        = int(grp["shortest_paths"].mode()[0])   # theoretical # optimal paths
        n_unique    = len(path_counts)

        H     = shannon_entropy(path_counts)
        H_max = max_entropy(n_sp)                            # based on graph structure
        H_norm = (H / H_max) if H_max > 0 else np.nan

        path_dist = {str(list(p)): round(c / total, 4)
                     for p, c in path_counts.most_common()}

        records.append(dict(
            start              = start,
            target             = target,
            n_trials           = total,
            n_unique_paths     = n_unique,
            n_shortest_paths   = n_sp,
            entropy_bits       = round(H, 4),
            max_entropy_bits   = round(H_max, 4),
            normalised_entropy = round(H_norm, 4) if not np.isnan(H_norm) else np.nan,
            path_distribution  = path_dist,
        ))

    summary = pd.DataFrame(records).sort_values("entropy_bits", ascending=True)
    return summary


# ── aggregate view ────────────────────────────────────────────────────────────

def aggregate_entropy_summary(summary: pd.DataFrame) -> dict:
    """
    Print and return global statistics across all (start, target) pairs.
    """
    stats = {
        "mean_entropy_bits"       : summary["entropy_bits"].mean(),
        "mean_normalised_entropy" : summary["normalised_entropy"].mean(),
        "median_normalised_entropy": summary["normalised_entropy"].median(),
        "most_biased_pairs"       : summary.nsmallest(5, "normalised_entropy")[
                                        ["start","target","n_trials",
                                         "n_shortest_paths","normalised_entropy"]
                                    ].to_dict("records"),
        "most_uniform_pairs"      : summary.nlargest(5, "normalised_entropy")[
                                        ["start","target","n_trials",
                                         "n_shortest_paths","normalised_entropy"]
                                    ].to_dict("records"),
    }

    print("=" * 60)
    print("DECISION ENTROPY SUMMARY")
    print("=" * 60)
    print(f"  Start-target pairs analysed : {len(summary)}")
    print(f"  Mean entropy (bits)         : {stats['mean_entropy_bits']:.3f}")
    print(f"  Mean normalised entropy     : {stats['mean_normalised_entropy']:.3f}")
    print(f"  Median normalised entropy   : {stats['median_normalised_entropy']:.3f}")
    print("\n  Most BIASED pairs (normalised entropy ↓):")
    for r in stats["most_biased_pairs"]:
        print(f"    {r['start']:>2} → {r['target']:<2}  "
              f"n={r['n_trials']:>4}  "
              f"n_opt={r['n_shortest_paths']}  "
              f"H_norm={r['normalised_entropy']:.3f}")
    print("\n  Most UNIFORM pairs (normalised entropy ↑):")
    for r in stats["most_uniform_pairs"]:
        print(f"    {r['start']:>2} → {r['target']:<2}  "
              f"n={r['n_trials']:>4}  "
              f"n_opt={r['n_shortest_paths']}  "
              f"H_norm={r['normalised_entropy']:.3f}")
    print("=" * 60)

    return stats


# ── usage example ─────────────────────────────────────────────────────────────

if __name__ == "__main__":

    # ── replace this block with your actual dataframe ──
    # df = pd.read_csv("your_data.csv")

    # minimal synthetic example ------------------------------------------------
    data = {
        "start":               [0, 0, 0, 0, 0,  3, 3, 3, 3, 3,  3, 3],
        "target":              [15,15,15,15,15, 12,12,12,12,12, 12,12],
        "nodes":               [
            [0,1,2,3,7,11,15], [0,1,2,3,7,11,15], [0,1,2,3,7,11,15],
            [0,4,8,12,13,14,15],[0,4,8,12,13,14,15],
            [3,2,1,0,4,8,12],  [3,2,1,0,4,8,12],  [3,2,1,0,4,8,12],
            [3,7,11,10,9,8,12],[3,7,11,10,9,8,12],
            [3,2,1,5,9,8,12],  [3,2,1,0,4,8,12],
        ],
        "shortest_paths":      [2, 2, 2, 2, 2,  3, 3, 3, 3, 3,  3, 3],
        "Shortest_path_length":[6, 6, 6, 6, 6,  6, 6, 6, 6, 6,  6, 6],
    }
    df = pd.DataFrame(data)
    # --------------------------------------------------------------------------

    # compute entropy
    summary = compute_decision_entropy(df, optimal_only=True, min_trials=3)

    print("\nPer-pair entropy table:")
    print(summary.drop(columns="path_distribution").to_string(index=False))

    print()
    stats = aggregate_entropy_summary(summary)

    print("\nPath distributions:")
    for _, row in summary.iterrows():
        print(f"\n  {int(row.start)} → {int(row.target)}")
        for path, prob in row["path_distribution"].items():
            print(f"    {path:40s}  p = {prob:.3f}")