"""
VEP Score Statistics
====================

* :func:`mannwhitney_power_grid`: bootstrap power analysis of a Mann-Whitney U
  test, i.e. how many variants per group would be needed for a score to separate
  two groups (e.g. variants with and without a measured effect) at a given power.
"""

import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu
from statsmodels.stats.proportion import proportion_confint


def mannwhitney_power_grid(
    df,
    x_col,
    y_cols,
    alternative="two-sided",
    n_range=(10, 20, 30, 50, 75, 100, 150, 200, 300, 500),
    n_sims=1000,
    alpha=0.05,
    target_power=0.8,
    ci_level=0.95,
    n_cols=4,
    random_state=None,
):
    """
    Estimate, via bootstrap simulation, the sample size needed per group to
    reach `target_power` for a Mann-Whitney U test, for each variable in
    y_cols. Resamples (with replacement) from the empirically observed
    values in df for each group, at each candidate n in n_range, and
    reports the empirical proportion of simulations reaching p < alpha,
    along with a Wilson score confidence interval on that proportion.

    Parameters
    ----------
    df : pandas.DataFrame
    x_col : str
        Categorical column with exactly two distinct non-null values.
    y_cols : str or list of str
        Column(s) to run the power analysis on.
    alternative : {'two-sided', 'greater', 'less'}
        Same semantics as scipy's mannwhitneyu and
        `cardiomics_s2f.vep.plotting.box_grid`.
        'greater'/'less' require x_col to be boolean-like
        ({True, False}, {0, 1}, or {-1, 1}).
    n_range : sequence of int
        Candidate per-group sample sizes to evaluate.
    n_sims : int
        Number of bootstrap simulations per (y_col, n) combination.
    alpha : float
        Significance threshold.
    target_power : float
        Power level used to report the minimum required n.
    ci_level : float
        Confidence level for the Wilson interval on each power estimate
        (and hence on the required-n estimate). Default 0.95.
    n_cols : int
        Number of columns in the plot grid.
    random_state : int or None
        Seed for reproducibility.

    Returns
    -------
    results : dict
        {y_col: pandas.DataFrame with columns
         ['n', 'power', 'power_ci_lower', 'power_ci_upper']}
    required_n : dict
        {y_col: {'estimate': int or None,
                  'ci_lower': int or None,
                  'ci_upper': int or None}}
        estimate = smallest n where the point-estimate power >= target_power
        ci_lower = smallest n where the UPPER bound of the power CI already
                   reaches target_power (optimistic: fewest samples that
                   could plausibly suffice)
        ci_upper = smallest n where the LOWER bound of the power CI reaches
                   target_power (conservative: n needed to be confident)
        Any of these is None if not reached within n_range.
    """
    if isinstance(y_cols, str):
        y_cols = [y_cols]

    def _identify_true_false_groups(unique_vals):
        uniq = set(unique_vals)
        if uniq == {True, False}:
            return True, False
        if uniq == {0, 1}:
            return 1, 0
        if uniq == {-1, 1}:
            return 1, -1
        return None, None

    groups = df[x_col].dropna().unique()
    if len(groups) != 2:
        raise ValueError(
            f"x_col '{x_col}' must have exactly 2 distinct values for a "
            f"Mann-Whitney U test; found {len(groups)}: {list(groups)}"
        )

    if alternative == "two-sided":
        group_first, group_second = groups
    else:
        true_val, false_val = _identify_true_false_groups(groups)
        if true_val is None:
            raise ValueError(
                f"alternative='{alternative}' requires x_col '{x_col}' to be "
                "boolean-like ({True, False}, {0, 1}, or {-1, 1}) so the "
                f"test direction is unambiguous; found values {list(groups)}."
            )
        group_first, group_second = true_val, false_val

    rng = np.random.default_rng(random_state)

    results = {}
    required_n = {}

    n_rows = math.ceil(len(y_cols) / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 4, n_rows * 3.5))
    axes = np.array(axes).flatten()

    for ax, y in zip(axes, y_cols):
        sub = df[[x_col, y]].dropna()
        vals_first = sub.loc[sub[x_col] == group_first, y].to_numpy()
        vals_second = sub.loc[sub[x_col] == group_second, y].to_numpy()

        powers, ci_lowers, ci_uppers = [], [], []
        for n in n_range:
            if len(vals_first) == 0 or len(vals_second) == 0:
                powers.append(np.nan)
                ci_lowers.append(np.nan)
                ci_uppers.append(np.nan)
                continue
            sig_count = 0
            valid_sims = 0
            for _ in range(n_sims):
                s1 = rng.choice(vals_first, size=n, replace=True)
                s2 = rng.choice(vals_second, size=n, replace=True)
                try:
                    _, p = mannwhitneyu(s1, s2, alternative=alternative)
                except ValueError:
                    continue
                valid_sims += 1
                if p < alpha:
                    sig_count += 1

            if valid_sims == 0:
                powers.append(np.nan)
                ci_lowers.append(np.nan)
                ci_uppers.append(np.nan)
                continue

            power_hat = sig_count / valid_sims
            lo, hi = proportion_confint(
                sig_count, valid_sims, alpha=1 - ci_level, method="wilson"
            )
            powers.append(power_hat)
            ci_lowers.append(lo)
            ci_uppers.append(hi)

        power_df = pd.DataFrame({
            "n": n_range,
            "power": powers,
            "power_ci_lower": ci_lowers,
            "power_ci_upper": ci_uppers,
        })
        results[y] = power_df

        def _first_crossing(series):
            reached = power_df.loc[series >= target_power, "n"]
            return int(reached.min()) if len(reached) > 0 else None

        est = _first_crossing(power_df["power"])
        ci_lo = _first_crossing(power_df["power_ci_upper"])  # optimistic
        ci_hi = _first_crossing(power_df["power_ci_lower"])  # conservative
        required_n[y] = {"estimate": est, "ci_lower": ci_lo, "ci_upper": ci_hi}

        yerr_lower = np.clip(power_df["power"] - power_df["power_ci_lower"], 0, None)
        yerr_upper = np.clip(power_df["power_ci_upper"] - power_df["power"], 0, None)
        yerr = np.array([yerr_lower, yerr_upper])
        ax.errorbar(
            power_df["n"], power_df["power"],
            yerr=yerr, marker="o", capsize=3, linewidth=1,
        )
        ax.axhline(target_power, color="red", linestyle="--", linewidth=1)
        ax.set_title(y.replace("_", " "))
        ax.set_xlabel("n per group")
        ax.set_ylabel("power")
        ax.set_ylim(0, 1.05)

        if est is not None:
            lo_txt = ci_lo if ci_lo is not None else "?"
            hi_txt = ci_hi if ci_hi is not None else ">max"
            label = f"n≈{est} ({lo_txt}\u2013{hi_txt})"
        else:
            label = "not reached"
        ax.text(
            0.5, 0.05, label,
            transform=ax.transAxes,
            ha="center", va="bottom",
            fontsize=9, fontweight="bold",
        )

    for j in range(len(y_cols), len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    plt.show()

    return results, required_n
