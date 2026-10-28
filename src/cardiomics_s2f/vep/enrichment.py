"""
VEP Score Enrichment in CHD Probands
====================================

Case/control enrichment of VEP scores on de novo variants (Richter et al. 2020):
for which scores, cutoffs and annotation strata are high-scoring variants
enriched in CHD probands (``is_CHD``) compared with controls?

``is_CHD`` is a patient-level label (every variant of a proband shares it), not a
per-variant ground truth. All cross-validation is therefore grouped by patient
(``StratifiedGroupKFold``): splitting the variants of one patient across training
and test folds would leak the label.

* Variant level: `youden_and_f1_cutoffs`, `fisher_odds_ratio_ci`,
  `grouped_cv_cutoff_odds_ratio` (cutoff fit per fold, pooled out-of-fold odds
  ratio), `evaluate_stratum_odds_ratio`, `build_stratum_grid`,
  `systematic_stratum_search` (all strata x scores, Benjamini-Hochberg FDR) and
  the forest plot `plot_top_strata_forest`.
* Patient level (one data point per proband): `patient_level_enrichment`
  (carriers of >= 1 variant in a stratum), `score_to_patient_carriers`,
  `patient_level_score_enrichment` and `systematic_patient_score_search`
  (carriers of >= 1 variant above a cross-validated score cutoff).

The result tables of both levels share their columns (``stratum``, ``score``,
``cutoff_method``, ``OR``, ``p_value``, ``ci_low``, ``ci_high``, ``fdr_q``,
``significant``, ``stratum_n``, ``n_pos``) and can all be plotted with
`plot_top_strata_forest`.
"""

from itertools import combinations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import fisher_exact
from sklearn.metrics import precision_recall_curve, roc_curve
from sklearn.model_selection import StratifiedGroupKFold


def _woolf_ci(a, b, c, d):
    """Woolf 95% confidence interval of the odds ratio of the 2x2 table [[a, b], [c, d]].

    Adds 0.5 to every cell if any cell is zero (Haldane-Anscombe correction).
    Returns ``(ci_low, ci_high)``.
    """
    a2, b2, c2, d2 = (x + 0.5 if 0 in (a, b, c, d) else x for x in (a, b, c, d))
    log_or = np.log((a2 * d2) / (b2 * c2))
    se = np.sqrt(1 / a2 + 1 / b2 + 1 / c2 + 1 / d2)
    z = 1.959963984540054  # ~95% two-sided
    ci_low = float(np.exp(log_or - z * se))
    ci_high = float(np.exp(log_or + z * se))
    return ci_low, ci_high


def _add_bh_fdr(results, fdr_alpha):
    """Sorts ``results`` by ``p_value`` and adds Benjamini-Hochberg ``fdr_q`` and ``significant``."""
    results = results.sort_values("p_value").reset_index(drop=True)
    m = len(results)
    rank = np.arange(1, m + 1)
    fdr_q = results["p_value"].to_numpy() * m / rank
    # enforce monotonicity (standard BH step-up)
    fdr_q = np.minimum.accumulate(fdr_q[::-1])[::-1]
    results["fdr_q"] = np.clip(fdr_q, 0, 1)
    results["significant"] = results["fdr_q"] < fdr_alpha

    return results


# ---------------------------------------------------------------------------
# 1. Cutoff selection
# ---------------------------------------------------------------------------

def youden_and_f1_cutoffs(y_true, y_score):
    """
    Compute the Youden's J-optimal and F1-optimal cutoffs for a single score.

    y_true : array-like of 0/1 (is_CHD)
    y_score : array-like, the numeric score column

    Returns dict with youden_cutoff, youden_j, f1_cutoff, f1_score.
    """
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score, dtype=float)

    fpr, tpr, roc_thresh = roc_curve(y_true, y_score)
    j = tpr - fpr
    youden_idx = np.argmax(j)

    precision, recall, pr_thresh = precision_recall_curve(y_true, y_score)
    # precision/recall have one more element than pr_thresh; drop the last
    f1 = 2 * precision[:-1] * recall[:-1] / (precision[:-1] + recall[:-1] + 1e-12)
    f1_idx = np.argmax(f1)

    return {
        "youden_cutoff": float(roc_thresh[youden_idx]),
        "youden_j": float(j[youden_idx]),
        "f1_cutoff": float(pr_thresh[f1_idx]),
        "f1_score": float(f1[f1_idx]),
    }


# ---------------------------------------------------------------------------
# 2. Odds ratio at a given cutoff
# ---------------------------------------------------------------------------

def fisher_odds_ratio_ci(y_true, y_pred_binary, alpha=0.05):
    """
    Fisher's exact test + Woolf log-OR confidence interval for a 2x2 table
    of (score high/low) x (is_CHD true/false).

    Returns dict with OR, p_value, ci_low, ci_high, table.
    """
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred_binary).astype(int)

    a = int(np.sum((y_pred == 1) & (y_true == 1)))  # score high, CHD
    b = int(np.sum((y_pred == 1) & (y_true == 0)))  # score high, control
    c = int(np.sum((y_pred == 0) & (y_true == 1)))  # score low, CHD
    d = int(np.sum((y_pred == 0) & (y_true == 0)))  # score low, control

    table = [[a, b], [c, d]]
    or_val, p_val = fisher_exact(table)

    ci_low, ci_high = _woolf_ci(a, b, c, d)

    return {"OR": float(or_val), "p_value": float(p_val),
            "ci_low": ci_low, "ci_high": ci_high, "table": table}


# ---------------------------------------------------------------------------
# 3. Grouped k-fold cutoff + pooled odds ratio for one score, one subset
# ---------------------------------------------------------------------------

def grouped_cv_cutoff_odds_ratio(df, score_col, label_col="is_CHD", group_col="patient_id",
                              n_splits=5, cutoff_method="youden", random_state=42):
    """
    StratifiedGroupKFold CV: on each fold, the cutoff is learned on the
    training portion only and applied to the held-out portion. Out-of-fold
    binarized predictions are pooled across all folds, then a single
    Fisher's exact test is run on the pooled 2x2 table (more stable than
    averaging per-fold p-values).

    cutoff_method : "youden" or "f1"
    """
    sub = df.dropna(subset=[score_col, label_col, group_col])
    X = sub[score_col].to_numpy(dtype=float)
    y = sub[label_col].astype(int).to_numpy()
    groups = sub[group_col].to_numpy()

    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    oof_pred = np.full(len(sub), np.nan)
    fold_cutoffs = []

    for train_idx, test_idx in sgkf.split(X, y, groups):
        cutoffs = youden_and_f1_cutoffs(y[train_idx], X[train_idx])
        cutoff = cutoffs["youden_cutoff"] if cutoff_method == "youden" else cutoffs["f1_cutoff"]
        fold_cutoffs.append(cutoff)
        oof_pred[test_idx] = (X[test_idx] >= cutoff).astype(int)

    or_result = fisher_odds_ratio_ci(y, oof_pred)

    return {
        "cutoff_method": cutoff_method,
        "fold_cutoffs": fold_cutoffs,
        "cutoff_mean": float(np.mean(fold_cutoffs)),
        "cutoff_std": float(np.std(fold_cutoffs)),
        "n": int(len(sub)),
        "n_pos": int(y.sum()),
        **or_result,
    }


# ---------------------------------------------------------------------------
# 4. Single-stratum evaluation with N filtering and safe failure
# ---------------------------------------------------------------------------

def evaluate_stratum_odds_ratio(df, score_col, mask, label_col="is_CHD", group_col="patient_id",
                      min_n=100, n_splits=5, cutoff_method="youden", random_state=42):
    """
    Evaluate one score column within one boolean mask (stratum).
    Returns None (with no exception) if the stratum is too small, has only
    one class, or doesn't have enough distinct patient groups per class for
    the requested number of folds -- callers should just skip None results.
    """
    sub = df.loc[mask].dropna(subset=[score_col, label_col, group_col])

    if len(sub) < min_n:
        return None
    if sub[label_col].nunique() < 2:
        return None

    # need at least n_splits distinct groups within each class
    group_counts = sub.groupby(label_col)[group_col].nunique()
    if (group_counts < n_splits).any():
        return None

    try:
        result = grouped_cv_cutoff_odds_ratio(
            sub, score_col, label_col, group_col, n_splits, cutoff_method, random_state
        )
    except ValueError:
        return None

    result["stratum_n"] = len(sub)
    return result


# ---------------------------------------------------------------------------
# 5. Building combinatorial strata from base categorical conditions
# ---------------------------------------------------------------------------

def build_stratum_grid(base_conditions, max_depth=2, include_base=True):
    """
    base_conditions : dict {name: boolean pd.Series}, e.g. the stratification dicts of the notebooks
        {"cCRE Promoter": ..., "cCRE prox. enhancer": ..., "cardioid_CA": ...}
    max_depth : combine up to this many conditions with AND
    include_base : include the depth-1 (single condition) strata too

    Returns dict {combined_name: combined boolean pd.Series}
    """
    grid = {}
    if include_base:
        grid.update(base_conditions)

    names = list(base_conditions.keys())
    for depth in range(2, max_depth + 1):
        for combo in combinations(names, depth):
            mask = base_conditions[combo[0]]
            for n in combo[1:]:
                mask = mask & base_conditions[n]
            grid[" & ".join(combo)] = mask

    return grid


# ---------------------------------------------------------------------------
# 6. Systematic search over strata x scores, with BH-FDR correction
# ---------------------------------------------------------------------------

def systematic_stratum_search(df, score_cols, stratum_grid, label_col="is_CHD",
                                group_col="patient_id", min_n=100, n_splits=5,
                                cutoff_methods=("youden", "f1"), random_state=42,
                                fdr_alpha=0.05):
    """
    Evaluate every (stratum, score, cutoff_method) combination, skipping
    strata below min_n or with insufficient group counts, then apply
    Benjamini-Hochberg FDR correction across all tests actually run.

    Returns a results DataFrame sorted by p_value, with 'fdr_q' and
    'significant' columns.
    """
    records = []
    for stratum_name, mask in stratum_grid.items():
        for score_col in score_cols:
            for method in cutoff_methods:
                res = evaluate_stratum_odds_ratio(
                    df, score_col, mask, label_col, group_col,
                    min_n, n_splits, method, random_state
                )
                if res is None:
                    continue
                row = {"stratum": stratum_name, "score": score_col}
                row.update({k: v for k, v in res.items() if k not in ("table", "fold_cutoffs")})
                records.append(row)

    results = pd.DataFrame.from_records(records)
    if results.empty:
        return results

    results = _add_bh_fdr(results, fdr_alpha)

    return results

def plot_top_strata_forest(results, top_n=15, rank_by="ci_low", min_ci_low=None,
                     cutoff_method=None, figsize=None, save_path=None):
    """
    Forest plot (OR + CI) of the top results from systematic_stratum_search.

    results : DataFrame returned by systematic_stratum_search
    top_n : how many rows to show
    rank_by : "ci_low" (recommended -- favors stable, well-powered hits over
        flashy small-N ORs), or "OR", "p_value", "fdr_q"
    min_ci_low : optionally drop rows with ci_low below this value before
        ranking (e.g. 1.0 to only show strata where the interval clears
        no-effect)
    cutoff_method : optionally restrict to "youden" or "f1" only, to avoid
        showing both cutoff variants of the same stratum/score
    save_path : if given, also saves the figure to this path

    Rows whose OR lies outside its CI are skipped (with a printed count): with
    an empty cell in the 2x2 table, Fisher's OR is 0 or inf while the Woolf CI
    is Haldane-Anscombe corrected, so no whisker can be drawn. The results
    table itself is not changed.

    Returns (fig, ax).
    """
    df = results.copy()

    outside_ci = (df["OR"] < df["ci_low"]) | (df["OR"] > df["ci_high"])
    if outside_ci.any():
        print(f"Skipped {int(outside_ci.sum())} row(s) with OR outside its CI (OR 0 or inf, empty 2x2 cell).")
        df = df[~outside_ci]

    if cutoff_method is not None:
        df = df[df["cutoff_method"] == cutoff_method]
    if min_ci_low is not None:
        df = df[df["ci_low"] >= min_ci_low]

    ascending = rank_by in ("p_value", "fdr_q")
    df = df.sort_values(rank_by, ascending=ascending).head(top_n)
    df = df.iloc[::-1]  # so the top-ranked row plots at the top

    labels = [f"{s}\n({sc}, {m})" for s, sc, m in
              zip(df["stratum"], df["score"], df["cutoff_method"])]
    y = np.arange(len(df))
    or_vals = df["OR"].to_numpy()
    err_low = or_vals - df["ci_low"].to_numpy()
    err_high = df["ci_high"].to_numpy() - or_vals

    if figsize is None:
        figsize = (9, max(3, 0.4 * len(df) + 1))
    fig, ax = plt.subplots(figsize=figsize)

    # whiskers plotted plain first, dots (colored/sized by significance) on top
    for yi, orv, lo, hi in zip(y, or_vals, err_low, err_high):
        ax.errorbar(orv, yi, xerr=[[lo], [hi]], fmt="none", ecolor="#999999",
                    capsize=3, zorder=1)

    neglog_fdr = -np.log10(np.clip(df["fdr_q"].to_numpy(dtype=float), 1e-300, 1.0))
    max_neglog = max(neglog_fdr.max(), 1e-9)
    sizes = 40 + 360 * (neglog_fdr / max_neglog)


    sc = ax.scatter(or_vals, y, c=neglog_fdr, s=sizes, cmap="Reds",
                     edgecolors="black", linewidths=0.5, zorder=2, vmin=0)
    cbar = fig.colorbar(sc, ax=ax, pad=0.25, shrink=0.6, )
    cbar.set_label("-log10(FDR)")

    ax.axvline(1.0, color="black", linestyle="--", linewidth=1)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("Odds ratio (95% CI)")
    ax.set_title(f"Top {len(df)} strata x score combinations (ranked by {rank_by})")

    x_right = ax.get_xlim()[1]
    for yi, (n, n_pos) in enumerate(zip(df["stratum_n"], df["n_pos"])):
        ax.annotate(f"N={n}, n_pos={n_pos}", xy=(x_right, yi),
                    xytext=(6, 0), textcoords="offset points",
                    va="center", fontsize=7, color="dimgray")

    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig, ax


# ---------------------------------------------------------------------------
# 9. Patient-level (proband) carrier burden enrichment
# ---------------------------------------------------------------------------

def patient_level_enrichment(df, strat_dict, label_col="is_CHD", group_col="Blinded ID",
                              min_carriers=5, alternative="greater", fdr_alpha=0.05):
    """
    For each stratification, tests whether patients carrying >=1 qualifying
    variant (i.e. >=1 variant matching that stratum's mask) are enriched
    among CHD probands vs. controls. One data point per patient (not per
    variant), so this sidesteps the non-independence issue of testing
    variant-level scores directly when a patient can carry several
    qualifying variants.

    df : the variant-level DataFrame (one row per variant)
    strat_dict : dict {name: boolean mask aligned to df.index}, same shape
        as used with build_stratum_grid. A value of df.index
        is accepted and treated as all-True.
    label_col : patient-level case/control column (boolean)
    group_col : patient ID column
    min_carriers : minimum total carriers (CHD + control combined) required
        to report a stratum -- strata below this are skipped, since Fisher's
        exact on a handful of carriers is unstable regardless of the OR
    alternative : passed to scipy.stats.fisher_exact ("greater" tests
        enrichment specifically, not depletion)

    Returns a results DataFrame with the same column set as
    systematic_stratum_search's output (stratum, score, cutoff_method, OR,
    p_value, ci_low, ci_high, fdr_q, significant, stratum_n, n_pos), so it
    can be passed directly to plot_top_strata_forest. "score" is fixed to
    "carrier burden" and "cutoff_method" to "patient-level" as placeholders
    to keep that compatibility -- there's no score cutoff being fit here.
    """
    label = df[label_col].astype(bool)
    total_chd = df.loc[label, group_col].nunique()
    total_control = df.loc[~label, group_col].nunique()

    records = []
    for name, mask in strat_dict.items():
        if isinstance(mask, pd.Index):
            m = pd.Series(True, index=df.index)
        else:
            m = mask.reindex(df.index).fillna(False).astype(bool)
        sub = df.loc[m]
        sub_label = sub[label_col].astype(bool)

        carrier_chd = sub.loc[sub_label, group_col].nunique()
        carrier_control = sub.loc[~sub_label, group_col].nunique()

        if (carrier_chd + carrier_control) < min_carriers:
            continue

        non_carrier_chd = total_chd - carrier_chd
        non_carrier_control = total_control - carrier_control

        table = [[carrier_chd, carrier_control], [non_carrier_chd, non_carrier_control]]
        or_val, p_val = fisher_exact(table, alternative=alternative)

        ci_low, ci_high = _woolf_ci(carrier_chd, carrier_control, non_carrier_chd, non_carrier_control)

        records.append({
            "stratum": name,
            "score": "carrier burden",
            "cutoff_method": "patient-level",
            "OR": float(or_val),
            "p_value": float(p_val),
            "ci_low": ci_low,
            "ci_high": ci_high,
            "carrier_chd": int(carrier_chd),
            "carrier_control": int(carrier_control),
            "total_chd": int(total_chd),
            "total_control": int(total_control),
            "n_pos": int(carrier_chd),
            "stratum_n": int(carrier_chd + carrier_control),
        })

    results = pd.DataFrame.from_records(records)
    if results.empty:
        return results

    results = _add_bh_fdr(results, fdr_alpha)

    return results


# ---------------------------------------------------------------------------
# 10. Patient-level burden test using a data-driven (CV-fit) score cutoff
# ---------------------------------------------------------------------------

def score_to_patient_carriers(df, score_col, label_col="is_CHD", group_col="patient_id",
                               n_splits=5, cutoff_method="youden", random_state=42):
    """
    Same grouped-CV cutoff fitting as grouped_cv_cutoff_odds_ratio, but instead of
    pooling out-of-fold binarized variants into a variant-level OR, aggregates
    them to patient level: a patient is a "carrier" if >=1 of their (held-out)
    variants was called score-high by that fold's cutoff.

    Returns (patient_carriers, fold_cutoffs):
        patient_carriers : pd.Series indexed by group_col value -> bool,
            only for patients present in df after dropping NaNs
        fold_cutoffs : list of per-fold cutoff values
    """
    sub = df.dropna(subset=[score_col, label_col, group_col])
    X = sub[score_col].to_numpy(dtype=float)
    y = sub[label_col].astype(bool).to_numpy()
    groups = sub[group_col].to_numpy()

    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    oof_pred = np.zeros(len(sub), dtype=bool)
    fold_cutoffs = []

    for train_idx, test_idx in sgkf.split(X, y, groups):
        cutoffs = youden_and_f1_cutoffs(y[train_idx], X[train_idx])
        cutoff = cutoffs["youden_cutoff"] if cutoff_method == "youden" else cutoffs["f1_cutoff"]
        fold_cutoffs.append(cutoff)
        oof_pred[test_idx] = X[test_idx] >= cutoff

    carrier_df = pd.DataFrame({group_col: groups, "carrier": oof_pred})
    patient_carriers = carrier_df.groupby(group_col)["carrier"].any()
    return patient_carriers, fold_cutoffs


def patient_level_score_enrichment(df, score_col, label_col="is_CHD", group_col="patient_id",
                                    mask=None, n_splits=5, cutoff_method="youden",
                                    min_carriers=5, alternative="greater", random_state=42):
    """
    Patient-level version of grouped_cv_cutoff_odds_ratio: fits a Youden/F1 cutoff
    on score_col via grouped CV (optionally restricted to a metadata stratum
    via `mask`), then tests whether patients carrying >=1 score-high variant
    are enriched among CHD probands vs. controls (same logic as
    patient_level_enrichment, but the "carrier" definition comes from a
    data-driven score cutoff instead of a fixed boolean mask).

    mask : optional boolean Series aligned to df.index restricting which
        variants are eligible ("high score AND in a proximal enhancer", say).
        None = use all variants in df.

    Returns a dict of results, or None if the (sub)stratum is too small, has
    only one class, or has too few patient groups per class for n_splits.
    """
    label_full = df[label_col].astype(bool)
    total_chd = df.loc[label_full, group_col].nunique()
    total_control = df.loc[~label_full, group_col].nunique()

    sub = df if mask is None else df.loc[mask.reindex(df.index).fillna(False).astype(bool)]
    sub = sub.dropna(subset=[score_col, label_col, group_col])

    if sub[label_col].nunique() < 2:
        return None
    group_counts = sub.groupby(sub[label_col].astype(bool))[group_col].nunique()
    if (group_counts < n_splits).any():
        return None

    try:
        patient_carriers, fold_cutoffs = score_to_patient_carriers(
            sub, score_col, label_col, group_col, n_splits, cutoff_method, random_state
        )
    except ValueError:
        return None

    patient_label = df.groupby(group_col)[label_col].agg(lambda s: bool(s.astype(bool).iloc[0]))
    carrier_status = patient_carriers.reindex(patient_label.index).fillna(False)

    carrier_chd = int((carrier_status & patient_label).sum())
    carrier_control = int((carrier_status & ~patient_label).sum())

    if (carrier_chd + carrier_control) < min_carriers:
        return None

    non_carrier_chd = total_chd - carrier_chd
    non_carrier_control = total_control - carrier_control
    table = [[carrier_chd, carrier_control], [non_carrier_chd, non_carrier_control]]
    or_val, p_val = fisher_exact(table, alternative=alternative)

    ci_low, ci_high = _woolf_ci(carrier_chd, carrier_control, non_carrier_chd, non_carrier_control)

    return {
        "score": score_col,
        "cutoff_method": cutoff_method,
        "cutoff_mean": float(np.mean(fold_cutoffs)),
        "cutoff_std": float(np.std(fold_cutoffs)),
        "OR": float(or_val),
        "p_value": float(p_val),
        "ci_low": ci_low,
        "ci_high": ci_high,
        "carrier_chd": carrier_chd,
        "carrier_control": carrier_control,
        "total_chd": int(total_chd),
        "total_control": int(total_control),
        "n_pos": carrier_chd,
        "stratum_n": carrier_chd + carrier_control,
    }


def systematic_patient_score_search(df, score_cols, stratum_grid=None, label_col="is_CHD",
                                     group_col="patient_id", n_splits=5,
                                     cutoff_methods=("youden", "f1"), min_carriers=5,
                                     alternative="greater", random_state=42, fdr_alpha=0.05):
    """
    Patient-level analog of systematic_stratum_search: for every combination
    of metadata stratum (optional) x score column x cutoff method, fits a
    CV cutoff restricted to that stratum's variants, aggregates to patient
    carrier status, and tests CHD-proband enrichment -- recapitulating the
    variant-level systematic search, but at the patient burden level.

    stratum_grid : dict of {name: boolean mask}, e.g. from build_stratum_grid.
        None runs on the whole dataset as a single "All Vars" stratum.

    Returns a results DataFrame with the same columns as
    patient_level_enrichment's batch output (stratum, score, cutoff_method,
    OR, p_value, ci_low, ci_high, fdr_q, significant, stratum_n, n_pos) --
    directly compatible with plot_top_strata_forest.
    """
    if stratum_grid is None:
        stratum_grid = {"All Vars": pd.Series(True, index=df.index)}

    records = []
    for stratum_name, mask in stratum_grid.items():
        m = (pd.Series(True, index=df.index) if isinstance(mask, pd.Index)
             else mask.reindex(df.index).fillna(False).astype(bool))
        for score_col in score_cols:
            for method in cutoff_methods:
                res = patient_level_score_enrichment(
                    df, score_col, label_col, group_col, mask=m,
                    n_splits=n_splits, cutoff_method=method,
                    min_carriers=min_carriers, alternative=alternative,
                    random_state=random_state,
                )
                if res is None:
                    continue
                row = {"stratum": stratum_name}
                row.update(res)
                records.append(row)

    results = pd.DataFrame.from_records(records)
    if results.empty:
        return results

    results = _add_bh_fdr(results, fdr_alpha)

    return results
