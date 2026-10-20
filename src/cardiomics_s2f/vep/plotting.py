"""Variant Effect Prediction Plots
===============================

Figures to benchmark VEP scores against observed variant effects:

* Classification of effect variants: `plot_AUPRCs`, `plot_AUROCs`,
  `plot_auc_bars`, `plot_contingency_grid`, `plot_sign_confusion_matrix`.
* Score distributions and effect sizes: `box_grid` (with Mann-Whitney U tests),
  `plot_scatter_panel` (with Spearman correlations).
* Stratified performance: `plot_stratified_auprc_heatmap` (strata e.g. from
  `cardiomics_s2f.vep.processing.create_tss_stratifications`).

All functions show their figure with ``plt.show()``, except
`plot_contingency_grid`, which returns it.
"""
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
    confusion_matrix
)
from scipy.stats import chi2_contingency, mannwhitneyu, spearmanr


def plot_AUPRCs(df, true_label, score_columns, n_cols=6):
    """Plots a grid of precision-recall curves, one panel per score column.

    Panels are ordered by AUPRC (highest first). Each panel shows the curve, the
    baseline (positive rate) and the cutoff that maximises the F1 score. Rows with
    a missing label or score are excluded per column; columns without both classes
    get an AUPRC of 0 and an "Insufficient Data" panel.

    Args:
        df (pd.DataFrame): Table with the labels and scores.
        true_label (str): Column with the binary ground truth (0/1 or bool).
        score_columns (list of str): Columns with the model scores.
        n_cols (int, optional): Panels per row. Defaults to 6.
    """
    
    # 1. Pre-calculate AUPRCs to rank them
    auprc_data = []
    for col in score_columns:
        # Create a mask to drop rows where either the true label or the score is NaN
        mask = df[true_label].notna() & df[col].notna()
        y_true = df.loc[mask, true_label]
        y_score = df.loc[mask, col]
        
        # Guard against columns that are completely empty or have only one class left
        if len(y_true) == 0 or len(y_true.unique()) < 2:
            auprc = 0.0  # Assign a 0 score so it sorts to the bottom
        else:
            auprc = average_precision_score(y_true, y_score)
            
        auprc_data.append((col, auprc))
        
    # Sort columns by AUPRC in descending order (highest to lowest)
    auprc_data.sort(key=lambda x: x[1], reverse=True)
    sorted_score_columns = [item[0] for item in auprc_data]
    
    # 2. Set up the plot grid
    n_rows = math.ceil(len(sorted_score_columns) / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols*4, n_rows*4))
    
    # Flatten axes for easy iteration (handles 1D or 2D arrays)
    # If there's only 1 plot, axes might not be an array, so we ensure it is
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])
    axes = axes.flatten()
        
    for i, score_col in enumerate(sorted_score_columns):
        ax = axes[i]
        
        # Apply the same mask for plotting
        mask = df[true_label].notna() & df[score_col].notna()
        y_true = df.loc[mask, true_label]
        y_score = df.loc[mask, score_col]
        
        # Calculate a subset-specific baseline
        # (Since dropping NaNs might slightly change the prevalence of the positive class)
        baseline = y_true.mean() if len(y_true) > 0 else 0
        
        # Retrieve the pre-calculated AUPRC
        auprc_value = next(item[1] for item in auprc_data if item[0] == score_col)

        # If data is valid, calculate curve and cutoff
        if len(y_true.unique()) >= 2:
            precision, recall, thresholds = precision_recall_curve(y_true, y_score)
            
            # Calculate optimal cutoff using the F1-score 
            f1_scores = 2 * (precision[:-1] * recall[:-1]) / (precision[:-1] + recall[:-1] + 1e-10)
            optimal_idx = np.argmax(f1_scores)
            optimal_cutoff = thresholds[optimal_idx]

            # Plot PR curve
            ax.plot(recall, precision, label='Model')
        else:
            optimal_cutoff = np.nan
            ax.text(0.5, 0.5, "Insufficient Data", ha='center', va='center', color='gray')
        
        # Plot Baseline as a dashed line
        ax.axhline(y=baseline, color='r', linestyle='--', label=f'Baseline ({baseline:.2f})')
        
        ax.set_title(score_col.replace("_", " "))
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")

        ax.text(
            0.95, 0.95,
            f"AUPRC = {auprc_value:.3f}\nBaseline = {baseline:.3f}\nCutoff = {optimal_cutoff:.3f}",
            transform=ax.transAxes,
            fontsize=11,
            horizontalalignment='right',
            verticalalignment='top'
        )
        
    # Hide extra axes
    for j in range(len(sorted_score_columns), len(axes)):
        axes[j].axis('off')
    
    plt.tight_layout()
    plt.show()
    

def plot_auc_bars(df, true_label, score_columns):
    """Plots AUROC and AUPRC of several models as two bar charts side by side.

    Models are ordered by AUPRC (highest first); dashed lines mark the baselines
    (0.5 for AUROC, the positive rate for AUPRC).

    Args:
        df (pd.DataFrame): Table with the labels and scores (no missing values in
            ``true_label`` or ``score_columns``).
        true_label (str): Column with the binary ground truth (0/1 or bool).
        score_columns (list of str): Columns with the model scores.
    """
    # 1. Calculate Baselines
    baseline_pr = df[true_label].mean()
    baseline_roc = 0.50
    
    # 2. Compute AUROC and AUPRC for each model
    results = []
    for col in score_columns:
        auroc = roc_auc_score(df[true_label], df[col])
        auprc = average_precision_score(df[true_label], df[col])
        results.append({'model': col, 'auroc': auroc, 'auprc': auprc})
        
    # 3. Sort models by AUPRC (descending) to establish a consistent x-axis order
    results.sort(key=lambda x: x['auprc'], reverse=True)
    
    models = [r['model'].replace("_", " ") for r in results]
    aurocs = [r['auroc'] for r in results]
    auprcs = [r['auprc'] for r in results]
    
    # 4. Set up the 1x2 plot grid
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # --- LEFT PLOT: AUROC ---
    ax1 = axes[0]
    bars1 = ax1.bar(models, aurocs, color='skyblue', edgecolor='black')
    ax1.axhline(y=baseline_roc, color='blue', linestyle='--', alpha=0.6, label='Baseline (0.50)')
    
    ax1.set_title('AUROC by Classifier')
    ax1.set_ylabel('AUROC Score')
    ax1.set_ylim(0, 1.1)  # Leave room for text labels
    ax1.set_xticks(range(len(models)))
    ax1.set_xticklabels(models, rotation=45, ha='right', rotation_mode='anchor')
    ax1.bar_label(bars1, fmt='%.3f', padding=3, fontsize=10)
    ax1.legend(loc='lower right')
    
    # --- RIGHT PLOT: AUPRC ---
    ax2 = axes[1]
    bars2 = ax2.bar(models, auprcs, color='lightcoral', edgecolor='black')
    ax2.axhline(y=baseline_pr, color='red', linestyle='--', alpha=0.6, label=f'Baseline ({baseline_pr:.2f})')
    
    ax2.set_title('AUPRC by Classifier')
    ax2.set_ylabel('AUPRC Score')
    ax2.set_ylim(0, 1.1)
    ax2.set_xticks(range(len(models)))
    ax2.set_xticklabels(models, rotation=45, ha='right', rotation_mode='anchor')
    ax2.bar_label(bars2, fmt='%.3f', padding=3, fontsize=10)
    ax2.legend(loc='lower right')
    
    plt.tight_layout()
    plt.show()
    

def plot_sign_confusion_matrix(y_observed, y_predicted, labels=[-1, 1]):
    """Plots the confusion matrix of predicted vs. observed effect signs.

    The title reports the p-value of a chi-square test of independence.

    Args:
        y_observed (array-like): Observed sign per variant (e.g. -1 / 1).
        y_predicted (array-like): Predicted sign per variant.
        labels (list, optional): Classes to include, in display order.
            Defaults to ``[-1, 1]``.
    """
    cm = confusion_matrix(y_observed, y_predicted, labels=labels)
    chi2, p_value, dof, expected = chi2_contingency(cm)
    
    if p_value < 0.001:
        p_str = "p < 0.001"
    else:
        p_str = f"p = {p_value:.3f}"
        
    plt.figure(figsize=(3, 3))
    sns.heatmap(
        cm, 
        annot=True,          
        fmt='d',             
        cmap='Blues',        
        cbar=False,          
        xticklabels=labels, 
        yticklabels=labels
    )
    
    plt.title(f'Predicted vs. Observed RNA Signs\n(Chi-Square {p_str})', pad=15)
    plt.xlabel('Predicted Sign', labelpad=10)
    plt.ylabel('Observed Sign', labelpad=10)

    plt.tight_layout()
    plt.show()


def plot_contingency_grid(
    df,
    true_label,
    score_cols,
    threshold=0.5,
    optimal_threshold=False,
    titles=None,
    ncols=3,
    cmap="Blues",
    figsize_per_cell=3.2,
    fontsize=11,
):
    """Plots a grid of 2x2 contingency tables of binarised scores vs. a binary label.

    Each score column is binarised at a fixed cutoff (``score >= threshold`` is
    positive) or at the cutoff that maximises the F1 score along its
    precision-recall curve; each panel reports the chi-square test p-value.
    Rows with a missing label or score are excluded per column.

    Args:
        df (pd.DataFrame): Table with the label and score columns.
        true_label (str): Column with the observed binary outcome, encoded as
            {True, False}, {0, 1} or {-1, 1}.
        score_cols (str or list of str): Score column(s) to binarise.
        threshold (float or dict, optional): Fixed cutoff for all columns, or a
            ``{column: cutoff}`` mapping (missing columns use 0.5). Ignored for
            columns selected by ``optimal_threshold``. Defaults to 0.5.
        optimal_threshold (bool or list of str, optional): True to use the
            F1-optimal cutoff for all columns, or a list of the columns that should
            use it. Cutoffs that would call every point positive (or negative) are
            skipped. Defaults to False.
        titles (dict, optional): ``{column: panel title}``; defaults to the column name.
        ncols (int, optional): Panels per row. Defaults to 3.
        cmap (str, optional): Colormap for the cell counts. Defaults to ``"Blues"``.
        figsize_per_cell (float, optional): Panel size in inches. Defaults to 3.2.
        fontsize (int, optional): Base font size. Defaults to 11.

    Returns:
        tuple:
            - matplotlib.figure.Figure: The figure.
            - np.ndarray: The panel axes.
            - dict: Per column ``{"threshold", "table", "chi2_pvalue"}``; ``table`` is
              a 2x2 DataFrame (rows: true negative/positive, columns: predicted
              negative/positive); the p-value is NaN for degenerate tables.
    """

    def _normalize_true_label(series):
        """Convert a true-label column (bool, 0/1, or -1/1) to a boolean array."""
        unique_vals = set(pd.unique(series.dropna()))
        if unique_vals <= {True, False}:
            return series.astype(bool)
        if unique_vals <= {0, 1}:
            return series.astype(int).astype(bool)
        if unique_vals <= {-1, 1}:
            return series.astype(int) == 1
        raise ValueError(
            "true_label column must contain only True/False, 0/1, or -1/1 "
            f"values; found {unique_vals}"
        )


    def _find_optimal_threshold(y_true, scores):
        """
        Threshold that maximizes F1 score along the precision-recall curve,
        excluding degenerate cutoffs that would classify every point as
        positive or every point as negative.

        Note: sklearn's precision_recall_curve returns thresholds as the sorted
        unique score values themselves, and the very first one is always the
        minimum score in the data -- using `score >= min(score)` as the cutoff
        trivially predicts positive for everyone. If a column's scores barely
        separate the two classes (or the true label is heavily imbalanced),
        F1-argmax can land exactly on that trivial cutoff. We exclude such
        degenerate cutoffs so the returned threshold always yields a genuine
        2x2 comparison.
        """
        precision, recall, thresholds = precision_recall_curve(y_true, scores)
        # precision/recall have one more element than thresholds -> drop the last
        denom = precision[:-1] + recall[:-1]
        f1 = np.zeros_like(denom)
        np.divide(
            2 * precision[:-1] * recall[:-1], denom,
            out=f1, where=denom > 0,
        )

        n = len(scores)
        sorted_scores = np.sort(scores)
        # number of points with score >= threshold, for each candidate threshold
        counts_pos = n - np.searchsorted(sorted_scores, thresholds, side="left")
        valid = (counts_pos > 0) & (counts_pos < n)

        if valid.any():
            f1 = np.where(valid, f1, -1)
        # if every cutoff is degenerate (e.g. all scores identical), fall back
        # to the unfiltered argmax rather than raising.

        best_idx = np.argmax(f1)
        return thresholds[best_idx]


    def _safe_chi2_pvalue(counts):
        """
        Chi-square test p-value for a 2x2 table, returning NaN instead of
        raising when a row or column sums to zero (e.g. a threshold that
        predicts a single class for everyone, or a true label with only one
        observed class) so the whole grid doesn't fail on one degenerate panel.
        """
        row_sums = counts.sum(axis=1)
        col_sums = counts.sum(axis=0)
        if np.any(row_sums == 0) or np.any(col_sums == 0):
            return np.nan
        try:
            _, p, _, _ = chi2_contingency(counts)
            return p
        except ValueError:
            return np.nan


    def _draw_single_contingency(ax, counts, pvalue, title, cmap, fontsize):
        ax.imshow(counts, cmap=cmap, aspect="equal", vmin=0)

        vmax = counts.max() if counts.max() > 0 else 1
        for (r, c), val in np.ndenumerate(counts):
            color = "white" if val > vmax * 0.6 else "black"
            ax.text(c, r, f"{val}", ha="center", va="center",
                    fontsize=fontsize + 2, color=color, fontweight="bold")

        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["Pred neg", "Pred pos"], fontsize=fontsize - 1)
        ax.set_yticklabels(["True neg", "True pos"], fontsize=fontsize - 1)
        ax.xaxis.set_ticks_position("bottom")
        ax.tick_params(length=0)

        # square 1pt black border
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color("black")
            spine.set_linewidth(1)
        ax.set_xlim(-0.5, 1.5)
        ax.set_ylim(1.5, -0.5)

        if np.isnan(pvalue):
            p_str = "n/a (degenerate table)"
        else:
            p_str = f"{pvalue:.2e}" if pvalue < 0.001 else f"{pvalue:.3f}"
        ax.set_title(f"{title}\nchi2 p = {p_str}", fontsize=fontsize)


    if isinstance(score_cols, str):
        score_cols = [score_cols]
    titles = titles or {}

    y_true = _normalize_true_label(df[true_label])

    if optimal_threshold is True:
        optimal_cols = set(score_cols)
    elif optimal_threshold is False:
        optimal_cols = set()
    else:
        optimal_cols = set(optimal_threshold)

    if isinstance(threshold, dict):
        fixed_thresholds = threshold
    else:
        fixed_thresholds = {col: threshold for col in score_cols}

    n = len(score_cols)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(figsize_per_cell * ncols, figsize_per_cell * nrows),
    )
    axes = np.atleast_1d(axes).flatten()

    results = {}

    for i, col in enumerate(score_cols):
        ax = axes[i]
        scores = df[col].values
        mask = ~pd.isna(scores) & ~pd.isna(y_true.values)
        y = y_true.values[mask]
        s = scores[mask]

        if col in optimal_cols:
            cutoff = _find_optimal_threshold(y, s)
        else:
            cutoff = fixed_thresholds.get(col, 0.5)

        pred = s >= cutoff

        table = pd.crosstab(
            pd.Categorical(y, categories=[False, True]),
            pd.Categorical(pred, categories=[False, True]),
            dropna=False,
        )
        table = table.reindex(index=[False, True], columns=[False, True], fill_value=0)

        p = _safe_chi2_pvalue(table.values)
        results[col] = {"threshold": cutoff, "table": table, "chi2_pvalue": p}

        _draw_single_contingency(
            ax, table.values, p,
            title=titles.get(col, col),
            cmap=cmap, fontsize=fontsize,
        )

    for j in range(n, len(axes)):
        axes[j].axis("off")

    fig.tight_layout()
    return fig, axes, results


def plot_AUROCs(df, true_label, score_columns, n_cols=6):
    """Plots a grid of ROC curves, one panel per score column.

    Panels are ordered by AUROC (highest first). Each panel reports the AUROC and
    the cutoff that maximises Youden's J statistic (TPR - FPR).

    Args:
        df (pd.DataFrame): Table with the labels and scores (no missing values in
            ``true_label`` or ``score_columns``).
        true_label (str): Column with the binary ground truth (0/1 or bool).
        score_columns (list of str): Columns with the model scores.
        n_cols (int, optional): Panels per row. Defaults to 6.
    """
    # 1. Pre-calculate AUCs to rank them
    auc_data = []
    for col in score_columns:
        auc = roc_auc_score(df[true_label], df[col])
        auc_data.append((col, auc))
    
    # Sort columns by AUC in descending order (highest to lowest)
    auc_data.sort(key=lambda x: x[1], reverse=True)
    sorted_score_columns = [item[0] for item in auc_data]
    
    # 2. Set up the plot grid
    n_rows = math.ceil(len(sorted_score_columns) / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols*4, n_rows*4))
    
    # Flatten axes for easy iteration (handles 1D or 2D arrays)
    axes = np.array(axes).flatten()
        
    for i, score_col in enumerate(sorted_score_columns):
        ax = axes[i]
       
        # Compute ROC
        fpr, tpr, thresholds = roc_curve(df[true_label], df[score_col])
        
        # Retrieve the pre-calculated AUC
        auc_value = next(item[1] for item in auc_data if item[0] == score_col)

        # Calculate Youden's J index to find the optimal cutoff
        optimal_idx = np.argmax(tpr - fpr)
        optimal_cutoff = thresholds[optimal_idx]

        # Plot ROC
        ax.plot(fpr, tpr)
        ax.plot([0, 1], [0, 1], linestyle='--')
        
        ax.set_title(score_col.replace("_", " "))
        ax.set_xlabel("FPR")
        ax.set_ylabel("TPR")

        # Add AUROC and Optimal Cutoff text 
        # Shifted x-coordinate to 0.50 to accommodate the longer text block
        ax.text(
            0.50, 0.05,
            f"AUC = {auc_value:.3f}\nCutoff = {optimal_cutoff:.3f}",
            transform=ax.transAxes,
            fontsize=11,
            verticalalignment='bottom'
        )
    
    # Hide extra axes if the number of plots isn't a multiple of n_cols
    for j in range(len(sorted_score_columns), len(axes)):
        axes[j].axis('off')
    
    plt.tight_layout()
    plt.show()
    

def box_grid(df, x_col, y_cols, n_cols=5, alternative="two-sided"):
    """Plots box plots of several variables split by a two-group column.

    Each panel is annotated with the p-value of a Mann-Whitney U test between the
    two groups (rows with missing values are excluded per panel).

    Args:
        df (pd.DataFrame): Table with the grouping and value columns.
        x_col (str): Grouping column with exactly two distinct non-missing values.
        y_cols (str or list of str): Column(s) to compare, one panel each.
        n_cols (int, optional): Panels per row. Defaults to 5.
        alternative (str, optional): ``"two-sided"``, ``"greater"`` or ``"less"``,
            passed to ``scipy.stats.mannwhitneyu``. One-sided tests compare the
            True (or 1) group against the False (or 0, or -1) group and therefore
            require a boolean-like ``x_col`` ({True, False}, {0, 1} or {-1, 1}).
            Defaults to ``"two-sided"``.

    Raises:
        ValueError: If ``x_col`` does not have exactly two groups, or a one-sided
            test is requested for a column that is not boolean-like.
    """
    if isinstance(y_cols, str):
        y_cols = [y_cols]

    def _format_pvalue(p):
        if np.isnan(p):
            return "P = n/a"
        return "P < 0.001" if p < 0.001 else f"P = {p:.3f}"


    def _identify_true_false_groups(unique_vals):
        """
        Given the two unique values of a boolean-like column, return
        (true_value, false_value). Accepts {True, False}, {0, 1}, or {-1, 1}
        encodings. Returns (None, None) if the values aren't boolean-like.
        """
        uniq = set(unique_vals)
        if uniq == {True, False}:
            return True, False
        if uniq == {0, 1}:
            return 1, 0
        if uniq == {-1, 1}:
            return 1, -1
        return None, None

    n_rows = math.ceil(len(y_cols) / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 4, n_rows * 4))

    # Flatten axes for easy iteration (handles a single Axes, 1D or 2D arrays)
    axes = np.atleast_1d(axes).flatten()

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

    for ax, y in zip(axes, y_cols):
        sns.boxplot(
            x=x_col,
            y=y,
            data=df,
            ax=ax,
            hue=x_col,
            legend=False,
            flierprops={"marker": "o", "markersize": 3, "markerfacecolor": "black", "markeredgecolor": "black"},
            width=0.4,
        )
        ax.set_title(y.replace("_", " "))

        sub = df[[x_col, y]].dropna()
        vals_first = sub.loc[sub[x_col] == group_first, y]
        vals_second = sub.loc[sub[x_col] == group_second, y]

        if len(vals_first) > 0 and len(vals_second) > 0:
            _, p = mannwhitneyu(vals_first, vals_second, alternative=alternative)
        else:
            p = np.nan

        ax.text(
            0.5, 0.98, _format_pvalue(p),
            transform=ax.transAxes,
            ha="center", va="top",
            fontsize=10, fontweight="bold",
        )

    # Hide extra axes if any
    for j in range(len(y_cols), len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    plt.show()


def plot_scatter_panel(df, x, y, hue, stratification_dict, cols=3, figsize_per_plot=(4, 4), save_fig = False, fig_path = "VEP_effect_size_scatter.pdf"):
    """Plots one scatter panel per stratum with a Spearman correlation and trend line.

    All panels share the axis limits and the colour of each ``hue`` level; a single
    legend is drawn outside the grid. Panels follow the order of
    ``stratification_dict``; rows with a missing ``x`` or ``y`` are excluded.

    Args:
        df (pd.DataFrame): Table with the plotted columns.
        x (str): Column for the x-axis (e.g. a predicted effect).
        y (str): Column for the y-axis (e.g. an observed effect).
        hue (str): Column used to colour the points.
        stratification_dict (dict): Panel title -> boolean mask or ``pd.Index``
            selecting rows of ``df`` (e.g. from `create_tss_stratifications`).
        cols (int, optional): Panels per row. Defaults to 3.
        figsize_per_plot (tuple of float, optional): Width and height of each panel
            in inches. Defaults to ``(4, 4)``.
        save_fig (bool, optional): Also save the figure to ``fig_path``. Defaults to False.
        fig_path (str, optional): Output path (300 dpi, transparent background).
            Defaults to ``"VEP_effect_size_scatter.pdf"``.
    """

    # --- GLOBAL CALCULATIONS ---
    # 1. Lock the hue order so colors never switch between plots
    global_hue_order = sorted(df[hue].dropna().unique())

    # 2. Calculate global x and y limits (adding a 5% padding margin)
    x_min, x_max = df[x].min(), df[x].max()
    y_min, y_max = df[y].min(), df[y].max()

    x_margin = (x_max - x_min) * 0.05
    y_margin = (y_max - y_min) * 0.05

    global_xlim = (x_min - x_margin, x_max + x_margin)
    global_ylim = (y_min - y_margin, y_max + y_margin)

    # --- PASS 1: Data Preparation & Pre-calculation ---
    plot_data = []

    for title, mask in stratification_dict.items():
        if isinstance(mask, pd.Index):
            subset = df.loc[mask].dropna(subset=[x, y])
        else:
            subset = df[mask].dropna(subset=[x, y])

        n_samples = len(subset)

        if n_samples > 1:
            rho, p_val = spearmanr(subset[x], subset[y])
        else:
            rho, p_val = np.nan, float('inf')

        plot_data.append({
            'title': title,
            'subset': subset,
            'n_samples': n_samples,
            'rho': rho,
            'p_val': p_val
        })

    # --- PASS 2: Plotting Setup ---
    n_plots = len(plot_data)
    rows = math.ceil(n_plots / cols)

    # Create grid WITHOUT sharex/sharey
    fig, axes = plt.subplots(
        rows, cols,
        figsize=(cols * figsize_per_plot[0], rows * figsize_per_plot[1])
    )

    axes = np.atleast_1d(axes).flatten()

    handles, labels = None, None

    # --- PASS 3: Generating Plots ---
    for i, data in enumerate(plot_data):
        ax = axes[i]
        subset = data['subset']
        title = data['title']
        n_samples = data['n_samples']
        p_val = data['p_val']
        rho = data['rho']

        # Apply the global limits to EVERY axis (even empty ones)
        ax.set_xlim(global_xlim)
        ax.set_ylim(global_ylim)

        if subset.empty or n_samples == 0:
            ax.set_title(f"{title}\n($n$=0)")
            continue

        # 1. Scatter plot
        sns.scatterplot(
            x=x,
            y=y,
            hue=hue,
            hue_order=global_hue_order, # FORCE consistent colors
            data=subset,
            ax=ax,
            legend=True
        )

        # Extract legend handles from the first valid plot
        if handles is None and ax.get_legend() is not None:
            handles, labels = ax.get_legend_handles_labels()

        # Remove the local legend from the subplot
        if ax.get_legend() is not None:
            ax.get_legend().remove()

        # 2. Trendline & Stats Annotation
        if n_samples > 1:
            if p_val < 0.001:
                p_str = "< 0.001"
            else:
                p_str = f"= {p_val:.3f}"

            # Plot dashed trendline
            sns.regplot(
                x=x,
                y=y,
                data=subset,
                scatter=False,
                ax=ax,
                color='black',
                truncate=False, # FORCE line to extend to plot edges
                line_kws={'linestyle': '--', 'alpha': 0.7}
            )

            # Add annotation box
            textstr = f'$n$ = {n_samples}\nSpearman $\\rho$: {rho:.2f}\n$p$-val {p_str}'
            props = dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='lightgray')
            ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=10,
                    verticalalignment='top', bbox=props)

        # Set Subplot Title
        ax.set_title(title)

    # Hide any unused subplots
    for j in range(n_plots, len(axes)):
        fig.delaxes(axes[j])

    # Add a single global legend outside the panel
    if handles is not None:
        fig.legend(
            handles, labels,
            loc='center left',
            bbox_to_anchor=(1.02, 0.5),
            title=hue,
            frameon=False
        )

    plt.tight_layout()
    if save_fig:
        plt.savefig(fig_path, dpi=300, bbox_inches='tight', transparent=True)
        print("Fig. saved")
    plt.show()


def plot_stratified_auprc_heatmap(df, true_label, score_columns, strat_dict, alternate_score_names=None, title = None, save_fig=False, fig_path="Causality_AUPRC_heatmap.pdf"):
    """Plots model AUPRCs per stratum as a three-part heatmap.

    Each row is a stratum; the panels show (1) the number of labelled variants,
    (2) the baseline AUPRC (positive rate) and (3) for each model
    ``log2(AUPRC / baseline)`` on a diverging scale centred at 0. Rows with a
    missing label or score are excluded per model; strata without both classes
    get NaN. Plot styling is applied only within this function.

    Args:
        df (pd.DataFrame): Table with the labels and scores.
        true_label (str): Column with the binary ground truth (0/1 or bool).
        score_columns (list of str): Columns with the model scores.
        strat_dict (dict): Stratum name -> boolean mask or index selecting rows of
            ``df`` (e.g. from `create_tss_stratifications`).
        alternate_score_names (list of str, optional): Display names for
            ``score_columns`` (same length). Defaults to the column names with
            underscores replaced by spaces.
        title (str, optional): Figure title. Defaults to None.
        save_fig (bool, optional): Also save the figure to ``fig_path``. Defaults to False.
        fig_path (str, optional): Output path (300 dpi, transparent background).
            Defaults to ``"Causality_AUPRC_heatmap.pdf"``.

    Returns:
        pd.DataFrame: Per stratum the sample size ``N``, the ``Baseline`` and the
        ``log2(AUPRC / baseline)`` of each model (columns named by display name).

    Raises:
        ValueError: If ``alternate_score_names`` and ``score_columns`` differ in length.
    """

    # Validate alternate score names if provided
    if alternate_score_names is not None:
        if len(alternate_score_names) != len(score_columns):
            raise ValueError("The length of 'alternate_score_names' must match the length of 'score_columns'.")
        # Map original names to alternate names for quick renaming later
        name_mapping = dict(zip(score_columns, alternate_score_names))
    else:
        name_mapping = {col: col.replace("_", " ") for col in score_columns}

    # Plot styling is applied only inside this function (via rc_context below),
    # so the caller's global matplotlib/seaborn settings are left untouched.
    # Embeds true fonts for pdf/ps export and keeps text editable in svg.
    style_rc = {
        **sns.axes_style("ticks"),
        **sns.plotting_context("talk", font_scale=0.85),
        'pdf.fonttype': 42,
        'ps.fonttype': 42,
        'svg.fonttype': 'none',
        'font.family': 'sans-serif',
        'font.sans-serif': ['DejaVu Sans', 'Helvetica', 'Arial', 'sans-serif'],
    }

    results_metrics = {}
    results_n = {}

    for strat_name, strat_idx in strat_dict.items():
        # Subset the dataframe
        df_subset = df.loc[strat_idx]

        # 1. Calculate N (Number of valid samples with a ground truth label)
        valid_y = df_subset[true_label].dropna()
        n_samples = len(valid_y)
        results_n[strat_name] = {'N': n_samples}

        # 2. Calculate the baseline (positive rate)
        baseline = valid_y.mean() if n_samples > 0 else np.nan
        row_data = {'Baseline': baseline}

        # 3. Calculate the AUPRC / baseline ratio for each model independently
        for score_col in score_columns:
            mask = df_subset[true_label].notna() & df_subset[score_col].notna()
            y_true = df_subset.loc[mask, true_label]
            y_score = df_subset.loc[mask, score_col]

            if len(y_true) == 0 or len(y_true.unique()) < 2:
                pct_delta_auprc = np.nan
            else:
                abs_auprc = average_precision_score(y_true, y_score)

                # Compute ratio to baseline protecting against division by zero
                if baseline == 0 or pd.isna(baseline):
                    pct_delta_auprc = np.nan
                else:
                    pct_delta_auprc = ((abs_auprc) / baseline)

            row_data[score_col] = pct_delta_auprc

        results_metrics[strat_name] = row_data

    # Convert results to DataFrames
    df_n = pd.DataFrame.from_dict(results_n, orient='index')
    df_metrics = pd.DataFrame.from_dict(results_metrics, orient='index')

    # Split Baseline and Models into two separate DataFrames
    df_baseline = df_metrics[['Baseline']]

    # Compute log2 values
    model_cols = [col for col in score_columns if col in df_metrics.columns]
    df_delta = np.log2(df_metrics[model_cols].copy())

    # Rename columns using either the alternate names or the fallback space-replacement
    df_delta = df_delta.rename(columns=name_mapping)

    # Setup the Split Grid
    width = len(model_cols) * 0.6 + 3
    height = len(strat_dict) * 0.4 + 1

    with plt.rc_context(style_rc):
        # Create side-by-side subplots with barely any space between them
        fig, (ax_n, ax_base, ax_delta) = plt.subplots(
            1, 3,
            figsize=(width, height),
            gridspec_kw={'width_ratios': [1, 1, len(model_cols)], 'wspace': 0.05}
        )

        def add_border(ax):
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_color('black')
                spine.set_linewidth(1)

        # --- PLOT 1: Sample Size Heatmap ---
        sns.heatmap(
            df_n,
            annot=True,
            fmt="d",
            cmap="Blues",
            vmin=0,
            cbar=False,
            linewidths=0.5,
            annot_kws={"size": 8, "weight":'bold'},
            linecolor='black',
            ax=ax_n
        )
        add_border(ax_n)

        # --- PLOT 2: The Baseline (Absolute) Heatmap ---
        sns.heatmap(
            df_baseline,
            annot=True,
            fmt=".2f",
            cmap="Greys",
            vmin=0,
            vmax=1,
            cbar=False,
            annot_kws={"size": 8, "weight": "bold"},
            linewidths=0.5,
            linecolor='black',
            ax=ax_base
        )
        add_border(ax_base)

        # --- PLOT 3: The Delta Heatmap ---
        max_abs_delta = df_delta.abs().max().max()
        if pd.isna(max_abs_delta) or max_abs_delta < 0.1:
            vmax_div = 0.1
        else:
            vmax_div = max_abs_delta

        vmin_div = -vmax_div

        sns.heatmap(
            df_delta,
            annot=True,
            fmt=".2f",
            cmap="icefire",
            center=0,
            vmin=vmin_div,
            vmax=vmax_div,
            linewidths=0.5,
            linecolor='black',
            annot_kws={"size": 7, "weight": "bold"},
            cbar_kws={'label': 'log2(AUPRC/Baseline)', 'shrink': 0.6},
            ax=ax_delta
        )
        add_border(ax_delta)

        cbar = ax_delta.collections[0].colorbar
        if cbar:
            cbar.outline.set_linewidth(1)
            cbar.outline.set_color('black')
            cbar.outline.set_visible(True)

        # --- STYLING & CLEANUP ---
        # Ax 1: N
        ax_n.set_yticklabels(ax_n.get_yticklabels(), rotation=0)
        ax_n.set_ylabel('Stratification', fontweight='bold', fontsize=12)
        ax_n.set_xticklabels(['n Variants'], rotation=45, ha='right')

        # Ax 2: Baseline
        ax_base.set_yticks([])
        ax_base.set_ylabel('')
        ax_base.set_xticklabels(['Baseline positive rate'], rotation=45, ha='right')

        # Ax 3: Delta Models
        ax_delta.set_yticks([])
        ax_delta.set_ylabel('')
        ax_delta.set_xticklabels(ax_delta.get_xticklabels(), rotation=45, ha='right')

        # Add a thick black line on the Delta plot to heavily divide it from the absolute columns
        ax_delta.axvline(0, color='black', linewidth=3)

        # Return the unified table
        results_df_full = pd.concat([df_n, df_baseline, df_delta], axis=1)

        if title:
            plt.suptitle(title,fontweight='bold', fontsize=12, y=1.05)

        if save_fig:
            plt.savefig(fig_path, dpi=300, bbox_inches='tight', transparent=True)
            print("Fig. saved")
        plt.show()

    return results_df_full