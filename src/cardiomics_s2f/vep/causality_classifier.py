"""
VEP Causality Classifier (Random Forest)
========================================

Random forest that combines VEP scores and variant annotation to classify de novo
variants of CHD probands vs controls (``is_CHD``), trained with patient-grouped,
stratified k-fold cross-validation so that every variant gets an out-of-fold
(OOF) prediction:

For each outer fold, a hyperparameter search (``RandomizedSearchCV``, scored by
average precision) runs on that fold's training part only, with its own grouped
inner CV; the best model is refit on the training part, predicts the held-out
variants (never scored by a model that saw their patient) and its feature
importances are recorded.

* `train_grouped_oof_rf_classifier`: the training loop; returns OOF predictions,
  per-fold feature importances and hyperparameters.
* `plot_rf_feature_importances`: mean (+- SD across folds) importances.
* Helpers: `add_variant_type`, `fill_conditional_categoricals`,
  `coerce_categorical_dtypes`, `build_preprocessor`, `build_search`.

The default column lists match the Richter benchmark table
(``analysis_notebooks/benchmarks/richter_vep_benchmark.ipynb``).
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import randint
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import RandomizedSearchCV, StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


# --------------------------------------------------------------------------- #
# Default column lists
# --------------------------------------------------------------------------- #

DEFAULT_NUMERIC_COLS = [
    "CADD_PHRED_score", "abs_COUNTSxJSD_cPBN_raw_score",
    "abs_AG_ft_ATAC_score", "abs_AG_ft_H3K27ac_score",
    "abs_AG_ft_H3K4me3_score", "abs_AG_ft_RNA_score",
    "abs_AG_pre_ATAC_raw_score", "abs_AG_pre_CTCF_TF_raw_score",
    "abs_AG_pre_H3K27ac_HIST_raw_score", "abs_AG_pre_H3K27me3_HIST_raw_score",
    "abs_AG_pre_H3K36me3_HIST_raw_score", "abs_AG_pre_H3K4me1_HIST_raw_score",
    "abs_AG_pre_H3K4me3_HIST_raw_score", "abs_AG_pre_H3K9ac_HIST_raw_score",
    "abs_AG_pre_H3K9me3_HIST_raw_score", "abs_AG_pre_POLR2A_TF_raw_score",
    "abs_AG_pre_RNA_raw_score", "H3K4me3_maxZ", "H3K27ac_maxZ",
]
DEFAULT_CATEGORICAL_COLS = [
    "cCRE_class", "gtf_feature", "gene_type", "is_cardioid_CA",
    "enhancer_regulatory_position", "is_cardioid_H3K27ac",
    "celltypes_in_enhancer", "in_cardioid_enhancer",
]
DEFAULT_CONDITIONAL_COLS = ["celltypes_in_enhancer", "is_cardioid_H3K27ac"]
DEFAULT_BALANCE_COLS = ["CHROM", "variant_type"]


# --------------------------------------------------------------------------- #
# 1. Feature engineering helpers
# --------------------------------------------------------------------------- #

def add_variant_type(df: pd.DataFrame, ref_col: str = "REF", alt_col: str = "ALT") -> pd.DataFrame:
    """Derive SNV / insertion / deletion from REF/ALT allele lengths."""
    df = df.copy()

    def _classify(row):
        ref, alt = row[ref_col], row[alt_col]
        if pd.isna(ref) or pd.isna(alt):
            return "other"
        if len(ref) == 1 and len(alt) == 1:
            return "SNV"
        if len(alt) > len(ref):
            return "insertion"
        if len(alt) < len(ref):
            return "deletion"
        return "mnv"

    df["variant_type"] = df.apply(_classify, axis=1)
    return df


def fill_conditional_categoricals(df: pd.DataFrame, conditional_cols: list[str],
                                   fill_value: str = "Not_applicable") -> pd.DataFrame:
    """
    Recode NaNs that mean "condition not met elsewhere" (e.g.
    celltypes_in_enhancer only set when in_cardioid_enhancer is True) as an
    explicit category, rather than imputing or dropping -- the missingness
    itself is informative.
    """
    df = df.copy()
    for col in conditional_cols:
        if col in df.columns:
            df[col] = df[col].astype("object").where(df[col].notna(), fill_value)
    return df


def coerce_categorical_dtypes(df: pd.DataFrame, categorical_cols: list[str]) -> pd.DataFrame:
    """Uniformly stringify categorical columns (NaNs preserved) so OneHotEncoder
    doesn't choke on mixed bool/str values."""
    df = df.copy()
    for col in categorical_cols:
        if col in df.columns:
            df[col] = df[col].map(lambda v: v if pd.isna(v) else str(v))
    return df


def _build_strat_key(df: pd.DataFrame, label_col: str, balance_cols: list[str],
                      min_group_size: int = 4) -> pd.Series:
    strat_key = df[label_col].astype(str)
    for col in balance_cols:
        strat_key = df[col].astype(str) + "_" + strat_key
    counts = strat_key.value_counts()
    rare = counts[counts < min_group_size].index
    return strat_key.where(~strat_key.isin(rare), "rare_combo")


# --------------------------------------------------------------------------- #
# 2. Preprocessing + model pipeline
# --------------------------------------------------------------------------- #

def build_preprocessor(numeric_cols: list[str], categorical_cols: list[str]) -> ColumnTransformer:
    numeric_pipe = Pipeline([("impute", SimpleImputer(strategy="median"))])
    categorical_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="constant", fill_value="Missing")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=True)),
    ])
    return ColumnTransformer([
        ("num", numeric_pipe, numeric_cols),
        ("cat", categorical_pipe, categorical_cols),
    ])


def build_search(preprocessor: ColumnTransformer, random_state: int, n_iter: int,
                  n_splits: int) -> RandomizedSearchCV:
    pipe = Pipeline([
        ("prep", preprocessor),
        ("clf", RandomForestClassifier(random_state=random_state, n_jobs=-1)),
    ])
    param_dist = {
        "clf__n_estimators": randint(200, 800),
        "clf__max_depth": [None, 6, 10, 15, 20, 30],
        "clf__min_samples_split": randint(2, 20),
        "clf__min_samples_leaf": randint(1, 10),
        "clf__max_features": ["sqrt", "log2", 0.3, 0.5],
        "clf__class_weight": [None, "balanced", "balanced_subsample"],
    }
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    return RandomizedSearchCV(
        pipe, param_distributions=param_dist, n_iter=n_iter,
        scoring="average_precision", cv=cv, random_state=random_state,
        n_jobs=-1, refit=True, verbose=0,
    )


# --------------------------------------------------------------------------- #
# 3. K-fold out-of-fold training
# --------------------------------------------------------------------------- #

def train_grouped_oof_rf_classifier(
    df: pd.DataFrame,
    label_col: str,
    group_col: str,
    id_col: str,
    numeric_cols: list[str],
    categorical_cols: list[str],
    conditional_cols: list[str],
    balance_cols: list[str],
    outer_splits: int = 5,
    inner_splits: int = 5,
    n_iter: int = 40,
    random_state: int = 42,
):
    """
    Patient-grouped, stratified outer k-fold CV: every variant gets exactly
    one out-of-fold prediction, from a model that never saw that variant's
    patient during its own training or hyperparameter search.

    Returns (oof_df, importances_df, params_df).
    """
    df = add_variant_type(df)
    df = fill_conditional_categoricals(df, conditional_cols)
    df = df.reset_index(drop=True)  # guarantee clean positional indexing below

    categorical_cols = list(categorical_cols)
    if "variant_type" not in categorical_cols:
        categorical_cols = categorical_cols + ["variant_type"]

    if label_col in categorical_cols:
        categorical_cols = [c for c in categorical_cols if c != label_col]
    numeric_cols = [c for c in numeric_cols if c != label_col]
    if group_col in balance_cols:
        raise ValueError(f"'{group_col}' (group_col) must not also appear in balance_cols.")

    df = coerce_categorical_dtypes(df, categorical_cols)

    strat_key = _build_strat_key(df, label_col, balance_cols)
    outer_cv = StratifiedGroupKFold(n_splits=outer_splits, shuffle=True, random_state=random_state)

    n = len(df)
    oof_proba = np.full(n, np.nan)
    oof_pred = np.full(n, np.nan)
    oof_fold = np.full(n, -1, dtype=int)
    importance_records = []
    param_records = []

    feature_cols = numeric_cols + categorical_cols

    for fold_i, (train_pos, test_pos) in enumerate(outer_cv.split(df, strat_key, groups=df[group_col])):
        train_df = df.iloc[train_pos]
        test_df = df.iloc[test_pos]

        # sanity check: no patient overlap between this fold's train/test
        assert set(train_df[group_col]) & set(test_df[group_col]) == set(), (
            f"Fold {fold_i}: patient overlap between train and held-out fold."
        )

        X_train, y_train = train_df[feature_cols], train_df[label_col]
        X_test = test_df[feature_cols]

        preprocessor = build_preprocessor(numeric_cols, categorical_cols)
        search = build_search(preprocessor, random_state=random_state, n_iter=n_iter, n_splits=inner_splits)
        search.fit(X_train, y_train, groups=train_df[group_col])
        best_model = search.best_estimator_

        proba = best_model.predict_proba(X_test)[:, 1]
        pred = best_model.predict(X_test)
        oof_proba[test_pos] = proba
        oof_pred[test_pos] = pred
        oof_fold[test_pos] = fold_i

        feature_names = best_model.named_steps["prep"].get_feature_names_out()
        importances = best_model.named_steps["clf"].feature_importances_
        for feat, imp in zip(feature_names, importances):
            importance_records.append({"fold": fold_i, "feature": feat, "importance": float(imp)})

        param_records.append({"fold": fold_i, **search.best_params_})

        fold_y_test = test_df[label_col].astype(int)
        fold_auc = roc_auc_score(fold_y_test, proba) if fold_y_test.nunique() > 1 else float("nan")
        fold_ap = average_precision_score(fold_y_test, proba) if fold_y_test.nunique() > 1 else float("nan")
        print(f"[fold {fold_i}] train={len(train_df)} test={len(test_df)} "
              f"AUC={fold_auc:.4f} AP={fold_ap:.4f} best_params={search.best_params_}")

    assert not np.isnan(oof_proba).any(), "Some variants never received an OOF prediction -- check outer CV."

    oof_df = pd.DataFrame({
        id_col: df[id_col],
        group_col: df[group_col],
        label_col: df[label_col],
        "oof_predicted_proba": oof_proba,
        "oof_predicted_label": oof_pred.astype(bool) if df[label_col].dtype == bool else oof_pred,
        "fold": oof_fold,
    })

    importances_df = pd.DataFrame(importance_records)
    params_df = pd.DataFrame(param_records)

    overall_y = df[label_col].astype(int)
    print(f"\n[overall OOF] AUC={roc_auc_score(overall_y, oof_proba):.4f} "
          f"AP={average_precision_score(overall_y, oof_proba):.4f}")

    return oof_df, importances_df, params_df


# --------------------------------------------------------------------------- #
# 4. Feature importances
# --------------------------------------------------------------------------- #

def plot_rf_feature_importances(
    importances_df: pd.DataFrame,
    numeric_cols: list[str],
    categorical_cols: list[str],
    top_n: int = 25,
    collapse_onehot: bool = True,
    save_path: str | None = None,
):
    """Plots the top random forest feature importances, averaged over the outer folds.

    Args:
        importances_df (pd.DataFrame): Per-fold importances from
            `train_grouped_oof_rf_classifier` (``fold``, ``feature``, ``importance``;
            feature names with the ``num__`` / ``cat__`` prefixes of the
            preprocessor).
        numeric_cols (list of str): Numeric feature columns used for training.
        categorical_cols (list of str): Categorical feature columns used for
            training (``variant_type`` is added automatically).
        top_n (int, optional): Number of features shown. Defaults to 25.
        collapse_onehot (bool, optional): Sum the importances of all one-hot
            columns of a categorical column (e.g. ``cCRE_class_Promoter``,
            ``cCRE_class_TF`` -> ``cCRE_class``) per fold, so that the source
            columns are compared. Set False to see each category. Defaults to True.
        save_path (str, optional): Also save the figure to this path. Defaults to None.

    Returns:
        tuple: ``(fig, imp_df)`` with ``imp_df`` holding ``feature``,
        ``importance`` (mean over folds) and ``importance_sd``, sorted by
        importance.
    """
    imp = importances_df.copy()
    # strip the ColumnTransformer's "num__"/"cat__" prefixes for readability
    imp["feature"] = imp["feature"].str.split("__", n=1).str[-1]

    if collapse_onehot:
        cat_cols = list(categorical_cols)
        if "variant_type" not in cat_cols:
            cat_cols = cat_cols + ["variant_type"]

        def source_col(fname: str) -> str:
            if fname in numeric_cols:
                return fname
            # match the longest categorical column name that prefixes fname
            candidates = [c for c in cat_cols if fname.startswith(c + "_")]
            return max(candidates, key=len) if candidates else fname

        imp["feature"] = imp["feature"].apply(source_col)
        imp = imp.groupby(["fold", "feature"], as_index=False)["importance"].sum()

    imp_df = (
        imp.groupby("feature")["importance"]
        .agg(importance="mean", importance_sd="std")
        .sort_values("importance", ascending=False)
        .reset_index()
    )

    plot_df = imp_df.head(top_n).iloc[::-1]  # reverse so biggest is on top

    fig, ax = plt.subplots(figsize=(8, max(4, 0.35 * len(plot_df))))
    ax.barh(plot_df["feature"], plot_df["importance"], xerr=plot_df["importance_sd"],
            color="#4C72B0", ecolor="dimgray", capsize=2)
    ax.set_xlabel("Feature importance (RF impurity-based, mean +- SD over folds)")
    ax.set_title(f"Top {len(plot_df)} feature importances")
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig, imp_df
