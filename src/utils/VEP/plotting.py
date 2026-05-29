import math
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    average_precision_score, 
    precision_recall_curve, 
    roc_auc_score, 
    roc_curve, 
    confusion_matrix
)
from scipy.stats import chi2_contingency


def plot_AUPRCs(df, true_label, score_columns, n_cols=6):
    """
    Plots a grid of Precision-Recall (PR) curves for multiple model score columns.
    The plots are sorted in descending order based on their Area Under the PR Curve (AUPRC).
    Robust to NaN values in score columns.
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
    """
    Generates side-by-side bar charts comparing AUROC and AUPRC scores across multiple models.
    Models are sorted in descending order of their AUPRC score.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The dataframe containing true labels and predicted scores.
    true_label : str
        The column name for ground truth binary labels.
    score_columns : list of str
        Column names representing model predictions.
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
    """
    Plots a confusion matrix heatmap for categorical/sign predictions and 
    calculates the Chi-Square test of independence.
    
    Parameters:
    -----------
    y_observed : array-like
        The true observed labels (e.g., -1 for negative, 1 for positive).
    y_predicted : array-like
        The predicted labels from the model.
    labels : list, default=[-1, 1]
        The specific class labels to include in the confusion matrix.
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
    

def plot_AUROCs(df, true_label, score_columns, n_cols=6):
    """
    Plots a grid of Receiver Operating Characteristic (ROC) curves for multiple models.
    The plots are sorted in descending order based on their AUROC score. Optimal
    cutoffs are calculated using Youden's J statistic.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The dataframe containing the true labels and predicted scores.
    true_label : str
        The column name representing ground truth binary labels.
    score_columns : list of str
        Column names representing model predictions.
    n_cols : int, default=6
        The number of columns in the plot grid.
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
    

def box_grid(df, x_col, y_cols, n_cols=5):
    """
    Plots a grid of Seaborn box plots comparing multiple continuous variables 
    against a single categorical variable.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The dataframe containing the data to plot.
    x_col : str
        The column name to use for the x-axis (usually categorical).
    y_cols : list of str
        A list of column names to plot on the y-axis (usually continuous).
    n_cols : int, default=5
        The number of columns in the plot grid.
    """
    n_rows = math.ceil(len(y_cols) / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols*4, n_rows*4))
    
    # Flatten axes for easy iteration
    axes = np.array(axes).flatten()

    if len(y_cols) == 1:
        axes = [axes]

    for ax, y in zip(axes, y_cols):
        sns.boxplot(
            x=x_col,
            y=y,
            data=df,
            ax=ax,
            hue = x_col,
            legend = False,
            flierprops = {"marker": "o", "markersize": 3, 'markerfacecolor': 'black',  'markeredgecolor': 'black'},
            width = 0.4
        )
        ax.set_title(y.replace("_", " "))

    # Hide extra axes if any
    for j in range(len(y_cols), len(axes)):
        axes[j].axis('off')
        
    plt.tight_layout()
    plt.show()