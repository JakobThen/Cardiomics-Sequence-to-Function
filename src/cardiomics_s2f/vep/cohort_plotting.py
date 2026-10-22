"""
Variant Cohort Plotting
=======================

Descriptive figures comparing the de novo variants (DNVs) of CHD probands and
controls (Richter et al. 2020): DNV burden, genomic and coding annotation,
conservation, ENCODE4 cCREs and overlap with cardioid ATAC peaks and ABC
enhancers.

Inputs are the annotated variant tables written by
``analysis_notebooks/variant_datasets/richter_variant_dataset.ipynb``. Cohort
numbers taken from the paper (proband counts, gene-list DNV counts) are fixed in
the figures. The functions use the current matplotlib/seaborn style.
"""

import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import chi2_contingency, mannwhitneyu
from statsmodels.stats.multitest import multipletests

from cardiomics_s2f.io.variant_annotation import CONSEQUENCE_RANK, IMPACT

# ENCODE4 cCRE classes in plotting order, and their colours
CRE_ORDER = [
    "Promoter", "Proximal enhancer", "Distal enhancer",
    "TF", "CA", "CA-CTCF", "CA-TF", "CA-H3K4me3"
]
CRE_COLORS = {
    "Promoter":          "#E63946",
    "Proximal enhancer": "#F4A261",
    "Distal enhancer":   "#2A9D8F",
    "TF":                "#6A4C93",
    "CA":                "#457B9D",
    "CA-CTCF":           "#1D3557",
    "CA-TF":             "#A8DADC",
    "CA-H3K4me3":        "#F1FAEE"
}


def format_p_value(p):
    """Formats a p-value as a figure label.

    Args:
        p (float): P-value (NaN allowed).

    Returns:
        str: ``"P < 0.001"``, ``"P = <p, 3 decimals>"`` or ``""`` for NaN.
    """
    if pd.isna(p): return ""
    if p < 0.001: return "P < 0.001"
    return f"P = {p:.3f}"


def _count_ccre_classes(df):
    return df["cCRE_class"].value_counts().reindex(CRE_ORDER, fill_value=0).values


def _ccre_class_chi2_pvalue(df_a, df_b):
    table = np.array([_count_ccre_classes(df_a), _count_ccre_classes(df_b)])
    col_mask = (table[0] > 0) & (table[1] > 0)
    table = table[:, col_mask]
    if table.shape[1] < 2 or table.sum(axis=1).min() == 0:
        return np.nan
    _, p, _, _ = chi2_contingency(table)
    return p


def _significance_stars(p):
    if np.isnan(p): return "" 
    if p < 0.001:   return "***"
    if p < 0.01:    return "**"
    if p < 0.05:    return "*"
    return "ns"


def _draw_significance_bracket(ax, x1, x2, y, label, color):
    pad = 1.2
    ax.plot([x1, x1, x2, x2], [y, y+pad, y+pad, y], lw=1, color=color)
    ax.text((x1+x2)/2, y+pad+0.2, label, ha="center", va="bottom", fontsize=10,
            color=color, fontweight="bold" if label != "ns" else "normal",
            style="italic" if label == "ns" else "normal")


def label_cardioid_enhancer_state(row):
    """Labels a variant by its cardioid enhancer state.

    Args:
        row (pd.Series): Variant with ``in_cardioid_enhancer``,
            ``enhancer_regulatory_position`` and ``is_cardioid_H3K27ac``.

    Returns:
        str: ``"Not Active in Cardioid"``, or the enhancer position and
        acetylation, e.g. ``"Distal H3K27ac+"``.
    """
    if row['in_cardioid_enhancer'] == False or pd.isna(row['in_cardioid_enhancer']):
        return "Not Active in Cardioid"
    
    # Format the position and acetylation status into a clean string
    pos = str(row['enhancer_regulatory_position']).capitalize()
    ac_status = "H3K27ac+" if row['is_cardioid_H3K27ac'] == True else "H3K27ac-"
    
    return f"{pos} {ac_status}"


def plot_richter_cohort_overview(df_chd, df_ctrl, CA_results_dict=None, save_fig=False, filepath='CHD_Extended_Panel.pdf', celltype_filter=None):
    """Plots the main descriptive figure of the CHD and control DNV cohorts.

    Twelve panels: (A) DNVs per proband, (B) distance to the nearest TSS,
    (C) Richter gene-list annotation (fixed counts from Richter et al. 2020),
    (D) GTF features, (E) coding effects, (F) ENCODE4 cCRE classes, (G) CADD,
    (H) phyloP, (I) ENCODE4 maxZ signal at cCREs, (J) overlap with cardioid ATAC
    peaks, (K) cCRE classes of variants inside vs outside ATAC peaks and
    (L) overlap with ABC candidate enhancers. Cohorts are compared with
    chi-squared tests (categorical panels; panel K BH-corrected across cell
    types) and Mann-Whitney U tests (continuous panels). Panels J and K are only
    drawn if ``CA_results_dict`` is given.

    Args:
        df_chd (pd.DataFrame): Annotated CHD proband variants (``cCRE_class``,
            ``gtf_feature``, ``coding_effect``, ``closest_TSS``, ``phyloP_100way``,
            ``CADD_PHRED_score``, ``DNase/H3K4me3/H3K27ac/CTCF_maxZ``, ``Blinded ID``,
            ``in_cardioid_enhancer``, ``celltypes_in_enhancer``,
            ``enhancer_regulatory_position``, ``is_cardioid_H3K27ac``).
        df_ctrl (pd.DataFrame): Annotated control variants with the same columns.
        CA_results_dict (dict, optional): Variants inside and outside each cell
            type's ATAC peaks, ``{cell_type: {"CHD"|"Control": {"in"|"out": pd.DataFrame}}}``.
            Defaults to None.
        save_fig (bool, optional): Whether to save the figure to ``filepath``.
            Defaults to False.
        filepath (str, optional): Output path. Defaults to 'CHD_Extended_Panel.pdf'.
        celltype_filter (list of str, optional): Cell types (and their order) shown
            in panels J-L. Defaults to None (all).
    """
    # Beautiful, high-contrast colors from your preferred style
    color_chd = "#E63946"
    color_ctrl = "#2A9D8F"
    element_width = 0.6 
    
    fig = plt.figure(figsize=(20, 22))
    
    # Establish structural grid to independently control spatial relationships of different sub-figures
    gs = gridspec.GridSpec(6, 3, height_ratios=[0.7, 0.05, 1, 0.45, 0.7, 1], wspace=0.3, hspace=0.4)
    
    ax_g = fig.add_subplot(gs[0, 0])
    ax_m = fig.add_subplot(gs[0, 1]) 
    ax_b = fig.add_subplot(gs[0, 2])
    #dummy sapcer 1
    
    ax_e = fig.add_subplot(gs[2, 0])
    ax_j = fig.add_subplot(gs[2, 1])
    ax_c = fig.add_subplot(gs[2, 2])  
    
    #dummy spacer row 3
    
    ax_o = fig.add_subplot(gs[4, 0])
    ax_d = fig.add_subplot(gs[4, 1])
    ax_f = fig.add_subplot(gs[4, 2])
    
    ax_k = fig.add_subplot(gs[5, 0]) 
    ax_l = fig.add_subplot(gs[5, 1]) 
    ax_n = fig.add_subplot(gs[5, 2]) 
    

    # ==========================================
    # DATA PREP
    # ==========================================
    df_chd = df_chd.copy()
    df_ctrl = df_ctrl.copy()
    df_chd['Cohort'] = 'CHD'
    df_ctrl['Cohort'] = 'Control'
    df_combined = pd.concat([df_chd, df_ctrl], ignore_index=True)

            
   
        
    # ==========================================
    # PANEL C: cCRE Class Distribution
    # ==========================================
    df_combined["cCRE_class"] = df_combined["cCRE_class"].fillna("No CRE")
    
    # Calculate counts and unstack to wide format (rows=Cohort, cols=cCRE_class)
    ccre_counts = df_combined.groupby(['Cohort', 'cCRE_class']).size().unstack(fill_value=0)
    
    # 1. Enforce the global CRE_ORDER, appending "No CRE" at the end
    full_c_order = CRE_ORDER + ["No CRE"]
    
    # Reindex columns to guarantee the order (fills missing classes with 0)
    ccre_counts = ccre_counts.reindex(columns=full_c_order, fill_value=0)
    
    # Drop any columns that are completely empty (in case some CRE classes don't exist in this specific dataset)
    ccre_counts = ccre_counts.loc[:, (ccre_counts != 0).any(axis=0)]
    
    # Convert counts to percentages across each row (Cohort)
    ccre_pct = ccre_counts.div(ccre_counts.sum(axis=1), axis=0) * 100
    
    # 2. Build the color list mapping to CRE_COLORS (defaulting to Gray for "No CRE")
    ccre_colors = [CRE_COLORS.get(col, "#CCCCCC") for col in ccre_pct.columns]
    
    # Plot stacked bar (turn off default legend)
    ccre_pct.plot(kind='bar', stacked=True, color=ccre_colors, ax=ax_c, edgecolor="black", width=element_width, legend=False)
    
    ax_c.set_title('F) ENCODE4 cCRE Annotations', loc='left', fontweight='bold')
    ax_c.set_ylabel('Proportion of DNVs (%)')
    ax_c.set_xlabel('')
    ax_c.set_xticklabels(ax_c.get_xticklabels(), rotation=0) 
    ax_c.spines[['top', 'right']].set_visible(False)
    
    # Custom Legend with integrated counts placed below the plot
    legend_labels_c = []
    for i, col in enumerate(ccre_pct.columns):
        chd_n = ccre_counts.loc['CHD', col] if 'CHD' in ccre_counts.index else 0
        ctrl_n = ccre_counts.loc['Control', col] if 'Control' in ccre_counts.index else 0
        legend_labels_c.append(f"{col} (n={int(chd_n)} vs {int(ctrl_n)})")
        
    handles_c = [plt.Rectangle((0,0),1,1, color=ccre_colors[i], ec="black") for i in range(len(ccre_pct.columns))]
    
    # Note: If this legend gets too tall, you can change ncol=2
    ax_c.legend(handles_c, legend_labels_c, title='cCRE Class (CHD vs Ctrl)', 
                bbox_to_anchor=(0.5, -0.15), loc='upper center', frameon=False, fontsize=9, ncol=1)
    
    # Calculate and place P-value
    contingency_table = pd.crosstab(df_combined['Cohort'], df_combined['cCRE_class'])
    if not contingency_table.empty:
        _, p_val, _, _ = chi2_contingency(contingency_table)
        ax_c.text(0.5, 0.95, format_p_value(p_val), transform=ax_c.transAxes, 
                  ha='center', va='bottom', fontweight='bold', fontsize=10)

    # ==========================================
    # PANEL D: phyloP_100way Distribution
    # ==========================================
    df_phylo = df_combined.dropna(subset=['phyloP_100way'])
    sns.violinplot(data=df_phylo, x='Cohort', y='phyloP_100way', hue='Cohort',
                   palette=[color_chd, color_ctrl], ax=ax_d, legend=False, inner="quartile")
    
    ax_d.set_title('H) Conservation (phyloP100)', loc='left', fontweight='bold')
    ax_d.set_ylabel('phyloP Score')
    ax_d.set_xlabel('')
    ax_d.spines[['top', 'right']].set_visible(False)
    
    scores_chd = df_phylo[df_phylo['Cohort'] == 'CHD']['phyloP_100way']
    scores_ctrl = df_phylo[df_phylo['Cohort'] == 'Control']['phyloP_100way']
    if len(scores_chd) > 0 and len(scores_ctrl) > 0:
        _, p_val_mw = mannwhitneyu(scores_chd, scores_ctrl, alternative='greater')
        ax_d.text(0.5, 0.95, format_p_value(p_val_mw), transform=ax_d.transAxes, 
                  ha='center', va='top', fontweight='bold', fontsize=10)
        
    # ==========================================
    # PANEL O: CADD Distribution
    # ==========================================
    df_phylo = df_combined.dropna(subset=['CADD_PHRED_score'])
    sns.violinplot(data=df_phylo, x='Cohort', y='CADD_PHRED_score', hue='Cohort',
                   palette=[color_chd, color_ctrl], ax=ax_o, legend=False, inner="quartile")
    
    ax_o.set_title('G) CADD', loc='left', fontweight='bold')
    ax_o.set_ylabel('PHRED Score')
    ax_o.set_xlabel('')
    ax_o.spines[['top', 'right']].set_visible(False)
    
    scores_chd = df_phylo[df_phylo['Cohort'] == 'CHD']['CADD_PHRED_score']
    scores_ctrl = df_phylo[df_phylo['Cohort'] == 'Control']['CADD_PHRED_score']
    if len(scores_chd) > 0 and len(scores_ctrl) > 0:
        _, p_val_mw = mannwhitneyu(scores_chd, scores_ctrl, alternative='greater')
        ax_o.text(0.5, 0.95, format_p_value(p_val_mw), transform=ax_o.transAxes, 
                  ha='center', va='top', fontweight='bold', fontsize=10)

   # ==========================================
    # PANEL E: GTF Feature Distribution (Stacked Bar)
    # ==========================================
    if 'gtf_feature' in df_combined.columns:
        df_combined["gtf_feature"] = df_combined["gtf_feature"].fillna("Intergenic")
        
        # 1. Define the explicit order and color mapping
        feature_order = [
            'start_codon', 'stop_codon', 'CDS', 'exon', 
            'UTR', 'intron', 'gene', 'Intergenic'
        ]
        
        feature_colors = {
            # Greys (Pure Black -> Medium Grey)
            'start_codon':  '#999999',
            'stop_codon':   '#666666',
            'CDS':          '#333333',
            'exon':         '#000000',
            # Reds (Dark Red -> Light Red)
            'UTR':          '#FFB2B2',
            'intron':       '#FF6666',
            'gene':         '#CC0000',
            'Intergenic':    '#8B0000',
        }
        
        # Calculate unstacked counts
        gtf_counts = df_combined.groupby(['Cohort', 'gtf_feature']).size().unstack(fill_value=0)
        
        # 2. Enforce the order (only keep categories that actually exist in your data)
        valid_order = [col for col in feature_order if col in gtf_counts.columns]
        gtf_counts = gtf_counts.reindex(columns=valid_order, fill_value=0)
        
        gtf_pct = gtf_counts.div(gtf_counts.sum(axis=1), axis=0) * 100
        
        # 3. Build the color palette based on the valid order
        gtf_colors_mapped = [feature_colors[col] for col in valid_order]
        
        # Plot stacked bar using our custom colors
        gtf_pct.plot(kind='bar', stacked=True, color=gtf_colors_mapped, ax=ax_e, 
                     edgecolor="black", width=element_width, legend=False)
        
        ax_e.set_title('D) GTF Annotations', loc='left', fontweight='bold')
        ax_e.set_ylabel('Proportion of DNVs (%)')
        ax_e.set_xlabel('')
        ax_e.set_xticklabels(ax_e.get_xticklabels(), rotation=0)
        ax_e.spines[['top', 'right']].set_visible(False)
        
        # Custom Legend with integrated counts placed below the plot
        legend_labels_e = []
        for i, col in enumerate(gtf_pct.columns):
            chd_n = gtf_counts.loc['CHD', col] if 'CHD' in gtf_counts.index else 0
            ctrl_n = gtf_counts.loc['Control', col] if 'Control' in gtf_counts.index else 0
            legend_labels_e.append(f"{col} (n={int(chd_n)} vs {int(ctrl_n)})")
            
        handles_e = [plt.Rectangle((0,0),1,1, color=gtf_colors_mapped[i], ec="black") for i in range(len(gtf_pct.columns))]
        
        # Depending on how long this list is, you might want to switch ncol=2 to save vertical space
        ax_e.legend(handles_e, legend_labels_e, title='Feature (CHD vs Ctrl)', 
                    bbox_to_anchor=(0.5, -0.15), loc='upper center', frameon=False, fontsize=9, ncol=1)
        
        contingency_table_e = pd.crosstab(df_combined['Cohort'], df_combined['gtf_feature'])
        if not contingency_table_e.empty:
            _, p_val_e, _, _ = chi2_contingency(contingency_table_e)
            ax_e.text(0.5, 0.95, format_p_value(p_val_e), transform=ax_e.transAxes, 
                      ha='center', va='bottom', fontweight='bold', fontsize=10)
    else:
        ax_e.text(0.5, 0.5, "Missing 'gtf_feature' column", ha='center', va='center')
        ax_e.axis('off')
        
    # ==========================================
    # PANEL J: Coding Effect Distribution (Stacked Bar)
    # ==========================================
    if 'coding_effect' in df_combined.columns:
        # Drop NaNs so we only look at variants that actually have a coding effect
        df_coding = df_combined.dropna(subset=['coding_effect'])
        
        if not df_coding.empty:
            # 1. Calculate unstacked counts
            effect_counts = df_coding.groupby(['Cohort', 'coding_effect']).size().unstack(fill_value=0)
            
            # 2. Identify present effects and sort them strictly by CONSEQUENCE_RANK
            present_effects = list(effect_counts.columns)
            # Default to 999 if an effect isn't in the dict so it goes to the bottom safely
            present_effects = sorted(present_effects, key=lambda x: CONSEQUENCE_RANK.get(x, 999))
            
            # Reorder the dataframe columns to match the rank
            effect_counts = effect_counts[present_effects]
            effect_pct = effect_counts.div(effect_counts.sum(axis=1), axis=0) * 100
            
            # 3. Dynamically assign colors based on IMPACT
            # Count how many of each impact level are present in the data
            impact_counts = {'HIGH': 0, 'MODERATE': 0, 'LOW': 0, 'MODIFIER': 0}
            for eff in present_effects:
                imp = IMPACT.get(eff, 'MODIFIER') # Default unknown to MODIFIER
                impact_counts[imp] += 1
                
            # Generate gradients (using _r to go from dark to light)
            # We request N+1 colors and drop the last one [:-1] so we don't end up with pure white
            palettes = {
                'HIGH': sns.color_palette("Greys_r", impact_counts['HIGH'] + 1)[:-1] if impact_counts['HIGH'] else [],
                'MODERATE': sns.color_palette("Reds_r", impact_counts['MODERATE'] + 1)[:-1] if impact_counts['MODERATE'] else [],
                'LOW': sns.color_palette("Greens_r", impact_counts['LOW'] + 1)[:-1] if impact_counts['LOW'] else [],
                'MODIFIER': sns.color_palette("Blues_r", impact_counts['MODIFIER'] + 1)[:-1] if impact_counts['MODIFIER'] else []
            }
            
            # Map the generated colors to the sorted effects
            color_map = {}
            pal_idx = {'HIGH': 0, 'MODERATE': 0, 'LOW': 0, 'MODIFIER': 0}
            
            for eff in present_effects:
                imp = IMPACT.get(eff, 'MODIFIER')
                color_map[eff] = palettes[imp][pal_idx[imp]]
                pal_idx[imp] += 1
                
            # Create the final ordered list of colors for plotting
            effect_colors = [color_map[eff] for eff in present_effects]
            
            # 4. Plot stacked bar
            effect_pct.plot(kind='bar', stacked=True, color=effect_colors, ax=ax_j, 
                            edgecolor="black", width=element_width, legend=False)
            
            ax_j.set_title('E) Coding Effects', loc='left', fontweight='bold')
            ax_j.set_ylabel('Proportion of Coding DNVs (%)')
            ax_j.set_xlabel('')
            ax_j.set_xticklabels(ax_j.get_xticklabels(), rotation=0)
            ax_j.spines[['top', 'right']].set_visible(False)
            
            # 5. Custom Legend with integrated counts placed below the plot
            legend_labels_j = []
            for i, col in enumerate(present_effects):
                chd_n = effect_counts.loc['CHD', col] if 'CHD' in effect_counts.index else 0
                ctrl_n = effect_counts.loc['Control', col] if 'Control' in effect_counts.index else 0
                legend_labels_j.append(f"{col} (n={int(chd_n)} vs {int(ctrl_n)})")
                
            handles_j = [plt.Rectangle((0,0),1,1, color=effect_colors[i], ec="black") for i in range(len(present_effects))]
            
            # Depending on how many unique coding effects you have, you might want to set ncol=2 here if the list gets too long
            ax_j.legend(handles_j, legend_labels_j, title='Effect (CHD vs Ctrl)', 
                        bbox_to_anchor=(0.5, -0.15), loc='upper center', frameon=False, fontsize=9, ncol=2)
                        
            # 6. Calculate and place P-value
            contingency_table_j = pd.crosstab(df_coding['Cohort'], df_coding['coding_effect'])
            if not contingency_table_j.empty:
                try: 
                    _, p_val_j, _, _ = chi2_contingency(contingency_table_j)
                    ax_j.text(0.5, 0.95, format_p_value(p_val_j), transform=ax_j.transAxes, 
                              ha='center', va='bottom', fontweight='bold', fontsize=10)
                except ValueError:
                    pass 

    else:
        ax_j.text(0.5, 0.5, "Missing 'coding_effect' column", ha='center', va='center')
        ax_j.axis('off')
    # ==========================================
    # PANEL M: Distance to Nearest TSS (Log10 Scale)
    # ==========================================
    if 'closest_TSS' in df_combined.columns:
        import matplotlib.ticker as ticker
        
        # 1. Helper function to format large numbers dynamically
        def human_format(x, pos):
            if x < 1:
                return '0'
            elif x < 1000:
                return f"{int(x)}"
            elif x < 1_000_000:
                return f"{int(x/1000)}k"
            else:
                return f"{x/1_000_000:g}M"
                
        # Drop missing values
        df_tss = df_combined.dropna(subset=['closest_TSS']).copy()
        
        # Add 1 to distances to prevent log10(0) errors for variants that land exactly on a TSS
        df_tss['closest_TSS_offset'] = df_tss['closest_TSS'] + 1
        
        # Plot using Seaborn's built-in log_scale argument 
        sns.histplot(data=df_tss, x='closest_TSS_offset', hue='Cohort', element='step', stat='density',
                     common_norm=False, palette=[color_chd, color_ctrl], ax=ax_m, 
                     log_scale=True, bins=30)
        
        ax_m.set_title('B) Distance to Nearest TSS', loc='left', fontweight='bold')
        ax_m.set_ylabel('Density')
        ax_m.set_xlabel('Distance to TSS (bp)')
        ax_m.spines[['top', 'right']].set_visible(False)
        
        # 2. Apply the dynamic human-readable formatter to the X-axis
        ax_m.xaxis.set_major_formatter(ticker.FuncFormatter(human_format))
        
        # Remove the local legend to rely on the global shared legend
        if ax_m.get_legend():
            ax_m.get_legend().remove()
        
        # Calculate statistics
        dist_chd = df_tss[df_tss['Cohort'] == 'CHD']['closest_TSS']
        dist_ctrl = df_tss[df_tss['Cohort'] == 'Control']['closest_TSS']
        
        if len(dist_chd) > 0 and len(dist_ctrl) > 0:
            _, p_val_tss = mannwhitneyu(dist_chd, dist_ctrl, alternative='two-sided')
            ax_m.text(0.95, 0.95, format_p_value(p_val_tss), transform=ax_m.transAxes, 
                      ha='right', va='top', fontweight='bold', fontsize=10)
    else:
        ax_m.text(0.5, 0.5, "Missing 'closest_TSS' column", ha='center', va='center')
        ax_m.axis('off')

    # ==========================================
    # PANEL F: Epigenetic Signal (Split Violins)
    # ==========================================
    marks = ['DNase_maxZ', 'H3K4me3_maxZ', 'H3K27ac_maxZ', 'CTCF_maxZ']
    
    if all(m in df_combined.columns for m in marks):
        # Filter to cCRE variants and drop rows missing these scores
        df_epigenetic = df_combined[df_combined['cCRE_class'] != 'No CRE'].dropna(subset=marks)
        
        # Melt dataframe to long format for Seaborn's split violin
        df_melted = df_epigenetic.melt(id_vars=['Cohort'], value_vars=marks, 
                                       var_name='Mark', value_name='MaxZ')
        
        # Clean up labels for the X-axis (e.g., 'H3K27ac_maxZ' -> 'H3K27ac')
        df_melted['Mark'] = df_melted['Mark'].str.replace('_maxZ', '')
        
        # Plot split violin
        # Note: 'split=True' requires exactly two hue levels
        sns.violinplot(data=df_melted, x='Mark', y='MaxZ', hue='Cohort', split=True,
                       palette=[color_chd, color_ctrl], ax=ax_f, inner="quartile", 
                       linewidth=1)
        
        ax_f.set_title('I) Epigenetic Signal at DNV cCREs', loc='left', fontweight='bold')
        ax_f.set_ylabel('ENCODE4 maxZ Score')
        ax_f.set_xlabel('')
        ax_f.spines[['top', 'right']].set_visible(False)
        ax_f.get_legend().remove() # Rely on the global master legend
        
        # Calculate and annotate P-values for each epigenetic mark
        y_max = df_melted['MaxZ'].max()
        for i, mark in enumerate(marks):
            chd_vals = df_epigenetic[df_epigenetic['Cohort'] == 'CHD'][mark]
            ctrl_vals = df_epigenetic[df_epigenetic['Cohort'] == 'Control'][mark]
            
            if len(chd_vals) > 0 and len(ctrl_vals) > 0:
                _, p_val = mannwhitneyu(chd_vals, ctrl_vals, alternative='greater')
                
                # Place the text slightly above the maximum value of the violins
                ax_f.text(i, y_max * 1.05, format_p_value(p_val), ha='center', va='bottom', 
                          fontweight='bold', fontsize=9)
        
        # Expand Y-axis slightly to prevent p-values from being cut off
        ax_f.set_ylim(bottom=df_melted['MaxZ'].min() * 1.1, top=y_max * 1.3) 
    else:
        ax_f.text(0.5, 0.5, "Missing one or more maxZ columns", ha='center', va='center')
        ax_f.axis('off')

    # ==========================================
    # PANEL G: DNVs per Proband Distribution
    # ==========================================
    if 'Blinded ID' in df_combined.columns:
        dnvs_per_proband = df_combined.groupby(['Cohort', 'Blinded ID']).size().reset_index(name='DNV_Count')
        
        sns.histplot(data=dnvs_per_proband, x='DNV_Count', hue='Cohort', element='step', stat='density',
                     common_norm=False, palette=[color_chd, color_ctrl], ax=ax_g, bins=25)
        
        ax_g.set_title('A) DNV Burden per Proband', loc='left', fontweight='bold')
        ax_g.set_ylabel('Density')
        ax_g.set_xlabel('DNVs per Proband')
        ax_g.spines[['top', 'right']].set_visible(False)
        ax_g.get_legend().remove()
        
        counts_chd = dnvs_per_proband[dnvs_per_proband['Cohort'] == 'CHD']['DNV_Count']
        counts_ctrl = dnvs_per_proband[dnvs_per_proband['Cohort'] == 'Control']['DNV_Count']
        if len(counts_chd) > 0 and len(counts_ctrl) > 0:
            _, p_val_burden = mannwhitneyu(counts_chd, counts_ctrl, alternative='two-sided')
            ax_g.text(0.95, 0.95, format_p_value(p_val_burden), transform=ax_g.transAxes, 
                      ha='right', va='top', fontweight='bold', fontsize=10)
    else:
        ax_g.text(0.5, 0.5, "Missing 'Blinded ID' column", ha='center', va='center')
        ax_g.axis('off')

    # ==========================================
    # PANEL B: DNVs by Gene List (Filtered Plotting)
    # ==========================================
    # 1. Define the FULL lists for accurate math
    gene_lists_full = ["Chromatin Mod.", "Cilia", "Highly Exp. Heart", "OMIM CHD", 
                       "Mouse CHD", "Multi-Damaging", "Non-implicated"]
    chd_dnvs_full = [215, 123, 1825, 88, 260, 43, 53552]
    ctrl_dnvs_full = [536, 291, 3569, 187, 504, 64, 108914]
    
    # 2. Calculate overall P-value using the FULL distribution
    contingency_table = np.array([chd_dnvs_full, ctrl_dnvs_full])
    _, p_val_overall, _, _ = chi2_contingency(contingency_table)
    
    # 3. Prepare Data and calculate percentages against the FULL totals
    df_genes_full = pd.DataFrame({
        'CHD': chd_dnvs_full,
        'Control': ctrl_dnvs_full
    }, index=gene_lists_full).T 
    
    # Calculate row-wise percentages (using the true 100% cohort sums)
    df_pct_full = df_genes_full.div(df_genes_full.sum(axis=1), axis=0) * 100
    
    # 4. Filter out "Non-implicated" just for the visual plot
    df_pct_plot = df_pct_full.drop(columns=['Non-implicated'])
    gene_lists_plot = gene_lists_full[:-1] # Remove last element
    
    # Plotting on a single axis
    element_width = 0.6
    gene_colors = sns.color_palette("Set3", len(gene_lists_plot))
    
    df_pct_plot.plot(kind='bar', stacked=True, color=gene_colors, ax=ax_b, edgecolor="black", width=element_width, legend=False)
    
    # 5. Styling and Labels
    # Dynamically scale the Y-axis to fit the plotted ~5% data plus some headroom for the legend/title
    max_y = df_pct_plot.sum(axis=1).max()
    ax_b.set_ylim(0, max_y * 1.15)
    ax_b.spines[['top', 'right']].set_visible(False)
       
    # Append the overall P-value to the title
    ax_b.set_title('C) Richter Gene List Annotation', loc='left', fontweight='bold')
    ax_b.set_ylabel('Proportion of Total DNVs (%)')
    ax_b.set_xticklabels(ax_b.get_xticklabels(), rotation=0)
    ax_b.text(0.5, 0.95, format_p_value(p_val_overall), transform=ax_b.transAxes, 
        ha='center', va='bottom', fontweight='bold', fontsize=10)
    
    # 6. Custom Legend with integrated counts
    legend_labels = []
    for i, gene in enumerate(gene_lists_plot):
        label = f"{gene} (n={chd_dnvs_full[i]} vs {ctrl_dnvs_full[i]})"
        legend_labels.append(label)
        
    handles = [plt.Rectangle((0,0),1,1, color=gene_colors[i], ec="black") for i in range(len(gene_lists_plot))]
    
    # Note: ncol=2 perfectly balances the 6 remaining categories into two neat rows
    ax_b.legend(handles, legend_labels, title='Gene List (CHD vs Ctrl)', 
                bbox_to_anchor=(0.5, -0.15), loc='upper center', frameon=False, fontsize=9, ncol=2)
    
    # ==========================================
    # PANEL N: Enhancer Overlaps by Cell Type
    # ==========================================
    # 1. Define explicit colors (Light versions instead of just alpha)
    color_dict = {
        'Proximal_ac':   '#F4A261',  # Original Proximal
        'Proximal_nac':  '#FCD5B5',  # Lighter Proximal
        'Distal_ac':     '#2A9D8F',  # Original Distal
        'Distal_nac':    '#82C2BA'   # Lighter Distal
    }

    # 2. Extract and explode cell types for variants in enhancers
    df_enh = df_combined[df_combined['in_cardioid_enhancer'] == True].copy()
    
    # Split the comma-separated string into a list and explode so each row is one cell type match
    df_enh['celltype_list'] = df_enh['celltypes_in_enhancer'].astype(str).str.split(', ')
    df_exploded = df_enh.explode('celltype_list')
    
    # Clean up any potential whitespace and get unique cell types
    df_exploded['celltype_list'] = df_exploded['celltype_list'].str.strip()
    all_enh_cts = sorted([ct for ct in df_exploded['celltype_list'].unique() if ct != 'nan' and ct != 'None'])

    # --- FILTER CELLTYPES FOR PANEL N ---
    if celltype_filter is not None:
        # Keep only the items that exist in our data, but strictly follow the order of celltype_filter
        unique_enh_cts = [ct for ct in celltype_filter if ct in all_enh_cts]
    else:
        unique_enh_cts = all_enh_cts

    # Ensure cohort_totals exists (fallback if not defined earlier in your script)
    if 'cohort_totals' not in locals():
        cohort_totals = df_combined['Cohort'].value_counts().to_dict()

    # 3. Plotting Setup
    xtick_pos_enh, xtick_label_enh = [], []
    bar_order_enh = ["CHD", "Control"]
    
    x_step = 1.0
    bar_w = 0.35
    max_pct_total = 0 

    for gi, ct in enumerate(unique_enh_cts):
        gc = gi * x_step
        xtick_pos_enh.append(gc)
        xtick_label_enh.append(ct)
        
        # Subset to the current cell type
        df_ct = df_exploded[df_exploded['celltype_list'] == ct]

        for bi, cohort in enumerate(bar_order_enh):
            x = gc + (bi - 0.5) * bar_w
            df_cohort = df_ct[df_ct['Cohort'] == cohort]
            
            n_in = len(df_cohort)
            n_total = cohort_totals.get(cohort, 1) # Prevent division by zero
            
            if n_in > 0:
                # Safely standardize text
                reg_pos = df_cohort['enhancer_regulatory_position'].astype(str).str.lower()
                is_ac = df_cohort['is_cardioid_H3K27ac'] == True
                
                # Calculate counts for the 4 categories
                n_prox_ac  = ((reg_pos == 'proximal') & is_ac).sum()
                n_prox_nac = ((reg_pos == 'proximal') & ~is_ac).sum()
                n_dist_ac  = ((reg_pos == 'distal') & is_ac).sum()
                n_dist_nac = ((reg_pos == 'distal') & ~is_ac).sum()
                
                # Convert to percentages of the FULL cohort
                pct_prox_ac  = 100 * n_prox_ac / n_total
                pct_prox_nac = 100 * n_prox_nac / n_total
                pct_dist_ac  = 100 * n_dist_ac / n_total
                pct_dist_nac = 100 * n_dist_nac / n_total
                
                pct_total = pct_prox_ac + pct_prox_nac + pct_dist_ac + pct_dist_nac
                
                if pct_total > max_pct_total:
                    max_pct_total = pct_total

                # Stack the bars: bottom to top
                current_bottom = 0
                
                if pct_prox_ac > 0:
                    ax_n.bar(x, pct_prox_ac, bar_w, bottom=current_bottom, color=color_dict['Proximal_ac'], edgecolor="black", linewidth=0.5)
                    current_bottom += pct_prox_ac
                    
                if pct_prox_nac > 0:
                    ax_n.bar(x, pct_prox_nac, bar_w, bottom=current_bottom, color=color_dict['Proximal_nac'], edgecolor="black", linewidth=0.5)
                    current_bottom += pct_prox_nac
                    
                if pct_dist_ac > 0:
                    ax_n.bar(x, pct_dist_ac, bar_w, bottom=current_bottom, color=color_dict['Distal_ac'], edgecolor="black", linewidth=0.5)
                    current_bottom += pct_dist_ac
                    
                if pct_dist_nac > 0:
                    ax_n.bar(x, pct_dist_nac, bar_w, bottom=current_bottom, color=color_dict['Distal_nac'], edgecolor="black", linewidth=0.5)
                    current_bottom += pct_dist_nac

                text_col = "#E63946" if cohort == "CHD" else "#2A9D8F"
                prefix = "CHD" if cohort == "CHD" else "Ctrl"
                
                ax_n.text(x, pct_total + (max_pct_total * 0.02 if max_pct_total > 0 else 0.1), f"{prefix} ({n_in})",
                          ha="center", va="bottom", rotation=90, fontsize=8, color=text_col, fontweight="bold")

    # 4. Axis formatting
    ax_n.set_xticks(xtick_pos_enh)
    ax_n.set_xticklabels(xtick_label_enh, rotation=25, ha="right", fontsize=10)
    ax_n.set_ylabel("Variants in Enhancers\n(% of full cohort)")
    ax_n.set_title("L) ABC Candidate Enhancers", loc='left', fontweight="bold")
    
    if max_pct_total > 0:
        ax_n.set_ylim(0, max_pct_total * 1.35)
    ax_n.spines[["top", "right"]].set_visible(False)

    # 5. Legend
    leg_enh_handles = [
        mpatches.Patch(facecolor=color_dict['Proximal_ac'], edgecolor='black', linewidth=0.5, label="Proximal (H3K27ac+)"),
        mpatches.Patch(facecolor=color_dict['Proximal_nac'], edgecolor='black', linewidth=0.5, label="Proximal (H3K27ac-)"),
        mpatches.Patch(facecolor=color_dict['Distal_ac'], edgecolor='black', linewidth=0.5, label="Distal (H3K27ac+)"),
        mpatches.Patch(facecolor=color_dict['Distal_nac'], edgecolor='black', linewidth=0.5, label="Distal (H3K27ac-)")
    ]
    ax_n.legend(handles=leg_enh_handles, loc="upper center", bbox_to_anchor=(0.5, -0.35), 
                frameon=False, fontsize=10, title="Enhancer Annotation", title_fontsize=11, ncol=2)
    
    # ==========================================
    # PANELS K & L: Chromatin Accessibility (from CA_results_dict)
    # ==========================================
    if CA_results_dict:
        # --- FILTER CELLTYPES FOR PANELS K & L ---
        if celltype_filter is not None:
            # Keep order from filter, but only plot if it actually exists in the dictionary
            sorted_cts = [ct for ct in celltype_filter if ct in CA_results_dict]
        else:
            sorted_cts = list(CA_results_dict.keys())
        
        bar_w   = 0.15
        group_w = bar_w * 4 + 0.05
        x_step  = group_w + 0.15
        
        cohort_totals = df_combined.groupby("Cohort").size()
        
        # ------------------------------------------
        # PANEL K: In-peak percentages (Stacked by cCRE status)
        # ------------------------------------------
        xtick_pos1, xtick_label1 = [], []
        bar_order_p1   = [("CHD", "in"), ("Control", "in")]
        bar_colors_p1  = {"CHD": "#E63946", "Control": "#2A9D8F"}

        max_pct_total = 0 

        for gi, ct in enumerate(sorted_cts):
            gc = gi * x_step
            xtick_pos1.append(gc)
            xtick_label1.append(ct)

            for bi, (cohort, kind) in enumerate(bar_order_p1):
                x       = gc + (bi - 0.5) * bar_w
                df_in   = CA_results_dict[ct][cohort][kind]
                
                n_in    = len(df_in)
                n_total = cohort_totals[cohort]
                
                if n_in > 0:
                    n_ccre     = df_in['cCRE_class'].notna().sum()
                    n_non_ccre = df_in['cCRE_class'].isna().sum()
                    
                    pct_ccre     = 100 * n_ccre / n_total
                    pct_non_ccre = 100 * n_non_ccre / n_total
                    pct_total    = pct_ccre + pct_non_ccre
                    
                    if pct_total > max_pct_total:
                        max_pct_total = pct_total
                    
                    col = bar_colors_p1[cohort]

                    if pct_ccre > 0:
                        ax_k.bar(x, pct_ccre, bar_w, color=col, edgecolor="black", linewidth=0.5)
                    
                    if pct_non_ccre > 0:
                        ax_k.bar(x, pct_non_ccre, bar_w, bottom=pct_ccre, color=col, alpha=0.3, edgecolor="black", linewidth=0.5)

                    ax_k.text(x, pct_total + 0.1, f"n={n_in}",
                              ha="center", va="bottom", fontsize=8, color=col, fontweight="bold")

        ax_k.set_xticks(xtick_pos1)
        ax_k.set_xticklabels(xtick_label1, rotation=20, ha="right", fontsize=10)
        ax_k.set_ylabel("Variants in ATAC peaks\n(% of full cohort)")
        ax_k.set_title("J) Cardioid ATAC peak overlap", loc='left', fontweight="bold")
        
        ax_k.set_ylim(0, max_pct_total * 1.15)
        ax_k.spines[["top", "right"]].set_visible(False)

        leg1_handles = [
            mpatches.Patch(facecolor="#E63946", label="CHD (cCRE)"),
            mpatches.Patch(facecolor="#E63946", alpha=0.3, label="CHD (Not cCRE)"),
            mpatches.Patch(facecolor="#2A9D8F", label="Control (cCRE)"),
            mpatches.Patch(facecolor="#2A9D8F", alpha=0.3, label="Control (Not cCRE)")
        ]
        ax_k.legend(handles=leg1_handles, loc="upper center", bbox_to_anchor=(0.5, -0.35), 
                    frameon=False, fontsize=10, title="Cohort & Annotation", title_fontsize=11, ncol=2)

        # ------------------------------------------
        # PANEL L: 100% stacked by CRE type
        # ------------------------------------------
        # Run Stats First
        comparisons = {
            "CHD_in_vs_Ctrl_in":   ("CHD",     "in",  "Control", "in"),
            "CHD_in_vs_CHD_out":   ("CHD",     "in",  "CHD",     "out"),
            "Ctrl_in_vs_Ctrl_out": ("Control", "in",  "Control", "out"),
        }
        
        pvals = {ct: {} for ct in sorted_cts}
        for ct in sorted_cts:
            for label, (c1, k1, c2, k2) in comparisons.items():
                pvals[ct][label] = _ccre_class_chi2_pvalue(CA_results_dict[ct][c1][k1], CA_results_dict[ct][c2][k2])

        sig = {ct: {} for ct in sorted_cts}
        for comp in comparisons:
            raw = [pvals[ct][comp] for ct in sorted_cts]
            valid = [(i, p) for i, p in enumerate(raw) if not np.isnan(p)]
            if valid:
                idxs, ps = zip(*valid)
                _, corrected, _, _ = multipletests(ps, method="fdr_bh")
                for i, p_corr in zip(idxs, corrected):
                    sig[sorted_cts[i]][comp] = p_corr
            for ct in sorted_cts:
                if comp not in sig[ct]:
                    sig[ct][comp] = np.nan

        # Plotting
        xtick_pos2, xtick_label2 = [], []
        bar_tops = {ct: {} for ct in sorted_cts}
        y_base    = 103
        y_bracket = 9 
        bar_order = [("CHD","in"), ("CHD","out"), ("Control","in"), ("Control","out")]
        out_hatch = "///"

        comp_styles = {
            "CHD_in_vs_Ctrl_in":   ("CHD_in",    "Control_in",  0, "#333333"),
            "CHD_in_vs_CHD_out":   ("CHD_in",    "CHD_out",     1, "#E63946"),
            "Ctrl_in_vs_Ctrl_out": ("Control_in","Control_out", 1, "#2A9D8F"),
        }

        for gi, ct in enumerate(sorted_cts):
            gc = gi * x_step
            xtick_pos2.append(gc)
            xtick_label2.append(ct)

            for bi, (cohort, kind) in enumerate(bar_order):
                x     = gc + (bi - 1.5) * bar_w
                df    = CA_results_dict[ct][cohort][kind].dropna(subset=["cCRE_class"])
                total = len(df)
                bar_tops[ct][f"{cohort}_{kind}"] = x

                if total == 0:
                    ax_l.bar(x, 100, bar_w, color="#EEEEEE", edgecolor="black", linewidth=1)
                    continue

                counts = _count_ccre_classes(df)
                pcts   = counts / total * 100
                bottom = 0
                for ci, cre in enumerate(CRE_ORDER):
                    h = pcts[ci]
                    if h > 0:
                        ax_l.bar(x, h, bar_w, bottom=bottom, color=CRE_COLORS[cre],
                                 edgecolor="black", linewidth=1,
                                 hatch=out_hatch if kind == "out" else "")
                        bottom += h

        # Draw Brackets
        for ct in sorted_cts:
            for comp, (bA, bB, yoff, col) in comp_styles.items():
                label = _significance_stars(sig[ct][comp])
                if not label: continue 
                x1, x2 = bar_tops[ct][bA], bar_tops[ct][bB]
                y  = y_base + yoff * y_bracket
                _draw_significance_bracket(ax_l, x1, x2, y, label, col)

        ax_l.set_xticks(xtick_pos2)
        ax_l.set_xticklabels(xtick_label2, rotation=20, ha="right", fontsize=10)
        ax_l.set_ylim(0, y_base + len(comp_styles) * y_bracket + 6)
        ax_l.set_ylabel("% Annotated variants")
        ax_l.set_title("K) cCRE-type distribution in ATAC peaks", loc='left', fontweight="bold")
        ax_l.spines[["top","right"]].set_visible(False)
        ax_l.axhline(100, color="grey", lw=0.5, ls="--", zorder=0)

        # Sub-labels for ax_l
        ylim2 = ax_l.get_ylim()
        yann2 = ylim2[0] - (ylim2[1] - ylim2[0]) * 0.06
        for gc in xtick_pos2:
            ax_l.annotate("CHD",     (gc - bar_w, yann2), ha="center", fontsize=9, color="#C1121F", annotation_clip=False)
            ax_l.annotate("Control", (gc + bar_w, yann2), ha="center", fontsize=9, color="#168AAD", annotation_clip=False)

        # Legend Panel L
        cre_patches = [mpatches.Patch(facecolor=CRE_COLORS[c], label=c) for c in CRE_ORDER]
        style_patches = [
            mpatches.Patch(facecolor="black", edgecolor="black", label="solid = in peaks"),
            mpatches.Patch(facecolor="white", edgecolor="black", hatch="//", label="hatched = outside peaks"),
        ]
        stat_patches = [
            mpatches.Patch(color="#333333", label="CHD-in vs Ctrl-in"),
            mpatches.Patch(color="#E63946", label="CHD-in vs CHD-out"),
            mpatches.Patch(color="#2A9D8F", label="Ctrl-in vs Ctrl-out"),
            plt.Line2D([0], [0], color="none", label="FDR (BH corrected)"),
        ]
        ax_l.legend(handles=cre_patches + style_patches + stat_patches, loc="upper center", bbox_to_anchor=(0.5, -0.35),
                    frameon=False, fontsize=10, title="CRE class / stats", title_fontsize=11, ncol=2)
        
    # # ==========================================
    # # GLOBAL SHARED LEGEND
    # # ==========================================
    # master_handles = [
    #     mpatches.Patch(facecolor=color_chd, label="CHD"),
    #     mpatches.Patch(facecolor=color_ctrl, label="Control")
    # ]
    # fig.legend(handles=master_handles, loc='upper right', bbox_to_anchor=(0.98, 0.98), 
    #            title="Cohort", fontsize=11, title_fontsize=12, frameon=False) 
    
    # ==========================================
    # FINALIZE
    # ==========================================   
    if save_fig:
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {filepath}")
        
    plt.show()


def plot_richter_cohort_qc(df_chd, df_ctrl, save_fig=False, filepath='CHD_Extended_Panel.pdf', celltype_filter=None):
    """Plots the supplementary QC figure of the CHD and control DNV cohorts.

    Three panels: (A) proband filtering (fixed counts from Richter et al. 2020
    plus the three probands excluded for causal coding DNVs), (B) variant types
    (SNV, insertion, deletion, MNV) and (C) transition/transversion ratio with
    the expected WGS value (2.05).

    Args:
        df_chd (pd.DataFrame): CHD proband variants with ``Ref`` and ``Alt`` columns.
        df_ctrl (pd.DataFrame): Control variants with ``Ref`` and ``Alt`` columns.
        save_fig (bool, optional): Whether to save the figure to ``filepath``.
            Defaults to False.
        filepath (str, optional): Output path. Defaults to 'CHD_Extended_Panel.pdf'.
        celltype_filter (list of str, optional): Not used. Defaults to None.
    """
    # Beautiful, high-contrast colors from your preferred style
    color_chd = "#E63946"
    color_ctrl = "#2A9D8F"
    element_width = 0.6 
    
    fig = plt.figure(figsize=(15, 4))
    
    # 4 rows, 3 columns. Top two rows for standard plots, bottom two for the split-axis bar plot
    gs = gridspec.GridSpec(1, 3, wspace=0.4, hspace=0.4)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_h = fig.add_subplot(gs[0, 1])
    ax_i = fig.add_subplot(gs[0, 2])
    
    # ==========================================
    # DATA PREP
    # ==========================================
    df_chd = df_chd.copy()
    df_ctrl = df_ctrl.copy()
    df_chd['Cohort'] = 'CHD'
    df_ctrl['Cohort'] = 'Control'
    df_combined = pd.concat([df_chd, df_ctrl], ignore_index=True)

    # ==========================================
    # PANEL A: CHD Probands (Filtering)
    # ==========================================
    total_chd, n_noncausal, n_control = 763, 749, 1611
    n_causal = total_chd - n_noncausal + 3 # the three we filtered out
    labels = ['CHD removed\nCausal Found', 'CHD Retained\nNon-Coding', 'Control Retained\nNon-Coding']
    ax_a.bar(labels, 
             [np.log10(n_causal), np.log10(n_noncausal), np.log10(n_control)], 
             color=['#333333', color_chd, color_ctrl], 
             width=element_width, edgecolor="black")

    ax_a.set_xticks(range(len(labels)))
    ax_a.set_xticklabels(labels, rotation=30, ha='right', rotation_mode='anchor')
    ax_a.set_ylabel('Log10 Number of Probands')
    ax_a.set_title('A) CHD Proband Filtering', loc='left', fontweight='bold')
    ax_a.spines[['top', 'right']].set_visible(False)

    for i, v in enumerate([n_causal, n_noncausal, n_control]):
        ax_a.text(i, np.log10(v) + 0.05, str(v), ha='center', va='bottom', fontweight='bold', fontsize=10)
        
   # ==========================================
    # PANEL H: Variant Type Distribution (Percentages)
    # ==========================================
    if 'Ref' in df_combined.columns and 'Alt' in df_combined.columns:
        import matplotlib.ticker as ticker
        
        # Fast vectorized variant type classification based on string length
        ref_len = df_combined['Ref'].astype(str).str.len()
        alt_len = df_combined['Alt'].astype(str).str.len()
        
        conditions = [
            (ref_len == 1) & (alt_len == 1),       # SNV
            (ref_len < alt_len),                   # Insertion
            (ref_len > alt_len),                   # Deletion
            (ref_len == alt_len) & (ref_len > 1)   # MNV
        ]
        choices = ['SNV', 'Insertion', 'Deletion', 'MNV']
        df_combined['Variant_Type'] = np.select(conditions, choices, default='Other')
        
        # Filter out 'Other' (usually missing data or complex structural variants)
        df_types = df_combined[df_combined['Variant_Type'] != 'Other']
        
        # Calculate percentages within each cohort
        type_counts = df_types.groupby(['Cohort', 'Variant_Type']).size().reset_index(name='Count')
        cohort_totals = df_types.groupby('Cohort').size()
        
        type_counts['Percentage'] = type_counts.apply(
            lambda row: (row['Count'] / cohort_totals[row['Cohort']]) * 100, axis=1)
        
        # Plot grouped barplot using percentages
        sns.barplot(data=type_counts, x='Variant_Type', y='Percentage', hue='Cohort', 
                    palette=[color_chd, color_ctrl], ax=ax_h, edgecolor="black", 
                    order=['SNV', 'Insertion', 'Deletion', 'MNV'])
        
        ax_h.set_title('B) Variant Types', loc='left', fontweight='bold')
        ax_h.set_ylabel('Proportion of DNVs (%)')
        ax_h.set_xlabel('')
        ax_h.spines[['top', 'right']].set_visible(False)
        ax_h.get_legend().remove()
        ax_h.set_xticks(ax_h.get_xticks()) # Grabs the tick locations Seaborn just made
        ax_h.set_xticklabels(ax_h.get_xticklabels(), rotation=30, ha='right', rotation_mode='anchor')
        
        # Use log scale so Indels/MNVs are clearly visible next to SNVs
        ax_h.set_yscale('log')
        
        # Format log ticks to show clean numbers (e.g., 0.1, 1, 10, 100)
        ax_h.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: '{:g}'.format(y)))
        if 'Variant_Type' in locals().get('df_types', pd.DataFrame()).columns:
            contingency_table_h = pd.crosstab(df_types['Cohort'], df_types['Variant_Type'])
            if not contingency_table_h.empty:
                _, p_val_h, _, _ = chi2_contingency(contingency_table_h)
                ax_h.text(0.5, 0.85, format_p_value(p_val_h), transform=ax_h.transAxes, 
                          ha='center', va='bottom', fontweight='bold', fontsize=10)
    else:
        ax_h.text(0.5, 0.5, "Missing 'Ref' or 'Alt' columns", ha='center', va='center')
        ax_h.axis('off')
        
    # ==========================================
    # PANEL I: Ti/Tv Ratio (QC Metric)
    # ==========================================
    if 'Ref' in df_combined.columns and 'Alt' in df_combined.columns:
        # Filter strictly for SNVs
        is_snv = (df_combined['Ref'].astype(str).str.len() == 1) & \
                 (df_combined['Alt'].astype(str).str.len() == 1)
        df_snvs = df_combined[is_snv].copy()
        
        # Define Transitions (Purine <-> Purine or Pyrimidine <-> Pyrimidine)
        transitions = [{'A', 'G'}, {'C', 'T'}]
        
        # Classify as Ti or Tv
        def classify_titv(row):
            mut_set = {row['Ref'].upper(), row['Alt'].upper()}
            if mut_set in transitions:
                return 'Ti'
            return 'Tv'
            
        df_snvs['TiTv'] = df_snvs.apply(classify_titv, axis=1)
        
        # Calculate the ratio per cohort
        titv_counts = df_snvs.groupby(['Cohort', 'TiTv']).size().unstack(fill_value=0)
        titv_counts['Ratio'] = titv_counts['Ti'] / titv_counts['Tv']
        titv_ratio_df = titv_counts.reset_index()
        
        contingency_table = titv_counts[['Ti', 'Tv']].values       
        _, p_val, _, _ = chi2_contingency(contingency_table)
        
        # Plot the ratio as a bar plot
        sns.barplot(data=titv_ratio_df, x='Cohort', y='Ratio', 
                    palette=[color_chd, color_ctrl], ax=ax_i, edgecolor="black", width=element_width)
        
        ax_i.set_title('C) Ti/Tv Ratio', loc='left', fontweight='bold')
        ax_i.set_ylabel('Ti/Tv Ratio')
        ax_i.set_xlabel('')
        ax_i.spines[['top', 'right']].set_visible(False)
        ax_i.text(0.5, 0.85, format_p_value(p_val), transform=ax_i.transAxes, 
                          ha='center', va='bottom', fontweight='bold', fontsize=10)
        
        # Add a baseline indicating the expected WGS ratio (~2.0 - 2.1)
        ax_i.axhline(2.05, color='gray', linestyle='--', linewidth=1, zorder=0)
        ax_i.text(0.5, 2.05, ' Expected WGS', va='bottom', ha='center', color='gray', fontsize=8)
        
        # Annotate the exact ratio values on top of the bars
        for i, row in titv_ratio_df.iterrows():
            ax_i.text(i, row['Ratio'] + 0.05, f"{row['Ratio']:.2f}", 
                      ha='center', va='bottom', fontweight='bold', fontsize=10)
            
        # Ensure the Y-axis has a bit of headroom for the text
        ax_i.set_ylim(0, titv_ratio_df['Ratio'].max() * 1.25)
    else:
        ax_i.text(0.5, 0.5, "Missing 'Ref'/'Alt' columns", ha='center', va='center')
        ax_i.axis('off')
       
    # ==========================================
    # GLOBAL SHARED LEGEND
    # ==========================================
    master_handles = [
        mpatches.Patch(facecolor=color_chd, label="CHD"),
        mpatches.Patch(facecolor=color_ctrl, label="Control")
    ]
    fig.legend(handles=master_handles, loc='upper right', bbox_to_anchor=(0.98, 0.98), 
               title="Cohort", fontsize=11, title_fontsize=12, frameon=False) 
    
    # ==========================================
    # FINALIZE
    # ==========================================   
    if save_fig:
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {filepath}")
        
    plt.show()
