import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any

def process_row(phases: List[int], variants: List[str]) -> List[Tuple[List[int], List[str]]]:
    """
    Processes a single row's phasing and variant information, resolving phase signs
    and formatting the variant strings accordingly (forward or reverse).
    
    Parameters:
    -----------
    phases : list of int
        A list of phase values (positive or negative integers).
    variants : list of str
        A list of variant string identifiers.
        
    Returns:
    --------
    list of tuples
        A list of (phases, new_variants) tuples. Returns one tuple if all signs 
        are the same, or two tuples (one for positive, one for negative phases) 
        if signs are mixed.
    """
    phases_sign = [1 if p > 0 else -1 for p in phases]
    unique_signs = set(phases_sign)
    
    result_rows = []
    
    if len(unique_signs) == 1:
        # All same sign
        sign_str = "(for)" if list(unique_signs)[0] > 0 else "(rev)"
        new_variants = [f"{v}-{sign_str}" for v in variants]
        result_rows.append((phases, new_variants))
    else:
        # Mixed signs
        # Append sign to each variant
        new_variants = [f"{v}-({'for' if p>0 else 'rev'})" for v, p in zip(variants, phases)]
        result_rows.append((phases, new_variants))
        
        # Split into two rows: positives and negatives
        pos_phases = [p for p in phases if p > 0]
        pos_vars = [f"{v}-(for)" for v, p in zip(variants, phases) if p > 0]
        
        neg_phases = [p for p in phases if p < 0]
        neg_vars = [f"{v}-(rev)" for v, p in zip(variants, phases) if p < 0]
        
        pos_vars_concat = pos_vars + [f"{v}-phased" for v in neg_vars]
        neg_vars_concat = neg_vars + [f"{v}-phased" for v in pos_vars]
        
        if pos_phases:
            result_rows.append((pos_phases, pos_vars_concat))
        if neg_phases:
            result_rows.append((neg_phases, neg_vars_concat))
    
    return result_rows

def expand_dataframe(df: pd.DataFrame, phase_col: str = "phase", var_col: str = "variant_id") -> pd.DataFrame:
    """
    Expands a dataframe by unnesting and processing the phasing and variant lists
    using `process_row`.
    
    Parameters:
    -----------
    df : pd.DataFrame
        The input dataframe containing phase and variant lists.
    phase_col : str, default="phase"
        The column name containing lists of phases.
    var_col : str, default="variant_id"
        The column name containing lists of variant IDs.
        
    Returns:
    --------
    pd.DataFrame
        A new expanded dataframe where phased rows are unnested.
    """
    out_rows = []
    cols = list(df.columns)
    
    if phase_col not in cols or var_col not in cols:
        raise KeyError(f"'{phase_col}' and '{var_col}' must be columns of the DataFrame")

    other_cols = [c for c in cols if c not in (phase_col, var_col)]

    for row in df.itertuples(index=False, name=None):
        row_dict = dict(zip(cols, row))
        phases = list(row_dict[phase_col])       
        variants = list(row_dict[var_col])

        expanded = process_row(phases, variants) 

        for phases_new, variants_new in expanded:
            new_row = {c: row_dict[c] for c in other_cols}
            new_row[phase_col] = phases_new
            new_row[var_col] = variants_new
            out_rows.append(new_row)

    return pd.DataFrame(out_rows)

def expand_indels(row: pd.Series) -> pd.DataFrame:
    """
    Expands insertions/deletions (indels) to cover the maximum length of the 
    REF or ALT alleles, incrementing the POS index accordingly.
    
    Parameters:
    -----------
    row : pd.Series
        A single row containing at least 'CHROM', 'POS', 'REF', 'ALT', and 'STRAND'.
        
    Returns:
    --------
    pd.DataFrame
        A dataframe of expanded positional rows for the indel.
    """
    ref = row["REF"]
    alt = row["ALT"]
    pos = row["POS"]
    
    # Find max length
    max_len = max(len(ref), len(alt))
    
    # Expand REF and ALT, pad with None if shorter
    ref_exp = list(ref) + [None] * (max_len - len(ref))
    alt_exp = list(alt) + [None] * (max_len - len(alt))
    
    # Expand POS: increment by index for multi-base
    pos_exp = [pos + i for i in range(max_len)]
    
    expanded_rows = pd.DataFrame({
        "CHROM": [row["CHROM"]] * max_len,
        "POS": pos_exp,
        "REF": ref_exp,
        "ALT": alt_exp,
        "STRAND": [row["STRAND"]] * max_len
    })
    
    return expanded_rows

def make_VCF_cols(df: pd.DataFrame, variant_id_col: str, hg38: Any) -> pd.DataFrame:
    """
    Parses variant IDs to reconstruct VCF-style columns (CHROM, POS, REF, ALT, STRAND),
    filling in missing reference bases using a provided genome object.
    
    Parameters:
    -----------
    df : pd.DataFrame
        The dataframe containing the variants to be processed.
    variant_id_col : str
        The column containing lists of variant strings.
    hg38 : object/dict
        A genome dictionary or sequence object (e.g., from pyfaidx) used to 
        fetch missing reference bases by doing `hg38[chrom][pos-1]`.
        
    Returns:
    --------
    pd.DataFrame
        The original dataframe horizontally concatenated with new VCF columns.
    """
    variant_ids = df[variant_id_col].copy()
    VCF_cols = []

    for var_list in variant_ids:
        # Removes all the phased variants on the other allele
        var_list = [var for var in var_list if "-phased" not in var] 
        parts = []
        for var in var_list:               
            part = var.split("-")[:5]
            parts.append(part)
            
        parts = pd.DataFrame(parts, columns=["CHROM", "POS", "REF", "ALT", "STRAND"]).sort_values(by="POS", ascending=True)    
        parts.POS = parts.POS.astype("int")

        chrom = parts.CHROM[0]
        min_pos = parts.POS.min()
        max_pos = parts.POS.max()

        extend_pos = pd.concat([expand_indels(row) for _, row in parts.iterrows()], ignore_index=True)
        missing_pos = []
        
        for pos in range(min_pos, max_pos + 1):
            if pos not in extend_pos.POS.values:
                missing_pos.append(pd.Series({
                    "CHROM": chrom,
                    "POS": pos,
                    "REF": str(hg38[chrom][pos - 1]).upper(),
                    "ALT": str(hg38[chrom][pos - 1]).upper(),
                    "STRAND": "(.)"
                }))
                
        if missing_pos:
            extend_pos = pd.concat([extend_pos, pd.DataFrame(missing_pos)]).sort_values(by="POS", ascending=True)  

        REF = extend_pos.REF.str.cat()
        ALT = extend_pos.ALT.str.cat()

        ret = extend_pos.iloc[0, :].copy()
        ret["REF"] = REF
        ret["ALT"] = ALT

        if ((extend_pos.STRAND == "(for)") | (extend_pos.STRAND == "(.)")).all():
            ret["STRAND"] = "(for)"
        elif ((extend_pos.STRAND == "(rev)") | (extend_pos.STRAND == "(.)")).all():
            ret["STRAND"] = "(rev)"
        else:
            ret["STRAND"] = "(.)"

        VCF_cols.append(ret)
    
    df[variant_id_col] = variant_ids.str.join("|")

    return pd.concat([df.reset_index(drop=True), pd.DataFrame(VCF_cols).reset_index(drop=True)], axis=1)

def DE_to_VCF(DE_df: pd.DataFrame, hg38: Any, col_to_keep: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Turns a Differential Expression (DE) output from SDR-seq into a VCF style file 
    with additional columns for Variant Effect Prediction (VEP).
    
    Parameters:
    -----------
    DE_df : pd.DataFrame
        The differential expression dataframe to process.
    hg38 : object/dict
        A genome object passed down to `make_VCF_cols` to fill in missing reference bases.
    col_to_keep : list of str, optional
        List of column names belonging to DE_df to keep in the final VCF style file.
        If None, all columns are kept.
        
    Returns:
    --------
    pd.DataFrame
        A formatted VCF-style dataframe with expanded variant info.
    """
    # Extract variant ids with phasing info
    phasing = [
        [
            (int(phase), var)
            for phase, var in zip(phase_geno, var_geno)
            if phase != "0"
        ]
        for phase_geno, var_geno in zip(
            DE_df.ID_Geno_Var1.str.split("|"),
            DE_df.ID_Geno_conc_Var1.str.split("|")
        )
    ]

    phasing = [
        (
            [phase for phase, var in row],
            [var for phase, var in row]
        )
        for row in phasing
    ]
    
    variants = pd.DataFrame(phasing, columns=["phase", "variant_id"])
    
    if col_to_keep is not None:   
        variants = pd.concat([
            variants.reset_index(drop=True),
            DE_df.loc[:, col_to_keep].reset_index(drop=True)
        ], axis=1)
    else:  
        variants = pd.concat([variants, DE_df], axis=1) 
    
    variants = expand_dataframe(variants, phase_col="phase", var_col="variant_id")
    vcf = make_VCF_cols(variants, variant_id_col="variant_id", hg38=hg38)
        
    return vcf

def pivot_tf_scores(df: pd.DataFrame) -> pd.DataFrame:
    """
    Pivots transcription factor (TF) scores from a long format into a wide format,
    creating unique columns for each TF (e.g., 'CTCF_TF_raw_score').
    
    Parameters:
    -----------
    df : pd.DataFrame
        The input dataframe in long format containing 'transcription_factor', 
        'TF_raw_score', and 'TF_quantile_score'.
        
    Returns:
    --------
    pd.DataFrame
        A wide-format dataframe with prefixed TF columns.
    """
    tf_columns = ['transcription_factor', 'TF_raw_score', 'TF_quantile_score']
    index_cols = [col for col in df.columns if col not in tf_columns]
    
    # Pivot the DataFrame using pivot_table with aggfunc='first'
    df_wide = df.pivot_table(
        index=index_cols,
        columns='transcription_factor',
        values=['TF_raw_score', 'TF_quantile_score'],
        aggfunc='first'
    )
     
    df_wide.columns = [f"{tf}_{measure}" for measure, tf in df_wide.columns]
    df_wide = df_wide.reset_index()
    
    return df_wide

def add_abs_score_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Finds all columns containing 'raw_score' or 'quantile_score',
    creates new columns with their absolute values prefixed by ``abs_``, 
    and returns the updated DataFrame.
    
    Parameters:
    -----------
    df : pd.DataFrame
        The input dataframe containing raw and quantile score columns.
        
    Returns:
    --------
    pd.DataFrame
        The updated dataframe including the new absolute value columns.
    """
    new_df = df.copy()
    for col in df.columns:
        if ("raw_score" in col or "quantile_score" in col) and "abs" not in col:
            new_col = f"abs_{col}"
            new_df[new_col] = new_df[col].abs()
    return new_df

#takes a fold mean chromBPNet VEP tsv files as pandas and refroamts into better columns
def format_chrombpnet_vep(vep_df, keep_abs=True):
    col_names = {
        "chr": "CHROM", "pos": "POS", "allele1": "REF", "allele2": "ALT",
        "logfc.mean": "COUNTS_cPBN_raw_score",
        "abs_logfc.mean": "abs_COUNTS_cPBN_raw_score",
        "logfc.mean.pval": "COUNTS_cPBN_pval",
        "abs_logfc.mean.pval": "abs_COUNTS_cPBN_pval",
        "jsd.mean": "JSD_cPBN_raw_score",
        "jsd.mean.pval": "JSD_cPBN_pval",
        "logfc_x_jsd.mean": "COUNTSxJSD_cPBN_raw_score",
        "abs_logfc_x_jsd.mean": "abs_COUNTSxJSD_cPBN_raw_score",
        "logfc_x_jsd.mean.pval": "COUNTSxJSD_pval",
        "abs_logfc_x_jsd.mean.pval": "abs_COUNTSxJSD_pval",
    }

    vep_df = vep_df[["variant_id"] + list(col_names.keys())].drop_duplicates()
    vep_df = vep_df.rename(columns=col_names)
    
    if not keep_abs:
        vep_df = vep_df.loc[:, ~vep_df.columns.str.contains("abs")]

    return vep_df

def create_tss_stratifications(df: pd.DataFrame, tss_col: str = "dist_target_gene_TSS", n: int = 4) -> Dict[str, Any]:
    """
    Splits variants into TSS-distance strata. Computes n quantile-based intervals
    on the absolute TSS distance, rounds the cutoffs to readable kb values and
    returns a dictionary of masks (e.g. for `plot_scatter_panel` or
    `plot_stratified_auprc_heatmap`).

    Parameters:
    -----------
    df : pd.DataFrame
        The dataframe containing the variants.
    tss_col : str, default="dist_target_gene_TSS"
        Column holding the signed distance (bp) of each variant to its target gene TSS.
    n : int, default=4
        Number of quantiles used to derive the cutoffs.

    Returns:
    --------
    dict
        Mapping of stratum name to selector. "All Vars" maps to ``df.index``,
        all other strata map to boolean masks over ``df``. Strata are
        right-closed intervals (``<= c1``, ``c1-c2`` = (c1, c2], ``> cn``).
        If no positive cutoff can be derived (e.g. all distances are 0),
        only "All Vars" is returned.
    """
    abs_dist = df[tss_col].abs()

    quantiles = np.linspace(0, 1, n + 1)[1:-1]
    raw_cutoffs = (
        abs_dist.quantile(quantiles)
        .dropna()
        .tolist()
    )
    def round_to_nice_kb(bp_val):
        kb = bp_val / 1000
        if kb < 10:
            return int(round(kb / 1) * 1)        # Round to nearest 1kb
        elif kb < 50:
            return int(round(kb / 5) * 5)        # Round to nearest 5kb
        elif kb < 100:
            return int(round(kb / 10) * 10)      # Round to nearest 10kb
        elif kb < 200:
            return int(round(kb / 20) * 20)      # Round to nearest 20kb
        elif kb < 500:
            return int(round(kb / 50) * 50)      # Round to nearest 50kb
        else:
            return int(round(kb / 100) * 100)    # Round to nearest 100kb

    # Apply rounding and ensure cutoffs are unique and strictly positive
    # (a cutoff that rounds down to 0kb would give an empty/degenerate stratum)
    nice_cutoffs_kb = sorted(set(c for c in (round_to_nice_kb(x) for x in raw_cutoffs) if c > 0))
    stratifications = {
        "All Vars": df.index
    }

    # No usable cutoff (e.g. all variants at TSS distance 0): only return "All Vars"
    if not nice_cutoffs_kb:
        return stratifications

    # Intervals are (lower, upper], so the first one includes its upper cutoff
    c_first = nice_cutoffs_kb[0]
    stratifications[f"TSS Distance <= {c_first}kb"] = (abs_dist <= c_first * 1000)

    # Middle intervals
    for i in range(len(nice_cutoffs_kb) - 1):
        lower_kb = nice_cutoffs_kb[i]
        upper_kb = nice_cutoffs_kb[i+1]

        mask = (abs_dist > lower_kb * 1000) & (abs_dist <= upper_kb * 1000)
        stratifications[f"TSS Distance {lower_kb}kb-{upper_kb}kb"] = mask

    # Last interval
    c_last = nice_cutoffs_kb[-1]
    stratifications[f"TSS Distance > {c_last}kb"] = (abs_dist > c_last * 1000)

    return stratifications