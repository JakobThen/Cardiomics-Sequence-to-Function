"""Variant Table Processing
========================

Converts SDR-seq differential-expression (DE) results into VCF-style variant
tables and prepares variant effect prediction (VEP) scores for analysis:

* `DE_to_VCF` (with `process_row`, `expand_dataframe`, `expand_indels` and
  `make_VCF_cols`): phased variant combinations -> one VCF-style row per
  haplotype.
* `pivot_tf_scores`, `add_abs_score_columns`, `format_chrombpnet_vep`: reshape
  model scores into one column per score.
* `create_tss_stratifications`: variant strata by distance to the target gene TSS.
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any

def process_row(phases: List[int], variants: List[str]) -> List[Tuple[List[int], List[str]]]:
    """Splits one combination of phased variants by haplotype.

    The sign of a phase value gives the haplotype: positive values are tagged
    ``(for)``, all others ``(rev)``, by appending ``-(for)`` / ``-(rev)`` to the
    variant ID.

    Args:
        phases (list of int): Signed phase value per variant.
        variants (list of str): Variant IDs, in the same order as ``phases``.

    Returns:
        list of tuple: ``(phases, variant_ids)`` pairs.

        - All signs equal: one pair with every variant tagged.
        - Mixed signs: three pairs: all variants (each tagged by its own sign),
          then the positive-phase variants and then the negative-phase variants.
          In the last two, the variants of the other haplotype are kept with an
          extra ``-phased`` suffix so that `make_VCF_cols` can drop them.
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
    """Applies `process_row` to every row of a variant table.

    Rows whose phases mix both signs are split into several rows (see
    `process_row`); all other columns are copied to each resulting row.

    Args:
        df (pd.DataFrame): Table with one list of phases and one list of variant IDs per row.
        phase_col (str, optional): Column holding the phase lists. Defaults to ``"phase"``.
        var_col (str, optional): Column holding the variant ID lists. Defaults to ``"variant_id"``.

    Returns:
        pd.DataFrame: The expanded table with a fresh index.

    Raises:
        KeyError: If ``phase_col`` or ``var_col`` is not a column of ``df``.
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
    """Expands one variant into one row per affected base.

    Multi-base alleles are split into single bases at consecutive positions
    starting at ``POS``; the shorter allele is padded with ``None`` (e.g. the
    deleted bases of a deletion have ``ALT=None``).

    Args:
        row (pd.Series): Variant with ``CHROM``, ``POS`` (int), ``REF``, ``ALT`` and ``STRAND``.

    Returns:
        pd.DataFrame: ``max(len(REF), len(ALT))`` rows with the columns
        ``CHROM, POS, REF, ALT, STRAND``.
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
    """Merges the variants of each row into one VCF-style allele.

    Variant IDs are parsed as ``CHROM-POS-REF-ALT-STRAND``; IDs tagged
    ``-phased`` (other haplotype, see `process_row`) are ignored. The remaining
    variants of a row are expanded base by base (`expand_indels`), gaps between
    them are filled with the reference base (``REF == ALT``), and the bases are
    concatenated into a single ``REF``/``ALT`` pair starting at the first variant.
    ``STRAND`` is ``(for)`` or ``(rev)`` if all variants agree, else ``(.)``.

    Note:
        ``df[variant_id_col]`` is modified in place: its lists are replaced by
        ``"|"``-joined strings.

    Args:
        df (pd.DataFrame): Table with one list of variant IDs per row.
        variant_id_col (str): Column holding the variant ID lists.
        hg38: Reference genome indexable as ``hg38[chrom][pos - 1]`` (0-based),
            e.g. a ``pyfaidx.Fasta`` or a dict of sequences.

    Returns:
        pd.DataFrame: ``df`` (fresh index) with the added columns
        ``CHROM, POS, REF, ALT, STRAND``, one row per input row.
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
    """Converts SDR-seq differential-expression results into a VCF-style table.

    Reads the variants carried by each tested genotype from ``ID_Geno_Var1``
    (``"|"``-separated phase values; ``"0"`` = not carried) and
    ``ID_Geno_conc_Var1`` (``"|"``-separated variant IDs), splits mixed phases
    into haplotypes (`expand_dataframe`) and merges each haplotype's variants into
    one allele (`make_VCF_cols`), ready for variant effect prediction.

    Args:
        DE_df (pd.DataFrame): DE results with the columns ``ID_Geno_Var1`` and
            ``ID_Geno_conc_Var1``. Use a default ``RangeIndex`` when ``col_to_keep``
            is None, as the columns are then joined by index.
        hg38: Reference genome for gap filling, see `make_VCF_cols`.
        col_to_keep (list of str, optional): DE columns to carry over. Defaults
            to None (all columns).

    Returns:
        pd.DataFrame: One row per haplotype with ``phase``, ``variant_id``
        (``"|"``-joined), the kept DE columns and ``CHROM, POS, REF, ALT, STRAND``.
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
    """Pivots transcription factor (TF) scores from long to wide format.

    All columns other than ``transcription_factor``, ``TF_raw_score`` and
    ``TF_quantile_score`` identify a row. Each TF becomes two columns,
    ``<TF>_TF_raw_score`` and ``<TF>_TF_quantile_score`` (e.g.
    ``CTCF_TF_raw_score``); duplicates keep their first value.

    Note:
        Rows with missing values in an identifying column are dropped
        (``pandas.pivot_table`` default).

    Args:
        df (pd.DataFrame): Long table with ``transcription_factor``,
            ``TF_raw_score`` and ``TF_quantile_score`` columns.

    Returns:
        pd.DataFrame: Wide table with the identifying columns and two score
        columns per TF.
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
    """Adds the absolute value of every score column.

    For each column whose name contains ``raw_score`` or ``quantile_score``
    (and not ``abs``), an ``abs_<column>`` column is added.

    Args:
        df (pd.DataFrame): Table with score columns.

    Returns:
        pd.DataFrame: A copy of ``df`` with the added ``abs_*`` columns.
    """
    new_df = df.copy()
    for col in df.columns:
        if ("raw_score" in col or "quantile_score" in col) and "abs" not in col:
            new_col = f"abs_{col}"
            new_df[new_col] = new_df[col].abs()
    return new_df

def format_chrombpnet_vep(vep_df, keep_abs=True):
    """Renames a fold-averaged chromBPNet variant-scorer table to the VEP column scheme.

    Keeps ``variant_id`` and the mean scores across folds, removes duplicate
    rows and renames the columns: ``chr, pos, allele1, allele2`` ->
    ``CHROM, POS, REF, ALT``; ``logfc.mean`` -> ``COUNTS_cPBN_raw_score``;
    ``jsd.mean`` -> ``JSD_cPBN_raw_score``; ``logfc_x_jsd.mean`` ->
    ``COUNTSxJSD_cPBN_raw_score``; each with its ``abs_`` variant and ``*_pval``.

    Args:
        vep_df (pd.DataFrame): Fold-mean variant-scorer output (e.g. read from its TSV).
        keep_abs (bool, optional): Keep the ``abs_*`` columns. Defaults to True.

    Returns:
        pd.DataFrame: The renamed score table.
    """
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
    """Splits variants into strata by distance to their target gene TSS.

    Cutoffs are the ``n``-quantiles of the absolute distance, rounded to readable
    kb values (1 kb steps below 10 kb, up to 100 kb steps above 500 kb). Strata
    are right-closed: ``<= c1``, ``c1-c2`` (= (c1, c2]) and ``> cn``.

    Args:
        df (pd.DataFrame): Variant table.
        tss_col (str, optional): Column with the signed distance (bp) to the target
            gene TSS. Defaults to ``"dist_target_gene_TSS"``.
        n (int, optional): Number of quantiles used for the cutoffs. Defaults to 4.

    Returns:
        dict: Stratum name -> selector for ``df``. ``"All Vars"`` maps to ``df.index``,
        the other strata (e.g. ``"TSS Distance <= 20kb"``) to boolean masks. If no
        positive cutoff can be derived (e.g. all distances are 0), only
        ``"All Vars"`` is returned.
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