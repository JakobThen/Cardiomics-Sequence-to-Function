"""Variant Table Processing
========================

Converts SDR-seq differential-expression (DE) results into VCF-style variant
tables and prepares variant effect prediction (VEP) scores for analysis:

* `DE_to_VCF` (with `process_row`, `expand_dataframe`, `expand_indels` and
  `make_VCF_cols`): phased variant combinations -> one VCF-style row per
  haplotype.
* `pivot_tf_scores`, `add_abs_score_columns`, `format_chrombpnet_vep`: reshape
  model scores into one column per score.
* `strip_phasing_suffix`: haplotype suffixes -> one ID per variant.
* `format_alphagenome_ft_vep_per_gene`, `format_borzoi_vep_per_gene`: long VEP
  tables of the fine-tuned models -> one row per variant, gene and cell type
  (cell-type-resolved effects, SDR).
* `format_alphagenome_ft_vep_top_gene`, `format_pretrained_alphagenome_vep_top_gene`:
  AlphaGenome VEP tables -> one row per variant with the largest-effect gene
  (no cell-type-resolved effects, Richter).
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

def strip_phasing_suffix(variant_ids: pd.Series) -> pd.Series:
    """Removes the phasing suffixes that `process_row` appends to variant IDs.

    ``-(for)``, ``-(rev)`` and ``-phased`` are removed, so that the scores of both
    haplotypes of a variant share one ID (e.g. to average them).

    Args:
        variant_ids (pd.Series): Variant IDs, e.g. ``chr1-100-A-G-(for)``.

    Returns:
        pd.Series: The IDs without suffix, e.g. ``chr1-100-A-G``.
    """
    return (
        variant_ids
        .str.replace("-(for)", "", regex=False)
        .str.replace("-(rev)", "", regex=False)
        .str.replace("-phased", "", regex=False)
    )

def _mean_alphagenome_ft_cutntag_days(vep: pd.DataFrame) -> pd.DataFrame:
    """Averages the fine-tuned AlphaGenome Cut&Tag scores over days, genes and haplotypes.

    Takes the ``CutnTag`` rows of a long fine-tuned AlphaGenome VEP table, removes
    the day from the track names, harmonises the mark names (``K4``,
    ``cutntag_4`` -> ``H3K4me3``; ``K27ac`` -> ``H3K27ac``) and returns one row per
    variant with one score column per mark.
    """
    #combine the cnt days by taking the mean
    cnt = vep[vep.output_type == "CutnTag"].copy()
    cnt.loc[:,"track_name"] = cnt["track_name"].str.replace(r"Day\d+", "", regex=True)
    cnt.loc[:,"track_name"] = (
        cnt["track_name"]
        .str.replace("K4", "H3K4me3", regex=False)
        .str.replace("cutntag_4", "H3K4me3", regex=False)
        .str.replace("K27ac", "H3K27ac", regex=False)
    )
    cnt = (
        cnt.groupby(["variant_id", "track_name", "output_type", "gene_name"], dropna=False)
          .mean(numeric_only=True)
          .reset_index()
    )
    #make avergge pahsing scores and pivot to histone score columns
    cnt = cnt.groupby(["variant_id", "output_type", "track_name"], observed=True).agg({
        "score": "mean"
    }).reset_index()
    cnt = cnt.pivot(columns = "track_name", index = "variant_id", values = "score").drop_duplicates().reset_index()
    return cnt

def format_alphagenome_ft_vep_per_gene(vep: pd.DataFrame, gtf: pd.DataFrame) -> pd.DataFrame:
    """Formats fine-tuned AlphaGenome VEP scores per variant, gene and cell type (CM, FB).

    For data with cell-type-resolved effects (SDR-seq DE results), so that every
    variant-gene pair is compared with the prediction of its own cell type:

    * phasing suffixes are removed from ``variant_id`` (`strip_phasing_suffix`);
    * Cut&Tag: mean over days, haplotypes and genes, one column per mark
      (``AG_ft_H3K27ac_score``, ``AG_ft_H3K4me3_score``; not cell-type-specific);
    * ATAC and RNA: only the Cardiomyocytes (``CM``) and Fibroblasts (``FB``) tracks;
      ATAC -> ``AG_ft_ATAC_score``;
    * RNA: mean over haplotypes, only the track on the strand of the gene (from
      ``gtf``) -> ``AG_ft_RNA_score``; all genes of a variant are kept (``Gene``).

    Args:
        vep (pd.DataFrame): Long VEP table of ``workflows/alphagenome_ft/04_vep.sh``
            (``variant_id``, ``gene_id``, ``gene_name``, ``output_type``,
            ``track_name``, ``scored_interval``, ``score``, ...).
        gtf (pd.DataFrame): Gene annotation with ``gene_id``, ``gene_name`` and
            ``strand`` (e.g. from `cardiomics_s2f.eval.track_prediction.get_gtf`).

    Returns:
        pd.DataFrame: One row per variant, cell type and gene (``Gene`` is missing
        for variants without RNA score) with ``AG_ft_ATAC_score``,
        ``AG_ft_H3K27ac_score``, ``AG_ft_H3K4me3_score`` and ``AG_ft_RNA_score``.
    """
    ag_ft_pred = vep.copy()
    ag_ft_pred["variant_id"] = strip_phasing_suffix(ag_ft_pred["variant_id"])

    cnt = _mean_alphagenome_ft_cutntag_days(ag_ft_pred)

    # filter ATAC and RNA to the celltypes we have in sdr
    ag_ft_pred = ag_ft_pred[ag_ft_pred.track_name.str.contains("Cardiomyocytes|Fibroblasts")].copy()
    assay_type = []
    for _, r in ag_ft_pred.iterrows():
        if "ATAC" in r.track_name.upper(): assay_type.append("ATAC")
        elif "RNA" in r.track_name.upper(): assay_type.append("RNA")
        else: assay_type.append("misc")
    ag_ft_pred.loc[:, "output_type"] = assay_type
    cell_type = []
    for _, r in ag_ft_pred.iterrows():
        if "Cardiomyocytes" in r.track_name: cell_type.append("CM")
        elif "Fibroblasts" in r.track_name: cell_type.append("FB")
        else: cell_type.append("misc")
    ag_ft_pred.loc[:, "cell_type"] = cell_type

    #merge wide cnt and atac as neitehr cotain genes
    atac = ag_ft_pred[ag_ft_pred.output_type == "ATAC"].drop(["gene_id","gene_name", "output_type","track_name", "scored_interval"], axis = 1).drop_duplicates()
    ag_ft_cnt_atac = pd.merge(atac, cnt, how="outer", on = "variant_id").rename(columns={"score": "AG_ft_ATAC_score", "H3K27ac": "AG_ft_H3K27ac_score", "H3K4me3": "AG_ft_H3K4me3_score"})

    #take rna with genes and merge with strand mathced genes from gtf
    rna = ag_ft_pred[ag_ft_pred.output_type == "RNA"]
    rna = rna.rename(columns = {"gene_name":"Gene"})
    rna = rna.groupby(["variant_id", "gene_id", "Gene", "output_type", "track_name", "cell_type"], observed=True).agg({
        "score": "mean"
    }).reset_index()
    gtf_genes = gtf[["gene_id", "strand", "gene_name"]].drop_duplicates()
    rna = pd.merge(rna, gtf_genes, how = "left", right_on = "gene_name", left_on = "Gene", suffixes=("", "_gtf")).drop_duplicates()
    rna = rna[((rna.track_name.str.contains("plus")) & (rna.strand == "+")) |
        (rna.track_name.str.contains("minus")) & (rna.strand == "-")]
    rna = rna.drop(["gene_id", "gene_id_gtf", "gene_name", "track_name", "output_type"], axis = 1)
    rna = rna.rename(columns={"score": "AG_ft_RNA_score"})

    #merge rna atac and cnt
    return pd.merge(ag_ft_cnt_atac, rna,
             how = "outer", on = ["variant_id", "cell_type"]).drop_duplicates()

def format_borzoi_vep_per_gene(vep: pd.DataFrame) -> pd.DataFrame:
    """Formats Borzoi VEP scores per variant, gene and cell type (CM, FB).

    Counterpart of `format_alphagenome_ft_vep_per_gene` for the table of
    ``workflows/borzoi/04_vep.sh``:

    * Cut&Tag: mean over days, haplotypes and genes; ``SUM``, ``L2`` and ``logSUM``
      scores per mark (``SUM_score_H3K27ac``, ``L2_score_H3K4me3``, ...;
      not cell-type-specific);
    * ATAC and RNA: only the Cardiomyocytes (``CM``) and Fibroblasts (``FB``)
      tracks, mean over haplotypes (phasing suffixes removed, see
      `strip_phasing_suffix`); ATAC -> ``*_score_ATAC``; RNA -> ``*_score_RNA``
      plus the gene-level ``L2ED_score`` and ``logSED_score`` per ``Gene``.

    Args:
        vep (pd.DataFrame): Long Borzoi VEP table (``variant_id``, ``track``,
            ``assay_type``, ``gene`` and the ``L2ED``, ``logSED``, ``SUM``, ``L2``,
            ``logSUM`` scores).

    Returns:
        pd.DataFrame: One row per variant, cell type and gene with the RNA, ATAC and
        Cut&Tag score columns.
    """
    borzoi_pred = vep.copy()

    #combine the cnt days by taking the mean
    cnt = borzoi_pred[borzoi_pred.assay_type == "Cut&Tag"].copy()
    cnt.loc[:,"track"] = cnt["track"].str.replace(r"Day\d+", "", regex=True)
    cnt = (
        cnt.groupby(["variant_id", "track", "assay_type", "gene"], dropna=False)
          .mean(numeric_only=True)
          .reset_index()
    )

    # #filteer to the celltypes we have in sdr
    borzoi_pred = borzoi_pred[borzoi_pred.track.str.contains("Cardiomyocytes|Fibroblasts")].copy()
    assay_type = []
    for _, r in borzoi_pred.iterrows():
        if "ATAC" in r.track.upper(): assay_type.append("ATAC")
        elif "RNA" in r.track.upper(): assay_type.append("RNA")
        else: assay_type.append("misc")

    borzoi_pred.loc[:, "assay_type"] = assay_type
    cell_type = []
    for _, r in borzoi_pred.iterrows():
        if "Cardiomyocytes" in r.track: cell_type.append("CM")
        elif "Fibroblasts" in r.track: cell_type.append("FB")
        else: cell_type.append("misc")
    borzoi_pred.loc[:, "cell_type"] = cell_type

    borzoi_pred = pd.concat([cnt, borzoi_pred]).rename(columns={"gene": "Gene"})

    #drop pahsing
    borzoi_pred["variant_id"] = strip_phasing_suffix(borzoi_pred["variant_id"])
    #agg pahsing
    borzoi_pred = borzoi_pred.groupby(["variant_id", "track", "assay_type", "Gene", "cell_type"], observed=True, dropna=False).agg({
        "L2ED_score": "mean",
        "logSED_score": "mean",
        "SUM_score": "mean",
        "L2_score": "mean",
        "logSUM_score": "mean",
    }).reset_index()

    #merge histone mods
    borzoi_pred_no_genes_cnt = borzoi_pred[borzoi_pred.assay_type== "Cut&Tag"].copy()
    borzoi_pred_no_genes_cnt['histone_mod'] = borzoi_pred_no_genes_cnt['track'].str.replace('_100M_norm.bw', '', regex=False)
    borzoi_pred_no_genes_cnt['histone_mod'] = borzoi_pred_no_genes_cnt['histone_mod'].str.replace('cutntag_4', 'H3K4me3', regex=False).replace('K4', 'H3K4me3', regex=False).replace('K27ac', 'H3K27ac', regex=False)
    borzoi_pred_no_genes_cnt = borzoi_pred_no_genes_cnt.groupby(["variant_id", "histone_mod"], observed=True, dropna=False).agg({
        "L2ED_score": "mean",
        "logSED_score": "mean",
        "SUM_score": "mean",
        "L2_score": "mean",
        "logSUM_score": "mean",
    }).reset_index()

    #pivot cnt
    borzoi_pred_no_genes_cnt = borzoi_pred_no_genes_cnt.pivot_table(
        index='variant_id',
        columns='histone_mod',
        values=['SUM_score', 'L2_score', 'logSUM_score'],
        aggfunc='mean'
    )
    borzoi_pred_no_genes_cnt.columns = [f"{metric}_{mod}" for metric, mod in borzoi_pred_no_genes_cnt.columns]
    borzoi_pred_no_genes_cnt = borzoi_pred_no_genes_cnt.reset_index()

    #pivot atac
    borzoi_pred_no_genes_atac = borzoi_pred[borzoi_pred.assay_type== "ATAC"].drop(
        ["assay_type", "track", "Gene", "L2ED_score", "logSED_score"], axis = 1
    ).rename(columns = {"SUM_score": "SUM_score_ATAC", "L2_score": "L2_score_ATAC", "logSUM_score": "logSUM_score_ATAC"}).reset_index(drop = True)

    #privot rna
    borzoi_pred_rna = borzoi_pred[borzoi_pred.assay_type == "RNA"].drop(
        ["assay_type", "track"], axis = 1).rename(
        columns = {"SUM_score": "SUM_score_RNA", "L2_score": "L2_score_RNA", "logSUM_score": "logSUM_score_RNA"})

    #merge back togehter
    borzoi_pred_cnt_atac = pd.merge(borzoi_pred_no_genes_cnt, borzoi_pred_no_genes_atac, how = "outer", on = ["variant_id"])
    return pd.merge(borzoi_pred_rna, borzoi_pred_cnt_atac, how = "outer", on = ["variant_id", "cell_type"])

def format_alphagenome_ft_vep_top_gene(vep: pd.DataFrame, gtf: pd.DataFrame,
                                       cell_type: str = "Cardiomyocytes") -> pd.DataFrame:
    """Formats fine-tuned AlphaGenome VEP scores to one row per variant (largest-effect gene).

    For data without cell-type-resolved effects (e.g. the Richter de novo
    variants), where the RNA effect of a variant is taken as its largest absolute
    effect on any gene. Counterpart of `format_alphagenome_ft_vep_per_gene`:

    * Cut&Tag: mean over days and genes, one column per mark
      (``AG_ft_H3K27ac_score``, ``AG_ft_H3K4me3_score``; not cell-type-specific);
    * ATAC and RNA: only the tracks of ``cell_type``; ATAC -> ``AG_ft_ATAC_score``;
    * RNA: only the track on the strand of the gene (from ``gtf``); per variant the
      gene with the largest absolute score -> ``AG_ft_RNA_score`` and ``AG_ft_Gene``.

    Args:
        vep (pd.DataFrame): Long VEP table of ``workflows/alphagenome_ft/04_vep.sh``
            (``variant_id``, ``gene_id``, ``gene_name``, ``output_type``
            (``CutnTag``, ``multiome_ATAC``, ``multiome_RNA``), ``track_name``,
            ``scored_interval``, ``score``).
        gtf (pd.DataFrame): Gene annotation with ``gene_id``, ``gene_name`` and
            ``strand`` (e.g. from `cardiomics_s2f.eval.track_prediction.get_gtf`).
        cell_type (str, optional): Substring of the ATAC and RNA track names to keep.
            Defaults to ``"Cardiomyocytes"``.

    Returns:
        pd.DataFrame: One row per variant with ``AG_ft_ATAC_score``,
        ``AG_ft_H3K27ac_score``, ``AG_ft_H3K4me3_score``, ``AG_ft_RNA_score`` and
        ``AG_ft_Gene``.
    """
    cnt = _mean_alphagenome_ft_cutntag_days(vep)
    # filter ATAC and RNA to the cell type
    vep = vep[vep.track_name.str.contains(cell_type)]
    #merge wide cnt and atac as neitehr cotain genes
    atac = vep[vep.output_type == "multiome_ATAC"].drop(["gene_id","gene_name", "output_type","track_name", "scored_interval"], axis = 1).drop_duplicates()
    ag_ft_cnt_atac = pd.merge(atac, cnt, how="outer", on = "variant_id").rename(columns={"score": "AG_ft_ATAC_score", "H3K27ac": "AG_ft_H3K27ac_score", "H3K4me3": "AG_ft_H3K4me3_score"})
    #take rna with genes and merge with strand mathced genes from gtf
    rna = vep[vep.output_type == "multiome_RNA"]
    rna = rna.rename(columns = {"gene_name":"AG_ft_Gene"})
    gtf_genes = gtf[["gene_id", "strand", "gene_name"]].drop_duplicates()
    rna = pd.merge(rna, gtf_genes, how = "left", right_on = "gene_name", left_on = "AG_ft_Gene", suffixes=("", "_gtf")).drop_duplicates()
    rna = rna[((rna.track_name.str.contains("plus")) & (rna.strand == "+")) |
        (rna.track_name.str.contains("minus")) & (rna.strand == "-")]
    rna = rna.drop(["gene_id", "gene_id_gtf", "gene_name", "track_name", "strand", "output_type", "scored_interval"], axis = 1)
    #take absmax gene
    idx = rna.groupby(["variant_id"])["score"].apply(lambda x: x.abs().idxmax())
    rna = rna.loc[idx].reset_index(drop= True)
    rna = rna.rename(columns={"score": "AG_ft_RNA_score"})
    #merge rna atac and cnt
    ret = pd.merge(ag_ft_cnt_atac, rna, how = "outer", on = ["variant_id"]).drop_duplicates()
    ret = ret.groupby('variant_id', as_index=False).agg({
        'AG_ft_ATAC_score': 'mean', 'AG_ft_H3K27ac_score': 'mean', 'AG_ft_H3K4me3_score': 'mean', 'AG_ft_RNA_score': 'mean', 'AG_ft_Gene': 'first'})

    return ret

def _pivot_tf_raw_scores_keep_nans(df: pd.DataFrame) -> pd.DataFrame:
    """Pivots ``TF_raw_score`` to one ``<TF>_TF_raw_score`` column per TF.

    Unlike `pivot_tf_scores`, rows with missing values in the identifying columns
    are kept (no ``pivot_table``), and only the raw scores are pivoted.
    """
    tf_cols = ['transcription_factor', 'TF_raw_score']
    base_cols = [col for col in df.columns if col not in tf_cols]
    df_unique = df.drop_duplicates(subset=base_cols + ['transcription_factor'])
    df_pivoted = (
        df_unique
        .set_index(base_cols + ['transcription_factor'])['TF_raw_score']
        .unstack('transcription_factor')
    )
    df_pivoted.columns = [f"{col}_TF_raw_score" for col in df_pivoted.columns]
    return df_pivoted.reset_index()

def format_pretrained_alphagenome_vep_top_gene(vep: pd.DataFrame) -> pd.DataFrame:
    """Formats pretrained AlphaGenome VEP scores to one row per variant (largest-effect gene).

    Takes an aggregated track-group table of ``workflows/alphagenome_api/04_vep.sh``
    (here: the whole-heart group) and keeps, per variant, the rows of the gene with
    the largest absolute ``RNA_raw_score`` (and variants without RNA score), drops
    the quantile scores, pivots the TF scores and averages the raw scores per
    variant. All score columns get the prefix ``AG_pre_``; the gene is
    ``AG_pre_Gene`` and ``variant_name`` becomes ``variant_id``.

    Note:
        The table must contain the RNA, ATAC, histone (H3K27ac, H3K27me3, H3K36me3,
        H3K4me1, H3K4me3, H3K9ac, H3K9me3) and TF (CTCF, POLR2A) scores of the
        whole-heart track group.

    Args:
        vep (pd.DataFrame): Aggregated pretrained AlphaGenome VEP table
            (``variant_name``, ``gene_id``, ``gene_name``, ``CHROM``, ``POS``,
            ``REF``, ``ALT``, ``transcription_factor`` and the ``*_raw_score`` /
            ``*_quantile_score`` columns).

    Returns:
        pd.DataFrame: One row per variant with ``AG_pre_Gene`` and the
        ``AG_pre_*_raw_score`` columns.
    """
    ag_pre_richter = vep
    #get max rna fast
    abs_rna = ag_pre_richter['RNA_raw_score'].abs()
    max_scores = abs_rna.groupby(ag_pre_richter['variant_name']).transform('max')
    mask_max_score = abs_rna == max_scores
    mask_nan = ag_pre_richter['RNA_raw_score'].isna()

    ag_pre_richter = ag_pre_richter[mask_max_score | mask_nan]

    ag_pre_richter = ag_pre_richter[[col for col in ag_pre_richter.columns if "quantile" not in col]]
    ag_pre_richter = ag_pre_richter.drop("gene_id", axis =1)
    ag_pre_richter = _pivot_tf_raw_scores_keep_nans(ag_pre_richter)
    ag_pre_richter = ag_pre_richter.rename(columns={"variant_name": "variant_id", "gene_name": "AG_pre_Gene"})
    ag_pre_richter = ag_pre_richter.groupby('variant_id', as_index=False).agg({
        'AG_pre_Gene': 'first',
        'CHROM': 'first',
        'POS': 'first',
        'REF': 'first',
        'ALT': 'first',
        'RNA_raw_score': 'mean',
        'H3K27ac_HIST_raw_score': 'mean',
        'H3K27me3_HIST_raw_score': 'mean',
        'H3K36me3_HIST_raw_score': 'mean',
        'H3K4me1_HIST_raw_score': 'mean',
        'H3K4me3_HIST_raw_score': 'mean',
        'H3K9ac_HIST_raw_score': 'mean',
        'H3K9me3_HIST_raw_score': 'mean',
        'ATAC_raw_score': 'mean',
        'CTCF_TF_raw_score': 'mean',
        'POLR2A_TF_raw_score': 'mean'
    })

    return ag_pre_richter.rename(columns={col: f"AG_pre_{col}" for col in ag_pre_richter.columns if "_score" in col})
