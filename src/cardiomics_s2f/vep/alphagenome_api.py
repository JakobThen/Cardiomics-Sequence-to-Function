"""
Pretrained AlphaGenome VEP
==========================

Helpers for variant effect prediction with the pretrained AlphaGenome model via
the AlphaGenome API (``alphagenome.models.dna_client``): tidying the scores
returned by ``score_variants`` and aggregating them over groups of tracks
(ontology CURIEs, e.g. all heart tracks) into one score per variant and modality.

The runnable script is ``workflows/alphagenome_api/run_vep_pretrained.py``.
"""

import os
from concurrent.futures import ProcessPoolExecutor, as_completed

import pandas as pd
from alphagenome.models import dna_client
from alphagenome.models import variant_scorers


def sequence_length_from_label(size_str):
    """Maps a sequence length label to the AlphaGenome constant.

    Args:
        size_str (str): ``"1MB"``, ``"500KB"``, ``"100KB"`` or ``"16KB"``.

    Returns:
        int: The sequence length in bp (``dna_client.SEQUENCE_LENGTH_*``).
    """
    mapping = {
        "1MB":   dna_client.SEQUENCE_LENGTH_1MB,
        "500KB": dna_client.SEQUENCE_LENGTH_500KB,
        "100KB": dna_client.SEQUENCE_LENGTH_100KB,
        "16KB":  dna_client.SEQUENCE_LENGTH_16KB,
    }
    return mapping[size_str]


def read_variant_table(file_path):
    """Reads a VCF-style variant table (CSV or TSV).

    The separator is taken from the file extension (``.tsv``/``.csv``) or, for
    other extensions, guessed from the first line.

    Args:
        file_path (str): Table with at least ``CHROM``, ``POS``, ``REF`` and ``ALT``.

    Returns:
        pd.DataFrame: The variant table.

    Raises:
        FileNotFoundError: If ``file_path`` does not exist.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Input file not found: {file_path}")
    if file_path.lower().endswith(".tsv"):
        return pd.read_csv(file_path, sep="\t")
    elif file_path.lower().endswith(".csv"):
        return pd.read_csv(file_path, sep=",")
    with open(file_path, "r") as f:
        first_line = f.readline()
        sep = "\t" if first_line.count("\t") > first_line.count(",") else ","
    return pd.read_csv(file_path, sep=sep)


def aggregate_modality_scores(modality, var_scores_mod):
    """Aggregates the scores of one output type over its tracks (mean).

    RNA-seq scores are restricted to protein-coding genes and kept per gene;
    ChIP-TF scores are averaged per transcription factor and ChIP-histone scores
    per histone mark (pivoted to ``<mark>_HIST_{raw,quantile}_score`` columns);
    all other supported output types become ``<PREFIX>_{raw,quantile}_score``.

    Args:
        modality (str): Output type, e.g. ``"RNA_SEQ"``, ``"ATAC"``, ``"CHIP_TF"``.
        var_scores_mod (pd.DataFrame): Tidy scores of this output type
            (``variant_name``, ``raw_score``, ``quantile_score`` and the
            modality's grouping columns).

    Returns:
        tuple: ``(aggregated DataFrame, merge columns, keep gene info)``;
        ``(None, [], False)`` for unsupported output types.
    """
    keep_gene_info = False
    merge_cols = []

    if modality == "RNA_SEQ":
        keep_gene_info = True
        if "gene_type" in var_scores_mod.columns:
            var_scores_mod = var_scores_mod[var_scores_mod.gene_type == "protein_coding"]
        var_scores_mod = var_scores_mod.groupby(
            ["variant_name", "gene_id", "gene_name"], as_index=False
        ).agg(RNA_raw_score=("raw_score", "mean"), RNA_quantile_score=("quantile_score", "mean"))
        merge_cols = ["variant_name", "gene_id", "gene_name"]

    elif modality == "CHIP_TF":
        var_scores_mod = var_scores_mod.groupby(
            ["variant_name", "transcription_factor"], as_index=False
        ).agg(TF_raw_score=("raw_score", "mean"), TF_quantile_score=("quantile_score", "mean"))
        merge_cols = ["variant_name"]

    elif modality == "CHIP_HISTONE":
        var_scores_mod = var_scores_mod.groupby(
            ["variant_name", "histone_mark"], as_index=False
        ).agg(HIST_raw_score=("raw_score", "mean"), HIST_quantile_score=("quantile_score", "mean"))
        df_raw = var_scores_mod.pivot(
            index="variant_name", columns="histone_mark", values="HIST_raw_score"
        )
        df_raw.columns = [f"{col}_HIST_raw_score" for col in df_raw.columns]
        df_quant = var_scores_mod.pivot(
            index="variant_name", columns="histone_mark", values="HIST_quantile_score"
        )
        df_quant.columns = [f"{col}_HIST_quantile_score" for col in df_quant.columns]
        var_scores_mod = pd.concat([df_raw, df_quant], axis=1).reset_index()
        merge_cols = ["variant_name"]

    elif modality in [
        "ATAC", "CAGE", "PROCAP", "DNASE", "CONTACT_MAPS",
        "POLYADENYLATION", "SPLICE_SITES", "SPLICE_SITE_USAGE",
        "SPLICE_JUNCTIONS"]:
        prefix_map = {
            "ATAC": "ATAC", "CAGE": "CAGE", "PROCAP": "PROCAP",
            "DNASE": "DNASE", "CONTACT_MAPS": "CONTACTS",
            "POLYADENYLATION": "POLYA", "SPLICE_SITES": "SPLICE_SITES",
            "SPLICE_SITE_USAGE": "SPLICE_USAGE", "SPLICE_JUNCTIONS": "SPLICE_JX",
        }
        prefix = prefix_map[modality]
        var_scores_mod = var_scores_mod.groupby(["variant_name"], as_index=False).agg(
            **{
                f"{prefix}_raw_score":      ("raw_score",      "mean"),
                f"{prefix}_quantile_score": ("quantile_score", "mean"),
            }
        )
        merge_cols = ["variant_name"]
    else:
        return None, [], False

    return var_scores_mod, merge_cols, keep_gene_info


def aggregate_scores_by_curies(var_scores, curies=None, max_workers=8):
    """Aggregates tidy variant scores over a set of tracks, per output type.

    Filters the scores to the given ontology CURIEs, aggregates each output type
    in parallel (`aggregate_modality_scores`) and merges the results into one
    wide table with ``CHROM``, ``POS``, ``REF`` and ``ALT`` split from
    ``variant_name``. Gene columns are only kept if RNA-seq scores are present.

    Args:
        var_scores (pd.DataFrame): Tidy scores (`tidy_variant_scores`).
        curies (list of str, optional): Ontology CURIEs to keep. Defaults to None (all tracks).
        max_workers (int, optional): Worker processes. Defaults to 8.

    Returns:
        pd.DataFrame: Aggregated scores (empty if no track matches).
    """
    if curies:
        var_scores_fil = var_scores[var_scores.ontology_curie.isin(curies)].copy()
    else:
        var_scores_fil = var_scores.copy()

    if var_scores_fil.empty:
        return pd.DataFrame()

    if "variant_name" not in var_scores_fil.columns:
        var_scores_fil["variant_name"] = var_scores_fil["variant_id"].apply(str)

    base_df = None
    any_keep_gene_info = False

    futures = []
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        for modality in var_scores_fil.output_type.unique():
            var_scores_mod = var_scores_fil[var_scores_fil.output_type == modality].copy()
            if var_scores_mod.empty:
                continue
            futures.append(executor.submit(aggregate_modality_scores, modality, var_scores_mod))

    for future in as_completed(futures):
        result_df, merge_cols, keep_gene_info = future.result()
        if result_df is None or result_df.empty:
            continue
        if keep_gene_info:
            any_keep_gene_info = True
        if base_df is None:
            base_df = result_df
        else:
            actual_merge_cols = [
                c for c in merge_cols if c in base_df.columns and c in result_df.columns
            ]
            base_df = base_df.merge(result_df, on=actual_merge_cols, how="outer")

    if base_df is not None:
        try:
            base_df[["CHROM", "POS", "REF", "ALT"]] = base_df["variant_name"].str.split(
                "-", expand=True
            )
            base_df["POS"] = base_df["POS"].astype(int)
        except Exception:
            pass

        if not any_keep_gene_info:
            base_df = base_df.drop(
                columns=[c for c in ["gene_id", "gene_name"] if c in base_df.columns]
            )

    return base_df


def tidy_variant_scores(batch_results, variants_list):
    """Converts ``score_variants`` output into a flat table.

    Adds ``variant_name`` (``CHROM-POS-REF-ALT``), converts object columns
    (variants, intervals) to strings and the score columns to numbers.

    Args:
        batch_results (list): Output of ``dna_client.DnaClient.score_variants``.
        variants_list (list): The scored variants (not used).

    Returns:
        pd.DataFrame: One row per variant, track (and gene) with ``raw_score``
        and ``quantile_score``.
    """
    vep = variant_scorers.tidy_scores(batch_results)

    # Build variant_name from the Variant objects in the variant_id column
    vep["variant_name"] = [
        f"{v.chromosome}-{v.position}-{v.reference_bases}-{v.alternate_bases}"
        for v in vep.variant_id
    ]

    # Convert object columns (Variant / Interval objects) to plain strings
    object_cols = vep.select_dtypes(include=["object"]).columns
    for col in object_cols:
        vep[col] = (
            vep[col]
            .apply(lambda x: str(x) if pd.notnull(x) else None)
            .astype("string")
        )

    # Ensure numeric score columns are typed correctly
    for score_col in ["raw_score", "quantile_score"]:
        if score_col in vep.columns:
            vep[score_col] = pd.to_numeric(vep[score_col], errors="coerce")

    return vep
