"""
ChromBPNet Variant Input
========================

Converts VCF-style variant tables into the ``chrombpnet`` schema read by the Kundaje lab
`variant-scorer <https://github.com/kundajelab/variant-scorer>`_ (columns ``chr, pos,
allele1, allele2, variant_id``; tab-separated, no header).

* :func:`read_variant_table`: CSV or TSV with automatic delimiter detection.
* :func:`drop_variants_with_ref_mismatch`: removes variants whose REF allele does not
  match the reference genome (``pysam``, imported when called).
* :func:`trim_shared_allele_prefix`: removes the bases REF and ALT share at the start of
  indels and shifts the position accordingly (``-`` for an empty allele).
* :func:`to_chrombpnet_variants` / :func:`write_chrombpnet_variants`: the full conversion.

variant-scorer takes ``allele1`` as the reference sequence starting at the 1-based
``pos`` and adjusts the flanks for the length difference of the alleles, so any REF/ALT
pair is valid input once the shared leading bases are removed.
"""

import csv

import pandas as pd

CHROMBPNET_COLUMNS = ["chr", "pos", "allele1", "allele2", "variant_id"]


def read_variant_table(path):
    """Reads a CSV or TSV file with automatic delimiter detection.

    Args:
        path (str): The file path to read.

    Returns:
        pd.DataFrame: The loaded data as a pandas DataFrame.
    """
    with open(path, "r") as f:
        sample = f.read(2048)

    dialect = csv.Sniffer().sniff(sample, delimiters=[",", "\t"])
    sep = dialect.delimiter

    return pd.read_csv(path, sep=sep)


def drop_variants_with_ref_mismatch(df, genome_path):
    """Filters out variants where the REF allele does not match the reference genome.

    Chromosome names are matched with and without the ``chr`` prefix. Variants on
    unknown chromosomes or outside the chromosome are dropped as well.

    Args:
        df (pd.DataFrame): The dataframe containing the variants with standard column names
            (``chr``, ``pos`` 1-based, ``allele1`` = REF).
        genome_path (str): The file path to the reference genome FASTA file (indexed).

    Returns:
        pd.DataFrame: A filtered dataframe containing only valid variants.
    """
    import pysam

    fasta = pysam.FastaFile(genome_path)
    valid_indices = []
    mismatches = 0

    for idx, row in df.iterrows():
        chrom = str(row["chr"])
        pos = int(row["pos"])
        ref = str(row["allele1"])

        # We must align chromosome nomenclature between the VCF and the FASTA to prevent false positive drops.
        if chrom not in fasta.references:
            alt_chrom = "chr" + chrom if not chrom.startswith("chr") else chrom[3:]
            if alt_chrom in fasta.references:
                chrom = alt_chrom
            else:
                mismatches += 1
                continue

        # PySAM uses zero-based half-open coordinates, requiring us to shift the one-based VCF position.
        start = pos - 1
        end = start + len(ref)

        try:
            genome_seq = fasta.fetch(chrom, start, end).upper()

            # The model requires exact sequence matches to safely substitute alleles without corrupting the surrounding sequence context.
            if genome_seq == ref.upper():
                valid_indices.append(idx)
            else:
                mismatches += 1
        except (ValueError, IndexError):
            # Drops the variant if the coordinates fall outside the boundaries of the defined chromosome.
            mismatches += 1

    fasta.close()

    print(f"Total input variants: {len(df)}", flush = True)
    print(f"Filtered out (mismatched REF): {mismatches}", flush = True)
    print(f"Remaining valid variants: {len(valid_indices)}", flush = True)

    return df.loc[valid_indices].copy()


def trim_shared_allele_prefix(df):
    """Removes the leading bases shared by REF and ALT of indels.

    VCF indels carry an anchor base (``AG -> A``); variant-scorer expects the changed
    sequence only, with ``-`` for an empty allele, at the position of its first base
    (``G -> -`` one base further). All shared leading bases are removed, so complex and
    non-normalised indels are converted correctly as well:

    * ``AG -> A`` at 100 becomes ``G -> -`` at 101 (deletion);
    * ``A -> AT`` at 100 becomes ``- -> T`` at 101 (insertion);
    * ``ACG -> AT`` at 100 becomes ``CG -> T`` at 101 (complex);
    * ``ACAC -> AC`` at 100 becomes ``AC -> -`` at 102.

    Variants with alleles of equal length (SNVs, MNVs) are left unchanged.

    Args:
        df (pd.DataFrame): Variants with columns ``pos``, ``allele1`` (REF) and ``allele2`` (ALT).

    Returns:
        pd.DataFrame: A copy with trimmed alleles and shifted positions.
    """
    out = df.copy()
    alleles1, alleles2, positions = [], [], []
    for pos, a1, a2 in zip(out["pos"], out["allele1"], out["allele2"]):
        a1, a2 = str(a1), str(a2)
        if len(a1) != len(a2):
            n = 0
            while n < min(len(a1), len(a2)) and a1[n] == a2[n]:
                n += 1
            a1, a2, pos = a1[n:] or "-", a2[n:] or "-", pos + n
        alleles1.append(a1)
        alleles2.append(a2)
        positions.append(pos)
    out["allele1"] = alleles1
    out["allele2"] = alleles2
    out["pos"] = positions
    return out


def to_chrombpnet_variants(df):
    """Converts a VCF-like variant table into the variant-scorer ``chrombpnet`` schema.

    Column names are normalised (``CHROM``/``#CHROM``/``chrom`` -> ``chr``, ``POS`` ->
    ``pos``, ``REF`` -> ``allele1``, ``ALT`` -> ``allele2``), alleles are upper-cased, a
    ``variant_id`` (``chr-pos-ref-alt``) is added if missing, and indels are trimmed with
    :func:`trim_shared_allele_prefix` unless the table already uses ``-`` alleles.

    Args:
        df (pd.DataFrame): Variant table (VCF columns or the ``chrombpnet`` names).

    Returns:
        pd.DataFrame: Columns ``chr, pos, allele1, allele2, variant_id``.

    Raises:
        ValueError: If a required column is missing.
    """
    df = df.copy()

    # Normalize column names to ensure downstream processing works correctly regardless of the input format.
    rename_map = {
        "#CHROM": "chr",
        "CHROM": "chr",
        "chrom": "chr",
        "POS": "pos",
        "position": "pos",
        "REF": "allele1",
        "ALT": "allele2",
    }

    df.rename(columns=rename_map, inplace=True)

    required = {"chr", "pos", "allele1", "allele2"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # variant-scorer compares allele1 with the upper-cased genome sequence
    df["allele1"] = df["allele1"].astype(str).str.upper()
    df["allele2"] = df["allele2"].astype(str).str.upper()

    # Build variant_id to track variants through the scoring pipeline.
    if "variant_id" not in df.columns:
        df["variant_id"] = (
            df["chr"].astype(str) + "-"
            + df["pos"].astype(str) + "-"
            + df["allele1"].astype(str) + "-"
            + df["allele2"].astype(str)
        )

    # Convert the indel representation to match what variant-scorer expects,
    # unless the table is already in that format.
    if not ((df["allele1"] == "-").any() or (df["allele2"] == "-").any()):
        df = trim_shared_allele_prefix(df)

    return df[CHROMBPNET_COLUMNS]


def write_chrombpnet_variants(in_path, out_file, genome_path=None):
    """Converts a variant CSV/TSV into a variant-scorer input file.

    Args:
        in_path (str): The path to the input CSV or TSV file.
        out_file (str): The path where the converted TSV will be saved (no header).
        genome_path (str, optional): The path to the reference genome FASTA. If given,
            variants whose REF does not match the genome are removed first.

    Returns:
        pd.DataFrame: The written table.
    """
    df = read_variant_table(in_path)
    df = df.rename(columns={"#CHROM": "chr", "CHROM": "chr", "chrom": "chr", "POS": "pos",
                            "position": "pos", "REF": "allele1", "ALT": "allele2"})

    # Filter bad variants before trimming the alleles, because the check needs the
    # original anchor bases of the VCF.
    if genome_path:
        df = drop_variants_with_ref_mismatch(df, genome_path)

    df = to_chrombpnet_variants(df)
    df.to_csv(out_file, sep="\t", index=False, header=False)
    print("Wrote:", out_file)
    return df
