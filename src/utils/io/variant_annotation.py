"""
Variant Annotation Module

This module provides functions to annotate a pandas DataFrame of genetic variants 
(VCF-style) with various genomic features, conservation scores, and coding effects.

Expected Input Data Format:
---------------------------
The core input to most of these functions is a pandas DataFrame representing genetic 
variants. At a minimum, this DataFrame must contain the following columns:
    * 'Chrom' or 'Chromosome' (str): The chromosome (e.g., '1', 'chr1', 'X').
    * 'Pos' (int): The 1-based genomic position of the variant.
    * 'Ref' (str): The reference allele.
    * 'Alt' (str): The alternate allele.

Some downstream functions (like coding effects) rely on the output of upstream 
functions (like GTF feature annotation), which will be explicitly noted in their 
respective docstrings.
"""

import time
from typing import Dict, List, Any, Optional

import numpy as np
import pandas as pd
import pyranges as pr
import requests
from ncls import NCLS
from tqdm.auto import tqdm
import pyBigWig 


# --- Constants ---
CONSEQUENCE_RANK: Dict[str, int] = {
    'transcript_ablation': 1, 'splice_acceptor_variant': 2, 'splice_donor_variant': 3,
    'stop_gained': 4, 'frameshift_variant': 5, 'stop_lost': 6, 'start_lost': 7,
    'transcript_amplification': 8, 'feature_elongation': 9, 'feature_truncation': 10,
    'inframe_insertion': 11, 'inframe_deletion': 12, 'missense_variant': 13,
    'protein_altering_variant': 14, 'splice_donor_5th_base_variant': 15,
    'splice_region_variant': 16, 'splice_donor_region_variant': 17,
    'splice_polypyrimidine_tract_variant': 18, 'incomplete_terminal_codon_variant': 19,
    'start_retained_variant': 20, 'stop_retained_variant': 21, 'synonymous_variant': 22,
    'coding_sequence_variant': 23, 'mature_miRNA_variant': 24, '5_prime_UTR_variant': 25,
    '3_prime_UTR_variant': 26, 'non_coding_transcript_exon_variant': 27,
    'intron_variant': 28, 'NMD_transcript_variant': 29, 'non_coding_transcript_variant': 30,
    'coding_transcript_variant': 31, 'upstream_gene_variant': 32,
    'downstream_gene_variant': 33, 'TFBS_ablation': 34, 'TFBS_amplification': 35,
    'TF_binding_site_variant': 36, 'regulatory_region_ablation': 37,
    'regulatory_region_amplification': 38, 'regulatory_region_variant': 39,
    'intergenic_variant': 40, 'sequence_variant': 41,
}

IMPACT: Dict[str, str] = {
    'transcript_ablation': 'HIGH', 'splice_acceptor_variant': 'HIGH',
    'splice_donor_variant': 'HIGH', 'stop_gained': 'HIGH', 'frameshift_variant': 'HIGH',
    'stop_lost': 'HIGH', 'start_lost': 'HIGH', 'transcript_amplification': 'HIGH',
    'feature_elongation': 'HIGH', 'feature_truncation': 'HIGH',
    'inframe_insertion': 'MODERATE', 'inframe_deletion': 'MODERATE',
    'missense_variant': 'MODERATE', 'protein_altering_variant': 'MODERATE',
    'splice_donor_5th_base_variant': 'LOW', 'splice_region_variant': 'LOW',
    'splice_donor_region_variant': 'LOW', 'splice_polypyrimidine_tract_variant': 'LOW',
    'incomplete_terminal_codon_variant': 'LOW', 'start_retained_variant': 'LOW',
    'stop_retained_variant': 'LOW', 'synonymous_variant': 'LOW',
    'coding_sequence_variant': 'MODIFIER', 'mature_miRNA_variant': 'MODIFIER',
    '5_prime_UTR_variant': 'MODIFIER', '3_prime_UTR_variant': 'MODIFIER',
    'non_coding_transcript_exon_variant': 'MODIFIER', 'intron_variant': 'MODIFIER',
    'NMD_transcript_variant': 'MODIFIER', 'non_coding_transcript_variant': 'MODIFIER',
    'coding_transcript_variant': 'MODIFIER', 'upstream_gene_variant': 'MODIFIER',
    'downstream_gene_variant': 'MODIFIER', 'TFBS_ablation': 'MODIFIER',
    'TFBS_amplification': 'MODIFIER', 'TF_binding_site_variant': 'MODIFIER',
    'regulatory_region_ablation': 'MODIFIER', 'regulatory_region_amplification': 'MODIFIER',
    'regulatory_region_variant': 'MODIFIER', 'intergenic_variant': 'MODIFIER',
    'sequence_variant': 'MODIFIER',
}


#helper to load
def standardize_vcf_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardizes column names to the expected internal format ('Chromosome', 'Pos', 'Ref', 'Alt').
    Handles common case/naming variations like '#CHROM', 'CHROM', 'chr', 'POS', 'position', etc.
    """
    rename_dict = {}
    for col in df.columns:
        col_upper = col.upper()
        if col_upper in ['CHROM', '#CHROM', 'CHROMOSOME', 'CHR', 'SEQNAMES']:
            rename_dict[col] = 'Chromosome'
        elif col_upper in ['POS', 'POSITION']:
            rename_dict[col] = 'Pos'
        elif col_upper in ['REF', 'REFERENCE']:
            rename_dict[col] = 'Ref'
        elif col_upper in ['ALT', 'ALTERNATE' 'ALTERNATIVE']:
            rename_dict[col] = 'Alt'
        elif col_upper in ['START']:
            rename_dict[col] = 'Start'
        elif col_upper in ['END']:
            rename_dict[col] = 'End'
            
    if rename_dict:
        return df.rename(columns=rename_dict)
    return df


def vcf_to_pr(vcf_df: pd.DataFrame) -> pr.PyRanges:
    """
    Converts a VCF-style pandas DataFrame into a PyRanges object.
    
    Translates 1-based 'Pos' coordinates to 0-based half-open intervals ('Start', 'End') 
    required by PyRanges.
    
    Args:
        vcf_df (pd.DataFrame): Input dataframe. Must contain 'Chrom' (or 'Chromosome') 
                               and 'Pos' columns.
                               
    Returns:
        pr.PyRanges: PyRanges object containing the standard interval coordinates.
    """
    vcf_pr = standardize_vcf_columns(vcf_df.copy())
    
    vcf_pr['Start'] = vcf_pr['Pos'] - 1
    vcf_pr['End'] = vcf_pr['Pos']
    return pr.PyRanges(vcf_pr)


def load_ccre_data(ccre_file: str) -> pr.PyRanges:
    """Loads ENCODE cCRE file into a PyRanges object once for reuse."""
    ccre_cols = [
        "Chromosome", "Start", "End", "cCRE_id", "score", "strand", 
        "thickStart", "thickEnd", "reserved", "cCRE_class", 
        "DNase_maxZ", "H3K4me3_maxZ", "H3K27ac_maxZ", "CTCF_maxZ"
    ]
    ccre_df = pd.read_csv(ccre_file, sep="\t", names=ccre_cols, header=0, compression='gzip')
    return pr.PyRanges(ccre_df)


def annotate_cCREs(ccre_pr: pr.PyRanges, vcf_pr: pr.PyRanges) -> pd.DataFrame:
    """Annotates variants using a pre-loaded cCRE PyRanges object."""
    annotated = vcf_pr.join(ccre_pr, how="left")
    df = annotated.df.replace([-1, -1.0, "-1"], np.nan)
    
    columns_to_drop = ["Start_b", "End_b", "reserved", "score", "strand"]
    return df.drop(columns=[col for col in columns_to_drop if col in df.columns], axis=1)


def annotate_gtf_features(variants_df: pd.DataFrame, gtf_df: pd.DataFrame) -> pd.DataFrame:
    """
    Annotates variants with the most specific overlapping GTF feature and calculates 
    absolute distance to the closest Transcription Start Site (TSS).
    
    Hierarchy (most to least specific):
    start_codon / stop_codon > CDS > UTR > exon > intron > gene
    
    Args:
        variants_df (pd.DataFrame): Variant dataframe. Must contain 'Chromosome', 
                                    'Start', and 'End' columns (0-based coordinates).
        gtf_df (pd.DataFrame): GTF dataframe parsed into standard columns ('chrom', 
                               'start', 'end', 'feature', 'strand', 'gene_name', 'gene_type').
                               
    Returns:
        pd.DataFrame: DataFrame aligned with the input index, containing columns: 
                      'gtf_feature', 'gene_name', 'gene_type', and 'closest_TSS'.
    """
    variants_df = standardize_vcf_columns(variants_df)
    
    PRIORITY = {
        'start_codon': 1, 'stop_codon': 2, 'CDS': 3, 'UTR': 4,
        'exon': 5, 'intron': 6, 'gene': 7,
    }
    RELEVANT = {'start_codon', 'stop_codon', 'CDS', 'UTR', 'exon', 'transcript', 'gene'}

    gtf_sub = gtf_df.loc[
        gtf_df['feature'].isin(RELEVANT), 
        ['chrom', 'start', 'end', 'feature', 'gene_name', 'gene_type']
    ].copy()
    
    gtf_sub['start'] = gtf_sub['start'] - 1  
    gtf_sub['mapped_feature'] = gtf_sub['feature'].replace({'transcript': 'intron'})
    gtf_sub['priority'] = gtf_sub['mapped_feature'].map(PRIORITY).astype(np.int32)

    tx_mask = gtf_df['feature'] == 'transcript'
    tss_df = gtf_df.loc[tx_mask, ['chrom', 'start', 'end', 'strand']].copy()
    tss_df['start'] = tss_df['start'] - 1
    tss_df['tss'] = np.where(tss_df['strand'] == '+', tss_df['start'], tss_df['end'] - 1)
    tss_dict = {chrom: np.sort(grp['tss'].to_numpy()) for chrom, grp in tss_df.groupby('chrom', observed=False)}

    var_pos = variants_df[['Chromosome', 'Start', 'End']].reset_index(drop=True)

    n_vars = len(var_pos)
    best_priority = np.full(n_vars, 999, dtype=np.int32)
    best_feature = np.full(n_vars, np.nan, dtype=object)   
    best_gene_name = np.full(n_vars, np.nan, dtype=object)
    best_gene_type = np.full(n_vars, np.nan, dtype=object)
    best_closest_tss = np.full(n_vars, np.nan, dtype=np.float64)

    for chrom, vg in var_pos.groupby('Chromosome', observed=False, sort=False):
        pos_idx = vg.index.to_numpy(dtype=np.int64)
        v_starts = vg['Start'].to_numpy(dtype=np.int64)
        v_ends = vg['End'].to_numpy(dtype=np.int64)
        local_ids = np.arange(len(vg), dtype=np.int64)

        if chrom in tss_dict:
            c_tss = tss_dict[chrom]
            if len(c_tss) > 0:
                v_mid = v_starts + (v_ends - v_starts) // 2
                idx = np.searchsorted(c_tss, v_mid)
                idx1 = np.clip(idx, 0, len(c_tss) - 1)
                idx0 = np.clip(idx - 1, 0, len(c_tss) - 1)
                dist = np.minimum(np.abs(c_tss[idx1] - v_mid), np.abs(c_tss[idx0] - v_mid))
                best_closest_tss[pos_idx] = dist

        gc = gtf_sub[gtf_sub['chrom'] == chrom]
        if gc.empty:
            continue

        gc = gc.reset_index(drop=True)
        g_starts = gc['start'].to_numpy(dtype=np.int64)
        g_ends = gc['end'].to_numpy(dtype=np.int64)
        g_ids = np.arange(len(gc), dtype=np.int64)
        g_prio = gc['priority'].to_numpy(dtype=np.int32)
        g_feat = gc['mapped_feature'].to_numpy()
        g_gname = gc['gene_name'].to_numpy()                       
        g_gtype = gc['gene_type'].to_numpy()                       

        tree = NCLS(g_starts, g_ends, g_ids)
        hit_v, hit_g = tree.all_overlaps_both(v_starts, v_ends, local_ids)

        if len(hit_v) == 0:
            continue

        best = (pd.DataFrame({
                'lv': hit_v,
                'prio': g_prio[hit_g],
                'feat': g_feat[hit_g],
                'gname': g_gname[hit_g],                             
                'gtype': g_gtype[hit_g],                             
            })
            .sort_values('prio')
            .drop_duplicates('lv', keep='first')
        )

        global_idx = pos_idx[best['lv'].to_numpy()]
        best_priority[global_idx] = best['prio'].to_numpy()
        best_feature[global_idx] = best['feat'].to_numpy()
        best_gene_name[global_idx] = best['gname'].to_numpy()       
        best_gene_type[global_idx] = best['gtype'].to_numpy()       

    return pd.DataFrame({
        'gtf_feature': pd.array(best_feature, dtype=object),
        'gene_name': pd.array(best_gene_name, dtype=object),
        'gene_type': pd.array(best_gene_type, dtype=object),
        'closest_TSS': best_closest_tss
    }, index=variants_df.index)


def annotate_phyloP_score(var_df: pd.DataFrame, phylo_bw: pyBigWig.pyBigWig) -> pd.Series:
    """
    Extracts phyloP conservation scores for variants from a BigWig file.
    
    Args:
        var_df (pd.DataFrame): Variant dataframe containing 'Chromosome', 'Start', 
                               and 'End' columns.
        phylo_bw (pyBigWig.pyBigWig): An open pyBigWig file object containing phyloP scores.
        
    Returns:
        pd.Series: Series containing phyloP scores, aligned with the input dataframe index.
    """
    var_df = standardize_vcf_columns(var_df)
    valid_chroms = set(phylo_bw.chroms().keys())

    def get_phylop_chrom(group: pd.DataFrame) -> pd.Series:
        chrom = group.name
        if chrom not in valid_chroms:
            return pd.Series(np.nan, index=group.index)

        starts = group['Start'].to_numpy(dtype=np.int64)
        ends = group['End'].to_numpy(dtype=np.int64)
        span = int(ends.max() - starts.min())
        n = len(group)

        if n > 1 and span < 10_000_000:     
            region_start = int(starts.min())
            region_end = int(ends.max())
            try:
                buf = np.array(
                    phylo_bw.values(chrom, region_start, region_end),
                    dtype=np.float64
                )
                offsets = starts - region_start 
                scores = buf[offsets]
                scores[np.isnan(scores)] = np.nan
                return pd.Series(np.round(scores, 4), index=group.index)
            except RuntimeError:
                pass 

        out = np.full(n, np.nan)
        for i, (s, e) in enumerate(zip(starts.tolist(), ends.tolist())):
            try:
                vals = phylo_bw.values(chrom, s, e)
                if vals and vals[0] is not None:
                    out[i] = round(vals[0], 4)
            except RuntimeError:
                pass
        return pd.Series(out, index=group.index)

    phyloP = var_df.groupby('Chromosome', sort=False, observed=False, group_keys=False).apply(get_phylop_chrom)
    return phyloP


def annotate_coding_effect(variants_df: pd.DataFrame, batch_size: int = 200, sleep: float = 0.5) -> pd.DataFrame:
    """
    Queries the Ensembl VEP REST API for coding consequence, amino acid changes, and impact.
    
    Note: This function filters internally to only request VEP data for variants 
    already annotated as coding (CDS, start/stop codon, exon).
    
    Args:
        variants_df (pd.DataFrame): Must contain 'Chromosome', 'Pos', 'Ref', 'Alt', 
                                    and 'gtf_feature'.
        batch_size (int): Number of variants to query in a single API POST request.
        sleep (float): Wait time in seconds between API requests to prevent rate limiting.
        
    Returns:
        pd.DataFrame: Contains columns 'all_effects', 'coding_effect', 'aa_change', 
                      and 'impact' aligned to the input variants_df index.
    """
    variants_df = standardize_vcf_columns(variants_df)
    
    CODING_FEATURES = {'CDS', 'start_codon', 'stop_codon', 'exon'}
    SERVER = "https://rest.ensembl.org/vep/human/region"
    HEADERS = {"Content-Type": "application/json", "Accept": "application/json"}

    coding_mask = variants_df['gtf_feature'].isin(CODING_FEATURES)
    coding = variants_df[coding_mask].copy()

    out = pd.DataFrame({
        'all_effects': pd.array(np.full(len(variants_df), np.nan), dtype=object),
        'coding_effect': pd.array(np.full(len(variants_df), np.nan), dtype=object),
        'aa_change': pd.array(np.full(len(variants_df), np.nan), dtype=object),
        'impact': pd.array(np.full(len(variants_df), np.nan), dtype=object),
    }, index=variants_df.index)

    if coding.empty:
        print("No coding variants found.")
        return out

    def to_vep(row: pd.Series) -> str:
        chrom = str(row['Chromosome']).replace('chr', '')
        pos = int(row['Pos'])
        ref = str(row['Ref'])
        alt = str(row['Alt'])

        while len(ref) > 1 and len(alt) > 1 and ref[0] == alt[0]:
            ref = ref[1:]
            alt = alt[1:]
            pos += 1

        if len(ref) == 1 and len(alt) == 1:
            return f"{chrom} {pos} . {ref} {alt} . . ."
        elif len(ref) == 1 and ref == alt[0]:
            return f"{chrom} {pos} {pos} - {alt[1:]} . . ."
        elif len(alt) == 1 and alt == ref[0]:
            return f"{chrom} {pos + 1} {pos + len(ref[1:])} {ref[1:]} - . . ."
        else:
            return f"{chrom} {pos} {pos + len(ref) - 1} {ref} {alt} . . ."

    coding['_vep'] = coding.apply(to_vep, axis=1)
    
    dupes = coding['_vep'].duplicated().sum()
    if dupes:
        print(f"Warning: {dupes} duplicate VEP strings — these will not match correctly")

    idx_list = coding.index.tolist()
    var_list = coding['_vep'].tolist()
    n_batches = (len(var_list) + batch_size - 1) // batch_size

    pbar = tqdm(
        range(0, len(var_list), batch_size),
        total=n_batches,
        desc="VEP annotation",
        unit="batch",
    )

    for i in pbar:
        batch_idx = idx_list[i : i + batch_size]
        batch_vars = var_list[i : i + batch_size]
        pbar.set_postfix({'variants': f"{min(i + batch_size, len(var_list))}/{len(var_list)}"})

        response = None
        for attempt in range(4):
            try:
                r = requests.post(
                    SERVER,
                    headers=HEADERS,
                    json={"variants": batch_vars},
                    timeout=60
                )
                if r.status_code == 200:
                    response = r.json()
                    break
                elif r.status_code == 429:
                    wait = 2 ** attempt
                    pbar.write(f"Rate limited — waiting {wait}s")
                    time.sleep(wait)
                else:
                    pbar.write(f"Batch {i//batch_size}: HTTP {r.status_code} — skipping")
                    break
            except requests.RequestException as e:
                pbar.write(f"Batch {i//batch_size} attempt {attempt}: {e}")
                time.sleep(2 ** attempt)

        if response is None:
            continue
            
        vep_to_idx = {vep_str: idx for idx, vep_str in zip(batch_idx, batch_vars)}

        for var_result in response:
            input_str = var_result.get('input', '').strip()
            idx = vep_to_idx.get(input_str)

            if idx is None:
                continue

            consequences = var_result.get('transcript_consequences', [])
            if not consequences:
                continue

            all_terms = sorted(
                {term for tc in consequences for term in tc.get('consequence_terms', [])},
                key=lambda t: CONSEQUENCE_RANK.get(t, 999)
            )
            if all_terms:
                out.at[idx, 'all_effects'] = ', '.join(all_terms)

            best_term = min(all_terms, key=lambda t: CONSEQUENCE_RANK.get(t, 999))
            out.at[idx, 'coding_effect'] = best_term
            out.at[idx, 'impact'] = IMPACT.get(best_term, np.nan)

            best_tc = min(
                consequences,
                key=lambda tc: min((CONSEQUENCE_RANK.get(t, 999) for t in tc.get('consequence_terms', [])), default=999)
            )
            aa_raw = best_tc.get('amino_acids')
            if aa_raw:
                parts = aa_raw.split('/')
                if len(parts) == 2:
                    ref_aa = parts[0].strip() or '-'
                    alt_aa = parts[1].strip() or '-'
                    if len(ref_aa) > 1: ref_aa = f"({ref_aa})"
                    if len(alt_aa) > 1: alt_aa = f"({alt_aa})"
                    out.at[idx, 'aa_change'] = f"{ref_aa}>{alt_aa}"

        time.sleep(sleep)
    pbar.close()

    effects = out['coding_effect'].dropna()
    if not effects.empty:
        print("\nCoding consequence breakdown:")
        print(effects.value_counts().rename_axis('consequence').reset_index(name='count').to_string(index=False))
        print("\nImpact breakdown:")
        print(out['impact'].dropna().value_counts().reindex(['HIGH', 'MODERATE', 'LOW', 'MODIFIER']).dropna().reset_index().rename(columns={'index': 'impact', 'impact': 'count'}).to_string(index=False))

    return out


def annotate_cadd_snv_scores(df: pd.DataFrame, bw_paths: Dict[str, str]) -> pd.DataFrame:
    """
    Extracts CADD PHRED scores for SNVs from allele-specific BigWig files.
    
    Filters out Indels automatically, mapping the remaining single nucleotide variants 
    to their corresponding target allele BigWig track.
    
    Args:
        df (pd.DataFrame): DataFrame containing 'Chromosome', 'Pos', 'Ref', and 'Alt'.
        bw_paths (Dict[str, str]): Dictionary mapping the alternate alleles to their 
                                   specific BigWig paths, e.g.:
                                   {'A': 'cadd1.7PhredSnvA.bw', 'C': '...'}
                                   
    Returns:
        pd.DataFrame: Contains only the filtered SNVs alongside their coordinates, 
                      alleles, and the resulting 'CADD_PHRED_score'.
    """
    df = standardize_vcf_columns(df)
    df = df[['Chromosome', 'Pos', 'Ref', 'Alt']].drop_duplicates() #selct dowwn to minmal set to spead up query
    
    bws = {allele: pyBigWig.open(path) for allele, path in bw_paths.items()}   
    
    is_snv = (df['Ref'].astype(str).str.len() == 1) & (df['Alt'].astype(str).str.len() == 1)
    df_snvs = df[is_snv].copy()
    df_snvs['Start'] = df_snvs['Pos'] - 1
    df_snvs['End'] = df_snvs['Pos']
    df_snvs['CADD_PHRED_score'] = np.nan   
    
    for (chrom, alt), group in df_snvs.groupby(['Chromosome', 'Alt'], observed=False):
        if alt not in bws:
            continue
            
        bw = bws[alt]
        valid_chroms = set(bw.chroms().keys())
        query_chrom = chrom if chrom in valid_chroms else f"chr{chrom}"
        
        if query_chrom not in valid_chroms:
            continue

        starts = group['Start'].to_numpy(dtype=np.int64)
        ends = group['End'].to_numpy(dtype=np.int64)
        span = int(ends.max() - starts.min())
        n = len(group)
        out = np.full(n, np.nan)

        if n > 1 and span < 10_000_000:     
            region_start = int(starts.min())
            region_end = int(ends.max())
            try:
                buf = np.array(
                    bw.values(query_chrom, region_start, region_end), 
                    dtype=np.float64
                )
                offsets = starts - region_start  
                scores = buf[offsets]
                out = np.round(scores, 4)
            except RuntimeError:
                pass 

        if np.isnan(out).all():
            for i, (s, e) in enumerate(zip(starts.tolist(), ends.tolist())):
                try:
                    vals = bw.values(query_chrom, s, e)
                    if vals and vals[0] is not None:
                        out[i] = round(vals[0], 4)
                except RuntimeError:
                    pass
                    
        df_snvs.loc[group.index, 'CADD_PHRED_score'] = out

    for bw in bws.values():
        bw.close()

    return df_snvs[['Chromosome', 'Ref', 'Alt', 'Pos', 'CADD_PHRED_score']]


def annotate_cardioid_enhancer_overlap(variants_df, regions_df):
    '''
    Gets all variants from variants_df that fall within regions_df and appends 
    their regulatory position and H3K27ac status.
    Expects CHROM, POS in variants_df and seqnames, start, end in regions_df.
    '''
    v_temp = standardize_vcf_columns(variants_df)
    v_temp['Start'] = v_temp['End'] - 1
    pr_variants = pr.PyRanges(v_temp)

    r_cols_to_keep = ['Chromosome', 'Start', 'End', 'regulatory_position', 'isH3K27ac', 'celltype']
    
    r_temp = standardize_vcf_columns(regions_df)
    r_temp = r_temp[r_cols_to_keep]
    pr_regions = pr.PyRanges(r_temp)

    overlapping_pr = pr_variants.join(pr_regions)
    final_variants = overlapping_pr.df

    cols_to_drop = [c for c in final_variants.columns if c.endswith('_b')]
    final_variants = final_variants.drop(columns=cols_to_drop, errors='ignore')

    final_variants = final_variants.sort_values('isH3K27ac', ascending=False)
    final_variants = final_variants.drop_duplicates(subset=['variant_id']).reset_index(drop=True)
    
    if 'celltype' in final_variants.columns:
        final_variants['celltype'] = final_variants['celltype'].astype(str)       
        final_variants['celltype'] = final_variants['celltype'].str.replace(r'[{}\'"]', '', regex=True)
        final_variants = final_variants.rename(columns={'celltype': 'celltypes_in_enhancer'})

    return final_variants.rename(columns = {"regulatory_position": "enhancer_regulatory_position", "isH3K27ac": "is_cardioid_H3K27ac"})