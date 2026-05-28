#Code by Jakob Then
#Script to run batch VEP from a VCF file using a custom AG model
#Using center mask scorering for CutnTag and ATAC and exon mask scoring for RNA

import os
import numpy as np
import pandas as pd
import h5py
from pathlib import Path

import jax
import jax.numpy as jnp
import jax.tree_util as jtu
from jax.experimental import mesh_utils
from jax.sharding import Mesh, PartitionSpec as P

from alphagenome_ft import load_checkpoint
from alphagenome_ft.finetune.config import load_targets_config, prepare_head_specs
from alphagenome_ft.finetune.train import register_predefined_heads
from alphagenome.models import dna_model as ag_dna_model
from alphagenome.data import genome
from alphagenome_research.io import fasta as fasta_lib
from alphagenome_research.io import genome as genome_io
from alphagenome_research.model import one_hot_encoder
from alphagenome_research.model import variant_scoring
from alphagenome_research.model.variant_scoring.variant_scoring import IndelMask, align_alternate
from alphagenome_research.model.variant_scoring.gene_mask_extractor import (
    GeneMaskExtractor, GeneMaskType, GeneQueryType
)


FASTA_PATH = Path("/g/steinmetz/calfonso/shared/reference_genomes/GRCh38_gencode_release29/genome_fasta/genome.fa")
CHROM_SIZES_PATH = Path("/g/steinmetz/calfonso/shared/reference_genomes/GRCh38_gencode_release29/genome_fasta/genome.chrom.sizes")
TARGETS_CONFIG_PATH = Path("/g/steinmetz/projects/variant2function_project/src/analysis/AG_ft/configs/all_modalities_ct_merged.yaml")
CHECKPOINT_DIR = Path("/g/steinmetz/projects/then/AG/heads_only")

#INPUT_VCF = Path("/g/steinmetz/projects/variant2function_project/data/processed/variant.selection/VCF_SDR007A_SDR010ABC_Comb.csv")
INPUT_VCF = Path("/g/steinmetz/projects/variant2function_project/data/processed/variant.selection/VCF_all_Richter_vars_REST.csv")

OUTPUT_PATH = Path("/g/steinmetz/projects/variant2function_project/results/variant.selection/VEP/AG_ft_VEP_out")

WINDOW_SIZE = 1_048_576
ORGANISM = "HOMO_SAPIENS"
ORGANISM_IDX = 0  # Human
MODEL_VERSION = "fold_1"
BATCH_SIZE = 24

OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
BASE_NAME = INPUT_VCF.stem

# 2. load variant vcf file
variants_df = pd.read_csv(INPUT_VCF)

missing_cols = {"variant_id", "REF", "ALT", "CHROM", "POS"} - set(variants_df.columns)
if missing_cols:
    raise ValueError(f"Missing required columns: {missing_cols}")
variants_df = variants_df[["variant_id", "CHROM", "POS", "REF", "ALT" ]].drop_duplicates() # remove redundant celltype info etc
print("VCF file loaded!", flush = True)

chrom_size_df = pd.read_csv(CHROM_SIZES_PATH,sep="\t",names=["chrom", "end"])
CHROM_SIZES = dict(zip(chrom_size_df['chrom'], chrom_size_df['end']))

# 3. define helper functions
#define new varaint effect batch genrate that works with our heads
def variant_batch_generator(df, fasta_path, batch_size, window_size, organism="HOMO_SAPIENS"):
    """Yields batches of (REF_one_hot, ALT_one_hot, indel_masks, variant_ids)"""
    extractor = fasta_lib.FastaExtractor(str(fasta_path))
    encoder = one_hot_encoder.DNAOneHotEncoder(dtype=np.float32)
    
    half_window = window_size // 2

    batch_ref, batch_alt, batch_masks, batch_ids = [], [], [], []
    batch_variants, batch_intervals = [], [] 
    
    for row in df.itertuples(index=False):
        chrom = str(row.CHROM)
        pos = int(row.POS) 
        
        # Create var objects
        variant = genome.Variant(
            chromosome=chrom, 
            position=pos, 
            reference_bases=str(row.REF).upper(), 
            alternate_bases=str(row.ALT).upper()
        )
        
        # 0-based interval centered on the variant
        ideal_start = pos - 1 - half_window
        
        # BOUNDARY LOGIC: Shift window to stay within chromosome while keeping size constant
        max_chrom_len = CHROM_SIZES.get(chrom, float('inf'))
        if ideal_start < 0:
            start = 0 
        elif ideal_start + window_size > max_chrom_len:
            start = max(0, max_chrom_len - window_size) 
        else:
            start = ideal_start

        interval = genome.Interval(start=start, end=start + window_size, chromosome=chrom)

        # AG Sequence Extraction (handles all indel shifting & padding)
        ref_seq_string, alt_seq_string = genome_io.extract_variant_sequences(
            interval, variant, extractor
        )
        
        # AG Indel Masks (for post-hoc alignment)
        indel_mask = IndelMask.from_variant(variant, interval)
        
        # Encode 
        ref_onehot = encoder.encode(ref_seq_string)
        alt_onehot = encoder.encode(alt_seq_string)
                
        # Store
        batch_ref.append(ref_onehot)
        batch_alt.append(alt_onehot)
        batch_masks.append(indel_mask)
        batch_ids.append(row.variant_id)
        batch_variants.append(variant)
        batch_intervals.append(interval)
        
        if len(batch_ref) == batch_size:
            yield np.stack(batch_ref), np.stack(batch_alt), batch_masks, batch_variants, batch_intervals, batch_ids, batch_size
            batch_ref, batch_alt, batch_masks, batch_variants, batch_intervals, batch_ids = [], [], [], [], [], []
            
    # if len(batch_ref) > 0:
    #     yield np.stack(batch_ref), np.stack(batch_alt), batch_masks, batch_variants, batch_intervals, batch_ids
    #ensure that last batch is padded to tobacth size if it contains less vairiants to not chrash the device mesh
    if len(batch_ref) > 0:
        actual_size = len(batch_ref)
        while len(batch_ref) < batch_size:  # pad to multiple of num_devices
            batch_ref.append(batch_ref[0])
            batch_alt.append(batch_alt[0])
            batch_masks.append(batch_masks[0])
            batch_variants.append(batch_variants[0])
            batch_intervals.append(batch_intervals[0])
            batch_ids.append("__PAD__")
        yield np.stack(batch_ref), np.stack(batch_alt), batch_masks, batch_variants, batch_intervals, batch_ids, actual_size

# def extract_resolution(raw_p, resolution=1):
#     if isinstance(raw_p, dict) or hasattr(raw_p, 'keys'):
#         if resolution in raw_p:
#             return np.array(raw_p[resolution])
#         elif 'predictions' in raw_p:
#             return np.array(raw_p['predictions'])
#         else:
#             return np.array(list(raw_p.values())[0])
#     return np.array(raw_p)


# Setup RNA Exon Extractor
GTF_URL = 'https://storage.googleapis.com/alphagenome/reference/gencode/hg38/gencode.v46.annotation.gtf.gz.feather'
print(f"Loaded GTF annotations from {GTF_URL}...", flush=True)
gtf_df = pd.read_feather(GTF_URL)

exon_extractor = GeneMaskExtractor(
    gtf=gtf_df,
    gene_mask_type=GeneMaskType.EXONS,
    gene_query_type=GeneQueryType.INTERVAL_CONTAINED,
    filter_protein_coding=False 
)

# 4. Mesh Setup
config_dict = load_targets_config(TARGETS_CONFIG_PATH, base_dir= Path("/g/steinmetz/projects/variant2function_project/src/analysis/AG_ft/configs"))
head_specs = prepare_head_specs(config_dict)
register_predefined_heads(head_specs)

model = load_checkpoint(CHECKPOINT_DIR / "best", base_model_version=MODEL_VERSION, init_seq_len=WINDOW_SIZE)
print("Loaded model onto GPU", flush=True)


# Prepare Track Metadata from config_dict
tracks_metadata = {}
for head in config_dict.get('heads', []):
    tracks_metadata[head['id']] = pd.DataFrame(head.get('targets', []))

organism_enum = getattr(ag_dna_model.Organism, ORGANISM)
strand_reindexing = jax.device_put(model._metadata[organism_enum].strand_reindexing, model._device_context._device)

num_devices = jax.local_device_count()
print(f"Setting up {num_devices}-GPU Device Mesh...", flush=True)
mesh = Mesh(mesh_utils.create_device_mesh((num_devices,)), axis_names=('data',))
data_sharding, replicated_sharding = P('data'), P()

@jax.jit(
    in_shardings=(replicated_sharding, replicated_sharding, data_sharding, data_sharding, data_sharding, replicated_sharding),
    out_shardings=replicated_sharding 
)
def parallel_predict(params, state, seq_batch, org_batch, mask_batch, strand_idx):
    raw_preds = model._predict(params, state, seq_batch, org_batch, negative_strand_mask=mask_batch, strand_reindexing=strand_idx)
    return {head.head_id: raw_preds[head.head_id] for head in head_specs}

batched_align_alternate = jax.jit(jax.vmap(align_alternate, in_axes=(0, 0))) #make align alternate function batch callable

# 5. VEP Evaluation, Aggregation & Long DataFrame Generation
# We will collect a list of dictionaries to build our long DataFrame
vep_records = []

print("Starting Variant Effect Prediction and Aggregation...", flush=True)
csv_path = OUTPUT_PATH / f"VEP_AG_ft_{MODEL_VERSION}_{BASE_NAME}.csv"
first_write = True # Flag to track whether we need to write the CSV headers

with model._device_context, jax.set_mesh(mesh):
    for batch_idx, (ref_seq, alt_seq, batch_masks, batch_variants, batch_intervals, var_ids, actual_size) in enumerate(variant_batch_generator(variants_df, FASTA_PATH, BATCH_SIZE, WINDOW_SIZE)):
        
        current_bsz = ref_seq.shape[0]
        
        org_idx = jnp.full((current_bsz,), ORGANISM_IDX, dtype=jnp.int32)
        neg_mask = jnp.zeros((current_bsz,), dtype=jnp.bool_)
        
        org_idx_sharded = jax.device_put(org_idx, data_sharding)
        neg_mask_sharded = jax.device_put(neg_mask, data_sharding)
        
        ref_sharded = jax.device_put(ref_seq, data_sharding)
        preds_ref = parallel_predict(model._params, model._state, ref_sharded, org_idx_sharded, neg_mask_sharded, strand_reindexing)

        alt_sharded = jax.device_put(alt_seq, data_sharding)
        preds_alt = parallel_predict(model._params, model._state, alt_sharded, org_idx_sharded, neg_mask_sharded, strand_reindexing)

        # stack the masks for GPU processing
        batched_masks = jtu.tree_map(lambda *x: jnp.stack(x), *batch_masks)

        #DEBUD check:
        if batch_idx == 0:
            print(f"Batched_masks shape: {jax.tree_util.tree_map(lambda x: x.shape, batched_masks)}", flush = True)
            print("Expected: IndelMask(variant_alt_mask=(8, 1048576), ...", flush = True)

        # # ALIGN, MASK, AGGREGATE
        # for head_spec in head_specs:
        #     head_name = head_spec.head_id
        #     p_ref_batch = extract_resolution(preds_ref[head_name], resolution=1)
        #     p_alt_batch = extract_resolution(preds_alt[head_name], resolution=1)
        #     #p_ref_batch = np.array(preds_ref[head_name].get('predictions', list(preds_ref[head_name].values())[0]) if isinstance(preds_ref[head_name], dict) else preds_ref[head_name])
        #     #p_alt_batch = np.array(preds_alt[head_name].get('predictions', list(preds_alt[head_name].values())[0]) if isinstance(preds_alt[head_name], dict) else preds_alt[head_name])
        #     seq_len = p_ref_batch.shape[1]
        #     num_tracks = p_ref_batch.shape[2] 
        #     # Grab track metadata if we have it (to populate track_name, etc.)
        #     head_metadata = tracks_metadata.get(head_name, pd.DataFrame())
            
        #    for i in range(actual_size):
        #       p_ref_i = p_ref_batch[i]
        #       p_alt_i = p_alt_batch[i]
        #       variant_id = var_ids[i]
        #       interval_str = f"{batch_intervals[i].chromosome}:{batch_intervals[i].start}-{batch_intervals[i].end}"    
        #       # ALIGN: Shift ALT to match REF
        #       p_alt_aligned_i = np.array(align_alternate(jnp.array(p_alt_i), batch_masks[i]))
                
                # RNA LOGIC (Exon Mask, Mean, Natural Log)

        # ALIGN, MASK, AGGREGATE
        for head_spec in head_specs:
            head_name = head_spec.head_id
            
            p_ref_batch_jax = preds_ref[head_name]['predictions_1bp']
            p_alt_batch_jax = preds_alt[head_name]['predictions_1bp']
            
            # batched alignment on GPU
            p_alt_aligned_batch_jax = batched_align_alternate(p_alt_batch_jax, batched_masks)
            p_ref_batch = np.array(p_ref_batch_jax)
            p_alt_aligned_batch = np.array(p_alt_aligned_batch_jax)
            
            seq_len = p_ref_batch.shape[1]
            num_tracks = p_ref_batch.shape[2]
            
            head_metadata = tracks_metadata.get(head_name, pd.DataFrame())
            
            # Iterate through actual batch size
            for i in range(actual_size):
                p_ref_i = p_ref_batch[i]
                p_alt_aligned_i = p_alt_aligned_batch[i] # Already aligned!
                variant_id = var_ids[i]
                interval_str = f"{batch_intervals[i].chromosome}:{batch_intervals[i].start}-{batch_intervals[i].end}"

                # RNA LOGIC (Exon Mask, Mean, Natural Log)
                if "RNA" in head_name.upper():
                    gene_mask_2d, annotations = exon_extractor.extract(
                        interval=batch_intervals[i], 
                        variant=batch_variants[i]
                    )

                    if batch_idx == 0:
                        print(f"gene_mask_2d shape: {gene_mask_2d.shape}", flush=True)
                        print(f"p_ref_i shape: {p_ref_i.shape}", flush=True)
                        assert gene_mask_2d.shape[0] == p_ref_i.shape[0], \
                            f"Mask/prediction length mismatch: {gene_mask_2d.shape[0]} vs {p_ref_i.shape[0]}"
                    
                    num_genes = gene_mask_2d.shape[1]
                    if num_genes == 0:
                        continue # Skip entirely if no genes are found in this window
                        
                    for g in range(num_genes):
                        g_mask = gene_mask_2d[:, g]
                        g_id = annotations['gene_id'].iloc[g]
                        g_name = annotations['gene_name'].iloc[g]
                        
                        mask_sum = g_mask.sum()
                        if mask_sum == 0:
                            continue # skip gene with no exons or invalid sequnce in window

                        ref_mean = np.sum(p_ref_i[g_mask, :], axis=0) / mask_sum
                        alt_mean = np.sum(p_alt_aligned_i[g_mask, :], axis=0) / mask_sum
                        
                        # Score: ln(alt_mean + 0.001) - ln(ref_mean + 0.001)
                        scores = np.log(alt_mean + 0.001) - np.log(ref_mean + 0.001)
                        
                        for trk in range(num_tracks):
                            trk_name = head_metadata.iloc[trk].get('label', f"Track_{trk}") if not head_metadata.empty else f"Track_{trk}"
                            
                            vep_records.append({
                                'variant_id': variant_id,
                                'scored_interval': interval_str,
                                'gene_id': g_id,
                                'gene_name': g_name,
                                'output_type': head_name,
                                'track_name': trk_name,
                                'score': scores[trk]
                            })
                            
                # ATAC / CUT&TAG LOGIC (Variant-Centric, Sum, Log2)
                else:
                    # Determine width based on assay type
                    if "ATAC" in head_name.upper():
                        half_width = 250  # 501 total width
                    elif "CUTNTAG" in head_name.upper() or "CNT" in head_name.upper() or "CUT&TAG" in head_name.upper() or "C&T" in head_name.upper():
                        half_width = 1000 # 2001 total width
                    else:
                        half_width = 250  # Default fallback
                        
                    variant_center = batch_variants[i].position - 1 - batch_intervals[i].start
                    
                    # Ensure we don't slice out of bounds
                    start_idx = max(0, variant_center - half_width)
                    end_idx = min(seq_len, variant_center + half_width + 1)
                    
                    spatial_mask = np.zeros(seq_len, dtype=bool)
                    spatial_mask[start_idx:end_idx] = True
                    
                    ref_sum = np.sum(p_ref_i[spatial_mask, :], axis=0)
                    alt_sum = np.sum(p_alt_aligned_i[spatial_mask, :], axis=0)
                    
                    # Score: log2(alt_sum + 1) - log2(ref_sum + 1)
                    scores = np.log2(alt_sum + 1) - np.log2(ref_sum + 1)
                    
                    for trk in range(num_tracks):
                        trk_name = head_metadata.iloc[trk].get('label', f"Track_{trk}") if not head_metadata.empty else f"Track_{trk}"
                        
                        vep_records.append({
                            'variant_id': variant_id,
                            'scored_interval': interval_str,
                            'gene_id': None,
                            'gene_name': None,
                            'output_type': head_name,
                            'track_name': trk_name,
                            'score': scores[trk]
                        })

        if (batch_idx + 1) % 25 == 0:
            if vep_records:
                #write csv fiel in chunks to save memory every 25 batches
                chunk_df = pd.DataFrame(vep_records)
                chunk_df.to_csv(csv_path, mode='a', header=first_write, index=False) # Append to CSV, writing headers only the first time
                first_write = False
                vep_records.clear() # clear mem
            print(f"Processed {batch_idx + 1} batches...", flush=True)

#write remaining batches
if vep_records:
    chunk_df = pd.DataFrame(vep_records)
    chunk_df.to_csv(csv_path, mode='a', header=first_write, index=False)
    vep_records.clear()


#SAving is done chunked to save memory!
# # Save as Long DataFrame
# print(f"\nCompiling long DataFrame and saving...", flush=True)
# final_df = pd.DataFrame(vep_records)
# # Save as csv
# csv_path = OUTPUT_PATH / f"VEP_AG_ft_{MODEL_VERSION}_{BASE_NAME}.csv"
# final_df.to_csv(csv_path, index=False)
# # Save as Parquet 
# parquet_path = OUTPUT_PATH / f"VEP_AG_ft_{MODEL_VERSION}_{BASE_NAME}.parquet"
# final_df.to_parquet(parquet_path, index=False)

print(f"Saved all records to:")
print(f" -> {csv_path}")
# print(f" -> {parquet_path}")
print("Variant Effect Prediction complete!", flush=True)