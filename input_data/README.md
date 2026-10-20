# input_data/

Model inputs. **Not tracked in git** — only this README is. Populate the folders
locally (copy or symlink); the pipelines locate them via `config/paths.yaml`.

```
input_data/
├── raw_bam/                          # pseudobulked multiome ATAC/RNA and bulk Cut&Tag BAMs
├── AlphaGenome_Borzoi_input/         # base-resolution BigWigs (100M-normalised), output of workflows/input_prep
│   ├── intervals/                    # AG_{train,valid,test}_intervals_fold1_HOMO_SAPIENS.bed
│   ├── multiome_ATAC_bigwig[_deduplicated]/
│   ├── multiome_RNA_3prime_bigwig[_deduplicated]/
│   └── CutnTag_bigwig/
├── chrombpnet_input/                 # ChromBPNet ATAC inputs
│   └── multiome_ATAC/
├── metadata/
│   ├── GRCh38_gencode_release29_genome_fasta/   # genome.fa(.fai), genome.chrom.sizes, genome.dict
│   └── bw_nonzero_means.csv          # per-BigWig non-zero means (alphagenome_ft step 00)
└── variant_data/                     # VCF-style variant tables for VEP
```

Data can also live elsewhere: `config/paths.yaml` may point to any location
directly (preferred over symlinks into this folder).
