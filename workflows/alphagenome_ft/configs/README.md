# AlphaGenome targets configs

Targets configs define the fine-tuned heads (one per modality) and their BigWig
tracks. They contain absolute cluster paths and are therefore **gitignored**
(`*.yaml` in this folder); keep them in the shared cluster checkout.

Point `ag_targets_config` in `config/paths.yaml` to the config to use. Step 01
(`01_make_targets_config.sh`) generates a new one from the current BigWig folders;
it never overwrites an existing config unless `OVERWRITE=1` is set.

## Format

```yaml
heads:
- dir_name: multiome_ATAC_bigwig_deduplicated   # informational: source folder
  id: multiome_ATAC                              # head name (checkpoints, VEP output_type)
  source: predefined                             # required by alphagenome_ft
  kind: atac                                     # atac | rna_seq
  targets:                                       # one entry per track; order = model output order
  - path: /g/.../ATAC_Cardiomyocytes_100M_norm.bw   # real file, not a symlink
    label: ATAC_Cardiomyocytes
    nonzero_mean: 0.717                          # informational in alphagenome_ft 0.1.5 (not used)
```

## Config of record

| Config | Model (`/g/steinmetz/projects/then/AG/`) | Cell types | Reads | Tracks |
|---|---|---|---|---|
| `simple_merged.yaml` | `heads_only_deduplicated` → `heads_only_deduplicated_resumed` (final) | normal definition, 11 types (≥400 cells) | deduplicated | 9 Cut&Tag + 11 ATAC + 22 RNA |

`simple_merged.yaml` is identical to the training config
`src/analysis/AG_ft/configs/simple_merged_deduplicated.yaml` (same paths, labels and
order) plus the non-zero means. Its BigWigs live in
`seq2fun_train_input_baseres/{CutnTag_bigwig, multiome_ATAC_bigwig, multiome_RNA_3prime_bigwig}`:
these folders hold **deduplicated** tracks although their names lack the suffix.
The `*_deduplicated` folders next to them are a later input-prep output of the
simple (7 cell type) definition.

To rebuild this config from the BigWigs:
`MODALITIES="CutnTag_bigwig multiome_ATAC_bigwig multiome_RNA_3prime_bigwig" TARGETS_CONFIG=workflows/alphagenome_ft/configs/rebuilt.yaml bash workflows/alphagenome_ft/01_make_targets_config.sh`
(heads follow the given folder order; tracks are sorted alphabetically, labels cut at `_100M`).

Older lineage (not kept here): `all_modalities_ct_merged.yaml` → `heads_only`
(non-deduplicated reads), used for the VEP runs until 19 May 2026.

The track **order** in a config must match the checkpoint it is used with:
labels in VEP and evaluation outputs are assigned by position.
