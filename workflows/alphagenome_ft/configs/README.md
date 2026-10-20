# AlphaGenome targets configs

Targets configs define the fine-tuned heads (one per modality) and their BigWig
tracks. Real configs contain absolute paths to your data and are **gitignored**;
only `targets_config.example.yaml` is tracked.

Point `ag_targets_config` in `config/paths.yaml` to the config to use. Step 01
(`01_make_targets_config.sh`) generates one from your BigWig folders; it never
overwrites an existing config unless `OVERWRITE=1` is set. Older configs can be
converted with `normalize_targets_config.py`.

## Format

See `targets_config.example.yaml`. Per head:

| Key | Meaning |
|---|---|
| `dir_name` | informational: folder the tracks came from |
| `id` | head name (used in checkpoints and as `output_type` in VEP output) |
| `source` | `predefined` (required by `alphagenome_ft`) |
| `kind` | `atac` (ATAC, Cut&Tag) or `rna_seq` (stranded RNA) |
| `targets` | one entry per track: `path` (real file, not a symlink), `label`, `nonzero_mean` (informational in `alphagenome_ft` 0.1.5) |

The track **order** in a config must match the checkpoint it is used with:
labels in VEP and evaluation outputs are assigned by position. Step 01 sorts
tracks by case-sensitive file name and cuts labels at the norm tag (`_100M`).
Pass `MODALITIES="folder1 folder2 ..."` to step 01 to choose the heads and their order.
