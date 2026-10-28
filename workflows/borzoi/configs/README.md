# borzoi configs

| File | Tracked | Used by |
|---|---|---|
| `bigwig_folders.txt` | no (data-specific) | step 01 (heads): BigWig folders of `bigwig_dir` staged to the GPU node |
| `bigwig_folders.example.txt` | yes | template |

Step 02 (decoder) stages the whole `bigwig_dir` and takes its tracks, in order, from the
checkpoint's task list. A file name must therefore occur in only one `*bigwig*` folder of
`bigwig_dir`; the training script warns when a folder and its `*_deduplicated` version
are both present and stops on ambiguous file names. See `PROVENANCE.md` for the folders
of the final model.
