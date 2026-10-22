# alphagenome_api configs

| File | Tracked | Used by |
|---|---|---|
| `track_aggregations.yaml` | no (data-specific) | steps 02 (track prediction) and 03 (VEP); path key `ag_track_aggregations` |
| `track_aggregations.example.yaml` | yes | template |

`track_aggregations.yaml` maps group names to ontology CURIEs of AlphaGenome
tracks. The groups used in this project come from the track-matching notebook
(`analysis_notebooks/track_matching/alphagenome_track_matching.ipynb`); see
`PROVENANCE.md` for the version used in the analyses.
