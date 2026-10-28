# models/

Trained / fine-tuned model checkpoints. **Not tracked in git** only this README is.
The pipelines locate checkpoints via `config/paths.yaml`.

```
models/
├── AlphaGenome_finetuned/   # alphagenome_ft head checkpoints (workflows/alphagenome_ft step 02)
├── Borzoi_finetuned/        # gReLU/Borzoi fine-tuned checkpoints
└── chromBPNet_trained/      # ChromBPNet bias + full models per pseudobulk
```
