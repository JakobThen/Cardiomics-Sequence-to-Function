from __future__ import annotations

import argparse
from pathlib import Path

from alphagenome_ft import create_model_with_heads, load_checkpoint
from alphagenome_ft.finetune.config import load_targets_config, prepare_head_specs
from alphagenome_ft.finetune.data import prepare_intervals_from_fold, BigWigDataModule, build_fasta_index
from alphagenome_ft.finetune.train import register_predefined_heads, train
from alphagenome.models import dna_model as ag_dna_model

def main():
    parser = argparse.ArgumentParser(description="AlphaGenome Finetuning Script")
    
    # -------------------------------------------------------------------------------------------
    # Core I/O Arguments
    # -------------------------------------------------------------------------------------------
    parser.add_argument("--fasta_path", type=str, required=True, help="Path to the genome.fa file")
    parser.add_argument("--config", type=str, required=True, help="Path to the target YAML config file")
    parser.add_argument("--base_dir", type=str, required=True, help="Base directory for resolving config paths (e.g., the local tmp dir)")
    parser.add_argument("--in_checkpoint_dir", type=str, required=True, help="Directory to save checkpoints")
    parser.add_argument("--out_checkpoint_dir", type=str, required=True, help="Directory to save checkpoints")
    
    # -------------------------------------------------------------------------------------------
    # Data & Model Options
    # -------------------------------------------------------------------------------------------
    parser.add_argument("--fold", type=str, default="1", help="Data fold to use (0, 1, 2, 3)")
    parser.add_argument("--window_size", type=int, default=1_048_576, help="Window size in bp (default: 1 Mbp)")
    parser.add_argument("--organism", type=str, default="HOMO_SAPIENS", help="Target organism")
    parser.add_argument("--model_version", type=str, default="fold_1", help="Base model version (e.g., fold_1, all_folds)")
    
    # -------------------------------------------------------------------------------------------
    # Training Hyperparameters
    # -------------------------------------------------------------------------------------------
    parser.add_argument("--batch_size", type=int, default=16, help="Global batch size (must be divisible by N_GPUs)")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-2, help="Weight decay")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--patience", type=int, default=5, help="Early stopping patience")
    parser.add_argument("--min_delta", type=float, default=0.0, help="Early stopping min delta")

    args = parser.parse_args()

    # Derived variables
    FASTA_PATH = Path(args.fasta_path)
    TARGETS_CONFIG_PATH = Path(args.config)
    BASE_DIR = Path(args.base_dir)
    IN_CHECKPOINT_DIR = Path(args.in_checkpoint_dir)
    OUT_CHECKPOINT_DIR = Path(args.out_checkpoint_dir)

    # Standard fixed training options
    SHUFFLE = True
    DROP_LAST = True
    MAX_TRAIN_STEPS = None
    VERBOSE = True
    HEADS_ONLY = True
    BEST_METRIC = "valid_loss"
    BEST_METRIC_MODE = "min"

    # ------------------------------------------------------------------------------------
    # Validate I/O
    # ------------------------------------------------------------------------------------
    FASTA_INDEX_PATH = Path(f"{FASTA_PATH}.fai")
    if FASTA_PATH.exists() and not FASTA_INDEX_PATH.exists():
        print(f"Building FASTA index: {FASTA_INDEX_PATH}", flush=True)
        build_fasta_index(FASTA_PATH)
    elif FASTA_INDEX_PATH.exists():
        print(f"Found FASTA index: {FASTA_INDEX_PATH}", flush=True)
    else:
        print("Skip FASTA indexing because FASTA is missing.", flush=True)

    for path in [FASTA_PATH, TARGETS_CONFIG_PATH]:
        if not path.exists():
            raise FileNotFoundError(f"Required file not found: {path}")

    OUT_CHECKPOINT_DIR = OUT_CHECKPOINT_DIR.resolve()
    OUT_CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    print("Input validation passed.", flush=True)

    # ------------------------------------------------------------------------------------
    # Model config
    # ------------------------------------------------------------------------------------
    config_dict = load_targets_config(
        TARGETS_CONFIG_PATH,
        base_dir=BASE_DIR
    )

    # Build head specifications from the config.
    head_specs = prepare_head_specs(config_dict)
    register_predefined_heads(head_specs)

    # Create dataframe of intervals for the specified fold and window size
    intervals = prepare_intervals_from_fold(
        fold=args.fold,
        window_size=args.window_size,
        organism=args.organism,
    )

    print("Heads:", [spec.head_id for spec in head_specs], flush=True)
    for split in ("train", "valid", "test"):
        print(f"{split}: {len(intervals.get(split, []))} intervals", flush=True)

    # Create model with new heads
    print("Loading model onto GPU...", flush = True)
    model = load_checkpoint(
        IN_CHECKPOINT_DIR / "last",
        base_model_version=args.model_version,
        init_seq_len=args.window_size
    )
    print("Model loaded successfully.", flush = True)

    if HEADS_ONLY:
        model.freeze_backbone()
        print("Model backbone frozen. Fine-tuning heads only.", flush=True)
    else:
        print("Model backbone UNFROZEN. Running full fine-tuning.", flush=True)

    print("Model ready.", flush=True)

    # Create data module
    data_module = BigWigDataModule(
        intervals=intervals,
        fasta_path=FASTA_PATH,
        head_specs=head_specs,
        batch_size=args.batch_size,
        shuffle=SHUFFLE,
        drop_last=DROP_LAST,
    )

    print("Data module ready. Starting training...", flush=True)

    # ------------------------------------------------------------------------------------
    # Start Model training
    # ------------------------------------------------------------------------------------
    train(
        model,
        data_module,
        head_specs,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        num_epochs=args.epochs,
        seed=args.seed,
        max_train_steps=MAX_TRAIN_STEPS,
        heads_only=HEADS_ONLY,
        checkpoint_dir=OUT_CHECKPOINT_DIR,
        organism=args.organism,
        best_metric=BEST_METRIC,
        best_metric_mode=BEST_METRIC_MODE,
        early_stopping_patience=args.patience,
        early_stopping_min_delta=args.min_delta,
        verbose=VERBOSE,
    )

if __name__ == "__main__":
    main()