"""Fine-tunes Borzoi (``human_rep0``, via gReLU) on the staged BigWig tracks.

One script for all stages:

* **Heads** (new run, no ``--prev_checkpoint``): pretrained Borzoi with a new output
  head, embedding frozen (``tune_on_dataset(..., freeze_embedding=True)``). Tracks are
  all normalised BigWigs (``*100M*``) in the staged ``*bigwig*`` folders, sorted by
  folder and file name.
* **Resumed stages** (``--prev_checkpoint``): continue from a checkpoint and unfreeze
  the embedding layers named in ``--unfreeze`` (decoder: ``unet_tower pointwise_conv``;
  transformer: ``transformer_tower unet_tower pointwise_conv``). Tracks and their order
  come from the checkpoint's task list.

RNA targets are trained on the Borzoi squashed scale
(:class:`~cardiomics_s2f.io.data_input.SelectiveSquashTransform`). After training, the
best checkpoint predicts the test intervals; the predictions (squashed scale, float16)
are saved as ``<prefix>_squashed_predictions.h5`` in ``--out_dir``.

Launched by ``01_finetune_heads.sh`` and ``02_finetune_decoder.sh`` (job
``jobs/finetune.slurm``); needs the Borzoi environment (``CS2F_ENV_BORZOI``).
"""

import argparse
import gc
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import grelu.lightning

from cardiomics_s2f.finetuning.borzoi.datasets import StreamingBigWigSeqDataset
from cardiomics_s2f.finetuning.borzoi.training import (
    find_checkpoint_file,
    find_normalized_bigwig_tracks,
    tracks_from_checkpoint,
    unfreeze_embedding_layers,
)
from cardiomics_s2f.io.data_input import SelectiveSquashTransform
from cardiomics_s2f.io.data_output import save_borzoi_predictions_with_fallback
from cardiomics_s2f.utils import log_stage_resources

# Borzoi human_rep0 geometry (used for new runs; resumed runs read it from the checkpoint)
SEQ_LEN = 524288
LABEL_LEN = 196608
BIN_SIZE = 32
GENOME = "hg38"


def get_parser() -> argparse.ArgumentParser:
    """Creates the argument parser (separate so Sphinx can document it).

    Returns:
        argparse.ArgumentParser: The configured argument parser.
    """
    parser = argparse.ArgumentParser(description="Borzoi Generic Finetuning Pipeline")

    # Core I/O
    parser.add_argument("--out_dir", type=str, required=True, help="Output directory for checkpoints and predictions")
    parser.add_argument("--prefix", type=str, default="Borzoi_run", help="Prefix for the prediction file")
    parser.add_argument("--prev_checkpoint", type=str, default=None,
                        help="Checkpoint (.ckpt file, or a run folder with exactly one epoch*.ckpt) to resume from; "
                             "omit to start from the pretrained Borzoi (heads only)")
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Staged input directory containing the *bigwig* folders")
    parser.add_argument("--train_bed", type=str, required=True, help="Path to the train intervals BED")
    parser.add_argument("--val_bed", type=str, required=True, help="Path to the validation intervals BED")
    parser.add_argument("--test_bed", type=str, required=True, help="Path to the test intervals BED")

    # Hyperparameters
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--epochs", type=int, default=15, help="Number of maximum epochs")
    parser.add_argument("--loss_weight", type=float, default=0.2, help="Multinomial loss weight")
    parser.add_argument("--unfreeze", nargs='*', default=[],
                        help="Layer name parts to unfreeze when resuming (e.g. unet_tower pointwise_conv). "
                             "Leave empty to keep the embedding frozen.")
    return parser


def main(args: argparse.Namespace) -> None:
    """Trains, then predicts the test intervals with the best checkpoint.

    Args:
        args (argparse.Namespace): Parsed command-line arguments.
    """
    global_start = time.time()
    run_type = "Base Initialization (Heads only)" if args.prev_checkpoint is None else "Resuming from Checkpoint"
    last_time = log_stage_resources(f"Script Started - {run_type}", global_start)

    out_path = Path(args.out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    num_gpus = torch.cuda.device_count()

    # 1. Load model and select tracks
    if args.prev_checkpoint is None:
        seq_len, label_len, bin_size, genome = SEQ_LEN, LABEL_LEN, BIN_SIZE, GENOME
        tracks = find_normalized_bigwig_tracks(args.data_dir)
    else:
        checkpoint = find_checkpoint_file(args.prev_checkpoint)
        print(f"Loading checkpoint from: {checkpoint}", flush=True)
        model = grelu.lightning.LightningModel.load_from_checkpoint(checkpoint, weights_only=False)
        seq_len = model.data_params["train"]['seq_len']
        label_len = model.data_params["train"]['label_len']
        bin_size = model.data_params["train"]['bin_size']
        genome = model.data_params["train"]['genome']
        tracks = tracks_from_checkpoint(model, args.data_dir)
    last_time = log_stage_resources("Model Loaded", last_time)

    # 2. Intervals and datasets
    print("Loading precomputed interval BED files...", flush=True)
    train = pd.read_csv(args.train_bed, sep="\t")
    val = pd.read_csv(args.val_bed, sep="\t")
    test = pd.read_csv(args.test_bed, sep="\t")

    rna_idx = tracks[tracks["modality"].str.contains("rna", case=False)].index.to_list()
    squash_transform = SelectiveSquashTransform(transform="Borzoi", rna_task_indices=rna_idx)

    dataset_kwargs = {
        "bw_files": tracks["path"].to_list(),
        "tasks": tracks["file_name"].to_list(),
        "bin_size": bin_size,
        "label_len": label_len,
        "label_aggfunc": "sum",
        "genome": genome,
        "label_transform_func": squash_transform,
    }
    train_ds = StreamingBigWigSeqDataset(
        intervals=train, rc=True, max_seq_shift=3, augment_mode="random", seed=0, **dataset_kwargs
    )
    val_ds = StreamingBigWigSeqDataset(
        intervals=val, rc=True, max_seq_shift=3, augment_mode="serial", seed=0, **dataset_kwargs
    )
    del train, val
    gc.collect()
    last_time = log_stage_resources("Datasets Initialized", last_time)

    # 3. Configure & train
    if args.prev_checkpoint is None:
        model_params = {
            'model_type': 'BorzoiPretrainedModel',
            'n_tasks': tracks.shape[0],
            'crop_len': int((seq_len - label_len) / (2 * bin_size)),
            'n_transformers': 8,
        }
        train_params = {
            'task': 'regression',
            'loss': 'poisson_multinomial',
            'logger': 'csv',
            'total_weight': args.loss_weight,
            'lr': args.lr,
            'batch_size': 2,
            'max_epochs': args.epochs,
            'devices': list(range(num_gpus)),
            'num_workers': num_gpus * 4,
            'save_dir': out_path,
            'checkpoint': True,
        }
        model = grelu.lightning.LightningModel(model_params, train_params)
        print(f"Starting Base Training with LR: {args.lr}, Weight: {args.loss_weight}, Epochs: {args.epochs}", flush=True)
        trainer = model.tune_on_dataset(train_ds, val_ds, freeze_embedding=True)
    else:
        unfreeze_embedding_layers(model, args.unfreeze)

        model.train_params['lr'] = args.lr
        model.train_params['total_weight'] = args.loss_weight
        model.train_params['max_epochs'] = args.epochs
        model.train_params['save_dir'] = str(out_path)
        # The checkpoint stores the devices of the previous stage (e.g. 8 GPUs);
        # use the GPUs of this job instead.
        model.train_params['devices'] = list(range(num_gpus))
        model.train_params['num_workers'] = num_gpus * 4

        # Re-initialize loss to catch new total_weight if changed
        model.initialize_loss()

        print(f"Resuming Training with LR: {args.lr}, Weight: {args.loss_weight}, Epochs: {args.epochs}", flush=True)
        trainer = model.train_on_dataset(train_ds, val_ds)

    best_checkpoint = trainer.checkpoint_callback.best_model_path
    print(f"Best Checkpoint Saved At: {best_checkpoint}", flush=True)

    del train_ds, val_ds
    gc.collect()
    last_time = log_stage_resources("Training Loop Completed", last_time)

    # 4. Test inference with the best checkpoint
    print("Starting test inference...", flush=True)
    test_ds = StreamingBigWigSeqDataset(
        intervals=test, rc=False, max_seq_shift=0, augment_mode="serial", seed=0, **dataset_kwargs
    )

    del model, trainer
    gc.collect()

    model = grelu.lightning.LightningModel.load_from_checkpoint(best_checkpoint, weights_only=False)
    preds = model.predict_on_dataset(test_ds, devices=[0], num_workers=4, batch_size=2, return_df=False)

    # Compress data type for storage (fine on the squashed scale)
    preds = preds.astype(np.float16)

    save_borzoi_predictions_with_fallback(
        preds=preds, out_dir=out_path,
        file_prefix=f"{args.prefix}_squashed_predictions",
        intervals_df=test, tracks_df=tracks,
    )

    log_stage_resources("Testing & Inference Completed", last_time)
    log_stage_resources("TOTAL SCRIPT EXECUTION", global_start)


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    main(args)
