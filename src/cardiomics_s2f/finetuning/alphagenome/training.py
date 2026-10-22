"""
AlphaGenome Data-Parallel Training
==================================

Multi-GPU fine-tuning loop for AlphaGenome heads.

``train_data_parallel`` is the training loop of ``alphagenome_ft.finetune.train.train``
(alphagenome-ft 0.1.5) with data parallelism added: every batch is sharded across all
local GPUs, while parameters, model state and optimizer state are replicated. This is
the version that trained the models of this project (it was originally a patch inside
the installed ``alphagenome_ft`` package). Changes to the original:

* device mesh from :func:`~cardiomics_s2f.finetuning.alphagenome.inference.setup_data_parallel_mesh`;
* ``train_step`` and ``eval_step`` are compiled with explicit input/output shardings;
* batches are placed with ``jax.device_put(batch, data_sharding)`` and the loop runs
  inside ``jax.set_mesh(mesh)``;
* with ``verbose=True``, the sharded train step is traced once on the first train
  batch (``jax.eval_shape``) before training starts.

The loop uses private attributes of alphagenome-ft 0.1.5 (``model._params``,
``data_module._intervals`` ...), so the extra pins that version.

Adapted from alphagenome-ft (https://github.com/genomicsxai/alphagenome_ft),
Copyright (c) 2026 Alan Murphy, released under the MIT License:

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction, including without limitation the rights
    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in all
    copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
    SOFTWARE.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Mapping, Sequence

import jax
import numpy as np
import optax
from alphagenome.models import dna_model as ag_dna_model
from alphagenome_research.model import dna_model as research_dna_model

from alphagenome_ft.custom_model import CustomAlphaGenomeModel
from alphagenome_ft.finetune.config import HeadSpec
from alphagenome_ft.finetune.data import BigWigDataModule, prepare_batch
from alphagenome_ft.finetune.train import create_optimizer

from cardiomics_s2f.finetuning.alphagenome.inference import setup_data_parallel_mesh


def train_data_parallel(
    model: CustomAlphaGenomeModel,
    data_module: BigWigDataModule,
    head_specs: Sequence[HeadSpec],
    *,
    learning_rate: float,
    weight_decay: float,
    num_epochs: int,
    seed: int = 42,
    max_train_steps: int | None = None,
    heads_only: bool = False,
    checkpoint_dir: Path | None = None,
    organism: str = "HOMO_SAPIENS",
    best_metric: str = "valid_loss",
    best_metric_mode: str = "min",
    early_stopping_patience: int = 0,
    early_stopping_min_delta: float = 0.0,
    verbose: bool = False,
    use_wandb: bool = False,
    wandb_project: str | None = None,
    wandb_entity: str | None = None,
    wandb_run_name: str | None = None,
    wandb_config: dict | None = None,
) -> None:
    """Run fine-tuning with JIT-compiled train/eval steps, data-parallel over all local GPUs.

    Same as ``alphagenome_ft.finetune.train.train`` (version 0.1.5), except that each
    batch is split along a one-dimensional ``'data'`` device mesh
    (:func:`~cardiomics_s2f.finetuning.alphagenome.inference.setup_data_parallel_mesh`)
    while parameters, model state and optimizer state are replicated. With ``verbose``,
    the sharded train step is traced once with ``jax.eval_shape`` on the first train
    batch before training starts, so a sharding error fails fast.

    Args:
        model: Initialized AlphaGenome model wrapper to fine-tune.
        data_module: Batch provider with train/valid intervals and BigWig targets.
        head_specs: Head definitions used to build losses and optimizer filters.
        learning_rate: Base AdamW learning rate.
        weight_decay: AdamW weight decay.
        num_epochs: Maximum number of epochs to run.
        seed: Base RNG seed used for per-epoch training shuffles.
        max_train_steps: Optional global cap on optimizer updates across all epochs.
        heads_only: If True, freeze backbone and optimize selected heads only.
        checkpoint_dir: Optional output directory for ``best``/``last`` checkpoints.
        organism: Organism enum name used for model organism indexing.
        best_metric: Metric name used for best-checkpoint and early-stopping tracking.
        best_metric_mode: Improvement direction for ``best_metric`` (``min`` or ``max``).
        early_stopping_patience: Stop after this many non-improving epochs (0 disables).
        early_stopping_min_delta: Minimum metric change required to count as improvement.
        verbose: If True, print per-step progress and extra diagnostics.
        use_wandb: If True, log metrics to Weights & Biases.
        wandb_project: Optional W&B project name override.
        wandb_entity: Optional W&B entity/team override.
        wandb_run_name: Optional W&B run-name override.
        wandb_config: Optional extra config keys to merge into W&B config.

    Notes:
        Total planned steps are computed before training from train-set size and
        batch settings as ``steps_per_epoch * num_epochs`` (or capped by
        ``max_train_steps`` when provided). Progress is reported with a global
        counter in ``current/total`` format.
    """
    train_intervals = list(data_module._intervals.get("train", ()))
    num_train_examples = len(train_intervals)
    if num_train_examples == 0:
        raise ValueError("No train intervals available for training.")

    if data_module._drop_last:
        steps_per_epoch = num_train_examples // data_module._batch_size
    else:
        steps_per_epoch = math.ceil(num_train_examples / data_module._batch_size)
    if steps_per_epoch == 0:
        raise ValueError(
            "Computed zero training steps per epoch. Check batch size, drop_last, and train intervals."
        )

    planned_steps = steps_per_epoch * num_epochs
    total_train_steps = (
        min(planned_steps, max_train_steps) if max_train_steps is not None else planned_steps
    )
    step_width = len(str(steps_per_epoch))

    if checkpoint_dir is not None:
        checkpoint_dir = Path(checkpoint_dir).expanduser().resolve()
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

    if use_wandb:
        import wandb

        wb_config = {
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
            "num_epochs": num_epochs,
            "batch_size": data_module._batch_size,
            "steps_per_epoch": steps_per_epoch,
            "total_train_steps": total_train_steps,
            "heads_only": heads_only,
            "organism": organism,
            "best_metric": best_metric,
            "best_metric_mode": best_metric_mode,
            "early_stopping_patience": early_stopping_patience,
            "seed": seed,
            **(wandb_config or {}),
        }
        wandb.init(
            project=wandb_project or "alphagenome-ft",
            entity=wandb_entity,
            name=wandb_run_name,
            config=wb_config,
        )

    head_names = [spec.head_id for spec in head_specs]
    if heads_only:
        model.freeze_backbone()

    optimizer = create_optimizer(
        model._params,
        head_names,
        learning_rate,
        weight_decay,
        heads_only,
    )
    opt_state = optimizer.init(model._params)

    organism_enum = getattr(ag_dna_model.Organism, organism)
    organism_index_value = research_dna_model.convert_to_organism_index(organism_enum)
    strand_reindexing = jax.device_put(
        model._metadata[organism_enum].strand_reindexing,
        model._device_context._device,
    )

    # Data parallelism: batches are split along 'data', everything else is replicated
    mesh, data_sharding, replicated_sharding = setup_data_parallel_mesh()

    loss_fns = {name: model.create_loss_fn_for_head(name) for name in head_names}

    @jax.jit(
        in_shardings=(replicated_sharding, replicated_sharding, replicated_sharding, data_sharding),
        out_shardings=(replicated_sharding, replicated_sharding, replicated_sharding)
    )
    def train_step(params, state, current_opt_state, batch):
        def loss_fn(current_params):
            predictions = model._predict(
                current_params,
                state,
                batch["sequences"],
                batch["organism_index"],
                negative_strand_mask=batch["negative_strand_mask"],
                strand_reindexing=strand_reindexing,
            )
            total_loss = 0.0
            for head_name in head_names:
                head_loss_dict = loss_fns[head_name](
                    predictions[head_name],
                    {
                        "targets": batch[f"targets_{head_name}"],
                        "organism_index": batch["organism_index"],
                    },
                )
                total_loss = total_loss + head_loss_dict["loss"]
            return total_loss

        loss_value, grads = jax.value_and_grad(loss_fn)(params)
        updates, new_opt_state = optimizer.update(grads, current_opt_state, params)
        new_params = optax.apply_updates(params, updates)
        return new_params, new_opt_state, loss_value

    @jax.jit(
        in_shardings=(replicated_sharding, replicated_sharding, data_sharding),
        out_shardings=replicated_sharding
    )
    def eval_step(params, state, batch):
        predictions = model._predict(
            params,
            state,
            batch["sequences"],
            batch["organism_index"],
            negative_strand_mask=batch["negative_strand_mask"],
            strand_reindexing=strand_reindexing,
        )
        head_losses = {}
        for head_name in head_names:
            loss_dict = loss_fns[head_name](
                predictions[head_name],
                {
                    "targets": batch[f"targets_{head_name}"],
                    "organism_index": batch["organism_index"],
                },
            )
            head_losses[head_name] = loss_dict["loss"]
        return head_losses

    if verbose:
        print("JIT-compiling step functions (first call will be slow)...")

    def aggregate_valid_loss(metrics: Mapping[str, float]) -> float | None:
        return float(sum(metrics.values())) if metrics else None

    def resolve_metric(
        metric_name: str,
        train_loss: float | None,
        valid_metrics: Mapping[str, float] | None,
    ):
        if metric_name in {"train", "train_loss"}:
            return "train_loss", train_loss
        if metric_name in {"valid", "val", "valid_loss", "val_loss"}:
            return "valid_loss", aggregate_valid_loss(valid_metrics or {})
        if metric_name.startswith("valid:") or metric_name.startswith("valid/"):
            head = metric_name.split(":", 1)[-1].split("/", 1)[-1]
            return f"valid/{head}", (valid_metrics or {}).get(head)
        if valid_metrics and metric_name in valid_metrics:
            return f"valid/{metric_name}", valid_metrics[metric_name]
        return metric_name, None

    def is_improved(current: float, best: float | None) -> bool:
        if best is None:
            return True
        if best_metric_mode == "max":
            return current > best + early_stopping_min_delta
        return current < best - early_stopping_min_delta

    # Dry run: trace the sharded train step once (jax.eval_shape) before training
    if verbose:
        print("\nRunning jax.eval_shape to validate mesh and sharding logic...")
        try:
            # Grab a single dummy batch from the data module
            dummy_batch_np = next(data_module.iter_batches("train", seed=0))
            dummy_batch = prepare_batch(dummy_batch_np, organism_index_value, head_names)

            # Apply the mesh context and shard the dummy batch
            with model._device_context, jax.set_mesh(mesh):
                dummy_batch_sharded = jax.device_put(dummy_batch, data_sharding)

                # Trace the function without executing GPU kernels
                out_shapes = jax.eval_shape(
                    train_step,
                    model._params,
                    model._state,
                    opt_state,
                    dummy_batch_sharded
                )

            print("Validation successful! JAX successfully traced the sharded computations.")
            # print the shape of the new parameters to confirm
            print("Output shapes:", jax.tree_util.tree_map(lambda x: x.shape, out_shapes[0]))

        except Exception as e:
            raise RuntimeError(f"jax.eval_shape validation failed! Check your mesh and sharding logic. Error: {e}")

    best_value: float | None = None
    epochs_since_improvement = 0
    global_step = 0

    print(
        "Train plan: "
        f"{num_train_examples} examples | "
        f"{steps_per_epoch} step(s)/epoch | "
        f"{num_epochs} epoch(s) | "
        f"total step(s) {total_train_steps}"
    )

    with model._device_context, jax.set_mesh(mesh):
        stop_training = False
        for epoch in range(1, num_epochs + 1):
            if verbose:
                print(f'\n{"=" * 60}')
                print(f"Epoch {epoch}/{num_epochs}")
                print(f'{"=" * 60}')
            else:
                print(f"Epoch {epoch}/{num_epochs}")

            epoch_step = 0
            train_losses: list[float] = []
            for batch_np in data_module.iter_batches("train", seed=seed + epoch):
                batch = prepare_batch(batch_np, organism_index_value, head_names)
                batch = jax.device_put(batch, data_sharding)  # shard the batch across devices

                model._params, opt_state, loss_value = train_step(
                    model._params,
                    model._state,
                    opt_state,
                    batch,
                )
                loss_scalar = float(loss_value)
                train_losses.append(loss_scalar)
                epoch_step += 1
                global_step += 1

                if verbose:
                    print(
                        # f"  step {global_step:0{step_width}d}/{total_train_steps:0{step_width}d}"
                        # f" | epoch_step {epoch_step:04d} | loss {loss_scalar:.4f}",
                        f"  step {epoch_step:0{step_width}d}/{steps_per_epoch:0{step_width}d}"
                        f" | loss {loss_scalar:.4f}",
                        end="\r",
                        flush=True,
                    )

                if use_wandb:
                    wandb.log(
                        {
                            "train/step_loss": loss_scalar,
                            "epoch": epoch,
                            "step": global_step,
                            # "epoch_step": epoch_step,
                        }
                    )

                if global_step >= total_train_steps:
                    stop_training = True
                    break

            train_loss_avg = float(np.mean(train_losses)) if train_losses else None
            if verbose:
                print()
            if train_loss_avg is not None:
                print(f"  Train loss: {train_loss_avg:.4f}")
                if use_wandb:
                    wandb.log({"train/epoch_loss": train_loss_avg, "epoch": epoch})

            valid_metrics: Mapping[str, float] | None = None
            if "valid" in data_module._intervals and len(data_module._intervals["valid"]) > 0:
                losses = {head: [] for head in head_names}
                for batch_np in data_module.iter_batches("valid"):
                    batch = prepare_batch(batch_np, organism_index_value, head_names)
                    batch = jax.device_put(batch, data_sharding)  # shard the batch across devices

                    head_losses = eval_step(model._params, model._state, batch)
                    for head_name in head_names:
                        losses[head_name].append(float(head_losses[head_name]))

                valid_metrics = {head: float(np.mean(values)) for head, values in losses.items() if values}
                print("  Validation metrics:", ", ".join(f"{k}={v:.4f}" for k, v in valid_metrics.items()))
                if use_wandb:
                    valid_log = {f"valid/{head}": v for head, v in valid_metrics.items()}
                    valid_log["valid/loss"] = float(sum(valid_metrics.values()))
                    valid_log["epoch"] = epoch
                    wandb.log(valid_log)

            metric_label, metric_value = resolve_metric(best_metric, train_loss_avg, valid_metrics)
            if metric_value is not None and math.isfinite(metric_value):
                if is_improved(metric_value, best_value):
                    best_value = metric_value
                    epochs_since_improvement = 0
                    if use_wandb:
                        wandb.log({"best/" + metric_label: metric_value, "epoch": epoch})
                    if checkpoint_dir:
                        print(
                            f"  Metric improved ({metric_label} = {metric_value:.4f}) "
                            " -- saving best checkpoint"
                        )
                        model.save_checkpoint(checkpoint_dir / "best", save_full_model=False)
                else:
                    epochs_since_improvement += 1
            else:
                print(f"  Best metric ({metric_label}): unavailable")

            if checkpoint_dir:
                model.save_checkpoint(checkpoint_dir / "last", save_full_model=False)

            if early_stopping_patience > 0 and epochs_since_improvement >= early_stopping_patience:
                print(f"\n  Early stopping: no improvement for {epochs_since_improvement} epoch(s)")
                break
            if stop_training:
                print(f"  Reached requested training steps: {global_step}/{total_train_steps}")
                break

    if checkpoint_dir and not (checkpoint_dir / "last").exists():
        model.save_checkpoint(checkpoint_dir / "last", save_full_model=False)

    print(f'\n{"=" * 60}')
    print("Training complete!")
    print(f'{"=" * 60}')

    if use_wandb:
        wandb.finish()
