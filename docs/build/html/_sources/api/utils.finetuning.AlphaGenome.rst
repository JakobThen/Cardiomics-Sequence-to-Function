utils.finetuning.AlphaGenome package
====================================

Module contents
---------------

.. automodule:: utils.finetuning.AlphaGenome
   :members:
   :show-inheritance:
   :undoc-members:

CLI Usage Examples
------------------

**evaluate_AG_on_test_streamed.py**

Usage:

.. code-block:: bash

    python src/utils/finetuning/AlphaGenome/evaluate_AG_on_test_streamed.py \
        --fasta_path [FASTA_PATH] \
        --config [CONFIG] \
        --input_dir [INPUT_DIR] \
        --out_dir [OUT_DIR] \
        --checkpoint_dir [CHECKPOINT_DIR] \
        [OTHER_ARGS]

Input arguments:

.. code-block:: text

    -h, --help: Show help message with arguments and their descriptions, and exit
    --fasta_path (required): Path to the genome.fa file
    --config (required): Path to the target YAML config file
    --input_dir (required): Directory containing input BigWig files
    --out_dir (required): Output directory for figures and results
    --checkpoint_dir (required): Directory containing the best model checkpoint
    --gtf_file: Optional GTF file for Gene-based correlation mapping
    --fold: Data fold to use (0, 1, 2, 3)
    --window_size: Window size in bp (must map to an AlphaGenome config)
    --organism: Target organism
    --model_version: Base model version (e.g., fold_1, all_folds)
    --batch_size: Global batch size (must be divisible by N_GPUs)
    --resolution: Embedding resolution to evaluate (1, 128)
    --save_predictions: Whether to save test interval predictions to HDF5.
    --minimal_test: Truncate run to the first 8 intervals for quick debugging.


**finetune_heads.py**

Usage:

.. code-block:: bash

    python src/utils/finetuning/AlphaGenome/finetune_heads.py \
        --fasta_path [FASTA_PATH] \
        --config [CONFIG] \
        --base_dir [BASE_DIR] \
        --checkpoint_dir [CHECKPOINT_DIR] \
        [OTHER_ARGS]

Input arguments:

.. code-block:: text

    -h, --help: Show help message with arguments and their descriptions, and exit
    --fasta_path (required): Path to the genome.fa file
    --config (required): Path to the target YAML config file
    --base_dir (required): Base directory for resolving config paths (e.g., the local tmp dir)
    --checkpoint_dir (required): Directory to save checkpoints
    --fold: Data fold to use (0, 1, 2, 3)
    --window_size: Window size in bp (default: 1 Mbp)
    --organism: Target organism
    --model_version: Base model version (e.g., fold_1, all_folds)
    --batch_size: Global batch size (must be divisible by N_GPUs)
    --epochs: Number of training epochs
    --lr: Learning rate
    --weight_decay: Weight decay
    --seed: Random seed
    --patience: Early stopping patience
    --min_delta: Early stopping min delta


**resume_finetune_heads.py**

Usage:

.. code-block:: bash

    python src/utils/finetuning/AlphaGenome/resume_finetune_heads.py \
        --fasta_path [FASTA_PATH] \
        --config [CONFIG] \
        --base_dir [BASE_DIR] \
        --in_checkpoint_dir [IN_CHECKPOINT_DIR] \
        --out_checkpoint_dir [OUT_CHECKPOINT_DIR] \
        [OTHER_ARGS]

Input arguments:

.. code-block:: text

    -h, --help: Show help message with arguments and their descriptions, and exit
    --fasta_path (required): Path to the genome.fa file
    --config (required): Path to the target YAML config file
    --base_dir (required): Base directory for resolving config paths (e.g., the local tmp dir)
    --in_checkpoint_dir (required): Directory containing the checkpoint to resume from
    --out_checkpoint_dir (required): Directory to save new checkpoints
    --fold: Data fold to use (0, 1, 2, 3)
    --window_size: Window size in bp (default: 1 Mbp)
    --organism: Target organism
    --model_version: Base model version (e.g., fold_1, all_folds)
    --batch_size: Global batch size (must be divisible by N_GPUs)
    --epochs: Number of training epochs
    --lr: Learning rate
    --weight_decay: Weight decay
    --seed: Random seed
    --patience: Early stopping patience
    --min_delta: Early stopping min delta


**stage_to_tmp.py**

Usage:

.. code-block:: bash

    python src/utils/finetuning/AlphaGenome/stage_to_tmp.py \
        --master_yaml [MASTER_YAML] \
        --tmp_dir [TMP_DIR] \
        [OTHER_ARGS]

Input arguments:

.. code-block:: text

    -h, --help: Show help message with arguments and their descriptions, and exit
    --master_yaml (required): Path to the input master YAML config
    --tmp_dir (required): Path to the local tmp directory on the compute node
    --out_yaml: Path for the output temporary YAML (defaults to <tmp_dir>/tmp_bw_config.yaml)


Submodules
----------

utils.finetuning.AlphaGenome.evaluate\_AG\_on\_test\_streamed module
--------------------------------------------------------------------

.. automodule:: utils.finetuning.AlphaGenome.evaluate_AG_on_test_streamed
   :members:
   :show-inheritance:
   :undoc-members:

utils.finetuning.AlphaGenome.finetune\_heads module
---------------------------------------------------

.. automodule:: utils.finetuning.AlphaGenome.finetune_heads
   :members:
   :show-inheritance:
   :undoc-members:

utils.finetuning.AlphaGenome.resume\_finetune\_heads module
-----------------------------------------------------------

.. automodule:: utils.finetuning.AlphaGenome.resume_finetune_heads
   :members:
   :show-inheritance:
   :undoc-members:

utils.finetuning.AlphaGenome.stage\_to\_tmp module
--------------------------------------------------

.. automodule:: utils.finetuning.AlphaGenome.stage_to_tmp
   :members:
   :show-inheritance:
   :undoc-members:
