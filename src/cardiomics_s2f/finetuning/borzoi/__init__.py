"""
Borzoi Finetuning Module
========================

Library code for fine-tuning Borzoi (``human_rep0``, via gReLU) on the project's
BigWig tracks (the runnable scripts live in ``workflows/borzoi/``).

Submodules:
-----------
* `datasets`: BigWig training dataset that streams labels from disk (the patched gReLU dataset of the cluster env).
* `intervals`: Train/validation/test windows of the Borzoi ``human_rep0`` split, filtered to chromosome ends and the blacklist.
* `training`: Track discovery in the staged BigWig folders, track order from a checkpoint and layer unfreezing.
"""
