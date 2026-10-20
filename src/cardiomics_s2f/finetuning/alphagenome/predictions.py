"""
AlphaGenome Prediction Helpers
==============================

Lightweight helpers (numpy/pandas only) to select prediction outputs from a
fine-tuned AlphaGenome model and to align per-track metadata with them.
"""

import numpy as np
import pandas as pd


def get_prediction_key(raw_p, bin_size=128, return_scaled=True):
    """Identifies the correct dictionary key for the desired embedding resolution.

    AlphaGenome predicts at multiple resolutions and data scales simultaneously.
    This helper finds the key matching the target target bin size.

    Args:
        raw_p (dict): Raw prediction dictionary from the model.
        bin_size (int, optional): The target bin size/resolution. Defaults to 128.
        return_scaled (bool, optional): Whether to look for scaled predictions. Defaults to True.

    Returns:
        str: The matching key found in the prediction dictionary.

    Raises:
        KeyError: If no matching prediction key is found in the dictionary.
    """
    prefix = "scaled_predictions" if return_scaled else "predictions"
    preferred = [f"{prefix}_{bin_size}bp", f"{prefix}_{128 if bin_size == 1 else 1}bp"]

    for key in preferred:
        if key in raw_p: return key
    raise KeyError(f"No matching prediction key found. Tried: {preferred}.")


def get_aligned_nonzero_means(tracks_df: pd.DataFrame, config_dict: dict) -> np.ndarray:
    """Extracts precomputed nonzero track means and aligns them to the tracks DataFrame.

    Strictly aligns the means based on file paths to ensure calculations match
    the correct BigWig files.

    Args:
        tracks_df (pd.DataFrame): DataFrame containing track metadata including paths.
        config_dict (dict): Target configuration dictionary containing head definitions.

    Returns:
        np.ndarray: A 1D array of aligned nonzero mean values.

    Raises:
        KeyError: If a path listed in `tracks_df` is missing from the configuration.
    """
    path_to_mean = {}
    for head in config_dict.get('heads', []):
        for target in head.get('targets', []):
            path = target.get('path')
            mean_val = target.get('nonzero_mean')
            if path is not None and mean_val is not None:
                path_to_mean[path] = mean_val

    aligned_means = []
    for path in tracks_df['path']:
        if path not in path_to_mean:
            raise KeyError(f"Path from tracks_df not found in config dictionary: {path}")
        aligned_means.append(path_to_mean[path])

    return np.array(aligned_means)
