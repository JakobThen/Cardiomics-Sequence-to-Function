"""
General Utilities
=================

Small helpers shared by the workflow scripts.
"""

import os
import time


def log_stage_resources(stage_name, start_time=None):
    """Prints the current RAM usage and the time elapsed since ``start_time``.

    Used to profile the stages of long training runs: pass the value returned by the
    previous call to time one stage, or the script's start time for the total.

    Args:
        stage_name (str): Label printed with the measurement.
        start_time (float, optional): ``time.time()`` value to measure from. Defaults to
            None (elapsed time 0).

    Returns:
        float: The current ``time.time()``, to pass to the next call.
    """
    import psutil

    mem_gb = psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024 * 1024)
    elapsed = time.time() - start_time if start_time else 0
    hours = int(elapsed // 3600)
    minutes = int((elapsed % 3600) // 60)
    seconds = int(elapsed % 60)
    time_str = f"{hours:02d}:{minutes:02d}:{seconds:02d}"    # Format hh:mm:ss

    print(f"\n[PROFILE - {stage_name}] RAM: {mem_gb:.2f} GB | Time Elapsed: {time_str}\n", flush=True)
    return time.time()
