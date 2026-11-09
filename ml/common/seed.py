import os
import random
import numpy as np


def seed_everything(seed: int = 42) -> int:
    """Set random seed across Python runtime and NumPy for complete reproducibility.

    Args:
        seed: The integer seed to fix.

    Returns:
        The fixed seed value.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    return seed
