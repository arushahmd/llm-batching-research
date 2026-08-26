from __future__ import annotations

import random

import numpy as np
import torch


def set_experiment_seed(seed: int) -> None:
    """
    Set the random seeds used by the matched experiment protocol.

    The seed is applied before fresh model initialization so that model
    initialization and other stochastic operations are controlled per run.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
