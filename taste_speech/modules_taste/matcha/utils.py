import logging

import numpy as np
import torch
from lightning.pytorch.utilities import rank_zero_only


def sequence_mask(length, max_length=None):
    if max_length is None:
        max_length = length.max()
    x = torch.arange(max_length, dtype=length.dtype, device=length.device)
    return x.unsqueeze(0) < length.unsqueeze(1)


def get_pylogger(name: str = __name__) -> logging.Logger:
    """Initializes a multi-GPU-friendly python command line logger.

    :param name: The name of the logger, defaults to ``__name__``.

    :return: A logger object.
    """
    logger = logging.getLogger(name)

    # this ensures all logging levels get marked with the rank zero decorator
    # otherwise logs would get multiplied for each GPU process in multi-GPU setup
    logging_levels = ("debug", "info", "warning", "error", "exception", "fatal", "critical")
    for level in logging_levels:
        setattr(logger, level, rank_zero_only(getattr(logger, level)))

    return logger
