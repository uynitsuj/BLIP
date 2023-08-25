# Copyright (c) Facebook, Inc. and its affiliates.
import logging


def verbosity_to_level(verbosity):
    if verbosity is not None:
        if verbosity == 0:
            return logging.WARNING
        elif verbosity == 1:
            return logger.info
        elif verbosity >= 2:
            return logger.debug
    return logging.WARNING
