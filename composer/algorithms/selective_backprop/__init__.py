# Copyright 2021 MosaicML. All Rights Reserved.

import importlib
import os
import sys
from pathlib import Path
from typing import Callable

from composer.algorithms.selective_backprop.selective_backprop import SelectiveBackprop as SelectiveBackprop
from composer.algorithms.selective_backprop.selective_backprop import \
    SelectiveBackpropHparams as SelectiveBackpropHparams
from composer.algorithms.selective_backprop.selective_backprop import selective_backprop as selective_backprop

_name = 'Selective Backprop'
_class_name = 'SelectiveBackprop'
_functional = 'selective_backprop'
_tldr = 'Drops examples with small loss contributions.'
_attribution = '(Jiang et al, 2019)'
_link = 'https://arxiv.org/abs/1910.00762'
_method_card = ''

SCORING_FXN_REGISTRY = {}


def register_scoring_fxn(name: str):
    """Registers scoring functions.
    This decorator allows composer to add custom scoring heuristics, even if the
    scoring heuristic is not part of composer. To use it, apply this decorator
    to a scoring function like this:
    .. code-block:: python
        @register_scoring_fxn('scoring_fxn_name')
        def scoring_fxn():
            ...
    and place the file in composer/algorithms/selective_backprop/scoring_functions"""

    def register_scoring_fxn_internal(fxn: Callable[..., Callable]):
        if name in SCORING_FXN_REGISTRY:
            raise ValueError("Cannot register duplicate scoring function ({})".format(name))

        SCORING_FXN_REGISTRY[name] = fxn
        return fxn

    return register_scoring_fxn_internal


def import_scoring_functions() -> None:
    scoring_fxns_path = os.path.join(Path(__file__).parent, "scoring_functions")
    base_module = "composer.algorithms.selective_backprop.scoring_functions"
    for file in os.listdir(scoring_fxns_path):
        if file.endswith((".py", ".pyc")) and not file.startswith("_"):
            module = file[:file.find(".py")]
            if module not in sys.modules:
                module_name = ".".join([base_module, module])
                importlib.import_module(module_name)


import_scoring_functions()
