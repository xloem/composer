# Copyright 2021 MosaicML. All Rights Reserved.

from __future__ import annotations

import ast
import csv
import inspect
import logging
from dataclasses import asdict, dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import yahp as hp
from torch.nn import functional as F

from composer.algorithms.algorithm_hparams import AlgorithmHparams
from composer.core.types import Algorithm, Batch, Event, Logger, State, Tensor

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)


def do_selective_backprop(
    epoch: int,
    batch_idx: int,
    start_epoch: int,
    end_epoch: int,
    interrupt: int,
) -> bool:
    """Decide if selective backprop should be run based on time in training.

    Returns true if the current ``epoch`` is between ``start_epoch`` and
    ``end_epoch``. Recommend that SB be applied during the later stages of
    a training run, once the model has already "learned" easy examples.

    To preserve convergence, SB can be interrupted with vanilla minibatch
    gradient steps every ``interrupt`` steps. When ``interrupt=0``, SB will be
    used at every step during the SB interval. When ``interrupt=2``, SB will
    alternate with vanilla minibatch steps.

    Args:
        epoch: The current epoch during training
        batch_idx: The current batch within the epoch
        start_epoch: The epoch at which selective backprop should be enabled
        end_epoch: The epoch at which selective backprop should be disabled
        interrupt: The number of batches between vanilla minibatch gradient updates

    Returns:
        bool: If selective backprop should be performed on this batch.
    """
    is_interval = ((epoch >= start_epoch) and (epoch < end_epoch))
    is_step = ((interrupt == 0) or ((batch_idx + 1) % interrupt != 0))

    return is_interval and is_step


# TODO this function should probably be part of the public API
def selective_backprop(batch: Union[Batch, Dict[str, Tensor], None] = None,
                       X: Union[torch.Tensor, None] = None,
                       y: Union[torch.Tensor, None] = None,
                       *,
                       model: torch.nn.Module,
                       scoring_fxn: Callable,
                       keep: float,
                       scale_factor: float = 1) -> Tuple[torch.Tensor, torch.Tensor]:
    """Select a subset of the batch on which to learn as per (`Jiang et al. 2019 <https://arxiv.org/abs/1910.00762>`_)

    Selective Backprop (SB) prunes minibatches according to the difficulty
    of the individual training examples and only computes weight gradients
    over the selected subset. This reduces iteration time and speeds up training.
    The fraction of the minibatch that is kept for gradient computation is
    specified by the argument ``0 <= keep <= 1``.

    To speed up SB's selection forward pass, the argument ``scale_factor`` can
    be used to spatially downsample input tensors. The full-sized inputs
    will still be used for the weight gradient computation.

    Args:
        X: Input tensor to prune
        y: Output tensor to prune
        model: Model with which to predict outputs
        scoring_fxn: Loss function of the form ``loss(outputs, targets, reduction='none')``.
            The function must take the keyword argument ``reduction='none'``
            to ensure that per-sample losses are returned.
        keep: Fraction of examples in the batch to keep
        scale_factor: Multiplier between 0 and 1 for spatial size. Downsampling
            requires the input tensor to be at least 3D.

    Returns:
        (torch.Tensor, torch.Tensor): The pruned batch of inputs and targets

    Raises:
        ValueError: If ``scale_factor > 1``
        TypeError: If ``loss_fun > 1`` has the wrong signature or is not callable

    Note:
    This function runs an extra forward pass through the model on the batch of data.
    If you are using a non-default precision, ensure that this forward pass
    runs in your desired precision. For example:

    .. code-block:: python

        with torch.cuda.amp.autocast(True):
            X_new, y_new = selective_backprop(X, y, model, loss_fun, keep, scale_factor)

    """
    INTERPOLATE_MODES = {3: "linear", 4: "bilinear", 5: "trilinear"}

    if batch:
        if not isinstance(batch, Dict):
            raise TypeError("Argument 'batch' must be a dict")
        if "data" not in batch.keys() or "target" not in batch.keys():
            raise ValueError("Argument 'batch' must be a dict that contains keys 'data' and 'target'.")
        X = batch["data"]
        y = batch["target"]

    assert isinstance(X, Tensor)
    assert isinstance(y, Tensor)

    interp_mode = "bilinear"
    if scale_factor != 1:
        if X.dim() not in INTERPOLATE_MODES:
            raise ValueError(f"Input must be 3D, 4D, or 5D if scale_factor != 1, got {X.dim()}")
        interp_mode = INTERPOLATE_MODES[X.dim()]

    if scale_factor > 1:
        raise ValueError("scale_factor must be <= 1")

    if not callable(scoring_fxn):
        raise TypeError("Loss function must be callable")

    with torch.no_grad():
        N = X.shape[0]

        # Maybe interpolate
        if scale_factor < 1:
            X_scaled = F.interpolate(X, scale_factor=scale_factor, mode=interp_mode)
        else:
            X_scaled = X

        # Get per-examples losses
        out = model(X_scaled)
        losses = scoring_fxn(out, y)

        # Sort losses
        sorted_idx = torch.argsort(losses)
        n_select = int(keep * N)

        # Sample by loss
        percs = np.arange(0.5, N, 1) / N
        probs = percs**((1.0 / keep) - 1.0)
        probs = probs / np.sum(probs)
        select_percs_idx = np.random.choice(N, n_select, replace=False, p=probs)
        select_idx = sorted_idx[select_percs_idx]

    return X[select_idx], y[select_idx]


@dataclass
class SelectiveBackpropHparams(AlgorithmHparams):
    """See :class:`SelectiveBackprop`"""

    start: float = hp.optional(doc="SB interval start, as fraction of training duration", default=0.5)
    end: float = hp.optional(doc="SB interval end, as fraction of training duration", default=0.9)
    keep: float = hp.optional(doc="fraction of minibatch to select and keep for gradient computation", default=0.5)
    scale_factor: float = hp.optional(doc="scale for downsampling input for selection forward pass", default=0.5)
    interrupt: int = hp.optional(doc="interrupt SB with a vanilla minibatch step every 'interrupt' batches", default=2)
    scoring_fxn: str = hp.optional(
        doc=
        "Scoring function that will be used to rank samples for backpropagation. Options: ['loss', 'irreducible_loss']. Can also write custom scoring function.",
        default="loss")
    score_path: Union[str, None] = hp.optional(doc="path to .csv containing sample scores", default=None)

    def initialize_object(self) -> SelectiveBackprop:
        return SelectiveBackprop(**asdict(self))


class SelectiveBackprop(Algorithm):
    """Selectively backpropagate gradients from a subset of each batch.

    Selective Backprop (SB) prunes minibatches according to the difficulty
    of the individual training examples, and only computes weight gradients
    over the pruned subset, reducing iteration time and speeding up training.
    The fraction of the minibatch that is kept for gradient computation is
    specified by the argument ``0 <= keep <= 1``.

    To speed up SB's selection forward pass, the argument ``scale_factor`` can
    be used to spatially downsample input image tensors. The full-sized inputs
    will still be used for the weight gradient computation.

    To preserve convergence, SB can be interrupted with vanilla minibatch
    gradient steps every ``interrupt`` steps. When ``interrupt=0``, SB will be
    used at every step during the SB interval. When ``interrupt=2``, SB will
    alternate with vanilla minibatch steps.

    Args:
        start: SB interval start as fraction of training duration
        end: SB interval end as fraction of training duration
        keep: fraction of minibatch to select and keep for gradient computation
        scale_factor: scale for downsampling input for selection forward pass
        interrupt: interrupt SB with a vanilla minibatch step every
            ``interrupt`` batches
        scoring_fxn: scoring function that will be used to rank samples for
            backpropagation. Must have the signature scoring_fxn(p, y, **kwargs).
        Options: ['loss', 'irreducible_loss']. 'loss' (default) = selective backprop (`Jiang et al. 2019 <https://arxiv.org/abs/1910.00762>`_).
            'irreducible_loss' is described in (`Mindermann et al. 2021
            <https://arxiv.org/abs/2107.02565>`_). You can also add a custom scoring
            function to composer/algorithms/selective_sampling/scoring_functions.
        score_path: path to .csv containing scores (or external data used for computing
        scores). 
    """

    def __init__(self, start: float, end: float, keep: float, scale_factor: float, interrupt: int, scoring_fxn: str,
                 score_path: Optional[None]):
        from composer.algorithms.selective_backprop import SCORING_FXN_REGISTRY
        self.scoring_fxn_registry = SCORING_FXN_REGISTRY
        if scoring_fxn not in self.scoring_fxn_registry.keys():
            ValueError(f"{scoring_fxn} is an invalid value for algorithms.selective_backprop.scoring_fxn. ")
        self.scoring_fxn = None
        self.hparams = SelectiveBackpropHparams(start=start,
                                                end=end,
                                                keep=keep,
                                                scale_factor=scale_factor,
                                                interrupt=interrupt,
                                                score_path=score_path,
                                                scoring_fxn=scoring_fxn)

    def match(self, event: Event, state: State) -> bool:
        """Match on Event.INIT or ``Event.AFTER_DATALOADER`` if time is between
        ``self.start`` and ``self.end``."""
        if event == Event.INIT and self.hparams.score_path:
            return True

        if event == Event.TRAINING_START:
            return True

        is_event = (event == Event.AFTER_DATALOADER)
        if not is_event:
            return False

        is_keep = (self.hparams.keep < 1)
        if not is_keep:
            return False

        is_chosen = do_selective_backprop(
            epoch=state.epoch,
            batch_idx=state.batch_idx,
            start_epoch=int(state.max_epochs * self.hparams.start),
            end_epoch=int(state.max_epochs * self.hparams.end),
            interrupt=self.hparams.interrupt,
        )
        return is_chosen

    def apply(self, event: Event, state: State, logger: Optional[Logger] = None) -> None:
        if event == Event.INIT and self.hparams.score_path:
            # Parse sample score .csv
            sample_score_dict = self.parse_score_csv()
            assert hasattr(state.train_dataloader.dataset, "samples"), "Dataset must have attribute 'samples'"
            # Add sample scores to samples in dataset
            state.train_dataloader.dataset.samples = self.append_scores_to_samples(  #  type: ignore - type not found
                sample_score_dict, state.train_dataloader.dataset.samples)  # type: ignore - ditto

        if event == Event.TRAINING_START:
            # Replace dummy loss scoring function in registry with actual loss function
            if self.hparams.scoring_fxn == "loss":
                assert callable(state.model.module.loss)  # type: ignore - type not found

                def loss(p, y, **kwargs):
                    return state.model.module.loss(  # type: ignore
                        p, (None, y), reduction="none")  # type: ignore

                self.scoring_fxn_registry["loss"] = loss
            self.scoring_fxn = self.scoring_fxn_registry[self.hparams.scoring_fxn]

        if event == Event.AFTER_DATALOADER:
            """Apply selective backprop to the current batch."""
            if isinstance(state.batch, Sequence):
                if not isinstance(state.batch_pair[0], Tensor) or not isinstance(state.batch_pair[1], Tensor):
                    raise TypeError("Batch items must be of type torch.Tensor")
                batch = {"data": state.batch_pair[0], "target": state.batch_pair[1]}
            elif isinstance(state.batch, Dict):
                batch = state.batch
            else:
                raise TypeError(f"state.batch must be a Sequence or Dict")

            assert callable(self.scoring_fxn)  # type: ignore - type not found

            # Model expected to only take in input, not the full batch
            model = lambda X: state.model((X, None))

            with state.precision_context:
                new_input, new_target = selective_backprop(
                    batch,
                    model=model,  # type: ignore - ditto because of loss
                    scoring_fxn=self.scoring_fxn,
                    keep=self.hparams.keep,
                    scale_factor=self.hparams.scale_factor)
            state.batch = (new_input, new_target)

    def parse_score_csv(self) -> Dict:
        """Parse .csv into a dict in which each item corresponds to a row, and each row
        is encoded as a dict. The data structure takes the following form:
        {
        sample0_path: {label0: value0, label1: value1, ...},
        sample1_path: {label0: value0, label1: value1, ...},
        ...
        sampleN_path: {label0: value0, label1: value1, ...}
        }"""
        parsed_rows = {}
        if self.hparams.score_path:
            try:
                with open(self.hparams.score_path, newline='') as csvfile:
                    reader = csv.DictReader(csvfile)
                    for row_n, row in enumerate(reader, start=1):
                        sample_path = None
                        for k, v in row.items():
                            # Default values
                            parsed_val = float("nan")
                            if k == "path":
                                parsed_rows[v] = {}
                                sample_path = v
                            else:
                                try:
                                    parsed_val = ast.literal_eval(v)
                                except ValueError:
                                    log.warning(
                                        f"Unable to parse column value {v} from row {row_n} in sample score file {self.hparams.score_path}. Strings must be 'quoted' to be parsed correctly."
                                    )
                                try:
                                    parsed_rows[sample_path][k] = parsed_val
                                except KeyError:
                                    log.warning(
                                        f"Unable to parse sample path from row {row_n} in sample score file {self.hparams.score_path}. Please ensure the first column is labeled 'path'."
                                    )
            except FileNotFoundError:
                log.error(f"Sample score file {self.hparams.score_path} not found.")
        return parsed_rows

    def append_scores_to_samples(self, score_dict: Dict, samples: Union[Sequence, Dict]) -> List[Dict]:
        """Takes dict of sample scores and adds them to each data sample in the dataset."""
        sample_dicts = []
        n_failed_supp_labels = 0
        for sample in samples:
            curr_sample_dict = {k: v for k, v in sample.items()}
            try:
                curr_row = score_dict[sample["path"]]
                for k, v in curr_row.items():
                    curr_sample_dict[k] = v
            except KeyError:
                n_failed_supp_labels += 1
            sample_dicts.append(curr_sample_dict)
        if n_failed_supp_labels > 0:
            log.warning(f"Unable to add one or more supplementary labels to {n_failed_supp_labels} samples.")
        return sample_dicts
