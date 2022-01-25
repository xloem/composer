from typing import Callable, Dict

from composer.algorithms.selective_backprop import register_scoring_fxn
from composer.core.types import Tensor


@register_scoring_fxn("irreducible_loss")  #  type: ignore
def irreducible_loss(logits: Tensor, targets: Tensor, loss_fxn: Callable, **kwargs: Dict):
    """'Irreducible_loss' as described in (`Mindermann et al. 2021
    <https://arxiv.org/abs/2107.02565>`_). Essentially surrogate_loss - current_loss."""
    if "score" not in kwargs.keys():
        raise KeyError(
            "The irreducible loss scoring function requires a dataset that returns samples as a dict with the key 'score'."
        )
    return kwargs["score"] - loss_fxn(logits, targets)
