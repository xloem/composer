from composer.algorithms.selective_backprop import register_scoring_fxn


@register_scoring_fxn("loss")  #  type: ignore
def dummy_fxn():
    """Dummy function to hold 'loss' in scoring function registry until actual loss
    function can be obtained."""
    pass
