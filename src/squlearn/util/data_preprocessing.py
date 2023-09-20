import numpy as np
from typing import Tuple


def adjust_input(x, x_length: int) -> Tuple[np.ndarray, bool]:
    """Adjust the input to the form [[]] if necessary.

    Args:
        x (np.ndarray): Input array.
        x_length (int): Dimension of the input array, e.g. feature dimension.

    Return:
        Adjusted input array and a boolean flag for multiple inputs.
    """
    multiple_inputs = False
    error = False
    shape = np.shape(x)
    if shape == () and x_length == 1:
        # Single floating point number
        xx = np.array([[x]])
    elif len(shape) == 1:
        if x_length == 1:
            xx = np.array([np.array([xx]) for xx in x])
            multiple_inputs = True
        else:
            # We have a single multi dimensional x (e.g. parameter vector)
            if len(x) == x_length:
                xx = np.array([x])
            else:
                error = True
    elif len(shape) == 2:
        if shape[1] == x_length:
            xx = x
            multiple_inputs = True
        else:
            error = True
    else:
        error = True

    if error:
        raise ValueError("Wrong format of an input variable.")

    return xx, multiple_inputs
