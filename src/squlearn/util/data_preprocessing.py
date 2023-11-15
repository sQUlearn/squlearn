import numpy as np
from typing import Tuple, Union


def adjust_features(x: Union[np.ndarray, float], x_length: int) -> Tuple[np.ndarray, bool]:
    """Adjust the feature vector to the form [[]] if necessary.

    Args:
        x (np.ndarray): Input array.
        x_length (int): Dimension of the input array, e.g. feature dimension.

    Return:
        Adjusted feature array and a boolean flag for multiple inputs.
    """

    return _adjust_input(x, x_length, allow_single_array=False)


def adjust_parameters(x: np.ndarray, x_length: int) -> Tuple[np.ndarray, bool]:
    """Adjust the parameter vector to the form [[]] if necessary.

    In contrast to feature vectors, one dimensional parameters are not considered
    as multiple inputs.

    Args:
        x (np.ndarray): Input array.
        x_length (int): Dimension of the input array, e.g. feature dimension.

    Return:
        Adjusted parameter array and a boolean flag for multiple inputs.
    """

    return _adjust_input(x, x_length, allow_single_array=True)


def _adjust_input(
    x: Union[float, np.ndarray], x_length: int, allow_single_array: bool
) -> Tuple[np.ndarray, bool]:
    """Adjust the input to the form [[]] if necessary.

    If allow_single_array is True, a one dimensional array is not considered as multiple outputs.

    Args:
        x (np.ndarray): Input array.
        x_length (int): Dimension of the input array, e.g. feature dimension.
        allow_single_array (bool): If True, a one dimensional array is not considered as
                                   multiple outputs.

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
            if allow_single_array:
                multiple_inputs = shape[0] != 1
            else:
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
