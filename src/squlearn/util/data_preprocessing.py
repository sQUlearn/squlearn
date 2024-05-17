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
    elif sum(shape) == 0 and x_length > 0:
        # Empty array although x_length not zero
        error = True
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

    return convert_to_float64(xx), multiple_inputs


def convert_to_float64(x: Union[float, np.ndarray, list]) -> np.ndarray:
    """Convert to float64 format, raise Error for complex values

    Args:
        x (Union[float, np.ndarray]): Data that is converted

    Returns:
        Converted numpy float64 array
    """
    if not isinstance(x, np.ndarray):
        x = np.array(x)
    if x.dtype != np.float64:
        x = np.real_if_close(x)
        if np.iscomplexobj(x):
            raise ValueError(
                "Only real values for parameters and features are supported in sQUlearn!"
            )
        x = np.array(x, dtype=np.float64)

    return x


def to_tuple(x: Union[float, np.ndarray, list, tuple], flatten: bool = True) -> Tuple:
    """Function for converting data into hashable tuples

    Args:
        x (Union[float,np.ndarray,list,tuple]): Input data.

    Return:
        Flattened tuple of the input data
    """

    if flatten:

        def recursive_flatten(container):
            for i in container:
                if isinstance(i, (list, tuple, np.ndarray)):
                    for j in recursive_flatten(i):
                        yield j
                else:
                    yield i

        if isinstance(x, float):
            return tuple([x])
        elif len(np.shape(x)) == 1:
            return tuple(list(x))
        else:
            return tuple(recursive_flatten(x))

    else:

        def array_to_nested_tuple(arr):
            if isinstance(arr, (list, tuple, np.ndarray)):
                return tuple(array_to_nested_tuple(subarr) for subarr in arr)
            else:
                return arr

        if isinstance(x, (list, tuple, np.ndarray)):
            return array_to_nested_tuple(x)
        else:
            return tuple([x])
