import numpy as np
from typing import Tuple

def assign_all_parameters(
    opflow,
    x=None,
    param=None,
    param_op=None,
    x_values=None,
    param_values=None,
    param_op_values=None,
):
    """
    Assigns circuit parameters
    """

    todo_list = []  # list for the variables
    multi_list = []  # list of return ListOp or no list
    param_list = []  # list of parameters that are substituted

    # check shape of the x and adjust to [[]] form if necessary
    if x is not None:
        xx, multi_x = adjust_input(x_values, len(x))
        todo_list.append(xx)
        param_list.append(x)
        multi_list.append(multi_x)
    if param is not None:
        pp, multi_p = adjust_input(param_values, len(param))
        todo_list.append(pp)
        param_list.append(param)
        multi_list.append(multi_p)
    if param_op is not None:
        pp_op, multi_op = adjust_input(param_op_values, len(param_op))
        todo_list.append(pp_op)
        param_list.append(param_op)
        multi_list.append(multi_op)

    # Recursive construction of the assignment dictionary and list structure
    def rec_assign(dic, todo_list, param_list, multi_list):
        if len(todo_list) <= 0:
            return None
        return_list = []
        for x_ in todo_list[0]:
            for A, B in zip(param_list[0], x_):
                dic[A] = B
            if len(multi_list[1:]) > 0:
                return_list.append(
                    rec_assign(dic.copy(), todo_list[1:], param_list[1:], multi_list[1:])
                )
            else:
                return_list.append(opflow.assign_parameters(dic))

        if multi_list[0]:
            from qiskit.opflow import ListOp

            return ListOp(return_list)
        else:
            return return_list[0]

    return rec_assign({}, todo_list, param_list, multi_list)


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
