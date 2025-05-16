from typing import Union, List
import numpy as np

from qiskit.circuit import ParameterVector
from qiskit.circuit.parametervector import ParameterVectorElement


class EvaluationBase:
    """Base class for evaluation of derivatives of the QNN

    Args:
        key (Union[str, tuple, ParameterVector, ParameterVectorElement]): Key of the derivative for
                                                                          the value dictionary
    """

    def __init__(self, key: Union[str, tuple, ParameterVector, ParameterVectorElement]):
        self.key = key


class DirectEvaluation(EvaluationBase):
    """Container class for evaluation of derivatives of the QNN

    Args:
        key (Union[str, tuple, ParameterVector, ParameterVectorElement]): Key of the derivative for
                                                                          the value dictionary
        order (int): Order of the derivative
        argnum (Union[None, List[int]]): List of which arguments (x,param,param_obs) to
                                         differentiate
        return_grad_param (bool): Calculate the gradient with respect to the parameters
        return_grad_param_obs (bool): Calculate the gradient with respect to the observable
                                      parameters
        return_grad_x (bool): Calculate the gradient with respect to the input
        squared (bool): Calculate the squared value of the derivative
    """

    def __init__(
        self,
        key: Union[str, tuple, ParameterVector, ParameterVectorElement],
        order: int = 0,
        argnum: Union[None, List[int]] = None,
        return_grad_param: bool = False,
        return_grad_param_obs: bool = False,
        return_grad_x: bool = False,
        squared: bool = False,
    ):
        super().__init__(key)
        self.order = order
        self.argnum = argnum
        self.return_grad_param = return_grad_param
        self.return_grad_param_obs = return_grad_param_obs
        self.return_grad_x = return_grad_x
        self.squared = squared


class PostProcessingEvaluation(EvaluationBase):
    """Container class for the post processing evaluation of derivatives of the QNN

    Args:
        key (Union[str, tuple, ParameterVector, ParameterVectorElement]): Key of the derivative
                                                                          for the value dictionary
        evaluation_tuple (tuple): Tuple of direct_evaluation objects that are used for evaluating
                                  the values that are used in the post processing
        evaluation_function (callable): Function to evaluate the post processing
    """

    def __init__(
        self,
        key: Union[str, tuple, ParameterVector, ParameterVectorElement],
        evaluation_tuple: tuple,
        evaluation_function: callable,
    ):
        super().__init__(key)
        self.evaluation_tuple = evaluation_tuple
        self.evaluation_function = evaluation_function


def get_evaluation_class(
    val: Union[
        str,
        tuple,
        DirectEvaluation,
        PostProcessingEvaluation,
        ParameterVector,
        ParameterVectorElement,
    ],
    not_implemented: Union[None, List[str]] = None,
) -> Union[DirectEvaluation, PostProcessingEvaluation]:
    """Converts an input string to the direct or post processing evaluation object

    Args:
        val (Union[str, tuple, direct_evaluation, post_processing_evaluation,
                ParameterVector, ParameterVectorElement]): Input string or evaluation object

    Returns:
        Associated Expec object

    """

    # Create the evaluation object from the input variables
    if isinstance(val, str):

        if not_implemented and val in not_implemented:
            raise NotImplementedError("Evaluation not implemented:", val)

        if val == "f":
            return DirectEvaluation("f")
        elif val == "dfdx":
            return DirectEvaluation("dfdx", 1, argnum=[1], return_grad_x=True)
        elif val == "dfdxdx":
            return DirectEvaluation("dfdxdx", 2, argnum=[1, 1], return_grad_x=True)
        elif val == "laplace":
            return PostProcessingEvaluation(val, ("dfdxdx",), get_eval_laplace("dfdxdx"))
        elif val == "laplace_dp":
            return PostProcessingEvaluation(val, ("dfdpdxdx",), get_eval_laplace("dfdpdxdx"))
        elif val == "laplace_dop":
            return PostProcessingEvaluation(val, ("dfdopdxdx",), get_eval_laplace("dfdopdxdx"))
        elif val == "dfdp":
            return DirectEvaluation("dfdp", 1, argnum=[0], return_grad_param=True)
        elif val == "dfdpdp":
            return DirectEvaluation("dfdpdp", 2, argnum=[0, 0], return_grad_param=True)
        elif val == "dfdopdp":
            return DirectEvaluation(
                "dfdopdp", 2, argnum=[2, 0], return_grad_param=True, return_grad_param_obs=True
            )
        elif val == "dfdpdop":
            return DirectEvaluation(
                "dfdpdop", 2, argnum=[0, 2], return_grad_param=True, return_grad_param_obs=True
            )
        elif val == "dfdop":
            return DirectEvaluation("dfdop", 1, argnum=[2], return_grad_param_obs=True)
        elif val == "dfdopdop":
            return DirectEvaluation("dfdopdop", 2, argnum=[2, 2], return_grad_param_obs=True)
        elif val == "dfdpdx":
            return DirectEvaluation(
                "dfdpdx", 2, argnum=[0, 1], return_grad_param=True, return_grad_x=True
            )
        elif val == "dfdxdp":
            return DirectEvaluation(
                "dfdxdp", 2, argnum=[1, 0], return_grad_param=True, return_grad_x=True
            )
        elif val == "dfdxdxdp":
            return DirectEvaluation(
                "dfdxdxdp", 3, argnum=[1, 1, 0], return_grad_param=True, return_grad_x=True
            )
        elif val == "dfdxdpdx":
            return DirectEvaluation(
                "dfdxdpdx", 3, argnum=[1, 0, 1], return_grad_param=True, return_grad_x=True
            )
        elif val == "dfdpdxdx":
            return DirectEvaluation(
                "dfdpdxdx", 3, argnum=[0, 1, 1], return_grad_param=True, return_grad_x=True
            )
        elif val == "dfdopdx":
            return DirectEvaluation(
                "dfdopdx", 2, argnum=[2, 1], return_grad_param_obs=True, return_grad_x=True
            )
        elif val == "dfdopdxdx":
            return DirectEvaluation(
                "dfdopdxdx",
                3,
                argnum=[2, 1, 1],
                return_grad_param_obs=True,
                return_grad_x=True,
            )
        elif val == "fcc":
            return DirectEvaluation("fcc", squared=True)
        elif val == "dfccdx":
            return DirectEvaluation("dfccdx", 1, argnum=[1], return_grad_x=True, squared=True)
        elif val == "dfccdxdx":
            return DirectEvaluation("dfccdxdx", 2, argnum=[1, 1], return_grad_x=True, squared=True)
        elif val == "dfccdp":
            return DirectEvaluation("dfccdp", 1, argnum=[0], return_grad_param=True, squared=True)
        elif val == "dfccdpdp":
            return DirectEvaluation(
                "dfccdpdp", 2, argnum=[0, 0], return_grad_param=True, squared=True
            )
        elif val == "dfccdopdx":
            return DirectEvaluation(
                "dfccdopdx",
                2,
                argnum=[2, 1],
                return_grad_param_obs=True,
                return_grad_x=True,
                squared=True,
            )
        elif val == "dfccdop":
            return DirectEvaluation(
                "dfccdop", 1, argnum=[2], return_grad_param_obs=True, squared=True
            )
        elif val == "dfccdopdop":
            return DirectEvaluation(
                "dfccdopdop", 2, argnum=[2, 2], return_grad_param_obs=True, squared=True
            )
        elif val in ("var", "varf"):
            return PostProcessingEvaluation(val, ("f", "fcc"), eval_var)
        elif val in ("dvardx", "dvarfdx"):
            return PostProcessingEvaluation(val, ("f", "dfccdx", "dfdx"), eval_dvardx)
        elif val in ("dvardp", "dvarfdp"):
            return PostProcessingEvaluation(val, ("f", "dfccdp", "dfdp"), eval_dvardp)
        elif val in ("dvardop", "dvarfdop"):
            return PostProcessingEvaluation(val, ("f", "dfccdop", "dfdop"), eval_dvardop)
        elif val == "fischer":
            return None
        else:
            raise ValueError("Unknown input string:", val)
    elif isinstance(val, tuple):
        return get_direct_evaluation_from_tuple(val)
    elif isinstance(val, ParameterVectorElement) or isinstance(val, ParameterVector):
        evaluation = get_direct_evaluation_from_tuple((val,))
        evaluation.key = val  # Replace it to remove tuple structure
        return evaluation
    elif isinstance(val, DirectEvaluation):
        return val  # Nothing to do
    elif isinstance(val, PostProcessingEvaluation):
        return val  # Nothing to do
    else:
        raise TypeError("String expected, found type:", type(val))


def eval_var(value_dict: dict) -> np.ndarray:
    """Evaluate the variance of the QNN output

    Args:
        value_dict (dict): Dictionary of QNN derivatives values

    Returns:
        Variance of the QNN output
    """
    return value_dict["fcc"] - np.square(value_dict["f"])


def eval_dvardx(value_dict: dict) -> np.ndarray:
    """Evaluate the derivative of the variance with respect to the input

    Args:
        value_dict (dict): Dictionary of QNN derivatives values

    Returns:
        Derivative of the variance with respect to the input
    """
    return_val = np.zeros(value_dict["dfccdx"].shape)
    for i in range(value_dict["dfccdx"].shape[-1]):
        return_val[..., i] = value_dict["dfccdx"][..., i] - 2.0 * (
            np.multiply(
                value_dict["dfdx"][..., i],
                value_dict["f"],
            )
        )
    return return_val


def eval_dvardp(value_dict: dict) -> np.ndarray:
    """Evaluate the derivative of the variance with respect to the parameters

    Args:
        value_dict (dict): Dictionary of QNN derivatives values

    Returns:
        Derivative of the variance with respect to the parameters
    """
    return_val = np.zeros(value_dict["dfccdp"].shape)
    for i in range(value_dict["dfccdp"].shape[-1]):
        return_val[..., i] = value_dict["dfccdp"][..., i] - 2.0 * (
            np.multiply(
                value_dict["dfdp"][..., i],
                value_dict["f"],
            )
        )
    return return_val


def eval_dvardop(value_dict: dict) -> np.ndarray:
    """Evaluate the derivative of the variance with respect to the observable parameters

    Args:
        value_dict (dict): Dictionary of QNN derivatives values

    Returns:
        Derivative of the variance with respect to the observable parameters
    """
    return_val = np.zeros(value_dict["dfccdop"].shape)
    for i in range(value_dict["dfccdop"].shape[-1]):
        return_val[..., i] = value_dict["dfccdop"][..., i] - 2.0 * (
            np.multiply(
                value_dict["dfdop"][..., i],
                value_dict["f"],
            )
        )
    return return_val


def get_eval_laplace(todo: str) -> callable:
    """Evaluate of the Laplace Operation on the QNN output.

    Args:
        todo (str): String that defines the type of Laplace operation

    Returns:
        Function to evaluate the Laplace operation
    """

    def eval_laplace(value_dict: dict) -> np.ndarray:
        """Summing the diagonal of the feature Hessian matrix

        Args:
            value_dict (dict): Dictionary of QNN derivatives values

        Returns:
            Sum of the diagonal of the feature Hessian matrix
        """
        return_val = np.zeros(value_dict[todo].shape[:-2])
        for i in range(value_dict[todo].shape[-1]):
            return_val += value_dict[todo][..., i, i]
        return return_val

    return eval_laplace


def get_pick_elements(pick_indices: list, label: tuple) -> callable:
    """
    Function for generating the post-processing function for general derivatives

    Args:
        pick_indices (list): List of indices to pick single elements of derivatives
                                integer indices for specific parameters or "all" for all elements
        label (str): Label of the derivative

    Returns:
        Function to pick single elements of derivatives
    """

    def pick_elements(value_dict: dict) -> np.ndarray:
        """
        Function to pick single elements of general derivatives

        Args:
            value_dict (dict): Dictionary of QNN derivatives values

        Returns:
            Picked (single) elements of the derivatives
        """
        val = value_dict[label]
        if all(i == "all" for i in pick_indices):
            # If only ParameterVectors are used, the return the full array
            return val
        else:
            # If single elements occur, pick them
            slice_tuple = tuple()  # -> tuple that is used for picking the elements
            for i in pick_indices:
                if i == "all":
                    # Keep all elements
                    slice_tuple += (slice(None),)
                else:
                    # Pick single element
                    slice_tuple += (slice(i, i + 1, 1),)
            # Dimension at the beginning of the array
            extra_dim = len(val.shape) - len(slice_tuple)
            if extra_dim > 0:
                slice_tuple = (slice(None),) * extra_dim + slice_tuple

            # Return the picked elements
            return val[slice_tuple]

    return pick_elements


def get_direct_evaluation_from_tuple(
    todo: tuple,
) -> Union[DirectEvaluation, PostProcessingEvaluation]:
    """
    Function for generating the evaluation object from a tuple

    Handles derivatives of the QNN output with respect to a tuple of
    ParameterVector and ParameterVectorElement

    Args:
        todo (tuple): Tuple of ParameterVector and ParameterVectorElement entries that
                        define the derivative

    Returns:
        Evaluation object for the derivative, either direct_evaluation or
        post_processing_evaluation
    """

    # Generate direct evaluation object for the derivative
    # in case of a single ParameterVectorElement, post processing is needed
    return_grad_param = False
    return_grad_param_obs = False
    return_grad_x = False
    argnum = []
    order = 0
    val_tuple = tuple()
    pick_list = []

    for sub_val in todo:
        if isinstance(sub_val, ParameterVector) or isinstance(sub_val, ParameterVectorElement):
            # Increase order of derivative
            order += 1
            # Generate list for picking the single elements in post processing
            # and the tuple for the direct evaluation
            if isinstance(sub_val, ParameterVectorElement):
                val_tuple += (sub_val.vector,)
                pick_list.append(sub_val.index)
            else:
                val_tuple += (sub_val,)
                pick_list.append("all")
            # Generate argnum list and set return_grad flags
            if "param_obs" in sub_val.name:
                return_grad_param_obs = True
                argnum.append(2)
            elif "param" in sub_val.name:
                return_grad_param = True
                argnum.append(0)
            elif "x" in sub_val.name:
                return_grad_x = True
                argnum.append(1)
            else:
                raise ValueError("Unsupported parameter name:", sub_val.name)
        else:
            raise ValueError("Unsupported input type:", type(sub_val))

    # Generate the direct evaluation object
    evaluation = DirectEvaluation(
        val_tuple, order, argnum, return_grad_param, return_grad_param_obs, return_grad_x
    )

    if all(i == "all" for i in pick_list):
        # In case of only ParameterVectors direct evaluation is enough
        return evaluation
    else:
        # In case of minimal one ParameterVectorsElement,
        # post processing is needed for picking
        return PostProcessingEvaluation(
            todo, (evaluation,), get_pick_elements(pick_list, val_tuple)
        )
