"""Loss function implementations for QNNs."""

import abc
from typing import Union
import numpy as np
from collections.abc import Callable


class LossBase(abc.ABC):
    """Base class implementation for loss functions."""

    def __init__(self):
        self._opt_param_op = True

    def set_opt_param_op(self, opt_param_op: bool = True):
        """Sets the `opt_param_op` flag.

        Args:
            opt_param_op (bool): True, if operator has trainable parameters
        """
        self._opt_param_op = opt_param_op

    @property
    def loss_variance_available(self) -> bool:
        """Returns True if the loss function has a variance function."""
        return False

    @property
    @abc.abstractmethod
    def loss_args_tuple(self) -> tuple:
        """Returns evaluation tuple for loss calculation."""
        raise NotImplementedError()

    @property
    def variance_args_tuple(self) -> tuple:
        """Returns evaluation tuple for loss calculation."""
        raise NotImplementedError()

    @property
    @abc.abstractmethod
    def gradient_args_tuple(self) -> tuple:
        """Returns evaluation tuple for loss gradient calculation."""
        raise NotImplementedError()

    @abc.abstractmethod
    def value(self, value_dict: dict, **kwargs) -> float:
        """Calculates and returns the loss value."""
        raise NotImplementedError()

    def variance(self, value_dict: dict, **kwargs) -> float:
        """Calculates and returns the variance of the loss value."""
        raise NotImplementedError()

    @abc.abstractmethod
    def gradient(
        self, value_dict: dict, **kwargs
    ) -> Union[np.ndarray, tuple[np.ndarray, np.ndarray]]:
        """Calculates and returns the gradient of the loss."""
        raise NotImplementedError()

    def __add__(self, x):
        """Adds two loss functions."""
        if isinstance(x, LossBase):
            return _ComposedLoss(self, x, "+")
        elif isinstance(x, float) or isinstance(x, int) or callable(x):
            return _ComposedLoss(self, ConstantLoss(x), "+")
        else:
            raise ValueError("Only the addition with another loss functions are allowed!")

    def __radd__(self, x):
        """Adds two loss functions."""
        if isinstance(x, LossBase):
            return _ComposedLoss(x, self, "+")
        elif isinstance(x, float) or isinstance(x, int) or callable(x):
            return _ComposedLoss(ConstantLoss(x), self, "+")
        else:
            raise ValueError("Only the addition with another loss functions are allowed!")

    def __mul__(self, x):
        """Multiplies two loss functions."""
        if isinstance(x, LossBase):
            return _ComposedLoss(self, x, "*")
        elif isinstance(x, float) or isinstance(x, int) or callable(x):
            return _ComposedLoss(self, ConstantLoss(x), "*")
        else:
            raise ValueError("Only the addition with another loss functions are allowed!")

    def __rmul__(self, x):
        """Multiplies two loss functions."""
        if isinstance(x, LossBase):
            return _ComposedLoss(x, self, "*")
        elif isinstance(x, float) or isinstance(x, int) or callable(x):
            return _ComposedLoss(ConstantLoss(x), self, "*")
        else:
            raise ValueError("Only the addition with another loss functions are allowed!")

    def __sub__(self, x):
        """Subtracts two loss functions."""
        if isinstance(x, LossBase):
            return _ComposedLoss(self, x, "-")
        elif isinstance(x, float) or isinstance(x, int) or callable(x):
            return _ComposedLoss(self, ConstantLoss(x), "-")
        else:
            raise ValueError("Only the addition with another loss functions are allowed!")

    def __rsub__(self, x):
        """Subtracts two loss functions."""
        if isinstance(x, LossBase):
            return _ComposedLoss(x, self, "-")
        elif isinstance(x, float) or isinstance(x, int) or callable(x):
            return _ComposedLoss(ConstantLoss(x), self, "-")
        else:
            raise ValueError("Only the addition with another loss functions are allowed!")

    def __truediv__(self, x):
        """Divides two loss functions."""
        if isinstance(x, LossBase):
            return _ComposedLoss(self, x, "/")
        elif isinstance(x, float) or isinstance(x, int) or callable(x):
            return _ComposedLoss(self, ConstantLoss(x), "/")
        else:
            raise ValueError("Only the addition with another loss functions are allowed!")

    def __rtruediv__(self, x):
        """Divides two loss functions."""
        if isinstance(x, LossBase):
            return _ComposedLoss(x, self, "/")
        elif isinstance(x, float) or isinstance(x, int) or callable(x):
            return _ComposedLoss(ConstantLoss(x), self, "/")
        else:
            raise ValueError("Only the addition with another loss functions are allowed!")


class _ComposedLoss(LossBase):
    """Special class for composed loss functions

    Class for addition, multiplication, subtraction, and division of loss functions.

    Args:
        l1 (LossBase): First loss function
        l2 (LossBase): Second loss function
        composition (str): Composition of the loss functions ("+", "-", "*", "/")

    """

    def __init__(self, l1: LossBase, l2: LossBase, composition: str = "+"):
        super().__init__()
        self._l1 = l1
        self._l2 = l2
        self._composition = composition
        self._opt_param_op = self._l1._opt_param_op or self._l2._opt_param_op
        self._l1.set_opt_param_op(self._opt_param_op)
        self._l2.set_opt_param_op(self._opt_param_op)

    def set_opt_param_op(self, opt_param_op: bool = True):
        """Sets the `opt_param_op` flag.

        Args:
            opt_param_op (bool): True, if operator has trainable parameters
        """
        self._opt_param_op = opt_param_op
        self._l1.set_opt_param_op(opt_param_op)
        self._l2.set_opt_param_op(opt_param_op)

    @property
    def loss_variance_available(self) -> bool:
        if self._composition in ("*", "/"):
            return False
        else:
            return self._l1.loss_variance_available and self._l2.loss_variance_available

    @property
    def loss_args_tuple(self) -> tuple:
        """Returns evaluation tuple for composed loss calculation."""
        return tuple(set(self._l1.loss_args_tuple + self._l2.loss_args_tuple))

    @property
    def variance_args_tuple(self) -> tuple:
        """Returns evaluation tuple for composed variance calculation."""
        if self._composition in ("*", "/"):
            raise ValueError("Variance not available for composition: ", self._composition)
        else:
            return tuple(set(self._l1.variance_args_tuple + self._l2.variance_args_tuple))

    @property
    def gradient_args_tuple(self) -> tuple:
        """Returns evaluation tuple for composed gradient calculation."""
        return tuple(set(self._l1.gradient_args_tuple + self._l2.gradient_args_tuple))

    def value(self, value_dict: dict, **kwargs) -> float:
        """Calculates and returns the composed loss value.

        Args:
            value_dict (dict): Dictionary with values for the evaluation of the loss function

        Returns:
            float: Composed loss value
        """

        value_l1 = self._l1.value(value_dict, **kwargs)
        value_l2 = self._l2.value(value_dict, **kwargs)

        if self._composition == "*":
            return value_l1 * value_l2
        elif self._composition == "/":
            return value_l1 / value_l2
        elif self._composition == "+":
            return value_l1 + value_l2
        elif self._composition == "-":
            return value_l1 - value_l2
        else:
            raise ValueError("Unknown composition: ", self._composition)

    def variance(self, value_dict: dict, **kwargs) -> float:
        """Calculates and returns the composed variance value.

        Args:
            value_dict (dict): Dictionary with values for the evaluation of the loss function

        Returns:
            float: Composed variance value
        """

        if self._composition in ("*", "/"):
            raise ValueError("Variance not available for composition: ", self._composition)

        var_l1 = self._l1.variance(value_dict, **kwargs)
        var_l2 = self._l2.variance(value_dict, **kwargs)

        if self._composition == "+":
            return var_l1 + var_l2
        elif self._composition == "-":
            return var_l1 + var_l2
        else:
            raise ValueError("Unknown composition: ", self._composition)

    def gradient(
        self, value_dict: dict, **kwargs
    ) -> Union[np.ndarray, tuple[np.ndarray, np.ndarray]]:
        """Calculates and returns the gradient of the composed loss.

        Args:
            value_dict (dict): Dictionary with values for the evaluation of the
                loss function gradient

        Returns:
            Union[np.ndarray, tuple[np.ndarray, np.ndarray]]: Gradient of the composed
                loss function

        """

        grad_l1 = self._l1.gradient(value_dict, **kwargs)
        grad_l2 = self._l2.gradient(value_dict, **kwargs)
        if self._composition in ("*", "/"):
            value_l1 = self._l1.value(value_dict, **kwargs)
            value_l2 = self._l2.value(value_dict, **kwargs)

        if isinstance(grad_l1, tuple) and isinstance(grad_l2, tuple):
            if self._composition == "*":
                # (f*g)' = f'*g + f*g'
                return tuple(
                    [
                        np.add(grad_l1[i] * value_l2, grad_l2[i] * value_l1)
                        for i in range(len(grad_l1))
                    ]
                )
            elif self._composition == "/":
                # (f/g)' = (f'*g - f*g')/g^2
                return tuple(
                    [
                        np.subtract(grad_l1[i] / value_l2, value_l1 / value_l2 * grad_l2[i])
                        for i in range(len(grad_l1))
                    ]
                )
            elif self._composition == "+":
                return tuple([np.add(grad_l1[i], grad_l2[i]) for i in range(len(grad_l1))])
            elif self._composition == "-":
                return tuple([np.subtract(grad_l1[i], grad_l2[i]) for i in range(len(grad_l1))])
            else:
                raise ValueError("Unknown composition: ", self._composition)

        elif not isinstance(grad_l1, tuple) and not isinstance(grad_l2, tuple):
            if self._composition == "*":
                # (f*g)' = f'*g + f*g'
                return np.add(grad_l1 * value_l2, grad_l2 * value_l1)
            elif self._composition == "/":
                # (f/g)' = (f'*g - f*g')/g^2
                return np.subtract(grad_l1 / value_l2, value_l1 / value_l2 * grad_l2)
            elif self._composition == "+":
                return np.add(grad_l1, grad_l2)
            elif self._composition == "-":
                return np.subtract(grad_l1, grad_l2)
            else:
                raise ValueError("Unknown composition: ", self._composition)
        else:
            raise ValueError("Gradient output structure types do not match!")


class ConstantLoss(LossBase):
    """Class for constant or independent loss functions.

    Args:
        value (Union[int, float, Callable[[int],float]]): Constant value or function depending
            on the iterations returning a constant value.
    """

    def __init__(self, value: Union[int, float, Callable[[int], float]] = 0.0):
        super().__init__()
        if callable(value):
            self._value = value
        else:
            self._value = float(value)

    @property
    def loss_variance_available(self) -> bool:
        """Returns True if the loss function has a variance function."""
        return True

    @property
    def loss_args_tuple(self) -> tuple:
        """Returns empty evaluation tuple for loss calculation."""
        return tuple()

    @property
    def variance_args_tuple(self) -> tuple:
        """Returns empty evaluation tuple for variance calculation."""
        return tuple()

    @property
    def gradient_args_tuple(self) -> tuple:
        """Returns empty evaluation tuple for gradient calculation."""
        return tuple()

    def value(self, value_dict: dict, **kwargs) -> float:
        """Returns constant or iteration dependent loss value

        Args:
            value_dict (dict): Contains calculated values of the model
            iteration (int): iteration number, if value is a callable function
        """
        if callable(self._value):
            if "iteration" not in kwargs:
                raise AttributeError("If value is callable, iteration is required.")
            return self._value(kwargs["iteration"])
        return self._value

    def variance(self, value_dict: dict, **kwargs) -> float:
        """Returns zero variance of the constant loss function."""
        return 0.0

    def gradient(
        self, value_dict: dict, **kwargs
    ) -> Union[np.ndarray, tuple[np.ndarray, np.ndarray]]:
        """Returns zero gradient value

        Args:
            value_dict (dict): Contains calculated values of the model
        """
        dp = np.zeros(value_dict["param"].shape)
        dop = np.zeros(value_dict["param_op"].shape)
        if self._opt_param_op:
            return dp, dop
        return dp


class SquaredLoss(LossBase):
    """Squared loss for regression."""

    @property
    def loss_variance_available(self) -> bool:
        """Returns True since the squared loss function has a variance function."""
        return True

    @property
    def loss_args_tuple(self) -> tuple:
        """Returns evaluation tuple for the squared loss calculation."""
        return ("f",)

    @property
    def variance_args_tuple(self) -> tuple:
        """Returns evaluation tuple for the squared loss variance calculation."""
        return ("f", "var")

    @property
    def gradient_args_tuple(self) -> tuple:
        """Returns evaluation tuple for the squared loss gradient calculation."""
        if self._opt_param_op:
            return ("f", "dfdp", "dfdop")
        return ("f", "dfdp")

    def value(self, value_dict: dict, **kwargs) -> float:
        r"""Calculates the squared loss.

        This function calculates the squared loss between the values in value_dict and ground_truth
        as

        .. math::
            \sum_i w_i \left|f\left(x_i\right)-f_ref\left(x_i\right)\right|^2

        Args:
            value_dict (dict): Contains calculated values of the model
            ground_truth (np.ndarray): The true values :math:`f_ref\left(x_i\right)`
            weights (np.ndarray): Weight for each data point, if None all data points count the same

        Returns:
            Loss value
        """
        if "ground_truth" not in kwargs:
            raise AttributeError("SquaredLoss requires ground_truth.")
        ground_truth = kwargs["ground_truth"]
        if "weights" in kwargs and kwargs["weights"] is not None:
            weights = kwargs["weights"]
        else:
            weights = np.ones_like(ground_truth)
        return np.sum(np.multiply(np.square(value_dict["f"] - ground_truth), weights))

    def variance(self, value_dict: dict, **kwargs) -> float:
        r"""Calculates the approximated variance of the squared loss.

        This function calculates the approximated variance of the squared loss

        .. math::
            4\sum_i w_i \left|f\left(x_i\right)-f_ref\left(x_i\right)\right|^2 \sigma_f^2(x_i)

        Args:
            value_dict (dict): Contains calculated values of the model
            ground_truth (np.ndarray): The true values :math:`f_ref\left(x_i\right)`
            weights (np.ndarray): Weight for each data point, if None all data points count the same

        Returns:
            Loss value
        """
        if "ground_truth" not in kwargs:
            raise AttributeError("SquaredLoss requires ground_truth.")
        ground_truth = kwargs["ground_truth"]
        if "weights" in kwargs and kwargs["weights"] is not None:
            weights = kwargs["weights"]
        else:
            weights = np.ones_like(ground_truth)

        diff_square = np.multiply(weights, np.square(value_dict["f"] - ground_truth))
        return np.sum(4 * np.multiply(diff_square, value_dict["var"]))

    def gradient(
        self, value_dict: dict, **kwargs
    ) -> Union[np.ndarray, tuple[np.ndarray, np.ndarray]]:
        r"""Returns the gradient of the squared loss.

        This function calculates the gradient of the squared loss between the values in value_dict
        and ground_truth as

        .. math::
           2\sum_j \sum_i w_i \left(f\left(x_i\right)-f_ref\left(x_i\right)\right) \frac{\partial f(x_i)}{\partial p_j}

        Args:
            value_dict (dict): Contains calculated values of the model
            ground_truth (np.ndarray): The true values :math:`f_ref\left(x_i\right)`
            weights (np.ndarray): Weight for each data point, if None all data points count the same
            multiple_output (bool): True if the QNN has multiple outputs

        Returns:
            Gradient values
        """

        if "ground_truth" not in kwargs:
            raise AttributeError("SquaredLoss requires ground_truth.")

        ground_truth = kwargs["ground_truth"]
        if "weights" in kwargs and kwargs["weights"] is not None:
            weights = kwargs["weights"]
        else:
            weights = np.ones_like(ground_truth)
        multiple_output = "multiple_output" in kwargs and kwargs["multiple_output"]

        weighted_diff = np.multiply((value_dict["f"] - ground_truth), weights)

        if value_dict["dfdp"].shape[0] == 0:
            d_p = np.array([])
        else:
            if multiple_output:
                d_p = 2.0 * np.einsum("ij,ijk->k", weighted_diff, value_dict["dfdp"])
            else:
                d_p = 2.0 * np.einsum("j,jk->k", weighted_diff, value_dict["dfdp"])

        # Extra code for the cost operator derivatives
        if not self._opt_param_op:
            return d_p

        if value_dict["dfdop"].shape[0] == 0:
            d_op = np.array([])
        else:
            if multiple_output:
                d_op = 2.0 * np.einsum("ij,ijk->k", weighted_diff, value_dict["dfdop"])
            else:
                d_op = 2.0 * np.einsum("j,jk->k", weighted_diff, value_dict["dfdop"])
        return d_p, d_op


class ODELoss(LossBase):
    """Squared loss for regression of Ordinary Differential Equations (ODEs)."""

    #ODELoss requires the ODE_functional and ODE_functional_gradient and initial values
    #ODELoss and ODE_functional_gradient are functions
    def __init__(self, ODE_functional: Union[bool] = None, ODE_functional_gradient: Union[bool] = None, initial_vec: np.ndarray = None, eta = np.float64(1.0), boundary_handling = "pinned", true_solution = None):
        super().__init__()
        self._ODE_functional = self.create_QNN_ode_loss_format(ODE_functional) #F[x, f, f_, f__] returns the value of the ODE functional shape: (n_samples, n_outputs)
        self._ODE_functional_gradient = self.create_QNN_ode_gradient_format(ODE_functional_gradient) #(dF/df, dF/df_, dF/df__) returns the value of the ODE functional shape: (n_samples, n_outputs)
        self.initial_vec = initial_vec
        self.eta = eta
        self.boundary_handling = boundary_handling 
        self.true_solution = true_solution

    @property
    def loss_args_tuple(self) -> tuple:
        """Returns evaluation tuple for the squared loss calculation."""
        if len(self.initial_vec) == 1:  #if only one initial value is given, we have a 1rst order ODE
            return ("f", "dfdx")
        elif len(self.initial_vec) == 2:
            return ("f", "dfdx", "dfdxdx")
        
    def get_true_solution(self) -> np.ndarray:
        return self.true_solution

    @property
    def gradient_args_tuple(self) -> tuple:
        """Returns evaluation tuple for the squared loss gradient calculation."""
        if self._opt_param_op:
            if len(self.initial_vec) == 1:  #if only one initial value is given, we have a 1rst order ODE
                return ("f", "dfdx", "dfdp", "dfdxdp", "dfdop", "dfdxdop")
            elif len(self.initial_vec) == 2:
                return ("f", "dfdx", "dfdxdx", "dfdp", "dfdxdp", "dfdxdxdp", "dfdop", "dfdxdop", "dfdxdxdop")        
            
        if len(self.initial_vec) == 1:  #if only one initial value is given, we have a 1rst order ODE
            return ("f", "dfdx", "dfdp", "dfdxdp")
        elif len(self.initial_vec) == 2:
            return ("f", "dfdx", "dfdxdx", "dfdp", "dfdxdp", "dfdxdxdp")
    
    def _Ansatz_to_Floating_Boundary_Ansatz(self, value_dict: dict, gradient_calculation = True, **kwargs) -> dict:
        """
        Converts the ansatz to a floating boundary ansatz by setting the initial values to the initial values of the ODE

        If 1rst order ODE: f(x_0) = f_0 and f'(x_0) free to optimize and f''(x) = 0 to save computational resources.
        If 2nd order ODE: f(x_0) = f_0 and f'(x_0) = f_0' and f''(x).

        Args:
            value_dict (dict): Contains calculated values of the model
            gradient_calculation (bool): True if the gradient is calculated
        
        Returns:
            value_dict_floating (dict): Contains the values of the model with the initial values set to the initial values of the ODE

        
        """
        value_dict_floating = value_dict
        value_dict_floating["f"][0] = self.initial_vec[0]  #f(x_0) = f_0

        if len(self.initial_vec) == 2:  #if only one initial value is given, we have a 1rst order ODE
            value_dict_floating["dfdx"][0] = self.initial_vec[1] #f'(x_0) = f_0'
            value_dict_floating["dfdxdx"] = value_dict_floating["dfdxdx"]
        else:
            value_dict_floating["dfdxdx"] = np.zeros_like(value_dict_floating["f"])



        if gradient_calculation:
            value_dict_floating["dfdp"][0] =  value_dict_floating["dfdp"][0]*0
            if len(self.initial_vec) == 2:  #if only one initial value is given, we have a 1rst order ODE
                value_dict_floating["dfdxdp"][0] = value_dict_floating["dfdxdp"][0]*0   
            else:
                value_dict_floating["dfdxdxdp"] = np.zeros((value_dict_floating["dfdxdx"].shape[0], 1, 1, value_dict_floating["dfdp"].shape[1]))

        return value_dict_floating
                #value_dict["dfdp"] shape: (n_samples, n_params) f
                #value_dict["dfdpdx"] shape: (n_samples, 1, n_params)

    def value(self, value_dict: dict, **kwargs) -> float:
        r"""Calculates the squared loss.

        This function calculates the squared loss between the values in value_dict and ground_truth
        as

        .. math::
            \sum_i w_i \left|f\left(x_i\right)-f_ref\left(x_i\right)\right|^2

        Args:
            value_dict (dict): Contains calculated values of the model
            ground_truth (np.ndarray): The true values :math:`f_ref\left(x_i\right)`
            weights (np.ndarray): Weight for each data point, if None all data points count the same

        Returns:
            Loss value
        """
        if "ground_truth" not in kwargs:
            raise AttributeError("SquaredLoss requires ground_truth.")
        ground_truth = kwargs["ground_truth"]
        if "weights" in kwargs and kwargs["weights"] is not None:
            weights = kwargs["weights"]
        else:
            weights = np.ones_like(ground_truth)

        functional_loss, initial_value_loss_f, initial_value_loss_df = 0, 0, 0
        if self.boundary_handling == "pinned":
            #print("Weights", weights)
            #print("Loss tensor", np.multiply(np.square(self._ODE_functional(value_dict) - ground_truth), weights))
            functional_loss = np.sum(np.multiply(np.square(self._ODE_functional(value_dict) - ground_truth), weights)) #L_theta = sum_i w_i (F(x_i, f_i, f_i', f_i'') - 0)^2, shape (n_samples, n_outputs)
            #print("Initial vec", self.initial_vec[0])
            #print("Initial value", value_dict["f"][0])

            initial_value_loss_f = self.eta*(np.square(value_dict["f"][0] - self.initial_vec[0]))     #L_theta +=  (f(x_i) - f_0)^2 #Pinned boundary to be included
            try:
                initial_value_loss_df = self.eta*(np.square(value_dict["dfdx"][0] - self.initial_vec[1])) #L_theta +=  (f'(x_i) - f_0')^2
            except:
                pass
        elif self.boundary_handling == "floating":
            value_dict = self._Ansatz_to_Floating_Boundary_Ansatz(value_dict, gradient_calculation = False)
            functional_loss = np.sum(np.multiply(np.square(self._ODE_functional(value_dict) - ground_truth), weights)) #L_theta = sum_i w_i (F(x_i, f_i, f_i', f_i'') - 0)^2, shape (n_samples, n_outputs)


        #print("Functional loss: ", functional_loss)
        #print("Initial value loss f: ", initial_value_loss_f)
        #print("Total: ", functional_loss + initial_value_loss_f + initial_value_loss_df)
        return functional_loss + initial_value_loss_f + initial_value_loss_df
    

  

    def gradient(
        self, value_dict: dict, **kwargs
    ) -> Union[np.ndarray, tuple[np.ndarray, np.ndarray]]:
        r"""Returns the gradient of the squared loss.

        This function calculates the gradient of the squared loss between the values in value_dict
        and ground_truth as

        .. math::
           \sum_j \sum_i w_i \left(f\left(x_i\right)-f_ref\left(x_i\right)\right) \frac{\partial f(x_i)}{\partial p_j}

        Args:
            value_dict (dict): Contains calculated values of the model
            ground_truth (np.ndarray): The true values :math:`f_ref\left(x_i\right)`
            weights (np.ndarray): Weight for each data point, if None all data points count the same
            multiple_output (bool): True if the QNN has multiple outputs

        Returns:
            Gradient values
        """

        if "ground_truth" not in kwargs:
            raise AttributeError("SquaredLoss requires ground_truth.")

        ground_truth = kwargs["ground_truth"]
        if "weights" in kwargs and kwargs["weights"] is not None:
            weights = kwargs["weights"]
        else:
            weights = np.ones_like(ground_truth)
        multiple_output = "multiple_output" in kwargs and kwargs["multiple_output"]

        weighted_diff = np.multiply((self._ODE_functional(value_dict) - ground_truth), weights) # shape: (n_samples, n_outputs) 
        #(F(x_0, f_0, f_0', f_0'')
        #(F(x_1, f_1, f_1', f_1''), ...

        #print("Params", value_dict["p"])
        if value_dict["dfdp"].shape[0] == 0:
            d_p = np.array([])
        else:
            if multiple_output:
                d_p = 2.0 * np.einsum("ij,ijk->k", weighted_diff, value_dict["dfdp"]) #shape: (n_samples, n_outputs, n_params) -> (n_params)
            else:
                #Mixed derivatives: dfdpdx = dfdxdp and dfdpdxdx = dfdxdxdp

                #value_dict["dfdp"] shape: (n_samples, n_params)
                #value_dict["dfdpdx"] shape: (n_samples, 1, n_params)
                #value_dict["dfdpdxdx"] shape: (n_samples, n_params)
                #print("dfdp shape: ", value_dict["dfdp"].shape) #shape: (n_samples, n_params)
                #print("dfdpdx shape: ", value_dict["dfdpdx"].shape) #shape: (n_samples, n_params, 1)
                #print("dfdpdxdx shape: ", value_dict["dfdpdxdx"].shape) #shape: (n_samples, n_params, 1, 1)

                #print("dfdp shape: ", value_dict["dfdp"].shape) #shape: (n_samples, n_params)
                #print("dfdxdp shape: ", value_dict["dfdxdp"].shape) #shape: (n_samples, 1, n_params)
                #print("dfdxdxdp shape: ", value_dict["dfdxdxdp"].shape) #shape: (n_samples, 1, 1, n_params)
                d_p = np.zeros(value_dict["dfdp"].shape[1])
                if self.boundary_handling == "pinned":
                    d_p += 2.0 * self.eta*(value_dict["f"][0] - self.initial_vec[0])*value_dict["dfdp"][0, :] #shape: (n_params)
                    try:
                        d_p += 2.0*self.eta*np.sum(value_dict["dfdx"][0] - self.initial_vec[1])*value_dict["dfdxdp"][0, 0, :] #shape: (n_params)
                    except:
                        pass
                elif self.boundary_handling == "floating":
                    value_dict = self._Ansatz_to_Floating_Boundary_Ansatz(value_dict, gradient_calculation = True)

                d_ODE_functional_dD = self._ODE_functional_gradient(value_dict) # shape: (3, n_samples, n_params)

                if len(self.initial_vec) == 1 and self.boundary_handling == "pinned":  
                    dfdp_like = d_ODE_functional_dD[0]*value_dict["dfdp"] + d_ODE_functional_dD[1]*value_dict["dfdxdp"][:,0,:] #shape: (n_samples, n_params)
                else:
                    dfdp_like = d_ODE_functional_dD[0]*value_dict["dfdp"] + d_ODE_functional_dD[1]*value_dict["dfdxdp"][:,0,:] +  d_ODE_functional_dD[2]*value_dict["dfdxdxdp"][:,0,0,:]

                d_p += 2.0 * np.einsum("j,jk->k", weighted_diff, dfdp_like) #shape: (n_samples, n_params) -> (n_params)

        if not self._opt_param_op:
            return d_p

        if value_dict["dfdop"].shape[0] == 0:
            d_op = np.array([])
        else:
            if multiple_output:
                d_op = 2.0 * np.einsum("ij,ijk->k", weighted_diff, value_dict["dfdop"]) #shape: (n_samples, n_outputs, n_params_op) -> (n_params_op)
            else:
                d_ODE_functional_dD = self._ODE_functional_gradient(value_dict) # shape: (3, n_samples, n_params)

                if len(self.initial_vec) == 1:  #if only one initial value is given, we have a 1rst order ODE
                    dfdp_like = d_ODE_functional_dD[0]*value_dict["dfdop"] + d_ODE_functional_dD[1]*value_dict["dfdxdop"][:,0,:] #shape: (n_samples, n_params)
                else:
                    dfdp_like = d_ODE_functional_dD[0]*value_dict["dfdop"] + d_ODE_functional_dD[1]*value_dict["dfdxdop"][:,0,:] +  d_ODE_functional_dD[2]*value_dict["dfdxdxdop"][:,0,0,:]

                d_op = 2.0 * np.einsum("j,jk->k", weighted_diff, dfdp_like) #shape: (n_samples, n_params) -> (n_params)
                d_op += 2.0 * self.eta*(value_dict["f"][0] - self.initial_vec[0])*value_dict["dfdop"][0, :] #shape: (n_params)
                try:
                    d_op += 2.0*self.eta*np.sum(value_dict["dfdx"][0] - self.initial_vec[1])*value_dict["dfdxdop"][0, 0, :] #shape: (n_params)
                except:
                    pass

        return d_p, d_op
    
    def get_derivatives_list_format(self, loss_values):
        """
        #TODO: changed to be similar to loss_args_tuple
        Args:
            loss_values (dict): Contains calculated values of the model
        Returns:
            x (np.ndarray): The input values
            f (np.ndarray): The output values
            dfdx (np.ndarray): The first derivative values
            dfdxdx (np.ndarray): The second derivative values
        
        """
        try: 
            dfdxdx = loss_values["dfdxdx"][:,0,0]
        except:
            dfdxdx = np.zeros_like(loss_values["f"])   
        return loss_values["x"], loss_values["f"], loss_values["dfdx"][:,0], dfdxdx
    
    def create_QNN_ode_loss_format(self, loss_functional):
        """
        Args:
            loss_functional (function): The loss function for the ODE problem
        Returns:
            QNN_loss (function): The loss function for the QNN with input in the format of the QNN tuple derivatives

        """
        def QNN_loss(QNN_derivatives_values):
            """
            Defines the loss function for the ODE problem
            f_array is assumed to be [x, f, dfdx, dfdxdx]
            
            """
            return loss_functional(self.get_derivatives_list_format(QNN_derivatives_values))
        return QNN_loss

    def create_QNN_ode_gradient_format(self, ODE_functional_gradient):
        """
        Args:
            ODE_functional_gradient (function): The analytical gradient function of the loss function for the ODE problem
        Returns:
            QNN_gradient (function): The gradient of the loss function for the QNN with input in the format of the QNN tuple derivatives        
        """
        def QNN_gradient(QNN_derivatives_values):
            """
            Defines the gradient of the loss function for the ODE problem
            f_array is assumed to be [x, f, dfdx, dfdxdx]
            
            """
            dFdf, dFdfdx, dFdfdxdx = ODE_functional_gradient(self.get_derivatives_list_format(QNN_derivatives_values))
            n_param = QNN_derivatives_values["dfdp"].shape[1]
            
            grad_envelope_list = np.zeros((3, QNN_derivatives_values["x"].shape[0], n_param)) # shape (3, n, p)
            grad_envelope_list[0,:,:] = np.tile(dFdf, (n_param, 1)).T  
            grad_envelope_list[1,:,:] = np.tile(dFdfdx, (n_param, 1)).T
            grad_envelope_list[2,:,:] =  np.tile(dFdfdxdx, (n_param, 1)).T
            return grad_envelope_list
        return QNN_gradient



class VarianceLoss(LossBase):
    r"""Variance loss for regression.

    Args:
        alpha (float, Callable[[int], float]): Weight value :math:`\alpha`
    """

    def __init__(self, alpha: Union[float, Callable[[int], float]] = 0.005):
        super().__init__()
        self._alpha = alpha

    @property
    def loss_variance_available(self) -> bool:
        """Returns True since we neglect the variance of the variance."""
        return True

    @property
    def loss_args_tuple(self) -> tuple:
        """Returns evaluation tuple for loss calculation."""
        return ("var",)

    @property
    def variance_args_tuple(self) -> tuple:
        """Returns evaluation tuple for variance calculation."""
        return tuple()

    @property
    def gradient_args_tuple(self) -> tuple:
        """Returns evaluation tuple for loss gradient calculation."""
        if self._opt_param_op:
            return ("var", "dvardp", "dvardop")
        return ("var", "dvardp")

    def value(self, value_dict: dict, **kwargs) -> float:
        r"""Returns the variance.

        This function returns the weighted variance as

        .. math::
            L_\operatorname{Var} = \alpha \sum_i \operatorname{Var}_i

        Args:
            value_dict (dict): Contains calculated values of the model
            iteration (int): iteration number, if alpha is a callable function

        Returns:
            Loss value
        """

        if callable(self._alpha):
            if "iteration" not in kwargs:
                raise AttributeError("If alpha is callable, iteration is required.")
            alpha = self._alpha(kwargs["iteration"])
        else:
            alpha = self._alpha

        return alpha * np.sum(value_dict["var"])

    def variance(self, value_dict: dict, **kwargs) -> float:
        """Returns 0 since we neglect the variance of the variance."""
        return 0.0

    def gradient(
        self, value_dict: dict, **kwargs
    ) -> Union[np.ndarray, tuple[np.ndarray, np.ndarray]]:
        """Returns the gradient of the variance.

        This function calculates the gradient of the variance values in value_dict.

        Args:
            value_dict (dict): Contains calculated values of the model
            iteration (int): iteration number, if variance_factor is a function
            multiple_output (bool): True if the QNN has multiple outputs

        Returns:
            Gradient values
        """
        if callable(self._alpha):
            if "iteration" not in kwargs:
                raise AttributeError("If alpha is callable, iteration is required.")
            alpha = self._alpha(kwargs["iteration"])
        else:
            alpha = self._alpha

        multiple_output = "multiple_output" in kwargs and kwargs["multiple_output"]
        if value_dict["dfdp"].shape[0] == 0:
            d_p = np.array([])
        else:
            if multiple_output:
                d_p = alpha * np.sum(value_dict["dvardp"], axis=(0, 1))
            else:
                d_p = alpha * np.sum(value_dict["dvardp"], axis=0)

        # Extra code for the cost operator derivatives
        if not self._opt_param_op:
            return d_p

        if value_dict["dfdop"].shape[0] == 0:
            d_op = np.array([])
        else:
            if multiple_output:
                d_op = alpha * np.sum(value_dict["dvardop"], axis=(0, 1))
            else:
                d_op = alpha * np.sum(value_dict["dvardop"], axis=0)

        return d_p, d_op


class ParameterRegularizationLoss(LossBase):
    r"""Loss for parameter regularization.

    Possible implementations:

    * ``"L1"``: :math:`L=\alpha \sum_i \left|p_i\right|`
    * ``"L2"``: :math:`L=\alpha \sum_i p_i^2`

    Args:
        alpha (float, Callable[[int], float]): Weight value :math:`\alpha`
        mode (str): Type of regularization, either 'L1' or 'L2' (default: 'L2').
        parameter_list (list): List of parameters to regularize, None: all (default: None).
        parameter_operator_list (list): List of operator parameters to regularize, None: all
            (default: []).
    """

    def __init__(
        self,
        alpha: Union[float, Callable[[int], float]] = 0.005,
        mode: str = "L2",
        parameter_list: Union[list, None] = None,
        parameter_operator_list: Union[list, None] = [],
    ):
        super().__init__()
        self._alpha = alpha
        self._mode = mode
        if self._mode not in ["L1", "L2"]:
            raise ValueError("Type must be 'L1' or 'L2'!")

        self._parameter_list = parameter_list
        self._parameter_operator_list = parameter_operator_list

    @property
    def loss_variance_available(self) -> bool:
        """Returns True since variance is zero (and available)."""
        return True

    @property
    def loss_args_tuple(self) -> tuple:
        """Returns evaluation tuple for loss calculation."""
        return tuple()

    @property
    def variance_args_tuple(self) -> tuple:
        """Returns evaluation tuple for loss calculation."""
        return tuple()

    @property
    def gradient_args_tuple(self) -> tuple:
        """Returns evaluation tuple for loss gradient calculation."""
        return tuple()

    def value(self, value_dict: dict, **kwargs) -> float:
        r"""Returns the variance.

        This function returns the weighted variance as

        .. math::
            L_\text{var} = \alpha \sum_i \var_i

        Args:
            value_dict (dict): Contains calculated values of the model
            iteration (int): iteration number, if alpha is a callable function

        Returns:
            Loss value
        """

        if callable(self._alpha):
            if "iteration" not in kwargs:
                raise AttributeError("If alpha is callable, iteration is required.")
            alpha = self._alpha(kwargs["iteration"])
        else:
            alpha = self._alpha

        loss = 0.0
        if self._parameter_list is None:
            if self._mode == "L1":
                loss += np.sum(np.abs(value_dict["param"]))
            elif self._mode == "L2":
                loss += np.sum(np.square(value_dict["param"]))
            else:
                raise ValueError("Type must be L1 or L2!")
        else:
            if self._mode == "L1":
                loss += np.sum(np.abs(value_dict["param"][self._parameter_list]))
            elif self._mode == "L2":
                loss += np.sum(np.square(value_dict["param"][self._parameter_list]))
            else:
                raise ValueError("Type must be L1 or L2!")

        if self._opt_param_op:
            if self._parameter_list is None:
                if self._mode == "L1":
                    loss += np.sum(np.abs(value_dict["param_op"]))
                elif self._mode == "L2":
                    loss += np.sum(np.square(value_dict["param_op"]))
                else:
                    raise ValueError("Type must be L1 or L2!")
            else:
                if self._mode == "L1":
                    loss += np.sum(np.abs(value_dict["param_op"][self._parameter_operator_list]))
                elif self._mode == "L2":
                    loss += np.sum(
                        np.square(value_dict["param_op"][self._parameter_operator_list])
                    )
                else:
                    raise ValueError("Type must be L1 or L2!")

        return alpha * loss

    def variance(self, value_dict: dict, **kwargs) -> float:
        """Returns 0 since the variance is equal to zero."""
        return 0.0

    def gradient(
        self, value_dict: dict, **kwargs
    ) -> Union[np.ndarray, tuple[np.ndarray, np.ndarray]]:
        """Returns the gradient of the variance.

        This function calculates the gradient of the variance values in value_dict.

        Args:
            value_dict (dict): Contains calculated values of the model
            iteration (int): iteration number, if variance_factor is a function

        Returns:
            Gradient values
        """
        if callable(self._alpha):
            if "iteration" not in kwargs:
                raise AttributeError("If alpha is callable, iteration is required.")
            alpha = self._alpha(kwargs["iteration"])
        else:
            alpha = self._alpha

        d_p = np.zeros_like(value_dict["param"])
        if self._parameter_list is None:
            if self._mode == "L1":
                d_p = alpha * np.sign(value_dict["param"])
            elif self._mode == "L2":
                d_p = alpha * 2.0 * value_dict["param"]
            else:
                raise ValueError("Type must be L1 or L2!")
        else:
            if self._mode == "L1":
                d_p[self._parameter_list] = alpha * np.sign(
                    value_dict["param"][self._parameter_list]
                )
            elif self._mode == "L2":
                d_p[self._parameter_list] = alpha * 2.0 * value_dict["param"][self._parameter_list]
            else:
                raise ValueError("Type must be L1 or L2!")

        # Extra code for the cost operator derivatives
        if not self._opt_param_op:
            return d_p

        d_op = np.zeros_like(value_dict["param_op"])
        if self._parameter_operator_list is None:
            if self._mode == "L1":
                d_op = alpha * np.sign(value_dict["param_op"])
            elif self._mode == "L2":
                d_op = alpha * 2.0 * value_dict["param_op"]
            else:
                raise ValueError("Type must be L1 or L2!")
        else:
            if self._mode == "L1":
                d_op[self._parameter_operator_list] = alpha * np.sign(
                    value_dict["param_op"][self._parameter_operator_list]
                )
            elif self._mode == "L2":
                d_op[self._parameter_operator_list] = (
                    alpha * 2.0 * value_dict["param_op"][self._parameter_operator_list]
                )
            else:
                raise ValueError("Type must be L1 or L2!")

        return d_p, d_op