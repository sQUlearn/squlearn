import numpy as np
import copy
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector,ParameterExpression
from typing import List, Union, Callable
import time
from qiskit.primitives import Estimator
from qiskit.converters import circuit_to_dag
from hashlib import blake2b

import dill as pickle

# TODO: better indexing of lists for tree objects for less verbose code

def hash_circuit(circuit:QuantumCircuit):
    from qiskit.primitives.utils import _circuit_key
    return _circuit_key(circuit)
    #return blake2b(str(_circuit_key(circuit)).encode("utf-8"), digest_size=20).hexdigest() # faster for comparison slower for generation


class OpTreeElementBase():

    def __init__(self,children_list=None,factor_list=None,operation_list=None) -> None:

        if children_list is not None:
            self._children_list = children_list
            if factor_list is not None:
                if len(children_list) != len(factor_list):
                    raise ValueError("circuit_list and factor_list must have the same length")
                self._factor_list = factor_list
            else:
                self._factor_list = [1.0 for i in range(len(children_list))]

            if operation_list is not None:
                if len(children_list) != len(operation_list):
                    raise ValueError("circuit_list and operation_list must have the same length")
                self._operation_list = operation_list
            else:
                self._operation_list = [None for i in range(len(children_list))]

        else:
            self._children_list = []
            self._factor_list = []
            self._operation_list = []

    def append(self, circuit, factor:float=1.0, operation=None):
        self._children_list.append(circuit)
        self._factor_list.append(factor)
        self._operation_list.append(operation)

    # Faster but dirty
    # def __eq__(self, other) -> bool:
    #     if isinstance(other,type(self)):
    #         return str(self) == str(other)
    #     return False

    # Maybe use hasing and string for better comparison
    
    # Save but very slow
    def __eq__(self, other) -> bool:
        # op is missing!
        if isinstance(other,type(self)):
            # Fast checks
            if len(self._children_list) != len(other._children_list):
                return False
            # Medium fast check
            fac_set_self = set(self._factor_list)
            fac_set_other = set(other._factor_list)
            if len(fac_set_self) != len(fac_set_other):
                return False
            if fac_set_self != fac_set_other:
                return False
            # Slow check
            for child in self._children_list:
                if child not in other._children_list:
                    return False
                else :
                    index = other._children_list.index(child)
                    if self._factor_list[self._children_list.index(child)] != other._factor_list[index]:
                        return False
            return True
        else:
            return False

class OpTreeList(OpTreeElementBase):
    def __str__(self) -> str:
        text="["
        for i in range(len(self._children_list)):
            text += str(self._factor_list[i]) +"*"+ str(self._children_list[i])
            if i < len(self._factor_list)-1:
                text += ", "
        text+="]"
        return text

class OpTreeSum(OpTreeElementBase):

    def __str__(self) -> str:
        text="["
        for i in range(len(self._children_list)):
            text += str(self._factor_list[i])+"*"
            if not isinstance(self._children_list[i],str):
                text += "\n"
            text += str(self._children_list[i])
            if i < len(self._factor_list)-1:
                text += "\n + "
        text+="]"
        return text


class OpTreeCircuit():

    def __init__(self,circuit,expectation=None,index:int=-1) -> None:
        self._circuit = circuit
        self._expectation = expectation
        self._index = index
        #self._circuit_str = str(circuit)
        # self._dag_representation = circuit_to_dag(circuit, copy_operations=True)
        self._hashvalue = hash_circuit(circuit)


    def __str__(self) -> str:
        #return self._circuit_str
        return str(self._circuit)

    def __eq__(self, other) -> bool:

        if isinstance(other,OpTreeCircuit):
            #this has to be faster! -> store dag maybe
            # return (equal_circuit_fast(self._circuit,other._circuit) and
            #         self._expectation == other._expectation and
            #         self._index == other._index)
            #return (self._dag_representation == other._dag_representation and
            #        self._expectation == other._expectation and
            #        self._index == other._index)
            return (self._hashvalue == other._hashvalue and
                    self._expectation == other._expectation and
                    self._index == other._index)

        return False

class OpTreeOperator():
    def __init__(self,operator,index:int=-1) -> None:
        self._operator = operator
        self._index = index

    def __str__(self) -> str:
        return str(self._operator)

def equal_circuit_fast(circuit1:QuantumCircuit,circuit2:QuantumCircuit):
        
        return circuit_to_dag(circuit1, copy_operations=True) == circuit_to_dag(
            circuit2, copy_operations=True
        )


class OpTree():

    def __init__(self) -> None:
        self.root = None
        self.num_circuits = 0

    def __str__(self) -> str:
        pass

def get_first_circuit(element: Union[OpTreeElementBase,OpTreeCircuit,QuantumCircuit]) -> QuantumCircuit:

    if isinstance(element, OpTreeElementBase):
        return get_first_circuit(element._children_list[0])
    elif isinstance(element, OpTreeCircuit):
        return element._circuit
    elif isinstance(element, QuantumCircuit):
        return element
    else:
        raise ValueError("element must be a CircuitTreeLeaf or a QuantumCircuit")

def build_total_str(element) ->  str:

    if isinstance(element,OpTreeCircuit):
        return element._circuit
    elif isinstance(element,OpTreeSum):
        str = "["
        for i in range(len(element._children_list)):
            str += build_total_str(element._children_list[i])
            if i < len(element._children_list)-1:
                str += " + "
        str +="]"
        return str
    elif isinstance(element,OpTreeList):
        str = "["
        for i in range(len(element._children_list)):
            str += build_total_str(element._children_list[i])
            if i < len(element._children_list)-1:
                str += " , "
        str +="]"
        return str

def circuit_parameter_shift(element: Union[OpTreeCircuit,QuantumCircuit], parameter: ParameterExpression):

    if isinstance(element,OpTreeCircuit):
        circuit = element._circuit
        type = "leaf"
    elif isinstance(element,QuantumCircuit):
        circuit = element
        type = "circuit"
    else:
        raise ValueError("element must be a CircuitTreeLeaf or a QuantumCircuit")

    iref_to_data_index = {id(inst.operation): idx for idx, inst in enumerate(circuit.data)}
    summe = OpTreeSum()
    for param_reference in circuit._parameter_table[parameter]:
        original_gate, param_index = param_reference
        m = iref_to_data_index[id(original_gate)]
        #print("original_gate",original_gate)
        #print("param_index",param_index)
        #print("m",m)

        #print("original_gate.params",original_gate.params)
        fac = original_gate.params[0].gradient(parameter)

        pshift_circ = copy.deepcopy(circuit)
        mshift_circ = copy.deepcopy(circuit)

        pshift_gate = pshift_circ.data[m].operation
        mshift_gate = mshift_circ.data[m].operation

        p_param = pshift_gate.params[param_index]
        m_param = mshift_gate.params[param_index]
        # For analytic gradients the circuit parameters are shifted once by +pi/2 and
        # once by -pi/2.
        shift_constant = 0.5
        pshift_gate.params[param_index] = p_param + (np.pi / (4 * shift_constant))
        mshift_gate.params[param_index] = m_param - (np.pi / (4 * shift_constant))

        #print("pshift_circ",pshift_circ)
        #print("mshift_circ",mshift_circ)
        if type == "leaf":
            summe.append(OpTreeCircuit(pshift_circ),shift_constant*fac)
            summe.append(OpTreeCircuit(mshift_circ),-shift_constant*fac)
        else:
            summe.append(pshift_circ,shift_constant*fac)
            summe.append(mshift_circ,-shift_constant*fac)

    return summe


def circuit_derivative_inplace(element: Union[OpTreeElementBase,OpTreeCircuit,QuantumCircuit],
                       parameter: ParameterExpression):

    if isinstance(element, OpTreeElementBase):
        for i in range(len(element._children_list)):
            if (isinstance(element._children_list[i], QuantumCircuit) or
               isinstance(element._children_list[i], OpTreeCircuit)):

                if isinstance(element._factor_list[i],ParameterExpression):
                    # get derivative of factor
                    f = element._factor_list[i].gradient(parameter)
                    l = element._children_list[i]

                    grad = circuit_parameter_shift(element._children_list[i],parameter)

                    if isinstance(f,float):
                        if f == 0.0:
                            element._children_list[i] = grad
                        else:
                            element._children_list[i] = OpTreeSum([l,grad],[f,element._factor_list[i]])
                            element._factor_list[i] = 1.0
                    else:
                        element._children_list[i] = OpTreeSum([l,grad],[f,element._factor_list[i]])
                        element._factor_list[i] = 1.0

                else:
                    element._children_list[i] = circuit_parameter_shift(element._children_list[i],parameter)
            else:
                circuit_derivative_inplace(element._children_list[i],parameter)
    else:
        element = circuit_parameter_shift(element,parameter)

def circuit_derivative(element: Union[OpTreeElementBase,OpTreeCircuit,QuantumCircuit],
               parameters: Union[ParameterExpression,List[ParameterExpression], ParameterVector]):

    # preprocessing

    is_list = True
    if isinstance(parameters,ParameterExpression):
        parameters = [parameters]
        is_list = False

    is_not_circuit = True
    if isinstance(element,QuantumCircuit) or isinstance(element,OpTreeCircuit):
        is_not_circuit = False
        start = OpTreeList([element],[1.0])
    else:
        start = element
    #start = element

    derivative_list = []
    fac_list=[]
    for dp in parameters:
        res = copy.deepcopy(start)
        circuit_derivative_inplace(res,dp)
        if is_not_circuit:
            derivative_list.append(res)
        else:
            derivative_list.append(res._children_list[0])
        fac_list.append(1.0)

    if is_list:
        return OpTreeList(derivative_list,fac_list)
    else:
        return derivative_list[0]

def circuit_derivative_copy(element: Union[OpTreeElementBase,OpTreeCircuit,QuantumCircuit],
                       parameter: ParameterExpression):

    if isinstance(element, OpTreeElementBase):
        children_list = []
        factor_list = []
        for i in range(len(element._children_list)):

            if isinstance(element._factor_list[i], ParameterExpression):
                # get derivative of factor
                df = element._factor_list[i].gradient(parameter)
                f = element._factor_list[i]
                l = element._children_list[i]
                grad = circuit_derivative_copy(element._children_list[i],parameter)

                if isinstance(df,float):
                    if df == 0.0:
                        children_list.append(grad)
                        factor_list.append(f)
                    else:
                        children_list.append(OpTreeSum([l,grad],[df,f]))
                        factor_list.append(1.0)
                else:
                    children_list.append(OpTreeSum([l,grad],[df,f]))
                    factor_list.append(1.0)

            else:
                children_list.append(circuit_derivative_copy(element._children_list[i],parameter))
                factor_list.append(element._factor_list[i])

        if isinstance(element,OpTreeSum):
            return OpTreeSum(children_list,factor_list)
        elif isinstance(element,OpTreeList):
            return OpTreeList(children_list,factor_list)
        else:
            raise ValueError("element must be a CircuitTreeSum or a CircuitTreeList")

    else:
        return circuit_parameter_shift(element,parameter)

def circuit_derivative_v2(element: Union[OpTreeElementBase,OpTreeCircuit,QuantumCircuit],
               parameters: Union[ParameterExpression,List[ParameterExpression], ParameterVector]):

    # preprocessing

    is_list = True
    if isinstance(parameters,ParameterExpression):
        parameters = [parameters]
        is_list = False

    start = element

    derivative_list = []
    fac_list=[]
    for dp in parameters:
        derivative_list.append(circuit_derivative_copy(start,dp))
        fac_list.append(1.0)

    if is_list:
        return OpTreeList(derivative_list,fac_list)
    else:
        return derivative_list[0]


def simplify_copy(element: Union[OpTreeElementBase,OpTreeCircuit,QuantumCircuit]):

    def _combine_two_ops(op1,op2):
        if op1 is None and op2 is None:
            return None
        elif op1 is None and op2 is not None:
            return op2
        elif op1 is not None and op2 is None:
            return op1
        else:
            return lambda x: op1(op2(x))

    if isinstance(element, OpTreeElementBase):

        if len(element._children_list) > 0:

            l = []
            f = []
            op =[]
            for i in range(len(element._children_list)):
                l.append(simplify_copy(element._children_list[i]))
                f.append(element._factor_list[i])
                op.append(element._operation_list[i])

            if isinstance(element, OpTreeSum):
                new_element = OpTreeSum(l,f,op)
            elif isinstance(element, OpTreeList):
                new_element = OpTreeList(l,f,op)
            else:
                raise ValueError("element must be a CircuitTreeSum or a CircuitTreeList")

            # Check for double sum
            if (isinstance(new_element, OpTreeSum) and
                isinstance(new_element._children_list[0], OpTreeSum)):
                # detected double sum -> combine double some

                l = []
                f = []
                op =[]
                for i in range(len(new_element._children_list)):
                    for j in range(len(new_element._children_list[i]._children_list)):
                        l.append(new_element._children_list[i]._children_list[j])
                        f.append(new_element._factor_list[i]*new_element._children_list[i]._factor_list[j])
                        op.append(_combine_two_ops(new_element._operation_list[i],new_element._children_list[i]._operation_list[j]))
                new_element = OpTreeSum(l,f,op)

            # Check for double circuits in sum
            if isinstance(new_element, OpTreeSum):
                dic={}
                l = []
                f = []
                op = []

                #Better but slower
                for i in range(len(new_element._children_list)):

                    if new_element._children_list[i] in l:
                        index = l.index(new_element._children_list[i])
                        f[index] += new_element._factor_list[i]
                    else:
                        l.append(new_element._children_list[i])
                        f.append(new_element._factor_list[i])
                        op.append(new_element._operation_list[i])

                # for i in range(len(new_element._children_list)):
                #     element_str = str(new_element._children_list[i])
                #     if element_str in dic:
                #         f[dic[element_str]] += new_element._factor_list[i]
                #         print("double circuit in sum")
                #     else:
                #         dic[element_str] = len(f)
                #         l.append(new_element._children_list[i])
                #         f.append(new_element._factor_list[i])
                #         op.append(new_element._operation_list[i])
                #
                new_element = OpTreeSum(l,f,op)

            return new_element

        else:
            return copy.deepcopy(element)
    else:
        return copy.deepcopy(element)


def evaluate_index_tree(element: Union[OpTreeElementBase,OpTreeCircuit],result_array):

    if isinstance(element, OpTreeElementBase):
        temp = np.array([element._factor_list[i]*evaluate_index_tree(element._children_list[i],result_array) for i in range(len(element._children_list))])
        if isinstance(element, OpTreeSum):
            return np.sum(temp,axis=0)
        elif isinstance(element, OpTreeList):
            return temp
        else:
            raise ValueError("element must be a CircuitTreeSum or a CircuitTreeList")
    elif isinstance(element, int):
        return result_array[element]
    else:
        raise ValueError("element must be a Tree element or a integer pointer")

def evaluate(element: Union[OpTreeElementBase,OpTreeCircuit,QuantumCircuit],
             estimator, operator, dictionary, detect_circuit_duplicates:bool=False):

    # dictionary might be slow!

    # create a list of circuit and a copy of the circuit tree with indices pointing to the circuit
    circuit_list = []
    if detect_circuit_duplicates:
        circuit_hash_list = []
    parameter_list = []
    global circuit_counter
    circuit_counter = 0

    def build_lists_and_index_tree(element):
        global circuit_counter
        if isinstance(element, OpTreeElementBase):
            l = [build_lists_and_index_tree(c) for c in element._children_list]
            f = []
            for i in range(len(element._factor_list)):
                if isinstance(element._factor_list[i],ParameterExpression):
                    f.append(float(element._factor_list[i].bind(dictionary,allow_unknown_parameters=True)))
                else:
                    f.append(element._factor_list[i])
            op = element._operation_list
            if isinstance(element, OpTreeSum):
                return OpTreeSum(l,f,op)
            elif isinstance(element, OpTreeList):
                return OpTreeList(l,f,op)
            else:
                raise ValueError("element must be a CircuitTreeSum or a CircuitTreeList")

        else:
            if isinstance(element,QuantumCircuit):
                circuit = element
                if detect_circuit_duplicates:
                    circuit_hash = hash_circuit(circuit)
            elif isinstance(element,OpTreeCircuit):
                circuit = element._circuit
                if detect_circuit_duplicates:
                    circuit_hash = element._hashvalue
            else:
                raise ValueError("element must be a CircuitTreeLeaf or a QuantumCircuit")

            if detect_circuit_duplicates:
                if circuit_hash in circuit_hash_list:
                    return circuit_list.index(circuit)
                circuit_hash_list.append(circuit_hash)

            circuit_list.append(circuit)

            parameter_list.append(np.array([dictionary[p] for p in circuit.parameters ]))
            circuit_counter += 1
            return circuit_counter - 1

    start=time.time()
    index_tree = build_lists_and_index_tree(element)
    print("build_lists_and_index_tree",time.time()-start)

    print("len(circuit_list)",len(circuit_list))

    #print(circuit_tree_index)


    parameter_list = [parameter_list[0]]*len(circuit_list)



    op_list = [operator]*len(circuit_list)

    # print("circuit_list",circuit_list)
    # print("op_list",op_list)
    # print("parameter_list",parameter_list)

    #print("inital_circuit",circuit_list[0])

    start=time.time()
    res1 = Estimator().run(circuit_list,op_list,parameter_list)
    print("run",time.time()-start)
    start=time.time()
    res2 = res1.result()
    print("res2",time.time()-start)
    start=time.time()
    result = res2.values
    print("result time",time.time()-start)

    print("result",result)

    return evaluate_index_tree(index_tree,result)
