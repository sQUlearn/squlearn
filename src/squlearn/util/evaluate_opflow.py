from qiskit.opflow import CircuitSampler
from qiskit.opflow import OperatorBase
from qiskit.opflow import StateFn, CircuitStateFn, OperatorStateFn, DictStateFn
from qiskit.opflow import SummedOp, ComposedOp
from qiskit.opflow import ListOp
from functools import partial
import numpy as np
import time

from qiskit.primitives import Estimator as Estimator_primitive
from qiskit.primitives import BackendEstimator as BackendEstimator_primitive

from executor import ExecutorEstimator


def evaluate_opflow_qi_slow(QuantumInstance, opflow):
    Sampler = CircuitSampler(QuantumInstance, caching="last")
    Sampler._transpile_before_bind = False
    eval_conv = Sampler.convert(opflow)
    return np.real(eval_conv.eval())


def evaluate_opflow_qi(QuantumInstance, opflow, mitigation_func=None):
    # build a list of circuits which have to be executed
    circuit_list = []

    def build_circuit_list(operator: OperatorBase) -> None:
        if isinstance(operator, CircuitStateFn):
            if QuantumInstance.is_statevector:
                circuit_list.append(operator.to_circuit(meas=False))
            else:
                circuit_list.append(operator.to_circuit(meas=True))
        elif isinstance(operator, ListOp):
            for op in operator.oplist:
                build_circuit_list(op)

    build_circuit_list(opflow)

    # build a list in the same order as circuit containing the associated operators
    CircuitStateFn_list = []

    def build_CircuitStateFn_list(operator: OperatorBase) -> None:
        if isinstance(operator, CircuitStateFn):
            CircuitStateFn_list.append(operator)
        elif isinstance(operator, ListOp):
            for op in operator.oplist:
                build_CircuitStateFn_list(op)

    build_CircuitStateFn_list(opflow)

    # Execute circuits
    start = time.time()
    results = QuantumInstance.execute(circuit_list, had_transpiled=True)
    # print("exec:", time.time() - start)

    # build StateFns from the results (copied from qiskit source code)
    sampled_statefn_dicts = {}
    for i, op_c in enumerate(CircuitStateFn_list):
        # Taking square root because we're replacing a statevector
        # representation of probabilities.

        circ_results = results.data(i)

        if "expval_measurement" in circ_results:
            avg = circ_results["expval_measurement"]
            # Will be replaced with just avg when eval is called later
            num_qubits = opflow[0].num_qubits
            result_sfn = DictStateFn(
                "0" * num_qubits,
                coeff=avg * op_c.coeff,
                is_measurement=op_c.is_measurement,
                from_operator=op_c.from_operator,
            )
        elif QuantumInstance.is_statevector:
            result_sfn = StateFn(
                op_c.coeff * results.get_statevector(i),
                is_measurement=op_c.is_measurement,
            )
        else:
            QI_shots = QuantumInstance._run_config.shots

            if mitigation_func is not None:
                proba = mitigation_func(results.get_counts(i).items())
            else:
                proba = {b: (v / QI_shots) for (b, v) in results.get_counts(i).items()}
            result_sfn = DictStateFn(
                {b: np.abs(p) ** 0.5 * op_c.coeff for (b, p) in proba.items()},
                is_measurement=op_c.is_measurement,
                from_operator=op_c.from_operator,
            )

        sampled_statefn_dicts[id(op_c)] = result_sfn

    def replace_circuits_with_dicts(operator):
        if isinstance(operator, CircuitStateFn):
            return sampled_statefn_dicts[id(operator)]
        elif isinstance(operator, ListOp):
            return operator.traverse(partial(replace_circuits_with_dicts))
        else:
            return operator

    # Restore original opflow structure with evaluated StateFn
    start = time.time()
    eval_circs = replace_circuits_with_dicts(opflow)
    return_val = np.real(eval_circs.eval())
    # print("eval:", time.time() - start)

    return return_val


def evaluate_opflow_estimator(estimator, opflow):
    # build a list of circuits which have to be executed
    circuit_list = []
    measure_list = []
    repeat_list = []

    def build_circuit_list(operator: OperatorBase) -> None:
        if isinstance(operator, ComposedOp):
            len_measure = 0
            for op in operator.oplist:
                if isinstance(op, ListOp):
                    for op2 in op.oplist:
                        measure_list.append(op2.primitive)
                        len_measure = len_measure + 1
                elif isinstance(op, OperatorStateFn):
                    measure_list.append(op.primitive)
                    len_measure = len_measure + 1
            repeat_list.append(len_measure)
            for op in operator.oplist:
                if isinstance(op, CircuitStateFn):
                    for i in range(len_measure):
                        circuit_list.append(op.to_circuit(meas=False))
        elif isinstance(operator, ListOp):
            for op in operator.oplist:
                build_circuit_list(op)

    build_circuit_list(opflow)

    if len(measure_list) != len(circuit_list):
        raise ValueError("Non-equal number of circuits and measurement operators!")

    # build a list in the same order as circuit containing the associated operators
    CircuitStateFn_list = []

    def build_CircuitStateFn_list(operator: OperatorBase) -> None:
        if (
            isinstance(operator, CircuitStateFn)
            or isinstance(operator, ComposedOp)
            or isinstance(operator, OperatorStateFn)
        ):
            CircuitStateFn_list.append(operator)
        elif isinstance(operator, ListOp):
            for op in operator.oplist:
                build_CircuitStateFn_list(op)

    build_CircuitStateFn_list(opflow)

    try:
        start = time.time()
        job = estimator.run(circuit_list, measure_list)
        # print("exec:", time.time() - start)
        job_result = job.result()

        # Clear cache of estimator, otherwise memory leak
        if isinstance(estimator, Estimator_primitive) or isinstance(
            estimator, BackendEstimator_primitive
        ):
            estimator._circuits = []
            estimator._observables = []
            estimator._parameters = []
            estimator._circuit_ids = {}
        elif isinstance(estimator, ExecutorEstimator):
            estimator.clear_cache()

    except:
        # second try
        start = time.time()
        job = estimator.run(circuit_list, measure_list)
        # print("exec:", time.time() - start)
        job_result = job.result()

        # Clear cache of estimator, otherwise memory leak
        if isinstance(estimator, Estimator_primitive) or isinstance(
            estimator, BackendEstimator_primitive
        ):
            estimator._circuits = []
            estimator._observables = []
            estimator._parameters = []
            estimator._circuit_ids = {}
        elif isinstance(estimator, ExecutorEstimator):
            estimator.clear_cache()


    sampled_statefn_dicts = {}

    repeat_list2 = []
    ioff = 0
    for i in range(len(repeat_list) - 1):
        ioff = ioff + repeat_list[i]
        repeat_list2.append(ioff)
    values = np.split(job_result.values, repeat_list2)

    for i, op_c in enumerate(CircuitStateFn_list):
        sampled_statefn_dicts[id(op_c)] = values[i]

    def replace_circuits_with_dicts(operator):
        if (
            isinstance(operator, CircuitStateFn)
            or isinstance(operator, ComposedOp)
            or isinstance(operator, OperatorStateFn)
        ):
            return sampled_statefn_dicts[id(operator)] * operator.coeff
        elif isinstance(operator, ListOp):
            value_list = np.array([replace_circuits_with_dicts(op) for op in operator.oplist])
            if isinstance(operator, SummedOp):
                return np.sum(value_list, axis=0) * operator.coeff
            else:
                return np.array(value_list) * operator.coeff
        else:
            raise RuntimeError("Wrong operator in result list!")

    return_val = replace_circuits_with_dicts(opflow)
    return return_val


def evaluate_opflow_sampler(sampler, opflow):
    # build a list of circuits which have to be executed
    circuit_list = []

    def build_circuit_list(operator: OperatorBase) -> None:
        if isinstance(operator, CircuitStateFn):
            circuit_list.append(operator.to_circuit(meas=True))
        elif isinstance(operator, ListOp):
            for op in operator.oplist:
                build_circuit_list(op)

    build_circuit_list(opflow)

    # build a list in the same order as circuit containing the associated operators
    CircuitStateFn_list = []

    def build_CircuitStateFn_list(operator: OperatorBase) -> None:
        if isinstance(operator, CircuitStateFn):
            CircuitStateFn_list.append(operator)
        elif isinstance(operator, ListOp):
            for op in operator.oplist:
                build_CircuitStateFn_list(op)

    build_CircuitStateFn_list(opflow)

    # Execute circuits
    try:
        start = time.time()
        job = sampler.run(circuit_list)
        results = job.result()
        # print("exec:", time.time() - start)
    except:
        # second try
        start = time.time()
        job = sampler.run(circuit_list)
        results = job.result()
        # print("exec:", time.time() - start)

    # build StateFns from the results (copied from qiskit source code)
    sampled_statefn_dicts = {}
    for i, op_c in enumerate(CircuitStateFn_list):
        # Taking square root because we're replacing a statevector
        # representation of probabilities.
        circ_results = results.quasi_dists[i]
        proba = circ_results.binary_probabilities()
        result_sfn = DictStateFn(
            {b: np.abs(p) ** 0.5 * op_c.coeff for (b, p) in proba.items()},
            is_measurement=op_c.is_measurement,
            from_operator=op_c.from_operator,
        )
        sampled_statefn_dicts[id(op_c)] = result_sfn

    def replace_circuits_with_dicts(operator):
        if isinstance(operator, CircuitStateFn):
            return sampled_statefn_dicts[id(operator)]
        elif isinstance(operator, ListOp):
            return operator.traverse(partial(replace_circuits_with_dicts))
        else:
            return operator

    # Restore original opflow structure with evaluated StateFn
    start = time.time()
    eval_circs = replace_circuits_with_dicts(opflow)
    return_val = np.real(eval_circs.eval())
    # print("eval:", time.time() - start)
    return return_val
