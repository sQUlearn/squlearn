import numpy as np
from qiskit import transpile
from typing import List
from qiskit import QuantumCircuit

def cost_func_advanced(circ: QuantumCircuit, layouts: List[List], backend):
    """
    A custom cost function that includes T1 and T2 computed during idle periods,
    that return the total error rate over all the layouts for the gates in the given circuit.
    Include 'rz' gates.

    Parameters:
        circ (QuantumCircuit): circuit of interest
        layouts (list of lists): List of specified layouts
        backend (IBMQBackend): An IBM Quantum backend instance

    Returns:
        list: Tuples of layout and cost
    """
    out = []
    props = backend.properties()
    dt = backend.configuration().dt
    num_qubits = backend.configuration().num_qubits
    t1s = [props.qubit_property(qq, "T1")[0] for qq in range(num_qubits)]
    t2s = [props.qubit_property(qq, "T2")[0] for qq in range(num_qubits)]
    for layout in layouts:
        sch_circ = transpile(
            circ, backend, initial_layout=layout, optimization_level=0, scheduling_method="alap"
        )
        error = 0
        fid = 1
        touched = set()
        for item in sch_circ._data:
            if item[0].name == "cx":
                q0 = sch_circ.find_bit(item[1][0]).index
                q1 = sch_circ.find_bit(item[1][1]).index
                fid *= 1 - props.gate_error("cx", [q0, q1])
                touched.add(q0)
                touched.add(q1)

            elif item[0].name in ["sx", "x", "rz"]:
                q0 = sch_circ.find_bit(item[1][0]).index
                fid *= 1 - props.gate_error(item[0].name, q0)
                touched.add(q0)

            elif item[0].name in ["measure", "reset"]:
                q0 = sch_circ.find_bit(item[1][0]).index
                fid *= 1 - props.readout_error(q0)
                touched.add(q0)

            elif item[0].name == "delay":
                q0 = sch_circ.find_bit(item[1][0]).index
                # Ignore delays that occur before gates
                # This assumes you are in ground state and errors
                # do not occur.
                if q0 in touched:
                    time = item[0].duration * dt
                    fid *= 1 - idle_error(time, t1s[q0], t2s[q0])
            # else:
            #     print('Uncomputed gate: {}'.format(item[0].name))

        error = 1 - fid
        out.append((layout, error))
    return out


def idle_error(time: float, t1: float, t2: float):
    """Compute the approx. idle error from T1 and T2
    Parameters:
        time (float): Delay time in sec
        t1 (float): T1 time in sec
        t2, (float): T2 time in sec
    Returns:
        float: Idle error
    """
    t2 = min(t1, t2)
    rate1 = 1 / t1
    rate2 = 1 / t2
    p_reset = 1 - np.exp(-time * rate1)
    p_z = (1 - p_reset) * (1 - np.exp(-time * (rate2 - rate1))) / 2
    return p_z + p_reset
