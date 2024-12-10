from qiskit.circuit.library import standard_gates

def DecomposeToStd(circuit):
        """Function to decompose the circuit to standard gates"""
        decompose_names = [None]
        while len(decompose_names)>0:
            decompose_names = []
            for instr in circuit.data:
                if instr.operation.name not in [*dir(standard_gates),"cx","cy","cz","measure"]:
                    decompose_names.append(instr.operation.name)
            circuit_new = circuit.decompose(decompose_names)
            if circuit == circuit_new:
                break
            circuit = circuit_new
        return circuit_new