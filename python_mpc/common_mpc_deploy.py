import os
import sys
sys.path.append(os.getcwd())

import numpy as np
import sympy as sp
import copy

from external_libraries.python_numpy_to_cpp.python_numpy.numpy_deploy import NumpyDeploy


def convert_SparseAvailable_for_deploy(SparseAvailable: sp.Matrix) -> np.ndarray:

    if not isinstance(SparseAvailable, sp.MatrixBase):
        raise TypeError("SparseAvailable must be a sympy Matrix")

    SparseAvailable_ndarray = np.zeros(SparseAvailable.shape, dtype=bool)

    for i in range(SparseAvailable.shape[0]):
        for j in range(SparseAvailable.shape[1]):
            if int(SparseAvailable[i, j]) != 0:
                SparseAvailable_ndarray[i, j] = True

    return SparseAvailable_ndarray


class MinMaxCodeGenerator:
    def __init__(
            self,
            min_max_array: np.ndarray,
            min_max_name: str
    ):
        self.values = min_max_array
        self.size = self.values.shape[0]
        self.min_max_name = min_max_name

    def generate_active_set(
        self,
        is_active_function: callable = None,
        is_active_array: np.ndarray = None
    ):
        if is_active_function is None and is_active_array is None:
            raise ValueError(
                "Either is_active_function or is_active_array must be provided")

        self.active_set = np.zeros((self.size, 1), dtype=bool)

        for i in range(self.size):
            if is_active_function is not None and is_active_function(i):
                self.active_set[i, 0] = True

            elif is_active_array is not None and is_active_array[i]:
                self.active_set[i, 0] = True

        return self.active_set

    def create_limits_code(
            self,
            data_type,
            variable_name: str,
            caller_file_name_without_ext: str
    ):

        active_set = np.array(
            copy.deepcopy(self.active_set), dtype=data_type).reshape(-1, 1)
        exec(f"{variable_name}_" + self.min_max_name +
             " = active_set")

        file_name = eval(
            f"NumpyDeploy.generate_matrix_cpp_code(matrix_in={variable_name}_" +
            self.min_max_name + ", " +
            "file_name=caller_file_name_without_ext)")

        file_name_no_extension = file_name.split(".")[0]

        return file_name, file_name_no_extension
