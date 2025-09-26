import os
import sys
sys.path.append(os.getcwd())

import numpy as np
import sympy as sp


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
            min_max_array: np.ndarray
    ):
        self.values = min_max_array
        self.size = self.values.shape[0]

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
