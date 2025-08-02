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
