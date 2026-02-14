"""
File: common_mpc_deploy.py
Description: Common functions for deploying MPC code,
 including conversion of SymPy matrices to NumPy arrays for use in C++ code generation.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import numpy as np
import sympy as sp
import copy

from external_libraries.python_numpy_to_cpp.python_numpy.numpy_deploy import NumpyDeploy

VALUE_IS_ZERO_TOL = 1e-30


def convert_SparseAvailable_for_deploy(SparseAvailable: sp.Matrix) -> np.ndarray:
    """
    Converts a SymPy sparse availability matrix to a NumPy boolean ndarray for deployment.

    Parameters
    ----------
    SparseAvailable : sympy.Matrix
        A SymPy matrix representing sparse availability,
          where nonzero entries indicate availability.

    Returns
    -------
    numpy.ndarray
        A boolean NumPy array of the same shape as `SparseAvailable`,
          where True indicates available (nonzero) entries.
    """
    if not isinstance(SparseAvailable, sp.MatrixBase):
        raise TypeError("SparseAvailable must be a sympy Matrix")

    SparseAvailable_ndarray = np.zeros(SparseAvailable.shape, dtype=bool)

    for i in range(SparseAvailable.shape[0]):
        for j in range(SparseAvailable.shape[1]):
            if int(SparseAvailable[i, j]) != 0:
                SparseAvailable_ndarray[i, j] = True

    return SparseAvailable_ndarray
