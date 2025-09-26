import os
import sys
sys.path.append(os.getcwd())

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


class MinMaxCodeGenerator:
    """
    A class to generate code for min/max limits and their active sets, 
    typically used in Model Predictive Control (MPC) deployments.
    Attributes:
        values (np.ndarray): Array of min/max values.
        size (int): Number of elements in the min/max array.
        active_set (np.ndarray or None): Boolean array indicating active limits.
        min_max_name (str): Name identifier for the min/max variable.
        file_name_no_extension (str or None): File name (without extension) for generated code.
    Methods:
        __init__(min_max_array, min_max_name, size):
            Initializes the generator with min/max values and configuration.
        generate_active_set(is_active_function=None, is_active_array=None):
            Generates the active set array using a function or a boolean array.
        create_limits_code(data_type, variable_name, caller_file_name_without_ext):
            Generates C++ code for the limits and returns the file name and its base name.
        write_limits_code(code_text, type_name):
            Appends code for setting active limits to the provided code text.
    """

    def __init__(
            self,
            min_max_array: np.ndarray,
            min_max_name: str,
            size: int
    ):
        if min_max_array is not None:
            self.values = min_max_array
            self.size = self.values.shape[0]
            self.active_set = None
        else:
            self.size = size
            self.values = np.ones((self.size, 1))
            self.active_set = np.zeros((self.size, 1), dtype=bool)

        self.min_max_name = min_max_name
        self.file_name_no_extension = None

    def generate_active_set(
        self,
        is_active_function: callable = None,
        is_active_array: np.ndarray = None
    ):
        """
        Generates and returns the active set for the current object.

        The active set is a boolean numpy array of shape (self.size, 1)
          indicating which elements are active.
        The active set can be determined either by a provided function or by
          a provided boolean array.

        Args:
            is_active_function (callable, optional): A function that takes an index
              (int) and returns True if the element is active, False otherwise.
            is_active_array (np.ndarray, optional): A boolean numpy array of length
              self.size indicating active elements.

        Raises:
            ValueError: If neither is_active_function nor is_active_array is provided.

        Returns:
            np.ndarray: A boolean numpy array of shape (self.size, 1)
              representing the active set.
        """
        if is_active_function is None and is_active_array is None:
            raise ValueError(
                "Either is_active_function or is_active_array must be provided")

        if self.active_set is None:
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
        """
        Generates C++ code for the active set limits and returns
          the generated file name and its name without extension.

        Args:
            data_type: The desired NumPy data type for the active set array.
            variable_name (str): The base name for the variable to store the active set.
            caller_file_name_without_ext (str): The file name
              (without extension) to use for the generated C++ code.

        Returns:
            tuple:
                - file_name (str): The name of the generated C++ code file.
                - file_name_no_extension (str): The file name without its extension.

        Side Effects:
            - Sets self.file_name_no_extension to the file name without extension.

        Notes:
            - Uses `exec` and `eval` to dynamically create variables
              and call code generation functions.
            - Relies on `self.active_set` and `self.min_max_name` attributes.
        """
        active_set = np.array(self.active_set, dtype=data_type).reshape(-1, 1)
        exec(f"{variable_name}_{self.min_max_name} = copy.deepcopy(active_set)")

        file_name = eval(
            f"NumpyDeploy.generate_matrix_cpp_code(matrix_in={variable_name}_" +
            self.min_max_name + ", file_name=caller_file_name_without_ext)")

        self.file_name_no_extension = file_name.split(".")[0]

        return file_name, self.file_name_no_extension

    def write_limits_code(
            self,
            code_text: str,
            type_name: str
    ):
        """
        Appends C++ code for setting limit values to the provided code text.

        This method generates code that creates an instance of a min-max limits object,
        and, if an active set is defined and non-zero, sets specific limit values using
        template-based setter calls for each active index.

        Args:
            code_text (str): The initial code text to append to.
            type_name (str): The C++ type name to use for static casting limit values.

        Returns:
            str: The updated code text with generated limit-setting code appended.
        """
        code_text += f"  auto {self.min_max_name} = {self.file_name_no_extension}::make();\n\n"
        if self.active_set is not None and \
                np.linalg.norm(self.active_set) > VALUE_IS_ZERO_TOL:
            for i in range(len(self.active_set)):
                if self.active_set[i]:
                    code_text += f"  {self.min_max_name}.template set<{i}, 0>("
                    code_text += \
                        f"static_cast<{type_name}>({self.values[i, 0]})"
                    code_text += ");\n"
            code_text += "\n"

        return code_text
