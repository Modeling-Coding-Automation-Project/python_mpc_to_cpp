"""
File: nonlinear_mpc_deploy.py

This module provides functionality to deploy a Nonlinear Model Predictive Control (MPC)
system to C++ code. It includes methods to generate C++ code for the Nonlinear MPC
based on the provided Python implementation.
It includes the `NonlinearMPC_Deploy` class with a static method
`generate_Nonlinear_MPC_cpp_code` that takes a `NonlinearMPC_TwiceDifferentiable` object
and generates the corresponding C++ code files.
"""
import os
import sys
sys.path.append(os.getcwd())
sys.path.append(os.path.join(
    os.getcwd(), "external_libraries", "python_control_to_cpp"))
sys.path.append(os.path.join(
    os.getcwd(), 'external_libraries', 'python_optimization_to_cpp'))

import inspect
import copy

from external_libraries.python_numpy_to_cpp.python_numpy.numpy_deploy import NumpyDeploy
from external_libraries.MCAP_python_control.python_control.control_deploy import ControlDeploy
from external_libraries.python_control_to_cpp.python_control.kalman_filter_deploy import KalmanFilterDeploy
from external_libraries.python_optimization_to_cpp.optimization_utility.sqp_matrix_utility_deploy import SQP_MatrixUtilityDeploy

from external_libraries.MCAP_python_mpc.python_mpc.nonlinear_mpc import NonlinearMPC_TwiceDifferentiable


class NonlinearMPC_Deploy:
    """
    NonlinearMPC_Deploy

    This class provides static methods for deploying a
      nonlinear Model Predictive Controller (MPC) by generating corresponding
        C++ code from a Python-based nonlinear MPC object.

    Methods
    -------
    generate_Nonlinear_MPC_cpp_code(nonlinear_mpc: NonlinearMPC_TwiceDifferentiable, file_name=None)
        Generates C++ header files for the provided nonlinear MPC object,
          including its Kalman filter, cost matrices, and all required type definitions.

        Parameters
        ----------
        nonlinear_mpc : NonlinearMPC_TwiceDifferentiable
            The nonlinear MPC object to be deployed.
              This object should contain all necessary model, filter,
                and cost matrix information.
        file_name : str, optional
            The base name for the generated C++ files. If not provided,
              the caller's file name will be used.

        Returns
        -------
        deployed_file_names : list of str
            A list of generated C++ file names, including EKF, parameter,
              cost matrices, and the main MPC header file.

        Raises
        ------
        ValueError
            If no parameter file is found in the EKF deployment files.

        Notes
        -----
        - The method inspects the caller's context to determine variable
          and file names if not explicitly provided.
        - The generated C++ code includes type definitions, namespace encapsulation,
          and a factory function for constructing the MPC object.
        - Relies on auxiliary deployment utilities for Kalman filter
          and cost matrices code generation.
    """

    @staticmethod
    def generate_Nonlinear_MPC_cpp_code(
            nonlinear_mpc: NonlinearMPC_TwiceDifferentiable,
            file_name=None):
        """
        Generates C++ header files for deploying a nonlinear Model Predictive Controller (MPC)
        with an Extended Kalman Filter (EKF) and associated cost matrices.

        This function inspects the provided `nonlinear_mpc` object, extracts its parameters,
        and generates C++ code that encapsulates the MPC, EKF, and cost matrices in a
        deployable format. The generated code includes type definitions, constants, and
        a factory function for constructing the MPC object in C++. The function also
        handles file naming based on the caller's context or a provided file name.

        Args:
            nonlinear_mpc (NonlinearMPC_TwiceDifferentiable):
                The nonlinear MPC object to be deployed. Must contain a Kalman filter,
                cost matrices, and model parameters.
            file_name (str, optional):
                The base name for the generated C++ files. If not provided, the name
                is inferred from the caller's file and variable name.

        Returns:
            list of str:
                A list of file names for the generated C++ header files, including
                EKF, cost matrices, and the main MPC deployment file.

        Raises:
            ValueError:
                If no parameter file is found in the EKF deployment files.

        Notes:
            - The function uses introspection to determine variable and file names
                if not explicitly provided.
            - The generated C++ code assumes the existence of certain type definitions
                and namespaces (e.g., PythonNumpy, PythonControl, PythonMPC).
            - Deep copies of cost matrices are made to avoid modifying the original object.
        """

        parameters = nonlinear_mpc.kalman_filter.Parameters
        number_of_delay = nonlinear_mpc.kalman_filter.Number_of_Delay

        deployed_file_names = []

        data_type = nonlinear_mpc.X_inner_model.dtype
        ControlDeploy.restrict_data_type(data_type.name)

        type_name = NumpyDeploy.check_dtype(nonlinear_mpc.kalman_filter.A)

        # %% inspect arguments
        # Get the caller's frame
        frame = inspect.currentframe().f_back
        # Get the caller's local variables
        caller_locals = frame.f_locals
        # Find the variable name that matches the matrix_in value
        variable_name = None
        for name, value in caller_locals.items():
            if value is nonlinear_mpc:
                variable_name = name
                break
        # Get the caller's file name
        if file_name is None:
            caller_file_full_path = frame.f_code.co_filename
            caller_file_name = os.path.basename(caller_file_full_path)
            caller_file_name_no_extension = os.path.splitext(caller_file_name)[
                0]
        else:
            caller_file_name_no_extension = file_name

        code_file_name = caller_file_name_no_extension + "_" + variable_name
        code_file_name_ext = code_file_name + ".hpp"

        # %% create EKF code
        locals_map = {
            f"{variable_name}_ekf": nonlinear_mpc.kalman_filter,
            "caller_file_name_no_extension": caller_file_name_no_extension,
            "number_of_delay": number_of_delay
        }
        ekf_file_names = eval(
            f"KalmanFilterDeploy.generate_EKF_cpp_code({variable_name}_ekf, caller_file_name_no_extension, number_of_delay=number_of_delay)",
            globals(),
            locals_map
        )

        deployed_file_names.append(ekf_file_names)
        ekf_file_name = ekf_file_names[-1]

        ekf_file_name_no_extension = ekf_file_name.split(".")[0]

        parameter_code_file_name = ""
        parameter_code_file_name_no_extension = ""
        for name in ekf_file_names:
            if "parameter" in name:
                parameter_code_file_name = name
                parameter_code_file_name_no_extension = name.split(".")[
                    0]
                break

        if parameter_code_file_name == "":
            raise ValueError(
                "No parameter file found in EKF deployment files.")

        # %% generate cost matrices code
        cost_matrices = copy.deepcopy(nonlinear_mpc.sqp_cost_matrices)

        cost_matrices_code_names = SQP_MatrixUtilityDeploy.generate_cpp_code(
            cost_matrices=cost_matrices,
            file_name=caller_file_name_no_extension,
        )
        deployed_file_names.extend(cost_matrices_code_names)

        cost_matrices_file_name_no_extension = caller_file_name_no_extension + \
            "_cost_matrices"
        cost_matrices_file_name = cost_matrices_file_name_no_extension + ".hpp"

        # %% main code generation
        code_text = ""

        file_header_macro_name = "__" + code_file_name.upper() + "_HPP__"

        code_text += "#ifndef " + file_header_macro_name + "\n"
        code_text += "#define " + file_header_macro_name + "\n\n"

        code_text += f"#include \"{ekf_file_name}\"\n"
        code_text += f"#include \"{cost_matrices_file_name}\"\n\n"

        code_text += "#include \"python_mpc.hpp\"\n\n"

        namespace_name = code_file_name

        code_text += "namespace " + namespace_name + " {\n\n"

        code_text += "using namespace PythonNumpy;\n"
        code_text += "using namespace PythonControl;\n"
        code_text += "using namespace PythonMPC;\n\n"

        code_text += f"constexpr std::size_t NP = {nonlinear_mpc.Np};\n\n"

        code_text += f"constexpr std::size_t INPUT_SIZE = {ekf_file_name_no_extension}::INPUT_SIZE;\n"
        code_text += f"constexpr std::size_t STATE_SIZE = {ekf_file_name_no_extension}::STATE_SIZE;\n"
        code_text += f"constexpr std::size_t OUTPUT_SIZE = {ekf_file_name_no_extension}::OUTPUT_SIZE;\n\n"

        code_text += f"constexpr std::size_t NUMBER_OF_DELAY = {ekf_file_name_no_extension}::NUMBER_OF_DELAY;\n\n"

        code_text += f"using X_Type = StateSpaceState_Type<{type_name}, STATE_SIZE>;\n\n"
        code_text += f"using U_Type = StateSpaceInput_Type<{type_name}, INPUT_SIZE>;\n\n"
        code_text += f"using Y_Type = StateSpaceOutput_Type<{type_name}, OUTPUT_SIZE>;\n\n"

        code_text += f"using EKF_Type = {ekf_file_name_no_extension}::type;\n\n"

        code_text += f"using Parameter_Type = {parameter_code_file_name_no_extension}::Parameter_Type;\n\n"

        code_text += f"using Cost_Matrices_Type = {cost_matrices_file_name_no_extension}::type;\n\n"

        code_text += f"using Reference_Type = DenseMatrix_Type<{type_name}, OUTPUT_SIZE, 1>;\n\n"

        code_text += f"using ReferenceTrajectory_Type = DenseMatrix_Type<{type_name}, OUTPUT_SIZE, NP>;\n\n"

        code_text += f"using type = NonlinearMPC_TwiceDifferentiable_Type<\n" + \
            "    EKF_Type, Cost_Matrices_Type>;\n\n"

        code_text += "inline auto make(void) -> type {\n\n"

        code_text += f"    auto kalman_filter = {ekf_file_name_no_extension}::make();\n\n"

        code_text += f"    auto cost_matrices = {cost_matrices_file_name_no_extension}::make();\n\n"

        code_text += f"    {type_name} delta_time = static_cast<{type_name}>({nonlinear_mpc.delta_time});\n\n"

        code_text += f"    X_Type X_initial;\n"

        for i in range(nonlinear_mpc.X_inner_model.shape[0]):
            code_text += f"    X_initial.template set<{i}, 0>(static_cast<{type_name}>({nonlinear_mpc.X_inner_model[i, 0]}));\n"
        code_text += "\n"

        code_text += f"    auto nonlinear_mpc = make_NonlinearMPC_TwiceDifferentiable(\n" + \
            f"        kalman_filter, cost_matrices, delta_time, X_initial);\n\n"

        code_text += f"    return nonlinear_mpc;\n"

        code_text += "}\n\n"

        code_text += "} // namespace " + namespace_name + "\n\n"

        code_text += "#endif // " + file_header_macro_name + "\n"

        code_file_name_ext = ControlDeploy.write_to_file(
            code_text, code_file_name_ext)

        deployed_file_names.append(code_file_name_ext)

        return deployed_file_names
