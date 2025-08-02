"""
File: adaptive_mpc_deploy.py

This module provides functionality to deploy an Adaptive Model Predictive Control
(MPC) system without constraints from Python to C++ code.
It includes the `AdaptiveMPC_Deploy` class, which contains static methods for generating
C++ header files representing the structure and parameters of an Adaptive MPC controller,
including its Kalman filter, prediction matrices, solver factors, and weight matrices.
"""
import os
import sys
sys.path.append(os.getcwd())
sys.path.append(os.path.join(
    os.getcwd(), "external_libraries", "python_control_to_cpp"))

import inspect
import numpy as np
import copy

from external_libraries.python_numpy_to_cpp.python_numpy.numpy_deploy import NumpyDeploy
from external_libraries.MCAP_python_control.python_control.control_deploy import ControlDeploy
from external_libraries.python_control_to_cpp.python_control.kalman_filter_deploy import KalmanFilterDeploy

from mpc_utility.adaptive_matrices_deploy import AdaptiveMatricesDeploy
from python_mpc.common_mpc_deploy import convert_SparseAvailable_for_deploy

from external_libraries.MCAP_python_mpc.python_mpc.adaptive_mpc import AdaptiveMPC_NoConstraints

TOL = 1e-30


class AdaptiveMPC_Deploy:
    """
    A utility class for deploying Adaptive Model Predictive Control (MPC)
      models without constraints
    by generating corresponding C++ header files from Python objects.

    This class provides static methods to automate the translation
      of Python-based Adaptive MPC models into C++ code,
        facilitating deployment in embedded or high-performance environments.

    Notes
    -----
    - This class is not intended to be instantiated.
    - The code generation process inspects the caller's context to
      determine variable names and file paths.
    - The generated C++ code includes all necessary type definitions, constants,
      and factory functions
      to reconstruct the Adaptive MPC object in C++.
    """

    @staticmethod
    def generate_Adaptive_MPC_NC_cpp_code(
            ada_mpc_nc: AdaptiveMPC_NoConstraints,
            file_name=None):
        """
        Generates C++ header files for deploying an Adaptive Model Predictive Control (MPC) system without constraints.
        This function automates the process of converting a configured Python AdaptiveMPC_NoConstraints object into a set of C++ header files.
        These files encapsulate the system's matrices, Kalman filter,
         prediction matrices, solver factors, weights, and updater functions,
        making them ready for integration into a C++ codebase.
        The function performs the following steps:
            - Inspects the calling context to determine variable and file names for code generation.
            - Generates C++ code for the B matrix, embedded integrator updater,
              prediction matrices updater, and adaptive MPC Phi/F updater.
            - Generates C++ code for the Kalman filter (EKF), F and Phi matrices,
              solver factor, and input weight matrices.
            - Assembles a main C++ header file that includes all generated components and defines the necessary types and a factory function for the Adaptive MPC object.
            - Writes all generated code to files and returns their names.

        Notes:
            - This function relies on several helper classes and functions (e.g., ControlDeploy, NumpyDeploy, KalmanFilterDeploy, AdaptiveMatricesDeploy).
            - The generated C++ code assumes a specific structure and naming convention for types and factory functions.
            - The function uses dynamic code execution (exec/eval) to handle variable names and code generation context.
        """

        parameters = ada_mpc_nc.kalman_filter.Parameters
        number_of_delay = ada_mpc_nc.kalman_filter.Number_of_Delay

        deployed_file_names = []

        ControlDeploy.restrict_data_type(ada_mpc_nc.kalman_filter.A.dtype.name)

        type_name = NumpyDeploy.check_dtype(ada_mpc_nc.kalman_filter.A)

        # %% inspect arguments
        # Get the caller's frame
        frame = inspect.currentframe().f_back
        # Get the caller's local variables
        caller_locals = frame.f_locals
        # Find the variable name that matches the matrix_in value
        variable_name = None
        for name, value in caller_locals.items():
            if value is ada_mpc_nc:
                variable_name = name
                break
        # Get the caller's file name
        if file_name is None:
            caller_file_full_path = frame.f_code.co_filename
            caller_file_name = os.path.basename(caller_file_full_path)
            caller_file_name_without_ext = os.path.splitext(caller_file_name)[
                0]
        else:
            caller_file_name_without_ext = file_name

        code_file_name = caller_file_name_without_ext + "_" + variable_name
        code_file_name_ext = code_file_name + ".hpp"

        # %% generate B matrix code
        _, _, B_symbolic_SparseAvailable_list = KalmanFilterDeploy.create_state_and_measurement_function_code(
            ada_mpc_nc.B_symbolic_file_name, "B_Type"
        )

        B_matrix_name = f"{variable_name}_B"

        exec(
            f"{B_matrix_name} = B_symbolic_SparseAvailable_list[0]")
        B_file_name = eval(
            f"NumpyDeploy.generate_matrix_cpp_code(matrix_in={B_matrix_name}, " +
            "file_name=caller_file_name_without_ext)")

        deployed_file_names.append(B_file_name)
        B_file_name_no_extension = B_file_name.split(".")[0]

        # %% generate Embedded Integrator Updater code
        embedded_integrator_updater_file_name = \
            ada_mpc_nc.state_space_initializer.embedded_integrator_updater_file_name
        embedded_integrator_updater_name_no_extension = \
            embedded_integrator_updater_file_name.split(".")[0]
        embedded_integrator_updater_cpp_name = embedded_integrator_updater_name_no_extension + ".hpp"

        embedded_integrator_updater_code = \
            AdaptiveMatricesDeploy.generate_embedded_integrator_updater_cpp_code(
                embedded_integrator_updater_file_name, embedded_integrator_updater_name_no_extension)

        embedded_integrator_updater_cpp_name_ext = ControlDeploy.write_to_file(
            embedded_integrator_updater_code, embedded_integrator_updater_cpp_name)

        deployed_file_names.append(embedded_integrator_updater_cpp_name_ext)

        # %% generate Prediction Matrices Phi F updater code
        prediction_matrices_updater_file_name = \
            ada_mpc_nc.state_space_initializer.prediction_matrices_phi_f_updater_file_name
        prediction_matrices_updater_name_no_extension = \
            prediction_matrices_updater_file_name.split(".")[0]
        prediction_matrices_updater_cpp_name = prediction_matrices_updater_name_no_extension + ".hpp"

        prediction_matrices_updater_code = \
            AdaptiveMatricesDeploy.generate_prediction_matrices_phi_f_updater_cpp_code(
                prediction_matrices_updater_file_name, prediction_matrices_updater_name_no_extension)

        prediction_matrices_updater_cpp_name_ext = ControlDeploy.write_to_file(
            prediction_matrices_updater_code, prediction_matrices_updater_cpp_name)

        deployed_file_names.append(prediction_matrices_updater_cpp_name_ext)

        # %% generate Adaptive MPC Phi F Updater code
        Adaptive_MPC_Phi_F_updater_file_name = \
            ada_mpc_nc.state_space_initializer.Adaptive_MPC_Phi_F_updater_file_name
        Adaptive_MPC_Phi_F_updater_name_no_extension = \
            Adaptive_MPC_Phi_F_updater_file_name.split(".")[0]
        Adaptive_MPC_Phi_F_updater_cpp_name = Adaptive_MPC_Phi_F_updater_name_no_extension + ".hpp"

        Adaptive_MPC_Phi_F_updater_code = \
            AdaptiveMatricesDeploy.generate_adaptive_mpc_phi_f_updater_cpp_code(
                Adaptive_MPC_Phi_F_updater_file_name, Adaptive_MPC_Phi_F_updater_name_no_extension,
                embedded_integrator_updater_cpp_name,
                prediction_matrices_updater_cpp_name)

        Adaptive_MPC_Phi_F_updater_cpp_name_ext = ControlDeploy.write_to_file(
            Adaptive_MPC_Phi_F_updater_code, Adaptive_MPC_Phi_F_updater_cpp_name)

        deployed_file_names.append(Adaptive_MPC_Phi_F_updater_cpp_name_ext)

        # %% create EKF, F, Phi, solver_factor, Weight_U_Nc code
        exec(f"{variable_name}_ekf = ada_mpc_nc.kalman_filter")
        ekf_file_names = eval(
            f"KalmanFilterDeploy.generate_EKF_cpp_code({variable_name}_ekf, caller_file_name_without_ext, number_of_delay={number_of_delay})")

        deployed_file_names.append(ekf_file_names)
        ekf_file_name = ekf_file_names[-1]

        ekf_file_name_no_extension = ekf_file_name.split(".")[0]

        parameter_code_file_name = ""
        parameter_code_file_name_without_ext = ""
        for name in ekf_file_names:
            if "parameter" in name:
                parameter_code_file_name = name
                parameter_code_file_name_without_ext = name.split(".")[
                    0]
                break

        if parameter_code_file_name == "":
            raise ValueError(
                "No parameter file found in EKF deployment files.")

        # create F
        F_SparseAvailable = convert_SparseAvailable_for_deploy(
            ada_mpc_nc.prediction_matrices.F_SparseAvailable)
        exec(f"{variable_name}_F = ada_mpc_nc.prediction_matrices.F_ndarray")
        F_file_name = eval(
            f"NumpyDeploy.generate_matrix_cpp_code(matrix_in={variable_name}_F, " +
            "SparseAvailable=F_SparseAvailable, " +
            "file_name=caller_file_name_without_ext)")

        deployed_file_names.append(F_file_name)
        F_file_name_no_extension = F_file_name.split(".")[0]

        # create Phi
        Phi_SparseAvailable = convert_SparseAvailable_for_deploy(
            ada_mpc_nc.prediction_matrices.Phi_SparseAvailable)
        exec(
            f"{variable_name}_Phi = ada_mpc_nc.prediction_matrices.Phi_ndarray")
        Phi_file_name = eval(
            f"NumpyDeploy.generate_matrix_cpp_code(matrix_in={variable_name}_Phi, " +
            "SparseAvailable=Phi_SparseAvailable, " +
            "file_name=caller_file_name_without_ext)")

        deployed_file_names.append(Phi_file_name)
        Phi_file_name_no_extension = Phi_file_name.split(".")[0]

        # create solver_factor
        solver_factor_SparseAvailable = convert_SparseAvailable_for_deploy(
            ada_mpc_nc.solver_factor_SparseAvailable)
        exec(f"{variable_name}_solver_factor = ada_mpc_nc.solver_factor")
        solver_factor_file_name = eval(
            f"NumpyDeploy.generate_matrix_cpp_code(matrix_in={variable_name}_solver_factor, " +
            "SparseAvailable=solver_factor_SparseAvailable, " +
            "file_name=caller_file_name_without_ext)")

        deployed_file_names.append(solver_factor_file_name)
        solver_factor_file_name_no_extension = solver_factor_file_name.split(".")[
            0]

        exec(f"{variable_name}_Weight_U_Nc = ada_mpc_nc.Weight_U_Nc")
        Weight_U_Nc_file_name = eval(
            f"NumpyDeploy.generate_matrix_cpp_code(matrix_in={variable_name}_Weight_U_Nc, " +
            "file_name=caller_file_name_without_ext)")

        deployed_file_names.append(Weight_U_Nc_file_name)
        Weight_U_Nc_file_name_no_extension = Weight_U_Nc_file_name.split(".")[
            0]

        # %% main code generation
        code_text = ""

        file_header_macro_name = "__" + code_file_name.upper() + "_HPP__"

        code_text += "#ifndef " + file_header_macro_name + "\n"
        code_text += "#define " + file_header_macro_name + "\n\n"

        code_text += f"#include \"{B_file_name}\"\n"
        code_text += f"#include \"{ekf_file_name}\"\n"
        code_text += f"#include \"{F_file_name}\"\n"
        code_text += f"#include \"{Phi_file_name}\"\n"
        code_text += f"#include \"{solver_factor_file_name}\"\n"
        code_text += f"#include \"{Weight_U_Nc_file_name}\"\n"
        code_text += f"#include \"{parameter_code_file_name}\"\n"
        code_text += f"#include \"{Adaptive_MPC_Phi_F_updater_cpp_name}\"\n\n"

        code_text += "#include \"python_mpc.hpp\"\n\n"

        namespace_name = code_file_name

        code_text += "namespace " + namespace_name + " {\n\n"

        code_text += "using namespace PythonNumpy;\n"
        code_text += "using namespace PythonControl;\n"
        code_text += "using namespace PythonMPC;\n\n"

        code_text += f"constexpr std::size_t NP = {ada_mpc_nc.Np};\n"
        code_text += f"constexpr std::size_t NC = {ada_mpc_nc.Nc};\n\n"

        code_text += f"constexpr std::size_t INPUT_SIZE = {ekf_file_name_no_extension}::INPUT_SIZE;\n"
        code_text += f"constexpr std::size_t STATE_SIZE = {ekf_file_name_no_extension}::STATE_SIZE;\n"
        code_text += f"constexpr std::size_t OUTPUT_SIZE = {ekf_file_name_no_extension}::OUTPUT_SIZE;\n\n"

        code_text += f"constexpr std::size_t AUGMENTED_STATE_SIZE = STATE_SIZE + OUTPUT_SIZE;\n\n"

        code_text += f"constexpr std::size_t NUMBER_OF_DELAY = {ekf_file_name_no_extension}::NUMBER_OF_DELAY;\n\n"

        code_text += f"using EKF_Type = {ekf_file_name_no_extension}::type;\n\n"

        code_text += f"using A_Type = typename EKF_Type::A_Type;\n\n"

        code_text += f"using B_Type = {B_file_name_no_extension}::type;\n\n"

        code_text += f"using C_Type = typename EKF_Type::C_Type;\n\n"

        code_text += f"using X_Type = StateSpaceState_Type<{type_name}, STATE_SIZE>;\n\n"

        code_text += f"using U_Type = StateSpaceOutput_Type<{type_name}, INPUT_SIZE>;\n\n"

        code_text += f"using F_Type = {F_file_name_no_extension}::type;\n\n"

        code_text += f"using Phi_Type = {Phi_file_name_no_extension}::type;\n\n"

        code_text += f"using SolverFactor_Type = {solver_factor_file_name_no_extension}::type;\n\n"

        code_text += f"using PredictionMatrices_Type = MPC_PredictionMatrices_Type<\n" + \
            "  F_Type, Phi_Type, NP, NC, INPUT_SIZE, AUGMENTED_STATE_SIZE, OUTPUT_SIZE>;\n\n"

        ref_row_size_text = "1"
        if ada_mpc_nc.is_ref_trajectory:
            ref_row_size_text = "NP"

        code_text += f"using Ref_Type = DenseMatrix_Type<{type_name}, OUTPUT_SIZE, " + \
            ref_row_size_text + ">;\n\n"

        code_text += f"using ReferenceTrajectory_Type = MPC_ReferenceTrajectory_Type<\n" + \
            "  Ref_Type, NP>;\n\n"

        code_text += f"using Parameter_Type = {parameter_code_file_name_without_ext}::Parameter_Type;\n\n"

        code_text += f"using Weight_U_Nc_Type = {Weight_U_Nc_file_name_no_extension}::type;\n\n"

        code_text += f"using EmbeddedIntegratorStateSpace_Type =\n" + \
            f"  typename EmbeddedIntegratorTypes<A_Type, B_Type, C_Type>::StateSpace_Type;\n\n"

        code_text += f"using type = AdaptiveMPC_NoConstraints_Type<B_Type,\n" + \
            "  EKF_Type, PredictionMatrices_Type, ReferenceTrajectory_Type,\n" + \
            "  Parameter_Type, SolverFactor_Type>;\n\n"

        code_text += "inline auto make() -> type {\n\n"

        code_text += f"  auto kalman_filter = {ekf_file_name_no_extension}::make();\n\n"

        for i in range(ada_mpc_nc.kalman_filter.x_hat.shape[0]):
            code_text += f"  kalman_filter.X_hat.template set<{i}, 0>(" + \
                f"static_cast<{type_name}>({ada_mpc_nc.kalman_filter.x_hat[i, 0]}));\n"

        code_text += "\n"

        code_text += f"  auto F = {F_file_name_no_extension}::make();\n\n"

        code_text += f"  auto Phi = {Phi_file_name_no_extension}::make();\n\n"

        code_text += f"  auto solver_factor = {solver_factor_file_name_no_extension}::make();\n\n"

        code_text += f"  PredictionMatrices_Type prediction_matrices(F, Phi);\n\n"

        code_text += f"  ReferenceTrajectory_Type reference_trajectory;\n\n"

        code_text += f"  Weight_U_Nc_Type Weight_U_Nc = {Weight_U_Nc_file_name_no_extension}::make();\n\n"

        adaptive_mpc_phi_f_updater_name = caller_file_name_without_ext + \
            "_adaptive_mpc_phi_f_updater"

        code_text += f"  Adaptive_MPC_Phi_F_Updater_Function_Object<\n" + \
            f"    X_Type, U_Type, Parameter_Type,\n" + \
            f"    Phi_Type, F_Type, EmbeddedIntegratorStateSpace_Type>\n" + \
            f"    Adaptive_MPC_Phi_F_Updater_Function =\n" + \
            f"    {adaptive_mpc_phi_f_updater_name}::Adaptive_MPC_Phi_F_Updater::update<\n" + \
            f"      X_Type, U_Type, Parameter_Type,\n" + \
            f"      Phi_Type, F_Type, EmbeddedIntegratorStateSpace_Type>;\n\n"

        code_text += f"  auto adaptive_mpc_nc = make_AdaptiveMPC_NoConstraints<B_Type,\n" + \
            f"      EKF_Type, PredictionMatrices_Type,\n" + \
            f"      ReferenceTrajectory_Type, Parameter_Type,\n" + \
            f"      SolverFactor_Type, Weight_U_Nc_Type,\n" + \
            f"      X_Type, U_Type,\n" + \
            f"      EmbeddedIntegratorStateSpace_Type>(\n" + \
            f"      kalman_filter, prediction_matrices, reference_trajectory, solver_factor,\n" + \
            f"      Weight_U_Nc, Adaptive_MPC_Phi_F_Updater_Function);\n\n"

        code_text += "  return adaptive_mpc_nc;\n\n"

        code_text += "}\n\n"

        code_text += "} // namespace " + namespace_name + "\n\n"

        code_text += "#endif // " + file_header_macro_name + "\n"

        code_file_name_ext = ControlDeploy.write_to_file(
            code_text, code_file_name_ext)

        deployed_file_names.append(code_file_name_ext)

        return deployed_file_names
