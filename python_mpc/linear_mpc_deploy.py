"""
File: linear_mpc_deploy.py

This module provides functionality for deploying Linear Model Predictive Control (MPC) objects to C++ code.
It contains utilities to generate C++ header files from Python-based MPC models, including both constrained
and unconstrained LTI MPCs. The generated code includes all necessary matrices, parameters, and type definitions
to instantiate and use the MPC controller in a C++ environment.
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
from python_mpc.ltv_matrices_deploy import LTVMatricesDeploy

from external_libraries.MCAP_python_mpc.mpc_utility.linear_solver_utility import DU_U_Y_Limits
from external_libraries.MCAP_python_mpc.python_mpc.linear_mpc import LTI_MPC_NoConstraints
from external_libraries.MCAP_python_mpc.python_mpc.linear_mpc import LTI_MPC
from external_libraries.MCAP_python_mpc.python_mpc.linear_mpc import LTV_MPC_NoConstraints

TOL = 1e-30


class LinearMPC_Deploy:
    """
    A class for deploying Linear Model Predictive Control (MPC) objects to C++ code.
    This class provides static methods to generate C++ header files from Python-based MPC models,
    including both constrained and unconstrained LTI MPCs.
    The generated code includes all necessary matrices, parameters, and type definitions
    to instantiate and use the MPC controller in a C++ environment.
    Attributes:
        None
    Methods:
        generate_LTI_MPC_NC_cpp_code(lti_mpc_nc, file_name=None, number_of_delay=0):
            Generates C++ code for an LTI MPC without constraints.
        generate_LTI_MPC_cpp_code(lti_mpc, file_name=None, number_of_delay=0):
            Generates C++ code for an LTI MPC with constraints.
    """

    @staticmethod
    def generate_LTI_MPC_NC_cpp_code(
            lti_mpc_nc: LTI_MPC_NoConstraints, file_name=None):
        """
        Generates C++ code for an LTI MPC without constraints.
        Args:
            lti_mpc_nc (LTI_MPC_NoConstraints): The LTI MPC without constraints object to deploy.
            file_name (str, optional): The name of the file to save the generated C++ code. If None, uses the caller's file name.
            number_of_delay (int, optional): The number of delays in the MPC. Defaults to 0.
        Returns:
            list: A list of file names of the generated C++ code files.
        """
        number_of_delay = lti_mpc_nc.Number_of_Delay

        deployed_file_names = []

        ControlDeploy.restrict_data_type(lti_mpc_nc.kalman_filter.A.dtype.name)

        type_name = NumpyDeploy.check_dtype(lti_mpc_nc.kalman_filter.A)

        # %% inspect arguments
        # Get the caller's frame
        frame = inspect.currentframe().f_back
        # Get the caller's local variables
        caller_locals = frame.f_locals
        # Find the variable name that matches the matrix_in value
        variable_name = None
        for name, value in caller_locals.items():
            if value is lti_mpc_nc:
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

        # %% code generation
        code_file_name = caller_file_name_without_ext + "_" + variable_name
        code_file_name_ext = code_file_name + ".hpp"

        # create LKF code
        exec(f"{variable_name}_lkf = lti_mpc_nc.kalman_filter")
        lkf_file_names = eval(
            f"KalmanFilterDeploy.generate_LKF_cpp_code({variable_name}_lkf, caller_file_name_without_ext, number_of_delay={number_of_delay})")

        deployed_file_names.append(lkf_file_names)
        lkf_file_name = lkf_file_names[-1]

        lkf_file_name_no_extension = lkf_file_name.split(".")[0]

        # create F code
        exec(f"{variable_name}_F = lti_mpc_nc.prediction_matrices.F_ndarray")
        F_file_name = eval(
            f"NumpyDeploy.generate_matrix_cpp_code({variable_name}_F, caller_file_name_without_ext)")

        deployed_file_names.append(F_file_name)
        F_file_name_no_extension = F_file_name.split(".")[0]

        # create Phi code
        exec(
            f"{variable_name}_Phi = lti_mpc_nc.prediction_matrices.Phi_ndarray")
        Phi_file_name = eval(
            f"NumpyDeploy.generate_matrix_cpp_code({variable_name}_Phi, caller_file_name_without_ext)")

        deployed_file_names.append(Phi_file_name)
        Phi_file_name_no_extension = Phi_file_name.split(".")[0]

        # create solver_factor code
        exec(f"{variable_name}_solver_factor = lti_mpc_nc.solver_factor")
        solver_factor_file_name = eval(
            f"NumpyDeploy.generate_matrix_cpp_code({variable_name}_solver_factor, caller_file_name_without_ext)")

        deployed_file_names.append(solver_factor_file_name)
        solver_factor_file_name_no_extension = solver_factor_file_name.split(".")[
            0]

        # create cpp code
        code_text = ""

        file_header_macro_name = "__" + code_file_name.upper() + "_HPP__"

        code_text += "#ifndef " + file_header_macro_name + "\n"
        code_text += "#define " + file_header_macro_name + "\n\n"

        code_text += f"#include \"{lkf_file_name}\"\n"
        code_text += f"#include \"{F_file_name}\"\n"
        code_text += f"#include \"{Phi_file_name}\"\n"
        code_text += f"#include \"{solver_factor_file_name}\"\n\n"
        code_text += "#include \"python_mpc.hpp\"\n\n"

        namespace_name = code_file_name

        code_text += "namespace " + namespace_name + " {\n\n"

        code_text += "using namespace PythonNumpy;\n"
        code_text += "using namespace PythonControl;\n"
        code_text += "using namespace PythonMPC;\n\n"

        code_text += f"constexpr std::size_t NP = {lti_mpc_nc.Np};\n"
        code_text += f"constexpr std::size_t NC = {lti_mpc_nc.Nc};\n\n"

        code_text += f"constexpr std::size_t INPUT_SIZE = {lkf_file_name_no_extension}::INPUT_SIZE;\n"
        code_text += f"constexpr std::size_t STATE_SIZE = {lkf_file_name_no_extension}::STATE_SIZE;\n"
        code_text += f"constexpr std::size_t OUTPUT_SIZE = {lkf_file_name_no_extension}::OUTPUT_SIZE;\n\n"

        code_text += f"constexpr std::size_t AUGMENTED_STATE_SIZE = STATE_SIZE + OUTPUT_SIZE;\n\n"

        code_text += f"constexpr std::size_t NUMBER_OF_DELAY = {lkf_file_name_no_extension}::NUMBER_OF_DELAY;\n\n"

        code_text += f"using LKF_Type = {lkf_file_name_no_extension}::type;\n\n"

        code_text += f"using F_Type = {F_file_name_no_extension}::type;\n\n"

        code_text += f"using Phi_Type = {Phi_file_name_no_extension}::type;\n\n"

        code_text += f"using SolverFactor_Type = {solver_factor_file_name_no_extension}::type;\n\n"

        code_text += f"using PredictionMatrices_Type = MPC_PredictionMatrices_Type<\n" + \
            "  F_Type, Phi_Type, NP, NC, INPUT_SIZE, AUGMENTED_STATE_SIZE, OUTPUT_SIZE>;\n\n"

        ref_row_size_text = "1"
        if lti_mpc_nc.is_ref_trajectory:
            ref_row_size_text = "NP"

        code_text += f"using Ref_Type = DenseMatrix_Type<{type_name}, OUTPUT_SIZE, " + \
            ref_row_size_text + ">;\n\n"

        code_text += f"using ReferenceTrajectory_Type = MPC_ReferenceTrajectory_Type<\n" + \
            "  Ref_Type, NP>;\n\n"

        code_text += f"using type = LTI_MPC_NoConstraints_Type<\n" + \
            "  LKF_Type, PredictionMatrices_Type, ReferenceTrajectory_Type,\n" + \
            "  SolverFactor_Type>;\n\n"

        code_text += "inline auto make() -> type {\n\n"

        code_text += f"  auto kalman_filter = {lkf_file_name_no_extension}::make();\n\n"

        code_text += f"  auto F = {F_file_name_no_extension}::make();\n\n"

        code_text += f"  auto Phi = {Phi_file_name_no_extension}::make();\n\n"

        code_text += f"  auto solver_factor = {solver_factor_file_name_no_extension}::make();\n\n"

        code_text += f"  PredictionMatrices_Type prediction_matrices(F, Phi);\n\n"

        code_text += f"  ReferenceTrajectory_Type reference_trajectory;\n\n"

        code_text += f"  auto lti_mpc_nc = make_LTI_MPC_NoConstraints(\n" + \
            "    kalman_filter, prediction_matrices, reference_trajectory, solver_factor);\n\n"

        code_text += "  return lti_mpc_nc;\n\n"

        code_text += "}\n\n"

        code_text += "} // namespace " + namespace_name + "\n\n"

        code_text += "#endif // " + file_header_macro_name + "\n"

        code_file_name_ext = ControlDeploy.write_to_file(
            code_text, code_file_name_ext)

        deployed_file_names.append(code_file_name_ext)

        return deployed_file_names

    @staticmethod
    def get_cpp_bool_text(flag: bool) -> str:
        """
        Converts a Python boolean to a C++ boolean string representation.
        Args:
            flag (bool): The boolean value to convert.
        Returns:
            str: "true" if flag is True, "false" if flag is False.
        """
        if flag:
            return "true"
        else:
            return "false"

    @staticmethod
    def generate_LTI_MPC_cpp_code(
            lti_mpc: LTI_MPC, file_name=None):
        """
        Generates C++ code for an LTI MPC with constraints.
        Args:
            lti_mpc (LTI_MPC): The LTI MPC object to deploy.
            file_name (str, optional): The name of the file to save the generated C++ code. If None, uses the caller's file name.
            number_of_delay (int, optional): The number of delays in the MPC. Defaults to 0.
        Returns:
            list: A list of file names of the generated C++ code files.
        """
        number_of_delay = lti_mpc.kalman_filter.Number_of_Delay

        deployed_file_names = []

        ControlDeploy.restrict_data_type(lti_mpc.kalman_filter.A.dtype.name)

        type_name = NumpyDeploy.check_dtype(lti_mpc.kalman_filter.A)

        # %% inspect arguments
        # Get the caller's frame
        frame = inspect.currentframe().f_back
        # Get the caller's local variables
        caller_locals = frame.f_locals
        # Find the variable name that matches the matrix_in value
        variable_name = None
        for name, value in caller_locals.items():
            if value is lti_mpc:
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

        # %% code generation
        code_file_name = caller_file_name_without_ext + "_" + variable_name
        code_file_name_ext = code_file_name + ".hpp"

        # create LKF code
        exec(f"{variable_name}_lkf = lti_mpc.kalman_filter")
        lkf_file_names = eval(
            f"KalmanFilterDeploy.generate_LKF_cpp_code({variable_name}_lkf, caller_file_name_without_ext, number_of_delay={number_of_delay})")

        deployed_file_names.append(lkf_file_names)
        lkf_file_name = lkf_file_names[-1]

        lkf_file_name_no_extension = lkf_file_name.split(".")[0]

        # create F code
        exec(f"{variable_name}_F = lti_mpc.prediction_matrices.F_ndarray")
        F_file_name = eval(
            f"NumpyDeploy.generate_matrix_cpp_code({variable_name}_F, caller_file_name_without_ext)")

        deployed_file_names.append(F_file_name)
        F_file_name_no_extension = F_file_name.split(".")[0]

        # create Phi code
        exec(f"{variable_name}_Phi = lti_mpc.prediction_matrices.Phi_ndarray")
        Phi_file_name = eval(
            f"NumpyDeploy.generate_matrix_cpp_code({variable_name}_Phi, caller_file_name_without_ext)")

        deployed_file_names.append(Phi_file_name)
        Phi_file_name_no_extension = Phi_file_name.split(".")[0]

        # create solver_factor code
        exec(f"{variable_name}_solver_factor = lti_mpc.solver_factor")
        solver_factor_file_name = eval(
            f"NumpyDeploy.generate_matrix_cpp_code({variable_name}_solver_factor, caller_file_name_without_ext)")

        deployed_file_names.append(solver_factor_file_name)
        solver_factor_file_name_no_extension = solver_factor_file_name.split(".")[
            0]

        # create Weight_U_Nc code
        exec(f"{variable_name}_Weight_U_Nc = lti_mpc.Weight_U_Nc")
        Weight_U_Nc_file_name = eval(
            f"NumpyDeploy.generate_matrix_cpp_code({variable_name}_Weight_U_Nc, caller_file_name_without_ext)")

        deployed_file_names.append(Weight_U_Nc_file_name)
        Weight_U_Nc_file_name_no_extension = Weight_U_Nc_file_name.split(".")[
            0]

        # Limits code
        delta_U_min_values = lti_mpc.qp_solver.DU_U_Y_Limits.delta_U_min
        if delta_U_min_values is not None:
            delta_U_min_active_set = np.zeros(
                np.size(delta_U_min_values), dtype=bool)
            for i in range(len(delta_U_min_values)):
                if lti_mpc.qp_solver.DU_U_Y_Limits.is_delta_U_min_active(i):
                    delta_U_min_active_set[i] = True

        delta_U_max_values = lti_mpc.qp_solver.DU_U_Y_Limits.delta_U_max
        if delta_U_max_values is not None:
            delta_U_max_active_set = np.zeros(
                np.size(delta_U_max_values), dtype=bool)
            for i in range(len(delta_U_max_values)):
                if lti_mpc.qp_solver.DU_U_Y_Limits.is_delta_U_max_active(i):
                    delta_U_max_active_set[i] = True

        U_min_values = lti_mpc.qp_solver.DU_U_Y_Limits.U_min
        if U_min_values is not None:
            U_min_active_set = np.zeros(np.size(U_min_values), dtype=bool)
            for i in range(len(U_min_values)):
                if lti_mpc.qp_solver.DU_U_Y_Limits.is_U_min_active(i):
                    U_min_active_set[i] = True

        U_max_values = lti_mpc.qp_solver.DU_U_Y_Limits.U_max
        if U_max_values is not None:
            U_max_active_set = np.zeros(np.size(U_max_values), dtype=bool)
            for i in range(len(U_max_values)):
                if lti_mpc.qp_solver.DU_U_Y_Limits.is_U_max_active(i):
                    U_max_active_set[i] = True

        Y_min_values = lti_mpc.qp_solver.DU_U_Y_Limits.Y_min
        if Y_min_values is not None:
            Y_min_active_set = np.zeros(np.size(Y_min_values), dtype=bool)
            for i in range(len(Y_min_values)):
                if lti_mpc.qp_solver.DU_U_Y_Limits.is_Y_min_active(i):
                    Y_min_active_set[i] = True

        Y_max_values = lti_mpc.qp_solver.DU_U_Y_Limits.Y_max
        if Y_max_values is not None:
            Y_max_active_set = np.zeros(np.size(Y_max_values), dtype=bool)
            for i in range(len(Y_max_values)):
                if lti_mpc.qp_solver.DU_U_Y_Limits.is_Y_max_active(i):
                    Y_max_active_set[i] = True

        # create Limits code
        delta_U_min = copy.deepcopy(delta_U_min_active_set)
        delta_U_min = np.array(
            delta_U_min, dtype=delta_U_max_values.dtype).reshape(-1, 1)
        exec(f"{variable_name}_delta_U_min = delta_U_min")
        delta_U_min_file_name = eval(
            f"NumpyDeploy.generate_matrix_cpp_code({variable_name}_delta_U_min, caller_file_name_without_ext)")

        deployed_file_names.append(delta_U_min_file_name)
        delta_U_min_file_name_no_extension = delta_U_min_file_name .split(".")[
            0]

        delta_U_max = copy.deepcopy(delta_U_max_active_set)
        delta_U_max = np.array(
            delta_U_max, dtype=delta_U_max_values.dtype).reshape(-1, 1)
        exec(f"{variable_name}_delta_U_max = delta_U_max")
        delta_U_max_file_name = eval(
            f"NumpyDeploy.generate_matrix_cpp_code({variable_name}_delta_U_max, caller_file_name_without_ext)")

        deployed_file_names.append(delta_U_max_file_name)
        delta_U_max_file_name_no_extension = delta_U_max_file_name .split(".")[
            0]

        U_min = copy.deepcopy(U_min_active_set)
        U_min = np.array(U_min, dtype=U_min_values.dtype).reshape(-1, 1)
        exec(f"{variable_name}_U_min = U_min")
        U_min_file_name = eval(
            f"NumpyDeploy.generate_matrix_cpp_code({variable_name}_U_min, caller_file_name_without_ext)")

        deployed_file_names.append(U_min_file_name)
        U_min_file_name_no_extension = U_min_file_name .split(".")[0]

        U_max = copy.deepcopy(U_max_active_set)
        U_max = np.array(U_max, dtype=U_max_values.dtype).reshape(-1, 1)
        exec(f"{variable_name}_U_max = U_max")
        U_max_file_name = eval(
            f"NumpyDeploy.generate_matrix_cpp_code({variable_name}_U_max, caller_file_name_without_ext)")

        deployed_file_names.append(U_max_file_name)
        U_max_file_name_no_extension = U_max_file_name .split(".")[0]

        Y_min = copy.deepcopy(Y_min_active_set)
        Y_min = np.array(Y_min, dtype=Y_min_values.dtype).reshape(-1, 1)
        exec(f"{variable_name}_Y_min = Y_min")
        Y_min_file_name = eval(
            f"NumpyDeploy.generate_matrix_cpp_code({variable_name}_Y_min, caller_file_name_without_ext)")

        deployed_file_names.append(Y_min_file_name)
        Y_min_file_name_no_extension = Y_min_file_name .split(".")[0]

        Y_max = copy.deepcopy(Y_max_active_set)
        Y_max = np.array(Y_max, dtype=Y_max_values.dtype).reshape(-1, 1)
        exec(f"{variable_name}_Y_max = Y_max")
        Y_max_file_name = eval(
            f"NumpyDeploy.generate_matrix_cpp_code({variable_name}_Y_max, caller_file_name_without_ext)")

        deployed_file_names.append(Y_max_file_name)
        Y_max_file_name_no_extension = Y_max_file_name .split(".")[0]

        # create cpp code
        code_text = ""

        file_header_macro_name = "__" + code_file_name.upper() + "_HPP__"

        code_text += "#ifndef " + file_header_macro_name + "\n"
        code_text += "#define " + file_header_macro_name + "\n\n"

        code_text += f"#include \"{lkf_file_name}\"\n"
        code_text += f"#include \"{F_file_name}\"\n"
        code_text += f"#include \"{Phi_file_name}\"\n"
        code_text += f"#include \"{solver_factor_file_name}\"\n"
        code_text += f"#include \"{Weight_U_Nc_file_name}\"\n\n"

        code_text += f"#include \"{delta_U_min_file_name}\"\n"
        code_text += f"#include \"{delta_U_max_file_name}\"\n"
        code_text += f"#include \"{U_min_file_name}\"\n"
        code_text += f"#include \"{U_max_file_name}\"\n"
        code_text += f"#include \"{Y_min_file_name}\"\n"
        code_text += f"#include \"{Y_max_file_name}\"\n\n"

        code_text += "#include \"python_mpc.hpp\"\n\n"

        namespace_name = code_file_name

        code_text += "namespace " + namespace_name + " {\n\n"

        code_text += "using namespace PythonNumpy;\n"
        code_text += "using namespace PythonControl;\n"
        code_text += "using namespace PythonMPC;\n\n"

        code_text += f"constexpr std::size_t NP = {lti_mpc.Np};\n"
        code_text += f"constexpr std::size_t NC = {lti_mpc.Nc};\n\n"

        code_text += f"constexpr std::size_t INPUT_SIZE = {lkf_file_name_no_extension}::INPUT_SIZE;\n"
        code_text += f"constexpr std::size_t STATE_SIZE = {lkf_file_name_no_extension}::STATE_SIZE;\n"
        code_text += f"constexpr std::size_t OUTPUT_SIZE = {lkf_file_name_no_extension}::OUTPUT_SIZE;\n\n"

        code_text += f"constexpr std::size_t AUGMENTED_STATE_SIZE = STATE_SIZE + OUTPUT_SIZE;\n\n"

        code_text += f"constexpr std::size_t NUMBER_OF_DELAY = {lkf_file_name_no_extension}::NUMBER_OF_DELAY;\n\n"

        code_text += f"using LKF_Type = {lkf_file_name_no_extension}::type;\n\n"

        code_text += f"using F_Type = {F_file_name_no_extension}::type;\n\n"

        code_text += f"using Phi_Type = {Phi_file_name_no_extension}::type;\n\n"

        code_text += f"using SolverFactor_Type = {solver_factor_file_name_no_extension}::type;\n\n"

        code_text += f"using Delta_U_Min_Type = {delta_U_min_file_name_no_extension}::type;\n\n"

        code_text += f"using Delta_U_Max_Type = {delta_U_max_file_name_no_extension}::type;\n\n"

        code_text += f"using U_Min_Type = {U_min_file_name_no_extension}::type;\n\n"

        code_text += f"using U_Max_Type = {U_max_file_name_no_extension}::type;\n\n"

        code_text += f"using Y_Min_Type = {Y_min_file_name_no_extension}::type;\n\n"

        code_text += f"using Y_Max_Type = {Y_max_file_name_no_extension}::type;\n\n"

        code_text += f"using PredictionMatrices_Type = MPC_PredictionMatrices_Type<\n" + \
            "  F_Type, Phi_Type, NP, NC, INPUT_SIZE, AUGMENTED_STATE_SIZE, OUTPUT_SIZE>;\n\n"

        ref_row_size_text = "1"
        if lti_mpc.is_ref_trajectory:
            ref_row_size_text = "NP"

        code_text += f"using Ref_Type = DenseMatrix_Type<{type_name}, OUTPUT_SIZE, " + \
            ref_row_size_text + ">;\n\n"

        code_text += f"using ReferenceTrajectory_Type = MPC_ReferenceTrajectory_Type<\n" + \
            "  Ref_Type, NP>;\n\n"

        code_text += f"using type = LTI_MPC_Type<\n" + \
            "  LKF_Type, PredictionMatrices_Type, ReferenceTrajectory_Type,\n" + \
            "  Delta_U_Min_Type, Delta_U_Max_Type,\n" + \
            "  U_Min_Type, U_Max_Type, Y_Min_Type, Y_Max_Type,\n" + \
            "  SolverFactor_Type>;\n\n"

        code_text += "inline auto make() -> type {\n\n"

        code_text += f"  auto kalman_filter = {lkf_file_name_no_extension}::make();\n\n"

        code_text += f"  auto F = {F_file_name_no_extension}::make();\n\n"

        code_text += f"  auto Phi = {Phi_file_name_no_extension}::make();\n\n"

        code_text += f"  auto solver_factor = {solver_factor_file_name_no_extension}::make();\n\n"

        code_text += f"  auto Weight_U_Nc = {Weight_U_Nc_file_name_no_extension}::make();\n\n"

        # limits
        code_text += f"  auto delta_U_min = {delta_U_min_file_name_no_extension}::make();\n\n"
        if delta_U_min is not None and np.linalg.norm(delta_U_min_active_set) > TOL:
            for i in range(len(delta_U_min)):
                if delta_U_min_active_set[i]:
                    code_text += f"  delta_U_min.template set<{i}, 0>("
                    code_text += f"static_cast<{type_name}>({delta_U_min_values[i, 0]})"
                    code_text += ");\n"
            code_text += "\n"

        code_text += f"  auto delta_U_max = {delta_U_max_file_name_no_extension}::make();\n\n"
        if delta_U_max is not None and np.linalg.norm(delta_U_max_active_set) > TOL:
            for i in range(len(delta_U_max)):
                if delta_U_max_active_set[i]:
                    code_text += f"  delta_U_max.template set<{i}, 0>("
                    code_text += f"static_cast<{type_name}>({delta_U_max_values[i, 0]})"
                    code_text += ");\n"
            code_text += "\n"

        code_text += f"  auto U_min = {U_min_file_name_no_extension}::make();\n\n"
        if U_min is not None and np.linalg.norm(U_min_active_set) > TOL:
            for i in range(len(U_min)):
                if U_min_active_set[i]:
                    code_text += f"  U_min.template set<{i}, 0>("
                    code_text += f"static_cast<{type_name}>({U_min_values[i, 0]})"
                    code_text += ");\n"
            code_text += "\n"

        code_text += f"  auto U_max = {U_max_file_name_no_extension}::make();\n\n"
        if U_max is not None and np.linalg.norm(U_max_active_set) > TOL:
            for i in range(len(U_max)):
                if U_max_active_set[i]:
                    code_text += f"  U_max.template set<{i}, 0>("
                    code_text += f"static_cast<{type_name}>({U_max_values[i, 0]})"
                    code_text += ");\n"
            code_text += "\n"

        code_text += f"  auto Y_min = {Y_min_file_name_no_extension}::make();\n\n"
        if Y_min is not None and np.linalg.norm(Y_min_active_set) > TOL:
            for i in range(len(Y_min)):
                if Y_min_active_set[i]:
                    code_text += f"  Y_min.template set<{i}, 0>("
                    code_text += f"static_cast<{type_name}>({Y_min_values[i, 0]})"
                    code_text += ");\n"
            code_text += "\n"

        code_text += f"  auto Y_max = {Y_max_file_name_no_extension}::make();\n\n"
        if Y_max is not None and np.linalg.norm(Y_max_active_set) > TOL:
            for i in range(len(Y_max)):
                if Y_max_active_set[i]:
                    code_text += f"  Y_max.template set<{i}, 0>("
                    code_text += f"static_cast<{type_name}>({Y_max_values[i, 0]})"
                    code_text += ");\n"
            code_text += "\n"

        # prediction matrices
        code_text += f"  PredictionMatrices_Type prediction_matrices(F, Phi);\n\n"

        code_text += f"  ReferenceTrajectory_Type reference_trajectory;\n\n"

        code_text += f"  auto lti_mpc = make_LTI_MPC(\n" + \
            "    kalman_filter, prediction_matrices, reference_trajectory, Weight_U_Nc,\n" + \
            "    delta_U_min, delta_U_max, U_min, U_max, Y_min, Y_max, solver_factor);\n\n"

        code_text += "  return lti_mpc;\n\n"

        code_text += "}\n\n"

        code_text += "} // namespace " + namespace_name + "\n\n"

        code_text += "#endif // " + file_header_macro_name + "\n"

        code_file_name_ext = ControlDeploy.write_to_file(
            code_text, code_file_name_ext)

        deployed_file_names.append(code_file_name_ext)

        return deployed_file_names

    @staticmethod
    def generate_LTV_MPC_NC_cpp_code(ltv_mpc_nc: LTV_MPC_NoConstraints,
                                     file_name=None):
        parameters = ltv_mpc_nc.parameters_struct
        number_of_delay = ltv_mpc_nc.Number_of_Delay

        deployed_file_names = []

        ControlDeploy.restrict_data_type(ltv_mpc_nc.kalman_filter.A.dtype.name)

        type_name = NumpyDeploy.check_dtype(ltv_mpc_nc.kalman_filter.A)

        # %% inspect arguments
        # Get the caller's frame
        frame = inspect.currentframe().f_back
        # Get the caller's local variables
        caller_locals = frame.f_locals
        # Find the variable name that matches the matrix_in value
        variable_name = None
        for name, value in caller_locals.items():
            if value is ltv_mpc_nc:
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

        # %% generate parameter class code
        parameter_code_file_name = caller_file_name_without_ext + "_parameters.hpp"
        parameter_code_file_name_no_extension = parameter_code_file_name.split(".")[
            0]

        parameter_code = LTVMatricesDeploy.generate_parameter_cpp_code(
            parameters, type_name, parameter_code_file_name_no_extension)

        parameter_code_file_name_ext = ControlDeploy.write_to_file(
            parameter_code, parameter_code_file_name)

        deployed_file_names.append(parameter_code_file_name_ext)

        # %% generate MPC State Space Updater code
        mpc_state_space_updater_python_name = \
            ltv_mpc_nc.state_space_initializer.mpc_state_space_updater_file_name
        mpc_state_space_updater_name_no_extension = \
            mpc_state_space_updater_python_name.split(".")[0]
        mpc_state_space_updater_cpp_name = mpc_state_space_updater_name_no_extension + ".hpp"

        mpc_state_space_updater_code = LTVMatricesDeploy.generate_mpc_state_space_updater_cpp_code(
            mpc_state_space_updater_python_name, mpc_state_space_updater_name_no_extension)

        mpc_state_space_updater_cpp_name_ext = ControlDeploy.write_to_file(
            mpc_state_space_updater_code, mpc_state_space_updater_cpp_name)

        deployed_file_names.append(mpc_state_space_updater_cpp_name_ext)

        # %% generate Embedded Integrator Updater code
        embedded_integrator_updater_file_name = \
            ltv_mpc_nc.state_space_initializer.embedded_integrator_updater_file_name
        embedded_integrator_updater_name_no_extension = \
            embedded_integrator_updater_file_name.split(".")[0]
        embedded_integrator_updater_cpp_name = embedded_integrator_updater_name_no_extension + ".hpp"

        # Embedded Integrator Updater is the same style as MPC State Space Updater
        embedded_integrator_updater_code = LTVMatricesDeploy.generate_mpc_state_space_updater_cpp_code(
            embedded_integrator_updater_file_name, embedded_integrator_updater_name_no_extension)

        embedded_integrator_updater_cpp_name_ext = ControlDeploy.write_to_file(
            embedded_integrator_updater_code, embedded_integrator_updater_cpp_name)

        deployed_file_names.append(embedded_integrator_updater_cpp_name_ext)

        # %% generate Prediction Matrices Phi F updater code
        prediction_matrices_updater_file_name = \
            ltv_mpc_nc.state_space_initializer.prediction_matrices_phi_f_updater_file_name
        prediction_matrices_updater_name_no_extension = \
            prediction_matrices_updater_file_name.split(".")[0]
        prediction_matrices_updater_cpp_name = prediction_matrices_updater_name_no_extension + ".hpp"

        prediction_matrices_updater_code = \
            LTVMatricesDeploy.generate_prediction_matrices_phi_f_updater_cpp_code(
                prediction_matrices_updater_file_name, prediction_matrices_updater_name_no_extension)

        prediction_matrices_updater_cpp_name_ext = ControlDeploy.write_to_file(
            prediction_matrices_updater_code, prediction_matrices_updater_cpp_name)

        deployed_file_names.append(prediction_matrices_updater_cpp_name_ext)

        # %% generate LTV MPC Phi F Updater code
        LTV_MPC_Phi_F_updater_file_name = \
            ltv_mpc_nc.state_space_initializer.LTV_MPC_Phi_F_updater_file_name
        LTV_MPC_Phi_F_updater_name_no_extension = \
            LTV_MPC_Phi_F_updater_file_name.split(".")[0]
        LTV_MPC_Phi_F_updater_cpp_name = LTV_MPC_Phi_F_updater_name_no_extension + ".hpp"

        LTV_MPC_Phi_F_updater_code = \
            LTVMatricesDeploy.generate_ltv_mpc_phi_f_updater_cpp_code(
                LTV_MPC_Phi_F_updater_file_name, LTV_MPC_Phi_F_updater_name_no_extension,
                embedded_integrator_updater_cpp_name,
                prediction_matrices_updater_cpp_name)

        LTV_MPC_Phi_F_updater_cpp_name_ext = ControlDeploy.write_to_file(
            LTV_MPC_Phi_F_updater_code, LTV_MPC_Phi_F_updater_cpp_name)

        deployed_file_names.append(LTV_MPC_Phi_F_updater_cpp_name_ext)

        # %% create LKF, F, Phi, solver_factor, Weight_U_Nc code
        exec(f"{variable_name}_lkf = ltv_mpc_nc.kalman_filter")
        lkf_file_names = eval(
            f"KalmanFilterDeploy.generate_LKF_cpp_code({variable_name}_lkf, caller_file_name_without_ext, number_of_delay={number_of_delay})")

        deployed_file_names.append(lkf_file_names)
        lkf_file_name = lkf_file_names[-1]

        lkf_file_name_no_extension = lkf_file_name.split(".")[0]

        exec(f"{variable_name}_F = ltv_mpc_nc.prediction_matrices.F_ndarray")
        F_file_name = eval(
            f"NumpyDeploy.generate_matrix_cpp_code({variable_name}_F, caller_file_name_without_ext)")

        deployed_file_names.append(F_file_name)
        F_file_name_no_extension = F_file_name.split(".")[0]

        exec(
            f"{variable_name}_Phi = ltv_mpc_nc.prediction_matrices.Phi_ndarray")
        Phi_file_name = eval(
            f"NumpyDeploy.generate_matrix_cpp_code({variable_name}_Phi, caller_file_name_without_ext)")

        deployed_file_names.append(Phi_file_name)
        Phi_file_name_no_extension = Phi_file_name.split(".")[0]

        exec(f"{variable_name}_solver_factor = ltv_mpc_nc.solver_factor")
        solver_factor_file_name = eval(
            f"NumpyDeploy.generate_matrix_cpp_code({variable_name}_solver_factor, caller_file_name_without_ext)")

        deployed_file_names.append(solver_factor_file_name)
        solver_factor_file_name_no_extension = solver_factor_file_name.split(".")[
            0]

        exec(f"{variable_name}_Weight_U_Nc = ltv_mpc_nc.Weight_U_Nc")
        Weight_U_Nc_file_name = eval(
            f"NumpyDeploy.generate_matrix_cpp_code({variable_name}_Weight_U_Nc, caller_file_name_without_ext)")

        deployed_file_names.append(Weight_U_Nc_file_name)
        Weight_U_Nc_file_name_no_extension = Weight_U_Nc_file_name.split(".")[
            0]

        # %% main code generation
        code_text = ""

        file_header_macro_name = "__" + code_file_name.upper() + "_HPP__"

        code_text += "#ifndef " + file_header_macro_name + "\n"
        code_text += "#define " + file_header_macro_name + "\n\n"

        code_text += f"#include \"{lkf_file_name}\"\n"
        code_text += f"#include \"{F_file_name}\"\n"
        code_text += f"#include \"{Phi_file_name}\"\n"
        code_text += f"#include \"{solver_factor_file_name}\"\n"
        code_text += f"#include \"{Weight_U_Nc_file_name}\"\n"
        code_text += f"#include \"{caller_file_name_without_ext}_parameters.hpp\"\n"
        code_text += f"#include \"{mpc_state_space_updater_cpp_name}\"\n"
        code_text += f"#include \"{LTV_MPC_Phi_F_updater_cpp_name}\"\n\n"

        code_text += "#include \"python_mpc.hpp\"\n\n"

        namespace_name = code_file_name

        code_text += "namespace " + namespace_name + " {\n\n"

        code_text += "using namespace PythonNumpy;\n"
        code_text += "using namespace PythonControl;\n"
        code_text += "using namespace PythonMPC;\n\n"

        code_text += f"constexpr std::size_t NP = {ltv_mpc_nc.Np};\n"
        code_text += f"constexpr std::size_t NC = {ltv_mpc_nc.Nc};\n\n"

        code_text += f"constexpr std::size_t INPUT_SIZE = {lkf_file_name_no_extension}::INPUT_SIZE;\n"
        code_text += f"constexpr std::size_t STATE_SIZE = {lkf_file_name_no_extension}::STATE_SIZE;\n"
        code_text += f"constexpr std::size_t OUTPUT_SIZE = {lkf_file_name_no_extension}::OUTPUT_SIZE;\n\n"

        code_text += f"constexpr std::size_t AUGMENTED_STATE_SIZE = STATE_SIZE + OUTPUT_SIZE;\n\n"

        code_text += f"constexpr std::size_t NUMBER_OF_DELAY = {lkf_file_name_no_extension}::NUMBER_OF_DELAY;\n\n"

        code_text += f"using LKF_Type = {lkf_file_name_no_extension}::type;\n\n"

        code_text += f"using A_Type = typename LKF_Type::DiscreteStateSpace_Type::A_Type;\n\n"

        code_text += f"using B_Type = typename LKF_Type::DiscreteStateSpace_Type::B_Type;\n\n"

        code_text += f"using C_Type = typename LKF_Type::DiscreteStateSpace_Type::C_Type;\n\n"

        code_text += f"using F_Type = {F_file_name_no_extension}::type;\n\n"

        code_text += f"using Phi_Type = {Phi_file_name_no_extension}::type;\n\n"

        code_text += f"using SolverFactor_Type = {solver_factor_file_name_no_extension}::type;\n\n"

        code_text += f"using PredictionMatrices_Type = MPC_PredictionMatrices_Type<\n" + \
            "  F_Type, Phi_Type, NP, NC, INPUT_SIZE, AUGMENTED_STATE_SIZE, OUTPUT_SIZE>;\n\n"

        ref_row_size_text = "1"
        if ltv_mpc_nc.is_ref_trajectory:
            ref_row_size_text = "NP"

        code_text += f"using Ref_Type = DenseMatrix_Type<{type_name}, OUTPUT_SIZE, " + \
            ref_row_size_text + ">;\n\n"

        code_text += f"using ReferenceTrajectory_Type = MPC_ReferenceTrajectory_Type<\n" + \
            "  Ref_Type, NP>;\n\n"

        code_text += f"using Parameter_Type = {parameter_code_file_name_no_extension}::Parameter;\n\n"

        code_text += f"using Weight_U_Nc_Type = {Weight_U_Nc_file_name_no_extension}::type;\n\n"

        code_text += f"using EmbeddedIntegratorSateSpace_Type =\n" + \
            f"  typename EmbeddedIntegratorTypes<A_Type, B_Type, C_Type>::StateSpace_Type;\n\n"

        code_text += f"using type = LTV_MPC_NoConstraints_Type<\n" + \
            "  LKF_Type, PredictionMatrices_Type, ReferenceTrajectory_Type,\n" + \
            "  Parameter_Type, SolverFactor_Type>;\n\n"

        code_text += "inline auto make() -> type {\n\n"

        code_text += f"  auto kalman_filter = {lkf_file_name_no_extension}::make();\n\n"

        code_text += f"  auto F = {F_file_name_no_extension}::make();\n\n"

        code_text += f"  auto Phi = {Phi_file_name_no_extension}::make();\n\n"

        code_text += f"  auto solver_factor = {solver_factor_file_name_no_extension}::make();\n\n"

        code_text += f"  PredictionMatrices_Type prediction_matrices(F, Phi);\n\n"

        code_text += f"  ReferenceTrajectory_Type reference_trajectory;\n\n"

        code_text += f"  Weight_U_Nc_Type Weight_U_Nc = {Weight_U_Nc_file_name_no_extension}::make();\n\n"

        mpc_state_space_updater_name = caller_file_name_without_ext + \
            "_mpc_state_space_updater"

        code_text += f"  MPC_StateSpace_Updater_Function_Object<\n" + \
            f"    Parameter_Type, typename LKF_Type::DiscreteStateSpace_Type>\n" + \
            f"    MPC_StateSpace_Updater_Function =\n" + \
            f"    {mpc_state_space_updater_name}::MPC_StateSpace_Updater::update<\n" + \
            f"      Parameter_Type, typename LKF_Type::DiscreteStateSpace_Type>;\n\n"

        ltv_mpc_phi_f_updater_name = caller_file_name_without_ext + "_ltv_mpc_phi_f_updater"

        code_text += f"  LTV_MPC_Phi_F_Updater_Function_Object<\n" + \
            f"    EmbeddedIntegratorSateSpace_Type, Parameter_Type, Phi_Type, F_Type>\n" + \
            f"    LTV_MPC_Phi_F_Updater_Function =\n" + \
            f"    {ltv_mpc_phi_f_updater_name}::LTV_MPC_Phi_F_Updater::update<\n" + \
            f"      EmbeddedIntegratorSateSpace_Type, Parameter_Type, Phi_Type, F_Type>;\n\n"

        code_text += f"  auto ltv_mpc_nc = make_LTV_MPC_NoConstraints(\n" + \
            "    kalman_filter, prediction_matrices, reference_trajectory, solver_factor,\n" + \
            "    Weight_U_Nc, MPC_StateSpace_Updater_Function,\n" + \
            "    LTV_MPC_Phi_F_Updater_Function);\n\n"

        code_text += "  return ltv_mpc_nc;\n\n"

        code_text += "}\n\n"

        code_text += "} // namespace " + namespace_name + "\n\n"

        code_text += "#endif // " + file_header_macro_name + "\n"

        code_file_name_ext = ControlDeploy.write_to_file(
            code_text, code_file_name_ext)

        deployed_file_names.append(code_file_name_ext)

        return deployed_file_names

    @staticmethod
    def generate_LTV_MPC_cpp_code(ltv_mpc: LTV_MPC_NoConstraints,
                                  file_name=None):
        parameters = ltv_mpc.parameters_struct
        number_of_delay = ltv_mpc.kalman_filter.Number_of_Delay

        deployed_file_names = []

        ControlDeploy.restrict_data_type(ltv_mpc.kalman_filter.A.dtype.name)

        type_name = NumpyDeploy.check_dtype(ltv_mpc.kalman_filter.A)

        # %% inspect arguments
        # Get the caller's frame
        frame = inspect.currentframe().f_back
        # Get the caller's local variables
        caller_locals = frame.f_locals
        # Find the variable name that matches the matrix_in value
        variable_name = None
        for name, value in caller_locals.items():
            if value is ltv_mpc:
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

        # %% generate parameter class code
        parameter_code_file_name = caller_file_name_without_ext + "_parameters.hpp"
        parameter_code_file_name_no_extension = parameter_code_file_name.split(".")[
            0]

        parameter_code = LTVMatricesDeploy.generate_parameter_cpp_code(
            parameters, type_name, parameter_code_file_name_no_extension)

        parameter_code_file_name_ext = ControlDeploy.write_to_file(
            parameter_code, parameter_code_file_name)

        deployed_file_names.append(parameter_code_file_name_ext)

        # %% generate MPC State Space Updater code
        mpc_state_space_updater_python_name = \
            ltv_mpc.state_space_initializer.mpc_state_space_updater_file_name
        mpc_state_space_updater_name_no_extension = \
            mpc_state_space_updater_python_name.split(".")[0]
        mpc_state_space_updater_cpp_name = mpc_state_space_updater_name_no_extension + ".hpp"

        mpc_state_space_updater_code = LTVMatricesDeploy.generate_mpc_state_space_updater_cpp_code(
            mpc_state_space_updater_python_name, mpc_state_space_updater_name_no_extension)

        mpc_state_space_updater_cpp_name_ext = ControlDeploy.write_to_file(
            mpc_state_space_updater_code, mpc_state_space_updater_cpp_name)

        deployed_file_names.append(mpc_state_space_updater_cpp_name_ext)

        # %% generate Embedded Integrator Updater code
        embedded_integrator_updater_file_name = \
            ltv_mpc.state_space_initializer.embedded_integrator_updater_file_name
        embedded_integrator_updater_name_no_extension = \
            embedded_integrator_updater_file_name.split(".")[0]
        embedded_integrator_updater_cpp_name = embedded_integrator_updater_name_no_extension + ".hpp"

        # Embedded Integrator Updater is the same style as MPC State Space Updater
        embedded_integrator_updater_code = LTVMatricesDeploy.generate_mpc_state_space_updater_cpp_code(
            embedded_integrator_updater_file_name, embedded_integrator_updater_name_no_extension)

        embedded_integrator_updater_cpp_name_ext = ControlDeploy.write_to_file(
            embedded_integrator_updater_code, embedded_integrator_updater_cpp_name)

        deployed_file_names.append(embedded_integrator_updater_cpp_name_ext)

        # %% generate Prediction Matrices Phi F updater code
        prediction_matrices_updater_file_name = \
            ltv_mpc.state_space_initializer.prediction_matrices_phi_f_updater_file_name
        prediction_matrices_updater_name_no_extension = \
            prediction_matrices_updater_file_name.split(".")[0]
        prediction_matrices_updater_cpp_name = prediction_matrices_updater_name_no_extension + ".hpp"

        prediction_matrices_updater_code = \
            LTVMatricesDeploy.generate_prediction_matrices_phi_f_updater_cpp_code(
                prediction_matrices_updater_file_name, prediction_matrices_updater_name_no_extension)

        prediction_matrices_updater_cpp_name_ext = ControlDeploy.write_to_file(
            prediction_matrices_updater_code, prediction_matrices_updater_cpp_name)

        deployed_file_names.append(prediction_matrices_updater_cpp_name_ext)

        # %% generate LTV MPC Phi F Updater code
        LTV_MPC_Phi_F_updater_file_name = \
            ltv_mpc.state_space_initializer.LTV_MPC_Phi_F_updater_file_name
        LTV_MPC_Phi_F_updater_name_no_extension = \
            LTV_MPC_Phi_F_updater_file_name.split(".")[0]
        LTV_MPC_Phi_F_updater_cpp_name = LTV_MPC_Phi_F_updater_name_no_extension + ".hpp"

        LTV_MPC_Phi_F_updater_code = \
            LTVMatricesDeploy.generate_ltv_mpc_phi_f_updater_cpp_code(
                LTV_MPC_Phi_F_updater_file_name, LTV_MPC_Phi_F_updater_name_no_extension,
                embedded_integrator_updater_cpp_name,
                prediction_matrices_updater_cpp_name)

        LTV_MPC_Phi_F_updater_cpp_name_ext = ControlDeploy.write_to_file(
            LTV_MPC_Phi_F_updater_code, LTV_MPC_Phi_F_updater_cpp_name)

        deployed_file_names.append(LTV_MPC_Phi_F_updater_cpp_name_ext)

        # %% create LKF, F, Phi, solver_factor, Weight_U_Nc code
        exec(f"{variable_name}_lkf = ltv_mpc.kalman_filter")
        lkf_file_names = eval(
            f"KalmanFilterDeploy.generate_LKF_cpp_code({variable_name}_lkf, caller_file_name_without_ext, number_of_delay={number_of_delay})")

        deployed_file_names.append(lkf_file_names)
        lkf_file_name = lkf_file_names[-1]

        lkf_file_name_no_extension = lkf_file_name.split(".")[0]

        exec(f"{variable_name}_F = ltv_mpc.prediction_matrices.F_ndarray")
        F_file_name = eval(
            f"NumpyDeploy.generate_matrix_cpp_code({variable_name}_F, caller_file_name_without_ext)")

        deployed_file_names.append(F_file_name)
        F_file_name_no_extension = F_file_name.split(".")[0]

        exec(
            f"{variable_name}_Phi = ltv_mpc.prediction_matrices.Phi_ndarray")
        Phi_file_name = eval(
            f"NumpyDeploy.generate_matrix_cpp_code({variable_name}_Phi, caller_file_name_without_ext)")

        deployed_file_names.append(Phi_file_name)
        Phi_file_name_no_extension = Phi_file_name.split(".")[0]

        exec(f"{variable_name}_solver_factor = ltv_mpc.solver_factor")
        solver_factor_file_name = eval(
            f"NumpyDeploy.generate_matrix_cpp_code({variable_name}_solver_factor, caller_file_name_without_ext)")

        deployed_file_names.append(solver_factor_file_name)
        solver_factor_file_name_no_extension = solver_factor_file_name.split(".")[
            0]

        exec(f"{variable_name}_Weight_U_Nc = ltv_mpc.Weight_U_Nc")
        Weight_U_Nc_file_name = eval(
            f"NumpyDeploy.generate_matrix_cpp_code({variable_name}_Weight_U_Nc, caller_file_name_without_ext)")

        deployed_file_names.append(Weight_U_Nc_file_name)
        Weight_U_Nc_file_name_no_extension = Weight_U_Nc_file_name.split(".")[
            0]

        # %% create limits code
        delta_U_min_values = ltv_mpc.qp_solver.DU_U_Y_Limits.delta_U_min
        if delta_U_min_values is not None:
            delta_U_min_active_set = np.zeros(
                np.size(delta_U_min_values), dtype=bool)
            for i in range(len(delta_U_min_values)):
                if ltv_mpc.qp_solver.DU_U_Y_Limits.is_delta_U_min_active(i):
                    delta_U_min_active_set[i] = True

        delta_U_max_values = ltv_mpc.qp_solver.DU_U_Y_Limits.delta_U_max
        if delta_U_max_values is not None:
            delta_U_max_active_set = np.zeros(
                np.size(delta_U_max_values), dtype=bool)
            for i in range(len(delta_U_max_values)):
                if ltv_mpc.qp_solver.DU_U_Y_Limits.is_delta_U_max_active(i):
                    delta_U_max_active_set[i] = True

        U_min_values = ltv_mpc.qp_solver.DU_U_Y_Limits.U_min
        if U_min_values is not None:
            U_min_active_set = np.zeros(np.size(U_min_values), dtype=bool)
            for i in range(len(U_min_values)):
                if ltv_mpc.qp_solver.DU_U_Y_Limits.is_U_min_active(i):
                    U_min_active_set[i] = True

        U_max_values = ltv_mpc.qp_solver.DU_U_Y_Limits.U_max
        if U_max_values is not None:
            U_max_active_set = np.zeros(np.size(U_max_values), dtype=bool)
            for i in range(len(U_max_values)):
                if ltv_mpc.qp_solver.DU_U_Y_Limits.is_U_max_active(i):
                    U_max_active_set[i] = True

        Y_min_values = ltv_mpc.qp_solver.DU_U_Y_Limits.Y_min
        if Y_min_values is not None:
            Y_min_active_set = np.zeros(np.size(Y_min_values), dtype=bool)
            for i in range(len(Y_min_values)):
                if ltv_mpc.qp_solver.DU_U_Y_Limits.is_Y_min_active(i):
                    Y_min_active_set[i] = True

        Y_max_values = ltv_mpc.qp_solver.DU_U_Y_Limits.Y_max
        if Y_max_values is not None:
            Y_max_active_set = np.zeros(np.size(Y_max_values), dtype=bool)
            for i in range(len(Y_max_values)):
                if ltv_mpc.qp_solver.DU_U_Y_Limits.is_Y_max_active(i):
                    Y_max_active_set[i] = True

        # Limits code
        delta_U_min = copy.deepcopy(delta_U_min_active_set)
        delta_U_min = np.array(
            delta_U_min, dtype=delta_U_max_values.dtype).reshape(-1, 1)
        exec(f"{variable_name}_delta_U_min = delta_U_min")
        delta_U_min_file_name = eval(
            f"NumpyDeploy.generate_matrix_cpp_code({variable_name}_delta_U_min, caller_file_name_without_ext)")

        deployed_file_names.append(delta_U_min_file_name)
        delta_U_min_file_name_no_extension = delta_U_min_file_name .split(".")[
            0]

        delta_U_max = copy.deepcopy(delta_U_max_active_set)
        delta_U_max = np.array(
            delta_U_max, dtype=delta_U_max_values.dtype).reshape(-1, 1)
        exec(f"{variable_name}_delta_U_max = delta_U_max")
        delta_U_max_file_name = eval(
            f"NumpyDeploy.generate_matrix_cpp_code({variable_name}_delta_U_max, caller_file_name_without_ext)")

        deployed_file_names.append(delta_U_max_file_name)
        delta_U_max_file_name_no_extension = delta_U_max_file_name .split(".")[
            0]

        U_min = copy.deepcopy(U_min_active_set)
        U_min = np.array(U_min, dtype=U_min_values.dtype).reshape(-1, 1)
        exec(f"{variable_name}_U_min = U_min")
        U_min_file_name = eval(
            f"NumpyDeploy.generate_matrix_cpp_code({variable_name}_U_min, caller_file_name_without_ext)")

        deployed_file_names.append(U_min_file_name)
        U_min_file_name_no_extension = U_min_file_name .split(".")[0]

        U_max = copy.deepcopy(U_max_active_set)
        U_max = np.array(U_max, dtype=U_max_values.dtype).reshape(-1, 1)
        exec(f"{variable_name}_U_max = U_max")
        U_max_file_name = eval(
            f"NumpyDeploy.generate_matrix_cpp_code({variable_name}_U_max, caller_file_name_without_ext)")

        deployed_file_names.append(U_max_file_name)
        U_max_file_name_no_extension = U_max_file_name .split(".")[0]

        Y_min = copy.deepcopy(Y_min_active_set)
        Y_min = np.array(Y_min, dtype=Y_min_values.dtype).reshape(-1, 1)
        exec(f"{variable_name}_Y_min = Y_min")
        Y_min_file_name = eval(
            f"NumpyDeploy.generate_matrix_cpp_code({variable_name}_Y_min, caller_file_name_without_ext)")

        deployed_file_names.append(Y_min_file_name)
        Y_min_file_name_no_extension = Y_min_file_name .split(".")[0]

        Y_max = copy.deepcopy(Y_max_active_set)
        Y_max = np.array(Y_max, dtype=Y_max_values.dtype).reshape(-1, 1)
        exec(f"{variable_name}_Y_max = Y_max")
        Y_max_file_name = eval(
            f"NumpyDeploy.generate_matrix_cpp_code({variable_name}_Y_max, caller_file_name_without_ext)")

        deployed_file_names.append(Y_max_file_name)
        Y_max_file_name_no_extension = Y_max_file_name .split(".")[0]

        # %% main code generation
        code_text = ""

        file_header_macro_name = "__" + code_file_name.upper() + "_HPP__"

        code_text += "#ifndef " + file_header_macro_name + "\n"
        code_text += "#define " + file_header_macro_name + "\n\n"

        code_text += f"#include \"{lkf_file_name}\"\n"
        code_text += f"#include \"{F_file_name}\"\n"
        code_text += f"#include \"{Phi_file_name}\"\n"
        code_text += f"#include \"{solver_factor_file_name}\"\n"
        code_text += f"#include \"{Weight_U_Nc_file_name}\"\n"
        code_text += f"#include \"{caller_file_name_without_ext}_parameters.hpp\"\n"
        code_text += f"#include \"{mpc_state_space_updater_cpp_name}\"\n"
        code_text += f"#include \"{LTV_MPC_Phi_F_updater_cpp_name}\"\n\n"

        code_text += f"#include \"{delta_U_min_file_name}\"\n"
        code_text += f"#include \"{delta_U_max_file_name}\"\n"
        code_text += f"#include \"{U_min_file_name}\"\n"
        code_text += f"#include \"{U_max_file_name}\"\n"
        code_text += f"#include \"{Y_min_file_name}\"\n"
        code_text += f"#include \"{Y_max_file_name}\"\n\n"

        code_text += "#include \"python_mpc.hpp\"\n\n"

        namespace_name = code_file_name

        code_text += "namespace " + namespace_name + " {\n\n"

        code_text += "using namespace PythonNumpy;\n"
        code_text += "using namespace PythonControl;\n"
        code_text += "using namespace PythonMPC;\n\n"

        code_text += f"constexpr std::size_t NP = {ltv_mpc.Np};\n"
        code_text += f"constexpr std::size_t NC = {ltv_mpc.Nc};\n\n"

        code_text += f"constexpr std::size_t INPUT_SIZE = {lkf_file_name_no_extension}::INPUT_SIZE;\n"
        code_text += f"constexpr std::size_t STATE_SIZE = {lkf_file_name_no_extension}::STATE_SIZE;\n"
        code_text += f"constexpr std::size_t OUTPUT_SIZE = {lkf_file_name_no_extension}::OUTPUT_SIZE;\n\n"

        code_text += f"constexpr std::size_t AUGMENTED_STATE_SIZE = STATE_SIZE + OUTPUT_SIZE;\n\n"

        code_text += f"constexpr std::size_t NUMBER_OF_DELAY = {lkf_file_name_no_extension}::NUMBER_OF_DELAY;\n\n"

        code_text += f"using LKF_Type = {lkf_file_name_no_extension}::type;\n\n"

        code_text += f"using A_Type = typename LKF_Type::DiscreteStateSpace_Type::A_Type;\n\n"

        code_text += f"using B_Type = typename LKF_Type::DiscreteStateSpace_Type::B_Type;\n\n"

        code_text += f"using C_Type = typename LKF_Type::DiscreteStateSpace_Type::C_Type;\n\n"

        code_text += f"using F_Type = {F_file_name_no_extension}::type;\n\n"

        code_text += f"using Phi_Type = {Phi_file_name_no_extension}::type;\n\n"

        code_text += f"using SolverFactor_Type = {solver_factor_file_name_no_extension}::type;\n\n"

        code_text += f"using Delta_U_Min_Type = {delta_U_min_file_name_no_extension}::type;\n\n"

        code_text += f"using Delta_U_Max_Type = {delta_U_max_file_name_no_extension}::type;\n\n"

        code_text += f"using U_Min_Type = {U_min_file_name_no_extension}::type;\n\n"

        code_text += f"using U_Max_Type = {U_max_file_name_no_extension}::type;\n\n"

        code_text += f"using Y_Min_Type = {Y_min_file_name_no_extension}::type;\n\n"

        code_text += f"using Y_Max_Type = {Y_max_file_name_no_extension}::type;\n\n"

        code_text += f"using PredictionMatrices_Type = MPC_PredictionMatrices_Type<\n" + \
            "  F_Type, Phi_Type, NP, NC, INPUT_SIZE, AUGMENTED_STATE_SIZE, OUTPUT_SIZE>;\n\n"

        ref_row_size_text = "1"
        if ltv_mpc.is_ref_trajectory:
            ref_row_size_text = "NP"

        code_text += f"using Ref_Type = DenseMatrix_Type<{type_name}, OUTPUT_SIZE, " + \
            ref_row_size_text + ">;\n\n"

        code_text += f"using ReferenceTrajectory_Type = MPC_ReferenceTrajectory_Type<\n" + \
            "  Ref_Type, NP>;\n\n"

        code_text += f"using Parameter_Type = {parameter_code_file_name_no_extension}::Parameter;\n\n"

        code_text += f"using Weight_U_Nc_Type = {Weight_U_Nc_file_name_no_extension}::type;\n\n"

        code_text += f"using EmbeddedIntegratorSateSpace_Type =\n" + \
            f"  typename EmbeddedIntegratorTypes<A_Type, B_Type, C_Type>::StateSpace_Type;\n\n"

        code_text += f"using type = LTV_MPC_Type<\n" + \
            "  LKF_Type, PredictionMatrices_Type, ReferenceTrajectory_Type,\n" + \
            "  Parameter_Type," + \
            "  Delta_U_Min_Type, Delta_U_Max_Type,\n" + \
            "  U_Min_Type, U_Max_Type, Y_Min_Type, Y_Max_Type,\n" + \
            "  SolverFactor_Type>;\n\n"

        code_text += "inline auto make() -> type {\n\n"

        code_text += f"  auto kalman_filter = {lkf_file_name_no_extension}::make();\n\n"

        code_text += f"  auto F = {F_file_name_no_extension}::make();\n\n"

        code_text += f"  auto Phi = {Phi_file_name_no_extension}::make();\n\n"

        code_text += f"  auto solver_factor = {solver_factor_file_name_no_extension}::make();\n\n"

        code_text += f"  auto Weight_U_Nc = {Weight_U_Nc_file_name_no_extension}::make();\n\n"

        # limits
        code_text += f"  auto delta_U_min = {delta_U_min_file_name_no_extension}::make();\n\n"
        if delta_U_min is not None and np.linalg.norm(delta_U_min_active_set) > TOL:
            for i in range(len(delta_U_min)):
                if delta_U_min_active_set[i]:
                    code_text += f"  delta_U_min.template set<{i}, 0>("
                    code_text += f"static_cast<{type_name}>({delta_U_min_values[i, 0]})"
                    code_text += ");\n"
            code_text += "\n"

        code_text += f"  auto delta_U_max = {delta_U_max_file_name_no_extension}::make();\n\n"
        if delta_U_max is not None and np.linalg.norm(delta_U_max_active_set) > TOL:
            for i in range(len(delta_U_max)):
                if delta_U_max_active_set[i]:
                    code_text += f"  delta_U_max.template set<{i}, 0>("
                    code_text += f"static_cast<{type_name}>({delta_U_max_values[i, 0]})"
                    code_text += ");\n"
            code_text += "\n"

        code_text += f"  auto U_min = {U_min_file_name_no_extension}::make();\n\n"
        if U_min is not None and np.linalg.norm(U_min_active_set) > TOL:
            for i in range(len(U_min)):
                if U_min_active_set[i]:
                    code_text += f"  U_min.template set<{i}, 0>("
                    code_text += f"static_cast<{type_name}>({U_min_values[i, 0]})"
                    code_text += ");\n"
            code_text += "\n"

        code_text += f"  auto U_max = {U_max_file_name_no_extension}::make();\n\n"
        if U_max is not None and np.linalg.norm(U_max_active_set) > TOL:
            for i in range(len(U_max)):
                if U_max_active_set[i]:
                    code_text += f"  U_max.template set<{i}, 0>("
                    code_text += f"static_cast<{type_name}>({U_max_values[i, 0]})"
                    code_text += ");\n"
            code_text += "\n"

        code_text += f"  auto Y_min = {Y_min_file_name_no_extension}::make();\n\n"
        if Y_min is not None and np.linalg.norm(Y_min_active_set) > TOL:
            for i in range(len(Y_min)):
                if Y_min_active_set[i]:
                    code_text += f"  Y_min.template set<{i}, 0>("
                    code_text += f"static_cast<{type_name}>({Y_min_values[i, 0]})"
                    code_text += ");\n"
            code_text += "\n"

        code_text += f"  auto Y_max = {Y_max_file_name_no_extension}::make();\n\n"
        if Y_max is not None and np.linalg.norm(Y_max_active_set) > TOL:
            for i in range(len(Y_max)):
                if Y_max_active_set[i]:
                    code_text += f"  Y_max.template set<{i}, 0>("
                    code_text += f"static_cast<{type_name}>({Y_max_values[i, 0]})"
                    code_text += ");\n"
            code_text += "\n"

        # prediction matrices
        code_text += f"  PredictionMatrices_Type prediction_matrices(F, Phi);\n\n"

        code_text += f"  ReferenceTrajectory_Type reference_trajectory;\n\n"

        mpc_state_space_updater_name = caller_file_name_without_ext + \
            "_mpc_state_space_updater"

        code_text += f"  MPC_StateSpace_Updater_Function_Object<\n" + \
            f"    Parameter_Type, typename LKF_Type::DiscreteStateSpace_Type>\n" + \
            f"    MPC_StateSpace_Updater_Function =\n" + \
            f"    {mpc_state_space_updater_name}::MPC_StateSpace_Updater::update<\n" + \
            f"      Parameter_Type, typename LKF_Type::DiscreteStateSpace_Type>;\n\n"

        ltv_mpc_phi_f_updater_name = caller_file_name_without_ext + "_ltv_mpc_phi_f_updater"

        code_text += f"  LTV_MPC_Phi_F_Updater_Function_Object<\n" + \
            f"    EmbeddedIntegratorSateSpace_Type, Parameter_Type, Phi_Type, F_Type>\n" + \
            f"    LTV_MPC_Phi_F_Updater_Function =\n" + \
            f"    {ltv_mpc_phi_f_updater_name}::LTV_MPC_Phi_F_Updater::update<\n" + \
            f"      EmbeddedIntegratorSateSpace_Type, Parameter_Type, Phi_Type, F_Type>;\n\n"

        code_text += f"  auto ltv_mpc = make_LTV_MPC(\n" + \
            "    kalman_filter, prediction_matrices, reference_trajectory, Weight_U_Nc,\n" + \
            "    MPC_StateSpace_Updater_Function,\n" + \
            "    LTV_MPC_Phi_F_Updater_Function,\n" + \
            "    delta_U_min, delta_U_max, U_min, U_max, Y_min, Y_max, solver_factor);\n\n"

        code_text += "  return ltv_mpc;\n\n"

        code_text += "}\n\n"

        code_text += "} // namespace " + namespace_name + "\n\n"

        code_text += "#endif // " + file_header_macro_name + "\n"

        code_file_name_ext = ControlDeploy.write_to_file(
            code_text, code_file_name_ext)

        deployed_file_names.append(code_file_name_ext)

        return deployed_file_names
