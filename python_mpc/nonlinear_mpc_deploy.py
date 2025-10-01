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

from mpc_utility.adaptive_matrices_deploy import AdaptiveMatricesDeploy
from python_mpc.common_mpc_deploy import convert_SparseAvailable_for_deploy
from external_libraries.python_optimization_to_cpp.optimization_utility.common_optimization_deploy import MinMaxCodeGenerator

from external_libraries.MCAP_python_mpc.python_mpc.nonlinear_mpc import NonlinearMPC_TwiceDifferentiable


class NonlinearMPC_Deploy:

    @staticmethod
    def generate_Nonlinear_MPC_cpp_code(
            nonlinear_mpc: NonlinearMPC_TwiceDifferentiable,
            file_name=None):

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
        exec(f"{variable_name}_ekf = nonlinear_mpc.kalman_filter")
        ekf_file_names = eval(
            f"KalmanFilterDeploy.generate_EKF_cpp_code({variable_name}_ekf, caller_file_name_no_extension, number_of_delay={number_of_delay})")

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

        code_text += "inline auto make() -> type {\n\n"

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
