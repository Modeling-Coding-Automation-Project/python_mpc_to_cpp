import os
import sys
sys.path.append(os.getcwd())
sys.path.append(os.path.join(
    os.getcwd(), "external_libraries", "python_control_to_cpp"))

import inspect

from external_libraries.python_numpy_to_cpp.python_numpy.numpy_deploy import NumpyDeploy
from external_libraries.python_control_to_cpp.python_control.control_deploy import ControlDeploy

from external_libraries.python_control_to_cpp.python_control.kalman_filter_deploy import KalmanFilterDeploy

from python_mpc.linear_mpc import LTI_MPC_NoConstraints


class LinearMPC_Deploy:
    def __init__(self):
        pass

    @staticmethod
    def generate_LTI_MPC_NC_cpp_code(
            lti_mpc_nc: LTI_MPC_NoConstraints, file_name=None, number_of_delay=0):
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

        number_of_delay = lti_mpc_nc.Number_of_Delay

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
        exec(f"{variable_name}_F = lti_mpc_nc.prediction_matrices.F_numeric")
        F_file_name = eval(
            f"NumpyDeploy.generate_matrix_cpp_code({variable_name}_F, caller_file_name_without_ext)")

        deployed_file_names.append(F_file_name)
        F_file_name_no_extension = F_file_name.split(".")[0]

        # create Phi code
        exec(f"{variable_name}_Phi = lti_mpc_nc.prediction_matrices.Phi_numeric")
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

        code_text += f"using type = LTI_MPC_NoConstraints<\n" + \
            "  LKF_Type, PredictionMatrices_Type, ReferenceTrajectory_Type,\n" + \
            "  SolverFactor_Type>;\n\n"

        code_text += "auto make() -> type {\n\n"

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
