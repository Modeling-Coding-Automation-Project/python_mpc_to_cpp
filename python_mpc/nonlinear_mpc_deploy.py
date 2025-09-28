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

        pass
