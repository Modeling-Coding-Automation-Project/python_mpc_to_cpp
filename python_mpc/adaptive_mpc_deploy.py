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
from python_mpc.adaptive_matrices_deploy import AdaptiveMatricesDeploy

from external_libraries.MCAP_python_mpc.python_mpc.adaptive_mpc import AdaptiveMPC_NoConstraints

TOL = 1e-30


class AdaptiveMPC_Deploy:

    @staticmethod
    def generate_Adaptive_MPC_NC_cpp_code(
            ada_mpc_nc: AdaptiveMPC_NoConstraints,
            file_name=None):
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

        # %% generate parameter class code
        parameter_code_file_name = caller_file_name_without_ext + "_parameters.hpp"
        parameter_code_file_name_no_extension = parameter_code_file_name.split(".")[
            0]

        parameter_code = AdaptiveMatricesDeploy.generate_parameter_cpp_code(
            parameters, type_name, parameter_code_file_name_no_extension)

        parameter_code_file_name_ext = ControlDeploy.write_to_file(
            parameter_code, parameter_code_file_name)

        deployed_file_names.append(parameter_code_file_name_ext)
