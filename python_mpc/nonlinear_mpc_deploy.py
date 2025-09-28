import os
import sys
sys.path.append(os.getcwd())
sys.path.append(os.path.join(
    os.getcwd(), "external_libraries", "python_control_to_cpp"))

import inspect

from external_libraries.python_numpy_to_cpp.python_numpy.numpy_deploy import NumpyDeploy
from external_libraries.MCAP_python_control.python_control.control_deploy import ControlDeploy
from external_libraries.python_control_to_cpp.python_control.kalman_filter_deploy import KalmanFilterDeploy

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

        pass
