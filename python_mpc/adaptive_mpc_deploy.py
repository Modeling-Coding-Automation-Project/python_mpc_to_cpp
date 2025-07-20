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

from external_libraries.MCAP_python_mpc.python_mpc.adaptive_mpc import AdaptiveMPC_NoConstraints

TOL = 1e-30


class AdaptiveMPC_Deploy:

    @staticmethod
    def generate_Adaptive_MPC_NC_cpp_code(
            ada_mpc_nc: AdaptiveMPC_NoConstraints,
            file_name=None):
        pass
