import os
import sys
sys.path.append(os.getcwd())
sys.path.append(os.path.join(
    os.getcwd(), "external_libraries", "python_control_to_cpp"))

import inspect
import numpy as np
from dataclasses import is_dataclass

from external_libraries.python_numpy_to_cpp.python_numpy.numpy_deploy import NumpyDeploy
from external_libraries.MCAP_python_control.python_control.control_deploy import ControlDeploy

from external_libraries.python_control_to_cpp.python_control.kalman_filter_deploy import KalmanFilterDeploy


class LTVMatricesDeploy:

    @staticmethod
    def generate_parameter_cpp_code(parameter_object,
                                    value_type_name: str,
                                    file_name: str = None):
        if not is_dataclass(parameter_object):
            raise TypeError("parameter_object must be a dataclass instance.")

        code_text = ""

        if file_name is not None:
            code_text += f"#ifndef __{file_name.upper()}_HPP__\n"
            code_text += f"#define __{file_name.upper()}_HPP__\n\n"

            code_text += f"namespace {file_name} {{\n\n"

        code_text += KalmanFilterDeploy.generate_parameter_cpp_code(
            parameter_object, value_type_name)

        if file_name is not None:
            code_text += "\n} // namespace " + file_name + "\n\n"

            code_text += f"#endif // __{file_name.upper()}_HPP__\n"

        return code_text
