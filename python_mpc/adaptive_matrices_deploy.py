"""
File: adaptive_matrices_deploy.py

"""
import os
import sys
sys.path.append(os.getcwd())
sys.path.append(os.path.join(
    os.getcwd(), "external_libraries", "python_control_to_cpp"))

import ast
import astor
import os
from dataclasses import is_dataclass
import textwrap

from external_libraries.python_numpy_to_cpp.python_numpy.numpy_deploy import python_to_cpp_types
from external_libraries.MCAP_python_control.python_control.control_deploy import ControlDeploy
from external_libraries.MCAP_python_control.python_control.control_deploy import IntegerPowerReplacer
from external_libraries.python_control_to_cpp.python_control.kalman_filter_deploy import NpArrayExtractor
from external_libraries.python_control_to_cpp.python_control.kalman_filter_deploy import KalmanFilterDeploy

from python_mpc.ltv_matrices_deploy import extract_class_methods
from python_mpc.ltv_matrices_deploy import MatrixUpdaterToCppVisitor
from python_mpc.ltv_matrices_deploy import StateSpaceUpdaterToCppVisitor
from python_mpc.ltv_matrices_deploy import PredictionMatricesPhiF_UpdaterToCppVisitor


class AdaptiveMatricesDeploy:
    """
    LTVMatricesDeploy
    A utility class providing static methods to generate C++ code
    for Linear Time-Varying (LTV) Model Predictive Control (MPC) matrix operations
    from Python source files and dataclass parameter objects.

    Notes
    -----
    - These methods are intended for code generation and automation
      in control systems and MPC applications.
    - The generated C++ code uses templates and namespaces
      for type safety and modularity.
    - The class assumes the existence of several helper classes
      and functions (e.g., KalmanFilterDeploy, ControlDeploy,
        extract_class_methods, and various Visitor classes)
          for code conversion and file handling.
    """

    @staticmethod
    def generate_parameter_cpp_code(parameter_object,

                                    value_type_name: str,
                                    file_name_no_extension: str = None):
        """
        Generates C++ code for a given dataclass parameter object,
          optionally wrapping it in a namespace and header guards.
        Args:
            parameter_object: An instance of a dataclass containing
              the parameters to be converted into C++ code.
            value_type_name (str): The C++ type name to be used
              for the parameter values (e.g., 'double', 'float').
            file_name_no_extension (str, optional): If provided,
              the generated code will be wrapped in a namespace and header guards
                using this string as the base name.
        Returns:
            str: The generated C++ code as a string.
        Raises:
            TypeError: If `parameter_object` is not a dataclass instance.
        """

        if not is_dataclass(parameter_object):
            raise TypeError("parameter_object must be a dataclass instance.")

        code_text = ""

        if file_name_no_extension is not None:
            code_text += f"#ifndef __{file_name_no_extension.upper()}_HPP__\n"
            code_text += f"#define __{file_name_no_extension.upper()}_HPP__\n\n"

            code_text += f"namespace {file_name_no_extension} {{\n\n"

        code_text += KalmanFilterDeploy.generate_parameter_cpp_code(
            parameter_object, value_type_name)

        if file_name_no_extension is not None:
            code_text += "\n} // namespace " + file_name_no_extension + "\n\n"

            code_text += f"#endif // __{file_name_no_extension.upper()}_HPP__\n"

        return code_text

    @staticmethod
    def generate_mpc_state_space_updater_cpp_code(input_python_file_name: str,
                                                  file_name_no_extension: str):
        """
        Generates C++ header code for MPC (Model Predictive Control) state-space updater classes
        based on the structure and methods of Python classes defined in the specified input file.
        This function reads a Python file containing class definitions for MPC state-space updaters,
        extracts their methods, and converts them into corresponding C++ class templates and methods.
        The generated C++ code is returned as a string, formatted as a header file with include guards
        and a namespace based on the provided file name.
        Args:
            input_python_file_name (str): The name of the Python file containing the class definitions
                to be converted.
            file_name_no_extension (str): The base name (without extension) to use for the C++ header
                file, include guards, and namespace.
        Returns:
            str: The generated C++ header code as a string.
        Raises:
            ValueError: If no classes are found in the specified Python file.
        Notes:
            - The function expects the input Python file to contain specific class structures for
                MPC state-space updaters.
            - Uses helper classes and functions such as `ControlDeploy.find_file`, `extract_class_methods`,
                `MatrixUpdaterToCppVisitor`, and `StateSpaceUpdaterToCppVisitor`
                  for file handling and code conversion.
            - The last class in the file is treated as the main MPC StateSpace Updater class, while others
                are treated as matrix updater classes (A, B, C, D, etc.).
        """

        function_file_path = ControlDeploy.find_file(
            input_python_file_name, os.getcwd())

        class_methods = extract_class_methods(function_file_path)

        if not class_methods:
            raise ValueError(f"No classes found in {input_python_file_name}.")

        code_text = ""

        code_text += f"#ifndef __{file_name_no_extension.upper()}_HPP__\n"
        code_text += f"#define __{file_name_no_extension.upper()}_HPP__\n\n"

        code_text += f"namespace {file_name_no_extension} {{\n\n"

        for i, (class_name, methods) in enumerate(class_methods.items()):
            if i < len(class_methods) - 1:
                # A, B, C, D updater class
                output_type_name = f"{class_name}_Output_Type"

                code_text += f"template <typename " + output_type_name + ">\n"
                code_text += f"class {class_name} {{\npublic:\n"

                visitor = MatrixUpdaterToCppVisitor(output_type_name)

                for _, method_source in methods.items():
                    function_code = visitor.convert(
                        textwrap.dedent(method_source))
                code_text += function_code
                code_text += "\n"

            else:
                # MPC StateSpace Updater class
                output_type_name = f"{class_name}_Output_Type"

                code_text += f"class {class_name} {{\npublic:\n"
                code_text += f"template <typename Parameter_Type, typename " + output_type_name + ">\n"
                code_text += "static inline void update(const Parameter_Type& parameter, " + \
                    output_type_name + "& output) {\n"

                visitor = StateSpaceUpdaterToCppVisitor()
                visitor.output_type_name = output_type_name

                function_code = visitor.convert(
                    textwrap.dedent(methods['update']))
                code_text += function_code
                code_text += "}\n"

            code_text += "};\n\n"

        code_text += f"}} // namespace {file_name_no_extension}\n\n"

        code_text += f"#endif // __{file_name_no_extension.upper()}_HPP__\n"

        return code_text

    @staticmethod
    def generate_prediction_matrices_phi_f_updater_cpp_code(input_python_file_name: str,
                                                            file_name_no_extension: str):
        """
        Generates C++ header code for a class that updates prediction matrices Phi and F, 
        based on a Python class method named 'update' found in the specified input file.

        This function locates the Python file, extracts the first class and its methods, 
        and generates a C++ class with a static templated 'update' method. The body of 
        the 'update' method is converted from the Python implementation using a visitor 
        pattern. The resulting C++ code is wrapped in a namespace and header guards.

        Args:
            input_python_file_name (str): Path to the Python file containing
              the class with the 'update' method.
            file_name_no_extension (str): The base name (without extension)
              to use for the generated C++ header and namespace.

        Returns:
            str: The generated C++ header code as a string.

        Raises:
            ValueError: If no classes are found in the specified Python file.
        """

        function_file_path = ControlDeploy.find_file(
            input_python_file_name, os.getcwd())

        class_methods = extract_class_methods(function_file_path)
        items = class_methods.items()
        class_name = next(iter(items))[0] if items else None
        methods = next(iter(items))[1] if items else {}

        if not class_methods:
            raise ValueError(f"No classes found in {input_python_file_name}.")

        code_text = ""

        code_text += f"#ifndef __{file_name_no_extension.upper()}_HPP__\n"
        code_text += f"#define __{file_name_no_extension.upper()}_HPP__\n\n"

        code_text += f"namespace {file_name_no_extension} {{\n\n"

        code_text += f"class {class_name} {{\npublic:\n"
        code_text += f"template <typename A_Type, typename B_Type, typename C_Type,\n" + \
            f"          typename Phi_Type, typename F_Type >\n"
        code_text += "static inline void update(const A_Type &A, const B_Type &B,\n" + \
            f"          const C_Type &C, Phi_Type& Phi, F_Type& F) {{\n"

        visitor = PredictionMatricesPhiF_UpdaterToCppVisitor()
        function_code = visitor.convert(
            textwrap.dedent(methods['update']))
        code_text += function_code
        code_text += "}\n"

        code_text += "};\n\n"

        code_text += f"}} // namespace {file_name_no_extension}\n\n"

        code_text += f"#endif // __{file_name_no_extension.upper()}_HPP__\n"

        return code_text

    @staticmethod
    def generate_ltv_mpc_phi_f_updater_cpp_code(input_python_file_name: str,
                                                file_name_no_extension: str,
                                                embedded_integrator_updater_cpp_name: str,
                                                prediction_matrices_phi_f_updater_cpp_name: str):
        """
        Generates C++ header code for an LTV MPC Phi/F updater class
          based on provided Python class definitions.

        This function reads a Python file containing class definitions,
          extracts the first class and its methods,
        and generates a C++ header file that defines a class
          with a static `update` method. The generated class
        uses two other updater classes (for embedded integrator
          and prediction matrices) and includes their headers.
        The resulting code is suitable for deployment in a C++ project.

        Args:
            input_python_file_name (str): Path to the Python file
              containing the class to convert.
            file_name_no_extension (str): Base name (without extension)
              for the generated C++ header file and namespace.
            embedded_integrator_updater_cpp_name (str):
            Name of the C++ header file for the embedded integrator updater.
            prediction_matrices_phi_f_updater_cpp_name (str):
            Name of the C++ header file for the prediction matrices Phi/F updater.

        Returns:
            str: The generated C++ header code as a string.

        Raises:
            ValueError: If no classes are found in the provided Python file.
        """

        function_file_path = ControlDeploy.find_file(
            input_python_file_name, os.getcwd())

        class_methods = extract_class_methods(function_file_path)
        items = class_methods.items()
        class_name = next(iter(items))[0] if items else None
        methods = next(iter(items))[1] if items else {}

        if not class_methods:
            raise ValueError(f"No classes found in {input_python_file_name}.")

        embedded_integrator_updater_cpp_name_no_extension = os.path.splitext(
            embedded_integrator_updater_cpp_name)[0]
        prediction_matrices_phi_f_updater_cpp_name_no_extension = os.path.splitext(
            prediction_matrices_phi_f_updater_cpp_name)[0]

        code_text = ""

        code_text += f"#ifndef __{file_name_no_extension.upper()}_HPP__\n"
        code_text += f"#define __{file_name_no_extension.upper()}_HPP__\n\n"

        code_text += f"#include \"{embedded_integrator_updater_cpp_name}\"\n"
        code_text += f"#include \"{prediction_matrices_phi_f_updater_cpp_name}\"\n\n"

        code_text += f"namespace {file_name_no_extension} {{\n\n"

        code_text += f"using namespace {embedded_integrator_updater_cpp_name_no_extension};\n"
        code_text += f"using namespace {prediction_matrices_phi_f_updater_cpp_name_no_extension};\n\n"

        code_text += f"class {class_name} {{\npublic:\n"
        code_text += f"template <typename StateSpace_Type, typename Parameter_Type,\n" + \
            "  typename Phi_Type, typename F_Type>\n"
        code_text += "static inline void update(const Parameter_Type &parameter, Phi_Type& Phi, F_Type& F) {\n\n"

        code_text += "  StateSpace_Type state_space;\n"
        code_text += "  EmbeddedIntegrator_Updater::update(parameter, state_space);\n\n"

        code_text += "  PredictionMatricesPhiF_Updater::update(\n"
        code_text += "      state_space.A, state_space.B, state_space.C, Phi, F);\n\n"

        code_text += "}\n"

        code_text += "};\n\n"

        code_text += f"}} // namespace {file_name_no_extension}\n\n"

        code_text += f"#endif // __{file_name_no_extension.upper()}_HPP__\n"

        return code_text
