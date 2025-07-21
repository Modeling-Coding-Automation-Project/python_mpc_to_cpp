"""
File: adaptive_matrices_deploy.py

This module provides the `AdaptiveMatricesDeploy` class,
which contains static utility methods for generating C++ code from Python source files and
dataclass parameter objects,
specifically for Linear Time-Varying (LTV) Model Predictive Control (MPC) matrix operations.
The generated C++ code is intended for use in control systems and MPC applications,
leveraging templates and namespaces for type safety and modularity.
"""
import os
import sys
sys.path.append(os.getcwd())
sys.path.append(os.path.join(
    os.getcwd(), "external_libraries", "python_control_to_cpp"))

from dataclasses import is_dataclass
import textwrap

from external_libraries.python_numpy_to_cpp.python_numpy.numpy_deploy import python_to_cpp_types
from external_libraries.MCAP_python_control.python_control.control_deploy import ControlDeploy
from external_libraries.python_control_to_cpp.python_control.kalman_filter_deploy import KalmanFilterDeploy

from mpc_utility.mpc_matrices_deploy import extract_class_methods
from mpc_utility.mpc_matrices_deploy import MatrixUpdaterToCppVisitor
from mpc_utility.mpc_matrices_deploy import StateSpaceUpdaterToCppVisitor
from mpc_utility.mpc_matrices_deploy import PredictionMatricesPhiF_UpdaterToCppVisitor


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
    def generate_embedded_integrator_updater_cpp_code(
            input_python_file_name: str,
            file_name_no_extension: str):
        """
        Generates C++ header code for embedded integrator updater classes
          based on Python class definitions.

        This function reads a Python file containing class definitions
          for matrix updaters and a state-space updater,
        and generates a corresponding C++ header file with templated classes
          and methods suitable for embedded deployment.

        Args:
            input_python_file_name (str): The path to the Python file
              containing the updater class definitions.
            file_name_no_extension (str): The base name (without extension)
              to use for the generated C++ header file and namespace.

        Returns:
            str: The generated C++ header code as a string.

        Raises:
            ValueError: If no classes are found in the specified Python file.

        Notes:
            - The generated C++ code includes template classes for each updater,
              with appropriate type parameters.
            - The function relies on helper functions and visitors
                to parse Python source and convert method bodies to C++.
            - The output is wrapped in include guards and a namespace
              based on `file_name_no_extension`.
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

        arg_list = []

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

                arg_list = visitor.arg_list
            else:
                # MPC StateSpace Updater class
                output_type_name = f"{class_name}_Output_Type"

                code_text += f"class {class_name} {{\npublic:\n"
                code_text += "template <typename X_Type, typename U_Type, " + \
                    f"typename Parameter_Type, typename {output_type_name}>\n"
                code_text += "static inline void update(const X_Type& X, const U_Type& U, " + \
                    "const Parameter_Type& parameter, " + \
                    f"{output_type_name}& output) {{\n"

                visitor = StateSpaceUpdaterToCppVisitor(
                    arg_list=arg_list
                )
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
    def generate_adaptive_mpc_phi_f_updater_cpp_code(
            input_python_file_name: str,
            file_name_no_extension: str,
            embedded_integrator_updater_cpp_name: str,
            prediction_matrices_phi_f_updater_cpp_name: str):
        """
        Generates C++ header code for an adaptive MPC Phi/F updater class.

        This function creates a C++ header file as a string, which defines a class for updating
        the prediction matrices (Phi, F) in an adaptive Model Predictive Control (MPC) context.
        The generated class integrates two updater modules: one for the embedded integrator and
        one for the prediction matrices. The resulting C++ code includes necessary headers,
        namespace usage, and a templated static update method that performs the update logic.

        Args:
            input_python_file_name (str): Path to the Python file containing the class
              to be converted.
            file_name_no_extension (str): Base name (without extension) for the generated
              C++ header file and namespace.
            embedded_integrator_updater_cpp_name (str): Filename of the embedded integrator
              updater C++ header to include.
            prediction_matrices_phi_f_updater_cpp_name (str): Filename of the
              prediction matrices Phi/F updater C++ header to include.

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
        code_text += f"template <typename X_Type, typename U_Type, typename Parameter_Type,\n" + \
            "  typename Phi_Type, typename F_Type, typename StateSpace_Type>\n"
        code_text += "static inline void update(const X_Type &X, const U_Type &U,\n" + \
            "  const Parameter_Type &parameter, Phi_Type& Phi, F_Type& F) {\n\n"

        code_text += "  StateSpace_Type state_space;\n"
        code_text += "  EmbeddedIntegrator_Updater::update(X, U, parameter, state_space);\n\n"

        code_text += "  PredictionMatricesPhiF_Updater::update(\n"
        code_text += "      state_space.A, state_space.B, state_space.C, Phi, F);\n\n"

        code_text += "}\n"

        code_text += "};\n\n"

        code_text += f"}} // namespace {file_name_no_extension}\n\n"

        code_text += f"#endif // __{file_name_no_extension.upper()}_HPP__\n"

        return code_text
