"""
File: ltv_matrices_deploy.py

This module provides utilities for converting Python-based 
Linear Time-Varying (LTV) Model Predictive Control (MPC) models into C++ code.
It includes classes and functions to parse Python classes and methods
 (especially those defining system matrices and update logic),
and generate corresponding C++ header files for use in
 embedded or high-performance environments.

Key features:

Extracts class methods from Python source files using AST parsing.
Converts Python matrix updater and state-space updater
 logic into C++ templates and classes.
Handles translation of NumPy-based matrix operations
 and assignments to C++ equivalents.
Supports generation of C++ code for parameter structures,
 state-space updaters, and prediction matrix updaters.
Designed for integration with external code generation
 and control libraries.
The generated C++ code enables deployment of LTV MPC models,
 including all necessary type definitions, update routines,
and matrix operations, for use in C++ projects.
"""
import os
import sys
sys.path.append(os.getcwd())
sys.path.append(os.path.join(
    os.getcwd(), "external_libraries", "python_control_to_cpp"))

import inspect
import ast
import astor
import os
import numpy as np
from dataclasses import is_dataclass
import textwrap

from external_libraries.python_numpy_to_cpp.python_numpy.numpy_deploy import NumpyDeploy, python_to_cpp_types
from external_libraries.MCAP_python_control.python_control.control_deploy import ControlDeploy
from external_libraries.MCAP_python_control.python_control.control_deploy import FunctionExtractor
from external_libraries.MCAP_python_control.python_control.control_deploy import IntegerPowerReplacer
from external_libraries.python_control_to_cpp.python_control.kalman_filter_deploy import NpArrayExtractor

from external_libraries.python_control_to_cpp.python_control.kalman_filter_deploy import KalmanFilterDeploy


def extract_class_methods(file_path):
    """
    Extracts class methods from a Python file and returns them as a dictionary.
    The keys are class names and the values are dictionaries of method names
    with their source code as strings.
    """
    with open(file_path, "r", encoding="utf-8") as f:
        source = f.read()
    source_lines = source.splitlines()
    tree = ast.parse(source)
    class_methods = {}

    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            class_name = node.name
            methods = {}
            for item in node.body:
                if isinstance(item, ast.FunctionDef):
                    # include decorators and line numbers
                    start = item.lineno - 1
                    if item.decorator_list:
                        start = min(
                            dec.lineno - 1 for dec in item.decorator_list)
                    end = item.end_lineno
                    method_source = "\n".join(source_lines[start:end])
                    methods[item.name] = method_source
            class_methods[class_name] = methods
    return class_methods


class MatrixUpdaterToCppVisitor(ast.NodeVisitor):
    """
    MatrixUpdaterToCppVisitor is an AST (Abstract Syntax Tree) visitor class
    that traverses Python code and generates equivalent C++ code
    for matrix updater classes, particularly for use 
    in Model Predictive Control (MPC) applications. 
    The class is designed to convert Python class and function definitions,
    especially those involving matrix operations and sympy functions,
    into C++ struct and method definitions
    with appropriate type annotations and conversions.

    Notes:
        - The visitor relies on external utilities such as
        IntegerPowerReplacer and NpArrayExtractor for specific code transformations.
        - The class is tailored for code generation scenarios
          where Python matrix manipulation code needs to be ported
          to C++ for performance-critical applications.
    """

    def __init__(self, Output_Type_name):
        self.cpp_code = ""
        self.Output_Type_name = Output_Type_name
        self.Value_Type_name = "double"
        self.class_name = ""
        self.in_class = False
        self.indent = ""
        self.SparseAvailable = None

    def visit_ClassDef(self, node):
        """
        Visits a Python class definition node and generates the corresponding C++ struct code.
        Args:
            node (ast.ClassDef): The AST node representing the class definition.
        Side Effects:
            - Updates self.class_name with the current class name.
            - Appends the generated C++ struct code to self.cpp_code.
            - Sets self.in_class to True during processing and resets it to False after.
            - Adjusts self.indent for proper code formatting.
        The method iterates over the class body statements and recursively
          visits each node to generate the appropriate C++ code.
        """

        self.class_name = node.name
        self.cpp_code += f"struct {self.class_name} {{\n"
        self.in_class = True
        self.indent = "    "
        for stmt in node.body:
            self.visit(stmt)
        self.cpp_code += "};\n"
        self.in_class = False
        self.indent = ""

    def visit_FunctionDef(self, node):
        """
        Visits a Python function definition (ast.FunctionDef node)
          and generates the corresponding C++ function code.
        This method performs the following tasks:
        - Checks if the function is decorated as a static method
          and adds the 'static' keyword if necessary.
        - Extracts argument names and their type annotations,
          mapping Python types to C++ types using a predefined mapping.
        - Constructs the C++ function signature, including argument types and return type.
        - Handles special cases for functions named 'update' and 'sympy_function':
            - For 'update', identifies unused arguments
              and adds static_cast<void> statements to suppress unused variable warnings.
            - For 'sympy_function', initializes a result variable of the return type.
        - Recursively visits the function body to generate corresponding C++ code.
        - Manages indentation for proper formatting of the generated C++ code.
        Args:
            node (ast.FunctionDef): The AST node representing
              the Python function definition.
        Side Effects:
            Modifies self.cpp_code by appending the generated C++ function code.
            Adjusts self.indent to manage code formatting.
        """

        # static method check
        is_static = any(isinstance(dec, ast.Name) and dec.id == "staticmethod"
                        for dec in node.decorator_list)
        static_str = "static " if is_static else ""

        # arguments and type annotations
        args = [arg.arg for arg in node.args.args]
        annotations = {}
        for arg in node.args.args:
            if arg.annotation:
                annotations[arg.arg] = ast.dump(arg.annotation)
            else:
                annotations[arg.arg] = None
        if node.returns:
            annotations['return'] = ast.dump(node.returns)
        else:
            annotations['return'] = None

        # arguments list
        arg_strs = []
        for arg in args:
            Value_Type_name = self.Value_Type_name
            if annotations[arg] is not None:
                annotation = annotations[arg]
                if "attr='" in annotation:
                    Value_Type_python_name = annotation.split("attr='")[
                        1].split("'")[0]
                    Value_Type_name = python_to_cpp_types.get(
                        Value_Type_python_name, "double")
            arg_strs.append(f"{Value_Type_name} {arg}")
        arg_list = ", ".join(arg_strs)

        # return type
        ret_type = self.Output_Type_name

        # function header
        self.cpp_code += f"{self.indent}{static_str}inline auto {node.name}({arg_list}) -> {ret_type} {{\n"

        if node.name == "update":
            update_args = [arg.arg for arg in node.args.args]

            sympy_func_args = []
            for stmt in node.body:
                if isinstance(stmt, ast.Return) and isinstance(stmt.value, ast.Call):
                    if hasattr(stmt.value.func, 'attr') and stmt.value.func.attr == "sympy_function":
                        sympy_func_args = [astor.to_source(
                            arg).strip() for arg in stmt.value.args]

            unused_args = [
                arg for arg in update_args if arg not in sympy_func_args]

            for arg in unused_args:
                self.cpp_code += f"{self.indent}static_cast<void>({arg});\n"
            self.cpp_code += "\n"

        if node.name == "sympy_function":
            self.cpp_code += f"{self.indent}    {ret_type} result;\n\n"

        # function body
        self.indent += "    "
        for stmt in node.body:
            self.visit(stmt)
        self.indent = self.indent[:-4]
        self.cpp_code += f"{self.indent}}}\n\n"

    def visit_Return(self, node):
        """
        Processes a Python AST Return node and generates corresponding C++ code.
        This method handles return statements in the AST.
          If the return value is a function call,
        it converts the call to source code and applies
          integer power transformation. If the return
        value contains a NumPy array, it extracts and converts it to C++ code,
          handling sparse arrays
        if necessary. The generated C++ code is appended
          to the class's cpp_code attribute.
        Args:
            node (ast.Return): The AST node representing the return statement.
        Raises:
            TypeError: If the return value is not a function call (ast.Call).
        """

        return_code = ""
        if isinstance(node.value, ast.Call):
            return_code += astor.to_source(node.value).strip()
        else:
            raise TypeError(f"Unsupported return type: {type(node.value)}")

        integer_power_replacer = IntegerPowerReplacer()
        return_code = integer_power_replacer.transform_code(return_code)

        if "np.array(" in return_code:
            np_array_extractor = NpArrayExtractor(
                return_code, self.Value_Type_name)
            np_array_extractor.extract()
            return_code = np_array_extractor.convert_to_cpp()
            self.SparseAvailable = np_array_extractor.SparseAvailable
            return_code = return_code.replace(
                "\n", "\n" + self.indent + "    ")
            self.cpp_code += self.indent + "    " + return_code + "\n"
            self.cpp_code += self.indent + "    return result;\n"
        else:
            self.cpp_code += self.indent + "    return " + return_code + ";\n"

    def visit_Assign(self, node):
        """
        Visits an assignment node in the AST
          and generates corresponding C++ assignment code.
        This method:
        - Converts the assignment targets and value from Python AST to source code strings.
        - Applies a transformation to handle integer power operations in the value.
        - Constructs a C++ assignment statement using the specified value type name.
        - Replaces Python-style indexing with C++ template-based indexing.
        - Appends the generated C++ code to the class's cpp_code attribute.
        Args:
            node (ast.Assign): The assignment node from the Python AST to process.
        Side Effects:
            Modifies self.cpp_code by appending the generated C++ assignment code.
        """

        integer_power_replacer = IntegerPowerReplacer()
        assign_code = ""
        targets = [astor.to_source(t).strip() for t in node.targets]
        value = astor.to_source(node.value).strip()
        value = integer_power_replacer.transform_code(value)
        assign_code += self.indent + self.Value_Type_name + \
            " " + ", ".join(targets) + " = " + value + ";\n"
        assign_code += "\n"
        assign_code = assign_code.replace("[", ".template get<")
        assign_code = assign_code.replace("]", ">()")
        self.cpp_code += assign_code

    def convert(self, python_code):
        """
        Converts a given Python code string into its
          C++ equivalent by parsing the code's AST,
        visiting the first function definition,
          and performing necessary string replacements
        for compatibility. Returns the generated C++ code as a string.
        Args:
            python_code (str): The Python source code to be converted.
        Returns:
            str: The converted C++ code.
        """

        tree = ast.parse(python_code)
        if tree.body and isinstance(tree.body[0], ast.FunctionDef):
            self.visit(tree.body[0])

        self.cpp_code = self.cpp_code.replace(
            ".sympy_function(", "::sympy_function(")

        return self.cpp_code


class StateSpaceUpdaterToCppVisitor(ast.NodeVisitor):
    """
    StateSpaceUpdaterToCppVisitor is an AST (Abstract Syntax Tree)
      visitor class that traverses a Python function definition
    and generates equivalent C++ code for updating
    state-space matrices (A, B, C, and optionally D) using updater classes.
    """

    def __init__(self):
        self.cpp_code = ""
        self.indent = "    "
        self.param_names = []
        self.has_D = False

        self.output_type_name = ""

    def visit_FunctionDef(self, node):
        """
        Visits a Python function definition node (ast.FunctionDef)
          and generates corresponding C++ code
          for parameter extraction and updater calls.
        This method performs the following steps:
        1. Extracts argument names from the function definition,
          assuming the first argument represents the parameters object.
        2. Scans the function body for assignments that extract attributes
          from the parameters object, collecting variable and attribute names.
        3. Detects if a variable 'D' is assigned via a call to a D_Updater,
          setting a flag if present.
        4. Generates C++ code to declare and initialize variables
          from the parameters object.
        5. Generates C++ code to call updater functions
          (A_Updater, B_Updater, C_Updater, and optionally D_Updater)
            using the extracted variables.
        6. Assigns the results of the updater calls to the
          corresponding fields in the output object.
        Args:
            node (ast.FunctionDef): The AST node representing the
              function definition to process.
        Side Effects:
            Updates self.cpp_code with the generated C++ code.
            Updates self.param_names with tuples of
              (variable name, attribute name) extracted from parameters.
            Sets self.has_D to True if a D updater call is detected.
        """

        # arguments names
        args = [arg.arg for arg in node.args.args]
        # assuming the first argument is 'parameters'
        param_arg = args[0] if args else "parameters"

        # Enumerate variables (those extracted from parameters)
        for stmt in node.body:
            if isinstance(stmt, ast.Assign) and isinstance(stmt.value, ast.Attribute):
                # Example: Lshaft = parameters.Lshaft
                var_name = stmt.targets[0].id
                attr_name = stmt.value.attr
                self.param_names.append((var_name, attr_name))
            elif isinstance(stmt, ast.Assign) and isinstance(stmt.value, ast.Call):
                # Example: D = D_Updater.update(...)
                if stmt.targets[0].id == "D":
                    self.has_D = True

        # Extracting variable
        for var_name, attr_name in self.param_names:
            self.cpp_code += f"{self.indent}double {var_name} = parameter.{attr_name};\n"

        self.cpp_code += "\n"

        # A, B, C
        for updater in ["A", "B", "C"]:
            self.cpp_code += (
                f"{self.indent}auto {updater} = {updater}_Updater<typename " +
                f"{self.output_type_name}::{updater}_Type>::update("
                + ", ".join([v[0] for v in self.param_names]) + ");\n\n"
            )

        # D
        if self.has_D:
            self.cpp_code += (
                f"{self.indent}auto D = D_Updater<typename {self.output_type_name}::D_Type>::update("
                + ", ".join([v[0] for v in self.param_names]) + ");\n\n"
            )

        # substitute the output
        for updater in ["A", "B", "C"]:
            self.cpp_code += f"{self.indent}output.{updater} = {updater};\n"
        if self.has_D:
            self.cpp_code += f"{self.indent}output.D = D;\n"

    def convert(self, python_code):
        """
        Converts a given Python function code into its C++ equivalent representation.
        Args:
            python_code (str): The source code of a Python function as a string.
        Returns:
            str: The generated C++ code as a string.
        """

        tree = ast.parse(python_code)
        for node in tree.body:
            if isinstance(node, ast.FunctionDef):
                self.visit(node)
                break

        return self.cpp_code


class PredictionMatricesPhiF_UpdaterToCppVisitor(ast.NodeVisitor):
    """
    Converts the body of update_Phi_F from Python to C++.
    - Ignores np.zeros lines.
    - Converts chained matrix multiplications (C @ A ...) to C++ auto lines.
    - Converts assignments to Phi[...] and F[...] to C++ set/get calls.
    """

    def __init__(self):
        self.cpp_code = ""
        self.indent = "    "
        self.defined_vars = set()

    def visit_FunctionDef(self, node):
        """
        Visits a function definition node in the AST.
        This method processes only the body of the function,
          skipping its arguments and decorators.
        It iterates through each statement in the function body
          and applies the visitor to them.
        Args:
            node (ast.FunctionDef): The function definition
              AST node to visit.
        """

        # Only process the body, skip arguments and decorators
        for stmt in node.body:
            self.visit(stmt)

    def visit_Assign(self, node):
        """
        Visits assignment nodes in the AST and generates
          corresponding C++ code for matrix operations and assignments.
        Handles the following cases:
        - Ignores assignments involving `np.zeros`.
        - Converts matrix multiplication assignments
          (using `@` in Python) to C++ code using `*` operator
            and `auto` type deduction.
        - Handles chained and variable-based matrix multiplications.
        - Processes assignments to subscripted variables
          (e.g., `Phi[i, j] = ...`), converting them
            to C++ template-based set operations.
        - Handles assignments where the right-hand side is a subscript,
          converting them to C++ template-based get operations.
        - Updates the set of defined variables as new variables are assigned.
        Parameters:
            node (ast.Assign): The assignment node to process.
        Returns:
            None
        """

        # Ignore np.zeros
        if isinstance(node.value, ast.Call):
            func = node.value.func
            if isinstance(func, ast.Attribute) and func.attr == "zeros":
                return

        # Matrix multiplication chain: C_A_1 = C @ A
        if isinstance(node.value, ast.BinOp) and isinstance(node.value.op, ast.MatMult):
            target = astor.to_source(node.targets[0]).strip()
            left = astor.to_source(node.value.left).strip()
            right = astor.to_source(node.value.right).strip()
            self.cpp_code += f"{self.indent}auto {target} = {left} * {right};\n"
            self.defined_vars.add(target)
            return

        # Chained matrix multiplication: C_A_2 = C_A_1 @ A
        if isinstance(node.value, ast.BinOp) and isinstance(node.value.op, ast.MatMult):
            target = astor.to_source(node.targets[0]).strip()
            left = astor.to_source(node.value.left).strip()
            right = astor.to_source(node.value.right).strip()
            self.cpp_code += f"{self.indent}auto {target} = {left} * {right};\n"
            self.defined_vars.add(target)
            return

        # Assignment from matrix multiplication result: C_A_0_B = C @ B
        if isinstance(node.value, ast.BinOp) and isinstance(node.value.op, ast.MatMult):
            target = astor.to_source(node.targets[0]).strip()
            left = astor.to_source(node.value.left).strip()
            right = astor.to_source(node.value.right).strip()
            self.cpp_code += f"{self.indent}auto {target} = {left} * {right};\n"
            self.defined_vars.add(target)
            return

        # Assignment from variable: e.g. C_A_1_B = C_A_1 @ B
        if isinstance(node.value, ast.BinOp) and isinstance(node.value.op, ast.MatMult):
            target = astor.to_source(node.targets[0]).strip()
            left = astor.to_source(node.value.left).strip()
            right = astor.to_source(node.value.right).strip()
            self.cpp_code += f"{self.indent}auto {target} = {left} * {right};\n"
            self.defined_vars.add(target)
            return

        # Assignment to Phi[...] or F[...]
        if isinstance(node.targets[0], ast.Subscript):
            target = node.targets[0]
            value = node.value
            # Only handle simple assignments: Phi[xx, yy] = something
            if isinstance(target.value, ast.Name):
                arr_name = target.value.id
                # Get indices
                if isinstance(target.slice, ast.Index):
                    idx = target.slice.value
                    if isinstance(idx, ast.Tuple):
                        i = astor.to_source(idx.elts[0]).strip()
                        j = astor.to_source(idx.elts[1]).strip()
                    else:
                        i = astor.to_source(idx).strip()
                        j = "0"
                elif isinstance(target.slice, ast.Tuple):
                    i = astor.to_source(target.slice.elts[0]).strip()
                    j = astor.to_source(target.slice.elts[1]).strip()
                else:
                    # fallback
                    i = j = "0"
                # Value
                val_code = astor.to_source(value).strip()
                # If value is subscript, convert get<...>
                if isinstance(value, ast.Subscript):
                    v = value
                    if isinstance(v.value, ast.Name):
                        v_arr = v.value.id
                        if isinstance(v.slice, ast.Index):
                            v_idx = v.slice.value
                            if isinstance(v_idx, ast.Tuple):
                                vi = astor.to_source(v_idx.elts[0]).strip()
                                vj = astor.to_source(v_idx.elts[1]).strip()
                            else:
                                vi = astor.to_source(v_idx).strip()
                                vj = "0"
                        elif isinstance(v.slice, ast.Tuple):
                            vi = astor.to_source(v.slice.elts[0]).strip()
                            vj = astor.to_source(v.slice.elts[1]).strip()
                        else:
                            vi = vj = "0"
                        val_code = f"{v_arr}.template get<{vi}, {vj}>()"

                i = i.strip("()")
                j = j.strip("()")
                self.cpp_code += f"{self.indent}{arr_name}.template set<{i}, {j}>({val_code});\n"
                return

    def convert(self, python_code):
        tree = ast.parse(python_code)
        for node in tree.body:
            if isinstance(node, ast.FunctionDef):
                self.visit(node)
                break
        return self.cpp_code


class LTVMatricesDeploy:
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
