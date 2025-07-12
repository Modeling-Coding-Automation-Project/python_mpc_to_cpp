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
    def __init__(self, Output_Type_name):
        self.cpp_code = ""
        self.Output_Type_name = Output_Type_name
        self.Value_Type_name = "double"
        self.class_name = ""
        self.in_class = False
        self.indent = ""
        self.SparseAvailable = None

    def visit_ClassDef(self, node):
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
        tree = ast.parse(python_code)
        if tree.body and isinstance(tree.body[0], ast.FunctionDef):
            self.visit(tree.body[0])
        return self.cpp_code


class StateSpaceUpdaterToCppVisitor(ast.NodeVisitor):
    def __init__(self):
        self.cpp_code = ""
        self.indent = "    "
        self.param_names = []
        self.has_D = False

    def visit_FunctionDef(self, node):
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
                f"{self.indent}auto {updater} = {updater}_Updater<MPC_StateSpace_Updater_Output_Type::{updater}_Type>::update("
                + ", ".join([v[0] for v in self.param_names]) + ");\n\n"
            )

        # D
        if self.has_D:
            self.cpp_code += (
                f"{self.indent}auto D = D_Updater<MPC_StateSpace_Updater_Output_Type::D_Type>::update("
                + ", ".join([v[0] for v in self.param_names]) + ");\n\n"
            )

        # substitute the output
        for updater in ["A", "B", "C"]:
            self.cpp_code += f"{self.indent}output.{updater} = {updater};\n"
        if self.has_D:
            self.cpp_code += f"{self.indent}output.D = D;\n"

    def convert(self, python_code):
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
        # Only process the body, skip arguments and decorators
        for stmt in node.body:
            self.visit(stmt)

    def visit_Assign(self, node):
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

    @staticmethod
    def generate_parameter_cpp_code(parameter_object,
                                    value_type_name: str,
                                    file_name_no_extension: str = None):
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
            f"          typename D_Type, typename Phi_Type, typename F_Type >\n"
        code_text += "static inline void update(const A_Type &A, const B_Type &B,\n" + \
            f"          const C_Type &C, const D_Type &D, Phi_Type& Phi, F_Type& F) {{\n"

        visitor = PredictionMatricesPhiF_UpdaterToCppVisitor()
        function_code = visitor.convert(
            textwrap.dedent(methods['update']))
        code_text += function_code
        code_text += "}\n"

        code_text += "};\n\n"

        code_text += f"}} // namespace {file_name_no_extension}\n\n"

        code_text += f"#endif // __{file_name_no_extension.upper()}_HPP__\n"

        return code_text
