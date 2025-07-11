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


class MatrixUpdatorToCppVisitor(ast.NodeVisitor):
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
        # static method判定
        is_static = any(isinstance(dec, ast.Name) and dec.id == "staticmethod"
                        for dec in node.decorator_list)
        static_str = "static " if is_static else ""

        # 引数と型
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

        # 引数リスト
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

        # 戻り値型
        ret_type = self.Output_Type_name

        # 関数ヘッダ
        self.cpp_code += f"{self.indent}{static_str}inline auto {node.name}({arg_list}) -> {ret_type} {{\n"

        if node.name == "sympy_function":
            self.cpp_code += f"{self.indent}    {ret_type} result;\n\n"

        # 関数本体
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
    def generate_mpc_state_space_updater_cpp_code(python_file_name: str,
                                                  file_name_no_extension: str):

        function_file_path = ControlDeploy.find_file(
            python_file_name, os.getcwd())

        class_methods = extract_class_methods(function_file_path)

        if not class_methods:
            raise ValueError(f"No classes found in {python_file_name}.")

        code_text = ""

        code_text += f"#ifndef __{file_name_no_extension.upper()}_HPP__\n"
        code_text += f"#define __{file_name_no_extension.upper()}_HPP__\n\n"

        code_text += f"namespace {file_name_no_extension} {{\n\n"

        output_type_names = []

        for i, (class_name, methods) in enumerate(class_methods.items()):
            if i < len(class_methods) - 1:
                output_type_name = f"{class_name}_Output_Type"
                output_type_names.append(output_type_name)

                code_text += f"template <typename " + output_type_name + ">\n"
                code_text += f"class {class_name} {{\npublic:\n"

                converter = MatrixUpdatorToCppVisitor(output_type_name)

                for _, method_source in methods.items():
                    function_code = converter.convert(
                        textwrap.dedent(method_source))
                code_text += function_code
                code_text += "\n"

            else:
                code_text += f"template <"
                for output_type_name in output_type_names:
                    code_text += f"typename {output_type_name}, "
                code_text = code_text[:-2]  # Remove the last comma and space
                code_text += ">\n"
                code_text += f"class {class_name} {{\npublic:\n"

                for method_name, method_source in methods.items():
                    pass

            code_text += "};\n\n"

        code_text += f"}} // namespace {file_name_no_extension}\n\n"

        code_text += f"#endif // __{file_name_no_extension.upper()}_HPP__\n"

        return code_text
