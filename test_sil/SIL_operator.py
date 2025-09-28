import os
import subprocess


def snake_to_camel(snake_str):
    components = snake_str.split('_')
    return ''.join(x.title() for x in components)


class CmakeGenerator:
    def __init__(self, SIL_folder):
        self.SIL_folder = SIL_folder

        self.folder_name = os.path.basename(os.path.normpath(self.SIL_folder))

    def generate_cmake_lists_txt(self):
        SIL_lib_file_name = snake_to_camel(self.folder_name) + "SIL"
        SIL_cpp_file_name = self.folder_name + "_SIL"

        code_text = ""
        code_text += "cmake_minimum_required(VERSION 3.14)\n"
        code_text += "cmake_policy(SET CMP0148 NEW)\n\n"

        code_text += f"project({SIL_lib_file_name})\n\n"

        code_text += "set(CMAKE_CXX_STANDARD 11)\n"
        code_text += "set(CMAKE_CXX_STANDARD_REQUIRED ON)\n"
        code_text += "set(CMAKE_CXX_FLAGS_RELEASE \"-O2\")\n"
        code_text += "set(CMAKE_CXX_FLAGS_RELEASE \"${CMAKE_CXX_FLAGS_RELEASE} -flto=auto\")\n\n"

        code_text += "find_package(pybind11 REQUIRED)\n\n"

        code_text += f"pybind11_add_module({SIL_lib_file_name} {SIL_cpp_file_name}.cpp)\n\n"

        code_text += f"target_compile_options({SIL_lib_file_name} PRIVATE -Werror)\n\n"

        code_text += f"target_include_directories({SIL_lib_file_name} PRIVATE\n"
        code_text += "    ${CMAKE_SOURCE_DIR}/../../mpc_utility\n"
        code_text += "    ${CMAKE_SOURCE_DIR}/../../python_mpc\n"
        code_text += "    ${CMAKE_SOURCE_DIR}/../../external_libraries/python_optimization_to_cpp/optimization_utility\n"
        code_text += "    ${CMAKE_SOURCE_DIR}/../../external_libraries/python_optimization_to_cpp/python_optimization\n"
        code_text += "    ${CMAKE_SOURCE_DIR}/../../external_libraries/python_control_to_cpp/python_control\n"
        code_text += "    ${CMAKE_SOURCE_DIR}/../../external_libraries/python_numpy_to_cpp/python_numpy\n"
        code_text += "    ${CMAKE_SOURCE_DIR}/../../external_libraries/python_numpy_to_cpp/base_matrix\n"
        code_text += "    ${CMAKE_SOURCE_DIR}/../../external_libraries/python_math_to_cpp/base_math\n"
        code_text += "    ${CMAKE_SOURCE_DIR}/../../external_libraries/python_math_to_cpp/python_math\n"
        code_text += "    ${CMAKE_SOURCE_DIR}/../../external_libraries/base_utility_cpp/base_utility\n"
        code_text += "    ${CMAKE_SOURCE_DIR}/../../test_sil/" + \
            f"{self.folder_name}\n"
        code_text += ")\n\n"

        with open(os.path.join(self.SIL_folder, "CMakeLists.txt"), "w", encoding="utf-8") as f:
            f.write(code_text)


class SIL_CodeGenerator:
    def __init__(self, deployed_file_names, SIL_folder):
        self.deployed_file_names = deployed_file_names
        self.SIL_folder = SIL_folder

        self.folder_name = os.path.basename(os.path.normpath(self.SIL_folder))

    def move_deployed_files(self, file_names):
        for file_name in file_names:
            if isinstance(file_name, list):
                self.move_deployed_files(file_name)
            else:
                src = os.path.join(os.getcwd(), file_name)
                dst = os.path.join(self.SIL_folder, file_name)
                if os.path.exists(dst):
                    os.remove(dst)
                os.rename(src, dst)

    def generate_wrapper_code(self):

        file_name = f"{self.folder_name}" + "_SIL_wrapper.hpp"
        file_name_without_extension = file_name.split(".")[0]

        deployed_file_name = self.deployed_file_names[-1]
        deployed_file_name_without_extension = deployed_file_name.split(".")[0]

        code_text = ""
        code_text += f"#include \"{deployed_file_name}\"\n\n"

        code_text += f"namespace {file_name_without_extension} = {deployed_file_name_without_extension};\n"

        with open(os.path.join(self.SIL_folder, file_name), "w", encoding="utf-8") as f:
            f.write(code_text)

    def build_pybind11_code(self):

        build_folder = os.path.join(self.SIL_folder, "build")
        generated_file_name = snake_to_camel(self.folder_name) + "SIL"

        subprocess.run(f"rm -rf {build_folder}", shell=True)
        subprocess.run(f"mkdir -p {build_folder}", shell=True)
        subprocess.run(
            f"cmake -S {self.SIL_folder} -B {build_folder} -DCMAKE_BUILD_TYPE=Release",
            shell=True
        )
        subprocess.run(
            f"cmake --build {build_folder} --config Release", shell=True)

        subprocess.run(
            f"mv {build_folder}/{generated_file_name}.*so {self.SIL_folder}", shell=True)

    def build_SIL_code(self):
        cmake_generator = CmakeGenerator(self.SIL_folder)
        cmake_generator.generate_cmake_lists_txt()

        self.move_deployed_files(self.deployed_file_names)
        self.generate_wrapper_code()
        self.build_pybind11_code()
