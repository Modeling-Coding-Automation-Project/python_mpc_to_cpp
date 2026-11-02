"""Generate and build the Python bindings used by the SIL tests.

Existing tests build beside their C++ source. A separate module name and working
directory also allow several controllers in one folder and temporary builds.
All builds use the Python interpreter running the test and stop on failure.
"""

import importlib.machinery
from pathlib import Path
import shutil
import subprocess
import sys


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def snake_to_camel(snake_str):
    components = snake_str.split('_')
    return ''.join(x.title() for x in components)


class CmakeGenerator:
    def __init__(self, SIL_folder, sil_name=None, work_dir=None):
        self.SIL_folder = Path(SIL_folder).resolve()
        self.folder_name = sil_name or self.SIL_folder.name
        self.work_dir = (Path(work_dir).resolve() if work_dir is not None
                         else self.SIL_folder)

    def generate_cmake_lists_txt(self):
        module_name = snake_to_camel(self.folder_name) + "SIL"
        source_name = self.folder_name + "_SIL.cpp"
        include_folders = [
            "mpc_utility",
            "python_mpc",
            "external_libraries/python_optimization_to_cpp/optimization_utility",
            "external_libraries/python_optimization_to_cpp/python_optimization",
            "external_libraries/python_control_to_cpp/python_control",
            "external_libraries/python_numpy_to_cpp/python_numpy",
            "external_libraries/python_numpy_to_cpp/base_matrix",
            "external_libraries/python_math_to_cpp/base_math",
            "external_libraries/python_math_to_cpp/python_math",
            "external_libraries/base_utility_cpp/base_utility",
        ]
        includes = [self.work_dir, self.SIL_folder]
        includes.extend(REPOSITORY_ROOT / folder for folder in include_folders)
        include_text = "\n".join(
            f'    "{path.as_posix()}"' for path in includes)

        code_text = f'''cmake_minimum_required(VERSION 3.14)
if(POLICY CMP0148)
    cmake_policy(SET CMP0148 NEW)
endif()

project({module_name} LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_FLAGS_RELEASE "-O2")
set(PYBIND11_FINDPYTHON ON)
find_package(pybind11 REQUIRED)

pybind11_add_module({module_name} {source_name})
target_compile_options({module_name} PRIVATE -Wall -Wextra -Werror)
target_include_directories({module_name} PRIVATE
{include_text}
)
'''
        self.work_dir.mkdir(parents=True, exist_ok=True)
        (self.work_dir / "CMakeLists.txt").write_text(code_text, encoding="utf-8")


class SIL_CodeGenerator:
    """Deploy headers and build a SIL module, returning its extension path.

    The two-argument form retains the existing folder-based naming. ``sil_name``
    selects another ``<name>_SIL.cpp`` in that folder; ``work_dir`` keeps generated
    files outside the source tree when a test uses a temporary directory.
    """

    def __init__(self, deployed_file_names, SIL_folder, sil_name=None,
                 work_dir=None):
        self.deployed_file_names = deployed_file_names
        self.SIL_folder = Path(SIL_folder).resolve()
        self.folder_name = sil_name or self.SIL_folder.name
        self.work_dir = (Path(work_dir).resolve() if work_dir is not None
                         else self.SIL_folder)

    def move_deployed_files(self, file_names):
        for file_name in file_names:
            if isinstance(file_name, list):
                self.move_deployed_files(file_name)
            else:
                source = Path(file_name).resolve()
                destination = self.work_dir / Path(file_name).name
                if source != destination:
                    if destination.exists():
                        destination.unlink()
                    shutil.move(str(source), str(destination))

    def generate_wrapper_code(self):
        wrapper_name = self.folder_name + "_SIL_wrapper"
        deployed_file = Path(self.deployed_file_names[-1])
        code_text = (f'#include "{deployed_file.name}"\n\n'
                     f'namespace {wrapper_name} = {deployed_file.stem};\n')
        (self.work_dir / (wrapper_name + ".hpp")).write_text(
            code_text, encoding="utf-8")

    def build_pybind11_code(self):
        import pybind11

        build_folder = self.work_dir / "build"
        module_name = snake_to_camel(self.folder_name) + "SIL"
        if build_folder.exists():
            shutil.rmtree(build_folder)

        subprocess.run([
            "cmake", "-S", str(self.work_dir), "-B", str(build_folder),
            "-DCMAKE_BUILD_TYPE=Release",
            f"-DPython_EXECUTABLE={sys.executable}",
            f"-Dpybind11_DIR={pybind11.get_cmake_dir()}",
        ], check=True)
        subprocess.run([
            "cmake", "--build", str(build_folder), "--config", "Release",
            "--parallel", "1",
        ], check=True)

        for suffix in importlib.machinery.EXTENSION_SUFFIXES:
            source = build_folder / (module_name + suffix)
            if source.is_file():
                destination = self.work_dir / source.name
                source.replace(destination)
                return destination
        raise FileNotFoundError(f"Built SIL module not found: {module_name}")

    def build_SIL_code(self):
        self.work_dir.mkdir(parents=True, exist_ok=True)
        source = self.SIL_folder / (self.folder_name + "_SIL.cpp")
        if self.work_dir != self.SIL_folder:
            # Quoted includes must select the headers generated for this build.
            shutil.copy2(source, self.work_dir / source.name)

        CmakeGenerator(self.SIL_folder, self.folder_name,
                       self.work_dir).generate_cmake_lists_txt()
        self.move_deployed_files(self.deployed_file_names)
        self.generate_wrapper_code()
        return self.build_pybind11_code()
