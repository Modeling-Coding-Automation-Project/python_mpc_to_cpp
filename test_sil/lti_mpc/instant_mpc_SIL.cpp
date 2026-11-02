/**
 * @file instant_mpc_SIL.cpp
 * @brief Python bindings for the generated servo iMPC SIL comparison.
 *
 * Exposes identical-input controller/plant steps and an independent closed
 * loop. Each cycle applies the previous input, then measures the updated plant
 * state.
 */
#include "instant_mpc_SIL_wrapper.hpp"
#include "python_control.hpp"

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <string>

namespace py = pybind11;
namespace model = instant_mpc_SIL_wrapper;

using FLOAT = typename model::type::Value_Type;
using Array = py::array_t<FLOAT, py::array::c_style | py::array::forcecast>;
using A_Type =
    PythonNumpy::DenseMatrix_Type<FLOAT, model::STATE_SIZE, model::STATE_SIZE>;
using B_Type =
    PythonNumpy::DenseMatrix_Type<FLOAT, model::STATE_SIZE, model::INPUT_SIZE>;
using C_Type =
    PythonNumpy::DenseMatrix_Type<FLOAT, model::OUTPUT_SIZE, model::STATE_SIZE>;
using D_Type =
    PythonNumpy::DenseMatrix_Type<FLOAT, model::OUTPUT_SIZE, model::INPUT_SIZE>;
using Plant_Type =
    PythonControl::DiscreteStateSpace_Type<A_Type, B_Type, C_Type, D_Type>;
using Input_Type =
    PythonControl::StateSpaceInput_Type<FLOAT, model::INPUT_SIZE>;
using Output_Type =
    PythonControl::StateSpaceOutput_Type<FLOAT, model::OUTPUT_SIZE>;

model::type controller;
Plant_Type plant;
Input_Type previous_input;

template <typename Matrix>
Matrix from_array(const Array &array, const char *name) {
  const auto info = array.request();
  if (info.ndim != 2 ||
      info.shape[0] != static_cast<py::ssize_t>(Matrix::ROWS) ||
      info.shape[1] != static_cast<py::ssize_t>(Matrix::COLS)) {
    throw py::value_error(std::string(name) + " must have shape (" +
                          std::to_string(Matrix::ROWS) + ", " +
                          std::to_string(Matrix::COLS) + ").");
  }
  const auto *values = static_cast<const FLOAT *>(info.ptr);
  Matrix result;
  for (std::size_t row = 0; row < Matrix::ROWS; ++row) {
    for (std::size_t column = 0; column < Matrix::COLS; ++column) {
      result(row, column) = values[row * Matrix::COLS + column];
    }
  }
  return result;
}

template <typename Matrix> py::array_t<FLOAT> to_array(const Matrix &matrix) {
  py::array_t<FLOAT> result({static_cast<py::ssize_t>(Matrix::ROWS),
                             static_cast<py::ssize_t>(Matrix::COLS)});
  for (std::size_t row = 0; row < Matrix::ROWS; ++row) {
    for (std::size_t column = 0; column < Matrix::COLS; ++column) {
      result.mutable_at(row, column) = matrix(row, column);
    }
  }
  return result;
}

void initialize(const Array &a, const Array &b, const Array &c, const Array &d,
                FLOAT dt) {
  controller = model::make();
  plant = PythonControl::make_DiscreteStateSpace(
      from_array<A_Type>(a, "A"), from_array<B_Type>(b, "B"),
      from_array<C_Type>(c, "C"), from_array<D_Type>(d, "D"), dt);
  previous_input = Input_Type{};
}

py::array_t<FLOAT> update(const Array &reference, const Array &measurement) {
  return to_array(controller.update(
      from_array<model::Reference_Type>(reference, "reference"),
      from_array<Output_Type>(measurement, "measurement")));
}

Output_Type advance_plant(const Input_Type &input) {
  plant.update(input);
  // update() stores Y before advancing X; the demo measures the new X.
  return plant.output_equation(plant.get_X(), input);
}

py::tuple plant_step(const Array &input) {
  const auto measurement =
      advance_plant(from_array<Input_Type>(input, "input"));
  return py::make_tuple(to_array(plant.get_X()), to_array(measurement));
}

py::tuple closed_loop_step(const Array &reference) {
  const auto measurement = advance_plant(previous_input);
  previous_input = controller.update(
      from_array<model::Reference_Type>(reference, "reference"), measurement);
  return py::make_tuple(to_array(previous_input), to_array(plant.get_X()),
                        to_array(measurement));
}

PYBIND11_MODULE(InstantMpcSIL, module) {
  module.def("initialize", &initialize, "Reset the controller and plant.");
  module.def("update", &update, "Update using an identical SIL measurement.");
  module.def("plant_step", &plant_step,
             "Apply the previous input and measure.");
  module.def("closed_loop_step", &closed_loop_step,
             "Advance the independent C++ closed loop.");
}
