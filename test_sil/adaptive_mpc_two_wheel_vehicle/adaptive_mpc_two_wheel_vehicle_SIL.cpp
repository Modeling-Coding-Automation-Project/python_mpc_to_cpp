#include "adaptive_mpc_two_wheel_vehicle_SIL_wrapper.hpp"

#include "two_wheel_vehicle_model_SIL_ada_mpc_ekf_parameter.hpp"

#include "python_control.hpp"

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

using FLOAT =
    typename adaptive_mpc_two_wheel_vehicle_SIL_wrapper::type::Value_Type;

constexpr std::size_t INPUT_SIZE =
    adaptive_mpc_two_wheel_vehicle_SIL_wrapper::INPUT_SIZE;
constexpr std::size_t STATE_SIZE =
    adaptive_mpc_two_wheel_vehicle_SIL_wrapper::STATE_SIZE;
constexpr std::size_t OUTPUT_SIZE =
    adaptive_mpc_two_wheel_vehicle_SIL_wrapper::OUTPUT_SIZE;

using Ref_Type = typename adaptive_mpc_two_wheel_vehicle_SIL_wrapper::Ref_Type;

adaptive_mpc_two_wheel_vehicle_SIL_wrapper::type ada_mpc;

void initialize(void) {
  ada_mpc = adaptive_mpc_two_wheel_vehicle_SIL_wrapper::make();
}

void update_parameters(FLOAT m) {

  two_wheel_vehicle_model_SIL_ada_mpc_ekf_parameter::Parameter_Type
      controller_parameters;
  controller_parameters.m = m;

  ada_mpc.update_parameters(controller_parameters);
}

py::array_t<FLOAT> update_manipulation(py::array_t<FLOAT> ref_in,
                                       py::array_t<FLOAT> Y_in) {

  py::buffer_info ref_info = ref_in.request();
  py::buffer_info Y_info = Y_in.request();

  /* check compatibility */
  if (OUTPUT_SIZE != ref_info.shape[0]) {
    throw std::runtime_error("ref must have " + std::to_string(OUTPUT_SIZE) +
                             " columns.");
  }

  if (OUTPUT_SIZE != Y_info.shape[0]) {
    throw std::runtime_error("Y must have " + std::to_string(OUTPUT_SIZE) +
                             " outputs.");
  }

  /* substitute */
  FLOAT *ref_data_ptr = static_cast<FLOAT *>(ref_info.ptr);
  Ref_Type ref;
  for (std::size_t i = 0; i < Ref_Type::COLS; ++i) {
    for (std::size_t j = 0; j < Ref_Type::ROWS; ++j) {
      ref.access(i, j) = ref_data_ptr[i * Ref_Type::ROWS + j];
    }
  }

  FLOAT *Y_data_ptr = static_cast<FLOAT *>(Y_info.ptr);
  PythonControl::StateSpaceOutput_Type<FLOAT, OUTPUT_SIZE> Y;
  for (std::size_t i = 0; i < OUTPUT_SIZE; ++i) {
    Y.access(i, 0) = Y_data_ptr[i];
  }

  /* update */
  auto U = ada_mpc.update_manipulation(ref, Y);

  /* output U */
  py::array_t<FLOAT> result;
  result.resize({static_cast<int>(INPUT_SIZE), static_cast<int>(1)});

  for (std::size_t i = 0; i < INPUT_SIZE; ++i) {
    result.mutable_at(i, 0) = U.access(i, 0);
  }

  return result;
}

PYBIND11_MODULE(AdaptiveMpcTwoWheelVehicleSIL, m) {
  py::class_<two_wheel_vehicle_model_SIL_ada_mpc_ekf_parameter::Parameter>(
      m, "Parameter")
      .def(py::init<>())
      .def_readwrite(
          "m", &two_wheel_vehicle_model_SIL_ada_mpc_ekf_parameter::Parameter::m)
      .def_readwrite(
          "l_f",
          &two_wheel_vehicle_model_SIL_ada_mpc_ekf_parameter::Parameter::l_f)
      .def_readwrite(
          "l_r",
          &two_wheel_vehicle_model_SIL_ada_mpc_ekf_parameter::Parameter::l_r)
      .def_readwrite(
          "I", &two_wheel_vehicle_model_SIL_ada_mpc_ekf_parameter::Parameter::I)
      .def_readwrite(
          "K_f",
          &two_wheel_vehicle_model_SIL_ada_mpc_ekf_parameter::Parameter::K_f)
      .def_readwrite(
          "K_r",
          &two_wheel_vehicle_model_SIL_ada_mpc_ekf_parameter::Parameter::K_r);

  m.def("initialize", &initialize, "initialize linear MPC");
  m.def("update_manipulation", &update_manipulation,
        "update MPC with ref and output");

  m.def(
      "update_parameters",
      [](const two_wheel_vehicle_model_SIL_ada_mpc_ekf_parameter::Parameter
             &param) { ada_mpc.update_parameters(param); },
      "update MPC parameters with Parameter struct");
}
