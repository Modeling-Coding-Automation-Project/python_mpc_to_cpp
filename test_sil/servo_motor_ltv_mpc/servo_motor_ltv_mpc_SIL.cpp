#include "servo_motor_ltv_mpc_SIL_wrapper.hpp"

#include "servo_motor_LTV_SIL_parameters.hpp"

#include "python_control.hpp"

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

using FLOAT = typename servo_motor_ltv_mpc_SIL_wrapper::type::Value_Type;

constexpr std::size_t INPUT_SIZE = servo_motor_ltv_mpc_SIL_wrapper::INPUT_SIZE;
constexpr std::size_t STATE_SIZE = servo_motor_ltv_mpc_SIL_wrapper::STATE_SIZE;
constexpr std::size_t OUTPUT_SIZE =
    servo_motor_ltv_mpc_SIL_wrapper::OUTPUT_SIZE;

using Reference_Type = typename servo_motor_ltv_mpc_SIL_wrapper::Reference_Type;

servo_motor_ltv_mpc_SIL_wrapper::type lmpc;

void initialize(void) { lmpc = servo_motor_ltv_mpc_SIL_wrapper::make(); }

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
  Reference_Type ref;
  for (std::size_t i = 0; i < Reference_Type::COLS; ++i) {
    for (std::size_t j = 0; j < Reference_Type::ROWS; ++j) {
      ref.access(i, j) = ref_data_ptr[i * Reference_Type::ROWS + j];
    }
  }

  FLOAT *Y_data_ptr = static_cast<FLOAT *>(Y_info.ptr);
  PythonControl::StateSpaceOutput_Type<FLOAT, OUTPUT_SIZE> Y;
  for (std::size_t i = 0; i < OUTPUT_SIZE; ++i) {
    Y.access(i, 0) = Y_data_ptr[i];
  }

  /* update */
  auto U = lmpc.update_manipulation(ref, Y);

  /* output U */
  py::array_t<FLOAT> result;
  result.resize({static_cast<int>(INPUT_SIZE), static_cast<int>(1)});

  for (std::size_t i = 0; i < INPUT_SIZE; ++i) {
    result.mutable_at(i, 0) = U.access(i, 0);
  }

  return result;
}

PYBIND11_MODULE(ServoMotorLtvMpcSIL, m) {
  py::class_<servo_motor_LTV_SIL_parameters::Parameter>(m, "Parameter")
      .def(py::init<>())
      .def_readwrite("Lshaft",
                     &servo_motor_LTV_SIL_parameters::Parameter::Lshaft)
      .def_readwrite("dshaft",
                     &servo_motor_LTV_SIL_parameters::Parameter::dshaft)
      .def_readwrite("shaftrho",
                     &servo_motor_LTV_SIL_parameters::Parameter::shaftrho)
      .def_readwrite("G", &servo_motor_LTV_SIL_parameters::Parameter::G)
      .def_readwrite("Mmotor",
                     &servo_motor_LTV_SIL_parameters::Parameter::Mmotor)
      .def_readwrite("Rmotor",
                     &servo_motor_LTV_SIL_parameters::Parameter::Rmotor)
      .def_readwrite("Bmotor",
                     &servo_motor_LTV_SIL_parameters::Parameter::Bmotor)
      .def_readwrite("R", &servo_motor_LTV_SIL_parameters::Parameter::R)
      .def_readwrite("Kt", &servo_motor_LTV_SIL_parameters::Parameter::Kt)
      .def_readwrite("Bload",
                     &servo_motor_LTV_SIL_parameters::Parameter::Bload);

  m.def("initialize", &initialize, "initialize linear MPC");
  m.def("update_manipulation", &update_manipulation,
        "update MPC with ref and output");

  m.def(
      "update_parameters",
      [](const servo_motor_LTV_SIL_parameters::Parameter &param) {
        lmpc.update_parameters(param);
      },
      "update MPC parameters with Parameter struct");
}
