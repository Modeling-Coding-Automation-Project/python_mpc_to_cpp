#include "linear_mpc_SIL_wrapper.hpp"
#include "python_control.hpp"

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

using FLOAT = typename linear_mpc_SIL_wrapper::type::Value_Type;

constexpr std::size_t INPUT_SIZE = linear_mpc_SIL_wrapper::INPUT_SIZE;
constexpr std::size_t STATE_SIZE = linear_mpc_SIL_wrapper::STATE_SIZE;
constexpr std::size_t OUTPUT_SIZE = linear_mpc_SIL_wrapper::OUTPUT_SIZE;

using Ref_Type = typename linear_mpc_SIL_wrapper::Ref_Type;

linear_mpc_SIL_wrapper::type lmpc;

void initialize(void) { lmpc = linear_mpc_SIL_wrapper::make(); }

py::array_t<FLOAT> update(py::array_t<FLOAT> ref_in, py::array_t<FLOAT> Y_in) {

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
  auto U = lmpc.update(ref, Y);

  /* output U */
  py::array_t<FLOAT> result;
  result.resize({static_cast<int>(INPUT_SIZE), static_cast<int>(1)});

  for (std::size_t i = 0; i < INPUT_SIZE; ++i) {
    result.mutable_at(i, 0) = U.access(i, 0);
  }

  return result;
}

PYBIND11_MODULE(LinearMpcSIL, m) {
  m.def("initialize", &initialize, "initialize linear MPC");
  m.def("update", &update, "update MPC with ref and output");
}
