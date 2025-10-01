#include "nonlinear_mpc_kinematic_bicycle_model_SIL_wrapper.hpp"

#include "python_control.hpp"
#include <memory>
#include <stdexcept>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

using FLOAT = typename nonlinear_mpc_kinematic_bicycle_model_SIL_wrapper::
    EKF_Type::Value_Type;

constexpr std::size_t INPUT_SIZE =
    nonlinear_mpc_kinematic_bicycle_model_SIL_wrapper::INPUT_SIZE;
constexpr std::size_t STATE_SIZE =
    nonlinear_mpc_kinematic_bicycle_model_SIL_wrapper::STATE_SIZE;
constexpr std::size_t OUTPUT_SIZE =
    nonlinear_mpc_kinematic_bicycle_model_SIL_wrapper::OUTPUT_SIZE;

constexpr std::size_t NP =
    nonlinear_mpc_kinematic_bicycle_model_SIL_wrapper::NP;

using ReferenceTrajectory_Type =
    typename nonlinear_mpc_kinematic_bicycle_model_SIL_wrapper::
        ReferenceTrajectory_Type;

static std::unique_ptr<nonlinear_mpc_kinematic_bicycle_model_SIL_wrapper::type>
    nonlinear_mpc;

void initialize(void) {

  nonlinear_mpc.reset(
      new nonlinear_mpc_kinematic_bicycle_model_SIL_wrapper::type(
          nonlinear_mpc_kinematic_bicycle_model_SIL_wrapper::make()));

  nonlinear_mpc->set_solver_max_iteration(5);
}

py::array_t<FLOAT> update_manipulation(py::array_t<FLOAT> reference_trajectory,
                                       py::array_t<FLOAT> Y_in) {

  if (!nonlinear_mpc) {
    throw std::runtime_error(
        "SIL module is not initialized. Call initialize() first.");
  }

  // Expect a 2D reference_trajectory with shape (OUTPUT_SIZE, horizon)
  py::buffer_info ref_info = reference_trajectory.request();
  py::buffer_info Y_info = Y_in.request();

  if (ref_info.ndim != 2) {
    throw std::runtime_error("reference_trajectory must be 2-dimensional");
  }

  std::size_t ref_cols = static_cast<std::size_t>(ref_info.shape[0]);
  std::size_t ref_rows = static_cast<std::size_t>(ref_info.shape[1]);

  if (OUTPUT_SIZE != ref_cols) {
    throw std::runtime_error("reference_trajectory must have " +
                             std::to_string(OUTPUT_SIZE) + " cols (outputs).");
  }

  if (NP != ref_rows) {
    throw std::runtime_error("reference_trajectory must have " +
                             std::to_string(ReferenceTrajectory_Type::COLS) +
                             " columns (horizon).");
  }

  if (Y_info.ndim != 1 && Y_info.ndim != 2) {
    throw std::runtime_error(
        "Y must be a 1D or 2D array with OUTPUT_SIZE elements");
  }

  if (OUTPUT_SIZE != static_cast<std::size_t>(Y_info.shape[0])) {
    throw std::runtime_error("Y must have " + std::to_string(OUTPUT_SIZE) +
                             " outputs.");
  }

  /* substitute reference trajectory: assume C-contiguous row-major layout */
  FLOAT *ref_data_ptr = static_cast<FLOAT *>(ref_info.ptr);
  ReferenceTrajectory_Type ref;
  for (std::size_t col = 0; col < OUTPUT_SIZE; ++col) {
    for (std::size_t row = 0; row < NP; ++row) {

      ref.access(col, row) = ref_data_ptr[col * NP + row];
    }
  }

  FLOAT *Y_data_ptr = static_cast<FLOAT *>(Y_info.ptr);
  PythonControl::StateSpaceOutput_Type<FLOAT, OUTPUT_SIZE> Y;
  for (std::size_t i = 0; i < OUTPUT_SIZE; ++i) {
    Y.access(i, 0) = Y_data_ptr[i];
  }

  /* update */
  auto U = nonlinear_mpc->update_manipulation(ref, Y);

  /* output U */
  py::array_t<FLOAT> result;
  result.resize({static_cast<int>(INPUT_SIZE), static_cast<int>(1)});

  for (std::size_t i = 0; i < INPUT_SIZE; ++i) {
    result.mutable_at(i, 0) = U.access(i, 0);
  }

  return result;
}

PYBIND11_MODULE(NonlinearMpcKinematicBicycleModelSIL, m) {

  m.def("initialize", &initialize, "initialize nonlinear MPC");
  m.def("update_manipulation", &update_manipulation,
        "update nonlinear MPC with ref and output");
}
