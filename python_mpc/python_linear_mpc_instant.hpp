/**
 * @file python_linear_mpc_instant.hpp
 * @brief Offline-designed instant MPC for discrete-time LTI models.
 *
 * Python prepares the prediction/equality-constraint projection, inequality
 * matrices, unconstrained initialization maps and semi-implicit inverse.
 * The generated C++ factory supplies those coefficients to InstantMPC_LTI.
 * Each update estimates the state with a fixed Kalman gain and performs the
 * configured number of primal-dual substeps; it does not solve a QP to a
 * convergence tolerance or invert matrices online.
 *
 * The decision vector contains input increments followed by augmented states
 * [delta_x; y - reference] over the prediction horizon. The final first input
 * increment is added to the previous control input. Bounds influence the dual
 * update, but the returned input is not clipped and finite substeps do not
 * establish strict constraint satisfaction or a general stability guarantee.
 */
#ifndef PYTHON_LINEAR_MPC_INSTANT_HPP_
#define PYTHON_LINEAR_MPC_INSTANT_HPP_

#include "base_utility.hpp"
#include "python_control.hpp"
#include "python_numpy.hpp"

#include <functional>
#include <tuple>
#include <type_traits>

namespace PythonMPC {

/** Reference changes larger than this infinity-norm tolerance reset mu. */
static constexpr double IMPC_REFERENCE_CHANGED_TOL = 1e-6;
/** Minimum signed denominator magnitude in the dual non-negativity update. */
static constexpr double IMPC_SAFE_GP_EPS = 1e-30;

/** Internal fixed-size operations shared by the iMPC update branches. */
namespace LMPC_Instant_Operation {

/**
 * @brief Extrapolates the delayed state estimate and forms an output
 * correction.
 *
 * @tparam Number_Of_Delay Positive measurement-delay length in samples.
 * @tparam X_Type State column-vector type.
 * @tparam Y_Type Output column-vector type.
 * @tparam Y_Store_Type Ring buffer of predicted outputs.
 * @tparam LKF_Type Kalman filter providing get_x_hat_without_delay().
 * @param X_in Delayed estimate; the filter provides the extrapolated state.
 * @param Y_in Delayed measured output.
 * @param[in,out] Y_store Predicted-output history, advanced once per call.
 * @param kalman_filter Filter with input history for delay compensation.
 * @return Extrapolated state and corrected output. InstantMPC_LTI::update uses
 * only the state; its output error is formed from C * X_compensated.
 */
template <std::size_t Number_Of_Delay, typename X_Type, typename Y_Type,
          typename Y_Store_Type, typename LKF_Type>
inline typename std::enable_if<(Number_Of_Delay > 0),
                               std::tuple<X_Type, Y_Type>>::type
compensate_X_Y_delay(const X_Type &X_in, const Y_Type &Y_in,
                     Y_Store_Type &Y_store, LKF_Type &kalman_filter) {

  static_cast<void>(X_in);

  Y_Type Y_measured = Y_in;

  X_Type X_out = kalman_filter.get_x_hat_without_delay();
  auto Y_raw = kalman_filter.state_space.C * X_out;

  Y_Type Y;
  PythonNumpy::substitute_matrix(Y, Y_raw);

  Y_store.push(Y);
  auto Y_diff = Y_measured - Y_store.get();

  Y_Type Y_out = Y + Y_diff;

  return std::make_tuple(X_out, Y_out);
}

/**
 * @brief Returns the estimate and measurement unchanged when delay is zero.
 * @tparam Number_Of_Delay Zero for this overload.
 * @param X_in Current estimated state.
 * @param Y_in Current measured output.
 * @param Y_store Unused in the zero-delay branch.
 * @param kalman_filter Unused in the zero-delay branch.
 * @return Copies of X_in and Y_in; the history is not advanced.
 */
template <std::size_t Number_Of_Delay, typename X_Type, typename Y_Type,
          typename Y_Store_Type, typename LKF_Type>
inline typename std::enable_if<(Number_Of_Delay == 0),
                               std::tuple<X_Type, Y_Type>>::type
compensate_X_Y_delay(const X_Type &X_in, const Y_Type &Y_in,
                     Y_Store_Type &Y_store, LKF_Type &kalman_filter) {

  static_cast<void>(Y_store);
  static_cast<void>(kalman_filter);

  return std::make_tuple(X_in, Y_in);
}

/** Elementwise max(x, matrix), traversing positive matrix dimensions. */
namespace MaximumScalarMatrixOperation {

/** @brief Visits columns of one row; column zero terminates recursion. */
template <typename T, typename Out_Type, typename In_M_Type, std::size_t M,
          std::size_t N, std::size_t I, std::size_t J_idx>
struct Row {
  static void compute(Out_Type &Out, const T &x, const In_M_Type &Matrix) {

    Out.template set<I, J_idx>(x > Matrix.template get<I, J_idx>()
                                   ? x
                                   : Matrix.template get<I, J_idx>());

    Row<T, Out_Type, In_M_Type, M, N, I, (J_idx - 1)>::compute(Out, x, Matrix);
  }
};

/** @brief Visits columns of one row; column zero terminates recursion. */
template <typename T, typename Out_Type, typename In_M_Type, std::size_t M,
          std::size_t N, std::size_t I>
struct Row<T, Out_Type, In_M_Type, M, N, I, 0> {
  static void compute(Out_Type &Out, const T &x, const In_M_Type &Matrix) {

    Out.template set<I, 0>(
        x > Matrix.template get<I, 0>() ? x : Matrix.template get<I, 0>());
  }
};

/** @brief Visits rows in descending order; row zero terminates recursion. */
template <typename T, typename Out_Type, typename In_M_Type, std::size_t M,
          std::size_t N, std::size_t I_idx>
struct Column {
  static void compute(Out_Type &Out, const T &x, const In_M_Type &Matrix) {
    Row<T, Out_Type, In_M_Type, M, N, I_idx, (N - 1)>::compute(Out, x, Matrix);
    Column<T, Out_Type, In_M_Type, M, N, (I_idx - 1)>::compute(Out, x, Matrix);
  }
};

/** @brief Visits rows in descending order; row zero terminates recursion. */
template <typename T, typename Out_Type, typename In_M_Type, std::size_t M,
          std::size_t N>
struct Column<T, Out_Type, In_M_Type, M, N, 0> {
  static void compute(Out_Type &Out, const T &x, const In_M_Type &Matrix) {
    Row<T, Out_Type, In_M_Type, M, N, 0, (N - 1)>::compute(Out, x, Matrix);
  }
};

/**
 * @brief Writes all entries of Out as max(x, Matrix).
 * @param[out] Out Result with the same positive dimensions as Matrix.
 * @param x Scalar comparison operand.
 * @param Matrix Input matrix; template indices address its stored elements.
 */
template <typename T, typename Out_Type, typename In_M_Type>
inline void compute(Out_Type &Out, const T &x, const In_M_Type &Matrix) {
  constexpr std::size_t M = In_M_Type::ROWS;
  constexpr std::size_t N = In_M_Type::COLS;

  Column<T, Out_Type, In_M_Type, M, N, (M - 1)>::compute(Out, x, Matrix);
}

} // namespace MaximumScalarMatrixOperation

/**
 * @brief Returns elementwise max(x, M), retaining M's storage type.
 * @param x Scalar lower limit; the dual update passes zero.
 * @param M Matrix to limit. The iMPC caller uses a dense dual vector.
 * @return A new matrix; M is not modified.
 */
template <typename T, typename Matrix_Type>
auto maximum(const T &x, const Matrix_Type &M) -> typename std::enable_if<
    std::is_convertible<T, typename Matrix_Type::Value_Type>::value,
    Matrix_Type>::type {

  Matrix_Type result;

  MaximumScalarMatrixOperation::compute(result, x, M);

  return result;
}

/** Elementwise maximum of equally sized matrices; output is dense. */
namespace MaximumMatrixMatrixOperation {

/** @brief Visits columns of one row; column zero terminates recursion. */
template <typename Out_Type, typename In_A_Type, typename In_B_Type,
          std::size_t M, std::size_t N, std::size_t I, std::size_t J_idx>
struct Row {
  static void compute(Out_Type &Out, const In_A_Type &A, const In_B_Type &B) {

    Out.template set<I, J_idx>(A.template get<I, J_idx>() >
                                       B.template get<I, J_idx>()
                                   ? A.template get<I, J_idx>()
                                   : B.template get<I, J_idx>());

    Row<Out_Type, In_A_Type, In_B_Type, M, N, I, (J_idx - 1)>::compute(Out, A,
                                                                       B);
  }
};

/** @brief Visits columns of one row; column zero terminates recursion. */
template <typename Out_Type, typename In_A_Type, typename In_B_Type,
          std::size_t M, std::size_t N, std::size_t I>
struct Row<Out_Type, In_A_Type, In_B_Type, M, N, I, 0> {
  static void compute(Out_Type &Out, const In_A_Type &A, const In_B_Type &B) {

    Out.template set<I, 0>(A.template get<I, 0>() > B.template get<I, 0>()
                               ? A.template get<I, 0>()
                               : B.template get<I, 0>());
  }
};

/** @brief Visits rows in descending order; row zero terminates recursion. */
template <typename Out_Type, typename In_A_Type, typename In_B_Type,
          std::size_t M, std::size_t N, std::size_t I_idx>
struct Column {
  static void compute(Out_Type &Out, const In_A_Type &A, const In_B_Type &B) {
    Row<Out_Type, In_A_Type, In_B_Type, M, N, I_idx, (N - 1)>::compute(Out, A,
                                                                       B);
    Column<Out_Type, In_A_Type, In_B_Type, M, N, (I_idx - 1)>::compute(Out, A,
                                                                       B);
  }
};

/** @brief Visits rows in descending order; row zero terminates recursion. */
template <typename Out_Type, typename In_A_Type, typename In_B_Type,
          std::size_t M, std::size_t N>
struct Column<Out_Type, In_A_Type, In_B_Type, M, N, 0> {
  static void compute(Out_Type &Out, const In_A_Type &A, const In_B_Type &B) {
    Row<Out_Type, In_A_Type, In_B_Type, M, N, 0, (N - 1)>::compute(Out, A, B);
  }
};

/**
 * @brief Writes elementwise maxima into an equally sized output matrix.
 * @param[out] Out Destination.
 * @param A First input matrix.
 * @param B Second input matrix.
 */
template <typename Out_Type, typename In_A_Type, typename In_B_Type>
inline void compute(Out_Type &Out, const In_A_Type &A, const In_B_Type &B) {
  constexpr std::size_t M = In_A_Type::ROWS;
  constexpr std::size_t N = In_A_Type::COLS;

  Column<Out_Type, In_A_Type, In_B_Type, M, N, (M - 1)>::compute(Out, A, B);
}

} // namespace MaximumMatrixMatrixOperation

/**
 * @brief Returns a dense elementwise maximum of equal-sized matrices.
 * @param A First operand.
 * @param B Second operand with matching dimensions and scalar type.
 * @return Dense matrix containing max(A(i,j), B(i,j)).
 */
template <typename Matrix_A_Type, typename Matrix_B_Type>
auto maximum(const Matrix_A_Type &A, const Matrix_B_Type &B)
    -> PythonNumpy::DenseMatrix_Type<typename Matrix_A_Type::Value_Type,
                                     Matrix_A_Type::ROWS, Matrix_A_Type::COLS> {

  static_assert(std::is_same<typename Matrix_A_Type::Value_Type,
                             typename Matrix_B_Type::Value_Type>::value,
                "Matrix A and B must have the same value type.");

  static_assert(Matrix_A_Type::ROWS == Matrix_B_Type::ROWS &&
                    Matrix_A_Type::COLS == Matrix_B_Type::COLS,
                "Matrix A and B must have the same dimensions.");

  PythonNumpy::DenseMatrix_Type<typename Matrix_A_Type::Value_Type,
                                Matrix_A_Type::ROWS, Matrix_A_Type::COLS>
      result;

  MaximumMatrixMatrixOperation::compute(result, A, B);

  return result;
}

/** Tiles U_A - U_B into one NP-by-INPUT_SIZE block of bg. */
namespace FillBgBoundOperation {

/**
 * @brief Writes one bound-difference block for prediction step K_idx.
 * @tparam ROW_OFFSET First row of this constraint group in bg.
 * @tparam J_idx Current input component, visited down to zero.
 */
template <std::size_t NP, std::size_t INPUT_SIZE, std::size_t ROW_OFFSET,
          std::size_t K_idx, std::size_t J_idx, typename Bg_Type,
          typename U_A_Type, typename U_B_Type>
struct Inner {
  static void compute(Bg_Type &bg, const U_A_Type &U_A, const U_B_Type &U_B) {

    bg(ROW_OFFSET + K_idx * INPUT_SIZE + J_idx, 0) =
        U_A(J_idx, 0) - U_B(J_idx, 0);

    Inner<NP, INPUT_SIZE, ROW_OFFSET, K_idx, (J_idx - 1), Bg_Type, U_A_Type,
          U_B_Type>::compute(bg, U_A, U_B);
  }
};

/** @brief Writes input component zero and terminates the inner recursion. */
template <std::size_t NP, std::size_t INPUT_SIZE, std::size_t ROW_OFFSET,
          std::size_t K_idx, typename Bg_Type, typename U_A_Type,
          typename U_B_Type>
struct Inner<NP, INPUT_SIZE, ROW_OFFSET, K_idx, 0, Bg_Type, U_A_Type,
             U_B_Type> {
  static void compute(Bg_Type &bg, const U_A_Type &U_A, const U_B_Type &U_B) {

    bg(ROW_OFFSET + K_idx * INPUT_SIZE + 0, 0) = U_A(0, 0) - U_B(0, 0);
  }
};

/**
 * @brief Repeats the bound difference over prediction steps K_idx down to zero.
 * @tparam NP Prediction horizon; each step contributes INPUT_SIZE rows.
 */
template <std::size_t NP, std::size_t INPUT_SIZE, std::size_t ROW_OFFSET,
          std::size_t K_idx, typename Bg_Type, typename U_A_Type,
          typename U_B_Type>
struct Outer {
  static void compute(Bg_Type &bg, const U_A_Type &U_A, const U_B_Type &U_B) {
    Inner<NP, INPUT_SIZE, ROW_OFFSET, K_idx, (INPUT_SIZE - 1), Bg_Type,
          U_A_Type, U_B_Type>::compute(bg, U_A, U_B);
    Outer<NP, INPUT_SIZE, ROW_OFFSET, (K_idx - 1), Bg_Type, U_A_Type,
          U_B_Type>::compute(bg, U_A, U_B);
  }
};

/** @brief Writes prediction step zero and terminates the outer recursion. */
template <std::size_t NP, std::size_t INPUT_SIZE, std::size_t ROW_OFFSET,
          typename Bg_Type, typename U_A_Type, typename U_B_Type>
struct Outer<NP, INPUT_SIZE, ROW_OFFSET, 0, Bg_Type, U_A_Type, U_B_Type> {
  static void compute(Bg_Type &bg, const U_A_Type &U_A, const U_B_Type &U_B) {
    Inner<NP, INPUT_SIZE, ROW_OFFSET, 0, (INPUT_SIZE - 1), Bg_Type, U_A_Type,
          U_B_Type>::compute(bg, U_A, U_B);
  }
};

/**
 * @brief Replaces one bound group with NP copies of U_A - U_B.
 * @param[in,out] bg Full inequality-bound column vector.
 * @param U_A Minuend input-sized vector.
 * @param U_B Subtrahend input-sized vector.
 */
template <std::size_t NP, std::size_t INPUT_SIZE, std::size_t ROW_OFFSET,
          typename Bg_Type, typename U_A_Type, typename U_B_Type>
inline void compute(Bg_Type &bg, const U_A_Type &U_A, const U_B_Type &U_B) {
  Outer<NP, INPUT_SIZE, ROW_OFFSET, (NP - 1), Bg_Type, U_A_Type,
        U_B_Type>::compute(bg, U_A, U_B);
}

} // namespace FillBgBoundOperation

/**
 * @brief Writes NP copies of U_max - U_latest into the active upper-U block.
 * @param[in,out] bg Inequality-bound vector; other groups are left unchanged.
 * @param U_max Per-input absolute upper limits.
 * @param U_latest Input returned on the previous control step.
 */
template <bool UseUUB, std::size_t NP, std::size_t INPUT_SIZE,
          std::size_t U_UB_ROW_OFFSET, typename Bg_Type, typename U_Max_Type,
          typename U_Type>
inline typename std::enable_if<UseUUB, void>::type
update_bg_u_ub(Bg_Type &bg, const U_Max_Type &U_max, const U_Type &U_latest) {

  FillBgBoundOperation::compute<NP, INPUT_SIZE, U_UB_ROW_OFFSET>(bg, U_max,
                                                                 U_latest);
}

/**
 * @brief Updates U upper-bound rows of bg (no-op when UseUUB = false).
 */
template <bool UseUUB, std::size_t NP, std::size_t INPUT_SIZE,
          std::size_t U_UB_ROW_OFFSET, typename Bg_Type, typename U_Max_Type,
          typename U_Type>
inline typename std::enable_if<!UseUUB, void>::type
update_bg_u_ub(Bg_Type &bg, const U_Max_Type &U_max, const U_Type &U_latest) {
  static_cast<void>(bg);
  static_cast<void>(U_max);
  static_cast<void>(U_latest);
}

/**
 * @brief Writes NP copies of U_latest - U_min into the active lower-U block.
 * @param[in,out] bg Inequality-bound vector; other groups are left unchanged.
 * @param U_min Per-input absolute lower limits.
 * @param U_latest Input returned on the previous control step.
 */
template <bool UseULB, std::size_t NP, std::size_t INPUT_SIZE,
          std::size_t U_LB_ROW_OFFSET, typename Bg_Type, typename U_Min_Type,
          typename U_Type>
inline typename std::enable_if<UseULB, void>::type
update_bg_u_lb(Bg_Type &bg, const U_Min_Type &U_min, const U_Type &U_latest) {

  FillBgBoundOperation::compute<NP, INPUT_SIZE, U_LB_ROW_OFFSET>(bg, U_latest,
                                                                 U_min);
}

/**
 * @brief Updates U lower-bound rows of bg (no-op when UseULB = false).
 */
template <bool UseULB, std::size_t NP, std::size_t INPUT_SIZE,
          std::size_t U_LB_ROW_OFFSET, typename Bg_Type, typename U_Min_Type,
          typename U_Type>
inline typename std::enable_if<!UseULB, void>::type
update_bg_u_lb(Bg_Type &bg, const U_Min_Type &U_min, const U_Type &U_latest) {
  static_cast<void>(bg);
  static_cast<void>(U_min);
  static_cast<void>(U_latest);
}

/**
 * @brief Updates time-varying rows of bg (no-op when IsNoConstraints = true).
 */
template <bool IsNoConstraints, bool UseUUB, bool UseULB, std::size_t NP,
          std::size_t INPUT_SIZE, std::size_t U_UB_ROW_OFFSET,
          std::size_t U_LB_ROW_OFFSET, typename Bg_Type, typename U_Max_Type,
          typename U_Min_Type, typename U_Type>
inline typename std::enable_if<IsNoConstraints, void>::type
update_bg(Bg_Type &bg, const U_Max_Type &U_max, const U_Min_Type &U_min,
          const U_Type &U_latest) {
  static_cast<void>(bg);
  static_cast<void>(U_max);
  static_cast<void>(U_min);
  static_cast<void>(U_latest);
}

/**
 * @brief Updates time-varying rows of bg (dispatches to UUB/ULB helpers
 *        when IsNoConstraints = false).
 *
 * Constraint blocks are ordered delta-U upper, delta-U lower, U upper and
 * U lower, omitting inactive groups. Delta-U bounds remain constant.
 */
template <bool IsNoConstraints, bool UseUUB, bool UseULB, std::size_t NP,
          std::size_t INPUT_SIZE, std::size_t U_UB_ROW_OFFSET,
          std::size_t U_LB_ROW_OFFSET, typename Bg_Type, typename U_Max_Type,
          typename U_Min_Type, typename U_Type>
inline typename std::enable_if<!IsNoConstraints, void>::type
update_bg(Bg_Type &bg, const U_Max_Type &U_max, const U_Min_Type &U_min,
          const U_Type &U_latest) {
  update_bg_u_ub<UseUUB, NP, INPUT_SIZE, U_UB_ROW_OFFSET>(bg, U_max, U_latest);
  update_bg_u_lb<UseULB, NP, INPUT_SIZE, U_LB_ROW_OFFSET>(bg, U_min, U_latest);
}

/** Projects the inequality residual onto the dual-variable tangent direction.
 */
namespace ComputeGpOperation {

/**
 * @brief Computes gp[i] element: g[i] when mu[i] > 0, else max(0, g[i]).
 *        Recursive template unrolls the loop at compile time (general case).
 */
template <typename T, typename Mu_Type, std::size_t NMU, std::size_t I_idx>
struct Compute {
  static void run(Mu_Type &gp, const Mu_Type &g, const Mu_Type &mu) {
    if (mu.template get<I_idx, 0>() > static_cast<T>(0)) {
      gp.template set<I_idx, 0>(g.template get<I_idx, 0>());
    } else {
      T g_val = g.template get<I_idx, 0>();
      gp.template set<I_idx, 0>(g_val > static_cast<T>(0) ? g_val
                                                          : static_cast<T>(0));
    }
    Compute<T, Mu_Type, NMU, (I_idx - 1)>::run(gp, g, mu);
  }
};

/**
 * @brief Base case: computes gp[0].
 */
template <typename T, typename Mu_Type, std::size_t NMU>
struct Compute<T, Mu_Type, NMU, 0> {
  static void run(Mu_Type &gp, const Mu_Type &g, const Mu_Type &mu) {
    if (mu.template get<0, 0>() > static_cast<T>(0)) {
      gp.template set<0, 0>(g.template get<0, 0>());
    } else {
      T g_val = g.template get<0, 0>();
      gp.template set<0, 0>(g_val > static_cast<T>(0) ? g_val
                                                      : static_cast<T>(0));
    }
  }
};

/**
 * @brief Writes the admissible dual-gradient direction for every constraint.
 * @param[out] gp g where mu > 0, otherwise max(0, g).
 * @param g Inequality residual Ag * wproj - bg.
 * @param mu Current nonnegative dual vector.
 */
template <typename T, typename Mu_Type>
inline void compute(Mu_Type &gp, const Mu_Type &g, const Mu_Type &mu) {
  constexpr std::size_t NMU = Mu_Type::ROWS;
  Compute<T, Mu_Type, NMU, (NMU - 1)>::run(gp, g, mu);
}

} // namespace ComputeGpOperation

/** Scales dual steps that would cross zero; eta is initialized to one. */
namespace ComputeEtaOperation {

/**
 * @brief Computes eta[i]: if (mu[i] + dtzeta_gp[i]) < 0, sets eta[i] to
 *        -mu[i] / avoid_zero_divide(dtzeta_gp[i], eps), otherwise leaves
 *        eta[i] unchanged (= 1).
 *        Recursive template unrolls the loop at compile time (general case).
 */
template <typename T, typename Mu_Type, std::size_t NMU, std::size_t I_idx>
struct Compute {
  static void run(Mu_Type &eta, const Mu_Type &mu, const Mu_Type &dtzeta_gp,
                  const T &eps) {
    if ((mu.template get<I_idx, 0>() + dtzeta_gp.template get<I_idx, 0>()) <
        static_cast<T>(0)) {
      eta.template set<I_idx, 0>(-mu.template get<I_idx, 0>() /
                                 Base::Utility::avoid_zero_divide(
                                     dtzeta_gp.template get<I_idx, 0>(), eps));
    }
    Compute<T, Mu_Type, NMU, (I_idx - 1)>::run(eta, mu, dtzeta_gp, eps);
  }
};

/**
 * @brief Base case: computes eta[0].
 */
template <typename T, typename Mu_Type, std::size_t NMU>
struct Compute<T, Mu_Type, NMU, 0> {
  static void run(Mu_Type &eta, const Mu_Type &mu, const Mu_Type &dtzeta_gp,
                  const T &eps) {
    if ((mu.template get<0, 0>() + dtzeta_gp.template get<0, 0>()) <
        static_cast<T>(0)) {
      eta.template set<0, 0>(-mu.template get<0, 0>() /
                             Base::Utility::avoid_zero_divide(
                                 dtzeta_gp.template get<0, 0>(), eps));
    }
  }
};

/**
 * @brief Adjusts only the dual components that would become negative.
 * @param[in,out] eta Vector initialized to ones by the caller.
 * @param mu Current dual vector.
 * @param dtzeta_gp Proposed unconstrained dual increment.
 * @note The signed denominator guard uses IMPC_SAFE_GP_EPS. This limits the
 * dual step, not the physical control input.
 */
template <typename T, typename Mu_Type>
inline void compute(Mu_Type &eta, const Mu_Type &mu, const Mu_Type &dtzeta_gp) {
  constexpr std::size_t NMU = Mu_Type::ROWS;
  Compute<T, Mu_Type, NMU, (NMU - 1)>::run(eta, mu, dtzeta_gp,
                                           static_cast<T>(IMPC_SAFE_GP_EPS));
}

} // namespace ComputeEtaOperation

/** Builds reference-gradient corrections in the state portion of w. */
namespace ComputeQTrajectoryOperation {

/**
 * @brief Extracts reference(:, K) - ref for one prediction step.
 * @tparam K Zero-based trajectory column; J_idx descends over output rows.
 */
template <std::size_t OUTPUT_SIZE, std::size_t K, std::size_t J_idx,
          typename Y_Type, typename Ref_Type>
struct FillDrK {
  static void run(Y_Type &dr_k, const Ref_Type &reference, const Y_Type &ref) {
    dr_k.template set<J_idx, 0>(reference.template get<J_idx, K>() -
                                ref.template get<J_idx, 0>());
    FillDrK<OUTPUT_SIZE, K, (J_idx - 1), Y_Type, Ref_Type>::run(dr_k, reference,
                                                                ref);
  }
};

/** @brief Writes output component zero and terminates extraction. */
template <std::size_t OUTPUT_SIZE, std::size_t K, typename Y_Type,
          typename Ref_Type>
struct FillDrK<OUTPUT_SIZE, K, 0, Y_Type, Ref_Type> {
  static void run(Y_Type &dr_k, const Ref_Type &reference, const Y_Type &ref) {
    dr_k.template set<0, 0>(reference.template get<0, K>() -
                            ref.template get<0, 0>());
  }
};

/**
 * @brief Adds one gradient block at offset NU + K * NS_SIZE.
 * @tparam NU Number of input-increment entries preceding the state blocks.
 * @tparam I_idx Augmented-state component, descending to zero.
 */
template <std::size_t NS_SIZE, std::size_t NU, std::size_t K, std::size_t I_idx,
          typename W_Type, typename XS_Type>
struct Accumulate {
  static void run(W_Type &q_trajectory, const XS_Type &WCzT_dr_k) {
    q_trajectory.template set<NU + K * NS_SIZE + I_idx, 0>(
        q_trajectory.template get<NU + K * NS_SIZE + I_idx, 0>() +
        WCzT_dr_k.template get<I_idx, 0>());
    Accumulate<NS_SIZE, NU, K, (I_idx - 1), W_Type, XS_Type>::run(q_trajectory,
                                                                  WCzT_dr_k);
  }
};

/** @brief Adds augmented-state component zero and terminates accumulation. */
template <std::size_t NS_SIZE, std::size_t NU, std::size_t K, typename W_Type,
          typename XS_Type>
struct Accumulate<NS_SIZE, NU, K, 0, W_Type, XS_Type> {
  static void run(W_Type &q_trajectory, const XS_Type &WCzT_dr_k) {
    q_trajectory.template set<NU + K * NS_SIZE + 0, 0>(
        q_trajectory.template get<NU + K * NS_SIZE + 0, 0>() +
        WCzT_dr_k.template get<0, 0>());
  }
};

/** @brief Fills gradient blocks for trajectory columns K_idx down to zero. */
template <std::size_t NP, std::size_t OUTPUT_SIZE, std::size_t NS_SIZE,
          std::size_t NU, std::size_t K_idx, typename W_Type, typename Y_Type,
          typename Ref_Type, typename WCzT_Qk_Type>
struct Outer {
  static void run(W_Type &q_trajectory, const Ref_Type &reference,
                  const Y_Type &ref, const WCzT_Qk_Type &WCzT_Qk) {
    Y_Type dr_k;
    FillDrK<OUTPUT_SIZE, K_idx, (OUTPUT_SIZE - 1), Y_Type, Ref_Type>::run(
        dr_k, reference, ref);
    auto WCzT_dr_k = WCzT_Qk * dr_k;
    using XS_Type = decltype(WCzT_dr_k);
    Accumulate<NS_SIZE, NU, K_idx, (NS_SIZE - 1), W_Type, XS_Type>::run(
        q_trajectory, WCzT_dr_k);
    Outer<NP, OUTPUT_SIZE, NS_SIZE, NU, (K_idx - 1), W_Type, Y_Type, Ref_Type,
          WCzT_Qk_Type>::run(q_trajectory, reference, ref, WCzT_Qk);
  }
};

/** @brief Fills the first trajectory block and terminates recursion. */
template <std::size_t NP, std::size_t OUTPUT_SIZE, std::size_t NS_SIZE,
          std::size_t NU, typename W_Type, typename Y_Type, typename Ref_Type,
          typename WCzT_Qk_Type>
struct Outer<NP, OUTPUT_SIZE, NS_SIZE, NU, 0, W_Type, Y_Type, Ref_Type,
             WCzT_Qk_Type> {
  static void run(W_Type &q_trajectory, const Ref_Type &reference,
                  const Y_Type &ref, const WCzT_Qk_Type &WCzT_Qk) {
    Y_Type dr_k;
    FillDrK<OUTPUT_SIZE, 0, (OUTPUT_SIZE - 1), Y_Type, Ref_Type>::run(
        dr_k, reference, ref);
    auto WCzT_dr_k = WCzT_Qk * dr_k;
    using XS_Type = decltype(WCzT_dr_k);
    Accumulate<NS_SIZE, NU, 0, (NS_SIZE - 1), W_Type, XS_Type>::run(
        q_trajectory, WCzT_dr_k);
  }
};

/**
 * @brief Accumulates all trajectory-gradient corrections.
 * @param[in,out] q_trajectory Zero-initialized Nw-by-1 vector. Input blocks
 * stay zero; each augmented-state block receives WCzT_Qk * (reference(:,k) -
 * ref).
 * @param reference Ny-by-NP trajectory, with one prediction step per column.
 * @param ref First reference column used as the augmented-state offset.
 * @param WCzT_Qk Offline Ns-by-Ny weighted-output gradient map.
 */
template <std::size_t NP, std::size_t OUTPUT_SIZE, std::size_t NS_SIZE,
          std::size_t NU, typename W_Type, typename Y_Type, typename Ref_Type,
          typename WCzT_Qk_Type>
inline void compute(W_Type &q_trajectory, const Ref_Type &reference,
                    const Y_Type &ref, const WCzT_Qk_Type &WCzT_Qk) {
  Outer<NP, OUTPUT_SIZE, NS_SIZE, NU, (NP - 1), W_Type, Y_Type, Ref_Type,
        WCzT_Qk_Type>::run(q_trajectory, reference, ref, WCzT_Qk);
}

} // namespace ComputeQTrajectoryOperation

/** Infinity-norm reference difference used to detect a base-reference change.
 */
namespace MaxAbsDiffOperation {

/**
 * @brief Recursively computes max(max_diff, |A[I_idx,0] - B[I_idx,0]|)
 *        across all elements (general case).
 */
template <typename T, typename Vec_Type, std::size_t N, std::size_t I_idx>
struct Compute {
  static T run(const Vec_Type &A, const Vec_Type &B, const T &max_diff) {
    T d = A.template get<I_idx, 0>() - B.template get<I_idx, 0>();
    if (d < static_cast<T>(0)) {
      d = -d;
    }
    T new_max = (d > max_diff) ? d : max_diff;
    return Compute<T, Vec_Type, N, (I_idx - 1)>::run(A, B, new_max);
  }
};

/**
 * @brief Base case: computes |A[0,0] - B[0,0]| and returns max with
 * max_diff.
 */
template <typename T, typename Vec_Type, std::size_t N>
struct Compute<T, Vec_Type, N, 0> {
  static T run(const Vec_Type &A, const Vec_Type &B, const T &max_diff) {
    T d = A.template get<0, 0>() - B.template get<0, 0>();
    if (d < static_cast<T>(0)) {
      d = -d;
    }
    return (d > max_diff) ? d : max_diff;
  }
};

/**
 * @brief Returns the maximum absolute difference of two column vectors.
 * @param A Current base reference.
 * @param B Last reference accepted by the reset detector.
 * @return Infinity norm of A - B; vectors must have a positive row count.
 */
template <typename T, typename Vec_Type>
inline T compute(const Vec_Type &A, const Vec_Type &B) {
  constexpr std::size_t N = Vec_Type::ROWS;
  return Compute<T, Vec_Type, N, (N - 1)>::run(A, B, static_cast<T>(0));
}

} // namespace MaxAbsDiffOperation

/** Integrates the first N decision-vector entries into the input vector. */
namespace AddAssignColumnOperation {

/**
 * @brief Recursively computes A[I_idx, 0] += B[I_idx, 0] (general case).
 */
template <typename Vec_A_Type, typename Vec_B_Type, std::size_t N,
          std::size_t I_idx>
struct Compute {
  static void run(Vec_A_Type &A, const Vec_B_Type &B) {
    A.template set<I_idx, 0>(A.template get<I_idx, 0>() +
                             B.template get<I_idx, 0>());
    Compute<Vec_A_Type, Vec_B_Type, N, (I_idx - 1)>::run(A, B);
  }
};

/**
 * @brief Base case: computes A[0, 0] += B[0, 0].
 */
template <typename Vec_A_Type, typename Vec_B_Type, std::size_t N>
struct Compute<Vec_A_Type, Vec_B_Type, N, 0> {
  static void run(Vec_A_Type &A, const Vec_B_Type &B) {
    A.template set<0, 0>(A.template get<0, 0>() + B.template get<0, 0>());
  }
};

/**
 * @brief Adds the first N rows of B to the input vector A, in place.
 * @tparam N Number of applied input channels, not the entire horizon length.
 * @param[in,out] A Previous input, replaced by the new absolute input.
 * @param B Projected decision vector whose first rows are delta-U.
 * @note No saturation or constraint projection is performed here.
 */
template <std::size_t N, typename Vec_A_Type, typename Vec_B_Type>
inline void compute(Vec_A_Type &A, const Vec_B_Type &B) {
  Compute<Vec_A_Type, Vec_B_Type, N, (N - 1)>::run(A, B);
}

} // namespace AddAssignColumnOperation

} // namespace LMPC_Instant_Operation

/**
 * @brief Instant Model Predictive Control (iMPC) for discrete-time LTI
 * systems.
 *
 * This class implements an iMPC algorithm that performs a fixed number of
 * sub-stepped primal-dual gradient flow updates per control step using a
 * semi-implicit Euler discretization. Equality constraints enter through an
 * affine projection K*w + L*xt; inequality residuals drive the dual update.
 * Only the dual variable is explicitly kept nonnegative. Finite substeps and
 * rounded offline matrices do not guarantee exact primal feasibility.
 *
 * Nx, Nu and Ny are plant state, input and output sizes. Ns = Nx + Ny and
 * Nw = NP * (Nu + Ns). The first NP*Nu entries are input increments, followed
 * by NP augmented-state blocks. NC is represented by offline equality rows
 * fixing later increments to zero; it does not reduce Nw.
 *
 * Each call rebuilds the primal starting point from the unconstrained maps.
 * The dual state carries over unless the first reference column differs from
 * the stored reset reference by more than IMPC_REFERENCE_CHANGED_TOL.
 * A fresh generated make() resets the
 * filter and controller state; copying this class preserves the current state.
 *
 * The precomputed matrices (K, L, q_K_matrix, w_unc_map, w_unc_traj_map,
 * M_sub_inv, Ag, bg, AgKT, WCzT_Qk) are passed directly to the constructor
 * and must be computed offline from the Python InstantMPC_LTI.__init__.
 *
 * @tparam NP_In Prediction horizon in samples, positive.
 * @tparam NC_In Control horizon used to construct the offline equalities.
 * @tparam LKF_Type_In Type of the fixed-gain linear Kalman filter.
 * @tparam ReferenceTrajectory_Type
 *           Reference type: DenseMatrix_Type<T,Ny,1> for point reference,
 *           DenseMatrix_Type<T,Ny,Np> for trajectory reference.
 * @tparam Delta_U_Min_Type   Type for delta-U lower bound (or
 * SparseMatrixEmpty).
 * @tparam Delta_U_Max_Type   Type for delta-U upper bound (or
 * SparseMatrixEmpty).
 * @tparam U_Min_Type         Type for U lower bound (or SparseMatrixEmpty).
 * @tparam U_Max_Type         Type for U upper bound (or SparseMatrixEmpty).
 * @tparam WCzT_Qk_Type       Type of the WCzT_Qk matrix  (Ns x Ny).
 * @tparam Ag_Type            Type of the constraint matrix Ag  (Nmu x Nw).
 * @tparam Bg_Type            Type of the constraint bound bg  (Nmu x 1).
 * @tparam K_Type             Type of the null-space projector K  (Nw x Nw).
 * @tparam L_Type             Type of the affine offset L  (Nw x Ns).
 * @tparam q_K_matrix_Type    Type of the projected gradient map  (Nw x Ns).
 * @tparam W_Unc_Map_Type Unconstrained KKT initialization map (Nw x Ns).
 * @tparam W_Unc_Traj_Map_Type Trajectory correction for that map (Nw x Nw).
 * @tparam M_Sub_Inv_Type Precomputed (I + h*K*P*K) inverse, h=dtzeta_sub.
 * @tparam AgKT_Type Type of (Ag*K)^T (Nw x Nmu).
 * @note These formulas describe the offline construction before coefficient
 * rounding. The constructor copies the prepared arrays without recomputing
 * them, checking convergence, or certifying their conditioning.
 */
template <std::size_t NP_In, std::size_t NC_In, typename LKF_Type_In,
          typename ReferenceTrajectory_Type, typename Delta_U_Min_Type,
          typename Delta_U_Max_Type, typename U_Min_Type, typename U_Max_Type,
          typename WCzT_Qk_Type, typename Ag_Type, typename Bg_Type,
          typename K_Type, typename L_Type, typename q_K_matrix_Type,
          typename W_Unc_Map_Type, typename W_Unc_Traj_Map_Type,
          typename M_Sub_Inv_Type, typename AgKT_Type>
class InstantMPC_LTI {
private:
  /* Type */
  using T_ = typename LKF_Type_In::Value_Type;

public:
  /* Type */
  using Value_Type = T_;
  using LKF_Type = LKF_Type_In;

public:
  /* Sizes */
  static constexpr std::size_t INPUT_SIZE = LKF_Type::INPUT_SIZE;
  static constexpr std::size_t OUTPUT_SIZE = LKF_Type::OUTPUT_SIZE;
  static constexpr std::size_t STATE_SIZE = LKF_Type::STATE_SIZE;
  static constexpr std::size_t NUMBER_OF_DELAY = LKF_Type::NUMBER_OF_DELAY;

  static constexpr std::size_t NS_SIZE = STATE_SIZE + OUTPUT_SIZE;
  static constexpr std::size_t NW_SIZE = K_Type::ROWS;
  static constexpr std::size_t NMU_SIZE = Bg_Type::ROWS;

  static constexpr std::size_t NP = NP_In;
  static constexpr std::size_t NC = NC_In;

public:
  /* Flags */

  /**
   * @brief True when all constraint bound types are empty (no constraints).
   */
  static constexpr bool IS_NO_CONSTRAINTS =
      std::is_same<Delta_U_Min_Type, PythonNumpy::SparseMatrixEmpty_Type<
                                         T_, INPUT_SIZE, 1>>::value &&
      std::is_same<Delta_U_Max_Type, PythonNumpy::SparseMatrixEmpty_Type<
                                         T_, INPUT_SIZE, 1>>::value &&
      std::is_same<U_Min_Type, PythonNumpy::SparseMatrixEmpty_Type<
                                   T_, INPUT_SIZE, 1>>::value &&
      std::is_same<U_Max_Type, PythonNumpy::SparseMatrixEmpty_Type<
                                   T_, INPUT_SIZE, 1>>::value;

  /**
   * @brief True when the reference is a trajectory
   *        (ReferenceTrajectory_Type::COLS > 1).
   */
  static constexpr bool IS_REFERENCE_TRAJECTORY =
      (ReferenceTrajectory_Type::COLS > 1);

private:
  /* Active constraint flags derived from template types */
  static constexpr bool _USE_DU_UB = !std::is_same<
      Delta_U_Max_Type,
      PythonNumpy::SparseMatrixEmpty_Type<T_, INPUT_SIZE, 1>>::value;

  static constexpr bool _USE_DU_LB = !std::is_same<
      Delta_U_Min_Type,
      PythonNumpy::SparseMatrixEmpty_Type<T_, INPUT_SIZE, 1>>::value;

  static constexpr bool _USE_U_UB = !std::is_same<
      U_Max_Type,
      PythonNumpy::SparseMatrixEmpty_Type<T_, INPUT_SIZE, 1>>::value;

  static constexpr bool _USE_U_LB = !std::is_same<
      U_Min_Type,
      PythonNumpy::SparseMatrixEmpty_Type<T_, INPUT_SIZE, 1>>::value;

  /* Row offsets in _bg for each constraint block */
  static constexpr std::size_t _DU_UB_ROW_OFFSET = 0;
  static constexpr std::size_t _DU_LB_ROW_OFFSET =
      _DU_UB_ROW_OFFSET + (_USE_DU_UB ? NP * INPUT_SIZE : 0);
  static constexpr std::size_t _U_UB_ROW_OFFSET =
      _DU_LB_ROW_OFFSET + (_USE_DU_LB ? NP * INPUT_SIZE : 0);
  static constexpr std::size_t _U_LB_ROW_OFFSET =
      _U_UB_ROW_OFFSET + (_USE_U_UB ? NP * INPUT_SIZE : 0);

private:
  /* Check Compatibility */
  static_assert(
      Delta_U_Max_Type::ROWS == INPUT_SIZE,
      "Delta_U_Max_Type must have the same number of rows as INPUT_SIZE.");

  static_assert(
      Delta_U_Max_Type::ROWS == Delta_U_Min_Type::ROWS &&
          Delta_U_Max_Type::COLS == Delta_U_Min_Type::COLS,
      "Delta_U_Max_Type and Delta_U_Min_Type must have the same dimensions.");

  static_assert(
      Delta_U_Min_Type::ROWS == U_Min_Type::ROWS &&
          Delta_U_Min_Type::COLS == U_Min_Type::COLS,
      "Delta_U_Min_Type and U_Min_Type must have the same dimensions.");

  static_assert(U_Max_Type::ROWS == U_Min_Type::ROWS &&
                    U_Max_Type::COLS == U_Min_Type::COLS,
                "U_Max_Type and U_Min_Type must have the same dimensions.");

  static_assert(K_Type::ROWS == K_Type::COLS,
                "K_Type must be a square matrix (Nw x Nw).");

  static_assert(K_Type::ROWS == NW_SIZE,
                "K_Type must have NW_SIZE cols and rows (Python shape).");

  static_assert(L_Type::ROWS == NW_SIZE,
                "L_Type must have NW_SIZE cols (Python shape).");

  static_assert(L_Type::COLS == NS_SIZE,
                "L_Type must have NS_SIZE rows (Python shape).");

private:
  /* Internal types */
  using U_Type_ = PythonControl::StateSpaceInput_Type<T_, INPUT_SIZE>;
  using X_Type_ = PythonControl::StateSpaceState_Type<T_, STATE_SIZE>;
  using Y_Type_ = PythonControl::StateSpaceOutput_Type<T_, OUTPUT_SIZE>;
  using Y_Store_Type_ =
      PythonControl::DelayedVectorObject<Y_Type_, NUMBER_OF_DELAY>;

  /* Augmented initial state: xt = [dx; y_dev] (Ns x 1) */
  using Xt_Type_ = PythonNumpy::DenseMatrix_Type<T_, NS_SIZE, 1>;

  /* Decision variable w (Nw x 1) */
  using W_Type_ = PythonNumpy::DenseMatrix_Type<T_, NW_SIZE, 1>;

  /* Dual variable mu (Nmu x 1) */
  using Mu_Type_ = PythonNumpy::DenseMatrix_Type<T_, NMU_SIZE, 1>;

  /* w_unc_map, w_unc_traj_map, M_sub_inv use template-parameter types */
  using W_Unc_Map_Type_ = W_Unc_Map_Type;
  using W_Unc_Traj_Map_Type_ = W_Unc_Traj_Map_Type;
  using M_Sub_Inv_Type_ = M_Sub_Inv_Type;

public:
  /* Constructor */

  /**
   * @brief Creates default matrices and zero controller state.
   * @note This is useful as an assignment target. It is not a designed
   * controller: use the generated make() before calling update().
   */
  InstantMPC_LTI()
      : _kalman_filter(), _WCzT_Qk(), _delta_U_min(), _delta_U_max(), _U_min(),
        _U_max(), _Ag(), _bg(), _N_sub(1), _dtzeta_sub(static_cast<T_>(0)),
        _K(), _L(), _q_K_matrix(), _w_unc_map(), _w_unc_traj_map(),
        _M_sub_inv(), _AgKT(), _X_inner_model(), _U_latest(), _Y_store(), _w(),
        _mu(), _ref_prev() {}

  /**
   * @brief Construct with all precomputed matrices.
   *
   * All matrices and h=dtzeta_sub must come from the same offline design.
   * Runtime inverse/projection construction is deliberately avoided.
   *
   * @param kalman_filter Linear Kalman filter with the offline fixed gain.
   * @param WCzT_Qk           Precomputed WCzT_Qk matrix  (Ns x Ny).
   * @param delta_U_min       Delta-U lower bound (or SparseMatrixEmpty).
   * @param delta_U_max       Delta-U upper bound (or SparseMatrixEmpty).
   * @param U_min             U lower bound (or SparseMatrixEmpty).
   * @param U_max             U upper bound (or SparseMatrixEmpty).
   * @param Ag                Inequality constraint matrix Ag  (Nmu x Nw).
   * @param bg                Constraint bound bg (Nmu x 1); U-bound rows
   *                          are updated every step.
   * @param N_sub Positive fixed number of substeps per control step.
   * @param dtzeta_sub        Sub-step size (dt * zeta / N_sub).
   * @param K                 Null-space projector K  (Nw x Nw).
   * @param L                 Affine offset L = -C_eq^T (C_eq C_eq^T)^{-1}
   * D_eq  (Nw x Ns).
   * @param q_K_matrix        Projected gradient map K P L  (Nw x Ns).
   * @param w_unc_map         Unconstrained-QP warm-start map  (Nw x Ns).
   * @param w_unc_traj_map    Trajectory warm-start correction  (Nw x Nw).
   * @param M_sub_inv Semi-implicit inverse (I + dtzeta_sub*K*P*K)^{-1}
   * (Nw x Nw).
   * @param AgKT              (Ag K)^T  (Nw x Nmu).
   */
  InstantMPC_LTI(const LKF_Type &kalman_filter, const WCzT_Qk_Type &WCzT_Qk,
                 const Delta_U_Min_Type &delta_U_min,
                 const Delta_U_Max_Type &delta_U_max, const U_Min_Type &U_min,
                 const U_Max_Type &U_max, const Ag_Type &Ag, const Bg_Type &bg,
                 const std::size_t &N_sub, const T_ &dtzeta_sub,
                 const K_Type &K, const L_Type &L,
                 const q_K_matrix_Type &q_K_matrix,
                 const W_Unc_Map_Type_ &w_unc_map,
                 const W_Unc_Traj_Map_Type_ &w_unc_traj_map,
                 const M_Sub_Inv_Type_ &M_sub_inv, const AgKT_Type &AgKT)
      : _kalman_filter(kalman_filter), _WCzT_Qk(WCzT_Qk),
        _delta_U_min(delta_U_min), _delta_U_max(delta_U_max), _U_min(U_min),
        _U_max(U_max), _Ag(Ag), _bg(bg), _N_sub(N_sub), _dtzeta_sub(dtzeta_sub),
        _K(K), _L(L), _q_K_matrix(q_K_matrix), _w_unc_map(w_unc_map),
        _w_unc_traj_map(w_unc_traj_map), _M_sub_inv(M_sub_inv), _AgKT(AgKT),
        _X_inner_model(), _U_latest(), _Y_store(), _w(), _mu(), _ref_prev() {}

  /** @brief Copies coefficients, estimator history and current controller
   * state. */
  InstantMPC_LTI(const InstantMPC_LTI &other)
      : _kalman_filter(other._kalman_filter), _WCzT_Qk(other._WCzT_Qk),
        _delta_U_min(other._delta_U_min), _delta_U_max(other._delta_U_max),
        _U_min(other._U_min), _U_max(other._U_max), _Ag(other._Ag),
        _bg(other._bg), _N_sub(other._N_sub), _dtzeta_sub(other._dtzeta_sub),
        _K(other._K), _L(other._L), _q_K_matrix(other._q_K_matrix),
        _w_unc_map(other._w_unc_map), _w_unc_traj_map(other._w_unc_traj_map),
        _M_sub_inv(other._M_sub_inv), _AgKT(other._AgKT),
        _X_inner_model(other._X_inner_model), _U_latest(other._U_latest),
        _Y_store(other._Y_store), _w(other._w), _mu(other._mu),
        _ref_prev(other._ref_prev) {}

  /** @brief Replaces this controller with a full copy, including dynamic state.
   */
  InstantMPC_LTI &operator=(const InstantMPC_LTI &other) {
    if (this != &other) {
      this->_kalman_filter = other._kalman_filter;
      this->_WCzT_Qk = other._WCzT_Qk;
      this->_delta_U_min = other._delta_U_min;
      this->_delta_U_max = other._delta_U_max;
      this->_U_min = other._U_min;
      this->_U_max = other._U_max;
      this->_Ag = other._Ag;
      this->_bg = other._bg;
      this->_N_sub = other._N_sub;
      this->_dtzeta_sub = other._dtzeta_sub;
      this->_K = other._K;
      this->_L = other._L;
      this->_q_K_matrix = other._q_K_matrix;
      this->_w_unc_map = other._w_unc_map;
      this->_w_unc_traj_map = other._w_unc_traj_map;
      this->_M_sub_inv = other._M_sub_inv;
      this->_AgKT = other._AgKT;
      this->_X_inner_model = other._X_inner_model;
      this->_U_latest = other._U_latest;
      this->_Y_store = other._Y_store;
      this->_w = other._w;
      this->_mu = other._mu;
      this->_ref_prev = other._ref_prev;
    }
    return *this;
  }

  /** @brief Transfers coefficients and current state from another controller.
   */
  InstantMPC_LTI(InstantMPC_LTI &&other) noexcept
      : _kalman_filter(std::move(other._kalman_filter)),
        _WCzT_Qk(std::move(other._WCzT_Qk)),
        _delta_U_min(std::move(other._delta_U_min)),
        _delta_U_max(std::move(other._delta_U_max)),
        _U_min(std::move(other._U_min)), _U_max(std::move(other._U_max)),
        _Ag(std::move(other._Ag)), _bg(std::move(other._bg)),
        _N_sub(other._N_sub), _dtzeta_sub(other._dtzeta_sub),
        _K(std::move(other._K)), _L(std::move(other._L)),
        _q_K_matrix(std::move(other._q_K_matrix)),
        _w_unc_map(std::move(other._w_unc_map)),
        _w_unc_traj_map(std::move(other._w_unc_traj_map)),
        _M_sub_inv(std::move(other._M_sub_inv)), _AgKT(std::move(other._AgKT)),
        _X_inner_model(std::move(other._X_inner_model)),
        _U_latest(std::move(other._U_latest)),
        _Y_store(std::move(other._Y_store)), _w(std::move(other._w)),
        _mu(std::move(other._mu)), _ref_prev(std::move(other._ref_prev)) {}

  /** @brief Move-assigns the complete controller and estimator state. */
  InstantMPC_LTI &operator=(InstantMPC_LTI &&other) noexcept {
    if (this != &other) {
      this->_kalman_filter = std::move(other._kalman_filter);
      this->_WCzT_Qk = std::move(other._WCzT_Qk);
      this->_delta_U_min = std::move(other._delta_U_min);
      this->_delta_U_max = std::move(other._delta_U_max);
      this->_U_min = std::move(other._U_min);
      this->_U_max = std::move(other._U_max);
      this->_Ag = std::move(other._Ag);
      this->_bg = std::move(other._bg);
      this->_N_sub = other._N_sub;
      this->_dtzeta_sub = other._dtzeta_sub;
      this->_K = std::move(other._K);
      this->_L = std::move(other._L);
      this->_q_K_matrix = std::move(other._q_K_matrix);
      this->_w_unc_map = std::move(other._w_unc_map);
      this->_w_unc_traj_map = std::move(other._w_unc_traj_map);
      this->_M_sub_inv = std::move(other._M_sub_inv);
      this->_AgKT = std::move(other._AgKT);
      this->_X_inner_model = std::move(other._X_inner_model);
      this->_U_latest = std::move(other._U_latest);
      this->_Y_store = std::move(other._Y_store);
      this->_w = std::move(other._w);
      this->_mu = std::move(other._mu);
      this->_ref_prev = std::move(other._ref_prev);
    }
    return *this;
  }

public:
  /* Function */

  /**
   * @brief Run one iMPC step.
   *
   * Matches the interface of LTI_MPC::update(reference, Y). The previous
   * returned U is used for the Kalman prediction. Supply the measurement after
   * that input has acted for one sampling interval (or its configured delayed
   * counterpart), then apply the returned U for the next interval.
   *
   * The method forms xt=[x_hat-x_hat_previous; C*x_hat-ref], updates absolute-U
   * bound offsets and runs exactly N_sub primal-dual substeps. It initializes
   * w from the offline unconstrained maps on every call; stored mu is reused.
   * No QP convergence test, online inversion, or final input clipping occurs.
   *
   * @param reference  Reference output: DenseMatrix_Type<T,Ny,1> (point) or
   *                   DenseMatrix_Type<T,Ny,Np> (trajectory).
   * @param Y          Measured output vector (Ny x 1).
   * @return New absolute control input U (Nu x 1), also stored as U_latest.
   * @note The object is stateful. Replaying the same measurement twice advances
   * its estimator and dual state twice. Reference changes reset w/mu, not the
   * filter, previous-state estimate, or accumulated input.
   */
  inline auto update(const ReferenceTrajectory_Type &reference,
                     const Y_Type_ &Y) -> U_Type_ {

    /* 1. Kalman filter predict + update */
    this->_kalman_filter.predict_and_update_with_fixed_G(this->_U_latest, Y);

    /* 2. State estimate and delay compensation */
    X_Type_ X = this->_kalman_filter.get_x_hat();

    X_Type_ X_compensated;
    Y_Type_ Y_compensated;
    std::tie(X_compensated, Y_compensated) = this->_compensate_X_Y_delay(X, Y);

    static_cast<void>(Y_compensated);

    /* 3. Extract base reference (first column for trajectory, direct otherwise)
     */
    Y_Type_ ref = this->_extract_ref(reference);

    /* 4. Augmented initial state  xt = [dx; y_dev] */
    auto dx = X_compensated - this->_X_inner_model;

    auto C_X = this->_kalman_filter.state_space.C * X_compensated;
    auto y_dev = C_X - ref;
    Xt_Type_ xt = PythonNumpy::concatenate_vertically(dx, y_dev);

    /* 5. Reset dual / primal variables on reference change */
    T_ max_diff = LMPC_Instant_Operation::MaxAbsDiffOperation::compute<T_>(
        ref, this->_ref_prev);

    if (max_diff > static_cast<T_>(IMPC_REFERENCE_CHANGED_TOL)) {
      this->_w = PythonNumpy::make_DenseMatrixZeros<T_, NW_SIZE, 1>();
      this->_mu = PythonNumpy::make_DenseMatrixZeros<T_, NMU_SIZE, 1>();
      this->_ref_prev = ref;
    }

    /* 6. Update time-varying rows of _bg (absolute U bounds) */
    this->_update_bg();

    /* 7. Projected gradient offset: q_K = q_K_matrix * xt */
    auto q_K = this->_q_K_matrix * xt;

    /* 8. Trajectory linear correction (trajectory mode only) */
    W_Type_ q_trajectory = this->_compute_q_trajectory(reference, ref);

    /* 9. Effective projected gradient including trajectory correction */
    W_Type_ q_K_eff = this->_compute_q_K_eff(q_K, q_trajectory);

    /* 10. Warm-start w from unconstrained QP solution */
    W_Type_ w = this->_compute_w_warmstart(xt, q_trajectory);

    /* 11. Precompute L * xt (constant across sub-steps) */
    auto Lxt = this->_L * xt;

    /* 12. iMPC primal-dual sub-steps */
    Mu_Type_ mu = this->_mu;

    for (std::size_t sub = 0; sub < this->_N_sub; ++sub) {

      /* Affine projection: wproj = K * w + L * xt */
      auto wproj = this->_K * w + Lxt;

      /* Constraint gradient and dual update (skip when no constraints) */
      Mu_Type_ mu_new;
      Mu_Type_ mu_times_eta;

      std::tie(mu_new, mu_times_eta) = this->_update_mu(wproj, mu);

      /* Semi-implicit w update:
       *   rhs = w - dtzeta_sub * (q_K_eff [+ AgKT * mu_times_eta])
       *   w_new = K * M_sub_inv * rhs
       */
      auto w_new = this->_compute_w_new(w, q_K_eff, mu_times_eta);

      w = w_new;
      mu = mu_new;
    }

    /* 13. Final projection and compute u */
    auto wproj_final = this->_K * w + Lxt;

    LMPC_Instant_Operation::AddAssignColumnOperation::compute<INPUT_SIZE>(
        this->_U_latest, wproj_final);

    /* 14. Store updated iMPC state */
    this->_w = w;
    this->_mu = mu;
    this->_X_inner_model = X_compensated;

    return this->_U_latest;
  }

  /** @brief Returns a copy of the last compensated state estimate used in dx.
   */
  inline auto get_X_inner_model() const -> X_Type_ {
    return this->_X_inner_model;
  }

  /** @brief Returns a copy of the most recently computed absolute input. */
  inline auto get_U_latest() const -> U_Type_ { return this->_U_latest; }

  /** @brief Returns a copy of the delay-compensation output-history buffer. */
  inline auto get_Y_store() const -> Y_Store_Type_ { return this->_Y_store; }

private:
  /** @brief Rewrites only active absolute-U bound groups using U_latest. */
  inline void _update_bg(void) {
    LMPC_Instant_Operation::update_bg<IS_NO_CONSTRAINTS, _USE_U_UB, _USE_U_LB,
                                      NP, INPUT_SIZE, _U_UB_ROW_OFFSET,
                                      _U_LB_ROW_OFFSET>(
        this->_bg, this->_U_max, this->_U_min, this->_U_latest);
  }

  /** @brief Uses the Ny-by-1 point reference directly. */
  template <bool IsTrajectory = IS_REFERENCE_TRAJECTORY>
  inline typename std::enable_if<!IsTrajectory, Y_Type_>::type
  _extract_ref(const ReferenceTrajectory_Type &reference) const {

    return reference;
  }

  /**
   * @brief Extracts reference(:,0) as the base offset and reset reference.
   * @note PythonNumpy::get_row<0> names a stored column in this matrix API.
   */
  template <bool IsTrajectory = IS_REFERENCE_TRAJECTORY>
  inline typename std::enable_if<IsTrajectory, Y_Type_>::type
  _extract_ref(const ReferenceTrajectory_Type &reference) const {
    Y_Type_ ref;

    ref = PythonNumpy::get_row<0>(reference);

    return ref;
  }

  /** @brief Returns a zero gradient correction for a point reference. */
  template <bool IsTrajectory = IS_REFERENCE_TRAJECTORY>
  inline typename std::enable_if<!IsTrajectory, W_Type_>::type
  _compute_q_trajectory(const ReferenceTrajectory_Type &reference,
                        const Y_Type_ &ref) const {
    static_cast<void>(reference);
    static_cast<void>(ref);
    return W_Type_();
  }

  /** @brief Accumulates weighted reference deviations into the state blocks. */
  template <bool IsTrajectory = IS_REFERENCE_TRAJECTORY>
  inline typename std::enable_if<IsTrajectory, W_Type_>::type
  _compute_q_trajectory(const ReferenceTrajectory_Type &reference,
                        const Y_Type_ &ref) const {
    /* q_trajectory[NU + k*Ns : NU + (k+1)*Ns] += WCzT_Qk * (ref_k - ref)
     * for k = 0 .. NP-1                                                    */
    W_Type_ q_trajectory;
    constexpr std::size_t NU = NP * INPUT_SIZE;

    LMPC_Instant_Operation::ComputeQTrajectoryOperation::compute<
        NP, OUTPUT_SIZE, NS_SIZE, NU>(q_trajectory, reference, ref,
                                      this->_WCzT_Qk);

    return q_trajectory;
  }

  /** @brief Rebuilds w from the regularized unconstrained KKT map and xt. */
  template <bool IsTrajectory = IS_REFERENCE_TRAJECTORY>
  inline typename std::enable_if<!IsTrajectory, W_Type_>::type
  _compute_w_warmstart(const Xt_Type_ &xt, const W_Type_ &q_trajectory) const {
    static_cast<void>(q_trajectory);
    return this->_w_unc_map * xt;
  }

  /** @brief Adds the offline trajectory correction to the fresh primal guess.
   */
  template <bool IsTrajectory = IS_REFERENCE_TRAJECTORY>
  inline typename std::enable_if<IsTrajectory, W_Type_>::type
  _compute_w_warmstart(const Xt_Type_ &xt, const W_Type_ &q_trajectory) const {
    return this->_w_unc_map * xt + this->_w_unc_traj_map * q_trajectory;
  }

  /** @brief Keeps the projected gradient offset for a point reference. */
  template <bool IsTrajectory = IS_REFERENCE_TRAJECTORY>
  inline typename std::enable_if<!IsTrajectory, W_Type_>::type
  _compute_q_K_eff(const W_Type_ &q_K, const W_Type_ &q_trajectory) const {

    static_cast<void>(q_trajectory);
    return q_K;
  }

  /** @brief Subtracts K*q_trajectory from the projected gradient offset. */
  template <bool IsTrajectory = IS_REFERENCE_TRAJECTORY>
  inline typename std::enable_if<IsTrajectory, W_Type_>::type
  _compute_q_K_eff(const W_Type_ &q_K, const W_Type_ &q_trajectory) const {

    return q_K - this->_K * q_trajectory;
  }

  /** @brief Applies the semi-implicit primal step without an inequality term.
   */
  template <bool NoConstraints = IS_NO_CONSTRAINTS>
  inline typename std::enable_if<NoConstraints, W_Type_>::type
  _compute_w_new(const W_Type_ &w, const W_Type_ &q_K_eff,
                 const Mu_Type_ &mu_times_eta) const {
    static_cast<void>(mu_times_eta);
    auto rhs = w - this->_dtzeta_sub * q_K_eff;
    return this->_K * (this->_M_sub_inv * rhs);
  }

  /**
   * @brief Applies K*M_sub_inv*(w - h*(q_K_eff + AgKT*mu_times_eta)).
   * @param w Current substep primal vector.
   * @param q_K_eff Gradient offset, constant across this control step.
   * @param mu_times_eta Previous substep mu multiplied elementwise by eta.
   */
  template <bool NoConstraints = IS_NO_CONSTRAINTS>
  inline typename std::enable_if<!NoConstraints, W_Type_>::type
  _compute_w_new(const W_Type_ &w, const W_Type_ &q_K_eff,
                 const Mu_Type_ &mu_times_eta) const {
    auto AgKT_mu_eta = this->_AgKT * mu_times_eta;
    auto rhs = w - this->_dtzeta_sub * (q_K_eff + AgKT_mu_eta);
    return this->_K * (this->_M_sub_inv * rhs);
  }

  /** @brief Preserves the dummy dual vector and returns a zero coupling term.
   */
  template <bool NoConstraints = IS_NO_CONSTRAINTS>
  inline typename std::enable_if<NoConstraints,
                                 std::tuple<Mu_Type_, Mu_Type_>>::type
  _update_mu(const W_Type_ &wproj, const Mu_Type_ &mu) const {
    static_cast<void>(wproj);
    Mu_Type_ mu_times_eta;
    return std::make_tuple(mu, mu_times_eta);
  }

  /**
   * @brief Updates mu using residual Ag*wproj-bg and the non-negativity guard.
   * @param wproj Affine-projected decision vector at the current substep.
   * @param mu Dual vector before this substep.
   * @return (mu_new, mu*eta). The primal step uses the old mu*eta, not mu_new.
   */
  template <bool NoConstraints = IS_NO_CONSTRAINTS>
  inline typename std::enable_if<!NoConstraints,
                                 std::tuple<Mu_Type_, Mu_Type_>>::type
  _update_mu(const W_Type_ &wproj, const Mu_Type_ &mu) const {

    Mu_Type_ g = this->_Ag * wproj - this->_bg;

    /* gp = max(0, g), but use g[i] when mu[i] > 0 (active constraint) */
    Mu_Type_ gp;
    LMPC_Instant_Operation::ComputeGpOperation::compute<T_>(gp, g, mu);

    /* dtzeta_gp = dtzeta_sub * gp */
    Mu_Type_ dtzeta_gp = this->_dtzeta_sub * gp;

    /* eta: safeguard against mu going negative */
    auto eta = PythonNumpy::make_DenseMatrixOnes<T_, NMU_SIZE, 1>();

    LMPC_Instant_Operation::ComputeEtaOperation::compute<T_>(eta, mu,
                                                             dtzeta_gp);

    /* mu_ = max(0, mu + dtzeta_gp * eta) */
    Mu_Type_ dtzeta_gp_eta;
    PythonNumpy::element_wise_multiply(dtzeta_gp_eta, dtzeta_gp, eta);
    Mu_Type_ mu_new =
        LMPC_Instant_Operation::maximum(static_cast<T_>(0), mu + dtzeta_gp_eta);

    /* mu_times_eta = mu * eta (element-wise) */
    Mu_Type_ mu_times_eta;
    PythonNumpy::element_wise_multiply(mu_times_eta, mu, eta);

    return std::make_tuple(mu_new, mu_times_eta);
  }

  /** @brief Dispatches delay compensation and advances history when enabled. */
  inline std::tuple<X_Type_, Y_Type_> _compensate_X_Y_delay(const X_Type_ &X,
                                                            const Y_Type_ &Y) {
    return LMPC_Instant_Operation::compensate_X_Y_delay<NUMBER_OF_DELAY>(
        X, Y, this->_Y_store, this->_kalman_filter);
  }

private:
  /* Variables */

  /* Kalman filter for state estimation */
  LKF_Type _kalman_filter;

  /* Offline coefficients and fixed bounds. _bg's absolute-U rows are mutable;
   * _N_sub and _dtzeta_sub must stay consistent with _M_sub_inv. */
  WCzT_Qk_Type _WCzT_Qk;
  Delta_U_Min_Type _delta_U_min;
  Delta_U_Max_Type _delta_U_max;
  U_Min_Type _U_min;
  U_Max_Type _U_max;
  Ag_Type _Ag;
  Bg_Type _bg;
  std::size_t _N_sub;
  T_ _dtzeta_sub;
  K_Type _K;
  L_Type _L;
  q_K_matrix_Type _q_K_matrix;
  W_Unc_Map_Type_ _w_unc_map;
  W_Unc_Traj_Map_Type_ _w_unc_traj_map;
  M_Sub_Inv_Type_ _M_sub_inv;
  AgKT_Type _AgKT;

  /* Dynamic state. _w stores the last substep result but is reinitialized from
   * the unconstrained map next call; _mu persists until a reference reset. */
  X_Type_ _X_inner_model;
  U_Type_ _U_latest;
  Y_Store_Type_ _Y_store;
  W_Type_ _w;
  Mu_Type_ _mu;
  Y_Type_
      _ref_prev; // Initially zero; updated only when reference reset occurs.
};

/* make InstantMPC_LTI */

/**
 * @brief Constructs a controller from one consistent set of offline matrices.
 *
 * The arguments correspond to the InstantMPC_LTI constructor. This factory
 * performs no matrix design or online solve; the generated model namespace's
 * make() supplies these types and values for normal use.
 *
 * @tparam NP_In Prediction horizon used by the offline design.
 * @tparam NC_In Control horizon encoded in the equality constraints.
 * @param kalman_filter State estimator with an offline fixed gain.
 * @param WCzT_Qk Trajectory gradient map (Ns x Ny).
 * @param delta_U_min Increment lower bounds, or a structurally empty type.
 * @param delta_U_max Increment upper bounds, or a structurally empty type.
 * @param U_min Absolute-input lower bounds, or a structurally empty type.
 * @param U_max Absolute-input upper bounds, or a structurally empty type.
 * @param Ag Inequality coefficient matrix (Nmu x Nw).
 * @param bg Initial inequality bound vector.
 * @param N_sub Positive substep count.
 * @param dtzeta_sub Integration step dt*zeta/N_sub.
 * @param K Equality null-space projection.
 * @param L Equality particular-solution map.
 * @param q_K_matrix Projected objective offset map.
 * @param w_unc_map Unconstrained primal initialization map.
 * @param w_unc_traj_map Trajectory contribution to primal initialization.
 * @param M_sub_inv Precomputed semi-implicit inverse.
 * @param AgKT Projected inequality coupling (Ag*K)^T.
 * @return Controller initialized with zero MPC history and the supplied filter.
 */
template <std::size_t NP_In, std::size_t NC_In, typename LKF_Type,
          typename ReferenceTrajectory_Type, typename Delta_U_Min_Type,
          typename Delta_U_Max_Type, typename U_Min_Type, typename U_Max_Type,
          typename WCzT_Qk_Type, typename Ag_Type, typename Bg_Type,
          typename K_Type, typename L_Type, typename q_K_matrix_Type,
          typename W_Unc_Map_Type, typename W_Unc_Traj_Map_Type,
          typename M_Sub_Inv_Type, typename AgKT_Type>
inline auto make_InstantMPC_LTI(
    const LKF_Type &kalman_filter, const WCzT_Qk_Type &WCzT_Qk,
    const Delta_U_Min_Type &delta_U_min, const Delta_U_Max_Type &delta_U_max,
    const U_Min_Type &U_min, const U_Max_Type &U_max, const Ag_Type &Ag,
    const Bg_Type &bg, std::size_t N_sub,
    typename LKF_Type::Value_Type dtzeta_sub, const K_Type &K, const L_Type &L,
    const q_K_matrix_Type &q_K_matrix, const W_Unc_Map_Type &w_unc_map,
    const W_Unc_Traj_Map_Type &w_unc_traj_map, const M_Sub_Inv_Type &M_sub_inv,
    const AgKT_Type &AgKT)
    -> InstantMPC_LTI<NP_In, NC_In, LKF_Type, ReferenceTrajectory_Type,
                      Delta_U_Min_Type, Delta_U_Max_Type, U_Min_Type,
                      U_Max_Type, WCzT_Qk_Type, Ag_Type, Bg_Type, K_Type,
                      L_Type, q_K_matrix_Type, W_Unc_Map_Type,
                      W_Unc_Traj_Map_Type, M_Sub_Inv_Type, AgKT_Type> {

  return InstantMPC_LTI<NP_In, NC_In, LKF_Type, ReferenceTrajectory_Type,
                        Delta_U_Min_Type, Delta_U_Max_Type, U_Min_Type,
                        U_Max_Type, WCzT_Qk_Type, Ag_Type, Bg_Type, K_Type,
                        L_Type, q_K_matrix_Type, W_Unc_Map_Type,
                        W_Unc_Traj_Map_Type, M_Sub_Inv_Type, AgKT_Type>(
      kalman_filter, WCzT_Qk, delta_U_min, delta_U_max, U_min, U_max, Ag, bg,
      N_sub, dtzeta_sub, K, L, q_K_matrix, w_unc_map, w_unc_traj_map, M_sub_inv,
      AgKT);
}

/**
 * @brief Public alias used by generated headers for the complete controller
 * type. Template parameters and dimensions have the same meaning as
 * InstantMPC_LTI.
 */
template <std::size_t NP_In, std::size_t NC_In, typename LKF_Type,
          typename ReferenceTrajectory_Type, typename Delta_U_Min_Type,
          typename Delta_U_Max_Type, typename U_Min_Type, typename U_Max_Type,
          typename WCzT_Qk_Type, typename Ag_Type, typename Bg_Type,
          typename K_Type, typename L_Type, typename q_K_matrix_Type,
          typename W_Unc_Map_Type, typename W_Unc_Traj_Map_Type,
          typename M_Sub_Inv_Type, typename AgKT_Type>
using InstantMPC_LTI_Type =
    InstantMPC_LTI<NP_In, NC_In, LKF_Type, ReferenceTrajectory_Type,
                   Delta_U_Min_Type, Delta_U_Max_Type, U_Min_Type, U_Max_Type,
                   WCzT_Qk_Type, Ag_Type, Bg_Type, K_Type, L_Type,
                   q_K_matrix_Type, W_Unc_Map_Type, W_Unc_Traj_Map_Type,
                   M_Sub_Inv_Type, AgKT_Type>;

} // namespace PythonMPC

#endif // PYTHON_LINEAR_MPC_INSTANT_HPP_
