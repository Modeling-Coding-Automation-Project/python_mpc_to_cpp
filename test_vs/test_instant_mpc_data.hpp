/**
 * @file test_instant_mpc_data.hpp
 * @brief Small, analytically solvable prediction problems for iMPC tests.
 *
 * The fixtures use the actual scalar Kalman filter and rational precomputed
 * matrices. Expected controls follow from minimizing one- or two-variable
 * quadratics, independently of the controller's matrix implementation.
 */
#ifndef TEST_INSTANT_MPC_DATA_HPP_
#define TEST_INSTANT_MPC_DATA_HPP_

#include "python_linear_mpc_instant.hpp"

namespace InstantMPCTestData {

using namespace PythonNumpy;
using namespace PythonControl;

// A scalar plant with exact measurement correction (G=1). Keeping the real
// Kalman filter exercises its input history and the controller's delay adapter.
template <typename T, std::size_t Delay>
using Filter = LinearKalmanFilter_Type<
    DiscreteStateSpace_Type<
        DenseMatrix_Type<T, 1, 1>, DenseMatrix_Type<T, 1, 1>,
        DenseMatrix_Type<T, 1, 1>, DenseMatrix_Type<T, 1, 1>, Delay>,
    KalmanFilter_Q_Type<T, 1>, KalmanFilter_R_Type<T, 1>>;

template <typename T, std::size_t Delay>
Filter<T, Delay> make_filter(T a = T(1)) {
  auto system = make_DiscreteStateSpace<Delay>(
      make_DenseMatrix<1, 1>(a), make_DenseMatrix<1, 1>(T(1)),
      make_DenseMatrix<1, 1>(T(1)), make_DenseMatrix<1, 1>(T(0)), T(1));
  auto filter = make_LinearKalmanFilter(system, make_KalmanFilter_Q<1>(T(1)),
                                        make_KalmanFilter_R<1>(T(1)));
  filter.G = make_DenseMatrix<1, 1>(T(1));
  return filter;
}

template <typename T, bool Active> struct Bound;
template <typename T> struct Bound<T, true> {
  using Type = DenseMatrix_Type<T, 1, 1>;
  static Type make(T value) { return make_DenseMatrix<1, 1>(value); }
};
template <typename T> struct Bound<T, false> {
  using Type = SparseMatrixEmpty_Type<T, 1, 1>;
  static Type make(T) { return Type(); }
};

// One-step problem for x+=x+u, y=x, unit input/output costs. For
// z=[du,dx_next,y_error_next], the constraints are dx_next=dx+du and
// y_error_next=y_error+dx+du. Hence K=ones(3,3)/3 and the exact unconstrained
// answer is du=-(dx+y_error)/2. All coefficients below are rational.
// Kind: 0=no bounds, 1=du<=1/4, 2=u<=1/4, 3=u>=-1/4, 4=du>=-1/4.
template <typename T, int Kind, std::size_t Delay = 0> struct OneStep {
  using Ref = DenseMatrix_Type<T, 1, 1>;
  using Empty = SparseMatrixEmpty_Type<T, 1, 1>;
  using DuMin = typename Bound<T, Kind == 4>::Type;
  using DuMax = typename Bound<T, Kind == 1>::Type;
  using UMin = typename Bound<T, Kind == 3>::Type;
  using UMax = typename Bound<T, Kind == 2>::Type;
  using Controller = PythonMPC::InstantMPC_LTI_Type<
      1, 1, Filter<T, Delay>, Ref, DuMin, DuMax, UMin, UMax,
      DenseMatrix_Type<T, 2, 1>, DenseMatrix_Type<T, 1, 3>, Ref,
      DenseMatrix_Type<T, 3, 3>, DenseMatrix_Type<T, 3, 2>,
      DenseMatrix_Type<T, 3, 2>, DenseMatrix_Type<T, 3, 2>,
      SparseMatrixEmpty_Type<T, 3, 3>, DenseMatrix_Type<T, 3, 3>,
      DenseMatrix_Type<T, 3, 1>>;

  static Controller make(std::size_t substeps = 2) {
    const T third = T(1) * (T(1) / T(3));
    auto K = make_DenseMatrixOnes<T, 3, 3>() * third;
    auto L =
        make_DenseMatrix<3, 2>(T(-2), T(-1), T(1), T(-1), T(1), T(2)) * third;
    auto q = make_DenseMatrix<3, 2>(T(-1), T(1), T(-1), T(1), T(-1), T(1)) *
             (T(1) / T(9));
    auto warm = make_DenseMatrix<3, 2>(T(1), T(-1), T(1), T(-1), T(1), T(-1)) *
                (T(1) / T(6));
    auto inverse = make_DenseMatrix<3, 3>(T(5), T(-1), T(-1), T(-1), T(5),
                                          T(-1), T(-1), T(-1), T(5)) *
                   (T(1) / T(6));
    const T sign = Kind == 3 || Kind == 4 ? T(-1) : T(1);
    auto Ag = make_DenseMatrix<1, 3>(sign, T(0), T(0));
    auto AgKT = make_DenseMatrix<3, 1>(sign, sign, sign) * third;
    // Even with a nonzero Ag and an infeasible dummy bg, empty bound types
    // must select the unconstrained branch.
    auto bg = make_DenseMatrix<1, 1>(Kind == 0 ? T(-100) : T(0.25));
    return PythonMPC::make_InstantMPC_LTI<1, 1, Filter<T, Delay>, Ref>(
        make_filter<T, Delay>(), make_DenseMatrix<2, 1>(T(0), T(1)),
        Bound<T, Kind == 4>::make(T(-0.25)), Bound<T, Kind == 1>::make(T(0.25)),
        Bound<T, Kind == 3>::make(T(-0.25)), Bound<T, Kind == 2>::make(T(0.25)),
        Ag, bg, substeps, T(1.5), K, L, q, warm,
        SparseMatrixEmpty_Type<T, 3, 3>(), inverse, AgKT);
  }
};

// Two-step problem for x+=u, y=x. The augmented prediction equations are
// dx1=du0, e1=e0+du0, dx2=du1, e2=e1+du1. Decision order is
// [du0,du1,dx1,e1,dx2,e2], with unit du/e costs. Minimizing the resulting
// two-variable quadratic gives du0=(-3*e0 + (r1-r0))/5.
// These fixed matrices are exact fractions derived from those four equalities;
// no controller implementation is used to calculate the expected answers.
template <typename T, bool Trajectory> struct TwoStep {
  using Ref = DenseMatrix_Type<T, 1, Trajectory ? 2 : 1>;
  using Empty = SparseMatrixEmpty_Type<T, 1, 1>;
  using Controller = PythonMPC::InstantMPC_LTI_Type<
      2, 2, Filter<T, 0>, Ref, Empty, Empty, Empty, Empty,
      DenseMatrix_Type<T, 2, 1>, SparseMatrixEmpty_Type<T, 1, 6>,
      DenseMatrix_Type<T, 1, 1>, DenseMatrix_Type<T, 6, 6>,
      DenseMatrix_Type<T, 6, 2>, DenseMatrix_Type<T, 6, 2>,
      DenseMatrix_Type<T, 6, 2>, DenseMatrix_Type<T, 6, 6>,
      DenseMatrix_Type<T, 6, 6>, SparseMatrixEmpty_Type<T, 6, 1>>;

  static Controller make(std::size_t substeps = 3) {
    auto K = make_DenseMatrix<6, 6>(
                 T(3), T(-1), T(3), T(3), T(-1), T(2), T(-1), T(4), T(-1),
                 T(-1), T(4), T(3), T(3), T(-1), T(3), T(3), T(-1), T(2), T(3),
                 T(-1), T(3), T(3), T(-1), T(2), T(-1), T(4), T(-1), T(-1),
                 T(4), T(3), T(2), T(3), T(2), T(2), T(3), T(5)) *
             (T(1) / T(11));
    auto L = make_DenseMatrix<6, 2>(T(0), T(-5), T(0), T(-2), T(0), T(-5), T(0),
                                    T(6), T(0), T(-2), T(0), T(4)) *
             (T(1) / T(11));
    auto q = make_DenseMatrix<6, 2>(T(0), T(13), T(0), T(3), T(0), T(13), T(0),
                                    T(13), T(0), T(3), T(0), T(16)) *
             (T(1) / T(121));
    auto warm = make_DenseMatrix<6, 2>(T(0), T(-8), T(0), T(-1), T(0), T(-8),
                                       T(0), T(-8), T(0), T(-1), T(0), T(-9)) *
                (T(1) / T(55));
    auto trajectory =
        make_DenseMatrix<6, 6>(T(2), T(-1), T(2), T(2), T(-1), T(1), T(-1),
                               T(3), T(-1), T(-1), T(3), T(2), T(2), T(-1),
                               T(2), T(2), T(-1), T(1), T(2), T(-1), T(2), T(2),
                               T(-1), T(1), T(-1), T(3), T(-1), T(-1), T(3),
                               T(2), T(1), T(2), T(1), T(1), T(2), T(3)) *
        (T(1) / T(5));
    auto inverse =
        make_DenseMatrix<6, 6>(
            T(1696), T(69), T(-273), T(-273), T(69), T(-204), T(69), T(1627),
            T(69), T(69), T(-342), T(-273), T(-273), T(69), T(1696), T(-273),
            T(69), T(-204), T(-273), T(69), T(-273), T(1696), T(69), T(-204),
            T(69), T(-342), T(69), T(69), T(1627), T(-273), T(-204), T(-273),
            T(-204), T(-204), T(-273), T(1492)) *
        (T(1) / T(1969));
    return PythonMPC::make_InstantMPC_LTI<2, 2, Filter<T, 0>, Ref>(
        make_filter<T, 0>(T(0)), make_DenseMatrix<2, 1>(T(0), T(1)), Empty(),
        Empty(), Empty(), Empty(), SparseMatrixEmpty_Type<T, 1, 6>(),
        make_DenseMatrix<1, 1>(T(0)), substeps, T(1.5), K, L, q, warm,
        trajectory, inverse, SparseMatrixEmpty_Type<T, 6, 1>());
  }
};

} // namespace InstantMPCTestData
#endif // TEST_INSTANT_MPC_DATA_HPP_
