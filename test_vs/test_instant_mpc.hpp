/**
 * @file test_instant_mpc.hpp
 * @brief iMPC helper and controller regression tests for float and double.
 *
 * Fixed helper expectations and small quadratic optima cover projection,
 * primal-dual continuation, changing bounds/references, trajectories, delay
 * compensation, and value semantics without requiring the servo demo data.
 */
#ifndef TEST_INSTANT_MPC_HPP_
#define TEST_INSTANT_MPC_HPP_

#include "MCAP_tester.hpp"
#include "test_instant_mpc_data.hpp"
#include <type_traits>
#include <utility>

namespace InstantMPCTest {
template <typename T> T tolerance() {
  return std::is_same<T, float>::value ? T(2e-5) : T(1e-11);
}
} // namespace InstantMPCTest

template <typename T> void check_InstantMPC_Operations() {
  using namespace PythonNumpy;
  namespace Op = PythonMPC::LMPC_Instant_Operation;
  Tester::MCAPTester<T> tester;
  const T eps = InstantMPCTest::tolerance<T>();

  auto a = make_DenseMatrix<2, 3>(T(-3), T(0), T(4), T(2), T(-1), T(5));
  auto b = make_DenseMatrix<2, 3>(T(-2), T(0), T(1), T(3), T(-4), T(6));
  auto scalar_max = Op::maximum(T(0), a);
  auto matrix_max = Op::maximum(a, b);
  tester.expect_near(
      scalar_max.matrix.data,
      make_DenseMatrix<2, 3>(T(0), T(0), T(4), T(2), T(0), T(5)).matrix.data,
      eps, "iMPC scalar maximum: negative, zero, positive entries");
  tester.expect_near(
      matrix_max.matrix.data,
      make_DenseMatrix<2, 3>(T(-2), T(0), T(4), T(3), T(-1), T(6)).matrix.data,
      eps, "iMPC elementwise maximum: both choices and ties");

  auto g = make_DenseMatrix<5, 1>(T(-2), T(-3), T(4), T(5), T(0));
  auto mu = make_DenseMatrix<5, 1>(T(1), T(0), T(0), T(2), T(0));
  DenseMatrix_Type<T, 5, 1> gp;
  Op::ComputeGpOperation::compute<T>(gp, g, mu);
  tester.expect_near(
      gp.matrix.data,
      make_DenseMatrix<5, 1>(T(-2), T(0), T(4), T(5), T(0)).matrix.data, eps,
      "iMPC gp permits negative residual only for active multipliers");

  mu = make_DenseMatrix<5, 1>(T(2), T(0), T(1), T(3), T(0));
  auto step = make_DenseMatrix<5, 1>(T(-4), T(-2), T(-1), T(2), T(0));
  auto eta = make_DenseMatrixOnes<T, 5, 1>();
  eta(4, 0) = T(0.75);
  Op::ComputeEtaOperation::compute<T>(eta, mu, step);
  tester.expect_near(
      eta.matrix.data,
      make_DenseMatrix<5, 1>(T(0.5), T(0), T(1), T(1), T(0.75)).matrix.data,
      eps,
      "iMPC eta clips crossing zero and leaves equality/positive cases "
      "unchanged");
  auto tiny_mu = make_DenseMatrix<1, 1>(T(1e-32));
  auto tiny_step = make_DenseMatrix<1, 1>(T(-2e-32));
  auto tiny_eta = make_DenseMatrixOnes<T, 1, 1>();
  Op::ComputeEtaOperation::compute<T>(tiny_eta, tiny_mu, tiny_step);
  tester.expect_near(tiny_eta(0, 0), T(0.01), eps,
                     "iMPC eta uses the signed denominator safeguard");

  auto bg = make_DenseMatrixOnes<T, 10, 1>() * T(99);
  auto upper = make_DenseMatrix<2, 1>(T(5), T(8));
  auto lower = make_DenseMatrix<2, 1>(T(-3), T(-4));
  auto input = make_DenseMatrix<2, 1>(T(1), T(2));
  Op::update_bg<false, true, true, 2, 2, 2, 6>(bg, upper, lower, input);
  tester.expect_near(
      bg.matrix.data,
      make_DenseMatrix<10, 1>(T(99), T(99), T(4), T(6), T(4), T(6), T(4), T(6),
                              T(4), T(6))
          .matrix.data,
      eps,
      "iMPC absolute bound blocks repeat over horizon and preserve DU prefix");
  auto unchanged = bg;
  input = make_DenseMatrix<2, 1>(T(2), T(0));
  Op::update_bg<true, true, true, 2, 2, 2, 6>(bg, upper, lower, input);
  Op::update_bg<false, false, false, 2, 2, 2, 6>(bg, upper, lower, input);
  tester.expect_near(bg.matrix.data, unchanged.matrix.data, eps,
                     "iMPC absent constraints/bounds leave bg unchanged");
  Op::update_bg<false, true, false, 2, 2, 2, 6>(bg, upper, lower, input);
  tester.expect_near(
      bg.matrix.data,
      make_DenseMatrix<10, 1>(T(99), T(99), T(3), T(8), T(3), T(8), T(4), T(6),
                              T(4), T(6))
          .matrix.data,
      eps, "iMPC upper-only bound update does not overwrite lower bounds");
  Op::update_bg<false, false, true, 2, 2, 2, 6>(bg, upper, lower, input);
  tester.expect_near(
      bg.matrix.data,
      make_DenseMatrix<10, 1>(T(99), T(99), T(3), T(8), T(3), T(8), T(5), T(4),
                              T(5), T(4))
          .matrix.data,
      eps, "iMPC lower-only bound update uses latest input minus lower bound");

  auto reference = make_DenseMatrix<2, 2>(T(10), T(13), T(20), T(18));
  auto ref = make_DenseMatrix<2, 1>(T(10), T(20));
  auto weight = make_DenseMatrix<3, 2>(T(1), T(2), T(-1), T(0), T(0), T(3));
  auto q = make_DenseMatrixOnes<T, 8, 1>();
  Op::ComputeQTrajectoryOperation::compute<2, 2, 3, 2>(q, reference, ref,
                                                       weight);
  tester.expect_near(
      q.matrix.data,
      make_DenseMatrix<8, 1>(T(1), T(1), T(1), T(1), T(1), T(0), T(-2), T(-5))
          .matrix.data,
      eps,
      "iMPC trajectory gradient places weighted differences after input block");
  Op::ComputeQTrajectoryOperation::compute<2, 2, 3, 2>(q, reference, ref,
                                                       weight);
  tester.expect_near(
      q.matrix.data,
      make_DenseMatrix<8, 1>(T(1), T(1), T(1), T(1), T(1), T(-1), T(-5), T(-11))
          .matrix.data,
      eps, "iMPC trajectory gradient accumulates rather than replaces");

  auto first = make_DenseMatrix<3, 1>(T(-1), T(4), T(7));
  auto second = make_DenseMatrix<3, 1>(T(2), T(0), T(6));
  tester.expect_near(
      Op::MaxAbsDiffOperation::compute<T>(first, second), T(4), eps,
      "iMPC reference distance takes maximum absolute difference");
  tester.expect_near(Op::MaxAbsDiffOperation::compute<T>(tiny_mu, tiny_mu),
                     T(0), eps, "iMPC reference distance singleton equality");
  auto target = make_DenseMatrix<3, 1>(T(1), T(2), T(99));
  auto increment = make_DenseMatrix<3, 1>(T(3), T(-4), T(-100));
  Op::AddAssignColumnOperation::compute<2>(target, increment);
  tester.expect_near(
      target.matrix.data,
      make_DenseMatrix<3, 1>(T(4), T(-2), T(99)).matrix.data, eps,
      "iMPC applies only the first INPUT_SIZE decision increments");
  tester.throw_error_if_test_failed();
}

template <typename T> void check_InstantMPC_UnconstrainedAndTrajectory() {
  using namespace PythonNumpy;
  using namespace InstantMPCTestData;
  Tester::MCAPTester<T> tester;
  const T eps = InstantMPCTest::tolerance<T>();
  using Fixture = OneStep<T, 0>;
  using Controller = typename Fixture::Controller;
  static_assert(Controller::IS_NO_CONSTRAINTS,
                "Empty bounds select unconstrained mode");
  static_assert(!Controller::IS_REFERENCE_TRAJECTORY,
                "One column is a point reference");
  static_assert(Controller::NP == 1 && Controller::NC == 1 &&
                    Controller::NW_SIZE == 3 && Controller::NS_SIZE == 2,
                "One-step dimensions must match the decision layout");
  auto controller = Fixture::make();
  auto ref = make_DenseMatrix<1, 1>(T(2));
  auto y = make_DenseMatrix<1, 1>(T(0));
  tester.expect_near(controller.get_U_latest()(0, 0), T(0), eps,
                     "iMPC starts at zero input");
  tester.expect_near(controller.update(ref, y)(0, 0), T(1), eps,
                     "iMPC unit quadratic optimum du=-(dx+error)/2");
  y(0, 0) = T(0.5);
  tester.expect_near(controller.update(ref, y)(0, 0), T(1.5), eps,
                     "iMPC second update includes estimated state increment");
  y(0, 0) = T(1);
  tester.expect_near(
      controller.update(ref, y)(0, 0), T(1.75), eps,
      "iMPC stores the previous state and accumulates control increments");
  tester.expect_near(controller.get_X_inner_model()(0, 0), T(1), eps,
                     "iMPC exposes the corrected Kalman state");
  ref(0, 0) = T(4);
  tester.expect_near(
      controller.update(ref, y)(0, 0), T(3.25), eps,
      "iMPC reference changes preserve accumulated absolute input");
  auto no_substeps = Fixture::make(0);
  tester.expect_near(
      no_substeps.update(make_DenseMatrix<1, 1>(T(2)),
                         make_DenseMatrix<1, 1>(T(0)))(0, 0),
      T(1), eps,
      "iMPC zero substeps still projects the unconstrained warm start");
  Controller default_constructed;
  tester.expect_near(default_constructed.get_U_latest()(0, 0), T(0), eps,
                     "iMPC default constructor initializes input");
  tester.expect_near(default_constructed.get_X_inner_model()(0, 0), T(0), eps,
                     "iMPC default constructor initializes state");

  using PointFixture = TwoStep<T, false>;
  using TrajectoryFixture = TwoStep<T, true>;
  static_assert(TrajectoryFixture::Controller::IS_REFERENCE_TRAJECTORY,
                "Two reference columns select trajectory mode");
  auto point = PointFixture::make();
  auto flat = TrajectoryFixture::make();
  auto rising = TrajectoryFixture::make();
  auto point_ref = make_DenseMatrix<1, 1>(T(2));
  auto flat_ref = make_DenseMatrix<1, 2>(T(2), T(2));
  auto rising_ref = make_DenseMatrix<1, 2>(T(2), T(4));
  y(0, 0) = T(0);
  tester.expect_near(point.update(point_ref, y)(0, 0), T(1.2), eps,
                     "iMPC two-step quadratic optimum for a point reference");
  tester.expect_near(flat.update(flat_ref, y)(0, 0), T(1.2), eps,
                     "iMPC flat trajectory agrees with point reference");
  tester.expect_near(
      rising.update(rising_ref, y)(0, 0), T(1.6), eps,
      "iMPC future target adds one fifth of its difference to du0");
  // Change only the future reference: the base reference and measured state
  // stay fixed, so stale trajectory corrections would give the wrong du0.
  rising_ref(0, 1) = T(1);
  tester.expect_near(
      rising.update(rising_ref, y)(0, 0), T(2.6), eps,
      "iMPC recomputes trajectory correction when only future target changes");
  auto direct_warm_start = TrajectoryFixture::make(0);
  tester.expect_near(
      direct_warm_start.update(make_DenseMatrix<1, 2>(T(2), T(4)), y)(0, 0),
      T(1.6), eps,
      "iMPC trajectory warm start already solves the unconstrained QP");
  tester.throw_error_if_test_failed();
}

template <typename T> void check_InstantMPC_ConstraintsAndValueSemantics() {
  using namespace PythonNumpy;
  using namespace InstantMPCTestData;
  Tester::MCAPTester<T> tester;
  const T eps = InstantMPCTest::tolerance<T>();
  using Fixture = OneStep<T, 1>;
  using Controller = typename Fixture::Controller;
  static_assert(!Controller::IS_NO_CONSTRAINTS,
                "A DU upper bound activates dual updates");
  auto ref = make_DenseMatrix<1, 1>(T(2));
  auto y = make_DenseMatrix<1, 1>(T(0));
  auto one_substep = Fixture::make(1);
  tester.expect_near(one_substep.update(ref, y)(0, 0), T(1), eps,
                     "iMPC first substep uses old mu, not newly computed mu");
  auto controller = Fixture::make(2);
  // Warm optimum du=1 violates du<=1/4. With h=3/2, mu1=9/8;
  // the second primal step reduces du by mu1/4, giving 23/32.
  tester.expect_near(controller.update(ref, y)(0, 0), T(23) / T(32), eps,
                     "iMPC second substep applies the retained dual penalty");
  Controller copied(controller);
  Controller assigned;
  assigned = controller;
  Controller moved(std::move(copied));
  Controller move_assigned;
  move_assigned = std::move(assigned);
  Controller &same_controller = controller;
  controller = same_controller;
  controller = std::move(same_controller);
  // Continuing from mu=9/4 gives du=-1/8 and U=19/32. This is
  // deliberately not hard clipping: finite primal-dual steps may violate
  // bounds.
  tester.expect_near(
      controller.update(ref, y)(0, 0), T(19) / T(32), eps,
      "iMPC carries dual state across updates and supports self assignment");
  tester.expect_near(
      moved.update(ref, y)(0, 0), T(19) / T(32), eps,
      "iMPC copy and move constructors preserve optimizer history");
  tester.expect_near(
      move_assigned.update(ref, y)(0, 0), T(19) / T(32), eps,
      "iMPC copy and move assignments preserve optimizer history");
  auto reset = Fixture::make();
  reset.update(ref, y);
  ref(0, 0) = T(4);
  tester.expect_near(
      reset.update(ref, y)(0, 0), T(33) / T(16), eps,
      "iMPC changed reference clears dual history before substeps");

  auto below_threshold = Fixture::make();
  auto above_threshold = Fixture::make();
  ref(0, 0) = T(2);
  below_threshold.update(ref, y);
  above_threshold.update(ref, y);
  ref(0, 0) = T(2) + T(PythonMPC::IMPC_REFERENCE_CHANGED_TOL) * T(0.25);
  tester.expect_near(
      below_threshold.update(ref, y)(0, 0),
      T(19) / T(32) + T(5) / T(16) * (ref(0, 0) - T(2)), eps,
      "iMPC reference jitter below tolerance retains dual history");
  ref(0, 0) = T(2) + T(PythonMPC::IMPC_REFERENCE_CHANGED_TOL) * T(2);
  tester.expect_near(
      above_threshold.update(ref, y)(0, 0),
      T(13) / T(16) + T(5) / T(16) * ref(0, 0), eps,
      "iMPC reference changes above tolerance reset dual history");

  ref(0, 0) = T(2);
  auto absolute_upper = OneStep<T, 2>::make();
  tester.expect_near(absolute_upper.update(ref, y)(0, 0), T(23) / T(32), eps,
                     "iMPC U upper bound starts with Umax-Ulatest");
  tester.expect_near(
      absolute_upper.update(ref, y)(0, 0), T(83) / T(256), eps,
      "iMPC U upper bound shifts after the previous input is applied");
  auto absolute_lower = OneStep<T, 3>::make();
  ref(0, 0) = T(-2);
  tester.expect_near(absolute_lower.update(ref, y)(0, 0), T(-23) / T(32), eps,
                     "iMPC U lower bound uses the opposite residual direction");
  tester.expect_near(
      absolute_lower.update(ref, y)(0, 0), T(-83) / T(256), eps,
      "iMPC U lower bound shifts with the latest negative input");
  auto delta_lower = OneStep<T, 4>::make();
  tester.expect_near(
      delta_lower.update(ref, y)(0, 0), T(-23) / T(32), eps,
      "iMPC DU lower bound activates the negative increment constraint");
  tester.expect_near(delta_lower.update(ref, y)(0, 0), T(-19) / T(32), eps,
                     "iMPC DU lower bound remains fixed across input updates");
  tester.throw_error_if_test_failed();
}

template <typename T> void check_InstantMPC_Delay() {
  using namespace PythonNumpy;
  using namespace InstantMPCTestData;
  namespace Op = PythonMPC::LMPC_Instant_Operation;
  Tester::MCAPTester<T> tester;
  const T eps = InstantMPCTest::tolerance<T>();
  using Vector = DenseMatrix_Type<T, 1, 1>;
  auto filter = make_filter<T, 1>();
  filter.predict_and_update_with_fixed_G(make_DenseMatrix<1, 1>(T(3)),
                                         make_DenseMatrix<1, 1>(T(2)));
  PythonControl::DelayedVectorObject<Vector, 1> history;
  auto ignored_state = make_DenseMatrix<1, 1>(T(99));
  auto measured = make_DenseMatrix<1, 1>(T(7));
  auto compensated =
      Op::compensate_X_Y_delay<1>(ignored_state, measured, history, filter);
  tester.expect_near(
      std::get<0>(compensated)(0, 0), T(5), eps,
      "iMPC delayed state predicts xhat=2 through pending input 3");
  tester.expect_near(
      std::get<1>(compensated)(0, 0), T(12), eps,
      "iMPC delayed output adds measured residual against initial history");
  filter.set_x_hat(make_DenseMatrix<1, 1>(T(4)));
  measured(0, 0) = T(8);
  compensated =
      Op::compensate_X_Y_delay<1>(ignored_state, measured, history, filter);
  tester.expect_near(
      std::get<0>(compensated)(0, 0), T(7), eps,
      "iMPC delayed state uses current filter estimate rather than X_in");
  tester.expect_near(
      std::get<1>(compensated)(0, 0), T(10), eps,
      "iMPC delayed output residual uses the previous predicted output");
  tester.expect_near(history.get()(0, 0), T(5), eps,
                     "iMPC delay helper advances output history");
  auto passthrough =
      Op::compensate_X_Y_delay<0>(ignored_state, measured, history, filter);
  tester.expect_near(std::get<0>(passthrough)(0, 0), T(99), eps,
                     "iMPC zero delay preserves supplied state");
  tester.expect_near(std::get<1>(passthrough)(0, 0), T(8), eps,
                     "iMPC zero delay preserves measured output");
  tester.expect_near(history.get()(0, 0), T(5), eps,
                     "iMPC zero delay leaves history untouched");

  using Fixture = OneStep<T, 0, 1>;
  auto controller = Fixture::make();
  auto ref = make_DenseMatrix<1, 1>(T(2));
  auto y = make_DenseMatrix<1, 1>(T(0));
  tester.expect_near(controller.update(ref, y)(0, 0), T(1), eps,
                     "iMPC delayed controller first update");
  tester.expect_near(
      controller.update(ref, y)(0, 0), T(1), eps,
      "iMPC delayed controller accounts for the pending unit input");
  tester.expect_near(controller.get_X_inner_model()(0, 0), T(1), eps,
                     "iMPC stores compensated state, not delayed filter state");
  typename Fixture::Controller copied(controller);
  typename Fixture::Controller moved(std::move(copied));
  tester.expect_near(controller.update(ref, y)(0, 0), T(1.5), eps,
                     "iMPC delayed controller continues with buffered state");
  tester.expect_near(moved.update(ref, y)(0, 0), T(1.5), eps,
                     "iMPC copy and move preserve delay/filter histories");
  tester.expect_near(moved.get_Y_store().get()(0, 0), T(1), eps,
                     "iMPC output history getter returns delayed prediction");
  auto detached_history = moved.get_Y_store();
  detached_history.push(make_DenseMatrix<1, 1>(T(100)));
  tester.expect_near(moved.get_Y_store().get_latest()(0, 0), T(1), eps,
                     "iMPC output history getter returns an independent copy");
  tester.throw_error_if_test_failed();
}

#endif // TEST_INSTANT_MPC_HPP_
