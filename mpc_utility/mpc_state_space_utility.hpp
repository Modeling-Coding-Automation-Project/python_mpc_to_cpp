#ifndef __MPC_STATE_SPACE_UTILITY_HPP__
#define __MPC_STATE_SPACE_UTILITY_HPP__

#include "python_control.hpp"
#include "python_numpy.hpp"

#include <functional>
#include <type_traits>

namespace PythonMPC {

/* MPC Prediction Matrices */
template <typename F_Type, typename Phi_Type, std::size_t Np, std::size_t Nc,
          std::size_t Number_Of_Input, std::size_t Number_Of_State,
          std::size_t Number_Of_Output>
class MPC_PredictionMatrices {
protected:
  /* Type */
  using _T = typename F_Type::Value_Type;

public:
  /* Type */
  using Value_Type = _T;

  static constexpr std::size_t INPUT_SIZE = Number_Of_Input;
  static constexpr std::size_t STATE_SIZE = Number_Of_State;
  static constexpr std::size_t OUTPUT_SIZE = Number_Of_Output;

  /* Check Compatibility */
  static_assert(std::is_same<typename Phi_Type::Value_Type, _T>::value,
                "F_Type and Phi_Type must have the same Value_Type");

  static_assert(F_Type::COLS == OUTPUT_SIZE * Np,
                "F_Type::COLS must be equal to OUTPUT_SIZE * Np");
  static_assert(F_Type::ROWS == STATE_SIZE,
                "F_Type::ROWS must be equal to STATE_SIZE");
  static_assert(Phi_Type::COLS == OUTPUT_SIZE * Np,
                "Phi_Type::ROWS must be equal to OUTPUT_SIZE * Np");
  static_assert(Phi_Type::ROWS == INPUT_SIZE * Nc,
                "Phi_Type::ROWS must be equal to INPUT_SIZE * Nc");

public:
  /* Constructor */
  MPC_PredictionMatrices() : F(), Phi() {}

  MPC_PredictionMatrices(const F_Type &F, const Phi_Type &Phi)
      : F(F), Phi(Phi) {}

  /* Copy Constructor */
  MPC_PredictionMatrices(
      const MPC_PredictionMatrices<F_Type, Phi_Type, Np, Nc, Number_Of_Input,
                                   Number_Of_State, Number_Of_Output> &input)
      : F(input.F), Phi(input.Phi) {}

  MPC_PredictionMatrices<F_Type, Phi_Type, Np, Nc, Number_Of_Input,
                         Number_Of_State, Number_Of_Output> &
  operator=(
      const MPC_PredictionMatrices<F_Type, Phi_Type, Np, Nc, Number_Of_Input,
                                   Number_Of_State, Number_Of_Output> &input) {
    if (this != &input) {
      this->F = input.F;
      this->Phi = input.Phi;
    }
    return *this;
  }

  /* Move Constructor */
  MPC_PredictionMatrices(
      MPC_PredictionMatrices<F_Type, Phi_Type, Np, Nc, Number_Of_Input,
                             Number_Of_State, Number_Of_Output>
          &&input) noexcept
      : F(std::move(input.F)), Phi(std::move(input.Phi)) {}

  MPC_PredictionMatrices<F_Type, Phi_Type, Np, Nc, Number_Of_Input,
                         Number_Of_State, Number_Of_Output> &
  operator=(MPC_PredictionMatrices<F_Type, Phi_Type, Np, Nc, Number_Of_Input,
                                   Number_Of_State, Number_Of_Output>
                &&input) noexcept {
    if (this != &input) {
      this->F = std::move(input.F);
      this->Phi = std::move(input.Phi);
    }
    return *this;
  }

public:
  /* Constant */
  static constexpr std::size_t NP = Np;
  static constexpr std::size_t NC = Nc;

public:
  /* Variables */
  F_Type F;
  Phi_Type Phi;
};

/* make MPC Prediction Matrices */
template <typename F_Type, typename Phi_Type, std::size_t Np, std::size_t Nc,
          std::size_t Number_Of_Input, std::size_t Number_Of_State,
          std::size_t Number_Of_Output>
inline auto make_MPC_PredictionMatrices(void)
    -> MPC_PredictionMatrices<F_Type, Phi_Type, Np, Nc, Number_Of_Input,
                              Number_Of_State, Number_Of_Output> {

  return MPC_PredictionMatrices<F_Type, Phi_Type, Np, Nc, Number_Of_Input,
                                Number_Of_State, Number_Of_Output>();
}

/* MPC Prediction Matrices Type */
template <typename F_Type, typename Phi_Type, std::size_t Np, std::size_t Nc,
          std::size_t Number_Of_Input, std::size_t Number_Of_State,
          std::size_t Number_Of_Output>
using MPC_PredictionMatrices_Type =
    MPC_PredictionMatrices<F_Type, Phi_Type, Np, Nc, Number_Of_Input,
                           Number_Of_State, Number_Of_Output>;

namespace MPC_ReferenceTrajectoryOperation {

template <std::size_t ROWS, std::size_t Np, std::size_t I, std::size_t J,
          typename Ref_Type, typename Fx_Type, typename Dif_Type>
inline typename std::enable_if<(ROWS > 1), void>::type
calculate_each_dif(const Ref_Type &ref, const Fx_Type &Fx, Dif_Type &dif) {
  static_assert(ROWS == Np, "ROWS must be equal to Np when ROWS > 1");

  dif.template set<(I * Ref_Type::COLS) + J, 0>(
      ref.template get<J, I>() -
      Fx.template get<(I * Ref_Type::COLS) + J, 0>());
}

template <std::size_t ROWS, std::size_t Np, std::size_t I, std::size_t J,
          typename Ref_Type, typename Fx_Type, typename Dif_Type>
inline typename std::enable_if<(ROWS == 1), void>::type
calculate_each_dif(const Ref_Type &ref, const Fx_Type &Fx, Dif_Type &dif) {
  static_assert(ROWS == 1, "ROWS must be equal to 1");

  dif.template set<(I * Ref_Type::COLS) + J, 0>(
      ref.template get<J, 0>() -
      Fx.template get<(I * Ref_Type::COLS) + J, 0>());
}

// when J_idx < N
template <typename Ref_Type, typename Fx_Type, typename Dif_Type,
          std::size_t Np, std::size_t I, std::size_t J_idx>
struct DifColumn {
  static void calculate(const Ref_Type &ref, const Fx_Type &Fx, Dif_Type &dif) {

    calculate_each_dif<Ref_Type::ROWS, Np, I, J_idx>(ref, Fx, dif);

    DifColumn<Ref_Type, Fx_Type, Dif_Type, Np, I, J_idx - 1>::calculate(ref, Fx,
                                                                        dif);
  }
};

// column recursion termination
template <typename Ref_Type, typename Fx_Type, typename Dif_Type,
          std::size_t Np, std::size_t I>
struct DifColumn<Ref_Type, Fx_Type, Dif_Type, Np, I, 0> {
  static void calculate(const Ref_Type &ref, const Fx_Type &Fx, Dif_Type &dif) {

    calculate_each_dif<Ref_Type::ROWS, Np, I, 0>(ref, Fx, dif);
  }
};

// when I_idx < M
template <typename Ref_Type, typename Fx_Type, typename Dif_Type,
          std::size_t Np, std::size_t M, std::size_t N, std::size_t I_idx>
struct DifRow {
  static void calculate(const Ref_Type &ref, const Fx_Type &Fx, Dif_Type &dif) {
    DifColumn<Ref_Type, Fx_Type, Dif_Type, Np, I_idx, N - 1>::calculate(ref, Fx,
                                                                        dif);
    DifRow<Ref_Type, Fx_Type, Dif_Type, Np, M, N, I_idx - 1>::calculate(ref, Fx,
                                                                        dif);
  }
};

// row recursion termination
template <typename Ref_Type, typename Fx_Type, typename Dif_Type,
          std::size_t Np, std::size_t M, std::size_t N>
struct DifRow<Ref_Type, Fx_Type, Dif_Type, Np, M, N, 0> {
  static void calculate(const Ref_Type &ref, const Fx_Type &Fx, Dif_Type &dif) {
    DifColumn<Ref_Type, Fx_Type, Dif_Type, Np, 0, N - 1>::calculate(ref, Fx,
                                                                    dif);
  }
};

template <std::size_t Np, std::size_t Number_Of_Output, typename Ref_Type,
          typename Fx_Type, typename Dif_Type>
inline void calculate_dif(const Ref_Type &ref, const Fx_Type &Fx,
                          Dif_Type &dif) {
  DifRow<Ref_Type, Fx_Type, Dif_Type, Np, Np, Number_Of_Output,
         (Np - 1)>::calculate(ref, Fx, dif);
}

} // namespace MPC_ReferenceTrajectoryOperation

/* MPC Reference Trajectory */
template <typename Ref_Type, std::size_t Np> class MPC_ReferenceTrajectory {
protected:
  /* Type */
  using _T = typename Ref_Type::Value_Type;

public:
  /* Type */
  using Value_Type = _T;

  /* Check Compatibility */
  static_assert((Ref_Type::ROWS == Np) || (Ref_Type::ROWS == 1),
                "Ref_Type::ROWS must be equal to Np, or 1");

  using Dif_Type = PythonNumpy::DenseMatrix_Type<_T, (Np * Ref_Type::COLS), 1>;

public:
  /* Constructor */
  MPC_ReferenceTrajectory() : reference() {}

  MPC_ReferenceTrajectory(const Ref_Type &reference_in)
      : reference(reference_in) {}

  /* Copy Constructor */
  MPC_ReferenceTrajectory(const MPC_ReferenceTrajectory<Ref_Type, Np> &input)
      : reference(input.reference) {}

  MPC_ReferenceTrajectory<Ref_Type, Np> &
  operator=(const MPC_ReferenceTrajectory<Ref_Type, Np> &input) {
    if (this != &input) {
      this->reference = input.reference;
    }
    return *this;
  }

  /* Move Constructor */
  MPC_ReferenceTrajectory(
      MPC_ReferenceTrajectory<Ref_Type, Np> &&input) noexcept
      : reference(std::move(input.reference)) {}

  MPC_ReferenceTrajectory<Ref_Type, Np> &
  operator=(MPC_ReferenceTrajectory<Ref_Type, Np> &&input) noexcept {
    if (this != &input) {
      this->reference = std::move(input.reference);
    }
    return *this;
  }

public:
  /* Function */
  template <typename Fx_Type>
  inline auto calculate_dif(const Fx_Type &Fx) -> Dif_Type {

    static_assert(std::is_same<typename Fx_Type::Value_Type, _T>::value,
                  "Fx_Type::Value_Type must be equal to Value_Type");

    Dif_Type dif;

    MPC_ReferenceTrajectoryOperation::calculate_dif<NP, NUMBER_OF_OUTPUT>(
        this->reference, Fx, dif);

    return dif;
  }

public:
  /* Constant */
  static constexpr std::size_t NP = Np;
  static constexpr std::size_t NUMBER_OF_OUTPUT = Ref_Type::COLS;

public:
  /* Variables */
  Ref_Type reference;
};

/* make MPC Reference Trajectory */
template <typename Ref_Type, std::size_t Np>
inline auto make_MPC_ReferenceTrajectory(void)
    -> MPC_ReferenceTrajectory<Ref_Type, Np> {
  return MPC_ReferenceTrajectory<Ref_Type, Np>();
}

/* MPC Reference Trajectory Type */
template <typename Ref_Type, std::size_t Np>
using MPC_ReferenceTrajectory_Type = MPC_ReferenceTrajectory<Ref_Type, Np>;

} // namespace PythonMPC

#endif // __MPC_STATE_SPACE_UTILITY_HPP__
