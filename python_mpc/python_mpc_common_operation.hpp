#ifndef __PYTHON_MPC_COMMON_OPERATION_HPP__
#define __PYTHON_MPC_COMMON_OPERATION_HPP__

#include "python_numpy.hpp"

namespace CommonOperation {

template <typename T, typename Matrix_Type, std::size_t Index>
struct SubstituteIdentityElements {
  static void apply(Matrix_Type &matrix) {
    matrix.template set<Index, Index>(static_cast<T>(1));

    SubstituteIdentityElements<T, Matrix_Type, Index - 1>::apply(matrix);
  }
};

template <typename T, typename Matrix_Type>
struct SubstituteIdentityElements<T, Matrix_Type, 0> {
  static void apply(Matrix_Type &matrix) {
    matrix.template set<0, 0>(static_cast<T>(1));
  }
};

template <typename T, typename Matrix_Type, std::size_t Size>
inline void substitute_identity_elements(Matrix_Type &matrix) {
  SubstituteIdentityElements<T, Matrix_Type, Size - 1>::apply(matrix);
}

template <typename T, typename Augmented_Phi_Size_Identity_Zero_Type>
inline auto create_augmented_phi_size_identity_zero(void)
    -> Augmented_Phi_Size_Identity_Zero_Type {
  Augmented_Phi_Size_Identity_Zero_Type augmented_phi_size_identity_zero;

  substitute_identity_elements<T, Augmented_Phi_Size_Identity_Zero_Type,
                               Augmented_Phi_Size_Identity_Zero_Type::ROWS>(
      augmented_phi_size_identity_zero);

  return augmented_phi_size_identity_zero;
}

} // namespace CommonOperation

#endif // __PYTHON_MPC_COMMON_OPERATION_HPP__
