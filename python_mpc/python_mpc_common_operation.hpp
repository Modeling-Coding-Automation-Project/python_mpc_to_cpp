#ifndef __PYTHON_MPC_COMMON_OPERATION_HPP__
#define __PYTHON_MPC_COMMON_OPERATION_HPP__

#include "python_numpy.hpp"

namespace CommonOperation {

template <typename T, typename Matrix_Type, std::size_t Index>
struct SubstituteIdentityElements {
  /**
   * @brief Substitutes the identity element at the specified index in the
   * matrix.
   *
   * This function sets the element at (Index, Index) to the specified value.
   *
   * @param matrix The matrix to modify.
   */
  static void apply(Matrix_Type &matrix) {
    matrix.template set<Index, Index>(static_cast<T>(1));

    SubstituteIdentityElements<T, Matrix_Type, Index - 1>::apply(matrix);
  }
};

template <typename T, typename Matrix_Type>
struct SubstituteIdentityElements<T, Matrix_Type, 0> {
  /**
   * @brief Base case for substituting the identity element in the matrix.
   *
   * This function sets the first element of the matrix to the specified value.
   *
   * @param matrix The matrix to modify.
   */
  static void apply(Matrix_Type &matrix) {
    matrix.template set<0, 0>(static_cast<T>(1));
  }
};

/**
 * @brief Substitutes identity elements in a matrix.
 *
 * This function replaces the diagonal elements of the matrix with the
 * specified value, effectively creating an identity-like structure.
 *
 * @tparam T The type of the elements in the matrix.
 * @tparam Matrix_Type The type of the matrix to modify.
 * @tparam Size The size of the matrix (number of rows/columns).
 */
template <typename T, typename Matrix_Type, std::size_t Size>
inline void substitute_identity_elements(Matrix_Type &matrix) {
  SubstituteIdentityElements<T, Matrix_Type, Size - 1>::apply(matrix);
}

/**
 * @brief Creates an augmented matrix with identity elements and zeros.
 *
 * This function initializes an augmented matrix with identity elements in the
 * upper part and zeros in the lower part, suitable for MPC operations.
 *
 * @tparam T The type of the elements in the matrix.
 * @tparam Augmented_Phi_Size_Identity_Zero_Type The type of the augmented
 * matrix.
 *
 * @return An instance of the augmented matrix with identity elements and zeros.
 */
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
