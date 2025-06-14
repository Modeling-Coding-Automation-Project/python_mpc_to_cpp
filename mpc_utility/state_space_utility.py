"""
File: state_space_utility.py

This module provides utility classes and functions for symbolic and numeric manipulation of state-space models, particularly for Model Predictive Control (MPC) applications. It leverages sympy for symbolic computation and numpy for numerical operations, enabling the construction, augmentation, and conversion of state-space representations, as well as the generation of prediction matrices for MPC.
"""
import numpy as np
import sympy as sp


def symbolic_to_numeric_matrix(symbolic_matrix: sp.Matrix) -> np.ndarray:
    """
    Convert a symbolic sympy matrix to a numeric numpy matrix.
    Args:
        symbolic_matrix (sp.Matrix): A sympy matrix containing symbolic expressions.
    Returns:
        np.ndarray: A numpy array with numeric values converted from the symbolic matrix.
    """
    numeric_matrix = np.zeros(
        (symbolic_matrix.shape[0], symbolic_matrix.shape[1]), dtype=float)

    for i in range(symbolic_matrix.shape[0]):
        for j in range(symbolic_matrix.shape[1]):
            numeric_matrix[i, j] = float(symbolic_matrix[i, j])

    return numeric_matrix


class SymbolicStateSpace:
    """
    A class representing a symbolic state-space model.
    Attributes:
        A (sp.Matrix): State matrix.
        B (sp.Matrix): Input matrix.
        C (sp.Matrix): Output matrix.
        D (sp.Matrix, optional): Feedthrough matrix.
        delta_time (float): Time step for discrete systems.
        Number_of_Delay (int): Number of delays in the system.
    """

    def __init__(self, A: sp.Matrix, B: sp.Matrix, C: sp.Matrix,
                 D: sp.Matrix = None, delta_time=0.0, Number_of_Delay=0):
        self.delta_time = delta_time
        self.STATE_SIZE = A.shape[0]
        self.INPUT_SIZE = B.shape[1]
        self.OUTPUT_SIZE = C.shape[0]

        if not isinstance(A, sp.Matrix):
            self.A = sp.Matrix(A)
        else:
            self.A = A

        if not isinstance(B, sp.Matrix):
            self.B = sp.Matrix(B)
        else:
            self.B = B

        if not isinstance(C, sp.Matrix):
            self.C = sp.Matrix(C)
        else:
            self.C = C

        if D is not None:
            if not isinstance(D, sp.Matrix):
                self.D = sp.Matrix(D)
            else:
                self.D = D

        self.Number_of_Delay = Number_of_Delay


class StateSpaceEmbeddedIntegrator:
    """
    A class that augments a state-space model with an embedded integrator.
    This class takes a symbolic state-space model and constructs an augmented model
    that includes the state, input, and output matrices, along with the necessary
    transformations to handle the output as an integral of the state.
    Attributes:
        delta_time (float): Time step for discrete systems.
        INPUT_SIZE (int): Number of inputs in the system.
        STATE_SIZE (int): Number of states in the system.
        OUTPUT_SIZE (int): Number of outputs in the system.
        A (sp.Matrix): Augmented state matrix.
        B (sp.Matrix): Augmented input matrix.
        C (sp.Matrix): Augmented output matrix.
    """

    def __init__(self, state_space: SymbolicStateSpace):
        if not isinstance(state_space.A, sp.Matrix):
            raise ValueError(
                "A must be of type sympy matrix.")
        if not isinstance(state_space.B, sp.Matrix):
            raise ValueError(
                "B must be of type sympy matrix.")
        if not isinstance(state_space.C, sp.Matrix):
            raise ValueError(
                "C must be of type sympy matrix.")

        self.delta_time = state_space.delta_time

        self.INPUT_SIZE = state_space.INPUT_SIZE
        self.STATE_SIZE = state_space.STATE_SIZE
        self.OUTPUT_SIZE = state_space.OUTPUT_SIZE

        self.A = sp.Matrix(self.STATE_SIZE + self.OUTPUT_SIZE,
                           self.STATE_SIZE + self.OUTPUT_SIZE,
                           lambda i, j: 0.0)
        self.B = sp.Matrix(self.STATE_SIZE + self.OUTPUT_SIZE,
                           self.INPUT_SIZE,
                           lambda i, j: 0.0)
        self.C = sp.Matrix(self.OUTPUT_SIZE,
                           self.STATE_SIZE + self.OUTPUT_SIZE,
                           lambda i, j: 0.0)

        self.construct_augmented_model(
            state_space.A, state_space.B, state_space.C)

    def construct_augmented_model(self, A_original: sp.Matrix,
                                  B_original: sp.Matrix, C_original: sp.Matrix):
        """
        Constructs the augmented state-space model with an embedded integrator.
        Args:
            A_original (sp.Matrix): Original state matrix.
            B_original (sp.Matrix): Original input matrix.
            C_original (sp.Matrix): Original output matrix.
        """

        o_xy_T = sp.Matrix(self.STATE_SIZE,
                           self.OUTPUT_SIZE, lambda i, j: 0.0)
        o_xy = sp.Matrix(self.OUTPUT_SIZE,
                         self.STATE_SIZE, lambda i, j: 0.0)
        I_yy = sp.eye(self.OUTPUT_SIZE)

        A_upper: sp.Matrix = A_original.row_join(o_xy_T)
        A_lower = (C_original * A_original).row_join(I_yy)
        self.A = A_upper.col_join(A_lower)

        B_upper = B_original
        B_lower = C_original * B_original
        self.B = B_upper.col_join(B_lower)

        self.C = o_xy.row_join(I_yy)


class MPC_PredictionMatrices:
    """
    A class to generate prediction matrices for Model Predictive Control (MPC).
    This class constructs the F and Phi matrices based on the state-space model
    and the specified prediction horizon (Np) and control horizon (Nc).

    Attributes:
        Np (int): Prediction horizon.
        Nc (int): Control horizon.
        INPUT_SIZE (int): Number of inputs in the system.
        STATE_SIZE (int): Number of states in the system.
        OUTPUT_SIZE (int): Number of outputs in the system.
    """

    def __init__(self, Np, Nc, INPUT_SIZE, STATE_SIZE, OUTPUT_SIZE):
        self.INPUT_SIZE = INPUT_SIZE
        self.STATE_SIZE = STATE_SIZE
        self.OUTPUT_SIZE = OUTPUT_SIZE

        self.Np = Np
        self.Nc = Nc

        self.A_symbolic = None
        self.B_symbolic = None
        self.C_symbolic = None
        self.A_numeric = None
        self.B_numeric = None
        self.C_numeric = None
        self.initialize_ABC()

        self._exponential_A_list = self._generate_exponential_A_list(
            self.A_symbolic)

        self.F_symbolic = None
        self.Phi_symbolic = None
        self.F_numeric = None
        self.Phi_numeric = None

        self.ABC_values = {}

    def initialize_ABC(self):
        """
        Initializes symbolic matrices A, B, and C with symbolic variables.
        This method creates symbolic matrices for the state, input, and output
        matrices, which will be used for symbolic manipulation and substitution.
        """
        self.A_symbolic = sp.Matrix(self.STATE_SIZE, self.STATE_SIZE,
                                    lambda i, j: sp.symbols(f'a{i+1}{j+1}'))
        self.B_symbolic = sp.Matrix(self.STATE_SIZE, self.INPUT_SIZE,
                                    lambda i, j: sp.symbols(f'b{i+1}{j+1}'))
        self.C_symbolic = sp.Matrix(self.OUTPUT_SIZE, self.STATE_SIZE,
                                    lambda i, j: sp.symbols(f'c{i+1}{j+1}'))

        self.A_numeric = sp.Matrix(self.STATE_SIZE, self.STATE_SIZE,
                                   lambda i, j: 0.0)
        self.B_numeric = sp.Matrix(self.STATE_SIZE, self.INPUT_SIZE,
                                   lambda i, j: 0.0)
        self.C_numeric = sp.Matrix(self.OUTPUT_SIZE, self.STATE_SIZE,
                                   lambda i, j: 0.0)

    def generate_symbolic_substitution(self, A: np.ndarray, B: np.ndarray, C: np.ndarray):
        """
        Generates symbolic variables for the elements of matrices A, B, and C,
        and maps them to their corresponding numeric values.
        Args:
            A (np.ndarray): State matrix.
            B (np.ndarray): Input matrix.
            C (np.ndarray): Output matrix.
        """
        # Generate symbolic variables and map them to A's elements
        for i in range(A.shape[0]):
            for j in range(A.shape[1]):
                symbol = sp.symbols(f'a{i+1}{j+1}')
                self.ABC_values[symbol] = A[i, j]

        # Generate symbolic variables and map them to B's elements
        for i in range(B.shape[0]):
            for j in range(B.shape[1]):
                symbol = sp.symbols(f'b{i+1}{j+1}')
                self.ABC_values[symbol] = B[i, j]

        # Generate symbolic variables and map them to C's elements
        for i in range(C.shape[0]):
            for j in range(C.shape[1]):
                symbol = sp.symbols(f'c{i+1}{j+1}')
                self.ABC_values[symbol] = C[i, j]

    def substitute_ABC_symbolic(self, A: sp.Matrix, B: sp.Matrix, C: sp.Matrix):
        """
        Substitutes symbolic variables in the state-space matrices A, B, and C
        with their corresponding numeric values.
        Args:
            A (sp.Matrix): State matrix.
            B (sp.Matrix): Input matrix.
            C (sp.Matrix): Output matrix.
        """
        self.A_symbolic = sp.Matrix(self.STATE_SIZE, self.STATE_SIZE,
                                    lambda i, j: sp.symbols(f'a{i+1}{j+1}'))
        for i in range(A.shape[0]):
            for j in range(A.shape[1]):
                self.A_symbolic[i, j].subs(f'a{i+1}{j+1}', A[i, j])

        self.B_symbolic = sp.Matrix(self.STATE_SIZE, self.INPUT_SIZE,
                                    lambda i, j: sp.symbols(f'b{i+1}{j+1}'))
        for i in range(B.shape[0]):
            for j in range(B.shape[1]):
                self.B_symbolic[i, j].subs(f'b{i+1}{j+1}', B[i, j])

        self.C_symbolic = sp.Matrix(self.OUTPUT_SIZE, self.STATE_SIZE,
                                    lambda i, j: sp.symbols(f'c{i+1}{j+1}'))
        for i in range(C.shape[0]):
            for j in range(C.shape[1]):
                self.C_symbolic[i, j].subs(f'c{i+1}{j+1}', C[i, j])

        self._generate_exponential_A_list(self.A_symbolic)

    def substitute_ABC_numeric(self, A: np.ndarray, B: np.ndarray, C: np.ndarray):
        """
        Substitutes numeric values into the symbolic matrices A, B, and C.
        Args:
            A (np.ndarray): State matrix.
            B (np.ndarray): Input matrix.
            C (np.ndarray): Output matrix.
        """
        self.A_symbolic = sp.Matrix(self.STATE_SIZE, self.STATE_SIZE,
                                    lambda i, j: sp.symbols(f'a{i+1}{j+1}'))
        for i in range(A.shape[0]):
            for j in range(A.shape[1]):
                self.A_numeric[i, j] = self.A_symbolic[i, j].subs(
                    self.ABC_values)

        self.B_symbolic = sp.Matrix(self.STATE_SIZE, self.INPUT_SIZE,
                                    lambda i, j: sp.symbols(f'b{i+1}{j+1}'))
        for i in range(B.shape[0]):
            for j in range(B.shape[1]):
                self.B_numeric[i, j] = self.B_symbolic[i, j].subs(
                    self.ABC_values)

        self.C_symbolic = sp.Matrix(self.OUTPUT_SIZE, self.STATE_SIZE,
                                    lambda i, j: sp.symbols(f'c{i+1}{j+1}'))
        for i in range(C.shape[0]):
            for j in range(C.shape[1]):
                self.C_numeric[i, j] = self.C_symbolic[i, j].subs(
                    self.ABC_values)

        self._exponential_A_list = self._generate_exponential_A_list(
            self.A_numeric)

    def substitute_numeric(self, A: np.ndarray, B: np.ndarray, C: np.ndarray) -> tuple:
        """
        Substitutes numeric values into the symbolic matrices A, B, and C,
        and builds the F and Phi matrices.
        Args:
            A (np.ndarray): State matrix.
            B (np.ndarray): Input matrix.
            C (np.ndarray): Output matrix.
        """
        if not isinstance(A, np.ndarray):
            A = symbolic_to_numeric_matrix(A)
        if not isinstance(B, np.ndarray):
            B = symbolic_to_numeric_matrix(B)
        if not isinstance(C, np.ndarray):
            C = symbolic_to_numeric_matrix(C)

        self.generate_symbolic_substitution(A, B, C)
        self.substitute_ABC_numeric(A, B, C)

        self.build_matrices(self.B_numeric, self.C_numeric)

        self.F_numeric = symbolic_to_numeric_matrix(
            self.F_symbolic)
        self.Phi_numeric = symbolic_to_numeric_matrix(
            self.Phi_symbolic)

    def build_matrices(self, B: sp.Matrix, C: sp.Matrix) -> tuple:
        """
        Builds the F and Phi matrices based on the symbolic state-space model.
        Args:
            B (sp.Matrix): Input matrix.
            C (sp.Matrix): Output matrix.
        """
        self.F_symbolic = self._build_F(C)
        self.Phi_symbolic = self._build_Phi(B, C)

    def _generate_exponential_A_list(self, A: sp.Matrix):
        """
        Generates a list of matrices representing the exponential of the state matrix A
        for each step in the prediction horizon.
        Args:
            A (sp.Matrix): State matrix.
        Returns:
            list: A list of matrices representing the exponential of A for each step.
        """
        exponential_A_list = []

        for i in range(self.Np):
            if i == 0:
                exponential_A_list.append(A)
            else:
                exponential_A_list.append(
                    exponential_A_list[i - 1] * A)

        return exponential_A_list

    def _build_F(self, C: sp.Matrix) -> sp.Matrix:
        """
        Builds the F matrix, which is used in the MPC prediction step.
        Args:
            C (sp.Matrix): Output matrix.
        Returns:
            sp.Matrix: The F matrix, which is a block matrix containing the outputs
            of the system at each step in the prediction horizon.
        """
        F = sp.zeros(self.OUTPUT_SIZE * self.Np, self.STATE_SIZE)
        for i in range(self.Np):
            # C A^{i+1}
            F[i * self.OUTPUT_SIZE:(i + 1) *
              self.OUTPUT_SIZE, :] = C * self._exponential_A_list[i]
        return F

    def _build_Phi(self, B: sp.Matrix, C: sp.Matrix) -> sp.Matrix:
        """
        Builds the Phi matrix, which is used in the MPC control step.
        Args:
            B (sp.Matrix): Input matrix.
            C (sp.Matrix): Output matrix.
        Returns:
            sp.Matrix: The Phi matrix, which is a block matrix containing the
            contributions of the inputs to the outputs at each step in the prediction horizon.
        """
        Phi = sp.zeros(self.OUTPUT_SIZE * self.Np,
                       self.INPUT_SIZE * self.Nc)

        for i in range(self.Nc):
            for j in range(i, self.Np):
                exponent = j - i
                if exponent == 0:
                    blok = C * B
                else:
                    blok = C * self._exponential_A_list[exponent - 1] * B

                r0, c0 = j * self.OUTPUT_SIZE, i * self.INPUT_SIZE
                Phi[r0:r0 + self.OUTPUT_SIZE,
                    c0:c0 + self.INPUT_SIZE] = blok
        return Phi


class MPC_ReferenceTrajectory:
    """
    A class to handle the reference trajectory for Model Predictive Control (MPC).
    This class manages the reference vector, which can either be a single row vector
    or multiple row vectors, and provides a method to calculate the difference
    between the reference vector and the predicted state.
    Attributes:
        reference_vector (np.ndarray): The reference trajectory vector.
        Np (int): Prediction horizon.
        OUTPUT_SIZE (int): Number of outputs in the system.
        follow_flag (bool): Indicates whether the reference vector has multiple rows.
    """

    def __init__(self, reference_vector: np.ndarray, Np: int):
        if reference_vector.shape[1] == Np:
            self.follow_flag = True
        elif reference_vector.shape[1] == 1:
            self.follow_flag = False
        else:
            raise ValueError(
                "Reference vector must be either a single row vector or a Np row vectors.")

        self.reference_vector = reference_vector

        self.Np = Np
        self.OUTPUT_SIZE = reference_vector.shape[0]

    def calculate_dif(self, Fx: np.ndarray) -> np.ndarray:
        """
        Calculates the difference between the reference vector and the predicted state.
        Args:
            Fx (np.ndarray): The predicted state vector.
        Returns:
            np.ndarray: The difference vector, which is the reference vector minus the predicted state.
        """
        dif = np.zeros((self.Np * self.OUTPUT_SIZE, 1))

        if self.follow_flag:
            for i in range(self.Np):
                for j in range(self.OUTPUT_SIZE):
                    dif[i * self.OUTPUT_SIZE + j, :] = \
                        self.reference_vector[j, i] - \
                        Fx[i * self.OUTPUT_SIZE + j, :]
        else:
            for i in range(self.Np):
                for j in range(self.OUTPUT_SIZE):
                    dif[i * self.OUTPUT_SIZE + j, :] = \
                        self.reference_vector[j, :] - \
                        Fx[i * self.OUTPUT_SIZE + j, :]

        return dif
