import numpy as np
import sympy as sp


def symbolic_to_numeric_matrix(symbolic_matrix: sp.Matrix) -> np.ndarray:
    numeric_matrix = np.zeros(
        (symbolic_matrix.shape[0], symbolic_matrix.shape[1]), dtype=float)

    for i in range(symbolic_matrix.shape[0]):
        for j in range(symbolic_matrix.shape[1]):
            numeric_matrix[i, j] = float(symbolic_matrix[i, j])

    return numeric_matrix


class SymbolicStateSpace:
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
        self.F_symbolic = self._build_F(C)
        self.Phi_symbolic = self._build_Phi(B, C)

    def _generate_exponential_A_list(self, A: sp.Matrix):
        exponential_A_list = []

        for i in range(self.Np):
            if i == 0:
                exponential_A_list.append(A)
            else:
                exponential_A_list.append(
                    exponential_A_list[i - 1] * A)

        return exponential_A_list

    def _build_F(self, C: sp.Matrix) -> sp.Matrix:

        F = sp.zeros(self.OUTPUT_SIZE * self.Np, self.STATE_SIZE)
        for i in range(self.Np):
            # C A^{i+1}
            F[i * self.OUTPUT_SIZE:(i + 1) *
              self.OUTPUT_SIZE, :] = C * self._exponential_A_list[i]
        return F

    def _build_Phi(self, B: sp.Matrix, C: sp.Matrix) -> sp.Matrix:

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
