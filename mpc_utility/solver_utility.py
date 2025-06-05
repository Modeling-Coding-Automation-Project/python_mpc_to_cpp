import numpy as np
import sympy as sp

from external_libraries.python_optimization_to_cpp.python_optimization.qp_active_set import QP_ActiveSetSolver


class DU_U_Y_Limits:
    def __init__(self,
                 delta_U_min: np.ndarray = None, delta_U_max: np.ndarray = None,
                 U_min: np.ndarray = None, U_max: np.ndarray = None,
                 Y_min: np.ndarray = None, Y_max: np.ndarray = None):
        self.delta_U_min = delta_U_min
        self.delta_U_max = delta_U_max
        self.U_min = U_min
        self.U_max = U_max
        self.Y_min = Y_min
        self.Y_max = Y_max

        self._U_size, self._Y_size = self.check_U_Y_size(
            delta_U_min=delta_U_min,
            delta_U_max=delta_U_max,
            U_min=U_min,
            U_max=U_max,
            Y_min=Y_min,
            Y_max=Y_max
        )

        self.check_min_max_compatibility()

        self._number_of_delta_U_constraints = None
        self._number_of_U_constraints = None
        self._number_of_Y_constraints = None

        self._number_of_delta_U_constraints, \
            self._number_of_U_constraints, \
            self._number_of_Y_constraints = self.count_check_constraints()

        self._delta_U_min_active_set, self._delta_U_max_active_set, \
            self._U_min_active_set, self._U_max_active_set, \
            self._Y_min_active_set, self._Y_max_active_set = \
            self.generate_DU_U_Y_active_set()

    def check_U_Y_size(self, delta_U_min: np.ndarray = None, delta_U_max: np.ndarray = None,
                       U_min: np.ndarray = None, U_max: np.ndarray = None,
                       Y_min: np.ndarray = None, Y_max: np.ndarray = None):

        U_size = 0
        if delta_U_min is not None:
            U_size = delta_U_min.shape[0]
        if delta_U_max is not None and delta_U_max.shape[0] != U_size:
            raise ValueError(
                "delta_U_max must have the same size as delta_U_min.")
        if U_min is not None and U_min.shape[0] != U_size:
            raise ValueError("U_min must have the same size as delta_U_min.")
        if U_max is not None and U_max.shape[0] != U_size:
            raise ValueError("U_max must have the same size as delta_U_min.")

        Y_size = 0
        if Y_min is not None:
            Y_size = Y_min.shape[0]
        if Y_max is not None and Y_max.shape[0] != Y_size:
            raise ValueError("Y_max must have the same size as Y_min.")

        return U_size, Y_size

    def check_min_max_compatibility(self):

        if self.delta_U_min is not None and self.delta_U_min.shape[1] > 1:
            raise ValueError("delta_U_min must be a (n, 1) vector.")
        if self.delta_U_max is not None and self.delta_U_max.shape[1] > 1:
            raise ValueError("delta_U_max must be a (n, 1) vector.")

        if self.U_min is not None and self.U_min.shape[1] > 1:
            raise ValueError("U_min must be a (n, 1) vector.")
        if self.U_max is not None and self.U_max.shape[1] > 1:
            raise ValueError("U_max must be a (n, 1) vector.")

        if self.Y_min is not None and self.Y_min.shape[1] > 1:
            raise ValueError("Y_min must be a (n, 1) vector.")
        if self.Y_max is not None and self.Y_max.shape[1] > 1:
            raise ValueError("Y_max must be a (n, 1) vector.")

        if self.delta_U_min is not None and self._U_size != self.delta_U_min.shape[0]:
            raise ValueError(
                "size of delta_U_min doesn't match the size of initialized ones.")
        if self.delta_U_max is not None and self._U_size != self.delta_U_max.shape[0]:
            raise ValueError(
                "size of delta_U_max doesn't match the size of initialized ones.")

        if self.U_min is not None and self._U_size != self.U_min.shape[0]:
            raise ValueError(
                "size of U_min doesn't match the size of initialized ones.")
        if self.U_max is not None and self._U_size != self.U_max.shape[0]:
            raise ValueError(
                "size of U_max doesn't match the size of initialized ones.")

        if self.Y_min is not None and self._Y_size != self.Y_min.shape[0]:
            raise ValueError(
                "size of Y_min doesn't match the size of initialized ones.")
        if self.Y_max is not None and self._Y_size != self.Y_max.shape[0]:
            raise ValueError(
                "size of Y_max doesn't match the size of initialized ones.")

    def count_check_constraints(self):
        self.check_min_max_compatibility()

        number_of_delta_U_constraints = 0

        if self.delta_U_min is not None:
            for i in range(self.delta_U_min.shape[0]):
                if np.isfinite(self.delta_U_min[i]):
                    number_of_delta_U_constraints += 1
        if self.delta_U_max is not None:
            for i in range(self.delta_U_max.shape[0]):
                if np.isfinite(self.delta_U_max[i]):
                    number_of_delta_U_constraints += 1

        if self._number_of_delta_U_constraints is not None and \
                (self._number_of_delta_U_constraints != number_of_delta_U_constraints):
            raise ValueError(
                "The number of delta_U constraints does not match the expected count.")

        number_of_U_constraints = 0

        if self.U_min is not None:
            for i in range(self.U_min.shape[0]):
                if np.isfinite(self.U_min[i]):
                    number_of_U_constraints += 1
        if self.U_max is not None:
            for i in range(self.U_max.shape[0]):
                if np.isfinite(self.U_max[i]):
                    number_of_U_constraints += 1

        if self._number_of_U_constraints is not None and \
                (self._number_of_U_constraints != number_of_U_constraints):
            raise ValueError(
                "The number of U constraints does not match the expected count.")

        number_of_Y_constraints = 0

        if self.Y_min is not None:
            for i in range(self.Y_min.shape[0]):
                if np.isfinite(self.Y_min[i]):
                    number_of_Y_constraints += 1
        if self.Y_max is not None:
            for i in range(self.Y_max.shape[0]):
                if np.isfinite(self.Y_max[i]):
                    number_of_Y_constraints += 1

        if self._number_of_Y_constraints is not None and \
                (self._number_of_Y_constraints != number_of_Y_constraints):
            raise ValueError(
                "The number of Y constraints does not match the expected count.")

        return (number_of_delta_U_constraints,
                number_of_U_constraints,
                number_of_Y_constraints)

    def generate_DU_U_Y_active_set(self):
        delta_U_min_active_set = np.zeros(self._U_size, dtype=bool)
        delta_U_max_active_set = np.zeros(self._U_size, dtype=bool)
        U_min_active_set = np.zeros(self._U_size, dtype=bool)
        U_max_active_set = np.zeros(self._U_size, dtype=bool)
        Y_min_active_set = np.zeros(self._Y_size, dtype=bool)
        Y_max_active_set = np.zeros(self._Y_size, dtype=bool)

        for i in range(self._U_size):
            if self.delta_U_min is not None and np.isfinite(self.delta_U_min[i]):
                delta_U_min_active_set[i] = True
            if self.delta_U_max is not None and np.isfinite(self.delta_U_max[i]):
                delta_U_max_active_set[i] = True

            if self.U_min is not None and np.isfinite(self.U_min[i]):
                U_min_active_set[i] = True
            if self.U_max is not None and np.isfinite(self.U_max[i]):
                U_max_active_set[i] = True

        for i in range(self._Y_size):
            if self.Y_min is not None and np.isfinite(self.Y_min[i]):
                Y_min_active_set[i] = True
            if self.Y_max is not None and np.isfinite(self.Y_max[i]):
                Y_max_active_set[i] = True

        return delta_U_min_active_set, delta_U_max_active_set, \
            U_min_active_set, U_max_active_set, \
            Y_min_active_set, Y_max_active_set

    def update_min_max(self, delta_U_min: np.ndarray = None, delta_U_max: np.ndarray = None,
                       U_min: np.ndarray = None, U_max: np.ndarray = None,
                       Y_min: np.ndarray = None, Y_max: np.ndarray = None):

        if delta_U_min is not None and delta_U_min.shape[0] != self._U_size:
            for i in range(self._U_size):
                if self._delta_U_active_set[i] and \
                        np.isfinite(delta_U_min[i, 0]):
                    self.delta_U_min[i, 0] = delta_U_min[i, 0]

        if delta_U_max is not None and delta_U_max.shape[0] != self._U_size:
            for i in range(self._U_size):
                if self._delta_U_active_set[i] and \
                        np.isfinite(delta_U_max[i, 0]):
                    self.delta_U_max[i, 0] = delta_U_max[i, 0]

        if U_min is not None and U_min.shape[0] != self._U_size:
            for i in range(self._U_size):
                if self._U_active_set[i] and \
                        np.isfinite(U_min[i, 0]):
                    self.U_min[i, 0] = U_min[i, 0]

        if U_max is not None and U_max.shape[0] != self._U_size:
            for i in range(self._U_size):
                if self._U_active_set[i] and \
                        np.isfinite(U_max[i, 0]):
                    self.U_max[i, 0] = U_max[i, 0]

        if Y_min is not None and Y_min.shape[0] != self._Y_size:
            for i in range(self._Y_size):
                if self._Y_active_set[i] and \
                        np.isfinite(Y_min[i, 0]):
                    self.Y_min[i, 0] = Y_min[i, 0]

        if Y_max is not None and Y_max.shape[0] != self._Y_size:
            for i in range(self._Y_size):
                if self._Y_active_set[i] and \
                        np.isfinite(Y_max[i, 0]):
                    self.Y_max[i, 0] = Y_max[i, 0]

    def is_delta_U_min_active(self, index: int) -> bool:
        if index < 0:
            index = 0
        if index >= self._U_size:
            index = self._U_size - 1

        return self._delta_U_min_active_set[index]

    def is_delta_U_max_active(self, index: int) -> bool:
        if index < 0:
            index = 0
        if index >= self._U_size:
            index = self._U_size - 1

        return self._delta_U_max_active_set[index]

    def is_U_min_active(self, index: int) -> bool:
        if index < 0:
            index = 0
        if index >= self._U_size:
            index = self._U_size - 1

        return self._U_min_active_set[index]

    def is_U_max_active(self, index: int) -> bool:
        if index < 0:
            index = 0
        if index >= self._U_size:
            index = self._U_size - 1

        return self._U_max_active_set[index]

    def is_Y_min_active(self, index: int) -> bool:
        if index < 0:
            index = 0
        if index >= self._Y_size:
            index = self._Y_size - 1

        return self._Y_min_active_set[index]

    def is_Y_max_active(self, index: int) -> bool:
        if index < 0:
            index = 0
        if index >= self._Y_size:
            index = self._Y_size - 1

        return self._Y_max_active_set[index]

    def get_number_of_all_constraints(self):
        return self._number_of_delta_U_constraints + \
            self._number_of_U_constraints + \
            self._number_of_Y_constraints

    def get_number_of_delta_U_constraints(self):
        return self._number_of_delta_U_constraints

    def get_number_of_U_constraints(self):
        return self._number_of_U_constraints

    def get_number_of_Y_constraints(self):
        return self._number_of_Y_constraints

    def get_U_size(self):
        return self._U_size

    def get_Y_size(self):
        return self._Y_size


class LTI_MPC_QP_Solver:
    def __init__(self, number_of_variables: int, U: np.ndarray, X: np.ndarray,
                 Phi: np.ndarray, F: np.ndarray, delta_U_Nc: np.ndarray,
                 delta_U_min: np.ndarray = None, delta_U_max: np.ndarray = None,
                 U_min: np.ndarray = None, U_max: np.ndarray = None,
                 Y_min: np.ndarray = None, Y_max: np.ndarray = None):

        self.number_of_variables = number_of_variables

        self.DU_U_Y_Limits = DU_U_Y_Limits(
            delta_U_min=delta_U_min,
            delta_U_max=delta_U_max,
            U_min=U_min,
            U_max=U_max,
            Y_min=Y_min,
            Y_max=Y_max
        )
        self.DU_U_Y_Limits.count_check_constraints()

        self.number_of_constraints = self.DU_U_Y_Limits.get_number_of_all_constraints()

        self.active_set_solver = QP_ActiveSetSolver(
            number_of_variables=number_of_variables,
            number_of_constraints=self.number_of_constraints,
            x=delta_U_Nc)

        self.M = np.zeros(
            (self.number_of_constraints, self.number_of_variables))
        self.gamma = np.zeros((self.number_of_constraints, 1))

        self.U_shape = U.shape
        self.X_shape = X.shape
        self.Phi_shape = Phi.shape
        self.F_shape = F.shape

        self.update_constraints(U, X, Phi, F)

    def update_min_max(self, delta_U_min: np.ndarray = None, delta_U_max: np.ndarray = None,
                       U_min: np.ndarray = None, U_max: np.ndarray = None,
                       Y_min: np.ndarray = None, Y_max: np.ndarray = None):

        self.DU_U_Y_Limits.update_min_max(
            delta_U_min=delta_U_min,
            delta_U_max=delta_U_max,
            U_min=U_min,
            U_max=U_max,
            Y_min=Y_min,
            Y_max=Y_max
        )

    def _calculate_M_gamma_delta_U(self, total_index: int):

        initial_position = total_index

        for i in range(self.DU_U_Y_Limits.get_U_size()):
            set_count = 0
            if self.DU_U_Y_Limits.is_delta_U_min_active(i):
                self.M[initial_position + 2 * i, i] = -1.0
                self.gamma[initial_position + 2 * i, 0] = - \
                    self.DU_U_Y_Limits.delta_U_min[i, 0]

                set_count += 1

            if self.DU_U_Y_Limits.is_delta_U_max_active(i):
                self.M[initial_position + 2 * i + 1,
                       i] = 1.0
                self.gamma[initial_position + 2 * i + 1, 0] = \
                    self.DU_U_Y_Limits.delta_U_max[i, 0]

                set_count += 1

            total_index += set_count

        return total_index

    def _calculate_M_gamma_U(self, total_index: int, U: np.ndarray):

        initial_position = total_index

        for i in range(self.DU_U_Y_Limits.get_U_size()):
            set_count = 0
            if self.DU_U_Y_Limits.is_U_min_active(i):
                self.M[initial_position + 2 * i, i] = -1.0
                self.gamma[initial_position + 2 * i, 0] = - \
                    self.DU_U_Y_Limits.U_min[i, 0] + U[i, 0]

                set_count += 1

            if self.DU_U_Y_Limits.is_U_max_active(i):
                self.M[initial_position + 2 * i + 1,
                       i] = 1.0
                self.gamma[initial_position + 2 * i + 1, 0] = \
                    self.DU_U_Y_Limits.U_max[i, 0] - U[i, 0]

                set_count += 1

            total_index += set_count

        return total_index

    def _calculate_M_gamma_Y(self, total_index: int, X: np.ndarray,
                             Phi: np.ndarray, F: np.ndarray):

        initial_position = total_index
        F_X = F @ X

        for i in range(self.DU_U_Y_Limits.get_Y_size()):
            if self.DU_U_Y_Limits.is_Y_min_active(i):
                self.M[initial_position + 2 * i, :] = -Phi[i, :]
                self.gamma[initial_position + 2 * i, 0] = -self.DU_U_Y_Limits.Y_min[i, 0] + \
                    F_X[i, 0]

            if self.DU_U_Y_Limits.is_Y_max_active(i):
                self.M[initial_position + 2 * i + 1,
                       :] = Phi[i, :]
                self.gamma[initial_position + 2 * i + 1, 0] = \
                    self.DU_U_Y_Limits.Y_max[i, 0] - F_X[i, 0]

    def update_constraints(self,
                           U: np.ndarray, X: np.ndarray,
                           Phi: np.ndarray, F: np.ndarray):

        if not (U.shape[0] == self.U_shape[0]) or \
                not (U.shape[1] == self.U_shape[1]):
            raise ValueError("U shape does not match the initialized shape.")

        if not (X.shape[0] == self.X_shape[0]) or \
                not (X.shape[1] == self.X_shape[1]):
            raise ValueError("X shape does not match the initialized shape.")

        if not (Phi.shape[0] == self.Phi_shape[0]) or \
                not (Phi.shape[1] == self.Phi_shape[1]):
            raise ValueError("Phi shape does not match the initialized shape.")

        if not (F.shape[0] == self.F_shape[0]) or \
                not (F.shape[1] == self.F_shape[1]):
            raise ValueError("F shape does not match the initialized shape.")

        if 0 == self.number_of_constraints:
            return

        total_index = 0

        total_index = self._calculate_M_gamma_delta_U(total_index)
        total_index = self._calculate_M_gamma_U(total_index, U)
        self._calculate_M_gamma_Y(total_index, X, Phi, F)
