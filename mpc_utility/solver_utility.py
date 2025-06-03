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

        self.U_size = delta_U_min.shape[0]
        self.Y_size = Y_min.shape[0]

        self.check_min_max_compatibility()

        self.number_of_delta_U_constraints = None
        self.number_of_U_constraints = None
        self.number_of_Y_constraints = None

        self.number_of_delta_U_constraints, \
            self.number_of_U_constraints, \
            self.number_of_Y_constraints = self.count_check_constraints()

        self.delta_U_active_set, self.U_active_set, self.Y_active_set = \
            self.generate_DU_U_Y_active_set()

    def check_min_max_compatibility(self):

        if self.delta_U_min is not None or self.delta_U_min.shape[1] > 1:
            raise ValueError("delta_U_min must be a (n, 1) vector.")
        if self.delta_U_max is not None or self.delta_U_max.shape[1] > 1:
            raise ValueError("delta_U_max must be a (n, 1) vector.")

        if self.U_min is not None or self.U_min.shape[1] > 1:
            raise ValueError("U_min must be a (n, 1) vector.")
        if self.U_max is not None or self.U_max.shape[1] > 1:
            raise ValueError("U_max must be a (n, 1) vector.")

        if self.Y_min is not None or self.Y_min.shape[1] > 1:
            raise ValueError("Y_min must be a (n, 1) vector.")
        if self.Y_max is not None or self.Y_max.shape[1] > 1:
            raise ValueError("Y_max must be a (n, 1) vector.")

        if self.U_size != self.delta_U_min.shape[0]:
            raise ValueError(
                "size of delta_U_min doesn't match the size of initialized ones.")
        if self.U_size != self.delta_U_max.shape[0]:
            raise ValueError(
                "size of delta_U_max doesn't match the size of initialized ones.")

        if self.U_size != self.U_min.shape[0]:
            raise ValueError(
                "size of U_min doesn't match the size of initialized ones.")
        if self.U_size != self.U_max.shape[0]:
            raise ValueError(
                "size of U_max doesn't match the size of initialized ones.")

        if self.Y_size != self.Y_min.shape[0]:
            raise ValueError(
                "size of Y_min doesn't match the size of initialized ones.")
        if self.Y_size != self.Y_max.shape[0]:
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

        if self.number_of_delta_U_constraints is not None and \
                (self.number_of_delta_U_constraints != number_of_delta_U_constraints):
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

        if self.number_of_U_constraints is not None and \
                (self.number_of_U_constraints != number_of_U_constraints):
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

        if self.number_of_Y_constraints is not None and \
                (self.number_of_Y_constraints != number_of_Y_constraints):
            raise ValueError(
                "The number of Y constraints does not match the expected count.")

        return (number_of_delta_U_constraints,
                number_of_U_constraints,
                number_of_Y_constraints)

    def generate_DU_U_Y_active_set(self):
        delta_U_active_set = np.zeros(self.U_size, dtype=bool)
        U_active_set = np.zeros(self.U_size, dtype=bool)
        Y_active_set = np.zeros(self.Y_size, dtype=bool)

        for i in range(self.U_size):
            if self.delta_U_min is not None and np.isfinite(self.delta_U_min[i]):
                delta_U_active_set[i] = True
            if self.delta_U_max is not None and np.isfinite(self.delta_U_max[i]):
                delta_U_active_set[i] = True

            if self.U_min is not None and np.isfinite(self.U_min[i]):
                U_active_set[i] = True
            if self.U_max is not None and np.isfinite(self.U_max[i]):
                U_active_set[i] = True

        for i in range(self.Y_size):
            if self.Y_min is not None and np.isfinite(self.Y_min[i]):
                Y_active_set[i] = True
            if self.Y_max is not None and np.isfinite(self.Y_max[i]):
                Y_active_set[i] = True

        return delta_U_active_set, U_active_set, Y_active_set

    def update_min_max(self, delta_U_min: np.ndarray = None, delta_U_max: np.ndarray = None,
                       U_min: np.ndarray = None, U_max: np.ndarray = None,
                       Y_min: np.ndarray = None, Y_max: np.ndarray = None):

        if delta_U_min is not None and delta_U_min.shape[0] != self.U_size:
            for i in range(self.U_size):
                if self.delta_U_active_set[i] and \
                        np.isfinite(delta_U_min[i, 0]):
                    self.delta_U_min[i, 0] = delta_U_min[i, 0]

        if delta_U_max is not None and delta_U_max.shape[0] != self.U_size:
            for i in range(self.U_size):
                if self.delta_U_active_set[i] and \
                        np.isfinite(delta_U_max[i, 0]):
                    self.delta_U_max[i, 0] = delta_U_max[i, 0]

        if U_min is not None and U_min.shape[0] != self.U_size:
            for i in range(self.U_size):
                if self.U_active_set[i] and \
                        np.isfinite(U_min[i, 0]):
                    self.U_min[i, 0] = U_min[i, 0]

        if U_max is not None and U_max.shape[0] != self.U_size:
            for i in range(self.U_size):
                if self.U_active_set[i] and \
                        np.isfinite(U_max[i, 0]):
                    self.U_max[i, 0] = U_max[i, 0]

        if Y_min is not None and Y_min.shape[0] != self.Y_size:
            for i in range(self.Y_size):
                if self.Y_active_set[i] and \
                        np.isfinite(Y_min[i, 0]):
                    self.Y_min[i, 0] = Y_min[i, 0]

        if Y_max is not None and Y_max.shape[0] != self.Y_size:
            for i in range(self.Y_size):
                if self.Y_active_set[i] and \
                        np.isfinite(Y_max[i, 0]):
                    self.Y_max[i, 0] = Y_max[i, 0]

    def get_number_of_all_constraints(self):
        return self.number_of_delta_U_constraints + \
            self.number_of_U_constraints + \
            self.number_of_Y_constraints


class LTI_MPC_QP_Solver:
    def __init__(self, number_of_variables: int, x: np.ndarray = None,
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
            x=x)

        self.M = np.zeros(
            (self.number_of_constraints, self.number_of_variables))
        self.gamma = np.zeros((self.number_of_constraints, 1))

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

    def update_constraints(self, U: np.ndarray, X: np.ndarray, Phi: np.ndarray):

        if 0 == self.number_of_constraints:
            return
