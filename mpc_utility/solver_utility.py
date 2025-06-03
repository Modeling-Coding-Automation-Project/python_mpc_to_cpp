import numpy as np
import sympy as sp

from external_libraries.python_optimization_to_cpp.python_optimization.qp_active_set import QP_ActiveSetSolver


class UXY_Limits:
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

    def update_min_max(self, delta_U_min: np.ndarray = None, delta_U_max: np.ndarray = None,
                       U_min: np.ndarray = None, U_max: np.ndarray = None,
                       Y_min: np.ndarray = None, Y_max: np.ndarray = None):

        if delta_U_min is not None:
            self.delta_U_min = delta_U_min
        if delta_U_max is not None:
            self.delta_U_max = delta_U_max

        if U_min is not None:
            self.U_min = U_min
        if U_max is not None:
            self.U_max = U_max

        if Y_min is not None:
            self.Y_min = Y_min
        if Y_max is not None:
            self.Y_max = Y_max

        self.check_min_max_compatibility()

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

        self.UXY_Limits = UXY_Limits(
            delta_U_min=delta_U_min,
            delta_U_max=delta_U_max,
            U_min=U_min,
            U_max=U_max,
            Y_min=Y_min,
            Y_max=Y_max
        )
        self.UXY_Limits.count_check_constraints()

        self.number_of_constraints = self.UXY_Limits.get_number_of_all_constraints()

        self.active_set_solver = QP_ActiveSetSolver(
            number_of_variables=number_of_variables,
            number_of_constraints=self.number_of_constraints,
            x=x)

        self.M = np.zeros(
            (self.number_of_constraints, self.number_of_variables))
        self.gamma = np.zeros((self.number_of_constraints, 1))

    def update_constraints(self, U: np.ndarray, X: np.ndarray, Phi: np.ndarray):

        if 0 == self.number_of_constraints:
            raise ValueError(
                "No constraints defined. Please set constraints before updating.")
