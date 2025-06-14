"""
File: linear_solver_utility.py

This module provides the DU_U_Y_Limits class, which is a utility for managing and validating constraints on control increments (delta_U), control variables (U), and output variables (Y) in the context of Model Predictive Control (MPC) or similar optimization problems. The class supports initialization, validation, and dynamic management of lower and upper bounds for these variables, as well as tracking which constraints are active or inactive.
"""
import numpy as np
import sympy as sp

from mpc_utility.state_space_utility import *
from external_libraries.python_optimization_to_cpp.python_optimization.qp_active_set import QP_ActiveSetSolver

MAX_ITERATION_DEFAULT = 10
TOL_DEFAULT = 1e-8


class DU_U_Y_Limits:
    """
    DU_U_Y_Limits class manages the constraints for control increments (delta_U), control variables (U), and output variables (Y) in a Model Predictive Control (MPC) context.
    It allows for initialization with specific bounds, checks for compatibility, counts constraints, and generates active sets for each type of constraint.
    Attributes:
        delta_U_min (np.ndarray): Lower bounds for control increments.
        delta_U_max (np.ndarray): Upper bounds for control increments.
        U_min (np.ndarray): Lower bounds for control variables.
        U_max (np.ndarray): Upper bounds for control variables.
        Y_min (np.ndarray): Lower bounds for output variables.
        Y_max (np.ndarray): Upper bounds for output variables.
    """

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

        self._delta_U_min_size, self._delta_U_max_size, \
            self._U_min_size, self._U_max_size, \
            self._Y_min_size, self._Y_max_size \
            = self.check_U_Y_size(
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
        """
        Checks the sizes of the provided constraints and ensures they are compatible.
        Args:
            delta_U_min (np.ndarray): Lower bounds for control increments.
            delta_U_max (np.ndarray): Upper bounds for control increments.
            U_min (np.ndarray): Lower bounds for control variables.
            U_max (np.ndarray): Upper bounds for control variables.
            Y_min (np.ndarray): Lower bounds for output variables.
            Y_max (np.ndarray): Upper bounds for output variables.
        Returns:
            tuple: Sizes of the constraints in the order of delta_U_min, delta_U_max, U_min, U_max, Y_min, Y_max.
        Raises:
            ValueError: If the sizes of the constraints are not compatible.
        """
        delta_U_min_size = 0
        delta_U_max_size = 0
        U_min_size = 0
        U_max_size = 0

        if delta_U_min is not None:
            delta_U_min_size = delta_U_min.shape[0]

        U_size = delta_U_min_size

        if delta_U_max is not None:
            if delta_U_max.shape[0] != U_size and U_size > 0:
                raise ValueError(
                    "delta_U_max must have the same size as delta_U_min.")

            delta_U_max_size = delta_U_max.shape[0]
            U_size = delta_U_max_size

        if U_min is not None:
            if U_min.shape[0] != U_size and U_size > 0:
                raise ValueError(
                    "U_min must have the same size as delta_U_min.")

            U_min_size = U_min.shape[0]
            U_size = U_min_size

        if U_max is not None:
            if U_max.shape[0] != U_size and U_size > 0:
                raise ValueError(
                    "U_max must have the same size as delta_U_min.")

            U_max_size = U_max.shape[0]

        Y_min_size = 0
        Y_max_size = 0

        if Y_min is not None:
            Y_min_size = Y_min.shape[0]
        if Y_max is not None:
            if Y_max.shape[0] != Y_min_size and Y_min_size > 0:
                raise ValueError("Y_max must have the same size as Y_min.")
            Y_max_size = Y_max.shape[0]

        return delta_U_min_size, delta_U_max_size, \
            U_min_size, U_max_size, Y_min_size, Y_max_size

    def check_min_max_compatibility(self):
        """
        Checks the compatibility of the provided constraints.
        Raises:
            ValueError: If the constraints are not compatible.
        """
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

        if self.delta_U_min is not None and \
                self._delta_U_min_size != self.delta_U_min.shape[0]:
            raise ValueError(
                "size of delta_U_min doesn't match the size of initialized ones.")
        if self.delta_U_max is not None and \
                self._delta_U_max_size != self.delta_U_max.shape[0]:
            raise ValueError(
                "size of delta_U_max doesn't match the size of initialized ones.")

        if self.U_min is not None and \
                self._U_min_size != self.U_min.shape[0]:
            raise ValueError(
                "size of U_min doesn't match the size of initialized ones.")
        if self.U_max is not None and \
                self._U_max_size != self.U_max.shape[0]:
            raise ValueError(
                "size of U_max doesn't match the size of initialized ones.")

        if self.Y_min is not None and \
                self._Y_min_size != self.Y_min.shape[0]:
            raise ValueError(
                "size of Y_min doesn't match the size of initialized ones.")
        if self.Y_max is not None and \
                self._Y_max_size != self.Y_max.shape[0]:
            raise ValueError(
                "size of Y_max doesn't match the size of initialized ones.")

    def count_check_constraints(self):
        """
        Counts the number of active constraints for delta_U, U, and Y.
        Returns:
            tuple: Number of active constraints for delta_U, U, and Y.
        Raises:
            ValueError: If the number of constraints does not match the expected count.
        """
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
        """
        Generates active sets for delta_U, U, and Y constraints based on their finite values.
        Returns:
            tuple: Active sets for delta_U_min, delta_U_max, U_min, U_max, Y_min, Y_max.
        """
        delta_U_min_active_set = np.zeros(self._delta_U_min_size, dtype=bool)
        delta_U_max_active_set = np.zeros(self._delta_U_max_size, dtype=bool)
        U_min_active_set = np.zeros(self._U_min_size, dtype=bool)
        U_max_active_set = np.zeros(self._U_max_size, dtype=bool)
        Y_min_active_set = np.zeros(self._Y_min_size, dtype=bool)
        Y_max_active_set = np.zeros(self._Y_max_size, dtype=bool)

        for i in range(self._delta_U_min_size):
            if self.delta_U_min is not None and np.isfinite(self.delta_U_min[i]):
                delta_U_min_active_set[i] = True
        for i in range(self._delta_U_max_size):
            if self.delta_U_max is not None and np.isfinite(self.delta_U_max[i]):
                delta_U_max_active_set[i] = True

        for i in range(self._U_min_size):
            if self.U_min is not None and np.isfinite(self.U_min[i]):
                U_min_active_set[i] = True
        for i in range(self._U_max_size):
            if self.U_max is not None and np.isfinite(self.U_max[i]):
                U_max_active_set[i] = True

        for i in range(self._Y_min_size):
            if self.Y_min is not None and np.isfinite(self.Y_min[i]):
                Y_min_active_set[i] = True
        for i in range(self._Y_max_size):
            if self.Y_max is not None and np.isfinite(self.Y_max[i]):
                Y_max_active_set[i] = True

        return delta_U_min_active_set, delta_U_max_active_set, \
            U_min_active_set, U_max_active_set, \
            Y_min_active_set, Y_max_active_set

    def update_min_max(self, delta_U_min: np.ndarray = None, delta_U_max: np.ndarray = None,
                       U_min: np.ndarray = None, U_max: np.ndarray = None,
                       Y_min: np.ndarray = None, Y_max: np.ndarray = None):
        """
        Updates the minimum and maximum constraints for delta_U, U, and Y.
        Args:
            delta_U_min (np.ndarray): New lower bounds for control increments.
            delta_U_max (np.ndarray): New upper bounds for control increments.
            U_min (np.ndarray): New lower bounds for control variables.
            U_max (np.ndarray): New upper bounds for control variables.
            Y_min (np.ndarray): New lower bounds for output variables.
            Y_max (np.ndarray): New upper bounds for output variables.
        """
        if delta_U_min is not None and delta_U_min.shape[0] != self._delta_U_min_size:
            for i in range(self._delta_U_min_size):
                if self._delta_U_min_active_set[i] and \
                        np.isfinite(delta_U_min[i, 0]):
                    self.delta_U_min[i, 0] = delta_U_min[i, 0]

        if delta_U_max is not None and delta_U_max.shape[0] != self._delta_U_max_size:
            for i in range(self._delta_U_max_size):
                if self._delta_U_max_active_set[i] and \
                        np.isfinite(delta_U_max[i, 0]):
                    self.delta_U_max[i, 0] = delta_U_max[i, 0]

        if U_min is not None and U_min.shape[0] != self._U_min_size:
            for i in range(self._U_min_size):
                if self._U_min_active_set[i] and \
                        np.isfinite(U_min[i, 0]):
                    self.U_min[i, 0] = U_min[i, 0]

        if U_max is not None and U_max.shape[0] != self._U_max_size:
            for i in range(self._U_max_size):
                if self._U_max_active_set[i] and \
                        np.isfinite(U_max[i, 0]):
                    self.U_max[i, 0] = U_max[i, 0]

        if Y_min is not None and Y_min.shape[0] != self._Y_min_size:
            for i in range(self._Y_min_size):
                if self._Y_min_active_set[i] and \
                        np.isfinite(Y_min[i, 0]):
                    self.Y_min[i, 0] = Y_min[i, 0]

        if Y_max is not None and Y_max.shape[0] != self._Y_max_size:
            for i in range(self._Y_max_size):
                if self._Y_max_active_set[i] and \
                        np.isfinite(Y_max[i, 0]):
                    self.Y_max[i, 0] = Y_max[i, 0]

    # Check if the constraints are active
    def is_delta_U_min_active(self, index: int) -> bool:
        """
        Checks if the delta_U_min constraint at the given index is active.
        Args:
            index (int): Index of the delta_U_min constraint.
        Returns:
            bool: True if the constraint is active, False otherwise.
        """
        if index < 0:
            index = 0
        if index >= self._delta_U_min_size:
            index = self._delta_U_min_size - 1

        return self._delta_U_min_active_set[index]

    def is_delta_U_max_active(self, index: int) -> bool:
        """
        Checks if the delta_U_max constraint at the given index is active.
        Args:
            index (int): Index of the delta_U_max constraint.
        Returns:
            bool: True if the constraint is active, False otherwise.
        """
        if index < 0:
            index = 0
        if index >= self._delta_U_max_size:
            index = self._delta_U_max_size - 1

        return self._delta_U_max_active_set[index]

    def is_U_min_active(self, index: int) -> bool:
        """
        Checks if the U_min constraint at the given index is active.
        Args:
            index (int): Index of the U_min constraint.
        Returns:
            bool: True if the constraint is active, False otherwise.
        """
        if index < 0:
            index = 0
        if index >= self._U_min_size:
            index = self._U_min_size - 1

        return self._U_min_active_set[index]

    def is_U_max_active(self, index: int) -> bool:
        """
        Checks if the U_max constraint at the given index is active.
        Args:
            index (int): Index of the U_max constraint.
        Returns:
            bool: True if the constraint is active, False otherwise.
        """
        if index < 0:
            index = 0
        if index >= self._U_max_size:
            index = self._U_max_size - 1

        return self._U_max_active_set[index]

    def is_Y_min_active(self, index: int) -> bool:
        """
        Checks if the Y_min constraint at the given index is active.
        Args:
            index (int): Index of the Y_min constraint.
        Returns:
            bool: True if the constraint is active, False otherwise.
        """
        if index < 0:
            index = 0
        if index >= self._Y_min_size:
            index = self._Y_min_size - 1

        return self._Y_min_active_set[index]

    def is_Y_max_active(self, index: int) -> bool:
        """
        Checks if the Y_max constraint at the given index is active.
        Args:
            index (int): Index of the Y_max constraint.
        Returns:
            bool: True if the constraint is active, False otherwise.
        """
        if index < 0:
            index = 0
        if index >= self._Y_max_size:
            index = self._Y_max_size - 1

        return self._Y_max_active_set[index]

    # set constraints inactive
    def set_delta_U_min_inactive(self, index: int):
        """
        Sets the delta_U_min constraint at the given index to inactive.
        Args:
            index (int): Index of the delta_U_min constraint.
        """
        if index < 0:
            index = 0
        if index >= self._delta_U_min_size:
            index = self._delta_U_min_size - 1

        self._delta_U_min_active_set[index] = False

    def set_delta_U_max_inactive(self, index: int):
        """
        Sets the delta_U_max constraint at the given index to inactive.
        Args:
            index (int): Index of the delta_U_max constraint.
        """
        if index < 0:
            index = 0
        if index >= self._delta_U_max_size:
            index = self._delta_U_max_size - 1

        self._delta_U_max_active_set[index] = False

    def set_U_min_inactive(self, index: int):
        """
        Sets the U_min constraint at the given index to inactive.
        Args:
            index (int): Index of the U_min constraint.
        """
        if index < 0:
            index = 0
        if index >= self._U_min_size:
            index = self._U_min_size - 1

        self._U_min_active_set[index] = False

    def set_U_max_inactive(self, index: int):
        """
        Sets the U_max constraint at the given index to inactive.
        Args:
            index (int): Index of the U_max constraint.
        """
        if index < 0:
            index = 0
        if index >= self._U_max_size:
            index = self._U_max_size - 1

        self._U_max_active_set[index] = False

    def set_Y_min_inactive(self, index: int):
        """
        Sets the Y_min constraint at the given index to inactive.
        Args:
            index (int): Index of the Y_min constraint.
        """
        if index < 0:
            index = 0
        if index >= self._Y_min_size:
            index = self._Y_min_size - 1

        if True == self._Y_min_active_set[index]:
            self._number_of_Y_constraints -= 1

            self._Y_min_active_set[index] = False

    def set_Y_max_inactive(self, index: int):
        """
        Sets the Y_max constraint at the given index to inactive.
        Args:
            index (int): Index of the Y_max constraint.
        """
        if index < 0:
            index = 0
        if index >= self._Y_max_size:
            index = self._Y_max_size - 1

        if True == self._Y_max_active_set[index]:
            self._number_of_Y_constraints -= 1

            self._Y_max_active_set[index] = False

    # Getters for the sizes and counts of constraints
    def get_number_of_all_constraints(self):
        """
        Returns the total number of constraints across delta_U, U, and Y.
        Returns:
            int: Total number of constraints.
        """
        return self._number_of_delta_U_constraints + \
            self._number_of_U_constraints + \
            self._number_of_Y_constraints

    def get_number_of_delta_U_constraints(self):
        """
        Returns the number of delta_U constraints.
        Returns:
            int: Number of delta_U constraints.
        """
        return self._number_of_delta_U_constraints

    def get_number_of_U_constraints(self):
        """
        Returns the number of U constraints.
        Returns:
            int: Number of U constraints.
        """
        return self._number_of_U_constraints

    def get_number_of_Y_constraints(self):
        """
        Returns the number of Y constraints.
        Returns:
            int: Number of Y constraints.
        """
        return self._number_of_Y_constraints

    def get_delta_U_min_size(self):
        """
        Returns the size of the delta_U_min constraints.
        Returns:
            int: Size of the delta_U_min constraints.
        """
        return self._delta_U_min_size

    def get_delta_U_max_size(self):
        """
        Returns the size of the delta_U_max constraints.
        Returns:
            int: Size of the delta_U_max constraints.
        """
        return self._delta_U_max_size

    def get_U_min_size(self):
        """
        Returns the size of the U_min constraints.
        Returns:
            int: Size of the U_min constraints.
        """
        return self._U_min_size

    def get_U_max_size(self):
        """
        Returns the size of the U_max constraints.
        Returns:
            int: Size of the U_max constraints.
        """
        return self._U_max_size

    def get_Y_min_size(self):
        """
        Returns the size of the Y_min constraints.
        Returns:
            int: Size of the Y_min constraints.
        """
        return self._Y_min_size

    def get_Y_max_size(self):
        """
        Returns the size of the Y_max constraints.
        Returns:
            int: Size of the Y_max constraints.
        """
        return self._Y_max_size


class LTI_MPC_QP_Solver:
    """
    LTI_MPC_QP_Solver class implements a solver for Linear Time-Invariant (LTI) Model Predictive Control (MPC) problems using Quadratic Programming (QP).
    It utilizes the DU_U_Y_Limits class to manage constraints on control increments (delta_U), control variables (U), and output variables (Y).
    Attributes:
        number_of_variables (int): Number of control variables.
        output_size (int): Size of the output variables.
        U (np.ndarray): Control input matrix.
        X_augmented (np.ndarray): Augmented state matrix.
        Phi (np.ndarray): Prediction matrix.
        F (np.ndarray): State transition matrix.
        Weight_U_Nc (np.ndarray): Weighting matrix for control increments.
        delta_U_Nc (np.ndarray): Control increment matrix.
        delta_U_min (np.ndarray): Lower bounds for control increments.
        delta_U_max (np.ndarray): Upper bounds for control increments.
        U_min (np.ndarray): Lower bounds for control variables.
        U_max (np.ndarray): Upper bounds for control variables.
        Y_min (np.ndarray): Lower bounds for output variables.
        Y_max (np.ndarray): Upper bounds for output variables.
    """

    def __init__(self, number_of_variables: int, output_size: int,
                 U: np.ndarray, X_augmented: np.ndarray,
                 Phi: np.ndarray, F: np.ndarray,
                 Weight_U_Nc: np.ndarray,
                 delta_U_Nc: np.ndarray,
                 delta_U_min: np.ndarray = None, delta_U_max: np.ndarray = None,
                 U_min: np.ndarray = None, U_max: np.ndarray = None,
                 Y_min: np.ndarray = None, Y_max: np.ndarray = None,
                 Y_constraints_prediction_offset: int = 0,
                 max_iteration: int = MAX_ITERATION_DEFAULT,
                 tol: float = TOL_DEFAULT):

        self.max_iteration = max_iteration
        self.tol = tol

        self.number_of_variables = number_of_variables

        self.U_size = U.shape[0]
        self.Y_size = output_size

        self.DU_U_Y_Limits = DU_U_Y_Limits(
            delta_U_min=delta_U_min,
            delta_U_max=delta_U_max,
            U_min=U_min,
            U_max=U_max,
            Y_min=Y_min,
            Y_max=Y_max
        )
        self.DU_U_Y_Limits.count_check_constraints()

        self.U_shape = U.shape
        self.X_shape = X_augmented.shape
        self.Phi_shape = Phi.shape
        self.F_shape = F.shape

        self.prediction_offset = Y_constraints_prediction_offset
        self.check_Y_constraints_feasibility(Phi)

        self.number_of_constraints = \
            self.DU_U_Y_Limits.get_number_of_all_constraints()

        self.M = np.zeros(
            (self.number_of_constraints, self.number_of_variables))
        self.gamma = np.zeros((self.number_of_constraints, 1))

        self.update_constraints(U, X_augmented, Phi, F)

        self.solver = QP_ActiveSetSolver(
            number_of_variables=number_of_variables,
            number_of_constraints=self.number_of_constraints,
            X=delta_U_Nc,
            max_iteration=max_iteration,
            tol=tol)

        self.update_E(Phi=Phi,
                      Weight_U_Nc=Weight_U_Nc)

    def check_Y_constraints_feasibility(self,
                                        Phi: np.ndarray):
        """
        Checks the feasibility of Y constraints based on the prediction matrix Phi.
        Args:
            Phi (np.ndarray): Prediction matrix.
        Raises:
            ValueError: If the Phi matrix does not match the expected shape.
        """

        for i in range(self.DU_U_Y_Limits.get_Y_min_size()):
            Phi_factor_norm = np.linalg.norm(
                Phi[self.prediction_offset + i, :])

            if Phi_factor_norm < self.tol:
                print("[Warning] " +
                      f"Y[{i}] min cannot be constrained because Phi row is zero. " +
                      f"Y[{i}] min constraint is no linger considered.")

                self.DU_U_Y_Limits.set_Y_min_inactive(i)

        for i in range(self.DU_U_Y_Limits.get_Y_max_size()):
            Phi_factor_norm = np.linalg.norm(
                Phi[self.prediction_offset + i, :])

            if Phi_factor_norm < self.tol:

                print("[Warning] " +
                      f"Y[{i}] max cannot be constrained because Phi row is zero. " +
                      f"Y[{i}] max constraint is no linger considered.")

                self.DU_U_Y_Limits.set_Y_max_inactive(i)

    def update_min_max(self, delta_U_min: np.ndarray = None, delta_U_max: np.ndarray = None,
                       U_min: np.ndarray = None, U_max: np.ndarray = None,
                       Y_min: np.ndarray = None, Y_max: np.ndarray = None):
        """
        Updates the minimum and maximum constraints for delta_U, U, and Y.
        Args:
            delta_U_min (np.ndarray): New lower bounds for control increments.
            delta_U_max (np.ndarray): New upper bounds for control increments.
            U_min (np.ndarray): New lower bounds for control variables.
            U_max (np.ndarray): New upper bounds for control variables.
            Y_min (np.ndarray): New lower bounds for output variables.
            Y_max (np.ndarray): New upper bounds for output variables.
        """

        self.DU_U_Y_Limits.update_min_max(
            delta_U_min=delta_U_min,
            delta_U_max=delta_U_max,
            U_min=U_min,
            U_max=U_max,
            Y_min=Y_min,
            Y_max=Y_max
        )

    def _calculate_M_gamma_delta_U(self, total_index: int):
        """
        Calculates the M and gamma matrices for delta_U constraints.
        Args:
            total_index (int): The starting index for the M and gamma matrices.
        Returns:
            int: The updated total index after processing delta_U constraints.
        """
        initial_position = total_index

        for i in range(self.DU_U_Y_Limits.get_delta_U_min_size()):
            set_count = 0
            if self.DU_U_Y_Limits.is_delta_U_min_active(i):
                self.M[initial_position + i, i] = -1.0
                self.gamma[initial_position + i, 0] = - \
                    self.DU_U_Y_Limits.delta_U_min[i, 0]

                set_count += 1

            total_index += set_count

        initial_position = total_index

        for i in range(self.DU_U_Y_Limits.get_delta_U_max_size()):
            set_count = 0
            if self.DU_U_Y_Limits.is_delta_U_max_active(i):
                self.M[initial_position + i,
                       i] = 1.0
                self.gamma[initial_position + i, 0] = \
                    self.DU_U_Y_Limits.delta_U_max[i, 0]

                set_count += 1

            total_index += set_count

        return total_index

    def _calculate_M_gamma_U(self, total_index: int, U: np.ndarray):
        """
        Calculates the M and gamma matrices for U constraints.
        Args:
            total_index (int): The starting index for the M and gamma matrices.
            U (np.ndarray): Control input matrix.
        Returns:
            int: The updated total index after processing U constraints.
        """
        initial_position = total_index

        for i in range(self.DU_U_Y_Limits.get_U_min_size()):
            set_count = 0
            if self.DU_U_Y_Limits.is_U_min_active(i):
                self.M[initial_position + i, i] = -1.0
                self.gamma[initial_position + i, 0] = - \
                    self.DU_U_Y_Limits.U_min[i, 0] + U[i, 0]

                set_count += 1

            total_index += set_count

        initial_position = total_index

        for i in range(self.DU_U_Y_Limits.get_U_max_size()):
            set_count = 0
            if self.DU_U_Y_Limits.is_U_max_active(i):
                self.M[initial_position + i,
                       i] = 1.0
                self.gamma[initial_position + i, 0] = \
                    self.DU_U_Y_Limits.U_max[i, 0] - U[i, 0]

                set_count += 1

            total_index += set_count

        return total_index

    def _calculate_M_gamma_Y(self, total_index: int, X_augmented: np.ndarray,
                             Phi: np.ndarray, F: np.ndarray):
        """
        Calculates the M and gamma matrices for Y constraints.
        Args:
            total_index (int): The starting index for the M and gamma matrices.
            X_augmented (np.ndarray): Augmented state matrix.
            Phi (np.ndarray): Prediction matrix.
            F (np.ndarray): State transition matrix.
        """
        initial_position = total_index
        F_X = F @ X_augmented

        for i in range(self.DU_U_Y_Limits.get_Y_min_size()):
            set_count = 0
            if self.DU_U_Y_Limits.is_Y_min_active(i):

                self.M[initial_position + i, :] = \
                    -Phi[self.prediction_offset + i, :]
                self.gamma[initial_position + i, 0] = \
                    - \
                    self.DU_U_Y_Limits.Y_min[i, 0] + \
                    F_X[self.prediction_offset + i, 0]

                set_count += 1

            total_index += set_count

        initial_position = total_index

        for i in range(self.DU_U_Y_Limits.get_Y_max_size()):
            if self.DU_U_Y_Limits.is_Y_max_active(i):

                self.M[initial_position + i,
                       :] = Phi[self.prediction_offset + i, :]
                self.gamma[initial_position + i, 0] = \
                    self.DU_U_Y_Limits.Y_max[i, 0] - \
                    F_X[self.prediction_offset + i, 0]

    def update_constraints(self,
                           U: np.ndarray, X_augmented: np.ndarray,
                           Phi: np.ndarray, F: np.ndarray):
        """
        Updates the M and gamma matrices based on the current constraints.
        Args:
            U (np.ndarray): Control input matrix.
            X_augmented (np.ndarray): Augmented state matrix.
            Phi (np.ndarray): Prediction matrix.
            F (np.ndarray): State transition matrix.
        Raises:
            ValueError: If the shapes of U, X_augmented, Phi, or F do not match the initialized shapes.
        """
        if not (U.shape[0] == self.U_shape[0]) or \
                not (U.shape[1] == self.U_shape[1]):
            raise ValueError("U shape does not match the initialized shape.")

        if not (X_augmented.shape[0] == self.X_shape[0]) or \
                not (X_augmented.shape[1] == self.X_shape[1]):
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
        self._calculate_M_gamma_Y(total_index, X_augmented, Phi, F)

    def update_E(self, Phi: np.ndarray, Weight_U_Nc: np.ndarray):
        """
        Updates the E matrix used in the QP solver.
        Args:
            Phi (np.ndarray): Prediction matrix.
            Weight_U_Nc (np.ndarray): Weighting matrix for control increments.
        """
        self.solver.update_E(Phi.T @ Phi + Weight_U_Nc)

    def solve(self, Phi: np.ndarray, F: np.ndarray,
              reference_trajectory: MPC_ReferenceTrajectory,
              X_augmented: np.ndarray, Weight_U_Nc: np.ndarray = None) -> np.ndarray:
        """
        Solves the MPC problem using the QP solver.
        Args:
            Phi (np.ndarray): Prediction matrix.
            F (np.ndarray): State transition matrix.
            reference_trajectory (MPC_ReferenceTrajectory): Reference trajectory for the MPC problem.
            X_augmented (np.ndarray): Augmented state matrix.
            Weight_U_Nc (np.ndarray, optional): Weighting matrix for control increments.
        Returns:
            np.ndarray: Optimal control input.
        Raises:
            ValueError: If the shapes of Phi, F, or X_augmented do not match the initialized shapes.
        """
        L = Phi.T @ reference_trajectory.calculate_dif(F @ X_augmented)

        if Weight_U_Nc is not None:
            self.update_E(Phi, Weight_U_Nc)

        x_opt = self.solver.solve(
            L=L,
            M=self.M,
            gamma=self.gamma)

        return x_opt
