import numpy as np
from trajectory_generators.trajectory_generator import TrajectoryGenerator


class Poly3(TrajectoryGenerator):
    def __init__(self, start_q, desired_q, T):
        self.T = T
        self.q_0 = start_q
        self.q_k = desired_q

        self.a_0 = self.q_0
        self.a_1 = 3 * self.q_0
        self.a_2 = 3 * self.q_k
        self.a_3 = self.q_k

    def generate(self, t):
        t /= self.T

        q = self.a_3 * t**3 + self.a_2 * t**2 * (1 - t) + self.a_1 * t * (1 - t)**2 + self.a_0 * (1 - t)**3
        q_dot = self.a_3 * t**2 + 2 * self.a_2 * t - 3 * self.a_2 * t ** 2 + self.a_1 - 4 * self.a_1 * t + 3 * self.a_1 * t ** 2 - 3 * self.a_0 + 6 * self.a_0 * t - 3 * self.a_0 * t ** 2
        q_ddot = 6 * self.a_3 * t + 2 * self.a_2 - 6 * self.a_2 * t - 4 * self.a_1 + 6 * self.a_1 * t + 6 * self.a_0 - 6 * self.a_0 * t

        return q, q_dot / self.T, q_ddot / self.T**2
