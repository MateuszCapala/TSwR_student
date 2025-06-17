import numpy as np
from models.manipulator_model import ManipulatorModel
from .controller import Controller


class FeedbackLinearizationController(Controller):
    def __init__(self, Tp):
        self.model = ManipulatorModel(Tp)

    def calculate_control(self, x, q_r, q_r_dot, q_r_ddot):
        Kp = 30
        Kd = 20

        q = x[:2]
        q_dot = x[2:]

        v = Kp * (q_r - q) + Kd * (q_r_dot - q_dot) + q_r_ddot
       
        M = self.model.M(x)
        C = self.model.C(x)

        u = M @ v + C @ q_dot

        return u
