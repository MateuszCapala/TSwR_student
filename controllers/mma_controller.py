import numpy as np
from .controller import Controller
from models.manipulator_model import ManipulatorModel


class MMAController(Controller):
    def __init__(self, Tp):
        Model_1 = ManipulatorModel(Tp = Tp)
        Model_1.m3 = 0.1
        Model_1.r3 = 0.05
        Model_2 = ManipulatorModel(Tp = Tp)
        Model_2.m3 = 0.01
        Model_2.r3 = 0.01
        Model_3 = ManipulatorModel(Tp = Tp)
        Model_3.m3 = 1.0
        Model_3.r3 = 0.3

        self.models = [Model_1, Model_2, Model_3]
        self.index = 0

        self.K_p = 30
        self.K_d = 20
        self.Tp = Tp

        self.u_previous = np.zeros((2, 1))
        self.x_previous = np.zeros((4, 1))

    def choose_model(self, x):
        error = []
        for model in self.models:
            M = model.M(x)
            C = model.C(x)
            q = x[:2]
            q_dot = x[2:]

            Mq = M @ q[:, None]
            Cq_dot = C @ q_dot[:, None]

            error.append(np.linalg.norm(Mq + Cq_dot))

        self.index = np.argmin(error)
        print(f"Model number: {self.index + 1} ")

    def calculate_control(self, x, q_r, q_r_dot, q_r_ddot):
        self.choose_model(x)
        q = x[:2]
        q_dot = x[2:]
        
        v = q_r_ddot + self.K_d * (q_r_dot - q_dot) + self.K_p * (q_r - q)
        M = self.models[self.index].M(x)
        C = self.models[self.index].C(x)
  
        u = M @ v[:, None] + C @ q_dot[:, None]
        
        self.x_previous = x
        self.u_previous = u

        return u
