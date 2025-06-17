import numpy as np
from observers.eso import ESO
from .controller import Controller


class ADRCJointController(Controller):
    def __init__(self, b, kp, kd, p, q0, Tp):
        self.u_previous = 0
        self.b = b
        self.kp = kp
        self.kd = kd

        A = np.array([[0, 1, 0],
                      [0, 0, 1],
                      [0, 0, 0]])
        B = np.array([[0],[self.b],[0]])
        L = np.array([[3 * p],[3 * p**2],[p**3]])
        W = np.array([[1, 0, 0]])
        
        self.eso = ESO(A, B, W, L, q0, Tp)

    def set_b(self, b):
        self.b = b
        self.eso.set_B(np.array([[0], [self.b], [0]]))
        return 

    def calculate_control(self, x, q_d, q_d_dot, q_d_ddot):
        q, _ = x
        _, q_hat_dot, f = self.eso.get_state()

        v = self.kp * (q_d - q) + self.kd * (q_d_dot - q_hat_dot) + q_d_ddot
        u = ((v - f) / self.b)

        self.u_previous = u
        self.eso.update(q, self.u_previous)

        return u
