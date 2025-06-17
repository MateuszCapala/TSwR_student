import numpy as np


class ManipulatorModel:
    def __init__(self, Tp):
        self.Tp = Tp
        self.l1 = 0.5
        self.r1 = 0.04
        self.m1 = 3                   
        self.d1 = self.l1 / 2
        self.l2 = 0.4
        self.r2 = 0.04
        self.m2 = 2.4               
        self.d2 = self.l2 / 2
        self.I_1 = 1 / 12 * self.m1 * (3 * self.r1 ** 2 + self.l1 ** 2)
        self.I_2 = 1 / 12 * self.m2 * (3 * self.r2 ** 2 + self.l2 ** 2)
        self.m3 = 1.2               
        self.r3 = 0.05
        self.I_3 = 2. / 5 * self.m3 * self.r3 ** 2

    def M(self, x):
        _, q2, _, _ = x

        alpha = self.m1 * (self.d1**2) + self.I_1 + self.m2*(self.l1**2+(self.d2**2)) + self.I_2 + self.m3 * (self.l1**2 + self.l2**2) + self.I_3
        beta = self.m2 * self.l1*self.d2 + self.m3 * self.l1 * self.l2
        gamma = self.m2 * (self.d2**2) + self.I_2 + self.m3 * self.l2**2 + self.I_3

        m11 = 2*beta * np.cos(q2) + alpha 
        m12 = beta * np.cos(q2) + gamma

        m21 = beta * np.cos(q2) + gamma
        m22 = gamma

        M = np.array([[m11, m12], [m21, m22]])
        
        return M

    def C(self, x):
        _, q2, q1_dot, q2_dot = x

        beta = self.m2 * self.l1*self.d2 + self.m3 * self.l1 * self.l2

        c11 = -beta * np.sin(q2) * q2_dot
        c12 = -beta * np.sin(q2) * (q1_dot+q2_dot)

        c21 = beta * np.sin(q2) * q1_dot
        c22 = 0

        C = np.array([[c11, c12], [c21, c22]])

        return C
 