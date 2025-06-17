import numpy as np

#from models.free_model import FreeModel
from observers.eso import ESO
from .adrc_joint_controller import ADRCJointController
from .controller import Controller
from models.manipulator_model import ManipulatorModel
#from models.ideal_model import IdealModel


class ADRFLController(Controller):
    def __init__(self, Tp, q0, Kp, Kd, p):
        self.model = ManipulatorModel(Tp)
        self.Kp = Kp
        self.Kd = Kd
        p1 = p[0]
        p2 = p[1]
        
        self.L = np.array([[3*p1, 0],[0, 3*p2],[3*p1**2, 0],[0, 3*p2**2],[p1**3, 0],[0, p2**3]])

        W = np.array([[1, 0, 0, 0, 0, 0],[0,1, 0, 0, 0, 0]])

        A = np.array([[0, 0, 1, 0, 0, 0],
                      [0, 0, 0, 1, 0, 0],
                      [0, 0, 0, 0, 1, 0],
                      [0, 0, 0, 0, 0, 1],
                      [0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0]])
        
        B = np.zeros((6, 2))

        self.eso = ESO(A, B, W, self.L, q0, Tp)
        self.update_params(q0[:2], q0[2:])

    def update_params(self, q, q_dot):
        x = np.concatenate([q, q_dot], axis=0)

        M = self.model.M(x)
        C = self.model.C(x)
        M_inv = np.linalg.inv(M)

        A = np.array([[0, 0, 1, 0, 0, 0],
                      [0, 0, 0, 1, 0, 0],
                      [0, 0, 0, 0, 1, 0],
                      [0, 0, 0, 0, 0, 1],
                      [0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0]])
        
        B = np.zeros((6, 2))

        A[2:4, 2:4] =-M_inv @ C
        B[2:4, :2] =M_inv

        self.eso.A = A
        self.eso.B = B

    def calculate_control(self, x, q_d, q_d_dot, q_d_ddot):
        M = self.model.M(x)
        C = self.model.C(x)

        q1, q2, _, _ = x
        q = np.array([q1, q2])

        estimated_state = self.eso.get_state()
        q_hat = estimated_state[0:2]
        q_dot_hat = estimated_state[2:4]
        disturbance_hat = estimated_state[4:6]

        v = self.Kp @ (q_d - q) + self.Kd @ (q_d_dot - q_dot_hat) + q_d_ddot
        u = M @ (v - disturbance_hat) + C @ q_dot_hat

        self.update_params(q_hat, q_dot_hat)
        self.eso.update(np.expand_dims(q, axis=1), np.expand_dims(u, axis=1))

        return u
