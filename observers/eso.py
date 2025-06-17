from copy import copy
import numpy as np


class ESO:
    def __init__(self, A, B, W, L, state, Tp):
        self.A = A
        self.B = B
        self.W = W
        self.L = L
        self.state = np.pad(np.array(state), (0, A.shape[0] - len(state)))
        self.Tp = Tp
        self.states = []

    def set_B(self, B):
        self.B = B

    def update(self, q, u):
        self.states.append(copy(self.state))
        ### TODO implement ESO update
        z = self.state.reshape(len(self.state), 1)
        z_dot = self.A @ z + self.B @ np.atleast_2d(u) + self.L @ (q - self.W @ z)
        z_dot_rsh = z_dot.reshape(1, len(z_dot))
        updated_state = (self.state + self.Tp * z_dot_rsh).flatten()
        self.state = updated_state

    def get_state(self):
        return self.state
