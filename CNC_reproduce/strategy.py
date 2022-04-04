import numpy as np
from random import choice
import numba as nb



class Strategy():
    def __init__(self, init_obs):
        self.horizon = 80
        self.beta = 1
        self.z_range = range(-3, 4)
        self.z_range_nb = nb.typed.List()
        for z in self.z_range:
            self.z_range_nb.append(z)
        self.a_range = [0, 2, 3]
        self.frames_per_step = 4
        self.t = 0
        self.s = np.array([self.obs_ppc(init_obs)] * self.frames_per_step)
        self.R = 0
        self.R_per_10k_steps = [] # 10k-step interval
        self.R_in_last_20_episodes = [0]
        self.average_score_per_episode = [] # 20-episode interval

        self.s_seq = []
        self.a_seq = []
        self.r_seq = []

        self.z_bucket = [{'sum': 0} for _ in self.a_range]

        self.eps = 1
        self.eps_ultimate = 0.02
        self.eps_decay_duration = 200000
        self.eps_decay = (self.eps - self.eps_ultimate) / self.eps_decay_duration

    def save(self):
        self.z_range_nb = None

    def load(self):
        self.z_range_nb = nb.typed.List()
        for z in self.z_range:
            self.z_range_nb.append(z)

        self.s_seq = []
        self.a_seq = []
        self.r_seq = []

    def get_action(self):
        if np.random.random() < self.eps:
            return choice(self.a_range)
        else:
            value_function = self.Q(self.s)
            max_idx = np.flatnonzero(np.isclose(value_function, value_function.max()))
            return self.a_range[choice(max_idx)]

    def feedback(self, action, observation, reward, done, training = True):
        self.t += 1
        observation = self.obs_ppc(observation)
        self.s_seq.append(np.array(self.s))
        self.r_seq.append(reward)
        self.a_seq.append(action)
        self.s = np.delete(self.s, -1, 0)
        self.s = np.insert(self.s, 0, observation, 0)
        if training:
            if len(self.r_seq) == self.horizon:
                z = sum(self.r_seq)
                self.add_to_bucket(self.s_seq[0], z, self.a_seq[0])
                del self.s_seq[0], self.a_seq[0], self.r_seq[0]
            if self.t <= self.eps_decay_duration:
                self.eps -= self.eps_decay

        self.R += reward
        if not self.t % 10000:
            self.R_per_10k_steps.append(self.R)
            self.R = 0
        self.R_in_last_20_episodes[-1] += reward
        if done:
            if len(self.R_in_last_20_episodes) == 20:
                self.average_score_per_episode.append(np.array(self.R_in_last_20_episodes).mean())
                self.R_in_last_20_episodes = [0]
            else:
                self.R_in_last_20_episodes.append(0)

    def Q(self, s):
        return Q(self.log_rho_s(s) + np.log(self.rho_z()), len(self.z_range), len(self.a_range), self.z_range_nb)

    def log_rho_s(self, s):
        return np.random.rand(len(self.z_range), len(self.a_range))

    def rho_z(self):
        return np.array([[self.SAD(self.z_bucket[a_idx], z, len(self.z_range)) for a_idx in range(len(self.a_range))]
                for z in self.z_range])

    def add_to_bucket(self, s, z, a):
        pass

    def obs_ppc(self, observation):
        return observation[33:193, :, 1] > 127

    def SAD(self, d, x, total_psblt):
        if x in d:
            return d[x] / (d['sum'] + self.beta)
        else:
            return 1 / (total_psblt - len(d) + 1) * self.beta / (d['sum'] + self.beta)


@nb.jit(nopython = True)
def Q(log_omg_td, Z, A, z_range):
    value_function = np.zeros(A)
    for z_idx in range(Z):
        for a_idx in range(A):
            denominator = np.exp(log_omg_td[:, a_idx] - log_omg_td[z_idx, a_idx]).sum()
            value_function[a_idx] += z_range[z_idx] * 1 / denominator

    return value_function