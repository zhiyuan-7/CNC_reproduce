import numpy as np
from random import choice



MAINTAIN = 100
ISO_10_PERCENT = 200
ISO_20_PERCENT = 300
ISO_30_PERCENT = 400
ISO_40_PERCENT = 500
ISO_50_PERCENT = 600
ISO_ALL = 700
# ACTION_SET = [MAINTAIN, ISO_10_PERCENT, ISO_20_PERCENT, ISO_30_PERCENT, ISO_40_PERCENT, ISO_50_PERCENT, ISO_ALL]
ACTION_SET = [MAINTAIN, ISO_20_PERCENT, ISO_50_PERCENT, ISO_ALL]
ISO_RATE_SET = [0, 0.2, 0.5, 1]
# ISO_RATE_SET = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 1]

class Agent():
    def __init__(self, init_obs):
        self.horizon = 20
        self.beta = 1
        self.z_range = range(-100, 20, 1)
        self.Z = len(self.z_range)
        self.a_range = ACTION_SET
        self.A = len(self.a_range)
        self.t = 0
        self.s = init_obs
        self.S = len(self.s)
        self.R = 0
        self.R_per_20_steps = []

        self.s_seq = []
        self.a_seq = []
        self.r_seq = []

        self.z_bucket = [{'count': 0} for _ in self.a_range]
        self.s_bucket = {'bucket': np.ones((self.Z, self.A, self.S)) / 3, 'count': np.ones((self.Z, self.A))}

        self.eps = 1
        self.eps_ultimate = 0.02
        self.eps_decay_duration = 300
        self.eps_decay = (self.eps - self.eps_ultimate) / self.eps_decay_duration

    def get_action(self):
        if np.random.random() < self.eps:
            return choice(self.a_range)
        else:
            value_function = self.Q(self.s)
            max_idx = np.flatnonzero(np.isclose(value_function, value_function.max()))
            a_idx = choice(max_idx)
            print(f'Action index: {a_idx}')
            return self.a_range[a_idx]

    def feedback(self, action, observation, reward):
        self.t += 1
        self.s_seq.append(self.s)
        self.r_seq.append(reward)
        self.a_seq.append(action)
        self.s = observation
        if len(self.r_seq) == self.horizon:
            z = sum(self.r_seq)
            print(z, f'ISOLATE_{ISO_RATE_SET[ACTION_SET.index(action)]}')
            self.add_to_bucket(self.s_seq[0], z, self.a_seq[0])
            del self.s_seq[0], self.a_seq[0], self.r_seq[0]
        if self.t <= self.eps_decay_duration:
            self.eps -= self.eps_decay

        self.R += reward
        if not self.t % 20:
            self.R_per_20_steps.append(self.R)
            self.R = 0

    def Q(self, s):
        log_omg_td = self.log_rho_s(s) + np.log(self.rho_z())
        denominators = np.exp(log_omg_td.reshape(1, self.Z, self.A) - log_omg_td.reshape(self.Z, 1, self.A)).sum(1)

        return (np.array(self.z_range).reshape(-1, 1) / denominators).sum(0)

    def log_rho_s(self, s):
        bucket = self.s_bucket['bucket']
        s_rsp = s.reshape(1, 1, -1)

        return (s_rsp * np.log(bucket) + (1 - s_rsp) * np.log(1 - bucket)).sum(-1)

    def rho_z(self):
        return np.array([[self.SAD(self.z_bucket[a_idx], z, self.Z) for a_idx in range(self.A)]
                for z in self.z_range])

    def add_to_bucket(self, s, z, a):
        z_idx = (np.array(self.z_range) - z).__abs__().argmin()
        z = self.z_range[z_idx]
        a_idx = self.a_range.index(a)

        if z in self.z_bucket[a_idx]:
            self.z_bucket[a_idx][z] += 1
        else:
            self.z_bucket[a_idx][z] = 1
        self.z_bucket[a_idx]['count'] += 1

        bkt, cnt = self.s_bucket['bucket'][z_idx, a_idx], self.s_bucket['count'][z_idx, a_idx]
        self.s_bucket['bucket'][z_idx, a_idx] = (bkt * cnt + s) / (cnt + 1)
        self.s_bucket['count'][z_idx, a_idx] = cnt + 1

    def SAD(self, d, x, total_psblt):
        if x in d:
            return d[x] / (d['count'] + self.beta)
        else:
            return 1 / (total_psblt - len(d) + 1) * self.beta / (d['count'] + self.beta)