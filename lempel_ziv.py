from strategy import Strategy
import numpy as np



class Lempel_Ziv(Strategy):
    def __init__(self, init_obs):
        Strategy.__init__(self, init_obs)

        self.search_buffer_size = int(self.s.size / self.frames_per_step)
        # self.look_ahead_buffer_size = int(self.s.size / self.frames_per_step)
        self.look_ahead_buffer_size = 256
        self.triple_size = (np.ceil(np.log2(self.search_buffer_size)) + np.ceil(np.log2(self.look_ahead_buffer_size)) + 1).astype(np.int32)

        self.s_bucket = [[self.s.tobytes()[-self.search_buffer_size:]] * len(self.a_range)] * len(self.z_range)

    def log_rho_s(self, s):
        s = s.tobytes()
        prob_table = np.zeros((len(self.z_range), len(self.a_range)))
        for z_idx in range(len(self.z_range)):
            for a_idx in range(len(self.a_range)):
                triple_num = 0
                pos = 0
                search_buffer = self.s_bucket[z_idx][a_idx]
                while pos < self.s.size:
                    substring_len = 0
                    max_substring_len = min(self.look_ahead_buffer_size, self.s.size - pos)
                    while substring_len <= max_substring_len and s[pos:pos + substring_len] in search_buffer:
                        substring_len += 1
                    triple_num += 1
                    pos += substring_len
                prob_table[z_idx, a_idx] = -triple_num * self.triple_size * np.log(2)

        return prob_table

    def add_to_bucket(self, s, z, a):
        z_idx = self.z_range.index(z)
        a_idx = self.a_range.index(a)

        s = s.tobytes()
        self.s_bucket[z_idx][a_idx] = (self.s_bucket[z_idx][a_idx] + s)[self.s.size:]

        if z in self.z_bucket[a_idx]:
            self.z_bucket[a_idx][z] += 1
        else:
            self.z_bucket[a_idx][z] = 1
        self.z_bucket[a_idx]['sum'] += 1