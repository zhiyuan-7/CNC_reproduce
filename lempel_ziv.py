from strategy import Strategy
import numpy as np
import cv2



class Lempel_Ziv(Strategy):
    def __init__(self, init_obs):
        self.scale_factor = 0.25
        self.unit_len = 8
        self.bi_cvt = 2 ** np.arange(self.unit_len ** 2)

        Strategy.__init__(self, init_obs)

        self.byte_len = 16
        self.s_bucket = [[[Phrase_Node(0), 1, []] for _ in range(len(self.a_range))] for _ in range(len(self.z_range))]

        self.dict_full_1st_time = True

    def log_rho_s(self, s):
        prob_table = np.zeros((len(self.z_range), len(self.a_range)))
        s = s.reshape(-1)
        for z_idx in range(len(self.z_range)):
            for a_idx in range(len(self.a_range)):
                self.rho_s_pt(prob_table, self.s_bucket[z_idx][a_idx], z_idx, a_idx, s)

        return prob_table

    def rho_s_pt(self, prob_table, bucket, z_idx, a_idx, s):
        count = 0
        prefix_node = bucket[0]
        next_idx = 0
        while next_idx < len(s):
            tmp = prefix_node.child.get(s[next_idx])
            if tmp:
                prefix_node = tmp
                next_idx += 1
            elif prefix_node == bucket[0]:
                next_idx += 1
                count += 2
            else:
                count += 1
                prefix_node = bucket[0]
        prob_table[z_idx, a_idx] = -count * self.byte_len * np.log(2)

    def add_to_bucket(self, s, z, a):
        z_idx = self.z_range.index(z)
        a_idx = self.a_range.index(a)

        s = s.reshape(-1)
        bucket = self.s_bucket[z_idx][a_idx]
        prefix_node = bucket[0]
        next_idx = 0
        while next_idx < len(s):
            prefix_node.last_usage = self.t - self.horizon + 1
            tmp = prefix_node.child.get(s[next_idx])
            if tmp:
                prefix_node = tmp
                next_idx += 1
            else:
                new_node = Phrase_Node(self.t - self.horizon + 1, value = s[next_idx], parent = prefix_node)
                prefix_node.child[s[next_idx]] = new_node
                bucket[2].append(new_node)
                bucket[1] += 1
                if bucket[1] > 2 ** self.byte_len:
                    if self.dict_full_1st_time:
                        print(f'The dictionary of size {2 ** self.byte_len} is full for the first time.')
                        self.dict_full_1st_time = False
                    bucket[2].sort(key = lambda x: x.last_usage)
                    idx = next(filter(lambda i: len(bucket[2][i].child) == 0, range(len(bucket[2]))))
                    node_to_del = bucket[2][idx]
                    del bucket[2][idx]
                    del node_to_del.parent.child[node_to_del.value]
                    del node_to_del
                    bucket[1] -= 1
                prefix_node = bucket[0]

        if z in self.z_bucket[a_idx]:
            self.z_bucket[a_idx][z] += 1
        else:
            self.z_bucket[a_idx][z] = 1
        self.z_bucket[a_idx]['sum'] += 1

    def obs_ppc(self, observation):
        obs = cv2.resize(observation[33:193, :, 1], (0, 0), fx = self.scale_factor, fy = self.scale_factor, interpolation = cv2.INTER_NEAREST) > 127
        obs_revised = obs[:obs.shape[0] - obs.shape[0] % self.unit_len, :obs.shape[1] - obs.shape[1] % self.unit_len]
        H = int(obs_revised.shape[0] // self.unit_len)
        W = int(obs_revised.shape[1] // self.unit_len)
        obs_revised = np.array([np.split(obs_, W, 1) for obs_ in np.split(obs_revised, H, 0)])
        obs_revised = obs_revised.reshape(H, W, -1)
        obs_revised = (obs_revised * self.bi_cvt).sum(-1)


        return obs_revised


class Phrase_Node():
    def __init__(self, last_usage, value = None, parent = None):
        self.last_usage = last_usage
        self.value = value

        self.parent = parent
        self.child = {}