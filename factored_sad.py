from strategy import Strategy
import numpy as np
import numba as nb



class Factored_SAD(Strategy):
    def __init__(self, init_obs):
        Strategy.__init__(self, init_obs)

        self.H = self.W = 16
        self.anchor_num = 22
        self.anchor = self.division(np.random.randint(0, 2, self.s.shape + (self.anchor_num,)))
        self.s_bucket = nb.typed.List()
        for _ in range(self.H * self.W * len(self.z_range) * len(self.a_range)):
            self.s_bucket.append(nb.typed.Dict.empty(key_type = nb.types.Tuple((nb.types.int64,) * 5), value_type = nb.types.int64))
        self.s_bucket_sum = np.zeros((self.H, self.W, len(self.z_range), len(self.a_range)), dtype = np.int64)
        self.s_patch_psblt = np.array(self.anchor.shape[2:5]).prod() * 128

    def save(self):
        Strategy.save(self)
        self.s_bucket = [dict(d) for d in self.s_bucket]

    def load(self):
        Strategy.load(self)

        s_bucket = nb.typed.List()
        for d in self.s_bucket:
            d0 = nb.typed.Dict.empty(key_type = nb.types.Tuple((nb.types.int64,) * 5), value_type = nb.types.int64)
            for key, value in d.items():
                d0[key] = value
            s_bucket.append(d0)
        self.s_bucket = s_bucket

    def log_rho_s(self, s):
        s_divided = self.division(s)
        coord = np.abs(s_divided.reshape(s_divided.shape + (1,)) - self.anchor).sum(2).sum(2).sum(2)
        prob_table = rho_s_pt(self.s_bucket, coord, self.H, self.W, len(self.z_range), len(self.a_range), self.beta, self.s_patch_psblt, self.s_bucket_sum)

        return prob_table

    def add_to_bucket(self, s, z, a):
        z_idx = self.z_range.index(z)
        a_idx = self.a_range.index(a)

        s_divided = self.division(s)
        coord = np.abs(s_divided.reshape(s_divided.shape + (1,)) - self.anchor).sum(2).sum(2).sum(2)
        add_to_s_bucket(self.s_bucket, coord, z_idx, a_idx, self.H, self.W, len(self.z_range), len(self.a_range), self.s_bucket_sum)

        if z in self.z_bucket[a_idx]:
            self.z_bucket[a_idx][z] += 1
        else:
            self.z_bucket[a_idx][z] = 1
        self.z_bucket[a_idx]['sum'] += 1

    def division(self, s):
        s_revised = s[:, :s.shape[1] - s.shape[1] % self.H, :s.shape[2] - s.shape[2] % self.W]
        return np.array([np.split(s_, self.W, 2) for s_ in np.split(s_revised, self.H, 1)])


@nb.jit(nopython = True)
def rho_s_pt(s_bucket, coord, H, W, Z, A, beta, s_patch_psblt, s_bucket_sum):
    prob_table = np.zeros((Z, A))
    for z_idx in range(Z):
        for a_idx in range(A):
            prob_table[z_idx, a_idx] = SAD_for_s(s_bucket, coord, z_idx, a_idx, H, W, Z, A, beta, s_patch_psblt, s_bucket_sum)

    return prob_table


@nb.jit(nopython = True)
def add_to_s_bucket(s_bucket, coord, z_idx, a_idx, H, W, Z, A, s_bucket_sum):
    for h in range(H):
        for w in range(W):
            bkt = s_bucket[h * W * Z * A + w * Z * A + z_idx * A + a_idx]
            k = (coord[h, w, 0], coord[h, w, 1], coord[h, w, 2], coord[h, w, 3], coord[h, w, 4])
            if k in bkt:
                bkt[k] += 1
            else:
                bkt[k] = 1
            s_bucket_sum[h, w, z_idx, a_idx] += 1


@nb.jit(nopython = True)
def SAD_for_s(s_bucket, coord, z_idx, a_idx, H, W, Z, A, beta, s_patch_psblt, s_bucket_sum):
    sum = 0
    for h in range(H):
        for w in range(W):
            d = s_bucket[h * W * Z * A + w * Z * A + z_idx * A + a_idx]
            k = (coord[h, w, 0], coord[h, w, 1], coord[h, w, 2], coord[h, w, 3], coord[h, w, 4])
            if k in d:
                p = d[k] / (s_bucket_sum[h, w, z_idx, a_idx] + beta)
            else:
                p = 1 / (s_patch_psblt - len(d)) * beta / (s_bucket_sum[h, w, z_idx, a_idx] + beta)
            sum += np.log(p)

    return sum