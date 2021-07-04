from strategy import Strategy
import numpy as np
import torch



class Logistic_Regression(Strategy):
    def __init__(self, init_obs):
        Strategy.__init__(self, init_obs)

        self.model = NeuralNetwork(self.z_range, self.a_range)
        self.optimizer = torch.optim.Adagrad(self.model.parameters())

        self.coord = np.ones_like(self.s)
        self.coord0 = (np.array(range(1, self.frames_per_step + 1)).reshape(-1, 1, 1) * self.coord).reshape(-1, 1)
        self.coord1 = (np.array(range(2, self.s.shape[1] + 2)).reshape(1, -1, 1) * self.coord).reshape(-1, 1)
        self.coord2 = (np.array(range(2, self.s.shape[2] + 2)).reshape(1, 1, -1) * self.coord).reshape(-1, 1)
        self.coord0 = (self.coord0 + np.array([[-1] * 25 + [0] * 12])).reshape(-1).astype(np.int32)
        self.coord1 = (self.coord1 + np.array([[-2] * 5 + [-1] * 5 + [0] * 5 + [1] * 5 + [2] * 5 + [-2] * 5 + [-1] * 5 + [0] * 2])).reshape(-1).astype(np.int32)
        self.coord2 = (self.coord2 + np.array([[-2, -1, 0, 1, 2] * 7 + [-2, -1]])).reshape(-1).astype(np.int32)

    def log_rho_s(self, s):
        context = self.context_extractor(s)
        pred = self.model(context).detach().numpy()
        s = s.reshape(-1)
        prob_table = (s * np.log(pred) + (1 - s) * np.log(1 - pred)).sum(-1)

        return prob_table

    def add_to_bucket(self, s, z, a):
        context = self.context_extractor(s)
        pred = self.model(context, z, a)
        s = s.reshape(-1)
        s = torch.from_numpy(s)
        loss = -(s * torch.log(pred) + ~s * torch.log(1 - pred)).sum()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        a_idx = self.a_range.index(a)
        if z in self.z_bucket[a_idx]:
            self.z_bucket[a_idx][z] += 1
        else:
            self.z_bucket[a_idx][z] = 1
        self.z_bucket[a_idx]['sum'] += 1

    def context_extractor(self, s):
        s_padded = np.zeros((self.frames_per_step + 1, s.shape[1] + 4, s.shape[2] + 4), dtype = np.int32)
        s_padded[1:, 2:-2, 2:-2] = s

        return s_padded[self.coord0, self.coord1, self.coord2].reshape(-1, 37)


class NeuralNetwork(torch.nn.Module):
    def __init__(self, z_range, a_range):
        super(NeuralNetwork, self).__init__()
        self.z_range = z_range
        self.a_range = a_range

        self.w = torch.nn.Parameter(torch.Tensor(len(z_range), len(a_range), 1, 37))
        torch.nn.init.xavier_uniform_(self.w)
        self.b = torch.nn.Parameter(torch.zeros(len(z_range), len(a_range), 1))
        self.sigmoid = torch.nn.Sigmoid()


    def forward(self, x, z = None, a = None):
        x = torch.from_numpy(x)

        if z is None and a is None:
            logits = (self.w * x).sum(-1) + self.b
        else:
            z_idx = self.z_range.index(z)
            a_idx = self.a_range.index(a)
            logits = (self.w[z_idx, a_idx] * x).sum(-1) + self.b[z_idx, a_idx]

        pred = self.sigmoid(logits)

        return pred