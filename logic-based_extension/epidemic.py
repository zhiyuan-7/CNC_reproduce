import numpy as np
from scipy.sparse.csr import csr_matrix

import sys
sys.path.append('.')
from config import MU_S, MU_E, MU_I, MU_R, ALPHA_S, ALPHA_E, ALPHA_I, ALPHA_R

print('MU_S: %g, MU_E: %g, MU_I: %g, MU_R: %g' % (MU_S, MU_E, MU_I, MU_R))
print('ALPHA_S: %g, ALPHA_E: %g, ALPHA_I: %g, ALPHA_R: %g' % (ALPHA_S, ALPHA_E, ALPHA_I, ALPHA_R))


# SEIRS: S --(\beta)--> E --(\sigma)--> I --(\gamma) --> R --(\rho)--> S
# State: s_i \in {S, E, I, R}
# Observation: o_i \in {P, N, U}

STATE_S = 1  # Susceptible
STATE_E = 2  # Exposed
STATE_I = 3  # Infectious
STATE_R = 4  # Recovered

OBS_P = 10   # Positive
OBS_N = -10  # Negative
OBS_U = 0    # Untested


class Epidemic:
    def __init__(self, param: np.ndarray, state: np.ndarray):
        self.n_nodes = state.shape[0]
        self.param = np.copy(param)
        self.state = np.copy(state)
    
    def evolve(self, adj_matrix: csr_matrix, SEIRS: bool=True, rng=None):
        self.state[:] = transition_states(self.param, self.state, adj_matrix, SEIRS, rng).reshape(-1)

    def observe(self, rng=None) -> np.ndarray:
        return _observe(self.state, rng)
 
    def is_absorbing_state(self) -> bool:
        """
        Check if the states of all nodes are 'S' (Susceptible) or 'R' (Recovered)
        """
        return np.logical_or(self.state == STATE_S, self.state == STATE_R).sum(dtype=np.int) == self.n_nodes


def _observe(state: np.ndarray, rng=None) -> np.ndarray:
    """
    Observe the node states in graph, i.e., the sensor model
    """
    rng = np.random.default_rng(rng)
    assert state.ndim == 1
    n_nodes = state.shape[0]

    obs = np.full(n_nodes, OBS_U, dtype=np.int8)
    rands_o = rng.random(size=n_nodes, dtype=np.float32)
    rands_p = rng.random(size=n_nodes, dtype=np.float32)

    for s, p, alpha in zip([STATE_S, STATE_E, STATE_I, STATE_R], [MU_S, MU_E, MU_I, MU_R], [ALPHA_S, ALPHA_E, ALPHA_I, ALPHA_R]):
        o_ind = np.logical_and(state == s, rands_o < alpha)
        p_ind = rands_p < p
        pos_ind = np.logical_and(o_ind, p_ind)
        neg_ind = np.logical_and(o_ind, np.logical_not(p_ind))
        obs[pos_ind] = OBS_P
        obs[neg_ind] = OBS_N

    return obs


def transition_states(param: np.ndarray, states: np.ndarray, adj_matrix: csr_matrix, SEIRS: bool=True, rng=None) -> np.ndarray:
    """
    Evolve the SEIRS model by one step, i.e., the transition model
    """
    rng = np.random.default_rng(rng)
    if states.ndim == 1:
        states = states.reshape(1, -1)
    n_nodes = states.shape[1]
    assert adj_matrix.shape == (n_nodes, n_nodes)
    assert param.ndim == 1
    if SEIRS:
        assert param.shape[0] == 4
        beta, sigma, gamma, rho = param
    else:
        assert param.shape[0] == 3
        beta, sigma, gamma = param
    
    new_states = np.copy(states)
    rands = rng.random(size=n_nodes, dtype=np.float32)
    i_ind = states == STATE_I
 
    # S -- (\beta) --> E
    ## PMF: n_choose_k * p^k * q^(n-k), here we want k >= 1, or equivalently, probability = 1. - (probability of k=0)
    ## p = beta, q = 1. - beta and pr = 1. - np.power(q, K)
    K = adj_matrix.dot(i_ind.T.astype(np.uint32)).T  # compute the (matrix) number of infectious neighbours
    q = 1. - beta
    pr = 1. - np.power(q, K, dtype=np.float32); del q, K
    s2e_ind = np.logical_and(rands < pr, states == STATE_S); del pr
    new_states[s2e_ind] = STATE_E; del s2e_ind
   
    # E -- (\sigma) --> I
    e2i_ind = np.logical_and(rands < sigma, states == STATE_E)
    new_states[e2i_ind] = STATE_I; del e2i_ind
    
    # I -- (\gamma) --> R
    i2r_ind = np.logical_and(rands < gamma, i_ind); del i_ind
    new_states[i2r_ind] = STATE_R; del i2r_ind

    # R -- (\xi) --> S
    if SEIRS:
        r2s_ind = np.logical_and(rands < rho, states == STATE_R)
        new_states[r2s_ind] = STATE_S
    
    return new_states

