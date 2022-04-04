import numpy as np
from scipy import stats
from typing import Iterable, List
from epidemic import STATE_S, STATE_E, STATE_I, STATE_R, OBS_P, OBS_N, OBS_U

import sys
sys.path.append('.')
from config import MU_S, MU_E, MU_I, MU_R, ALPHA_S, ALPHA_E, ALPHA_I, ALPHA_R


def guess_distros(obs: np.ndarray, prior: np.ndarray=np.array([0.997, 0.001, 0.001, 0.001])) -> np.ndarray:
    # assert np.equal(np.sum(prior), 1.0), np.sum(prior)
    
    C = 1e3
    n_nodes = obs.shape[0]
    mu_vec = np.array([MU_S, MU_E, MU_I, MU_R])
    alphas = np.array([ALPHA_S, ALPHA_E, ALPHA_I, ALPHA_R])

    # Posterior probabilities: Pr(state |observation, tested) where row -- observation, col -- state
    # - row order: STATE_S, STATE_E, STATE_I, STATE_R
    # - col order: OBS_P, OBS_N, OBS_U
    distros = np.vstack([C * prior * alphas * mu_vec,
                         C * prior * alphas * (1. - mu_vec),
                         C * prior * (1. - alphas)]).T
    distros /= distros.sum(axis=0, keepdims=True)  # normalisation

    # encoding observation type as indices: OBS_P -> 0, OBS_N -> 1, OBS_U -> 2
    cols = np.full(n_nodes, 2, dtype=np.int)
    cols[obs == OBS_P] = 0
    cols[obs == OBS_N] = 1

    return distros[:, cols]


def sample_states(obs: np.ndarray, size: int, prior: np.ndarray=np.array([0.997, 0.001, 0.001, 0.001]), rng=None) -> np.ndarray:
    # assert np.equal(np.sum(prior), 1.0), np.sum(prior)
    assert size > 0
    rng = np.random.default_rng(rng)

    STATE_SET = [STATE_S, STATE_E, STATE_I, STATE_R]
    ALPHA_SET = [ALPHA_S, ALPHA_E, ALPHA_I, ALPHA_R]
    MU_SET = [MU_S, MU_E, MU_I, MU_R]
    OBS_SET = [OBS_P, OBS_N, OBS_U]
    
    C = 1e3
    n_nodes = obs.shape[0]
    mu_vec = np.array(MU_SET)
    alphas = np.array(ALPHA_SET)

    # Posterior probabilities: Pr(state |observation, tested) where row -- observation, col -- state
    # - row order: OBS_P, OBS_N, OBS_U
    # - col order: STATE_S, STATE_E, STATE_I, STATE_R
    distros = np.vstack([C * prior * alphas * mu_vec,
                         C * prior * alphas * (1. - mu_vec),
                         C * prior * (1. - alphas)])  # 3 by len(STATE_SET)
    distros /= distros.sum(axis=1)[:, np.newaxis]  # normalisation
    # print(distros)

    distro_dict = {o: distros[ix, :] for ix, o in enumerate(OBS_SET)}
    states = np.empty((size, n_nodes), dtype=np.uint8)
    for j in range(n_nodes):
        # states[:, j] = np.random.choice(STATE_SET, p=distro_dict[obs[j]], size=size)
        states[:, j] = rng.choice(STATE_SET, p=distro_dict[obs[j]], size=size)
        # at least one E or I
        if np.sum(np.logical_or(states[:, j] == STATE_E, states[:, j] == STATE_I), dtype=np.int32) < 1:
            ix = rng.integers(0, size)
            states[ix, j] = rng.choice([STATE_E, STATE_I])
            # ix = np.random.randint(0, size)
            # states[ix, j] = np.random.choice([STATE_E, STATE_I])
    
    return states


def compute_population_properties(states: np.ndarray) -> tuple:
    if states.ndim == 1:
        states = states.reshape(1, -1)
    s_ind = states == STATE_S
    e_ind = states == STATE_E
    i_ind = states == STATE_I
    r_ind = states == STATE_R
    ret = np.vstack([ind.sum(axis=1, dtype=np.float32) / states.shape[1] for ind in [s_ind, e_ind, i_ind, r_ind]]).T
    return ret.squeeze().astype(np.float16)


def sample_params(n_samples: int, rng=None) -> np.ndarray:
    assert n_samples > 0
    rng = np.random.default_rng(rng)
    H1 = .8
    samples = rng.uniform(0, H1, size=(n_samples, 3))
    H2 = 0.1
    return np.hstack([samples, rng.uniform(0, H2, size=(n_samples, 1))])


def gaussian_jittering(params: np.ndarray, var_diag: np.ndarray, ranges: List[tuple], verbose=True, rng=None) -> np.ndarray:
    assert params.ndim == 2
    assert len(ranges) == params.shape[1], 'len(ranges)=%d, params.shape[1]=%d' % (len(ranges), params.shape[1])
    assert var_diag.shape[0] == params.shape[1]
    cov = np.diag(var_diag)
    if verbose:
        print('jittering, sigma: %s' % np.sqrt(var_diag))
    N = params.shape[0]
    eps = 1e-5
    lowers = np.array([r[0] + eps for r in ranges])
    uppers = np.array([r[1] - eps for r in ranges])
    # truncate = lambda x: np.maximum(lowers, np.minimum(uppers, x))
    truncate = lambda x: np.clip(x, lowers, uppers)
    return np.vstack([truncate([stats.multivariate_normal.rvs(mean=params[n, :], cov=cov, random_state=rng)]) for n in range(N)])


def softmax(log_weights: Iterable, temperature: float=1.0) -> np.ndarray:
    log_weights = np.asarray(log_weights) / temperature
    w = log_weights - np.max(log_weights)
    expw = np.exp(w)
    return expw / expw.sum()

