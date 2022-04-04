import numpy as np
from typing import List, Callable, Tuple
from util import softmax
from epidemic import STATE_S, STATE_E, STATE_I, STATE_R, OBS_P, OBS_N, OBS_U

import sys
sys.path.append('.')
from config import MU_S, MU_E, MU_I, MU_R, ALPHA_S, ALPHA_E, ALPHA_I, ALPHA_R


class Particle:
    def __init__(self, param: np.ndarray, states: np.ndarray):
        self.param = np.copy(param)
        self.states = np.copy(states)


def particle_filter(particles: List[Particle], obs: np.ndarray, adj_info: Tuple[np.ndarray,np.ndarray], n_samples: int, jittering: Callable=None, args: tuple=None, verbose: bool=False, rng=None) -> List[Particle]:
    """
    - particles: particles at time t-1
    - obs: observation at time t
    return particles with updated parameters
    """
    assert n_samples > 0
    N = len(particles)
    rng = np.random.default_rng(rng)
    
    indices = rng.integers(0, N, size=N)
    inplace_resample(particles, resampled_ix=indices)
    del indices
    
    params = np.vstack([p.param for p in particles])
    # if jittering is not None:
    #     assert args is not None
    #     params[:] = jittering(params, *args, rng=rng)
    for n in range(N):
        particles[n].param[:] = params[n, :]

    # if not verbose:
    #     del params

    log_weights = np.zeros(N)
    for n in range(N):
        states_bar = transition_states_factored(particles[n].param, particles[n].states, adj_info, n_samples, rng)
        likelihood_mat = compute_states_weights(states_bar, obs)
        log_weights[n] = np.log(likelihood_mat.mean(axis=0)).sum()
        del states_bar, likelihood_mat
        
        # (Crisan and Míguez, 2018) will update states here
        # likelihood_mat /= likelihood_mat.sum(axis=0, keepdims=True)
        # row_ix = np.vstack([np.random.choice(M, p=likelihood_mat[:,j], size=M, replace=True) for j in range(n_nodes)]).T
        # col_ix = np.arange(n_nodes, dtype=np.int32)
        # particles[n].states[:] = states_bar[row_ix, col_ix]

    weights = softmax(log_weights)

    if verbose:
        print('Before resampling')
        cnt = 0
        for ix in np.argsort(-weights):
            cnt += 1
            if cnt > 20:
                break
            print('__%4f: %s, %f' % (weights[ix], params[ix], log_weights[ix]))

    inplace_resample(particles, normalised_weights=weights, rng=rng)
    return particles


def conditional_particle_filter(particles: List[Particle], obs: np.ndarray, adj_info: Tuple[np.ndarray,np.ndarray], rng=None) -> List[Particle]:
    """
    - particles: particles at time t-1
    - obs: observation at time t
    return updated particles
    """
    rng = np.random.default_rng(rng)
    N = len(particles)
    M, n_nodes = particles[0].states.shape
    col_ix = np.arange(n_nodes, dtype=np.int32)

    for n in range(N):
        # compute likelihood
        states_bar = transition_states_factored(particles[n].param, particles[n].states, adj_info, M, rng)
        likelihood_mat = compute_states_weights(states_bar, obs)
        
        # update states
        likelihood_mat /= likelihood_mat.sum(axis=0, keepdims=True)
        row_ix = random_choice(likelihood_mat, col_ix, rng=rng)

        ## row_ix = np.asarray(np.vstack([np.random.choice(M, p=likelihood_mat[:,j], size=M, replace=True) for j in range(n_nodes)])).T
        # row_ix = np.empty((M, n_nodes), dtype=np.int32)
        # for j in range(n_nodes):
        #     row_ix[:, j] = rng.choice(M, p=likelihood_mat[:, j], size=M, replace=True)
        del likelihood_mat

        particles[n].states[:] = states_bar[row_ix, col_ix]
        del row_ix, states_bar

        # smoothing
        # rand_ix = np.vstack([np.random.choice(M, size=len(state_set), replace=False) for _ in range(n_nodes)]).T
        # for ix, ss in enumerate(state_set):
        #     rix = rand_ix[ix, :]
        #     particles[n].states[rix, col_ix] = ss
    return particles


def inplace_resample(particles: List[Particle], normalised_weights: np.ndarray=None, resampled_ix: np.ndarray=None, rng=None):
    """
    Perform the resampling step inplace for memeory efficiency
    """
    N = len(particles)
    if resampled_ix is None:
        assert normalised_weights is not None
        assert np.isclose(normalised_weights.sum(), 1.)
        # assert np.all(np.logical_not(normalised_weights < 0))
        rng = np.random.default_rng(rng)
        resampled_ix = rng.choice(N, p=normalised_weights, size=N, replace=True)
    else:
        assert len(resampled_ix) == N
        if normalised_weights is not None:
            print('inplace_resample(): ignore `normalised_weights` because `resampled_ix` is given')
  
    counts = np.zeros(N, dtype=np.int32)
    for ix in resampled_ix:
        counts[ix] += 1

    items_ix = np.nonzero(counts > 1)[0]  # indices of particles with more than one occurrence 
    items_cnt = counts[items_ix]
    items_cnt -= 1  # count only the extra copies needed

    places_ix = np.nonzero(counts == 0)[0]  # indices of particles that can be overwritten
    assert len(places_ix) == items_cnt.sum()

    j = 0
    for i in range(len(items_ix)):
        from_ix = items_ix[i]
        for _ in range(items_cnt[i]):
            to_ix = places_ix[j]
            particles[to_ix].param[:] = particles[from_ix].param
            particles[to_ix].states[:] = particles[from_ix].states
            j += 1


def random_choice(weights: np.ndarray, col_ix: np.ndarray, rng=None) -> np.ndarray:
    """
    vectorised random.choice() for sampling from multiple categorical distributions
    https://en.wikipedia.org/wiki/Categorical_distribution#Sampling
    """
    rng = np.random.default_rng(rng)
    M, L = weights.shape
    assert len(col_ix) == L

    W = weights.T
    t = col_ix.reshape(-1, 1) * 2

    bins = np.cumsum(W, axis=1)  # L by M
    bins += t
    bins[:, -1] = (col_ix + 1) * 2.  # make the right side of the last bin far bigger to avoid an edge case: rng.random() returns 1.0 when using dtype=float32
    
    rands = rng.random(size=(L, M), dtype=np.float32)
    rands += t

    ix = np.searchsorted(bins.reshape(-1), rands.reshape(-1), side='left')  # see np.digitize() and np.searchsorted()
    ix = ix.reshape(L, M) - col_ix.reshape(-1, 1) * M
    # ix = ix.reshape(L, M) % M  # mod is more expensive
    del bins, rands, t

    return ix.T


def transition_states_factored(param: np.ndarray, states: np.ndarray, adj_info: Tuple[np.ndarray,np.ndarray], n_samples: int, rng=None) -> np.ndarray:
    """
    Evolve the SIS transition model for all nodes using the factored approach
    """
    assert n_samples > 0
    rng = np.random.default_rng(rng)
    M, n_nodes = states.shape
    assert param.shape[0] == 4
    beta, sigma, gamma, rho = param
    assert len(adj_info) == 2
    adj_ix_flatten, split_ix = adj_info
        
    # compute the (matrix) number of infectious neighbours
    ind_mat = states == STATE_I
    col_ix = adj_ix_flatten
    row_ix = rng.integers(0, M, size=(n_samples, len(col_ix)), dtype=np.int32)
    D = np.add.reduceat(ind_mat[row_ix, col_ix], split_ix, axis=1, dtype=np.int32)
    del row_ix, ind_mat
 
    # S -- (\beta) --> E: part 1
    q = 1. - beta
    pr_s2e = 1. - np.power(q, D, dtype=np.float32); del D
    rands = rng.random(size=(n_samples, n_nodes), dtype=np.float32)
    s2e_ind_bar = rands < pr_s2e; del pr_s2e
    
    # new state particles
    col_ix = np.arange(n_nodes, dtype=np.int32)
    row_ix = rng.integers(0, M, size=(n_samples, n_nodes), dtype=np.int32)
    new_states = states[row_ix, col_ix]
    del row_ix, col_ix

    # compute indices as new_states will be overwritten
    s_ind = new_states == STATE_S
    e_ind = new_states == STATE_E
    i_ind = new_states == STATE_I
    r_ind = new_states == STATE_R
    
    # S -- (\beta) --> E: part 2
    s2e_ind = np.logical_and(s2e_ind_bar, s_ind); del s2e_ind_bar, s_ind
    new_states[s2e_ind] = STATE_E; del s2e_ind
    
    # E -- (\sigma) --> I
    e2i_ind = np.logical_and(rands < sigma, e_ind); del e_ind
    new_states[e2i_ind] = STATE_I; del e2i_ind
   
    # I -- (\gamma) --> R
    i2r_ind = np.logical_and(rands < gamma, i_ind); del i_ind
    new_states[i2r_ind] = STATE_R; del i2r_ind

    # R -- (\xi) --> S
    r2s_ind = np.logical_and(rands < rho, r_ind); del r_ind, rands
    new_states[r2s_ind] = STATE_S; del r2s_ind

    return new_states


def compute_states_weights(states: np.ndarray, obs: np.ndarray) -> np.ndarray:
    """
    Compute likelihoods of given observations
    """
    STATE_SET = [STATE_S, STATE_E, STATE_I, STATE_R]
    ALPHA_SET = [ALPHA_S, ALPHA_E, ALPHA_I, ALPHA_R]
    MU_SET = [MU_S, MU_E, MU_I, MU_R]
    OBS_SET = [OBS_P, OBS_N, OBS_U]
    alpha_vec = np.array(ALPHA_SET)
    mu_vec = np.array(MU_SET)
    n_nodes = obs.shape[0]
    
    # row order: STATE_SET
    # col order: OBS_SET
    obs_kernel = np.empty((len(STATE_SET), 3), dtype=np.float32)
    obs_kernel[:, 0] = alpha_vec * mu_vec
    obs_kernel[:, 1] = alpha_vec * (1. - mu_vec)
    obs_kernel[:, 2] = 1. - alpha_vec

    row_ix = np.empty(states.shape, dtype=np.int32)
    for ix, ss in enumerate(STATE_SET):
        row_ix[states == ss] = ix

    col_ix = np.empty(n_nodes, dtype=np.int32)
    for ix, oo in enumerate(OBS_SET):
        col_ix[obs == oo] = ix

    return obs_kernel[row_ix, col_ix]
 
