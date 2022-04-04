import sys
import os
import time
import random
import numpy as np
from typing import List, Tuple
from util import compute_population_properties, sample_states, sample_params, gaussian_jittering
from epidemic import Epidemic, OBS_P, OBS_N, OBS_U, STATE_S, STATE_E, STATE_I, STATE_R
from factored_particle_filter import Particle, particle_filter, conditional_particle_filter
from agent import *
import networkx as nx
from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator
import warnings
warnings.filterwarnings("ignore")

sys.path.append('.')
from config import *
np.set_printoptions(precision=5, suppress=True)


try:
    from config import SEED
    print('Set random seed=%d' % SEED)
    random.seed(SEED)
    RNG = np.random.default_rng(SEED)
except:
    print('No random seed set')
    RNG = np.random.default_rng()

try:
    from config import VERBOSE
except:
    VERBOSE = False


def estimate_param(particles: List[Particle]) -> List[float]:
    params = np.array([p.param for p in particles])
    ret = []
    for mu, std in zip(np.mean(params, axis=0), np.std(params, axis=0)):
        ret += [mu, std]
    return ret


def estimate_avgEval(true_state: np.ndarray, particles: List[Particle]) -> Tuple[float, float, float]:
    state_set = [STATE_S, STATE_E, STATE_I, STATE_R]
    acc_total = 0.
    l1e_total = 0.
    kld_total = 0.
    M, n_nodes = particles[0].states.shape
    truth = np.zeros((len(state_set), n_nodes), dtype=np.int8)
    for ix, ss in enumerate(state_set):
        truth[ix, true_state == ss] = 1
    max_ind = np.argmax(truth, axis=0)
    nz_ind = truth != 0
    for p in particles:
        preds = np.ones((len(state_set), n_nodes), dtype=np.float32)  # additive smoothing
        for ix, ss in enumerate(state_set):
            preds[ix, :] += np.sum(p.states == ss, dtype=np.float32, axis=0)
        preds /= (M + len(state_set))
        acc_total += np.mean(max_ind == np.argmax(preds, axis=0), dtype=np.float32)
        l1e_total += np.mean(np.abs(truth - preds).sum(axis=0))
        kld_total += np.mean(-np.log(preds[nz_ind]))  # simplified KLD when true distribution for a node is a one-hot vector

    N = len(particles)
    return acc_total / N, l1e_total / N, kld_total / N


t0 = time.time()
assert GRAPH_FILE.endswith('.npz')
adj_matrix = np.load(GRAPH_FILE, allow_pickle=True)['A'].item()
adj_matrix = adj_matrix.astype(np.uint16)
num_nodes = adj_matrix.shape[0]
num_edges = int(adj_matrix.sum() / 2)
print('#Nodes: {:,d}, #Edges: {:,d}'.format(num_nodes, num_edges))
print('Degrees (top 20):', np.sort(adj_matrix.sum(axis=0).A.reshape(-1))[-20:][::-1])
print('Parameter particles: %d' % N)
print('State     particles: %d' % M)
print('State       samples: %d' % N_SAMPLE)
if CONTROL:
    G = nx.Graph()
    G.add_edges_from(np.array(adj_matrix.nonzero()).T)
    bc_dict = nx.betweenness_centrality(G)
    bc_sort = list(range(num_nodes))
    bc_sort.sort(key = lambda x: bc_dict[x], reverse = True)
    bc_top15pct = bc_sort[:int(num_nodes * 0.15)]
    dg_sort = list(range(num_nodes))
    dg = adj_matrix.sum(0)
    dg_sort.sort(key = lambda x: dg[0, x], reverse = True)
    dg_top15pct = dg_sort[:int(num_nodes * 0.15)]
    del G, dg, dg_sort

adj_list = []
for j in range(num_nodes):
    adj_list.append(adj_matrix[j, :].indices)

adj_ix_flatten = np.concatenate(adj_list, dtype=np.int32)
split_ix = np.cumsum([0] + [len(a) for a in adj_list])[:-1]  # exclude the last element

del adj_list

while(True):
    root = random.randint(0, adj_matrix.shape[0]-1)
    print('Root: %d' % root)

    PARAM = np.array([BETA, SIGMA, GAMMA, RHO])
    STATE_SET = [STATE_S, STATE_E, STATE_I, STATE_R]
    state0 = np.full(adj_matrix.shape[0], STATE_S, dtype=np.int8)
    state0[root] = STATE_E
    epi = Epidemic(PARAM, state0)

    # for reproducibility
    print('Starting simulation ...')
    obs_all = []
    sts_all = []
    true_props = []
    T1 = -1
    t = 0

    adj_matrix_iso = adj_matrix
    adj_info_iso = (adj_ix_flatten, split_ix)
    if CONTROL:
        reward_all = []
        iso_rate_all = []
    # obs_all_ = []

    estimate_begin = False
    control_begin = False

    while t < MAX_ITER:
        epi.evolve(adj_matrix_iso, rng=RNG)
        obs = epi.observe(rng=RNG)
        obs_all.append(obs)
        sts_all.append(np.copy(epi.state))
        true_props.append(np.array([np.sum(epi.state == ss, dtype=np.float32) / num_nodes for ss in STATE_SET]))
        if epi.is_absorbing_state():
            print('No infections in population.')
            break
        if T1 == -1 and (obs == OBS_P).sum(dtype=np.int32) >= N_POS_INIT:
            T1 = t

            if ESTIMATE:
                print('Init particles ...')
                particles = [Particle(PARAM, sample_states(obs_all[T1], size=M, prior=np.array(PRIOR), rng=RNG)) for n in range(N)]
                acc_all = []
                l1e_all = []
                kld_all = []
                acc, l1e, kld = estimate_avgEval(sts_all[T1], particles)
                acc_all.append(acc)
                l1e_all.append(l1e)
                kld_all.append(kld)
                sts = sts_all[-1]
                tir_all = [np.mean(sts == STATE_E) + np.mean(sts == STATE_I)]
                est_states = particles[0].states
                eir_all = [np.mean(est_states == STATE_E) + np.mean(est_states == STATE_I)]
                t1 = time.time()
                estimate_begin = True

        if ESTIMATE and estimate_begin:
            sts = sts_all[-1]

            t2 = time.time()
            print(f'\nLAMBDA = {LAMBDA}, REP = {REP}, iteration {t}:')

            nobs = np.array([(obs == o).sum(dtype=np.int32) for o in [OBS_P, OBS_N]])
            nsts = np.array([(sts == s).sum(dtype=np.int32) for s in STATE_SET])

            print('STS: (S, E, I, R) = ({:4d}, {:4d}, {:4d}, {:4d})'.format(*nsts))
            print('OBS: (P, N) = ({:d}, {:d})'.format(*nobs))

            t2 = time.time()
            particles = particle_filter(particles, obs, adj_info=adj_info_iso, n_samples=N_SAMPLE, jittering=gaussian_jittering, verbose=VERBOSE, rng=RNG)
            print('Particle Filtering ... %.1f seconds' % (time.time() - t2))

            t3 = time.time()
            particles = conditional_particle_filter(particles, obs, adj_info_iso, rng=RNG)
            print('Conditional Particle Filtering ... %.1f seconds' % (time.time() - t3))

            if CALC_EVAL:
                acc, l1e, kld = estimate_avgEval(sts, particles)
                acc_all.append(acc)
                l1e_all.append(l1e)
                kld_all.append(kld)
                print('Accuracy: {:.1%}'.format(acc))

            tir = np.mean(sts == STATE_E) + np.mean(sts == STATE_I)
            print('True infection rate: {:.1%}'.format(tir))
            tir_all.append(tir)
            est_states = particles[0].states
            eir = np.mean(est_states == STATE_E) + np.mean(est_states == STATE_I)
            print('Estimated infection rate: {:.1%}'.format(eir))
            eir_all.append(eir)
            print('Positive rate: {:.1%}'.format(nobs[0] / nobs.sum()))

            if CONTROL:
                bc_top15pct_sts = est_states[:, bc_top15pct]
                bc_top15pct_eir = np.mean(bc_top15pct_sts == STATE_E) + np.mean(bc_top15pct_sts == STATE_I)
                dg_top15pct_sts = est_states[:, dg_top15pct]
                dg_top15pct_eir = np.mean(dg_top15pct_sts == STATE_E) + np.mean(dg_top15pct_sts == STATE_I)
                agt_obs = np.zeros(9)
                agt_obs[[min(int(eir / 0.15), 2), min(int(bc_top15pct_eir / 0.15), 2) + 3, min(int(dg_top15pct_eir / 0.15), 2) + 6]] = 1
                # agt_obs[min(int(tir / 0.15), 2)] = 1

                if not control_begin:
                    agt = Agent(agt_obs)
                    control_begin = True
                else:
                    reward = -LAMBDA * eir - iso_rate
                    # reward = -LAMBDA * (tir - tir_all[-2]) - iso_rate
                    # reward = -LAMBDA * tir - iso_rate
                    reward_all.append(reward)
                    print('Reward: %.1f' % reward)
                    agt.feedback(action, agt_obs, reward)

                    # if not agt.t % 1:
                    #     print(obs)
                    #     print(eir, bc_top15pct_eir, dg_top15pct_eir)
                    #     obs_all_.append(obs + 0)
                    #     for o in obs_all_:
                    #         print(o)

                if not agt.t % ACTION_DURATION:
                    action = agt.get_action()

                    iso_rate = ISO_RATE_SET[ACTION_SET.index(action)]
                    infection_prob = np.mean(est_states == STATE_E, axis = 0) + np.mean(est_states == STATE_I, axis = 0)

                    iso_nodes = bc_sort[:int(iso_rate * num_nodes)]

                    # # infection_truth = (sts == STATE_E) + (sts == STATE_I)
                    # # print(infection_prob.shape, infection_truth.shape)
                    # # print(infection_prob[:5], infection_truth[:5])
                    # # print(infection_prob[300:305], infection_truth[300:305])
                    # # print(infection_prob[700:705], infection_truth[700:705])
                    # # print(infection_prob[-5:], infection_truth[-5:])
                    # iso_nodes = list(range(num_nodes))
                    # iso_nodes.sort(key = lambda x: bc_dict[x] * infection_prob[x], reverse = True)
                    # iso_nodes = iso_nodes[:int(iso_rate * num_nodes)]

                    adj_matrix_iso = adj_matrix + 0
                    adj_matrix_iso[iso_nodes] = 0
                    adj_matrix_iso[:, iso_nodes] = 0
                    adj_list = []
                    for j in range(num_nodes):
                        adj_list.append(adj_matrix_iso[j, :].indices)
                    adj_ix_flatten = np.concatenate(adj_list, dtype=np.int32)
                    split_ix = np.cumsum([0] + [len(a) for a in adj_list])[:-1]  # exclude the last element
                    adj_info_iso = (adj_ix_flatten, split_ix)
                iso_rate_all.append(iso_rate)

            print('Time: %.1f seconds' % (time.time() - t1))
            sys.stdout.flush()

            t1 = time.time()

        t += 1
    print('\n%d observations generated.' % len(obs_all))
    print('T1 = %d' % T1)
    del epi, state0

    # if no valid outbreak
    if T1 == -1 or len(obs_all) < 80:
        print('No valid outbreak data collected.\n\nSimulating again...')
        continue
    del adj_matrix
    break

sys.stdout.flush()

if ESTIMATE:
    if CALC_EVAL:
        # np.save('./result/true_props_%d_%d.npy' % (M, MAX_ITER), np.vstack(true_props))
        # np.save('./result/acc_%d_%d.npy' % (M, MAX_ITER), np.array(acc_all))
        # np.save('./result/l1e_%d_%d.npy' % (M, MAX_ITER), np.array(l1e_all))
        # np.save('./result/kld_%d_%d.npy' % (M, MAX_ITER), np.array(kld_all))

        # np.save(os.path.join(SAVING_PATH, f'teir_lambda_{LAMBDA}.npy'), np.array([tir_all, eir_all]))
        np.save(os.path.join(SAVING_PATH, f'teir.npy'), np.array([tir_all, eir_all]))
        plt.plot(tir_all, label='True infection rate')
        plt.plot(eir_all, label='Estimated infection rate')
        plt.legend(loc='best')
        plt.xlabel('Time')
        plt.ylabel('Infection rate')
        plt.xlim(0, len(tir_all) - 1)
        plt.ylim(0, 1)
        ax = plt.gca()
        # ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.set_yticklabels([f'{y:>.0%}' for y in ax.get_yticks()])
        # plt.savefig(os.path.join(SAVING_PATH, f'infection_rate_lambda_{LAMBDA}.pdf'))
        plt.savefig(os.path.join(SAVING_PATH, f'infection_rate.pdf'))
        plt.close()

        if CONTROL:
            np.save(os.path.join(SAVING_PATH, f'reward_lambda_{LAMBDA}.npy'), np.array(reward_all))

            plt.plot(reward_all)
            plt.xlabel('Time')
            plt.ylabel('Reward')
            plt.xlim(0, len(reward_all) - 1)
            plt.savefig(os.path.join(SAVING_PATH, f'reward_lambda_{LAMBDA}.pdf'))
            plt.close()

            np.save(os.path.join(SAVING_PATH, f'iso_rate_lambda_{LAMBDA}.npy'), np.array(iso_rate_all))

            plt.plot(iso_rate_all)
            plt.xlabel('Time')
            plt.ylabel('Isolation rate')
            plt.xlim(0, len(iso_rate_all) - 1)
            plt.ylim(0, 1)
            ax = plt.gca()
            # ax.xaxis.set_major_locator(MaxNLocator(integer=True))
            ax.set_yticklabels([f'{y:>.0%}' for y in ax.get_yticks()])
            plt.savefig(os.path.join(SAVING_PATH, f'iso_rate_lambda_{LAMBDA}.pdf'))
            plt.close()

            import pickle
            f = open(os.path.join(SAVING_PATH, f'z_bucket_lambda_{LAMBDA}'), 'wb')
            pickle.dump(agt.z_bucket, f)
            f.close()
            f = open(os.path.join(SAVING_PATH, f's_bucket_lambda_{LAMBDA}'), 'wb')
            pickle.dump(agt.s_bucket, f)
            f.close()


    duration = time.time() - t0
    print('Total Time: %d minutes %d seconds' % (duration // 60, duration % 60))