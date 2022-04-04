import numpy as np
import pickle
import os
from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator
import warnings
warnings.filterwarnings("ignore")

from agent import Agent


MAX_ITER = 800
DIR = './result/bc_rank'
LAMBDA_SET = [0.1, 1, 3, 10]
REP_SET = range(1, 11)
ACTION_NAME_SET = ['No isolation', '20% isolation', '50% isolation', 'Full isolation']
PREDICATE_NAME_SET = ['0-15%', '15-30%', '30+%'] * 3
TEST_STATE_SET = np.array([[1, 0, 0] * 3, [0, 1, 0] * 3, [0, 0, 1] * 3])
TEST_STATE_NAME_SET = ['Mild', 'Moderate', 'Severe']
# C1 = np.array([176, 224, 230]) / 255
# C2 = np.array([65, 105, 225]) / 255
C1 = np.array([153, 204, 255]) / 255
C2 = np.array([0, 0, 128]) / 255
N = len(LAMBDA_SET)
COLOR_SET = [n / (N - 1) * C2 + (1 - n / (N - 1)) * C1 for n in range(N)]


tir_avg_all = []
iso_rate_avg_all = []
for LAMBDA in LAMBDA_SET:
    tir = []
    iso_rate = []
    for REP in REP_SET:
        tir.append(np.load(os.path.join(DIR, f'lambda_{LAMBDA}', f'rep_{REP}', f'teir_lambda_{LAMBDA}.npy'))[0])
        iso_rate.append(np.load(os.path.join(DIR, f'lambda_{LAMBDA}', f'rep_{REP}', f'iso_rate_lambda_{LAMBDA}.npy')))
    tir = np.array([np.concatenate([r, np.zeros(MAX_ITER)])[:MAX_ITER + 1] for r in tir])
    iso_rate = np.array([np.concatenate([r, np.zeros(MAX_ITER)])[:MAX_ITER + 1] for r in iso_rate])
    tir_avg = tir.mean(0)
    iso_rate_avg = iso_rate.mean(0)
    tir_avg_all.append(tir_avg)
    iso_rate_avg_all.append(iso_rate_avg)
    tir_std = tir.std(0)
    iso_rate_std = iso_rate.std(0)

    plt.plot(tir_avg)
    plt.xlabel('Time')
    plt.ylabel('Infection rate')
    plt.xlim(0, len(tir_avg) - 1)
    plt.ylim(0, 0.5)
    ax = plt.gca()
    ax.fill_between(range(len(tir_avg)), tir_avg - tir_std, tir_avg + tir_std, alpha = 0.2)
    # ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_yticklabels([f'{y:>.0%}' for y in ax.get_yticks()])
    plt.savefig(os.path.join(DIR, 'plot', f'infection_rate_agg_lambda_{LAMBDA}.pdf'))
    plt.close()

    plt.plot(iso_rate_avg)
    plt.xlabel('Time')
    plt.ylabel('Isolation rate')
    plt.xlim(0, len(iso_rate_avg) - 1)
    plt.ylim(0, 1)
    ax = plt.gca()
    ax.fill_between(range(len(iso_rate_avg)), iso_rate_avg - iso_rate_std, iso_rate_avg + iso_rate_std, alpha = 0.2)
    # ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_yticklabels([f'{y:>.0%}' for y in ax.get_yticks()])
    plt.savefig(os.path.join(DIR, 'plot', f'iso_rate_agg_lambda_{LAMBDA}.pdf'))
    plt.close()

for idx, LAMBDA in enumerate(LAMBDA_SET):
    plt.plot(tir_avg_all[idx], label = f'$\lambda={LAMBDA}$', c = COLOR_SET[idx])
plt.legend(loc = 'best')
plt.xlabel('Time')
plt.ylabel('Infection rate')
plt.xlim(0, MAX_ITER)
plt.ylim(0, 0.5)
ax = plt.gca()
# ax.xaxis.set_major_locator(MaxNLocator(integer=True))
ax.set_yticklabels([f'{y:>.0%}' for y in ax.get_yticks()])
plt.savefig(os.path.join(DIR, 'plot', f'infection_rate_agg_all.pdf'))
plt.close()

for idx, LAMBDA in enumerate(LAMBDA_SET):
    plt.plot(iso_rate_avg_all[idx], label = f'$\lambda={LAMBDA}$', c = COLOR_SET[idx])
plt.legend(loc = 'best')
plt.xlabel('Time')
plt.ylabel('Isolation rate')
plt.xlim(0, MAX_ITER)
plt.ylim(0, 1)
ax = plt.gca()
# ax.xaxis.set_major_locator(MaxNLocator(integer=True))
ax.set_yticklabels([f'{y:>.0%}' for y in ax.get_yticks()])
plt.savefig(os.path.join(DIR, 'plot', f'iso_rate_agg_all.pdf'))
plt.close()

# smoothing
WINDOW = 20
LEN = len(tir_avg)
infection_rate_avg_all_smooth = [np.convolve(r, np.ones(WINDOW), 'full')[:LEN] / np.concatenate([range(1, WINDOW + 1), [WINDOW] * (LEN - WINDOW)]) for r in tir_avg_all]
for idx, LAMBDA in enumerate(LAMBDA_SET):
    plt.plot(infection_rate_avg_all_smooth[idx], label = f'$\lambda={LAMBDA}$', c = COLOR_SET[idx])
plt.legend(loc = 'best')
plt.xlabel('Time')
plt.ylabel('Infection rate after smoothing')
plt.xlim(0, MAX_ITER)
plt.ylim(0, 0.5)
ax = plt.gca()
# ax.xaxis.set_major_locator(MaxNLocator(integer=True))
ax.set_yticklabels([f'{y:>.0%}' for y in ax.get_yticks()])
plt.savefig(os.path.join(DIR, 'plot', f'infection_rate_agg_all_smooth.pdf'))
plt.close()

WINDOW = 50
LEN = len(iso_rate_avg)
iso_rate_avg_all_smooth = [np.convolve(r, np.ones(WINDOW), 'full')[:LEN] / np.concatenate([range(1, WINDOW + 1), [WINDOW] * (LEN - WINDOW)]) for r in iso_rate_avg_all]
for idx, LAMBDA in enumerate(LAMBDA_SET):
    plt.plot(iso_rate_avg_all_smooth[idx], label = f'$\lambda={LAMBDA}$', c = COLOR_SET[idx])
plt.legend(loc = 'best')
plt.xlabel('Time')
plt.ylabel('Isolation rate after smoothing')
plt.xlim(0, MAX_ITER)
plt.ylim(0, 1)
ax = plt.gca()
# ax.xaxis.set_major_locator(MaxNLocator(integer=True))
ax.set_yticklabels([f'{y:>.0%}' for y in ax.get_yticks()])
plt.savefig(os.path.join(DIR, 'plot', f'iso_rate_agg_all_smooth.pdf'))
plt.close()


rt_avg_all = []
for LAMBDA in LAMBDA_SET:
    rt = []
    Q = []
    for REP in REP_SET:
        with open(os.path.join(DIR, f'lambda_{LAMBDA}', f'rep_{REP}', f'z_bucket_lambda_{LAMBDA}'), 'rb') as f:
            z_bucket = pickle.load(f)
        rt.append([sum(z * z_count for z, z_count in bucket.items() if type(z) == int) / bucket['count'] for bucket in z_bucket])
        with open(os.path.join(DIR, f'lambda_{LAMBDA}', f'rep_{REP}', f's_bucket_lambda_{LAMBDA}'), 'rb') as f:
            s_bucket = pickle.load(f)
        agt = Agent(TEST_STATE_SET[0])
        agt.z_bucket = z_bucket
        agt.s_bucket = s_bucket
        Q.append([agt.Q(s) for s in TEST_STATE_SET])
    rt = np.array(rt)
    mu, sigma = rt.mean(), rt.mean(0).std()
    rt = (rt - mu) / sigma
    rt_avg = rt.mean(0)
    rt_avg_all.append(rt_avg)
    rt_std = rt.std(0)
    Q_avg = np.array(Q).mean(0)

    plt.plot(rt_avg)
    plt.xlabel('Action')
    plt.ylabel('Average return')
    plt.xlim(0, 3)
    ax = plt.gca()
    ax.fill_between(range(len(rt_avg)), rt_avg - rt_std, rt_avg + rt_std, alpha = 0.2)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_xticklabels(ACTION_NAME_SET)
    plt.savefig(os.path.join(DIR, 'plot', f'z_a_agg_lambda_{LAMBDA}.pdf'))
    plt.close()

    for STATE_idx, _ in enumerate(TEST_STATE_SET):
        plt.plot(Q_avg[STATE_idx], label = TEST_STATE_NAME_SET[STATE_idx], c = COLOR_SET[STATE_idx])
    plt.legend(loc = 'best')
    plt.xlabel('Action')
    plt.ylabel('Q value')
    plt.xlim(0, 3)
    ax = plt.gca()
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_xticklabels(ACTION_NAME_SET)
    plt.savefig(os.path.join(DIR, 'plot', f'Q_value_agg_lambda_{LAMBDA}.pdf'))
    plt.close()

for idx, LAMBDA in enumerate(LAMBDA_SET):
    plt.plot(rt_avg_all[idx], label = f'$\lambda={LAMBDA}$', c = COLOR_SET[idx])
plt.legend(loc = 'best')
plt.xlabel('Action')
plt.ylabel('Average return')
plt.xlim(0, 3)
ax = plt.gca()
ax.xaxis.set_major_locator(MaxNLocator(integer=True))
ax.set_xticklabels(ACTION_NAME_SET)
plt.savefig(os.path.join(DIR, 'plot', f'z_a_agg_all.pdf'))
plt.close()