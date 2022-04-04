import numpy as np
import matplotlib.pyplot as plt
import os


strategies = {'FS': 'Factored SAD', 'LZ': 'Lempel-Ziv'}
# strategies = {'FS': 'Factored SAD', 'LZ': 'Lempel-Ziv', 'FS_an_5': 'Factored SAD anchor 5'}
# strategies = {'LR': 'Logistic regression'}
# strategies = {'FS': 'Factored SAD'}
# strategies = {'LZ': 'Lempel-Ziv'}

for strategy in strategies.keys():
    reward_per_100_steps = []
    average_score = []
    for i in range(10):
        if os.path.isfile(f'./evaluation/{strategy}/rp100s_trial_' + str(i + 1) + '.npy'):
            reward_per_100_steps.append(np.load(f'./evaluation/{strategy}/rp100s_trial_' + str(i + 1) + '.npy')) # 10k-step interval
            average_score.append(np.load(f'./evaluation/{strategy}/as_trial_' + str(i + 1) + '.npy')) # 20-episode interval
    # min_len = min(len(seq) for seq in reward_per_100_steps)
    # r = np.array([seq[:min_len] for seq in reward_per_100_steps])
    # print(strategy, r[:, -1].mean(), r[:, -1].std())
    # reward_per_100_steps = r.mean(0)
    # min_len = min(len(seq) for seq in average_score)
    # s = np.array([seq[:min_len] for seq in average_score])
    # print(strategy, s[:, -1].mean(), s[:, -1].std())
    # average_score = s.mean(0)
    strategies[strategy] = (strategies[strategy], reward_per_100_steps, average_score)

min_len = min(min(len(seq) for seq in reward_per_100_steps) for _, reward_per_100_steps, _ in strategies.values())
t = range(10, 10 * min_len + 1, 10)
for strategy_name, reward_per_100_steps, _ in strategies.values():
    r = np.array([seq[:min_len] for seq in reward_per_100_steps])
    print(f'{strategy_name}: mean {r[:, -1].mean()}, std {r[:, -1].std()}')
    plt.plot(t, r.mean(0), label = strategy_name)
plt.legend(loc = 'best')
plt.xlabel('Steps (1000\'s)')
plt.ylabel('Reward per 100 steps')
plt.xlim(0, max(t))
# plt.ylim(-2.5, 1)
# plt.show()
plt.savefig('./plot/reward_per_100_steps.pdf')
plt.close()

min_len = min(min(len(seq) for seq in average_score) for _, _, average_score in strategies.values())
t = range(20, 20 * min_len + 1, 20)
for strategy_name, _, average_score in strategies.values():
    a = np.array([seq[:min_len] for seq in average_score])
    print(f'{strategy_name}: mean {a[:, -1].mean()}, std {a[:, -1].std()}')
    plt.plot(t, a.mean(0), label = strategy_name)
plt.legend(loc = 'best')
plt.xlabel('Episodes')
plt.ylabel('Average score')
plt.xlim(0, max(t))
plt.ylim(-21, None)
# plt.show()
plt.savefig('./plot/average_score.pdf')
plt.close()