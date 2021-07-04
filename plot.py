import numpy as np
import matplotlib.pyplot as plt
import os


# strategies = {'FS': 'Factored SAD', 'LR': 'Logistic regression'}
strategies = {'LR': 'Logistic regression'}

for strategy in strategies.keys():
    reward_per_100_steps = []
    average_score = []
    for i in range(10):
        if os.path.isfile(f'./evaluation/{strategy}/rp100s_trial_' + str(i + 1) + '.npy'):
            reward_per_100_steps.append(np.load(f'./evaluation/{strategy}/rp100s_trial_' + str(i + 1) + '.npy')) # 10k-step interval
            average_score.append(np.load(f'./evaluation/{strategy}/as_trial_' + str(i + 1) + '.npy')) # 20-episode interval
    min_len = min(len(seq) for seq in reward_per_100_steps)
    reward_per_100_steps = np.array([seq[:min_len] for seq in reward_per_100_steps]).mean(0)
    min_len = min(len(seq) for seq in average_score)
    average_score = np.array([seq[:min_len] for seq in average_score]).mean(0)
    strategies[strategy] = (strategies[strategy], reward_per_100_steps, average_score)

max_len = max(len(reward_per_100_steps) for _, reward_per_100_steps, _ in strategies.values())
t = range(10, 10 * max_len + 1, 10)
for strategy_name, reward_per_100_steps, _ in strategies.values():
    plt.plot(t[:len(reward_per_100_steps)], reward_per_100_steps, label = strategy_name)
plt.legend(loc = 'best')
plt.xlabel('Steps (1000\'s)')
plt.ylabel('Reward per 100 steps')
plt.xlim(0, None)
# plt.ylim(-2.5, 1)
# plt.show()
plt.savefig('./plot/reward_per_100_steps.pdf')
plt.close()

max_len = max(len(average_score) for _, _, average_score in strategies.values())
t = range(20, 20 * max_len + 1, 20)
for strategy_name, _, average_score in strategies.values():
    plt.plot(t[:len(average_score)], average_score, label = strategy_name)
plt.legend(loc = 'best')
plt.xlabel('Episodes')
plt.ylabel('Average score')
plt.xlim(0, None)
# plt.ylim(-21, None)
# plt.show()
plt.savefig('./plot/average_score.pdf')
plt.close()