import time
import gym
import numpy as np
import pickle
import os



class Agent_control():
    def __init__(self, args):
        self.env = gym.make(args.game)
        self.strategy = args.strategy
        self.evaluation = args.evaluation
        self.trial_num = args.trial_num

        self.step_per_trial = 10000000
        self.save_dir = os.path.join('./evaluation/', self.strategy)
        self.pickle_file = os.path.join(self.save_dir, 'trial_' + str(self.trial_num))

        self.action_kb = 0


    def control(self):
        if self.strategy == 'keyboard':
            self.keyboard_control()
        elif self.strategy == 'random':
            self.random_control()
        else:
            self.strategy_control()


    def keyboard_control(self):
        self.env.reset()
        while(True):
            self.env.render()
            self.env.step(self.action_kb)
            time.sleep(0.05)


    def random_control(self):
        self.env.reset()
        count = 0
        R = 0
        while(True):
            self.env.render()
            _, reward, done, _ = self.env.step(self.env.action_space.sample())
            if done:
                self.env.reset()
            # time.sleep(0.02)
            count += 1
            R += reward
            if not count % 1000:
                print(R / count * 100)


    def strategy_control(self):
        t0 = time.time()
        if os.path.isfile(self.pickle_file):
            with open(self.pickle_file, 'rb') as ipt:
                s = pickle.load(ipt)
                s.load()
            self.env.reset()
            s.R_in_last_20_episodes[-1] = 0
        elif self.strategy == 'FS':
            from factored_sad import Factored_SAD
            s = Factored_SAD(self.env.reset())
        elif self.strategy == 'LR':
            from logistic_regression import Logistic_Regression
            s = Logistic_Regression(self.env.reset())
        elif self.strategy == 'LZ':
            from lempel_ziv import Lempel_Ziv
            s = Lempel_Ziv(self.env.reset())
        else:
            print('Warning: None of the strategy is being used!')
            from strategy import Strategy
            s = Strategy(self.env.reset())
        while(True):
            if not self.evaluation:
                s.eps = 0

            self.env.render()
            action = s.get_action()
            observation, reward, done, _ = self.env.step(action)

            if done:
                observation = self.env.reset()
            
            s.feedback(action, observation, reward, done, training = self.evaluation)

            if len(s.R_per_10k_steps) and len(s.average_score_per_episode) and not s.t % (1000 * 10):
                print('{}k steps, {} mins, average reward {} per 100 steps, average score {} per episode.'
                      .format(s.t // 1000, round((time.time() - t0) / 60, 1), round(s.R_per_10k_steps[-1] / 100, 2),
                              round(s.average_score_per_episode[-1], 2)))

            if self.evaluation:
                if not s.t % (1000 * 20):
                    np.save(os.path.join(self.save_dir, 'rp100s_trial_' + str(self.trial_num) + '.npy'), np.array(s.R_per_10k_steps) / 100)
                    np.save(os.path.join(self.save_dir, 'as_trial_' + str(self.trial_num) + '.npy'), np.array(s.average_score_per_episode))
                    s.save()
                    with open(self.pickle_file, 'wb') as output:
                        pickle.dump(s, output, pickle.HIGHEST_PROTOCOL)
                    s.load()

                if s.t >= self.step_per_trial:
                    break
        self.env.close()