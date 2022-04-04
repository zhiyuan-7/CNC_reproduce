from argparse import ArgumentParser

import agent_control

ACTION = 0



def read_command():
    parser = ArgumentParser()

    parser.add_argument('-g', '--game', dest = 'game',
                        help = 'the name of the Atari game', default = 'Pong-v0')
    parser.add_argument('-s', '--strategy', dest = 'strategy',
                        help = 'the strategy used to control the agent', default = 'FS')
    parser.add_argument('-e', '--evaluation', dest = 'evaluation', action = 'store_true',
                        help = 'evaluate the strategy, i.e. train mode', default = False)
    parser.add_argument('-t', '--trial_num', dest = 'trial_num', type = int,
                        help = 'number of trial', default = 1)

    args = parser.parse_args()

    return args


def run_game_interface(args):
    A = agent_control.Agent_control(args)
    A.control()

if __name__ == '__main__':
    args = read_command()  # Get game components based on input
    run_game_interface(args)