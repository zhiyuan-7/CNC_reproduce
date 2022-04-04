# Reproduce of CNC
 
## Author
Zhiyuan Wu

## Evaluation platform
The agent is evaluated using Atari 2600 video games.
The interface used is OpenAI Gym, a toolkit for developing and comparing RL algorithms.
Each frame, which is generated at 60Hz, contains 160X210 color pixels.
The agent is tested by the game of PONG with an action space of UP, DOWN, NOOP.
Whenever the agent scores, it will receive reward +1, and it will be rewarded -1 if the opponent scores.
If either of the players score 21 points, the episode ends, which means the score for an episode ranges from -21 to 21.

## Experimental setup
Two CNC agents are evaluated according to the procedure presented in the paper, both adopting the SAD estimator for $\rho_Z$ with $\beta=1$.
For $\rho_S$, the former adopts the Factored SAD model, while the latter adopts the Lempel-Ziv model.
For the $\epsilon$-greedy algorithm, $\epsilon$ is set to 1 initially and 0.02 after 200,000 time steps, decaying linearly in between.
The horizon $m=80$ steps.
The agents are tested over 10 trials with length of 2,000,000 time steps each.