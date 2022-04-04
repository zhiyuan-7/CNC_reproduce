# A Logic-based Extension of CNC in Partially Observable, Complex but Structured Environments

## Evaluation platform
The platform for epidemic evolution simulation is provided by Dawei Chen (dawei.chen@anu.edu.au).
The parameters of the transition model are $\beta=0.27,\sigma=0.5,\gamma=\frac{1}{7},\rho=0.1$, and the parameters of the observation model are $\alpha_S=0.2,\alpha_E=0.7,\alpha_I=0.9,\alpha_R=0.05,\lambda_1=0.1,\lambda_2=0.3$.

## Experimental setup
An email communication network with 1,133 nodes and 5,451 edges is adopted as the contact network $G$.
The number of particles for the factored particle filter is 50.
For the SAD estimator of $\rho_Z$, $\beta=1$.
For the $\epsilon$-greedy algorithm, $\epsilon$ is set to 1 initially and 0.02 after 300 time steps, decaying linearly in between.
Each action taken by the agent will last 3 time steps in order to improve the stability.
The horizon $m=20$ steps.
In the reward function, $\lambda\in\{0.1,1,3,10\}$.
For each possible value of $\lambda$, the agent is tested over 10 trials with length of 800 time steps each.