# True parameters
BETA = 0.27
SIGMA = 1./2.
GAMMA = 1./7.
# RHO = 1./90.
RHO = 0.1

# Testing accuracy
# sensitivity (ie. recall, true positive rate) = TP / (TP + FN) = 1 - FNR
# specificity (ie. true negative rate) = TN / (TN + FP) = 1 - FPR
FPR = 0.1  # false positive rate
FNR = 0.3  # false negative rate
# FPR = 0.01  # false positive rate
# FNR = 0.03  # false negative rate
MU_S = FPR  # if tested, Pr(O = P | s = S) = MU_S, Pr(O = N | s = S) = 1 - MU_S
MU_E = 1 - FNR
MU_I = 1 - FNR
MU_R = FPR

# Testing scope
ALPHA_S = 0.2
ALPHA_E = 0.7
ALPHA_I = 0.9
ALPHA_R = 0.05
# ALPHA_S = 0.4
# ALPHA_E = 0.4
# ALPHA_I = 0.4
# ALPHA_R = 0.4

# N = 500
# M = 512
N = 1
M = 50
# MAX_ITER = 600  # months
MAX_ITER = 800
N_SAMPLE = 1024

N_POS_INIT = 3
JITTER_VAR_ALL_MAX = [1e-4, 1e-4, 1e-4, 9e-6]
JITTER_VAR_ALL_MIN = [9e-6, 9e-6, 9e-6, 8.1e-7]
DECAY = 0.996

# VERBOSE = True
CALC_EVAL = True

PRIOR = [0.97, 0.01, 0.01, 0.01]

GRAPH_FILE = '../data/email-1k-5k.adj.npz'

ESTIMATE = True
CONTROL = False
LAMBDA = 10
ACTION_DURATION = 3

REP = 10
SAVING_PATH = f'./result/bc_rank/lambda_{LAMBDA}/rep_{REP}'
if not CONTROL:
    SAVING_PATH = './result'
import os
if not os.path.exists(SAVING_PATH):
    os.makedirs(SAVING_PATH)