import numpy as np
from scipy.linalg import expm
import random
import matplotlib.pyplot as plt

# parameters
lambda_0 = 1.0
lambda_1 = 0.2
lambda_2 = 1.0
c = 0.3
leakage_0 = 0.2
leakage_1 = 0.6

# plot settings and max time
T_MAX = 10.0
TIME_GRID = np.linspace(0, T_MAX, 200)

N_SIMS = 5000
SEED = 7
random.seed(SEED)
np.random.seed(SEED)


# state space (a,b) with a,b in {0,1,2} S x S
states = [(0,0),(1,0),(2,0),
          (0,1),(1,1),(2,1),
          (0,2),(1,2),(2,2)]

state_to_idx = {s:i for i,s in enumerate(states)}
idx_to_state = {i:s for s,i in state_to_idx.items()}

# synchronized states (0,0), (1,1), (2,2)
SYNC_STATES = {state_to_idx[(0,0)], state_to_idx[(1,1)], state_to_idx[(2,2)]}

# building the matrix and inputing values
# 0 -> 1 at rate lambda_0
# 1 -> 2 at rate lambda_1 = (1 - leakage_1)lambda_0
# 2 -> 0 at rate lamba_2 (flash)


def add_rate(Q, from_state, to_state, rate):
        if rate <= 0:
            return
        i = state_to_idx[from_state]
        j = state_to_idx[to_state]
        Q[i,j] += rate

def build_Q(lambda_0, lambda_1, lambda_2, c, leakage_0, leakage_1):
    Q = np.zeros((9,9), dtype=float)
    for (a,b) in states:
        if a == 0:
            add_rate(Q, (a,b), (1,b), lambda_0)
        elif a == 1:
            add_rate(Q, (a,b), (2,b), lambda_1)
        elif a == 2:
            add_rate(Q, (a,b), (0,b), lambda_2)
            if b == 0:
                add_rate(Q, (a,b), (0,1), c * (1 - leakage_0))
            elif b == 1:
                add_rate(Q, (a,b), (0,2), c * (1 - leakage_1))

        if b == 0:
            add_rate(Q, (a,b), (a,1), lambda_0)
        elif b == 1:
            add_rate(Q, (a,b), (a,2), lambda_1)
        elif b == 2:
            add_rate(Q, (a,b), (a,0), lambda_2)
            if a == 0:
                add_rate(Q, (a,b), (1,0), c * (1 - leakage_0))
            elif a == 1:
                add_rate(Q, (a,b), (2,0), c * (1 - leakage_1))
    for i in range(9):
        Q[i,i] = -np.sum(Q[i,:]) # exit rates are negative

    return Q

def lock_Q(Q): # creates aborbing states for synchronized states
    QL = Q.copy()
    for s in SYNC_STATES:
        QL[s,:] = 0.0
        QL[s,s] = 0.0
    return QL


def dist_at_time(Q, pi0, t):
    return pi0 @ expm(Q * t)

def p_sync_by_t(Q_locked, pi0, t):
    pi_t = dist_at_time(Q_locked, pi0, t)
    return float(np.sum([pi_t[s] for s in SYNC_STATES]))

Q = build_Q(lambda_0, lambda_1, lambda_2, c, leakage_0, leakage_1)
Q_locked = lock_Q(Q)


start_state = (0,1)
start_idx = state_to_idx[start_state]

pi0 = np.zeros(9)
pi0[start_idx] = 1.0

math_sync = np.array([p_sync_by_t(Q_locked, pi0, t) for t in TIME_GRID])
for t_chk in [0.5, 1.0, 2.0, 5.0, 10.0]:
    m = p_sync_by_t(Q_locked, pi0, t_chk)
    j = int(np.argmin(np.abs(TIME_GRID - t_chk)))
    print(f"t={t_chk:>4}: P(sync by t) = {m:.4f}")

plt.figure(figsize=(12,6))
plt.plot(TIME_GRID, math_sync, label="(exp(Q_locked t))")
plt.xlabel("Time")
plt.ylabel("P(sync by t)")
plt.title("synchronization")
plt.grid(True, alpha=0.3)
plt.legend()
plt.show()