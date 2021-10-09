import numpy as np
import ot
import sys


## Generate samples
def simul_two_Gaussians(num_samples, seed=49):
    np.random.seed(seed)

    mean_X = np.array([3, 3])
    cov_X = np.array([[1, 0], [0, 1]])
    X = np.random.multivariate_normal(mean_X, cov_X, num_samples)

    mean_Y = np.array([4, 4])
    cov_Y = np.array([[1, -0.2], [-0.2, 1]])
    Y = np.random.multivariate_normal(mean_Y, cov_Y, num_samples)

    return X, Y


num_samples = 100
X, Y = simul_two_Gaussians(num_samples, seed=49)
a, b = (1 / num_samples) * np.ones(num_samples), (1 / num_samples) * np.ones(
    num_samples
)


## Compute some costs cost matrices
C1 = EOT_sinkhorn.Square_Euclidean_Distance(X, Y)
C2 = EOT_sinkhorn.alpha_Euclidean_Distance(X, Y, alpha=2.2)
C3 = EOT_sinkhorn.Trivial_cost(X, Y)
C = np.zeros((3, num_samples, num_samples))
C[0, :, :] = C1
C[1, :, :] = C2
C[2, :, :] = C3

## Or consider the example defined in the sequential OT experiment
N = 2
C = EOT_sinkhorn.N_cost_matrices(0.7, X, Y, N, seed_init=49)

## Compute EOT distance
# Projected Sinkhorn algorithm
max_iter = 3000
reg = 5 * 1e-3
res, acc, times, lam, alpha, beta, denom, KC, K_trans = EOT_sinkhorn.EOT_PSinkhorn(
    C, reg, a, b, max_iter=max_iter
)
print(res)

Couplings = (alpha[:, np.newaxis] * K_trans * beta[np.newaxis, :]) / denom
P1, P2 = Couplings
P = P1 + P2
# check that P satisfies the marginal constraints
np.sum(P, axis=0)
np.sum(P, axis=1)

# Compute the primal formulation of the entropic EOT
res_primal = EOT_sinkhorn.compute_EOT_primal(lam, alpha, beta, KC, denom, reg, a, b)
print(res_primal)

## Accelerated  PGD method
max_iter = 3000
res, acc, times, lam, alpha, beta, denom, KC, K_trans = EOT_sinkhorn.EOT_APGD(
    C, reg, a, b, max_iter=max_iter
)
print(res)

# Compute the primal formulation of the entropic EOT
res_primal = EOT_sinkhorn.compute_EOT_primal(lam, alpha, beta, KC, denom, reg, a, b)
print(res_primal)


## Linear Program: primal formulation
res = EOT_sinkhorn.LP_solver_Primal(C, a, b)
print(res["fun"])

P1 = res["x"][1 : num_samples * num_samples + 1]
P2 = res["x"][num_samples * num_samples + 1 : 2 * num_samples * num_samples + 1]
np.sum(P1)
np.sum(P2)
