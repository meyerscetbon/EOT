import numpy as np
from scipy.optimize import linprog
import time

## Linear Program: primal formulation of EOT
def LP_solver_Primal(M, a, b):

    if len(np.shape(M)) != 3:
        M = np.expand_dims(M, axis=0)

    N, n, m = np.shape(M)

    M_flat = []
    for k in range(N):
        M_flat.append(M[k, :, :].flatten("C"))

    A_ub = np.zeros((N, N * n * m + 1))
    A_ub[:, 0] = -1
    for k in range(N):
        A_ub[k, 1 + k * n * m : 1 + (k + 1) * n * m] = M_flat[k]

    b_ub = np.zeros(N)

    A_eq = np.zeros((n + m, N * n * m + 1))
    for i in range(n):
        for k in range(N):
            A_eq[i, 1 + k * m * n + m * i : 1 + k * m * n + m * (i + 1)] = 1

    for j in range(m):
        for k in range(N):
            ind_j = [1 + k * m * n + i * m + j for i in range(n)]
            A_eq[n + j, ind_j] = 1

    A_eq = A_eq[:-1, :]

    b_eq = np.zeros(n + m - 1)
    b_eq[:n] = a
    b_eq[n:] = b[:-1]

    c = np.zeros(N * n * m + 1)
    c[0] = 1

    bounds = [(None, None)] + [(0, None)] * (N * n * m)
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds)
    return res


## Linear Program: dual formulation of EOT
def LP_solver_Dual(M, a, b):
    # x = (lambda,f,g)

    if len(np.shape(M)) != 3:
        M = np.expand_dims(M, axis=0)

    N, n, m = np.shape(M)

    M_flat = []
    for k in range(N):
        M_flat.append(M[k, :, :].flatten("C"))

    A_ub = np.zeros((N * n * m, N + n + m))
    for k in range(N):
        A_ub[k * n * m : (k + 1) * n * m, k] = -M_flat[k]

    for k in range(N):
        for q in range(n * m):
            i = q // m
            j = q % m
            A_ub[k * n * m + q, N + i] = 1
            A_ub[k * n * m + q, N + n + j] = 1

    b_ub = np.zeros(N * n * m)

    A_eq = np.zeros((1, N + n + m))
    for k in range(N):
        A_eq[0, k] = 1

    b_eq = np.ones(1)

    c = np.zeros(N + n + m)
    c[N : N + n] = a
    c[N + n :] = b

    bounds = [(0, 1)] * N + [(None, None)] * (n + m)
    res = linprog(-c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds)
    return res


## Projected Gradient Descent: for N costs
# C is of size (N,n,m)
def EOT_PGD(C, reg, a, b, max_iter=1000):
    start = time.time()
    acc = []
    times = []

    if len(np.shape(C)) != 3:
        C = np.expand_dims(C, axis=0)

    N, n, m = np.shape(C)
    K = np.exp(-(C / reg))

    L_1 = 2 * N / reg
    L_2 = np.max(np.abs(C)) ** 2 / reg
    L = max(L_1, L_2)

    lam = (1 / N) * np.ones(N)

    f, g = np.zeros(n), np.zeros(n)
    grad_lam, grad_f, grad_g, denom, K_lam_trans, K_trans = compute_grad_EOT(
        lam, f, g, K, C, a, b, reg
    )

    for k in range(max_iter):

        # Update f , g, lam
        lam_trans = lam + (1 / L) * grad_lam
        lam = projection_simplex_sort(lam_trans)
        f = f + (1 / L) * grad_f
        g = g + (1 / L) * grad_g

        # Compute grad
        grad_lam, grad_f, grad_g, denom, K_lam_trans, K_trans = compute_grad_EOT(
            lam, f, g, K, C, a, b, reg
        )

        # Compute loss
        GOT_trans = compute_EOT_dual_2(lam, f, g, denom, reg, a, b)
        if np.isnan(GOT_trans) == True:
            return "Error"
        else:
            acc.append(GOT_trans)
            end = time.time()
            times.append(end - start)

    alpha = np.exp(f / reg)
    beta = np.exp(g / reg)

    return acc[-1], acc, times, lam, alpha, beta, denom, K_lam_trans, K_trans


#### Accelerated Projected Gradient Ascent: for N costs ####
# C is of size (N,n,m)
def EOT_APGD(C, reg, a, b, max_iter=1000):
    start = time.time()
    acc = []
    times = []

    if len(np.shape(C)) != 3:
        C = np.expand_dims(C, axis=0)

    N, n, m = np.shape(C)
    K = np.exp(-(C / reg))

    L_1 = 2 * N / reg
    L_2 = np.max(np.abs(C)) ** 2 / reg
    L = max(L_1, L_2)

    lam_old = (1 / N) * np.ones(N)
    lam = lam_old.copy()

    f_old, g_old = np.zeros(n), np.zeros(n)
    f, g = f_old.copy(), g_old.copy()

    for k in range(max_iter):

        v = lam + ((k - 2) / (k + 1)) * (lam - lam_old)
        w = f + ((k - 2) / (k + 1)) * (f - f_old)
        z = g + ((k - 2) / (k + 1)) * (g - g_old)

        lam_old = lam.copy()
        f_old = f.copy()
        g_old = g.copy()

        # Update f, g, lam
        (
            grad_lam,
            grad_f,
            grad_g,
            denom_useless,
            K_lam_trans_useless,
            K_trans_useless,
        ) = compute_grad_EOT(v, w, z, K, C, a, b, reg)
        lam_trans = v + (1 / L) * grad_lam
        lam = projection_simplex_sort(lam_trans)
        f = w + (1 / L) * grad_f
        g = z + (1 / L) * grad_g

        # Update the total cost
        K_trans = np.zeros((N, n, m))
        K_lam_trans = np.zeros((N, n, m))
        for k in range(N):
            K_trans[k, :, :] = K[k, :, :] ** lam[k]
            K_lam_trans[k, :, :] = K_trans[k, :, :].copy() * C[k, :, :]
        alpha = np.exp(f / reg)
        beta = np.exp(g / reg)
        beta_trans = np.sum(np.dot(K_trans, beta), axis=0)
        denom_trans = alpha * beta_trans
        denom = np.sum(denom_trans)

        GOT_trans = compute_EOT_dual_2(lam, f, g, denom, reg, a, b)
        if np.isnan(GOT_trans) == True:
            return "Error"
        else:
            acc.append(GOT_trans)
            end = time.time()
            times.append(end - start)

    return acc[-1], acc, times, lam, alpha, beta, denom, K_lam_trans, K_trans


def compute_grad_EOT(lam, f, g, K, C, a, b, reg):
    N, n, m = np.shape(K)

    denom = 0
    alpha = np.exp(f / reg)
    beta = np.exp(g / reg)

    K_trans = np.zeros((N, n, m))
    K_lam_trans = np.zeros((N, n, m))

    for k in range(N):
        K_trans[k, :, :] = K[k, :, :] ** lam[k]
        K_lam_trans[k, :, :] = K_trans[k, :, :].copy() * C[k, :, :]

    beta_trans = np.sum(np.dot(K_trans, beta), axis=0)
    alpha_trans = np.sum(np.dot(np.transpose(K_trans, (0, 2, 1)), alpha), axis=0)

    f_num = alpha * beta_trans
    g_num = beta * alpha_trans
    denom = np.sum(f_num)

    lam_trans = np.dot(K_lam_trans, beta)
    grad_lam = np.sum(alpha * lam_trans, axis=1)

    grad_f = a - reg * (f_num / denom)
    grad_g = b - reg * (g_num / denom)
    grad_lam = grad_lam / denom

    return grad_lam, grad_f, grad_g, denom, K_lam_trans, K_trans


#### Projected Sinkhorn: for N costs ####
# C is of size (N,n,m)
def EOT_PSinkhorn(C, reg, a, b, max_iter=1000, tau=1e-20, stable=0):
    start = time.time()
    acc = []
    times = []

    if len(np.shape(C)) != 3:
        C = np.expand_dims(C, axis=0)

    N, n, m = np.shape(C)

    K = np.exp(-(C / reg))
    L = np.max(np.abs(C)) ** 2 / reg
    lam = (1 / N) * np.ones(N)
    alpha, beta = np.ones(n), np.ones(m)

    K_trans = np.zeros((N, n, m))
    K_lam_trans = np.zeros((N, n, m))

    denom = 1
    stop = 1
    k = 0

    while stop > tau and k < max_iter:

        for j in range(N):
            K_trans[j, :, :] = K[j, :, :] ** lam[j]
            K_lam_trans[j, :, :] = K_trans[j, :, :].copy() * C[j, :, :]

        K_update = np.sum(K_trans, axis=0)

        # Update alpha
        alpha_trans = np.dot(K_update, beta) + stable
        denom = np.dot(alpha, alpha_trans)
        alpha = denom * (a / alpha_trans)

        # Update beta
        beta_trans = np.dot(K_update.T, alpha) + stable
        denom = np.dot(beta, beta_trans)
        beta = denom * (b / beta_trans)

        # Update lam
        lam_trans = np.dot(K_lam_trans, beta)
        lam_trans = lam_trans * alpha
        grad_lam = np.sum(lam_trans, axis=1) / (denom + stable)
        lam_trans = lam + (1 / L) * grad_lam
        lam = projection_simplex_sort(lam_trans)

        # Update the total cost
        GOT_trans = compute_EOT_dual(lam, alpha, beta, denom, reg, a, b)
        if np.isnan(GOT_trans) == True:
            return "Error"
        else:
            acc.append(GOT_trans)
            end = time.time()
            times.append(end - start)

        # stop = np.abs((GOT_trans-GOT_trans_old)/GOT_trans_old)
        # GOT_trans_old = GOT_trans

        k = k + 1

    return acc[-1], acc, times, lam, alpha, beta, denom, K_lam_trans, K_trans


## Projection on the simplex
def projection_simplex_sort(v, z=1):
    n_features = v.shape[0]
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u) - z
    ind = np.arange(n_features) + 1
    cond = u - cssv / ind > 0
    rho = ind[cond][-1]
    theta = cssv[cond][-1] / float(rho)
    w = np.maximum(v - theta, 0)
    return w


## Evaluate GOT
def compute_EOT_dual(lam, alpha, beta, denom, reg, a, b):
    n = np.shape(a)[0]
    m = np.shape(b)[0]
    res = reg * (np.dot(a, np.log(alpha)) + np.dot(b, np.log(beta)))
    res_trans = reg * (np.log(denom))
    return res - res_trans + reg * np.log(n * m)


def compute_EOT_dual_2(lam, f, g, denom, reg, a, b):
    n = np.shape(a)[0]
    m = np.shape(b)[0]
    res = np.dot(a, f) + np.dot(b, g)
    res_trans = reg * np.log(denom)
    return res - res_trans + reg * np.log(n * m)


def compute_EOT_primal(lam, alpha, beta, KC, denom, reg, a, b):
    alpha_trans = np.dot(KC, beta)
    dot_products = np.dot(alpha_trans, alpha)
    res = np.dot(dot_products, lam) / denom
    return res


## Cost functions
def alpha_L1_Distance(X, Y, alpha=1):
    X_col = X[:, np.newaxis]
    Y_lin = Y[np.newaxis, :]
    C = np.sum(np.abs(X_col - Y_lin), 2)
    C = C ** (alpha)
    return C


def Square_Euclidean_Distance(X, Y):
    X_col = X[:, np.newaxis]
    Y_lin = Y[np.newaxis, :]
    C = np.sum((X_col - Y_lin) ** 2, 2)
    return C


def alpha_Euclidean_Distance(X, Y, alpha=1):
    X_col = X[:, np.newaxis]
    Y_lin = Y[np.newaxis, :]
    C = (np.sqrt(np.sum((X_col - Y_lin) ** 2, 2))) ** (alpha)
    return C


def alpha_Euclidean_cost_equivalent(X, Y, alpha=1):
    X_col = X[:, np.newaxis]
    Y_lin = Y[np.newaxis, :]
    C = (np.sqrt(np.sum((X_col - Y_lin) ** 2, 2))) ** (alpha)
    C_den = 1 + C
    C = C / C_den
    return C


def Lp_to_the_p_cost(X, Y, p=1):
    X_col = X[:, np.newaxis]
    Y_lin = Y[np.newaxis, :]
    C = np.sum(np.abs(X_col - Y_lin) ** p, 2)
    return C


def Trivial_cost(X, Y):
    X_col = X[:, np.newaxis]
    Y_lin = Y[np.newaxis, :]
    C = np.sqrt(np.sum((X_col - Y_lin) ** 2, 2))
    res = C == 0
    res = 1 - res.astype(float)

    return res


def Angle_cost(X, Y, stable=1e-6):
    C = np.dot(X, Y.T)
    norm_X = np.sqrt(np.sum(X ** 2, 1)) + stable
    norm_Y = np.sqrt(np.sum(Y ** 2, 1)) + stable
    C = C / norm_X
    C = C / norm_Y

    return C


def Teleport_cost_alpha(X, Y, stable=1e-6, alpha=1):
    X_col = X[:, np.newaxis]
    Y_lin = Y[np.newaxis, :]
    C_inv = np.sqrt(np.sum(np.abs(X_col - Y_lin) ** 2, 2)) ** alpha + stable
    C = 1 / C_inv

    return C


def Cos_cost(X, Y):
    C = np.dot(X, Y.T)
    C = np.cos(C) + 1

    return C


def cost_fixed_line(n, m, cost_function):
    X = np.arange(n).reshape(n, 1)
    Y = np.arange(m).reshape(m, 1)

    C = cost_function(X / n, Y / n)

    return C


def cost_matrix_sequential(alpha, w, x, y):
    res = alpha_Euclidean_Distance(x, y)

    xw = x.dot(w)
    yw = y.dot(w)

    wxy = np.tile(xw, (len(yw), 1)).T - np.tile(yw, (len(xw), 1))

    return res - alpha * wxy


def N_cost_matrices(alpha, x, y, N, seed_init=49):
    n = len(x)
    m = len(y)
    C = np.zeros((N, n, m))
    seed = seed_init
    np.random.seed(seed)
    for i in range(N):
        rho = np.sqrt(np.random.uniform())
        theta = np.random.uniform(0, 2 * np.pi)
        w = [rho * np.cos(theta), rho * np.sin(theta)]
        C[i, :, :] = cost_matrix_sequential(alpha, w, x, y)
        seed = seed + 1
        np.random.seed(seed)

    return C
