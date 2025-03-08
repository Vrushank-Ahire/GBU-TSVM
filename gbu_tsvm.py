import numpy as np
import time
from numpy import linalg
from cvxopt import matrix, solvers

def GBUTSVM(DataTrain, TestX, d1, d2, du, EPS):
    start = time.time()
    eps1 = 0.0005
    eps2 = 0.0005
    reg_term = 1e-6

    A = DataTrain[DataTrain[:, -1] == 1, :-1]
    B = DataTrain[DataTrain[:, -1] != 1, :-1]

    np.random.shuffle(A)
    np.random.shuffle(B)

    C_plus = A[:, :-1]
    C_minus = B[:, :-1]
    R_plus = A[:, -1]
    R_minus = B[:, -1]

    m1 = C_plus.shape[0]
    m2 = C_minus.shape[0]
    e_plus = np.ones((m1, 1))
    e_minus = np.ones((m2, 1))

    k = min(m1, m2) // 2
    U = C_plus[:k, :] + C_minus[:k, :]
    U = U / 2

    H = np.hstack((C_plus, e_plus))
    G = np.hstack((C_minus, e_minus))
    e_universum = np.ones((U.shape[0], 1))
    R_universum = np.zeros((U.shape[0], 1))
    O = np.hstack((U, e_universum))

    HH1 = np.dot(H.T, H) + eps1 * np.eye(H.shape[1]) + reg_term * np.eye(H.shape[1])
    GO = np.vstack([G, -O])
    HH_inv = linalg.solve(HH1, GO.T)
    kerH = np.dot(GO, HH_inv)
    kerH = (kerH + kerH.T) / 2

    z_dim = kerH.shape[0]
    tilde_P = kerH
    e_universum = np.ones((U.shape[0], 1))
    R_universum = np.zeros((U.shape[0], 1))
    tilde_q = np.hstack([(e_minus.T - R_minus.T), ((EPS - 1) * e_universum.T - R_universum.T)]).T

    G_neg = -np.eye(z_dim)
    h_neg = np.zeros(z_dim)
    G_pos = np.eye(z_dim)
    h_pos = np.hstack([d1 * np.ones(G.shape[0]), du * np.ones(O.shape[0])])
    G_combined = np.vstack([G_neg, G_pos])
    h_combined = np.hstack([h_neg, h_pos])

    solvers.options['show_progress'] = False
    solution = solvers.qp(matrix(tilde_P, tc='d'), matrix(tilde_q, tc='d'), matrix(G_combined, tc='d'), matrix(h_combined, tc='d'))

    z_sol = np.array(solution['x']).flatten()
    alpha = z_sol[:G.shape[0]]
    mu = z_sol[G.shape[0]:]

    GO = np.vstack([G, -O])
    w_b_plus = GO.T @ z_sol
    w_b_plus = linalg.solve(np.dot(H.T, H) + eps1 * np.eye(H.shape[1]) + reg_term * np.eye(H.shape[1]), w_b_plus)
    w_plus = w_b_plus[:-1]
    b_plus = w_b_plus[-1]

    P = np.hstack((C_plus, e_plus))
    Q = np.hstack((C_minus, e_minus))
    S = np.hstack((U, e_universum))
    e_universum = np.ones((S.shape[0], 1))
    R_universum = np.zeros((S.shape[0], 1))

    QQ1 = np.dot(Q.T, Q) + eps2 * np.eye(Q.shape[1]) + reg_term * np.eye(Q.shape[1])

    PS = np.vstack([P, -S])
    QQ_inv = linalg.solve(QQ1, PS.T)
    kerQ = np.dot(PS, QQ_inv)
    kerQ = (kerQ + kerQ.T) / 2

    z_dim = kerQ.shape[0]
    tilde_P = kerQ
    tilde_q = np.hstack([(e_plus.T - R_plus.T), ((EPS - 1) * e_universum.T - R_universum.T)]).T

    G_neg = -np.eye(z_dim)
    h_neg = np.zeros(z_dim)
    G_pos = np.eye(z_dim)
    h_pos = np.hstack([d1 * np.ones(P.shape[0]), du * np.ones(S.shape[0])])
    G_combined = np.vstack([G_neg, G_pos])
    h_combined = np.hstack([h_neg, h_pos])

    solvers.options['show_progress'] = False
    solution = solvers.qp(matrix(tilde_P, tc='d'), matrix(tilde_q, tc='d'), matrix(G_combined, tc='d'), matrix(h_combined, tc='d'))

    z_sol = np.array(solution['x']).flatten()
    lamb = z_sol[:P.shape[0]]
    nu = z_sol[P.shape[0]:]

    PS = np.vstack([P, -S])
    w_b_minus = PS.T @ z_sol
    w_b_minus = linalg.solve(np.dot(Q.T, Q) + eps2 * np.eye(Q.shape[1]) + reg_term * np.eye(Q.shape[1]), w_b_minus)
    w_minus = w_b_minus[:-1]
    b_minus = w_b_minus[-1]

    P_1 = TestX[:, :-1]
    y1 = np.dot(P_1, w_plus) + b_plus
    y2 = np.dot(P_1, w_minus) + b_minus

    Predict_Y = np.zeros((y1.shape[0], 1))
    for i in range(y1.shape[0]):
        if np.min([np.abs(y1[i]), np.abs(y2[i])]) == np.abs(y1[i]):
            Predict_Y[i] = 1
        else:
            Predict_Y[i] = -1

    end = time.time()
    Time = end - start
    return TestX[:, -1], Predict_Y.flatten(), Time