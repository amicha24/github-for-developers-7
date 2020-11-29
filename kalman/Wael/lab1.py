import numpy as np
import matplotlib.pyplot as plt


def simulate_motion_1D(var_q,N,T):
    # Transition matrix
    A = np.array([[1, T],
                  [0, 1]], dtype=float)
    # Vector gain for the process noise
    G = np.array([[T ** 2 / 2],
                  [T]], dtype=float)

    x = np.zeros((2, N))    # states are [position; speed; acceleration
    x[:, 0] = [0, 10]       # state initialization

    # import generated process noise with a given variance var_q
    w = np.genfromtxt(rf"w_var_q={var_q}.csv", delimiter=',')
    # simulate system dynamics
    for ii in range(1, N):
        x[:, ii] = A.dot(x[:, ii - 1]) + G.dot(w[ii]).T

    # import generated process noise with a given variance var_q
    v = np.genfromtxt("measurement_noise.csv", delimiter=',')

    z = x[0, :] + v  # position measurements assuming C = [1 0 ]
    return x,z


data_set1= simulate_motion_1D(0,100,1)
data_set2 = simulate_motion_1D(1,100,1)
data_set3 = simulate_motion_1D(9,100,1)




plt.plot(data_set1[0][0, :], data_set1[0][1, :], label='var_q = 0')
plt.legend(loc='best')
plt.show()

plt.plot(data_set2[0][0, :], data_set2[0][1, :], label='var_q = 1')
plt.legend(loc='best')

plt.show()

plt.plot(data_set3[0][0, :], data_set3[0][1, :], label='var_q = 9')
plt.legend(loc='best')

plt.show()

#
# plt.plot(data_set3[0, :], data_set3[1, :], label='var_q = 1')
# plt.legend(loc='best')
# plt.show()




# def KF(z,N,p0=1, R = 0.01, Q = 10**(-5),x0=0):
#     # initial x and P, add initial value x0 = 0, p0 = 1 to the list
#     X = np.zeros(N)
#     P = np.zeros(N)
#     P[0] = p0
#     X[0] = x0
#     K = np.zeros(N-1)
#
#     for idx, el in enumerate(z):
#         ahead_x = X[idx]
#         P[idx] = P[idx] + Q
#         K[idx] = (P[idx] / (P[idx] + R))
#         X[idx+1] = (ahead_x + K[idx] * (el - ahead_x))
#         P[idx+1] = ((1 - K[idx]) * P[idx])
#     return X, P, K




