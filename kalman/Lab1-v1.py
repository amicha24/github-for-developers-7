import numpy as np
import matplotlib.pyplot as plt

ns = 2 # number of states in the state vector (e.g. x = 2 for position/veloc)
# note even though there are no inputs we assume that there will be as many as the number of states
# they will all be 0 but it's necessary for computational purposes
nu = ns # number of inputs (u) to the system
nm = 1 # number of measurements in the output vector z
R = 1
T = 1

# Transition matrix
A = np.array([[1, T],
              [0, 1]], dtype=float)
# Vector gain for the process noise
G = np.array([[T ** 2 / 2],
              [T]], dtype=float)

C = np.array([1, 0]) #dimensions nm x ns in this case 1 measurement x 2 states = 1 x 2 matrix
H = np.array([1, 0]) #dimensions nm x ns
print(H)

def simulate_motion_1D(var_q, N, T):
    x = np.zeros((ns, N))    # states are [position; speed]
    x[:, 0] = [0, 10]       # state initialization

    # import generated process noise with a given variance var_q
    w = np.genfromtxt(rf"w_var_q={var_q}.csv", delimiter=',')
    # simulate system dynamics
    for ii in range(1, N):
        x[:, ii] = A.dot(x[:, ii - 1]) + G.dot(w[ii]).T

    # import generated process noise with a given variance var_q
    v = np.genfromtxt("measurement_noise.csv", delimiter=',')

    z = x[0, :] + v  # position measurements assuming C = [1 0 ]
    return x, z

# each data set contains the states and measurements
data_set1 = simulate_motion_1D(0, 100, 1) # var = 0
data_set2 = simulate_motion_1D(1, 100, 1) # var = 1
data_set3 = simulate_motion_1D(9, 100, 1) # var = 9


# plt.plot(data_set1[0][0, :], data_set1[0][1, :], label='var_q = 0')
# plt.legend(loc='best')
# plt.show()

# plt.plot(data_set2[0][0, :], data_set2[0][1, :], label='var_q = 1')
# plt.legend(loc='best')
#
# plt.show()
#
# plt.plot(data_set3[0][0, :], data_set3[0][1, :], label='var_q = 9')
# plt.legend(loc='best')
#
# plt.show()

print(len(data_set1[0][1]))

def kalman_filter(measurements, x_est_0, p_est_0, Q_f, R_f):
    N = len(measurements)
    z = measurements
    # create estimated state matrix
    x_est = np.zeros((ns, N), dtype=float)
    x_est[:, 0] = [x_est_0, 10]  # set initial pos and velocity

    # create estimated covariance
    p_est = np.zeros((N,), dtype=object)
    p_est[0] = p_est_0

    # predicted state
    x_pred = np.zeros((ns, N), dtype=float)

    u_k = np.zeros((nu, 1), dtype=float)
    # print(u_k)
    B = np.zeros((ns, nu), dtype=float)

    # predicted covariance
    p_pred = np.zeros((N,), dtype=object)

    # kalman gain
    K = np.zeros((ns,N), dtype=float)

    # R_f matrix- it has dimensions ns x ns
    R_f_mtrx = np.array([[R_f, 0],
                  [0, 0]], dtype=float)

    for ii in range(1,100):
        # print('b', A.dot(x_est[:, ii-1]))
        # x_pred[:, ii] = A.dot(x_est[:, ii-1]) + (B.dot(u_k)).T
        x_pred[:, ii] = A.dot(x_est[:, ii-1])
        #print(x_pred)
        # print(((A.dot(p_est[ii-1])).dot(A.T)).shape)
        # print(((G.dot(Q_f)).dot(G.T))) # makes sense for this to be 0 kuz Q is 0
        # p_pred[ii] = (A.dot(p_est[ii-1])).dot(A.T) + (G.dot(Q_f)).dot(G.T)
        p_pred[ii] = (A.dot(p_est[ii-1])).dot(A.T)

        # print(p_pred[ii])
        # inter = (C.dot(p_pred[ii])).dot(C.T) + (H.dot(R_f_mtrx)).dot(H.T)# intermediate term
        inter = (C.dot(p_pred[ii])).dot(C.T) + np.array(R_f)# intermediate term

        print(1/inter)
        if inter.shape != ():
            K[:,ii] = (p_pred[ii].dot(C.T)).dot(np.linalg.inv(inter))
            print('yo')
        else:
            K[:,ii] = (p_pred[ii].dot(C.T)).dot(1/inter)

        x_est[:,ii] = x_pred[:, ii] + K[:, ii].dot(z[ii]-C.dot(x_pred[:, ii]))
        p_est[ii] = (np.identity(ns) - K[:,ii].dot(C)).dot(p_pred[ii])

    return x_est, p_est
    #
    # print('x est', x_est[:,ii])
    # print('p est', p_est[ii])
    # print('x pred', x_pred[:,ii])
    # print('p pred', p_pred[ii])

# initialization
p_est_zero = np.array([[R, R/T],
                   [R/T, 2*R/T**2]])
p_est_zero = np.linalg.inv(p_est_zero)
print(p_est_zero)
# assume that the first predicted value is the average of the
# first two measurements. Note that we only measure position!!!
measurements = data_set1[0][0]
print('meas', measurements[0])
print('meas', measurements[1])

# print(measurements)
x_est_zero = (measurements[0] + measurements[1])*0.5
# x_est_zero = measurements[0]

# x_est, p_est = kalman_filter(measurements, x_est_zero, p_est_zero, 0, 1)
# measurements = data_set2[0][0]
# x_est, p_est = kalman_filter(measurements, x_est_zero, p_est_zero, 1, 1)

# position vs velocity works ok but weird result
# plt.plot(data_set1[0][0, :], data_set1[0][1, :], label='theoretical - var_q = 0')
# plt.plot(x_est[0, :], x_est[1, :], label='Kalman Estimate - var_q = 0')
#
# plt.legend(loc='best')
# plt.show()

# print(x_est)
# print(p_est)

# only position
# plt.plot(data_set1[0][0, :], label='theoretical - var_q = 0')
# plt.plot(x_est[0, :], label='Kalman Estimate - var_q = 0')
#
# plt.legend(loc='best')
# plt.show()

# only velocity
# plt.plot(data_set1[0][1, :], label='theoretical - var_q = 0')
# plt.plot(x_est[1, :], label='Kalman Estimate - var_q = 0')
#
# plt.legend(loc='best')
# plt.show()

measurements = data_set2[0][0]
x_est, p_est = kalman_filter(measurements, x_est_zero, p_est_zero, 1, 1)

# position vs velocity works ok but weird result
# plt.plot(data_set2[0][0, :], data_set2[0][1, :], label='theoretical - var_q = 1')
# plt.plot(x_est[0, :], x_est[1, :], label='Kalman Estimate - var_q = 1')
#
# plt.legend(loc='best')
# plt.show()

# # only position
# plt.plot(data_set2[0][0, :], label='theoretical - var_q = 1')
# plt.plot(x_est[0, :], label='Kalman Estimate - var_q = 1')
#
# plt.legend(loc='best')
# plt.show()
#
# # only velocity
# plt.plot(data_set2[0][1, :], label='theoretical - var_q = 1')
# plt.plot(x_est[1, :], label='Kalman Estimate - var_q = 1')
#
# plt.legend(loc='best')
# plt.show()

measurements = data_set3[0][0]
x_est, p_est = kalman_filter(measurements, x_est_zero, p_est_zero, 9, 1)

# position vs velocity works ok but weird result
plt.plot(data_set3[0][0, :], data_set2[0][1, :], label='theoretical - var_q = 9')
plt.plot(x_est[0, :], x_est[1, :], label='Kalman Estimate - var_q = 9')

plt.legend(loc='best')
plt.show()

# # only position
plt.plot(data_set3[0][0, :], label='theoretical - var_q = 9')
plt.plot(x_est[0, :], label='Kalman Estimate - var_q = 9')

plt.legend(loc='best')
plt.show()
#
# # only velocity
plt.plot(data_set3[0][1, :], label='theoretical - var_q = 9')
plt.plot(x_est[1, :], label='Kalman Estimate - var_q = 9')

plt.legend(loc='best')
plt.show()