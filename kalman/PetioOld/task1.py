#!/usr/bin/python
'''
This function generates the data for a 1D motion considering a
Piecewise constant, white noise acceleration model, equation (14)

var_a is the variance of the process noise
R is the variance of measurement noise
z is the measured data
x are the true values of the system states

example of use
python gen_data14.py 1e-3 1
'''

import sys
import numpy as np
import matplotlib.pyplot as plt

# var_q = float(sys.argv[1])
# R = float(sys.argv[2])

var_q_list = [0, 1.0, 9.0]
R = 1.0

N = 100 # data size
T = 1.0 # [s] Sampling time interval

def simulate_motion1D(var_q_list, R, N, T):
    states = [] # store x
    process_noise = [] # store w
    measurements = [] # store z
    measurement_noise = [] # store v

    for var_q in var_q_list:
        x = np.zeros((2, N)) # states are [position; speed; acceleration]
        x[:, 0] = [0, 10] # state initialization, change to give your own initial values
        # print(x)

        A = np.array([[1, T],
                      [0, 1]], dtype=float)  # Transition matrix

        G = np.array([[T**2/2],
                      [T]], dtype=float) # Vector gain for the process noise
        w = np.random.normal(0.0, np.sqrt(var_q), N) # process noise
        # testing
        # print(w, 'shape')
        # plt.plot(w)
        # plt.ylabel('some numbers')
        # plt.show()
        # print(np.array([100, 200, 300]))

        # print(G.shape)
        # print(w[1].shape)
        # print(G.dot(w[1]).T.shape)

        for col in range(1, N): # simulate system dynamics
            # here w[ii] is actually a scalar so you are not doing dot product
            # but you are doing vector*scalar instead and then taking the transpose
            x[:, col] = A.dot(x[:, col-1]) + (G.dot(w[col])).T # T is the transpose of this array

        v = np.random.normal(0.0, np.sqrt(R), N) # measurement noise
        z = x[0, :] + v  # position measurements assuming C = [1 0 ]
        #print(z.shape)
        states.append(x)
        process_noise.append(w)
        measurements.append(z)
        measurement_noise.append(v)

    return (states, process_noise, measurements, measurement_noise)

data = simulate_motion1D(var_q_list, R, N, T)
# print(data[2][0])
#print(len(data[0]))

def plot_motion1D(data):
    fig1 = plt.figure()
    # fig, axs = plt.subplots(3, sharex=True, sharey=True, gridspec_kw={'hspace': 0})
    # fig.suptitle('Position x with time')
    states = data[0]
    print(len(states))
    for i in range(3):
        plt.plot(states[i][0,:], states[i][1,:], label='var_q = {i}'.format(i=var_q_list[i]))
        plt.legend(loc='best')
    # axs[0].plot(states[0][0,:], states[0][1,:]) # ahhhhh you plot against the index if no other dimension is given
    # axs[1].plot(x, 0.3 * y, 'o')
    # axs[2].plot(x, y, '+')

    # Hide x labels and tick labels for all but bottom plot.
    # for ax in axs:
    #     ax.label_outer()
    plt.show()

plot_motion1D(data)

# f1 = plt.figure()
# plt.plot(z, label='linear')
# plt.xlabel('Time [s]')
# plt.ylabel('Measured position')
# # plt.show()
#
# f2 = plt.figure()
# plt.plot(x[0,:], label='linear')
# plt.xlabel('Time [s]')
# plt.ylabel('True position [m]')
# # plt.show()
#
# f3 = plt.figure()
# plt.plot(x[1,:], label='linear')
# plt.xlabel('Time [s]')
# plt.ylabel('True speed [m/s]')
# plt.show()


# APPLY THE KALMAN FILTER
# initialization
p_zero = np.array([[R, R/T],
                   [R/T, 2*R/T**2]])
p_zero = np.linalg.inv(p_zero)

# assume that the first predicted value is the average of the
# first two measurements
x_est_zero = (data[2][0][0] + data[2][0][1])*0.5

x_est =np.zeros(2,100)