import numpy as np
mean=0
R = 1
N = 100
# generate 100 the random errors
v = np.random.normal(mean, np.sqrt(R), N)  # measurement noise
np.savetxt("measurement_noise.csv", v, delimiter=',')