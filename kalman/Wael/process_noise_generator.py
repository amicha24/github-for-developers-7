# The aim of generating the noise this way is to be able to refer to the same data
# This file will generate 3 files, each one of them corresponds to
# a given a variance of the process noise
import numpy as np
mean=0
var_q = [0,1,9]
N = 100

# generate 100 random errors for different variances
for i in range(len(var_q)):
    w = np.random.normal(mean, np.sqrt(var_q[i]), N) # process noise
    file_name = fr"w_var_q={var_q[i]}"
    np.savetxt(fr"{file_name}.csv", w, delimiter=',')


