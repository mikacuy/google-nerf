import numpy as np

#### Use later ####
near = 0.05
far = 1.0
###################


### Define s_i
s_i = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])

### Define tau_i

# This is increasing
tau_i = np.array([0.01, 0.02, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.9])

### Calculate T(s_i) : Eq 11
dists = s_i[...,1:] - s_i[...,:-1]
interval_ave_tau = 0.5 * (tau_i[...,1:] + tau_i[...,:-1])
expr = np.exp(-interval_ave_tau*dists)

print(dists.shape)
print(interval_ave_tau.shape)
print(expr.shape)

T_si = np.cumprod(expr)

print(T_si.shape)


### Calculate PMF : Eq 10

### Derive CDF from PMF (cumsum)


###############################
########### TESTS #############
###############################

### 1. Check T(s_k) > (u - c_k) --> for increasing case