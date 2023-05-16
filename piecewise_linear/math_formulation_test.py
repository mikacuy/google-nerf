import numpy as np

def y_x(x, s_i, tau_i):
	ind = np.searchsorted(s_i, x, side="left")
	print(ind)

	s_k = s_i[ind]
	s_kp1 = s_i[ind+1]
	tau_k = tau_i[ind]
	tau_kp1 = tau_i[ind+1]

	numerator = ((tau_kp1 - tau_k)*x + (s_kp1*tau_k - s_k*tau_kp1))**2

	## Increasing case
	if tau_k <= tau_kp1:
		denominator = 2*(s_kp1 - s_k)*(tau_kp1 - tau_k)

	## Decreasing case
	else:
		denominator = 2*(s_kp1 - s_k)*(tau_k - tau_kp1)

	out = numerator/denominator

	return out

def get_ysk_term(u, C_i, s_i, tau_i):
	ind = np.searchsorted(C_i, u, side="left")
	print(ind)

	s_k = s_i[ind]
	s_kp1 = s_i[ind+1]
	tau_k = tau_i[ind]
	tau_kp1 = tau_i[ind+1]

	numerator = (tau_k**2) * (s_kp1 - s_k)

	## Increasing case
	if tau_k <= tau_kp1:
		denominator = 2*(tau_kp1 - tau_k)

	## Decreasing case
	else:
		denominator = 2*(tau_k - tau_kp1)

def get_positive_solution(u, C_i, s_i, tau_i, T_si):
	ind = np.searchsorted(C_i, u, side="left")
	print(ind)

	c_k = C_i[ind]
	s_k = s_i[ind]
	s_kp1 = s_i[ind+1]
	tau_k = tau_i[ind]
	tau_kp1 = tau_i[ind+1]
	T_sk = T_si[ind]

	nume1 = 2*(tau_kp1 - tau_k)*(np.log(T_sk) - np.log(T_sk - (u-c_k)))
	denom1 = (s_kp1 - s_k)

	determinant = tau_k**2 + nume1/denom1

	t = ((s_kp1 - s_k) * (-tau_k + np.sqrt(determinant))) / (tau_kp1 - tau_k)

	x = s_k + t

	return x



###################
near = 0.05
far = 1e10
###################

### Define s_i
# s_i = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
# s_i = np.array([0.25, 0.5, 0.75])

## Concatenate near and far plane
s_i = np.array([near, 0.25, 0.5, 0.75, far])

print("1. s_i")
print(s_i)
print()

### Define tau_i

# This is increasing
# tau_i = np.array([0.01, 0.02, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.9])
# tau_i = np.array([0.1, 0.4, 0.9])

## Concatenate near and far plane
# tau_i = np.array([0.0001, 0.1, 0.4, 0.9, 0.9])
tau_i = np.array([0.0001, 10., 10., 1e-6, 0.0])

print("2. tau_i")
print(tau_i)
print()

### Calculate T(s_i) : Eq 11
dists = s_i[1:] - s_i[:-1]

print("3. s_i - s_{i-1}")
print(dists)
print()

interval_ave_tau = 0.5 * (tau_i[1:] + tau_i[:-1])

print("4. (tau_i + tau_{i-1})/2")
print(interval_ave_tau)
print()

expr = np.exp(-interval_ave_tau*dists)

print("5. exp(-interval_ave_tau * dist)")
print(expr)
print()

T_si = np.cumprod(expr)
### T(s_0) = 1
T_si = np.insert(T_si, 0, 1.0)

print("6. T(s_i)")
print(T_si)
print()


### Calculate PMF : Eq 10
P_si = T_si[ :-1] * (1-expr) 

print("7. P_si")
print(P_si)
print()

### Derive CDF from PMF (cumsum)
C_si = np.cumsum(P_si)
C_si = np.insert(C_si, 0, 0.0)

print("8. C_si")
print(C_si)
print()


###############################
########### TESTS #############
###############################

### 1. Check T(s_k) > (u - c_k) --> for increasing case
print("#######################################")
print("Check if T(s_k) > (u - c_k)")
print()

C_diff = C_si[1:] - C_si[:-1]
print("9. C_diff = c_{k+1} - c_k")
print(C_diff)
print()

T_minus_C_Diff = T_si[:-1] - C_diff
print("10. T(s_k) - C_diff")
print(T_minus_C_Diff)
print()
print("#######################################")




#####################################################
####### This is wrong --> fixed math error ##########
#####################################################

# print("#######################################")
# print("Check if T(s_k) * W > (u - c_k) for decreasing case.")
# print()
# ### 2. Check T(s_k) * W > (u - c_k) --> for decreasing case
# ### W = exp(-\frac{tau_k^2*(s_{k+1}-s_k)}{tau_k-tau_{k+1}})
# nume = tau_i[:-1]**2 * (dists)
# denom = tau_i[:-1] + tau_i[1:]
# W = np.exp(-nume/denom)
# print("11. W")
# print(W)
# print()

# print("12. T(s_k)*W")
# print(W*T_si[:-1])
# print()

# TW_minus_C_Diff = W*T_si[:-1] - C_diff
# print("13. T(s_k)*W - C_diff")
# print(TW_minus_C_Diff)
# print()

### 3. Check solution equivalence for positive and negative cases
### 3.1 Positive

### 3.2 Negative







