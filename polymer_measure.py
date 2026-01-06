"""
Additional functions to analyze the Monte Carlo simulation results
"""

from numba import jit
import numpy as np

# ---------- Measure the order parameter ----------

def height(polymer):
    return np.mean(polymer[1:-1,-1])

# ---------- Measure the angles ----------

def closest_two_values(arr, num):
    differences = np.abs(arr - num)
    closest_idx = np.argsort(differences)[:3]
    if differences[closest_idx[0]]==0:
        indices = closest_idx[1:]
    else:
        indices = closest_idx[:2]
    return indices

def angles(polymer):
    langle = np.arctan(polymer[1,1]/polymer[1,0])
    rangle = np.arctan((polymer[-2,1]-polymer[-1,1])/(polymer[-1,0]-polymer[-2,0]))
    midpt = (polymer[-1,0]-polymer[0,0])/2
    middle = closest_two_values(polymer[:,0], midpt)
    mangle = np.arctan((polymer[middle[1],1]-polymer[middle[0],1])/(polymer[middle[1],0]-polymer[middle[0],0]))
    return langle, mangle, rangle

# ---------- Measure the virial stress ----------

@jit(nopython=True)
def virialstress(polymer):
    sigma_xx = np.zeros((N+1, N+1))
    diff = polymer[1:, :] - polymer[:-1, :]
    normdiff = np.zeros(N)
    unitdiff = np.zeros_like(diff)
    for i in range(N):
        normdiff[i] = np.linalg.norm(diff[i, :])
        unitdiff[i, :] = diff[i, :] / normdiff[i]
    cos = np.sum(unitdiff[1:, :] * unitdiff[:-1, :],axis=1)
    diff2 = polymer[2:, :] - polymer[:-2, :]
    for k in range(1,N):
        sigma_xx[k+1,k-1] += -(1/2) * diff2[k-1,0] * kb * diff2[k-1,0] / (normdiff[k-1] * normdiff[k])
        sigma_xx[k-1,k+1] += -(1/2) * diff2[k-1,0] * kb * diff2[k-1,0] / (normdiff[k-1] * normdiff[k])
        sigma_xx[k,k-1] += (1/2) * (-diff[k-1,0] * (-ks * (normdiff[k-1]-l) * unitdiff[k-1,0])\
                            + diff[k-1,0] * kb * (1/normdiff[k]+cos[k-1]/normdiff[k-1]) * unitdiff[k-1,0])
        sigma_xx[k-1,k] += (1/2) * (-diff[k-1,0] * (-ks * (normdiff[k-1]-l) * unitdiff[k-1,0])\
                            + diff[k-1,0] * kb * (1/normdiff[k]+cos[k-1]/normdiff[k-1]) * unitdiff[k-1,0])
        sigma_xx[k,k+1] += (1/2) * diff[k,0] * kb * (1/normdiff[k-1]+cos[k-1]/normdiff[k]) * unitdiff[k,0]
        sigma_xx[k+1,k] += (1/2) * diff[k,0] * kb * (1/normdiff[k-1]+cos[k-1]/normdiff[k]) * unitdiff[k,0]
        if k == N-1:
            sigma_xx[k,k+1] += (1/2) * (-diff[k,0] * (-ks*(normdiff[k]-l) * unitdiff[k,0]))
            sigma_xx[k+1,k] += (1/2) * (-diff[k,0] * (-ks*(normdiff[k]-l) * unitdiff[k,0]))
    return sigma_xx, np.sum(sigma_xx)/(separation)