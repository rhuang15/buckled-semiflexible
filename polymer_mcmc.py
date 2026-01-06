"""
Program to run a Markov chain Monte Carlo simulation for an extensible, semiflexible polymer in a fixed strain ensemble
"""

import sys
from numba import jit
import numpy as np

# ------------- Parameters -------------

N = int(sys.argv[1]) # System size: N+1 nodes, with i = 0 and i = N being fixed
l = 1 # Rest length of each segment
dc = 1 # Codimension
strain = float(sys.argv[2]) # Imposed compressional strain
separation = N * l * (1 - strain) # Fixed endpoint separation

# Set seed for Numba's RNG used by np.random.* inside nopython functions
@jit(nopython=True)
def seed_numba(seed):
    np.random.seed(seed)

rand_seed = int(sys.argv[3]) # Use job array taskid for seed
seed_numba(rand_seed)

kb = 1 # Bending constant
ks = 1000 # Stretching constant

beta = 1000 # Inverse temperature

lstep = float(sys.argv[4]) # Trial step size for local moves
wstep = float(sys.argv[5]) # Trial step size for wave moves
wmodes = int(N/10) # Only use the Fourier modes from 1 to wmodes for wave moves

nsteps = int(float(sys.argv[6])) # Number of MC local + wave step pairs
record = int(float(sys.argv[7])) # Record snapshot after this many step pairs

params = np.array([N, l, dc, strain, rand_seed, kb, ks, beta, lstep, wstep, wmodes, nsteps, record])

# ------------- Initial Conditions -------------

def flat_start():
    polymer = np.zeros((N + 1, dc + 1))
    polymer[:, 0] = np.linspace(0, separation, N + 1)
    return polymer

def sine_start():
    polymer = np.zeros((N + 1, dc + 1))
    polymer[:, 0] = np.linspace(0, separation, N + 1)
    strain_c = (kb/ks) * (np.pi/separation)**2
    polymer[:, -1] = (2/np.pi) * separation * np.sqrt(strain-strain_c) * np.sin((np.pi/separation) * polymer[:, 0])
    return polymer

# ------------- Measure the energy -------------

@jit(nopython=True)
def energy(polymer):
    diff = polymer[1:, :] - polymer[:-1, :]
    Es = (0.5) * ks * np.sum((np.sqrt(np.sum(diff**2, axis=1)) - l)**2)
    for i in range(N):
        diff[i, :] = diff[i, :] / np.linalg.norm(diff[i, :])
    Eb = kb * (N - 1 - np.sum(diff[1:, :] * diff[:-1, :]))
    return Es + Eb

@jit(nopython=True)
def energy_local(polymer, site):
    diff_r = polymer[site + 1, :] - polymer[site, :]
    diff_l = polymer[site, :] - polymer[site - 1, :]
    Es = (0.5) * ks * ((np.linalg.norm(diff_r) - l)**2 + (np.linalg.norm(diff_l) - l)**2)
    unit_r = diff_r / np.linalg.norm(diff_r)
    unit_l = diff_l / np.linalg.norm(diff_l)
    if site != 1 and site != N - 1:
        diff_rr = polymer[site + 2, :] - polymer[site + 1, :]
        unit_rr = diff_rr / np.linalg.norm(diff_rr)
        diff_ll = polymer[site - 1, :] - polymer[site - 2, :]
        unit_ll = diff_ll / np.linalg.norm(diff_ll)
        Eb = kb * (1 - np.dot(unit_ll, unit_l) + 1 - np.dot(unit_l, unit_r) + 1 - np.dot(unit_r, unit_rr))
    elif site == 1:
        diff_rr = polymer[site + 2, :] - polymer[site + 1, :]
        unit_rr = diff_rr / np.linalg.norm(diff_rr)
        Eb = kb * (1 - np.dot(unit_l, unit_r) + 1 - np.dot(unit_r, unit_rr))
    elif site == N-1:
        diff_ll = polymer[site - 1, :] - polymer[site - 2, :]
        unit_ll = diff_ll / np.linalg.norm(diff_ll)
        Eb = kb * (1 - np.dot(unit_ll, unit_l) + 1 - np.dot(unit_l, unit_r))
    return Es + Eb

# ------------- Trial moves -------------

@jit(nopython=True)
def wave_move(polymer, beta):
    yes = 0
    energy_before = energy(polymer)
    nmode = np.random.randint(1, wmodes + 1)
    qmode = (np.pi / separation) * nmode
    amp = np.random.uniform(-wstep, wstep) / (separation * qmode)
    polymer[:, -1] += amp * np.sin(qmode * polymer[:, 0])
    dE = energy(polymer) - energy_before
    if dE <= 0:
        yes += 1
    elif np.exp(-beta * dE) > np.random.rand():
        yes += 1
    else:
        polymer[:, -1] -= amp * np.sin(qmode * polymer[:, 0])
    return nmode, yes

@jit(nopython=True)
def local_move(polymer, beta):
    yes = 0
    site = np.random.randint(1, N)
    energy_before = energy_local(polymer, site)
    trial_move = np.random.uniform(-lstep, lstep, size=dc+1)
    polymer[site, :] += trial_move
    dE = energy_local(polymer, site) - energy_before
    if dE <= 0:
        yes += 1
    elif np.exp(-beta * dE) > np.random.rand():
        yes += 1
    else:
        polymer[site, :] -= trial_move
    return yes

# ------------- Run Markov chain Monte Carlo -------------

if strain > (kb/ks) * (np.pi/separation)**2:
    polymer = sine_start()
else:
    polymer = flat_start()

polymerlist = np.zeros((nsteps // record + 1, np.shape(polymer)[0], np.shape(polymer)[1]))

accepted = np.zeros(wmodes + 1)
attempts = np.zeros(wmodes + 1)

polymerlist[0, :, :] = polymer

for i in range(1, nsteps + 1):
    lyes = local_move(polymer, beta)
    accepted[0] += lyes
    nmode, yes = wave_move(polymer, beta)
    accepted[nmode] += yes
    attempts[nmode] += 1
    if i % record == 0:
        polymerlist[i // record, :, :] = polymer

attempts[0] = nsteps
acceptance = accepted / attempts

np.savez_compressed('../DATA/strain%0.2e/strain%0.2e_taskid%02d.npz'%(strain,strain,rand_seed),\
                    accept=acceptance,attempts=attempts,snapshot=polymerlist,params=params)