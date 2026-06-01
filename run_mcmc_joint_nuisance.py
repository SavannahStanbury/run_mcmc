# this script is a joint-likelihod mcmc, with a nuisance paramter to mitigate errors, and used on simulated data
import os
import numpy as np
import ephem
import matplotlib.pyplot as plt
import emcee
import mcmc_funcs as mf
from datetime import datetime
from multiprocessing import Pool
import matplotlib.cm as cm

# ----------------- directories -----------------
base_dir = "joint_likelihood_nuisance/"
os.makedirs(base_dir, exist_ok=True)

data_dir_1 = "mcmc_fit/seed_17_25"
data_dir_2 = "mcmc_fit/seed_16_25"
data_dir_3 = "mcmc_fit/seed_19_50/error_a"
data_dir_4 = "mcmc_fit/seed_18_30"

# ----------------- observer -----------------
reach = ephem.Observer()
reach.lat = '-30.84'
reach.lon = '21.38'
reach.elevation = 1151

# ----------------- global setup -----------------
thrsh = 25
nside = 64
freqs = np.arange(50., 121., 1)
freq_0 = 408.

# ----------------- LSTs -----------------
lst_1 = '2025/11/06 01:32:38.965461'
lst_2 = '2025/11/06 01:44:31'
lst_3 = '2025/11/06 01:50:00'
lst_4 = '2025/11/06 02:00:00'

# ----------------- true values -----------------
beta_Gal_true = 2.5
tR_true = 14
beta_R_true = 2.6
beta_plane_true = 2.2
beta_outer_true = 2.4

# ----------------- model -----------------
def model(theta, precomp_HG, precomp_LG):
    beta_Gal, tR, beta_R, beta_plane, beta_outer = theta
    high = mf.compute_high_lat_haslam(precomp_HG, tR, beta_R, beta_Gal)
    radio = mf.compute_radio_excess(freqs, tR, freq_0, beta_R)
    low_plane = mf.compute_low_latitude_plane(freqs, beta_plane, precomp_LG, use_beam=True)
    low_outer = mf.compute_low_latitude_outer(freqs, beta_outer, precomp_LG, use_beam=True)
    return high + radio + low_plane + low_outer

# ----------------- load dataset -----------------
def load_dataset(data_dir, lst):
    high_lat = np.load(os.path.join(data_dir, "high_lat.npy"))
    radio_excess = np.load(os.path.join(data_dir, "radio_excess.npy"))
    low_lat_plane = np.load(os.path.join(data_dir, "low_lat_plane.npy"))
    low_lat_outer = np.load(os.path.join(data_dir, "low_lat_outer.npy"))
    noise = np.load(os.path.join(data_dir, "noise.npy"))

    y_true = high_lat + radio_excess + low_lat_plane + low_lat_outer
    y_obs = y_true + noise
    sigma = y_true / np.sqrt(1e6 * 3600)

    precomp_HG = mf.prepare_high_lat_haslam(nside, reach, freqs, freq_0, lst, use_beam=True)
    precomp_LG = mf.compute_low_latitude_components_multispix(nside, reach, freqs, lst, threshold=thrsh, use_beam=True)

    return y_obs, y_true, sigma, precomp_HG, precomp_LG

# ----------------- load all datasets -----------------
y1, y1_true, sigma1, precomp_HG_1, precomp_LG_1 = load_dataset(data_dir_1, lst_1)
y2, y2_true, sigma2, precomp_HG_2, precomp_LG_2 = load_dataset(data_dir_2, lst_2)
y3, y3_true, sigma3, precomp_HG_3, precomp_LG_3 = load_dataset(data_dir_3, lst_3)
y4, y4_true, sigma4, precomp_HG_4, precomp_LG_4 = load_dataset(data_dir_4, lst_4)

sigma1_2 = sigma1**2
sigma2_2 = sigma2**2
sigma3_2 = sigma3**2
sigma4_2 = sigma4**2

# ----------------- likelihood -----------------
def log_likelihood(theta):
    beta_Gal, tR, beta_R, beta_plane, beta_outer, n1, n2, n3, n4 = theta

    model1 = model(theta[:5], precomp_HG_1, precomp_LG_1)
    model2 = model(theta[:5], precomp_HG_2, precomp_LG_2)
    model3 = model(theta[:5], precomp_HG_3, precomp_LG_3)
    model4 = model(theta[:5], precomp_HG_4, precomp_LG_4)

    sigma1_tot = sigma1_2 + n1**2
    sigma2_tot = sigma2_2 + n2**2
    sigma3_tot = sigma3_2 + n3**2
    sigma4_tot = sigma4_2 + n4**2

    logL1 = -0.5 * np.sum((y1 - model1)**2 / sigma1_tot + np.log(sigma1_tot))
    logL2 = -0.5 * np.sum((y2 - model2)**2 / sigma2_tot + np.log(sigma2_tot))
    logL3 = -0.5 * np.sum((y3 - model3)**2 / sigma3_tot + np.log(sigma3_tot))
    logL4 = -0.5 * np.sum((y4 - model4)**2 / sigma4_tot + np.log(sigma4_tot))

    return logL1 + logL2 + logL3 + logL4

# ----------------- prior -----------------
def log_prior(theta):
    beta_Gal, tR, beta_R, beta_plane, beta_outer, n1, n2, n3, n4 = theta

    if (
        1.5 < beta_Gal < 3.5 and
        0 < tR < 30 and
        1.5 < beta_R < 3.5 and
        1.5 < beta_plane < 3.5 and
        1.5 < beta_outer < 3.5 and
        0 < n1 < 150 and
        0 < n2 < 150 and
        0 < n3 < 150 and
        0 < n4 < 150
    ):
        return 0.0

    return -np.inf

# ----------------- posterior -----------------
def log_probability(theta):
    lp = log_prior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(theta)

# ----------------- MCMC setup -----------------
nwalkers = 60
n_steps = 20000
ndim = 9

initial = np.array([2.5, 14, 2.6, 2.2, 2.4, 1.0, 1.0, 1.0, 1.0])

np.random.seed(21)
pos = initial + 1e-2 * np.random.randn(nwalkers, ndim)
pos[:, 5:] = np.abs(pos[:, 5:])

# ----------------- run MCMC -----------------
print("\n===== starting joint nuisance mcmc =====\n")

with Pool(30) as pool:
    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability, pool=pool)
    sampler.run_mcmc(pos, n_steps, progress=True)

# ----------------- save chain -----------------
chain = sampler.get_chain()
np.save(os.path.join(base_dir, "chain.npy"), chain)

# ----------------- plot chains -----------------
labels = ["beta_Gal", "tR", "beta_R", "beta_plane", "beta_outer", "n1", "n2", "n3", "n4"]

fig, axes = plt.subplots(ndim, 1, figsize=(10, 2 * ndim), sharex=True)

colors = [cm.viridis(i) for i in np.linspace(0.1, 0.9, nwalkers)]

for i in range(ndim):
    for j in range(nwalkers):
        axes[i].plot(chain[:, j, i], color=colors[j], alpha=0.3)
    axes[i].set_ylabel(labels[i])

axes[-1].set_xlabel("step")

plt.tight_layout()
plt.savefig(os.path.join(base_dir, "chains.png"), dpi=300)
plt.close()

# ----------------- results -----------------
flat = sampler.get_chain(discard=2000, thin=10, flat=True)
best = np.median(flat, axis=0)

print("\n===== best fit =====\n")
for i in range(ndim):
    print(labels[i], best[i])

print("\nrun complete")