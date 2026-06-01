# This script will load multiple data sets and fit using a joint likelihood mcmc. The data sets and their noise should be precomputed from the single likelihood script.
# Note that the date and integration time for each data set should be changed. 

import os
import numpy as np
import ephem
import matplotlib.pyplot as plt
import emcee
import corner
import mcmc_funcs as mf
from datetime import datetime
from multiprocessing import Pool
import matplotlib.cm as cm

# ----------------- directories -----------------
base_dir = "joint_likelihood/"

data_dir_lst6 = "mcmc_fit/seed_17_25"
data_dir_lst62 = "mcmc_fit/seed_16_25"
#data_dir_lst6_dd = "mcmc_fit/seed_15_25"

os.makedirs(base_dir, exist_ok=True)

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
lst_6 = '2025/11/06 01:32:38.965461'
lst_62 = '2025/11/06 01:44:31'
#lst_6_dd = '2025/11/07 01:28:30'

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

    model_tot = high + radio + low_plane + low_outer

    return model_tot, high, radio, low_plane, low_outer

def model_l(theta, precomp_HG, precomp_LG):

    beta_Gal, tR, beta_R, beta_plane, beta_outer = theta

    model_tot = mf.compute_high_lat_haslam(precomp_HG, tR, beta_R, beta_Gal) + mf.compute_radio_excess(freqs, tR, freq_0, beta_R) + mf.compute_low_latitude_plane(freqs, beta_plane, precomp_LG, use_beam=True) + mf.compute_low_latitude_outer(freqs, beta_outer, precomp_LG, use_beam=True)

    return model_tot

# ----------------- load dataset -----------------
def load_dataset(data_dir, lst):

    print(f"\nloading dataset: {data_dir}")

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

# ----------------- load both datasets -----------------
y1, y1_true, sigma1, precomp_HG_1, precomp_LG_1 = load_dataset(data_dir_lst6, lst_6)

y2, y2_true, sigma2, precomp_HG_2, precomp_LG_2 = load_dataset(data_dir_lst62, lst_62)

#y2, y2_true, sigma2, precomp_HG_2, precomp_LG_2 = load_dataset(data_dir_lst6_dd, lst_6_dd)

sigma1_2 = sigma1**2
sigma2_2 = sigma2**2

# ----------------- MCMC setup -----------------
nwalkers = 50
n_steps = 20000
ndim = 5

# ----------------- likelihood -----------------
def log_likelihood(theta):

    model1 = model_l(theta, precomp_HG_1, precomp_LG_1)

    model2 = model_l(theta, precomp_HG_2, precomp_LG_2)

    logL1 = -0.5 * np.sum((y1 - model1)**2 / sigma1_2 + np.log(sigma1_2))

    logL2 = -0.5 * np.sum((y2 - model2)**2 / sigma2_2 + np.log(sigma2_2))

    return logL1 + logL2

# ----------------- prior -----------------
def log_prior(theta):

    beta_Gal, tR, beta_R, beta_plane, beta_outer = theta

    if 1.5 < beta_Gal < 3.5 and 0 < tR < 30 and 1.5 < beta_R < 3.5 and 1.5 < beta_plane < 3.5 and 1.5 < beta_outer < 3.5:
    #if 2.0 < beta_Gal < 3.0 and 0 < tR < 30 and 2.0 < beta_R < 3.0 and 2.0 < beta_plane < 3.0 and 2.0 < beta_outer < 3.0:
        return 0.0

    return -np.inf

# ----------------- posterior -----------------
def log_probability(theta):

    lp = log_prior(theta)

    if not np.isfinite(lp):
        return -np.inf

    return lp + log_likelihood(theta)

# ----------------- initial positions -----------------
initial = np.array([2.5, 14, 2.6, 2.2, 2.4])

np.random.seed(20)

pos = initial + 1e-2 * np.random.randn(nwalkers, ndim)

# ----------------- run MCMC -----------------
print("\n================ starting mcmc ================\n")

with Pool(30) as pool:

    sampler = emcee.EnsembleSampler(
        nwalkers,
        ndim,
        log_probability,
        pool=pool
    )

    sampler.run_mcmc(pos, n_steps, progress=True)

# ----------------- save chain -----------------
chain = sampler.get_chain()

np.save(os.path.join(base_dir, "chain.npy"), chain)

print(f"chain shape: {chain.shape}")

# ----------------- labels -----------------
labels = ["beta_Gal", "tR", "beta_R", "beta_plane", "beta_outer"]

# add nuisance labels automatically
for i in range(ndim - 5):
    labels.append(f"nuisance_{i}")

# ----------------- plot chains -----------------
fig, axes = plt.subplots(ndim, 1, figsize=(10, 2 * ndim), sharex=True)

# make axes iterable if ndim=1
if ndim == 1:
    axes = [axes]

colors = [cm.viridis(i) for i in np.linspace(0.1, 0.9, nwalkers)]

for i in range(ndim):

    for j in range(nwalkers):

        axes[i].plot(chain[:, j, i], color=colors[j], alpha=0.3)

    axes[i].set_ylabel(labels[i])

axes[-1].set_xlabel("step")

plt.tight_layout()

plt.savefig(os.path.join(base_dir, "chains_diagnostic.png"), dpi=300)

plt.close()

# ----------------- flatten samples -----------------
flat_samples = sampler.get_chain(discard=2000, thin=10, flat=True)

labels = ["beta_Gal", "tR", "beta_R", "beta_plane", "beta_outer"]

truths = [beta_Gal_true, tR_true, beta_R_true, beta_plane_true, beta_outer_true]

# ----------------- corner plot -----------------
fig = corner.corner(flat_samples, labels=labels, truths=truths)

plt.savefig(os.path.join(base_dir, "corner.png"), dpi=300)

plt.close()

# ----------------- best fit + errors -----------------
best_fit = np.median(flat_samples, axis=0)

lower = np.percentile(flat_samples, 16, axis=0)

upper = np.percentile(flat_samples, 84, axis=0)

err_low = best_fit - lower

err_high = upper - best_fit

print("\n===== best fit parameters =====\n")

for i in range(ndim):

    print(f"{labels[i]}")
    print(f"true   = {truths[i]}")
    print(f"fit    = {best_fit[i]:.6f}")
    print(f"+err   = {err_high[i]:.6f}")
    print(f"-err   = {err_low[i]:.6f}")
    print("")

# ----------------- best-fit models -----------------
model_best_1, high1, radio1, plane1, outer1 = model(best_fit, precomp_HG_1, precomp_LG_1)

model_best_2, high2, radio2, plane2, outer2 = model(best_fit, precomp_HG_2, precomp_LG_2)

# ----------------- plot dataset 1 -----------------
plt.figure(figsize=(8, 5))

plt.plot(freqs, y1, label="Observed", color="k")

plt.plot(freqs, y1_true, label="True", linestyle="--")

plt.plot(freqs, model_best_1, label="Best Fit", color="red")

plt.xlabel("Frequency (MHz)")

plt.ylabel("Temperature (K)")

plt.legend()

plt.grid()

plt.savefig(os.path.join(base_dir, "fit_lst6.png"), dpi=300)

plt.close()

# ----------------- plot dataset 2 -----------------
plt.figure(figsize=(8, 5))

plt.plot(freqs, y2, label="Observed", color="k")

plt.plot(freqs, y2_true, label="True", linestyle="--")

plt.plot(freqs, model_best_2, label="Best Fit", color="red")

plt.xlabel("Frequency (MHz)")

plt.ylabel("Temperature (K)")

plt.legend()

plt.grid()

plt.savefig(os.path.join(base_dir, "fit_lst62.png"), dpi=300)

plt.close()

# ----------------- log file -----------------
with open(os.path.join(base_dir, "output_log.txt"), "w") as f:

    f.write("==== JOINT MCMC RUN LOG ====\n")
    f.write(f"{datetime.now()}\n\n")

    f.write("Datasets:\n")
    f.write(f"LST_6  = {lst_6}\n")
    f.write(f"LST_62 = {lst_6_dd}\n\n")

    f.write("TRUE PARAMETERS:\n")
    f.write(f"beta_Gal = {beta_Gal_true}\n")
    f.write(f"tR = {tR_true}\n")
    f.write(f"beta_R = {beta_R_true}\n")
    f.write(f"beta_plane = {beta_plane_true}\n")
    f.write(f"beta_outer = {beta_outer_true}\n\n")

    f.write("BEST FIT PARAMETERS:\n")

    for i in range(ndim):

        f.write(f"{labels[i]}\n")
        f.write(f"true   = {truths[i]}\n")
        f.write(f"fit    = {best_fit[i]:.6f}\n")
        f.write(f"+err   = {err_high[i]:.6f}\n")
        f.write(f"-err   = {err_low[i]:.6f}\n\n")

    f.write(f"Mean acceptance fraction = {np.mean(sampler.acceptance_fraction):.5f}\n")

print("Joint MCMC complete.")

corner