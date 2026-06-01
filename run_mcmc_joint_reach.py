# This data set loads 3-4 REACH data sets and fits them using a joint-likelihood. 
# Radiometer noise is assumed. 
# Note that the date and integration time need to be changed. 

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
lst_name = "lst_6.0"

base_dir = "erb_pipeline/obs_data_LST_6.0"

date1 = "06"
date2 = "07"
date3 = "08"
#date4 = "12"

data_set_1 = os.path.join(base_dir,f"data_{date1}.npy")
data_set_2 = os.path.join(base_dir,f"data_{date2}.npy")
data_set_3 = os.path.join(base_dir,f"data_{date3}.npy")
#data_set_4 = os.path.join(base_dir,f"data_{date4}.npy")


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

# ----------------- LSTs ----------------
lst_1 = "2025/11/06 01:32:38.96"
lst_2 = "2025/11/07 01:27:00.00"
lst_3 = "2025/11/08 01:23:35.97"
#lst_4 = "2025/11/12 01:07:45.00"

int1 = 48 * 5
int2 = 48 * 4
int3 = 48 * 5
#int4 = 48 * 4

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

# ----------------- laod data ------------------
# TOGGLE THIS ON FOR REAL DATA
def load_data(data_npy, date, lst, int_time):

    print(f"\nloading dataset: {date}")

    y_obs = np.load(data_npy)[:71]

    sigma = y_obs/(np.sqrt(int_time*1e6))

    precomp_HG = mf.prepare_high_lat_haslam(nside, reach, freqs, freq_0, lst, use_beam=True)

    precomp_LG = mf.compute_low_latitude_components_multispix(nside, reach, freqs, lst, threshold=thrsh, use_beam=True)

    return y_obs, sigma, precomp_HG, precomp_LG


y1, sigma1, precomp_HG_1, precomp_LG_1 = load_data(data_set_1, date1, lst_1, int1)

y2, sigma2, precomp_HG_2, precomp_LG_2 = load_data(data_set_2, date2, lst_2, int2)

y3, sigma3, precomp_HG_3, precomp_LG_3 = load_data(data_set_3, date3, lst_3, int3)

#y4, sigma4, precomp_HG_4, precomp_LG_4 = load_data(data_set_4, date4, lst_4, int4)

sigma1_2 = sigma1**2
sigma2_2 = sigma2**2
sigma3_2 = sigma3**2
#sigma4_2 = sigma4**2

# ----------------- MCMC setup -----------------
nwalkers = 50
n_steps = 20000
ndim = 5

# ----------------- likelihood -----------------
def log_likelihood(theta):

    model1 = model_l(theta, precomp_HG_1, precomp_LG_1)

    model2 = model_l(theta, precomp_HG_2, precomp_LG_2)

    model3 = model_l(theta, precomp_HG_3, precomp_LG_3)

    #model4 = model_l(theta, precomp_HG_4, precomp_LG_4)

    logL1 = -0.5 * np.sum((y1 - model1)**2 / sigma1_2 + np.log(sigma1_2))

    logL2 = -0.5 * np.sum((y2 - model2)**2 / sigma2_2 + np.log(sigma2_2))

    logL3 = -0.5 * np.sum((y3 - model3)**2 / sigma3_2 + np.log(sigma3_2))

    #logL4 = -0.5 * np.sum((y4 - model4)**2 / sigma4_2 + np.log(sigma4_2))

    return logL1 + logL2 + logL3 #+ logL4

# ----------------- prior -----------------
def log_prior(theta):

    beta_Gal, tR, beta_R, beta_plane, beta_outer = theta

    if 1.5 < beta_Gal < 3.5 and 0 < tR < 30 and 1.5 < beta_R < 3.5 and 1.5 < beta_plane < 3.5 and 1.5 < beta_outer < 3.5:

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

    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability, pool=pool)

    sampler.run_mcmc(pos, n_steps, progress=True)

# ----------------- save chain -----------------
chain = sampler.get_chain()

np.save(os.path.join(base_dir, "chain.npy"), chain)

print(f"chain shape: {chain.shape}")

# ----------------- labels -----------------
labels = ["beta_Gal", "tR", "beta_R", "beta_plane", "beta_outer"]

for i in range(ndim - 5):

    labels.append(f"nuisance_{i}")

# ----------------- plot chains -----------------
fig, axes = plt.subplots(ndim, 1, figsize=(10, 2 * ndim), sharex=True)

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

model_best_3, high3, radio3, plane3, outer3 = model(best_fit, precomp_HG_3, precomp_LG_3)

#model_best_4, high4, radio4, plane4, outer4 = model(best_fit, precomp_HG_4, precomp_LG_4)

# ----------------- plot dataset 1 -----------------
plt.figure(figsize=(8, 5))

plt.plot(freqs, y1, label="Observed", color="k")

#plt.plot(freqs, y1_true, label="True", linestyle="--")

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

#plt.plot(freqs, y2_true, label="True", linestyle="--")

plt.plot(freqs, model_best_2, label="Best Fit", color="red")

plt.xlabel("Frequency (MHz)")

plt.ylabel("Temperature (K)")

plt.legend()

plt.grid()

plt.savefig(os.path.join(base_dir, "fit_lst62.png"), dpi=300)

plt.close()

# ----------------- plot dataset 3 -----------------
plt.figure(figsize=(8, 5))

plt.plot(freqs, y3, label="Observed", color="k")

#plt.plot(freqs, y3_true, label="True", linestyle="--")

plt.plot(freqs, model_best_3, label="Best Fit", color="red")

plt.xlabel("Frequency (MHz)")

plt.ylabel("Temperature (K)")

plt.legend()

plt.grid()

plt.savefig(os.path.join(base_dir, "fit_lst6_dd.png"), dpi=300)

plt.close()

# ----------------- plot dataset 4 -----------------
plt.figure(figsize=(8, 5))

plt.plot(freqs, y4, label="Observed", color="k")

#plt.plot(freqs, y4_true, label="True", linestyle="--")

#plt.plot(freqs, model_best_4, label="Best Fit", color="red")

plt.xlabel("Frequency (MHz)")

plt.ylabel("Temperature (K)")

plt.legend()

plt.grid()

plt.savefig(os.path.join(base_dir, "fit_lst62_dd.png"), dpi=300)

plt.close()

# ----------------- log file -----------------
with open(os.path.join(base_dir, "output_log.txt"), "w") as f:

    f.write("==== JOINT MCMC RUN LOG ====\n")
    f.write(f"{datetime.now()}\n\n")

    f.write("Datasets:\n")
    f.write(f"LST_6 = {lst_1}\n")
    f.write(f"LST_62 = {lst_2}\n")
    f.write(f"LST_6_DD = {lst_3}\n")
    f.write(f"LST_62_DD = {lst_4}\n\n")

    f.write("TRUE PARAMETERS:\n")
    f.write(f"beta_Gal = {beta_Gal_true}\n")
    f.write(f"tR = {tR_true}\n")
    f.write(f"beta_R = {beta_R_true}\n")
    f.write(f"beta_plane = {beta_plane_true}\n")
    f.write(f"beta_outer = {beta_outer_true}\n\n")

    f.write("BEST FIT PARAMETERS:\n")

    for i in range(ndim):

        f.write(f"{labels[i]}\n")
        f.write(f"true = {truths[i]}\n")
        f.write(f"fit = {best_fit[i]:.6f}\n")
        f.write(f"+err = {err_high[i]:.6f}\n")
        f.write(f"-err = {err_low[i]:.6f}\n\n")

    f.write(f"Mean acceptance fraction = {np.mean(sampler.acceptance_fraction):.5f}\n")

print("Joint MCMC complete.")

