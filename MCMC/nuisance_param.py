# this script runs the MCMC for simulated data that includes an error
# it has the addition of the nuisance parameter, sigma_sys when fitting

import os
import numpy as np
import ephem
import time
import matplotlib.pyplot as plt
import emcee
import corner
import matplotlib.cm as cm
from multiprocessing import Pool
import bean_functions as bf
import plot_beam_reach as bm
import mcmc_funcs as mf
from datetime import datetime

# ----------------- set base directory -----------------
#base_dir = "new_data/noise_param/case4.52"
base_dir = "new_data/sys_error/p_100_shift"
os.makedirs(base_dir, exist_ok=True)

# ----------------- observer -----------------
reach = ephem.Observer()
reach.lat = '-30.84'
reach.lon = '21.38'
reach.elevation = 1151


# ----------------- parameters -----------------
nside = 64
freqs_beam = np.arange(50., 101., 1)
freqs = np.arange(50., 101., 1)
beta_Gal = 2.5
tR = 14
beta_R = 2.6
beta_plane = 2.2
beta_outer = 2.4
A = 0.01  # 1% fractional amplitude (dimensionless)
freq_0 = 408
lst = '2024/02/11 19:09:04.87'

# initial guess for fitted parameters (nuisance_param ~ 1 K)
initial = np.array([beta_Gal, tR, beta_R, beta_plane, beta_outer, 1.0])
nwalkers = 50
ndim = 6
int_time = 1  # hrs
seed_noise = 23
seed_pos = 20

# ----------------- load precomputed data -----------------
radio_excess = np.load(os.path.join(base_dir, "408_radio_excess.npy"))
high_galactic_lat = np.load(os.path.join(base_dir, "408_high_lat.npy"))
low_galactic_lat = np.load(os.path.join(base_dir, "multispix_low_lat.npy"))

precomp_HG = mf.prepare_high_lat_haslam(64, reach, freqs_beam, freq_0, lst, use_beam=True)
precomp_LG = mf.compute_low_latitude_components_multispix(
    64, reach, freqs_beam, lst, threshold=100.0, use_beam=True
)

# ----------------- generate simulated data -----------------
y_tot = high_galactic_lat + radio_excess + low_galactic_lat
np.save(os.path.join(base_dir, "y_tot.npy"), y_tot)
np.random.seed(seed_noise)

# measurement noise (freq dependent)
sigma = y_tot / np.sqrt(1e6 * 3600 * int_time)  # units: K
noise = np.random.normal(0.0, sigma)
print(f"noise at 50 MHz (one realisation): {noise[0]:.6f} K")

# --------------------------- injected systematic --------------------
# A is fractional, so (A * y_tot) has units of K
noise_A = (A * y_tot) * (np.sin((2 * np.pi) * (freqs+50) / 100))#50)

# compute rms of injected sinusoid (K)
rms_sys = np.sqrt(np.mean(noise_A**2))
print(f"rms of injected sinusoid (A = {A}): {rms_sys:.6f} K")

# save injected noise array and rms
np.save(os.path.join(base_dir, "noise_A_injected.npy"), noise_A)
with open(os.path.join(base_dir, "injected_systematic_info.txt"), "w") as f:
    f.write(f"A (fractional) = {A}\n")
    f.write(f"injected sinusoid rms [K] = {rms_sys}\n")

# plot the injected sinusoidal error with units
plt.figure(figsize=(8, 4))
plt.plot(freqs, noise_A)
plt.axhline(0, color="k", linestyle="--", linewidth=1)
plt.xlabel("Frequency (MHz)")
plt.ylabel("Systematic error amplitude (K)")
plt.savefig(os.path.join(base_dir, "sinusoidal_injected_error.png"), dpi=300, bbox_inches="tight")
plt.show()
plt.close()

# combine to form simulated observed data
y = y_tot + noise + noise_A

# ----------------- log-likelihood (with nuisance_param) -----------------
def log_likelihood(theta, y, freqs):
    beta_Gal, tR, beta_R, beta_plane, beta_outer, nuisance_param = theta

    model = (
        mf.compute_high_lat_haslam(precomp_HG, tR, beta_R, beta_Gal)
        + mf.compute_radio_excess(freqs, tR, 408., beta_R)
        + mf.compute_low_latitude_multispix(freqs, beta_plane, beta_outer, precomp_LG, use_beam=True)
    )

    # include nuisance noise term in quadrature with sigma
    sigma2 = sigma**2 + nuisance_param**2
    return -0.5 * np.sum((y - model) ** 2 / sigma2 + np.log(sigma2))

# ----------------- prior -----------------
def log_prior(theta):
    beta_Gal, tR, beta_R, beta_plane, beta_outer, nuisance_param = theta
    if (
        0.0 < beta_Gal < 3.0
        and 0 < tR < 50
        and 2.0 < beta_R < 3.0
        and 1.0 < beta_plane < 3.0
        and 2.0 < beta_outer < 3.0
        and 0 < nuisance_param < 400
    ):
        return 0.0
    return -np.inf

# ----------------- full probability -----------------
def log_probability(theta, y, freqs):
    lp = log_prior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(theta, y, freqs)

# ----------------- run MCMC -----------------
np.random.seed(seed_pos)
pos = initial + 1e-2 * np.random.randn(nwalkers, ndim)
pos[:, -1] = np.abs(pos[:, -1]) + 1e-6
print("initial walker positions sample (first 5):")
print(pos[:5])

n_steps = 100000

start_time = time.time()
with Pool(30) as pool:
    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability, args=(y, freqs), pool=pool)
    sampler.run_mcmc(pos, n_steps, progress=True)

# ----------------- save data -----------------
chain = sampler.get_chain()
np.save(os.path.join(base_dir, "chain.npy"), chain)
np.save(os.path.join(base_dir, "noise.npy"), noise)

# ----------------- plot MCMC chains -----------------
labels = ["beta_Gal", "tR", "beta_R", "beta_plane", "beta_outer", "nuisance_param"]
fig, axes = plt.subplots(len(labels), figsize=(10, 7), sharex=True)
colors = [cm.viridis(x) for x in np.linspace(0.1, 0.9, nwalkers)]
for i in range(ndim):
    for j in range(nwalkers):
        axes[i].plot(chain[:, j, i], color=colors[j], alpha=0.4)
    axes[i].set_ylabel(labels[i])
axes[-1].set_xlabel("Step")
plt.tight_layout()
plt.savefig(os.path.join(base_dir, "mcmc_chains.png"), dpi=300, bbox_inches="tight")
plt.close()

# ----------------- log output -----------------
elapsed_time = time.time() - start_time
log_path = os.path.join(base_dir, "output_log.txt")
with open(log_path, "w") as f:
    f.write("==== Simulation Output Log ====\n")
    f.write(f"{datetime.now()}\n")
    f.write("True parameters:\n")
    f.write(f"  beta_Gal = {beta_Gal}\n")
    f.write(f"  tR = {tR}\n")
    f.write(f"  beta_R = {beta_R}\n")
    f.write(f"  beta_plane = {beta_plane}\n")
    f.write(f"  beta_outer = {beta_outer}\n")
    f.write(f"  A (fractional) = {A}\n")
    f.write(f"  RMS systematic [K] = {rms_sys}\n\n")

    f.write(f"Initial walker positions (shape {pos.shape}):\n{pos}\n\n")
    f.write(f"Number of steps: {n_steps}\n")
    f.write(f"Total time taken: {elapsed_time:.2f} seconds\n\n")

    logp = log_probability(initial, y, freqs)
    f.write(f"log(probability) at initial = {logp:.3f}\n")
    f.write(f"log(prior) at initial = {log_prior(initial):.3f}\n")
    f.write(f"log(likelihood) at initial = {log_likelihood(initial, y, freqs):.3f}\n")

print(f"Run completed. All outputs saved in {base_dir}")
