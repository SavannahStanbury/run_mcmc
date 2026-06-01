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
import reach_beam_script as bm
import mcmc_funcs as mf
from datetime import datetime

# ----------------- set base directory -----------------
base_dir = "erb_pipeline/obs_data_LST_6.0/lst_6.0_median"
os.makedirs(base_dir, exist_ok=True)

# ----------------- data directory (NEW) -----------------
data_dir = "erb_pipeline/obs_data_LST_6.0/"

thrsh = 25
seed_noise = 5

# ----------------- observer -----------------
reach = ephem.Observer()
reach.lat = '-30.84'
reach.lon = '21.38'
reach.elevation = 1151

# ----------------- parameters -----------------
nside = 64
freqs = np.arange(50., 121., 1)

beta_Gal = 2.5
tR = 14
beta_R = 2.6
beta_plane = 2.2
beta_outer = 2.4
freq_0 = 408.

lst = '2025/11/07 01:27:00.00'
reach.date = lst

initial = np.array([beta_Gal, tR, beta_R, beta_plane, beta_outer])

nwalkers = 50
int_time = 0.23

seed_pos = 40

# ----------------- LOAD DATA (REPLACES SIMULATION) -----------------
y = np.load(os.path.join(data_dir, "median_spectrum_lst_6.0.npy"))

# integration time (must match how data was produced)
int_time = 1/6  # hours (same as before)

# radiometer noise reconstruction
# (assumes system equivalent noise scaling used previously)
np.random.seed(seed_noise)
sigma = y / np.sqrt(1e6 * 3600 * int_time)

sigma2 = sigma**2
# ----------------- precompute -----------------
precomp_HG = mf.prepare_high_lat_haslam(nside, reach, freqs, freq_0, lst, use_beam=True)
precomp_LG = mf.compute_low_latitude_components_multispix(nside, reach, freqs, lst, threshold=thrsh, use_beam=True)

# ----------------- define forward model -----------------
def model(theta):
    beta_Gal, tR, beta_R, beta_plane, beta_outer = theta

    high = mf.compute_high_lat_haslam(precomp_HG, tR, beta_R, beta_Gal)
    radio = mf.compute_radio_excess(freqs, tR, freq_0, beta_R)
    low_outer = mf.compute_low_latitude_outer(freqs, beta_outer, precomp_LG, use_beam=True)
    low_plane = mf.compute_low_latitude_plane(freqs, beta_plane, precomp_LG, use_beam=True)

    total = high + radio + low_outer + low_plane
    return total, high, radio, low_plane, low_outer

# ----------------- (OPTIONAL) evaluate model components -----------------
y_tot, high_galactic_lat, radio_excess, low_plane, low_outer = model(initial)

# save components
np.save(os.path.join(base_dir, "high_lat.npy"), high_galactic_lat)
np.save(os.path.join(base_dir, "radio_excess.npy"), radio_excess)
np.save(os.path.join(base_dir, "low_lat_plane.npy"), low_plane)
np.save(os.path.join(base_dir, "low_lat_outer.npy"), low_outer)

# ----------------- plot components -----------------
plt.figure(figsize=(10, 6))
plt.plot(freqs, high_galactic_lat, label="High Galactic Lat")
plt.plot(freqs, radio_excess, label="Radio Excess")
plt.plot(freqs, low_plane, label="Low Lat Plane")
plt.plot(freqs, low_outer, label="Low Lat Outer")
plt.plot(freqs, y_tot, label="Total (model)", linestyle="--", linewidth=2)
plt.plot(freqs, y, label="Observed (loaded)", alpha=0.7)

plt.xlabel("Frequency (MHz)")
plt.ylabel("Temperature (K)")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(base_dir, "data_components.png"), dpi=300)
plt.close()

# ----------------- log-likelihood and prior -----------------
def log_likelihood(theta, y, freqs):
    model_tot, _, _, _, _ = model(theta)
    return -0.5 * np.sum((y - model_tot)**2 / sigma2 + np.log(sigma2))

def log_prior(theta):
    beta_Gal, tR, beta_R, beta_plane, beta_outer = theta
    if (2.0 < beta_Gal < 3.0 and 0 < tR < 50 and 2.0 < beta_R < 3.0 and 2.0 < beta_plane < 3.0 and 2.0 < beta_outer < 3.0):
        return 0.0
    return -np.inf

def log_probability(theta, y, freqs):
    lp = log_prior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(theta, y, freqs)

# ----------------- run MCMC -----------------
np.random.seed(seed_pos)
pos = initial + 1e-2 * np.random.randn(nwalkers, 5)
ndim = 5
n_steps = 20000

start_time = time.time()

with Pool(35) as pool:
    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability, args=(y, freqs), pool=pool)
    sampler.run_mcmc(pos, n_steps, progress=True)

# ----------------- save chain -----------------
chain = sampler.get_chain()
np.save(os.path.join(base_dir, "chain.npy"), chain)

# ----------------- chain plot -----------------
labels = ["beta_Gal", "tR", "beta_R", "beta_plane", "beta_outer"]

fig, axes = plt.subplots(len(labels), figsize=(10, 7), sharex=True)
colors = [cm.viridis(x) for x in np.linspace(0.1, 0.9, nwalkers)]

for i in range(ndim):
    for j in range(nwalkers):
        axes[i].plot(chain[:, j, i], color=colors[j], alpha=0.4)
    axes[i].set_ylabel(labels[i])

axes[-1].set_xlabel("Step")
plt.tight_layout()
plt.savefig(os.path.join(base_dir, "mcmc_chains.png"))
plt.close()

print(f"Run completed. Outputs saved in {base_dir}")

# ----------------- log output -----------------
elapsed_time = time.time() - start_time
log_path = os.path.join(base_dir, "output_log.txt")

with open(log_path, 'w') as f:
    f.write("==== MCMC Output Log ====\n")
    f.write(f"{datetime.now()}\n\n")

    f.write("Initial parameters:\n")
    f.write(f"beta_Gal = {beta_Gal}\n")
    f.write(f"tR = {tR}\n")
    f.write(f"beta_R = {beta_R}\n")
    f.write(f"beta_plane = {beta_plane}\n")
    f.write(f"beta_outer = {beta_outer}\n\n")

    f.write(f"Threshold:{thrsh}\n")
    f.write(f"LST:{lst}\n")
    f.write(f"Integration Time: {int_time}\n\n")

    f.write(f"Number of steps: {n_steps}\n")
    f.write(f"Total time: {elapsed_time:.2f} s\n\n")

    f.write("Acceptance fraction:\n")
    f.write(f"{sampler.acceptance_fraction}\n\n")

    logp = log_probability(initial, y, freqs)
    f.write(f"log(prob) at initial = {logp:.3f}\n")
    f.write(f"log(prior) = {log_prior(initial):.3f}\n")
    f.write(f"log(likelihood) = {log_likelihood(initial, y, freqs):.3f}\n")