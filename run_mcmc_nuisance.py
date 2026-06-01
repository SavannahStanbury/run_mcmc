import os
import numpy as np
import ephem
import time
import matplotlib.pyplot as plt
import emcee
import matplotlib.cm as cm
from multiprocessing import Pool
import mcmc_funcs as mf
from datetime import datetime

# ------------------ variables ------------------------
thrsh = 50
error = 0.01
seed_noise = 19
n_steps = 40000

# ----------------- set base directory -----------------
base_dir = "mcmc_fit/seed_19_50/error_a"
os.makedirs(base_dir, exist_ok=True)

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

A = error  # fractional systematic
freq_0 = 408.
lst = '2025/11/06 01:32:38.965461'

initial = np.array([beta_Gal, tR, beta_R, beta_plane, beta_outer, 1.0])

nwalkers = 50
ndim = 6
int_time = 1
seed_pos = 40

# ----------------- precompute -----------------
precomp_HG = mf.prepare_high_lat_haslam(nside, reach, freqs, freq_0, lst, use_beam=True)

precomp_LG = mf.compute_low_latitude_components_multispix(
    nside, reach, freqs, lst, threshold=thrsh, use_beam=True
)

# ----------------- forward model -----------------
def model(theta):

    beta_Gal, tR, beta_R, beta_plane, beta_outer = theta[:5]

    high = mf.compute_high_lat_haslam(precomp_HG, tR, beta_R, beta_Gal)

    radio = mf.compute_radio_excess(freqs, tR, freq_0, beta_R)

    low_outer = mf.compute_low_latitude_outer(
        freqs, beta_outer, precomp_LG, use_beam=True
    )

    low_plane = mf.compute_low_latitude_plane(
        freqs, beta_plane, precomp_LG, use_beam=True
    )

    total = high + radio + low_outer + low_plane

    return total, high, radio, low_plane, low_outer

# ----------------- generate data -----------------
y_tot, high_galactic_lat, radio_excess, low_plane, low_outer = model(initial)

# save components
np.save(os.path.join(base_dir, "high_lat.npy"), high_galactic_lat)
np.save(os.path.join(base_dir, "radio_excess.npy"), radio_excess)
np.save(os.path.join(base_dir, "low_lat_plane.npy"), low_plane)
np.save(os.path.join(base_dir, "low_lat_outer.npy"), low_outer)
np.save(os.path.join(base_dir, "y_tot.npy"), y_tot)

# ----------------- generate thermal noise -----------------
np.random.seed(seed_noise)

sigma = y_tot / np.sqrt(1e6 * 3600 * int_time)

noise = np.random.normal(0.0, sigma)

print(f"noise at 50 MHz: {noise[0]:.6f} K")

np.save(os.path.join(base_dir, "noise.npy"), noise)

# ----------------- injected systematic -----------------
noise_A = (A * y_tot) * np.sin((2 * np.pi * freqs) / 25)

rms_sys = np.sqrt(np.mean(noise_A**2))
max_sys = np.max(np.abs(noise_A))

print(f"systematic RMS: {rms_sys:.6f} K")
print(f"systematic max amplitude: {max_sys:.6f} K")

np.save(os.path.join(base_dir, "noise_A_injected.npy"), noise_A)


# ----------------- plot systematic -----------------
plt.figure(figsize=(8, 4))

plt.plot(freqs, noise_A)

plt.axhline(0, color="k", linestyle="--")

plt.xlabel("Frequency (MHz)")
plt.ylabel("Systematic error (K)")

plt.tight_layout()

plt.savefig(os.path.join(base_dir, "sinusoidal_injected_error.png"), dpi=300)

plt.close()

# ----------------- observed data -----------------
y = y_tot + noise + noise_A


# ----------------- RMSE calculation -----------------
# y = simulated data + noise
# y_hat = simulated data + noise + error (or model perturbed version)

y_hat = y # replace this if you have a separate "error-added" dataset
y_noise = y_tot + noise
rmse = np.sqrt(np.mean((y_hat - y_noise)**2))

print(f"RMSE: {rmse:.6e} K")

# ----------------- plot components -----------------
plt.figure(figsize=(10, 6))

plt.plot(freqs, high_galactic_lat, label="High Galactic Lat")
plt.plot(freqs, radio_excess, label="Radio Excess")
plt.plot(freqs, low_plane, label="Low Lat Plane")
plt.plot(freqs, low_outer, label="Low Lat Outer")

plt.plot(freqs, y_tot, label="Total (no noise)", linestyle="--", linewidth=2)

plt.plot(freqs, y, label="Observed", alpha=0.7)

plt.xlabel("Frequency (MHz)")
plt.ylabel("Temperature (K)")

plt.legend()

plt.tight_layout()

plt.savefig(os.path.join(base_dir, "data_components.png"), dpi=300)

plt.close()

# ----------------- likelihood -----------------
def log_likelihood(theta, y, freqs):

    model_tot, _, _, _, _ = model(theta)

    nuisance = theta[-1]

    sigma2 = sigma**2 + nuisance**2

    return -0.5 * np.sum(
        (y - model_tot)**2 / sigma2 + np.log(sigma2)
    )

# ----------------- prior -----------------
def log_prior(theta):

    beta_Gal, tR, beta_R, beta_plane, beta_outer, nuisance = theta

    if (
        2.0 < beta_Gal < 3.0 and
        0 < tR < 50 and
        2.0 < beta_R < 3.0 and
        2.0 < beta_plane < 3.0 and
        2.0 < beta_outer < 3.0 and
        0 < nuisance < 150
    ):
        return 0.0

    return -np.inf

# ----------------- probability -----------------
def log_probability(theta, y, freqs):

    lp = log_prior(theta)

    if not np.isfinite(lp):
        return -np.inf

    return lp + log_likelihood(theta, y, freqs)

# ----------------- run MCMC -----------------
np.random.seed(seed_pos)

pos = initial + 1e-2 * np.random.randn(nwalkers, ndim)

pos[:, -1] = np.abs(pos[:, -1]) + 1e-6



start_time = time.time()

with Pool(30) as pool:

    sampler = emcee.EnsembleSampler(
        nwalkers,
        ndim,
        log_probability,
        args=(y, freqs),
        pool=pool
    )

    sampler.run_mcmc(pos, n_steps, progress=True)

# ----------------- save outputs -----------------
chain = sampler.get_chain()

np.save(os.path.join(base_dir, "chain.npy"), chain)

# ----------------- chain plot -----------------
labels = ["beta_Gal","tR","beta_R","beta_plane","beta_outer","nuisance"]

fig, axes = plt.subplots(len(labels), figsize=(10, 7), sharex=True)

colors = [cm.viridis(x) for x in np.linspace(0.1, 0.9, nwalkers)]

for i in range(ndim):

    for j in range(nwalkers):
        axes[i].plot(chain[:, j, i], color=colors[j], alpha=0.4)

    axes[i].set_ylabel(labels[i])

axes[-1].set_xlabel("Step")

plt.tight_layout()

plt.savefig(os.path.join(base_dir, "mcmc_chains.png"), dpi=300)

plt.close()

# ----------------- logging -----------------
elapsed_time = time.time() - start_time

with open(os.path.join(base_dir, "output_log.txt"), "w") as f:

    f.write("==== Simulation Output Log ====\n")
    f.write(f"{datetime.now()}\n\n")

    f.write("True parameters:\n")
    f.write(f"beta_Gal = {beta_Gal}\n")
    f.write(f"tR = {tR}\n")
    f.write(f"beta_R = {beta_R}\n")
    f.write(f"beta_plane = {beta_plane}\n")
    f.write(f"beta_outer = {beta_outer}\n\n")

    f.write("Systematic info:\n")
    f.write(f"A (fractional) = {A}\n")
    f.write(f"A (percentage) = {A*100:.2f}%\n")
    f.write(f"Max systematic amplitude = {max_sys:.6f} K\n")
    f.write(f"RMS systematic = {rms_sys:.6f} K\n\n")
    f.write(f"RMSE: {rmse:.6e} K\n")

    f.write(f"Steps = {n_steps}\n")
    f.write(f"Runtime = {elapsed_time:.2f} s\n\n")

    f.write("Acceptance fraction:\n")
    f.write(f"{sampler.acceptance_fraction}\n\n")

    logp = log_probability(initial, y, freqs)

    f.write(f"log(prob) at truth = {logp:.3f}\n")
    f.write(f"log(prior) = {log_prior(initial):.3f}\n")
    f.write(f"log(likelihood) = {log_likelihood(initial, y, freqs):.3f}\n")

print(f"Run completed. Outputs saved in {base_dir}")