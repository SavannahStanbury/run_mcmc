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
save_dir = "erb_pipeline/joint_likelihood/nuisance_LST_6.0"

date1 = "06"
date2 = "07"
date3 = "08"
#date4 = "12"

data_set_1 = os.path.join(base_dir, f"data_{date1}.npy")
data_set_2 = os.path.join(base_dir, f"data_{date2}.npy")
data_set_3 = os.path.join(base_dir, f"data_{date3}.npy")
#data_set_4 = os.path.join(base_dir, f"data_{date4}.npy")

os.makedirs(base_dir, exist_ok=True)
os.makedirs(save_dir, exist_ok=True)

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
nuisance_true = 1.0

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

    return (
        mf.compute_high_lat_haslam(precomp_HG, tR, beta_R, beta_Gal)
        + mf.compute_radio_excess(freqs, tR, freq_0, beta_R)
        + mf.compute_low_latitude_plane(freqs, beta_plane, precomp_LG, use_beam=True)
        + mf.compute_low_latitude_outer(freqs, beta_outer, precomp_LG, use_beam=True)
    )

# ----------------- load data ------------------
def load_data(data_npy, date, lst, int_time):

    print(f"\nloading dataset: {date}")

    y_obs = np.load(data_npy)[:71]
    sigma = y_obs / (np.sqrt(int_time * 1e6))

    precomp_HG = mf.prepare_high_lat_haslam(nside, reach, freqs, freq_0, lst, use_beam=True)
    precomp_LG = mf.compute_low_latitude_components_multispix(nside, reach, freqs, lst, threshold=thrsh, use_beam=True)

    return y_obs, sigma, precomp_HG, precomp_LG

y1, sigma1, precomp_HG_1, precomp_LG_1 = load_data(data_set_1, date1, lst_1, int1)
y2, sigma2, precomp_HG_2, precomp_LG_2 = load_data(data_set_2, date2, lst_2, int2)
y3, sigma3, precomp_HG_3, precomp_LG_3 = load_data(data_set_3, date3, lst_3, int3)

sigma1_2 = sigma1**2
sigma2_2 = sigma2**2
sigma3_2 = sigma3**2

# ----------------- MCMC setup -----------------
nwalkers = 50
n_steps = 20000
ndim = 6

initial = np.array([2.5, 14, 2.6, 2.2, 2.4, 1.0])

np.random.seed(20)
pos = initial + 1e-2 * np.random.randn(nwalkers, ndim)
pos[:, -1] = np.abs(pos[:, -1]) + 1e-6  # nuisance > 0

# ----------------- likelihood -----------------
def log_likelihood(theta):

    beta_Gal, tR, beta_R, beta_plane, beta_outer, nuisance = theta

    model1 = model_l(theta[:-1], precomp_HG_1, precomp_LG_1)
    model2 = model_l(theta[:-1], precomp_HG_2, precomp_LG_2)
    model3 = model_l(theta[:-1], precomp_HG_3, precomp_LG_3)

    sigma1_eff = sigma1_2 + nuisance**2
    sigma2_eff = sigma2_2 + nuisance**2
    sigma3_eff = sigma3_2 + nuisance**2

    logL1 = -0.5 * np.sum((y1 - model1)**2 / sigma1_eff + np.log(sigma1_eff))
    logL2 = -0.5 * np.sum((y2 - model2)**2 / sigma2_eff + np.log(sigma2_eff))
    logL3 = -0.5 * np.sum((y3 - model3)**2 / sigma3_eff + np.log(sigma3_eff))

    return logL1 + logL2 + logL3

# ----------------- prior -----------------
def log_prior(theta):

    beta_Gal, tR, beta_R, beta_plane, beta_outer, nuisance = theta

    if (1.5 < beta_Gal < 3.5 and 0 < tR < 30 and 1.5 < beta_R < 3.5 and 1.5 < beta_plane < 3.5 and 1.5 < beta_outer < 3.5 and 0 < nuisance < 150 ):
        return 0.0

    return -np.inf

# ----------------- posterior -----------------
def log_probability(theta):

    lp = log_prior(theta)
    if not np.isfinite(lp):
        return -np.inf

    return lp + log_likelihood(theta)

# ----------------- run MCMC -----------------
print("\n================ starting mcmc ================\n")

with Pool(30) as pool:
    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability, pool=pool)
    sampler.run_mcmc(pos, n_steps, progress=True)

# ----------------- save chain -----------------
chain = sampler.get_chain()
np.save(os.path.join(save_dir, "chain.npy"), chain)

# ----------------- plot chains -----------------
labels = ["beta_Gal", "tR", "beta_R", "beta_plane", "beta_outer", "nuisance"]

fig, axes = plt.subplots(ndim, 1, figsize=(10, 2 * ndim), sharex=True)

colors = [cm.viridis(i) for i in np.linspace(0.1, 0.9, nwalkers)]

for i in range(ndim):
    for j in range(nwalkers):
        axes[i].plot(chain[:, j, i], color=colors[j], alpha=0.3)
    axes[i].set_ylabel(labels[i])

axes[-1].set_xlabel("step")

plt.tight_layout()
plt.savefig(os.path.join(save_dir, "chains_diagnostic.png"), dpi=300)
plt.close()

# ----------------- corner plot -----------------
flat_samples = sampler.get_chain(discard=2000, thin=10, flat=True)

truths = [
    beta_Gal_true,
    tR_true,
    beta_R_true,
    beta_plane_true,
    beta_outer_true,
    nuisance_true
]

fig = corner.corner(flat_samples, labels=labels, truths=truths)
plt.savefig(os.path.join(save_dir, "corner.png"), dpi=300)
plt.close()

# ----------------- best fit -----------------
best_fit = np.median(flat_samples, axis=0)

lower = np.percentile(flat_samples, 16, axis=0)
upper = np.percentile(flat_samples, 84, axis=0)

err_low = best_fit - lower
err_high = upper - best_fit

print("\n===== best fit parameters =====\n")

for i in range(ndim):
    print(labels[i])
    print(f"true   = {truths[i]}")
    print(f"fit    = {best_fit[i]:.6f}")
    print(f"+err   = {err_high[i]:.6f}")
    print(f"-err   = {err_low[i]:.6f}")
    print("")

# ----------------- best-fit models -----------------
model_best_1, *_ = model(best_fit[:-1], precomp_HG_1, precomp_LG_1)
model_best_2, *_ = model(best_fit[:-1], precomp_HG_2, precomp_LG_2)
model_best_3, *_ = model(best_fit[:-1], precomp_HG_3, precomp_LG_3)

# ----------------- plots -----------------
plt.figure()
plt.plot(freqs, y1, 'k')
plt.plot(freqs, model_best_1, 'r')
plt.savefig(os.path.join(save_dir, "fit1.png"))
plt.close()

plt.figure()
plt.plot(freqs, y2, 'k')
plt.plot(freqs, model_best_2, 'r')
plt.savefig(os.path.join(save_dir, "fit2.png"))
plt.close()

plt.figure()
plt.plot(freqs, y3, 'k')
plt.plot(freqs, model_best_3, 'r')
plt.savefig(os.path.join(save_dir, "fit3.png"))
plt.close()

print("Joint MCMC with nuisance complete.")