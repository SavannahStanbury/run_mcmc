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
import mcmc_funcs_RIP as mf

# ----------------- set base directory -----------------
base_dir = "new_data/multi_LST"#LG_one_spix"
os.makedirs(base_dir, exist_ok=True)

# ----------------- observer -----------------
reach = ephem.Observer()
reach.lat = '-30.84'
reach.lon = '21.38'
reach.elevation = 1151
reach.date = '2024/02/11 15:09:44.19'

# ----------------- parameters -----------------
nside = 64

freqs_beam = np.arange(50., 101., 1)
freqs = np.arange(50., 101., 1)
beta_Gal = 2.5
tR = 14
beta_R = 2.6
beta_plane = 2.2
beta_outer = 2.4
freq_0 = 408

#lst = '2024/02/11 21:09:04.87' #lst 8
lst = '2024/02/11 19:09:04.87'

initial = np.array([beta_Gal, tR, beta_R, beta_plane, beta_outer])
nwalkers = 50
int_time = 1 #9  # hrs
seed_noise = 20
seed_pos = 20

# ----------------- frequencies -----------------
freq_0 = 408.
#freqs_beam = np.arange(50., 101., 1)
#freqs =  np.arange(50., 101., 1)
# ----------------- load precomputed data -----------------
'''
radio_excess = np.load(os.path.join(base_dir, "408_radio_excess.npy"))
high_galactic_lat = np.load(os.path.join(base_dir, "408_high_lat.npy"))
#low_galactic_lat = np.load(os.path.join(base_dir, "multispix_low_lat.npy"))
low_galactic_lat_plane = np.load(os.path.join(base_dir, "multispix_low_lat_plane.npy"))
low_galactic_lat_outer = np.load(os.path.join(base_dir, "multispix_low_lat_outer.npy"))
'''

radio_excess = np.load(os.path.join(base_dir, "AVG_radio_excess.npy"))
high_galactic_lat = np.load(os.path.join(base_dir, "AVG_high_lat.npy"))
#low_galactic_lat = np.load(os.path.join(base_dir, "multispix_low_lat.npy"))
low_galactic_lat_plane = np.load(os.path.join(base_dir, "AVG_low_lat_plane.npy"))
low_galactic_lat_outer = np.load(os.path.join(base_dir, "AVG_low_lat_outer.npy"))


precomp_HG = mf.prepare_high_lat_haslam(64, reach, freqs_beam, freq_0, lst, use_beam=True)
precomp_LG = mf.compute_low_latitude_components_multispix(64, reach, freqs_beam, lst, threshold=100.0, use_beam=True)

# ----------------- generate signal with noise -----------------
y_tot = high_galactic_lat + radio_excess + low_galactic_lat_plane + low_galactic_lat_outer
np.random.seed(seed_noise)
sigma = y_tot / np.sqrt(1e6 * 3600 * int_time)
noise = np.random.normal(0.0, sigma)
y = y_tot + noise

# ----------------- log-likelihood and prior -----------------
def log_likelihood(theta, y, freqs):
    beta_Gal, tR, beta_R, beta_plane, beta_outer = theta
    #beta_Gal, tR, beta_R, beta_LG = theta
    model = mf.compute_high_lat_haslam(precomp_HG, tR, beta_R, beta_Gal) + \
            mf.compute_radio_excess(freqs, tR, 408., beta_R) + \
            mf.compute_low_latitude_multispix(freqs, beta_plane, beta_outer, precomp_LG, use_beam=True)
            
    sigma2 = sigma**2
    return -0.5 * np.sum((y - model)**2 / sigma2 + np.log(sigma2))

def log_prior(theta):
    beta_Gal, tR, beta_R, beta_plane, beta_outer = theta
    #beta_Gal, tR, beta_R, beta_LG = theta
    if 2.0 < beta_Gal < 3.0 and 0 < tR < 50 and 2.0 < beta_R < 3.0 and 2.0 < beta_plane < 3.0 and 2.0 < beta_outer < 3.0:
    #if 2.0 < beta_Gal < 3.0 and 0 < tR < 50 and 2.0 < beta_R < 3.0 and 2.0 < beta_LG < 3.0: 
        return 0.0
    return -np.inf

def log_probability(theta, y, freqs):
    lp = log_prior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(theta, y, freqs)

# ----------------- run MCMC -----------------
np.random.seed(seed_pos)
#pos = initial + 1e-2 * np.random.randn(nwalkers, 5)
#ndim = 5
pos = initial + 1e-2 * np.random.randn(nwalkers, 5)
ndim = 5
n_steps = 50000

start_time = time.time()
with Pool(30) as pool:
    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability, args=(y, freqs_beam), pool=pool)
    sampler.run_mcmc(pos, n_steps, progress=True)

# ----------------- save data -----------------
chain = sampler.get_chain()
np.save(os.path.join(base_dir, "chain.npy"), chain)
np.save(os.path.join(base_dir, "noise.npy"), noise)

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

print(f"Run completed. All outputs saved in {base_dir}")

# ----------------- log output -----------------
from datetime import datetime

elapsed_time = time.time() - start_time
log_path = os.path.join(base_dir, "output_log.txt")
with open(log_path, 'w') as f:
    f.write("==== Simulation Output Log ====\n")
    f.write(f"{datetime.now()}\n")
    f.write(f"True parameters:\n")
    f.write(f"  beta_Gal = {beta_Gal}\n")
    f.write(f"  tR = {tR}\n")
    f.write(f"  beta_R = {beta_R}\n")
    #f.write(f"  beta_LG = {beta_LG}\n")
    f.write(f"  beta_plane = {beta_plane}\n")
    f.write(f"  beta_outer = {beta_outer}\n\n")
    
    f.write(f"Initial walker positions (shape {pos.shape}):\n{pos}\n\n")
    f.write(f"Number of steps: {n_steps}\n")
    f.write(f"Total time taken: {elapsed_time:.2f} seconds\n\n")

    f.write("Prior bounds:\n")
    f.write("  beta_Gal: (2.0, 3.0)\n")
    f.write("  tR: (0, 50)\n")
    f.write("  beta_R: (2.0, 3.0)\n")
    f.write("  beta_LG: (2.0, 3.0)\n")
    #f.write("  beta_plane: (2.0, 3.0)\n")
    #f.write("  beta_outer: (2.0, 3.0)\n\n")
    
    f.write(f"Noise (stddev shape {sigma.shape}):\n{noise}\n\n")
    f.write("Sampler acceptance fraction:\n")
    f.write(f"{sampler.acceptance_fraction}\n\n")
    
    logp = log_probability(initial, y, freqs_beam)
    f.write(f"log(probability) at initial = {logp:.3f}\n")
    f.write(f"log(prior) at initial = {log_prior(initial):.3f}\n")
    f.write(f"log(likelihood) at initial = {log_likelihood(initial, y, freqs_beam):.3f}\n")