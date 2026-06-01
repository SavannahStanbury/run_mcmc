'''
import os
import numpy as np
import matplotlib.pyplot as plt
import ephem
import mcmc_funcs as mf

base_root = "mcmc_fit/seed_19_50/"

datasets = {
    "1.95 K": {"dir": os.path.join(base_root, "error_b"), "color": "#4A90E2", "ls": "--"},
    "19.55 K": {"dir": os.path.join(base_root, "error_a"), "color": "#C80470", "ls": "-"},
    "97.75 K": {"dir": os.path.join(base_root, "error_d"), "color": "#F4A300", "ls": "-."},
    "195.5 K": {"dir": os.path.join(base_root, "error_c"), "color": "#2E8B57", "ls": ":"}
}

out_dir = os.path.join(base_root, "comparison_residuals")
os.makedirs(out_dir, exist_ok=True)

thrsh = 50.0

reach = ephem.Observer()
reach.lat = '-30.84'
reach.lon = '21.38'
reach.elevation = 1151
reach.date = '2024/02/11 15:09:44.19'

nside = 64
freq_0 = 408
lst = '2024/02/11 19:09:04.87'

freqs = np.arange(50, 121, 1)

precomp_HG = mf.prepare_high_lat_haslam(nside, reach, freqs, freq_0, lst, use_beam=True)

precomp_LG = mf.compute_low_latitude_components_multispix(nside, reach, freqs, lst, threshold=thrsh, use_beam=True)

def model(theta):

    theta = np.asarray(theta).flatten()

    beta_Gal, tR, beta_R, beta_plane, beta_outer = theta[:5]

    high = mf.compute_high_lat_haslam(precomp_HG, tR, beta_R, beta_Gal)

    radio = mf.compute_radio_excess(freqs, tR, freq_0, beta_R)

    low_outer = mf.compute_low_latitude_outer(freqs, beta_outer, precomp_LG, use_beam=True)

    low_plane = mf.compute_low_latitude_plane(freqs, beta_plane, precomp_LG, use_beam=True)

    return high + radio + low_outer + low_plane

def reconstruct(base_dir):

    chain = np.load(os.path.join(base_dir, "chain.npy"))

    nwalkers, nsteps, ndim = chain.shape

    burn_in = int(0.3 * nsteps)

    flat = chain[:, burn_in:, :].reshape(-1, ndim)

    n_draws = 500

    idx = np.random.choice(len(flat), size=n_draws, replace=False)

    spectra = np.array([model(flat[i]) for i in idx])

    recon = np.median(spectra, axis=0)

    high_lat = np.load(os.path.join(base_dir, "high_lat.npy"))
    low_outer = np.load(os.path.join(base_dir, "low_lat_outer.npy"))
    low_plane = np.load(os.path.join(base_dir, "low_lat_plane.npy"))
    radio_excess = np.load(os.path.join(base_dir, "radio_excess.npy"))

    sim = high_lat + low_outer + low_plane + radio_excess

    percent_residuals = 100 * (sim - recon) / recon

    return sim, recon, percent_residuals

results = {}

for label, info in datasets.items():

    sim, recon, residuals = reconstruct(info["dir"])

    results[label] = {
        "sim": sim,
        "recon": recon,
        "residuals": residuals,
        "color": info["color"],
        "ls": info["ls"]
    }

def make_plot(with_legend=True):

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 7), sharex=True, gridspec_kw={'height_ratios': [3, 1]})

    first_key = list(results.keys())[0]

    ax1.plot(freqs, results[first_key]["sim"], color="#BDBDBD", linewidth=2.5, label="Simulated")

    for label, result in results.items():

        ax1.plot(freqs, result["recon"], linewidth=1.7, linestyle=result["ls"], color=result["color"], label=label)

    ax1.set_ylabel("T (K)")
    ax1.set_ylim(0, 7000)

    if with_legend:
        ax1.legend(frameon=False)

    for label, result in results.items():

        ax2.plot(freqs, result["residuals"], linewidth=1.5, linestyle=result["ls"], color=result["color"], label=label)

    ax2.axhline(0, color="black", linestyle="--", linewidth=1)

    ax2.set_xlabel(r"$\nu$ (MHz)")
    ax2.set_ylabel("Residual (%)")

    tag = "with_legend" if with_legend else "no_legend"

    path = os.path.join(out_dir, f"comparison_plot_{tag}.png")

    plt.savefig(path, dpi=300, bbox_inches="tight")

    plt.close()

    print(f"saved: {path}")

make_plot(with_legend=True)
make_plot(with_legend=False)
'''
import os
import numpy as np
import matplotlib.pyplot as plt
import ephem
import mcmc_funcs as mf

import os
import numpy as np
import matplotlib.pyplot as plt
import ephem
import mcmc_funcs as mf

base_root = "mcmc_fit/"

datasets = {
    "threshold 25": {"dir": os.path.join(base_root, "seed_19_25/no_error"), "color": "#CE659F", "ls": "--", "thrsh": 25.0},
    "threshold 100": {"dir": os.path.join(base_root, "seed_19_100/"), "color": "#8923A2", "ls": "-", "thrsh": 100.0}
}

out_dir = os.path.join(base_root, "comparison_residuals")
os.makedirs(out_dir, exist_ok=True)

reach = ephem.Observer()
reach.lat = "-30.84"
reach.lon = "21.38"
reach.elevation = 1151
reach.date = "2024/02/11 15:09:44.19"

nside = 64
freq_0 = 408
lst = "2024/02/11 19:09:04.87"

freqs = np.arange(50, 121, 1)

precomp_HG = mf.prepare_high_lat_haslam(nside, reach, freqs, freq_0, lst, use_beam=True)

def model(theta, precomp_LG):
    theta = np.asarray(theta).flatten()
    beta_Gal, tR, beta_R, beta_plane, beta_outer = theta[:5]
    high = mf.compute_high_lat_haslam(precomp_HG, tR, beta_R, beta_Gal)
    radio = mf.compute_radio_excess(freqs, tR, freq_0, beta_R)
    low_outer = mf.compute_low_latitude_outer(freqs, beta_outer, precomp_LG, use_beam=True)
    low_plane = mf.compute_low_latitude_plane(freqs, beta_plane, precomp_LG, use_beam=True)
    return high + radio + low_outer + low_plane

def reconstruct(base_dir, thrsh):
    chain = np.load(os.path.join(base_dir, "chain.npy"))
    nwalkers, nsteps, ndim = chain.shape
    burn_in = int(0.3 * nsteps)
    flat = chain[:, burn_in:, :].reshape(-1, ndim)
    n_draws = 500
    idx = np.random.choice(len(flat), size=n_draws, replace=False)
    precomp_LG = mf.compute_low_latitude_components_multispix(nside, reach, freqs, lst, threshold=thrsh, use_beam=True)
    spectra = np.array([model(flat[i], precomp_LG) for i in idx])
    recon = np.median(spectra, axis=0)
    high_lat = np.load(os.path.join(base_dir, "high_lat.npy"))
    low_outer = np.load(os.path.join(base_dir, "low_lat_outer.npy"))
    low_plane = np.load(os.path.join(base_dir, "low_lat_plane.npy"))
    radio_excess = np.load(os.path.join(base_dir, "radio_excess.npy"))
    sim = high_lat + low_outer + low_plane + radio_excess
    percent_residuals = 100 * (sim - recon) / recon
    return sim, recon, percent_residuals

results = {}

for label, info in datasets.items():
    sim, recon, residuals = reconstruct(info["dir"], info["thrsh"])
    results[label] = {"sim": sim, "recon": recon, "residuals": residuals, "color": info["color"], "ls": info["ls"]}

def make_plot(with_legend=True):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 7), sharex=True, gridspec_kw={"height_ratios": [3, 1]})
    first_key = list(results.keys())[0]
    ax1.plot(freqs, results[first_key]["sim"], color="#BDBDBD", linewidth=2.5, label="Simulated")
    for label, result in results.items():
        ax1.plot(freqs, result["recon"], linewidth=1.7, linestyle=result["ls"], color=result["color"], label=label)
    ax1.set_ylabel("T (K)")
    ax1.set_ylim(0, 7000)
    if with_legend:
        ax1.legend(frameon=False)
    for label, result in results.items():
        ax2.plot(freqs, result["residuals"], linewidth=1.5, linestyle=result["ls"], color=result["color"], label=label)
    ax2.axhline(0, color="black", linestyle="--", linewidth=1)
    ax2.set_xlabel(r"$\nu$ (MHz)")
    ax2.set_ylabel("Residual (%)")
    tag = "with_legend" if with_legend else "no_legend"
    path = os.path.join(out_dir, f"comparison_plot_{tag}.png")
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"saved: {path}")

make_plot(with_legend=True)
make_plot(with_legend=False)