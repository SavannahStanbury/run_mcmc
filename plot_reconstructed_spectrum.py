import os
import numpy as np
import matplotlib.pyplot as plt
import ephem
import mcmc_funcs as mf

# --------------------------------------------------
# directories
# --------------------------------------------------

base_dir = "joint_likelihood/diff_days_final/" #"mcmc_fit/seed_19_25/no_error"
out_dir = os.path.join(base_dir, "residuals")
os.makedirs(out_dir, exist_ok=True)
thrsh = 25

# --------------------------------------------------
# load data
# --------------------------------------------------

freqs = np.arange(50,121,1)
chain = np.load(os.path.join(base_dir, "chain.npy"))

nwalkers, nsteps, ndim = chain.shape

# burn-in + flatten
burn_in = int(0.3 * nsteps)
flat = chain[:, burn_in:, :].reshape(-1, ndim)

# --------------------------------------------------
# observer setup
# --------------------------------------------------

reach = ephem.Observer()
reach.lat = '-30.84'
reach.lon = '21.38'
reach.elevation = 1151
reach.date = '2024/02/11 15:09:44.19'

nside = 64
freq_0 = 408
lst = '2024/02/11 19:09:04.87'

# --------------------------------------------------
# precompute
# --------------------------------------------------

precomp_HG = mf.prepare_high_lat_haslam(
    nside, reach, freqs, freq_0, lst, use_beam=True
)

precomp_LG = mf.compute_low_latitude_components_multispix(
    nside, reach, freqs, lst, threshold=thrsh, use_beam=True
)

# --------------------------------------------------
# model
# --------------------------------------------------

def model(theta):
    theta = np.asarray(theta).flatten()
    beta_Gal, tR, beta_R, beta_plane, beta_outer = theta[:5]

    high = mf.compute_high_lat_haslam(precomp_HG, tR, beta_R, beta_Gal)
    radio = mf.compute_radio_excess(freqs, tR, freq_0, beta_R)
    low_outer = mf.compute_low_latitude_outer(freqs, beta_outer, precomp_LG, use_beam=True)
    low_plane = mf.compute_low_latitude_plane(freqs, beta_plane, precomp_LG, use_beam=True)

    return high + radio + low_outer + low_plane

# --------------------------------------------------
# posterior reconstruction
# --------------------------------------------------

n_draws = 500
idx = np.random.choice(len(flat), size=n_draws, replace=False)

spectra = np.array([model(flat[i]) for i in idx])

recon = np.median(spectra, axis=0)

# --------------------------------------------------
# simulated spectrum
# --------------------------------------------------

high_lat = np.load(os.path.join(base_dir, "high_lat.npy"))
low_outer = np.load(os.path.join(base_dir, "low_lat_outer.npy"))
low_plane = np.load(os.path.join(base_dir, "low_lat_plane.npy"))
radio_excess = np.load(os.path.join(base_dir, "radio_excess.npy"))

sim = high_lat + low_outer + low_plane + radio_excess

np.save(os.path.join(out_dir, "simulated_spectrum.npy"), sim)
np.save(os.path.join(out_dir, "reconstructed_median.npy"), recon)

# --------------------------------------------------
# residuals
# --------------------------------------------------

percent_residuals = 100 * (sim - recon) / recon

# --------------------------------------------------
# plotting function
# --------------------------------------------------

def make_plot(with_legend=True):

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(9, 7), sharex=True,
        gridspec_kw={'height_ratios': [3, 1]}
    )

    # ---------------- top panel ----------------
    ax1.plot(freqs, sim, color="#C3B3B4", linewidth=2.5, label="Simulated")

    ax1.plot(freqs, recon, color="#C80470", linestyle="--", linewidth=1.5, label="Best-fit Posterior")

    ax1.set_ylabel("T (K)")

    ax1.set_ylim(0, 7000)

    if with_legend:
        ax1.legend(frameon=False)

    # ---------------- bottom panel ----------------
    ax2.plot(freqs, percent_residuals, color="#C80470", linewidth=1.5)
    ax2.axhline(0, color="black", linestyle="--", linewidth=1)

    ax2.set_xlabel(r"$\nu$ (MHz)")
    ax2.set_ylabel("Residual (%)")

    # ---------------- save ----------------
    tag = "with_legend" if with_legend else "no_legend"
    path = os.path.join(out_dir, f"residual_plot_{tag}.png")

    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"saved: {path}")

# --------------------------------------------------
# run both versions
# --------------------------------------------------

make_plot(with_legend=True)
make_plot(with_legend=False)