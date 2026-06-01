import numpy as np
import matplotlib.pyplot as plt
import corner
import os

# set directory for data input/output
directory ="erb_pipeline/obs_data_LST_6.0/lst_6.0_median"# "testing_og_code/sub_tR/5_param_fit_with_error/case_4/"#sim_data_v6"
#directory ="joint_likelihood"
# true values for comparison

steps = '50k'
discard = '20k'

freqs_beam = np.arange(50., 121., 1)
beta_Gal = 2.5#2.5
tR = 14
beta_R = 2.6
beta_plane = 2.2 #2.2
beta_outer = 2.4 #2.4
freq_0 = 408.

'''
# load the chain and flatten it
chain = np.load(os.path.join(directory, "chain_35_walkers_nstep40000.npy"))
samples = chain[2000::15, :, :].reshape(-1, chain.shape[2])  # discard=6000, thin=15, flat=True
np.save(os.path.join(directory, "samples.npy"), samples)
'''

################# to do a corner plot with discarded walkers ########################
exclude_walkers = []

# load the chain and discard burn-in and apply thinning
chain = np.load(os.path.join(directory, "chain.npy"))  # shape: (nsteps, nwalkers, ndim)
discard = 5000
#chain = chain[discard::15, :, :]  # shape: (nsteps, nwalkers, ndim)
chain = chain[discard:, :, :]

# optionally exclude walkers
include_walkers = [i for i in range(chain.shape[1]) if i not in exclude_walkers]
chain = chain[:, include_walkers, :]

# flatten the chain
samples = chain.reshape(-1, chain.shape[2])
#samples = samples[:, :5]  

# save the filtered samples
np.save(os.path.join(directory, "samples_excl_chains.npy"), samples)
'''
# get mean and error estimates
params_mcmc = np.mean(samples, axis=0)
params_err = np.percentile(samples, [16, 84], axis=0)

# unpack estimates
#
beta_Gal_mcmc, tR_mcmc, beta_R_mcmc, beta_plane_mcmc, beta_outer_mcmc = params_mcmc
beta_Gal_err_lower, beta_Gal_err_upper = beta_Gal_mcmc - params_err[0, 0], params_err[1, 0] - beta_Gal_mcmc
tR_err_lower, tR_err_upper = tR_mcmc - params_err[0, 1], params_err[1, 1] - tR_mcmc
beta_R_err_lower, beta_R_err_upper = beta_R_mcmc - params_err[0, 2], params_err[1, 2] - beta_R_mcmc
#beta_LG_err_lower, beta_LG_err_upper = beta_LG_mcmc - params_err[0, 3], params_err[1, 3] - beta_R_mcmc
beta_plane_err_lower, beta_plane_err_upper = beta_plane_mcmc - params_err[0, 3], params_err[1, 3] - beta_plane_mcmc
beta_outer_err_lower, beta_outer_err_upper = beta_outer_mcmc - params_err[0, 4], params_err[1, 4] - beta_outer_mcmc
'''

# ---------------------------------------------------------
# compute statistics properly (median + 1σ + 95% CI)
# ---------------------------------------------------------

percentiles = np.percentile(samples, [2.5, 16, 50, 84, 97.5], axis=0)

p2p5  = percentiles[0]
p16   = percentiles[1]
p50   = percentiles[2]   # median (best fit)
p84   = percentiles[3]
p97p5 = percentiles[4]

# unpack medians
beta_Gal_mcmc, tR_mcmc, beta_R_mcmc, beta_plane_mcmc, beta_outer_mcmc = p50

# 1σ errors (16–84)
beta_Gal_err_lower = beta_Gal_mcmc - p16[0]
beta_Gal_err_upper = p84[0] - beta_Gal_mcmc

tR_err_lower = tR_mcmc - p16[1]
tR_err_upper = p84[1] - tR_mcmc

beta_R_err_lower = beta_R_mcmc - p16[2]
beta_R_err_upper = p84[2] - beta_R_mcmc

beta_plane_err_lower = beta_plane_mcmc - p16[3]
beta_plane_err_upper = p84[3] - beta_plane_mcmc

beta_outer_err_lower = beta_outer_mcmc - p16[4]
beta_outer_err_upper = p84[4] - beta_outer_mcmc

# 95% confidence intervals
ci95 = list(zip(p2p5, p97p5))

# ---------------------------------------------------------
# print clean summary
# ---------------------------------------------------------

true_vals = [beta_Gal, tR, beta_R, beta_plane, beta_outer]
names = ["beta_Gal (HG)", "tR", "beta_R", "beta_plane", "beta_outer"]

print("\n================ Parameter Summary ================\n")

for i, name in enumerate(names):
    print(f"{name}")
    print(f"  True value        : {true_vals[i]}")
    print(f"  Best fit (median) : {p50[i]:.6f}")
    print(f"  1σ interval       : [{p16[i]:.6f}, {p84[i]:.6f}]")
    print(f"  95% CI            : [{ci95[i][0]:.6f}, {ci95[i][1]:.6f}]")
    print()

plt.rcParams.update({
    "font.size": 14,        # base font
    "axes.labelsize": 14,   # axis labels
    "axes.titlesize": 14,   # plot title
    "xtick.labelsize": 14,  # x ticks
    "ytick.labelsize": 14,  # y ticks
    "legend.fontsize": 14   # legend
})

# save the best fits
best_fits = np.array([
    [beta_Gal_mcmc, beta_Gal_err_lower, beta_Gal_err_upper],
    [tR_mcmc, tR_err_lower, tR_err_upper],
    [beta_R_mcmc, beta_R_err_lower, beta_R_err_upper],
    #[beta_LG_mcmc, beta_LG_err_lower, beta_LG_err_upper]
    [beta_plane_mcmc, beta_plane_err_lower, beta_plane_err_upper],
    [beta_outer_mcmc, beta_outer_err_lower, beta_outer_err_upper]
])
np.save(os.path.join(directory, "best_fits.npy"), best_fits)

# plot corner plot
#labels = ["beta_Gal", "tR", "beta_R", "beta_plane", "beta_outer"]
labels = [
    r"$\beta_{\mathrm{HG}}$",
    r"$T_{R}$",
    r"$\beta_{R}$",
    r"$\beta_{\mathrm{P}}$",
    r"$\beta_{\mathrm{O}}$"
]

truths = [beta_Gal, tR, beta_R, beta_plane, beta_outer]#, 0.5]

# set plotting ranges with padding
ranges = []
for i in range(samples.shape[1]):
    s_min, s_max = np.min(samples[:, i]), np.max(samples[:, i])
    padding = 0.4 * (s_max - s_min)
    ranges.append((s_min - padding, s_max + padding))

fig = corner.corner(
    samples,
    smooth = True,
    labels=labels,
    levels=[0.68, 0.95, 0.997],
    #bins=[9] * len(labels),
    bins=[14] * len(labels),
    truths=truths,
    range=ranges,
    label_kwargs={"fontsize": 20}
)

# annotation text

text_str = rf"""
True $T_{{\mathrm{{R}}}}$: {tR}
Fitted $T_{{\mathrm{{R}}}}$: {tR_mcmc:.3f} K $^{{+{tR_err_upper:.3g}}}_{{-{tR_err_lower:.3g}}}$

True $\beta_{{\mathrm{{R}}}}$: {beta_R}
Fitted $\beta_{{\mathrm{{R}}}}$: {beta_R_mcmc:.3f} $^{{+{beta_R_err_upper:.3g}}}_{{-{beta_R_err_lower:.3g}}}$

True $\beta_{{\mathrm{{HG}}}}$: {beta_Gal}
Fitted $\beta_{{\mathrm{{HG}}}}$: {beta_Gal_mcmc:.3f} $^{{+{beta_Gal_err_upper:.3g}}}_{{-{beta_Gal_err_lower:.3g}}}$

True $\beta_{{\mathrm{{LG,plane}}}}$: {beta_plane}
Fitted $\beta_{{\mathrm{{LG,plane}}}}$: {beta_plane_mcmc:.3f} $^{{+{beta_plane_err_upper:.3g}}}_{{-{beta_plane_err_lower:.3g}}}$

True $\beta_{{\mathrm{{LG,outer}}}}$: {beta_outer}
Fitted $\beta_{{\mathrm{{LG,outer}}}}$: {beta_outer_mcmc:.3f} $^{{+{beta_outer_err_upper:.3g}}}_{{-{beta_outer_err_lower:.3g}}}$
"""



#plt.gcf().text(0.70, 0.95, text_str, fontsize=14, verticalalignment='top', bbox=dict(facecolor='white', alpha=0.8))
#plt.suptitle(f"Triangle for {steps} steps (discard = {discard}) with 1hr Integration Time and Beam Error")

# save figure
print(text_str)

# choose filename based on whether walkers were excluded
if len(exclude_walkers) == 0:
    fig_name = "corner_plot"
else:
    fig_name = "plot_corner_excl_walkers"

# save figure
plt.savefig(os.path.join(directory, fig_name))
print(f"Corner plot saved as {fig_name}")

plt.savefig(os.path.join(directory, f"{fig_name}.png"))
print(f"Corner plot saved as {fig_name}.png")


import corner

# ---------------------------------------------------------
# TR-only corner plot (exact same style as full corner)
# ---------------------------------------------------------

tr_samples = samples[:, 1].reshape(-1, 1)

fig_tr = corner.corner(
    tr_samples,
    labels=[r"$T_R$"],
    truths=[tR],
    truth_color="red",
    bins=14,
    smooth=True,
    plot_datapoints=False,
    fill_contours=True,
    show_titles=False
)

# resize for cleaner single-panel look
fig_tr.set_size_inches(5, 4)

tr_name = "TR_posterior_corner_exact"
fig_tr.savefig(os.path.join(directory, f"{tr_name}.png"),
               dpi=300, bbox_inches="tight")

plt.close(fig_tr)

print(f"T_R corner-style posterior saved as {tr_name}.png")

######################################## wider plot #########################################
'''
# define wider ranges to ensure truths are visible
ranges = []
for i in range(samples.shape[1]):
    t = truths[i]
    max_dev = max(np.abs(samples[:, i] - t))
    pad = 0.2 * max_dev
    ranges.append((t - max_dev - pad, t + max_dev + pad))


# plot
fig_prior = corner.corner(
    samples,
    labels=labels,
    levels=[0.68, 0.95, 0.997],
    bins=9,
    truths=truths,
    range=ranges
)

# annotate
plt.gcf().text(0.55, 0.95, text_str, fontsize=10, verticalalignment='top',
               bbox=dict(facecolor='white', alpha=0.8))

# save
plt.savefig(os.path.join(directory, "corner_plot_wider_axes_excl_walkers.png"))
'''