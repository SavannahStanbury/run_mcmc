import numpy as np
import matplotlib.pyplot as plt
import corner
import os

# set directory for data input/output
#directory = "new_data/noise_param/case4.5"
directory = 'erb_pipeline/obs_data_LST_6.0/lst_6.0_median/nuisance_2'#"testing_og_code/tR_6"#"testing_og_code/sub_tR/5_param_fit_with_error/case_4" #"new_data/sys_error/A_0.05"
# true values for comparison
steps = '50k'
discard = '20k'

beta_Gal = 2.5
tR = 14
beta_R = 2.6
beta_plane = 2.2
beta_outer = 2.4
A_true = 0.001  # fixed sinusoidal systematic (not fitted)

################# to do a corner plot with discarded walkers ########################
exclude_walkers = []
# load the chain and discard burn-in and apply thinning
chain = np.load(os.path.join(directory, "chain.npy"))  # shape: (nsteps, nwalkers, ndim)
discard = 5000
chain = chain[discard:, :, :]

# optionally exclude walkers
include_walkers = [i for i in range(chain.shape[1]) if i not in exclude_walkers]
chain = chain[:, include_walkers, :]

# flatten the chain
samples = chain.reshape(-1, chain.shape[2])

# save the filtered samples
np.save(os.path.join(directory, "samples_excl_chains.npy"), samples)

# get mean and error estimates
params_mcmc = np.mean(samples, axis=0)
params_err = np.percentile(samples, [16, 84], axis=0)
params_95 = np.percentile(samples, [2.5, 97.5], axis=0) #95% interval


# unpack estimates
beta_Gal_mcmc, tR_mcmc, beta_R_mcmc, beta_plane_mcmc, beta_outer_mcmc, nuisance_mcmc = params_mcmc
beta_Gal_err_lower, beta_Gal_err_upper = beta_Gal_mcmc - params_err[0, 0], params_err[1, 0] - beta_Gal_mcmc
tR_err_lower, tR_err_upper = tR_mcmc - params_err[0, 1], params_err[1, 1] - tR_mcmc
beta_R_err_lower, beta_R_err_upper = beta_R_mcmc - params_err[0, 2], params_err[1, 2] - beta_R_mcmc
beta_plane_err_lower, beta_plane_err_upper = beta_plane_mcmc - params_err[0, 3], params_err[1, 3] - beta_plane_mcmc
beta_outer_err_lower, beta_outer_err_upper = beta_outer_mcmc - params_err[0, 4], params_err[1, 4] - beta_outer_mcmc
nuisance_err_lower, nuisance_err_upper = nuisance_mcmc - params_err[0, 5], params_err[1, 5] - nuisance_mcmc

# save the best fits (now including nuisance_param)
best_fits = np.array([
    [beta_Gal_mcmc, beta_Gal_err_lower, beta_Gal_err_upper],
    [tR_mcmc, tR_err_lower, tR_err_upper],
    [beta_R_mcmc, beta_R_err_lower, beta_R_err_upper],
    [beta_plane_mcmc, beta_plane_err_lower, beta_plane_err_upper],
    [beta_outer_mcmc, beta_outer_err_lower, beta_outer_err_upper],
    [nuisance_mcmc, nuisance_err_lower, nuisance_err_upper]
])
np.save(os.path.join(directory, "best_fits.npy"), best_fits)

# plot corner plot
labels = [
    r"$\beta_{\mathrm{HG}}$",
    r"$T_{R}$",
    r"$\beta_{R}$",
    r"$\beta_{\mathrm{P}}$",
    r"$\beta_{\mathrm{O}}$",
    r"$\sigma_{\mathrm{sys}}$"
]

# true values — nuisance has no true value, so use None
truths = [beta_Gal, tR, beta_R, beta_plane, beta_outer, None]

# set plotting ranges with padding
ranges = []
padding_frac = 0.2  # 20% extra padding on both sides

for i in range(samples.shape[1]):
    s_min, s_max = np.min(samples[:, i]), np.max(samples[:, i])
    true_val = truths[i]
    if true_val is not None:
        s_min = min(s_min, true_val)
        s_max = max(s_max, true_val)
    delta = (s_max - s_min) * padding_frac
    ranges.append((s_min - delta, s_max + delta))

plt.rcParams.update({
    "font.size": 14,        # base font
    "axes.labelsize": 14,   # axis labels
    "axes.titlesize": 14,   # plot title
    "xtick.labelsize": 14,  # x ticks
    "ytick.labelsize": 14,  # y ticks
    "legend.fontsize": 14   # legend
})

fig = corner.corner(
    samples,
    labels=labels,
    levels=[0.68, 0.95, 0.997],
    bins=[12] * len(labels),
    #truths=truths,
    range=ranges,
    smooth=1.0,
    label_kwargs={"fontsize": 20}
)

for ax in fig.get_axes():
    ax.tick_params(axis='both', labelsize=14)

# annotation text for simulated data
text_str = f"""
==================== MCMC FIT RESULTS ====================

beta_HG:
  true value     = {beta_Gal}
  fitted value   = {beta_Gal_mcmc:.3f}  (+{beta_Gal_err_upper:.3g} / -{beta_Gal_err_lower:.3g})
  95% CI         = [{params_95[0,0]:.3f}, {params_95[1,0]:.3f}]

tR (K):
  true value     = {tR}
  fitted value   = {tR_mcmc:.3f}  (+{tR_err_upper:.3g} / -{tR_err_lower:.3g})
  95% CI         = [{params_95[0,1]:.3f}, {params_95[1,1]:.3f}]

beta_R:
  true value     = {beta_R}
  fitted value   = {beta_R_mcmc:.3f}  (+{beta_R_err_upper:.3g} / -{beta_R_err_lower:.3g})
  95% CI         = [{params_95[0,2]:.3f}, {params_95[1,2]:.3f}]

beta_plane:
  true value     = {beta_plane}
  fitted value   = {beta_plane_mcmc:.3f}  (+{beta_plane_err_upper:.3g} / -{beta_plane_err_lower:.3g})
  95% CI         = [{params_95[0,3]:.3f}, {params_95[1,3]:.3f}]

beta_outer:
  true value     = {beta_outer}
  fitted value   = {beta_outer_mcmc:.3f}  (+{beta_outer_err_upper:.3g} / -{beta_outer_err_lower:.3g})
  95% CI         = [{params_95[0,4]:.3f}, {params_95[1,4]:.3f}]

nuisance_param (K):
  fitted value   = {nuisance_mcmc:.3f}  (+{nuisance_err_upper:.3g} / -{nuisance_err_lower:.3g})
  95% CI         = [{params_95[0,5]:.3f}, {params_95[1,5]:.3f}]

===========================================================
"""

print(text_str)

#output_path = os.path.join(directory, "mcmc_outputs.txt")
#with open(output_path, "w") as f:
#    f.write(text_str)


# save figure
fig_name = "corner_plot"
plt.savefig(os.path.join(directory, f"{fig_name}.png"), dpi=300, bbox_inches="tight")
print(f"Corner plot saved as {fig_name}.png")
