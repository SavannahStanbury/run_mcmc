import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.lines import Line2D

# load the chain
chain = np.load("chain_35_walkers_nstep40000.npy")
nsteps, nwalkers, ndim = chain.shape
chain_500 = chain[:500, :, :]

labels = ["beta_Gal", "tR", "beta_R", "beta_plane", "beta_outer"]
#colors = [cm.viridis(x) for x in np.linspace(0.1, 0.9, nwalkers)]
#colors = [cm.hsv(x) for x in np.linspace(0, 1, nwalkers)]
colors = [cm.nipy_spectral(x) for x in np.linspace(0, 1, nwalkers)]


# --- plot first 500 steps ---
fig, axes = plt.subplots(len(labels), figsize=(10, 7), sharex=True)
for i in range(len(labels)):
    ax = axes[i]
    for walker in range(nwalkers):
        ax.plot(chain_500[:, walker, i], alpha=0.4, color=colors[walker])
    ax.set_ylabel(labels[i])
    ax.grid(True)
axes[-1].set_xlabel("Step number")
plt.suptitle("MCMC Chains (First 500 Steps)", fontsize=14)
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig("mcmc_chains_500steps.png")
print("Saved as mcmc_chains_500steps.png")

# --- plot full chain ---
fig1, axes1 = plt.subplots(len(labels), figsize=(10, 7), sharex=True)
for i in range(len(labels)):
    ax = axes1[i]
    for walker in range(nwalkers):
        ax.plot(chain[:, walker, i], alpha=0.4, color=colors[walker])
    ax.set_ylabel(labels[i])
    ax.grid(True)
axes1[-1].set_xlabel("Step number")
plt.suptitle("MCMC Chains (All Steps)", fontsize=14)
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig("mcmc_chains_all_steps.png")
print("Saved as mcmc_chains_all_steps.png")

# --- create separate legend image ---
legend_elements = [Line2D([0], [0], color=colors[i], lw=2, label=f'Walker {i}') for i in range(nwalkers)]

fig_legend = plt.figure(figsize=(8, 1.5))
ax_legend = fig_legend.add_subplot(111)
ax_legend.axis("off")
legend = ax_legend.legend(handles=legend_elements, loc='center', ncol=5, fontsize=8, frameon=False)
plt.tight_layout()
plt.savefig("mcmc_chain_legend.png")
print("Saved as mcmc_chain_legend.png")