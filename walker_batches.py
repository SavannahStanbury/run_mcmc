import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.lines import Line2D
import os

# load chain
chain = np.load('erb_pipeline/joint_likelihood/lst_6.2/chain.npy')  # shape: (nsteps, nwalkers, ndim)
nsteps, nwalkers, ndim = chain.shape

# updated labels including the extra parameter A
labels = ["beta_Gal", "tR", "beta_R", "beta_plane", "beta_outer"] #, "nuisance parameter"]  # <-- added A

# list of walkers to exclude (0-based indexing)

exclude_walkers = [1,9,11,13,10,18,19,29,26,21,24,20,36,39,38,42,49,45,43, 17,23,32,47]
include_walkers = [i for i in range(nwalkers) if i not in exclude_walkers]

# use tab10 for consistent color mapping
base_colors = plt.get_cmap('tab10')
color_map = [base_colors(i % 10) for i in range(nwalkers)]
gray = (0.7, 0.7, 0.7)

# output folder
os.makedirs("walker_batches", exist_ok=True)

# --- loop through batches of 10 walkers ---
for batch_idx in range(5):
    start = batch_idx * 10
    end = start + 10
    batch_walkers = [i for i in range(start, end) if i not in exclude_walkers]

    fig, axes = plt.subplots(len(labels), figsize=(10, 7), sharex=True)

    for i in range(len(labels)):
        ax = axes[i]
        for walker in batch_walkers:  # Only plot walkers in the current batch
            color = color_map[walker]
            ax.plot(chain[:, walker, i], alpha=0.5, color=color)
        ax.set_ylabel(labels[i])
        ax.grid(True)

    axes[-1].set_xlabel("Step number")
    plt.suptitle(f"MCMC Chains (Walkers {start+1}–{end}, excluding {', '.join(str(w+1) for w in exclude_walkers if start <= w < end)})", fontsize=13)
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    # legend for active walkers only
    legend_elements = [
        Line2D([0], [0], color=color_map[walker], lw=2, label=f"Walker {walker+1}")
        for walker in batch_walkers
    ]
    fig.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1, 1), fontsize='small')

    # filename
    filename = f"walker_batches/mcmc_chains_walkers_{start+1:02d}_to_{end:02d}_excluded.png"
    plt.savefig(filename)
    plt.close()
    print(f"Saved {filename}")

# --- Plot all walkers together (excluding excluded ones) without legend ---
fig, axes = plt.subplots(len(labels), figsize=(10, 7), sharex=True)

for i in range(len(labels)):
    ax = axes[i]
    for walker in include_walkers:
        ax.plot(chain[:, walker, i], alpha=0.5, color=color_map[walker])
    ax.set_ylabel(labels[i])
    ax.grid(True)

axes[-1].set_xlabel("Step number")
plt.suptitle("MCMC Chains: All Included Walkers", fontsize=13)
plt.tight_layout(rect=[0, 0, 1, 0.96])

# do NOT include legend for cleaner final plot
plt.savefig("walker_batches/mcmc_chains_all_walkers.png")
plt.close()
print("Saved walker_batches/mcmc_chains_all_walkers.png")
