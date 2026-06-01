# this script plots a histogram for T_R only based of different errors. 

import os
import numpy as np
import matplotlib.pyplot as plt

base_root = "mcmc_fit/seed_19_50/"

datasets = {
    "RMS 1.95 K": {"dir": os.path.join(base_root, "error_b"), "color": "#4A90E2"},
    "RMS 19.55 K": {"dir": os.path.join(base_root, "error_a"), "color": "#C80470"},
    "RMS 97.75": {"dir": os.path.join(base_root, "error_d"), "color": "#F4A300"},
    "RMS 195.50": {"dir": os.path.join(base_root, "error_c"), "color": "#2E8B57"}
    
}

out_dir = os.path.join(base_root, "corner_TR")
os.makedirs(out_dir, exist_ok=True)

burn_frac = 0.3

tr_samples = {}
all_tr = []

for label, info in datasets.items():

    chain = np.load(os.path.join(info["dir"], "chain.npy"))

    nwalkers, nsteps, ndim = chain.shape

    burn_in = int(burn_frac * nsteps)

    flat = chain[:, burn_in:, :].reshape(-1, ndim)

    tr = flat[:, 1]

    tr_samples[label] = {"tr": tr, "color": info["color"]}

    all_tr.append(tr)

all_tr = np.concatenate(all_tr)

bins = np.linspace(np.min(all_tr), np.max(all_tr), 45)

plt.figure(figsize=(8, 5))

for label, data in tr_samples.items():

    plt.hist(
        data["tr"],
        bins=bins,
        density=True,
        alpha=0.35,
        color=data["color"],
        label=label
    )

plt.xlabel(r"$T_R$")
plt.ylabel("Posterior density")

plt.xlim(0, 20)
plt.axvline(14, color="black", linewidth=2.0)

plt.legend(frameon=False)

plt.tight_layout()

plt.savefig(os.path.join(out_dir, "TR_corner_overlay.png"), dpi=300)

plt.close()

print("saved: TR_corner_overlay.png")