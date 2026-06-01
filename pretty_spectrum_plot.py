import os
import numpy as np
import matplotlib.pyplot as plt

base_dir = 'mcmc_fit/seed_19_50/'
freqs = np.arange(50, 121, 1)
# ----------------- load data -----------------
high_galactic_lat = np.load(os.path.join(base_dir, "high_lat.npy"))
radio_excess      = np.load(os.path.join(base_dir, "radio_excess.npy"))
low_plane         = np.load(os.path.join(base_dir, "low_lat_plane.npy"))
low_outer         = np.load(os.path.join(base_dir, "low_lat_outer.npy"))
noise             = np.load(os.path.join(base_dir, "noise.npy"))

# recompute totals
y_tot = high_galactic_lat + radio_excess + low_plane + low_outer
y = y_tot + noise

# ----------------- DARK3-STYLE PALETTE -----------------

dark3 = {
    "ERB": "#50A6D7",                     # blue
    "High Galactic Latitude": "#FFB71B",  # orange
    "Low Galactic Latitude": "#6A29B5",   # green
    "Galactic Plane": "#CE659F",          # pink/magenta
}

styles = {
    "ERB": dict(color=dark3["ERB"], linestyle="-"),
    "High Galactic Latitude": dict(color=dark3["High Galactic Latitude"], linestyle="--"),
    "Low Galactic Latitude": dict(color=dark3["Low Galactic Latitude"], linestyle=":"),
    "Galactic Plane": dict(color=dark3["Galactic Plane"], linestyle="-."),
}

# ----------------- main plot -----------------
def make_plot(show_legend=True, filename="data_components.png"):
    plt.figure(figsize=(10, 6))

    plt.plot(freqs, radio_excess,
             label="ERB",
             **styles["ERB"])

    plt.plot(freqs, high_galactic_lat,
             label="High Galactic Latitude",
             **styles["High Galactic Latitude"])

    plt.plot(freqs, low_outer,
             label="Low Galactic Latitude",
             **styles["Low Galactic Latitude"])

    plt.plot(freqs, low_plane,
             label="Galactic Plane",
             **styles["Galactic Plane"])

    # total (bold black for clarity)
    plt.plot(freqs, y_tot,
             color="black",
             linewidth=2,
             linestyle="-",
             label="Total")

    # observed
    #plt.plot(freqs, y,
    #         color="0.3",
    #         alpha=0.6,
    #         linewidth=1,
    #         label="Observed (with noise)")

    plt.xlabel(r"$\nu$ (MHz)")
    plt.ylabel("T (K)")

    if show_legend:
        plt.legend()
    
    plt.ylim(0, 7000)
    plt.tight_layout()
    plt.savefig(os.path.join(base_dir, filename), dpi=300)
    plt.close()

# ----------------- residual plot -----------------
def plot_residual(filename="residual.png"):
    residual = y - y_tot

    plt.figure(figsize=(10, 5))
    plt.plot(freqs, residual, color="black", linewidth=1)

    plt.axhline(0, color="0.5", linestyle="--", linewidth=1)

    plt.xlabel(r"$\nu$ (MHz)")
    plt.ylabel(r"$T_{\mathrm{model with noise}} - T_{\mathrm{model}}$ (K)")

    plt.tight_layout()
    plt.savefig(os.path.join(base_dir, filename), dpi=300)
    plt.close()

# ----------------- save plots -----------------
make_plot(show_legend=True,  filename="data_components_with_legend.png")
make_plot(show_legend=False, filename="data_components_no_legend.png")
plot_residual(filename="residual_spectrum.png")