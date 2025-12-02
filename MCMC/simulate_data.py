import ephem
import numpy as np
import matplotlib.pyplot as plt
import healpy as hp
import os
import emcee
import corner
from scipy import stats
from matplotlib import rc
from scipy.integrate import quad
from astropy.time import Time
from scipy.interpolate import interp1d
import bean_functions as bf
import plot_beam_reach as bm
import mcmc_funcs as mf

# -------- directory to save all outputs --------
#save_dir = "new_data/noise_param/case4" #sim_data_v6"
#save_dir = "new_data/sys_error/A_0.001"
save_dir = "new_data/5_params_simulation"
os.makedirs(save_dir, exist_ok=True)

# -------- initial settings --------
nside = 64

freqs_beam = np.arange(50., 101., 1)
beta_Gal = 2.5
tR = 14
beta_R = 2.6
beta_plane = 2.2
beta_outer = 2.4
freq_0 = 408.

beta_LG = 2.3
threshold = 100 #K

params = {
    'freqs_beam': freqs_beam,
    'beta_Gal': beta_Gal,
    'tR': tR,
    'beta_R': beta_R,
    'beta_plane': beta_plane,
    'beta_outer': beta_outer,
    'freq_0': freq_0
}

np.save('params.npy', params)


lst_6 = '2024/02/11 19:09:04.87'
lst_8 = '2024/02/11 21:09:03.00'
lst_10 = '2024/02/11 23:08:30.00'
lst_12 = '2024/02/12 01:08:00.00'
# -------- observer setup --------
reach = ephem.Observer()
reach.lat = '-30.84'
reach.lon = '21.38'
reach.elevation = 1151

#--------- LST conversion --------------
reach.date = lst_6

lst_rad = reach.sidereal_time()
lst_hours = float(lst_rad) * 12 / np.pi  # convert radians → hours

print(f"LST (hours): {lst_hours:.2f} h")

# -------- simulate radio excess --------
radio_excess = mf.compute_radio_excess(freqs_beam, tR, freq_0, beta_R)
np.save(f"{save_dir}/408_radio_excess.npy", radio_excess)
print(f"Radio BG: {radio_excess}")

# -------- simulate HG with Haslam --------
precomp = mf.prepare_high_lat_haslam(nside, reach, freqs_beam, freq_0, lst_6, use_beam=True)
high_lat = mf.compute_high_lat_haslam(precomp, tR, beta_R, beta_Gal)
np.save(f"{save_dir}/408_high_lat.npy", high_lat)
print(f"High Lat: {high_lat}")
'''
# -------- simulate LG with one spix --------
components = mf.compute_low_latitude_components(nside, reach, freqs_beam, lst_6, threshold=100.0, use_beam=True)
low_lat = mf.compute_low_latitude(freqs_beam, beta_LG, components, use_beam=True)
np.save(f"{save_dir}/low_lat.npy", low_lat)
print(f"Low Lat: {low_lat}")

# -------- simulate LG with multispix --------
components = mf.compute_low_latitude_components_multispix(nside, reach, freqs_beam, lst_6, threshold, use_beam=True)
low_lat = mf.compute_low_latitude_multispix(freqs_beam, beta_plane, beta_outer, components, use_beam=True)
np.save(f"{save_dir}/multispix_low_lat.npy", low_lat)
print(f"Low Lat: {low_lat}")
'''

# -------- simulate LG with multispix --------
components = mf.compute_low_latitude_components_multispix(
    nside, reach, freqs_beam, lst_6, threshold=25.0, use_beam=True
)
low_lat_total, low_lat_plane, low_lat_outer = mf.compute_low_latitude_multispix(
    freqs_beam, beta_plane, beta_outer, components, use_beam=True
)

np.save(f"{save_dir}/multispix_low_lat_total.npy", low_lat_total)
np.save(f"{save_dir}/multispix_low_lat_plane.npy", low_lat_plane)
np.save(f"{save_dir}/multispix_low_lat_outer.npy", low_lat_outer)

print("Low Lat total:", low_lat_total)
print("Plane only:", low_lat_plane)
print("Outer only:", low_lat_outer)

# -------- simulate beam error --------

y = mf.beam_calibration_error(0.01, 50., freqs_beam, 0.)
print(y)
np.save(f"{save_dir}/beam_calibration_error.npy", y)

plt.plot(freqs_beam, y)
plt.xlabel("Frequency (MHz)")
plt.ylabel("Relative Gain Error")
plt.title("Simulated Beam Calibration Error")
plt.grid()
plt.savefig(f"{save_dir}/systematic_error.png")

int_time = 9
seed_noise=20

# ----------------- generate signal with noise -----------------
y_tot = high_lat + radio_excess + low_lat_outer + low_lat_plane
np.random.seed(23)
np.save(f"{save_dir}/total_signal.npy")
sigma = y_tot / np.sqrt(1e6 * 3600 * 1)
noise = np.random.normal(0.0, sigma)
y = y_tot + noise
'''
# ----------------- plot noiseless components + total -----------
plt.figure(figsize=(8, 5))
plt.plot(freqs_beam, high_lat, label="High Galactic Latitude")
plt.plot(freqs_beam, radio_excess, label="Radio Excess")
plt.plot(freqs_beam, low_lat_plane, label="Low Lat (Plane)")
plt.plot(freqs_beam, low_lat_outer, label="Low Lat (Outer)")
plt.plot(freqs_beam, y_tot, label="Total Signal", color='k', linewidth=1.2)
plt.xlabel("Frequency [MHz]")
plt.ylabel("Temperature [K]")
plt.title("Noiseless Simulated Spectrum")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(save_dir, "plot_noiseless_spectrum.png"))
plt.close()

# ------------------- plot noisy total ----------------------------
plt.figure(figsize=(8, 5))
plt.plot(freqs_beam, y, label="Total Signal + Noise", color='purple')
plt.plot(freqs_beam, y_tot, label="Noiseless Total", color='k', linestyle='--')
plt.xlabel("Frequency [MHz]")
plt.ylabel("Temperature [K]")
plt.title("Noisy Simulated Spectrum")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(save_dir, "plot_noisy_spectrum.png"))
plt.close()

# ------------------------- plot residual --------------------
plt.figure(figsize=(8, 5))
plt.plot(freqs_beam, y - y_tot, label="Residual (Noisy - Noiseless)", color='red')
plt.axhline(0, color='black', linestyle='--', linewidth=1)
plt.xlabel("Frequency [MHz]")
plt.ylabel("Temperature Residual [K]")
plt.title("Residual Spectrum")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(save_dir, "plot_residual_spectrum.png"))
plt.close()
'''