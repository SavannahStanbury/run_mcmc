#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# ============================================================
#                   IMPORTS
# ============================================================
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

# ---- your modules ----
import bean_functions as bf
import plot_beam_reach as bm
import mcmc_funcs as mf

# ============================================================
#                   SETUP
# ============================================================

# -------- directory to save all outputs --------
save_dir = "new_data/multi_LST"
os.makedirs(save_dir, exist_ok=True)
print(f"Main save directory: {save_dir}")

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
threshold = 100.  # not used directly in current code

params = {
    'freqs_beam': freqs_beam,
    'beta_Gal': beta_Gal,
    'tR': tR,
    'beta_R': beta_R,
    'beta_plane': beta_plane,
    'beta_outer': beta_outer,
    'freq_0': freq_0
}

np.save(os.path.join(save_dir, 'params.npy'), params)
print("Saved parameter dictionary.")

# -------- observer setup --------
reach = ephem.Observer()
reach.lat = '-30.84'
reach.lon = '21.38'
reach.elevation = 1151

# -------- define base LST --------
lst_6 = '2024-02-11 19:09:04.87'
# You can switch to lst_8, lst_10, lst_12 manually if needed
base_lst = lst_6


# ============================================================
#        LST Handling — compute ±5, 10, 15 minutes
# ============================================================

def lst_offset(lst_string, minutes):
    """Return LST string shifted by 'minutes'."""
    t = Time(lst_string, scale="utc")
    t_offset = t + minutes * 60.0
    return t_offset.strftime("%Y/%m/%d %H:%M:%S.%f")[:-3]

offsets = [-15, -10, -5, 0, 5, 10, 15]
lst_list = [lst_offset(base_lst, m) for m in offsets]

print("\nLST sequence to process:")
for L in lst_list:
    print(" →", L)


# ============================================================
#            STORAGE FOR COMBINED TOTALS
# ============================================================

all_high_lat = []
all_radio_excess = []
all_low_lat_total = []
all_low_lat_plane = []
all_low_lat_outer = []

# Per-LST subfolder
sub_dir = os.path.join(save_dir, "per_LST")
os.makedirs(sub_dir, exist_ok=True)
print(f"\nCreated per-LST data folder: {sub_dir}")


# ============================================================
#            MAIN LST LOOP
# ============================================================

for L in lst_list:

    print("\n===================================================")
    print(f"→ Running simulation for LST = {L}")
    print("===================================================")

    # Prepare per-LST directory
    LST_clean = L.replace(":", "-").replace(" ", "_")
    this_out = os.path.join(sub_dir, f"LST_{LST_clean}")
    os.makedirs(this_out, exist_ok=True)
    print(f"Saving outputs into: {this_out}")

    # Update observer time
    reach.date = L
    lst_hours = float(reach.sidereal_time()) * 12.0 / np.pi
    print(f"Computed LST = {lst_hours:.2f} hours")

    # ============================================================
    #               RADIO EXCESS
    # ============================================================
    print("Computing radio excess...")
    radio_excess = mf.compute_radio_excess(freqs_beam, tR, freq_0, beta_R)
    np.save(f"{this_out}/radio_excess.npy", radio_excess)
    print(" → Done.")

    # ============================================================
    #            HIGH GALACTIC LATITUDE
    # ============================================================
    print("Preparing and computing high-lat Haslam...")
    precomp = mf.prepare_high_lat_haslam(
        nside, reach, freqs_beam, freq_0, L, use_beam=True
    )
    high_lat = mf.compute_high_lat_haslam(precomp, tR, beta_R, beta_Gal)
    np.save(f"{this_out}/high_lat.npy", high_lat)
    print(" → Done.")

    # ============================================================
    #            LOW GALACTIC LATITUDE (MULTISPIX)
    # ============================================================
    print("Computing low-lat components (multispix)...")
    components = mf.compute_low_latitude_components_multispix(
        nside, reach, freqs_beam, L, threshold, use_beam=True
    )
    low_lat_total, low_lat_plane, low_lat_outer = mf.compute_low_latitude_multispix(
        freqs_beam, beta_plane, beta_outer, components, use_beam=True
    )

    np.save(f"{this_out}/low_lat_total.npy", low_lat_total)
    np.save(f"{this_out}/low_lat_plane.npy", low_lat_plane)
    np.save(f"{this_out}/low_lat_outer.npy", low_lat_outer)
    print(" → Done low-lat.")

    # ============================================================
    #         APPEND TO COMBINED TOTAL STORAGE
    # ============================================================
    all_high_lat.append(high_lat)
    all_radio_excess.append(radio_excess)
    all_low_lat_total.append(low_lat_total)
    all_low_lat_plane.append(low_lat_plane)
    all_low_lat_outer.append(low_lat_outer)

    print(f"✔ Finished simulation for LST = {L}")


# ============================================================
#        SAVE COMBINED (STACKED OVER ALL LSTs) DATA
# ============================================================

print("\n\n===================================================")
print("Saving COMBINED (all-LST) datasets...")
print("===================================================")

# stacked arrays: shape = (n_LST, n_freq)
all_high_lat = np.array(all_high_lat)
all_radio_excess = np.array(all_radio_excess)
all_low_lat_total = np.array(all_low_lat_total)
all_low_lat_plane = np.array(all_low_lat_plane)
all_low_lat_outer = np.array(all_low_lat_outer)

# ---- TIME AVERAGING PER FREQUENCY ----
# each averaged spectrum now has shape (n_freq,)
avg_high_lat = np.mean(all_high_lat, axis=0)
avg_radio_excess = np.mean(all_radio_excess, axis=0)
avg_low_lat_total = np.mean(all_low_lat_total, axis=0)
avg_low_lat_plane = np.mean(all_low_lat_plane, axis=0)
avg_low_lat_outer = np.mean(all_low_lat_outer, axis=0)

# ---- save stacked (raw) ----
np.save(f"{save_dir}/ALL_high_lat.npy", all_high_lat)
np.save(f"{save_dir}/ALL_radio_excess.npy", all_radio_excess)
np.save(f"{save_dir}/ALL_low_lat_total.npy", all_low_lat_total)
np.save(f"{save_dir}/ALL_low_lat_plane.npy", all_low_lat_plane)
np.save(f"{save_dir}/ALL_low_lat_outer.npy", all_low_lat_outer)

# ---- save TIME-AVERAGED spectra ----
np.save(f"{save_dir}/AVG_high_lat.npy", avg_high_lat)
np.save(f"{save_dir}/AVG_radio_excess.npy", avg_radio_excess)
np.save(f"{save_dir}/AVG_low_lat_total.npy", avg_low_lat_total)
np.save(f"{save_dir}/AVG_low_lat_plane.npy", avg_low_lat_plane)
np.save(f"{save_dir}/AVG_low_lat_outer.npy", avg_low_lat_outer)

print("✔ Combined + averaged files saved successfully.")
print("SCRIPT COMPLETED.\n")
