# this script converts a normal time to an LST for REACH
# it is basically to check what times correspond to which LSTs
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

lst_6 = '2025/11/06 01:32:38.965461'
lst_62 = '2025/11/06 01:44:31'
lst_10 = '2024/02/11 23:08:30.00'
lst_12 = '2024/02/12 01:08:00.00'
lst = '2025/11/06 01:50:30' #lst 6 on a different day

# -------- observer setup --------
reach = ephem.Observer()
reach.lat = '-30.84'
reach.lon = '21.38'
reach.elevation = 1151

#--------- LST conversion --------------
reach.date = lst

lst_rad = reach.sidereal_time()
lst_hours = float(lst_rad) * 12 / np.pi  # convert radians → hours

print(f"LST (hours): {lst_hours:.2f} h")
#-----------------------------------------------

#reach.date = lst_8

#lst_rad = reach.sidereal_time()
#lst_hours = float(lst_rad) * 12 / np.pi  # convert radians → hours

#print(f"LST (hours): {lst_hours:.2f} h")

#-----------------------------------------------

#reach.date = lst_10

#lst_rad = reach.sidereal_time()
#lst_hours = float(lst_rad) * 12 / np.pi  # convert radians → hours

#print(f"LST (hours): {lst_hours:.2f} h")
#-----------------------------------------------