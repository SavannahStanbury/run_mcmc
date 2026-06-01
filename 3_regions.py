# produces a healpy map that shows the different regions with different spix
# a demonstrative image, this script isnt used for any fitting

import healpy as hp
import ephem
import numpy as np
import matplotlib.pyplot as plt
import mcmc_funcs as mf

import matplotlib.colors as mcolors
cmap = mcolors.ListedColormap(['#b53688', '#7003a5', '#f3bb3a'])  # blue, orange, green
bounds = [-0.5, 0.5, 1.5, 2.5]
norm = mcolors.BoundaryNorm(bounds, cmap.N)

save_dir = "/home/stanbury/run_mcmc/regional_plots"

# ----------------------------
# Observer and LST
# ----------------------------
reach = ephem.Observer()
reach.lat = '-30.84'
reach.lon = '21.38'
reach.elevation = 1151
reach.date = '2024/02/11 19:09:04.87'
lst = '2024/02/11 19:09:04.87'

# ----------------------------
# Parameters
# ----------------------------
nside = 512
threshold = 100
freqs_beam = np.linspace(50, 100, 51)

# ----------------------------
# Load and prepare Haslam map
# ----------------------------
rotator_GC = hp.Rotator(coord=['G', 'C'])
rotator_CG = hp.Rotator(coord=['C', 'G'])
haslam_408_512_map = hp.read_map("haslam408.fits")
haslam_map = hp.ud_grade(haslam_408_512_map, nside)
celestial_haslam = rotator_GC.rotate_map_pixel(haslam_map)

# ----------------------------
# LST mask
# ----------------------------
LST_mask = mf.LST_map(np.ones_like(celestial_haslam), reach, lst)

# ----------------------------
# Low and high latitude masks
# ----------------------------
low_mask = mf.create_mask(nside, -70, -90, mask_type="celestial")
high_mask = np.where(np.isnan(low_mask), 1, np.nan)

# ----------------------------
# Split low-lat into bright/dim
# ----------------------------
T_408 = celestial_haslam * low_mask
bright_mask = (T_408 >= threshold) & (~np.isnan(T_408))
dim_mask = (T_408 < threshold) & (~np.isnan(T_408))

# ----------------------------
# Apply LST mask
# ----------------------------
bright_mask_LST = bright_mask * LST_mask
dim_mask_LST = dim_mask * LST_mask
dim_mask_LST[dim_mask_LST == 0] = np.nan
high_mask_LST = high_mask * LST_mask

# ----------------------------
# Save masks
# ----------------------------
np.save(f"{save_dir}/bright_mask_LST.npy", bright_mask_LST)
np.save(f"{save_dir}/dim_mask_LST.npy", dim_mask_LST)
np.save(f"{save_dir}/high_mask_LST.npy", high_mask_LST)

# ----------------------------
# Plot
# ----------------------------
hp.mollview(bright_mask_LST, title="Bright Low-Lat (LST visible)")
#plt.savefig(f"{save_dir}/bright_mask_LST.png", bbox_inches='tight', dpi=300)
plt.show()

hp.mollview(dim_mask_LST, title="Dim Low-Lat (LST visible)")
#plt.savefig(f"{save_dir}/dim_mask_LST.png", bbox_inches='tight', dpi=300)
plt.show()

hp.mollview(high_mask_LST, title="High-Lat (LST visible)")
#plt.savefig(f"{save_dir}/high_mask_LST.png", bbox_inches='tight', dpi=300)
plt.show()

bright_map_LST = bright_mask * LST_mask
dim_map_LST = dim_mask * LST_mask
high_map_LST = high_mask * LST_mask

# set zeros to nan for plotting
bright_map_LST[bright_map_LST == 0] = np.nan
dim_map_LST[dim_map_LST == 0] = np.nan
high_map_LST[high_map_LST == 0] = np.nan

# ----------------------------
# Plot individual regions
# ----------------------------
hp.mollview(bright_map_LST, title="Bright Region (LST visible)")
#plt.savefig(f"{save_dir}/bright_mask_LST_plot.png", bbox_inches='tight', dpi=300)
plt.show()

hp.mollview(dim_map_LST, title="Dim Region (LST visible)")
#plt.savefig(f"{save_dir}/dim_mask_LST_plot.png", bbox_inches='tight', dpi=300)
plt.show()

hp.mollview(high_map_LST, title="High-Lat Region (LST visible)")
#plt.savefig(f"{save_dir}/high_mask_LST_plot.png", bbox_inches='tight', dpi=300)
plt.show()

# ----------------------------
# Combined map
# ----------------------------
combined_map = np.full_like(bright_map_LST, np.nan)
combined_map[np.isfinite(high_map_LST)] = 0
combined_map[np.isfinite(dim_map_LST)] = 1
combined_map[np.isfinite(bright_map_LST)] = 2

hp.mollview(combined_map, title="", cmap='plasma', cbar=False)
plt.savefig(f"{save_dir}/combined_mask_LST_plot_threshold_{threshold}.png", bbox_inches='tight', dpi=300)
plt.show()


# ----------------------------
# Colourised plot in Galactic coordinates (full-sky)
# ----------------------------
# rotate bright and dim masks to Galactic
rotator_CG = hp.Rotator(coord=['C','G'])
bright_mask_gal = rotator_CG.rotate_map_pixel(bright_mask.astype(float))
dim_mask_gal = rotator_CG.rotate_map_pixel(dim_mask.astype(float))
high_mask_gal = rotator_CG.rotate_map_pixel(high_mask.astype(float))

# create combined map
gal_map_full = np.full_like(bright_mask_gal, np.nan)
gal_map_full[np.isfinite(high_mask_gal)] = 0
gal_map_full[np.isfinite(dim_mask_gal)] = 1
gal_map_full[np.isfinite(bright_mask_gal)] = 2

# define colours
import matplotlib.colors as mcolors
cmap = mcolors.ListedColormap(['#b53688', '#7003a5', '#f3bb3a'])  # blue, orange, green
bounds = [-0.5, 0.5, 1.5, 2.5]
norm = mcolors.BoundaryNorm(bounds, cmap.N)

# plot
hp.mollview(
    gal_map_full,
    title="Regions in Galactic Coordinates (full-sky)",
    cmap=cmap,
    norm=norm,
    cbar=False
)

#plt.savefig(f"{save_dir}/combined_mask_galactic_fullsky_threshold_{threshold}.png",
 #           bbox_inches='tight', dpi=300)
#plt.show()