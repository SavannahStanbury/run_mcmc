import ephem
import numpy as np
import healpy as hp
import matplotlib.pyplot as plt
import bean_functions as bf
import plot_beam_reach as bm

# disable LaTeX in matplotlib
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['text.usetex'] = False

import warnings
warnings.filterwarnings("ignore", category=np.RankWarning)

# ------------------ REACH Observer ------------------
reach = ephem.Observer()
reach.lat = '-30.84'
reach.lon = '21.38'
reach.elevation = 1151

# ------------------ Coordinate Rotators ------------------
rotator_GC = hp.Rotator(coord=['G', 'C'])
rotator_CG = hp.Rotator(coord=['C', 'G'])

# ------------------ General Functions ------------------

def LST_map(map_data, observer, time_utc_str):
    """
    mask pixels below the horizon at a given UTC time for an observer.
    """
    observer.date = ephem.Date(time_utc_str)
    nside = hp.get_nside(map_data)
    npix = len(map_data)
    theta, phi = hp.pix2ang(nside, np.arange(npix))
    dec = np.degrees(np.pi/2 - theta)
    ra = np.degrees(phi)
    
    altitudes = np.empty(npix)
    for i in range(npix):
        body = ephem.FixedBody()
        body._ra = np.radians(ra[i])
        body._dec = np.radians(dec[i])
        body._epoch = observer.date
        body.compute(observer)
        altitudes[i] = body.alt

    masked_map = map_data.copy()
    masked_map[altitudes < 0] = np.nan
    return masked_map

def create_mask(nside, lat_threshold_upper=-70, lat_threshold_lower=-90, mask_type='galactic'):
    """
    create a mask for pixels based on latitude thresholds.
    """
    npix = hp.nside2npix(nside)
    theta, _ = hp.pix2ang(nside, np.arange(npix))
    b_lat = 90 - np.degrees(theta)
    
    mask = np.ones(npix)
    mask[(b_lat > lat_threshold_lower) & (b_lat < lat_threshold_upper)] = np.nan

    if mask_type == 'celestial':
        mask = rotator_GC.rotate_map_pixel(mask)
    return mask

def compute_radio_excess(freqs, tR, freq_0, beta_R):
    """
    compute extragalactic radio excess for a given frequency range.
    """
    freqs = np.array(freqs)  # ensure numpy array to avoid TypeError
    return tR * (freqs / freq_0) ** -beta_R

def scale_haslam_map(map_408, target_freq, spix):
    """
    scale the Haslam map to a new frequency using a spectral index.
    """
    return map_408 * (target_freq / 408.) ** -spix

# ------------------ High-latitude (Haslam-based) ------------------

def prepare_high_lat_haslam(nside, observer, freqs, freq_0, lst, use_beam=True):
    """
    prepare high-latitude patch and optionally beams for computation.
    """
    haslam_408 = hp.ud_grade(hp.read_map("haslam408.fits"), nside)
    celestial_haslam = rotator_GC.rotate_map_pixel(haslam_408)
    
    low_mask = create_mask(nside, -70, -90, mask_type="celestial")
    high_mask = np.where(np.isnan(low_mask), 1, np.nan)
    LST_mask_vals = LST_map(np.ones_like(celestial_haslam), observer, lst)
    
    HG_patch_haslam_408 = celestial_haslam * high_mask * LST_mask_vals
    
    rotated_beams = []
    if use_beam:
        for freq in freqs:
            beam = bm.generate_beam(freq, lst)
            rotated = rotator_GC.rotate_map_pixel(beam)
            rotated[rotated < 0] = np.nan
            rotated_beams.append(rotated)

    return {
        "HG_patch_haslam_408": HG_patch_haslam_408,
        "high_mask": high_mask,
        "rotated_beams": rotated_beams if use_beam else None,
        "use_beam": use_beam,
        "freqs": freqs,
        "freq_0": freq_0
    }

def compute_high_lat_haslam(precomputed, tR, beta_R, beta_HG):
    """
    compute high-latitude temperatures for each frequency.
    """
    T_H = np.nanmean(precomputed["HG_patch_haslam_408"])
    T_R = tR
    
    freqs = precomputed["freqs"]
    T_H_minus_T_R_scaled = np.zeros(len(freqs))
    for i, freq in enumerate(freqs):
        T_H_minus_T_R_scaled[i] = (T_H - T_R) * (freq / precomputed["freq_0"])**-beta_HG

    integral_omega = np.zeros(len(freqs))
    integral_omega_prime = np.zeros(len(freqs))

    if precomputed["use_beam"]:
        for i, freq in enumerate(freqs):
            A = precomputed["rotated_beams"][i]
            integral_omega[i] = np.nansum(A)
            integral_omega_prime[i] = np.nansum(T_H_minus_T_R_scaled[i] * A * precomputed["high_mask"])
    else:
        for i, freq in enumerate(freqs):
            integral_omega[i] = np.sum(~np.isnan(precomputed["HG_patch_haslam_408"]))
            integral_omega_prime[i] = T_H_minus_T_R_scaled[i]

    return integral_omega_prime / integral_omega

# ------------------ Low-latitude (multispix) ------------------

def compute_low_latitude_components_multispix(nside, observer, freqs, lst, threshold=100.0, use_beam=True):
    """
    compute low-latitude map components for each frequency.
    """
    haslam_408 = hp.ud_grade(hp.read_map("haslam408.fits"), nside)
    celestial_haslam = rotator_GC.rotate_map_pixel(haslam_408)

    low_mask = create_mask(nside, -70, -90, mask_type="celestial")
    T_408 = celestial_haslam * low_mask

    bright_mask = (T_408 >= threshold) & (~np.isnan(T_408))
    dim_mask = (T_408 < threshold) & (~np.isnan(T_408))

    LST_mask_vals = LST_map(np.ones_like(celestial_haslam), observer, lst)
    integral_omega_408 = np.nansum(T_408)

    A = None
    integral_omega = None
    if use_beam:
        A = np.zeros((len(freqs), len(T_408)))
        integral_omega = np.zeros(len(freqs))
        for i, freq in enumerate(freqs):
            beam = bm.generate_beam(freq, lst)
            A[i] = rotator_GC.rotate_map_pixel(beam)
            A[i][A[i] < 0] = np.nan
            integral_omega[i] = np.nansum(A[i])
    
    return T_408, integral_omega_408, LST_mask_vals, A, integral_omega, bright_mask, dim_mask

def compute_low_latitude_multispix(freqs, beta_plane, beta_outer, components, use_beam=True):
    """
    compute low-latitude temperatures split into plane, outer, and combined.
    """
    T_408, _, LST_mask_vals, A, integral_omega, bright_mask, dim_mask = components
    low_lat_combined = np.zeros(len(freqs))
    low_lat_plane = np.zeros(len(freqs))
    low_lat_outer = np.zeros(len(freqs))

    for i, freq in enumerate(freqs):
        plane_map = np.full_like(T_408, np.nan)
        outer_map = np.full_like(T_408, np.nan)

        plane_map[bright_mask] = T_408[bright_mask] * (freq / 408.0) ** -beta_plane
        outer_map[dim_mask] = T_408[dim_mask] * (freq / 408.0) ** -beta_outer

        plane_map *= LST_mask_vals
        outer_map *= LST_mask_vals

        if use_beam:
            low_lat_plane[i] = np.nansum(plane_map * A[i]) / np.nansum(A[i])
            low_lat_outer[i] = np.nansum(outer_map * A[i]) / np.nansum(A[i])
            low_lat_combined[i] = low_lat_plane[i] + low_lat_outer[i]
            #low_lat_combined[i] = np.nansum((plane_map + outer_map) * A[i]) / np.nansum(A[i])
        else:
            low_lat_plane[i] = np.nansum(plane_map) / np.count_nonzero(~np.isnan(plane_map))
            low_lat_outer[i] = np.nansum(outer_map) / np.count_nonzero(~np.isnan(outer_map))
            combined_map = plane_map + outer_map
            low_lat_combined[i] = np.nansum(combined_map) / np.count_nonzero(~np.isnan(combined_map))

    return low_lat_combined, low_lat_plane, low_lat_outer
