import ephem
import numpy as np
import matplotlib.pyplot as plt
import random
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
#import reach_beam_script as bm

# Disable LaTeX in Matplotlib
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['text.usetex'] = False

import warnings
warnings.filterwarnings("ignore", category=np.RankWarning)

"""### **REACH**"""

# Observer setup
reach = ephem.Observer()
reach.lat = '-30.84'
reach.lon = '21.38'
reach.elevation = 1151


"""# **Parameters**"""

lst = '2024/02/11 19:09:04.87'
nside=64
npix = hp.nside2npix(nside)

reach.date = lst

"""# **General Definitions**"""

# Rotate map from Galactic to Celestial coordinates
rotator_GC = hp.Rotator(coord=['G', 'C'])
rotator_CG = hp.Rotator(coord=['C', 'G'])

def LST_map(map_data, observer, time_utc_str, save_plots=False):
    """
    mask pixels below the horizon at a given UTC time for an observer.

    parameters:
        map_data: numpy array (HEALPix map in celestial coordinates)
        observer (ephem.Observer): ephem observer with lat/lon/elevation set
        time_utc_str (str): UTC time string, e.g. '2024/02/11 14:09:54.02'
        save_plots (bool): whether to save diagnostic plots

    returns:
        numpy array: copy of map_data with below-horizon pixels set to np.nan
    """
    #print(f"Setting observer date to {time_utc_str}")
    observer.date = ephem.Date(time_utc_str)

    nside = hp.get_nside(map_data)
    npix = len(map_data)
    #print(f"Map nside: {nside}, npix: {npix}")

    # convert pixel indices to theta, phi
    theta, phi = hp.pix2ang(nside, np.arange(npix))
    dec = np.degrees(np.pi/2 - theta)  # declination
    ra = np.degrees(phi)               # right ascension

    #print(f"RA range: {ra.min()} - {ra.max()}, Dec range: {dec.min()} - {dec.max()}")

    # compute altitudes for each pixel
    altitudes = np.empty(npix)
    for i in range(npix):
        body = ephem.FixedBody()
        body._ra = np.radians(ra[i])
        body._dec = np.radians(dec[i])
        body._epoch = observer.date
        body.compute(observer)
        altitudes[i] = body.alt

    #print(f"Altitude stats: min={np.min(altitudes)}, max={np.max(altitudes)}, mean={np.mean(altitudes)}")
    #print(f"Number of pixels above horizon: {(altitudes >= 0).sum()}")

    # mask below horizon
    masked_map = map_data.copy()
    masked_map[altitudes < 0] = np.nan

    if save_plots:
        # original map
        hp.mollview(map_data, title="Original Map", min=np.nanmin(map_data), max=np.nanmax(map_data))
        hp.graticule()
        plt.savefig("test_results/LST_original_map.png")
        plt.clf()

        # masked map
        hp.mollview(masked_map, title="Masked Map (Below Horizon NaN)", min=np.nanmin(masked_map), max=np.nanmax(masked_map))
        hp.graticule()
        plt.savefig("test_results/LST_masked_map.png")
        plt.clf()

    return masked_map

def scale_map(map_408, freq_408, freq_new, beta):
    """
    scale a map from freq_408 to freq_new using spectral index beta.

    map_408 : healpix map at freq_408 (e.g., 408 MHz)
    freq_408 : reference frequency (float)
    freq_new : target frequency (float)
    beta : spectral index

    returns: map scaled to freq_new
    """
    return map_408 * (freq_new / freq_408) ** (-beta)


def create_mask(nside, lat_threshold_upper=-70, lat_threshold_lower=-90, mask_type='galactic', save_plots=False):
    """
    create a mask for pixels based on latitude thresholds.

    parameters:
        nside: healpy nside resolution
        lat_threshold_upper: upper latitude threshold for masking
        lat_threshold_lower: lower latitude threshold for masking
        mask_type: 'galactic' or 'celestial'
        save_plots: save diagnostic plots

    returns:
        mask: the generated mask

        **note that the part that is masked off is the part in between the thresholds**
    """
    npix = hp.nside2npix(nside)
    #print(f"nside={nside}, npix={npix}")

    theta, _ = hp.pix2ang(nside, np.arange(npix))  # theta in radians
    b_lat = 90 - np.degrees(theta)                 # latitude in degrees
    #print(f"Latitude range: {b_lat.min()} to {b_lat.max()}")

    # create initial mask: 1 everywhere
    mask = np.ones(npix)

    # mask pixels where latitude is within thresholds
    condition = (b_lat >= lat_threshold_lower) & (b_lat <= lat_threshold_upper)
    #print(f"Number of pixels to mask (set to NaN): {np.sum(condition)}")
    mask[condition] = np.nan

    if mask_type == 'celestial':
        mask = rotator_GC.rotate_map_pixel(mask)
        #print("Applied celestial rotation to mask")

    if save_plots:
        # plot the mask
        hp.mollview(mask, title=f"{mask_type.capitalize()} Mask", min=0, max=1)
        hp.graticule()
        plt.savefig("test_results/mask_plot.png")
        plt.clf()

        # histogram of latitudes
        plt.hist(b_lat, bins=50)
        plt.title("Latitude distribution of pixels")
        plt.xlabel("Latitude [deg]")
        plt.ylabel("Number of pixels")
        plt.savefig("test_results/mask_lat_hist.png")
        plt.clf()

    return mask


def compute_radio_excess(freqs, tR, freq_0, beta_R):
    """
    compute extragalactic radio excess for a given frequency range.

    parameters:
        freqs: array of frequency values
        tR: reference temperature
        freq_0: reference frequency
        beta_R: spectral index

    returns:
        array of radio excess values for each frequency
    """
    return (tR * (freqs / freq_0) ** -beta_R)

def scale_haslam_map(map_408, target_freq, spix):
    return map_408 * (target_freq / 408.) ** -spix


###################### produces high latitude from haslam  ################]
'''
def high_lat_haslam(tR, beta_R, beta_HG, nside, observer, freqs, freq_0, lst, use_beam=True):
    # make a high galactic patch, usingthe haslam map, at the correct LST
    haslam_408_512_map = hp.read_map("haslam408.fits")
    haslam_map = hp.ud_grade(haslam_408_512_map, nside)
    celestial_haslam = rotator_GC.rotate_map_pixel(haslam_map)
    low_mask = create_mask(nside, -70, -90, mask_type="celestial")
    high_mask = np.where(np.isnan(low_mask), 1, np.nan)
    LST_mask = LST_map(np.ones_like(celestial_haslam), observer, lst)
    haslam_at_LST = LST_map(celestial_haslam, observer, lst)
    #hp.mollview(haslam_at_LST, title='Haslam at LST 2')
    #hp.graticule()
    #plt.savefig("Haslam_at_LST.png")


    HG_patch_haslam_408 = celestial_haslam * high_mask * LST_mask
    #hp.mollview(HG_patch_haslam_408, title="HG Patch, from the Haslam Map, at the correct LST")
    #plt.savefig("HG_patch_Haslam_408.png")

    T_H = np.nanmean(HG_patch_haslam_408)
    T_R = tR * (408./ freq_0) ** -beta_R
    #print(f"T_H: {T_H}, T_R: {T_R}")

    T_H_minus_T_R_scaled = np.zeros(len(freqs))
    for i, freq in enumerate(freqs): 
        T_H_minus_T_R_scaled[i] = (T_H - T_R) * (freq / freq_0)**-beta_HG
        #print(T_H_minus_T_R_scaled[i])


    integral_omega = np.zeros(len(freqs))
    integral_omega_prime = np.zeros(len(freqs))

    if use_beam:
        #A = np.zeros(len(freqs))
        for i, freq in enumerate(freqs):
            beam = bm.generate_beam(freq, lst)
            A = rotator_GC.rotate_map_pixel(beam)
            A[A < 0] = np.nan
            integral_omega[i] = np.nansum(A)
            #print(f"Integral Omega at {freq}: {integral_omega[i]}")
            integral_omega_prime[i] = np.nansum(T_H_minus_T_R_scaled[i] * A * high_mask)
            #print(f"Integral Omega Prime at {freq}: {integral_omega_prime[i]}")
            #hp.mollview(T_H_minus_T_R_scaled[i] * A * high_mask, title="TH - TR * A")
            #plt.savefig("beamy.png")
     
    else:
        for i, freq in enumerate(freqs):
            integral_omega[i] = np.sum(~np.isnan(haslam_at_LST))
            integral_omega_prime[i] = T_H_minus_T_R_scaled[i]

    return integral_omega_prime/integral_omega

def prepare_high_lat_haslam(nside, observer, freqs, freq_0, lst, use_beam=True):
    # load and rotate haslam map
    haslam_408_512_map = hp.read_map("haslam408.fits")
    haslam_map = hp.ud_grade(haslam_408_512_map, nside)
    celestial_haslam = rotator_GC.rotate_map_pixel(haslam_map)

    #hp.mollview(haslam_map, title="Haslam Map in Equitorial Coordinates", min=0, max=200)
    #hp.graticule()
    #plt.savefig("test_results/haslam_equitortl.png")
    #plt.clf()

    #hp.mollview(celestial_haslam, title="Haslam Map in Celestial Coordinates", min=0, max=200)
    #hp.graticule()
    #plt.savefig("test_results/haslam_celestial.png")
    #plt.clf()


    # create high-lat mask in celestial coords
    low_mask = create_mask(nside, -70, -90, mask_type="celestial")
    high_mask = np.where(np.isnan(low_mask), 1, np.nan)

    # apply LST mask
    LST_mask = LST_map(np.ones_like(celestial_haslam), observer, lst)
    haslam_at_LST = LST_map(celestial_haslam, observer, lst)

    # extract patch
    HG_patch_haslam_408 = celestial_haslam * high_mask * LST_mask 

    # optionally precompute beams
    beams = []
    rotated_beams = []
    if use_beam:
        for freq in freqs:
            beam = bm.generate_beam(freq, lst)
            rotated = rotator_GC.rotate_map_pixel(beam)
            rotated[rotated < 0] = np.nan
            rotated_beams.append(rotated)

    # return all precomputed quantities
    return {
        "HG_patch_haslam_408": HG_patch_haslam_408,
        "high_mask": high_mask,
        "rotated_beams": rotated_beams if use_beam else None,
        "use_beam": use_beam,
        "freqs": freqs,
        "freq_0": freq_0
    }

def compute_high_lat_haslam(precomputed, tR, beta_R, beta_HG):
    T_H = np.nanmean(precomputed["HG_patch_haslam_408"])
    T_R = tR #* (408. / precomputed["freq_0"]) ** -beta_R

    freqs = precomputed["freqs"]
    #T_H_minus_T_R_scaled = np.zeros(len(freqs))
    #for i, freq in enumerate(freqs): 
    #    T_H_minus_T_R_scaled[i] = (T_H - T_R) * (freq / precomputed["freq_0"])**-beta_HG

    for i, freq in enumerate(freqs): 
        scaled_map = (precomputed["HG_patch_haslam_408"] - T_R) * (freq / precomputed["freq_0"])**-beta_HG
        #hp.mollview(scaled_map, title=f"Haslam Patch Scaled at {int(freq)} MHz", min=0, max= 1000)
        #hp.graticule()
        #plt.savefig(f"test_results/high_mask_and_LST_{int(freq)}MHz.png")
# CORRECTION HERE
    integral_omega = np.zeros(len(freqs))
    integral_omega_prime = np.zeros(len(freqs))

    if precomputed["use_beam"]:
        for i, freq in enumerate(freqs):
            A = precomputed["rotated_beams"][i]
            integral_omega[i] = np.nansum(A)
            #integral_omega_prime[i] = np.nansum(T_H_minus_T_R_scaled[i] * A * precomputed["high_mask"])
            integral_omega_prime[i] = np.nansum(scaled_map * A)
    else:
        for i, freq in enumerate(freqs):
            integral_omega[i] = np.sum(~np.isnan(precomputed["HG_patch_haslam_408"]))
            integral_omega_prime[i] = T_H_minus_T_R_scaled[i]

    return integral_omega_prime / integral_omega
'''

def prepare_high_lat_haslam(nside, observer, freqs, freq_0, lst, use_beam=True, save_plots=False):
    #print("Loading Haslam map...")
    haslam_408_512_map = hp.read_map("gsm2016.fits")#("haslam408.fits")
    #print(f"Original map size: {len(haslam_408_512_map)} pixels")

    haslam_map = hp.ud_grade(haslam_408_512_map, nside)
    #print(f"Upgraded map size: {len(haslam_map)} pixels")

    celestial_haslam = rotator_GC.rotate_map_pixel(haslam_map)
    #print("Applied celestial rotation")

    if save_plots:
        hp.mollview(haslam_map, title="Haslam Map Equatorial", min=0, max=200)
        hp.graticule()
        plt.savefig("test_results/haslam_equatorial.png")
        plt.clf()

        hp.mollview(celestial_haslam, title="Haslam Map Celestial", min=0, max=200)
        hp.graticule()
        plt.savefig("test_results/haslam_celestial.png")
        plt.clf()

    # high-latitude mask
    low_mask = create_mask(nside, -70, -90, mask_type="celestial")
    #print(f"Low mask created: {np.sum(~np.isnan(low_mask))} valid pixels")
    
    high_mask = np.where(np.isnan(low_mask), 1, np.nan)
    #print(f"High mask created: {np.nansum(high_mask)} pixels")

    # LST mask
    LST_mask = LST_map(np.ones_like(celestial_haslam), observer, lst)
    haslam_at_LST = LST_map(celestial_haslam, observer, lst)
    #print(f"LST mask applied: {np.nansum(LST_mask)} visible pixels")

    HG_patch_haslam_408 = celestial_haslam * high_mask * LST_mask
    #print(f"High-lat patch mean: {np.nanmean(HG_patch_haslam_408)}, min: {np.nanmin(HG_patch_haslam_408)}, max: {np.nanmax(HG_patch_haslam_408)}")

    beams = []
    rotated_beams = []
    if use_beam:
        for freq in freqs:
            beam = bm.generate_beam(freq, lst)
            rotated = rotator_GC.rotate_map_pixel(beam)
            rotated[rotated < 0] = np.nan
            rotated_beams.append(rotated)
            #print(f"Beam at {freq} MHz: mean={np.nanmean(rotated)}, min={np.nanmin(rotated)}, max={np.nanmax(rotated)}")

    return {
        "HG_patch_haslam_408": HG_patch_haslam_408,
        "high_mask": high_mask,
        "rotated_beams": rotated_beams if use_beam else None,
        "use_beam": use_beam,
        "freqs": freqs,
        "freq_0": freq_0
    }

def compute_high_lat_haslam(precomputed, tR, beta_R, beta_HG, save_plots=False):
    """
    Compute high galactic latitude contribution according to:
    
    HighLat(nu) = ∫ [T_H(nu0) - T_R(nu0)] * (nu/nu0)^β_HG * A(nu) dΩ' / ∫ A(nu) dΩ
    
    Parameters
    ----------
    precomputed : dict
        Output of prepare_high_lat_haslam
    tR : float
        Reference extragalactic radio background temperature at freq_0
    beta_R : float
        Spectral index of radio excess (not used here but included for API)
    beta_HG : float
        Spectral index of high-latitude component
    save_plots : bool
        Save diagnostic plots for each frequency

    Returns
    -------
    np.ndarray
        High-latitude temperature for each frequency
    """

    HG_patch = precomputed["HG_patch_haslam_408"]
    freqs = precomputed["freqs"]
    freq_0 = precomputed["freq_0"]
    use_beam = precomputed["use_beam"]

    result = np.zeros(len(freqs))

    for i, freq in enumerate(freqs):
        # 1) subtract T_R at reference frequency
        map_minus_TR = HG_patch - tR  # T_H(nu0) - T_R(nu0)

        # 2) scale the map pixelwise
        scaled_map = map_minus_TR * (freq / freq_0) ** -beta_HG

        if save_plots:
            hp.mollview(scaled_map, title=f"Scaled high-lat map at {int(freq)} MHz", min=-100, max=500)
            hp.graticule()
            plt.savefig(f"test_results/high_lat_scaled_{int(freq)}MHz.png")
            plt.clf()

        # 3) get beam
        if use_beam:
            A = precomputed["rotated_beams"][i]
            # numerator: ∫ (scaled_map * A) ignoring NaNs
            numerator = np.nansum(scaled_map * A)
            # denominator: ∫ A ignoring NaNs
            denominator = np.nansum(A)
        else:
            numerator = np.nansum(scaled_map)
            denominator = np.sum(~np.isnan(scaled_map))

        if denominator == 0:
            print(f"Warning: denominator zero at {freq} MHz! Setting result to nan")
            result[i] = np.nan
        else:
            result[i] = numerator / denominator

        # debug print
        #print(f"{freq} MHz: numerator={numerator}, denominator={denominator}, mean={result[i]}")

    return result



#############################   THIS MAKES & FITS THE LOW LATITUDE DATA USING 2 SPIX #############################################

def compute_low_latitude_components_multispix(nside, observer, freqs, lst, threshold=100.0, use_beam=True):
    haslam_408_512_map = hp.read_map("gsm2016.fits")#("haslam408.fits")
    haslam_map = hp.ud_grade(haslam_408_512_map, nside)
    celestial_haslam = rotator_GC.rotate_map_pixel(haslam_map)

    low_mask = create_mask(nside, -70, -90, mask_type="celestial")
    T_408 = celestial_haslam * low_mask
     
    # new: create brightness masks
    bright_mask = (T_408 >= threshold) & (~np.isnan(T_408))
    dim_mask = (T_408 < threshold) & (~np.isnan(T_408))


    # visualize bright and dim masks
    bright_map = np.where(bright_mask, T_408, np.nan)
    dim_map = np.where(dim_mask, T_408, np.nan)

    #hp.mollview(bright_map, title="Bright pixels (>= threshold)", min = 0, max = 200)
    #hp.graticule()
    #plt.savefig("test_results/bright_mask.png")

    #hp.mollview(dim_map, title="Dim pixels (< threshold)", min=0, max = 200)
    #hp.graticule()
    #plt.savefig("test_results/dim_mask.png")


    LST_mask = LST_map(np.ones_like(celestial_haslam), observer, lst)
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
    
    return T_408, integral_omega_408, LST_mask, A, integral_omega, bright_mask, dim_mask

def compute_low_latitude_multispix(freqs, beta_plane, beta_outer, components, use_beam=True):
    T_408, _, LST_mask, A, integral_omega, bright_mask, dim_mask = components
    low_lat = np.zeros(len(freqs))

    for i, freq in enumerate(freqs):
        T_LST = np.full_like(T_408, np.nan)
        T_LST[bright_mask] = T_408[bright_mask] * (freq / 408.0) ** -beta_plane
        T_LST[dim_mask] = T_408[dim_mask] * (freq / 408.0) ** -beta_outer
        T_LST = T_LST * LST_mask

        if use_beam:
            integral_omega_prime = np.nansum(T_LST * A[i])
        else:
            integral_omega_prime = np.nansum(T_LST)
            integral_omega[i] = np.count_nonzero(~np.isnan(T_LST))

        low_lat[i] = integral_omega_prime / integral_omega[i]

    return low_lat

def compute_low_latitude(freqs, beta_plane, beta_outer, components, use_beam=True):
    """
    Computes the low-latitude temperature contribution for bright (plane) 
    and dim (outer) regions separately, then returns total, plane, and outer.

    Parameters
    ----------
    freqs : array_like
        Frequencies in MHz.
    beta_plane : float
        Spectral index for bright/plane region.
    beta_outer : float
        Spectral index for dim/outer region.
    components : tuple
        Tuple containing:
        T_408 : np.array, Haslam map at 408 MHz
        _ : placeholder (not used)
        LST_mask : np.array, LST visibility mask
        A : np.array, beam weights per frequency
        integral_omega : np.array, normalization for integration
        bright_mask : np.array, boolean mask for bright region
        dim_mask : np.array, boolean mask for dim region
    use_beam : bool, optional
        Whether to weight by the beam (default True).

    Returns
    -------
    total_low_lat : np.array
        Low-latitude temperature integrated over all regions.
    plane_low_lat : np.array
        Contribution from bright (plane) region.
    outer_low_lat : np.array
        Contribution from dim (outer) region.
    """
    
    T_408, _, LST_mask, A, integral_omega, bright_mask, dim_mask = components
    nfreq = len(freqs)
    
    # initialize outputs
    total_low_lat = np.zeros(nfreq)
    plane_low_lat = np.zeros(nfreq)
    outer_low_lat = np.zeros(nfreq)
    
    for i, freq in enumerate(freqs):
        # compute plane and outer separately
        T_plane = np.full_like(T_408, np.nan)
        T_outer = np.full_like(T_408, np.nan)
        
        T_plane[bright_mask] = T_408[bright_mask] * (freq / 408.0) ** -beta_plane
        T_outer[dim_mask] = T_408[dim_mask] * (freq / 408.0) ** -beta_outer
        
        # apply LST mask
        T_plane *= LST_mask
        T_outer *= LST_mask
        
        if use_beam:
            plane_sum = np.nansum(T_plane * A[i])
            outer_sum = np.nansum(T_outer * A[i])
        else:
            plane_sum = np.nansum(T_plane)
            outer_sum = np.nansum(T_outer)
            integral_omega[i] = np.count_nonzero(~np.isnan(T_plane + T_outer))
        
        plane_low_lat[i] = plane_sum / integral_omega[i]
        outer_low_lat[i] = outer_sum / integral_omega[i]
        total_low_lat[i] = plane_low_lat[i] + outer_low_lat[i]
    
    return total_low_lat, plane_low_lat, outer_low_lat

def compute_low_latitude_multispix_tR(freqs, beta_plane, beta_outer, components, tR, use_beam=True):

    T_408, _, LST_mask, A, integral_omega, bright_mask, dim_mask = components

    # subtract tR at 408 MHz (copy so original is untouched)
    T_outer_408 = T_408.copy()
    T_outer_408[dim_mask] -= tR

    low_lat = np.zeros(len(freqs))

    for i, freq in enumerate(freqs):

        T_LST = np.full_like(T_408, np.nan)

        # plane (no tR subtraction)
        T_LST[bright_mask] = (
            T_408[bright_mask] *
            (freq / 408.0) ** -beta_plane
        )

        # outer (tR subtracted at 408 first)
        T_LST[dim_mask] = (
            T_outer_408[dim_mask] *
            (freq / 408.0) ** -beta_outer
        )

        T_LST *= LST_mask

        if use_beam:
            numerator = np.nansum(T_LST * A[i])
        else:
            numerator = np.nansum(T_LST)
            integral_omega[i] = np.count_nonzero(~np.isnan(T_LST))

        low_lat[i] = numerator / integral_omega[i]

    return low_lat

def beam_calibration_error(amp, period, freqs, shift):
    freqs = np.array(freqs)
    y = 1 + amp * np.sin(2 * np.pi * (freqs - shift) / period)
    return y

def compute_low_latitude_plane(freqs, beta_plane, components, use_beam=True):
    """
    Computes the low-latitude contribution from the bright/plane region only.

    Returns
    -------
    plane_low_lat : np.array
        Contribution from bright (plane) region.
    """
    T_408, _, LST_mask, A, integral_omega, bright_mask, _ = components
    nfreq = len(freqs)
    plane_low_lat = np.zeros(nfreq)

    for i, freq in enumerate(freqs):
        # compute bright region
        T_plane = np.full_like(T_408, np.nan)
        T_plane[bright_mask] = T_408[bright_mask] * (freq / 408.0) ** -beta_plane
        T_plane *= LST_mask

        if use_beam:
            plane_sum = np.nansum(T_plane * A[i])
        else:
            plane_sum = np.nansum(T_plane)
            integral_omega[i] = np.count_nonzero(~np.isnan(T_plane))

        plane_low_lat[i] = plane_sum / integral_omega[i]

    return plane_low_lat


def compute_low_latitude_outer(freqs, beta_outer, components, use_beam=True):
    """
    Computes the low-latitude contribution from the dim/outer region only.

    Returns
    -------
    outer_low_lat : np.array
        Contribution from dim (outer) region.
    """
    T_408, _, LST_mask, A, integral_omega, _, dim_mask = components
    nfreq = len(freqs)
    outer_low_lat = np.zeros(nfreq)

    for i, freq in enumerate(freqs):
        # compute dim region
        T_outer = np.full_like(T_408, np.nan)
        T_outer[dim_mask] = T_408[dim_mask] * (freq / 408.0) ** -beta_outer
        T_outer *= LST_mask

        if use_beam:
            outer_sum = np.nansum(T_outer * A[i])
        else:
            outer_sum = np.nansum(T_outer)
            integral_omega[i] = np.count_nonzero(~np.isnan(T_outer))

        outer_low_lat[i] = outer_sum / integral_omega[i]

    return outer_low_lat
