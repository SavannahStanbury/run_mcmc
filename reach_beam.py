import numpy as np
import healpy as hp
import ephem
from scipy.interpolate import LinearNDInterpolator

pattern = np.load("reach_log_spiral_pattern.npz")["log_spiral"]
freq_axis = np.arange(pattern.shape[2])
nside = 64

_interps_cache = None

def transform(in_position, in_system, out_system, observer=None):
    observer = observer if observer else ephem.Observer()
    in_position = np.radians(in_position)

    if in_system == "galactic":
        gal = ephem.Galactic(in_position[0], in_position[1])
        eq = ephem.Equatorial(gal)
        ra = eq.ra
        dec = eq.dec
    elif in_system == "equatorial":
        ra, dec = in_position
    else:
        raise RuntimeError("unsupported input system")

    if out_system == "equatorial":
        return np.degrees([ra, dec])

    if out_system == "horizon":
        eq = ephem.Equatorial(ra, dec)
        body = ephem.FixedBody()
        body._ra = eq.ra
        body._dec = eq.dec
        body._epoch = eq.epoch
        body.compute(observer)
        return np.degrees([body.az, body.alt])

    raise RuntimeError("unsupported output system")


def build_beam_interp(pattern):
    fN = pattern.shape[2]
    interps = []

    for f in range(fN):
        dec = pattern[:, 0, f]
        az = pattern[:, 1, f]
        gain = np.clip(pattern[:, 2, f], 1e-12, None)

        log_gain = np.log(gain)
        points = np.column_stack((dec, az))

        interps.append(
            LinearNDInterpolator(points, log_gain, fill_value=-np.inf)
        )

    return interps


def get_freq_index(freq_axis, freq):
    return int(np.argmin(np.abs(freq_axis - freq)))


def generate_beam(freq, time):
    global _interps_cache

    if _interps_cache is None:
        _interps_cache = build_beam_interp(pattern)

    fidx = get_freq_index(freq_axis, freq)

    REACH = ephem.Observer()
    REACH.lat = '-30:50:19.5'
    REACH.long = '21:22:29.71'
    REACH.elevation = 1151
    REACH.date = time

    npix = hp.nside2npix(nside)
    ipix = np.arange(npix)
    l, b = hp.pix2ang(nside, ipix, lonlat=True)

    ra_dec = np.array([
        transform((l[i], b[i]), 'galactic', 'equatorial', observer=REACH)
        for i in range(npix)
    ])

    az_alt = np.array([
        transform((ra_dec[i, 0], ra_dec[i, 1]), 'equatorial', 'horizon', observer=REACH)
        for i in range(npix)
    ])

    dec_val = 90.0 - az_alt[:, 1]
    az_val = az_alt[:, 0]

    pts = np.column_stack((dec_val, az_val))

    log_g = _interps_cache[fidx](pts)

    beam = np.exp(log_g)
    beam[~np.isfinite(beam)] = hp.UNSEEN

    return beam