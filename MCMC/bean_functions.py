import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import healpy as hp
from scipy import stats
import ephem
from scipy.integrate import quad
from astropy.time import Time
from scipy.interpolate import interp1d

def transform(in_position, in_system, out_system, observer=None):
    """Simple coordinate transforms between commonly used coordinate systems.
    
    position: input position as a float tuple (longitude, latitude) in degrees
    in_system, out_system: coordinate system strings; one of
        horizon:    Alt,  Az
        equatorial: RA,   DEC
        galactic:   GLON, GLAT
    observer: ephem.Observer (needed when converting to / from horizon system
    
    Returns: transformed input position as a float tuple (longitude, latitude) in degrees
    See http://stackoverflow.com/questions/11169523/how-to-compute-alt-az-for-given-galactic-coordinate-glon-glat-with-pyephem
    """
    # Set the default observer
    observer = observer if observer else HESS()
    # Internally use radians;
    in_position = np.radians(in_position)
    
    # Transform in_position to Equatorial coordinates ra, dec:
    if in_system == 'horizon':
        ra, dec = map(float, observer.radec_of(in_position[0], in_position[1]))
    elif in_system == 'equatorial':
        ra, dec = in_position
    elif in_system == 'galactic':
        galactic = ephem.Galactic(in_position[0], in_position[1])
        equatorial = ephem.Equatorial(galactic)
        ra, dec = equatorial.ra.real, equatorial.dec.real
    else:
        raise RuntimeError('in_system = %s not supported' % in_system)

    # Here we have ra, dec in radians as floats

    # Now transform Equatorial coordinates to out_system:
    if out_system == 'horizon':
        equatorial = ephem.Equatorial(ra, dec)
        body = ephem.FixedBody()
        body._ra = equatorial.ra
        body._dec = equatorial.dec
        body._epoch = equatorial.epoch
        body.compute(observer)
        out_position = body.az, body.alt
    elif out_system == 'equatorial':
        out_position = ra, dec
    elif out_system == 'galactic':
        equatorial = ephem.Equatorial(ra, dec)
        galactic = ephem.Galactic(equatorial)
        out_position = galactic.lon.real, galactic.lat.real
    else:
        raise RuntimeError('out_system = %s not supported' % out_system)
    
    # Clip longitude to 0 .. 360 deg range
    if out_position[0] > 360:
        out_position[0] = out_position[0] - 360

    # Return out position in degrees
    return np.degrees(out_position) 





def equ2hor(ra, dec, LST, lat0, zenith=False):
    """
    Convert Equatorial RA DEC to horizontal coordinates 
    The coordinates should be given in radiants
    NEED to specify LST, lat0 in hour angle and degree
    """
    h=np.radians(LST-ra/(2*np.pi)*24) #hour angle
    lat0=np.radians(lat0)
    A = np.arctan2(np.sin(h),(np.cos(h)*np.sin(lat0)-np.tan(dec)*np.cos(lat0)))
    sina = np.sin(lat0)*np.sin(dec)+np.cos(lat0)*np.cos(dec)*np.cos(h)
    a=np.arcsin(sina) 
    if zenith==True: return  A, np.pi/2-a
    else: return A, a


def times2lst(times):
    lst=np.zeros(len(times))
    for t in range(len(times)):
        lst[t]=jd2lst(times[t])
    return lst


def jd2lst(jd):
    jdt = Time(jd, format='jd',location=('21d', '-30d'))
    #jdt.delta_ut1_utc = 0.
    lst = jdt.sidereal_time('apparent').to_string('hour').split('m')
    #lst = lst[0].split('h') uncomment if mins not needed
    return lst[0]


def func_A(h,theta,phi):
    return 2*np.sin(2*np.pi*h*np.cos(theta))*np.sqrt(1-(np.sin(theta)*np.sin(phi))**2)


def p_i(nu,theta,a):
    return (1-(theta/(np.pi/2))**a[0])*(np.cos(theta))**a[1]+a[2]*(theta/(np.pi/2))*(np.cos(theta))**a[3]


def func_LWA(nu,theta,phi,aE,aH):
    return np.sqrt((p_i(nu,theta,aE)*np.cos(phi))**2+(p_i(nu,theta,aH)*np.sin(phi))**2)


def a_pol_fromfit(coef_a,nu):
    #nu in Mhz
    N=len(coef_a[:,0])
    M=len(coef_a[0,:])
    a_nu=np.zeros(M)
    for j in range(M):
        for n in range(N):
            a_nu[j]+=coef_a[n,j]*(nu/10)**n
    return a_nu


def a_interpol(filename,nu,pol):
    #nu in Mhz
    key='a_'+pol
    lcoef=np.load(filename)
    freq=lcoef['freq']
    a=lcoef[key]
    f0=interp1d(freq, a[:,0], kind='cubic')
    f1=interp1d(freq, a[:,1], kind='cubic')
    f2=interp1d(freq, a[:,2], kind='cubic')
    f3=interp1d(freq, a[:,3], kind='cubic')
    a_nu=[f0(nu),f1(nu),f2(nu),f3(nu)]
    return np.array(a_nu)


def a_fit(filename,nu,pol):
    #nu in Mhz
    key='a_'+pol
    lcoef=np.load(filename)
    freq=lcoef['freq']
    a=lcoef[key]
    f0=np.polyfit(freq, a[:,0], 3)
    f1=np.polyfit(freq, a[:,1], 3)
    f2=np.polyfit(freq, a[:,2], 3)
    f3=np.polyfit(freq, a[:,3], 3)
    a_nu=[cube(nu,f0),cube(nu,f1),cube(nu,f2),cube(nu,f3)]
    return np.array(a_nu)


def hasnu(nside,haslam,alpha,freq):
    #nu in MHz
    m=hp.pixelfunc.ud_grade(haslam,nside)
    return m*(freq/408)**(alpha)


def h(freq):
    #freq in MHz
    C=3e8 #in m/s
    return 0.8/(C/(freq*1e6))


def gen_beam_LWA(nside,date,freq,aE,aH):
    REACH = ephem.Observer()
    REACH.lat, REACH.long, REACH.elevation = '-30:50:19.5', '21:22:29.71', 1151
    horizon=90.
    npix=hp.nside2npix(nside)
    ipix=np.arange(npix)
    l, b=hp.pix2ang(nside, ipix, lonlat=True) 
    beam=np.zeros(npix)
    REACH.date=date
    for i in range(npix):
        ra, dec= transform((l[i],b[i]), 'galactic', 'equatorial', observer=REACH)
        A, alt= transform((ra,dec), 'equatorial', 'horizon', observer=REACH)
        z = 90.- alt
        if (z>horizon): 
            beam[i]=0    #use hp.UNSEEN for plotting purposes
        else: 
            beam[i]=func_LWA(freq,np.radians(z),np.radians(A),aE,aH)
    return beam



def cube(x,pars):
    return pars[0]*x**3+pars[1]*x**2+pars[2]*x+pars[3]


def gen_horizon(nside,date):
    REACH = ephem.Observer()
    REACH.lat, REACH.long, REACH.elevation = '-30:50:19.5', '21:22:29.71', 1151
    horizon=90.
    npix=hp.nside2npix(nside)
    ipix=np.arange(npix)
    l, b=hp.pix2ang(nside, ipix, lonlat=True) 
    beam=np.zeros(npix)
    REACH.date=date
    for i in range(npix):
        ra, dec= transform((l[i],b[i]), 'galactic', 'equatorial', observer=REACH)
        A, alt= transform((ra,dec), 'equatorial', 'horizon', observer=REACH)
        z = 90.- alt
        if (z>horizon): 
            beam[i]=0    #use hp.UNSEEN for plotting purposes
        else: 
            beam[i]=1.
    return beam
    
    
def a_value(filename,nu,pol):
    #nu in Hz
    nu=nu*1e6
    beamDict = np.load(filename)
    if pol=='EH': beamCoeff=beamDict['fitX']
    elif pol=='HE': beamCoeff=beamDict['fitY']
    a_nu=np.zeros((2,4))
    for i in range(2):
        for j in range(4):
            a_nu[i,j]=np.polyval(beamCoeff[i,j,:], nu)
    return a_nu
