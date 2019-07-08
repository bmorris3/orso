import numpy as np
from fleck import Star, generate_spots
from batman import TransitParams, TransitModel
import astropy.units as u
import h5py
from astropy.table import Table
from astropy.constants import R_sun, R_earth 
from scipy.stats import binned_statistic

archive = h5py.File('../data/kepler_lcs/archive.hdf5', 'r')

keys = list(archive)

koi_table = Table.read('../data/cumulative_2019.07.05_01.52.59.votable')
koi_table.add_index('kepid')
        
u_ld = [0.5079, 0.2239]
times = np.linspace(-0.1, 0.1, 300)
bin_times = np.linspace(-0.5, 0.5, 51)

# transit_model = TransitModel(params, times).light_curve(params)

star = Star(spot_contrast=0., u_ld=u_ld, rotation_period=26)

koi_stdevs = np.load('../data/oot_scatter.npy')

n_transits = 500
transits = []
residuals = []
spots_occulted = []

counter = 0

while len(transits) < n_transits: 
    counter += 1
    try: 
        kepid = keys[np.random.randint(0, len(keys))]

        bstr = str(int(kepid)).encode()

        props = koi_table.loc[bstr]

        if not isinstance(koi_table.loc[bstr]['kepid'], bytes): 
            props = props[np.argmax(props['koi_depth'])]

        period = props['koi_period']
        duration = props['koi_duration'] / 24

        epoch = props['koi_time0bk'] + 2454833
        b = props['koi_impact']

        params = TransitParams()
        params.per = period
        params.t0 = 0
        params.duration = duration
        params.rp = float(props['koi_prad']*R_earth/(props['koi_srad']*R_sun))
        a = (np.sin(duration * np.pi / period) / np.sqrt((1 + params.rp)**2 - b**2))**-1
        params.a = a
        params.inc = np.degrees(np.arccos(b/params.a))
        params.limb_dark = 'quadratic'
        params.u = u_ld
        params.ecc = 0
        params.w = 90
        
        stddev = koi_stdevs[np.random.randint(0, len(koi_stdevs))]
        
        if params.rp**2 > 0.005 and stddev < params.rp**2 and duration < 0.2: 

            if counter % 2 == 0: 
                n_spots = 50
            else: 
                n_spots = 0
            spot_lons, spot_lats, spot_radii, inc_stellar = generate_spots(-90, 90, 0.1, n_spots, inclinations=90*u.deg)
            lc, so = star.light_curve(spot_lons, spot_lats, spot_radii,
                                      inc_stellar, planet=params, times=times, 
                                      return_spots_occulted=True, fast=True)
            # Add Kepler-scale white noise
            lc += stddev * np.random.randn(len(lc))[:, np.newaxis]
            # Add single-point positive outliers
            lc[np.random.randint(0, len(lc), size=2)] *= 1.001
            lc[np.random.randint(0, len(lc), size=2)] *= 0.999


#             interped_lc = np.interp(interp_times, times / duration, lc[:, 0])
            bs = binned_statistic(times / duration, lc[:, 0], bins=bin_times, statistic='median')
            interped_lc = bs.statistic
            interped_lc -= interped_lc.mean()
            interped_lc /= interped_lc.ptp()
            
            transits.append(interped_lc)
            spots_occulted.append(so)
    except KeyError:
        pass

#     lc = lc[:, 0] - transit_model + 1e-4 * np.random.randn(len(times))
    
#     fit = np.polyval(np.polyfit(times - times.mean(), lc, 3), times - times.mean())
#     residual = (lc - fit) / params.rp**2
#     transits.append(residuals)
#     residuals.append(residual)

# import matplotlib.pyplot as plt
# plt.plot(np.vstack(transits).T)
# plt.show()


rand = np.random.rand()

np.save('data/parallel_normed/{0:09d}_simulated_transit_lcs.npy'.format(int(1e9*rand)), np.vstack(transits).T)
np.save('data/parallel_normed/{0:09d}_simulated_spots_occulted.npy'.format(int(1e9*rand)), spots_occulted)