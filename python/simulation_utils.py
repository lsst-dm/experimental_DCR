
import imp
import os

import numpy as np
import pickle
import galsim

from lsst.afw.geom import Angle
from lsst.sims.catUtils.matchSED.matchUtils import matchStar

imp.load_source('calc_refractive_index', '/Users/sullivan/LSST/code/StarFast/calc_refractive_index.py')
imp.load_source('fast_dft', '/Users/sullivan/LSST/code/StarFast/fast_dft.py')
imp.load_source('StarFast', '/Users/sullivan/LSST/code/StarFast/StarFast.py')
from StarFast import StarSim  # noqa E402


lsst_lat = Angle(np.radians(-30.244639))
lsst_lon = Angle(np.radians(-70.749417))
lsst_alt = 2663.
filter_name = 'g'


def simulation_wrapper(sim=None, seed=7, band_name='g', dimension=1024, pixel_scale=0.25, psf_fwhm=0.25,
                       n_star=10000, hottest_star='B', coolest_star='M',
                       n_quasar=1000, quasar_seed=None,
                       wavelength_step=3., use_bandpass=True,
                       ra_offset=Angle(0.), dec_offset=Angle(0.), sky_rotation=0., exposureId=0,
                       instrument_noise=0., photon_noise=1./15, sky_noise=0.,
                       elevation_min=30., elevation_max=90., elevation_step=5.,
                       attenuation=1., do_simulate=True,
                       output_directory=None, write_catalog=False, write_fits=False):
    """Call StarFast and generate realistic simulations of stars and quasars.

    Parameters
    ----------
    sim : `StarFast.StarSim`, optional
        Optionally supply a previously-generated simulation.
    seed : `int`, optional
        Seed for the random number generator.
        Simulations with the same seed are reproduceable
    band_name : `str`, optional
        Name of the filter.
    dimension : `int`, optional
        Number of pixels on a side for the simulated image
    pixel_scale : `float`, optional
        plate scale, in arcseconds/pixel
    psf_fwhm : `float`, optional
        FWHM of the PSF, in arcseconds
    n_star : `int`, optional
        Number of stars to model in the simulated catalog.
        The catalog covers an area ~4x larger than the area,
        to allow simulated rotations and dithering observations
    hottest_star : `str`, optional
        Hottest star to include (types are 'OBAFGKMR')
    coolest_star : `str`, optional
        Coolest star to include
    n_quasar : `int`, optional
        Number of quasars to model in the simulated catalog.
    quasar_seed : `int`, optional
        Seed for the random number generator.
        Simulations with the same seed are reproduceable
    wavelength_step : `float`, optional
        Wavelength resolution of the spectra and calculation of filter and DCR effects. In nm.
    use_bandpass : `bool`, optional
        Include the LSST filter throughput.
    ra_offset : `lsst.afw.geom.Angle`, optional
        Additional offset in RA from the field center, for dithering.
        In radians as an LSST Angle object
    dec_offset : `lsst.afw.geom.Angle`, optional
        Additional offset in Dec from the field center, for dithering.
        In radians as an LSST Angle object
    sky_rotation : `float`, optional
        Sky rotation angle, in Degrees. I realize this is different than RA and Dec
    exposureId : `int`, optional
        Unique exposure identification number. Also used as the "OBSID"
    instrument_noise : `float`, optional
        Adds noise akin to instrumental noise (post-PSF).
        Set to 1.0 for default value, can be scaled up or down
    photon_noise : `float`, optional
        Adds poisson noise akin to photon shot noise.
        Set to 1.0 for default value, can be scaled up or down
    sky_noise : `float`, optional
        Adds noise prior to convolving with the PSF.
    elevation_min : `float`, optional
        Minimum observation elevation angle to simulate, in degrees
    elevation_max : `float`, optional
        Open maximum observation angle, in degrees.
        Only angles less than elevation_max will be simulated
    elevation_step : `float`, optional
        Elevation angle step size, in degrees.
    attenuation : `float`, optional
        Attenuation factor to be used in the simulations to decrease flux.
    do_simulate : `bool`, optional
        If set, construct the the raw simulation.
        Turn off to save time if you only want the catalog.
    output_directory : `str`, optional
        Path to the directory to save output
    write_catalog : `bool`, optional
        Write a reference catalog using the simulated sources?
    write_fits : `bool`, optional
        Generate realizations of the simulation under different observing conditions, and write the fits files

    Returns
    -------
    `StarFast.StarSim` : simulation
        Gridded simulation of the "true" sky, before convolving
        with the instrument's beam or modeling atmospheric refraction.
    """
    if output_directory is None:
        output_directory = "/Users/sullivan/LSST/simulations/test%1i_quasars3nm/" % seed
    band_dict = {'u': 0, 'g': 1, 'r': 2, 'i': 3, 'z': 4, 'y': 5}  # LSST filter numbers used by the butler
    if sim is None:
        ra = lsst_lon + ra_offset
        dec = lsst_lat + dec_offset
        if quasar_seed is None:
            quasar_seed = seed + 1
        pickle_file = "sed_list.pickle"
        if os.path.exists(pickle_file):
            with open(pickle_file, 'rb') as dumpfile:
                sed_list = pickle.load(dumpfile)
        else:
            matchStarObj = matchStar()
            sed_list = matchStarObj.loadKuruczSEDs()
            with open(pickle_file, 'wb') as dumpfile:
                pickle.dump(sed_list, dumpfile)

        gsp = galsim.GSParams(folding_threshold=1.0/dimension, maximum_fft_size=12288)
        psf = galsim.Kolmogorov(fwhm=psf_fwhm/pixel_scale, flux=1, gsparams=gsp)
        sim = StarSim(psf=psf, pixel_scale=pixel_scale, x_size=dimension, y_size=dimension,
                      band_name=band_name, wavelength_step=wavelength_step,
                      sed_list=sed_list, ra=ra, dec=dec, sky_rotation=sky_rotation,
                      use_mirror=use_bandpass, use_lens=use_bandpass, use_atmos=use_bandpass,
                      use_filter=use_bandpass, use_detector=use_bandpass, attenuation=attenuation)

        if n_star > 0:
            # Simulate a catalog of stars, with fluxes and SEDs
            sim.load_catalog(n_star=n_star, hottest_star=hottest_star, coolest_star=coolest_star, seed=seed)
            # Generate the raw simulation
            if do_simulate:
                sim.simulate()
        if n_quasar > 0:
            # Now additionally simulate a catalog of quasars
            sim.load_quasar_catalog(n_quasar=n_quasar, seed=quasar_seed)
            # Generate a new raw simulation and add to the previous one of stars
            if do_simulate:
                sim.simulate(useQuasars=True)
    if write_catalog:
        sim.make_reference_catalog(output_directory=output_directory + "input_data/",
                                   filter_list=['u', 'g', 'r', 'i', 'z'], magnitude_limit=16.0)
    if write_fits:
        expId = exposureId + 100*band_dict[band_name]
        for elevation in np.arange(elevation_min, elevation_max, elevation_step):
            for azimuth in [0.0, 180.0]:
                exposure = sim.convolve(elevation=elevation, azimuth=azimuth,
                                        instrument_noise=instrument_noise, sky_noise=sky_noise,
                                        photon_noise=photon_noise, exposureId=expId, obsid=expId)
                filename = "lsst_e_%3.3i_f%i_R22_S11_E000.fits" % (expId, band_dict[band_name])
                expId += 1
                exposure.writeFits(output_directory + "images/" + filename)
    return sim
