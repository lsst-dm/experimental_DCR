
import imp
import os
import sqlite3

import numpy as np
import pickle
from shutil import copy as copyfile

from lsst.afw.geom import Angle, arcseconds
import galsim
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
                       simulation_size=1.,
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
    simulation_size : `float`, optional
        Size of the underlying simulated sky relative to the size of the output images.
        Increase to allow a greater number of images to be mosaiced.
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
    # if output_directory is None:
    #     output_directory = "/Users/sullivan/LSST/simulations/test%1i_quasars3nm/" % seed
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
        x_size_use = sim.coord.xsize(base=True)  # / 2.0
        y_size_use = sim.coord.ysize(base=True)  # / 2.0
        if sim.edge_dist is not None:
            x_size_use += sim.edge_dist
            y_size_use += sim.edge_dist
        sky_radius = np.sqrt(x_size_use**2.0 + y_size_use**2.0)*sim.wcs.getPixelScale().asDegrees()
        if simulation_size > 1:
            sky_radius *= simulation_size

        if n_star > 0:
            # Simulate a catalog of stars, with fluxes and SEDs
            sim.load_catalog(n_star=n_star, hottest_star=hottest_star, coolest_star=coolest_star,
                             seed=seed, sky_radius=sky_radius)
            # Generate the raw simulation
            if do_simulate:
                sim.simulate()
        if n_quasar > 0:
            # Now additionally simulate a catalog of quasars
            sim.load_quasar_catalog(n_quasar=n_quasar, seed=quasar_seed, sky_radius=sky_radius)
            # Generate a new raw simulation and add to the previous one of stars
            if do_simulate:
                sim.simulate(useQuasars=True)
            sim.seed = seed
    if write_catalog:
        sim.make_reference_catalog(output_directory=output_directory + "input_data/",
                                   filter_list=[band_name, ], magnitude_limit=16.0)
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


class OpSim_wrapper:

    """Connect to the OpSim database and use the observing conditions
    to generate sets of simulated data
    """

    def __init__(self, opsim_db=None, year=1, filt='g', sim_directory=None, airmass_threshold=None,
                 conditions_db=None, field_radius=1.5):
        if opsim_db is None:
            opsim_db = '/Users/sullivan/LSST/OpSim/baseline2018a.db'
        if sim_directory is None:
            self.sim_directory = "/Users/sullivan/LSST/simulations/OpSim/baseline2018a/"
        else:
            self.sim_directory = sim_directory
        sql_connection = sqlite3.connect(opsim_db)
        opsim = sql_connection.cursor()
        if conditions_db is not None:
            sql_connection = sqlite3.connect(conditions_db)
            self.opsim = sql_connection.cursor()
            self.conditions_flag = True
            self.field_radius = field_radius
        else:
            self.conditions_flag = False
            self.field_radius = None
        self.filter = filt
        self.year = year
        night0 = 365*(year - 1)
        night1 = 365*year
        query = "select fieldId, fieldRA, fieldDec from SummaryAllProps where "\
                "night > %i and night < %i and filter = '%s';"\
                % (night0, night1, filt)
        result = opsim.execute(query)
        field_list0 = result.fetchall()
        self.field_list = np.array([res[0] for res in field_list0])
        self.field_Ids = list(set(self.field_list))
        if airmass_threshold is not None:
            # Optionally only include fields with at least one observation
            # at an airmass above the given threshold
            # NOTE: The database is queried twice. The first time gathers all
            # observations of each field, while this second time only gathers
            # those observations above the airmass_threshold.
            # It is done this way so that all observations of a field are
            # returned, as long as at least one is above the threshold.
            query = "select fieldId, fieldRA, fieldDec from SummaryAllProps where "\
                    "night > %i and night < %i and filter = '%s' and airmass > %f;"\
                    % (night0, night1, filt, airmass_threshold)
            result = opsim.execute(query)
            field_list0 = result.fetchall()
            field_list = np.array([res[0] for res in field_list0])
            self.field_Ids = list(set(field_list))
        self.ra_list = np.array([res[1] for res in field_list0])
        self.dec_list = np.array([res[2] for res in field_list0])
        self.field_ra = {}
        self.field_dec = {}
        for field in self.field_Ids:
            ra = self.ra_list[field_list == field]
            dec = self.dec_list[field_list == field]
            self.field_ra[field] = ra[0]
            self.field_dec[field] = dec[0]
        self.field_Id = None
        self.altitude = None
        self.azimuth = None
        self.airmass = None
        self.seeing = None

    def set_field(self, n_obs=None, index=0, max_n_obs=40, min_n_obs=4, year=None):
        """Pick a field from the observations database that meets all criteria.

        Parameters
        ----------
        n_obs : `int`, optional
            Select only fields with ``n_obs`` observations in the current year.
            No constraint on the number of observations if not set.
        index : `int`, optional
            Element of the list of fields matching the set of conditions to use.
            Wrapped at the number of matching fields.
        max_n_obs : `int`, optional
            Maximum number of observations for a field. Only used if ``n_obs`` is not set.
        min_n_obs : `int`, optional
            Minimum number of observations for a field. Only used if ``n_obs`` is not set.
        year : `int`, optional
            Optionally supply the year of the survey. Overrides the current ``self.year``
        """
        if year is not None:
            self.year = year
        n_obs_list = [np.sum(np.array(self.field_list) == field) for field in self.field_Ids]
        if n_obs is None:
            obs_use = np.where((np.array(n_obs_list) <= max_n_obs) &
                               (np.array(n_obs_list) >= min_n_obs))[0]
        else:
            obs_use = np.where(np.array(n_obs_list) == n_obs)[0]
        index_use = index % len(obs_use)
        self.field_Id = self.field_Ids[obs_use[index_use]]
        self.set_conditions_for_field()

    def build_conditions_query(self):
        night0 = 365*(self.year - 1)
        night1 = 365*self.year
        if self.conditions_flag:
            ra = self.field_ra[self.field_Id]
            dec = self.field_dec[self.field_Id]
            ra_delta = self.field_radius/np.cos(dec*np.pi/180.)
            dec_delta = self.field_radius
            if (ra + ra_delta) > 360:
                ra_string = "fieldRA > %f or fieldRA < %f" % (ra - ra_delta, ra + ra_delta - 360.)
            elif (ra - ra_delta) < 0:
                ra_string = "fieldRA > %f or fieldRA < %f" % (360. + ra - ra_delta, ra + ra_delta)
            else:
                ra_string = "fieldRA > %f and fieldRA < %f" % (ra - ra_delta, ra + ra_delta)
            dec_string = "fieldDec > %f and fieldDec < %f" % (dec - dec_delta, dec + dec_delta)
            field_string = "%s and %s" % (ra_string, dec_string)
        else:
            field_string = "fieldId = %i" % self.field_Id
        query = "select altitude,azimuth,airmass,seeingFwhmGeom from SummaryAllProps "\
                + "where night > %i and night < %i and filter = '%s' and %s;"\
                % (night0, night1, self.filter, field_string)
        return query

    def set_conditions_for_field(self, verbose=True, use_coordinates=False):
        """Query the database and store the observing conditions for the chosen field.
        """
        if self.field_Id is None:
            raise RuntimeError("The target field must be chosen with `set_field()` first.")
        query = self.build_conditions_query()
        obs_conditions_cmd = self.opsim.execute(query)
        obs_conditions_list = obs_conditions_cmd.fetchall()
        self.altitude = [obs[0] for obs in obs_conditions_list]
        azimuth = [obs[1] for obs in obs_conditions_list]
        azimuth = [np.floor(az/180)*180 for az in azimuth]
        self.azimuth = azimuth
        self.airmass = [obs[2] for obs in obs_conditions_list]
        self.seeing = [obs[3] for obs in obs_conditions_list]
        min_seeing, max_seeing = np.min(self.seeing), np.max(self.seeing)
        min_airmass, max_airmass = np.min(self.airmass), np.max(self.airmass)
        n_obs = len(self.seeing)
        if verbose:
            print("Selecting %i obs from field %i, "
                  "with seeing range %3.3f to %3.3f "
                  "and airmass range %3.3f to %3.3f"
                  % (n_obs, self.field_Id, min_seeing, max_seeing, min_airmass, max_airmass))

    def set_randomized_conditions_for_field(self, n_obs, verbose=True, force_seeing_range=None):
        """Query the database and store the observing conditions for the chosen field.
        """
        if self.field_Id is None:
            raise RuntimeError("The target field must be chosen with `set_field()` first.")
        night0 = 365*(self.year - 1)
        night1 = 365*self.year
        query = "select altitude,azimuth,airmass,seeingFwhmGeom from SummaryAllProps "\
                + "where night > %i and night < %i and filter = '%s';"\
                % (night0, night1, self.filter)
        obs_conditions_cmd = self.opsim.execute(query)
        obs_conditions_list = obs_conditions_cmd.fetchall()
        altitude = [obs[0] for obs in obs_conditions_list]
        azimuth = [obs[1] for obs in obs_conditions_list]
        azimuth = [np.floor(az/180)*180 for az in azimuth]
        airmass = [obs[2] for obs in obs_conditions_list]
        seeing = [obs[3] for obs in obs_conditions_list]
        rng = np.random.RandomState(self.field_Id + self.year)
        indices = np.floor(rng.rand(n_obs)*len(seeing)).astype(int)
        self.altitude = [altitude[i] for i in indices]
        self.azimuth = [azimuth[i] for i in indices]
        self.airmass = [airmass[i] for i in indices]
        self.seeing = [seeing[i] for i in indices]
        min_seeing, max_seeing = np.min(self.seeing), np.max(self.seeing)
        if force_seeing_range is not None:
            scale = (force_seeing_range - 1)/((max_seeing - min_seeing)/min_seeing)
            self.seeing = [(seeing0 - min_seeing)*scale + min_seeing for seeing0 in self.seeing]
            min_seeing, max_seeing = np.min(self.seeing), np.max(self.seeing)
        min_airmass, max_airmass = np.min(self.airmass), np.max(self.airmass)
        if verbose:
            print("Selecting %i randomized obs from field %i, "
                  "with seeing range %3.3f to %3.3f "
                  "and airmass range %3.3f to %3.3f"
                  % (n_obs, self.field_Id, min_seeing, max_seeing, min_airmass, max_airmass))

    def update_year(self, year, randomize_conditions=False, set_n_obs=0, force_seeing_range=None):
        """Change the year and load new observing conditions for the same target field.

        Parameters
        ----------
        year : `int`
            The new year to load scheduled observations for.
        """
        self.year = year
        self.set_conditions_for_field(verbose=False)
        if randomize_conditions:
            if set_n_obs > 0:
                n_obs = set_n_obs
            else:
                n_obs = len(self.seeing)
            self.set_randomized_conditions_for_field(n_obs, force_seeing_range=force_seeing_range)

    def initialize_simulation(self, n_star=10000, n_quasar=1000,
                              attenuation=20., wavelength_step=10., seed=None,
                              dimension=1024, pixel_scale=0.25, dither=(0., 0.), simulation_size=1.):
        """Set up the simulation using the stored observing conditions.

        Parameters
        ----------
        n_star : `int`, optional
            Number of stars to model in the simulated catalog.
        n_quasar : `int`, optional
            Number of quasars to model in the simulated catalog.
        attenuation : `float`, optional
            Attenuation factor that was used in the simulations
        wavelength_step : `float`, optional
            Wavelength resolution of the spectra and calculation of filter and DCR effects. In nm.
        seed : None, optional
            Description
        dimension : int, optional
            Description
        pixel_scale : float, optional
            Description
        dither : tuple, optional
            Description

        Returns
        -------
        TYPE
            Description

        Raises
        ------
        RuntimeError
            Description
        """
        if self.field_Id is None:
            raise RuntimeError("The target field must be chosen with `set_field()` first.")
        if seed is None:
            self.seed = self.field_Id
        else:
            self.seed = seed
        ra_offset = dither[0]*dimension*pixel_scale*arcseconds/np.cos(lsst_lat.asRadians())
        dec_offset = dither[1]*dimension*pixel_scale*arcseconds
        sim = simulation_wrapper(seed=self.seed, n_star=n_star, n_quasar=n_quasar,
                                 output_directory=self.sim_directory,
                                 attenuation=attenuation, wavelength_step=wavelength_step,
                                 dimension=dimension, pixel_scale=pixel_scale,
                                 ra_offset=ra_offset, dec_offset=dec_offset, simulation_size=simulation_size,
                                 write_catalog=False, write_fits=False, do_simulate=True)
        return sim

    def run_simulation(self, sim, use_seeing=False, write_catalog=False,
                       instrument_noise=0., photon_noise=1./15, sky_noise=0.,
                       initialize_directory=True, mosaic=False, randomize_conditions=False):
        """Generate realizations of the chosen field using the stored set of observing conditions.

        Parameters
        ----------
        sim : `StarFast.StarSim`
            The previously-generated simulation.
        use_seeing : `bool`, optional
            Use database seeing value to scale the PSF width of each observation?
        write_catalog : `bool`, optional
            Write a reference catalog using the simulated sources?
        instrument_noise : `float`, optional
            Adds noise akin to instrumental noise (post-PSF).
            Set to 1.0 for default value, can be scaled up or down
        photon_noise : `float`, optional
            Adds poisson noise akin to photon shot noise.
            Set to 1.0 for default value, can be scaled up or down
        sky_noise : `float`, optional
            Adds noise prior to convolving with the PSF.
        """
        if self.seeing is None:
            raise RuntimeError("The observing conditions must be loaded with  "
                               "`set_conditions_for_field()` first.")
        band_dict = {'u': 0, 'g': 1, 'r': 2, 'i': 3, 'z': 4, 'y': 5}
        psf_name = "var" if use_seeing else "const"
        if mosaic:
            output_directory = self.sim_directory + "mosaic_%i_%sPSF" % (self.seed, psf_name)
            expId = 100*self.field_Id + 1000000*self.year
        else:
            output_directory = self.sim_directory + "field_%i_%sPSF" % (self.field_Id, psf_name)
            expId = 100*band_dict[self.filter] + 1000*self.year
        if randomize_conditions:
            output_directory += "_rand/"
        else:
            output_directory += "/"
        image_directory = output_directory + "images/"
        ingest_directory = output_directory + "input_data/"
        if initialize_directory:
            os.mkdir(output_directory)
            os.mkdir(image_directory)
            os.mkdir(ingest_directory)
            copyfile(self.sim_directory + "_mapper", image_directory)
            copyfile(self.sim_directory + "processEimage_config.py", output_directory)
        if write_catalog:
            sim = simulation_wrapper(sim, output_directory=output_directory,
                                     write_catalog=True, write_fits=False, do_simulate=False)
        psf_fwhm = sim.psf.calculateFWHM()
        gsp = galsim.GSParams(folding_threshold=1.0 /sim.coord.xsize(), maximum_fft_size=12288)
        for az, alt, seeing in zip(self.azimuth, self.altitude, self.seeing):
            if use_seeing:
                psf_fwhm_use = psf_fwhm*seeing/np.median(self.seeing)
            else:
                psf_fwhm_use = psf_fwhm
            psf = galsim.Kolmogorov(fwhm=psf_fwhm_use, flux=1, gsparams=gsp)
            exposure = sim.convolve(elevation=alt, azimuth=az,
                                    instrument_noise=instrument_noise, sky_noise=sky_noise,
                                    photon_noise=photon_noise, exposureId=expId, obsid=expId,
                                    psf=psf)
            filename = "lsst_e_%i_f%i_R22_S11_E000.fits" % (expId, band_dict[self.filter])
            exposure.writeFits(image_directory + filename)
            expId += 1
