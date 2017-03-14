"""Attempts to create airmass-matched template images for existing images in an LSST repository."""

# LSST Data Management System
# Copyright 2016 LSST Corporation.
#
# This product includes software developed by the
# LSST Project (http://www.lsst.org/).
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the LSST License Statement and
# the GNU General Public License along with this program.  If not,
# see <http://www.lsstcorp.org/LegalNotices/>.
#

from __future__ import print_function, division, absolute_import
from collections import namedtuple

import numpy as np
from scipy import constants
from scipy.ndimage.interpolation import shift as scipy_shift
from scipy.ndimage.morphology import binary_dilation
import scipy.optimize.nnls

from lsst.daf.base import DateTime
import lsst.daf.persistence as daf_persistence
from lsst.afw.coord import Coord, IcrsCoord
import lsst.afw.geom as afwGeom
from lsst.afw.geom import Angle
import lsst.afw.image as afwImage
import lsst.afw.math as afwMath
import lsst.meas.algorithms as measAlg
import lsst.pex.exceptions
import lsst.pex.policy as pexPolicy
from lsst.sims.photUtils import Bandpass
from lsst.utils import getPackageDir

from .calc_refractive_index import diff_refraction
from .lsst_defaults import lsst_observatory, lsst_weather

# lsst_lat = -30.244639*lsst.afw.geom.degrees
# lsst_lon = -70.749417*lsst.afw.geom.degrees
# lsst_alt = 2663.

__all__ = ("DcrModel", "DcrCorrection")

nanFloat = float("nan")
nanAngle = Angle(nanFloat)

# Temporary debugging parameters, used if debug_mode=True or self.debug=True is set.
x0 = 300
dx = 200
y0 = 500
dy = 200


class DcrModel:
    """Lightweight object with only the minimum needed to generate DCR-matched template exposures.

    A model must first be generated with DcrCorrection (below). That model can then be used directly, or
    persisted and later read back in needing only this more lightweight class that doesn't need to carry
    around the input exposures. This class will generate template exposures suitable for
    image differencing that are matched to existing exposures in a repository.

    Attributes
    ----------
    bandpass : lsst.sims.photUtils.Bandpass object
        Bandpass object returned by load_bandpass
    bbox : lsst.afw.geom.Box2I object
        A bounding box.
    butler : lsst.daf.persistence Butler object
        The butler handles persisting and depersisting data to and from a repository.
    debug : bool
        Temporary debugging option.
        If set, calculations are performed on only a small region of the full images.
    detected_bit : int
        Value of the detected bit in the bit plane mask.
    exposure_time : float
        Length of the exposure, in seconds.
    filter_name : str
        Name of the bandpass-defining filter of the data. Expected values are u,g,r,i,z,y.
    instrument : str
        Name of the observatory. Used to format dataIds for the butler.
    mask : np.ndarray
        Combined bit plane mask of the model, which is used as the mask plane for generated templates.
    model : list of np.ndarrays
        The DCR model to be used to generate templates. Contains one array for each wavelength step.
    n_step : int
        Number of sub-filter wavelength planes to model.
    observatory : lsst.afw.coord.coordLib.Observatory
        Class containing the longitude, latitude, and altitude of the observatory.
    pixel_scale : lsst.afw.geom.Angle
            Plate scale, as an Angle.
    psf : lsst.meas.algorithms KernelPsf object
        Representation of the point spread function (PSF) of the model.
    psf_size : int
        Dimension of the PSF, in pixels.
    wcs : lsst.afw.image Wcs object
        World Coordinate System of the model.
    weights : np.ndarray
        Weights of the model, calculated from the combined inverse variance of the input exposures.
    """

    def __init__(self, model_repository=None, band_name='g', **kwargs):
        """Restore a persisted DcrModel.

        Only run when restoring a model or for testing; otherwise superceded by DcrCorrection __init__.

        Parameters
        ----------
        model_repository : None, optional
            Path to the repository where the previously-generated DCR model is stored.
        band_name : str, optional
            Name of the bandpass-defining filter of the data. Expected values are u,g,r,i,z,y.
        **kwargs : TYPE
            Any additional keyword arguments to pass to load_bandpass
        """
        self.butler = None
        self.load_model(model_repository=model_repository, band_name=band_name, **kwargs)

    def generate_templates_from_model(self, obsid_range=None, exposures=None,
                                      repository=None, output_repository=None,
                                      instrument='lsstSim', warp=False, verbose=True,
                                      output_obsid_offset=None):
        """Use the previously generated model and construct a dcr template image.

        Parameters
        ----------
        obsid_range : int, or list of ints, optional
            Single, or list of observation IDs in repository to create matched
            templates for. Ignored if exposures are supplied directly.
        exposures : List or generator of lsst.afw.image.ExposureD objects, optional
            List or generator of exposure objects that will have matched templates created.
        repository : str, optional
            Path to the repository where the exposure data to be matched are stored.
            Ignored if exposures are supplied directly.
        output_repository : str, optional
            Path to repository directory where templates will be saved.
            The templates will not be written to disk if `output_repository` is None.
        instrument : str, optional
            Name of the observatory.
        warp : bool, optional
            Set to true if the exposures have different wcs from the model.
            If True, the generated templates will be warped to match the wcs of each exposure.
        verbose : bool, optional
            Set to True to print progress messages.
        output_obsid_offset : int, optional
            Optional offset to add to the output obsids.
            Use if writing to the same repository as the input to avoid over-writing the input data.

        Yields
        ------
        Returns a generator that builds DCR-matched templates for each exposure.

        Raises
        ------
        ValueError
            If neither `repository` or `exposures` is set.
            If neither `obsid_range` or `exposures` is set
        """
        butler_out = None  # Overwritten later if a butler is used
        if exposures is None:
            if repository is not None:
                butler = daf_persistence.Butler(repository)
                if self.butler is None:
                    self.butler = butler
            elif self.butler is not None:
                butler = self.butler
            else:
                raise ValueError("Either repository or exposures must be set.")
            if obsid_range is not None:
                dataId_gen = self._build_dataId(obsid_range, self.filter_name, instrument=instrument)
                exposures = (calexp for calexp in
                             (butler.get("calexp", dataId=dataId) for dataId in dataId_gen))
            else:
                raise ValueError("One of obsid_range or exposures must be set.")

        self.instrument = instrument
        if self.psf is None:
            self.calc_psf_model()

        for exp_i, calexp in enumerate(exposures):
            visitInfo = calexp.getInfo().getVisitInfo()
            obsid = visitInfo.getExposureId()
            if verbose:
                print("Working on observation %s" % obsid, end="")
            bbox_exp = calexp.getBBox()
            wcs_exp = calexp.getInfo().getWcs()
            el = visitInfo.getBoresightAzAlt().getLatitude()
            az = visitInfo.getBoresightAzAlt().getLongitude()
            rotation_angle = calculate_rotation_angle(calexp)
            weather = visitInfo.getWeather()
            template, variance = self.build_matched_template(calexp, el=el, rotation_angle=rotation_angle,
                                                             weather=weather)

            if verbose:
                print(" ... Done!")

            if output_obsid_offset is not None:
                obsid_out = obsid + output_obsid_offset
            else:
                obsid_out = obsid
            dataId_out = self._build_dataId(obsid_out, self.filter_name, instrument=instrument)[0]
            exposure = self.create_exposure(template, variance=variance, snap=0,
                                            boresightRotAngle=rotation_angle, weather=weather,
                                            elevation=el, azimuth=az, obsid=obsid_out)
            if warp:
                wrap_warpExposure(exposure, wcs_exp, bbox_exp)
            if output_repository is not None:
                if butler_out is None:
                    butler_out = daf_persistence.Butler(output_repository)
                butler_out.put(exposure, "calexp", dataId=dataId_out)
            yield exposure

    def build_matched_template(self, exposure, model=None, el=None, rotation_angle=None,
                               return_weights=True, weather=None):
        """Sub-routine to calculate the sum of the model images shifted by DCR for a given exposure.

        Parameters
        ----------
        exposure : lsst.afw.image.ExposureD object
            Single exposure to create a DCR-matched template for from the model.
        model : List of numpy ndarrays, optional
            The DCR model. If not set, then self.model is used.
        el : lsst.afw.geom.Angle, optional
            Elevation angle of the observation. If not set, it is read from the exposure.
        rotation_angle : lsst.afw.geom.Angle, optional
            Sky rotation angle of the observation. If not set it is calculated from the exposure metadata.
        return_weights : bool, optional
            Set to True to return the variance plane, as well as the image.
        weather : lsst.afw.coord Weather, optional
            Class containing the measured temperature, pressure, and humidity
            at the observatory during an observation
            Weather data is read from the exposure metadata if not supplied.

        Returns
        -------
        np.ndarrary or (np.ndarray, np.ndarray)
            Returns a numpy ndarray of the image values for the template.
            If `return_weights` is set, then it returns a tuple of the image and variance arrays.
        """
        if el is None:
            el = exposure.getInfo().getVisitInfo().getBoresightAzAlt().getLatitude()
        if rotation_angle is None:
            rotation_angle = calculate_rotation_angle(exposure)
        if weather is None:
            weather = exposure.getInfo().getVisitInfo().getWeather()
        dcr_gen = self._dcr_generator(self.bandpass, pixel_scale=self.pixel_scale,
                                      observatory=self.observatory, weather=weather,
                                      elevation=el, rotation_angle=rotation_angle, use_midpoint=True)
        template = np.zeros((self.y_size, self.x_size))
        if return_weights:
            weights = np.zeros((self.y_size, self.x_size))
        if model is None:
            model_use = self.model
        else:
            model_use = model
        for f, dcr in enumerate(dcr_gen):
            shift = (dcr.dy, dcr.dx)
            template += scipy_shift(model_use[f], shift)
            if return_weights:
                weights += scipy_shift(self.weights, shift)
        if return_weights:
            weights /= self.n_step
            variance = np.zeros((self.y_size, self.x_size))
            variance[weights > 0] = 1./weights[weights > 0]
            return (template, variance)
        else:
            return template

    def extract_image(self, exposure, airmass_weight=False, calculate_dcr_gen=True):
    def build_matched_psf(self, el, rotation_angle, weather):
        """Sub-routine to calculate the PSF as elongated by DCR for a given exposure.

        Once the matched templates incorporate variable seeing, this function should also match the seeing.

        Parameters
        ----------
        el : lsst.afw.geom.Angle
            Elevation angle of the observation. If not set, it is read from the exposure.
        rotation_angle : lsst.afw.geom.Angle
            Sky rotation angle of the observation. If not set it is calculated from the exposure metadata.
        weather : lsst.afw.coord Weather
            Class containing the measured temperature, pressure, and humidity
            at the observatory during an observation

        Returns
        -------
        lsst.meas.algorithms KernelPsf object
            Designed to be passed to a lsst.afw.image ExposureD through the method setPsf()
        """
        dcr_gen = DcrModel._dcr_generator(self.bandpass, pixel_scale=self.pixel_scale,
                                          observatory=self.observatory, weather=weather,
                                          elevation=el, rotation_angle=rotation_angle, use_midpoint=True)
        psf_vals = self.psf.computeKernelImage().getArray()
        psf_vals_out = np.zeros((self.psf_size, self.psf_size))

        for f, dcr in enumerate(dcr_gen):
            shift = (dcr.dy, dcr.dx)
            psf_vals_out += scipy_shift(psf_vals, shift)
        psf_image = afwImage.ImageD(self.psf_size, self.psf_size)
        psf_image.getArray()[:, :] = psf_vals_out
        psfK = afwMath.FixedKernel(psf_image)
        psf = measAlg.KernelPsf(psfK)
        return psf

        """Helper function to extract image array values from an exposure.

        Parameters
        ----------
        exposure : lsst.afw.image.ExposureD object
            Input single exposure to extract the image and variance planes
        airmass_weight : bool, optional
            Set to True to scale the variance by the airmass of the observation.
        calculate_dcr_gen : bool, optional
            Set to True to also return a DcrModel.dcr_generator generator.
        use_only_detected : bool, optional
            If True, set all pixels to zero that do not have the detected bit set in the mask plane.
        use_variance : bool, optional
            If True, return the true inverse variance.
            Otherwise, return calculated weights in the range 0 - 1 for each pixel.

        Returns
        -------
        Returns a tuple of the image and weights (inverse variance) arrays.
        If `calculate_dcr_gen` is set, returns a tuple of the image, weights, and dcr generator.
        """
        img_vals = exposure.getMaskedImage().getImage().getArray()
        nan_inds = np.isnan(img_vals)
        img_vals[nan_inds] = 0.
        variance = exposure.getMaskedImage().getVariance().getArray()
        variance[nan_inds] = 0
        inverse_var = np.zeros_like(variance)
        inverse_var[variance > 0] = 1./variance[variance > 0]

        mask = exposure.getMaskedImage().getMask().getArray()
        ind_cut = (mask | self.detected_bit) != self.detected_bit
        inverse_var[ind_cut] = 0.
        # Create a buffer of lower-weight pixels surrounding masked pixels.
        ind_cut2 = binary_dilation(ind_cut, iterations=2)
        inverse_var[ind_cut2] /= 2.

        if self.debug:
            slice_inds = np.s_[y0: y0 + dy, x0: x0 + dx]
            img_vals = img_vals[slice_inds]
            inverse_var = inverse_var[slice_inds]

        visitInfo = exposure.getInfo().getVisitInfo()
        if airmass_weight:
            inverse_var /= visitInfo.getBoresightAirmass()
        if calculate_dcr_gen:
            el = visitInfo.getBoresightAzAlt().getLatitude()
            rotation_angle = calculate_rotation_angle(exposure)
            weather = visitInfo.getWeather()
            dcr_gen = DcrModel._dcr_generator(self.bandpass, pixel_scale=self.pixel_scale,
                                              observatory=self.observatory, weather=weather,
                                              elevation=el, rotation_angle=rotation_angle, use_midpoint=True)
            return (img_vals, inverse_var, dcr_gen)
        else:
            return (img_vals, inverse_var)

    @staticmethod
    def _fetch_metadata(metadata, property_name, default_value=None):
        """Simple wrapper to extract metadata from an exposure, with some error handling.

        Parameters
        ----------
        metadata : obj
            An LSST exposure metadata object, obtained with exposure.getMetadata()
        property_name : str
            Name of the property to be extracted
        default_value : None, optional
            Value to be returned if the property is not found in the exposure metadata.

        Returns
        -------
        Returns the value of `property_name` from the metadata of exposure.
        If the given property is not found, returns `default_value` if supplied, or None otherwise.
        """
        try:
            value = metadata.get(property_name)
        except lsst.pex.exceptions.wrappers.NotFoundError as e:
            if default_value is not None:
                print("WARNING: " + str(e) + ". Using default value: %s" % repr(default_value))
                return default_value
            else:
                return None
        return value

    @staticmethod
    def _build_dataId(obsid_range, band, instrument='lsstSim'):
        """Construct a dataId dictionary for the butler to find a calexp.

        Parameters
        ----------
        obsid_range : int, or list of ints
            The observation IDs of the data to load.
        band : str
            Name of the bandpass-defining filter of the data. Expected values are u,g,r,i,z,y.
        instrument : str, optional
            Name of the observatory.

        Returns
        -------
        Return a list of dataIds for the butler to use to load a calexp from a repository
        """
        if instrument == 'lsstSim':
            if hasattr(obsid_range, '__iter__'):
                dataId = [{'visit': obsid, 'raft': '2,2', 'sensor': '1,1', 'filter': band}
                          for obsid in obsid_range]
            else:
                dataId = [{'visit': obsid, 'raft': '2,2', 'sensor': '1,1', 'filter': band}
                          for obsid in [obsid_range]]
        elif instrument == 'decam':
            if hasattr(obsid_range, '__iter__'):
                dataId = [{'visit': obsid, 'ccdnum': 10}
                          for obsid in obsid_range]
            else:
                dataId = [{'visit': obsid, 'ccdnum': 10}
                          for obsid in [obsid_range]]
        return dataId

    @staticmethod
    def _build_model_dataId(band, subfilter=None):
        """Construct a dataId dictionary for the butler to find a dcrModel.

        Parameters
        ----------
        band : str
            Name of the bandpass-defining filter of the data. Expected values are u,g,r,i,z,y.
        subfilter : int, optional
            DCR model index within the band.

        Returns
        -------
        Return a dataId for the butler to use to load a dcrModel from a repository
        """
        if subfilter is None:
            dataId = {'filter': band, 'tract': 0, 'patch': '0'}
        else:
            dataId = {'filter': band, 'tract': 0, 'patch': '0', 'subfilter': subfilter}
        return(dataId)

    @staticmethod
    def _create_wcs(bbox, pixel_scale, ra, dec, sky_rotation):
        """Create a wcs (coordinate system).

        Parameters
        ----------
        bbox : lsst.afw.geom.Box2I object
            A bounding box.
        pixel_scale : lsst.afw.geom.Angle
            Plate scale, as an Angle.
        ra : lsst.afw.geom.Angle
            Right Ascension of the reference pixel, as an Angle.
        dec : lsst.afw.geom.Angle
            Declination of the reference pixel, as an Angle.
        sky_rotation : lsst.afw.geom.Angle
            Rotation of the image axis, East from North.

        Returns
        -------
        Returns a lsst.afw.image.wcs object.
        """
        crval = IcrsCoord(ra, dec)
        crpix = afwGeom.Box2D(bbox).getCenter()
        cd1_1 = np.cos(sky_rotation.asRadians())*pixel_scale.asDegrees()
        cd1_2 = -np.sin(sky_rotation.asRadians())*pixel_scale.asDegrees()
        cd2_1 = np.sin(sky_rotation.asRadians())*pixel_scale.asDegrees()
        cd2_2 = np.cos(sky_rotation.asRadians())*pixel_scale.asDegrees()
        return(afwImage.makeWcs(crval, crpix, cd1_1, cd1_2, cd2_1, cd2_2))

    @staticmethod
    def load_bandpass(band_name='g', wavelength_step=None, use_mirror=True, use_lens=True, use_atmos=True,
                      use_filter=True, use_detector=True):
        """Load in Bandpass object from sims_photUtils.

        Parameters
        ----------
        band_name : str, optional
            Common name of the filter used. For LSST, use u, g, r, i, z, or y
        wavelength_step : float, optional
            Wavelength resolution in nm, also the wavelength range of each sub-band plane.
            If not set, the entire band range is used.
        use_mirror : bool, optional
            Set to include the mirror in the filter throughput calculation.
        use_lens : bool, optional
            Set to use the LSST lens in the filter throughput calculation
        use_atmos : bool, optional
            Set to use the standard atmosphere transmission in the filter throughput calculation
        use_filter : bool, optional
            Set to use the LSST filters in the filter throughput calculation.
        use_detector : bool, optional
            Set to use the LSST detector efficiency in the filter throughput calculation.

        Returns
        -------
        Returns a lsst.sims.photUtils.Bandpass object.
        """
        class BandpassMod(Bandpass):
            """Customize a few methods of the Bandpass class from sims_photUtils."""

            def calc_eff_wavelen(self, wavelength_min=None, wavelength_max=None):
                """Calculate effective wavelengths for filters.

                Parameters
                ----------
                wavelength_min : float, optional
                    Starting wavelength, in nm
                wavelength_max : float, optional
                    End wavelength, in nm

                Returns
                -------
                Returns the weighted average wavelength within the range given, taken over the bandpass.
                """
                if self.phi is None:
                    self.sbTophi()
                if wavelength_min is None:
                    wavelength_min = np.min(self.wavelen)
                if wavelength_max is None:
                    wavelength_max = np.max(self.wavelen)
                w_inds = (self.wavelen >= wavelength_min) & (self.wavelen <= wavelength_max)
                effwavelenphi = (self.wavelen[w_inds]*self.phi[w_inds]).sum()/self.phi[w_inds].sum()
                return effwavelenphi

            def calc_bandwidth(self):
                f0 = constants.speed_of_light/(self.wavelen_min*1.0e-9)
                f1 = constants.speed_of_light/(self.wavelen_max*1.0e-9)
                f_cen = constants.speed_of_light/(self.calc_eff_wavelen()*1.0e-9)
                return(f_cen*2.0*(f0 - f1)/(f0 + f1))

        """
        Define the wavelength range and resolution for a given ugrizy band.
        These are defined in case the LSST filter throughputs are not used.
        """
        band_dict = {'u': (324.0, 395.0), 'g': (405.0, 552.0), 'r': (552.0, 691.0),
                     'i': (818.0, 921.0), 'z': (922.0, 997.0), 'y': (975.0, 1075.0)}
        band_range = band_dict[band_name]
        if wavelength_step is None:
            wavelength_step = band_range[1] - band_range[0]
        bandpass = BandpassMod(wavelen_min=band_range[0], wavelen_max=band_range[1],
                               wavelen_step=wavelength_step)
        throughput_dir = getPackageDir('throughputs')
        lens_list = ['baseline/lens1.dat', 'baseline/lens2.dat', 'baseline/lens3.dat']
        mirror_list = ['baseline/m1.dat', 'baseline/m2.dat', 'baseline/m3.dat']
        atmos_list = ['atmos/atmos_11.dat']
        detector_list = ['baseline/detector.dat']
        filter_list = ['baseline/filter_' + band_name + '.dat']
        component_list = []
        if use_mirror:
            component_list += mirror_list
        if use_lens:
            component_list += lens_list
        if use_atmos:
            component_list += atmos_list
        if use_detector:
            component_list += detector_list
        if use_filter:
            component_list += filter_list
        bandpass.readThroughputList(rootDir=throughput_dir, componentList=component_list)
        # Calculate bandpass phi value if required.
        if bandpass.phi is None:
            bandpass.sbTophi()
        return bandpass

    @staticmethod
    def _wavelength_iterator(bandpass, use_midpoint=False):
        """Define iterator to ensure that loops over wavelength are consistent.

        Parameters
        ----------
        bandpass : lsst.sims.photUtils.Bandpass object
            Bandpass object returned by load_bandpass
        use_midpoint : bool, optional
            If set to True return the filter-weighted average wavelength.
            Otherwise, return a tuple of the starting and end wavelength.

        Yields
        -----
        If `use_midpoint` is set, yields the effective wavelength of the next sub-band.
        Otherwise, yields the start and end wavelength of the next sub-band as a tuple.
        """
        wave_start = bandpass.wavelen_min
        while np.ceil(wave_start) < bandpass.wavelen_max:
            wave_end = wave_start + bandpass.wavelen_step
            if wave_end > bandpass.wavelen_max:
                wave_end = bandpass.wavelen_max
            if use_midpoint:
                yield bandpass.calc_eff_wavelen(wavelength_min=wave_start, wavelength_max=wave_end)
            else:
                yield (wave_start, wave_end)
            wave_start = wave_end

    @staticmethod
    def _dcr_generator(bandpass, pixel_scale, elevation, rotation_angle,
                       weather=lsst_weather,
                       observatory=lsst_observatory, use_midpoint=False):
        """Call the functions that compute Differential Chromatic Refraction (relative to mid-band).

        Parameters
        ----------
        bandpass : lsst.sims.photUtils.Bandpass object
            Bandpass object returned by load_bandpass
        pixel_scale : lsst.afw.geom.Angle
            Plate scale, as an Angle.
        elevation : lsst.afw.geom.Angle
            Elevation angle of the observation
        rotation_angle : lsst.afw.geom.Angle
            Sky rotation angle of the observation
        weather : lsst.afw.coord Weather, optional
            Class containing the measured temperature, pressure, and humidity
            at the observatory during an observation
            Weather data is read from the exposure metadata if not supplied.
        observatory : lsst.afw.coord.coordLib.Observatory, optional
            Class containing the longitude, latitude, and altitude of the observatory.
        use_midpoint : bool, optional
            Set to True to use the effective wavelength of the sub-band.

        Yields
        ------
            If `use_midpoint` is True, yields the x and y DCR offsets for the mid-point of the next sub-band.
            Otherwise yields a tuple of the x and y DCR offsets for the start and end of the next sub-band.

        """
        zenith_angle = Angle(np.pi/2) - elevation
        wavelength_midpoint = bandpass.calc_eff_wavelen()
        delta = namedtuple("delta", ["start", "end"])
        dcr = namedtuple("dcr", ["dx", "dy"])
        if use_midpoint:
            for wl in DcrModel._wavelength_iterator(bandpass, use_midpoint=True):
                # Note that refract_amp can be negative, since it's relative to the midpoint of the full band
                refract_mid = diff_refraction(wavelength=wl, wavelength_ref=wavelength_midpoint,
                                              zenith_angle=zenith_angle,
                                              observatory=observatory, weather=weather)
                refract_mid_pixels = refract_mid.asArcseconds()/pixel_scale.asArcseconds()
                yield dcr(dx=refract_mid_pixels*np.sin(rotation_angle.asRadians()),
                          dy=refract_mid_pixels*np.cos(rotation_angle.asRadians()))
        else:
            for wl_start, wl_end in DcrModel._wavelength_iterator(bandpass, use_midpoint=False):
                # Note that refract_amp can be negative, since it's relative to the midpoint of the full band
                refract_start = diff_refraction(wavelength=wl_start, wavelength_ref=wavelength_midpoint,
                                                zenith_angle=zenith_angle,
                                                observatory=observatory, weather=weather)
                refract_end = diff_refraction(wavelength=wl_end, wavelength_ref=wavelength_midpoint,
                                              zenith_angle=zenith_angle,
                                              observatory=observatory, weather=weather)
                refract_start_pixels = refract_start.asArcseconds()/pixel_scale.asArcseconds()
                refract_end_pixels = refract_end.asArcseconds()/pixel_scale.asArcseconds()
                dx = delta(start=refract_start_pixels*np.sin(rotation_angle.asRadians()),
                           end=refract_end_pixels*np.sin(rotation_angle.asRadians()))
                dy = delta(start=refract_start_pixels*np.cos(rotation_angle.asRadians()),
                           end=refract_end_pixels*np.cos(rotation_angle.asRadians()))
                yield dcr(dx=dx, dy=dy)

    def create_exposure(self, array, elevation, azimuth, variance=None, era=None, snap=0,
                        exposureId=0, ra=nanAngle, dec=nanAngle, boresightRotAngle=nanAngle,
                        weather=lsst_weather, **kwargs):
        """Convert a numpy array to an LSST exposure, and units of electron counts.

        Parameters
        ----------
        array : np.ndarray
            Numpy array to use as the data for the exposure
        elevation : lsst.afw.geom Angle
            Elevation angle of the observation
        azimuth : lsst.afw.geom Angle
            Azimuth angle of the observation
        variance : np.ndarray, optional
            Numpy array to use as the variance plane of the exposure.
            If None, the absoulte value of 'array' is used for the variance plane.
        era : lsst.afw.geom Angle, optional
            Earth rotation angle (ERA) of the observation.
            If not set it will be calculated from the latitude, longitude, RA, Dec, and elevation angle
        snap : int, optional
            Snap ID to add to the metadata of the exposure. Required to mimic Phosim output.
        exposureId : int, optional
            Observation ID of the exposure, a long int.
        ra : lsst.afw.geom Angle, optional
            The right ascension of the boresight of the target field.
        dec : lsst.afw.geom Angle, optional
            The declination of the boresight of the target field
        boresightRotAngle : lsst.afw.geom Angle, optional
            The rotation angle of the field around the boresight.
        weather : lsst.afw.coord Weather, optional
            Class containing the measured temperature, pressure, and humidity
            at the observatory during an observation
            Weather data is read from the exposure metadata if not supplied.
        **kwargs : TYPE
            Any additional keyword arguments will be added to the metadata of the exposure.

        Returns
        -------
        lsst.afw.image.ExposureD object
        """
        exposure = afwImage.ExposureD(self.bbox)
        exposure.setWcs(self.wcs)
        # We need the filter name in the exposure metadata, and it can't just be set directly
        try:
            exposure.setFilter(afwImage.Filter(self.filter_name))
        except:
            filterPolicy = pexPolicy.Policy()
            filterPolicy.add("lambdaEff", self.bandpass.calc_eff_wavelen())
            afwImage.Filter.define(afwImage.FilterProperty(self.filter_name, filterPolicy))
            exposure.setFilter(afwImage.Filter(self.filter_name))
            # Need to reset afwImage.Filter to prevent an error in future calls to daf_persistence.Butler
            afwImage.FilterProperty_reset()
        if self.debug:
            array_temp = array
            array = np.zeros_like(exposure.getMaskedImage().getImage().getArray())
            array[y0: y0 + dy, x0: x0 + dx] = array_temp
            if variance is not None:
                variance_temp = variance
                variance = np.zeros_like(array)
                variance[y0: y0 + dy, x0: x0 + dx] = variance_temp
        exposure.getMaskedImage().getImage().getArray()[:, :] = array
        if variance is None:
            variance = np.abs(array)
        exposure.getMaskedImage().getVariance().getArray()[:, :] = variance

        if self.mask is not None:
            exposure.getMaskedImage().getMask().getArray()[:, :] = self.mask
        ha_term1 = np.sin(elevation.asRadians())
        ha_term2 = np.sin(dec.asRadians())*np.sin(self.observatory.getLatitude().asRadians())
        ha_term3 = np.cos(dec.asRadians())*np.cos(self.observatory.getLatitude().asRadians())
        hour_angle = np.arccos((ha_term1 - ha_term2) / ha_term3)
        mjd = 59000.0 + (self.observatory.getLatitude().asDegrees()/15.0 - hour_angle*180/np.pi)/24.0
        airmass = 1.0/np.sin(elevation.asRadians())
        if era is None:
            era = Angle(hour_angle - self.observatory.getLongitude().asRadians())
        meta = exposure.getMetadata()
        meta.add("CHIPID", "R22_S11")
        # Required! Phosim output stores the snap ID in "OUTFILE" as the last three characters in a string.
        meta.add("OUTFILE", ("SnapId_%3.3i" % snap))

        meta.add("TAI", mjd)
        meta.add("MJD-OBS", mjd)

        meta.add("EXTTYPE", "IMAGE")
        meta.add("EXPTIME", self.exposure_time)
        meta.add("AIRMASS", airmass)
        meta.add("ZENITH", 90. - elevation.asDegrees())
        meta.add("AZIMUTH", azimuth.asDegrees())

        # Add all additional keyword arguments to the metadata.
        for add_item in kwargs:
            meta.add(add_item, kwargs[add_item])

        visitInfo = afwImage.makeVisitInfo(exposureId=int(exposureId),
                                           exposureTime=self.exposure_time,
                                           darkTime=self.exposure_time,
                                           date=DateTime(mjd),
                                           ut1=mjd,
                                           era=era,
                                           boresightRaDec=IcrsCoord(ra, dec),
                                           boresightAzAlt=Coord(azimuth, elevation),
                                           boresightAirmass=airmass,
                                           boresightRotAngle=boresightRotAngle,
                                           observatory=self.observatory,
                                           weather=weather
                                           )
        exposure.getInfo().setVisitInfo(visitInfo)

        #Set the DCR-matched PSF
        psf_single = self.build_matched_psf(elevation, calculate_rotation_angle(exposure), weather)
        exposure.setPsf(psf_single)
        return exposure

    def export_model(self, model_repository=None):
        """Persist a DcrModel with metadata to a repository.

        Parameters
        ----------
        model_repository : None, optional
            Full path to the directory of the repository to save the dcrModel in
            If not set, uses the existing self.butler

        Returns
        -------
        None
        """
        if model_repository is None:
            butler = self.butler
        else:
            butler = daf_persistence.Butler(model_repository)
        wave_gen = DcrModel._wavelength_iterator(self.bandpass, use_midpoint=False)
        for f in range(self.n_step):
            wl_start, wl_end = wave_gen.next()
            exp = self.create_exposure(self.model[f], variance=self.weights,
                                       elevation=Angle(np.pi/2), azimuth=Angle(0),
                                       detectbit=self.detected_bit,
                                       subfilt=f, nstep=self.n_step, wavelow=wl_start, wavehigh=wl_end,
                                       wavestep=self.bandpass.wavelen_step, telescop=self.instrument)
            butler.put(exp, "dcrModel", dataId=self._build_model_dataId(self.filter_name, f))

    def load_model(self, model_repository=None, band_name='g', **kwargs):
        """Depersist a DcrModel from a repository and set up the metadata.

        Parameters
        ----------
        model_repository : None, optional
            Full path to the directory of the repository to load the dcrModel from.
            If not set, uses the existing self.butler
        band_name : str, optional
            Common name of the filter used. For LSST, use u, g, r, i, z, or y
        **kwargs : TYPE
            Any additional keyword arguments to pass to load_bandpass

        Returns
        -------
        None, but loads self.model and sets up all the needed quantities such as the psf and bandpass objects.
        """
        if model_repository is None:
            butler = self.butler
        else:
            butler = daf_persistence.Butler(model_repository)
        model_arr = []
        weights_arr = []
        f = 0
        while butler.datasetExists("dcrModel", dataId=self._build_model_dataId(band_name, subfilter=f)):
            dcrModel = butler.get("dcrModel", dataId=self._build_model_dataId(band_name, subfilter=f))
            model_arr.append(dcrModel.getMaskedImage().getImage().getArray())
            weights_arr.append(dcrModel.getMaskedImage().getVariance().getArray())
            f += 1

        self.model = model_arr
        self.weights = weights_arr[0]  # The weights should be identical for all subfilters.
        self.mask = dcrModel.getMaskedImage().getMask().getArray()

        # This only uses the mask of the last image. For real data all masks should be used.
        meta = dcrModel.getMetadata()
        self.wcs = dcrModel.getWcs()
        self.n_step = len(model_arr)
        wave_step = self._fetch_metadata(meta, "WAVESTEP")
        self.detected_bit = self._fetch_metadata(meta, "DETECTBIT")
        self.y_size, self.x_size = dcrModel.getDimensions()
        self.pixel_scale = self.wcs.pixelScale()
        self.exposure_time = dcrModel.getInfo().getVisitInfo().getExposureTime()
        self.observatory = dcrModel.getInfo().getVisitInfo().getObservatory()
        self.filter_name = band_name
        self.bbox = dcrModel.getBBox()
        self.instrument = self._fetch_metadata(meta, "TELESCOP", default_value='lsstSim')
        self.bandpass = self.load_bandpass(band_name=band_name, wavelength_step=wave_step, **kwargs)

        self.psf = dcrModel.getPsf()
        psf_avg = self.psf.computeKernelImage().getArray()
        self.psf_size = psf_avg.shape[0]
        self.debug = False

    @staticmethod
    def _calc_offset_phase(dcr_gen, exposure=None, size=None, size_out=None, center_only=False):
        """Calculate the covariance matrix for a simple shift with no psf.

        Parameters
        ----------
        dcr_gen : generator
             A dcr generator of offsets, returned by _dcr_generator.
        exposure : lsst.afw.image.ExposureD object, optional
            An LSST exposure object. Only needed if size is not specified.
        size : int, optional
            Width in pixels of the region used in the origin image. Default is entire image
        size_out : int, optional
            Width in pixels of the region used in the destination image. Default is same as size
        center_only : bool, optional
            Set to True to calculate the covariance for only the center pixel.

        Returns
        -------
        np.ndarray
            Returns the covariance matrix of an offset generated by _dcr_generator in the form (dx, dy)
        """
        phase_arr = []
        if size is None:
            size = min([exposure.getHeight(), exposure.getWidth()])
        if size_out is None:
            size_out = size
        for dx, dy in dcr_gen:
            kernel_x = _kernel_1d(dx, size, n_substep=100)
            kernel_y = _kernel_1d(dy, size, n_substep=100)
            kernel = np.einsum('i,j->ij', kernel_y, kernel_x)

            if center_only:
                size_out = 1
            shift_mat = _calc_psf_kernel_subroutine(kernel, size=size, size_out=size_out)
            phase_arr.append(shift_mat)
        phase_arr = np.hstack(phase_arr)
        return phase_arr

    def _build_dcr_kernel(self, size, expand_intermediate=False, exposure=None,
                          bandpass=None, n_step=None):
        """Calculate the DCR covariance matrix for a set of exposures, or a single exposure.

        Parameters
        ----------
        size : int
            Width in pixels of the region used in the origin image.
        expand_intermediate : bool, optional
            If set, calculate the covariance matrix between the region of pixels in
            the origin image and a region twice as wide in the destination image.
            This helps avoid edge effects when computing A^T A.
        exposure : lsst.afw.image.ExposureD object, optional
            If not supplied, the covariance matrix for all exposures in self.exposures is calculated.
        bandpass : lsst.sims.photUtils.Bandpass object
            Bandpass object returned by load_bandpass
        n_step : int, optional
            Number of sub-band planes to use. Default is to use self.n_step

        Returns
        -------
        np.ndarray
            Returns the covariance matrix for the exposure(s).
        """
        n_pix = size**2
        if expand_intermediate:
            kernel_size_intermediate = size*2
        else:
            kernel_size_intermediate = size
        n_pix_int = kernel_size_intermediate**2

        if exposure is None:
            exp_gen = (exp for exp in self.exposures)
            n_images = self.n_images
        else:
            exp_gen = (exposure for i in range(1))
            n_images = 1
        if n_step is None:
            n_step = self.n_step
        if bandpass is None:
            bandpass = self.bandpass
        dcr_kernel = np.zeros((n_images*n_pix_int, n_step*n_pix))
        for exp_i, exp in enumerate(exp_gen):
            visitInfo = exp.getInfo().getVisitInfo()
            el = visitInfo.getBoresightAzAlt().getLatitude()
            az = visitInfo.getBoresightAzAlt().getLongitude()
            weather = visitInfo.getWeather()
            dcr_gen = DcrModel._dcr_generator(bandpass, pixel_scale=self.pixel_scale, weather=weather,
                                              observatory=self.observatory, elevation=el, rotation_angle=az)
            kernel_single = self._calc_offset_phase(dcr_gen=dcr_gen, size=size,
                                                    size_out=kernel_size_intermediate)
            dcr_kernel[exp_i*n_pix_int: (exp_i + 1)*n_pix_int, :] = kernel_single
        return dcr_kernel

    def calc_psf_model_single(self, exposure):
        """Calculate the fiducial psf for a single exposure, accounting for DCR.

        Parameters
        ----------
        exposure : lsst.afw.image.ExposureD object
            A single LSST exposure object

        Returns
        -------
        np.ndarray
            Returns the fiducial PSF for an exposure, after taking out DCR effects.
        """
        visitInfo = exposure.getInfo().getVisitInfo()
        el = visitInfo.getBoresightAzAlt().getLatitude()
        az = visitInfo.getBoresightAzAlt().getLongitude()
        weather = visitInfo.getWeather()

        # Take the measured PSF as the true PSF, smeared out by DCR.
        psf_img = exposure.getPsf().computeKernelImage().getArray()
        psf_size_test = psf_img.shape[0]
        if psf_size_test > self.psf_size:
            p0 = psf_size_test//2 - self.psf_size//2
            p1 = p0 + self.psf_size
            psf_img = psf_img[p0:p1, p0:p1]
            psf_size_use = self.psf_size
        else:
            psf_size_use = psf_size_test

        # Calculate the expected shift (with no psf) due to DCR
        dcr_gen = DcrModel._dcr_generator(self.bandpass, pixel_scale=self.pixel_scale, weather=weather,
                                          observatory=self.observatory, elevation=el, azimuth=az)
        dcr_shift = self._calc_offset_phase(exposure=exposure, dcr_gen=dcr_gen,
                                            size=psf_size_use)
        # Assume that the PSF does not change between sub-bands.
        regularize_psf = None
        # Use the entire psf provided, even if larger than than the kernel we will use to solve DCR for images
        # If the original psf is much larger than the kernel, it may be trimmed slightly by fit_psf_size above
        psf_model_gen = solve_model(psf_size_use, np.ravel(psf_img), n_step=self.n_step,
                                    use_nonnegative=True, regularization=regularize_psf, kernel_dcr=dcr_shift)

        # After solving for the (potentially) large psf, store only the central portion of size kernel_size.
        psf_vals = np.sum(psf_model_gen, axis=0)/self.n_step
        return psf_vals


class DcrCorrection(DcrModel):
    """Class that loads LSST calibrated exposures and produces airmass-matched template images.

    Input exposures are read with a butler, and an initial model is made by coadding the images.
    An improved model of the sky is built for a series of sub-bands within the full bandwidth of the filter
    used for the observations by iteratively forward-modeling the template using the calculated
    Differential Chromatic Refration for the exposures in each sub-band.

    Attributes
    ----------
    bandpass : lsst.sims.photUtils.Bandpass object
        Bandpass object returned by `load_bandpass`
    bbox : lsst.afw.geom.Box2I object
        A bounding box.
    butler : lsst.daf.persistence Butler object
        The butler handles persisting and depersisting data to and from a repository.
    debug : bool
        Temporary debugging option.
        If set, calculations are performed on only a small region of the full images.
    detected_bit : int
        Value of the detected bit in the bit plane mask.
    exposure_time : float
        Length of the exposure, in seconds.
    exposures : list
        Description
    filter_name : str
        Name of the bandpass-defining filter of the data. Expected values are u,g,r,i,z,y.
    instrument : str
        Name of the observatory. Used to format dataIds for the butler.
    mask : np.ndarray
        Combined bit plane mask of the model, which is used as the mask plane for generated templates.
    model : list of np.ndarrays
        The DCR model to be used to generate templates. Contains one array for each wavelength step.
    model_base : np.ndarray
        Coadded model built from the input exposures, without accounting for DCR.
        Used as the starting point for the iterative solution.
    n_images : int
        Number of input images used to calculate the model.
    n_step : int
        Number of sub-filter wavelength planes to model.
    observatory : lsst.afw.coord.coordLib.Observatory
        Class containing the longitude, latitude, and altitude of the observatory.
    pixel_scale : lsst.afw.geom.Angle
            Plate scale, as an Angle.
    psf : lsst.meas.algorithms KernelPsf object
        Representation of the point spread function (PSF) of the model.
    psf_size : int
        Dimension of the PSF, in pixels.
    wcs : lsst.afw.image Wcs object
        World Coordinate System of the model.
    weights : np.ndarray
        Weights of the model, calculated from the combined inverse variance of the input exposures.
    x_size : int
        Width of the model, in pixels.
    y_size : int
        Height of the model, in pixels.

    Example
    -------

    Set up:
    dcrCorr = DcrCorrection(n_step=3, repository="./test_data/",
                            obsid_range=np.arange(100, 124, 3), band_name='g')

    Generate the model:
    dcrCorr.build_model(max_iter=10)

    Use the model to make matched templates for several observations:
    template_exposure_gen = dcrCorr.generate_templates_from_model(obsid_range=[108, 109, 110],
                                                                  output_repository="./test_data_templates/")
    im_arr = []
    for exp in template_exposure_gen:
        im_arr.append(exp.getMaskedImage().getImage().getArray())
    """

    def __init__(self, obsid_range=None, repository=".", band_name='g', wavelength_step=10.,
                 n_step=None, exposures=None, detected_bit=32,
                 warp=False, instrument='lsstSim', debug_mode=False, **kwargs):
        """Load images from the repository and set up parameters.

        Parameters
        ----------
        obsid_range : int or list of ints, optional
            The observation IDs of the data to load. Not used if `exposures` is set.
        repository : str, optional
            Full path to repository with the data. Defaults to working directory
        band_name : str, optional
            Name of the bandpass-defining filter of the data. Expected values are u,g,r,i,z,y.
        wavelength_step : float, optional
            Wavelength resolution in nm, also the wavelength range of each sub-band plane.
            Overridden if `n_step` is supplied.
        n_step : int, optional
            Number of sub-band planes to use. Takes precendence over `wavelength_step`.
        exposures : List of lsst.afw.image.ExposureD objects, optional
            List of exposures to use to calculate the model.
        detected_bit : int, optional
            Value of the detected bit in the bit plane mask. This should really be read from the data!
        warp : bool, optional
            Set to true if the exposures have different wcs from the model.
            If True, the generated templates will be warped to match the wcs of each exposure.
        instrument : str, optional
            Name of the observatory.
        debug_mode : bool, optional
            Description
        **kwargs : TYPE
            Allows additional keyword arguments to be passed to `load_bandpass`.

        Raises
        ------
        ValueError
            If `exposures` is not set and no valid exposures are found in `repository`.
        """
        if exposures is None:
            self.butler = daf_persistence.Butler(repository)
            dataId_gen = self._build_dataId(obsid_range, band_name, instrument=instrument)
            self.exposures = []
            for dataId in dataId_gen:
                calexp = self.butler.get("calexp", dataId=dataId)
                self.exposures.append(calexp)
        else:
            self.exposures = exposures

        if len(self.exposures) == 0:
            raise ValueError("No valid exposures found.")

        self.debug = debug_mode
        self.instrument = instrument
        self.n_images = len(self.exposures)
        self.detected_bit = detected_bit
        self.filter_name = band_name
        psf_size_arr = []
        hour_angle_arr = []
        ref_exp_i = 0
        self.bbox = self.exposures[ref_exp_i].getBBox()
        self.wcs = self.exposures[ref_exp_i].getWcs()
        self.observatory = self.exposures[ref_exp_i].getInfo().getVisitInfo().getObservatory()

        for i, calexp in enumerate(self.exposures):
            psf_size_arr.append(calexp.getPsf().computeKernelImage().getArray().shape[0])

            hour_angle_arr.append(calexp.getInfo().getVisitInfo().getBoresightHourAngle().asRadians())

            if (i != ref_exp_i) & warp:
                wrap_warpExposure(calexp, self.wcs, self.bbox)

        if np.any(np.isnan(hour_angle_arr)):
            print("Warning: invalid hour angle in metadata. Azimuth will be used instead.")
        self.x_size, self.y_size = self.exposures[ref_exp_i].getDimensions()
        self.pixel_scale = self.exposures[ref_exp_i].getWcs().pixelScale()
        self.exposure_time = self.exposures[ref_exp_i].getInfo().getVisitInfo().getExposureTime()
        self.psf_size = int(np.min(psf_size_arr))
        self.psf = None
        self.mask = self._combine_masks()

        bandpass = DcrModel.load_bandpass(band_name=band_name, wavelength_step=wavelength_step, **kwargs)
        if n_step is not None:
            wavelength_step = (bandpass.wavelen_max - bandpass.wavelen_min) / n_step
            bandpass = DcrModel.load_bandpass(band_name=band_name, wavelength_step=wavelength_step, **kwargs)
        else:
            n_step = int(np.ceil((bandpass.wavelen_max - bandpass.wavelen_min) / bandpass.wavelen_step))
        if n_step >= self.n_images:
            print("Warning! Under-constrained system. Reducing number of frequency planes.")
            wavelength_step *= n_step / self.n_images
            bandpass = DcrModel.load_bandpass(band_name=band_name, wavelength_step=wavelength_step, **kwargs)
            n_step = int(np.ceil((bandpass.wavelen_max - bandpass.wavelen_min) / bandpass.wavelen_step))
        self.n_step = n_step
        self.bandpass = bandpass

    def calc_psf_model(self):
        """Calculate the fiducial psf from a given set of exposures, accounting for DCR.

        Returns
        -------
        None
            Sets self.psf with a lsst.meas.algorithms KernelPsf object.
        """
        n_step = 1
        bandpass = DcrModel.load_bandpass(band_name=self.filter_name, wavelength_step=None)
        n_pix = self.psf_size**2
        psf_mat = np.zeros(self.n_images*self.psf_size**2)
        for exp_i, exp in enumerate(self.exposures):
            # Use the measured PSF as the solution of the shifted PSFs.
            psf_img = exp.getPsf().computeKernelImage().getArray()
            psf_y_size, psf_x_size = psf_img.shape
            x0 = int(psf_x_size//2 - self.psf_size//2)
            x1 = x0 + self.psf_size
            y0 = int(psf_y_size//2 - self.psf_size//2)
            y1 = y0 + self.psf_size
            psf_mat[exp_i*n_pix: (exp_i + 1)*n_pix] = np.ravel(psf_img[y0:y1, x0:x1])

        dcr_shift = self._build_dcr_kernel(size=self.psf_size, bandpass=bandpass, n_step=n_step)
        psf_model_gen = solve_model(self.psf_size, psf_mat, n_step=n_step, kernel_dcr=dcr_shift)

        psf_vals = np.sum(psf_model_gen)/n_step
        psf_image = afwImage.ImageD(self.psf_size, self.psf_size)
        psf_image.getArray()[:, :] = psf_vals
        psfK = afwMath.FixedKernel(psf_image)
        self.psf = measAlg.KernelPsf(psfK)

    def build_model(self, verbose=True, max_iter=10, gain=None, clamp=None,
                    frequency_regularization=True, max_slope=None,
                    test_convergence=False, convergence_threshold=None):
        """Build a model of the sky in multiple sub-bands.

        Parameters
        ----------
        verbose : `bool`, optional
            Print additional status messages.
        max_iter : `int`, optional
            The maximum number of iterations of forward modeling allowed.
        min_iter : int, optional
            The minimum number of iterations of forward modeling before checking for convergence.
        gain : float, optional
            The weight of the new solution relative to the last solution
            when calculating the model to use for the next iteration.
            The defualt value is 1.0, and should only be changed if you know what you are doing.
        clamp : float, optional
            Restrict new solutions from being more than a factor of ``clamp`` different from the last solution
            before `gain` is applied.
            The default value is 3, chosen so that a gain of 1 restricts the change of the solution between
            iterations to less than a factor of 2.
        frequency_regularization : bool, optional
            Set to restrict variations between frequency planes
        max_slope : float, optional
            Maximum slope to allow between sub-band model planes.
            Only used if ``frequency_regularization`` is set.
        test_convergence : bool, optional
            If True, then matched templates will be generated for each image for every iteration,
            and the difference with the image will be checked to see if it is less than the previous iteration
            Any images where the difference is increasing will be excluded from the next iteration.
        convergence_threshold : float, optional
            Return once the convergence metric changes by less than ``convergence_threshold``
            between iterations.
        use_variance : bool, optional
            Set to weight pixels by their inverse variance when combining images.
        """
        if verbose:
            print("Calculating initial solution...", end="")

        if self.debug:
            self.x_size = dx
            self.y_size = dy
        # Set up an initial guess with all model planes equal as a starting point of the iterative solution
        initial_solution = np.zeros((self.y_size, self.x_size))
        initial_weights = np.zeros((self.y_size, self.x_size))
        for exp in self.exposures:
            img, inverse_var = self.extract_image(exp, airmass_weight=True, calculate_dcr_gen=False)
            initial_solution += img*inverse_var
            initial_weights += inverse_var

        weight_inds = initial_weights > 0
        self.model_base = initial_solution
        initial_solution[weight_inds] /= initial_weights[weight_inds]
        if verbose:
            print(" Done!")

        self.build_model_subroutine(initial_solution, verbose=verbose, max_iter=max_iter,
                                    frequency_regularization=frequency_regularization, max_slope=None,
                                    gain=gain, clamp=clamp,
                                    test_convergence=test_convergence,
                                    convergence_threshold=convergence_threshold)
        if verbose:
            print("\nFinished building model.")

    def build_model_subroutine(self, initial_solution, verbose=True, max_iter=10,
                               test_convergence=False, frequency_regularization=True, max_slope=None,
                               gain=None, clamp=None, convergence_threshold=None):
        """Extract the math from building the model so it can be re-used.

        Parameters
        ----------
        initial_solution : float or np.ndarray
            The model to use as a starting point for iteration.
            If a float, then a constant value is used for all pixels.
        verbose : bool, optional
            Print additional status messages.
        max_iter : int, optional
            The maximum number of iterations of forward modeling allowed.
        min_iter : int, optional
            The minimum number of iterations of forward modeling before checking for convergence.
        test_convergence : bool, optional
            If True, then matched templates will be generated for each image for every iteration,
            and the difference with the image will be checked to see if it is less than the previous iteration
            Any images where the difference is increasing will be excluded from the next iteration.
        frequency_regularization : bool, optional
            Set to restrict variations between frequency planes
        max_slope : float, optional
            Maximum slope to allow between sub-band model planes.
        gain : float, optional
            The weight of the new solution when calculating the model to use for the next iteration.
            The defualt value is 1.0, and should only be changed if you know what you are doing.
        clamp : float, optional
            Restrict new solutions from being more than a factor of `clamp` different from the last solution.
        convergence_threshold : float, optional
            Return once the convergence metric changes by less than this amount between iterations.
        use_variance : bool, optional
            Set to weight pixels by their inverse variance when combining images.

        Returns
        -------
        bool
            False if the solutions failed to converge, True otherwise.
        Sets self.model as a list of np.ndarrays
        Sets self.weights as a np.ndarray
        """
        if gain is None:
            gain = 1.
        if clamp is None:
            # The value of clamp is chosen so that the solution never changes by
            #  more than a factor of 2 between iterations: if new = old*3 then (old + new)/2 = 2*old
            clamp = 3.
        if convergence_threshold is None:
            convergence_threshold = 1e-3
        min_images = self.n_step + 1
        min_iter = 2
        last_solution = [np.zeros((self.y_size, self.x_size)) for f in range(self.n_step)]
        for f in range(self.n_step):
            last_solution[f] += np.abs(initial_solution/self.n_step)

        if verbose:
            print("Fractional change per iteration:")
        if test_convergence:
            last_convergence_metric_full = self.calc_model_metric(last_solution)
            print("Full initial convergence metric: ", last_convergence_metric_full)
            last_convergence_metric = np.mean(last_convergence_metric_full)

        exp_cut = [False for exp_i in range(self.n_images)]
        final_soln_iter = None
        converge_error = False
        for sol_iter in range(int(max_iter)):
            new_solution, inverse_var_arr = self._calculate_new_model(last_solution, exp_cut)

            # Optionally restrict variations between frequency planes
            if frequency_regularization:
                self._regularize_model_solution(new_solution, self.bandpass, max_slope=max_slope)

            # Restrict new solutions from being wildly different from the last solution
            self._clamp_model_solution(new_solution, last_solution, clamp, model_base=self.model_base)

            inds_use = inverse_var_arr[-1] > 0
            for f in range(self.n_step - 1):
                inds_use *= inverse_var_arr[f] > 0

            # Use the average of the new and last solution for the next iteration. This reduces oscillations.
            new_solution_use = [np.abs((last_solution[f] + gain*new_solution[f])/(1 + gain))
                                for f in range(self.n_step)]

            delta = (np.sum(np.abs([last_solution[f][inds_use] - new_solution_use[f][inds_use]
                                    for f in range(self.n_step)])) /
                     np.sum(np.abs([soln[inds_use] for soln in last_solution])))
            if verbose:
                print("Iteration %i: delta=%f" % (sol_iter, delta))
                last_soln_use = [soln[inds_use] for soln in last_solution]
                print("Stddev(last_solution): %f, mean(abs(last_solution)): %f"
                      % (np.std(last_soln_use), np.mean(np.abs(last_soln_use))))
                new_soln_use = [soln[inds_use] for soln in new_solution_use]
                print("Stddev(new_solution): %f, mean(abs(new_solution)): %f"
                      % (np.std(new_soln_use), np.mean(np.abs(new_soln_use))))
            if test_convergence:
                convergence_metric_full = self.calc_model_metric(new_solution_use)
                if verbose:
                    print("Full convergence metric:", convergence_metric_full)
                if sol_iter >= min_iter:
                    exp_cut = convergence_metric_full > last_convergence_metric_full
                n_exp_cut = np.sum(exp_cut)
                if n_exp_cut > 0:
                    print("%i exposure(s) cut from lack of convergence." % int(n_exp_cut))
                if (self.n_images - n_exp_cut) < min_images:
                    print("Exiting iterative solution: Too few images left.")
                    final_soln_iter = sol_iter - 1
                    converge_error = True
                    break
                last_convergence_metric_full = convergence_metric_full
                convergence_metric = np.mean(convergence_metric_full[np.logical_not(exp_cut)])
                print("Convergence metric: %f" % convergence_metric)

                if sol_iter > min_iter:
                    if convergence_metric > last_convergence_metric:
                        print("BREAK from lack of convergence")
                        final_soln_iter = sol_iter - 1
                        converge_error = True
                        break
                    convergence_check2 = (1 - convergence_threshold)*last_convergence_metric
                    if convergence_metric > convergence_check2:
                        print("BREAK after reaching convergence threshold")
                        final_soln_iter = sol_iter
                        last_solution = new_solution_use
                        break
                last_convergence_metric = convergence_metric
            last_solution = new_solution_use
        if final_soln_iter is None:
            final_soln_iter = sol_iter
        if verbose:
            print("Final solution from iteration: %i" % final_soln_iter)
        self.model = last_solution
        self.weights = np.sum(inverse_var_arr, axis=0)/self.n_step
        return converge_error

    def _calculate_new_model(self, last_solution, exp_cut, use_variance):
        """Sub-routine to calculate a new model from the residuals of forward-modeling the previous solution.

        Parameters
        ----------
        last_solution : list of np.ndarrays
            One np.ndarray for each model sub-band, from the previous iteration.
        exp_cut : List of bools
            Exposures that failed to converge in the previous iteration are flagged,
            and not included in the current iteration solution.

        Returns
        -------
        Tuple of two lists of np.ndarrays
            One np.ndarray for each model sub-band, and the associated inverse variance array.
        """
        residual_arr = [np.zeros((self.y_size, self.x_size)) for f in range(self.n_step)]
        inverse_var_arr = [np.zeros((self.y_size, self.x_size)) for f in range(self.n_step)]
        for exp_i, exp in enumerate(self.exposures):
            if exp_cut[exp_i]:
                continue
            img, inverse_var, dcr_gen = self._extract_image(exp, use_variance=use_variance)
            dcr_list = [dcr for dcr in dcr_gen]
            last_model_shift = []
            for f, dcr in enumerate(dcr_list):
                shift = (dcr.dy, dcr.dx)
                last_model_shift.append(scipy_shift(last_solution[f], shift))
            for f, dcr in enumerate(dcr_list):
                inv_shift = (-dcr.dy, -dcr.dx)
                last_model = np.zeros((self.y_size, self.x_size))
                for f2 in range(self.n_step):
                    if f2 != f:
                        last_model += last_model_shift[f2]
                img_residual = img - last_model
                residual_shift = scipy_shift(img_residual, inv_shift)
                inv_var_shift = scipy_shift(inverse_var, inv_shift)

                residual_arr[f] += residual_shift*inv_var_shift  # *weights_shift
                inverse_var_arr[f] += inv_var_shift
        new_solution = [np.zeros((self.y_size, self.x_size)) for f in range(self.n_step)]
        for f in range(self.n_step):
            inds_use = inverse_var_arr[f] > 0
            new_solution[f][inds_use] = residual_arr[f][inds_use]/inverse_var_arr[f][inds_use]
        return (new_solution, inverse_var_arr)

    @staticmethod
    def _clamp_model_solution(new_solution, last_solution, clamp, model_base=None):
        """Restrict new solutions from being wildly different from the last solution.

        Parameters
        ----------
        new_solution : list of np.ndarrays
            The model solution from the current iteration.
        last_solution : list of np.ndarrays
            The model solution from the previous iteration.
        clamp : float
            Restrict new solutions from being more than a factor of `clamp` different from the last solution.

        Returns
        -------
        None
            Modifies new_solution in place.
        """
        for s_i, solution in enumerate(new_solution):
            # Note: last_solution is always positive
            clamp_high_i = solution > clamp*last_solution[s_i]
            solution[clamp_high_i] = clamp*last_solution[s_i][clamp_high_i]
            clamp_low_i = solution < last_solution[s_i]/clamp
            solution[clamp_low_i] = last_solution[s_i][clamp_low_i]/clamp
            if model_base is not None:
                noise_threshold = np.std(solution)
                clamp_high_i2 = solution > (model_base + 3.*noise_threshold)
                solution[clamp_high_i2] = model_base[clamp_high_i2]

    @staticmethod
    def _regularize_model_solution(new_solution, bandpass, max_slope=None):
        """Calculate a slope across sub-band model planes, and clip outlier values beyond a given threshold.

        Parameters
        ----------
        new_solution : list of np.ndarrays
            The model solution from the current iteration.
        max_slope : float, optional
            Maximum slope to allow between sub-band model planes.

        Returns
        -------
        None
            Modifies new_solution in place.
        """
        if max_slope is None:
            max_slope = 1.
        n_step = len(new_solution)
        y_size, x_size = new_solution[0].shape
        solution_avg = np.sum(new_solution, axis=0)/n_step
        slope_ratio = max_slope
        sum_x = 0.
        sum_y = np.zeros((y_size, x_size))
        sum_xy = np.zeros((y_size, x_size))
        sum_xx = 0.
        wl_cen = bandpass.calc_eff_wavelen()
        for f, wl in enumerate(DcrModel._wavelength_iterator(bandpass, use_midpoint=True)):
            sum_x += wl - wl_cen
            sum_xx += (wl - wl_cen)**2
            sum_xy += (wl - wl_cen)*(new_solution[f] - solution_avg)
            sum_y += new_solution[f] - solution_avg
        slope = (n_step*sum_xy - sum_x*sum_y)/(n_step*sum_xx + sum_x**2)
        slope_cut_high = slope*bandpass.wavelen_step > solution_avg*slope_ratio
        slope_cut_low = slope*bandpass.wavelen_step < -solution_avg*slope_ratio
        slope[slope_cut_high] = solution_avg[slope_cut_high]*slope_ratio/bandpass.wavelen_step
        slope[slope_cut_low] = -solution_avg[slope_cut_low]*slope_ratio/bandpass.wavelen_step
        offset = solution_avg
        for f, wl in enumerate(DcrModel._wavelength_iterator(bandpass, use_midpoint=True)):
            new_solution[f] = offset + slope*(wl - wl_cen)

    def calc_model_metric(self, model=None):
        """Calculate a quality of fit metric for the DCR model given the set of exposures.

        Parameters
        ----------
        model : None, optional
            The DCR model. If not set, then self.model is used.

        Returns
        -------
        np.ndarray
            The calculated metric for each exposure.
        """
        metric = np.zeros(self.n_images)
        for exp_i, exp in enumerate(self.exposures):
            img_use, inverse_var = self._extract_image(exp, calculate_dcr_gen=False, use_only_detected=True)
            template = self.build_matched_template(exp, model=model, return_weights=False)
            diff = np.abs(img_use - template)
            metric[exp_i] = np.sum(diff*inverse_var)/np.sum(inverse_var)
        return metric

    def _combine_masks(self):
        """Combine multiple mask planes.

        Sets the detected mask bit if any image has a detection,
        and sets other bits only if set in all images.

        Returns
        -------
        np.ndarray
            The combined mask plane.
        """
        mask_arr = (exp.getMaskedImage().getMask().getArray() for exp in self.exposures)

        detected_mask = None
        mask_use = None
        for mask in mask_arr:
            if mask_use is None:
                mask_use = mask
            else:
                mask_use = np.bitwise_and(mask_use, mask)

            if detected_mask is None:
                detected_mask = mask & self.detected_bit
            else:
                detected_mask = np.bitwise_or(detected_mask, (mask & self.detected_bit))
        mask = np.bitwise_or(mask_use, detected_mask)
        return mask


def _calc_psf_kernel_subroutine(psf_img, size=None, size_out=None):
    """Subroutine to build a covariance matrix from an image of a PSF.

    Parameters
    ----------
    psf_img : np.ndarray
        An image of the point spread function.
    size : int, optional
        Width of the kernel in the origin image, in pixels. Default is to use the entire image.
    size_out : int, optional
        Width of the kernel in the destination image, in pixels. Default is equal to size.

    Returns
    -------
    np.ndarray
        The covariance matrix, with dimensions (size_out**2, size**2)
    """
    if size is None:
        y_size, x_size = psf_img.shape
    else:
        y_size = size
        x_size = size
    if size_out is None:
        size_out = size
    psf_y_size, psf_x_size = psf_img.shape
    if psf_x_size < x_size:
        x0 = int(x_size//2 - psf_x_size//2)
        x1 = x0 + psf_x_size
    else:
        x0 = int(psf_x_size//2 - x_size//2)
        x1 = x0 + x_size
    if psf_y_size < y_size:
        y0 = int(y_size//2 - psf_y_size//2)
        y1 = y0 + psf_y_size
    else:
        y0 = int(psf_y_size//2 - y_size//2)
        y1 = y0 + y_size
    sub_image = psf_img[y0:y1, x0:x1]

    # sub_image_use below will have dimensions (size*2, size*2), we want central (size_out, size_out)
    slice_inds = np.s_[y_size - size_out//2: y_size - size_out//2 + size_out,
                       x_size - size_out//2: x_size - size_out//2 + size_out]

    psf_mat = np.zeros((size_out*size_out, x_size*y_size))
    for j in range(y_size):
        for i in range(x_size):
            ij = i + j * x_size
            x_shift = (i, x_size - i)
            y_shift = (j, y_size - j)
            sub_image_use = np.pad(sub_image, (y_shift, x_shift), 'constant', constant_values=0.)
            psf_mat[:, ij] = np.ravel(sub_image_use[slice_inds])
    return psf_mat


def _kernel_1d(offset, size, n_substep=None, lanczos=None, debug_sinc=False):
    """Pre-compute the 1D sinc function values along each axis.

    Calculate the kernel as a simple numerical integration over the width of the offset with n_substep steps

    Parameters
    ----------
    offset : named tuple
        Tuple of start/end pixel offsets of dft locations along single axis (either x or y)
    size : int
        Dimension in pixels of the given axis.
    n_substep : int, optional
        Number of points in the numerical integration. Default is 1.
    lanczos : None, optional
        If set, the order of lanczos interpolation to use.
    debug_sinc : bool, optional
        Set to use a simple linear interpolation between nearest neighbors, instead of a sinc kernel.

    Returns
    -------
    np.ndarray
        An array containing the values of the calculated kernel.
    """
    if n_substep is None:
        n_substep = 1
    else:
        n_substep = int(n_substep)
    pi = np.pi
    pix = np.arange(size, dtype=np.float64)

    kernel = np.zeros(size, dtype=np.float64)
    for n in range(n_substep):
        loc = (size + 1)/2. + (offset.start*(n_substep - (n + 0.5)) + offset.end*(n + 0.5))/n_substep
        if loc % 1.0 == 0:
            kernel[int(loc)] += 1.0
        else:
            if debug_sinc:
                i_low = int(np.floor(loc))
                i_high = i_low + 1
                frac_high = loc - i_low
                frac_low = 1. - frac_high
                kernel[i_low] += frac_low
                kernel[i_high] += frac_high
            else:
                x = pi*(pix - loc)
                if lanczos is None:
                    kernel += np.sin(x)/x
                else:
                    kernel += (np.sin(x)/x)*(np.sin(x/lanczos)/(x/lanczos))
    return kernel/n_substep


def parallactic_angle(hour_angle, dec, lat):
    """Compute the parallactic angle given hour angle, declination, and latitude.

    Parameters
    ----------
    hour_angle : lsst.afw.geom.Angle
        Hour angle of the observation
    dec : lsst.afw.geom.Angle
        Declination of the observation.
    lat : lsst.afw.geom.Angle
        Latitude of the observatory.
    """
    y_term = np.sin(hour_angle.asRadians())
    x_term = (np.cos(dec.asRadians())*np.tan(lat.asRadians()) -
              np.sin(dec.asRadians())*np.cos(hour_angle.asRadians()))
    return np.arctan2(y_term, x_term)


def wrap_warpExposure(exposure, wcs, BBox, warpingControl=None):
    """Warp an exposure to fit a given WCS and bounding box.

    Parameters
    ----------
    exposure : lsst.afw.image.ExposureD
        An LSST exposure object. The image values will be overwritten!
    wcs : lsst.afw.image.Wcs object
        World Coordinate System to warp the image to.
    BBox : lsst.afw.geom.Box2I object
        Bounding box of the new image.
    warpingControl : afwMath.WarpingControl, optional
        Sets the interpolation parameters. Loads defualt values if None.

    Returns
    -------
    None
        Modifies exposure in place.
    """
    if warpingControl is None:
        interpLength = 10
        warpingControl = afwMath.WarpingControl("lanczos4", "", 0, interpLength)
    warpExp = afwImage.ExposureD(BBox, wcs)
    afwMath.warpExposure(warpExp, exposure, warpingControl)

    warpImg = warpExp.getMaskedImage().getImage().getArray()
    exposure.getMaskedImage().getImage().getArray()[:, :] = warpImg
    warpMask = warpExp.getMaskedImage().getMask().getArray()
    exposure.getMaskedImage().getMask().getArray()[:, :] = warpMask
    warpVariance = warpExp.getMaskedImage().getVariance().getArray()
    exposure.getMaskedImage().getVariance().getArray()[:, :] = warpVariance
    exposure.setWcs(wcs)


def solve_model(kernel_size, img_vals, n_step, kernel_dcr, kernel_ref=None, kernel_restore=None):
    """Wrapper to call a fitter using a given covariance matrix, image values, and any regularization.

    Parameters
    ----------
    kernel_size : int
        Size of the kernel to use for calculating the covariance matrix, in pixels.
    img_vals : np.ndarray
        Image data values for the pixels being used for the calculation, as a 1D vector.
    n_step : int, optional
        Number of sub-filter wavelength planes to model.
    kernel_dcr : np.ndarray
        The covariance matrix describing the effect of DCR
    kernel_ref : np.ndarray, optional
        The covariance matrix for the reference image
    kernel_restore : np.ndarray, optional
        The covariance matrix for the final restored image

    Returns
    -------
    np.ndarray
        Array of the solution values.
    """
    x_size = kernel_size
    y_size = kernel_size
    if (kernel_restore is None) or (kernel_ref is None):
        vals_use = img_vals
        kernel_use = kernel_dcr
    else:
        vals_use = kernel_restore.dot(img_vals)
        kernel_use = kernel_ref.dot(kernel_dcr)

    model_solution = scipy.optimize.nnls(kernel_use, vals_use)
    model_vals = model_solution[0]
    n_pix = x_size*y_size
    for f in range(n_step):
        yield np.reshape(model_vals[f*n_pix: (f + 1)*n_pix], (y_size, x_size))


def calculate_rotation_angle(exposure):
    """Calculate the sky rotation angle of an exposure.

    Parameters
    ----------
    exposure : lsst.afw.image.ExposureD
        An LSST exposure object.

    Returns
    -------
    lsst.afw.geom.Angle
        The rotation of the image axis, East from North.
    """
    visitInfo = exposure.getInfo().getVisitInfo()

    az = visitInfo.getBoresightAzAlt().getLongitude()
    hour_angle = visitInfo.getBoresightHourAngle()
    # Some simulated data contains invalid hour_angle metadata.
    if np.isfinite(hour_angle.asRadians()):
        dec = visitInfo.getBoresightRaDec().getDec()
        lat = visitInfo.getObservatory().getLatitude()
        p_angle = parallactic_angle(hour_angle, dec, lat)
    else:
        p_angle = az.asRadians()
    cd = exposure.getInfo().getWcs().getCDMatrix()
    cd_rot = (np.arctan2(-cd[0, 1], cd[0, 0]) + np.arctan2(cd[1, 0], cd[1, 1]))/2.
    rotation_angle = Angle(cd_rot + p_angle)
    return rotation_angle
