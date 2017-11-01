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
from builtins import next
from builtins import range
from builtins import object
from collections import namedtuple

import numpy as np
from scipy.ndimage.interpolation import shift as scipy_shift
from scipy.ndimage.morphology import binary_dilation

from lsst.afw.coord import Coord, IcrsCoord
import lsst.afw.geom as afwGeom
from lsst.afw.geom import Angle
import lsst.afw.image as afwImage
import lsst.afw.math as afwMath
from lsst.daf.base import DateTime
import lsst.daf.persistence as daf_persistence
from lsst.geom import convexHull as convexHull
import lsst.meas.algorithms as measAlg
import lsst.pex.policy as pexPolicy
from lsst.pipe.tasks import coaddBase
from lsst.skymap import DiscreteSkyMap

from .lsst_defaults import lsst_observatory, lsst_weather
from .dcr_utils import calculate_rotation_angle
from .dcr_utils import diff_refraction
from .dcr_utils import fft_shift_convolve
from .dcr_utils import wrap_warpExposure
from .dcr_utils import calculate_hour_angle
from .dcr_utils import BandpassHelper

__all__ = ['GenerateTemplate']

nanFloat = float("nan")
nanAngle = Angle(nanFloat)

# Temporary debugging parameters, used if debug_mode=True or self.debug=True is set.
# In debug mode, the DCR model is only calculated for pixels within [y0: y0 + dy, x0: x0 + dx]
x0 = 300
dx = 200
y0 = 500
dy = 200


class GenerateTemplate(object):
    """Lightweight object with only the minimum needed to generate DCR-matched template exposures.

    This class will generate template exposures suitable for
    image differencing that are matched to existing exposures in a repository.
    A model must first be generated with BuildDcrModel. That model can then be used directly, or
    persisted and later read back in with the ``load_model`` method.
    GenerateTemplate requires less memory than BuildDcrModel, since it does not store the input exposures.

    Attributes
    ----------
    bandpass : BandpassHelper object
        Bandpass object returned by load_bandpass
    bandpass_highres : BandpassHelper object
        A second Bandpass object returned by load_bandpass, at the highest resolution available.
    bbox : lsst.afw.geom.Box2I object
        A bounding box.
    butler : lsst.daf.persistence Butler object
        The butler handles persisting and depersisting data to and from a repository.
    debug : bool
        Temporary debugging option.
        If set, only a small region [y0: y0 + dy, x0: x0 + dx] of the full images are used.
    default_repository : str
        Full path to repository with the data.
    exposure_time : float
        Length of the exposure, in seconds.
    filter_name : str
        Name of the bandpass-defining filter of the data. Expected values are u,g,r,i,z,y.
        Filter names are restricted by the filter profiles stored in dcr_utils.
    mask : np.ndarray
        Mask plane of the model. This mask is saved as the mask plane of the template exposure.
    model : list of np.ndarrays
        The DCR model to be used to generate templates, calculate with ``BuildDcrModel.build_model``.
        Contains one array for each wavelength step.
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
    skyMap : lsst.skymap DiscreteSkyMap object
        A skymap defining the the tracts and patches of the model.
    wcs : lsst.afw.image Wcs object
        World Coordinate System of the model.
    weights : np.ndarray
        Weights of the model. Calculated as the sum of the inverse variances of the input exposures to
        ``BuildDcrModel.build_model``. The same ``weights`` are used for each wavelength step of the ``model``
    x_size : int
        Width of the model, in pixels.
    y_size : int
        Height of the model, in pixels.
    """

    def __init__(self, model_repository=None, filter_name='g', butler=None):
        """Restore a persisted DCR model created with BuildDcrModel.

        Only run when restoring a model or for testing; otherwise superceded by BuildDcrModel __init__.

        Parameters
        ----------
        model_repository : None, optional
            Path to the repository where the previously-generated DCR model is stored.
        filter_name : str, optional
            Name of the bandpass-defining filter of the data. Expected values are u,g,r,i,z,y.
        butler : None, lsst.daf.persistence Butler object
            Optionally pass in a pre-initialized butler to manage persistence to and from a repository.
        """
        self.butler = butler
        self.default_repository = model_repository
        self.bbox = None
        self.wcs = None
        self.load_model(model_repository=model_repository, filter_name=filter_name)

    def generate_templates_from_model(self, obsids=None, exposures=None,
                                      input_repository=None, output_repository=None,
                                      warp=False, verbose=True,
                                      output_obsid_offset=None,
                                      stretch_threshold=None):
        """Use the previously generated model and construct a dcr template image.

        Parameters
        ----------
        obsids : int, or list of ints, optional
            Single, or list of observation IDs in ``input_repository`` to load and create matched
            templates for.
            Ignored if exposures are supplied directly.
        exposures : List or generator of lsst.afw.image.ExposureF objects, optional
            List or generator of exposure objects that will have matched templates created.
            Intended primarily for unit tests that separate reading and writing from processing data.
        input_repository : str, optional
            Path to the repository where the exposure data to be matched are stored.
            Ignored if exposures are supplied directly.
        output_repository : str, optional
            Path to repository directory where templates will be saved.
            The templates will not be written to disk if ``output_repository`` is None.
        warp : bool, optional
            Set to true if the exposures have different wcs from the model.
            If True, the generated templates will be warped to match the wcs of each exposure.
        verbose : bool, optional
            Set to True to print progress messages.
        output_obsid_offset : int, optional
            Optional offset to add to the output obsids.
            Use if writing to the same repository as the input to avoid over-writing the input data.
        stretch_threshold : float, optional
            Set to simulate the effect of DCR across each sub-band by stretching the model.
            If any sub-band has DCR greater than this amount for an exposure the finite bandwidth
            FFT-based shift will be used for that instance rather than a simple shift.
        Yields
        ------
        lsst.afw.image ExposureF object.
            Returns a generator that builds DCR-matched templates for each exposure.
            The exposure is also written to disk if ``output_repository`` is set.
        """
        if self.psf is None:
            self.calc_psf_model()

        if exposures is None:
            exposures = self.read_exposures(obsids, input_repository=input_repository)

        if output_repository is not None:
            # Need to also specify ``inputs`` to be able to query the butler for the required keys.
            output_args = {'root': output_repository}
            butler = daf_persistence.Butler(outputs=output_args, inputs=output_repository)
        else:
            butler = self.butler

        for calexp in exposures:
            visitInfo = calexp.getInfo().getVisitInfo()
            obsid = visitInfo.getExposureId()
            if verbose:
                print("Working on observation %s" % obsid, end="")
            bbox_exp = calexp.getBBox()
            wcs_exp = calexp.getInfo().getWcs()
            el = visitInfo.getBoresightAzAlt().getLatitude()
            az = visitInfo.getBoresightAzAlt().getLongitude()
            ra = visitInfo.getBoresightRaDec().getLongitude()
            dec = visitInfo.getBoresightRaDec().getLatitude()
            rotation_angle = calculate_rotation_angle(calexp)
            weather = visitInfo.getWeather()
            template, variance = self.build_matched_template(calexp, el=el, weather=weather,
                                                             rotation_angle=rotation_angle,
                                                             stretch_threshold=stretch_threshold)

            if verbose:
                print(" ... Done!")

            if output_obsid_offset is not None:
                obsid_out = obsid + output_obsid_offset
            else:
                obsid_out = obsid
            exposure = self.create_exposure(template, variance=variance, snap=0, ra=ra, dec=dec,
                                            boresightRotAngle=rotation_angle, weather=weather,
                                            elevation=el, azimuth=az, exposureId=obsid_out,
                                            )
            if warp:
                wrap_warpExposure(exposure, wcs_exp, bbox_exp)
            else:
                exposure.setWcs(wcs_exp)
            if output_repository is not None:
                self.write_exposure(exposure, butler=butler)
            yield exposure

    def read_exposures(self, obsids=None, input_repository=None, datasetType="calexp"):
        """Initialize a butler and read data from the given repository.

        Parameters
        ----------
        obsids : int or list of ints, optional
            Single, or list of observation IDs in ``input_repository`` to load and create matched
            templates for.
        input_repository : str, optional
            Path to the repository where the exposure data to be matched are stored.
        datasetType : str, optional
            The type of data to be persisted. Expected values are ``'calexp'`` or ``'dcrCoadd'``

        Yields
        ------
        lsst.afw.image ExposureF object
            The specified exposures from the given repository.
        lsst.afw.table SourceTable object
            Reads the source catalogs for the given dataId instead, if ``datasetType="src"``.

        Raises
        ------
        ValueError
            If no repository is set.
        """
        if input_repository is None:
            input_repository = self.default_repository
        if input_repository is not None:
            butler = daf_persistence.Butler(inputs=input_repository)
            if self.butler is None:
                self.butler = butler
        else:
            butler = self.butler
        if butler is None:
            raise ValueError("Can't initialize butler: input_repository not set.")
        if datasetType == "calexp":
            if hasattr(obsids, '__iter__'):
                obsids_list = obsids
            else:
                obsids_list = [obsids]
            refList = [self.makeDataRef(datasetType, butler=butler, visit=obs) for obs in obsids_list]
        elif datasetType == "dcrCoadd":
            # We want to read in all of the model planes, but we don't know ahead of time how many there are.
            max_subfilters = 100
            refList = []
            for s in range(max_subfilters):
                dataRef = self.makeDataRef(datasetType, butler=butler, subfilter=s)
                if dataRef.datasetExists():
                    refList.append(dataRef)
                else:
                    break
        elif datasetType == "src":
            if hasattr(obsids, '__iter__'):
                obsids_list = obsids
            else:
                obsids_list = [obsids]
            refList = [self.makeDataRef(datasetType, butler=butler, filter_name=self.filter_name, visit=obs)
                       for obs in obsids_list]
        else:
            raise ValueError("Invalid `datasetType`")
        for dataRef in refList:
            yield dataRef.get()

    def write_exposure(self, exposure, datasetType="calexp", subfilter=None, obsid=None, butler=None):
        """Persist data using a butler.

        Parameters
        ----------
        exposure : lsst.afw.image.ExposureF object
            The exposure to be persisted to the given repository.
        datasetType : str, optional
            The type of data to be persisted. Expected values are ``'calexp'`` or ``'dcrCoadd'``
        subfilter : int, optional
            The DCR model subfilter index, only used for ```datasetType`='dcrCoadd'``
        obsid : int, optional
            Observation ID of the data to persist.
        butler : None, lsst.daf.persistence Butler object
            Optionally pass in a pre-initialized butler to manage persistence to and from a repository.
            If not set, self.butler is used.

        Returns
        -------
        None
            The data is persisted to the repository.

        Raises
        ------
        ValueError
            If an unknown ``datasetType`` is supplied.
        """
        if obsid is None:
            obsid = exposure.getInfo().getVisitInfo().getExposureId()
        if butler is None:
            butler = self.butler
        if datasetType == "calexp":
            dataRef = self.makeDataRef(datasetType, butler=butler, visit=obsid)
        elif datasetType == "dcrCoadd":
            dataRef = self.makeDataRef(datasetType, butler=butler, subfilter=subfilter)
        elif datasetType == "deepCoadd":
            dataRef = self.makeDataRef(datasetType, butler=butler)
        elif datasetType == "src":
            dataRef = self.makeDataRef(datasetType, butler=butler, visit=obsid)
        else:
            raise ValueError("Invalid `datasetType`")
        dataRef.put(exposure)

    def build_matched_template(self, exposure=None, model=None, el=None, rotation_angle=None,
                               return_weights=True, weather=None, stretch_threshold=None):
        """Sub-routine to calculate the sum of the model images shifted by DCR for a given exposure.

        Parameters
        ----------
        exposure : lsst.afw.image.ExposureF object, optional if all metadata is supplied directly.
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
        stretch_threshold : float, optional
            Set to simulate the effect of DCR across each sub-band by stretching the model.
            If any sub-band has DCR greater than this amount for an exposure the finite bandwidth
            FFT-based shift will be used for that instance rather than a simple shift.

        Returns
        -------
        np.ndarrary or (np.ndarray, np.ndarray)
            Returns a numpy ndarray of the image values for the template.
            If ``return_weights`` is set, then it returns a tuple of the image and variance arrays.
        """
        if el is None:
            el = exposure.getInfo().getVisitInfo().getBoresightAzAlt().getLatitude()
        if rotation_angle is None:
            rotation_angle = calculate_rotation_angle(exposure)
        if weather is None:
            try:
                weather = exposure.getInfo().getVisitInfo().getWeather()
            except:
                weather = lsst_weather
        dcr_gen = self._dcr_generator(self.bandpass, pixel_scale=self.pixel_scale,
                                      observatory=self.observatory, weather=weather,
                                      elevation=el, rotation_angle=rotation_angle, use_midpoint=False)
        dcr_list = [dcr for dcr in dcr_gen]
        if stretch_threshold is None:
            stretch_test = [False for dcr in dcr_list]
        else:
            stretch_test = [(abs(dcr.dy.start - dcr.dy.end) > stretch_threshold) or
                            (abs(dcr.dx.start - dcr.dx.end) > stretch_threshold) for dcr in dcr_list]
        template = np.zeros((self.y_size, self.x_size))
        if return_weights:
            weights = np.zeros((self.y_size, self.x_size))
        if model is None:
            model_use = self.model
        else:
            model_use = model
        subband_weights = self._subband_weights()
        for f, dcr in enumerate(dcr_list):
            if stretch_test[f]:
                template += fft_shift_convolve(model_use[f], dcr, weights=subband_weights[f])
                if return_weights:
                    weights += fft_shift_convolve(self.weights, dcr, weights=subband_weights[f])
            else:
                shift = ((dcr.dy.start + dcr.dy.end)/2., (dcr.dx.start + dcr.dx.end)/2.)
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
            Designed to be passed to a lsst.afw.image ExposureF through the method setPsf()
        """
        dcr_gen = self._dcr_generator(self.bandpass, pixel_scale=self.pixel_scale,
                                      observatory=self.observatory, weather=weather,
                                      elevation=el, rotation_angle=rotation_angle, use_midpoint=True)
        psf_vals = self.psf.computeKernelImage().getArray()
        psf_vals_out = np.zeros((self.psf_size, self.psf_size))

        for dcr in dcr_gen:
            shift = (dcr.dy, dcr.dx)
            psf_vals_out += scipy_shift(psf_vals, shift)
        psf_image = afwImage.ImageD(self.psf_size, self.psf_size)
        psf_image.getArray()[:, :] = psf_vals_out
        psfK = afwMath.FixedKernel(psf_image)
        psf = measAlg.KernelPsf(psfK)
        return psf

    def _extract_image(self, exposure, airmass_weight=False,
                       use_only_detected=False, use_variance=True):
        """Helper function to extract image array values from an exposure.

        Parameters
        ----------
        exposure : lsst.afw.image.ExposureF object
            Input single exposure to extract the image and variance planes
        airmass_weight : bool, optional
            Set to True to scale the variance by the airmass of the observation.
        use_only_detected : bool, optional
            If True, set all pixels to zero that do not have the detected bit set in the mask plane.
        use_variance : bool, optional
            If True, return the true inverse variance.
            Otherwise, return calculated weights in the range 0 - 1 for each pixel.

        Returns
        -------
        Returns a tuple of the image and weights (inverse variance) arrays.
        The image and weights are returned separately instead of as a MaskedImage,
        so that all the math and logic can be contained in one place.
        If `calculate_dcr_gen` is set, returns a tuple of the image, weights, and dcr generator.
        """
        img_vals = exposure.getMaskedImage().getImage().getArray()
        nan_inds = np.isnan(img_vals)
        img_vals[nan_inds] = 0.
        variance = exposure.getMaskedImage().getVariance().getArray()
        variance[nan_inds] = 0
        inverse_var = np.zeros_like(variance)
        if use_variance:
            inverse_var[variance > 0] = 1./variance[variance > 0]
        else:
            inverse_var[variance > 0] = 1.

        mask = exposure.getMaskedImage().getMask()
        # Mask all pixels with any flag other than 'DETECTED' set
        detected_bit = mask.getPlaneBitMask('DETECTED')
        ind_cut = (mask.getArray() | detected_bit) != detected_bit
        inverse_var[ind_cut] = 0.
        # Create a buffer of lower-weight pixels surrounding masked pixels.
        ind_cut2 = binary_dilation(ind_cut, iterations=2)
        inverse_var[ind_cut2] /= 2.
        if use_only_detected:
            ind_cut3 = (mask.getArray() & detected_bit) != detected_bit
            inverse_var[ind_cut3] = 0.

        if self.debug:
            slice_inds = np.s_[y0: y0 + dy, x0: x0 + dx]
            img_vals = img_vals[slice_inds]
            inverse_var = inverse_var[slice_inds]

        if airmass_weight:
            visitInfo = exposure.getInfo().getVisitInfo()
            inverse_var /= visitInfo.getBoresightAirmass()
        return (img_vals, inverse_var)

    def makeDataRef(self, datasetType, butler=None, level=None, **kwargs):
        """Construct a dataRef to a repository from the given data IDs.

        Parameters
        ----------
        datasetType : str
            The type of dataset to get keys for, entire collection if None.
        butler : lsst.daf.persistence Butler object, optional
            The butler handles persisting and depersisting data to and from a repository.
        level : None, optional
            The hierarchy level for the butler to descend to. None if it should not be restricted.
        **kwargs
            Pass in the data IDs relevant for the datasetType.

        Returns
        -------
        lsst.daf.persistence.Butler subset
            Data reference that can be used to persist and depersist data from a repository.
        """
        if butler is None:
            butler = self.butler
        # Determine the keys needed for the given datasetType
        idKeyTypeDict = butler.getKeys(datasetType=datasetType, level=level)

        # Load default values. This is a hack, and should go away once DM-9616 is completed.
        default_keys = {'filter': self.filter_name, 'tract': 0, 'patch': '0,0',
                        'raft': '2,2', 'sensor': '1,1', 'ccdnum': 10}

        key_dict = {}
        for key in idKeyTypeDict:
            try:
                key_dict[key] = kwargs[key]
            except KeyError:
                key_dict[key] = default_keys[key]

        dataId = daf_persistence.DataId(**key_dict)
        dataRef = list(butler.subset(datasetType=datasetType, level=level, dataId=dataId))
        return dataRef[0]

    @staticmethod
    def load_bandpass(filter_name='g', profile='semi', wavelength_step=None):
        """Load in Bandpass object from sims_photUtils.

        Parameters
        ----------
        filter_name : str, optional
            Common name of the filter used. For LSST, use u, g, r, i, z, or y
        profile : str, optional
            Name of the filter profile approximation to use.
            The defualt profile is a semicircle.
        wavelength_step : float, optional
            Wavelength resolution in nm, also the wavelength range of each sub-band plane.
            If not set, the native resolution of the bandpass model is used.

        Returns
        -------
        Returns a BandpassHelper object that has an interface similar to lsst.sims.photUtils.Bandpass.
        """
        bandpass = BandpassHelper(filter_name=filter_name, profile=profile, wavelen_step=wavelength_step)
        return bandpass

    @staticmethod
    def _wavelength_iterator(bandpass, use_midpoint=False):
        """Helper function to set up consistent iterators over wavelength sub-bands of a ``bandpass``.

        Parameters
        ----------
        bandpass : BandpassHelper object
            Bandpass object returned by load_bandpass
        use_midpoint : bool, optional
            If set to True return the filter-weighted average wavelength.
            Otherwise, return a tuple of the starting and end wavelength.

        Yields
        ------
        If ``use_midpoint`` is set, yields the effective wavelength of the next sub-band.
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
        bandpass : BandpassHelper object
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
            If ``use_midpoint`` is True, yields the x and y DCR offsets
            for the mid-point of the next sub-band.
            Otherwise yields a tuple of the x and y
            DCR offsets for the start and end of the next sub-band.

        """
        zenith_angle = Angle(np.pi/2) - elevation
        wavelength_midpoint = bandpass.calc_eff_wavelen()
        delta = namedtuple("delta", ["start", "end"])
        dcr = namedtuple("dcr", ["dx", "dy"])
        if use_midpoint:
            for wl in GenerateTemplate._wavelength_iterator(bandpass, use_midpoint=True):
                # Note that refract_amp can be negative, since it's relative to the midpoint of the full band
                refract_mid = diff_refraction(wavelength=wl, wavelength_ref=wavelength_midpoint,
                                              zenith_angle=zenith_angle,
                                              observatory=observatory, weather=weather)
                refract_mid_pixels = refract_mid.asArcseconds()/pixel_scale.asArcseconds()
                yield dcr(dx=refract_mid_pixels*np.sin(rotation_angle.asRadians()),
                          dy=refract_mid_pixels*np.cos(rotation_angle.asRadians()))
        else:
            for wl_start, wl_end in GenerateTemplate._wavelength_iterator(bandpass, use_midpoint=False):
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

    def create_skyMap(self, butler=None, doWrite=True):
        """Create a skyMap that is matched to the dcrCoadd.

        Parameters
        ----------
        butler : lsst.daf.persistence Butler object, optional
            The butler handles persisting and depersisting data to and from a repository.
        doWrite : bool, optional
            Set to True to persist the skyMap with the butler.

        Returns
        -------
        None
            Sets ``self.skyMap`` and persists ``skyMap`` to the repository if ``doWrite=True`` is set.
        """
        if butler is None:
            butler = self.butler
        datasetName = "dcrCoadd_skyMap"
        skyMapConfig = DiscreteSkyMap.ConfigClass()
        skyMapConfig.update(pixelScale=self.pixel_scale.asArcseconds())
        skyMapConfig.update(patchInnerDimensions=[self.x_size, self.y_size])

        boxI = afwGeom.Box2I(afwGeom.Point2I(0, 0), afwGeom.Extent2I(self.x_size, self.y_size))
        boxD = afwGeom.Box2D(boxI)
        points = [tuple(self.wcs.pixelToSky(corner).getVector()) for corner in boxD.getCorners()]
        polygon = convexHull(points)
        circle = polygon.getBoundingCircle()

        skyMapConfig.raList.append(circle.center[0])
        skyMapConfig.decList.append(circle.center[1])
        skyMapConfig.radiusList.append(circle.radius)
        self.skyMap = DiscreteSkyMap(skyMapConfig)
        if doWrite:
            butler.put(self.skyMap, datasetName)

    def create_exposure(self, array, elevation=nanAngle, azimuth=nanAngle,
                        variance=None, mask=None, bbox=None,
                        exposureId=0, ra=nanAngle, dec=nanAngle, boresightRotAngle=nanAngle, era=None, snap=0,
                        weather=lsst_weather, isCoadd=False, **kwargs):
        """Convert a numpy array to an LSST exposure with all the required metadata.

        Parameters
        ----------
        array : np.ndarray
            Numpy array to use as the image data for the exposure.
            The variance plane is suppllied as a separate optional parameter ``variance``,
            and the mask plane is read from ``self.mask``.
        elevation : lsst.afw.geom Angle
            Elevation angle of the observation
        azimuth : lsst.afw.geom Angle
            Azimuth angle of the observation
        variance : np.ndarray, optional
            Numpy array to use as the variance plane of the exposure.
            If None, the absoulte value of 'array' is used for the variance plane.
        mask : np.ndarray, optional
            Mask plane to set for the exposure.
        bbox : None, optional
            Bounding box of the exposure. If ``None`` will be set to ``self.bbox``
        exposureId : int, optional
            Observation ID of the exposure, a long int.
        ra : lsst.afw.geom Angle, optional
            The right ascension of the boresight of the target field.
        dec : lsst.afw.geom Angle, optional
            The declination of the boresight of the target field
        boresightRotAngle : lsst.afw.geom Angle, optional
            The rotation angle of the field around the boresight.
        era : lsst.afw.geom Angle, optional
            Earth rotation angle (ERA) of the observation.
            If not set it will be calculated from the latitude, longitude, RA, Dec, and elevation angle
        snap : int, optional
            Snap ID to add to the metadata of the exposure. Required to mimic Phosim output.
        weather : lsst.afw.coord Weather, optional
            Class containing the measured temperature, pressure, and humidity
            at the observatory during an observation
            Weather data is read from the exposure metadata if not supplied.
        isCoadd : bool, optional
            Description
        **kwargs
            Any additional keyword arguments will be added to the metadata of the exposure.

        Returns
        -------
        lsst.afw.image.ExposureF object
        """
        if bbox is None:
            bbox = self.bbox
        if mask is None:
            mask = self.mask
        if self.psf is None:
            self.calc_psf_model()
        exposure = afwImage.ExposureF(bbox)
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
            afwImage.FilterProperty.reset()
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

        exposure.getMaskedImage().getMask().getArray()[:, :] = mask
        meta = exposure.getMetadata()

        # Add all additional keyword arguments to the metadata.
        for add_item in kwargs:
            try:
                meta.add(add_item, kwargs[add_item])
            except:
                print("Warning: not adding keyword %s to metadata; invalid type" % add_item)
        if isCoadd:
            boresightRotAngle = Angle(0.)
            visitInfo = afwImage.VisitInfo(exposureId=int(exposureId),
                                           exposureTime=self.exposure_time,
                                           darkTime=self.exposure_time,
                                           observatory=self.observatory,
                                           weather=weather
                                           )
            psf_single = self.psf
        else:
            hour_angle = calculate_hour_angle(elevation, dec, self.observatory.getLatitude())
            mjd = 59000.0 + (self.observatory.getLatitude().asDegrees()/15.0 - hour_angle.asDegrees())/24.0
            airmass = 1.0/np.sin(elevation.asRadians())
            if era is None:
                era = Angle(hour_angle.asRadians() - self.observatory.getLongitude().asRadians())
            meta.add("CHIPID", "R22_S11")
            # Required! Phosim output stores the snap ID in "OUTFILE" as the last three characters in a string
            meta.add("OUTFILE", ("SnapId_%3.3i" % snap))
            meta.add("OBSID", int(exposureId))

            meta.add("TAI", mjd)
            meta.add("MJD-OBS", mjd)

            meta.add("EXTTYPE", "IMAGE")
            meta.add("EXPTIME", self.exposure_time)
            meta.add("AIRMASS", airmass)
            meta.add("ZENITH", 90. - elevation.asDegrees())
            meta.add("AZIMUTH", azimuth.asDegrees())

            visitInfo = afwImage.VisitInfo(exposureId=int(exposureId),
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
            psf_single = self.build_matched_psf(elevation, boresightRotAngle, weather)
        exposure.getInfo().setVisitInfo(visitInfo)

        # Set the DCR-matched PSF
        exposure.setPsf(psf_single)
        return exposure

    def export_model(self, model_repository=None):
        """Persist a DcrModel with metadata to a repository.

        Parameters
        ----------
        model_repository : None, optional
            Full path to the directory of the repository to save the dcrCoadd in
            If not set, uses the existing self.butler

        Returns
        -------
        None
        """
        if model_repository is not None:
            # Need to also specify ``inputs`` to be able to query the butler for the required keys.
            butler = daf_persistence.Butler(outputs=model_repository, inputs=model_repository)
        else:
            butler = self.butler
        tract = 0
        patch = (0, 0)
        patchInfo = self.skyMap[tract].getPatchInfo(patch)
        patch_bbox = patchInfo.getOuterBBox()

        wave_gen = self._wavelength_iterator(self.bandpass, use_midpoint=False)
        variance = np.zeros_like(self.weights)
        nonzero_inds = self.weights > 0
        variance[nonzero_inds] = 1./self.weights[nonzero_inds]
        reference_image = np.sum(self.model, axis=0)
        # variance = reference_image[:, :]
        image_use, var_use, mask_use = _resize_image(reference_image, variance*self.n_step, self.mask,
                                                     bbox_old=self.bbox, bbox_new=patch_bbox,
                                                     expand=True)
        ref_exp = self.create_exposure(image_use, variance=var_use, mask=mask_use, bbox=patch_bbox,
                                       isCoadd=True)
        ref_exp.getMaskedImage().getMask().addMaskPlane("CLIPPED")
        self.write_exposure(ref_exp, datasetType="deepCoadd", butler=butler)
        butler.put(self.skyMap, "dcrCoadd_skyMap")
        for f in range(self.n_step):
            wl_start, wl_end = next(wave_gen)
            # variance = self.model[f][:, :]
            image_use, var_use, mask_use = _resize_image(self.model[f], variance, self.mask,
                                                         bbox_old=self.bbox, bbox_new=patch_bbox,
                                                         expand=True)
            exp = self.create_exposure(image_use, variance=var_use, mask=mask_use, bbox=patch_bbox,
                                       isCoadd=True,
                                       subfilt=f, nstep=self.n_step, wavelow=wl_start, wavehigh=wl_end)
            exp.getMaskedImage().getMask().addMaskPlane("CLIPPED")
            self.write_exposure(exp, datasetType="dcrCoadd", subfilter=f, butler=butler)

    def load_model(self, model_repository=None, filter_name='g', doWarp=False):
        """Depersist a DCR model from a repository and set up the metadata.

        Parameters
        ----------
        model_repository : None, optional
            Full path to the directory of the repository to load the ``dcrCoadd`` from.
            If not set, uses the existing self.butler
        filter_name : str, optional
            Common name of the filter used. For LSST, use u, g, r, i, z, or y
        doWarp : bool, optional
            Set if the input coadds need to be warped to the reference wcs.

        Returns
        ------------------
        None, but loads self.model and sets up all the needed quantities such as the psf and bandpass objects.
        """
        self.filter_name = filter_name
        model_arr = []
        dcrCoadd_gen = self.read_exposures(datasetType="dcrCoadd", input_repository=model_repository)
        for dcrCoadd in dcrCoadd_gen:
            if doWarp:
                wrap_warpExposure(dcrCoadd, self.wcs, self.bbox)
            model_in = dcrCoadd.getMaskedImage().getImage().getArray()
            var_in = dcrCoadd.getMaskedImage().getVariance().getArray()
            mask_in = dcrCoadd.getMaskedImage().getMask().getArray()
            model_use, var_use, mask_use = _resize_image(model_in, var_in, mask_in, bbox_new=self.bbox,
                                                         bbox_old=dcrCoadd.getBBox(), expand=False)
            model_arr.append(model_use)

        self.model = model_arr
        self.n_step = len(self.model)
        # The weights should be identical for all subfilters.
        self.weights = np.zeros_like(var_use)
        nonzero_inds = var_use > 0
        self.weights[nonzero_inds] = 1./var_use[nonzero_inds]
        # self.weights = var_use*self.n_step
        # The masks should be identical for all subfilters
        self.mask = mask_use

        skyInfo = coaddBase.getSkyInfo("dcr", self.makeDataRef("dcrCoadd", subfilter=0))
        self.skyMap = skyInfo.skyMap

        self.wcs = dcrCoadd.getWcs()
        self.bbox = skyInfo.patchInfo.getInnerBBox()
        x_size, y_size = self.bbox.getDimensions()
        self.n_step = len(self.model)
        self.x_size = x_size
        self.y_size = y_size
        self.pixel_scale = self.wcs.pixelScale()
        self.exposure_time = dcrCoadd.getInfo().getVisitInfo().getExposureTime()
        self.observatory = dcrCoadd.getInfo().getVisitInfo().getObservatory()
        bandpass_init = self.load_bandpass(filter_name=filter_name)
        wavelength_step = (bandpass_init.wavelen_max - bandpass_init.wavelen_min) / self.n_step
        self.bandpass = self.load_bandpass(filter_name=filter_name, wavelength_step=wavelength_step)
        self.bandpass_highres = self.load_bandpass(filter_name=filter_name, wavelength_step=None)

        self.psf = dcrCoadd.getPsf()
        psf_avg = self.psf.computeKernelImage().getArray()
        self.psf_size = psf_avg.shape[0]
        self.debug = False

    def _subband_weights(self):
        """Helper function to evaluate the filter profile across each sub-band.

        Returns
        -------
        list of np.ndarrays
            An array of bandpass values across each sub-band.
        """
        weights = []
        for wl_start, wl_end in self._wavelength_iterator(self.bandpass, use_midpoint=False):
            inds_use = (self.bandpass_highres.wavelen < wl_end) & (self.bandpass_highres.wavelen > wl_start)
            weights.append(self.bandpass_highres.sb[inds_use])
        return weights


def _resize_image(image, variance, mask, bbox_old, bbox_new=None, bitmask=255, expand=True):
    """Temporary function to resize an image to match a given bounding box."""
    if bbox_new is None:
        x_full = bbox_old.getDimensions().getX()
        y_full = bbox_old.getDimensions().getY()
        x_size = np.sum([((mask[:, i] & bitmask) == 0).any() for i in range(x_full)])
        y_size = np.sum([((mask[j, :] & bitmask) == 0).any() for j in range(y_full)])
        bbox_new = afwGeom.Box2I(afwGeom.Point2I(0, 0), afwGeom.ExtentI(x_size, y_size))
    shape = (bbox_new.getDimensions().getY(), bbox_new.getDimensions().getX())
    image_return = np.zeros(shape, dtype=image.dtype)
    variance_return = np.zeros(shape, dtype=variance.dtype)
    mask_return = np.zeros(shape, dtype=mask.dtype) + bitmask

    if expand:
        slice_inds = bbox_old.getSlices()
        image_return[slice_inds] = image
        variance_return[slice_inds] = variance
        mask_return[slice_inds] = mask
    else:
        slice_inds = bbox_new.getSlices()
        image_return = image[slice_inds]
        variance_return = variance[slice_inds]
        mask_return = mask[slice_inds]
    return (image_return, variance_return, mask_return)
