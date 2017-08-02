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

from lsst.afw.coord import Coord, IcrsCoord
import lsst.afw.geom as afwGeom
from lsst.afw.geom import Angle
import lsst.afw.image as afwImage
import lsst.afw.math as afwMath
from lsst.daf.base import DateTime
import lsst.daf.persistence as daf_persistence
from lsst.geom import convexHull as convexHull
import lsst.meas.algorithms as measAlg
# import lsst.pex.exceptions
import lsst.pex.policy as pexPolicy
from lsst.pipe.tasks import coaddBase
from lsst.sims.photUtils import Bandpass
from lsst.skymap import DiscreteSkyMap
from lsst.utils import getPackageDir

from .lsst_defaults import lsst_observatory, lsst_weather
from .dcr_utils import calculate_rotation_angle
from .dcr_utils import diff_refraction
from .dcr_utils import solve_model
from .dcr_utils import wrap_warpExposure
from .dcr_utils import calculate_hour_angle

__all__ = ['GenerateTemplate']

nanFloat = float("nan")
nanAngle = Angle(nanFloat)

# Temporary debugging parameters, used if debug_mode=True or self.debug=True is set.
# In debug mode, the DCR model is only calculated for pixels within [y0: y0 + dy, x0: x0 + dx]
x0 = 300
dx = 200
y0 = 500
dy = 200


class GenerateTemplate:
    """Lightweight object with only the minimum needed to generate DCR-matched template exposures.

    This class will generate template exposures suitable for
    image differencing that are matched to existing exposures in a repository.
    A model must first be generated with BuildDcrModel. That model can then be used directly, or
    persisted and later read back in with the ``load_model`` method.
    GenerateTemplate requires less memory than BuildDcrModel, since it does not store the input exposures.

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
        If set, only a small region [y0: y0 + dy, x0: x0 + dx] of the full images are used.
    default_repository : str
        Full path to repository with the data.
    exposure_time : float
        Length of the exposure, in seconds.
    filter_name : str
        Name of the bandpass-defining filter of the data. Expected values are u,g,r,i,z,y.
        Filter names are restricted by the filter profiles stored in lsst.sims.photUtils.Bandpass.
        If other filters are used, the profiles should be provided with a new Bandpass class.
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

    def __init__(self, model_repository=None, filter_name='g', butler=None, **kwargs):
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
        **kwargs : TYPE
            Any additional keyword arguments to pass to load_bandpass
        """
        self.butler = butler
        self.default_repository = model_repository
        self.bbox = None
        self.wcs = None
        self.load_model(model_repository=model_repository, filter_name=filter_name, **kwargs)

    def generate_templates_from_model(self, obsids=None, exposures=None,
                                      input_repository=None, output_repository=None,
                                      warp=False, verbose=True,
                                      output_obsid_offset=None):
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

        Yields
        ------
        lsst.afw.image ExposureF object.
            Returns a generator that builds DCR-matched templates for each exposure.
            The exposure is also written to disk if ``output_repository`` is set.

        Raises
        ------
        ValueError
            If a butler has not been previously instantiated and input_repository is not supplied.
        """
        if self.psf is None:
            self.calc_psf_model()

        if exposures is None:
            exposures = self.read_exposures(obsids, input_repository=input_repository)

        if output_repository is not None:
            # Need to also specify ``inputs`` to be able to query the butler for the required keys.
            butler = daf_persistence.Butler(outputs=output_repository, inputs=output_repository)
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
            template, variance = self.build_matched_template(calexp, el=el, rotation_angle=rotation_angle,
                                                             weather=weather)

            if verbose:
                print(" ... Done!")

            if output_obsid_offset is not None:
                obsid_out = obsid + output_obsid_offset
            else:
                obsid_out = obsid
            exposure = self.create_exposure(template, variance=variance, snap=0, ra=ra, dec=dec,
                                            boresightRotAngle=rotation_angle, weather=weather,
                                            elevation=el, azimuth=az, exposureId=obsid_out,
                                            psf=calexp.getPsf())
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
                               return_weights=True, weather=None):
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

    def _extract_image(self, exposure, airmass_weight=False, calculate_dcr_gen=True,
                       use_only_detected=False, use_variance=True):
        """Helper function to extract image array values from an exposure.

        Parameters
        ----------
        exposure : lsst.afw.image.ExposureF object
            Input single exposure to extract the image and variance planes
        airmass_weight : bool, optional
            Set to True to scale the variance by the airmass of the observation.
        calculate_dcr_gen : bool, optional
            Set to True to also return a GenerateTemplate.dcr_generator generator.
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

        visitInfo = exposure.getInfo().getVisitInfo()
        if airmass_weight:
            inverse_var /= visitInfo.getBoresightAirmass()
        if calculate_dcr_gen:
            el = visitInfo.getBoresightAzAlt().getLatitude()
            rotation_angle = calculate_rotation_angle(exposure)
            weather = visitInfo.getWeather()
            dcr_gen = self._dcr_generator(self.bandpass, pixel_scale=self.pixel_scale,
                                          observatory=self.observatory, weather=weather,
                                          elevation=el, rotation_angle=rotation_angle, use_midpoint=True)
            return (img_vals, inverse_var, dcr_gen)
        else:
            return (img_vals, inverse_var)

    def makeDataRef(self, datasetType, butler=None, level=None, **kwargs):
        """Construct a dataRef to a repository from the given data IDs.

        Parameters
        ----------
        datasetType : str
            The type of dataset to get keys for, entire collection if None.
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
    def load_bandpass(filter_name='g', wavelength_step=None, use_mirror=True, use_lens=True, use_atmos=True,
                      use_filter=True, use_detector=True, **kwargs):
        """Load in Bandpass object from sims_photUtils.

        Parameters
        ----------
        filter_name : str, optional
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
        **kwargs
            Accept and ignore any additional keyword arguments.

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
        band_range = band_dict[filter_name]
        if wavelength_step is None:
            wavelength_step = band_range[1] - band_range[0]
        bandpass = BandpassMod(wavelen_min=band_range[0], wavelen_max=band_range[1],
                               wavelen_step=wavelength_step)
        throughput_dir = getPackageDir('throughputs')
        lens_list = ['baseline/lens1.dat', 'baseline/lens2.dat', 'baseline/lens3.dat']
        mirror_list = ['baseline/m1.dat', 'baseline/m2.dat', 'baseline/m3.dat']
        atmos_list = ['atmos/atmos_11.dat']
        detector_list = ['baseline/detector.dat']
        filter_list = ['baseline/filter_' + filter_name + '.dat']
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
        """Helper function to set up consistent iterators over wavelength sub-bands of a ``bandpass``.

        Parameters
        ----------
        bandpass : lsst.sims.photUtils.Bandpass object
            Bandpass object returned by load_bandpass
        use_midpoint : bool, optional
            If set to True return the filter-weighted average wavelength.
            Otherwise, return a tuple of the starting and end wavelength.

        Yields
        -----
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

    def create_exposure(self, array, elevation, azimuth, variance=None, mask=None, bbox=None, psf=None,
                        exposureId=0, ra=nanAngle, dec=nanAngle, boresightRotAngle=nanAngle, era=None, snap=0,
                        weather=lsst_weather, **kwargs):
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
        era : lsst.afw.geom Angle, optional
            Earth rotation angle (ERA) of the observation.
            If not set it will be calculated from the latitude, longitude, RA, Dec, and elevation angle
        snap : int, optional
            Snap ID to add to the metadata of the exposure. Required to mimic Phosim output.
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
        weather : lsst.afw.coord Weather, optional
            Class containing the measured temperature, pressure, and humidity
            at the observatory during an observation
            Weather data is read from the exposure metadata if not supplied.
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
        hour_angle = calculate_hour_angle(elevation, dec, self.observatory.getLatitude())
        mjd = 59000.0 + (self.observatory.getLatitude().asDegrees()/15.0 - hour_angle.asDegrees())/24.0
        airmass = 1.0/np.sin(elevation.asRadians())
        if era is None:
            era = Angle(hour_angle.asRadians() - self.observatory.getLongitude().asRadians())
        meta = exposure.getMetadata()
        meta.add("CHIPID", "R22_S11")
        # Required! Phosim output stores the snap ID in "OUTFILE" as the last three characters in a string.
        meta.add("OUTFILE", ("SnapId_%3.3i" % snap))
        meta.add("OBSID", int(exposureId))

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
        exposure.getInfo().setVisitInfo(visitInfo)

        # Set the DCR-matched PSF
        if psf is None:
            if self.psf is None:
                self.calc_psf_model()
            psf_single = self.build_matched_psf(elevation, calculate_rotation_angle(exposure), weather)
        else:
            psf_single = psf
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
        variance[self.weights > 0] = 1./self.weights[self.weights > 0]
        reference_image = np.sum(self.model, axis=0)
        image_use, var_use, mask_use = _resize_image(reference_image, variance, self.mask, self.bbox,
                                                     patch_bbox, expand=True)
        ref_exp = self.create_exposure(image_use, variance=var_use, mask=mask_use, bbox=patch_bbox,
                                       elevation=Angle(np.pi/2), azimuth=Angle(0))
        ref_exp.getMaskedImage().getMask().addMaskPlane("CLIPPED")
        self.write_exposure(ref_exp, datasetType="deepCoadd", butler=butler)
        butler.put(self.skyMap, "dcrCoadd_skyMap")
        variance /= self.n_step
        for f in range(self.n_step):
            wl_start, wl_end = wave_gen.next()

            image_use, var_use, mask_use = _resize_image(self.model[f], variance, self.mask,
                                                         bbox_old=self.bbox, bbox_new=patch_bbox,
                                                         expand=True)

            exp = self.create_exposure(image_use, variance=var_use, mask=mask_use, bbox=patch_bbox,
                                       elevation=Angle(np.pi/2), azimuth=Angle(0),
                                       subfilt=f, nstep=self.n_step, wavelow=wl_start, wavehigh=wl_end)
            exp.getMaskedImage().getMask().addMaskPlane("CLIPPED")
            self.write_exposure(exp, datasetType="dcrCoadd", subfilter=f, butler=butler)

    def load_model(self, model_repository=None, filter_name='g', warp=False, **kwargs):
        """Depersist a DCR model from a repository and set up the metadata.

        Parameters
        ----------
        model_repository : None, optional
            Full path to the directory of the repository to load the ``dcrCoadd`` from.
            If not set, uses the existing self.butler
        filter_name : str, optional
            Common name of the filter used. For LSST, use u, g, r, i, z, or y
        **kwargs :
            Any additional keyword arguments to pass to ``load_bandpass``

        Returns
        -------
        None, but loads self.model and sets up all the needed quantities such as the psf and bandpass objects.
        """
        self.filter_name = filter_name
        model_arr = []
        dcrCoadd_gen = self.read_exposures(datasetType="dcrCoadd", input_repository=model_repository)
        for dcrCoadd in dcrCoadd_gen:
            if warp:
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
        self.weights = var_use*self.n_step
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
        bandpass_init = self.load_bandpass(filter_name=filter_name, **kwargs)
        wavelength_step = (bandpass_init.wavelen_max - bandpass_init.wavelen_min) / self.n_step
        self.bandpass = self.load_bandpass(filter_name=filter_name, wavelength_step=wavelength_step, **kwargs)

        self.psf = dcrCoadd.getPsf()
        psf_avg = self.psf.computeKernelImage().getArray()
        self.psf_size = psf_avg.shape[0]
        self.debug = False

    @staticmethod
    def _calc_offset_phase(dcr_gen, exposure=None, size=None, size_out=None, center_only=False):
        """Calculate the covariance matrix for a simple shift with no psf.

        Parameters
        ----------
        dcr_gen : generator
             A dcr generator of offsets, returned by ``_dcr_generator``.
        exposure : lsst.afw.image.ExposureF object, optional
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
            Returns the covariance matrix of an offset generated by _dcr_generator.
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
        exposure : lsst.afw.image.ExposureF object, optional
            If not supplied, the covariance matrix for all exposures in ``self.exposures`` is calculated.
        bandpass : lsst.sims.photUtils.Bandpass object
            Bandpass object returned by load_bandpass
        n_step : int, optional
            Number of sub-band planes to use. Default is to use ``self.n_step``

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
            weather = visitInfo.getWeather()
            rot_ang = calculate_rotation_angle(exp)
            dcr_gen = self._dcr_generator(bandpass, pixel_scale=self.pixel_scale, weather=weather,
                                          observatory=self.observatory, elevation=el, rotation_angle=rot_ang)
            kernel_single = self._calc_offset_phase(dcr_gen=dcr_gen, size=size,
                                                    size_out=kernel_size_intermediate)
            dcr_kernel[exp_i*n_pix_int: (exp_i + 1)*n_pix_int, :] = kernel_single
        return dcr_kernel

    def calc_psf_model_single(self, exposure):
        """Calculate the fiducial psf for a single exposure, accounting for DCR.

        Parameters
        ----------
        exposure : lsst.afw.image.ExposureF object
            A single LSST exposure object

        Returns
        -------
        np.ndarray
            Returns the fiducial PSF for an exposure, after taking out DCR effects.
        """
        visitInfo = exposure.getInfo().getVisitInfo()
        el = visitInfo.getBoresightAzAlt().getLatitude()
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
        rot_ang = calculate_rotation_angle(exposure)
        dcr_gen = self._dcr_generator(self.bandpass, pixel_scale=self.pixel_scale, weather=weather,
                                      observatory=self.observatory, elevation=el, rotation_angle=rot_ang)
        dcr_shift = self._calc_offset_phase(exposure=exposure, dcr_gen=dcr_gen,
                                            size=psf_size_use)
        # Use the entire psf provided, even if larger than than the kernel we will use to solve DCR for images
        # If the original psf is much larger than the kernel, it may be trimmed slightly by fit_psf_size above
        psf_model_gen = solve_model(psf_size_use, np.ravel(psf_img), n_step=self.n_step, kernel_dcr=dcr_shift)

        # After solving for the (potentially) large psf, store only the central portion of size kernel_size.
        psf_vals = np.sum(psf_model_gen, axis=0)/self.n_step
        return psf_vals


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
    lanczos : int, optional
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
