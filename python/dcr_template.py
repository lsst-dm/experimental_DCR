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
from scipy.linalg import pinv2 as scipy_invert
import scipy.optimize.nnls
from scipy.signal import tukey as scipy_tukey

from lsst.daf.base import DateTime
import lsst.daf.persistence as daf_persistence
from lsst.afw.coord import Coord, IcrsCoord, Observatory
import lsst.afw.geom as afwGeom
from lsst.afw.geom import Angle
import lsst.afw.image as afwImage
import lsst.afw.math as afwMath
import lsst.meas.algorithms as measAlg
import lsst.pex.exceptions
import lsst.pex.policy as pexPolicy
from lsst.sims.photUtils import Bandpass, PhotometricParameters
from lsst.utils import getPackageDir

from .calc_refractive_index import diff_refraction

__all__ = ("DcrModel", "DcrCorrection")

nanFloat = float("nan")
nanAngle = Angle(nanFloat)
lsst_lat = Angle(np.radians(-30.244639))
lsst_lon = Angle(np.radians(-70.749417))
lsst_alt = 2663.


class DcrModel:
    """!Lightweight object with only the minimum needed to generate DCR-matched template exposures."""

    def __init__(self, model_repository=None, band_name='g', debug_mode=False, **kwargs):
        """!Restore a persisted DcrModel.

        Only run when restoring a model or for testing; otherwise superceded by DcrCorrection __init__.
        @param model_repository  path to the repository where the previously-generated DCR model is stored.
        @param band_name  name of the bandpass-defining filter of the data. Expected values are u,g,r,i,z,y.
        @param debug_mode  if set to True, only use a subset of the data for speed (used in _edge_test)
        @param **kwargs  Any additional keyword arguments to pass to load_bandpass
        """
        self.debug = debug_mode
        self.butler = None
        self.load_model(model_repository=model_repository, band_name=band_name, **kwargs)

    def generate_templates_from_model(self, obsid_range=None, exposures=None, add_noise=False,
                                      repository=None, output_repository=None,
                                      instrument='lsstSim', warp=False, verbose=True,
                                      output_obsid_offset=None):
        """!Use the previously generated model and construct a dcr template image.

        @param obsid_range  single, or list of observation IDs in repository to create matched
                            templates for. Ignored if exposures are supplied directly.
        @param exposures  optional, list or generator of exposure objects that will
                          have matched templates created.
        @param add_noise  If set to true, add Poisson noise to the template based on the variance.
        @param repository  path to the repository where the exposure data to be matched are stored.
                           Ignored if exposures are supplied directly.
        @param output_repository  path to repository directory where templates will be saved.
        @param kernel_size  [optional] size, in pixels, of the region surrounding each image pixel that DCR
                            shifts are calculated. Default is to use the same value the model was created with
        @param instrument  Name of the observatory.
        @param warp  Flag. Set to true if the exposures have different wcs from the model.
                     If True, the generated templates will be warped to match the wcs of each exposure.
        @param verbose  Flag, set to True to print progress messages.
        @param use_nonnegative  Flag, set to True to use a true non-negative least squares solution [SLOW]
        @return Returns a generator that builds DCR-matched templates for each exposure.
        """
        if output_repository is not None:
            butler_out = daf_persistence.Butler(output_repository)
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
                dataId_gen = self._build_dataId(obsid_range, self.photoParams.bandpass, instrument=instrument)
                exposures = (calexp for calexp in
                             (butler.get("calexp", dataId=dataId) for dataId in dataId_gen))
            else:
                raise ValueError("One of obsid_range or exposures must be set.")

        self.instrument = instrument
        if self.psf_avg is None:
            self.calc_psf_model()

        model_inverse_weights = np.zeros_like(self.weights)
        weight_inds = self.weights > 0
        model_inverse_weights[weight_inds] = 1./self.weights[weight_inds]

        if obsid_range is not None:
            if not hasattr(obsid_range, '__iter__'):
                obsid_range = [obsid_range]
        for exp_i, calexp in enumerate(exposures):
            if obsid_range is not None:
                obsid = obsid_range[exp_i]
            else:
                obsid = self._fetch_metadata(calexp.getMetadata(), "OBSID", default_value=0)
            if verbose:
                print("Working on observation %s" % obsid)
            visitInfo = calexp.getInfo().getVisitInfo()
            bbox_exp = calexp.getBBox()
            wcs_exp = calexp.getInfo().getWcs()
            el = visitInfo.getBoresightAzAlt().getLatitude()
            kernel_dcr = self.build_dcr_kernel(exposure=calexp)
            kernel_model = self.build_psf_kernel(exposure=calexp, use_full=False, expand_intermediate=True)
            kernel_exp = self.build_psf_kernel(exposure=calexp, use_full=True, expand_intermediate=True)
            lstsq_kernel = build_lstsq_kernel(kernel_dcr=kernel_dcr, kernel_ref=kernel_model,
                                              kernel_restore=kernel_exp, invert=True)

            template, weights = self.solver_wrapper(self.model, kernel_dcr=kernel_dcr,
                                                    kernel_ref=kernel_model,
                                                    inverse_weights=model_inverse_weights,
                                                    kernel_restore=kernel_exp, lstsq_kernel=lstsq_kernel,
                                                    verbose=verbose, use_nonnegative=use_nonnegative,
                                                    center_only=debug_solver, **debug_kwargs)
            az = visitInfo.getBoresightAzAlt().getLongitude()
            lat = visitInfo.getObservatory().getLatitude()
            lon = visitInfo.getObservatory().getLongitude()
            alt = visitInfo.getObservatory().getElevation()
            rotation_angle = calculate_rotation_angle(calexp)
            if verbose:
                print("Finished building template.")
            template = template[0]  # template is returned as a single element list.
            # Weights may be different for each exposure
            template[weights > 0] /= weights[weights > 0]
            template[weights == 0] = 0.0
            if add_noise:
                variance_level = np.median(calexp.getMaskedImage().getVariance().getArray())
                rand_gen = np.random
                template += rand_gen.normal(scale=np.sqrt(variance_level), size=template.shape)

            if output_obsid_offset is not None:
                obsid_out = obsid + output_obsid_offset
            else:
                obsid_out = obsid
            dataId_out = self._build_dataId(obsid_out, self.photoParams.bandpass, instrument=instrument)[0]
            exposure = self.create_exposure(template, variance=variance, snap=0,
                                            boresightRotAngle=rotation_angle.asDegrees(),
                                            elevation=el, azimuth=az, latitude=lat,
                                            longitude=lon, altitude=alt, obsid=obsid_out)
            if warp:
                wrap_warpExposure(exposure, wcs_exp, bbox_exp)
            if output_repository is not None:
                butler_out.put(exposure, "calexp", dataId=dataId_out)
            yield exposure

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
        """Helper function to extract image array values from an exposure."""
        img_residual = exposure.getMaskedImage().getImage().getArray()
        nan_inds = np.isnan(img_residual)
        img_residual[nan_inds] = 0.
        variance = exposure.getMaskedImage().getVariance().getArray()
        variance[nan_inds] = 0
        inverse_var = np.zeros_like(variance)
        inverse_var[variance > 0] = 1./variance[variance > 0]

        mask = exposure.getMaskedImage().getMask().getArray()
        ind_cut = (mask | self.detected_bit) != self.detected_bit
        inverse_var[ind_cut] = 0
        visitInfo = exposure.getInfo().getVisitInfo()
        if airmass_weight:
            inverse_var /= visitInfo.getBoresightAirmass()
        if calculate_dcr_gen:
            el = visitInfo.getBoresightAzAlt().getLatitude()
            rotation_angle = calculate_rotation_angle(exposure)
            dcr_gen = DcrModel.dcr_generator(self.bandpass, pixel_scale=self.pixel_scale,
                                             elevation=el, rotation_angle=rotation_angle, use_midpoint=True)
            return (img_residual, inverse_var, dcr_gen)
        else:
            return (img_residual, inverse_var)

    @staticmethod
    def _fetch_metadata(metadata, property_name, default_value=None):
        """!Simple wrapper to extract metadata from an exposure, with some error handling.

        @param metadata  An LSST exposure metadata object
        @param property_name  String, name of the property to be extracted
        @param default_value  Value to be returned if the property is not found in the exposure metadata.
        @return Returns the value of property_name from the metadata of exposure.
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
        """!Construct a dataId dictionary for the butler to find a calexp.

        @param obsid_range  A single obsid or list of obsids.
        @param band  name of the bandpass-defining filter of the data. Expected values are u,g,r,i,z,y.
        @param instrument  Name of the observatory. Each observatory defines their own dataIds.
        @return Return a list of dataIds for the butler to use to load a calexp from a repository
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
        """!Construct a dataId dictionary for the butler to find a dcrModel.

        @param band  name of the bandpass-defining filter of the data. Expected values are u,g,r,i,z,y.
        @param subfilter  DCR model index within the band.
        @return Return a dataId for the butler to use to load a dcrModel from a repository
        """
        if subfilter is None:
            dataId = {'filter': band, 'tract': 0, 'patch': '0'}
        else:
            dataId = {'filter': band, 'tract': 0, 'patch': '0', 'subfilter': subfilter}
        return(dataId)

    @staticmethod
    def create_wcs(bbox=None, pixel_scale=None, ra=nanAngle, dec=nanAngle, sky_rotation=nanAngle):
        """!Create a wcs (coordinate system).

        @param bbox  A bounding box.
        @param pixel_scale  Plate scale, in arcseconds.
        @param ra  Right Ascension of the reference pixel, as an Angle.
        @param dec  Declination of the reference pixel, as an Angle.
        @param sky_rotation  Rotation of the image axis, East from North.
        @return  Returns a WCS object.
        """
        crval = IcrsCoord(ra, dec)
        crpix = afwGeom.Box2D(bbox).getCenter()
        cd1_1 = (pixel_scale * afwGeom.arcseconds * np.cos(sky_rotation.asRadians())).asDegrees()
        cd1_2 = (-pixel_scale * afwGeom.arcseconds * np.sin(sky_rotation.asRadians())).asDegrees()
        cd2_1 = (pixel_scale * afwGeom.arcseconds * np.sin(sky_rotation.asRadians())).asDegrees()
        cd2_2 = (pixel_scale * afwGeom.arcseconds * np.cos(sky_rotation.asRadians())).asDegrees()
        return(afwImage.makeWcs(crval, crpix, cd1_1, cd1_2, cd2_1, cd2_2))

    # NOTE: This function was copied from StarFast.py
    @staticmethod
    def load_bandpass(band_name='g', wavelength_step=None, use_mirror=True, use_lens=True, use_atmos=True,
                      use_filter=True, use_detector=True, **kwargs):
        """!Load in Bandpass object from sims_photUtils.

        @param band_name  Common name of the filter used. For LSST, use u, g, r, i, z, or y
        @param wavelength_step  Wavelength resolution in nm, also the wavelength range of each sub-band plane
        @param use_mirror  Flag, include mirror in filter throughput calculation?
        @param use_lens  Flag, use LSST lens in filter throughput calculation?
        @param use_atmos  Flag, use standard atmosphere transmission in filter throughput calculation?
        @param use_filter  Flag, use LSST filters in filter throughput calculation?
        @param use_detector  Flag, use LSST detector efficiency in filter throughput calculation?
        @return Returns a bandpass object.
        """
        class BandpassMod(Bandpass):
            """Customize a few methods of the Bandpass class from sims_photUtils."""

            def calc_eff_wavelen(self, wavelength_min=None, wavelength_max=None):
                """Calculate effective wavelengths for filters."""
                # This is useful for summary numbers for filters.
                # Calculate effective wavelength of filters.
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

    # NOTE: This function was copied from StarFast.py
    @staticmethod
    def _wavelength_iterator(bandpass, use_midpoint=False):
        """!Define iterator to ensure that loops over wavelength are consistent.

        @param bandpass  Bandpass object returned by load_bandpass
        @param use_midpoint  if set, return the filter-weighted average wavelength.
                             Otherwise, return a tuple of the starting and end wavelength.
        @return Returns a generator that iterates through sub-bands of a given bandpass.
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

    # NOTE: This function was modified from StarFast.py
    @staticmethod
    def dcr_generator(bandpass, pixel_scale=None, elevation=Angle(np.radians(50.0)),
                      rotation_angle=Angle(np.radians(0.0)), use_midpoint=False):
        """!Call the functions that compute Differential Chromatic Refraction (relative to mid-band).

        @param bandpass  bandpass object created with load_bandpass
        @param pixel_scale  plate scale in arcsec/pixel
        @param elevation  elevation angle of the center of the image, as a lsst.afw.geom Angle.
        @param azimuth  azimuth angle of the observation, as a lsst.afw.geom Angle.
        @return  Returns a generator that produces named tuples containing the x and y offsets, in pixels.
        """
        zenith_angle = Angle(np.pi/2) - elevation
        wavelength_midpoint = bandpass.calc_eff_wavelen()
        delta = namedtuple("delta", ["start", "end"])
        dcr = namedtuple("dcr", ["dx", "dy"])
        for wl_start, wl_end in DcrModel._wavelength_iterator(bandpass, use_midpoint=False):
            # Note that refract_amp can be negative, since it's relative to the midpoint of the full band
            refract_start = diff_refraction(wavelength=wl_start, wavelength_ref=wavelength_midpoint,
                                            zenith_angle=zenith_angle.asDegrees())
            refract_end = diff_refraction(wavelength=wl_end, wavelength_ref=wavelength_midpoint,
                                          zenith_angle=zenith_angle.asDegrees())
            refract_start *= 3600.0 / pixel_scale  # Refraction initially in degrees, convert to pixels.
            refract_end *= 3600.0 / pixel_scale
            dx = delta(start=refract_start*np.sin(azimuth.asRadians()),
                       end=refract_end*np.sin(azimuth.asRadians()))
            dy = delta(start=refract_start*np.cos(azimuth.asRadians()),
                       end=refract_end*np.cos(azimuth.asRadians()))
            yield dcr(dx=dx, dy=dy)

        else:

    # NOTE: This function was copied from StarFast.py
    def create_exposure(self, array, variance=None, elevation=None, azimuth=None,
                        latitude=lsst_lat, longitude=lsst_lon, altitude=lsst_alt, snap=0,
                        exposureId=0, ra=nanAngle, dec=nanAngle, boresightRotAngle=nanFloat, **kwargs):
        """Convert a numpy array to an LSST exposure, and units of electron counts.

        @param array  numpy array to use as the data for the exposure
        @param variance  optional numpy array to use as the variance plane of the exposure.
                         If None, the absoulte value of 'array' is used for the variance plane.
        @param elevation  Elevation angle of the observation, as a lsst.afw.geom Angle.
        @param azimuth  Azimuth angle of the observation, as a lsst.afw.geom Angle.
        @param snap  snap ID to add to the metadata of the exposure. Required to mimic Phosim output.
        @param exposureId: observation ID of the exposure, a long int.
        @param ra  The right ascension of the boresight of the target field, as an Angle.
        @param dec  The declination of the boresight of the target field, as an Angle.
        @param boresightRotAngle  The rotation angle of the field around the boresight, in degrees.
        @param **kwargs  Any additional keyword arguments will be added to the metadata of the exposure.
        @return  Returns an LSST exposure.
        """
        exposure = afwImage.ExposureD(self.bbox)
        exposure.setWcs(self.wcs)
        # We need the filter name in the exposure metadata, and it can't just be set directly
        try:
            exposure.setFilter(afwImage.Filter(self.photoParams.bandpass))
        except:
            filterPolicy = pexPolicy.Policy()
            filterPolicy.add("lambdaEff", self.bandpass.calc_eff_wavelen())
            afwImage.Filter.define(afwImage.FilterProperty(self.photoParams.bandpass, filterPolicy))
            exposure.setFilter(afwImage.Filter(self.photoParams.bandpass))
            # Need to reset afwImage.Filter to prevent an error in future calls to daf_persistence.Butler
            afwImage.FilterProperty_reset()
        exposure.setPsf(self.psf)
        exposure.getMaskedImage().getImage().getArray()[:, :] = array
        if variance is None:
            variance = np.abs(array)
        exposure.getMaskedImage().getVariance().getArray()[:, :] = variance

        if self.mask is not None:
            exposure.getMaskedImage().getMask().getArray()[:, :] = self.mask
        ha_term1 = np.sin(elevation.asRadians())
        ha_term2 = np.sin(dec.asRadians())*np.sin(latitude.asRadians())
        ha_term3 = np.cos(dec.asRadians())*np.cos(latitude.asRadians())
        hour_angle = np.arccos((ha_term1 - ha_term2) / ha_term3)
        mjd = 59000.0 + (latitude.asDegrees()/15.0 - hour_angle*180/np.pi)/24.0
        airmass = 1.0/np.sin(elevation.asRadians())
        meta = exposure.getMetadata()
        meta.add("CHIPID", "R22_S11")
        # Required! Phosim output stores the snap ID in "OUTFILE" as the last three characters in a string.
        meta.add("OUTFILE", ("SnapId_%3.3i" % snap))

        meta.add("TAI", mjd)
        meta.add("MJD-OBS", mjd)

        meta.add("EXTTYPE", "IMAGE")
        meta.add("EXPTIME", self.photoParams.exptime)
        meta.add("AIRMASS", airmass)
        meta.add("ZENITH", 90. - elevation.asDegrees())
        meta.add("AZIMUTH", azimuth.asDegrees())
        # Add all additional keyword arguments to the metadata.
        for add_item in kwargs:
            meta.add(add_item, kwargs[add_item])

        visitInfo = afwImage.makeVisitInfo(
            exposureId=int(exposureId),
            exposureTime=self.photoParams.exptime,
            darkTime=self.photoParams.exptime,
            date=DateTime(mjd),
            ut1=mjd,
            boresightRaDec=IcrsCoord(ra, dec),
            boresightAzAlt=Coord(azimuth, elevation),
            boresightAirmass=airmass,
            boresightRotAngle=Angle(np.radians(boresightRotAngle)),
            observatory=Observatory(longitude, latitude, altitude),)
        exposure.getInfo().setVisitInfo(visitInfo)
        return exposure

    def export_model(self, model_repository=None):
        """!Persist a DcrModel with metadata to a repository.

        Parameters
        ----------
        model_repository : None, optional
            Full path to the directory of the repository to save the dcrModel in
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
                                       subfilt=f, nstep=self.n_step, wavelow=wl_start, wavehigh=wl_end,
                                       wavestep=self.bandpass.wavelen_step, telescop=self.instrument)
            butler.put(exp, "dcrModel", dataId=self._build_model_dataId(self.photoParams.bandpass, f))

    def load_model(self, model_repository=None, band_name='g', **kwargs):
        """!Depersist a DcrModel from a repository and set up the metadata.

        @param model_repository  full path to the directory of the repository to load the dcrModel from.
        @param band_name  Common name of the filter used. For LSST, use u, g, r, i, z, or y
        @param **kwargs  Any additional keyword arguments to pass to load_bandpass
        @return No return value, but sets up all the needed quantities such as the psf and bandpass objects.
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
        self.y_size, self.x_size = dcrModel.getDimensions()
        self.pixel_scale = self.wcs.pixelScale().asArcseconds()
        exposure_time = dcrModel.getInfo().getVisitInfo().getExposureTime()
        self.photoParams = PhotometricParameters(exptime=exposure_time, nexp=1, platescale=self.pixel_scale,
                                                 bandpass=band_name)
        self.bbox = dcrModel.getBBox()
        self.instrument = self._fetch_metadata(meta, "TELESCOP", default_value='lsstSim')
        self.bandpass = DcrModel.load_bandpass(band_name=band_name, wavelength_step=wave_step, **kwargs)

        self.psf = dcrModel.getPsf()
        psf_avg = self.psf.computeKernelImage().getArray()
        self.psf_size = psf_avg.shape[0]
        self.psf_avg = psf_avg

    @staticmethod
    def calc_offset_phase(exposure=None, dcr_gen=None, size=None, size_out=None,
                          center_only=False):
        """!Calculate the covariance matrix for a simple shift with no psf.

        @param exposure  An LSST exposure object. Only needed if size is not specified.
        @param dcr_gen  A dcr generator of offsets, returned by dcr_generator.
        @param size  Width in pixels of the region used in the origin image. Default is entire image
        @param size_out  Width in pixels of the region used in the destination image. Default is same as size
        @param center_only  Flag, set to True to calculate the covariance for only the center pixel.
        @return Returns the covariance matrix of an offset generated by dcr_generator in the form (dx, dy)
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


    def build_dcr_kernel(self, size, expand_intermediate=False, exposure=None,
                         bandpass=None, n_step=None):
        """!Calculate the DCR covariance matrix for a set of exposures, or a single exposure.

        @param size  Width in pixels of the region used in the origin image. Default is entire image
        @param expand_intermediate  If set, calculate the covariance matrix between the region of pixels in
                                    the origin image and a region twice as wide in the destination image.
                                    This helps avoid edge effects when computing A^T A.
        @param exposure Optional, an LSST exposure object. If not supplied, the covariance matrix for all
                        exposures in self.exposures is calculated.
        @param bandpass  Optional. Bandpass object created with load_bandpass
        @return Returns the covariance matrix for the exposure(s).
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
            dcr_gen = DcrModel.dcr_generator(bandpass, pixel_scale=self.pixel_scale,
                                             elevation=el, rotation_angle=az)
            kernel_single = DcrModel.calc_offset_phase(dcr_gen=dcr_gen, size=size_use,
                                                       size_out=kernel_size_intermediate)
            dcr_kernel[exp_i*n_pix_int: (exp_i + 1)*n_pix_int, :] = kernel_single
        return dcr_kernel

    def calc_psf_model_single(self, exposure):
        """!Calculate the fiducial psf for a single exposure, accounting for DCR.

        @param exposure  A single LSST exposure object.
        @return  Returns the fiducial PSF for an exposure, after taking out DCR effects.
        """
        visitInfo = exposure.getInfo().getVisitInfo()
        el = visitInfo.getBoresightAzAlt().getLatitude()
        az = visitInfo.getBoresightAzAlt().getLongitude()

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
        dcr_gen = DcrModel.dcr_generator(self.bandpass, pixel_scale=self.pixel_scale,
                                         elevation=el, azimuth=az)
        dcr_shift = DcrModel.calc_offset_phase(exposure=exposure, dcr_gen=dcr_gen,
                                               size=psf_size_use)
        # Assume that the PSF does not change between sub-bands.
        reg_weight = np.max(np.abs(dcr_shift))*100.
        regularize_psf = DcrCorrection.build_regularization(psf_size_use, n_step=self.n_step,
                                                            weight=reg_weight, frequency_regularization=True)
        # Use the entire psf provided, even if larger than than the kernel we will use to solve DCR for images
        # If the original psf is much larger than the kernel, it may be trimmed slightly by fit_psf_size above
        psf_model_gen = solve_model(psf_size_use, np.ravel(psf_img), n_step=self.n_step,
                                    use_nonnegative=True, regularization=regularize_psf, kernel_dcr=dcr_shift)

        # After solving for the (potentially) large psf, store only the central portion of size kernel_size.
        psf_vals = np.sum(psf_model_gen, axis=0)/self.n_step
        return psf_vals


class DcrCorrection(DcrModel):
    """!Class that loads LSST calibrated exposures and produces airmass-matched template images."""

    def __init__(self, repository=".", obsid_range=None, band_name='g', wavelength_step=10,
                 n_step=None, debug_mode=False, kernel_size=None, exposures=None,
                 warp=False, instrument='lsstSim', **kwargs):
        """!Load images from the repository and set up parameters.

        @param repository  path to repository with the data. String, defaults to working directory
        @param obsid_range  obsid or range of obsids to process.
        @param band_name  Common name of the filter used. For LSST, use u, g, r, i, z, or y
        @param wavelength_step  Overridden by n_step. Sub-filter width, in nm.
        @param n_step  Number of sub-filter wavelength planes to model. Optional if wavelength_step supplied.
        @param debug_mode  Flag. Set to True to run in debug mode, which may have unpredictable behavior.
        @param kernel_size  Size of the kernel to use for calculating the covariance matrix, in pixels.
                            Note that kernel_size must be odd, so even values will be increased by one.
                            Optional. If missing, will be calculated from the maximum shift predicted from DCR
        @param exposures  A list of LSST exposures to use as input to the DCR calculation.
                          Optional. If missing, exposures will be loaded from the specified repository.
        @param warp  Flag. Set to true if the exposures have different wcs from the model.
                     If True, the generated templates will be warped to match the wcs of each exposure.
        @param instrument  Name of the observatory.
        @param **kwargs  Allows additional keyword arguments to be passed to load_bandpass.
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

        self.instrument = instrument
        self.n_images = len(self.exposures)
        self.detected_bit = detected_bit
        psf_size_arr = np.zeros(self.n_images)
        self.airmass_arr = np.zeros(self.n_images, dtype=np.float64)
        self.elevation_arr = []
        self.azimuth_arr = []
        self.sky_rotation_arr = []
        ref_exp_i = 0
        self.bbox = self.exposures[ref_exp_i].getBBox()
        self.wcs = self.exposures[ref_exp_i].getWcs()

        for i, calexp in enumerate(self.exposures):
            visitInfo = calexp.getInfo().getVisitInfo()
            self.airmass_arr[i] = visitInfo.getBoresightAirmass()
            psf_size_arr[i] = calexp.getPsf().computeKernelImage().getArray().shape[0]

            el = visitInfo.getBoresightAzAlt().getLatitude()
            az = visitInfo.getBoresightAzAlt().getLongitude()
            rotation_angle = calculate_rotation_angle(calexp)
            self.elevation_arr.append(el)
            self.azimuth_arr.append(az)
            self.sky_rotation_arr.append(rotation_angle)

            if (i != ref_exp_i) & warp:
                wrap_warpExposure(calexp, self.wcs, self.bbox)

        self.x_size, self.y_size = self.exposures[ref_exp_i].getDimensions()
        self.pixel_scale = self.exposures[ref_exp_i].getWcs().pixelScale().asArcseconds()
        exposure_time = self.exposures[ref_exp_i].getInfo().getVisitInfo().getExposureTime()
        self.psf_size = int(np.min(psf_size_arr))
        self.psf_avg = None
        self.mask = None
        self._combine_masks()

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
        self.photoParams = PhotometricParameters(exptime=exposure_time, nexp=1, platescale=self.pixel_scale,
                                                 bandpass=band_name)


    def calc_psf_model(self):
        """!Calculate the fiducial psf from a given set of exposures, accounting for DCR."""
        n_step = 1
        bandpass = DcrModel.load_bandpass(band_name=self.photoParams.bandpass, wavelength_step=None)
        regularize_psf = None
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

        dcr_shift = self.build_dcr_kernel(size=self.psf_size)
        # Assume that the PSF does not change between sub-bands.
        reg_weight = np.max(np.abs(dcr_shift))*100.
        regularize_psf = self.build_regularization(self.psf_size, n_step=self.n_step, weight=reg_weight,
                                                   frequency_regularization=True)
        # Use the entire psf provided, even if larger than than the kernel we will use to solve DCR for images
        # If the original psf is much larger than the kernel, it may be trimmed slightly by fit_psf_size above

        psf_model_gen = solve_model(self.psf_size, psf_mat, n_step=n_step, kernel_dcr=dcr_shift,
                                    regularization=regularize_psf)

        # After solving for the (potentially) large psf, store only the central portion of size kernel_size.

        psf_vals = np.sum(psf_model_gen)/n_step
        self.psf_avg = psf_vals
        psf_image = afwImage.ImageD(self.psf_size, self.psf_size)
        psf_image.getArray()[:, :] = psf_vals
        psfK = afwMath.FixedKernel(psf_image)
        self.psf = measAlg.KernelPsf(psfK)

        if self.psf_avg is None:
            self.calc_psf_model()
        for exp in self.exposures:

            if verbose:
        if verbose:

    def _combine_masks(self):
        """!Compute the bitwise OR of the input masks."""
        mask_arr = (exp.getMaskedImage().getMask().getArray() for exp in self.exposures)

        # Flags a pixel if ANY image is flagged there.
        for mask in mask_arr:
            if self.mask is None:
                self.mask = mask
            else:
                self.mask = np.bitwise_or(self.mask, mask)


def _calc_psf_kernel_subroutine(psf_img, size=None, size_out=None):
    """!Subroutine to build a covariance matrix from an image of a PSF.

    @param psf_img  Numpy array, containing an image of the PSF.
    @param size  width of the kernel in the origin image, in pixels.
    @param size_out  width of the kernel in the destination image, in pixels.
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
    """!Pre-compute the 1D sinc function values along each axis.

    @param offset  tuple of start/end pixel offsets of dft locations along single axis (either x or y)
    @params size  dimension in pixels of the given axis.
    @param n_substep  Number of points in the numerical integration.
    @param lanczos  If set, the order of lanczos interpolation to use.
    """
    # Calculate the kernel as a simple numerical integration over the width of the offset with n_substep steps
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
    """!Compute the parallactic angle given hour angle, declination, and latitude.

    @param hour_angle  Hour angle of the observation, in radians.
    @param dec  Declination of the observation, in radians.
    @param lat  Latitude of the observatory, in radians
    @return  Returns the parallactic angle of the observation, in radians.
    """
    return np.arctan2(np.sin(hour_angle), np.cos(dec)*np.tan(lat) - np.sin(dec)*np.cos(hour_angle))


def wrap_warpExposure(exposure, wcs, BBox, warpingControl=None):
    """!Warp an exposure to fit a given WCS and bounding box.

    @param exposure  An LSST exposure object. The image values will be overwritten!
    @param wcs  World Coordinate System (wcs) to warp the image to.
    @param BBox  Bounding box of the new image.
    @param warpingControl  [optional] afwMath.WarpingControl that sets the interpolation parameters.

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


def solve_model(kernel_size, img_vals, n_step=None, regularization=None,
                kernel_dcr=None, kernel_ref=None, kernel_restore=None):
    """!Wrapper to call a fitter using a given covariance matrix, image values, and any regularization.

    @param kernel_size  Size of the kernel to use for calculating the covariance matrix, in pixels.
    @param img_vals  Image data values for the pixels being used for the calculation, as a 1D vector.
    @param n_step  Number of sub-filter wavelength planes to model. Optional if wavelength_step supplied.
    @param lstsq_kernel  Pre-computed matrix for solving the linear least squares solution.
                         Built with build_lstsq_kernel.
    @param use_nonnegative  Flag, set to True to use a true non-negative least squares solution [SLOW]
    @param regularization  Regularization matrix created by build_regularization. If None, it is not used.
                             The type of regularization is set previously with build_regularization.
                             Used to build lstsq_kernel if not supplied, or if use_nonnegative is set.
    @param center_only  Flag, set to True to calculate the covariance for only the center pixel.
    @param kernel_dcr  The covariance matrix describing the effect of DCR
    @param kernel_ref  The covariance matrix for the reference image, used only if use_nonnegative is set
                       or lstsq_kernel is None.
    @param kernel_restore  The covariance matrix for the final restored image, used only if
                           use_nonnegative is set or lstsq_kernel is None.
    @return  Returns a numpy array containing the solution values.
    """
    x_size = kernel_size
    y_size = kernel_size
    if use_nonnegative:
        center_only = False
        if (kernel_restore is None) or (kernel_ref is None):
            vals_use = img_vals
            kernel_use = kernel_dcr
        else:
            vals_use = kernel_restore.dot(img_vals)
            kernel_use = kernel_ref.dot(kernel_dcr)

        if regularization is not None:
            regularize_dim = regularization.shape
            vals_use = np.append(vals_use, np.zeros(regularize_dim[0]))
            kernel_use = np.append(kernel_use, regularization, axis=0)
        model_solution = scipy.optimize.nnls(kernel_use, vals_use)
        model_vals = model_solution[0]
    else:
        if lstsq_kernel is None:
            lstsq_kernel = build_lstsq_kernel(kernel_dcr=kernel_dcr, kernel_ref=kernel_ref,
                                              kernel_restore=kernel_restore, regularization=regularization)
        model_vals = lstsq_kernel.dot(img_vals)
    if n_step is None:
        if center_only:
            yield model_vals
        else:
            yield np.reshape(model_vals, (y_size, x_size))
    else:
        if center_only:
            for model_val in model_vals:
                yield model_val
        else:
            n_pix = x_size*y_size
            for f in range(n_step):
                yield np.reshape(model_vals[f*n_pix: (f + 1)*n_pix], (y_size, x_size))

def calculate_rotation_angle(exposure):
    visitInfo = exposure.getInfo().getVisitInfo()

    az = visitInfo.getBoresightAzAlt().getLongitude()
    hour_angle = visitInfo.getBoresightHourAngle()
    if np.isfinite(hour_angle.asRadians()):
        dec = visitInfo.getBoresightRaDec().getDec()
        lat = visitInfo.getObservatory().getLatitude()
        p_angle = parallactic_angle(hour_angle.asRadians(), dec.asRadians(), lat.asRadians())
    else:
        p_angle = az.asRadians()
    cd = exposure.getInfo().getWcs().getCDMatrix()
    cd_rot = (np.arctan2(-cd[0, 1], cd[0, 0]) + np.arctan2(cd[1, 0], cd[1, 1]))/2.
    rotation_angle = Angle(cd_rot + p_angle)
    return rotation_angle
