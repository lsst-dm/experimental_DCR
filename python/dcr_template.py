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

__all__ = ["DcrModel", "DcrCorrection"]

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
                                      repository=None, output_repository=None, kernel_size=None,
                                      instrument='lsstSim', warp=False, verbose=True, debug_solver=False,
                                      use_nonnegative=False):
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

        self.model_sigma = None
        if kernel_size is not None:
            if kernel_size != self.kernel_size:
                print("Recalculating the PSF model!")
                self.kernel_size = 2*(kernel_size//2) + 1  # kernel must have odd dimensions.
                self.calc_psf_model()  # Also have to recalculate the psf model
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
            if instrument == 'decam':
                hour_angle = visitInfo.getBoresightHourAngle()
                dec = visitInfo.getBoresightRaDec().getDec()
                lat = visitInfo.getObservatory().getLatitude()
                p_angle = parallactic_angle(hour_angle.asRadians(), dec.asRadians(), lat.asRadians())
                cd = calexp.getInfo().getWcs().getCDMatrix()
                cd_rot = (np.arctan2(-cd[0, 1], cd[0, 0]) + np.arctan2(cd[1, 0], cd[1, 1]))/2.
                az = Angle(cd_rot + p_angle)
            else:
                az = visitInfo.getBoresightAzAlt().getLongitude()
            kernel_dcr = self.build_dcr_kernel(exposure=calexp)
            kernel_model = self.build_psf_kernel(exposure=calexp, use_full=False, expand_intermediate=True)
            kernel_exp = self.build_psf_kernel(exposure=calexp, use_full=True, expand_intermediate=True)
            lstsq_kernel = self.build_lstsq_kernel(kernel_dcr, kernel_ref=kernel_model,
                                                   kernel_restore=kernel_exp, invert=True)

            template = np.zeros((self.y_size, self.x_size))
            weights = np.zeros((self.y_size, self.x_size))
            pix_radius = self.kernel_size//2
            if verbose:
                print("Working on column")
            for j in range(self.y_size):
                if verbose:
                    if j % 100 == 0:
                        print("%i" % j, end="")
                    elif j % 10 == 0:
                        print(".", end="")
                for i in range(self.x_size):
                    if self._edge_test(j, i):
                        continue
                    model_vals = self._extract_model_vals(j, i, radius=pix_radius, model_arr=self.model,
                                                          inverse_weights=model_inverse_weights)
                    template_vals = self.solve_model(self.kernel_size, model_vals,
                                                     lstsq_kernel=lstsq_kernel, kernel_dcr=kernel_dcr,
                                                     center_only=debug_solver,
                                                     kernel_ref=kernel_model, kernel_restore=kernel_exp,
                                                     use_nonnegative=use_nonnegative)
                    self._insert_template_vals(j, i, template_vals, template=template, weights=weights,
                                               radius=pix_radius, kernel=self.psf_avg,
                                               center_only=debug_solver)
            if verbose:
                print("\nFinished building template.")
            # Weights may be different for each exposure
            template[weights > 0] /= weights[weights > 0]
            template[weights == 0] = 0.0
            if add_noise:
                variance_level = np.median(calexp.getMaskedImage().getVariance().getArray())
                rand_gen = np.random
                template += rand_gen.normal(scale=np.sqrt(variance_level), size=template.shape)

            dataId_out = self._build_dataId(obsid, self.photoParams.bandpass, instrument=instrument)[0]
            exposure = self.create_exposure(template, variance=np.abs(template), snap=0,
                                            elevation=el, azimuth=az, obsid=dataId_out['visit'])
            if warp:
                wrap_warpExposure(exposure, wcs_exp, bbox_exp)
            if output_repository is not None:
                butler_out.put(exposure, "calexp", dataId=dataId_out)
            yield exposure

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

    @staticmethod
    def _extract_model_vals(j, i, model_arr, inverse_weights, radius=None):
        """!Return all pixels within a box surrounding a given point as a 1D vector for each dcr plane model.

        @param j  Vertical pixel index
        @param i  Horizontal pixel index
        @param model_arr  dcrModel read in with DcrModel._load_model or DcrCorrection, a list of 2D arrays
        @param inverse_weights   A 2D array containing the inverse of the nonzero elements of the weights
                                 returned by DcrCorrection, with zeros everywhere else.
        @param radius  Half the width, in pixels, of the box surrounding (j, i) to be extracted.
        @return Returns the weighted model values within the range of pixels, formatted for _apply_dcr_kernel
        """
        n_step = len(model_arr)
        n_pix = (2*radius + 1)**2
        model_return = np.zeros(n_step*n_pix, dtype=np.float64)
        slice_inds = np.s_[j - radius: j + radius + 1, i - radius: i + radius + 1]
        inv_weights_use = inverse_weights[slice_inds]
        for f, model in enumerate(model_arr):
            model_use = model[slice_inds]
            model_return[f*n_pix: (f + 1)*n_pix] = np.ravel(model_use*inv_weights_use)
        return model_return

    @staticmethod
    def _insert_template_vals(j, i, vals, template=None, weights=None, radius=None, kernel=None,
                              center_only=False):
        """!Update a template image in place.

        @param j  Vertical pixel index
        @param i  Horizontal pixel index
        @param vals  DCR-corrected template values, from solve_model
        @param template  The existing template image, which will be modified in place.
        @param weights  An array of weights, to be modified in place.
        @param radius  Half the width, in pixels, of the box surrounding (j, i) to be updated.
        @param kernel  [optional] Weight inserted template values with this kernel.
        @param center_only  Set to True to update only the value of the center pixel.
        """
        if kernel is None:
            kernel_use = 1.0
        else:
            kernel_use = kernel
        if center_only:
            template[j, i] += vals
            weights[j, i] += 1.
        else:
            slice_inds = np.s_[j - radius: j + radius + 1, i - radius: i + radius + 1]
            template[slice_inds] += vals*kernel_use
            weights[slice_inds] += kernel_use

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
                      azimuth=Angle(np.radians(0.0)), **kwargs):
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
                                            zenith_angle=zenith_angle.asDegrees(), **kwargs)
            refract_end = diff_refraction(wavelength=wl_end, wavelength_ref=wavelength_midpoint,
                                          zenith_angle=zenith_angle.asDegrees(), **kwargs)
            refract_start *= 3600.0 / pixel_scale  # Refraction initially in degrees, convert to pixels.
            refract_end *= 3600.0 / pixel_scale
            dx = delta(start=refract_start*np.sin(azimuth.asRadians()),
                       end=refract_end*np.sin(azimuth.asRadians()))
            dy = delta(start=refract_start*np.cos(azimuth.asRadians()),
                       end=refract_end*np.cos(azimuth.asRadians()))
            yield dcr(dx=dx, dy=dy)

    def _edge_test(self, j, i):
        """!Check if a given pixel is near the edge of the image.

            #TODO I expect this function to go away in production code. It exists to simplify other code
                  that I don't want cluttered with these tests.
                  Also, this is where the debugging option is checked, which speeds up imaging during tests
                  by skipping ALL pixels outside of a narrow range.
        @return Returns True if the pixel is near the edge or outside the debugging region if debug_mode
                is turned on, returns False otherwise.
        """
        # Debugging parameters. Only pixels in the range [y0: y0 + dy, x0: x0 + dx] will be used.
        x0 = 150
        dx = 165
        y0 = 480
        dy = 170

        pix_radius = self.kernel_size//2

        # Deal with the edges later. Probably by padding the image with zeroes.
        if self.debug:
            if i < x0:
                return True
            elif i > x0+dx:
                return True
            elif j < y0:
                return True
            elif j > y0+dy:
                return True
            else:
                return False
        elif i < pix_radius + 1:
            return True
        elif self.x_size - i < pix_radius + 1:
            return True
        elif j < pix_radius + 1:
            return True
        elif self.y_size - j < pix_radius + 1:
            return True
        else:
            return False

    # NOTE: This function was copied from StarFast.py
    def create_exposure(self, array, variance=None, elevation=None, azimuth=None, snap=0,
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

        hour_angle = (90.0 - elevation.asDegrees())*np.cos(azimuth.asRadians())/15.0
        mjd = 59000.0 + (lsst_lat.asDegrees()/15.0 - hour_angle)/24.0
        airmass = 1.0/np.sin(elevation.asRadians())
        meta = exposure.getMetadata()
        meta.add("CHIPID", "R22_S11")
        # Required! Phosim output stores the snap ID in "OUTFILE" as the last three characters in a string.
        meta.add("OUTFILE", ("SnapId_%3.3i" % snap))

        meta.add("TAI", mjd)
        meta.add("MJD-OBS", mjd)

        meta.add("EXTTYPE", "IMAGE")
        meta.add("EXPTIME", 30.0)
        meta.add("AIRMASS", airmass)
        meta.add("ZENITH", 90. - elevation.asDegrees())
        meta.add("AZIMUTH", azimuth.asDegrees())
        # Add all additional keyword arguments to the metadata.
        for add_item in kwargs:
            meta.add(add_item, kwargs[add_item])

        visitInfo = afwImage.makeVisitInfo(
            exposureId=int(exposureId),
            exposureTime=30.0,
            darkTime=30.0,
            date=DateTime(mjd),
            ut1=mjd,
            boresightRaDec=IcrsCoord(ra, dec),
            boresightAzAlt=Coord(azimuth, elevation),
            boresightAirmass=airmass,
            boresightRotAngle=Angle(np.radians(boresightRotAngle)),
            observatory=Observatory(lsst_lon, lsst_lat, lsst_alt),)
        exposure.getInfo().setVisitInfo(visitInfo)
        return exposure

    def export_model(self, model_repository=None):
        """!Persist a DcrModel with metadata to a repository.

        @param model_repository  full path to the directory of the repository to save the dcrModel in.
        """
        if model_repository is None:
            butler = self.butler
        else:
            butler = daf_persistence.Butler(model_repository)
        wave_gen = DcrModel._wavelength_iterator(self.bandpass, use_midpoint=False)
        for f in range(self.n_step):
            wl_start, wl_end = wave_gen.next()
            exp = self.create_exposure(self.model[f], variance=self.weights,
                                       elevation=Angle(np.pi/2), azimuth=Angle(0), ksupport=self.kernel_size,
                                       subfilt=f, nstep=self.n_step, wavelow=wl_start, wavehigh=wl_end,
                                       wavestep=self.bandpass.wavelen_step)
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
        wave_step = meta.get("WAVESTEP")
        self.y_size, self.x_size = dcrModel.getDimensions()
        self.pixel_scale = self.wcs.pixelScale().asArcseconds()
        exposure_time = dcrModel.getInfo().getVisitInfo().getExposureTime()
        self.photoParams = PhotometricParameters(exptime=exposure_time, nexp=1, platescale=self.pixel_scale,
                                                 bandpass=band_name)
        self.bbox = dcrModel.getBBox()
        self.kernel_size = meta.get("KSUPPORT")
        self.bandpass = DcrModel.load_bandpass(band_name=band_name, wavelength_step=wave_step, **kwargs)

        self.psf = dcrModel.getPsf()
        self.psf_size = self.psf.computeKernelImage().getArray().shape[0]
        # Store the central part of the image of the psf for use as a kernel later.
        p0 = self.psf_size//2 - self.kernel_size//2
        p1 = p0 + self.kernel_size
        self.psf_avg = self.psf.computeKernelImage().getArray()[p0: p1, p0: p1]

    def view_model(self, index):
        """!Display a slice of the DcrModel with the proper weighting applied.

        @param index  sub-band slice of the DcrModel to extract
        @return Returns a 2D numpy array.
        """
        model = np.zeros_like(self.weights)
        weight_inds = self.weights > 0
        model[weight_inds] = self.model[index][weight_inds]/self.weights[weight_inds]
        return model

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

    def build_dcr_kernel(self, size=None, expand_intermediate=False, exposure=None):
        """!Calculate the DCR covariance matrix for a set of exposures, or a single exposure.

        @param size  Width in pixels of the region used in the origin image. Default is entire image
        @param expand_intermediate  If set, calculate the covariance matrix between the region of pixels in
                                    the origin image and a region twice as wide in the destination image.
                                    This helps avoid edge effects when computing A^T A.
        @param exposure Optional, an LSST exposure object. If not supplied, the covariance matrix for all
                        exposures in self.exposures is calculated.
        @return Returns the covariance matrix for the exposure(s).
        """
        if size is None:
            size_use = self.kernel_size
        else:
            size_use = size
        n_pix = size_use**2
        if expand_intermediate:
            kernel_size_intermediate = size_use*2
        else:
            kernel_size_intermediate = size_use
        n_pix_int = kernel_size_intermediate**2

        if exposure is None:
            exp_gen = (exp for exp in self.exposures)
            n_images = self.n_images
        else:
            exp_gen = (exposure for i in range(1))
            n_images = 1
        dcr_kernel = np.zeros((n_images*n_pix_int, self.n_step*n_pix))
        for exp_i, exp in enumerate(exp_gen):
            visitInfo = exp.getInfo().getVisitInfo()
            el = visitInfo.getBoresightAzAlt().getLatitude()
            az = visitInfo.getBoresightAzAlt().getLongitude()
            dcr_gen = DcrModel.dcr_generator(self.bandpass, pixel_scale=self.pixel_scale,
                                             elevation=el, azimuth=az)
            kernel_single = DcrModel.calc_offset_phase(dcr_gen=dcr_gen, size=size_use,
                                                       size_out=kernel_size_intermediate)
            dcr_kernel[exp_i*n_pix_int: (exp_i + 1)*n_pix_int, :] = kernel_single
        return dcr_kernel

    def build_psf_kernel(self, size=None, expand_intermediate=True, exposure=None, use_full=True):
        """!Compute the covariance matrix within the kernel for all exposures.

        @param size  Width in pixels of the region used in the origin image. Default is entire image
        @param expand_intermediate  If set, calculate the covariance matrix between the region of pixels in
                                    the origin image and a region twice as wide in the destination image.
                                    This helps avoid edge effects when computing A^T A.
        @param exposure  Optional, an LSST exposure object. If not supplied, the covariance matrix for all
                        exposures in self.exposures is calculated.
        @param use_full  Flag. Set to True to use the measured PSF, or False to use the fiducial PSF.
        @return Returns the covariance matrix for the exposure(s).
        """
        if size is None:
            size_use = self.kernel_size
        else:
            size_use = size
        n_pix = size_use**2
        if expand_intermediate:
            kernel_size_intermediate = size_use*2
        else:
            kernel_size_intermediate = size_use
        n_pix_int = kernel_size_intermediate**2

        if exposure is None:
            exp_gen = (exp for exp in self.exposures)
            n_images = self.n_images
        else:
            exp_gen = (exposure for i in range(1))
            n_images = 1
        kernel = np.zeros((n_images*n_pix_int, n_images*n_pix))
        if use_full:
                self.psf_arr = []
        for exp_i, exp in enumerate(exp_gen):
            if use_full:
                psf_img = self.calc_psf_model_single(exp)
                self.psf_arr.append(psf_img)
            else:
                psf_img = self.psf_avg  # /self.n_step
            kernel_single = _calc_psf_kernel_subroutine(psf_img, size=self.kernel_size,
                                                        size_out=kernel_size_intermediate)
            kernel[exp_i*n_pix_int: (exp_i + 1)*n_pix_int, exp_i*n_pix: (exp_i + 1)*n_pix] = kernel_single
        return kernel

    def build_weight_kernel(self, size=None, expand_intermediate=True, n_images=None):
        """!Calculate the inverse variance weighting kernel for the weighted linear least squares fit.

        @param size  Width in pixels of the region used in the origin image. Default is entire image
        @param expand_intermediate  If set, calculate the covariance matrix between the region of pixels in
                                    the origin image and a region twice as wide in the destination image.
        @param n_images Optional, number of images to create weights for. Default is to use all images.
        @return  Returns the weights as a block diagonal matrix.
        """
        if n_images is None:
            n_images = self.n_images
        if size is None:
            size_use = self.kernel_size
        else:
            size_use = size
        if expand_intermediate:
            kernel_size_int = size_use*2
        else:
            kernel_size_int = size_use
        n_pix = kernel_size_int**2
        kernel = np.zeros((n_images*n_pix, n_images*n_pix))

        sub_kernel = np.zeros((kernel_size_int, kernel_size_int))
        i0 = kernel_size_int//2 - size_use//2
        i1 = i0 + size_use
        sub_kernel[i0:i1, i0:i1] = self.psf_avg/np.max(self.psf_avg)
        diagonal_vals = np.diag(np.einsum('i,j->ij', np.ravel(sub_kernel), np.ravel(sub_kernel[::-1, ::-1])))

        for exp_i in range(n_images):
            kernel[exp_i*n_pix: (exp_i + 1)*n_pix, exp_i*n_pix: (exp_i + 1)*n_pix] = np.diag(diagonal_vals)
        return kernel

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
        psf_model_large = DcrCorrection.solve_model(psf_size_use, np.ravel(psf_img), n_step=self.n_step,
                                                    use_nonnegative=True, regularization=regularize_psf,
                                                    kernel_dcr=dcr_shift)

        # After solving for the (potentially) large psf, store only the central portion of size kernel_size.
        psf_vals = np.sum(psf_model_large, axis=0)/self.n_step
        p0 = self.psf_size//2 - self.kernel_size//2
        p1 = p0 + self.kernel_size
        return psf_vals[p0: p1, p0: p1]

    @staticmethod
    def build_lstsq_kernel(kernel_dcr, kernel_ref=None, kernel_restore=None, regularization=None, n_pix=None,
                           kernel_dcr_intermediate=None, invert=False, center_only=False, kernel_weight=None):
        """!Build the matrix M for a linear least squares solution to solve Q B y = P s for y.

        M = (B^T B)^-1 B^T (Q^T W Q)^-1 Q^T W P
        @param kernel_dcr  The covariance matrix B describing the effect of DCR
        @param kernel_ref  The covariance matrix Q for the reference image values s.
        @param kernel_restore  The covariance matrix P for the final restored image values y.
        @param regularization  Regularization matrix created with build_regularization
                               The type of regularization is set previously with build_regularization.
                               If set to None, no regularization will be used.
        @param n_pix  Total number of pixels in the origin image kernel, only used if center_only is set.
        @param kernel_dcr_intermediate Alternate DCR covariance matrix to use for calculating (B^T B)^-1
        @param invert Set to instead solve for s, so that M = (P^T W P)^-1 P^T W Q B y
        @param center_only Flag, set to True to calculate the covariance for only the center pixel.
        @param kernel_weight  [optional] Additional weighting W to include when computing the matrix inverse.
        @return  Returns the matrix M for the linear least squares solution
        """
        if kernel_dcr_intermediate is None:
            kernel_dcr_intermediate = kernel_dcr
        if regularization is not None:
            kernel_use = np.append(kernel_dcr_intermediate, regularization, axis=0)
        else:
            kernel_use = kernel_dcr_intermediate

        if not invert:
            gram_matrix = kernel_use.T.dot(kernel_use)  # compute A^T A
            kernel_inv = scipy_invert(gram_matrix)
            lstsq_dcr = kernel_inv.dot(kernel_dcr.T)

        if (kernel_ref is None) or (kernel_restore is None):
            if invert:
                lstsq_kernel = kernel_dcr
            else:
                lstsq_kernel = lstsq_dcr
        else:
            if kernel_weight is None:
                kernel_ref_T = kernel_ref.T
            else:
                kernel_ref_T = kernel_ref.T.dot(kernel_weight)
            gram_matrix_psf = kernel_ref_T.dot(kernel_ref)
            kernel_inv_psf = scipy_invert(gram_matrix_psf)

            if invert:
                lstsq_kernel = ((kernel_ref_T.dot(kernel_restore)).dot(kernel_inv_psf)).dot(kernel_dcr)
            else:
                lstsq_kernel = (lstsq_dcr.dot(kernel_inv_psf)).dot(kernel_ref_T.dot(kernel_restore))

        if center_only:
            lstsq_kernel = lstsq_kernel[n_pix//2::n_pix, :]
        return lstsq_kernel

    @staticmethod
    def solve_model(kernel_size, img_vals, n_step=None, lstsq_kernel=None, use_nonnegative=False,
                    regularization=None, center_only=False,
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
                lstsq_kernel = DcrCorrection.build_lstsq_kernel(kernel_dcr, kernel_ref=kernel_ref,
                                                                kernel_restore=kernel_restore,
                                                                regularization=regularization)
            model_vals = lstsq_kernel.dot(img_vals)
        if n_step is None:
            if center_only:
                return model_vals
            else:
                return np.reshape(model_vals, (y_size, x_size))
        else:
            if center_only:
                return np.reshape(model_vals, (n_step))
            else:
                return np.reshape(model_vals, (n_step, y_size, x_size))


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

        self.n_images = len(self.exposures)
        psf_size_arr = np.zeros(self.n_images)
        self.airmass_arr = np.zeros(self.n_images, dtype=np.float64)
        self.elevation_arr = []
        self.azimuth_arr = []
        ref_exp_i = 0
        self.bbox = self.exposures[ref_exp_i].getBBox()
        self.wcs = self.exposures[ref_exp_i].getWcs()

        for i, calexp in enumerate(self.exposures):
            visitInfo = calexp.getInfo().getVisitInfo()
            self.elevation_arr.append(visitInfo.getBoresightAzAlt().getLatitude())
            self.airmass_arr[i] = visitInfo.getBoresightAirmass()
            psf_size_arr[i] = calexp.getPsf().computeKernelImage().getArray().shape[0]
            if instrument == 'decam':
                hour_angle = visitInfo.getBoresightHourAngle()
                dec = visitInfo.getBoresightRaDec().getDec()
                lat = visitInfo.getObservatory().getLatitude()
                p_angle = parallactic_angle(hour_angle.asRadians(), dec.asRadians(), lat.asRadians())
                cd = calexp.getInfo().getWcs().getCDMatrix()
                cd_rot = (np.arctan2(-cd[0, 1], cd[0, 0]) + np.arctan2(cd[1, 0], cd[1, 1]))/2.
                self.azimuth_arr.append(Angle(cd_rot + p_angle))
            else:
                self.azimuth_arr.append(visitInfo.getBoresightAzAlt().getLongitude())
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

        # Calculate slightly worse DCR than maximum.
        elevation_min = np.min(self.elevation_arr) - Angle(np.radians(5.))
        dcr_test = DcrModel.dcr_generator(bandpass, pixel_scale=self.pixel_scale,
                                          elevation=elevation_min, azimuth=Angle(0.))
        self.dcr_max = int(np.ceil(np.max([dcr.dy for dcr in dcr_test])) + 1)
        if kernel_size is None:
            kernel_size = 2*self.dcr_max + 1
        else:
            kernel_size = 2*(kernel_size//2) + 1  # Ensure kernel is odd
        self.kernel_size = int(kernel_size)
        self.debug = bool(debug_mode)

    @staticmethod
    def build_regularization(x_size=None, y_size=None, n_step=None, weight=1.,
                             spatial_regularization=False,
                             frequency_regularization=False, frequency_second_regularization=False,
                             positive_regularization=False, test_solution=None):
        """!Regularization adapted from Nate Lust's DCR Demo iPython notebook.

        Calculate a difference matrix for regularization as if each wavelength were a pixel, then scale
        the difference matrix to the size of the number of pixels times number of wavelengths
        @param x_size  width of the kernel, in pixels.
        @param y_size  height of the kernel, in pixels.
        @param n_step  Number of sub-filter wavelength planes.
        @param weight  Scale factor for the regularization kernel.
                       Weight equal to the mean of the covariance matrix is recommended.
        @param spatial_regularization  Flag, set to True to include spatial regularization (not recommended)
        @param frequency_regularization  Flag, set to True to regularize using frequency smoothness
        @param frequency_second_regularization  Flag, set to True to regularize using smoothness of the
               derivative of the frequency.
        @param positive_regularization  Flag, set to true to use an initial test solution
               to down-weight negative solutions
        @param test_solution  Numpy array of length x_size*y_size.
               The initial test solution to use for positive_regularization.
        @return Returns a regularization matrix, or None if all regularization options are set to False.
        """
        reg_lambda = None
        reg_lambda2 = None
        reg_positive = None
        regularization_full = None

        if frequency_regularization:
            # regularization that forces the SED to be smooth
            reg_lambda = np.zeros((n_step*x_size*y_size, (n_step - 1)*x_size*y_size))
            for f in range(n_step - 1):
                for ij in range(x_size*y_size):
                    reg_lambda[f*x_size*y_size + ij, f*x_size*y_size + ij] = 2*weight
                    reg_lambda[(f + 1)*x_size*y_size + ij, f*x_size*y_size + ij] = -2*weight
            # We should include both positive and negative slopes, which could be dealt with by taking
            # reg_lambda = np.append(reg_lambda, -reg_lambda, axis=1). That works out to be the same as
            # just doubling the weight of reg_lambda, so we use 2*weight here instead of inflating the
            # size of the matrix
            if regularization_full is None:
                regularization_full = reg_lambda
            else:
                regularization_full = np.append(regularization_full, reg_lambda, axis=1)

        if frequency_second_regularization:
            # regularization that forces the derivative of the SED to be smooth
            reg_lambda2 = np.zeros((n_step*x_size*y_size, (n_step - 2)*x_size*y_size))
            for f in range(n_step - 2):
                for ij in range(x_size*y_size):
                    reg_lambda2[f*x_size*y_size + ij, f*x_size*y_size + ij] = -weight
                    reg_lambda2[(f + 1)*x_size*y_size + ij, f*x_size*y_size + ij] = 2.*weight
                    reg_lambda2[(f + 2)*x_size*y_size + ij, f*x_size*y_size + ij] = -weight
            if regularization_full is None:
                regularization_full = reg_lambda2
            else:
                regularization_full = np.append(regularization_full, reg_lambda2, axis=1)

        if positive_regularization:
            #regularization that down-weights negative elements in a trial solution.
            if test_solution is None:
                n_zero = 0
            else:
                test_solution_use = np.reshape(test_solution, (n_step*x_size*y_size))
                zero_inds = np.where(np.ravel(test_solution_use) < 0)[0]
                n_zero = len(zero_inds)
            if n_zero > 0:
                reg_positive = np.zeros((n_step*x_size*y_size, n_zero))
                for zi in range(n_zero):
                    reg_positive[zero_inds[zi], zi] = weight*test_solution_use[zero_inds[zi]]
            if regularization_full is None:
                regularization_full = reg_positive
            else:
                regularization_full = np.append(regularization_full, reg_positive, axis=1)

        return regularization_full

    @staticmethod
    def _extract_image_vals(j, i, image_arr, mask=None, radius=None):
        """!Return all pixels within a radius of a given point as a 1D vector for each exposure.

        @param j  Vertical index of the center pixel
        @param i  Horizontal index of the center pixel
        @param image_arr  List of 2D numpy arrays containing the image data.
        @param mask  [Optional] List of mask planes associated with each image. #TODO Not yet implemented!
        @param radius  Half the width, in pixels, of the box surrounding (j, i) to be updated.
        @return Returns a numpy array of size (number of exposures, (2*radius + 1)**2)
        """
        n_pix = (2*radius + 1)**2
        n_img = len(image_arr)
        sub_img_arr = np.zeros(n_pix*n_img, dtype=np.float64)
        for ii, img in enumerate(image_arr):
            sub_img = img[j - radius: j + radius + 1, i - radius: i + radius + 1]
            sub_img_arr[ii*n_pix: (ii + 1)*n_pix] = np.ravel(sub_img)
        return sub_img_arr

    @staticmethod
    def _insert_model_vals(j, i, vals, model, weights, radius=None, kernel=1., center_only=False):
        """!Insert the given values into the model and update the weights.

        @param j  Vertical index of the center pixel
        @param i  Horizontal index of the center pixel
        @param vals  Numpy array of values to be inserted, with size (n_step, 2*radius + 1, 2*radius + 1)
        @param model  The weighted model to be updated. Modified in place.
        @param weights  The weights to use for calculating a weighted average of the model. Modified in place.
        @param radius  Half the width, in pixels, of the box surrounding (j, i) to be updated.
        @param kernel  A scalar or numpy array of weights to multiply vals by for a weighted average.
        @param center_only  Flag, set to True to calculate the covariance for only the center pixel.
        """
        if center_only:
            model[:, j, i] += vals
            weights[j, i] += 1.
        else:
            model[:, j - radius: j + radius + 1, i - radius: i + radius + 1] += vals*kernel
            weights[j - radius: j + radius + 1, i - radius: i + radius + 1] += kernel

    @staticmethod
    def fit_psf_size(exposures, minimum_psf_size=5, threshold=1e-2):
        """!Fit for the size of the psf to use from a set of exposures.

        @param exposures  List of LSST exposures.
        @param minimum_psf_size  Force the fit psf size to be at least this size, in pixels
        @param threshold  Fraction of total PSF power to cut from edge of PSF.
        @return  Returns the cropped size of the psf that preserves most of the flux, as set by threshold.
        """
        psf_size_arr = np.zeros(len(exposures))
        for exp_i, exp in enumerate(exposures):
            psf_img = exp.getPsf().computeKernelImage().getArray()
            psf_dim = psf_img.shape[0]
            if psf_dim <= minimum_psf_size:
                psf_size_arr[exp_i] = psf_dim
                continue
            psf_sum = np.sum(psf_img)
            psf_sum_arr = np.zeros(psf_dim//2)
            for i in range(0, psf_dim//2 - minimum_psf_size//2):
                psf_sum_arr[i] = np.sum(psf_img[i: psf_dim - i, i: psf_dim - i])
            psf_rel_diff = np.abs(psf_sum_arr - psf_sum)/psf_sum
            psf_size_arr[exp_i] = 2*(psf_dim//2 - np.max(np.where(psf_rel_diff < threshold))) + 1
        return(int(np.min(psf_size_arr)))

    def calc_psf_model(self):
        """!Calculate the fiducial psf from a given set of exposures, accounting for DCR."""
        self.psf_size = self.fit_psf_size(self.exposures, minimum_psf_size=self.kernel_size, threshold=5e-2)
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
        psf_model_large = DcrCorrection.solve_model(self.psf_size, psf_mat, n_step=self.n_step,
                                                    kernel_dcr=dcr_shift,
                                                    use_nonnegative=True, regularization=regularize_psf)

        p0 = self.psf_size//2 - self.kernel_size//2
        p1 = p0 + self.kernel_size
        # After solving for the (potentially) large psf, store only the central portion of size kernel_size.
        self.psf_model = psf_model_large[:, p0: p1, p0: p1]
        psf_vals = np.sum(psf_model_large, axis=0)/self.n_step
        self.psf_avg = psf_vals[p0: p1, p0: p1]
        psf_image = afwImage.ImageD(self.psf_size, self.psf_size)
        psf_image.getArray()[:, :] = psf_vals
        psfK = afwMath.FixedKernel(psf_image)
        self.psf = measAlg.KernelPsf(psfK)

    def build_model(self, use_only_detected=False, verbose=True, kernel_size=None,
                    use_nonnegative=False, positive_regularization=False, frequency_regularization=True,
                    debug_solver=False, debug_regularize=None):
        """!Calculate a model of the true sky using the known DCR offset for each freq plane.

        @param use_only_detected  Flag, set to True to only calculate the DCR model for the footprint
                                    of detected sources.
        @param verbose  Flag, set to True to print progress messages.
        @param kernel_size  [optional] Override the previously set kernel_size with the given value.
        @param use_nonnegative  Flag, set to True to use a true non-negative least squares fit. Very slow!
        @param positive_regularization  Flag, set to True to use an approximate non-negative least squares fit
        @param frequency_regularization Flag, set to True to add constraints on the slope of the frequency
                                        spectrum of the solution.
        @return  No return value, but modifies self.model and self.weights in place.
        """
        self.model_sigma = None
        if kernel_size is not None:
            if kernel_size != self.kernel_size:
                print("Recalculating the PSF model!")
                self.kernel_size = 2*(kernel_size//2) + 1  # kernel must have odd dimensions.
                self.calc_psf_model()  # Also have to recalculate the psf model
        if self.psf_avg is None:
            self.calc_psf_model()
        if use_nonnegative:
            expand_intermediate = False
        else:
            expand_intermediate = True
        kernel_restore = self.build_psf_kernel(use_full=False, expand_intermediate=expand_intermediate)
        kernel_full = self.build_psf_kernel(use_full=True, expand_intermediate=expand_intermediate)
        kernel_dcr = self.build_dcr_kernel(expand_intermediate=False)
        kernel_weight = self.build_weight_kernel(expand_intermediate=expand_intermediate)

        self.kernel_restore = kernel_restore
        self.kernel_full = kernel_full
        self.kernel_dcr = kernel_dcr

        regularize_scale = np.max(kernel_dcr)
        if debug_regularize is not None:
            regularize_scale *= debug_regularize
            kernel_dcr_intermediate = None
        if expand_intermediate:
            kernel_size_intermediate = self.kernel_size*2
            kernel_dcr_intermediate = self.build_dcr_kernel(expand_intermediate=True)
        else:
            kernel_size_intermediate = self.kernel_size
            kernel_dcr_intermediate = None
        # if positive_regularization:
        #     test_values = np.ones(self.n_images*self.kernel_size**2)
        #     test_solution = self.solve_model(self.kernel_size, self.n_step, kernel_base, test_values,
        #                                      use_nonnegative=use_nonnegative, kernel_weight=kernel_weight)
        # else:
        #     test_solution = None
        self.regularize = self.build_regularization(kernel_size_intermediate, size_out=self.kernel_size,
                                                    n_step=self.n_step, weight=regularize_scale,
                                                    frequency_regularization=frequency_regularization,
                                                    positive_regularization=positive_regularization,
                                                    test_solution=None)
        n_pix = self.kernel_size**2
        lstsq_kernel = self.build_lstsq_kernel(kernel_dcr, kernel_ref=kernel_full,
                                               kernel_restore=kernel_restore, kernel_weight=kernel_weight,
                                               regularization=self.regularize, n_pix=n_pix,
                                               kernel_dcr_intermediate=kernel_dcr_intermediate,
                                               center_only=debug_solver)
        self.lstsq = lstsq_kernel
        model = np.zeros((self.n_step, self.y_size, self.x_size))
        weights = np.zeros((self.y_size, self.x_size))
        pix_radius = self.kernel_size//2

        variance = np.zeros((self.y_size, self.x_size))
        for exp in self.exposures:
            variance += exp.getMaskedImage().getVariance().getArray()**2.
        self.variance = np.sqrt(variance) / self.n_images
        detected_bit = exp.getMaskedImage().getMask().getPlaneBitMask("DETECTED")
        image_arr = [exp.getMaskedImage().getImage().getArray() for exp in self.exposures]
        mask_arr = [exp.getMaskedImage().getMask().getArray() for exp in self.exposures]
        if verbose:
            print("Working on column", end="")
        for j in range(self.y_size):
            if verbose:
                if j % 100 == 0:
                    print("\n %i" % j, end="")
                elif j % 10 == 0:
                    print("|", end="")
                else:
                    print(".", end="")
            for i in range(self.x_size):
                if self._edge_test(j, i):
                    continue
                # This option saves time by only performing the fit if the center pixel is masked as detected
                # Note that by gridding the results with the psf and maintaining a separate 'weights' array
                # this allows us to construct a variance-weighted average of the model.
                if use_only_detected:
                    if self.mask[j, i] & detected_bit == 0:
                        continue
                img_vals = self._extract_image_vals(j, i, image_arr, mask=mask_arr, radius=pix_radius)

                model_vals = self.solve_model(self.kernel_size, img_vals, n_step=self.n_step,
                                              lstsq_kernel=lstsq_kernel, use_nonnegative=use_nonnegative,
                                              regularization=self.regularize, center_only=debug_solver,
                                              kernel_ref=kernel_full, kernel_restore=kernel_restore,
                                              kernel_dcr=kernel_dcr)
                self._insert_model_vals(j, i, model_vals, model, weights, center_only=debug_solver,
                                        radius=pix_radius, kernel=self.psf_avg)
        if verbose:
            print("\nFinished building model.")
        self.weights = weights
        self.model = [model[f, :, :] for f in range(self.n_step)]

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


def _kernel_1d(offset, size, n_substep=None, lanczos=None):
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
