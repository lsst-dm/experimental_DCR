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
import scipy.optimize.nnls

from .calc_refractive_index import diff_refraction
import lsst.daf.persistence as daf_persistence
import lsst.afw.image as afwImage
import lsst.afw.math as afwMath
import lsst.meas.algorithms as measAlg
import lsst.pex.exceptions
import lsst.pex.policy as pexPolicy
from lsst.sims.photUtils import Bandpass, PhotometricParameters
from lsst.utils import getPackageDir

__all__ = ["DcrModel", "DcrCorrection"]

lsst_lat = -30.244639
lsst_lon = -70.749417


class DcrModel:
    """Lightweight object that contains only the minimum needed to generate DCR-matched template exposures."""

    def __init__(self, model_repository=None, band_name='g', debug_mode=False, **kwargs):
        """Only run when restoring a model or for testing; otherwise superceded by DcrCorrection __init__."""
        """
        @param model_repository: path to the repository where the previously-generated DCR model is stored.
        @param band_name: name of the bandpass-defining filter of the data. Expected values are u,g,r,i,z,y.
        @param debug_mode: if set, only use a subset of the data for speed (used in _edge_test)
        """
        self.debug = debug_mode
        self.butler = None
        self.load_model(model_repository=model_repository, band_name=band_name, **kwargs)

    def generate_templates_from_model(self, obsid_range=None, exposures=None, add_noise=False, use_full=True,
                                      repository=None, output_repository=None, kernel_size=None, **kwargs):
        """Use the previously generated model and construct a dcr template image."""
        """
        @param obsid_range: single, range, or list of observation IDs in repository to create matched
                            templates for. Ignored if exposures are supplied directly.
        @param exposures: optional, list of exposure objects that will have matched templates created.
        @param add_noise: If set to true, add Poisson noise to the template based on the variance.
        @param use_full: debugging keyword. Set to use the full psf kernel instead of the average.
        @param repository: path to the repository where the exposure data to be matched are stored.
                           Ignored if exposures are supplied directly.
        @param output_repository: path to repository directory where templates will be saved.
        """
        if output_repository is not None:
            butler_out = daf_persistence.Butler(output_repository)
        if exposures is None:
            if repository is not None:
                butler = daf_persistence.Butler(repository)
                if self.butler is None:
                    self.butler = butler
            else:
                butler = self.butler
            exposures = []
            if obsid_range is not None:
                dataId_gen = self._build_dataId(obsid_range, self.photoParams.bandpass)
                for dataId in dataId_gen:
                    calexp = butler.get("calexp", dataId=dataId)
                    exposures.append(calexp)
            else:
                raise ValueError("One of obsid_range or exposures must be set.")
        else:
            if obsid_range is None:
                obsid_range = [self._fetch_metadata(calexp, "OBSID", default_value=0) for calexp in exposures]
        dataId_out_arr = self._build_dataId(obsid_range, self.photoParams.bandpass)

        if kernel_size is not None:
            self.kernel_size = kernel_size
        for exp_i, calexp in enumerate(exposures):
            el = 90 - calexp.getMetadata().get("ZENITH")
            az = calexp.getMetadata().get("AZIMUTH")
            pix = self.photoParams.platescale
            dcr_gen = self._dcr_generator(self.bandpass, pixel_scale=pix, elevation=el, azimuth=az)

            make_kernel_kwargs = dict(exposure=calexp, dcr_gen=dcr_gen, return_matrix=True,
                                      x_size=self.kernel_size, y_size=self.kernel_size)
            if self.use_psf:
                if use_full:
                    dcr_kernel = self._calc_psf_kernel_full(psf_img=self.psf_avg, **make_kernel_kwargs)
                else:
                    dcr_kernel = self._calc_psf_kernel(psf_img=self.psf_avg, **make_kernel_kwargs)
            else:
                dcr_kernel = self._calc_offset_phase(**make_kernel_kwargs)

            template = np.zeros((self.y_size, self.x_size))
            weights = np.zeros((self.y_size, self.x_size))
            pix_radius = self.kernel_size//2
            for j in range(self.y_size):
                for i in range(self.x_size):
                    if self._edge_test(j, i):
                        continue
                    model_vals = self._extract_model_vals(j, i, radius=pix_radius,
                                                          model=self.model, weights=self.weights)
                    template_vals = self._apply_dcr_kernel(dcr_kernel, model_vals,
                                                           x_size=self.kernel_size,
                                                           y_size=self.kernel_size)
                    self._insert_template_vals(j, i, template_vals, template=template, weights=weights,
                                               radius=pix_radius, kernel=self.psf_avg)
            template[weights > 0] /= weights[weights > 0]
            template[weights == 0] = 0.0
            if add_noise:
                variance_level = np.median(calexp.getMaskedImage().getVariance().getArray())
                rand_gen = np.random
                template += rand_gen.normal(scale=np.sqrt(variance_level), size=template.shape)

            dataId_out = dataId_out_arr[exp_i]
            exposure = self._create_exposure(template, variance=np.abs(template), snap=0,
                                             elevation=el, azimuth=az, obsid=dataId_out['visit'])
            if output_repository is not None:
                butler_out.put(exposure, "calexp", dataId=dataId_out)
            yield exposure

    @staticmethod
    def _fetch_metadata(exposure, property_name, default_value=None):
        try:
            value = exposure.getMetadata().get(property_name)
        except lsst.pex.exceptions.wrappers.NotFoundError as e:
            if default_value is not None:
                print("WARNING: " + str(e) + ". Using default value: %s" % repr(default_value))
                return default_value
            else:
                raise e
        return value

    @staticmethod
    def _build_dataId(obsid_range, band):
        if hasattr(obsid_range, '__iter__'):
            if len(obsid_range) > 2:
                if obsid_range[2] < obsid_range[0]:
                    dataId = [{'visit': obsid, 'raft': '2,2', 'sensor': '1,1', 'filter': band}
                              for obsid in np.arange(obsid_range[0], obsid_range[1], obsid_range[2])]
                else:
                    dataId = [{'visit': obsid, 'raft': '2,2', 'sensor': '1,1', 'filter': band}
                              for obsid in obsid_range]
            else:
                dataId = [{'visit': obsid, 'raft': '2,2', 'sensor': '1,1', 'filter': band}
                          for obsid in np.arange(obsid_range[0], obsid_range[1])]
        else:
            dataId = [{'visit': obsid, 'raft': '2,2', 'sensor': '1,1', 'filter': band}
                      for obsid in [obsid_range]]
        return dataId

    @staticmethod
    def _build_model_dataId(band, subfilter=None):
        if subfilter is None:
            dataId = {'filter': band, 'tract': 0, 'patch': '0'}
        else:
            dataId = {'filter': band, 'tract': 0, 'patch': '0', 'subfilter': subfilter}
        return(dataId)

    @staticmethod
    def _apply_dcr_kernel(dcr_kernel, model_vals, x_size=None, y_size=None):
        template_vals = np.dot(dcr_kernel.T, model_vals)
        return(np.reshape(template_vals, (y_size, x_size)))

    @staticmethod
    def _extract_model_vals(j, i, radius=None, model=None, weights=None):
        """Return all pixles within a radius of a given point as a 1D vector for each dcr plane model."""
        model_arr = []
        if weights is None:
            weights = np.ones_like(model)
        for f in range(model.shape[0]):
            model_use = model[f, j - radius: j + radius + 1, i - radius: i + radius + 1].copy()
            weights_use = weights[f, j - radius: j + radius + 1, i - radius: i + radius + 1]
            model_use[weights_use > 0] /= weights_use[weights_use > 0]
            model_use[weights_use <= 0] = 0.0
            model_arr.append(np.ravel(model_use))
        return np.hstack(model_arr)

    @staticmethod
    def _insert_template_vals(j, i, vals, template=None, weights=None, radius=None, kernel=None):
        if kernel is None:
            kernel = 1.0
        template[j - radius: j + radius + 1, i - radius: i + radius + 1] += vals*kernel
        weights[j - radius: j + radius + 1, i - radius: i + radius + 1] += kernel

    # NOTE: This function was copied from StarFast.py
    @staticmethod
    def _load_bandpass(band_name='g', wavelength_step=None, use_mirror=True, use_lens=True, use_atmos=True,
                       use_filter=True, use_detector=True, **kwargs):
        """Load in Bandpass object from sims_photUtils."""
        """
        @param band_name: Common name of the filter used. For LSST, use u, g, r, i, z, or y
        @param wavelength_step: Wavelength resolution, also the wavelength range of each sub-band plane
        @param use_mirror: Flag, include mirror in filter throughput calculation?
        @param use_lens: Flag, use LSST lens in filter throughput calculation?
        @param use_atmos: Flag, use standard atmosphere transmission in filter throughput calculation?
        @param use_filter: Flag, use LSST filters in filter throughput calculation?
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
        """Define iterator to ensure that loops over wavelength are consistent."""
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
    def _dcr_generator(bandpass, pixel_scale=None, elevation=50.0, azimuth=0.0, use_midpoint=False, **kwargs):
        """Call the functions that compute Differential Chromatic Refraction (relative to mid-band)."""
        """
        @param bandpass: bandpass object created with load_bandpass
        @param pixel_scale: plate scale in arcsec/pixel
        @param elevation: elevation angle of the center of the image, in decimal degrees.
        @param azimuth: azimuth angle of the observation, in decimal degrees.
        """
        zenith_angle = 90.0 - elevation
        wavelength_midpoint = bandpass.calc_eff_wavelen()
        delta = namedtuple("delta", ["start", "end"])
        dcr = namedtuple("dcr", ["dx", "dy"])
        for wl_start, wl_end in DcrModel._wavelength_iterator(bandpass, use_midpoint=False):
            # Note that refract_amp can be negative, since it's relative to the midpoint of the full band
            if use_midpoint:
                wl_mid = bandpass.calc_eff_wavelen(wavelength_min=wl_start, wavelength_max=wl_end)
                refract_mid = diff_refraction(wavelength=wl_mid, wavelength_ref=wavelength_midpoint,
                                              zenith_angle=zenith_angle, **kwargs)
                refract_mid *= 3600.0 / pixel_scale
                dx_mid = refract_mid * np.sin(np.radians(azimuth))
                dy_mid = refract_mid * np.cos(np.radians(azimuth))
                yield dcr(dx=dx_mid, dy=dy_mid)
            else:
                refract_start = diff_refraction(wavelength=wl_start, wavelength_ref=wavelength_midpoint,
                                                zenith_angle=zenith_angle, **kwargs)
                refract_end = diff_refraction(wavelength=wl_end, wavelength_ref=wavelength_midpoint,
                                              zenith_angle=zenith_angle, **kwargs)
                refract_start *= 3600.0 / pixel_scale  # Refraction initially in degrees, convert to pixels.
                refract_end *= 3600.0 / pixel_scale
                dx = delta(start=refract_start*np.sin(np.radians(azimuth)),
                           end=refract_end*np.sin(np.radians(azimuth)))
                dy = delta(start=refract_start*np.cos(np.radians(azimuth)),
                           end=refract_end*np.cos(np.radians(azimuth)))
                yield dcr(dx=dx, dy=dy)

    @staticmethod
    def _calc_offset_phase(exposure=None, dcr_gen=None, x_size=None, y_size=None,
                           return_matrix=False, **kwargs):
        """Return the 2D FFT of an offset generated by _dcr_generator in the form (dx, dy)."""
        phase_arr = []
        if y_size is None:
            y_size = exposure.getHeight()
        if x_size is None:
            x_size = exposure.getWidth()
        for dx, dy in dcr_gen:
            kernel_x = _kernel_1d(dx, x_size)
            kernel_y = _kernel_1d(dy, y_size)
            kernel = np.einsum('i,j->ij', kernel_y, kernel_x)
            if return_matrix:
                shift_mat = np.zeros((x_size*y_size, x_size*y_size))
                for j in range(y_size):
                    for i in range(x_size):
                        ij = i + j*x_size
                        shift_mat[ij, :] = np.ravel(scipy_shift(kernel, (j - y_size//2, i - x_size//2),
                                                    mode='constant', cval=0.0))
                phase_arr.append(shift_mat)
            else:
                phase_arr.append(np.ravel(kernel))
        phase_arr = np.vstack(phase_arr)
        return phase_arr

    @staticmethod
    def _calc_psf_kernel(exposure=None, dcr_gen=None, x_size=None, y_size=None,
                         return_matrix=False, psf_img=None, **kwargs):

        if y_size is None:
            y_size = exposure.getHeight()
        if x_size is None:
            x_size = exposure.getWidth()
        psf_kernel_arr = []
        for dcr in dcr_gen:
            psf_y_size, psf_x_size = psf_img.shape
            psf = np.zeros((y_size, x_size), dtype=psf_img.dtype)
            if return_matrix:
                psf_kernel_arr.append(_calc_psf_kernel_subroutine(psf_img, dcr))
            else:
                psf_kernel_arr.append(np.ravel(psf))

        psf_kernel_arr = np.vstack(psf_kernel_arr)
        return psf_kernel_arr

    @staticmethod
    def _calc_psf_kernel_full(exposure=None, dcr_gen=None, x_size=None, y_size=None,
                              return_matrix=False, **kwargs):
        if y_size is None:
            y_size = exposure.getHeight()
        if x_size is None:
            x_size = exposure.getWidth()
        psf_kernel_arr = []
        psf_img = exposure.getPsf().computeKernelImage().getArray()
        for dcr in dcr_gen:
            kernel_single = _calc_psf_kernel_subroutine(psf_img, dcr, x_size=x_size, y_size=y_size)
            if return_matrix:
                psf_kernel_arr.append(kernel_single)
            else:
                kernel_single = kernel_single[y_size*x_size//2, :]
                psf_kernel_arr.append(kernel_single)

        psf_kernel_arr = np.vstack(psf_kernel_arr)
        return psf_kernel_arr

    def _edge_test(self, j, i):
        x0 = 150
        dx = 65
        y0 = 480
        dy = 70
        pix_radius = self.kernel_size//2

        # Deal with the edges later. Probably by padding the image with zeroes.
        if i < pix_radius + 1:
            edge = True
        elif self.x_size - i < pix_radius + 1:
            edge = True
        elif j < pix_radius + 1:
            edge = True
        elif self.y_size - j < pix_radius + 1:
            edge = True
        elif self.debug:
            if i < x0:
                edge = True
            elif i > x0+dx:
                edge = True
            elif j < y0:
                edge = True
            elif j > y0+dy:
                edge = True
            else:
                edge = False
        else:
            edge = False
        return edge

    # NOTE: This function was copied from StarFast.py
    def _create_exposure(self, array, variance=None, elevation=None, azimuth=None, snap=0, **kwargs):
        """Convert a numpy array to an LSST exposure, and units of electron counts."""
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
        calib = afwImage.Calib()
        calib.setExptime(self.photoParams.exptime)
        exposure.setCalib(calib)
        exposure.setPsf(self.psf)
        exposure.getMaskedImage().getImage().getArray()[:, :] = array
        if variance is None:
            variance = np.abs(array)
        exposure.getMaskedImage().getVariance().getArray()[:, :] = variance

        exposure.getMaskedImage().getMask().getArray()[:, :] = self.mask

        hour_angle = (90.0 - elevation)*np.cos(np.radians(azimuth))/15.0
        mjd = 59000.0 + (lsst_lat/15.0 - hour_angle)/24.0
        meta = exposure.getMetadata()
        meta.add("CHIPID", "R22_S11")
        # Required! Phosim output stores the snap ID in "OUTFILE" as the last three characters in a string.
        meta.add("OUTFILE", ("SnapId_%3.3i" % snap))

        meta.add("TAI", mjd)
        meta.add("MJD-OBS", mjd)

        meta.add("EXTTYPE", "IMAGE")
        meta.add("EXPTIME", 30.0)
        meta.add("AIRMASS", 1.0/np.sin(np.radians(elevation)))
        meta.add("ZENITH", 90 - elevation)
        meta.add("AZIMUTH", azimuth)
        for add_item in kwargs:
            meta.add(add_item, kwargs[add_item])
        return exposure

    def export_model(self, model_repository=None):
        """Persist a DcrModel with metadata to a repository."""
        if model_repository is None:
            butler = self.butler
        else:
            butler = daf_persistence.Butler(model_repository)
        wave_gen = DcrModel._wavelength_iterator(self.bandpass, use_midpoint=False)
        for f in range(self.n_step):
            wl_start, wl_end = wave_gen.next()
            exp = self._create_exposure(self.model[f, :, :], variance=self.weights[f, :, :],
                                        elevation=90., azimuth=0., ksupport=self.kernel_size,
                                        subfilt=f, nstep=self.n_step, wavelow=wl_start, wavehigh=wl_end,
                                        wavestep=self.bandpass.wavelen_step, psf_flag=self.use_psf)
            butler.put(exp, "dcrModel", dataId=self._build_model_dataId(self.photoParams.bandpass, f))

    def load_model(self, model_repository=None, band_name='g', **kwargs):
        """Depersist a DcrModel from a repository and set up the metadata."""
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

        self.model = np.array(model_arr)
        self.weights = np.array(weights_arr)
        self.mask = dcrModel.getMaskedImage().getMask().getArray()

        # This only uses the mask of the last image. For real data all masks should be used.
        meta = dcrModel.getMetadata()
        self.wcs = dcrModel.getWcs()
        self.n_step = len(model_arr)
        wave_step = meta.get("WAVESTEP")
        self.use_psf = meta.get("PSF_FLAG")
        self.y_size, self.x_size = dcrModel.getDimensions()
        self.pixel_scale = self.wcs.pixelScale().asArcseconds()
        exposure_time = dcrModel.getInfo().getCalib().getExptime()
        self.photoParams = PhotometricParameters(exptime=exposure_time, nexp=1, platescale=self.pixel_scale,
                                                 bandpass=band_name)
        self.bbox = dcrModel.getBBox()
        self.kernel_size = meta.get("KSUPPORT")
        self.bandpass = DcrModel._load_bandpass(band_name=band_name, wavelength_step=wave_step, **kwargs)

        self.psf = dcrModel.getPsf()
        self.psf_size = self.psf.computeKernelImage().getArray().shape[0]
        p0 = self.psf_size//2 - self.kernel_size//2
        p1 = p0 + self.kernel_size
        self.psf_avg = self.psf.computeKernelImage().getArray()[p0: p1, p0: p1]

    def view_model(self, index):
        """Display a slice of the DcrModel with the proper weighting applied."""
        model = self.model[index, :, :].copy()
        weights = self.weights[index, :, :]
        model[weights > 0] /= weights[weights > 0]
        model[weights <= 0] = 0.0
        return model


class DcrCorrection(DcrModel):
    """Class that loads LSST calibrated exposures and produces airmass-matched template images."""

    def __init__(self, repository=".", obsid_range=None, band_name='g', wavelength_step=10,
                 n_step=None, use_psf=True, debug_mode=False,
                 kernel_size=None, exposures=None, **kwargs):
        """Load images from the repository and set up parameters."""
        """
        @param repository: path to repository with the data. String, defaults to working directory
        @param obsid_range: obsid or range of obsids to process.
        """
        if exposures is None:
            self.butler = daf_persistence.Butler(repository)
            dataId_gen = self._build_dataId(obsid_range, band_name)
            self.exposures = []
            for dataId in dataId_gen:
                calexp = self.butler.get("calexp", dataId=dataId)
                self.exposures.append(calexp)
        else:
            self.exposures = exposures

        self.n_images = len(self.exposures)
        self.elevation_arr = np.zeros(self.n_images, dtype=np.float64)
        self.azimuth_arr = np.zeros(self.n_images, dtype=np.float64)
        self.airmass_arr = np.zeros(self.n_images, dtype=np.float64)
        for i, calexp in enumerate(self.exposures):
            self.elevation_arr[i] = 90 - calexp.getMetadata().get("ZENITH")
            self.azimuth_arr[i] = calexp.getMetadata().get("AZIMUTH")
            self.airmass_arr[i] = calexp.getMetadata().get("AIRMASS")

        self.y_size, self.x_size = self.exposures[0].getDimensions()
        self.pixel_scale = calexp.getWcs().pixelScale().asArcseconds()
        exposure_time = calexp.getInfo().getCalib().getExptime()
        self.bbox = calexp.getBBox()
        self.wcs = calexp.getWcs()
        self.psf_size = calexp.getPsf().computeKernelImage().getArray().shape[0]
        self.mask = None
        self._combine_masks()

        bandpass = DcrModel._load_bandpass(band_name=band_name, wavelength_step=wavelength_step, **kwargs)
        if n_step is not None:
            wavelength_step = (bandpass.wavelen_max - bandpass.wavelen_min) / n_step
            bandpass = DcrModel._load_bandpass(band_name=band_name, wavelength_step=wavelength_step, **kwargs)
        else:
            n_step = int(np.ceil((bandpass.wavelen_max - bandpass.wavelen_min) / bandpass.wavelen_step))
        if n_step >= self.n_images:
            print("Warning! Under-constrained system. Reducing number of frequency planes.")
            wavelength_step *= n_step / self.n_images
            bandpass = DcrModel._load_bandpass(band_name=band_name, wavelength_step=wavelength_step, **kwargs)
            n_step = int(np.ceil((bandpass.wavelen_max - bandpass.wavelen_min) / bandpass.wavelen_step))
        self.n_step = n_step
        self.bandpass = bandpass
        self.photoParams = PhotometricParameters(exptime=exposure_time, nexp=1, platescale=self.pixel_scale,
                                                 bandpass=band_name)

        elevation_min = np.min(self.elevation_arr) - 5.  # Calculate slightly worse DCR than maximum.
        dcr_test = DcrModel._dcr_generator(bandpass, pixel_scale=self.pixel_scale,
                                           elevation=elevation_min, azimuth=0.)
        self.dcr_max = int(np.ceil(np.max(dcr_test.next())) + 1)
        if kernel_size is None:
            kernel_size = 2*self.dcr_max + 1
        self.kernel_size = int(kernel_size)
        self.use_psf = bool(use_psf)
        self.debug = bool(debug_mode)
        self.regularize = self._build_regularization(x_size=self.kernel_size, y_size=self.kernel_size,
                                                     n_step=self.n_step, frequency_regularization=True)
        self._calc_psf_model()

    @staticmethod
    def _build_regularization(x_size=None, y_size=None, n_step=None, spatial_regularization=False,
                              frequency_regularization=False, frequency_second_regularization=False):
        """Regularization adapted from Nate Lust's DCR Demo iPython notebook."""
        """
        Calculate a difference matrix for regularization as if each wavelength were a pixel, then scale
        the difference matrix to the size of the number of pixels times number of wavelengths
        """
        reg_pix = None
        reg_lambda = None
        reg_lambda2 = None

        if spatial_regularization:
            reg_pix_x = np.zeros((n_step*x_size*y_size,
                                  n_step*x_size*y_size - x_size))
            for ij in range(n_step*x_size*y_size - x_size):
                reg_pix_x[ij, ij] = 1
                reg_pix_x[ij + x_size, ij] = -1
            reg_pix_x = np.append(reg_pix_x, -reg_pix_x, axis=1)

            reg_pix_y = np.zeros((n_step*x_size*y_size,
                                  n_step*x_size*y_size - 1))
            for ij in range(n_step*x_size*y_size - 1):
                reg_pix_y[ij, ij] = 1
                reg_pix_y[ij + 1, ij] = -1
            reg_pix_y = np.append(reg_pix_y, -reg_pix_y, axis=1)
            reg_pix = np.append(reg_pix_x, reg_pix_y, axis=1)

        if frequency_regularization:
            # regularization that forces the SED to be smooth
            reg_lambda = np.zeros((n_step*x_size*y_size, (n_step - 1)*x_size*y_size))
            for f in range(n_step - 1):
                for ij in range(x_size*y_size):
                    reg_lambda[f*x_size*y_size + ij, f*x_size*y_size + ij] = 1
                    reg_lambda[(f + 1)*x_size*y_size + ij, f*x_size*y_size + ij] = -1
            reg_lambda = np.append(reg_lambda, -reg_lambda, axis=1)

        if frequency_second_regularization:
            # regularization that forces the derivative of the SED to be smooth
            reg_lambda2 = np.zeros((n_step*x_size*y_size, (n_step - 2)*x_size*y_size))
            for f in range(n_step - 2):
                for ij in range(x_size*y_size):
                    reg_lambda2[f*x_size*y_size + ij, f*x_size*y_size + ij] = -1
                    reg_lambda2[(f + 1)*x_size*y_size + ij, f*x_size*y_size + ij] = 2
                    reg_lambda2[(f + 2)*x_size*y_size + ij, f*x_size*y_size + ij] = -1
        if reg_lambda is None:
            reg_lambda = reg_lambda2
        elif reg_lambda2 is not None:
            reg_lambda = np.append(reg_lambda, reg_lambda2, axis=1)

        if reg_pix is None:
            return reg_lambda
        elif reg_lambda is None:
            return reg_pix
        else:
            return np.append(reg_pix, reg_lambda, axis=1)

    def _extract_image_vals(self, j, i, radius=None):
        """Return all pixels within a radius of a given point as a 1D vector for each exposure."""
        img_arr = []
        for exp in self.exposures:
            img = exp.getMaskedImage().getImage().getArray()
            img = img[j - radius: j + radius + 1, i - radius: i + radius + 1]
            img_arr.append(np.ravel(img))
        return np.hstack(img_arr)

    def _insert_model_vals(self, j, i, vals, radius=None):
        if self.use_psf:
            psf_use = self.psf_avg
            self.model[:, j - radius: j + radius + 1, i - radius: i + radius + 1] += vals*psf_use
            self.weights[:, j - radius: j + radius + 1, i - radius: i + radius + 1] += psf_use
        else:
            psf_use = self.psf_avg
            self.model[:, j - radius: j + radius + 1, i - radius: i + radius + 1] += vals*psf_use
            self.weights[:, j - radius: j + radius + 1, i - radius: i + radius + 1] += psf_use

    def _calc_psf_model(self):
        psf_mat = []
        dcr_shift = []
        p0 = self.psf_size//2 - self.kernel_size//2
        p1 = p0 + self.kernel_size
        for img, exp in enumerate(self.exposures):
            el = self.elevation_arr[img]
            az = self.azimuth_arr[img]

            # Use the measured PSF as the solution of the shifted PSFs.
            # Taken at zenith, since we're solving for the shift and don't want to introduce any extra.
            dcr_genZ = DcrModel._dcr_generator(self.bandpass, pixel_scale=self.pixel_scale,
                                               elevation=90., azimuth=az)
            psf_mat.append(DcrModel._calc_psf_kernel_full(exposure=exp, dcr_gen=dcr_genZ, return_matrix=False,
                                                          x_size=self.psf_size, y_size=self.psf_size))
            # Calculate the expected shift (with no psf) due to DCR
            dcr_gen = DcrModel._dcr_generator(self.bandpass, pixel_scale=self.pixel_scale,
                                              elevation=el, azimuth=az)
            dcr_shift.append(DcrModel._calc_offset_phase(exposure=exp, dcr_gen=dcr_gen, return_matrix=True,
                                                         x_size=self.psf_size, y_size=self.psf_size))
        psf_mat = np.sum(np.hstack(psf_mat), axis=0)
        dcr_shift = np.hstack(dcr_shift)
        regularize_psf = self._build_regularization(x_size=self.psf_size, y_size=self.psf_size,
                                                    n_step=self.n_step, frequency_regularization=True)
        regularize_dim = regularize_psf.shape
        vals_use = np.append(psf_mat, np.zeros(regularize_dim[1]))
        kernel_use = np.append(dcr_shift.T, regularize_psf.T, axis=0)
        psf_soln = scipy.optimize.nnls(kernel_use, vals_use)
        psf_model = np.reshape(psf_soln[0], (self.n_step, self.psf_size, self.psf_size))
        self.psf_model = psf_model[:, p0: p1, p0: p1]
        psf_vals = np.sum(psf_model, axis=0)/self.n_step
        self.psf_avg = psf_vals[p0: p1, p0: p1]
        psf_image = afwImage.ImageD(self.psf_size, self.psf_size)
        psf_image.getArray()[:, :] = psf_vals
        psfK = afwMath.FixedKernel(psf_image)
        self.psf = measAlg.KernelPsf(psfK)

    def build_model(self, use_full=True, use_regularization=True, use_only_detected=False, verbose=True):
        """Calculate a model of the true sky using the known DCR offset for each freq plane."""
        dcr_kernel = self._build_dcr_kernel(use_full=use_full)
        self.model = np.zeros((self.n_step, self.y_size, self.x_size))
        self.weights = np.zeros_like(self.model)
        pix_radius = self.kernel_size//2

        variance = np.zeros((self.y_size, self.x_size))
        for exp in self.exposures:
            variance += exp.getMaskedImage().getVariance().getArray()**2.
        self.variance = np.sqrt(variance) / self.n_images
        detected_bit = exp.getMaskedImage().getMask().getPlaneBitMask("DETECTED")
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
                # this
                if use_only_detected:
                    if self.mask[j, i] & detected_bit == 0:
                        continue
                img_vals = self._extract_image_vals(j, i, radius=pix_radius)

                model_vals = self._solve_model(dcr_kernel, img_vals, use_regularization=use_regularization)
                self._insert_model_vals(j, i, model_vals, radius=pix_radius)
        if verbose:
            print("\nFinished building model.")

    def _build_dcr_kernel(self, use_full=None):
        dcr_kernel = []
        for img, exp in enumerate(self.exposures):
            el = self.elevation_arr[img]
            az = self.azimuth_arr[img]
            dcr_gen = DcrModel._dcr_generator(self.bandpass, pixel_scale=self.pixel_scale,
                                              elevation=el, azimuth=az)
            make_kernel_kwargs = dict(exposure=exp, dcr_gen=dcr_gen, return_matrix=True, psf_img=self.psf_avg,
                                      x_size=self.kernel_size, y_size=self.kernel_size)
            if self.use_psf:
                if use_full:
                    dcr_kernel.append(DcrModel._calc_psf_kernel_full(**make_kernel_kwargs))
                else:
                    dcr_kernel.append(DcrModel._calc_psf_kernel(**make_kernel_kwargs))
            else:
                dcr_kernel.append(DcrModel._calc_offset_phase(**make_kernel_kwargs))
        dcr_kernel = np.hstack(dcr_kernel)
        return dcr_kernel

    def _combine_masks(self):
        """Compute the bitwise OR of the input masks."""
        mask_arr = (exp.getMaskedImage().getMask().getArray() for exp in self.exposures)

        # Flags a pixel if ANY image is flagged there.
        for mask in mask_arr:
            if self.mask is None:
                self.mask = mask
            else:
                self.mask = np.bitwise_or(self.mask, mask)

    def _solve_model(self, dcr_kernel, img_vals, use_regularization=True):
        x_size = self.kernel_size
        y_size = self.kernel_size
        if use_regularization:
            regularize_dim = self.regularize.shape
            vals_use = np.append(img_vals, np.zeros(regularize_dim[1]))
            kernel_use = np.append(dcr_kernel.T, self.regularize.T, axis=0)
        else:
            vals_use = img_vals
            kernel_use = dcr_kernel.T
        model_solution = scipy.optimize.nnls(kernel_use, vals_use)
        model_vals = model_solution[0]
        return np.reshape(model_vals, (self.n_step, y_size, x_size))


def _calc_psf_kernel_subroutine(psf_img, dcr, x_size=None, y_size=None):
    if (x_size is None) | (y_size is None):
        y_size, x_size = psf_img.shape
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

    n_substep = 10
    psf_mat = np.zeros((x_size * y_size, x_size * y_size))
    for j in range(y_size):
        for i in range(x_size):
            ij = i + j * x_size
            sub_image = np.zeros_like(psf_img)
            for n in range(n_substep):
                j_use = j - y_size//2 + (dcr.dy.start*(n_substep - n) + dcr.dy.end*n)/n_substep
                i_use = i - x_size//2 + (dcr.dx.start*(n_substep - n) + dcr.dx.end*n)/n_substep
                sub_image += scipy_shift(psf_img, (j_use, i_use), mode='constant', cval=0.0)
            psf_mat[ij, :] = np.ravel(sub_image[x0:x1, y0:y1]) / n_substep
    return psf_mat


def _kernel_1d(offset, size, width=0.0, min_width=0.1):
    """Pre-compute the 1D sinc function values along each axis."""
    """
    @param offset: tuple of start/end pixel offsets of dft locations along single axis (either x or y)
    @params size: dimension in pixels of the given axis
    """
    n_substep = 10
    pi = np.pi
    pix = np.arange(size, dtype=np.float64)

    kernel = np.zeros(size, dtype=np.float64)
    for n in range(n_substep):
        loc = size//2. + (offset.start*(n_substep - n) + offset.end*n)/n_substep
        if loc % 1.0 == 0:
            kernel[int(loc)] += 1.0
        else:
            kernel += np.sin(pi*(pix - loc))/(pi*(pix - loc))
    return kernel/n_substep


class NonnegLstsqIterFit():
    def __init__():
        pass
