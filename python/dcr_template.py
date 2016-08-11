#
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
"""
Attempts to create airmass-matched template images for existing images in an LSST repository.
This version works in 2D and uses Fourier transforms.
This is a work in progress!
"""
from __future__ import print_function, division, absolute_import
import imp
import numpy as np
from scipy import constants
from scipy.ndimage.interpolation import shift as scipy_shift
import scipy.optimize.nnls as positive_lstsq

import lsst.daf.persistence as daf_persistence
import lsst.afw.geom as afwGeom
import lsst.afw.image as afwImage
import lsst.pex.policy as pexPolicy
# from lsst.pipe.base.struct import Struct
from lsst.sims.photUtils import Bandpass, PhotometricParameters
from lsst.utils import getPackageDir
imp.load_source('calc_refractive_index', '/Users/sullivan/LSST/code/StarFast/calc_refractive_index.py')
from calc_refractive_index import diff_refraction

__all__ = ["DcrCorrection"]

lsst_lat = -30.244639
lsst_lon = -70.749417


class DcrModel:
    """Lightweight object that contains only the minimum needed to generate DCR-matched template exposures."""

    def __init__(self, ):
        print("Running DcrModel init!")
        pass

    def generate_templates_from_model(self, obsid_range=None, elevation_arr=None, azimuth_arr=None,
                                      repository=None, output_directory=None):
        """Use the previously generated model and construct a dcr template image."""
        exposures = []
        if repository is not None:
            butler = daf_persistence.Butler(repository)
        else:
            butler = self.butler
        if obsid_range is not None:
            dataId = _build_dataId(obsid_range, self.photoParams.bandpass)
            elevation_arr = []
            azimuth_arr = []
            for _id in dataId:
                calexp = butler.get("calexp", dataId=_id)
                exposures.append(calexp)
                elevation_arr.append(90 - calexp.getMetadata().get("ZENITH"))
                azimuth_arr.append(calexp.getMetadata().get("AZIMUTH"))
        else:
            print("This version REQUIRES an exposure to match the template.")

        for _img, exp in enumerate(exposures):
            el = elevation_arr[_img]
            az = azimuth_arr[_img]
            pix = self.photoParams.platescale
            dcr_gen = _dcr_generator(self.bandpass, pixel_scale=pix, elevation=el, azimuth=az)
            if self.use_psf:
                dcr_phase = _calc_psf_kernel(exp, dcr_gen)
            else:
                dcr_phase = _calc_offset_phase(exp, dcr_gen)
            image = np.fft.irfft2(np.sum(self.model * dcr_phase, axis=0))
            # dcr_matrix = _calc_dcr_matrix(elevation=elevation, azimuth=azimuth_arr[_img],
            #                               size=self.kernel_size, pixel_scale=self.photoParams.platescale,
            #                               bandpass=self.bandpass)
            # image = np.zeros((self.y_size, self.x_size))
            # radius = self.kernel_size//2
            # for _j in range(radius, self.y_size - radius):
            #     dcr_matrix_use = dcr_matrix[0: self.kernel_size]
            #     for _i in range(self.x_size):
            #         model_vals = np.ravel(np.array(self.model[_i][_j - radius: _j + radius + 1]))
            #         image[_j - radius: _j + radius + 1, _i] += np.dot(dcr_matrix_use, model_vals.T)

            # seed and obsid will be over-written each iteration, but are needed to use _create_exposure as-is
            self.seed = None
            self.obsid = dataId[_img]['visit'] + 500 % 1000
            exposure = self._create_exposure(image, variance=self.variance, snap=0,
                                             elevation=el, azimuth=az)
            if output_directory is not None:
                band_dict = {'u': 0, 'g': 1, 'r': 2, 'i': 3, 'z': 4, 'y': 5}
                bn = band_dict[self.photoParams.bandpass]
                filename = "lsst_e_%3i_f%i_R22_S11_E%3.3i.fits" % (exposure.getMetadata().get("OBSID"), bn, 2)
                exposure.writeFits(output_directory + "images/" + filename)
            yield(exposure)

    def view_model(self, index=0):
        # """Simple function to convert the awkward array indexing of the model to a standard array."""
        # model = np.zeros((self.y_size, self.x_size))
        # for _j in range(self.y_size):
        #     for _i in range(self.x_size):
        #         model[_j, _i] = self.model[_i][_j][index]
        # return(model)
        pass

    # NOTE: This function was copied from StarFast.py
    def _create_exposure(self, array, variance=None, elevation=None, azimuth=None, snap=0):
        """Convert a numpy array to an LSST exposure, and units of electron counts."""
        exposure = afwImage.ExposureF(self.bbox)
        exposure.setWcs(self.wcs)
        # We need the filter name in the exposure metadata, and it can't just be set directly
        try:
            exposure.setFilter(afwImage.Filter(self.photoParams.bandpass))
        except:
            filterPolicy = pexPolicy.Policy()
            filterPolicy.add("lambdaEff", self.bandpass.calc_eff_wavelen())
            afwImage.Filter.define(afwImage.FilterProperty(self.photoParams.bandpass, filterPolicy))
            exposure.setFilter(afwImage.Filter(self.photoParams.bandpass))
        calib = afwImage.Calib()
        calib.setExptime(self.photoParams.exptime)
        exposure.setCalib(calib)
        exposure.getMaskedImage().getImage().getArray()[:, :] = array
        if variance is None:
            variance = np.abs(array)
        exposure.getMaskedImage().getVariance().getArray()[:, :] = variance

        exposure.getMaskedImage().getMask().getArray()[:, :] = self.mask

        hour_angle = (90.0 - elevation) * np.cos(np.radians(azimuth)) / 15.0
        mjd = 59000.0 + (lsst_lat / 15.0 - hour_angle) / 24.0
        meta = exposure.getMetadata()
        meta.add("CHIPID", "R22_S11")
        # Required! Phosim output stores the snap ID in "OUTFILE" as the last three characters in a string.
        meta.add("OUTFILE", ("SnapId_%3.3i" % snap))

        meta.add("TAI", mjd)
        meta.add("MJD-OBS", mjd)

        meta.add("EXTTYPE", "IMAGE")
        meta.add("EXPTIME", 30.0)
        meta.add("AIRMASS", 1.0 / np.sin(np.radians(elevation)))
        meta.add("ZENITH", 90 - elevation)
        meta.add("AZIMUTH", azimuth)
        # meta.add("FILTER", self.band_name)
        if self.seed is not None:
            meta.add("SEED", self.seed)
        if self.obsid is not None:
            meta.add("OBSID", self.obsid)
            self.obsid += 1
        return(exposure)


class DcrCorrection(DcrModel):
    """!Class that loads LSST calibrated exposures and produces airmass-matched template images."""

    def __init__(self, repository=".", obsid_range=None, band_name='g', wavelength_step=10,
                 n_step=None, elevation_min=40., use_psf=True, use_fft=False, **kwargs):
        """
        Load images from the repository and set up parameters.
        @param repository: path to repository with the data. String, defaults to working directory
        @param obsid_range: obsid or range of obsids to process.
        """
        self.butler = daf_persistence.Butler(repository)
        dataId_gen = _build_dataId(obsid_range, band_name)
        self.elevation_arr = []
        self.azimuth_arr = []
        self.airmass_arr = []
        self.exposures = []

        for _id in dataId_gen:
            calexp = self.butler.get("calexp", dataId=_id)
            self.exposures.append(calexp)
            self.elevation_arr.append(90 - calexp.getMetadata().get("ZENITH"))
            self.azimuth_arr.append(calexp.getMetadata().get("AZIMUTH"))
            self.airmass_arr.append(calexp.getMetadata().get("AIRMASS"))

        self.n_images = len(self.elevation_arr)
        self.y_size, self.x_size = self.exposures[0].getDimensions()
        self.pixel_scale = calexp.getWcs().pixelScale().asArcseconds()
        exposure_time = calexp.getInfo().getCalib().getExptime()
        self.bbox = calexp.getBBox()
        self.wcs = calexp.getWcs()
        self._combine_masks()

        bandpass = _load_bandpass(band_name=band_name, wavelength_step=wavelength_step, **kwargs)
        if n_step is not None:
            wavelength_step = (bandpass.wavelen_max - bandpass.wavelen_min) / n_step
            bandpass = _load_bandpass(band_name=band_name, wavelength_step=wavelength_step, **kwargs)
        else:
            n_step = int(np.ceil((bandpass.wavelen_max - bandpass.wavelen_min) / bandpass.wavelen_step))
        if n_step >= self.n_images:
            print("Warning! Under-constrained system. Reducing number of frequency planes.")
            wavelength_step *= n_step / self.n_images
            bandpass = _load_bandpass(band_name=band_name, wavelength_step=wavelength_step, **kwargs)
            n_step = int(np.ceil((bandpass.wavelen_max - bandpass.wavelen_min) / bandpass.wavelen_step))
        self.n_step = n_step
        self.bandpass = bandpass
        self.photoParams = PhotometricParameters(exptime=exposure_time, nexp=1, platescale=self.pixel_scale,
                                                 bandpass=band_name)
        dcr_test = _dcr_generator(bandpass, pixel_scale=self.pixel_scale, elevation=elevation_min, azimuth=0.)
        self.dcr_max = int(np.ceil(np.max(dcr_test.next())) + 1)
        self.kernel_size = 2 * self.dcr_max + 1
        self.use_psf = use_psf
        self.use_fft = use_fft
        self._calc_psf_model()

    def _extract_image_vals(self, _j, _i, radius=None, fft=False):
        """Return all pixles within a radius of a given point as a 1D vector for each exposure."""
        img_arr = []
        for calexp in self.exposures:
            img = calexp.getMaskedImage().getImage().getArray()
            img = img[_j - radius: _j + radius + 1, _i - radius: _i + radius + 1]
            if fft:
                img = np.fft.fft2(np.fft.fftshift(img))
            img_arr.append(np.ravel(img))
        return(np.hstack(img_arr))

    def _insert_model_vals(self, _j, _i, vals, radius=None):
        if self.use_psf:
            psf_use = self.psf_sum
            self.model[:, _j - radius: _j + radius + 1, _i - radius: _i + radius + 1] += vals * psf_use
            self.weights[:, _j - radius: _j + radius + 1, _i - radius: _i + radius + 1] += psf_use
        else:
            psf_use = self.psf_sum
            self.model[:, _j - radius: _j + radius + 1, _i - radius: _i + radius + 1] += vals * psf_use
            self.weights[:, _j - radius: _j + radius + 1, _i - radius: _i + radius + 1] += psf_use

    def _calc_psf_model(self):
        psf_image = []
        dcr_shift = []
        for _img, calexp in enumerate(self.exposures):
            el = self.elevation_arr[_img]
            az = self.azimuth_arr[_img]
            dcr_gen0 = _dcr_generator(self.bandpass, pixel_scale=self.pixel_scale, elevation=90., azimuth=az)
            dcr_gen = _dcr_generator(self.bandpass, pixel_scale=self.pixel_scale, elevation=el, azimuth=az)
            psf_image.append(_calc_psf_kernel(calexp, dcr_gen0, fft=False,
                             x_size=self.kernel_size, y_size=self.kernel_size, return_matrix=False))
            dcr_shift.append(_calc_offset_phase(calexp, dcr_gen, fft=False,
                             x_size=self.kernel_size, y_size=self.kernel_size, return_matrix=True))
        psf_image = np.sum(np.hstack(psf_image), axis=0)
        dcr_shift = np.hstack(dcr_shift)
        psf_soln = np.linalg.lstsq(dcr_shift.T, psf_image)

        self.psf_model = np.reshape(psf_soln[0], (self.n_step, self.kernel_size, self.kernel_size))
        self.psf_sum = np.sum(self.psf_model,axis=0)

    def build_model(self):
        """Calculate a model of the true sky using the known DCR offset for each freq plane."""
        dcr_kernel = []
        for _img, calexp in enumerate(self.exposures):
            el = self.elevation_arr[_img]
            az = self.azimuth_arr[_img]
            dcr_gen = _dcr_generator(self.bandpass, pixel_scale=self.pixel_scale, elevation=el, azimuth=az)
            if self.use_psf:
                dcr_kernel.append(_calc_psf_kernel2(calexp, dcr_gen, fft=self.use_fft, reverse_offset=True,
                                  x_size=self.kernel_size, y_size=self.kernel_size, return_matrix=True, psf_img=self.psf_sum))
            else:
                dcr_kernel.append(_calc_offset_phase(calexp, dcr_gen, fft=self.use_fft, reverse_offset=True,
                                  x_size=self.kernel_size, y_size=self.kernel_size, return_matrix=True))
        dcr_kernel = np.hstack(dcr_kernel)
        self.dcr_kernel = dcr_kernel

        self.model = np.zeros((self.n_step, self.y_size, self.x_size))
        self.weights = np.zeros_like(self.model)
        for _j in range(self.y_size):
            for _i in range(self.x_size):
                # Deal with the edges later. Probably by padding the image with zeroes.
                if _i < self.dcr_max + 1:
                    continue
                elif self.x_size - _i < self.dcr_max + 1:
                    continue
                elif _j < self.dcr_max + 1:
                    continue
                elif self.y_size - _j < self.dcr_max + 1:
                    continue
                else:
                    if _i < 625+25: continue
                    if _i > 670+25: continue
                    if _j < 815-53: continue
                    if _j > 860-53: continue
                    img_vals = self._extract_image_vals(_j, _i, radius=self.dcr_max, fft=self.use_fft)

                self.debug_dcr_kernel = dcr_kernel
                self.debug_img_vals = img_vals
                self.debug_i = _i
                self.debug_j = _j
                model_vals = self._solve_model(dcr_kernel, img_vals)
                self.debug_model_vals = model_vals
                self._insert_model_vals(_j, _i, model_vals, radius=self.dcr_max)
                # model[:, _j, _i] = _solve_model(dcr_kernel, img_vals, fft=self.use_fft)
        # self.model = model

        variance = np.zeros((self.y_size, self.x_size))
        for calexp in self.exposures:
            variance += calexp.getMaskedImage().getVariance().getArray()**2.
        self.variance = np.sqrt(variance) / self.n_images

    def _combine_masks(self):
        """Compute the bitwise OR of the input masks."""
        mask_arr = (exp.getMaskedImage().getMask().getArray() for exp in self.exposures)
        mask = mask_arr.next()
        for m in mask_arr:
            mask = np.bitwise_or(mask, m)  # Flags a pixel if ANY image is flagged there.
        self.mask = mask

    def _solve_model(self, dcr_kernel, img_vals):
        if self.use_fft:
            x_size = self.kernel_size
            y_size = self.kernel_size
            model_fft = np.zeros((self.n_step, y_size, x_size), dtype=img_vals.dtype)
            kernel_use = np.reshape(dcr_kernel, (self.n_step, y_size, x_size, self.n_images))
            img_use = np.reshape(img_vals, (y_size, x_size, self.n_images))
            for _j in range(y_size):
                for _i in range(x_size):
                    _ij = _j + _i * y_size
                    # kernel_single = dcr_kernel[:, _ij:: x_size*y_size]
                    kernel_single = kernel_use[:, _j, _i, :]
                    # img_single = img_vals[_ij:: x_size*y_size]
                    img_single = img_use[_j, _i, :]
                    model_solution = np.linalg.lstsq(kernel_single.T, img_single)
                    model_fft[:, _j, _i] = model_solution[0]
            # for _p in range(self.n_step):
            #     model_fft[_p, :, :] = np.fft.fftshift(model_fft[_p, :, :])
            return(np.real(np.fft.ifftn(model_fft, axes=[1, 2])))
        else:
            x_size = self.kernel_size
            y_size = self.kernel_size
            model_solution = positive_lstsq(dcr_kernel.T, img_vals)
            model_vals = model_solution[0]
            # if len(model_solution[1]) == 1:
            #     res_use = np.sqrt(model_solution[1])
            #     self.residuals[_j, _i] = res_use
            #     model_vals[res_use > np.abs(model_vals)] = 0.
            return(np.reshape(model_vals, (self.n_step, y_size, x_size)))

    def view_model(self, index=0):
        model = self.model[index, :, :].copy()
        weights = self.weights[index, :, :]
        model[weights > 0] /= weights[weights > 0]
        model[weights <= 0] = 0.0
        return(model)


def _calc_offset_phase(exposure, offset_gen, fft=False, x_size=None, y_size=None, return_matrix=False,
                       reverse_offset=False):
    """Return the 2D FFT of an offset generated by _dcr_generator in the form (dx, dy)."""
    phase_arr = []
    if y_size is None:
        y_size = exposure.getHeight()
    if x_size is None:
        x_size = exposure.getWidth()
    for offset in offset_gen:
        if reverse_offset:
            offset_x = -offset[0]
            offset_y = -offset[1]
        else:
            offset_x = offset[0]
            offset_y = offset[1]
        kernel_x = _kernel_1d(offset_x + x_size//2, x_size)
        kernel_y = _kernel_1d(offset_y + y_size//2, y_size)
        kernel = np.einsum('i,j->ij', kernel_y, kernel_x)
        if fft:
            kernel = np.fft.fft2(np.fft.fftshift(kernel))
            phase_arr.append(np.ravel(kernel))
        elif return_matrix:
            shift_mat = np.zeros((x_size * y_size, x_size * y_size))
            for _j in range(y_size):
                for _i in range(x_size):
                    _ij = _j + _i * y_size
                    # NOTE: This might need to be transposed
                    shift_mat[_ij, :] = np.ravel(scipy_shift(kernel, (_j - y_size//2, _i - x_size//2),
                                                 order=0, mode='constant', cval=0.0))
            phase_arr.append(shift_mat)
        else:
            phase_arr.append(np.ravel(kernel))
    phase_arr = np.vstack(phase_arr)
    # if return_matrix:
    #     phase_arr = np.einsum("i,j->ij", phase_arr, phase_arr)
    return(phase_arr)


def _calc_psf_kernel2(exposure, offset_gen, fft=False, x_size=None, y_size=None, return_matrix=False,
                      reverse_offset=False, psf_img=None):

    if y_size is None:
        y_size = exposure.getHeight()
    if x_size is None:
        x_size = exposure.getWidth()
    psf_kernel_arr = []
    for offset in offset_gen:
        if reverse_offset:
            offset_x = -offset[0]
            offset_y = -offset[1]
        else:
            offset_x = offset[0]
            offset_y = offset[1]
        psf_y_size, psf_x_size = psf_img.shape
        psf = np.zeros((y_size, x_size), dtype=psf_img.dtype)
        if fft:
            psf = np.fft.fft2(np.fft.fftshift(psf_img))
            psf_kernel_arr.append(np.ravel(psf))
        elif return_matrix:
            psf_mat = np.zeros((x_size * y_size, x_size * y_size))
            for _j in range(y_size):
                for _i in range(x_size):
                    _ij = _j + _i * y_size
                    # NOTE: This might need to be transposed
                    j_use = _j - y_size//2 + offset_y
                    i_use = _i - x_size//2 + offset_x
                    psf_mat[_ij, :] = np.ravel(scipy_shift(psf_img, (j_use, i_use), mode='constant', cval=0.0))
            psf_kernel_arr.append(psf_mat)
        else:
            psf_kernel_arr.append(np.ravel(psf))

    psf_kernel_arr = np.vstack(psf_kernel_arr)
    # if return_matrix:
    #     psf_kernel_arr = np.einsum("i,j->ij", psf_kernel_arr, psf_kernel_arr)
    return(psf_kernel_arr)


def _calc_psf_kernel(exposure, offset_gen, fft=False, x_size=None, y_size=None, return_matrix=False,
                     reverse_offset=False):

    if y_size is None:
        y_size = exposure.getHeight()
    if x_size is None:
        x_size = exposure.getWidth()
    psf_kernel_arr = []
    for offset in offset_gen:
        if reverse_offset:
            offset_x = -offset[0]
            offset_y = -offset[1]
        else:
            offset_x = offset[0]
            offset_y = offset[1]
        psf_pt = afwGeom.Point2D(offset_x, offset_y)
        psf_img = exposure.getPsf().computeImage(psf_pt).getArray()
        psf_y_size, psf_x_size = psf_img.shape
        psf = np.zeros((y_size, x_size), dtype=psf_img.dtype)
        dx = np.floor(offset_x)
        dy = np.floor(offset_y)
        if psf_x_size < x_size:
            y0 = int(y_size//2 - psf_y_size//2 + dx)
            y1 = y0 + psf_y_size
            x0 = int(x_size//2 - psf_x_size//2 + dy)
            x1 = x0 + psf_x_size
            psf[y0: y1, x0: x1] = psf_img
        else:
            y0 = int(psf_y_size//2 - y_size//2 - dx)
            y1 = y0 + y_size
            x0 = int(psf_x_size//2 - x_size//2 - dy)
            x1 = x0 + x_size
            psf = psf_img[y0: y1, x0: x1]
        if fft:
            psf = np.fft.fft2(np.fft.fftshift(psf))
            psf_kernel_arr.append(np.ravel(psf))
        elif return_matrix:
            psf_mat = np.zeros((x_size * y_size, x_size * y_size))
            for _j in range(y_size):
                for _i in range(x_size):
                    _ij = _j + _i * y_size
                    # NOTE: This might need to be transposed
                    psf_mat[_ij, :] = np.ravel(scipy_shift(psf, (_j - y_size//2, _i - x_size//2),
                                               order=0, mode='constant', cval=0.0))
            psf_kernel_arr.append(psf_mat)
        else:
            psf_kernel_arr.append(np.ravel(psf))

    psf_kernel_arr = np.vstack(psf_kernel_arr)
    # if return_matrix:
    #     psf_kernel_arr = np.einsum("i,j->ij", psf_kernel_arr, psf_kernel_arr)
    return(psf_kernel_arr)


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
        dataId = [{'visit': obsid, 'raft': '2,2', 'sensor': '1,1', 'filter': band} for obsid in [obsid_range]]
    return(dataId)


# NOTE: This function was copied from StarFast.py
def _load_bandpass(band_name='g', wavelength_step=None, use_mirror=True, use_lens=True, use_atmos=True,
                   use_filter=True, use_detector=True, **kwargs):
    """
    !Load in Bandpass object from sims_photUtils.
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
            effwavelenphi = (self.wavelen[w_inds] * self.phi[w_inds]).sum() / self.phi[w_inds].sum()
            return effwavelenphi

        def calc_bandwidth(self):
            f0 = constants.speed_of_light / (self.wavelen_min * 1.0e-9)
            f1 = constants.speed_of_light / (self.wavelen_max * 1.0e-9)
            f_cen = constants.speed_of_light / (self.calc_eff_wavelen() * 1.0e-9)
            return(f_cen * 2.0 * (f0 - f1) / (f0 + f1))

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
    return(bandpass)


# NOTE: This function was copied from StarFast.py
def _wavelength_iterator(bandpass, use_midpoint=False):
    """Define iterator to ensure that loops over wavelength are consistent."""
    wave_start = bandpass.wavelen_min
    while np.ceil(wave_start) < bandpass.wavelen_max:
        wave_end = wave_start + bandpass.wavelen_step
        if wave_end > bandpass.wavelen_max:
            wave_end = bandpass.wavelen_max
        if use_midpoint:
            yield(bandpass.calc_eff_wavelen(wavelength_min=wave_start, wavelength_max=wave_end))
        else:
            yield((wave_start, wave_end))
        wave_start = wave_end


# NOTE: This function was copied from StarFast.py
def _dcr_generator(bandpass, pixel_scale=None, elevation=50.0, azimuth=0.0, **kwargs):
    """
    !Call the functions that compute Differential Chromatic Refraction (relative to mid-band).
    @param bandpass: bandpass object created with load_bandpass
    @param pixel_scale: plate scale in arcsec/pixel
    @param elevation: elevation angle of the center of the image, in decimal degrees.
    @param azimuth: azimuth angle of the observation, in decimal degrees.
    """
    zenith_angle = 90.0 - elevation
    wavelength_midpoint = bandpass.calc_eff_wavelen()
    for wavelength in _wavelength_iterator(bandpass, use_midpoint=True):
        # Note that refract_amp can be negative, since it's relative to the midpoint of the band
        refract_amp = diff_refraction(wavelength=wavelength, wavelength_ref=wavelength_midpoint,
                                      zenith_angle=zenith_angle, **kwargs)
        refract_amp *= 3600.0 / pixel_scale  # Refraction initially in degrees, convert to pixels.
        dx = refract_amp * np.sin(np.radians(azimuth))
        dy = refract_amp * np.cos(np.radians(azimuth))
        yield((dx, dy))


def _kernel_1d(loc, size):
    """
    pre-compute the 1D sinc function values along each axis.

    @param locs: pixel coordinates of dft locations along single axis (either x or y)
    @params size: dimension in pixels of the given axis
    """
    pi = np.pi
    pix = np.arange(size, dtype=np.float64)
    sign = np.power(-1.0, pix)
    offset = int(np.floor(loc))
    delta = loc - offset
    kernel = np.zeros(size, dtype=np.float64)
    if delta == 0:
        kernel[offset] = 1.0
    else:
        kernel[:] = np.sin(-pi * loc) / (pi * (pix - loc)) * sign
    return kernel
