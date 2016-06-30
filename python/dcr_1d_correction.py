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
This is a work in progress!
"""
from __future__ import print_function, division, absolute_import
import imp
import numpy as np
from scipy import constants

import lsst.daf.persistence as daf_persistence
from lsst.sims.photUtils import Bandpass, PhotometricParameters
from lsst.utils import getPackageDir
imp.load_source('calc_refractive_index', '/Users/sullivan/LSST/code/StarFast/calc_refractive_index.py')
from calc_refractive_index import diff_refraction


class DcrCorrection:
    """!Class that loads LSST calibrated exposures and produces airmass-matched template images."""

    def __init__(self, repository=".", obsid_range=None, band_name='g', wavelength_step=10,
                 n_step=None, use_bandpass=False, **kwargs):
        """
        Load images from the repository and set up parameters.
        @param repository: path to repository with the data. String, defaults to working directory
        @param obsid_range: obsid or range of obsids to process.
        """
        butler = daf_persistence.Butler(repository)
        dataId_gen = _build_dataId(obsid_range, band_name)
        elevation_arr = []
        azimuth_arr = []
        airmass_arr = []
        image_arr = []
        mask_arr = []
        variance_arr = []

        for _id in dataId_gen:
            calexp = butler.get("calexp", dataId=_id)
            metadata = calexp.getMetadata()
            elevation_arr.append(90 - metadata.get("ZENITH"))
            azimuth_arr.append(metadata.get("AZIMUTH"))
            airmass_arr.append(metadata.get("AIRMASS"))
            img = calexp.getMaskedImage().getImage().getArray()
            image_arr.append(img[487: 487+128, 128: 256])
            # mask_arr.append(calexp.getMaskedImage().getMask().getArray())
            # variance_arr.append(calexp.getMaskedImage().getVariance().getArray())

        # if np.max(azimuth_arr) != np.min(azimuth_arr):
        #     print("Multiple azimuth angles detected! Only one angle is supported for now. Returning")
        #     return

        self.n_images = len(elevation_arr)
        self.y_size = (image_arr[0].shape)[0]
        self.x_size = (image_arr[0].shape)[1]
        azimuth = np.min(azimuth_arr)
        elevation_min = np.min(elevation_arr)
        pixel_scale = calexp.getWcs().pixelScale().asArcseconds()
        exposure_time = calexp.getInfo().getCalib().getExptime()

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
        self.use_bandpass = use_bandpass
        self.photoParams = PhotometricParameters(exptime=exposure_time, nexp=1, platescale=pixel_scale,
                                                 bandpass=band_name)
        self.band_name = band_name
        self.image_arr = image_arr
        self.airmass_arr = airmass_arr
        self.elevation_arr = elevation_arr
        self.azimuth_arr = azimuth_arr

        # Initial calculation to set boundaries, using the lowest elevation observation.
        dcr_gen = _dcr_generator(bandpass, pixel_scale=pixel_scale, elevation=elevation_min, azimuth=azimuth)
        # NOTE: This is purely 1D for now! Offset from _dcr_generator is a tuple of the y and x offsets.
        refraction_pix = [offset[1] for offset in dcr_gen]
        refract_min = np.floor(np.min(refraction_pix)).astype(int)
        if refract_min > -1:
            refract_min = -1
        refract_max = np.ceil(np.max(refraction_pix)).astype(int)
        if refract_max < 1:
            refract_max = 1
        self.refract_min = refract_min
        self.refract_max = refract_max

    def build_matrix(self):
        # kernel_radius = 5
        if self.use_bandpass:
            bandpass_normalized = self.bandpass.sb / self.bandpass.sb.sum()
        else:
            bandpass_normalized = np.ones(self.n_step, dtype=np.float64)

        refract_matrix = np.zeros((self.n_images * self.y_size, self.n_step * self.y_size), dtype=np.float64)
        for img_i, elevation in enumerate(self.elevation_arr):
            azimuth = self.azimuth_arr[img_i]
            pixel_scale = self.photoParams.platescale
            dcr_gen = _dcr_generator(self.bandpass, pixel_scale=pixel_scale,
                                     elevation=elevation, azimuth=azimuth)
            # NOTE: This is purely 1D for now! Offset from _dcr_generator is a tuple of the y and x offsets.
            for f_i, offset in enumerate(dcr_gen):
                offset_use = -offset[1]
                refract_kernel = kernel_1d(np.arange(self.y_size) + offset_use, self.y_size)  # kernel is 2D
                refract_kernel *= bandpass_normalized[f_i]
                # for _j in range(kernel_radius):
                #     refract_kernel[_j, _j + kernel_radius:] = 0.
                # for _j in range(kernel_radius, self.y_size - kernel_radius):
                #     refract_kernel[_j, _j + kernel_radius:] = 0.
                #     refract_kernel[_j, :_j - kernel_radius] = 0.
                # for _j in range(self.y_size - kernel_radius, self.y_size):
                #     refract_kernel[_j, :_j - kernel_radius] = 0.
                refract_matrix[img_i * self.y_size: (img_i + 1) * self.y_size,
                               f_i * self.y_size: (f_i + 1) * self.y_size] = refract_kernel
                # print(offset_use)
                # _i_low = np.floor(offset_use).astype(int)
                # _i_high = np.ceil(offset_use).astype(int)
                # if _i_low == _i_high:
                #     frac_low = 0.0
                #     frac_high = 1.0
                # else:
                #     frac_high = offset_use - _i_low
                #     frac_low = _i_high - offset_use
                # for _j in range(-self.refract_min + 1):
                #     ind_0 = _j + f_i * self.y_size
                #     ind_1 = _j + img_i * self.y_size
                #     refract_matrix[ind_0, ind_1] = bandpass_normalized[f_i]
                # for _j in range(self.y_size - self.refract_max - 1, self.y_size):
                #     ind_0 = _j + f_i * self.y_size
                #     ind_1 = _j + img_i * self.y_size
                #     refract_matrix[ind_0, ind_1] = bandpass_normalized[f_i]
                # for _j in range(-self.refract_min, self.y_size - self.refract_max):
                #     ind_0 = _j + f_i * self.y_size
                #     ind_1 = _j + img_i * self.y_size
                #     if (_j + _i_low >= 0) & (_j + _i_low < self.y_size):
                #         refract_matrix[ind_0, ind_1 + _i_low] = bandpass_normalized[f_i] * frac_low
                #     if (_j + _i_high >= 0) & (_j + _i_high < self.y_size):
                #         refract_matrix[ind_0, ind_1 + _i_high] = bandpass_normalized[f_i] * frac_high
        self.refract_matrix = refract_matrix

    def build_inverse_squared_matrix(self):
        """Break out the computationally expensive step of computing the matrix inverse."""
        matrix_squared = np.einsum('ij,ik->jk', self.refract_matrix, self.refract_matrix)
        coefficient_matrix = np.zeros((self.n_step, self.n_step), dtype=np.float64)
        for _j in range(self.n_step):
            j0 = _j * self.y_size
            j1 = (_j + 1) * self.y_size
            for _i in range(self.n_step):
                i0 = _i * self.y_size
                i1 = (_i + 1) * self.y_size
                coefficient_matrix[_i, _j] = np.max(matrix_squared[i0: i1, j0: j1])
        coefficient_matrix /= np.mean(coefficient_matrix)
        coefficient_inv = np.linalg.inv(coefficient_matrix)
        normalization = 1. / self.n_step**2.0
        for _j in range(self.n_step):
            j0 = _j * self.y_size
            j1 = (_j + 1) * self.y_size
            for _i in range(self.n_step):
                i0 = _i * self.y_size
                i1 = (_i + 1) * self.y_size
                matrix_squared[i0: i1, j0: j1] = (normalization * coefficient_inv[_i, _j]
                                                  * (matrix_squared[i0: i1, j0: j1]).T)
        self.matrix_squared_inv = matrix_squared
        # edge_pix = 1
        # matrix_squared_arr = []
        # for f_i in range(self.n_step):
        #     f0 = f_i * self.y_size
        #     f1 = (f_i + 1) * self.y_size
        #     matrix_single = np.zeros((self.y_size, self.y_size), dtype=np.float64)
        #     for img_i in range(self.n_images):
        #         i0 = img_i * self.y_size
        #         i1 = (img_i + 1) * self.y_size
        #         # matrix_single += (self.refract_matrix[i0: i1, f0: f1]
        #         #                   * np.abs(self.refract_matrix[i0: i1, f0: f1]))
        #         matrix_single += np.einsum('ij,ik->jk', self.refract_matrix[i0: i1, f0: f1],
        #                                    np.abs(self.refract_matrix[i0: i1, f0: f1]))
        #     matrix_squared_arr.append(matrix_single)
        # for matrix in matrix_squared_arr:
        #     # matrix = np.linalg.inv(matrix)
        #     matrix = _safe_divide(matrix)
        #     matrix[0: edge_pix, :] = 0
        #     matrix[-edge_pix:, :] = 0
        #     matrix[:, 0: edge_pix] = 0
        #     matrix[:, -edge_pix:] = 0
        # # for _j in range(self.n_step):
        # #     for _i in range(self.n_step):
        # #         j0 = _j * self.y_size
        # #         j1 = (_j + 1) * self.y_size
        # #         i0 = _i * self.y_size
        # #         i1 = (_i + 1) * self.y_size
        # #         sub_matrix = matrix_squared[j0:j1, i0:i1]
        # #         # large array, so perform operation in place
        # #         matrix_squared[j0:j1, i0:i1] = np.linalg.inv(sub_matrix)
        # #         matrix_squared[j0, i0:i1] = 0.
        # #         matrix_squared[j1 - 1, i0:i1] = 0.
        # #         matrix_squared[j0:j1, i0] = 0.
        # #         matrix_squared[j0:j1, i1 - 1] = 0.
        # self.matrix_squared_inv = matrix_squared_arr

    def build_template(self):
        """Now we have to solve the linear equation for the above matrix for each pixel, across all images."""
        # NOTE: This is purely 1D for now, and assumed to ALWAYS be constrained to the y-axis!!
        # Start with a straightforward loop over the pixels to verify the algorithm. We'll optimize later.

        if self.use_bandpass:
            bandpass_normalized = self.bandpass.sb / self.bandpass.sb.sum()
        else:
            bandpass_normalized = np.ones(self.n_step, dtype=np.float64)
        # Matrix version of a linear least squares fit
        template = np.zeros((self.y_size, self.x_size, self.n_step))
        for _i in range(self.x_size):
            img_vec = np.zeros(self.y_size * self.n_images)
            for s_i, image in enumerate(self.image_arr):
                img_vec[s_i * self.y_size: (s_i + 1) * self.y_size] = image[:, _i] / self.n_images
            # img_vec = np.hstack([image[:, _i] for image in self.image_arr])
            moment_vec = np.einsum('ij,i->j', self.refract_matrix, img_vec)  # transpose of refract_matrix

            template_vec = np.einsum('ij,i->j', self.matrix_squared_inv, moment_vec)

            # matrix_squared_inv should be the identity matrix times a scale for each slice
            # template_vec = moment_vec
            for f_i in range(self.n_step):
                template[:, _i, f_i] = template_vec[f_i * self.y_size: (f_i + 1) * self.y_size]
            # for f_i in range(self.n_step):
            #     moment_single = moment_vec[f_i * self.y_size: (f_i + 1) * self.y_size]
            #     template[:, _i, f_i] = np.einsum('ji,j->i', self.matrix_squared_inv[f_i], moment_single)

        self.template = [template[:, :, f_i] * bandpass_normalized[f_i] for f_i in range(self.n_step)]

    def correct_template(self):
        for img_i, elevation in enumerate(self.elevation_arr):
            pass


def _safe_divide(array):
    result = np.zeros(array.shape)
    result[np.nonzero(array)] = 1.0 / array[np.nonzero(array)]
    return(result)


def _refract_inverse(refraction_vector, refract_min=None, refract_max=None, dimension=None):
    if refract_min > -1:
        refract_min = -1
    if refract_max < 1:
        refract_max = 1
    refract_matrix = np.zeros((dimension, dimension), dtype=np.float64)
    print(refract_min, refract_max, refraction_vector)
    for _i in range(np.abs(refract_min)):
        refract_matrix[0: _i + refract_max + 1, _i] = refraction_vector[np.abs(refract_min + _i):]
    for _i in range(np.abs(refract_min), dimension - refract_max):
        refract_matrix[_i + refract_min: _i + refract_max + 1, _i] = refraction_vector
    for _i in range(dimension - refract_max, dimension):
        refract_matrix[_i + refract_min:, _i] = refraction_vector[: dimension - (_i + refract_max + 1)]
    try:
        refract_inverse = np.linalg.inv(refract_matrix)
        return refract_inverse
    except np.linalg.LinAlgError:
        print("The refraction matrix was not invertable!")
        return refract_matrix


def _build_dataId(obsid_range, band):
    if hasattr(obsid_range, '__iter__'):
        if len(obsid_range) > 2:
            if obsid_range[2] < obsid_range[0]:
                dataId = ({'visit': obsid, 'raft': '2,2', 'sensor': '1,1', 'filter': band}
                          for obsid in np.arange(obsid_range[0], obsid_range[1], obsid_range[2]))
            else:
                dataId = ({'visit': obsid, 'raft': '2,2', 'sensor': '1,1', 'filter': band}
                          for obsid in obsid_range)
        else:
            dataId = ({'visit': obsid, 'raft': '2,2', 'sensor': '1,1', 'filter': band}
                      for obsid in np.arange(obsid_range[0], obsid_range[1]))
    else:
        dataId = ({'visit': obsid, 'raft': '2,2', 'sensor': '1,1', 'filter': band} for obsid in [obsid_range])
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


# NOTE: This function was copied from fast_dft.py
def kernel_1d(locs, size):
    """
    pre-compute the 1D sinc function values along each axis.

    @param locs: pixel coordinates of dft locations along single axis (either x or y)
    @params size: dimension in pixels of the given axis
    """
    pi = np.pi
    pix = np.arange(size, dtype=np.float64)
    sign = np.power(-1.0, pix)
    offset = np.floor(locs)
    delta = locs - offset
    kernel = np.zeros((len(locs), size), dtype=np.float64)
    for i, loc in enumerate(locs):
        if delta[i] == 0:
            kernel[i, :][offset[i]] = 1.0
        else:
            kernel[i, :] = np.sin(-pi * loc) / (pi * (pix - loc)) * sign
    return kernel
