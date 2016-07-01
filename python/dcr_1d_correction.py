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
from scipy.linalg import toeplitz

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
        self.butler = daf_persistence.Butler(repository)
        dataId_gen = _build_dataId(obsid_range, band_name)
        elevation_arr = []
        azimuth_arr = []
        airmass_arr = []
        image_arr = []
        # mask_arr = []
        # variance_arr = []

        for _id in dataId_gen:
            exposure = self._load_exposures(_id)
            elevation_arr.append(exposure["elevation"])
            azimuth_arr.append(exposure["azimuth"])
            airmass_arr.append(exposure["airmass"])
            image_arr.append(exposure["image"])
            # mask_arr.append(exposure["mask"])
            # variance_arr.append(exposure["variance"])

        # if np.max(azimuth_arr) != np.min(azimuth_arr):
        #     print("Multiple azimuth angles detected! Only one angle is supported for now. Returning")
        #     return

        self.n_images = len(elevation_arr)
        self.y_size = (image_arr[0].shape)[0]
        self.x_size = (image_arr[0].shape)[1]
        azimuth = np.min(azimuth_arr)
        elevation_min = np.min(elevation_arr)
        pixel_scale = exposure["scale"]
        exposure_time = exposure["exptime"]

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
        self.refract_matrix = []

        # Regularization adapted from Nate Lust's DCR Demo iPython notebook
        # Calculate a difference matrix for regularization as if each wavelength were a pixel, then scale
        # The difference matrix to the size of the number of pixels times number of wavelengths
        baseReg = _difference(self.y_size)
        Regular = np.zeros((n_step*self.y_size, n_step*self.y_size))

        for i in range(n_step):
            Regular[i::n_step, i::n_step] = baseReg

        # Do the same thing as above but with the second derivative
        baseReg2 = _difference2(self.y_size)
        Regular2 = np.zeros((n_step*self.y_size, n_step*self.y_size))

        for i in range(n_step):
            Regular2[i::n_step, i::n_step] = baseReg2

        # Extra regularization that we force the SED to be smooth
        baseLam = _difference(n_step)
        smthLam = np.zeros(Regular.shape)

        for i in range(self.y_size):
            smthLam[i*n_step:i*n_step + n_step, i*n_step:i*n_step + n_step] = baseLam
        self.regular = Regular
        self.regular2 = Regular2
        self.smthLam = smthLam

    def build_matrix(self):
        for img_i, elevation in enumerate(self.elevation_arr):
            azimuth = self.azimuth_arr[img_i]
            self.refract_matrix.append(self._calc_refract_matrix(elevation, azimuth))

    def _calc_refract_matrix(self, elevation, azimuth):
        refract_matrix = np.zeros((self.y_size, self.n_step * self.y_size))
        pixel_scale = self.photoParams.platescale
        dcr_gen = _dcr_generator(self.bandpass, pixel_scale=pixel_scale,
                                 elevation=elevation, azimuth=azimuth)
        # NOTE: This is purely 1D for now! Offset from _dcr_generator is a tuple of the y and x offsets.
        for f_i, offset in enumerate(dcr_gen):
            offset_use = offset[1]
            i_high = np.ceil(offset_use)
            i_low = np.floor(offset_use)
            frac_high = offset_use - np.floor(offset_use)
            frac_low = np.ceil(offset_use) - offset_use
            for _i in range(self.y_size):
                if _i + i_low < 0:
                    refract_matrix[0, f_i + _i * self.n_step] = 1.
                elif _i + i_high >= self.y_size:
                    refract_matrix[-1, f_i + _i * self.n_step] = 1.
                else:
                    refract_matrix[_i + i_low, f_i + _i * self.n_step] = frac_low
                    refract_matrix[_i + i_high, f_i + _i * self.n_step] = frac_high
        return(refract_matrix)

    def build_inverse_squared_matrix(self):
        """Break out the computationally expensive step of computing the matrix inverse."""
        r_matrix_extend = reduce(lambda mat1, mat2: np.append(mat1, mat2, axis=0), self.refract_matrix)
        frac = .1
        r_matrix_squared = np.dot(r_matrix_extend.T, r_matrix_extend)
        reg_squared = np.dot(self.regular.T, self.regular)
        reg2_squared = np.dot(self.regular2.T, self.regular2)
        smthLam_squared = np.dot(self.smthLam.T, self.smthLam)
        mat_sum = r_matrix_squared + frac * (reg_squared + reg2_squared + smthLam_squared)
        mat_inv = np.linalg.pinv(mat_sum)
        self.transfer = np.dot(mat_inv, r_matrix_extend.T)

    def build_template(self):
        """Now we have to solve the linear equation for the above matrix for each pixel, across all images."""
        # NOTE: This is purely 1D for now, and assumed to ALWAYS be constrained to the y-axis!!
        # Start with a straightforward loop over the pixels to verify the algorithm. We'll optimize later.

        template = []
        for _i in range(self.x_size):
            img_vec = reduce(lambda vec1, vec2: np.append(vec1, vec2),
                             [image[:, _i] for image in self.image_arr])
            template.append(np.dot(self.transfer, img_vec.T))
        self.template = template
        # if self.use_bandpass:
        #     bandpass_normalized = self.bandpass.sb / self.bandpass.sb.sum()
        # else:
        #     bandpass_normalized = np.ones(self.n_step, dtype=np.float64)
        # # Matrix version of a linear least squares fit
        # template = np.zeros((self.y_size, self.x_size, self.n_step))
        # for _i in range(self.x_size):
        #     img_vec = reduce(lambda vec1, vec2: np.append(vec1, vec2),
        #                      [image[:, _i] for image in self.image_arr])
        #     template_vec = np.dot(self.transfer, img_vec.T)
        #     for f_i in range(self.n_step):
        #         template[:, _i, f_i] = template_vec[f_i::self.n_step]

        # self.template = [template[:, :, f_i] * bandpass_normalized[f_i] for f_i in range(self.n_step)]

    def apply_template(self, obsid=None, elevation=None, azimuth=None):
        if obsid is not None:
            dataId = _build_dataId(obsid, self.band_name)
            exposure = self._load_exposures(dataId)
            elevation = exposure["elevation"]
            azimuth = exposure["azimuth"]
        r_matrix = self._calc_refract_matrix(elevation, azimuth)
        image = np.zeros((self.y_size, self.x_size))
        for _i, template in enumerate(self.template):
            image[:, _i] = np.dot(r_matrix, template)
        return(image)

    def _load_exposures(self, dataId):
        calexp = self.butler.get("calexp", dataId=dataId)
        metadata = calexp.getMetadata()
        elevation = 90 - metadata.get("ZENITH")
        azimuth = metadata.get("AZIMUTH")
        airmass = metadata.get("AIRMASS")
        scale = calexp.getWcs().pixelScale().asArcseconds()
        exptime = calexp.getInfo().getCalib().getExptime()
        image = calexp.getMaskedImage().getImage().getArray()
        image = image[487: 487 + 128, 128: 256]  # Keep image small to run faster in testing

        mask = calexp.getMaskedImage().getMask().getArray()
        variance = calexp.getMaskedImage().getVariance().getArray()
        return {"elevation": elevation, "azimuth": azimuth, "airmass": airmass, "image": image,
                "mask": mask, "variance": variance, "scale": scale, "exptime": exptime}


def _difference(size):
    # adapted from Nate Lust's DCR Demo iPython notebook
    """ returns a toeplitz matrix
    difference regularization
    """
    r = np.zeros(size)
    c = np.zeros(size)
    r[0] = 1
    r[size - 1] = -1
    c[1] = -1
    return toeplitz(r, c).T


def _difference2(size):
    # adapted from Nate Lust's DCR Demo iPython notebook
    r = np.zeros(size)
    r[0] = 2
    r[1] = -1
    r[-1] = -1
    matrix = np.zeros((size, size))
    for i in range(size):
        matrix[i] = np.roll(r, i)
    return matrix


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
