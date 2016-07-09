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


class DcrCorrection:
    """!Class that loads LSST calibrated exposures and produces airmass-matched template images."""

    def __init__(self, repository=".", obsid_range=None, band_name='g', wavelength_step=10,
                 n_step=None, **kwargs):
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

        # if np.max(azimuth_arr) != np.min(azimuth_arr):
        #     print("Multiple azimuth angles detected! Only one angle is supported for now. Returning")
        #     return

        self.n_images = len(self.elevation_arr)
        self.y_size, self.x_size = self.exposures[0].getDimensions()
        pixel_scale = calexp.getWcs().pixelScale().asArcseconds()
        exposure_time = calexp.getInfo().getCalib().getExptime()
        self.bbox = calexp.getBBox()
        self.wcs = calexp.getWcs()

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
        self.photoParams = PhotometricParameters(exptime=exposure_time, nexp=1, platescale=pixel_scale,
                                                 bandpass=band_name)
        min_elevation = np.min(self.elevation_arr)
        dcr_test = _dcr_generator(bandpass, pixel_scale=pixel_scale, elevation=min_elevation, azimuth=0.)
        self.dcr_max = np.ceil((dcr_test.next())[1])
        self._build_regularization()

        # Construct a matrix for each input image that maps multiple sub-bandwidth frequency planes to
        #  a DCR-affected continuum image
        self.dcr_matrix = []
        for img_i, elevation in enumerate(self.elevation_arr):
            azimuth = self.azimuth_arr[img_i]
            self.dcr_matrix.append(_calc_dcr_matrix(elevation=elevation, azimuth=azimuth, size=self.y_size,
                                                    pixel_scale=pixel_scale, bandpass=self.bandpass))

    def _build_regularization(self):
        # Regularization adapted from Nate Lust's DCR Demo iPython notebook
        # Calculate a difference matrix for regularization as if each wavelength were a pixel, then scale
        # The difference matrix to the size of the number of pixels times number of wavelengths
        baseReg = _difference(self.y_size)
        Regular = np.zeros((self.n_step*self.y_size, self.n_step*self.y_size))

        for i in range(self.n_step):
            Regular[i::self.n_step, i::self.n_step] = baseReg

        # Do the same thing as above but with the second derivative
        baseReg2 = _difference2(self.y_size)
        Regular2 = np.zeros((self.n_step*self.y_size, self.n_step*self.y_size))

        for i in range(self.n_step):
            Regular2[i::self.n_step, i::self.n_step] = baseReg2

        # Extra regularization that we force the SED to be smooth
        baseLam = _difference(self.n_step)
        smthLam = np.zeros(Regular.shape)

        for i in range(self.y_size):
            smthLam[i*self.n_step: i*self.n_step + self.n_step,
                    i*self.n_step: i*self.n_step + self.n_step] = baseLam
        self.regular = Regular.T
        self.regular2 = Regular2.T
        self.smthLam = smthLam.T

    def build_transfer_matrix(self):
        """Break out the computationally expensive step of computing the matrix inverse."""
        """Note: this is very slow."""
        dcr_matrix_extend = reduce(lambda mat1, mat2: np.append(mat1, mat2, axis=0), self.dcr_matrix)
        frac = 1.
        dcr_matrix_squared = dcr_matrix_extend.T.dot(dcr_matrix_extend)
        reg_squared = self.regular.T.dot(self.regular)
        reg2_squared = self.regular2.T.dot(self.regular2)
        smthLam_squared = self.smthLam.T.dot(self.smthLam)
        mat_sum = dcr_matrix_squared + frac * (reg_squared + reg2_squared + smthLam_squared)
        # mat_sum = dcr_matrix_squared + frac * (reg_squared + reg2_squared + smthLam_squared)
        mat_inv = np.linalg.pinv(mat_sum)
        dcr_radius = int(self.dcr_max * self.n_step)
        for _i in range(self.y_size * self.n_step):
            if _i > dcr_radius:
                mat_inv[_i, 0:_i - dcr_radius] = 0.
            if _i + dcr_radius < self.y_size * self.n_step:
                mat_inv[_i, _i + dcr_radius:] = 0.
        self.transfer = mat_inv.dot(dcr_matrix_extend.T)

    def build_model(self):
        """Now we have to solve the linear equation for the above matrix for each pixel, across all images."""
        # NOTE: This is purely 1D for now, and assumed to ALWAYS be constrained to the y-axis!!
        # Start with a straightforward loop over the pixels to verify the algorithm. We'll optimize later.
        """Note: this is very slow."""

        model = []
        for _i in range(self.x_size):
            img_vec = reduce(lambda vec1, vec2: np.append(vec1, vec2),
                             [(calexp.getMaskedImage().getImage().getArray())[:, _i]
                              for calexp in self.exposures])
            model.append(self.transfer.dot(img_vec.T))
        # model_struct = Struct(model=model, azimuths=self.azimuth_arr, elevations=self.elevation_arr,
        #                       bandpass=self.bandpass, photoParams=self.photoParams,
        #                       bbox=self.bbox, wcs=self.wcs)
        self.model = model

        # Calculate the variance of the model.
        # This will need to change if the math in build_transfer_matrix changes.
        variance = np.zeros((self.y_size, self.x_size))
        for calexp in self.exposures:
            variance += calexp.getMaskedImage().getVariance().getArray()**2.
        self.variance = np.sqrt(variance) / self.n_images

    def view_model(self, index=0):
        """Simple function to convert the awkward array indexing of the model to a standard array."""
        model = np.zeros((self.y_size, self.x_size))
        for _i in range(self.x_size):
            model[:, _i] = self.model[_i][index::self.n_step]
        return(model)


    # def persist_model(self, filepath=None):
        # pass

    def generate_templates_from_model(self, obsid_range=None, elevation_arr=None, azimuth_arr=None,
                                      repository=None, output_directory=None):
        """Use the previously generated model and construct a dcr template image."""
        if repository is not None:
            butler = daf_persistence.Butler(repository)
        else:
            butler = self.butler
        if obsid_range is not None:
            dataId = _build_dataId(obsid_range, self.photoParams.bandpass)
            elevation_arr = []
            azimuth_arr = []
            for _id in dataId:
                calexp = butler.get("calexp", dataId=dataId[0])
                elevation_arr.append(90 - calexp.getMetadata().get("ZENITH"))
                azimuth_arr.append(calexp.getMetadata().get("AZIMUTH"))
        else:
            if not hasattr(elevation_arr, '__iter__'):
                elevation_arr = [elevation_arr]
                obsid_range = [{"visit": None}]
            else:
                obsid_range = [{"visit": None} for elevation in elevation_arr]
            if azimuth_arr is None:
                azimuth_arr = [0 for _i in range(len(elevation_arr))]
            elif not hasattr(azimuth_arr, '__iter__'):
                azimuth = azimuth_arr
                azimuth_arr = [azimuth for _i in range(len(elevation_arr))]

        for _i, elevation in enumerate(elevation_arr):
            dcr_matrix = _calc_dcr_matrix(elevation=elevation, azimuth=azimuth_arr[_i], size=self.y_size,
                                          pixel_scale=self.photoParams.platescale, bandpass=self.bandpass)
            image = np.zeros((self.y_size, self.x_size))
            for _j, model in enumerate(self.model):
                image[:, _j] = np.dot(dcr_matrix, model)

            # seed and obsid will be over-written each iteration, but are needed to use _create_exposure as-is
            self.seed = None
            self.obsid = dataId[_i]['visit'] + 500 % 1000
            exposure = self._create_exposure(image, variance=self.variance, snap=0,
                                             elevation=elevation, azimuth=azimuth_arr[_i])
            if output_directory is not None:
                band_dict = {'u': 0, 'g': 1, 'r': 2, 'i': 3, 'z': 4, 'y': 5}
                bn = band_dict[self.photoParams.bandpass]
                filename = "lsst_e_%3i_f%i_R22_S11_E%3.3i.fits" % (exposure.getMetadata().get("OBSID"), bn, 2)
                exposure.writeFits(output_directory + "images/" + filename)
            yield(exposure)

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

        # mask = exposure.getMaskedImage().getMask().getArray()
        # edge_maskval = afwImage.MaskU_getPlaneBitMask("EDGE")
        # edge_mask_dist = np.ceil(self.psf.getFWHM())
        # mask[0: edge_mask_dist, :] = edge_maskval
        # mask[:, 0: edge_mask_dist] = edge_maskval
        # mask[-edge_mask_dist:, :] = edge_maskval
        # mask[:, -edge_mask_dist:] = edge_maskval
        # # Check for saturation, and mask any saturated pixels
        # sat_maskval = afwImage.MaskU_getPlaneBitMask("SAT")
        # if np.max(array) > self.saturation:
        #     y_sat, x_sat = np.where(array >= self.saturation)
        #     mask[y_sat, x_sat] += sat_maskval

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


def _calc_dcr_matrix(elevation=None, azimuth=None, size=None, pixel_scale=None, bandpass=None):
    n_step = int(np.ceil((bandpass.wavelen_max - bandpass.wavelen_min) / bandpass.wavelen_step))
    dcr_matrix = np.zeros((size, n_step * size))
    # kernel_size = 4
    dcr_gen = _dcr_generator(bandpass, pixel_scale=pixel_scale,
                             elevation=elevation, azimuth=azimuth)
    # NOTE: This is purely 1D for now! Offset from _dcr_generator is a tuple of the y and x offsets.
    for f_i, offset in enumerate(dcr_gen):
        offset_use = offset[1]
        i_high = int(np.ceil(offset_use))
        i_low = int(np.floor(offset_use))
        frac_high = offset_use - np.floor(offset_use)
        frac_low = np.ceil(offset_use) - offset_use
        # kernel = kernel_1d([1 + frac_high], kernel_size)
        for _i in range(size):
            if _i + i_low < 0:
                dcr_matrix[0, f_i + _i * n_step] = 1.
            elif _i + i_high >= size:
                dcr_matrix[-1, f_i + _i * n_step] = 1.
            else:
                dcr_matrix[_i + i_low, f_i + _i * n_step] = frac_low
                dcr_matrix[_i + i_high, f_i + _i * n_step] = frac_high
    return(dcr_matrix)


def _difference(size):
    # adapted from Nate Lust's DCR Demo iPython notebook
    """ returns a toeplitz matrix
    difference regularization
    """
    size_use = size + 1
    r = np.zeros(size_use)
    c = np.zeros(size_use)
    r[0] = 1
    r[size_use - 1] = -1
    c[1] = -1
    result = toeplitz(r, c).T
    return result[0: size, 0: size]


def _difference2(size):
    # adapted from Nate Lust's DCR Demo iPython notebook
    size_use = size + 1
    r = np.zeros(size_use)
    r[0] = 2
    r[1] = -1
    r[-1] = -1
    matrix = np.zeros((size_use, size_use))
    for i in range(size_use):
        matrix[i] = np.roll(r, i)
    return matrix[0: size, 0: size]


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

def kernel_1d_sparse(locs, size):
    """Pre-compute a subset of pixels of a 1D sinc function."""
    pi = np.pi

    sign = [-1., 1, 1, -1]


    offset = np.floor(locs)
    delta = locs - offset
    kernel = np.zeros((len(locs), size), dtype=np.float64)
    for i, loc in enumerate(locs):
        if delta[i] == 0:
            kernel[i, :][offset[i]] = 1.0
        else:
            kernel[i, :] = np.sin(-pi * loc) / (pi * (pix - loc)) * sign
    return kernel
