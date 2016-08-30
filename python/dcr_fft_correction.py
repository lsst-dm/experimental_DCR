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
                 n_step=None, elevation_min=40., use_psf=True, **kwargs):
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
        self.dcr_max = int(np.ceil((dcr_test.next())[1]) + 1)
        self._fourier_transform()
        self.use_psf = use_psf

    def _fourier_transform(self, correct_mask=False):
        """Take the (real) Fourier transform of the input exposures, and mitigate masking."""
        fourier = []
        for exp in self.exposures:
            image = exp.getMaskedImage().getImage().getArray()
            # This mask includes only the footprints of detected sources.
            mask = np.zeros_like(image)
            detected_bit = exp.getMaskedImage().getMask().getPlaneBitMask("DETECTED")
            mask[np.bitwise_and(exp.getMaskedImage().getMask().getArray(), detected_bit) >= detected_bit] = 1.
            if correct_mask:
                fimage = np.fft.rfft2(image * mask)
                fmask = np.fft.rfft2(1.0 - mask) * np.std(image[mask == 0])
                fimage -= fmask
            else:
                fimage = np.fft.rfft2(image)
            fourier.append(fimage)
        self.fourier = np.asarray(fourier)

    # def _calc_dcr_shift(self):
    #     dcr_gen = _dcr_generator(self.bandpass, pixel_scale=self.pixel_scale,
    #                              elevation=elevation_min, azimuth=0.)

    def build_model(self, threshold=0.01):
        """Calculate a model of the true sky using the known DCR offset for each freq plane."""
        dcr_phase = []
        for _img, exp in enumerate(self.exposures):
            el = self.elevation_arr[_img]
            az = self.azimuth_arr[_img]
            dcr_gen = _dcr_generator(self.bandpass, pixel_scale=self.pixel_scale, elevation=el, azimuth=az)
            # dcr_gen2 = _dcr_generator(self.bandpass, pixel_scale=self.pixel_scale, elevation=90., azimuth=az)
            if self.use_psf:
                dcr_phase.append(_calc_psf_kernel(exp, dcr_gen))
            else:
                dcr_phase.append(_calc_offset_phase(exp, dcr_gen))
        dcr_phase = np.asarray(dcr_phase)
        self.dcr_phase = dcr_phase

        model_fourier = np.zeros((self.n_step, self.y_size, self.x_size//2 + 1), dtype=self.fourier.dtype)
        max_threshold = np.amax(np.abs(dcr_phase), axis=(0,1))
        self.residuals = np.zeros((self.y_size, self.x_size//2 + 1))
        for _j in range(self.y_size):
            for _i in range(self.x_size//2 + 1):
                if max_threshold[_j, _i] > threshold:
                    img_vals = self.fourier[:, _j, _i]
                    dcr_subarr = dcr_phase[:, :, _j, _i]

                    # dcr_mat = dcr_subarr.T.dot(dcr_subarr)
                    # dcr_mat_inv = np.linalg.pinv(dcr_mat)
                    # model_vals = dcr_mat_inv.dot(dcr_subarr.T.dot(img_vals))
                    model_solution = np.linalg.lstsq(dcr_subarr, img_vals)
                    model_vals = model_solution[0]
                    if len(model_solution[1]) == 1:
                        res_use = np.sqrt(model_solution[1])
                        self.residuals[_j, _i] = res_use
                        model_vals[res_use > np.abs(model_vals)] = 0.
                    # else:
                    #     model_vals[:] = 0.

                    # model_vals[np.abs(model_vals) > max_threshold[_j, _i]] = 0

                    # dcr_mat = np.real(dcr_subarr).T.dot(np.real(dcr_subarr))
                    # dcr_mat_inv = np.linalg.pinv(dcr_mat)
                    # model_vals_real = dcr_mat_inv.dot(np.real(dcr_subarr).T.dot(np.real(img_vals)))

                    # dcr_mat = np.imag(dcr_subarr).T.dot(np.imag(dcr_subarr))
                    # dcr_mat_inv = np.linalg.pinv(dcr_mat)
                    # model_vals_imag = dcr_mat_inv.dot(np.imag(dcr_subarr).T.dot(np.imag(img_vals)))

                    # dcr_mat = np.abs(dcr_subarr).T.dot(np.abs(dcr_subarr))
                    # dcr_mat_inv = np.linalg.pinv(dcr_mat)
                    # model_vals_real = dcr_mat_inv.dot(np.abs(dcr_subarr).T.dot(np.real(img_vals)))
                    # model_vals_imag = dcr_mat_inv.dot(np.abs(dcr_subarr).T.dot(np.imag(img_vals)))
                    model_fourier[:, _j, _i] = model_vals  #_real + 1j * model_vals_imag
        # self.model = np.fft.irfft2(model_fourier[_plane, :, :] for _plane in range(self.n_step))
        self.model = model_fourier

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


def _calc_offset_phase(exposure, offset_gen):
    """Return the 2D FFT of an offset generated by _dcr_generator in the form (dx, dy)."""
    phase_arr = []
    y_size = exposure.getHeight()
    x_size = exposure.getWidth()
    for offset in offset_gen:
        kernel_x = _kernel_1d(offset[0] + x_size//2, x_size)
        kernel_y = _kernel_1d(offset[1] + y_size//2, y_size)
        kernel = np.einsum('i,j->ij', kernel_y, kernel_x)
        phase_arr.append(np.fft.rfft2(np.fft.fftshift(kernel)))
    return(np.asarray(phase_arr))


def _calc_psf_kernel(exposure, offset_gen):

    y_size = exposure.getHeight()
    x_size = exposure.getWidth()
    psf_kernel_arr = []
    for offset in offset_gen:
        psf_pt = afwGeom.Point2D(offset[0], offset[1])
        psf_img = exposure.getPsf().computeImage(psf_pt).getArray()
        psf_y_size, psf_x_size = psf_img.shape
        psf = np.zeros((y_size, x_size), dtype=psf_img.dtype)
        dx = np.floor(offset[0])
        dy = np.floor(offset[1])
        y0 = int(y_size//2 - psf_y_size//2 + dx)
        y1 = y0 + psf_y_size
        x0 = int(x_size//2 - psf_x_size//2 + dy)
        x1 = x0 + psf_x_size
        psf[y0: y1, x0: x1] = psf_img
        psf_kernel_arr.append(np.fft.rfft2(np.fft.fftshift(psf)))
    return(np.asarray(psf_kernel_arr))


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
