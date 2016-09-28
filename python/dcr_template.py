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
from collections import namedtuple
from functools import reduce
import numpy as np
from scipy import constants
from scipy.ndimage.interpolation import shift as scipy_shift
import scipy.optimize.nnls as positive_lstsq

import lsst.daf.persistence as daf_persistence
import lsst.afw.coord as afwCoord
import lsst.afw.geom as afwGeom
import lsst.afw.image as afwImage
import lsst.afw.math as afwMath
import lsst.meas.algorithms as measAlg
import lsst.pex.policy as pexPolicy
from lsst.sims.photUtils import Bandpass, PhotometricParameters
from lsst.utils import getPackageDir
import unittest
import lsst.utils.tests
from .calc_refractive_index import diff_refraction

__all__ = ["DcrModel", "DcrCorrection"]

lsst_lat = -30.244639
lsst_lon = -70.749417


class DcrModel:
    """Lightweight object that contains only the minimum needed to generate DCR-matched template exposures."""

    def __init__(self, model_repository=None, band_name='g', debug_mode=False, **kwargs):
        self.debug = debug_mode
        self.butler = None
        self.load_model(model_repository=model_repository, band_name=band_name, **kwargs)
        self.use_fft = False

    def generate_templates_from_model(self, obsid_range=None, elevation_arr=None, azimuth_arr=None,
                                      repository=None, output_repository=None, add_noise=False, use_full=True,
                                      kernel_size=None, **kwargs):
        """Use the previously generated model and construct a dcr template image."""
        exposures = []
        if repository is not None:
            butler = daf_persistence.Butler(repository)
            if self.butler is None:
                self.butler = butler
        else:
            butler = self.butler
        if output_repository is not None:
            butler_out = daf_persistence.Butler(output_repository)
        if obsid_range is not None:
            dataId = self._build_dataId(obsid_range, self.photoParams.bandpass)
            elevation_arr = []
            azimuth_arr = []
            for _id in dataId:
                calexp = butler.get("calexp", dataId=_id)
                exposures.append(calexp)
                elevation_arr.append(90 - calexp.getMetadata().get("ZENITH"))
                azimuth_arr.append(calexp.getMetadata().get("AZIMUTH"))
        else:
            print("This version REQUIRES an exposure to match the template.")

        if kernel_size is not None:
            self.kernel_size = kernel_size
            self._build_regularization()
            self._calc_psf_model()
        for _img, exp in enumerate(exposures):
            el = elevation_arr[_img]
            az = azimuth_arr[_img]
            pix = self.photoParams.platescale
            dcr_gen = _dcr_generator(self.bandpass, pixel_scale=pix, elevation=el, azimuth=az)

            if self.use_psf:
                if use_full:
                    dcr_kernel = _calc_psf_kernel_full(exp, dcr_gen, return_matrix=True, psf_img=self.psf_avg,
                                                       x_size=self.kernel_size, y_size=self.kernel_size)
                else:
                    dcr_kernel = _calc_psf_kernel(exp, dcr_gen, return_matrix=True, psf_img=self.psf_avg,
                                                  x_size=self.kernel_size, y_size=self.kernel_size)
            else:
                dcr_kernel = _calc_offset_phase(exp, dcr_gen, return_matrix=True,
                                                x_size=self.kernel_size, y_size=self.kernel_size)
            if self.use_fft:
                template = np.fft.ifft2(np.sum(self.model * dcr_kernel, axis=0))
            else:
                template = np.zeros((self.y_size, self.x_size))
                weights = np.zeros((self.y_size, self.x_size))
                pix_radius = self.kernel_size//2
                for _j in range(self.y_size):
                    for _i in range(self.x_size):
                        if self._edge_test(_j, _i):
                            continue
                        model_vals = self._extract_model_vals(_j, _i, radius=pix_radius, fft=self.use_fft,
                                                              model=self.model, weights=self.weights)
                        template_vals = self._apply_dcr_kernel(dcr_kernel, model_vals,
                                                               x_size=self.kernel_size,
                                                               y_size=self.kernel_size)
                        self._insert_template_vals(_j, _i, template_vals, template=template, weights=weights,
                                                   radius=pix_radius, kernel=self.psf_avg)
                template[weights > 0] /= weights[weights > 0]
                template[weights == 0] = 0.0
                if add_noise:
                    variance_level = np.median(exp.getMaskedImage().getVariance().getArray())
                    rand_gen = np.random
                    template += rand_gen.normal(scale=np.sqrt(variance_level), size=template.shape)

            dataId_out = dataId[_img]
            exposure = self._create_exposure(template, variance=np.abs(template), snap=0,
                                             elevation=el, azimuth=az, obsid=dataId_out['visit'])
            if output_repository is not None:
                butler_out.put(exposure, "calexp", dataId=dataId_out)
            yield(exposure)

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
        return(dataId)

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
    def _extract_model_vals(_j, _i, radius=None, fft=False, model=None, weights=None):
        """Return all pixles within a radius of a given point as a 1D vector for each dcr plane model."""
        model_arr = []
        if weights is None:
            weights = np.ones_like(model)
        for _f in range(model.shape[0]):
            model_use = model[_f, _j - radius: _j + radius + 1, _i - radius: _i + radius + 1].copy()
            weights_use = weights[_f, _j - radius: _j + radius + 1, _i - radius: _i + radius + 1]
            model_use[weights_use > 0] /= weights_use[weights_use > 0]
            model_use[weights_use <= 0] = 0.0
            if fft:
                model = np.fft.fft2(np.fft.fftshift(model))
            model_arr.append(np.ravel(model_use))
        return(np.hstack(model_arr))

    @staticmethod
    def _insert_template_vals(_j, _i, vals, template=None, weights=None, radius=None, kernel=None):
        if kernel is None:
            kernel = 1.0
        template[_j - radius: _j + radius + 1, _i - radius: _i + radius + 1] += vals*kernel
        weights[_j - radius: _j + radius + 1, _i - radius: _i + radius + 1] += kernel

    def _edge_test(self, _j, _i):
        x0 = 150
        dx = 65
        y0 = 480
        dy = 70
        pix_radius = self.kernel_size//2

        # Deal with the edges later. Probably by padding the image with zeroes.
        if _i < pix_radius + 1:
            edge = True
        elif self.x_size - _i < pix_radius + 1:
            edge = True
        elif _j < pix_radius + 1:
            edge = True
        elif self.y_size - _j < pix_radius + 1:
            edge = True
        elif self.debug:
            if _i < x0:
                edge = True
            elif _i > x0+dx:
                edge = True
            elif _j < y0:
                edge = True
            elif _j > y0+dy:
                edge = True
            else:
                edge = False
        else:
            edge = False
        return(edge)

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
        return(exposure)

    def export_model(self, model_repository=None):
        """Persist a DcrModel with metadata to a repository."""
        if model_repository is None:
            butler = self.butler
        else:
            butler = daf_persistence.Butler(model_repository)
        wave_gen = _wavelength_iterator(self.bandpass, use_midpoint=False)
        for _f in range(self.n_step):
            wl_start, wl_end = wave_gen.next()
            exp = self._create_exposure(self.model[_f, :, :], variance=self.weights[_f, :, :],
                                        elevation=90., azimuth=0., ksupport=self.kernel_size,
                                        subfilt=_f, nstep=self.n_step, wavelow=wl_start, wavehigh=wl_end,
                                        wavestep=self.bandpass.wavelen_step, psf_flag=self.use_psf)
            butler.put(exp, "dcrModel", dataId=self._build_model_dataId(self.photoParams.bandpass, _f))

    def load_model(self, model_repository=None, band_name='g', **kwargs):
        """Depersist a DcrModel from a repository and set up the metadata."""
        if model_repository is None:
            butler = self.butler
        else:
            butler = daf_persistence.Butler(model_repository)
        model_arr = []
        weights_arr = []
        _f = 0
        while butler.datasetExists("dcrModel", dataId=self._build_model_dataId(band_name, subfilter=_f)):
            dcrModel = butler.get("dcrModel", dataId=self._build_model_dataId(band_name, subfilter=_f))
            model_arr.append(dcrModel.getMaskedImage().getImage().getArray())
            weights_arr.append(dcrModel.getMaskedImage().getVariance().getArray())
            _f += 1

        self.model = np.array(model_arr)
        self.weights = np.array(weights_arr)
        self.mask = dcrModel.getMaskedImage().getMask().getArray()

        meta = dcrModel.getMetadata()
        self.wcs = dcrModel.getWcs()
        self.n_step = len(model_arr)
        wavelength_step = meta.get("WAVESTEP")
        self.use_psf = meta.get("PSF_FLAG")
        self.y_size, self.x_size = dcrModel.getDimensions()
        self.pixel_scale = self.wcs.pixelScale().asArcseconds()
        exposure_time = dcrModel.getInfo().getCalib().getExptime()
        self.photoParams = PhotometricParameters(exptime=exposure_time, nexp=1, platescale=self.pixel_scale,
                                                 bandpass=band_name)
        self.bbox = dcrModel.getBBox()
        self.kernel_size = meta.get("KSUPPORT")
        self.bandpass = _load_bandpass(band_name=band_name, wavelength_step=wavelength_step, **kwargs)

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
        return(model)


class DcrCorrection(DcrModel):
    """Class that loads LSST calibrated exposures and produces airmass-matched template images."""

    def __init__(self, repository=".", obsid_range=None, band_name='g', wavelength_step=10,
                 n_step=None, elevation_min=40., use_psf=True, use_fft=False, debug_mode=False,
                 kernel_size=None, **kwargs):
        """Load images from the repository and set up parameters."""
        """
        @param repository: path to repository with the data. String, defaults to working directory
        @param obsid_range: obsid or range of obsids to process.
        """
        self.butler = daf_persistence.Butler(repository)
        dataId_gen = self._build_dataId(obsid_range, band_name)
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
        self.psf_size = calexp.getPsf().computeKernelImage().getArray().shape[0]
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
        if kernel_size is None:
            self.kernel_size = 2*self.dcr_max + 1
        else:
            self.kernel_size = kernel_size
        self.use_psf = use_psf
        self.use_fft = use_fft
        self.debug = debug_mode
        self.regularize = self._build_regularization(x_size=self.kernel_size,
                                                     y_size=self.kernel_size, n_step=self.n_step)
        self._calc_psf_model()

    @staticmethod
    def _build_regularization(x_size=None, y_size=None, n_step=None):
        """
        Regularization adapted from Nate Lust's DCR Demo iPython notebook.
        Calculate a difference matrix for regularization as if each wavelength were a pixel, then scale
        the difference matrix to the size of the number of pixels times number of wavelengths
        """
        reg_pix = None
        reg_lambda = None
        reg_lambda2 = None

        # baseReg = _difference(kernel_size)
        # reg_pix = np.zeros((n_step * kernel_size, n_step * kernel_size))

        # for i in range(n_step):
        #     reg_pix[i::n_step, i::n_step] = baseReg
        # reg_pix_x = np.zeros((n_step * x_size * y_size,
        #                       n_step * x_size * y_size - x_size))
        # for _ij in range(n_step * x_size * y_size - x_size):
        #     reg_pix_x[_ij, _ij] = 1
        #     reg_pix_x[_ij + x_size, _ij] = -1
        # reg_pix_x = np.append(reg_pix_x, -reg_pix_x, axis=1)

        # reg_pix_y = np.zeros((n_step * x_size * y_size,
        #                       n_step * x_size * y_size - 1))
        # for _ij in range(n_step * x_size * y_size - 1):
        #     reg_pix_y[_ij, _ij] = 1
        #     reg_pix_y[_ij + 1, _ij] = -1
        # reg_pix_y = np.append(reg_pix_y, -reg_pix_y, axis=1)
        # reg_pix = np.append(reg_pix_x, reg_pix_y, axis=1)

        # # Extra regularization that we force the SED to be smooth
        reg_lambda = np.zeros((n_step*x_size*y_size, (n_step - 1)*x_size*y_size))
        for _f in range(n_step - 1):
            for _ij in range(x_size*y_size):
                reg_lambda[_f*x_size*y_size + _ij, _f*x_size*y_size + _ij] = 1
                reg_lambda[(_f + 1)*x_size*y_size + _ij, _f*x_size*y_size + _ij] = -1
        reg_lambda = np.append(reg_lambda, -reg_lambda, axis=1)

        reg_lambda2 = np.zeros((n_step*x_size*y_size, (n_step - 2)*x_size*y_size))
        for _f in range(n_step - 2):
            for _ij in range(x_size * y_size):
                reg_lambda2[_f * x_size * y_size + _ij, _f * x_size * y_size + _ij] = -1
                reg_lambda2[(_f + 1) * x_size * y_size + _ij, _f * x_size * y_size + _ij] = 2
                reg_lambda2[(_f + 2) * x_size * y_size + _ij, _f * x_size * y_size + _ij] = -1
        if reg_lambda is None:
            reg_lambda = reg_lambda2
        elif reg_lambda2 is not None:
            reg_lambda = np.append(reg_lambda, reg_lambda2, axis=1)
        if reg_pix is None:
            return(reg_lambda)
        else:
            return(np.append(reg_pix, reg_lambda, axis=1))

    def _extract_image_vals(self, _j, _i, radius=None, fft=False):
        """Return all pixels within a radius of a given point as a 1D vector for each exposure."""
        img_arr = []
        for exp in self.exposures:
            img = exp.getMaskedImage().getImage().getArray()
            img = img[_j - radius: _j + radius + 1, _i - radius: _i + radius + 1]
            if fft:
                img = np.fft.fft2(np.fft.fftshift(img))
            img_arr.append(np.ravel(img))
        return(np.hstack(img_arr))

    def _insert_model_vals(self, _j, _i, vals, radius=None):
        if self.use_psf:
            psf_use = self.psf_avg
            self.model[:, _j - radius: _j + radius + 1, _i - radius: _i + radius + 1] += vals*psf_use
            self.weights[:, _j - radius: _j + radius + 1, _i - radius: _i + radius + 1] += psf_use
        else:
            psf_use = self.psf_avg
            self.model[:, _j - radius: _j + radius + 1, _i - radius: _i + radius + 1] += vals*psf_use
            self.weights[:, _j - radius: _j + radius + 1, _i - radius: _i + radius + 1] += psf_use

    def _calc_psf_model(self):
        psf_mat = []
        dcr_shift = []
        p0 = self.psf_size//2 - self.kernel_size//2
        p1 = p0 + self.kernel_size
        for _img, exp in enumerate(self.exposures):
            el = self.elevation_arr[_img]
            az = self.azimuth_arr[_img]

            dcr_genZ = _dcr_generator(self.bandpass, pixel_scale=self.pixel_scale, elevation=90., azimuth=az)
            dcr_gen = _dcr_generator(self.bandpass, pixel_scale=self.pixel_scale, elevation=el, azimuth=az)
            psf_mat.append(_calc_psf_kernel_full(exp, dcr_genZ, return_matrix=False,
                           x_size=self.psf_size, y_size=self.psf_size))
            dcr_shift.append(_calc_offset_phase(exp, dcr_gen, return_matrix=True,
                             x_size=self.psf_size, y_size=self.psf_size))
        psf_mat = np.sum(np.hstack(psf_mat), axis=0)
        dcr_shift = np.hstack(dcr_shift)
        regularize_psf = self._build_regularization(x_size=self.psf_size,
                                                    y_size=self.psf_size, n_step=self.n_step)
        regularize_dim = regularize_psf.shape
        vals_use = np.append(psf_mat, np.zeros(regularize_dim[1]))
        kernel_use = np.append(dcr_shift.T, regularize_psf.T, axis=0)
        psf_soln = positive_lstsq(kernel_use, vals_use)
        psf_model = np.reshape(psf_soln[0], (self.n_step, self.psf_size, self.psf_size))
        self.psf_model = psf_model[:, p0: p1, p0: p1]
        psf_vals = np.sum(psf_model, axis=0)/self.n_step
        self.psf_avg = psf_vals[p0: p1, p0: p1]
        psf_image = afwImage.ImageD(self.psf_size, self.psf_size)
        psf_image.getArray()[:, :] = psf_vals
        psfK = afwMath.FixedKernel(psf_image)
        self.psf = measAlg.KernelPsf(psfK)

    def build_model(self, use_full=True, use_regularization=True, use_only_detected=False):
        """Calculate a model of the true sky using the known DCR offset for each freq plane."""
        self.dcr_kernel = self._build_dcr_kernel(use_full=use_full)
        self.model = np.zeros((self.n_step, self.y_size, self.x_size))
        self.weights = np.zeros_like(self.model)
        pix_radius = self.kernel_size//2

        variance = np.zeros((self.y_size, self.x_size))
        for exp in self.exposures:
            variance += exp.getMaskedImage().getVariance().getArray()**2.
        self.variance = np.sqrt(variance) / self.n_images
        detected_bit = exp.getMaskedImage().getMask().getPlaneBitMask("DETECTED")
        print("Working on column", end="")
        for _j in range(self.y_size):
            if _j % 100 == 0:
                print("\n %i" % _j, end="")
            elif _j % 10 == 0:
                print("|", end="")
            else:
                print(".", end="")
            for _i in range(self.x_size):
                if self._edge_test(_j, _i):
                    continue
                # This option saves time by only performing the fit if the center pixel is masked as detected
                # Note that by gridding the results with the psf and maintaining a separate 'weights' array
                # this
                if use_only_detected:
                    if self.mask[_j, _i] & detected_bit == 0:
                        continue
                img_vals = self._extract_image_vals(_j, _i, radius=pix_radius, fft=self.use_fft)

                model_vals = self._solve_model(img_vals, use_regularization=use_regularization)
                self._insert_model_vals(_j, _i, model_vals, radius=pix_radius)
        print("\nFinished building model.")

    def _build_dcr_kernel(self, use_full=None):
        dcr_kernel = []
        for _img, exp in enumerate(self.exposures):
            el = self.elevation_arr[_img]
            az = self.azimuth_arr[_img]
            dcr_gen = _dcr_generator(self.bandpass, pixel_scale=self.pixel_scale, elevation=el, azimuth=az)

            if self.use_psf:
                if use_full:
                    dcr_kernel.append(_calc_psf_kernel_full(exp, dcr_gen, psf_img=self.psf_avg,
                                      x_size=self.kernel_size, y_size=self.kernel_size, return_matrix=True))
                else:
                    dcr_kernel.append(_calc_psf_kernel(exp, dcr_gen, psf_img=self.psf_avg,
                                      x_size=self.kernel_size, y_size=self.kernel_size, return_matrix=True))
            else:
                dcr_kernel.append(_calc_offset_phase(exp, dcr_gen, return_matrix=True,
                                  x_size=self.kernel_size, y_size=self.kernel_size))
        dcr_kernel = np.hstack(dcr_kernel)
        return(dcr_kernel)

    def _combine_masks(self):
        """Compute the bitwise OR of the input masks."""
        mask_arr = (exp.getMaskedImage().getMask().getArray() for exp in self.exposures)

        # Flags a pixel if ANY image is flagged there.
        self.mask = reduce(lambda m1, m2: np.bitwise_or(m1, m2), mask_arr)

    def _solve_model(self, img_vals, use_regularization=True):
        if self.use_fft:
            x_size = self.kernel_size
            y_size = self.kernel_size
            model_fft = np.zeros((self.n_step, y_size, x_size), dtype=img_vals.dtype)
            kernel_use = np.reshape(self.dcr_kernel, (self.n_step, y_size, x_size, self.n_images))
            img_use = np.reshape(img_vals, (y_size, x_size, self.n_images))
            for _j in range(y_size):
                for _i in range(x_size):
                    kernel_single = kernel_use[:, _j, _i, :]
                    img_single = img_use[_j, _i, :]
                    model_solution = np.linalg.lstsq(kernel_single.T, img_single)
                    model_fft[:, _j, _i] = model_solution[0]
            return(np.real(np.fft.ifftn(model_fft, axes=[1, 2])))
        else:
            x_size = self.kernel_size
            y_size = self.kernel_size
            if use_regularization:
                regularize_dim = self.regularize.shape
                vals_use = np.append(img_vals, np.zeros(regularize_dim[1]))
                kernel_use = np.append(self.dcr_kernel.T, self.regularize.T, axis=0)
                model_solution = positive_lstsq(kernel_use, vals_use)
            else:
                vals_use = img_vals
                kernel_use = self.dcr_kernel.T
                model_solution = positive_lstsq(kernel_use, vals_use)
            model_vals = model_solution[0]
            return(np.reshape(model_vals, (self.n_step, y_size, x_size)))


def _calc_offset_phase(exposure, dcr_gen, x_size=None, y_size=None, return_matrix=False):
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
            for _j in range(y_size):
                for _i in range(x_size):
                    _ij = _i + _j * x_size
                    shift_mat[_ij, :] = np.ravel(scipy_shift(kernel, (_j - y_size//2, _i - x_size//2),
                                                 mode='constant', cval=0.0))
            phase_arr.append(shift_mat)
        else:
            phase_arr.append(np.ravel(kernel))
    phase_arr = np.vstack(phase_arr)
    return(phase_arr)


def _calc_psf_kernel(exposure, dcr_gen, x_size=None, y_size=None, return_matrix=False, psf_img=None):

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
    return(psf_kernel_arr)


def _calc_psf_kernel_full(exposure, dcr_gen, x_size=None, y_size=None, return_matrix=False, psf_img=None):
    # psf_img is passed in but not used so that this function can be swapped in for _calc_psf_kernel
    # without having to change the keywords
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
    return(psf_kernel_arr)


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
    for _j in range(y_size):
        for _i in range(x_size):
            _ij = _i + _j * x_size
            sub_image = np.zeros_like(psf_img)
            for _n in range(n_substep):
                j_use = _j - y_size//2 + (dcr.dy.start * (n_substep - _n) + dcr.dy.end*_n)/n_substep
                i_use = _i - x_size//2 + (dcr.dx.start * (n_substep - _n) + dcr.dx.end*_n)/n_substep
                sub_image += scipy_shift(psf_img, (j_use, i_use), mode='constant', cval=0.0)
            psf_mat[_ij, :] = np.ravel(sub_image[x0:x1, y0:y1]) / n_substep
    return(psf_mat)


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


# NOTE: This function was modified from StarFast.py
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
    for wl_start, wl_end in _wavelength_iterator(bandpass, use_midpoint=False):
        # Note that refract_amp can be negative, since it's relative to the midpoint of the full band
        if use_midpoint:
            wl_mid = bandpass.calc_eff_wavelen(wavelength_min=wl_start, wavelength_max=wl_end)
            refract_mid = diff_refraction(wavelength=wl_mid, wavelength_ref=wavelength_midpoint,
                                          zenith_angle=zenith_angle, **kwargs)
            refract_mid *= 3600.0 / pixel_scale
            dx_mid = refract_mid * np.sin(np.radians(azimuth))
            dy_mid = refract_mid * np.cos(np.radians(azimuth))
            yield(dcr(dx=dx_mid, dy=dy_mid))
        else:
            refract_start = diff_refraction(wavelength=wl_start, wavelength_ref=wavelength_midpoint,
                                            zenith_angle=zenith_angle, **kwargs)
            refract_end = diff_refraction(wavelength=wl_end, wavelength_ref=wavelength_midpoint,
                                          zenith_angle=zenith_angle, **kwargs)
            refract_start *= 3600.0 / pixel_scale  # Refraction initially in degrees, convert to pixels.
            refract_end *= 3600.0 / pixel_scale
            dx = delta(start=refract_start * np.sin(np.radians(azimuth)),
                       end=refract_end * np.sin(np.radians(azimuth)))
            dy = delta(start=refract_start * np.cos(np.radians(azimuth)),
                       end=refract_end * np.cos(np.radians(azimuth)))
            yield(dcr(dx=dx, dy=dy))


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
    for _n in range(n_substep):
        loc = size//2. + (offset.start*(n_substep - _n) + offset.end*_n)/n_substep
        if loc % 1.0 == 0:
            kernel[int(loc)] += 1.0
        else:
            kernel += np.sin(pi*(pix - loc))/(pi*(pix - loc))
    return kernel/n_substep


class NonnegLstsqIterFit():
    def __init__():
        pass

# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-


class _BasicBandpass:
    """Dummy bandpass object for testing."""

    def __init__(self, band_name='g', wavelength_step=1):
        """Define the wavelength range and resolution for a given ugrizy band."""
        band_dict = {'u': (324.0, 395.0), 'g': (405.0, 552.0), 'r': (552.0, 691.0),
                     'i': (818.0, 921.0), 'z': (922.0, 997.0), 'y': (975.0, 1075.0)}
        band_range = band_dict[band_name]
        self.wavelen_min = band_range[0]
        self.wavelen_max = band_range[1]
        self.wavelen_step = wavelength_step

    def calc_eff_wavelen(self, wavelength_min=None, wavelength_max=None):
        """Mimic the calc_eff_wavelen method of the real bandpass class."""
        if wavelength_min is None:
            wavelength_min = self.wavelen_min
        if wavelength_max is None:
            wavelength_max = self.wavelen_max
        return((wavelength_min + wavelength_max) / 2.0)

    def calc_bandwidth(self):
        f0 = constants.speed_of_light / (self.wavelen_min * 1.0e-9)
        f1 = constants.speed_of_light / (self.wavelen_max * 1.0e-9)
        f_cen = constants.speed_of_light / (self.calc_eff_wavelen() * 1.0e-9)
        return(f_cen * 2.0 * (f0 - f1) / (f0 + f1))

    def getBandpass(self):
        """Mimic the getBandpass method of the real bandpass class."""
        wl_gen = _wavelength_iterator(self)
        wavelengths = [wl[0] for wl in wl_gen]
        wavelengths += [self.wavelen_max]
        bp_vals = [1] * len(wavelengths)
        return((wavelengths, bp_vals))


def _create_wcs(bbox=None, pixel_scale=None, ra=None, dec=None, sky_rotation=None):
    """Create a wcs (coordinate system)."""
    crval = afwCoord.IcrsCoord(ra * afwGeom.degrees, dec * afwGeom.degrees)
    crpix = afwGeom.Box2D(bbox).getCenter()
    cd1_1 = (pixel_scale * afwGeom.arcseconds * np.cos(np.radians(sky_rotation))).asDegrees()
    cd1_2 = (-pixel_scale * afwGeom.arcseconds * np.sin(np.radians(sky_rotation))).asDegrees()
    cd2_1 = (pixel_scale * afwGeom.arcseconds * np.sin(np.radians(sky_rotation))).asDegrees()
    cd2_2 = (pixel_scale * afwGeom.arcseconds * np.cos(np.radians(sky_rotation))).asDegrees()
    return(afwImage.makeWcs(crval, crpix, cd1_1, cd1_2, cd2_1, cd2_2))


class _BasicDcrModel(DcrModel):
    """Dummy DcrModel object for testing without a repository."""

    def __init__(self, size=None, kernel_size=5, n_step=3, band_name='g', exposure_time=30.,
                 pixel_scale=0.25, wavelength_step=10.0):
        seed = 5
        rand_gen = np.random
        rand_gen.seed(seed)
        self.butler = None
        self.use_fft = False
        self.use_psf = False

        bandpass = _BasicBandpass(band_name=band_name, wavelength_step=wavelength_step)
        if n_step is not None:
            wavelength_step = (bandpass.wavelen_max - bandpass.wavelen_min) / n_step
            bandpass = _BasicBandpass(band_name=band_name, wavelength_step=wavelength_step)
        else:
            n_step = int(np.ceil((bandpass.wavelen_max - bandpass.wavelen_min) / bandpass.wavelen_step))
        self.bandpass = bandpass
        self.model = rand_gen.random(size=(n_step, size, size))
        self.weights = np.ones((n_step, size, size))
        self.mask = np.zeros((size, size), dtype=np.int32)

        self.n_step = n_step
        self.y_size = size
        self.x_size = size
        self.pixel_scale = pixel_scale
        self.kernel_size = kernel_size
        self.photoParams = PhotometricParameters(exptime=exposure_time, nexp=1, platescale=pixel_scale,
                                                 bandpass=band_name)
        self.bbox = afwGeom.Box2I(afwGeom.Point2I(0, 0), afwGeom.ExtentI(size, size))
        self.wcs = _create_wcs(bbox=self.bbox, pixel_scale=pixel_scale, ra=0., dec=0., sky_rotation=0.)

        psf_vals = np.zeros((kernel_size, kernel_size))
        psf_vals[kernel_size//2 - 1: kernel_size//2 + 1,
                 kernel_size//2 - 1: kernel_size//2 + 1] = 0.5
        psf_vals[kernel_size//2, kernel_size//2] = 1.
        psf_image = afwImage.ImageD(kernel_size, kernel_size)
        psf_image.getArray()[:, :] = psf_vals
        psfK = afwMath.FixedKernel(psf_image)
        self.psf = measAlg.KernelPsf(psfK)

        self.psf_avg = psf_vals  # self.psf.computeKernelImage().getArray()


class DCRTestCase(lsst.utils.tests.TestCase):
    """Test the the calculations of Differential Chromatic Refraction."""

    def setUp(self):
        """Define parameters used by every test."""
        band_name = 'g'
        wavelength_step = 10.0
        self.pixel_scale = 0.25
        self.bandpass = _BasicBandpass(band_name=band_name, wavelength_step=wavelength_step)

    def tearDown(self):
        """Clean up."""
        del self.bandpass

    def test_dcr_generator(self):
        """Check that _dcr_generator returns a generator with n_step iterations, and (0,0) at zenith."""
        azimuth = 0.0
        elevation = 90.0
        zenith_dcr = 0.
        bp = self.bandpass
        dcr_gen = _dcr_generator(bp, pixel_scale=self.pixel_scale, elevation=elevation, azimuth=azimuth)
        n_step = int(np.ceil((bp.wavelen_max - bp.wavelen_min) / bp.wavelen_step))
        for _i in range(n_step):
            dcr = next(dcr_gen)
            self.assertFloatsEqual(dcr.dx.start, zenith_dcr)
            self.assertFloatsEqual(dcr.dx.end, zenith_dcr)
            self.assertFloatsEqual(dcr.dy.start, zenith_dcr)
            self.assertFloatsEqual(dcr.dy.end, zenith_dcr)
        # Also check that the generator is now exhausted
        with self.assertRaises(StopIteration):
            next(dcr_gen)

    def test_dcr_values(self):
        """Check DCR against pre-computed values."""
        azimuth = 0.0
        elevation = 50.0
        dcr_ref_vals = [(1.9847367904770623, 1.6467981843302726),
                        (1.6467981843302726, 1.3341803407311699),
                        (1.3341803407311699, 1.0443731947908652),
                        (1.0443731947908652, 0.77517513542339489),
                        (0.77517513542339489, 0.52464791969238367),
                        (0.52464791969238367, 0.29107920440002155),
                        (0.29107920440002155, 0.072951238300825172),
                        (0.072951238300825172, -0.13108543143740825),
                        (-0.13108543143740825, -0.3222341473268886),
                        (-0.3222341473268886, -0.50157094957602733),
                        (-0.50157094957602733, -0.6700605866796161),
                        (-0.6700605866796161, -0.8285701993878597),
                        (-0.8285701993878597, -0.97788106062563773),
                        (-0.97788106062563773, -1.1186986838806061),
                        (-1.1186986838806061, -1.2125619138571659)]
        bp = self.bandpass
        dcr_gen = _dcr_generator(bp, pixel_scale=self.pixel_scale, elevation=elevation,
                                 azimuth=azimuth, use_midpoint=False)
        n_step = int(np.ceil((bp.wavelen_max - bp.wavelen_min) / bp.wavelen_step))
        for _i in range(n_step):
            dcr = next(dcr_gen)
            self.assertFloatsEqual(dcr.dy.start, dcr_ref_vals[_i][0])
            self.assertFloatsEqual(dcr.dy.end, dcr_ref_vals[_i][1])


class BandpassTestCase(lsst.utils.tests.TestCase):
    """Tests of the interface to Bandpass from lsst.sims.photUtils."""

    def setUp(self):
        """Define parameters used by every test."""
        self.band_name = 'g'
        self.wavelength_step = 10
        self.bandpass = _load_bandpass(band_name=self.band_name, wavelength_step=self.wavelength_step)

    def test_step_bandpass(self):
        """Check that the bandpass has necessary methods, and those return the correct number of values."""
        bp = self.bandpass
        bp_wavelen, bandpass_vals = bp.getBandpass()
        n_step = int(np.ceil((bp.wavelen_max - bp.wavelen_min) / bp.wavelen_step))
        self.assertEqual(n_step + 1, len(bandpass_vals))


class DcrModelTestBase(lsst.utils.tests.TestCase):

    def setUp(self):
        band_name = 'g'
        n_step = 3
        pixel_scale = 0.25
        self.kernel_size = 5
        self.size = 20
        # NOTE that this array is randomly generated for each instance.
        self.array = np.random.random(size=(self.size, self.size))
        self.dcrModel = _BasicDcrModel(size=self.size, kernel_size=self.kernel_size, band_name=band_name,
                                       n_step=n_step, pixel_scale=pixel_scale)
        azimuth = 0.0
        elevation = 70.0
        self.dcr_gen = _dcr_generator(self.dcrModel.bandpass, pixel_scale=self.dcrModel.pixel_scale,
                                      elevation=elevation, azimuth=azimuth, use_midpoint=False)
        self.exposure = self.dcrModel._create_exposure(self.array, variance=None, elevation=elevation,
                                                       azimuth=azimuth)

    def tearDown(self):
        del self.dcrModel
        del self.exposure


class KernelTestCase(DcrModelTestBase):
    """Tests of the various kernels that incorporate dcr-based shifts."""

    def test_simple_phase_kernel(self):
        data_file = "test_data/simple_phase_kernel.npy"
        psf = self.exposure.getPsf()
        psf_size = psf.computeKernelImage().getArray().shape[0]
        phase_arr = _calc_offset_phase(self.exposure, self.dcr_gen,
                                       x_size=psf_size, y_size=psf_size, return_matrix=True)
        phase_arr_ref = np.load(data_file)
        self.assertFloatsEqual(phase_arr, phase_arr_ref)

    def test_simple_psf_kernel(self):
        data_file = "test_data/simple_psf_kernel.npy"
        psf = self.exposure.getPsf()
        psf_size = psf.computeKernelImage().getArray().shape[0]
        phase_arr = _calc_psf_kernel(self.exposure, self.dcr_gen, x_size=psf_size, y_size=psf_size,
                                     return_matrix=True, psf_img=self.dcrModel.psf_avg)
        phase_arr_ref = np.load(data_file)
        self.assertFloatsEqual(phase_arr, phase_arr_ref)

    def test_full_psf_kernel(self):
        data_file = "test_data/full_psf_kernel.npy"
        psf = self.exposure.getPsf()
        psf_size = psf.computeKernelImage().getArray().shape[0]
        phase_arr = _calc_psf_kernel_full(self.exposure, self.dcr_gen, x_size=psf_size, y_size=psf_size,
                                          return_matrix=True, psf_img=self.dcrModel.psf_avg)
        phase_arr_ref = np.load(data_file)
        self.assertFloatsEqual(phase_arr, phase_arr_ref)


class DcrModelTestCase(DcrModelTestBase):
    """Tests for the functions in the DcrModel class."""

    def test_dataId_single(self):
        id_ref = 100
        band_ref = 'g'
        ref_id = {'visit': id_ref, 'raft': '2,2', 'sensor': '1,1', 'filter': band_ref}
        dataId = self.dcrModel._build_dataId(id_ref, band_ref)
        self.assertEqual(ref_id, dataId[0])

    def test_dataId_range(self):
        id_ref = [100, 103]
        band_ref = 'g'
        ref_id = {'visit': id_ref, 'raft': '2,2', 'sensor': '1,1', 'filter': band_ref}
        dataId = self.dcrModel._build_dataId(id_ref, band_ref)
        for _i, _id in enumerate(range(id_ref[0], id_ref[1])):
            ref_id = {'visit': _id, 'raft': '2,2', 'sensor': '1,1', 'filter': band_ref}
            self.assertEqual(ref_id, dataId[_i])

    def test_extract_model_no_weights(self):
        # Make j and i different slightly so we can tell if the indices get swapped
        _i = self.size//2 + 1
        _j = self.size//2 - 1
        radius = self.kernel_size//2
        model_use = self.dcrModel.model
        model_vals = self.dcrModel._extract_model_vals(_j, _i, radius=radius, model=model_use)
        input_vals = [np.ravel(model_use[_f, _j - radius: _j + radius + 1, _i - radius: _i + radius + 1])
                      for _f in range(self.dcrModel.n_step)]
        self.assertFloatsEqual(np.hstack(input_vals), model_vals)

    def test_extract_model_with_weights(self):
        # Make j and i different slightly so we can tell if the indices get swapped
        _i = self.size//2 + 1
        _j = self.size//2 - 1
        radius = self.kernel_size//2
        model = self.dcrModel.model
        weight_scale = 2.2
        weights = self.dcrModel.weights * weight_scale
        weights[:, _j, _i] = 0.
        model_vals = self.dcrModel._extract_model_vals(_j, _i, radius=radius, model=model, weights=weights)
        input_arr = []
        for _f in range(self.dcrModel.n_step):
            input_vals = model[_f, _j - radius: _j + radius + 1, _i - radius: _i + radius + 1] / weight_scale
            input_vals[radius, radius] = 0.
            input_arr.append(np.ravel(input_vals))

        # input_vals = [model[_f, _j - radius: _j + radius + 1, _i - radius: _i + radius + 1] * weight_scale
        #               for _f in range(self.dcrModel.n_step)]
        # # input_vals = np.asarray(input_vals)
        # input_vals[:][radius, radius] = 0.
        # input_vals = [np.ravel(input_vals[_f]) for _f in range(self.dcrModel.n_step)]
        self.assertFloatsEqual(np.hstack(input_arr), model_vals)

    def test_apply_kernel(self):
        data_file = "test_data/dcr_kernel_vals.npy"
        i_use = self.size//2
        j_use = self.size//2
        radius = self.kernel_size//2
        model_vals = self.dcrModel._extract_model_vals(j_use, i_use, radius=radius, model=self.dcrModel.model,
                                                       weights=self.dcrModel.weights)
        dcr_kernel = _calc_offset_phase(self.exposure, self.dcr_gen, return_matrix=True,
                                        x_size=self.kernel_size, y_size=self.kernel_size)
        dcr_vals = self.dcrModel._apply_dcr_kernel(dcr_kernel, model_vals, x_size=self.kernel_size,
                                                   y_size=self.kernel_size)
        dcr_ref = np.load(data_file)
        self.assertFloatsEqual(dcr_vals, dcr_ref)


class PersistanceTestCase(DcrModelTestBase):
    """Tests that read and write exposures and dcr models to disk."""

    def test_create_exposure(self):
        self.assertFloatsEqual(self.exposure.getMaskedImage().getImage().getArray(), self.array)
        meta = self.exposure.getMetadata()
        # Check that the required metadata is present:
        try:
            meta.get("ZENITH")
        except Exception as e:
            raise e
        try:
            meta.get("AZIMUTH")
        except Exception as e:
            raise e

    def test_persist_dcr_model_roundtrip(self):
        # Instantiating the butler takes several seconds, so all butler-related tests are condensed into one.
        model_repository = "test_data"

        # First test that the model values are not changed from what is expected

        # The type "dcrModel" is read in as a 32 bit float, set in the lsst.obs.lsstSim.LsstSimMapper policy
        model = np.float32(self.dcrModel.model)
        self.dcrModel.export_model(model_repository=model_repository)
        dcrModel2 = DcrModel(model_repository=model_repository)
        # self.dcrModel.load_model(model_repository=model_repository)
        # Note that butler.get() reads the FITS file in 32 bit precision.
        self.assertFloatsEqual(model, dcrModel2.model)

        # Next, test that the required parameters have been restored
        param_ref = self.dcrModel.__dict__
        param_new = dcrModel2.__dict__
        for key in param_ref.keys():
            self.assertIn(key, param_new)


class MemoryTester(lsst.utils.tests.MemoryTestCase):
    pass


def setup_module(module):
    lsst.utils.tests.init()


if __name__ == "__main__":
    lsst.utils.tests.init()
    unittest.main()
