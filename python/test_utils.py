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
import numpy as np

import lsst.afw.geom as afwGeom
from lsst.afw.geom import Angle
import lsst.afw.image as afwImage
import lsst.afw.math as afwMath
import lsst.meas.algorithms as measAlg
from .dcr_template import DcrModel
from .dcr_template import DcrCorrection
from .dcr_template import parallactic_angle
from .lsst_defaults import lsst_observatory


nanFloat = float("nan")
nanAngle = Angle(nanFloat)


def BasicBandpass(band_name='g', wavelength_step=1):
    """Return a dummy bandpass object for testing."""
    bandpass = DcrModel.load_bandpass(band_name=band_name, wavelength_step=wavelength_step,
                                      use_mirror=False, use_lens=False, use_atmos=False,
                                      use_filter=False, use_detector=False)
    return(bandpass)


class BasicDcrModel(DcrModel):
    """Dummy DcrModel object for testing without a repository."""

    def __init__(self, size=None, kernel_size=5, n_step=3, band_name='g', exposure_time=30.,
                 pixel_scale=Angle(afwGeom.arcsecToRad(0.25)), wavelength_step=None):
        """
        @param size  Number of pixels on a side of the image and model.
        @param kernel_size  size, in pixels, of the region surrounding each image pixel that DCR
                            shifts are calculated.
        @param n_step  Number of sub-filter wavelength planes to model. Optional if wavelength_step supplied.
        @param band_name  Common name of the filter used. For LSST, use u, g, r, i, z, or y
        @param exposure_time  Length of the exposure, in seconds. Needed only for exporting to FITS.
        @param pixel_scale  Plate scale of the images, as an Angle
        @param wavelength_step  Overridden by n_step. Sub-filter width, in nm.
        """
        seed = 5
        rand_gen = np.random
        rand_gen.seed(seed)
        self.butler = None
        self.debug = False
        self.instrument = 'lsstSim'
        self.detected_bit = 32

        bandpass_init = BasicBandpass(band_name=band_name, wavelength_step=wavelength_step)
        wavelength_step = (bandpass_init.wavelen_max - bandpass_init.wavelen_min) / n_step
        self.bandpass = BasicBandpass(band_name=band_name, wavelength_step=wavelength_step)
        self.model = [rand_gen.random(size=(size, size)) for f in range(n_step)]
        self.weights = np.ones((size, size))
        self.mask = np.zeros((size, size), dtype=np.int32)

        self.n_step = n_step
        self.y_size = size
        self.x_size = size
        self.pixel_scale = pixel_scale
        self.psf_size = kernel_size
        self.exposure_time = exposure_time
        self.filter_name = band_name
        self.observatory = lsst_observatory
        self.bbox = afwGeom.Box2I(afwGeom.Point2I(0, 0), afwGeom.ExtentI(size, size))
        self.wcs = DcrModel._create_wcs(bbox=self.bbox, pixel_scale=pixel_scale, ra=Angle(0.),
                                        dec=Angle(0.), sky_rotation=Angle(0.))

        psf_vals = np.zeros((kernel_size, kernel_size))
        psf_vals[kernel_size//2 - 1: kernel_size//2 + 1,
                 kernel_size//2 - 1: kernel_size//2 + 1] = 0.5
        psf_vals[kernel_size//2, kernel_size//2] = 1.
        psf_image = afwImage.ImageD(kernel_size, kernel_size)
        psf_image.getArray()[:, :] = psf_vals
        psfK = afwMath.FixedKernel(psf_image)
        self.psf = measAlg.KernelPsf(psfK)


class BasicDcrCorrection(DcrCorrection):
    """Dummy DcrCorrection object for testing without a repository."""

    def __init__(self, band_name='g', n_step=3, kernel_size=5, exposures=None):
        """
        @param band_name  Common name of the filter used. For LSST, use u, g, r, i, z, or y.
        @param n_step  Number of sub-filter wavelength planes to model. Optional if wavelength_step supplied.
        @param kernel_size  size, in pixels, of the region surrounding each image pixel that DCR
                            shifts are calculated.
        @param exposures  A list of LSST exposures to use as input to the DCR calculation.
        """
        self.butler = None
        self.debug = False
        self.mask = None
        self.model_base = None
        self.instrument = 'lsstSim'
        self.detected_bit = 32
        self.filter_name = band_name

        self.exposures = exposures

        bandpass_init = BasicBandpass(band_name=band_name, wavelength_step=None)
        wavelength_step = (bandpass_init.wavelen_max - bandpass_init.wavelen_min) / n_step
        self.bandpass = BasicBandpass(band_name=band_name, wavelength_step=wavelength_step)
        self.n_step = n_step
        self.n_images = len(exposures)
        self.y_size, self.x_size = exposures[0].getDimensions()
        self.pixel_scale = exposures[0].getWcs().pixelScale()
        self.exposure_time = exposures[0].getInfo().getVisitInfo().getExposureTime()
        self.bbox = exposures[0].getBBox()
        self.wcs = exposures[0].getWcs()
        self.observatory = exposures[0].getInfo().getVisitInfo().getObservatory()
        psf = exposures[0].getPsf().computeKernelImage().getArray()
        self.psf_size = psf.shape[0]


class DcrModelTestBase:

    def setUp(self):
        band_name = 'g'
        n_step = 3
        pixel_scale = Angle(afwGeom.arcsecToRad(0.25))
        kernel_size = 5
        self.size = 20
        lsst_lat = lsst_observatory.getLatitude()
        # NOTE that this array is randomly generated with a new seed for each instance.
        self.array = np.random.random(size=(self.size, self.size))
        self.dcrModel = BasicDcrModel(size=self.size, kernel_size=kernel_size, band_name=band_name,
                                      n_step=n_step, pixel_scale=pixel_scale)
        dec = self.dcrModel.wcs.getSkyOrigin().getLatitude()
        ra = self.dcrModel.wcs.getSkyOrigin().getLongitude()
        self.azimuth = Angle(np.radians(140.0))
        self.elevation = Angle(np.radians(50.0))
        ha_term1 = np.sin(self.elevation.asRadians())
        ha_term2 = np.sin(dec.asRadians())*np.sin(lsst_lat.asRadians())
        ha_term3 = np.cos(dec.asRadians())*np.cos(lsst_lat.asRadians())
        hour_angle = Angle(np.arccos((ha_term1 - ha_term2) / ha_term3))
        p_angle = parallactic_angle(hour_angle, dec, lsst_lat)
        self.rotation_angle = Angle(p_angle)
        self.dcr_gen = BasicDcrModel._dcr_generator(self.dcrModel.bandpass,
                                                    pixel_scale=self.dcrModel.pixel_scale,
                                                    elevation=self.elevation,
                                                    rotation_angle=self.rotation_angle,
                                                    use_midpoint=False)
        self.exposure = self.dcrModel.create_exposure(self.array, variance=None, elevation=self.elevation,
                                                      azimuth=self.azimuth,
                                                      boresightRotAngle=self.rotation_angle, dec=dec, ra=ra)

    def tearDown(self):
        del self.dcrModel
        del self.exposure
        del self.dcr_gen