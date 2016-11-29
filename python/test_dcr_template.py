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
from lsst.sims.photUtils import PhotometricParameters
import unittest
import lsst.utils.tests
from .dcr_template import DcrModel
from .dcr_template import DcrCorrection


def basicBandpass(band_name='g', wavelength_step=1):
    """Return a dummy bandpass object for testing."""
    bandpass = DcrModel.load_bandpass(band_name=band_name, wavelength_step=wavelength_step,
                                      use_mirror=False, use_lens=False, use_atmos=False,
                                      use_filter=False, use_detector=False)
    return(bandpass)


class _BasicDcrModel(DcrModel):
    """Dummy DcrModel object for testing without a repository."""

    def __init__(self, size=None, kernel_size=5, n_step=3, band_name='g', exposure_time=30.,
                 pixel_scale=0.25, wavelength_step=None):
        """
        @param size  Number of pixels on a side of the image and model.
        @param kernel_size  size, in pixels, of the region surrounding each image pixel that DCR
                            shifts are calculated.
        @param n_step  Number of sub-filter wavelength planes to model. Optional if wavelength_step supplied.
        @param band_name  Common name of the filter used. For LSST, use u, g, r, i, z, or y
        @param exposure_time  Length of the exposure, in seconds. Needed only for exporting to FITS.
        @param pixel_scale  Plate scale of the images, in arcseconds
        @param wavelength_step  Overridden by n_step. Sub-filter width, in nm.
        """
        seed = 5
        rand_gen = np.random
        rand_gen.seed(seed)
        self.butler = None
        self.debug = False

        bandpass_init = basicBandpass(band_name=band_name, wavelength_step=None)
        wavelength_step = (bandpass_init.wavelen_max - bandpass_init.wavelen_min) / n_step
        self.bandpass = basicBandpass(band_name=band_name, wavelength_step=wavelength_step)
        self.model = [rand_gen.random(size=(size, size)) for f in range(n_step)]
        self.weights = np.ones((size, size))
        self.mask = np.zeros((size, size), dtype=np.int32)

        self.n_step = n_step
        self.y_size = size
        self.x_size = size
        self.pixel_scale = pixel_scale
        self.kernel_size = kernel_size
        self.photoParams = PhotometricParameters(exptime=exposure_time, nexp=1, platescale=pixel_scale,
                                                 bandpass=band_name)
        self.bbox = afwGeom.Box2I(afwGeom.Point2I(0, 0), afwGeom.ExtentI(size, size))
        self.wcs = DcrModel.create_wcs(bbox=self.bbox, pixel_scale=pixel_scale, ra=Angle(0.),
                                       dec=Angle(0.), sky_rotation=Angle(0.))

        psf_vals = np.zeros((kernel_size, kernel_size))
        psf_vals[kernel_size//2 - 1: kernel_size//2 + 1,
                 kernel_size//2 - 1: kernel_size//2 + 1] = 0.5
        psf_vals[kernel_size//2, kernel_size//2] = 1.
        psf_image = afwImage.ImageD(kernel_size, kernel_size)
        psf_image.getArray()[:, :] = psf_vals
        psfK = afwMath.FixedKernel(psf_image)
        self.psf = measAlg.KernelPsf(psfK)

        self.psf_avg = psf_vals  # self.psf.computeKernelImage().getArray()


class _BasicDcrCorrection(DcrCorrection):
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

        self.elevation_arr = []
        self.azimuth_arr = []
        self.airmass_arr = []
        for calexp in exposures:
            visitInfo = calexp.getInfo().getVisitInfo()
            self.elevation_arr.append(visitInfo.getBoresightAzAlt().getLatitude())
            self.azimuth_arr.append(visitInfo.getBoresightAzAlt().getLongitude())
            self.airmass_arr.append(visitInfo.getBoresightAirmass())
        self.exposures = exposures

        bandpass_init = basicBandpass(band_name=band_name, wavelength_step=None)
        wavelength_step = (bandpass_init.wavelen_max - bandpass_init.wavelen_min) / n_step
        self.bandpass = basicBandpass(band_name=band_name, wavelength_step=wavelength_step)
        self.n_step = n_step
        self.n_images = len(exposures)
        self.y_size, self.x_size = exposures[0].getDimensions()
        self.pixel_scale = calexp.getWcs().pixelScale().asArcseconds()
        exposure_time = visitInfo.getExposureTime()
        self.bbox = calexp.getBBox()
        self.wcs = calexp.getWcs()
        psf = calexp.getPsf().computeKernelImage().getArray()
        self.psf_size = psf.shape[0]
        self.psf_avg = psf

        self.kernel_size = kernel_size
        self.photoParams = PhotometricParameters(exptime=exposure_time, nexp=1, platescale=self.pixel_scale,
                                                 bandpass=band_name)
        # Calculate slightly worse DCR than maximum.
        elevation_min = np.min(self.elevation_arr) - Angle(np.radians(5.))
        dcr_test = DcrModel.dcr_generator(self.bandpass, pixel_scale=self.pixel_scale,
                                          elevation=elevation_min, azimuth=Angle(0.))
        self.dcr_max = int(np.ceil(np.max(dcr_test.next())) + 1)
        if kernel_size is None:
            self.kernel_size = 2*self.dcr_max + 1
        else:
            self.kernel_size = kernel_size


class DCRTestCase(lsst.utils.tests.TestCase):
    """Test the the calculations of Differential Chromatic Refraction."""

    def setUp(self):
        """Define parameters used by every test."""
        band_name = 'g'
        wavelength_step = 10.0  # nanometers
        self.pixel_scale = 0.25  # arcseconds/pixel
        self.bandpass = basicBandpass(band_name=band_name, wavelength_step=wavelength_step)

    def tearDown(self):
        """Clean up."""
        del self.bandpass

    def test_dcr_generator(self):
        """Check that _dcr_generator returns a generator with n_step iterations, and (0,0) at zenith."""
        azimuth = Angle(0.0)
        elevation = Angle(np.pi/2)
        zenith_dcr = 0.
        bp = self.bandpass
        dcr_gen = DcrModel.dcr_generator(bp, pixel_scale=self.pixel_scale,
                                         elevation=elevation, azimuth=azimuth)
        n_step = int(np.ceil((bp.wavelen_max - bp.wavelen_min) / bp.wavelen_step))
        for f in range(n_step):
            dcr = next(dcr_gen)
            self.assertFloatsAlmostEqual(dcr.dx.start, zenith_dcr)
            self.assertFloatsAlmostEqual(dcr.dx.end, zenith_dcr)
            self.assertFloatsAlmostEqual(dcr.dy.start, zenith_dcr)
            self.assertFloatsAlmostEqual(dcr.dy.end, zenith_dcr)
        # Also check that the generator is now exhausted
        with self.assertRaises(StopIteration):
            next(dcr_gen)

    def test_dcr_values(self):
        """Check DCR against pre-computed values."""
        azimuth = Angle(0.)
        elevation = Angle(np.radians(50.0))
        dcr_ref_vals = [(1.92315562164, 1.58521701549),
                        (1.58521701549, 1.27259917189),
                        (1.27259917189, 0.98279202595),
                        (0.98279202595, 0.713593966582),
                        (0.713593966582, 0.463066750851),
                        (0.463066750851, 0.229498035559),
                        (0.229498035559, 0.0113700694595),
                        (0.0113700694595, -0.192666600279),
                        (-0.192666600279, -0.383815316168),
                        (-0.383815316168, -0.563152118417),
                        (-0.563152118417, -0.731641755521),
                        (-0.731641755521, -0.890151368229),
                        (-0.890151368229, -1.03946222947),
                        (-1.03946222947, -1.18027985272),
                        (-1.18027985272, -1.2741430827)]
        bp = self.bandpass
        dcr_gen = DcrModel.dcr_generator(bp, pixel_scale=self.pixel_scale, elevation=elevation,
                                         azimuth=azimuth)
        n_step = int(np.ceil((bp.wavelen_max - bp.wavelen_min) / bp.wavelen_step))
        for f in range(n_step):
            dcr = next(dcr_gen)
            self.assertFloatsAlmostEqual(dcr.dy.start, dcr_ref_vals[f][0], rtol=1e-10)
            self.assertFloatsAlmostEqual(dcr.dy.end, dcr_ref_vals[f][1], rtol=1e-10)


class BandpassTestCase(lsst.utils.tests.TestCase):
    """Tests of the interface to Bandpass from lsst.sims.photUtils."""

    def setUp(self):
        """Define parameters used by every test."""
        self.band_name = 'g'
        self.wavelength_step = 10
        self.bandpass = DcrModel.load_bandpass(band_name=self.band_name, wavelength_step=self.wavelength_step)

    def test_step_bandpass(self):
        """Check that the bandpass has necessary methods, and those return the correct number of values."""
        bp = self.bandpass
        bp_wavelen, bandpass_vals = bp.getBandpass()
        n_step = int(np.ceil((bp.wavelen_max - bp.wavelen_min) / bp.wavelen_step))
        self.assertEqual(n_step + 1, len(bandpass_vals))


class DcrModelTestBase:

    def setUp(self):
        band_name = 'g'
        n_step = 3
        pixel_scale = 0.25
        self.kernel_size = 5
        self.size = 20
        # NOTE that this array is randomly generated with a new seed for each instance.
        self.array = np.random.random(size=(self.size, self.size))
        self.dcrModel = _BasicDcrModel(size=self.size, kernel_size=self.kernel_size, band_name=band_name,
                                       n_step=n_step, pixel_scale=pixel_scale)
        azimuth = Angle(np.radians(0.0))
        elevation = Angle(np.radians(70.0))
        self.dcr_gen = DcrModel.dcr_generator(self.dcrModel.bandpass, pixel_scale=self.dcrModel.pixel_scale,
                                              elevation=elevation, azimuth=azimuth, use_midpoint=False)
        self.exposure = self.dcrModel.create_exposure(self.array, variance=None, elevation=elevation,
                                                      azimuth=azimuth)

    def tearDown(self):
        del self.dcrModel
        del self.exposure
        del self.dcr_gen


class KernelTestCase(DcrModelTestBase, lsst.utils.tests.TestCase):
    """Tests of the various kernels that incorporate dcr-based shifts."""

    def test_simple_phase_kernel(self):
        """Compare the result of _calc_offset_phase to previously computed values."""
        data_file = "test_data/simple_phase_kernel.npy"
        psf = self.exposure.getPsf()
        psf_size = psf.computeKernelImage().getArray().shape[0]
        phase_arr = DcrModel.calc_offset_phase(exposure=self.exposure, dcr_gen=self.dcr_gen, x_size=psf_size,
                                               y_size=psf_size, return_matrix=True)
        phase_arr_ref = np.load(data_file)
        self.assertFloatsAlmostEqual(phase_arr, phase_arr_ref)

    def test_simple_psf_kernel(self):
        """Compare the result of _calc_psf_kernel to previously computed values."""
        data_file = "test_data/simple_psf_kernel.npy"
        psf = self.exposure.getPsf()
        psf_size = psf.computeKernelImage().getArray().shape[0]
        phase_arr = DcrModel.calc_psf_kernel(exposure=self.exposure, dcr_gen=self.dcr_gen,
                                             x_size=psf_size, y_size=psf_size, return_matrix=True,
                                             psf_img=self.dcrModel.psf_avg)
        phase_arr_ref = np.load(data_file)
        self.assertFloatsAlmostEqual(phase_arr, phase_arr_ref)

    def test_full_psf_kernel(self):
        """Compare the result of _calc_psf_kernel_full to previously computed values."""
        data_file = "test_data/full_psf_kernel.npy"
        psf = self.exposure.getPsf()
        psf_size = psf.computeKernelImage().getArray().shape[0]
        phase_arr = DcrModel.calc_psf_kernel_full(exposure=self.exposure, dcr_gen=self.dcr_gen,
                                                  x_size=psf_size, y_size=psf_size, return_matrix=True,
                                                  psf_img=self.dcrModel.psf_avg)
        phase_arr_ref = np.load(data_file)
        self.assertFloatsAlmostEqual(phase_arr, phase_arr_ref)


class DcrModelTestCase(DcrModelTestBase, lsst.utils.tests.TestCase):
    """Tests for the functions in the DcrModel class."""

    def test_dataId_single(self):
        id_ref = 100
        band_ref = 'g'
        ref_id = {'visit': id_ref, 'raft': '2,2', 'sensor': '1,1', 'filter': band_ref}
        dataId = DcrModel._build_dataId(id_ref, band_ref)
        self.assertEqual(ref_id, dataId[0])

    def test_dataId_list(self):
        id_ref = [100, 103]
        band_ref = 'g'
        ref_id = {'visit': id_ref, 'raft': '2,2', 'sensor': '1,1', 'filter': band_ref}
        dataId = DcrModel._build_dataId(id_ref, band_ref)
        for i, id in enumerate(id_ref):
            ref_id = {'visit': id, 'raft': '2,2', 'sensor': '1,1', 'filter': band_ref}
            self.assertEqual(ref_id, dataId[i])

    # def test_extract_model_no_weights(self):
    #     # Make j and i different slightly so we can tell if the indices get swapped
    #     i = self.size//2 + 1
    #     j = self.size//2 - 1
    #     radius = self.kernel_size//2
    #     model_vals = DcrModel._extract_model_vals(j, i, radius=radius, model_arr=self.dcrModel.model)
    #     input_vals = [np.ravel(model[j - radius: j + radius + 1, i - radius: i + radius + 1])
    #                   for model in self.dcrModel.model]
    #     self.assertFloatsAlmostEqual(np.hstack(input_vals), model_vals)

    def test_extract_model_with_weights(self):
        # Make j and i different slightly so we can tell if the indices get swapped
        i = self.size//2 + 1
        j = self.size//2 - 1
        radius = self.kernel_size//2
        weight_scale = 2.2
        inverse_weights = 1./(self.dcrModel.weights * weight_scale)
        inverse_weights[j, i] = 0.
        model_vals = DcrModel._extract_model_vals(j, i, radius=radius, model_arr=self.dcrModel.model,
                                                  inverse_weights=inverse_weights)
        input_arr = []
        for model in self.dcrModel.model:
            input_vals = model[j - radius: j + radius + 1, i - radius: i + radius + 1] / weight_scale
            input_vals[radius, radius] = 0.
            input_arr.append(np.ravel(input_vals))
        self.assertFloatsAlmostEqual(np.hstack(input_arr), model_vals)

    def test_apply_kernel(self):
        """Compare the result of _apply_dcr_kernel to previously computed values."""
        data_file = "test_data/dcr_kernel_vals.npy"
        i_use = self.size//2
        j_use = self.size//2
        radius = self.kernel_size//2
        inverse_weights = 1./self.dcrModel.weights
        model_vals = DcrModel._extract_model_vals(j_use, i_use, radius=radius, model_arr=self.dcrModel.model,
                                                  inverse_weights=inverse_weights)
        dcr_kernel = DcrModel.calc_offset_phase(self.exposure, self.dcr_gen, return_matrix=True,
                                                x_size=self.kernel_size, y_size=self.kernel_size)
        dcr_vals = DcrModel._apply_dcr_kernel(dcr_kernel, model_vals)
        dcr_ref = np.load(data_file)
        self.assertFloatsAlmostEqual(dcr_vals, dcr_ref)

    @unittest.expectedFailure
    def test_apply_even_kernel(self):
        """Only odd kernel sizes are currently supported, so this test is expected to fail for now."""
        data_file = "test_data/dcr_kernel_even_vals.npy"
        kernel_size = 6
        i_use = self.size//2
        j_use = self.size//2
        radius = kernel_size//2
        model_vals = DcrModel._extract_model_vals(j_use, i_use, radius=radius, model=self.dcrModel.model,
                                                  weights=self.dcrModel.weights)
        dcr_kernel = DcrModel.calc_offset_phase(self.exposure, self.dcr_gen, return_matrix=True,
                                                x_size=kernel_size, y_size=kernel_size)
        dcr_vals = DcrModel._apply_dcr_kernel(dcr_kernel, model_vals)
        dcr_ref = np.load(data_file)
        self.assertFloatsAlmostEqual(dcr_vals, dcr_ref)


class PersistanceTestCase(DcrModelTestBase, lsst.utils.tests.TestCase):
    """Tests that read and write exposures and dcr models to disk."""

    def test_create_exposure(self):
        self.assertFloatsAlmostEqual(self.exposure.getMaskedImage().getImage().getArray(), self.array)
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
        # Note that butler.get() reads the FITS file in 32 bit precision.
        self.assertFloatsAlmostEqual(model, dcrModel2.model)

        # Next, test that the required parameters have been restored
        param_ref = self.dcrModel.__dict__
        param_new = dcrModel2.__dict__
        for key in param_ref.keys():
            self.assertIn(key, param_new)

    def test_generate_template(self):
        """Compare the result of generate_templates_from_model to previously computed values."""
        data_file = "test_data/template.npy"
        elevation_arr = np.radians([50., 70., 85.])
        az = Angle(0.)
        # Note that self.array is randomly generated each call. That's okay, because the template should
        # depend only on the metadata.
        exposures = [self.dcrModel.create_exposure(self.array, variance=None, elevation=Angle(el), azimuth=az)
                     for el in elevation_arr]
        model_gen = self.dcrModel.generate_templates_from_model(exposures=exposures, kernel_size=5)
        model_test = [model for model in model_gen]
        model_ref = np.load(data_file)
        for m_i in range(len(model_test)):
            m_test = model_test[m_i].getMaskedImage().getImage().getArray()
            m_ref = model_ref[m_i].getMaskedImage().getImage().getArray()
            self.assertFloatsAlmostEqual(m_test, m_ref)


class DcrModelGenerationTestCase(lsst.utils.tests.TestCase):

    def setUp(self):
        band_name = 'g'
        self.n_step = 3
        self.n_images = 5
        pixel_scale = 0.25
        self.kernel_size = 5
        self.size = 20
        use_psf = False

        dcrModel = _BasicDcrModel(size=self.size, kernel_size=self.kernel_size, band_name=band_name,
                                  n_step=self.n_step, pixel_scale=pixel_scale)

        exposures = []
        self.ref_vals = []
        for i in range(self.n_images):
            # NOTE that this array is randomly generated for each instance.
            array = np.random.random(size=(self.size, self.size))*1000.
            self.ref_vals.append(array)
            el = Angle(np.radians(np.random.random()*50. + 40.))
            az = Angle(np.random.random()*2*np.pi)
            exposures.append(dcrModel.create_exposure(array, variance=None, elevation=el, azimuth=az))
        # Call the actual DcrCorrection class here, not just _BasicDcrCorrection
        self.dcrCorr = DcrCorrection(kernel_size=self.kernel_size, band_name=band_name,
                                     n_step=self.n_step, exposures=exposures, use_psf=use_psf)
        self.dcrCorr.calc_psf_model()

    def tearDown(self):
        del self.dcrCorr

    def test_extract_image(self):
        # Make j and i different slightly so we can tell if the indices get swapped
        i = self.size//2 + 1
        j = self.size//2 - 1
        radius = self.kernel_size//2
        image_arr = [exp.getMaskedImage().getImage().getArray() for exp in self.dcrCorr.exposures]
        mask_arr = [exp.getMaskedImage().getMask().getArray() for exp in self.dcrCorr.exposures]
        image_vals = self.dcrCorr._extract_image_vals(j, i, image_arr, mask=mask_arr, radius=radius)
        input_vals = [np.ravel(self.ref_vals[f][j - radius: j + radius + 1, i - radius: i + radius + 1])
                      for f in range(self.n_images)]
        self.assertFloatsAlmostEqual(np.hstack(input_vals), image_vals)

    def test_insert_model_vals(self):
        # Make j and i different slightly so we can tell if the indices get swapped
        i = self.size//2 + 1
        j = self.size//2 - 1
        radius = self.kernel_size//2
        test_vals = np.random.random(size=(self.n_step, self.kernel_size, self.kernel_size))
        model_ref = np.zeros((self.n_step, self.size, self.size))
        model = np.zeros((self.n_step, self.size, self.size))
        weights_ref = np.zeros((self.size, self.size))
        weights = np.zeros((self.size, self.size))
        psf_use = self.dcrCorr.psf_avg
        self.dcrCorr._insert_model_vals(j, i, test_vals, model, weights, radius=radius, kernel=psf_use)
        model_ref[:, j - radius: j + radius + 1, i - radius: i + radius + 1] += test_vals*psf_use
        weights_ref[j - radius: j + radius + 1, i - radius: i + radius + 1] += psf_use
        self.assertFloatsAlmostEqual(model_ref, model)
        self.assertFloatsAlmostEqual(weights_ref, weights)

    def test_calculate_psf(self):
        """Compare the result of _calc_psf_model (run in setUp) to previously computed values."""
        data_file = "test_data/calculate_psf.npy"
        psf_size = self.dcrCorr.psf.computeKernelImage().getArray().shape[0]
        p0 = psf_size//2 - self.kernel_size//2
        p1 = p0 + self.kernel_size
        psf_new = self.dcrCorr.psf.computeKernelImage().getArray()[p0: p1, p0: p1]
        psf_ref = np.load(data_file)
        self.assertFloatsAlmostEqual(psf_ref, psf_new)


class RegularizationTestCase(lsst.utils.tests.TestCase):
    def setUp(self):
        self.kernel_size = 5
        self.n_step = 3

    def test_spatial_regularization(self):
        """Compare the result of _build_regularization to previously computed values."""
        data_file = "test_data/spatial_regularization.npy"
        reg = DcrCorrection.build_regularization(x_size=self.kernel_size, y_size=self.kernel_size,
                                                 n_step=self.n_step, spatial_regularization=True)
        test_reg = np.load(data_file)
        self.assertFloatsAlmostEqual(reg, test_reg)

    def test_no_regularization(self):
        # All regularization is set to False by default
        reg = DcrCorrection.build_regularization(x_size=self.kernel_size, y_size=self.kernel_size,
                                                 n_step=self.n_step)
        self.assertIsNone(reg)

    def test_frequency_regularization(self):
        """Compare the result of _build_regularization to previously computed values."""
        data_file = "test_data/frequency_regularization.npy"
        reg = DcrCorrection.build_regularization(x_size=self.kernel_size, y_size=self.kernel_size,
                                                 n_step=self.n_step, frequency_regularization=True)
        test_reg = np.load(data_file)
        self.assertFloatsAlmostEqual(reg, test_reg)

    def test_frequency_derivative_regularization(self):
        """Compare the result of _build_regularization to previously computed values."""
        data_file = "test_data/frequency_derivative_regularization.npy"
        reg = DcrCorrection.build_regularization(x_size=self.kernel_size, y_size=self.kernel_size,
                                                 n_step=self.n_step, frequency_second_regularization=True)
        test_reg = np.load(data_file)
        self.assertFloatsAlmostEqual(reg, test_reg)

    def test_multiple_regularization(self):
        """Compare the result of _build_regularization to previously computed values."""
        spatial_file = "test_data/spatial_regularization.npy"
        freq_file = "test_data/frequency_regularization.npy"
        deriv_file = "test_data/frequency_derivative_regularization.npy"

        reg = DcrCorrection.build_regularization(x_size=self.kernel_size, y_size=self.kernel_size,
                                                 n_step=self.n_step, spatial_regularization=True,
                                                 frequency_regularization=True,
                                                 frequency_second_regularization=True)
        freq_reg = np.append(np.load(freq_file), np.load(deriv_file), axis=1)
        test_reg = np.append(np.load(spatial_file), freq_reg, axis=1)
        self.assertFloatsAlmostEqual(reg, test_reg)


class SolverTestCase(lsst.utils.tests.TestCase):
    def setUp(self):
        data_file = "test_data/exposures.npy"
        exposures = np.load(data_file)
        # Use _BasicDcrCorrection here to save execution time.
        self.dcrCorr = _BasicDcrCorrection(band_name='g', n_step=3, kernel_size=5, exposures=exposures)

    def tearDown(self):
        del self.dcrCorr

    def test_build_dcr_kernel_full(self):
        """Compare the result of _build_dcr_kernel to previously computed values."""
        data_file = "test_data/build_dcr_kernel_full_vals.npy"
        kernel = self.dcrCorr.build_dcr_kernel(self.dcrCorr.exposures, use_full=True, use_psf=True)
        kernel_ref = np.load(data_file)
        self.assertFloatsAlmostEqual(kernel, kernel_ref)

    def test_build_dcr_kernel(self):
        """Compare the result of _build_dcr_kernel to previously computed values."""
        data_file = "test_data/build_dcr_kernel_vals.npy"
        kernel = self.dcrCorr.build_dcr_kernel(self.dcrCorr.exposures, use_full=False, use_psf=False)
        kernel_ref = np.load(data_file)
        self.assertFloatsAlmostEqual(kernel, kernel_ref)

    def test_build_model(self):
        """Call build_model with as many options as possible turned off."""
        """Compare the result of build_model to previously computed values."""
        data_file = "test_data/build_model_vals.npy"
        self.dcrCorr.calc_psf_model()
        self.dcrCorr.build_model(use_full=False, use_regularization=False,
                                 use_only_detected=False, verbose=False, use_nonnegative=False)
        model_vals = self.dcrCorr.model
        model_ref = np.load(data_file)
        self.assertFloatsAlmostEqual(model_vals, model_ref)

    def test_solve_model_no_regularization(self):
        """Compare the result of _solve_model to previously computed values."""
        data_file = "test_data/solve_model_vals.npy"
        y_size, x_size = self.dcrCorr.exposures[0].getDimensions()
        kernel_size = self.dcrCorr.kernel_size
        n_step = self.dcrCorr.n_step
        pix_radius = kernel_size//2
        # Make j and i different slightly so we can tell if the indices get swapped
        i = x_size//2 + 1
        j = y_size//2 - 1
        image_arr = [exp.getMaskedImage().getImage().getArray() for exp in self.dcrCorr.exposures]
        mask_arr = [exp.getMaskedImage().getMask().getArray() for exp in self.dcrCorr.exposures]
        image_vals = self.dcrCorr._extract_image_vals(j, i, image_arr, mask=mask_arr, radius=pix_radius)
        dcr_kernel = self.dcrCorr.build_dcr_kernel(self.dcrCorr.exposures, use_full=False, use_psf=False)
        model_vals = self.dcrCorr.solve_model(kernel_size, n_step, dcr_kernel, image_vals,
                                              use_regularization=False)
        model_ref = np.load(data_file)
        self.assertFloatsAlmostEqual(model_vals, model_ref)

    def test_solve_model_with_regularization(self):
        """Compare the result of _solve_model to previously computed values."""
        data_file = "test_data/solve_model_reg_vals.npy"
        y_size, x_size = self.dcrCorr.exposures[0].getDimensions()
        kernel_size = self.dcrCorr.kernel_size
        n_step = self.dcrCorr.n_step
        pix_radius = kernel_size//2
        i = x_size//2 + 1
        j = y_size//2 - 1
        image_arr = [exp.getMaskedImage().getImage().getArray() for exp in self.dcrCorr.exposures]
        mask_arr = [exp.getMaskedImage().getMask().getArray() for exp in self.dcrCorr.exposures]
        image_vals = self.dcrCorr._extract_image_vals(j, i, image_arr, mask=mask_arr, radius=pix_radius)
        dcr_kernel = self.dcrCorr.build_dcr_kernel(self.dcrCorr.exposures, use_full=False, use_psf=False)
        self.dcrCorr.regularize = DcrCorrection.build_regularization(x_size=kernel_size, y_size=kernel_size,
                                                                     n_step=n_step,
                                                                     frequency_regularization=True)
        model_vals = self.dcrCorr.solve_model(kernel_size, n_step, dcr_kernel, image_vals,
                                              use_regularization=True)
        model_ref = np.load(data_file)
        self.assertFloatsAlmostEqual(model_vals, model_ref)


class MemoryTester(lsst.utils.tests.MemoryTestCase):
    pass


def setup_module(module):
    lsst.utils.tests.init()


if __name__ == "__main__":
    lsst.utils.tests.init()
    unittest.main()
