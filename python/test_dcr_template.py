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
from .dcr_template import solve_model
from .dcr_template import wrap_warpExposure


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
        self.instrument = 'lsstSim'

        bandpass_init = basicBandpass(band_name=band_name, wavelength_step=wavelength_step)
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
        self.psf_size = kernel_size
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

        self.psf_avg = psf_vals


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
        self.mask = None
        self.instrument = 'lsstSim'

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
        self.kernel_size = kernel_size
        exposure_time = visitInfo.getExposureTime()
        self.bbox = calexp.getBBox()
        self.wcs = calexp.getWcs()
        psf = calexp.getPsf().computeKernelImage().getArray()
        psf_size_test = psf.shape[0]
        if psf_size_test > 2*kernel_size:
            self.psf_size = 2*kernel_size
            p0 = psf_size_test//2 - self.psf_size//2
            p1 = p0 + self.psf_size
            self.psf_avg = psf[p0:p1, p0:p1]
        else:
            self.psf_size = psf_size_test
            self.psf_avg = psf

        self.kernel_size = kernel_size
        self.photoParams = PhotometricParameters(exptime=exposure_time, nexp=1, platescale=self.pixel_scale,
                                                 bandpass=band_name)
        # Calculate slightly worse DCR than maximum.
        elevation_min = np.min(self.elevation_arr) - Angle(np.radians(5.))
        dcr_test = DcrModel.dcr_generator(self.bandpass, pixel_scale=self.pixel_scale,
                                          elevation=elevation_min, rotation_angle=Angle(0.))
        self.dcr_max = int(np.ceil(np.max(dcr_test.next())) + 1)


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
        rotation_angle = Angle(0.0)
        elevation = Angle(np.pi/2)
        zenith_dcr = 0.
        bp = self.bandpass
        dcr_gen = DcrModel.dcr_generator(bp, pixel_scale=self.pixel_scale,
                                         elevation=elevation, rotation_angle=rotation_angle)
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
        rotation_angle = Angle(0.)
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
                                         rotation_angle=rotation_angle)
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
        rotation_angle = Angle(np.radians(0.0))
        azimuth = Angle(np.radians(0.0))
        elevation = Angle(np.radians(70.0))
        self.dcr_gen = DcrModel.dcr_generator(self.dcrModel.bandpass, pixel_scale=self.dcrModel.pixel_scale,
                                              elevation=elevation, rotation_angle=rotation_angle,
                                              use_midpoint=False)
        self.exposure = self.dcrModel.create_exposure(self.array, variance=None, elevation=elevation,
                                                      rotation_angle=rotation_angle.asDegrees(),
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
        phase_arr = DcrModel.calc_offset_phase(exposure=self.exposure, dcr_gen=self.dcr_gen, size=psf_size)
        # np.save(data_file, phase_arr)
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
        # np.save(data_file, model_test)
        model_ref = np.load(data_file)
        for m_i in range(len(model_test)):
            m_test = model_test[m_i].getMaskedImage().getImage().getArray()
            m_ref = model_ref[m_i].getMaskedImage().getImage().getArray()
            self.assertFloatsAlmostEqual(m_test, m_ref)

    def test_warp_exposure(self):
        wcs = self.exposure.getWcs()
        bbox = self.exposure.getBBox()
        wrap_warpExposure(self.exposure, wcs, bbox)
        array_warped = self.exposure.getMaskedImage().getImage().getArray()
        # For some reason the edges are all NAN.
        valid_inds = np.isfinite(array_warped)
        self.assertGreater(np.sum(valid_inds), (self.size/2)**2)
        array_ref = self.array[valid_inds]
        array_warped = array_warped[valid_inds]
        self.assertFloatsAlmostEqual(array_ref, array_warped, rtol=1e-7)


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


    def test_calculate_psf(self):
        """Compare the result of calc_psf_model (run in setUp) to previously computed values."""
        data_file = "test_data/calculate_psf.npy"
        psf_size = self.dcrCorr.psf.computeKernelImage().getArray().shape[0]
        p0 = psf_size//2 - self.kernel_size//2
        p1 = p0 + self.kernel_size
        psf_new = self.dcrCorr.psf.computeKernelImage().getArray()[p0: p1, p0: p1]
        # np.save(data_file, psf_new)
        psf_ref = np.load(data_file)
        self.assertFloatsAlmostEqual(psf_ref, psf_new)


class SolverTestCase(lsst.utils.tests.TestCase):
    def setUp(self):
        data_file = "test_data/exposures.npy"
        exposures = np.load(data_file)
        # Use _BasicDcrCorrection here to save execution time.
        self.dcrCorr = _BasicDcrCorrection(band_name='g', n_step=3, kernel_size=5, exposures=exposures)

    def tearDown(self):
        del self.dcrCorr

    def test_build_dcr_kernel(self):
        """Compare the result of _build_dcr_kernel to previously computed values."""
        data_file = "test_data/build_dcr_kernel_vals.npy"
        kernel = self.dcrCorr.build_dcr_kernel()
        # np.save(data_file, kernel)
        kernel_ref = np.load(data_file)
        self.assertFloatsAlmostEqual(kernel, kernel_ref)

    def test_build_model(self):
        """Call build_model with as many options as possible turned off."""
        """Compare the result of build_model to previously computed values."""
        data_file = "test_data/build_model_vals.npy"
        self.dcrCorr.calc_psf_model()
        self.dcrCorr.build_model(verbose=False)
        model_vals = self.dcrCorr.model
        # np.save(data_file, model_vals)
        model_ref = np.load(data_file)
        self.assertFloatsAlmostEqual(model_vals, model_ref)



class MemoryTester(lsst.utils.tests.MemoryTestCase):
    pass


def setup_module(module):
    lsst.utils.tests.init()


if __name__ == "__main__":
    lsst.utils.tests.init()
    unittest.main()
