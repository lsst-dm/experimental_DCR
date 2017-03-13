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
import copy
import numpy as np

import lsst.afw.geom as afwGeom
from lsst.afw.geom import Angle
import lsst.afw.image as afwImage
import lsst.afw.math as afwMath
import lsst.meas.algorithms as measAlg
import unittest
import lsst.utils.tests
from .dcr_template import DcrModel
from .dcr_template import DcrCorrection
from .dcr_template import solve_model
from .dcr_template import wrap_warpExposure
from .dcr_template import calculate_rotation_angle
from .dcr_template import parallactic_angle
from .lsst_defaults import lsst_observatory


nanFloat = float("nan")
nanAngle = Angle(nanFloat)


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
        self.instrument = 'lsstSim'
        self.detected_bit = 32

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
        self.model_base = None
        self.instrument = 'lsstSim'
        self.detected_bit = 32
        self.filter_name = band_name

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
        # self.kernel_size = kernel_size
        self.exposure_time = visitInfo.getExposureTime()
        self.bbox = calexp.getBBox()
        self.wcs = calexp.getWcs()
        psf = calexp.getPsf().computeKernelImage().getArray()
        self.observatory = exposures[0].getInfo().getVisitInfo().getObservatory()
        self.psf_size = psf.shape[0]


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
        dcr_gen = DcrModel._dcr_generator(bp, pixel_scale=self.pixel_scale,
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
        dcr_gen = DcrModel._dcr_generator(bp, pixel_scale=self.pixel_scale, elevation=elevation,
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
        kernel_size = 5
        self.size = 20
        lsst_lat = lsst_observatory.getLatitude()
        # NOTE that this array is randomly generated with a new seed for each instance.
        self.array = np.random.random(size=(self.size, self.size))
        self.dcrModel = _BasicDcrModel(size=self.size, kernel_size=kernel_size, band_name=band_name,
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
        self.dcr_gen = DcrModel._dcr_generator(self.dcrModel.bandpass, pixel_scale=self.dcrModel.pixel_scale,
                                               elevation=self.elevation, rotation_angle=self.rotation_angle,
                                               use_midpoint=False)
        self.exposure = self.dcrModel.create_exposure(self.array, variance=None, elevation=self.elevation,
                                                      azimuth=self.azimuth,
                                                      boresightRotAngle=self.rotation_angle, dec=dec, ra=ra)

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
        phase_arr = DcrModel._calc_offset_phase(exposure=self.exposure, dcr_gen=self.dcr_gen, size=psf_size)
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

    def test_model_dataId(self):
        subfilter = 1
        band_ref = 'g'
        ref_id = {'filter': band_ref, 'tract': 0, 'patch': '0', 'subfilter': subfilter}
        dataId = DcrModel._build_model_dataId(band_ref, subfilter=subfilter)
        self.assertEqual(ref_id, dataId)

    def test_fetch_metadata(self):
        az = DcrModel._fetch_metadata(self.exposure.getMetadata(), "AZIMUTH")
        self.assertFloatsAlmostEqual(az, self.azimuth.asDegrees())

    def test_fetch_missing_metadata(self):
        bogus_ref = 3.7
        bogus = DcrModel._fetch_metadata(self.exposure.getMetadata(), "BOGUS", default_value=bogus_ref)
        self.assertFloatsAlmostEqual(bogus, bogus_ref)

    def test_generate_template(self):
        """Compare the result of generate_templates_from_model to previously computed values."""
        data_file = "test_data/template.npy"
        elevation_arr = np.radians([50., 70., 85.])
        az = Angle(0.)
        # Note that self.array is randomly generated each call. That's okay, because the template should
        # depend only on the metadata.
        exposures = [self.dcrModel.create_exposure(self.array, variance=None, elevation=Angle(el), azimuth=az)
                     for el in elevation_arr]
        model_gen = self.dcrModel.generate_templates_from_model(exposures=exposures)
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

    def test_rotation_angle(self):
        rotation_angle = calculate_rotation_angle(self.exposure)
        self.assertFloatsAlmostEqual(self.rotation_angle.asDegrees(), rotation_angle.asDegrees())


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


class DcrModelGenerationTestCase(lsst.utils.tests.TestCase):

    def setUp(self):
        band_name = 'g'
        self.n_step = 3
        self.n_images = 5
        pixel_scale = 0.25
        kernel_size = 5
        self.size = 20

        dcrModel = _BasicDcrModel(size=self.size, kernel_size=kernel_size, band_name=band_name,
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
        self.dcrCorr = DcrCorrection(band_name=band_name, n_step=self.n_step, exposures=exposures)

    def tearDown(self):
        del self.dcrCorr

    def test_calculate_psf(self):
        """Compare the result of calc_psf_model (run in setUp) to previously computed values."""
        data_file = "test_data/calculate_psf.npy"
        self.dcrCorr.calc_psf_model()
        psf_new = self.dcrCorr.psf.computeKernelImage().getArray()
        # np.save(data_file, psf_new)
        psf_ref = np.load(data_file)
        self.assertFloatsAlmostEqual(psf_ref, psf_new)

    def test_extract_image(self):
        for exp_i, exp in enumerate(self.dcrCorr.exposures):
            image, inverse_var = self.dcrCorr._extract_image(exp, calculate_dcr_gen=False)
            self.assertFloatsAlmostEqual(self.ref_vals[exp_i], image)


class SolverTestCase(lsst.utils.tests.TestCase):
    def setUp(self):
        data_file = "test_data/exposures.npy"
        exposures = np.load(data_file)
        self.kernel_size = 5
        # Use _BasicDcrCorrection here to save execution time.
        self.dcrCorr = _BasicDcrCorrection(band_name='g', n_step=3, kernel_size=self.kernel_size,
                                           exposures=exposures)

    def tearDown(self):
        del self.dcrCorr

    def test_build_dcr_kernel(self):
        """Compare the result of _build_dcr_kernel to previously computed values."""
        data_file = "test_data/build_dcr_kernel_vals.npy"
        kernel = self.dcrCorr._build_dcr_kernel(self.kernel_size)
        # np.save(data_file, kernel)
        kernel_ref = np.load(data_file)
        self.assertFloatsAlmostEqual(kernel, kernel_ref)

    def test_build_model(self):
        """Call build_model with as many options as possible turned off."""
        """Compare the result of build_model to previously computed values."""
        data_file = "test_data/build_model_vals.npy"
        self.dcrCorr.build_model(verbose=False)
        model_vals = self.dcrCorr.model
        # np.save(data_file, model_vals)
        model_ref = np.load(data_file)
        for f, model in enumerate(model_vals):
            self.assertFloatsAlmostEqual(model, model_ref[f])

    def test_build_matched_template(self):
        data_file = "test_data/build_matched_template_vals.npy"
        exposure = self.dcrCorr.exposures[0]
        self.dcrCorr.build_model(verbose=False)
        template, variance = self.dcrCorr.build_matched_template(exposure)
        # np.save(data_file, (template, variance))
        template_ref, variance_ref = np.load(data_file)
        self.assertFloatsAlmostEqual(template, template_ref)
        self.assertFloatsAlmostEqual(variance, variance_ref)

    def test_calculate_new_model(self):
        data_file = "test_data/calculate_new_model_vals.npy"
        rand_gen = np.random
        rand_gen.seed(5)
        n_step = self.dcrCorr.n_step
        x_size = self.dcrCorr.x_size
        y_size = self.dcrCorr.y_size
        last_solution = [rand_gen.random((y_size, x_size)) for f in range(n_step)]
        exp_cut = [False for exp_i in range(self.dcrCorr.n_images)]
        new_solution, inverse_var_arr = self.dcrCorr._calculate_new_model(last_solution, exp_cut)
        # np.save(data_file, (new_solution, inverse_var_arr))
        new_solution_ref, inverse_var_arr_ref = np.load(data_file)
        for f, soln in enumerate(new_solution):
            self.assertFloatsAlmostEqual(soln, new_solution_ref[f])
        for f, var in enumerate(inverse_var_arr):
            self.assertFloatsAlmostEqual(var, inverse_var_arr_ref[f])

    def test_clamp_model_solution(self):
        clamp = 3.
        rand_gen = np.random
        rand_gen.seed(5)
        n_step = self.dcrCorr.n_step
        x_size = self.dcrCorr.x_size
        y_size = self.dcrCorr.y_size
        last_solution = [rand_gen.random((y_size, x_size)) for f in range(n_step)]
        new_solution = [10.*(rand_gen.random((y_size, x_size)) - 0.5) for f in range(n_step)]
        ref_solution = copy.deepcopy(new_solution)
        DcrCorrection._clamp_model_solution(new_solution, last_solution, clamp)
        ref_max = np.max(ref_solution)
        ref_min = np.min(ref_solution)
        last_max = np.max(last_solution)
        last_min = np.min(last_solution)
        clamp_max = np.max(new_solution)
        clamp_min = np.min(new_solution)
        self.assertLessEqual(ref_min, clamp_min)
        self.assertGreaterEqual(ref_max, clamp_max)
        self.assertGreaterEqual(clamp_min, last_min/clamp)
        self.assertLessEqual(clamp_max, last_max*clamp)

    def test_calc_model_metric(self):
        model_file = "test_data/build_model_vals.npy"
        metric_ref = np.array([127.962191286, 118.032111041, 165.288144737,
                               204.071081167, 234.698022211, 247.949131707])
        model = np.load(model_file)
        metric = self.dcrCorr.calc_model_metric(model=model)
        self.assertFloatsAlmostEqual(metric, metric_ref, rtol=1e-8, atol=1e-10)

    def test_build_model_convergence_failure(self):
        """Test that the iterative solver fails to converge if given a negative gain."""
        converge_error = self.dcrCorr._build_model_subroutine(initial_solution=1, verbose=False, gain=-2,
                                                              test_convergence=True)
        self.assertTrue(converge_error)

    def test_solve_model(self):
        """Compare the result of _solve_model to previously computed values."""
        data_file = "test_data/solve_model_vals.npy"
        y_size, x_size = self.dcrCorr.exposures[0].getDimensions()
        kernel_size = self.kernel_size
        n_step = self.dcrCorr.n_step
        pix_radius = kernel_size//2
        # Make j and i different slightly so we can tell if the indices get swapped
        i = x_size//2 + 1
        j = y_size//2 - 1
        slice_inds = np.s_[j - pix_radius: j + pix_radius + 1, i - pix_radius: i + pix_radius + 1]
        image_arr = []
        for exp in self.dcrCorr.exposures:
            image_arr.append(np.ravel(exp.getMaskedImage().getImage().getArray()[slice_inds]))
        image_vals = np.hstack(image_arr)
        dcr_kernel = self.dcrCorr._build_dcr_kernel(kernel_size)
        model_vals_gen = solve_model(kernel_size, image_vals, n_step=n_step, kernel_dcr=dcr_kernel)
        model_arr = [model for model in model_vals_gen]
        # np.save(data_file, model_arr)
        model_ref = np.load(data_file)
        for f, model in enumerate(model_arr):
            self.assertFloatsAlmostEqual(model, model_ref[f])


class MemoryTester(lsst.utils.tests.MemoryTestCase):
    pass


def setup_module(module):
    lsst.utils.tests.init()


if __name__ == "__main__":
    lsst.utils.tests.init()
    unittest.main()
