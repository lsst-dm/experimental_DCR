"""Tests for functions and methods that are primarily used for calculating the DCR model."""
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
from itertools import izip
import unittest

import numpy as np

import lsst.daf.persistence as daf_persistence
import lsst.utils.tests

from python.dcr_utils import solve_model
from python.test_utils import BasicBuildDcrCoadd


class DcrCoaddGenerationTestCase(lsst.utils.tests.TestCase):
    """Tests for functions and methods that are primarily used for calculating the DCR model.

    Attributes
    ----------
    dcrCoadd : `BuildDcrCoadd`
        Class that loads LSST calibrated exposures and produces airmass-matched template images.
    n_images : int
        Number of input images used to calculate the model.
    n_step : int
        Number of sub-filter wavelength planes to model.
    ref_vals : list of np.ndarrays
        The list of reference values used to set up the input exposures.
    """

    @classmethod
    def setUpClass(self):
        """Define parameters used by every test."""
        band_name = 'g'
        self.n_step = 3
        self.n_images = 5

        butler = daf_persistence.Butler(inputs="./test_data/")
        exposures = []
        n_exp = 6
        for exp_i in range(n_exp):
            dataId = {'visit': exp_i, 'raft': '2,2', 'sensor': '1,1', 'filter': 'g'}
            exposures.append(butler.get("calexp", dataId=dataId))
        # Use BasicBuildDcrCoadd here to save execution time.
        self.dcrCoadd = BasicBuildDcrCoadd(band_name=band_name, n_step=self.n_step, exposures=exposures)
        self.ref_vals = []
        detected_bit = self.dcrCoadd.exposures[0].getMaskedImage().getMask().getPlaneBitMask('DETECTED')
        for exp in self.dcrCoadd.exposures:
            exp.getMaskedImage().getMask().getArray()[:, :] = detected_bit
            self.ref_vals.append(exp.getMaskedImage().getImage().getArray())

    @classmethod
    def tearDownClass(self):
        """Clean up."""
        del self.dcrCoadd

    def test_extract_image(self):
        """Test that the extracted values are the same as `ref_vals`."""
        for exp_i, exp in enumerate(self.dcrCoadd.exposures):
            image, inverse_var = self.dcrCoadd._extract_image(exp, calculate_dcr_gen=False)
            self.assertFloatsAlmostEqual(self.ref_vals[exp_i], image)

    def test_build_model(self):
        """Call build_model with as many options as possible turned off."""
        """Compare the result of build_model to previously computed values."""
        data_file = "test_data/build_model_vals.npy"
        self.dcrCoadd.build_model(verbose=False)
        model_vals = self.dcrCoadd.model
        # Uncomment the following code to over-write the reference data:
        # np.save(data_file, model_vals, allow_pickle=False)
        model_ref = np.load(data_file)
        for m_new, m_ref in izip(model_vals, model_ref):
            self.assertFloatsAlmostEqual(m_new, m_ref)

    def test_build_matched_template(self):
        """Compare the image and variance plane of the template to previously computed values."""
        data_file = "test_data/build_matched_template_vals.npy"
        exposure = self.dcrCoadd.exposures[0]
        self.dcrCoadd.build_model(verbose=False)
        template, variance = self.dcrCoadd.build_matched_template(exposure)
        # Uncomment the following code to over-write the reference data:
        # np.save(data_file, (template, variance), allow_pickle=False)
        template_ref, variance_ref = np.load(data_file)
        self.assertFloatsAlmostEqual(template, template_ref)
        self.assertFloatsAlmostEqual(variance, variance_ref)

    def test_calculate_new_model(self):
        """Compare the new model prediction for one iteration of the solver to previously-computed values."""
        data_file = "test_data/calculate_new_model_vals.npy"
        rand_gen = np.random
        random_seed = 5
        rand_gen.seed(random_seed)
        n_step = self.dcrCoadd.n_step
        x_size = self.dcrCoadd.x_size
        y_size = self.dcrCoadd.y_size
        last_solution = [rand_gen.random((y_size, x_size)) for f in range(n_step)]
        new_solution, inverse_var_arr = self.dcrCoadd._calculate_new_model(last_solution)
        # Uncomment the following code to over-write the reference data:
        # np.save(data_file, (new_solution, inverse_var_arr), allow_pickle=False)
        new_solution_ref, inverse_var_arr_ref = np.load(data_file)
        for soln_new, soln_ref in izip(new_solution, new_solution_ref):
            self.assertFloatsAlmostEqual(soln_new, soln_ref)
        for inv_var_new, inv_var_ref in izip(inverse_var_arr, inverse_var_arr_ref):
            self.assertFloatsAlmostEqual(inv_var_new, inv_var_ref)

    def test_clamp_model_solution(self):
        """Test that extreme solutions that changed by more than a factor of ``clamp`` are reduced."""
        clamp = 3.
        rand_gen = np.random
        rand_gen.seed(5)
        n_step = self.dcrCoadd.n_step
        x_size = self.dcrCoadd.x_size
        y_size = self.dcrCoadd.y_size
        last_solution = [rand_gen.random((y_size, x_size)) for f in range(n_step)]
        new_solution = [10.*(rand_gen.random((y_size, x_size)) - 0.5) for f in range(n_step)]
        ref_solution = copy.deepcopy(new_solution)
        self.dcrCoadd._clamp_model_solution(new_solution, last_solution, clamp)
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
        """Test that the DCR model convergence metric is calculated consistently."""
        model_file = "test_data/build_model_vals.npy"
        metric_ref = np.array([0.0326935559509, 0.0299110529484, 0.0312179049219,
                               0.0347479542217, 0.0391646272523, 0.0421978103681])
        model = np.load(model_file)
        metric = self.dcrCoadd.calc_model_metric(model=model)
        self.assertFloatsAlmostEqual(metric, metric_ref, rtol=1e-8, atol=1e-10)

    def test_build_model_convergence_failure(self):
        """Test that the iterative solver fails to converge if given a negative gain."""
        did_converge = self.dcrCoadd._build_model_subroutine(initial_solution=1, verbose=False, gain=-2,
                                                             test_convergence=True)
        self.assertFalse(did_converge)

    def test_calculate_psf(self):
        """Compare the result of calc_psf_model (run in setUp) to previously computed values."""
        data_file = "test_data/calculate_psf.npy"
        self.dcrCoadd.calc_psf_model()
        psf_new = self.dcrCoadd.psf.computeKernelImage().getArray()
        # Uncomment the following code to over-write the reference data:
        # np.save(data_file, psf_new, allow_pickle=False)
        psf_ref = np.load(data_file)
        self.assertFloatsAlmostEqual(psf_ref, psf_new)

    def test_build_dcr_kernel(self):
        """Compare the result of _build_dcr_kernel to previously computed values."""
        data_file = "test_data/build_dcr_kernel_vals.npy"
        kernel_size = 5
        kernel = self.dcrCoadd._build_dcr_kernel(kernel_size)
        # Uncomment the following code to over-write the reference data:
        # np.save(data_file, kernel, allow_pickle=False)
        kernel_ref = np.load(data_file)
        self.assertFloatsAlmostEqual(kernel, kernel_ref)

    def test_solve_model(self):
        """Compare the result of _solve_model to previously computed values."""
        data_file = "test_data/solve_model_vals.npy"
        y_size, x_size = self.dcrCoadd.exposures[0].getDimensions()
        kernel_size = 5
        n_step = self.dcrCoadd.n_step
        pix_radius = kernel_size//2
        # Make j and i different slightly so we can tell if the indices get swapped
        i = x_size//2 + 1
        j = y_size//2 - 1
        slice_inds = np.s_[j - pix_radius: j + pix_radius + 1, i - pix_radius: i + pix_radius + 1]
        image_arr = []
        for exp in self.dcrCoadd.exposures:
            image_arr.append(np.ravel(exp.getMaskedImage().getImage().getArray()[slice_inds]))
        image_vals = np.hstack(image_arr)
        dcr_kernel = self.dcrCoadd._build_dcr_kernel(kernel_size)
        model_vals_gen = solve_model(kernel_size, image_vals, n_step=n_step, kernel_dcr=dcr_kernel)
        model_arr = [model for model in model_vals_gen]
        # Uncomment the following code to over-write the reference data:
        # np.save(data_file, model_arr, allow_pickle=False)
        model_ref = np.load(data_file)
        for m_new, m_ref in izip(model_arr, model_ref):
            self.assertFloatsAlmostEqual(m_new, m_ref)


class MemoryTester(lsst.utils.tests.MemoryTestCase):
    """Test for memory leaks."""

    pass


def setup_module(module):
    """Setup helper for pytest."""
    lsst.utils.tests.init()


if __name__ == "__main__":
    lsst.utils.tests.init()
    unittest.main()
