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

import unittest
import lsst.utils.tests
from python.dcr_template import DcrCorrection
from python.dcr_template import solve_model
from python.test_utils import BasicDcrCorrection


class SolverTestCase(lsst.utils.tests.TestCase):
    def setUp(self):
        data_file = "test_data/exposures.npy"
        exposures = np.load(data_file)
        self.kernel_size = 5
        # Use _BasicDcrCorrection here to save execution time.
        self.dcrCorr = BasicDcrCorrection(band_name='g', n_step=3, kernel_size=self.kernel_size,
                                          exposures=exposures)
        for exp in self.dcrCorr.exposures:
            exp.getMaskedImage().getMask().getArray()[:, :] = self.dcrCorr.detected_bit

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
        use_variance = True
        rand_gen = np.random
        rand_gen.seed(5)
        n_step = self.dcrCorr.n_step
        x_size = self.dcrCorr.x_size
        y_size = self.dcrCorr.y_size
        last_solution = [rand_gen.random((y_size, x_size)) for f in range(n_step)]
        exp_cut = [False for exp_i in range(self.dcrCorr.n_images)]
        new_solution, inverse_var_arr = self.dcrCorr._calculate_new_model(last_solution, exp_cut,
                                                                          use_variance)
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
        metric_ref = np.array([0.0326935547581, 0.0299110561613, 0.0312179049219,
                               0.0347479538541, 0.0391646266206, 0.0421978090644])
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
