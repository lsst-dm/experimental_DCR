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
import numpy as np
import unittest

import lsst.utils.tests

from python.test_utils import BasicBuildDcrModel


class DcrModelGenerationTestCase(lsst.utils.tests.TestCase):
    """Tests for functions and methods that are primarily used for calculating the DCR model.

    Attributes
    ----------
    dcrModel : `BuildDcrModel`
        Class that loads LSST calibrated exposures and produces airmass-matched template images.
    n_images : int
        Number of input images used to calculate the model.
    n_step : int
        Number of sub-filter wavelength planes to model.
    ref_vals : list of np.ndarrays
        The list of reference values used to set up the input exposures.
    """

    def setUp(self):
        """Define parameters used by every test."""
        band_name = 'g'
        self.n_step = 3
        self.n_images = 5

        data_file = "test_data/exposures.npy"
        exposures = np.load(data_file)
        # Use BasicBuildDcrModel here to save execution time.
        self.dcrModel = BasicBuildDcrModel(band_name=band_name, n_step=self.n_step, exposures=exposures)
        self.ref_vals = []
        detected_bit = self.dcrModel.exposures[0].getMaskedImage().getMask().getPlaneBitMask('DETECTED')
        for exp in self.dcrModel.exposures:
            exp.getMaskedImage().getMask().getArray()[:, :] = detected_bit
            self.ref_vals.append(exp.getMaskedImage().getImage().getArray())

    def tearDown(self):
        """Clean up."""
        del self.dcrModel

    def test_extract_image(self):
        """Test that the extracted values are the same as `ref_vals`."""
        for exp_i, exp in enumerate(self.dcrModel.exposures):
            image, inverse_var = self.dcrModel._extract_image(exp, calculate_dcr_gen=False)
            self.assertFloatsAlmostEqual(self.ref_vals[exp_i], image)

    def test_build_model(self):
        """Call build_model with as many options as possible turned off."""
        """Compare the result of build_model to previously computed values."""
        data_file = "test_data/build_model_vals.npy"
        self.dcrModel.build_model(verbose=False)
        model_vals = self.dcrModel.model
        # np.save(data_file, model_vals)
        model_ref = np.load(data_file)
        for f, model in enumerate(model_vals):
            self.assertFloatsAlmostEqual(model, model_ref[f])

    def test_build_matched_template(self):
        """Compare the image and variance plane of the template to previously computed values."""
        data_file = "test_data/build_matched_template_vals.npy"
        exposure = self.dcrModel.exposures[0]
        self.dcrModel.build_model(verbose=False)
        template, variance = self.dcrModel.build_matched_template(exposure)
        # np.save(data_file, (template, variance))
        template_ref, variance_ref = np.load(data_file)
        self.assertFloatsAlmostEqual(template, template_ref)
        self.assertFloatsAlmostEqual(variance, variance_ref)

    def test_calculate_new_model(self):
        """Compare the new model prediction for one iteration of the solver to previously-computed values."""
        data_file = "test_data/calculate_new_model_vals.npy"
        use_variance = True
        rand_gen = np.random
        random_seed = 5
        rand_gen.seed(random_seed)
        n_step = self.dcrModel.n_step
        x_size = self.dcrModel.x_size
        y_size = self.dcrModel.y_size
        last_solution = [rand_gen.random((y_size, x_size)) for f in range(n_step)]
        exp_cut = [False for exp_i in range(self.dcrModel.n_images)]
        new_solution, inverse_var_arr = self.dcrModel._calculate_new_model(last_solution, exp_cut,
                                                                           use_variance)
        # np.save(data_file, (new_solution, inverse_var_arr))
        new_solution_ref, inverse_var_arr_ref = np.load(data_file)
        for f, soln in enumerate(new_solution):
            self.assertFloatsAlmostEqual(soln, new_solution_ref[f])
        for f, var in enumerate(inverse_var_arr):
            self.assertFloatsAlmostEqual(var, inverse_var_arr_ref[f])

    def test_clamp_model_solution(self):
        """Test that extreme solutions are reduced."""
        clamp = 3.
        rand_gen = np.random
        rand_gen.seed(5)
        n_step = self.dcrModel.n_step
        x_size = self.dcrModel.x_size
        y_size = self.dcrModel.y_size
        last_solution = [rand_gen.random((y_size, x_size)) for f in range(n_step)]
        new_solution = [10.*(rand_gen.random((y_size, x_size)) - 0.5) for f in range(n_step)]
        ref_solution = copy.deepcopy(new_solution)
        self.dcrModel._clamp_model_solution(new_solution, last_solution, clamp)
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
        metric_ref = np.array([0.0326935547581, 0.0299110561613, 0.0312179049219,
                               0.0347479538541, 0.0391646266206, 0.0421978090644])
        model = np.load(model_file)
        metric = self.dcrModel.calc_model_metric(model=model)
        self.assertFloatsAlmostEqual(metric, metric_ref, rtol=1e-8, atol=1e-10)

    def test_build_model_convergence_failure(self):
        """Test that the iterative solver fails to converge if given a negative gain."""
        converge_error = self.dcrModel._build_model_subroutine(initial_solution=1, verbose=False, gain=-2,
                                                               test_convergence=True)
        self.assertTrue(converge_error)


class MemoryTester(lsst.utils.tests.MemoryTestCase):
    """Test for memory leaks."""

    pass


def setup_module(module):
    """Setup helper for pytest."""
    lsst.utils.tests.init()


if __name__ == "__main__":
    lsst.utils.tests.init()
    unittest.main()
