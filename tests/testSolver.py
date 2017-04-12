"""Tests related to the linear least squares solver and calculating the fiducial PSF."""
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
from itertools import izip
import unittest

import numpy as np

import lsst.daf.persistence as daf_persistence
import lsst.utils.tests

from python.dcr_utils import solve_model
from python.test_utils import BasicBuildDcrModel


class SolverTestCase(lsst.utils.tests.TestCase):
    """Tests related to the linear least squares solver.

    Attributes
    ----------
    dcrModel : `BasicBuildDcrModel`
        Class that loads LSST calibrated exposures and produces airmass-matched template images.
    """

    @classmethod
    def setUpClass(self):
        """Define parameters used by every test."""
        butler = daf_persistence.Butler(inputs="./test_data/")
        exposures = []
        n_exp = 6
        for exp_i in range(n_exp):
            dataId = {'visit': exp_i, 'raft': '2,2', 'sensor': '1,1', 'filter': 'g'}
            exposures.append(butler.get("calexp", dataId=dataId))
        # Use BasicBuildDcrModel here to save execution time.
        self.dcrModel = BasicBuildDcrModel(band_name='g', n_step=3, exposures=exposures)
        detected_bit = self.dcrModel.exposures[0].getMaskedImage().getMask().getPlaneBitMask('DETECTED')
        for exp in self.dcrModel.exposures:
            exp.getMaskedImage().getMask().getArray()[:, :] = detected_bit

    @classmethod
    def tearDownClass(self):
        """Clean up."""
        del self.dcrModel

    def test_calculate_psf(self):
        """Compare the result of calc_psf_model (run in setUp) to previously computed values."""
        data_file = "test_data/calculate_psf.npy"
        self.dcrModel.calc_psf_model()
        psf_new = self.dcrModel.psf.computeKernelImage().getArray()
        # np.save(data_file, psf_new, allow_pickle=False)
        psf_ref = np.load(data_file)
        self.assertFloatsAlmostEqual(psf_ref, psf_new)

    def test_build_dcr_kernel(self):
        """Compare the result of _build_dcr_kernel to previously computed values."""
        data_file = "test_data/build_dcr_kernel_vals.npy"
        kernel_size = 5
        kernel = self.dcrModel._build_dcr_kernel(kernel_size)
        # np.save(data_file, kernel, allow_pickle=False)
        kernel_ref = np.load(data_file)
        self.assertFloatsAlmostEqual(kernel, kernel_ref)

    def test_solve_model(self):
        """Compare the result of _solve_model to previously computed values."""
        data_file = "test_data/solve_model_vals.npy"
        y_size, x_size = self.dcrModel.exposures[0].getDimensions()
        kernel_size = 5
        n_step = self.dcrModel.n_step
        pix_radius = kernel_size//2
        # Make j and i different slightly so we can tell if the indices get swapped
        i = x_size//2 + 1
        j = y_size//2 - 1
        slice_inds = np.s_[j - pix_radius: j + pix_radius + 1, i - pix_radius: i + pix_radius + 1]
        image_arr = []
        for exp in self.dcrModel.exposures:
            image_arr.append(np.ravel(exp.getMaskedImage().getImage().getArray()[slice_inds]))
        image_vals = np.hstack(image_arr)
        dcr_kernel = self.dcrModel._build_dcr_kernel(kernel_size)
        model_vals_gen = solve_model(kernel_size, image_vals, n_step=n_step, kernel_dcr=dcr_kernel)
        model_arr = [model for model in model_vals_gen]
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
