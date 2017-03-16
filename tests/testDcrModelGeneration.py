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
import unittest

import lsst.afw.geom as afwGeom
from lsst.afw.geom import Angle
import lsst.utils.tests

from python.buildDcrModel import BuildDcrModel
from python.test_utils import BasicGenerateTemplate


class DcrModelGenerationTestCase(lsst.utils.tests.TestCase):

    def setUp(self):
        band_name = 'g'
        self.n_step = 3
        self.n_images = 5
        pixel_scale = Angle(afwGeom.arcsecToRad(0.25))
        kernel_size = 5
        self.size = 20

        dcrTemplate = BasicGenerateTemplate(size=self.size, kernel_size=kernel_size, band_name=band_name,
                                            n_step=self.n_step, pixel_scale=pixel_scale)

        exposures = []
        self.ref_vals = []
        for i in range(self.n_images):
            # NOTE that this array is randomly generated for each instance.
            array = np.random.random(size=(self.size, self.size))*1000.
            self.ref_vals.append(array)
            el = Angle(np.radians(np.random.random()*50. + 40.))
            az = Angle(np.random.random()*2*np.pi)
            exposures.append(dcrTemplate.create_exposure(array, variance=None, elevation=el, azimuth=az))
        # Call the actual BuildDcrModel class here, not just _BasicDcrCorrection
        self.dcrModel = BuildDcrModel(band_name=band_name, n_step=self.n_step, exposures=exposures)

    def tearDown(self):
        del self.dcrModel

    def test_calculate_psf(self):
        """Compare the result of calc_psf_model (run in setUp) to previously computed values."""
        data_file = "test_data/calculate_psf.npy"
        self.dcrModel.calc_psf_model()
        psf_new = self.dcrModel.psf.computeKernelImage().getArray()
        # np.save(data_file, psf_new)
        psf_ref = np.load(data_file)
        self.assertFloatsAlmostEqual(psf_ref, psf_new)

    def test_extract_image(self):
        for exp_i, exp in enumerate(self.dcrModel.exposures):
            image, inverse_var = self.dcrModel._extract_image(exp, calculate_dcr_gen=False)
            self.assertFloatsAlmostEqual(self.ref_vals[exp_i], image)


class MemoryTester(lsst.utils.tests.MemoryTestCase):
    pass


def setup_module(module):
    lsst.utils.tests.init()


if __name__ == "__main__":
    lsst.utils.tests.init()
    unittest.main()
