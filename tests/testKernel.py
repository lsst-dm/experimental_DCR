"""Tests of the kernels that incorporate dcr-based shifts."""
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

import lsst.utils.tests

from python.test_utils import BasicGenerateTemplate
from python.test_utils import DcrModelTestBase


class KernelTestCase(DcrModelTestBase, lsst.utils.tests.TestCase):
    """Tests of the various kernels that incorporate dcr-based shifts."""

    def test_simple_phase_kernel(self):
        """Compare the result of _calc_offset_phase to previously computed values."""
        data_file = "test_data/simple_phase_kernel.npy"
        psf = self.exposure.getPsf()
        psf_size = psf.computeKernelImage().getArray().shape[0]
        phase_arr = BasicGenerateTemplate._calc_offset_phase(exposure=self.exposure,
                                                             dcr_gen=self.dcr_gen, size=psf_size)
        # np.save(data_file, phase_arr, allow_pickle=False)
        phase_arr_ref = np.load(data_file)
        self.assertFloatsAlmostEqual(phase_arr, phase_arr_ref)


class MemoryTester(lsst.utils.tests.MemoryTestCase):
    """Test for memory leaks."""

    pass


def setup_module(module):
    """Setup helper for pytest."""
    lsst.utils.tests.init()


if __name__ == "__main__":
    lsst.utils.tests.init()
    unittest.main()
