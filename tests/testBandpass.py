"""Tests of the interface to Bandpass from lsst.sims.photUtils."""
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


class BandpassTestCase(lsst.utils.tests.TestCase):
    """Tests of the interface to Bandpass from lsst.sims.photUtils.

    Attributes
    ----------
    filter_name : str
        Name of the bandpass-defining filter of the data. Expected values are u,g,r,i,z,y.
    bandpass : lsst.sims.photUtils.Bandpass object
        Bandpass object returned by load_bandpass
    wavelength_step : float, optional
            Wavelength resolution in nm, also the wavelength range of each sub-band plane.
            If not set, the entire band range is used.
    """

    def setUp(self):
        """Define parameters used by every test."""
        self.filter_name = 'g'
        self.wavelength_step = 10.
        self.bandpass = BasicGenerateTemplate.load_bandpass(filter_name=self.filter_name,
                                                            wavelength_step=self.wavelength_step)

    def test_step_bandpass(self):
        """Check that the bandpass has necessary methods, and those return the correct number of values."""
        bp = self.bandpass
        bp_wavelen, bandpass_vals = bp.getBandpass()
        n_step = int(np.ceil((bp.wavelen_max - bp.wavelen_min) / bp.wavelen_step))
        self.assertEqual(n_step + 1, len(bandpass_vals))

    def test_bandpass(self):
        """Verify the calculated bandpass values."""
        data_file = "test_data/bandpass.npy"
        bp_wavelen, bandpass_vals = self.bandpass.getBandpass()
        # Uncomment the following code to over-write the reference data:
        # np.save(data_file, (bp_wavelen, bandpass_vals), allow_pickle=False)
        bp_wavelen_ref, bandpass_vals_ref = np.load(data_file)
        self.assertFloatsAlmostEqual(bp_wavelen, bp_wavelen_ref)
        self.assertFloatsAlmostEqual(bandpass_vals, bandpass_vals_ref)


class MemoryTester(lsst.utils.tests.MemoryTestCase):
    """Test for memory leaks."""

    pass


def setup_module(module):
    """Setup helper for pytest."""
    lsst.utils.tests.init()


if __name__ == "__main__":
    lsst.utils.tests.init()
    unittest.main()
