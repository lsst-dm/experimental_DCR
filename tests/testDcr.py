"""Test the the calculations of Differential Chromatic Refraction."""
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

from python.test_utils import BasicBandpass
from python.test_utils import BasicGenerateTemplate


class DCRTestCase(lsst.utils.tests.TestCase):
    """Test the the calculations of Differential Chromatic Refraction.

    Attributes
    ----------
    bandpass : lsst.sims.photUtils.Bandpass object
        Bandpass object returned by load_bandpass
    pixel_scale : lsst.afw.geom.Angle
            Plate scale, as an Angle.
    """

    def setUp(self):
        """Define parameters used by every test."""
        band_name = 'g'
        wavelength_step = 10.0  # nanometers
        self.pixel_scale = Angle(afwGeom.arcsecToRad(0.25))  # angle/pixel
        self.bandpass = BasicBandpass(band_name=band_name, wavelength_step=wavelength_step)

    def tearDown(self):
        """Clean up."""
        del self.bandpass

    def test_dcr_generator(self):
        """Check that _dcr_generator returns a generator with n_step iterations, and (0,0) at zenith."""
        rotation_angle = Angle(0.0)
        elevation = Angle(np.pi/2)
        zenith_dcr = 0.
        bp = self.bandpass
        dcr_gen = BasicGenerateTemplate._dcr_generator(bp, pixel_scale=self.pixel_scale,
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
        dcr_ref_vals = [(1.92388200784, 1.58581288354),
                        (1.58581288354, 1.27307598239),
                        (1.27307598239, 0.983159546125),
                        (0.983159546125, 0.713860598946),
                        (0.713860598946, 0.463239789237),
                        (0.463239789237, 0.229583865848),
                        (0.229583865848, 0.0113743279352),
                        (0.0113743279352, -0.192738901197),
                        (-0.192738901197, -0.383959687465),
                        (-0.383959687465, -0.563364513693),
                        (-0.563364513693, -0.731918505611),
                        (-0.731918505611, -0.890489127945),
                        (-0.890489127945, -1.03985793393),
                        (-1.03985793393, -1.18073068133),
                        (-1.18073068133, -1.27463093578),
                        ]
        bp = self.bandpass
        dcr_gen = BasicGenerateTemplate._dcr_generator(bp, pixel_scale=self.pixel_scale, elevation=elevation,
                                                       rotation_angle=rotation_angle)
        n_step = int(np.ceil((bp.wavelen_max - bp.wavelen_min) / bp.wavelen_step))
        for f in range(n_step):
            dcr = next(dcr_gen)
            self.assertFloatsAlmostEqual(dcr.dy.start, dcr_ref_vals[f][0], rtol=1e-10)
            self.assertFloatsAlmostEqual(dcr.dy.end, dcr_ref_vals[f][1], rtol=1e-10)


class MemoryTester(lsst.utils.tests.MemoryTestCase):
    """Test for memory leaks."""

    pass


def setup_module(module):
    """Setup helper for pytest."""
    lsst.utils.tests.init()


if __name__ == "__main__":
    lsst.utils.tests.init()
    unittest.main()
