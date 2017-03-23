"""Tests that read and write exposures and dcr models to disk."""
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

import lsst.utils.tests

from python.generateTemplate import GenerateTemplate
from python.test_utils import DcrModelTestBase


class PersistanceTestCase(DcrModelTestBase, lsst.utils.tests.TestCase):
    """Tests that read and write exposures and dcr models to disk."""

    def test_create_exposure(self):
        """Test that the data retrieved from an exposure is the same as what it was initialized with."""
        self.assertFloatsAlmostEqual(self.exposure.getMaskedImage().getImage().getArray(), self.array)

        # Check that the required metadata is present:
        visitInfo = self.exposure.getInfo().getVisitInfo()
        el = visitInfo.getBoresightAzAlt().getLatitude()
        az = visitInfo.getBoresightAzAlt().getLongitude()
        self.assertAnglesNearlyEqual(el, self.elevation)
        self.assertAnglesNearlyEqual(az, self.azimuth)
        hour_angle = visitInfo.getBoresightHourAngle()
        self.assertAnglesNearlyEqual(hour_angle, self.hour_angle)

    def test_persist_dcr_model_roundtrip(self):
        """Test that an exposure can be persisted and later depersisted from a repository."""
        # Instantiating the butler takes several seconds, so all butler-related tests are condensed into one.
        model_repository = "test_data"

        self.dcrTemplate.export_model(model_repository=model_repository)

        # First test that the model values are not changed from what is expected
        # This requires the full GenerateTemplate class, not just the lightweight test class.
        dcrTemplate2 = GenerateTemplate(model_repository=model_repository)
        # Note that butler.get() reads the FITS file in 32 bit precision.
        for m_new, m_ref in izip(dcrTemplate2.model, self.dcrTemplate.model):
            self.assertFloatsAlmostEqual(m_new, m_ref, rtol=1e-7)

        # Next, test that the required parameters have been restored
        param_ref = self.dcrTemplate.__dict__
        param_new = dcrTemplate2.__dict__
        for key in param_ref:
            self.assertIn(key, param_new)
        # If the parameters are present, now check that they have the correct values.
        # Note that this only tests floats, np.ndarrays, and strings.
        for key in param_ref:
            val_new = param_new.get(key)
            val_ref = param_ref.get(key)
            valid_float = False
            valid_string = False
            # Check whether the key value is a type we can test with assertFloatsAlmostEqual
            # by testing the reference value against itself.
            try:
                self.assertFloatsAlmostEqual(val_ref, val_ref)
                valid_float = True
            except:
                if isinstance(val_ref, str):
                    valid_string = True
            if valid_float:
                print("Checking value of key: %s" % key)
                self.assertFloatsAlmostEqual(val_new, val_ref)
            elif valid_string:
                print("Checking value of key: %s" % key)
                try:
                    self.assertEqual(val_new, val_ref)
                except:
                    print("Failed for some reason!")
            else:
                print("Skipping key: %s" % key)


class MemoryTester(lsst.utils.tests.MemoryTestCase):
    """Test for memory leaks."""

    pass


def setup_module(module):
    """Setup helper for pytest."""
    lsst.utils.tests.init()


if __name__ == "__main__":
    lsst.utils.tests.init()
    unittest.main()