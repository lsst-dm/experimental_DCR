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

from python.generateTemplate import GenerateTemplate
from python.test_utils import DcrModelTestBase


class PersistanceTestCase(DcrModelTestBase, lsst.utils.tests.TestCase):
    """Tests that read and write exposures and dcr models to disk."""

    def test_create_exposure(self):
        """Summary.

        Returns
        -------
        None

        Raises
        ------
        Exception
            Any exception is collected and passed up to the caller
        """
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
        """Summary.

        Returns
        -------
        None
        """
        # Instantiating the butler takes several seconds, so all butler-related tests are condensed into one.
        model_repository = "test_data"

        # First test that the model values are not changed from what is expected

        # The type "dcrTemplate" is read in as a 32 bit float,
        # set in the lsst.obs.lsstSim.LsstSimMapper policy
        model = np.float32(self.dcrTemplate.model)
        self.dcrTemplate.export_model(model_repository=model_repository)

        # This requires the full GenerateTemplate class, not just the lightweight test class.
        dcrTemplate2 = GenerateTemplate(model_repository=model_repository)
        # Note that butler.get() reads the FITS file in 32 bit precision.
        self.assertFloatsAlmostEqual(model, dcrTemplate2.model)

        # Next, test that the required parameters have been restored
        param_ref = self.dcrTemplate.__dict__
        param_new = dcrTemplate2.__dict__
        for key in param_ref.keys():
            self.assertIn(key, param_new)


class MemoryTester(lsst.utils.tests.MemoryTestCase):
    pass


def setup_module(module):
    lsst.utils.tests.init()


if __name__ == "__main__":
    lsst.utils.tests.init()
    unittest.main()
