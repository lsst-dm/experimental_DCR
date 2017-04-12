"""Tests for functions and methods that are primarily used for generating templates."""
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
import numpy as np
import unittest

from lsst.afw.geom import Angle
import lsst.utils.tests

from python.dcr_utils import wrap_warpExposure
from python.dcr_utils import calculate_rotation_angle
from python.test_utils import BasicGenerateTemplate
from python.test_utils import DcrModelTestBase


class DcrTemplateTestCase(DcrModelTestBase, lsst.utils.tests.TestCase):
    """Tests for the functions in the GenerateTemplate class."""

    def test_dataId_single(self):
        """Test that the dataId for the `calexp` data type is correct for a single observation."""
        id_ref = 100
        band_ref = 'g'
        ref_id = {'visit': id_ref, 'raft': '2,2', 'sensor': '1,1', 'filter': band_ref}
        dataId = BasicGenerateTemplate._build_dataId(id_ref, band_ref)
        self.assertEqual(ref_id, dataId[0])

    def test_dataId_list(self):
        """Test that the dataIds for the `calexp` data type are correct for a list of observations."""
        id_ref = [100, 103]
        band_ref = 'g'
        dataId = BasicGenerateTemplate._build_dataId(id_ref, band_ref)
        for i, obsid in enumerate(id_ref):
            ref_dataid = {'visit': obsid, 'raft': '2,2', 'sensor': '1,1', 'filter': band_ref}
            self.assertEqual(ref_dataid, dataId[i])

    def test_model_dataId(self):
        """Test that the dataIds for the `dcrModel` data type are correct."""
        subfilter = 1
        band_ref = 'g'
        ref_id = {'filter': band_ref, 'tract': 0, 'patch': '0', 'subfilter': subfilter}
        dataId = BasicGenerateTemplate._build_model_dataId(band_ref, subfilter=subfilter)
        self.assertEqual(ref_id, dataId)

    def test_generate_template(self):
        """Compare the result of generate_templates_from_model to previously computed values."""
        repository = "./test_data/"
        elevation_arr = np.radians([50., 70., 85.])
        az = Angle(0.)
        # Note that self.array is randomly generated each call. That's okay, because the template should
        # depend only on the metadata.
        exposures = []
        obsids = np.arange(len(elevation_arr)) + 500
        for el, obsid in zip(elevation_arr, obsids):
            exposures.append(self.dcrTemplate.create_exposure(self.array, variance=None, exposureId=obsid,
                                                              elevation=Angle(el), azimuth=az))
        template_gen = self.dcrTemplate.generate_templates_from_model(exposures=exposures)
        # Uncomment the following code to re-generate the reference data:
        # for exposure in model_gen:
        #     self.dcrTemplate.write_exposure(exposure, output_repository=repository)
        template_ref_gen = self.dcrTemplate.read_exposures(obsids=obsids, input_repository=repository)

        for template_test, template_ref in izip(template_gen, template_ref_gen):
            self.assertMaskedImagesNearlyEqual(template_test.getMaskedImage(), template_ref.getMaskedImage())

    def test_warp_exposure(self):
        """Test that an exposure warped to its own wcs is unchanged."""
        wcs = self.exposure.getWcs()
        bbox = self.exposure.getBBox()
        image_ref = self.exposure.getMaskedImage().clone()
        wrap_warpExposure(self.exposure, wcs, bbox)
        image_warped = self.exposure.getMaskedImage()
        mask = image_warped.getMask()
        no_data_bit = mask.getPlaneBitMask('NO_DATA')
        no_data_mask = (mask.getArray() & no_data_bit) == no_data_bit
        self.assertMaskedImagesNearlyEqual(image_ref, image_warped, rtol=1e-7, skipMask=no_data_mask)

    def test_rotation_angle(self):
        """Test that we can calculate the same rotation angle that was originally supplied in setup."""
        rotation_angle = calculate_rotation_angle(self.exposure)
        self.assertAnglesNearlyEqual(self.rotation_angle, rotation_angle)


class MemoryTester(lsst.utils.tests.MemoryTestCase):
    """Test for memory leaks."""

    pass


def setup_module(module):
    """Setup helper for pytest."""
    lsst.utils.tests.init()


if __name__ == "__main__":
    lsst.utils.tests.init()
    unittest.main()
