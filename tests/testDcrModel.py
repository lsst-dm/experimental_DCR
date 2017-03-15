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

from lsst.afw.geom import Angle
import unittest
import lsst.utils.tests
from python.dcr_template import wrap_warpExposure
from python.dcr_template import calculate_rotation_angle

from python.test_utils import BasicDcrModel
from python.test_utils import DcrModelTestBase


nanFloat = float("nan")
nanAngle = Angle(nanFloat)


class DcrModelTestCase(DcrModelTestBase, lsst.utils.tests.TestCase):
    """Tests for the functions in the DcrModel class."""

    def test_dataId_single(self):
        id_ref = 100
        band_ref = 'g'
        ref_id = {'visit': id_ref, 'raft': '2,2', 'sensor': '1,1', 'filter': band_ref}
        dataId = BasicDcrModel._build_dataId(id_ref, band_ref)
        self.assertEqual(ref_id, dataId[0])

    def test_dataId_list(self):
        id_ref = [100, 103]
        band_ref = 'g'
        ref_id = {'visit': id_ref, 'raft': '2,2', 'sensor': '1,1', 'filter': band_ref}
        dataId = BasicDcrModel._build_dataId(id_ref, band_ref)
        for i, id in enumerate(id_ref):
            ref_id = {'visit': id, 'raft': '2,2', 'sensor': '1,1', 'filter': band_ref}
            self.assertEqual(ref_id, dataId[i])

    def test_model_dataId(self):
        subfilter = 1
        band_ref = 'g'
        ref_id = {'filter': band_ref, 'tract': 0, 'patch': '0', 'subfilter': subfilter}
        dataId = BasicDcrModel._build_model_dataId(band_ref, subfilter=subfilter)
        self.assertEqual(ref_id, dataId)

    def test_fetch_metadata(self):
        az = BasicDcrModel._fetch_metadata(self.exposure.getMetadata(), "AZIMUTH")
        self.assertFloatsAlmostEqual(az, self.azimuth.asDegrees())

    def test_fetch_missing_metadata(self):
        bogus_ref = 3.7
        bogus = BasicDcrModel._fetch_metadata(self.exposure.getMetadata(), "BOGUS", default_value=bogus_ref)
        self.assertFloatsAlmostEqual(bogus, bogus_ref)

    def test_generate_template(self):
        """Compare the result of generate_templates_from_model to previously computed values."""
        data_file = "test_data/template.npy"
        elevation_arr = np.radians([50., 70., 85.])
        az = Angle(0.)
        # Note that self.array is randomly generated each call. That's okay, because the template should
        # depend only on the metadata.
        exposures = [self.dcrModel.create_exposure(self.array, variance=None, elevation=Angle(el), azimuth=az)
                     for el in elevation_arr]
        model_gen = self.dcrModel.generate_templates_from_model(exposures=exposures)
        model_test = [model for model in model_gen]
        # np.save(data_file, model_test)
        model_ref = np.load(data_file)
        for m_i in range(len(model_test)):
            m_test = model_test[m_i].getMaskedImage().getImage().getArray()
            m_ref = model_ref[m_i].getMaskedImage().getImage().getArray()
            self.assertFloatsAlmostEqual(m_test, m_ref)

    def test_warp_exposure(self):
        wcs = self.exposure.getWcs()
        bbox = self.exposure.getBBox()
        wrap_warpExposure(self.exposure, wcs, bbox)
        array_warped = self.exposure.getMaskedImage().getImage().getArray()
        # For some reason the edges are all NAN.
        valid_inds = np.isfinite(array_warped)
        self.assertGreater(np.sum(valid_inds), (self.size/2)**2)
        array_ref = self.array[valid_inds]
        array_warped = array_warped[valid_inds]
        self.assertFloatsAlmostEqual(array_ref, array_warped, rtol=1e-7)

    def test_rotation_angle(self):
        rotation_angle = calculate_rotation_angle(self.exposure)
        self.assertFloatsAlmostEqual(self.rotation_angle.asDegrees(), rotation_angle.asDegrees())


class MemoryTester(lsst.utils.tests.MemoryTestCase):
    pass


def setup_module(module):
    lsst.utils.tests.init()


if __name__ == "__main__":
    lsst.utils.tests.init()
    unittest.main()
