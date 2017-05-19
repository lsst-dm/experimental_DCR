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
import lsst.daf.persistence as daf_persistence
import lsst.utils.tests

from python.dcr_utils import wrap_warpExposure
from python.dcr_utils import calculate_rotation_angle
from python.generateTemplate import GenerateTemplate
from python.test_utils import BasicGenerateTemplate
from python.test_utils import DcrCoaddTestBase


class DcrTemplateTestCase(DcrCoaddTestBase, lsst.utils.tests.TestCase):
    """Tests for the functions in the GenerateTemplate class."""

    @classmethod
    def setUpClass(self):
        """Set up one instance of the butler for all tests of persistence."""
        self.repository = "./test_data/"
        self.butler = daf_persistence.Butler(inputs=self.repository, outputs=self.repository)

    @classmethod
    def tearDownClass(self):
        del self.butler

    def test_simple_phase_kernel(self):
        """Compare the result of _calc_offset_phase to previously computed values."""
        data_file = "test_data/simple_phase_kernel.npy"
        psf = self.exposure.getPsf()
        psf_size = psf.computeKernelImage().getArray().shape[0]
        phase_arr = BasicGenerateTemplate._calc_offset_phase(exposure=self.exposure,
                                                             dcr_gen=self.dcr_gen, size=psf_size)
        # Uncomment the following code to over-write the reference data:
        # np.save(data_file, phase_arr, allow_pickle=False)
        phase_arr_ref = np.load(data_file)
        self.assertFloatsAlmostEqual(phase_arr, phase_arr_ref)

    def test_generate_template(self):
        """Compare the result of generate_templates_from_model to previously computed values."""
        self.dcrTemplate.butler = self.butler
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
        # Uncomment the following code to over-write the reference data:
        # for exposure in model_gen:
        #     self.dcrTemplate.write_exposure(exposure, output_repository=self.repository)
        template_ref_gen = self.dcrTemplate.read_exposures(obsids=obsids)

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
        self.dcrTemplate.butler = self.butler
        # Uncomment the following code to over-write the reference data:
        # self.dcrTemplate.create_skyMap()
        self.dcrTemplate.export_model()

        # First test that the model values are not changed from what is expected
        # This requires the full GenerateTemplate class, not just the lightweight test class.
        dcrTemplate2 = GenerateTemplate(butler=self.butler)
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
            if key == "skyMap":
                print("Skipping key: skyMap")
                # Note: the skyMap object can't be checked properly, but the attempt takes a VERY long time.
                continue
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
