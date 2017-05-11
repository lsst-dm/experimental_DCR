"""Estimate a spectrum for forced sources using the DCR model."""

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

import lsst.afw.coord as afwCoord
import lsst.afw.detection as afwDet
import lsst.afw.geom as afwGeom
from lsst.afw.geom import Angle
import lsst.afw.table as afwTable
from lsst.meas.algorithms.detection import SourceDetectionTask
import lsst.meas.base as measBase
from lsst.meas.base import SingleFrameMeasurementTask

from .generateTemplate import GenerateTemplate
from .lsst_defaults import lsst_weather


class MeasureSources(GenerateTemplate):
    """Estimate a spectrum for forced sources using the DCR model.

    Inherits from GenerateTemplate to use its read and write exposure methods, and load_model method.

    Attributes
    ----------
    butler : lsst.daf.persistence Butler object
        The butler handles persisting and depersisting data to and from a repository.
    default_repository : str
        Full path to repository with the data.
    """

    def __init__(self, model_repository=None, band_name='g', **kwargs):
        """Restore a persisted DCR model created with BuildDcrModel.

        Parameters
        ----------
        model_repository : None, optional
            Path to the repository where the previously-generated DCR model is stored.
        band_name : str, optional
            Name of the bandpass-defining filter of the data. Expected values are u,g,r,i,z,y.
        **kwargs : TYPE
            Any additional keyword arguments to pass to load_bandpass
        """
        self.butler = None
        self.default_repository = model_repository
        self.load_model(model_repository=model_repository, band_name=band_name, **kwargs)

    def measure_spectrum(self, obsids, input_repository=None):
        # catalog_gen = self.read_exposures(obsids=obsids, input_repository=input_repository, data_type="src")

        # Generate a mock template at Zenith to perform initial source detection
        zenith_el = Angle(np.pi/2)
        zenith_rotation = Angle(0.0)
        zenith_template, variance = self.build_matched_template(el=zenith_el, rotation_angle=zenith_rotation)
        ref_exposure = self.create_exposure(zenith_template, variance=variance, snap=0,
                                            boresightRotAngle=zenith_rotation, weather=lsst_weather,
                                            elevation=zenith_el, azimuth=zenith_rotation, exposureId=-1)

        # schema = sourceCat.getSchema()
        # measurement_config = measBase.ForcedMeasurementConfig()
        # measurement_config.plugins.names = ["base_TransformedCentroid", "base_PsfFlux"]
        # measurement_config.slots.shape = None
        # measurement = measBase.ForcedMeasurementTask(schema, config=measurement_config)

        # # Update the coordinates of the reference catalog in-place to match the model
        # afwTable.utils.updateRefCentroids(self.wcs, sourceCat)

        # measCat = measurement.generateMeasCat(exposure, sourceCat, self.wcs)

        # measurement.attachTransformedFootprints(measCat, sourceCat, exposure, self.wcs)
        # measurement.run(measCat, exposure, sourceCat, self.wcs)
        # xv_full = catalog.getX()
        # yv_full = catalog.getY()

        measure_config = SingleFrameMeasurementTask.ConfigClass()
        schema = afwTable.SourceTable.makeMinimalSchema()
        measure_config.plugins.names.clear()
        plugin_list = ["base_SdssCentroid", "base_SdssShape", "base_CircularApertureFlux",
                       "base_GaussianFlux"]
        for plugin in plugin_list:
            measure_config.plugins.names.add(plugin)
        measure_config.slots.psfFlux = "base_GaussianFlux"
        measure_config.slots.apFlux = "base_CircularApertureFlux_3_0"
        measureTask = SingleFrameMeasurementTask(schema, config=measure_config)
        detect_config = SourceDetectionTask.ConfigClass()
        detect_config.background.isNanSafe = True
        detect_config.thresholdValue = 3
        detectionTask = SourceDetectionTask(config=detect_config, schema=schema)

        model_exp = [self.create_exposure(model, variance=variance, snap=0, oresightRotAngle=Angle(0.0),
                                          bweather=self.exposures[0].getInfo().getVisitInfo().getWeather(),
                                          elevation=Angle(np.pi/2), azimuth=Angle(0.0), exposureId=-1)
                     for model in self.model]

        tab = afwTable.SourceTable.make(schema)
        refCat = detectionTask.run(tab, ref_exposure)
        measureTask.run(refCat.sources, ref_exposure)



def make_sourceCat_from_astropy(astropy_catalog, exposure):
    schema = afwTable.SourceTable.makeMinimalSchema()
    x_key = schema.addField("centroid_x", type="D")
    y_key = schema.addField("centroid_y", type="D")
    alias = schema.getAliasMap()
    alias.set("slot_Centroid", "centroid")

    sourceCat = afwTable.SourceCatalog(schema)
    for source_rec in astropy_catalog:
        rec = sourceCat.addNew()
        coord = afwCoord.IcrsCoord(source_rec['raICRS']*afwGeom.degrees,
                                   source_rec['decICRS']*afwGeom.degrees)
        rec.setCoord(coord)
        point = exposure.getWcs().skyToPixel(coord)
        fp = afwDet.Footprint(afwGeom.Point2I(point), 6.0)
        rec.setFootprint(fp)
        rec[x_key] = point.getX()
        rec[y_key] = point.getY()
    return sourceCat


def do_forcephot(exposure, sourceCat):
    measurement_config = measBase.ForcedMeasurementConfig()
    measurement_config.plugins.names = ["base_TransformedCentroid", "base_PsfFlux"]
    measurement_config.slots.shape = None

    measurement = measBase.ForcedMeasurementTask(schema, config=measurement_config)

    measCat = measurement.generateMeasCat(exposure, sourceCat, exposure.getWcs())

    measurement.attachTransformedFootprints(measCat, sourceCat, exposure, exposure.getWcs())
    measurement.run(measCat, exposure, sourceCat, exposure.getWcs())
    return measCat
