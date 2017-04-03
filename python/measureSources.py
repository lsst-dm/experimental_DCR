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
import lsst.afw.table as afwTable
import lsst.meas.base as measBase
import lsst.afw.coord as afwCoord
import lsst.afw.table as afwTable
import lsst.afw.geom as afwGeom
import lsst.afw.detection as afwDet

from .generateTemplate import GenerateTemplate


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
        catalog_gen = self.read_exposures(obsids=obsids, input_repository=input_repository, data_type="src")
        measurement_config = measBase.ForcedMeasurementConfig()
        measurement_config.plugins.names = ["base_TransformedCentroid", "base_PsfFlux"]
        measurement_config.slots.shape = None

        measurement = measBase.ForcedMeasurementTask(schema, config=measurement_config)
        for sourceCat in catalog_gen:
            # Update the coordinates of the catalog in-place to match the model
            afwTable.utils.updateRefCentroids(self.wcs, sourceCat)

            measCat = measurement.generateMeasCat(exposure, sourceCat, self.wcs)

            measurement.attachTransformedFootprints(measCat, sourceCat, exposure, self.wcs)
            measurement.run(measCat, exposure, sourceCat, self.wcs)
            xv_full = catalog.getX()
            yv_full = catalog.getY()


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
