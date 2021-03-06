"""Lightweight instances of the classes GenerateTemplate and BuildDcrCoadd for unit testing."""
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

import numpy as np

from lsst.afw.geom import makeCdMatrix, makeSkyWcs
from lsst.geom import Angle
import lsst.afw.image as afwImage
import lsst.afw.math as afwMath
import lsst.meas.algorithms as measAlg
import lsst.geom as geom
from .generateTemplate import GenerateTemplate
from .buildDcrCoadd import BuildDcrCoadd
from .dcr_utils import parallactic_angle
from .lsst_defaults import lsst_observatory

__all__ = ["BasicBandpass", "BasicGenerateTemplate", "BasicBuildDcrCoadd", "DcrCoaddTestBase"]

nanFloat = float("nan")
nanAngle = Angle(nanFloat)


def BasicBandpass(filter_name='g', wavelength_step=1.):
    """Return a dummy bandpass object for testing.

    Parameters
    ----------
    filter_name : str, optional
        Common name of the filter used. For LSST, use u, g, r, i, z, or y
    wavelength_step : float, optional
        Wavelength resolution in nm, also the wavelength range of each sub-band plane.
            If not set, the entire band range is used.

    Returns
    -------
    Returns a lsst.sims.photUtils.Bandpass object.
    """
    bandpass = GenerateTemplate.load_bandpass(filter_name=filter_name, wavelength_step=wavelength_step)
    return(bandpass)


class BasicGenerateTemplate(GenerateTemplate):
    """Dummy GenerateTemplate object for testing without a repository.

    Attributes
    ----------
    bandpass : lsst.sims.photUtils.Bandpass object
        Bandpass object returned by load_bandpass
    bbox : lsst.afw.geom.Box2I object
        A bounding box.
    butler : lsst.daf.persistence Butler object
        The butler handles persisting and depersisting data to and from a repository.
    debug : bool
        Temporary debugging option.
        If set, only a small region [y0: y0 + dy, x0: x0 + dx] of the full images are used.
    default_repository : str
        Full path to repository with the data.
    exposure_time : float
        Length of the exposure, in seconds.
    filter_name : str
        Name of the bandpass-defining filter of the data. Expected values are u,g,r,i,z,y.
    mask : np.ndarray
        Mask plane of the model. This mask is saved as the mask plane of the template exposure.
    model : list of np.ndarrays
        The DCR model to be used to generate templates, calculate with `BuildDcrCoadd.build_model`.
        Contains one array for each wavelength step.
    n_step : int
        Number of sub-filter wavelength planes to model.
    observatory : lsst.afw.coord.coordLib.Observatory
        Class containing the longitude, latitude, and altitude of the observatory.
    pixel_scale : lsst.afw.geom.Angle
            Plate scale, as an Angle.
    psf : lsst.meas.algorithms KernelPsf object
        Representation of the point spread function (PSF) of the model.
    psf_size : int
        Dimension of the PSF, in pixels.
    wcs : lsst.afw.image Wcs object
        World Coordinate System of the model.
    weights : np.ndarray
        Weights of the model. Calculated as the sum of the inverse variances of the input exposures to
        `BuildDcrCoadd.build_model`. The same `weights` are used for each wavelength step of the `model`.
    x_size : int
        Width of the model, in pixels.
    y_size : int
        Height of the model, in pixels.
    """

    def __init__(self, size=None, n_step=3, filter_name='g', exposure_time=30.,
                 pixel_scale=Angle(geom.arcsecToRad(0.25)), wavelength_step=None):
        """Initialize the lightweight version of GenerateTemplate for testing.

        Parameters
        ----------
        size : int, optional
            Number of pixels on a side of the image and model.
        n_step : int, optional
            Number of sub-filter wavelength planes to model. Optional if `wavelength_step` supplied.
        filter_name : str, optional
            Name of the bandpass-defining filter of the data. Expected values are u,g,r,i,z,y.
        exposure_time : float, optional
            Length of the exposure, in seconds. Needed only for exporting to FITS.
        pixel_scale : lsst.afw.geom.Angle, optional
            Plate scale of the images, as an Angle
        wavelength_step : int, optional
            Overridden by `n_step`, if that is supplied. Sub-filter width, in nm.
        """
        seed = 5
        rand_gen = np.random
        rand_gen.seed(seed)
        self.butler = None
        self.default_repository = None
        self.debug = False

        bandpass_init = BasicBandpass(filter_name=filter_name, wavelength_step=wavelength_step)
        wavelength_step = (bandpass_init.wavelen_max - bandpass_init.wavelen_min) / n_step
        self.bandpass = BasicBandpass(filter_name=filter_name, wavelength_step=wavelength_step)
        self.bandpass_highres = BasicBandpass(filter_name=filter_name, wavelength_step=None)
        self.model = [rand_gen.random(size=(size, size)) for f in range(n_step)]
        # self.weights = np.ones((size, size))
        self.mask = np.zeros((size, size), dtype=np.int32)

        self.n_step = n_step
        self.y_size = size
        self.x_size = size
        self.pixel_scale = pixel_scale
        self.psf_size = 5
        self.exposure_time = exposure_time
        self.filter_name = filter_name
        self.observatory = lsst_observatory
        self.bbox = geom.Box2I(geom.Point2I(0, 0), geom.ExtentI(size, size))
        self.wcs = self._create_wcs(bbox=self.bbox, pixel_scale=pixel_scale, ra=Angle(0.),
                                    dec=Angle(0.), sky_rotation=Angle(0.))

        psf_vals = np.zeros((self.psf_size, self.psf_size))
        psf_vals[self.psf_size//2 - 1: self.psf_size//2 + 1,
                 self.psf_size//2 - 1: self.psf_size//2 + 1] = 0.5
        psf_vals[self.psf_size//2, self.psf_size//2] = 1.
        psf_image = afwImage.ImageD(self.psf_size, self.psf_size)
        psf_image.getArray()[:, :] = psf_vals
        psfK = afwMath.FixedKernel(psf_image)
        self.psf = measAlg.KernelPsf(psfK)

    @staticmethod
    def _create_wcs(bbox, pixel_scale, ra, dec, sky_rotation):
        """Create a wcs (coordinate system).

        Parameters
        ----------
        bbox : lsst.afw.geom.Box2I object
            A bounding box.
        pixel_scale : lsst.afw.geom.Angle
            Plate scale, as an Angle.
        ra : lsst.afw.geom.Angle
            Right Ascension of the reference pixel, as an `Angle`.
        dec : lsst.afw.geom.Angle
            Declination of the reference pixel, as an Angle.
        sky_rotation : lsst.afw.geom.Angle
            Rotation of the image axis, East from North.

        Returns
        -------
        Returns a lsst.afw.image.wcs object.
        """
        crval = geom.SpherePoint(ra, dec)
        crpix = geom.Box2D(bbox).getCenter()
        cd_matrix = makeCdMatrix(scale=pixel_scale, orientation=sky_rotation, flipX=True)
        wcs = makeSkyWcs(crpix=crpix, crval=crval, cdMatrix=cd_matrix)
        return(wcs)


class BasicBuildDcrCoadd(BuildDcrCoadd):
    """Dummy BuildDcrCoadd object for testing without a repository.

    Attributes
    ----------
    bandpass : lsst.sims.photUtils.Bandpass object
        Bandpass object returned by `load_bandpass`
    bbox : lsst.afw.geom.Box2I object
        A bounding box.
    butler : lsst.daf.persistence Butler object
        The butler handles persisting and depersisting data to and from a repository.
    debug : bool
        Temporary debugging option.
        If set, calculations are performed on only a small region of the full images.
    default_repository : str
        Full path to repository with the data.
    exposure_time : float
        Length of the exposure, in seconds.
    exposures : list
        List of input exposures used to calculate the model.
    filter_name : str
        Name of the bandpass-defining filter of the data. Expected values are u,g,r,i,z,y.
    mask : np.ndarray
        Combined bit plane mask of the model, which is used as the mask plane for generated templates.
    model_base : np.ndarray
        Coadded model built from the input exposures, without accounting for DCR.
        Used as the starting point for the iterative solution.
    n_images : int
        Number of input images used to calculate the model.
    n_step : int
        Number of sub-filter wavelength planes to model.
    observatory : lsst.afw.coord.coordLib.Observatory
        Class containing the longitude, latitude, and altitude of the observatory.
    pixel_scale : lsst.afw.geom.Angle
            Plate scale, as an Angle.
    psf_size : int
        Dimension of the PSF, in pixels.
    wcs : lsst.afw.image Wcs object
        World Coordinate System of the model.
    x_size : int
        Width of the model, in pixels.
    y_size : int
        Height of the model, in pixels.
    """

    def __init__(self, filter_name='g', n_step=3, exposures=None, psf_size=None):
        """Initialize the lightweight version of BuildDcrCoadd for testing.

        Parameters
        ----------
        filter_name : str, optional
            Name of the bandpass-defining filter of the data. Expected values are u,g,r,i,z,y.
        n_step : int, optional
            Number of sub-filter wavelength planes to model.
        exposures : lsst.afw.image ExposureD object, optional
            A list of LSST exposures to use as input to the DCR calculation.
        """
        self.butler = None
        self.default_repository = None
        self.debug = False
        self.mask = None
        self.model_base = None
        self.filter_name = filter_name

        self.exposures = exposures

        self.bandpass_highres = BasicBandpass(filter_name=filter_name, wavelength_step=None)
        wavelength_step = (self.bandpass_highres.wavelen_max - self.bandpass_highres.wavelen_min) / n_step
        self.bandpass = BasicBandpass(filter_name=filter_name, wavelength_step=wavelength_step)
        self.n_step = n_step
        self.n_images = len(exposures)
        y_size, x_size = exposures[0].getDimensions()
        self.x_size = x_size
        self.y_size = y_size
        self.pixel_scale = exposures[0].getWcs().getPixelScale()
        self.exposure_time = exposures[0].getInfo().getVisitInfo().getExposureTime()
        self.bbox = exposures[0].getBBox()
        self.wcs = exposures[0].getWcs()
        self.observatory = exposures[0].getInfo().getVisitInfo().getObservatory()
        psf = exposures[0].getPsf().computeKernelImage().getArray()
        if psf_size is None:
            psf_size = psf.shape[0]
        self.psf_size = psf_size


class DcrCoaddTestBase(object):
    """Base class many unit tests can inherit from to simplify setup.

    Attributes
    ----------
    array : np.ndarray
        Random array of input data values.
    azimuth : lsst.afw.geom Angle
        Azimuth angle of the observation
    dcr_gen : GenerateTemplate.dcr_generator
        Generator of Differential Chromatic Refraction (DCR) values per sub-band.
    dcrTemplate : `GenerateTemplate`
        Basic instance of the `GenerateTemplate` class for testing.
    elevation : lsst.afw.geom Angle
        Elevation angle of the observation
    exposure : lsst.afw.image.ExposureD object
        The exposure containing the data from `array` and associated metadata.
    rotation_angle : lsst.afw.geom Angle, optional
            The rotation angle of the field around the boresight.
    """

    def setUp(self):
        """Define parameters used by every test."""
        filter_name = 'g'
        n_step = 3
        pixel_scale = Angle(geom.arcsecToRad(0.25))
        size = 20
        self.latitude = lsst_observatory.getLatitude()
        # NOTE that this array is randomly generated
        random_seed = 3
        rand_gen = np.random
        rand_gen.seed(random_seed)
        self.array = np.float32(rand_gen.random(size=(size, size)))
        self.dcrTemplate = BasicGenerateTemplate(size=size, filter_name=filter_name,
                                                 n_step=n_step, pixel_scale=pixel_scale)
        self.dcrTemplate.create_skyMap(doWrite=False)
        self.dec = self.dcrTemplate.wcs.getSkyOrigin().getLatitude()
        self.ra = self.dcrTemplate.wcs.getSkyOrigin().getLongitude()
        self.azimuth = Angle(np.radians(140.0))
        self.elevation = Angle(np.radians(50.0))
        ha_term1 = np.sin(self.elevation.asRadians())
        ha_term2 = np.sin(self.dec.asRadians())*np.sin(self.latitude.asRadians())
        ha_term3 = np.cos(self.dec.asRadians())*np.cos(self.latitude.asRadians())
        self.hour_angle = Angle(np.arccos((ha_term1 - ha_term2) / ha_term3))
        p_angle = parallactic_angle(self.hour_angle, self.dec, self.latitude)
        self.rotation_angle = Angle(p_angle)
        self.dcrTemplate.weights = np.zeros_like(self.array)
        nonzero_inds = self.array > 0
        self.dcrTemplate.weights[nonzero_inds] = 1./np.abs(self.array[nonzero_inds])
        self.dcr_gen = self.dcrTemplate._dcr_generator(self.dcrTemplate.bandpass,
                                                       pixel_scale=self.dcrTemplate.pixel_scale,
                                                       elevation=self.elevation,
                                                       rotation_angle=self.rotation_angle,
                                                       use_midpoint=False)
        self.exposure = self.dcrTemplate.create_exposure(self.array, self.elevation, self.azimuth,
                                                         variance=None, boresightRotAngle=self.rotation_angle,
                                                         dec=self.dec, ra=self.ra)

    def tearDown(self):
        """Free memory."""
        del self.dcrTemplate
        del self.exposure
        del self.dcr_gen
