"""Utilities used by the DCR template generation code, including atmospheric refraction calculations."""

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

from astropy import units as u
from astropy.units import cds as u_cds
import numpy as np
import scipy.optimize.nnls

from lsst.afw.geom import Angle
import lsst.afw.image as afwImage
import lsst.afw.math as afwMath

from .lsst_defaults import lsst_observatory, lsst_weather

__all__ = ["parallactic_angle", "wrap_warpExposure", "solve_model", "calculate_rotation_angle",
           "refraction", "diff_refraction"]


def parallactic_angle(hour_angle, dec, lat):
    """Compute the parallactic angle given hour angle, declination, and latitude.

    Parameters
    ----------
    hour_angle : lsst.afw.geom.Angle
        Hour angle of the observation
    dec : lsst.afw.geom.Angle
        Declination of the observation.
    lat : lsst.afw.geom.Angle
        Latitude of the observatory.
    """
    y_term = np.sin(hour_angle.asRadians())
    x_term = (np.cos(dec.asRadians())*np.tan(lat.asRadians()) -
              np.sin(dec.asRadians())*np.cos(hour_angle.asRadians()))
    return np.arctan2(y_term, x_term)


def wrap_warpExposure(exposure, wcs, BBox, warpingControl=None):
    """Warp an exposure to fit a given WCS and bounding box.

    Parameters
    ----------
    exposure : lsst.afw.image.ExposureD
        An LSST exposure object. The image values will be overwritten!
    wcs : lsst.afw.image.Wcs object
        World Coordinate System to warp the image to.
    BBox : lsst.afw.geom.Box2I object
        Bounding box of the new image.
    warpingControl : afwMath.WarpingControl, optional
        Sets the interpolation parameters. Loads defualt values if None.

    Returns
    -------
    None
        Modifies exposure in place.
    """
    if warpingControl is None:
        interpLength = 10
        warpingControl = afwMath.WarpingControl("lanczos4", "", 0, interpLength)
    warpExp = afwImage.ExposureD(BBox, wcs)
    afwMath.warpExposure(warpExp, exposure, warpingControl)

    warpImg = warpExp.getMaskedImage().getImage().getArray()
    exposure.getMaskedImage().getImage().getArray()[:, :] = warpImg
    warpMask = warpExp.getMaskedImage().getMask().getArray()
    exposure.getMaskedImage().getMask().getArray()[:, :] = warpMask
    warpVariance = warpExp.getMaskedImage().getVariance().getArray()
    exposure.getMaskedImage().getVariance().getArray()[:, :] = warpVariance
    exposure.setWcs(wcs)


def solve_model(kernel_size, img_vals, n_step, kernel_dcr, kernel_ref=None, kernel_restore=None):
    """Wrapper to call a fitter using a given covariance matrix, image values, and any regularization.

    Parameters
    ----------
    kernel_size : int
        Size of the kernel to use for calculating the covariance matrix, in pixels.
    img_vals : np.ndarray
        Image data values for the pixels being used for the calculation, as a 1D vector.
    n_step : int, optional
        Number of sub-filter wavelength planes to model.
    kernel_dcr : np.ndarray
        The covariance matrix describing the effect of DCR
    kernel_ref : np.ndarray, optional
        The covariance matrix for the reference image
    kernel_restore : np.ndarray, optional
        The covariance matrix for the final restored image

    Returns
    -------
    np.ndarray
        Array of the solution values.
    """
    x_size = kernel_size
    y_size = kernel_size
    if (kernel_restore is None) or (kernel_ref is None):
        vals_use = img_vals
        kernel_use = kernel_dcr
    else:
        vals_use = kernel_restore.dot(img_vals)
        kernel_use = kernel_ref.dot(kernel_dcr)

    model_solution = scipy.optimize.nnls(kernel_use, vals_use)
    model_vals = model_solution[0]
    n_pix = x_size*y_size
    for f in range(n_step):
        yield np.reshape(model_vals[f*n_pix: (f + 1)*n_pix], (y_size, x_size))


def calculate_rotation_angle(exposure):
    """Calculate the sky rotation angle of an exposure.

    Parameters
    ----------
    exposure : lsst.afw.image.ExposureD
        An LSST exposure object.

    Returns
    -------
    lsst.afw.geom.Angle
        The rotation of the image axis, East from North.
    """
    visitInfo = exposure.getInfo().getVisitInfo()

    az = visitInfo.getBoresightAzAlt().getLongitude()
    hour_angle = visitInfo.getBoresightHourAngle()
    # Some simulated data contains invalid hour_angle metadata.
    if np.isfinite(hour_angle.asRadians()):
        dec = visitInfo.getBoresightRaDec().getDec()
        lat = visitInfo.getObservatory().getLatitude()
        p_angle = parallactic_angle(hour_angle, dec, lat)
    else:
        p_angle = az.asRadians()
    cd = exposure.getInfo().getWcs().getCDMatrix()
    cd_rot = (np.arctan2(-cd[0, 1], cd[0, 0]) + np.arctan2(cd[1, 0], cd[1, 1]))/2.
    rotation_angle = Angle(cd_rot + p_angle)
    return rotation_angle


def refraction(wavelength, zenith_angle, weather=lsst_weather, observatory=lsst_observatory):
    """Calculate overall refraction under atmospheric and observing conditions.

    Parameters
    ----------
    wavelength : float
        wavelength is in nm (valid for 230.2 < wavelength < 2058.6)
    zenith_angle : lsst.afw.geom Angle
        Zenith angle of the observation, as an Angle.
    weather : lsst.afw.coord Weather, optional
        Class containing the measured temperature, pressure, and humidity
        at the observatory during an observation
    observatory : lsst.afw.coord Observatory, optional
        Class containing the longitude, latitude, and altitude of the observatory.

    Returns
    -------
    lsst.afw.geom Angle
        The angular refraction for light of the given wavelength, under the given observing conditions.
    """
    latitude = observatory.getLatitude()
    altitude = observatory.getElevation()

    reduced_n = n_delta(wavelength, weather)*1E-8

    temperature = _extract_temperature(weather, units_kelvin=True)
    atmos_scaleheight_ratio = float(4.5908E-6*temperature/u.Kelvin)

    # Account for oblate Earth
    relative_gravity = (1. + 0.005302*np.sin(latitude.asRadians())**2. -
                        0.00000583*np.sin(2.*latitude.asRadians())**2. - 0.000000315*altitude)

    tanZ = np.tan(zenith_angle.asRadians())

    atmos_term_1 = reduced_n*relative_gravity*(1. - atmos_scaleheight_ratio)
    atmos_term_2 = reduced_n*relative_gravity*(atmos_scaleheight_ratio - reduced_n/2.)
    result = Angle(float(atmos_term_1*tanZ + atmos_term_2*tanZ**3.))
    return result


def diff_refraction(wavelength, wavelength_ref, zenith_angle,
                    weather=lsst_weather, observatory=lsst_observatory):
    """Calculate the differential refraction between two wavelengths.

    Parameters
    ----------
    wavelength : float
        wavelength is in nm (valid for 230.2 < wavelength < 2058.6)
    wavelength_ref : float
        Reference wavelength, typically the effective wavelength of a filter.
    zenith_angle : lsst.afw.geom Angle
        Zenith angle of the observation, as an Angle.
    weather : lsst.afw.coord Weather, optional
        Class containing the measured temperature, pressure, and humidity
        at the observatory during an observation
    observatory : lsst.afw.coord Observatory, optional
        Class containing the longitude, latitude, and altitude of the observatory.

    Returns
    -------
    lsst.afw.geom Angle
        The refraction at `wavelength` - the refraction at `wavelength_ref`.
    """
    refraction_start = refraction(wavelength, zenith_angle, weather=weather, observatory=observatory)
    refraction_end = refraction(wavelength_ref, zenith_angle, weather=weather, observatory=observatory)
    return refraction_start - refraction_end


def n_delta(wavelength, weather):
    """Calculate the differential refractive index of air.

    The differential refractive index is the difference of the refractive index from 1.,
    multiplied by 1E8 to simplify the notation and equations.
    Calculated as (n_air - 1)*10^8

    This replicates equation 14 of Stone 1996 "An Accurate Method for Computing Atmospheric Refraction"

    Parameters
    ----------
    wavelength : float
        wavelength is in nanometers
    weather : lsst.afw.coord Weather
        Class containing the measured temperature, pressure, and humidity
        at the observatory during an observation

    Returns
    -------
    float
        The difference of the refractive index of air from 1., calculated as (n_air - 1)*10^8
    """
    wave_num = 1E3/wavelength  # want wave number in units 1/micron

    dry_air_term = 2371.34 + (683939.7/(130. - wave_num**2.)) + (4547.3/(38.9 - wave_num**2.))

    wet_air_term = 6487.31 + 58.058*wave_num**2. - 0.71150*wave_num**4. + 0.08851*wave_num**6.

    return (dry_air_term * density_factor_dry(weather) +
            wet_air_term * density_factor_water(weather))


def density_factor_dry(weather):
    """Calculate dry air pressure term to refractive index calculation.

    This replicates equation 15 of Stone 1996 "An Accurate Method for Computing Atmospheric Refraction"

    Parameters
    ----------
    weather : lsst.afw.coord Weather, optional
        Class containing the measured temperature, pressure, and humidity
        at the observatory during an observation

    Returns
    -------
    float
        Returns the relative density of dry air at the given pressure and temperature.
    """
    temperature = _extract_temperature(weather, units_kelvin=True)
    water_vapor_pressure = humidity_to_pressure(weather)
    air_pressure = _extract_pressure(weather)
    dry_pressure = air_pressure - water_vapor_pressure

    eqn_1 = (dry_pressure/u_cds.mbar)*(57.90E-8 - 9.3250E-4*u.Kelvin/temperature +
                                       0.25844*u.Kelvin**2/temperature**2.)

    density_factor = float((1. + eqn_1)*(dry_pressure/u_cds.mbar)/(temperature/u.Kelvin))

    return density_factor


def density_factor_water(weather):
    """Calculate water vapor pressure term to refractive index calculation.

    This replicates equation 16 of Stone 1996 "An Accurate Method for Computing Atmospheric Refraction"

    Parameters
    ----------
    weather : lsst.afw.coord Weather, optional
        Class containing the measured temperature, pressure, and humidity
        at the observatory during an observation

    Returns
    -------
    float
        Returns the relative density of water vapor at the given pressure and temperature.
    """
    temperature = _extract_temperature(weather, units_kelvin=True)
    water_vapor_pressure = humidity_to_pressure(weather)

    eqn_1 = float(-2.37321E-3 + 2.23366*u.Kelvin/temperature -
                  710.792*u.Kelvin**2/temperature**2. +
                  7.75141E-4*u.Kelvin**3/temperature**3.)

    eqn_2 = float(water_vapor_pressure/u_cds.mbar)*(1. + 3.7E-4*water_vapor_pressure/u_cds.mbar)

    relative_density = float(water_vapor_pressure*u.Kelvin/(temperature*u_cds.mbar))
    density_factor = (1 + eqn_2*eqn_1)*relative_density

    return density_factor


def humidity_to_pressure(weather):
    """Simple function that converts humidity and temperature to water vapor pressure.

    This replicates equations 18 & 20 of Stone 1996 "An Accurate Method for Computing Atmospheric Refraction"

    Parameters
    ----------
    weather : lsst.afw.coord Weather
        Class containing the measured temperature, pressure, and humidity
        at the observatory during an observation

    Returns
    -------
    float
        The water vapor pressure in millibar, calculated from the given humidity and temperature.
    """
    if np.isnan(weather.getHumidity()):
        humidity = lsst_weather.getHumidity()
    else:
        humidity = weather.getHumidity()
    x = np.log(humidity/100.0)
    temperature = _extract_temperature(weather)
    eqn_1 = (temperature + 238.3*u.Celsius)*x + 17.2694*temperature
    eqn_2 = (temperature + 238.3*u.Celsius)*(17.2694 - x) - 17.2694*temperature
    dew_point = 238.3*float(eqn_1/eqn_2)

    water_vapor_pressure = (4.50874 + 0.341724*dew_point + 0.0106778*dew_point**2 + 0.184889E-3*dew_point**3 +
                            0.238294E-5*dew_point**4 + 0.203447E-7*dew_point**5)*133.32239*u.pascal

    return water_vapor_pressure


def _extract_temperature(weather, units_kelvin=False):
    """Thin wrapper to return the measured temperature from an observation with astropy units attached.

    Parameters
    ----------
    weather : lsst.afw.coord Weather
        Class containing the measured temperature, pressure, and humidity
        at the observatory during an observation
    units_kelvin : bool, optional
        Set to True to return the temperature in Kelvin instead of Celsius
        This is needed because Astropy can't easily convert between Kelvin and Celsius.

    Returns
    -------
    astropy.units
        The temperature in Celsius, unless `unit_kelvin` is set.
    """
    temperature = weather.getAirTemperature()
    if np.isnan(temperature):
        temperature = lsst_weather.getAirTemperature()*u.Celsius
    else:
        temperature *= u.Celsius
    if units_kelvin:
        temperature = temperature.to(u.Kelvin, equivalencies=u.temperature())
    return temperature


def _extract_pressure(weather):
    """Thin wrapper to return the measured pressure from an observation with astropy units attached.

    Parameters
    ----------
    weather : lsst.afw.coord Weather
        Class containing the measured temperature, pressure, and humidity
        at the observatory during an observation

    Returns
    -------
    astropy.units
        The air pressure in pascals.
    """
    pressure = weather.getAirPressure()
    if np.isnan(pressure):
        pressure = lsst_weather.getAirPressure()*u.pascal
    else:
        pressure *= u.pascal
    return pressure
