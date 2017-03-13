#
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
"""Calculate atmospheric refraction under different observing conditions."""
from __future__ import print_function, division, absolute_import
import numpy as np

from lsst.afw.geom import Angle
from .lsst_defaults import lsst_observatory, lsst_temperature, lsst_humidity
__all__ = ("refraction", "diff_refraction")

lsst_lat = Angle(np.radians(-30.244639))
lsst_alt = 2663.


def refraction(wavelength, zenith_angle, atmospheric_pressure=1.,
               temperature=lsst_temperature, humidity=lsst_humidity,
               observatory=lsst_observatory):
    """Calculate overall refraction under atmospheric and observing conditions.

    Parameters
    ----------
    wavelength : float
        wavelength is in nm (valid for 230.2 < wavelength < 2058.6)
    zenith_angle : lsst.afw.geom Angle
        Zenith angle of the observation, as an Angle.
    atmospheric_pressure : float, optional
        Atmospheric pressure is in atmospheres.
    temperature : float
        Observatory site temperature in Celcius (valid for -20 < T < 50)
    humidity : float, optional
    observatory : lsst.afw.coord.coordLib.Observatory, optional
        Class containing the longitude, latitude, and altitude of the observatory.

    Returns
    -------
    lsst.afw.geom Angle
        The angular refraction for light of the given wavelength, under the given observing conditions.
    """
    latitude = observatory.getLatitude()
    altitude = observatory.getElevation()
    temperature_Kelvin = temperature + 273.15
    water_vapor_pressure = humidity_to_pressure(humidity=humidity, temperature=temperature)

    atm_to_millibar = 760.*1.333224
    dry_pressure = atm_to_millibar*atmospheric_pressure - water_vapor_pressure

    reduced_n = n_delta(wavelength, dry_pressure, water_vapor_pressure, temperature_Kelvin)*1E-8

    atmos_scaleheight_ratio = 4.5908E-6*temperature_Kelvin

    # Account for oblate Earth
    relative_gravity = (1. + 0.005302*np.sin(latitude.asRadians())**2. -
                        0.00000583*np.sin(2.*latitude.asRadians())**2. - 0.000000315*altitude)

    tanZ = np.tan(zenith_angle.asRadians())

    atmos_term_1 = reduced_n*relative_gravity*(1. - atmos_scaleheight_ratio)
    atmos_term_2 = reduced_n*relative_gravity*(atmos_scaleheight_ratio - reduced_n/2.)
    result = Angle(atmos_term_1*tanZ + atmos_term_2*tanZ**3.)
    return result


def diff_refraction(wavelength, wavelength_ref, zenith_angle, atmospheric_pressure=1.,
                    temperature=lsst_temperature, humidity=lsst_humidity,
                    observatory=lsst_observatory):
    """Calculate the differential refraction between two wavelengths.

    Parameters
    ----------
    wavelength : float
        wavelength is in nm (valid for 230.2 < wavelength < 2058.6)
    wavelength_ref : TYPE
        Description
    zenith_angle : lsst.afw.geom Angle
        Zenith angle of the observation, as an Angle.
    atmospheric_pressure : float, optional
        Atmospheric pressure is in atmospheres.
    temperature : float, optional
        Observatory site temperature in Celcius (valid for -20 < T < 50)
    humidity : float, optional
        Observatory site humidity, in percent (0-100)
    observatory : lsst.afw.coord.coordLib.Observatory, optional
        Class containing the longitude, latitude, and altitude of the observatory.

    Returns
    -------
    lsst.afw.geom Angle
        The differential angular refraction of light between the two given wavelengths,
        under the given observing conditions
    """
    refraction_start = refraction(wavelength, zenith_angle, atmospheric_pressure, temperature,
                                  humidity=humidity, observatory=observatory)
    refraction_end = refraction(wavelength_ref, zenith_angle, atmospheric_pressure, temperature,
                                humidity=humidity, observatory=observatory)
    return refraction_start - refraction_end


def n_delta(wavelength, dry_pressure, water_vapor_pressure, temperature):
    """Calculate the differential refractive index of air.

    The differential refractive index is the difference of the refractive index from 1.,
    multiplied by 1E8 to simplify the notation and equations.

    Parameters
    ----------
    wavelength : float
        wavelength is in nanometers
    dry_pressure : float
        Atmospheric pressure, excluding water vapor. In millibar.
    water_vapor_pressure : float
        Pressure of the water vapor component of the atmosphere, in millibar.
    temperature : float
        Observatory site temperature in Kelvin

    Returns
    -------
    float
        The difference of the refractive index of air from 1., multiplied by 1E8.
    """
    wave_num = 1E3/wavelength  # want wave number in units 1/micron

    dry_air_term = 2371.34 + (683939.7/(130. - wave_num**2.)) + (4547.3/(38.9 - wave_num**2.))

    wet_air_term = 6487.31 + 58.058*wave_num**2. - 0.71150*wave_num**4. + 0.08851*wave_num**6.

    return (dry_air_term * density_factor_dry(dry_pressure, temperature) +
            wet_air_term * density_factor_water(water_vapor_pressure, temperature))


def density_factor_dry(dry_pressure, temperature):
    """Calculate dry air pressure term to refractive index calculation.

    Parameters
    ----------
    dry_pressure : float
        Atmospheric pressure, excluding water vapor. In millibar.
    temperature : float
        Observatory site temperature in Kelvin

    Returns
    -------
    float
        Returns the relative density of dry air at the given pressure and temperature.
    """
    eqn_1 = 1. + dry_pressure*(57.90E-8 - 9.3250E-4/temperature + 0.25844/temperature**2.)

    return eqn_1*dry_pressure/temperature


def density_factor_water(water_vapor_pressure, temperature):
    """Calculate water vapor pressure term to refractive index calculation.

    Parameters
    ----------
    water_vapor_pressure : float
        Pressure of the water vapor component of the atmosphere, in millibar.
    temperature : float
        Observatory site temperature in Kelvin

    Returns
    -------
    float
        Returns the relative density of water vapor at the given pressure and temperature.
    """
    eqn_1 = -2.37321E-3 + 2.23366/temperature - 710.792/temperature**2. + 7.75141E-4/temperature**3.

    eqn_2 = 1. + water_vapor_pressure*(1. + 3.7E-4*water_vapor_pressure)*eqn_1

    return eqn_2*water_vapor_pressure/temperature


def humidity_to_pressure(humidity, temperature):
    """Simple function that converts humidity and temperature to water vapor pressure.

    Parameters
    ----------
    humidity : float
        Observatory site humidity, in percent (0-100)
    temperature : float
        Observatory site temperature in Celcius

    Returns
    -------
    float
        The water vapor pressure in millibar, calculated from the given humidity and temperature.
    """
    pascals_to_mbar = 60.*1.333224/101325.0
    temperature_Kelvin = temperature + 273.15
    exponent_term = 77.3450 + 0.0057*temperature_Kelvin - 7235.0/temperature_Kelvin
    saturation_pressure = (pascals_to_mbar*np.exp(exponent_term)/temperature_Kelvin**8.2)
    return (humidity/100.0)*saturation_pressure
