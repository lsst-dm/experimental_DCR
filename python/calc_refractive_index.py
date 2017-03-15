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
from astropy import units as u
from astropy.units import cds as u_cds
import numpy as np

from lsst.afw.geom import Angle
from .lsst_defaults import lsst_observatory, lsst_weather
__all__ = ("refraction", "diff_refraction")


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

    temperature_Kelvin = _extract_temperature(weather, units_kelvin=True)
    atmos_scaleheight_ratio = float(4.5908E-6/u.Kelvin*temperature_Kelvin)

    # Account for oblate Earth
    relative_gravity = (1. + 0.005302*np.sin(latitude.asRadians())**2. -
                        0.00000583*np.sin(2.*latitude.asRadians())**2. - 0.000000315*altitude)

    tanZ = np.tan(zenith_angle.asRadians())

    atmos_term_1 = reduced_n*relative_gravity*(1. - atmos_scaleheight_ratio)
    atmos_term_2 = reduced_n*relative_gravity*(atmos_scaleheight_ratio - reduced_n/2.)
    result = Angle(atmos_term_1*tanZ + atmos_term_2*tanZ**3.)
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
    temperature_Kelvin = _extract_temperature(weather, units_kelvin=True)
    water_vapor_pressure = humidity_to_pressure(weather)
    water_vapor_pressure_mbar = water_vapor_pressure.to(u_cds.mbar)
    air_pressure_mbar = _extract_pressure(weather, units_mbar=True)
    dry_pressure_mbar = air_pressure_mbar - water_vapor_pressure_mbar

    eqn_1 = (dry_pressure_mbar/u_cds.mbar)*(57.90E-8 - 9.3250E-4*u.Kelvin/temperature_Kelvin +
                                            0.25844*u.Kelvin**2/temperature_Kelvin**2.)

    density_factor = float((1. + eqn_1)*(dry_pressure_mbar/u_cds.mbar)/(temperature_Kelvin/u.Kelvin))

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
    temperature_Kelvin = _extract_temperature(weather, units_kelvin=True)
    water_vapor_pressure = humidity_to_pressure(weather)
    water_vapor_pressure_mbar = water_vapor_pressure.to(u_cds.mbar)

    eqn_1 = (-2.37321E-3 + 2.23366/temperature_Kelvin.value -
             710.792/temperature_Kelvin.value**2. + 7.75141E-4/temperature_Kelvin.value**3.)

    eqn_2 = water_vapor_pressure_mbar.value*(1. + 3.7E-4*water_vapor_pressure_mbar.value)

    density_factor = (1 + eqn_2*eqn_1)*water_vapor_pressure_mbar.value/temperature_Kelvin.value

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
    t_celsius = _extract_temperature(weather)
    eqn_1 = (t_celsius.value + 238.3)*x + 17.2694*t_celsius.value
    eqn_2 = (t_celsius.value + 238.3)*(17.2694 - x) - 17.2694*t_celsius.value
    dew_point = 238.3*eqn_1/eqn_2

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


def _extract_pressure(weather, units_mbar=False):
    """Thin wrapper to return the measured pressure from an observation with astropy units attached.

    Parameters
    ----------
    weather : lsst.afw.coord Weather
        Class containing the measured temperature, pressure, and humidity
        at the observatory during an observation
    units_mbar : bool, optional
        Set to True to return the pressure in millibar instead of pascals.

    Returns
    -------
    astropy.units
        The air pressure in pascals, unless `units_mbar` is set.
    """
    pressure = weather.getAirPressure()
    if np.isnan(pressure):
        pressure = lsst_weather.getAirPressure()*u.pascal
    else:
        pressure *= u.pascal
    if units_mbar:
        pressure = pressure.to(u_cds.mbar)
    return pressure
