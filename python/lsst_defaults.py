"""Define default values used in the code and tests."""

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

from astropy import units as u

from lsst.afw.coord import Observatory, Weather
from lsst.afw.geom import degrees

lsst_lat = -30.244639*degrees
lsst_lon = -70.749417*degrees
lsst_alt = 2663.
lsst_temperature = 20.*u.Celsius  # in degrees Celcius
lsst_humidity = 40.  # in percent
lsst_pressure = 73892.*u.pascal

lsst_weather = Weather(lsst_temperature.value, lsst_pressure.value, lsst_humidity)
lsst_observatory = Observatory(lsst_lon, lsst_lat, lsst_alt)
