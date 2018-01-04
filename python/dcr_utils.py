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

from builtins import range
import numpy as np
from scipy import constants

from lsst.afw.geom import Angle
import lsst.afw.image as afwImage
import lsst.afw.math as afwMath

__all__ = ["calculate_hour_angle", "parallactic_angle", "wrap_warpExposure",
           "calculate_rotation_angle"]


class BandpassHelper(object):
    def __init__(self, filter_name='g', profile='lsstSim', wavelen_step=None):
        """A dummy bandpass object that mimics Bandpass from sims_photUtils.

        This approximates the filter profile with the supplied function.

        Parameters
        ----------
        filter_name : str, optional
            Common name of the filter used. For LSST, use u, g, r, i, z, or y
        profile : str, optional
            Name of the filter profile approximation to use.
            The defualt profile is a semicircle.
        wavelen_step : float, optional
            Wavelength resolution in nm.
        """
        if wavelen_step is None:
            wavelen_step = 1.0
        band_dict = {'u': (324.0, 395.0), 'g': (405.0, 552.0), 'r': (552.0, 691.0),
                     'i': (818.0, 921.0), 'z': (922.0, 997.0), 'y': (975.0, 1075.0)}
        band_range = band_dict[filter_name]
        self.wavelen_min = band_range[0]
        self.wavelen_max = band_range[1]
        self.wavelen = np.arange(self.wavelen_min, self.wavelen_max + wavelen_step, wavelen_step)
        self.wavelen_step = wavelen_step
        n_step = len(self.wavelen)
        self.phi = None
        if profile == 'semi':
            self.sb = np.sqrt(1. - (2.*np.arange(n_step)/(n_step - 1) - 1.)**2.)
        elif profile == 'quadratic':
            self.sb = -(self.wavelen - np.median(self.wavelen))**2.
            self.sb -= np.min(self.sb)
            self.sb /= np.max(self.sb)
        elif profile == 'lsstSim':
            data_file = str("test_data/LsstSim_bandpass_%s.npy" % filter_name)
            self.wavelen, self.sb = np.load(data_file)
        else:
            self.sb = np.ones(n_step)
        self.sbTophi()

    def calc_eff_wavelen(self, wavelength_min=None, wavelength_max=None):
        """Calculate effective wavelengths for filters.

        Parameters
        ----------
        wavelength_min : float, optional
            Starting wavelength, in nm
        wavelength_max : float, optional
            End wavelength, in nm

        Returns
        -------
        Returns the weighted average wavelength within the range given, taken over the bandpass.
        """
        if self.phi is None:
            self.sbTophi()
        if wavelength_min is None:
            wavelength_min = np.min(self.wavelen)
        if wavelength_max is None:
            wavelength_max = np.max(self.wavelen)
        w_inds = (self.wavelen >= wavelength_min) & (self.wavelen <= wavelength_max)
        effwavelenphi = (self.wavelen[w_inds]*self.phi[w_inds]).sum()/self.phi[w_inds].sum()
        return effwavelenphi

    def calc_bandwidth(self):
        """Calculate the bandwidth of a filter."""
        f0 = constants.speed_of_light/(self.wavelen_min*1.0e-9)
        f1 = constants.speed_of_light/(self.wavelen_max*1.0e-9)
        f_cen = constants.speed_of_light/(self.calc_eff_wavelen()*1.0e-9)
        return(f_cen*2.0*(f0 - f1)/(f0 + f1))

    def getBandpass(self):
        wavelen = np.copy(self.wavelen)
        sb = np.copy(self.sb)
        return wavelen, sb

    def sbTophi(self):
        """
        Calculate and set phi - the normalized system response.

        This function only updates self.phi.
        Copied verbatim from sims_photUtils.
        """
        # The definition of phi = (Sb/wavelength)/\int(Sb/wavelength)dlambda.
        # Due to definition of class, self.sb and self.wavelen are guaranteed equal-gridded.
        dlambda = self.wavelen[1]-self.wavelen[0]
        self.phi = self.sb/self.wavelen
        # Normalize phi so that the integral of phi is 1.
        phisum = self.phi.sum()
        if phisum < 1e-300:
            raise Exception("Phi is poorly defined (nearly 0) over bandpass range.")
        norm = phisum * dlambda
        self.phi = self.phi / norm
        return


def kernel_1d(offset, size, n_substep=None, lanczos=None, debug_sinc=False, useInverse=False, weights=None):
    """Pre-compute the 1D sinc function values along each axis.

        Calculate the kernel as a simple numerical integration over the width of the offset.

        Parameters
        ----------
        offset : named tuple
            Tuple of start/end pixel offsets of dft locations along single axis (either x or y)
        size : int
            Dimension in pixels of the given axis.
        n_substep : int, optional
            Number of points in the numerical integration. Default is 1.
        lanczos : int, optional
            If set, the order of lanczos interpolation to use.
        debug_sinc : bool, optional
            Set to use a simple linear interpolation between nearest neighbors, instead of a sinc kernel.
        useInverse : bool, optional
            If set, calculate the kernel with the reverse of the supplied shift.
        weights : np.ndarray, optional
            Array of values to weight the substeps by. Will be interpolated to `n_substep` points.
    =
        Returns
        -------
        np.ndarray
            An array containing the values of the calculated kernel.
    """
    if n_substep is None:
        n_substep = 1
    else:
        n_substep = int(n_substep)
    pi = np.pi
    pix = np.arange(size, dtype=np.float64)
    tol = 1e-4

    kernel = np.zeros(size, dtype=np.float64)
    if weights is None:
        weights_interp = np.ones(n_substep)
    else:
        interp_x = np.linspace(0, len(weights), num=n_substep)
        weights_interp = np.interp(interp_x, np.arange(len(weights)), weights)
        weights_interp *= np.sum(weights)/np.sum(weights_interp)
    for n in range(n_substep):
        if useInverse:
            loc = size/2. + (-offset.start*(n_substep - (n + 0.5)) - offset.end*(n + 0.5))/n_substep
        else:
            loc = size/2. + (offset.start*(n_substep - (n + 0.5)) + offset.end*(n + 0.5))/n_substep
        if np.abs(loc - np.round(loc)) < tol:
            kernel[int(np.round(loc))] += weights_interp[n]
        else:
            if debug_sinc:
                i_low = int(np.floor(loc))
                i_high = i_low + 1
                frac_high = loc - i_low
                frac_low = 1. - frac_high
                kernel[i_low] += weights_interp[n]*frac_low
                kernel[i_high] += weights_interp[n]*frac_high
            else:
                x = pi*(pix - loc)
                if lanczos is None:
                    kernel += weights_interp[n]*np.sin(x)/x
                else:
                    kernel += weights_interp[n]*(np.sin(x)/x)*(np.sin(x/lanczos)/(x/lanczos))
    return kernel/np.sum(weights_interp)


def fft_shift_convolve(image, shift, n_substep=100, useInverse=False, weights=None):
    """Shift an image using a Fourier space convolution.

    Parameters
    ----------
    image : np.ndarray
        Input image to be shifted.
    shift :  named tuple
        Tuple of x and y tuples of start/end pixel offsets of dft locations.
    n_substep : int, optional
            Number of points in the numerical integration. Default is 100.
    useInverse : bool, optional
            If set, calculate the kernel with the reverse of the supplied shift.
    weights : np.ndarray, optional
            Array of values to weight the substeps by. Will be interpolated to `n_substep` points.

    Returns
    -------
    np.ndarray
        The shifted image.
    """
    y_size, x_size = image.shape
    # A 4th order lanczos kernel preserves the first few sidelobes of the sinc function,
    #   but ensures that artifacts from any poorly modeled sources
    #   are damped within the typical psf footprint size.
    lanczos_order = 4
    kernel_x = kernel_1d(shift.dx, x_size, n_substep=n_substep, debug_sinc=False,
                         useInverse=useInverse, weights=weights, lanczos=lanczos_order)
    kernel_y = kernel_1d(shift.dy, y_size, n_substep=n_substep, debug_sinc=False,
                         useInverse=useInverse, weights=weights, lanczos=lanczos_order)
    kernel = np.einsum('i,j->ij', kernel_y, kernel_x)
    fft_image = np.fft.rfft2(image)
    fft_kernel = np.fft.rfft2(kernel)
    fft_image *= fft_kernel
    return_image = np.fft.fftshift(np.fft.irfft2(fft_image))
    return return_image


def calculate_hour_angle(elevation, dec, latitude):
    """Compute the hour angle.

    Parameters
    ----------
    elevation : lsst.afw.geom.Angle
        Elevation angle of the observation.
    dec : lsst.afw.geom.Angle
        Declination of the observation.
    latitude : lsst.afw.geom.Angle
        Latitude of the observatory.

    Returns
    -------
    lsst.afw.geom.Angle
        The hour angle.
    """
    ha_term1 = np.sin(elevation.asRadians())
    ha_term2 = np.sin(dec.asRadians())*np.sin(latitude.asRadians())
    ha_term3 = np.cos(dec.asRadians())*np.cos(latitude.asRadians())
    if (ha_term1 - ha_term2) > ha_term3:
        # Inexact values can lead to singularities close to 1.
        # Those values should correspond to locations straight overhead, or through the ground.
        # Assuming these are real observations, we choose the overhead option.
        hour_angle = 0.
    else:
        hour_angle = np.arccos((ha_term1 - ha_term2) / ha_term3)
    return Angle(hour_angle)


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
        A rotation angle of 0 degrees is defined with North along the +y axis and East along the +x axis.
        A rotation angle of 90 degrees is defined with North along the +x axis and East along the -y axis.
    """
    visitInfo = exposure.getInfo().getVisitInfo()

    az = visitInfo.getBoresightAzAlt().getLongitude()
    hour_angle = visitInfo.getBoresightHourAngle()
    # Some simulated data contains invalid hour_angle metadata.
    # Once DM-9900 is completed, invalid data should instead raise an exception.
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
