"""Attempts to create airmass-matched template images for existing images in an LSST repository."""

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

from builtins import zip
from builtins import range
import numpy as np
import scipy.ndimage as ndimage
from scipy.ndimage.interpolation import shift as scipy_shift

import lsst.afw.image as afwImage
import lsst.afw.math as afwMath
import lsst.meas.algorithms as measAlg

from .dcr_utils import calculate_rotation_angle  # Likely can remove soon
from .dcr_utils import fft_shift_convolve
from .dcr_utils import wrap_warpExposure
# from .dcr_utils import solve_model
from .generateTemplate import GenerateTemplate

__all__ = ["BuildDcrCoadd"]


class BuildDcrCoadd(GenerateTemplate):
    """Class that loads LSST calibrated exposures and produces airmass-matched template images.

    Input exposures are read with a butler, and an initial model is made by coadding the images.
    An improved model of the sky is built for a series of sub-bands within the full bandwidth of the filter
    used for the observations by iteratively forward-modeling the template using the calculated
    Differential Chromatic Refration for the exposures in each sub-band.

    Attributes
    ----------
    bandpass : BandpassHelper object
        Bandpass object returned by `load_bandpass`
    bandpass_highres : BandpassHelper object
        A second Bandpass object returned by load_bandpass, at the highest resolution available.
    bbox : lsst.afw.geom.Box2I object
        A bounding box.
    butler : lsst.daf.persistence Butler object
        The butler handles persisting and depersisting data to and from a repository.
    debug : bool
        Temporary debugging option.
        If set, calculations are performed on only a small region of the full images.
    default_repository : str
        Full path to repository with the data
    exposure_time : float
        Length of the exposure, in seconds.
    exposures : list
        List of input exposures used to calculate the model.
    filter_name : str
        Name of the bandpass-defining filter of the data. Expected values are u,g,r,i,z,y.
    mask : np.ndarray
        Combined bit plane mask of the model, which is used as the mask plane for generated templates.
    model : list of np.ndarrays
        The DCR model to be used to generate templates. Contains one array for each wavelength step.
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
    psf : lsst.meas.algorithms KernelPsf object
        Representation of the point spread function (PSF) of the model.
    psf_size : int
        Dimension of the PSF, in pixels.
    wcs : lsst.afw.image Wcs object
        World Coordinate System of the model.
    weights : np.ndarray
        Weights of the model, calculated from the combined inverse variance of the input exposures.
    x_size : int
        Width of the model, in pixels.
    y_size : int
        Height of the model, in pixels.

    Example
    -------

    Set up:
    dcrCoadd = BuildDcrCoadd(n_step=3, input_repository="./test_data/",
                             obsids=np.arange(100, 124, 3), filter_name='g')

    Generate the model:
    dcrCoadd.build_model(max_iter=10)

    Use the model to make matched templates for several observations:
    template_exposure_gen = dcrCoadd.generate_templates_from_model(obsids=[108, 109, 110],
                                                                   output_repository="./test_data_templates/")
    im_arr = []
    for exp in template_exposure_gen:
        # The exposures are written to the given ``output_repository``
        # and returned as the yield value of the generator
        im_arr.append(exp.getMaskedImage().getImage().getArray())
    """

    def __init__(self, obsids=None, input_repository='.', filter_name='g',
                 wavelength_step=10., n_step=None, exposures=None,
                 warp=False, debug_mode=False, verbose=False):
        """Load images from the repository and set up parameters.

        Parameters
        ----------
        obsids : int or list of ints, optional
            The observation IDs of the data to load. Not used if `exposures` is set.
        input_repository : str, optional
            Full path to repository with the data. Defaults to working directory
        filter_name : str, optional
            Name of the bandpass-defining filter of the data. Expected values are u,g,r,i,z,y.
        wavelength_step : float, optional
            Wavelength resolution in nm, also the wavelength range of each sub-band plane.
            Overridden if `n_step` is supplied.
        n_step : int, optional
            Number of sub-band planes to use. Takes precendence over `wavelength_step`.
        exposures : List of lsst.afw.image.ExposureF objects, optional
            List of exposures to use to calculate the model.
        warp : bool, optional
            Set to true if the exposures have different wcs from the model.
            If True, the generated templates will be warped to match the wcs of each exposure.
        debug_mode : bool, optional
            Temporary debugging option.
        verbose : bool, optional
            Set to print additional status messages.

        Raises
        ------
        ValueError
            If  no valid exposures are found in `input_repository` and `exposures` is not set.
        """
        self.filter_name = filter_name
        self.default_repository = input_repository
        self.butler = None  # Placeholder. The butler is instantiated in read_exposures.
        if exposures is None:
            exposures = self.read_exposures(obsids, input_repository=input_repository)
        self.exposures = [calexp for calexp in exposures]

        if len(self.exposures) == 0:
            raise ValueError("No valid exposures found.")

        self.debug = debug_mode
        self.n_images = len(self.exposures)
        psf_size_arr = []
        hour_angle_arr = []
        ref_exp_i = 0
        self.bbox = self.exposures[ref_exp_i].getBBox()
        self.wcs = self.exposures[ref_exp_i].getWcs()
        self.observatory = self.exposures[ref_exp_i].getInfo().getVisitInfo().getObservatory()

        for i, calexp in enumerate(self.exposures):
            psf_size_arr.append(calexp.getPsf().computeKernelImage().getArray().shape[0])
            visitInfo = calexp.getInfo().getVisitInfo()
            hour_angle_arr.append(visitInfo.getBoresightHourAngle().asRadians())

            if verbose:
                obsid = visitInfo.getExposureId()
                print("Working on observation %s" % obsid)
                el = visitInfo.getBoresightAzAlt().getLatitude()
                az = visitInfo.getBoresightAzAlt().getLongitude()
                ra = visitInfo.getBoresightRaDec().getLongitude()
                dec = visitInfo.getBoresightRaDec().getLatitude()
                rotation_angle = calculate_rotation_angle(calexp)
                print("El: %2.4f, Az: %2.4f, RA: %2.4f, Dec: %2.4f, Rotation: %2.4f" %
                      (el.asDegrees(), az.asDegrees(), ra.asDegrees(),
                       dec.asDegrees(), rotation_angle.asDegrees()))

            if (i != ref_exp_i) & warp:
                wrap_warpExposure(calexp, self.wcs, self.bbox)

        if np.any(np.isnan(hour_angle_arr)):
            print("Warning: invalid hour angle in metadata. Azimuth will be used instead.")
        x_size, y_size = self.exposures[ref_exp_i].getDimensions()
        self.x_size = x_size
        self.y_size = y_size
        self.pixel_scale = self.exposures[ref_exp_i].getWcs().pixelScale()
        self.exposure_time = self.exposures[ref_exp_i].getInfo().getVisitInfo().getExposureTime()
        # Use the largest input PSF to calculate the fiducial PSF so that no information is lost.
        self.psf_size = int(np.max(psf_size_arr))
        self.psf = None
        self.mask = self._combine_masks()

        bandpass = self.load_bandpass(filter_name=filter_name, wavelength_step=wavelength_step)
        bandpass_highres = self.load_bandpass(filter_name=filter_name, wavelength_step=None)
        if n_step is not None:
            wavelength_step = (bandpass.wavelen_max - bandpass.wavelen_min) / n_step
            bandpass = self.load_bandpass(filter_name=filter_name, wavelength_step=wavelength_step)
        else:
            n_step = int(np.ceil((bandpass.wavelen_max - bandpass.wavelen_min) / bandpass.wavelen_step))
        if n_step >= self.n_images:
            print("Warning! Under-constrained system. Reducing number of frequency planes.")
            wavelength_step *= n_step / self.n_images
            bandpass = self.load_bandpass(filter_name=filter_name, wavelength_step=wavelength_step)
            n_step = int(np.ceil((bandpass.wavelen_max - bandpass.wavelen_min) / bandpass.wavelen_step))
        self.n_step = n_step
        self.bandpass = bandpass
        self.bandpass_highres = bandpass_highres

    def calc_psf_model(self):
        """Calculate the fiducial psf from a given set of exposures, accounting for DCR.

        Parameters
        ----------

        Returns
        -------
        None
            Sets self.psf with a lsst.meas.algorithms KernelPsf object.
        """
        psf_vals = np.zeros((self.psf_size, self.psf_size))
        psf_weights = 0.
        for exp in self.exposures:
            airmass_weight = 1./exp.getInfo().getVisitInfo().getBoresightAirmass()
            # Use the measured PSF as the solution of the shifted PSFs.
            psf_img = exp.getPsf().computeKernelImage().getArray()
            psf_y_size, psf_x_size = psf_img.shape
            if self.psf_size > psf_x_size:
                i0 = int(self.psf_size//2 - psf_x_size//2)
                i1 = i0 + psf_x_size
                psf_img_use = np.zeros((self.psf_size, self.psf_size))
                psf_img_use[i0:i1, i0:i1] = psf_img
            else:
                i0 = int(psf_x_size//2 - self.psf_size//2)
                i1 = i0 + self.psf_size
                psf_img_use = psf_img[i0:i1, i0:i1]
            psf_vals += psf_img_use*airmass_weight
            psf_weights += airmass_weight
        psf_vals /= psf_weights
        psf_image = afwImage.ImageD(self.psf_size, self.psf_size)
        psf_image.getArray()[:, :] = psf_vals
        psfK = afwMath.FixedKernel(psf_image)
        self.psf = measAlg.KernelPsf(psfK)
        return None

    def build_model(self, verbose=True, max_iter=None, min_iter=None, clamp=None, use_variance=True,
                    frequency_regularization=False, max_slope=None,
                    initial_solution=None,
                    test_convergence=False, convergence_threshold=None,
                    spatial_filter=None, airmass_weight=False,
                    stretch_threshold=None,
                    divergence_threshold=None, obs_divergence_threshold=None,
                    refine_solution=False, refine_max_iter=None, refine_convergence_threshold=None):
        """Build a model of the sky in multiple sub-bands.

        Parameters
        ----------
        verbose : `bool`, optional
            Print additional status messages.
        max_iter : `int`, optional
            The maximum number of iterations of forward modeling allowed.
        min_iter : int, optional
            The minimum number of iterations of forward modeling before checking for convergence.
        clamp : float, optional
            Restrict new solutions from being more than a factor of ``clamp`` different from the last solution
            before `gain` is applied.
            The default value is 3, chosen so that a gain of 1 restricts the change of the solution between
            iterations to less than a factor of 2.
        use_variance : bool, optional
            Set to weight pixels by their inverse variance when combining images.
        frequency_regularization : bool, optional
            Set to restrict variations between frequency planes
        max_slope : float, optional
            Maximum slope to allow between sub-band model planes.
            Only used if ``frequency_regularization`` is set.
        initial_solution : list of np.ndarrays, optional
            Optionally supply the starting point of the iterative solution.
        test_convergence : bool, optional
            If True, then matched templates will be generated for each image for every iteration,
            and the difference with the image will be checked to see if it is less than the previous iteration
            Any images where the difference is increasing will be excluded from the next iteration.
        convergence_threshold : float, optional
            Return once the convergence metric changes by less than ``convergence_threshold``
            between iterations.
        spatial_filter : bool, optional
            For testing only! Apply a spatial filter to the solution at each iteration of forward modeling.
        airmass_weight : bool, optional
            Set to weight each observation by its airmass.
        stretch_threshold : float, optional
            Set to simulate the effect of DCR across each sub-band by stretching the model.
            If any sub-band has DCR greater than this amount for an exposure the finite bandwidth
            FFT-based shift will be used for that instance rather than a simple shift.
        divergence_threshold : float, optional
            Maximum increase in the difference between old and new solutions before
            exiting from lack of convergence.
        obs_divergence_threshold : float, optional
            Maximum degradation of convergence for one observation before it is cut.
        refine_solution : bool, optional
            If set, recalculate the initial solution from the model, and repeat iteration.
        refine_max_iter : bool, optional
            The maximum number of iterations of forward modeling allowed when refining the solution.
        refine_convergence_threshold : None, optional
            Optionally set a separate convergence threshold for a second iteration of refining the solution.
            If not set but `refine_solution` is set, the default is 1/10th of `convergence_threshold`.

        Returns
        -------
        bool
            Returns `True` if the convergence threshold was reached.
        """
        # Set up an initial guess with all model planes equal as a starting point of the iterative solution
        # The solution is initialized to 0. and not an array so that it can adapt
        # to the size of the array returned by _extract_image. This should only matter in debugging.
        if initial_solution is None:
            if verbose:
                print("Calculating initial solution...", end="")
            initial_solution = 0.
            initial_weights = 0.
            for exp in self.exposures:
                img, inverse_var = self._extract_image(exp, airmass_weight=True, use_variance=use_variance)
                initial_solution += img*inverse_var
                initial_weights += inverse_var

            weight_inds = initial_weights > 0
            initial_solution[weight_inds] /= initial_weights[weight_inds]
            wl_iter = self._wavelength_iterator(self.bandpass, use_midpoint=False)
            flux_out = []
            wl_in = self.bandpass_highres.wavelen
            flux_in = self.bandpass_highres.sb
            for wl_start, wl_end in wl_iter:
                wl_use = (wl_in >= wl_start) & (wl_in < wl_end)
                flux = np.sum(flux_in[wl_use])
                flux_out.append(flux)
            band_power = np.array(flux_out)
            band_power /= np.sum(band_power)
            initial_solution = [np.abs(initial_solution)*power for power in band_power]
            if verbose:
                print(" Done!")
        self.model_base = initial_solution
        # When debugging, the image returned by _extract_image might be cropped to speed up calculations.
        if self.debug:
            self.y_size, self.x_size = initial_solution[0].shape

        did_converge = self._build_model_subroutine(initial_solution, verbose=verbose,
                                                    min_iter=min_iter, max_iter=max_iter,
                                                    frequency_regularization=frequency_regularization,
                                                    clamp=clamp, max_slope=max_slope,
                                                    test_convergence=test_convergence,
                                                    convergence_threshold=convergence_threshold,
                                                    use_variance=use_variance,
                                                    spatial_filter=spatial_filter,
                                                    airmass_weight=airmass_weight,
                                                    stretch_threshold=stretch_threshold,
                                                    divergence_threshold=divergence_threshold,
                                                    obs_divergence_threshold=obs_divergence_threshold,
                                                    )
        if refine_solution:
            print("Refining model")
            if spatial_filter is None:
                spatial_filter_use = self.psf_size//2
            else:
                spatial_filter_use = spatial_filter
            initial_solution = self._model_spatial_filter(self.model, spatial_filter_use)
            if refine_max_iter is None:
                refine_max_iter = max_iter*2
            if refine_convergence_threshold is None:
                # If the refined convergence threshold is not set, simply drop the initial threshold by 10.
                refine_convergence_threshold = convergence_threshold/10.
            did_converge = self._build_model_subroutine(initial_solution, verbose=verbose,
                                                        max_iter=refine_max_iter, min_iter=min_iter,
                                                        frequency_regularization=frequency_regularization,
                                                        clamp=clamp, max_slope=max_slope,
                                                        test_convergence=test_convergence,
                                                        convergence_threshold=refine_convergence_threshold,
                                                        use_variance=use_variance,
                                                        spatial_filter=spatial_filter,
                                                        airmass_weight=airmass_weight,
                                                        stretch_threshold=stretch_threshold,
                                                        divergence_threshold=divergence_threshold,
                                                        obs_divergence_threshold=obs_divergence_threshold,
                                                        )

        if verbose:
            print("\nFinished building model.")
        return did_converge

    def _build_model_subroutine(self, initial_solution, verbose=True, max_iter=None, min_iter=None,
                                test_convergence=False, frequency_regularization=True, max_slope=None,
                                clamp=None, convergence_threshold=None, use_variance=True,
                                spatial_filter=None, airmass_weight=False, stretch_threshold=None,
                                divergence_threshold=None, obs_divergence_threshold=None):
        """Extract the math from building the model so it can be re-used.

        Parameters
        ----------
        initial_solution : float or np.ndarray
            The model to use as a starting point for iteration.
            If a float, then a constant value is used for all pixels.
        verbose : bool, optional
            Print additional status messages.
        max_iter : int, optional
            The maximum number of iterations of forward modeling allowed.
        min_iter : int, optional
            The minimum number of iterations of forward modeling before checking for convergence.
        test_convergence : bool, optional
            If True, then matched templates will be generated for each image for every iteration,
            and the difference with the image will be checked to see if it is less than the previous iteration
            Any images where the difference is increasing will be excluded from the next iteration.
        frequency_regularization : bool, optional
            Set to restrict variations between frequency planes
        max_slope : float, optional
            Maximum slope to allow between sub-band model planes.
        clamp : float, optional
            Restrict new solutions from being more than a factor of `clamp` different from the last solution.
        convergence_threshold : float, optional
            Return once the convergence metric changes by less than this amount between iterations.
        use_variance : bool, optional
            Set to weight pixels by their inverse variance when combining images.
        spatial_filter : bool, optional
            For testing only! Apply a spatial filter to the solution at each iteration of forward modeling.
        airmass_weight : bool, optional
            Set to weight each observation by its airmass.
        stretch_threshold : float, optional
            Set to simulate the effect of DCR across each sub-band by stretching the model.
            If any sub-band has DCR greater than this amount for an exposure the finite bandwidth
            FFT-based shift will be used for that instance rather than a simple shift.
        divergence_threshold : float, optional
            Maximum increase in the difference between old and new solutions before
            exiting from lack of convergence.
        obs_divergence_threshold : float, optional
            Maximum degradation of convergence for one observation before it is cut.

        Returns
        -------
        bool
            Returns `True` if the convergence threshold was reached.
        Sets self.model as a list of np.ndarrays
        Sets self.weights as a np.ndarray
        """
        if divergence_threshold is None:
            # A new solution may slightly degrade the overall convergence, but as long it does not get
            #   much worse than a few percent overall future iterations may still lead to an improvement.
            # From trial and error, 5% seems about right
            divergence_threshold = 1.05
        if obs_divergence_threshold is None:
            # Individual observations may degrade slightly even when the overall solution improves.
            # Allow a slight tolerance in that degradation before excluding them from the calculation.
            # From trial and error, 5% seems about right.
            obs_divergence_threshold = 1.05
        if clamp is None:
            # The value of clamp is chosen so that the solution never changes by
            #  more than a factor of 2 between iterations: if new = old*3 then (old + new)/2 = 2*old
            clamp = 3.
        if convergence_threshold is None:
            convergence_threshold = 1e-3
        # if stretch_threshold is None:
        #     stretch_threshold = 0.25  # Use a default of a quarter of a pixel
        min_images = self.n_step + 1
        if min_iter is None:
            min_iter = 2
        if max_iter is None:
            max_iter = 20
        last_solution = [np.zeros((self.y_size, self.x_size)) for f in range(self.n_step)]
        for f in range(self.n_step):
            last_solution[f] += initial_solution[f]

        if verbose:
            print("Fractional change per iteration:")
        last_convergence_metric_full = self.calc_model_metric(last_solution,
                                                              stretch_threshold=stretch_threshold)
        test_convergence_metric_full = last_convergence_metric_full
        last_convergence_metric = np.mean(last_convergence_metric_full)
        if verbose:
            print("Full initial convergence metric: ", last_convergence_metric_full)
            print("Initial convergence metric: %f" % last_convergence_metric)

        exp_cut = [False for exp_i in range(self.n_images)]
        final_soln_iter = None
        did_converge = False
        last_delta = 1.
        n_exp_cut = 0
        n_exp_cut_last = 0
        for sol_iter in range(int(max_iter)):
            new_solution, inverse_var_arr = self._calculate_new_model(last_solution, exp_cut,
                                                                      use_variance=use_variance,
                                                                      airmass_weight=airmass_weight,
                                                                      stretch_threshold=stretch_threshold)

            # Optionally restrict variations between frequency planes
            if frequency_regularization:
                self._regularize_model_solution(new_solution, self.bandpass, max_slope=max_slope)

            # Optionally restrict spatial variations in frequency structure
            if spatial_filter is not None:
                new_solution = self._model_spatial_filter(new_solution, spatial_filter, mask=None)

            # Restrict new solutions from being wildly different from the last solution
            if clamp > 0:
                self._clamp_model_solution(new_solution, last_solution, clamp)

            inds_use = inverse_var_arr[-1] > 0
            for f in range(self.n_step - 1):
                inds_use *= inverse_var_arr[f] > 0

            convergence_metric_full = self.calc_model_metric(new_solution,
                                                             stretch_threshold=stretch_threshold)
            convergence_metric = np.mean(convergence_metric_full[np.logical_not(exp_cut)])
            gain = last_convergence_metric/convergence_metric
            # Use the average of the new and last solution for the next iteration. This reduces oscillations.
            new_solution_use = [np.abs((last_solution[f] + gain*new_solution[f])/(1 + gain))
                                for f in range(self.n_step)]

            delta = (np.sum(np.abs([last_solution[f][inds_use] - new_solution_use[f][inds_use]
                                    for f in range(self.n_step)])) /
                     np.sum(np.abs([soln[inds_use] for soln in last_solution])))
            if verbose:
                print("Iteration %i: delta=%f" % (sol_iter, delta))
                print("Convergence-weighted gain used: %f" % gain)
            if n_exp_cut != n_exp_cut_last:
                # When an exposure is removed (or added back in), the next step size may be
                #   significantly larger without indicating that the solutions are diverging.
                # However, it is also possible for the solutions to diverge wildly once exposures start
                #   being cut, so we can't eliminate the divergence test entirely.
                # I allow a factor of 2, but the ideal threshold has not been investigated
                divergence_threshold_use = 2.
            else:
                divergence_threshold_use = divergence_threshold
            if delta > divergence_threshold_use*last_delta:
                print("BREAK from diverging model difference.")
                final_soln_iter = sol_iter - 1
                break
            else:
                last_delta = delta
            if test_convergence:
                convergence_metric_full = self.calc_model_metric(new_solution_use,
                                                                 stretch_threshold=stretch_threshold)
                if verbose:
                    print("Full convergence metric:", convergence_metric_full)
                if sol_iter >= min_iter:
                    exp_cut = convergence_metric_full > last_convergence_metric_full*obs_divergence_threshold
                    exp_cut1 = convergence_metric_full > test_convergence_metric_full*obs_divergence_threshold
                    exp_cut[exp_cut1] = True
                    # Only cut exposures that are beginning to diverge if they are also worse than most.
                    exp_cut_test = convergence_metric_full > np.median(convergence_metric_full)
                    exp_cut *= exp_cut_test
                else:
                    test_convergence_metric_full = last_convergence_metric_full

                n_exp_cut_last = n_exp_cut
                n_exp_cut = np.sum(exp_cut)
                if n_exp_cut > 0:
                    print("%i exposure(s) cut from lack of convergence: " % int(n_exp_cut),
                          (np.where(exp_cut))[0])
                    last_convergence_metric = np.mean(last_convergence_metric_full[np.logical_not(exp_cut)])
                if (self.n_images - n_exp_cut) < min_images:
                    print("Exiting iterative solution: Too few images left.")
                    final_soln_iter = sol_iter - 1
                    break
                last_convergence_metric_full = convergence_metric_full
                convergence_metric = np.mean(convergence_metric_full[np.logical_not(exp_cut)])
                conv_change = (1. - convergence_metric/last_convergence_metric)*100.
                print("Iteration %i convergence metric: %f (%f%% change)" %
                      (sol_iter, convergence_metric, conv_change))
                convergence_check2 = (1 - convergence_threshold)*last_convergence_metric

                if sol_iter > min_iter:
                    if convergence_metric > last_convergence_metric:
                        print("BREAK from lack of convergence")
                        final_soln_iter = sol_iter - 1
                        break
                    if convergence_metric > convergence_check2:
                        print("BREAK after reaching convergence threshold")
                        final_soln_iter = sol_iter
                        last_solution = new_solution_use
                        did_converge = True
                        break
                last_convergence_metric = convergence_metric
            last_solution = new_solution_use
        if final_soln_iter is None:
            final_soln_iter = sol_iter
        if verbose:
            print("Final solution from iteration: %i" % final_soln_iter)
        self.model = last_solution
        self.weights = np.sum(inverse_var_arr, axis=0)/self.n_step
        return did_converge

    def _calculate_new_model(self, last_solution, exp_cut=None, use_variance=True,
                             airmass_weight=False, stretch_threshold=None):
        """Sub-routine to calculate a new model from the residuals of forward-modeling the previous solution.

        Parameters
        ----------
        last_solution : list of np.ndarrays
            One np.ndarray for each model sub-band, from the previous iteration.
        exp_cut : List of bools
            Exposures that failed to converge in the previous iteration are flagged,
            and not included in the current iteration solution.
        use_variance : bool, optional
            Set to weight pixels by their inverse variance when combining images.
        airmass_weight : bool, optional
            Set to weight each observation by its airmass.
        stretch_threshold : float, optional
            Set to simulate the effect of DCR across each sub-band by stretching the model.
            If any sub-band has DCR greater than this amount for an exposure the finite bandwidth
            FFT-based shift will be used for that instance rather than a simple shift.

        Returns
        -------
        Tuple of two lists of np.ndarrays
            One np.ndarray for each model sub-band, and the associated inverse variance array.
        """
        residual_arr = [np.zeros((self.y_size, self.x_size)) for f in range(self.n_step)]
        inverse_var_arr = [np.zeros((self.y_size, self.x_size)) for f in range(self.n_step)]
        sub_weights = self._subband_weights()
        if exp_cut is None:
            exp_cut = [False for exp_i in range(self.n_images)]
        for exp_i, exp in enumerate(self.exposures):
            if exp_cut[exp_i]:
                continue
            img, inverse_var = self._extract_image(exp, use_variance=use_variance)

            visitInfo = exp.getInfo().getVisitInfo()
            el = visitInfo.getBoresightAzAlt().getLatitude()
            rotation_angle = calculate_rotation_angle(exp)
            weather = visitInfo.getWeather()
            dcr_gen = self._dcr_generator(self.bandpass, pixel_scale=self.pixel_scale,
                                          observatory=self.observatory, weather=weather,
                                          elevation=el, rotation_angle=rotation_angle,
                                          use_midpoint=False)
            if airmass_weight:
                inverse_var *= visitInfo.getBoresightAirmass()
            if use_variance:
                inverse_var = np.sqrt(inverse_var)
            dcr_list = [dcr for dcr in dcr_gen]
            last_model_shift = []
            if stretch_threshold is None:
                stretch_test = [False for dcr in dcr_list]
            else:
                stretch_test = [(abs(dcr.dy.start - dcr.dy.end) > stretch_threshold) or
                                (abs(dcr.dx.start - dcr.dx.end) > stretch_threshold) for dcr in dcr_list]

            for f, dcr in enumerate(dcr_list):
                if stretch_test[f]:
                    last_model_shift.append(fft_shift_convolve(last_solution[f], dcr, weights=sub_weights[f]))
                else:
                    shift = ((dcr.dy.start + dcr.dy.end)/2., (dcr.dx.start + dcr.dx.end)/2.)
                    last_model_shift.append(scipy_shift(last_solution[f], shift))
            for f, dcr in enumerate(dcr_list):
                last_model = np.zeros((self.y_size, self.x_size))
                for f2 in range(self.n_step):
                    if f2 != f:
                        last_model += last_model_shift[f2]
                img_residual = img - last_model
                # NOTE: We do NOT use the finite bandwidth shift here,
                #       since that would require a deconvolution.
                inv_shift = (-(dcr.dy.start + dcr.dy.end)/2., -(dcr.dx.start + dcr.dx.end)/2.)
                residual_shift = scipy_shift(img_residual, inv_shift)
                inv_var_shift = scipy_shift(inverse_var, inv_shift)
                inv_var_shift[inv_var_shift < 0] = 0.

                residual_arr[f] += residual_shift*inv_var_shift  # *weights_shift
                inverse_var_arr[f] += inv_var_shift
        new_solution = [np.zeros((self.y_size, self.x_size)) for f in range(self.n_step)]
        for f in range(self.n_step):
            inds_use = inverse_var_arr[f] > 0
            new_solution[f][inds_use] = residual_arr[f][inds_use]/inverse_var_arr[f][inds_use]
        return (new_solution, inverse_var_arr)

    @staticmethod
    def _model_spatial_filter(new_solution, filter_width, mask=None, filter_fn=ndimage.gaussian_filter):
        """Restrict spatial variations between model planes.

        Parameters
        ----------
        new_solution : list of np.ndarrays
            One np.ndarray for each model sub-band
        filter_width : float
            Width parameter passed in to the filtering function.
        mask : np.ndarray, optional
            Combined bit plane mask of the model, which is used as the mask plane for generated templates.
        filter_fn : function, optional
            The function used to perform the spatial filtering.

        Returns
        -------
        list of np.ndarrays
            One np.ndarray for each model sub-band
        """
        avg_solution = np.mean(new_solution, axis=0)
        if mask is not None:
            detected_bit = 32  # Should eventually use actual mask with mask plane info.
            mask_use = (mask & detected_bit) == detected_bit
            inds_use = mask_use == 1
            print("Before %f" % np.mean(avg_solution[inds_use]))
            mask_use = filter_fn(mask_use, filter_width, mode='constant')
            avg_solution *= mask_use
            print("After %f" % np.mean(avg_solution[inds_use]))
            scale_arr = [np.mean(soln[inds_use])/np.mean(avg_solution[inds_use]) for soln in new_solution]
        else:
            scale_arr = [np.mean(soln/np.mean(avg_solution)) for soln in new_solution]
        scale_arr = [scale if scale > 0 else 1. for scale in scale_arr]

        # Avoid dividing by zero
        avg_solution_inverse = np.zeros_like(avg_solution)
        non_zero_inds = avg_solution != 0
        avg_solution_inverse[non_zero_inds] = 1./avg_solution[non_zero_inds]

        delta_solution = [soln*avg_solution_inverse - scale
                          for soln, scale in zip(new_solution, scale_arr)]
        filter_solution = []
        ii = 0
        for delta, scale in zip(delta_solution, scale_arr):
            filter_delta = filter_fn(delta, filter_width, mode='constant')
            single_soln = (filter_delta + scale)*avg_solution
            if mask is not None:
                single_soln *= mask_use
                single_soln += new_solution[ii]*(1 - mask_use)
            filter_solution.append(single_soln)

        return filter_solution

    @staticmethod
    def _clamp_model_solution(new_solution, last_solution, clamp):
        """Restrict new solutions from being wildly different from the last solution.

        Parameters
        ----------
        new_solution : list of np.ndarrays
            The model solution from the current iteration.
        last_solution : list of np.ndarrays
            The model solution from the previous iteration.
        clamp : float
            Restrict new solutions from being more than a factor of `clamp` different from the last solution.

        Returns
        -------
        None
            Modifies new_solution in place.
        """
        for s_i, solution in enumerate(new_solution):
            # Note: last_solution is always positive
            clamp_high_i = solution > clamp*last_solution[s_i]
            high_excess = solution - clamp*last_solution[s_i]
            solution[clamp_high_i] = clamp*last_solution[s_i][clamp_high_i] + high_excess[clamp_high_i]/2.
            clamp_low_i = solution < last_solution[s_i]/clamp
            low_excess = solution - last_solution[s_i]/clamp
            solution[clamp_low_i] = last_solution[s_i][clamp_low_i]/clamp + low_excess[clamp_low_i]/2.

    @staticmethod
    def _regularize_model_solution(new_solution, bandpass, max_slope=None):
        """Calculate a slope across sub-band model planes, and clip outlier values beyond a given threshold.

        Parameters
        ----------
        new_solution : list of np.ndarrays
            The model solution from the current iteration.
        bandpass : BandpassHelper object
            Bandpass object returned by `load_bandpass`
        max_slope : float, optional
            Maximum slope to allow between sub-band model planes.

        Returns
        ------------------
        None
            Modifies new_solution in place.
        """
        if max_slope is None:
            max_slope = 1.
        n_step = len(new_solution)
        y_size, x_size = new_solution[0].shape
        solution_avg = np.median(new_solution, axis=0)

        slope_ratio = max_slope
        sum_x = 0.
        sum_y = np.zeros((y_size, x_size))
        sum_xy = np.zeros((y_size, x_size))
        sum_xx = 0.
        wl_cen = bandpass.calc_eff_wavelen()
        for f, wl in enumerate(GenerateTemplate._wavelength_iterator(bandpass, use_midpoint=True)):
            sum_x += wl - wl_cen
            sum_xx += (wl - wl_cen)**2
            sum_xy += (wl - wl_cen)*(new_solution[f] - solution_avg)
            sum_y += new_solution[f] - solution_avg
        slope = (n_step*sum_xy - sum_x*sum_y)/(n_step*sum_xx + sum_x**2)
        slope_cut_high = slope*bandpass.wavelen_step > solution_avg*slope_ratio
        slope_cut_low = slope*bandpass.wavelen_step < -solution_avg*slope_ratio
        slope[slope_cut_high] = solution_avg[slope_cut_high]*slope_ratio/bandpass.wavelen_step
        slope[slope_cut_low] = -solution_avg[slope_cut_low]*slope_ratio/bandpass.wavelen_step
        offset = solution_avg
        for f, wl in enumerate(GenerateTemplate._wavelength_iterator(bandpass, use_midpoint=True)):
            new_solution[f][slope_cut_high] = (offset + slope*(wl - wl_cen))[slope_cut_high]
            new_solution[f][slope_cut_low] = (offset + slope*(wl - wl_cen))[slope_cut_low]

    def calc_model_metric(self, model=None, stretch_threshold=None):
        """Calculate a quality of fit metric for the DCR model given the set of exposures.

        Parameters
        ----------
        model : None, optional
            The DCR model. If not set, then self.model is used.
        stretch_threshold : float, optional
            Set to simulate the effect of DCR across each sub-band by stretching the model.
            If any sub-band has DCR greater than this amount for an exposure the finite bandwidth
            FFT-based shift will be used for that instance rather than a simple shift.

        Returns
        -------
        np.ndarray
            The calculated metric for each exposure.
        """
        metric = np.zeros(self.n_images)
        if model is None:
            avg_model = np.mean(self.model, axis=0)
        else:
            avg_model = np.mean(model, axis=0)
        for exp_i, exp in enumerate(self.exposures):
            img_use, inverse_var = self._extract_image(exp, use_only_detected=True)
            template = self.build_matched_template(exp, model=model, return_weights=False,
                                                   stretch_threshold=stretch_threshold)
            inds_use = inverse_var > 0
            diff_vals = np.abs(img_use - template)*avg_model
            ref_vals = np.abs(img_use)*avg_model
            if np.sum(inds_use) == 0:
                metric[exp_i] = float("inf")
            else:
                metric[exp_i] = np.sum(diff_vals[inds_use])/np.sum(ref_vals[inds_use])
        return metric

    def _combine_masks(self):
        """Combine multiple mask planes.

        Sets the detected mask bit if any image has a detection,
        and sets other bits only if set in all images.

        Returns
        -------
        np.ndarray
            The combined mask plane.
        """
        mask_arr = (exp.getMaskedImage().getMask() for exp in self.exposures)

        detected_mask = None
        mask_use = None
        for mask in mask_arr:
            mask_vals = mask.getArray()
            if mask_use is None:
                mask_use = mask_vals
            else:
                mask_use = np.bitwise_and(mask_use, mask_vals)

            detected_bit = mask.getPlaneBitMask('DETECTED')
            if detected_mask is None:
                detected_mask = mask_vals & detected_bit
            else:
                detected_mask = np.bitwise_or(detected_mask, (mask_vals & detected_bit))
        mask_vals_return = np.bitwise_or(mask_use, detected_mask)
        return mask_vals_return
