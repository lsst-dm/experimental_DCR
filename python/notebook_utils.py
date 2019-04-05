import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
from astropy.table import Table

import lsst.afw.display as afwDisplay
import lsst.afw.geom as afwGeom
import lsst.daf.persistence as daf_persistence
from lsst.ip.diffim.dcrModel import wavelengthGenerator, calculateDcr


class DummyFilter:
    def __init__(self, lambdaEff=476.31, lambdaMin=405.0, lambdaMax=552.0):
        self.filter = FilterProperty(lambdaEff, lambdaMin, lambdaMax)

    def getFilterProperty(self):
        return self.filter


class FilterProperty:
    def __init__(self, lambdaEff, lambdaMin, lambdaMax):
        self.lambdaEff=lambdaEff
        self.lambdaMin=lambdaMin
        self.lambdaMax=lambdaMax

    def getLambdaMin(self):
        return self.lambdaMin

    def getLambdaMax(self):
        return self.lambdaMax

    def getLambdaEff(self):
        return self.lambdaEff


def look(img, range=None, x_range=None, y_range=None, outfile=None, window=1):
    """Simple function to wrap matplotlib and display an image with a colorbar."""
    afwDisplay.Display(window)
    if range is None:
        range = [np.min(img), np.max(img)]
    img_use = img.copy()
    if x_range is not None:
        x0 = int(x_range[0])
        x1 = int(x_range[1])
        if x0 < 0:
            img_use = np.roll(img_use, -x0, axis=1)
            x1 -= x0
            x0 = 0
        img_use = img_use[:, x0: x1]
    if y_range is not None:
        y0 = int(y_range[0])
        y1 = int(y_range[1])
        if y0 < 0:
            img_use = np.roll(img_use, -y0, axis=0)
            y1 -= y0
            y0 = 0
        img_use = img_use[y0: y1, :]
    fig_show = plt.imshow(img_use, interpolation='none', origin='lower', cmap=cm.rainbow, clim=range)
    plt.colorbar(fig_show, orientation='vertical', shrink=1)
    if outfile is not None:
        plt.savefig(outfile)
    plt.show()


class SimMatcher:
    def __init__(self, sim, tolerance=.1/3600, verbose=True):
        self.tolerance = tolerance
        self.wcs = sim.wcs
        self.coord = sim.coord
        self.quasar_coord = sim.quasar_coord
        pad_store = self.coord.pad
        flag_store = self.coord.flag[:]
        quasar_flag_store = self.quasar_coord.flag[:]
        self.coord.pad = 1
        self.coord.flag[:] = False
        self.quasar_coord.pad = 1
        self.quasar_coord.flag[:] = False
        star_ri = {}
        self.star_i = src_match(sim.coord, sim.catalog, tolerance, sim.wcs,
                                reverse_match=star_ri, verbose=verbose, use_flags=False)
        self.star_ri = star_ri
        quasar_ri = {}
        self.quasar_i = src_match(sim.quasar_coord, sim.quasar_catalog, tolerance, sim.wcs,
                                  reverse_match=quasar_ri, verbose=verbose, use_flags=False)
        self.quasar_ri = quasar_ri
        self.quasar_Z = extract_redshift(sim, ind_map=self.quasar_i)
        self.star_T = extract_temperature(sim, ind_map=self.star_i)
        self.coord.pad = pad_store
        self.coord.flag[:] = flag_store
        self.quasar_coord.pad = pad_store
        self.quasar_coord.flag[:] = quasar_flag_store

    def match_measurement(self, meas_cat, use_quasars=False, tolerance=None, verbose=True):
        if tolerance is None:
            tolerance = self.tolerance
        if use_quasars:
            pad_store = self.coord.pad
            quasar_flag_store = self.quasar_coord.flag[:]
            self.quasar_coord.pad = 1
            self.quasar_coord.flag[:] = False
            meas_matches = src_match(self.quasar_coord, meas_cat, tolerance, self.wcs,
                                     verbose=verbose, ref_use=list(self.quasar_i.values()))
            self.quasar_coord.pad = pad_store
            self.quasar_coord.flag[:] = quasar_flag_store
        else:
            pad_store = self.coord.pad
            flag_store = self.coord.flag[:]
            self.coord.pad = 1
            self.coord.flag[:] = False
            meas_matches = src_match(self.coord, meas_cat, tolerance, self.wcs,
                                     verbose=verbose, ref_use=list(self.star_i.values()))
            self.coord.pad = pad_store
            self.coord.flag[:] = flag_store
        return meas_matches

    def find_spectrum(self, cat_matches, temperature):
        sim_ind = -1
        best_temp = -1
        for i0, i1 in zip(cat_matches.keys(), cat_matches.values()):
            temperature_check = self.star_T[i1]
            if abs(temperature_check - temperature) <= abs(best_temp - temperature):
                sim_ind = i0
                best_temp = temperature_check
        return(sim_ind)

    def find_quasar_spectrum(self, cat_matches, redshift):
        sim_ind = -1
        best_redshift = -1
        for i0, i1 in zip(cat_matches.keys(), cat_matches.values()):
            redshift_check = self.quasar_Z[i1]
            if abs(redshift_check - redshift) <= abs(best_redshift - redshift):
                sim_ind = i0
                best_redshift = redshift_check
        return(sim_ind)


def find_location(src_cat, x=None, y=None, ra=None, dec=None):
    if x is None:
        cat_x = src_cat.getCoord().getLongitude().asDegrees()
        x_use = ra
    else:
        cat_x = src_cat.getX()
        x_use = x
    if y is None:
        cat_y = src_cat.getCoord().getLatitude().asDegrees()
        y_use = dec
    else:
        cat_y = src_cat.getY()
        y_use = y
    dist = np.sqrt((cat_x - x_use)**2. + (cat_y - y_use)**2.)
    ind_use = int(np.argmin(dist))
    ra_use = src_cat[ind_use].getCoord().getLongitude().asDegrees()
    dec_use = src_cat[ind_use].getCoord().getLatitude().asDegrees()
    print("RA: %f, Dec: %f" % (ra_use, dec_use))
    return(ind_use)


def src_match(coords_ref, src_cat, err_match, wcs, reverse_match={},
              verbose=True, ref_use=None, use_flags=True):
    """Match simulated sources between the full catalog and the subset in the image.

    Parameters
    ----------
    coords_ref : TYPE
        Description
    cat_match : TYPE
        Description
    err_match : TYPE
        Description
    wcs : TYPE
        Description
    reverse_match : dict, optional
        Description
    verbose : bool, optional
        Description
    ref_use : None, optional
        Description

    Returns
    -------
    TYPE
        Description
    """
    ref_sky = [wcs.pixelToSky(x_loc, y_loc) for x_loc, y_loc in zip(coords_ref.x_loc(), coords_ref.y_loc())]
    ref_ra = np.array([pt.getLongitude().asDegrees() for pt in ref_sky])
    ref_dec = np.array([pt.getLatitude().asDegrees() for pt in ref_sky])
    if ref_use is not None:
        ref_ra = ref_ra[ref_use]
        ref_dec = ref_dec[ref_use]
    ind_match = {}
    i = -1
    i2 = 0
    psf_flux_slot = src_cat.getPsfFluxSlot()
    for src in src_cat:
        i += 1
        if use_flags:
            if src[psf_flux_slot.getFlagKey()]:
                continue
        src_ra = src.getCoord().getLongitude().asDegrees()
        src_dec = src.getCoord().getLatitude().asDegrees()
        dist = np.sqrt((ref_ra - src_ra)**2. + (ref_dec - src_dec)**2.)
        match_test = dist < err_match
        n_match = np.sum(match_test)
        if n_match == 1:
            ind_match[i] = np.where(match_test)[0][0]
            reverse_match[np.where(match_test)[0][0]] = i
            i2+=1
        elif n_match > 1:
            if verbose:
                print("Multiple matches")
    if verbose:
        print("Matched: %i, Unmatched: %i" % (i2, i - i2))
    return(ind_match)


def convert_flux(flux_in, wl_in, filterInfo, numDcrSubfilters):
    flux_out = []
    for wl_start, wl_end in wavelengthGenerator(filterInfo, numDcrSubfilters):
        wl_use = (wl_in >= wl_start) & (wl_in < wl_end)
        flux = np.sum(flux_in[wl_use])
        flux_out.append(flux)
    return np.array(flux_out)


def extract_temperature(sim, ind_map=None):
    if ind_map is None:
        temps = [sim.catalog[ii]["temperature"] for ii in range(len(sim.catalog))]
    else:
        temps = [sim.catalog[ii]["temperature"] for ii in ind_map.keys()]
    return temps


def extract_redshift(sim, ind_map=None):
    if ind_map is None:
        redshift = [sim.quasar_catalog[ii]["redshift"] for ii in range(len(sim.quasar_catalog))]
    else:
        redshift = [sim.quasar_catalog[ii]["redshift"] for ii in ind_map.keys()]
    return redshift


def plot_spectrum(sim, filterInfo, meas_cats, src_ind, cat_matches=None,
                  rescale=False, use_throughput=False, star_T=None,
                  oplot=False, color='r', markersize=15, verbose=True, window=1,
                  fudge=1.):
    numDcrSubfilters = len(meas_cats)
    numDcrSubfiltersSim = sim.n_step
    meas_wl = [np.mean(wl) for wl in wavelengthGenerator(filterInfo, numDcrSubfilters)]
    sim_wl = [np.mean(wl) for wl in wavelengthGenerator(filterInfo, numDcrSubfiltersSim)]
    ratio_meas_sim = numDcrSubfiltersSim/numDcrSubfilters
    filter_throughput = sim.bandpass_highres.getBandpass()[1]
    filter_wl = sim.bandpass_highres.getBandpass()[0]
    throughput_correction = convert_flux(filter_throughput, filter_wl, DummyFilter(), numDcrSubfilters)
    i0 = src_ind
    if cat_matches is not None:
        i1 = cat_matches[i0]
        throughput_correction_sim = convert_flux(filter_throughput, filter_wl, DummyFilter(), numDcrSubfiltersSim)
        sim_flux_raw = np.zeros_like(throughput_correction_sim)
        sim_flux_raw += sim.star_flux[i1, :]
        sim_flux = convert_flux(sim_flux_raw, np.array(sim_wl), DummyFilter(), numDcrSubfilters)
        sim_fit_params = np.polyfit(meas_wl, sim_flux, 1)
        sim_fit = [sim_fit_params[0]*wl + sim_fit_params[1] for wl in meas_wl]
        sim_x = sim.coord.x_loc()[i1]
        sim_y = sim.coord.y_loc()[i1]
        ref_sky = sim.wcs.pixelToSky(afwGeom.Point2D(sim_x, sim_y))
        sim_ra = ref_sky.getLongitude().asDegrees()
        sim_dec = ref_sky.getLatitude().asDegrees()
    else:
        rescale = False

    meas_flux = np.array([cat.getPsfInstFlux()[i0] for cat in meas_cats])
    if fudge != 1:
        mean_flux = np.mean(meas_flux)
        meas_flux = np.array([(flux - mean_flux)*fudge + mean_flux for flux in meas_flux])
    meas_x = meas_cats[0][i0].getX()
    meas_y = meas_cats[0][i0].getY()
    meas_ra = meas_cats[0][i0].getCoord().getLongitude().asDegrees()
    meas_dec = meas_cats[0][i0].getCoord().getLatitude().asDegrees()
    if use_throughput:
        meas_flux /= throughput_correction
    if verbose:
        if cat_matches is not None:
            if star_T is not None:
                print("Temperature: %fK" % star_T[i1])
                if star_T[i1] < 3700 and star_T[i1] > 2400:
                    print("Type M")
                if star_T[i1] < 5200 and star_T[i1] > 3700:
                    print("Type K")
                if star_T[i1] < 6000 and star_T[i1] > 5200:
                    print("Type G")
                if star_T[i1] < 7500 and star_T[i1] > 6000:
                    print("Type F")
                if star_T[i1] < 10000 and star_T[i1] > 7500:
                    print("Type A")
                if star_T[i1] < 30000 and star_T[i1] > 10000:
                    print("Type B")
                if star_T[i1] < 50000 and star_T[i1] > 30000:
                    print("Type O")
            print("sim coordinates: %f, %f" % (sim_ra, sim_dec))
            print("sim x,y: %f, %f" % (sim_x, sim_y))
        print("meas coordinates: %f, %f" % (meas_ra, meas_dec))
        print("meas x,y: %f, %f" % (meas_x, meas_y))

        if cat_matches is not None:
            print(sim_flux)
        print(meas_flux)
    if rescale:
        if verbose:
            print("Rescale factor: %f" % (np.mean(sim_flux)/np.mean(meas_flux)))
        meas_flux *= np.mean(sim_flux)/np.mean(meas_flux)
    if not oplot:
        afwDisplay.Display(window)
        if cat_matches is not None:
            plt.plot(sim_wl, sim_flux_raw*ratio_meas_sim)
            plt.plot(meas_wl, sim_flux, 'bx', markersize=10, markeredgewidth=2)
    plt.plot(meas_wl, meas_flux, color+'+', markersize=markersize, markeredgewidth=2)
    if cat_matches is not None:
        plt.plot(meas_wl, sim_fit, 'b+', markersize=10, markeredgewidth=2)
    plt.xlabel('Wavelength (nm)')
    plt.ylabel('Flux (counts)')


def plot_quasar_spectrum(sim, filterInfo, meas_cats, src_ind, cat_matches=None,
                         rescale=False, use_throughput=False, quasar_Z=None,
                         oplot=False, color='r', markersize=15, verbose=True, window=1,
                         fudge=1.):
    numDcrSubfilters = len(meas_cats)
    numDcrSubfiltersSim = sim.n_step
    meas_wl = [np.mean(wl) for wl in wavelengthGenerator(filterInfo, numDcrSubfilters)]
    sim_wl = [np.mean(wl) for wl in wavelengthGenerator(filterInfo, numDcrSubfiltersSim)]
    ratio_meas_sim = numDcrSubfiltersSim/numDcrSubfilters
    filter_throughput = sim.bandpass_highres.getBandpass()[1]
    filter_wl = sim.bandpass_highres.getBandpass()[0]
    throughput_correction = convert_flux(filter_throughput, filter_wl, DummyFilter(), numDcrSubfilters)
    throughput_correction_sim = convert_flux(filter_throughput, filter_wl, DummyFilter(), numDcrSubfiltersSim)
    i0 = src_ind
    if cat_matches is not None:
        i1 = cat_matches[i0]
        sim_flux_raw = np.zeros_like(throughput_correction_sim)
        sim_flux_raw += sim.quasar_flux[i1, :]
        sim_flux = convert_flux(sim_flux_raw, np.array(sim_wl), DummyFilter(), numDcrSubfilters)
        sim_fit_params = np.polyfit(meas_wl, sim_flux, 1)
        sim_fit = [sim_fit_params[0]*wl + sim_fit_params[1] for wl in meas_wl]
        sim_x = sim.quasar_coord.x_loc()[i1]
        sim_y = sim.quasar_coord.y_loc()[i1]
        ref_sky = sim.wcs.pixelToSky(afwGeom.Point2D(sim_x, sim_y))
        sim_ra = ref_sky.getLongitude().asDegrees()
        sim_dec = ref_sky.getLatitude().asDegrees()
    else:
        rescale = False

    meas_flux = np.array([cat.getPsfInstFlux()[i0] for cat in meas_cats])
    if fudge != 1:
        mean_flux = np.mean(meas_flux)
        meas_flux = np.array([(flux - mean_flux)*fudge + mean_flux for flux in meas_flux])
    meas_x = meas_cats[0][i0].getX()
    meas_y = meas_cats[0][i0].getY()
    meas_ra = meas_cats[0][i0].getCoord().getLongitude().asDegrees()
    meas_dec = meas_cats[0][i0].getCoord().getLatitude().asDegrees()

    if use_throughput:
        meas_flux /= throughput_correction
    if verbose:
        if cat_matches is not None:
            if quasar_Z is not None:
                print("Redshift: %f" % quasar_Z[i1])
            print("sim coordinates: %f, %f" % (sim_ra, sim_dec))
            print("sim x,y: %f, %f" % (sim_x, sim_y))
            print("meas coordinates: %f, %f" % (meas_ra, meas_dec))
            print("meas x,y: %f, %f" % (meas_x, meas_y))
            print(sim_flux)
        print(meas_flux)
    if rescale:
        if verbose:
            print("Rescale factor: %f" % (np.mean(sim_flux)/np.mean(meas_flux)))
        meas_flux *= np.mean(sim_flux)/np.mean(meas_flux)
    if not oplot:
        afwDisplay.Display(window)
        if cat_matches is not None:
            plt.plot(sim_wl, sim_flux_raw*ratio_meas_sim)
            plt.plot(meas_wl, sim_flux, 'bx', markersize=10, markeredgewidth=2)
    plt.plot(meas_wl, meas_flux, color+'+', markersize=markersize, markeredgewidth=2)
    if cat_matches is not None:
        plt.plot(meas_wl, sim_fit, 'b+', markersize=10, markeredgewidth=2)
    plt.xlabel('Wavelength (nm)')
    plt.ylabel('Flux (counts)')


def plot_color(sim, filterInfo, meas_cats, matches, use_throughput=True, window=1,
               flux_min=None, flux_max=None, fudge=1.):
    meas_color = []
    sim_color = []
    numDcrSubfilters = len(meas_cats)
    numDcrSubfiltersSim = sim.n_step
    meas_wl = [np.mean(wl) for wl in wavelengthGenerator(filterInfo, numDcrSubfilters)]
    sim_wl = [np.mean(wl) for wl in wavelengthGenerator(filterInfo, numDcrSubfiltersSim)]
    ratio_meas_sim = numDcrSubfiltersSim/numDcrSubfilters
    filter_throughput = sim.bandpass_highres.getBandpass()[1]
    filter_wl = sim.bandpass_highres.getBandpass()[0]
    throughput_correction = convert_flux(filter_throughput, filter_wl, DummyFilter(), numDcrSubfilters)
    for key in list(matches.keys()):
        i0 = key
        i1 = matches[i0]
        sim_flux = convert_flux(sim.star_flux[i1, :], np.array(sim_wl), DummyFilter(), numDcrSubfilters)
        sim_fit_params = np.polyfit(meas_wl, sim_flux*ratio_meas_sim, 1)
        sim_fit = [sim_fit_params[0]*wl + sim_fit_params[1] for wl in meas_wl]
        meas_flux = np.array([cat.getPsfInstFlux()[i0] for cat in meas_cats])
        if fudge != 1:
            mean_flux = np.mean(meas_flux)
            meas_flux = np.array([(flux - mean_flux)*fudge + mean_flux for flux in meas_flux])
        if use_throughput:
            meas_flux /= throughput_correction
        if flux_min is not None:
            if np.median(meas_flux) < flux_min:
                continue
        if flux_max is not None:
            if np.median(meas_flux) > flux_max:
                continue
        meas_mag0 = -2.512 * np.log10(meas_flux[0])
        meas_mag1 = -2.512 * np.log10(meas_flux[numDcrSubfilters-1])
        sim_mag0 = -2.512 * np.log10(sim_fit[0])
        sim_mag1 = -2.512 * np.log10(sim_fit[numDcrSubfilters-1])
        if abs(meas_mag1 - meas_mag0 - (sim_mag1 - sim_mag0)) < 1.9:
            meas_color.append(meas_mag0 - meas_mag1)
            sim_color.append(sim_mag0 - sim_mag1)
    afwDisplay.Display(window)
    plt.plot(np.array(sim_color), (np.array(meas_color)), 'k+')
    plt.plot([-1, 2], [-1, 2], linestyle='--', color='b')
    plt.xlim = [-.5, 2]
    plt.ylim = [-.5, 2]
    plt.ylabel('Measured g(b)-g(r) color')
    plt.xlabel('True g(b)-g(r) color')


def plot_quasar_color(sim, filterInfo, meas_cats, matches, use_throughput=True, window=1,
                      flux_min=None, flux_max=None, fudge=1.):
    meas_color = []
    sim_color = []
    numDcrSubfilters = len(meas_cats)
    numDcrSubfiltersSim = sim.n_step
    meas_wl = [np.mean(wl) for wl in wavelengthGenerator(filterInfo, numDcrSubfilters)]
    sim_wl = [np.mean(wl) for wl in wavelengthGenerator(filterInfo, numDcrSubfiltersSim)]
    ratio_meas_sim = numDcrSubfiltersSim/numDcrSubfilters
    filter_throughput = sim.bandpass_highres.getBandpass()[1]
    filter_wl = sim.bandpass_highres.getBandpass()[0]
    throughput_correction = convert_flux(filter_throughput, filter_wl, DummyFilter(), numDcrSubfilters)
    for key in list(matches.keys()):
        i0 = key
        i1 = matches[i0]
        sim_flux = convert_flux(sim.quasar_flux[i1, :], np.array(sim_wl), DummyFilter(), numDcrSubfilters)
        sim_fit_params = np.polyfit(meas_wl, sim_flux*ratio_meas_sim, 1)
        sim_fit = [sim_fit_params[0]*wl + sim_fit_params[1] for wl in meas_wl]
        meas_flux = np.array([cat.getPsfInstFlux()[i0] for cat in meas_cats])
        if fudge != 1:
            mean_flux = np.mean(meas_flux)
            meas_flux = np.array([(flux - mean_flux)*fudge + mean_flux for flux in meas_flux])
        if use_throughput:
            meas_flux /= throughput_correction
        if flux_min is not None:
            if np.median(meas_flux) < flux_min:
                continue
        if flux_max is not None:
            if np.median(meas_flux) > flux_max:
                continue
        meas_mag0 = -2.512 * np.log10(meas_flux[0])
        meas_mag1 = -2.512 * np.log10(meas_flux[numDcrSubfilters-1])
        sim_mag0 = -2.512 * np.log10(sim_fit[0])
        sim_mag1 = -2.512 * np.log10(sim_fit[numDcrSubfilters-1])
        if abs(meas_mag1 - meas_mag0 - (sim_mag1 - sim_mag0)) < 1.9:
            meas_color.append(meas_mag0 - meas_mag1)
            sim_color.append(sim_mag0 - sim_mag1)
    afwDisplay.Display(window)
    plt.plot(np.array(sim_color), (np.array(meas_color)), 'r+')
    plt.plot([-1, 2], [-1, 2], linestyle='--', color='b')
    plt.xlim = [-.5, 2]
    plt.ylim = [-.5, 2]
    plt.ylabel('Measured g(b)-g(r) color')
    plt.xlabel('True g(b)-g(r) color')


def load_diffim(simulation_dir, deep_rerun, dcr_rerun,
                visits_template, visits_alerts,
                filter_name='g', coaddName='deep'):
    if coaddName == 'dcr':
        butler = daf_persistence.Butler(simulation_dir + dcr_rerun)
    else:
        butler = daf_persistence.Butler(simulation_dir + deep_rerun)
    airmasses = []
    par_angs = []
    fields = []
    years = []
    visit_list = []
    dcr_list = []
    filterInfo = None
    wcs = None
    for visits in visits_template:
        for visit in range(visits[0], visits[1] + 1):
            dataId = {'visit': visit, 'filter': filter_name}
            try:
                visitInfo = butler.get("calexp_visitInfo", dataId=dataId)
                if filterInfo is None:
                    filterInfo = butler.get("calexp_filter", dataId=dataId)
                if wcs is None:
                    wcs = butler.get("calexp_wcs", dataId=dataId)
            except:
                print("Skipping %i, no data." % visit)
#             psf = butler.get("calexp_psf", dataId=dataId)
#             psf_fwhm = psf.getFWHM()
            visit_list.append(visit)
            year = 2 if visit > 2000000 else 1
            years.append(year)
            field = "%.4i" % (visit//100 - year*10000)
            fields.append(field)
            airmass = visitInfo.getBoresightAirmass()
            airmasses.append(airmass)
            par_ang = visitInfo.getBoresightParAngle()
            par_angs.append(par_ang.asDegrees())
            dcr_list.append(calculateDcr(visitInfo, wcs, filterInfo, 5)[0])
#             psf_list.append(psf_fwhm)
    template_properties = Table([fields, years, visit_list, airmasses, par_angs, dcr_list],
                                names=('fieldId', 'year', 'visit', 'airmass', 'parAng', 'dcr'))
    airmasses = []
    par_angs = []
    n_dipoles = []
    dia_fluxes = []
    fields = []
    years = []
    visit_list = []
    dcr_list = []
    for visits in visits_alerts:
        for visit in range(visits[0], visits[1] + 1):
            dataId = {'visit': visit, 'filter': filter_name}
            try:
                dia_src = butler.get(coaddName + "Diff_diaSrc", dataId=dataId)
                visitInfo = butler.get("calexp_visitInfo", dataId=dataId)
            except:
                print("Skipping %i, no data." % visit)
                continue
            visit_list.append(visit)
            year = 2 if visit > 2000000 else 1
            years.append(year)
            field = "%.4i" % (visit//100 - year*10000)
            fields.append(field)
            n_dipole = np.sum(dia_src["ip_diffim_DipoleFit_flag_classification"])
            n_dipoles.append(n_dipole)
            dia_flux = np.nansum(np.abs(dia_src["slot_PsfFlux_instFlux"]))
            dia_fluxes.append(dia_flux)
            airmass = visitInfo.getBoresightAirmass()
            airmasses.append(airmass)
            par_ang = visitInfo.getBoresightParAngle()
            par_angs.append(par_ang.asDegrees())
            dcr_list.append(calculateDcr(visitInfo, wcs, filterInfo, 5)[0])
    alerts_properties = Table([fields, years, visit_list, airmasses,
                              par_angs, dcr_list, n_dipoles, dia_fluxes],
                              names=('fieldId', 'year', 'visit', 'airmass',
                                     'parAng', 'dcr', 'nDipole', 'diaFlux'))
    return(template_properties, alerts_properties)


def plot_diffim_quiver(deep_constPSF, dcr_constPSF, deep_varPSF, dcr_varPSF,
                       const_template=None, var_template=None,
                       quantity="nDipole", window=1, **kwargs):
    field_list = list(set(deep_constPSF["fieldId"]))
    afwDisplay.Display(window)
    color_gen = get_xkcd_color()
    names = []
    for field in field_list:
        if const_template is None:
            const_metric = dcr_metric(dcr_constPSF, field, binsize=0.05)
        else:
            const_metric = dcr_metric(const_template, field, binsize=0.05)
        if var_template is None:
            var_metric = dcr_metric(dcr_varPSF, field, binsize=0.05)
        else:
            var_metric = dcr_metric(var_template, field, binsize=0.05)
        names.append("Field %s with metrics const:%4.2f var:%4.2f" % (field, const_metric, var_metric))
        q1a = deep_constPSF[quantity][deep_constPSF["fieldId"] == field]
        q2a = dcr_constPSF[quantity][dcr_constPSF["fieldId"] == field]
        q1b = deep_varPSF[quantity][deep_varPSF["fieldId"] == field]
        q2b = dcr_varPSF[quantity][dcr_varPSF["fieldId"] == field]
        x = q1a
        y = q1b
        dx = q2a - q1a
        dy = q2b - q1b
        c = 'xkcd:' + next(color_gen)
        plt.quiver(x, y, dx, dy, color=c, angles='xy', scale_units='xy', scale=1., **kwargs)
    plt.xlabel('False detections using constant PSF (deep -> dcr)')
    plt.ylabel('False detections using variable PSF with good seeing cut')
    plt.legend(names)


def get_xkcd_color(index=None):
    colors = ["purple", "green", "blue", "pink", "brown", "red", "light blue",
              "teal", "orange", "light green", "magenta", "yellow", "sky blue",
              "grey", "lime green", "light purple", "violet", "dark green",
              "turquoise", "lavender", "dark blue", "tan", "cyan", "aqua",
              "forest green", "mauve", "dark purple", "bright green", "maroon",
              "olive", "salmon", "beige", "royal blue", "navy blue", "lilac",
              "black"]
    if index is not None:
        colors = [colors[index], ]
    for c in colors:
        yield c


def get_dcr(field_table, fieldId):
    fields = set(field_table["fieldId"])
    if fieldId not in fields:
        print('ValueError("fieldId not found in table")')
    rows = field_table["fieldId"] == fieldId
    dcr = field_table["dcr"].data[rows]
    return dcr


def dcr_metric(field_table, fieldId, binsize=0.1):
    dcr = get_dcr(field_table, fieldId)
    x_arr = dcr[:, 0]/binsize
    y_arr = dcr[:, 1]/binsize
    x_arr -= np.min(x_arr)
    y_arr -= np.min(y_arr)
    xsize = int(np.ceil(np.max(x_arr)) + 1)
    ysize = int(np.ceil(np.max(y_arr)) + 1)
    metric_arr = np.zeros((ysize, xsize), dtype=float)
    n_arr = np.zeros((ysize, xsize), dtype=float)
    for x, y in zip(x_arr, y_arr):
        x0 = int(np.floor(x))
        y0 = int(np.floor(y))
        dx = x - x0
        dy = y - y0
        metric_arr[y0, x0] += (1 - dx)*(1 - dy)
        metric_arr[y0 + 1, x0] += (1 - dx)*(dy)
        metric_arr[y0, x0 + 1] += (dx)*(1 - dy)
        metric_arr[y0 + 1, x0 + 1] += (dx)*(dy)
        n_arr[y0: y0 + 2, x0: x0 + 2] += 1
    pix_use = n_arr > 0
    metric_arr[pix_use] /= np.sqrt(n_arr[pix_use])
    metric = np.sum(metric_arr)
    return metric