from __future__ import print_function, division, absolute_import
import numpy as np

import lsst.daf.persistence as daf_persistence


def diasrc_metric(repository=".", repository_src=None, obsid_range=None, band='g'):
    """
    Compute statistics on diaSrc tables and return the results.
    @param repository: path to repository with the data. String, defaults to working directory
    @param obsid_range: obsid or range of obsids to process.
    """

    saturation = 65000.0
    butler = daf_persistence.Butler(repository)
    if repository_src is not None:
        butler2 = daf_persistence.Butler(repository_src)

    if hasattr(obsid_range, '__iter__'):
        if len(obsid_range) > 2:
            if obsid_range[2] < obsid_range[0]:
                dataId = ({'visit': obsid, 'raft': '2,2', 'sensor': '1,1', 'filter': band}
                          for obsid in np.arange(obsid_range[0], obsid_range[1], obsid_range[2]))
            else:
                dataId = ({'visit': obsid, 'raft': '2,2', 'sensor': '1,1', 'filter': band}
                          for obsid in obsid_range)
        else:
            dataId = ({'visit': obsid, 'raft': '2,2', 'sensor': '1,1', 'filter': band}
                      for obsid in np.arange(obsid_range[0], obsid_range[1]))
    else:
        dataId = ({'visit': obsid, 'raft': '2,2', 'sensor': '1,1', 'filter': band} for obsid in [obsid_range])

    schema = None
    for _id in dataId:
        dia_src = butler.get("deepDiff_diaSrc", dataId=_id)
        if dia_src.isContiguous() is False:
            dia_src = dia_src.copy(True)
        if schema is None:
            schema = dia_src.getSchema()
            # dipoleKey = schema.find("ip_diffim_DipoleFit_flag_classification").key
            posFluxKey = schema.find("ip_diffim_PsfDipoleFlux_pos_flux").key
            negFluxKey = schema.find("ip_diffim_PsfDipoleFlux_neg_flux").key
            fluxKey = schema.find("base_PsfFlux_flux").key
            sigmaKey = schema.find("base_PsfFlux_fluxSigma").key
            flagKey = schema.find("base_PsfFlux_flag").key
            # flag2Key = schema.find("base_PixelFlags_flag_saturated").key
            # orientationKey = schema.find("ip_diffim_DipoleFit_orientation").key
            # separationKey = schema.find("ip_diffim_DipoleFit_separation").key

        if repository_src is not None:
            ref_src = butler2.get("src", dataId=_id)
            if ref_src.isContiguous() is False:
                ref_src = ref_src.copy(True)
            schema2 = ref_src.getSchema()
            fluxKey2 = schema2.find("base_PsfFlux_flux").key
            flagKey2 = schema2.find("base_PsfFlux_flag").key
        else:
            ref_src = dia_src
            fluxKey2 = fluxKey
            flagKey2 = flagKey

        flux_ref = np.abs(ref_src[fluxKey2])
        flux_ref = flux_ref[~ref_src[flagKey2]]
        flux_ref = np.clip(flux_ref, 1.0, saturation)
        flux_ref = np.sum(flux_ref[np.isfinite(flux_ref)])

        combined_flux = (np.abs(dia_src[posFluxKey]) + np.abs(dia_src[negFluxKey]))
        metric = np.clip(combined_flux, 1.0, saturation) / np.abs(dia_src[sigmaKey])
        metric = metric[~dia_src[flagKey]]
        metric = np.log(1.0 + np.sum(metric[np.isfinite(metric)]) / np.sqrt(flux_ref))
        yield (_id['visit'], _id['filter'], metric)


# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

if __name__ == "__main__":
    repo = "/Users/sullivan/LSST/simulations/test6/Diffim_test"
    base_range = np.append(np.arange(0, 23, 2), np.arange(1, 23, 2))
    diasrc_metric(repository=repo, obsid_range=base_range + 100, band='u')
    diasrc_metric(repository=repo, obsid_range=base_range + 200, band='g')
    diasrc_metric(repository=repo, obsid_range=base_range + 300, band='r')
