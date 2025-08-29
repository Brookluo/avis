from astropy.table import Table
from astropy.io import fits
from pathlib import Path
import numpy as np


def get_sdss_spec(plate, mjd, fiberid):
    """Retrieve a SDSS spectrum beased on its plate, mjd and fiberid
    """
    
    sdss_path_tmpl = "/global/cfs/cdirs/sdss/data/sdss/dr16/eboss/spectro/redux/v5_13_0/spectra/lite/{plate:0>4d}/spec-{plate:0>4d}-{MJD}-{FIBERID:0>4d}.fits"
    # m_number = 26, 103 or 104
    sdss_old_path_tmpl = "/global/cfs/cdirs/sdss/data/sdss/dr8/sdss/spectro/redux/{m_number}/spectra/lite/{plate:0>4d}/spec-{plate:0>4d}-{MJD}-{FIBERID:0>4d}.fits"
    old = False
    try:
        sdss_spec = fits.open(sdss_path_tmpl.format(plate=plate, MJD=mjd, FIBERID=fiberid))
    except FileNotFoundError:
        print("Didn't find QSO in DR 16, try DR 8 now.")
        sdss_spec = fits.open(sdss_old_path_tmpl.format(m_number=26, plate=plate, MJD=mjd, FIBERID=fiberid))
        old = True
    sdss_spec_tab = Table.read(sdss_spec[1])
    sdss_spec.close()
    if old:
        sdss_spec_tab.rename_columns(['flux', 'loglam', 'ivar', 'and_mask',
                                      'or_mask', 'wdisp', 'sky', 'model'],
                                     ['FLUX', 'LOGLAM', 'IVAR', 'AND_MASK',
                                      'OR_MASK', 'WDISP', 'SKY', 'MODEL'])
    return sdss_spec_tab


def get_desi_spec_healpix(targetid, healpix, redux, survey, program):
    # healpix=str(healpix)
    
    specfile = find_desi_spec_path(redux, survey, program, healpix)

    if not specfile.exists():
        raise FileNotFoundError(f"nonexist spectrum file: {str(specfile)}")
    #read fibermap and specobj seperately? (TODO: could this be combined?)
    print(specfile)
    return read_desi_spec_from_path(specfile, targetid)


def find_desi_spec_path(redux, survey, program, healpix):
    #form full file paths
    healpix=str(healpix)
    hp_dir=Path(f'/global/cfs/cdirs/desi/spectro/redux/{redux}',
                'healpix', survey, program, healpix[0:-2], healpix)
    specfile=hp_dir / f'coadd-{survey}-{program}-{healpix}.fits'
    return specfile


def read_desi_spec_from_path(specpath: "str | Path", targetid=None, coadd=True):
    """Read spectra from a DESI specgtra FITS file path
    if targetid is None, read all spectra, else read only 
    spectra corresponding to the targetids
    
    Parameters
    ----------
    specpath: str | Path
        path to the DESI spectra file
    targetid: int | List[int] | Any
        Either one targetid, or a list of targetids to read from the file.
        If None, return all spectra in a file. Defaults to None
        
    Returns
    -------
    desispec.spectra.Spectra
        The DESI spectra object holding the wavelength and flux data and meta info
    """
    from desispec.io import read_spectra
    from desispec.coaddition import coadd_cameras
    
    # print(specfile)
    specobj = read_spectra(specpath)
    pick_id = np.ones(len(specobj.target_ids()), dtype=bool)
    if targetid is not None:
        if not isinstance(targetid, list):
            targetid = [targetid]
        pick_id &= np.logical_or.reduce([specobj.target_ids() == tid for tid in targetid])
        if sum(pick_id) == 0:
            # shall I raise an error or just throw an warning?
            raise RuntimeError(f"TARGETID {targetid} not in {str(specfile)}")
    if coadd:
        ret_specobj = coadd_cameras(specobj[pick_id])
    else:
        ret_specobj = specobj[pick_id]
    return ret_specobj
