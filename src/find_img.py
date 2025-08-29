import astropy
from astropy.table import Table
from astropy.coordinates import SkyCoord
from astropy.wcs import WCS
from astropy.io import fits
from pathlib import Path


def find_brick_with_radec(ra, dec, dr="dr10", bricks=None):
    '''find the brick directory given an ra and dec.
    Based on the description: 
    https://www.legacysurvey.org/dr10/files/#tractor-catalogs-south-tractor
    Note, the brick is [left-Inclusive, right-Exclusive)
    
    Args:
        ra (float): right acesion in degrees
        dec (float): declination in degrees
        dr (str): data release version
        bricks (astropy.table.Table): bricks file, optional, can be provided by
            users to speed up the find process.
        
    Return:
        (tuple): a tuple of (RA directory, brick subdirectory)
    '''
    if bricks is None:
        bricks = Table.read(f"/global/cfs/cdirs/cosmo/data/legacysurvey/{dr}/survey-bricks.fits.gz")
    find_ra = (bricks["RA1"] <= ra) &  (ra < bricks["RA2"])
    find_dec = (bricks["DEC1"] <= dec) &  (dec < bricks["DEC2"])
    brick_dir = bricks["BRICKNAME"][find_ra & find_dec][0]
    ra_dir = brick_dir[:3]
    # ra_ro = int(ra*10)
    # dec_ro = int(dec*10)
    # assign_pm = lambda num: 'm' if num < 0 else 'p'
    # dec_prefix = assign_pm(dec_ro)
    # ra_dir = f"{int(np.floor(ra)):03}"
    # brick_dir = f"{ra_ro:04}{dec_prefix}{abs(dec_ro):03}"
    return ra_dir, brick_dir


def find_sweep_with_radec(ra, dec):
    '''find the sweep file given an ra and dec
    
    Args:
        ra (float): right acesion in degrees
        dec (float): declination in degrees
        
    Return:
        (string): a file name contains this ra and dec 
    '''
    ra_floor = int(ra // 5 * 5)
    dec_floor = int(dec // 5 * 5)
    ra_roof = int(ra_floor + 5)
    dec_roof = int(dec_floor + 5)
    assign_pm = lambda num: 'm' if num < 0 else 'p'
#     floor_prefix = 'p'
#     roof_prefix = 'p'
#     if dec_floor < 0:
#         floor_prefix = 'm'
#     if dec_roof < 0:
#         roof_prefix = 'm'
    floor_prefix = assign_pm(dec_floor)
    roof_prefix = assign_pm(dec_roof)
    filename = f"sweep-{ra_floor:03}{floor_prefix}{abs(dec_floor):03}-{ra_roof:03}{roof_prefix}{abs(dec_roof):03}.fits"
    return filename


def get_brick_dir(ra, dec):
    """Find the brick dirctory given ra and dec in degree
    """
    data_root = "/global/cfs/cdirs/cosmo/data/legacysurvey/{dr}/{ns}/coadd"
    bricks = Table.read("/global/cfs/cdirs/cosmo/data/legacysurvey/dr10/survey-bricks.fits.gz")
    # definiation of north
    # https://ui.adsabs.harvard.edu/abs/2023AJ....165...50M/abstract
    # section 4.1.3
    coord = SkyCoord(ra=ra, dec=dec, unit="deg")
    north = True if dec >= 32.375 and coord.galactic.b.deg > 0 else False        
    if north:
        dr = "dr9"
        brick_dir = Path(data_root.format(dr=dr, ns="north"),
                         *find_brick_with_radec(ra, dec, dr, bricks))
        if not brick_dir.exists():
            raise FileNotFoundError(f"cannot found northern brick with ra={ra}, dec={dec}")
    else:
        dr = "dr10"
        brick_dir = Path(data_root.format(dr=dr, ns="south"),
                         *find_brick_with_radec(ra, dec, dr, bricks))
        if not brick_dir.exists():
            dr = "dr9"
            brick_dir = Path(data_root.format(dr=dr, ns="south"),
                             *find_brick_with_radec(ra, dec, dr, bricks))
            if not brick_dir.exists():
                raise FileNotFoundError(f"cannot found southern brick with ra={ra}, dec={dec}")
    return brick_dir


def read_image(path: "str | Path"):
    """Read a FITS image by its path

    Parameters
    ----------
    path : str | Path
        path to the FITS file to be loaded. The file should have
        two HDUs, where the second HDU contains the image data
        and its header has the WCS information.

    Returns
    -------
    numpy.ndarray
        an array contains the image data
    astropy.wcs.WCS
        an astropy WCS object from this image
    """
    with fits.open(path) as hdu:
        # hdu.info()
        imwcs = WCS(hdu[1].header)
        imdata = hdu[1].data
    return imdata, imwcs