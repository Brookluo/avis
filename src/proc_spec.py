import numpy as np
from find_spec import find_desi_spec_path, get_desi_spec_healpix, read_desi_spec_from_path
from astropy.table import Table
from pathlib import Path
import logging
import argparse
import signal
import sys


def write_one_spec_to_h5(zcat_row, redux: str, h5fpath: "str | Path", chkpt_dir=None, idx_component=None):
    # User must delete the original file if it exists
    import h5py
    
    h5fpath = Path(h5fpath)
    # h5fpath = new_spec_root / f"{redux}-targetid-healpix-coadd.h5"
    targetid = zcat_row["TARGETID"]
    hp_num = zcat_row["HEALPIX"]
    survey = zcat_row["SURVEY"].astype("str")
    program = zcat_row["PROGRAM"].astype("str")
    desi_spec = get_desi_spec_healpix(targetid, hp_num,
                               redux=redux, survey=survey,
                               program=program)
    desi_spec_path = find_desi_spec_path(redux=redux, survey=survey,
                                       program=program, healpix=hp_num)
    wave_flux_ivar_mask = np.stack(
        [
            desi_spec.wave["brz"].flatten(),
            desi_spec.flux["brz"].flatten(),
            desi_spec.ivar["brz"].flatten(),
            desi_spec.mask["brz"].flatten()
        ],
        axis=1
    )
    # arr_names = ["wave", "flux", "ivar", "mask"]
    targetid = str(targetid)
    # file format
    # coadd-main-bright-7511.fits
    # coadd-{survey}-{program}-{healpix}.fits
    dset_name = f"{redux}-{desi_spec_path.stem}"
    # suffix = ""
    if idx_component is not None:
        # dset_name = f"{int(idx_component)}-{dset_name}"
        # suffix = f"{idx_component:04}-"
        filename = f"{idx_component:04}-{h5fpath.name}"
        h5fpath = h5fpath.parent / filename
    with h5py.File(h5fpath, mode='a') as h5f:
        if "targetid_spec" not in h5f.keys():
            h5f.create_group("targetid_spec")
        spec_dset = h5f["targetid_spec"]
        if targetid not in spec_dset.keys():
            spec_dset.create_group(targetid)
        tar_dset = spec_dset[targetid]
        tar_dset.create_dataset(dset_name, data=wave_flux_ivar_mask)
        # put inside context manager to ensure write checkpoint at the same time
        if chkpt_dir is not None:
            # checkpoint file format f"{i:04}-targetid_{h5fpath.stem}.txt"
            # line in the file foramt {redux}-coadd-{survey}-{program}-{healpix}-{targetid}"
            chkpt_file = chkpt_dir / f"targetid_{h5fpath.stem}.txt"
            with open(chkpt_file, 'a') as fp:
                fp.write(f"{dset_name}-{targetid}\n")


def get_finished_idx(one_line: set, zcat_tab):
    info = one_line.split("-")
    # logging.info(one_line)
    # need to fix negative targetid
    # eg. fuji-coadd-special-dark-9189--407193578
    if len(info) == 7:
        redux, _, survey, program, healpix, _, targetid = info
        targetid = '-' + targetid
    else:
        redux, _, survey, program, healpix, targetid = info
    matched = (zcat_tab["SURVEY"].astype("str") == survey) \
                & (zcat_tab["PROGRAM"].astype("str") == program) \
                & (zcat_tab["HEALPIX"] == int(healpix)) \
                & (zcat_tab["TARGETID"] == int(targetid))
    return np.where(matched)[0][0]


def create_spec_h5_by_zcat(zcat: str, redux: str, new_spec_root: "str | Path", 
                   new_spec_fname: str, num_workers=10, chkpt_dir=None, n_component=1):
    from joblib import Parallel, delayed
    
    # f"{redux}-targetid-healpix-coadd.h5"
    h5fpath = Path(new_spec_root, new_spec_fname)
    if h5fpath.exists():
        logging.warning("Loading and appending to previously saved h5 file: " + str(h5fpath))
    # checkpointing:
    # chkpt_dir = Path("/pscratch/sd/b/brookluo/spectra/proc_checkpoints")
    zcat_tab = Table.read(zcat)
    if chkpt_dir is not None:
        if not chkpt_dir.exists():
            chkpt_dir.mkdir()
        progress = set()
        for chkpt_file in chkpt_dir.iterdir():
            logging.info(f"Loading checkpoint: {str(chkpt_file)}")
            with open(chkpt_file, 'r') as fp:
                progress |= set([l.strip() for l in fp])
        # finished_tab_idx = []
        zcat_arr = zcat_tab["SURVEY", "PROGRAM", "HEALPIX", "TARGETID"].as_array()
        finished_tab_idx = Parallel(n_jobs=num_workers)(delayed(get_finished_idx)
                                                    (one, zcat_arr)
                                                    for one in progress)
        # for i, row in enumerate(zcat_tab):
        #     fname = "-".join([str(x) for x in [redux, "coadd"] + list(row["SURVEY", "PROGRAM", "HEALPIX", "TARGETID"])])
        #     if fname in progress:
        #         finished_tab_idx.append(i)
        zcat_tab.remove_rows(finished_tab_idx)
        del finished_tab_idx, progress
    zcat_arr = zcat_tab["SURVEY", "PROGRAM", "HEALPIX", "TARGETID"].as_array()
    # can finally use number of lines in checkpoint file and the number of rows in zcat to
    # verify whether the progress is completed
    logging.info("Starting to process files")
    Parallel(n_jobs=num_workers, prefer="threads")(
        delayed(write_one_spec_to_h5)
        (row, redux, h5fpath, chkpt_dir, row["TARGETID"] % n_component if n_component > 1 else None)
        for row in zcat_arr
    )

    
def write_one_fits_to_h5(fits_path: Path, redux: str, h5_dir: "str | Path", chkpt_dir=None,
                         n_component=1, proc_idx=-1):
    import h5py
    h5_dir = Path(h5_dir)
    fits_spec = read_desi_spec_from_path(fits_path)
    target_ids = fits_spec.target_ids().data
    h5parts = target_ids % n_component
    unique_idx = np.unique(h5parts)
    # shuffle to avoid racing for resource in parallel
    np.random.shuffle(unique_idx)
    for idx in unique_idx:
        pick_tar = h5parts == idx
        # make the desi_spec array
        to_write = [
            (
                str(target_ids[i]),
                np.stack(
                    [
                        fits_spec.wave["brz"].flatten(), # all wave the same
                        fits_spec.flux["brz"][i].flatten(),
                        fits_spec.ivar["brz"][i].flatten(),
                        fits_spec.mask["brz"][i].flatten()
                    ],
                    axis=1
                )
            )
            for i in np.where(pick_tar)[0]
        ]
        filename = f"{idx:04}-{redux}-targetid-healpix-coadd.h5"
        dset_name = f"{redux}-{fits_path.stem}"
        # this might corrupt files for a very long hanging files
        with h5py.File(h5_dir / filename, mode='a') as h5f:
            for i, (targetid, wave_flux_ivar_mask) in enumerate(to_write):
                if "targetid_spec" not in h5f.keys():
                    h5f.create_group("targetid_spec")
                spec_dset = h5f["targetid_spec"]
                if targetid not in spec_dset.keys():
                    spec_dset.create_group(targetid)
                tar_dset = spec_dset[targetid]
                if dset_name in tar_dset.keys():
                    # this would mean that data has written into this file before
                    # but we are interrupted somehow
                    logging.warning(f"{dset_name} exists in {filename}")
                    if not np.array_equal(tar_dset[dset_name], wave_flux_ivar_mask):
                        tar_dset[dset_name] = wave_flux_ivar_mask
                else:
                    tar_dset.create_dataset(dset_name, data=wave_flux_ivar_mask)
                if i > 0 and i % 5 == 0:
                    # try to minimize the risk of corrupting files by
                    # flushing more frequently, so context manager will
                    # close faster after cataching exception
                    # 1.2 MB per flush
                    h5f.flush()
    if chkpt_dir is not None:
        chkpt_dir = Path(chkpt_dir)
        # checkpoint file format f"{i:04}-targetid_{h5fpath.stem}.txt"
        # line in the file foramt {redux}-coadd-{survey}-{program}-{healpix}-{targetid}"
        chkpt_file = f"{redux}_progress.txt"
        if proc_idx >= 0:
            chkpt_file = f"proc_{proc_idx}_" + chkpt_file
        with open(chkpt_dir / chkpt_file, 'a') as fp:
            fp.write(fits_path.stem + "\n")


def list_all_fits(dr, redux, catalog="healpix"):
    import os
    
    root_dir = Path("/global/cfs/cdirs/desi/public", dr, "spectro/redux", redux, catalog)
    fits_files = []
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if file.startswith("coadd") and file.endswith(".fits"):
                fits_files.append(os.path.join(file))
    return fits_files


def load_checkpoint(chkpt_dir: "str | Path"):
    chkpt_dir = Path(chkpt_dir)
    progress = set()
    for file in chkpt_dir.iterdir():
        if file.is_dir():
            continue
        with open(file, 'r') as fp:
            progress |= set([l.strip() for l in fp])
    return progress


def infer_healpix_path_from_name(fits_name, redux):
    # root_dir = Path("/global/cfs/cdirs/desi/public", dr, "spectro/redux", redux, catalog)
    # edr/spectro/redux/fuji/healpix/special/dark/93/9306/coadd-special-dark-9306.fits
    # format: /global/cfs/cdirs/desi/public/{dr}/spectra/reudx/{redux}/{catalog}/{survey}/{program}/{hp}/{healpix}/{fits}
    # file name: f'coadd-{survey}-{program}-{healpix}.fits'
    _, survey, program, healpix = fits_name.rsplit(".")[0].split("-")
    return find_desi_spec_path(redux, survey, program, healpix)


def create_spec_h5_by_fits(dr: str, redux: str, new_spec_root: Path, 
                           num_workers=10, chkpt_dir=None, n_component=1):
    from joblib import Parallel, delayed
    
    all_fits_file = new_spec_root / f"{dr}_{redux}_all_fits.txt"
    if all_fits_file.exists():
        logging.info("Loading checkpoint from: " + str(all_fits_file))
        with open(all_fits_file, 'r') as fp:
            files = set([l.strip() for l in fp])
        progress = load_checkpoint(chkpt_dir)
        todo_files = list(files - progress)
    else:
        logging.info("Finding all FITS and making checkpoint file")
        todo_files = list_all_fits(dr, redux, catalog="healpix")
        with open(all_fits_file, 'w') as fp:
            fp.write("\n".join(todo_files))
    if chkpt_dir is not None and not chkpt_dir.exists():        
        chkpt_dir.mkdir()
    logging.info("Starting to process files")
    np.random.shuffle(todo_files)
    Parallel(n_jobs=num_workers, prefer="threads", verbose=100)(
        delayed(write_one_fits_to_h5)
        (infer_healpix_path_from_name(file, redux), redux, new_spec_root, chkpt_dir, n_component, i % num_workers)
        for i, file in enumerate(todo_files)
    )
    logging.info("Done.")


def make_h5_index(redux, new_spec_root: Path, index_dir: Path):
    idx_fname = f"index-{redux}-targetid-healpix-coadd.h5"
    with h5py.File(index_dir / idx_fname, 'w') as h5f:
        for fpath in new_spec_root.glob("[0-9]"*4 + "*.h5"):
            # this requires h5 spectra are exactly one level down the index h5 file dir
            h5f["tm" + fpath.stem.split("-")[0]] = h5py.ExternalLink(fpath.parent.name + "/" + fpath.name, "/targetid_spec")


def get_parser():
    parser = argparse.ArgumentParser(description="Create new DESI spectra dataset from FITS files")
    parser.add_argument("--zcat", type=Path, default=None,
                        help='Path to the zcatalog.fits file. If None, will try to convert by FITS files rather than ' \
                            "rows in the zcat")
    parser.add_argument("--dr", type=str, required=True,
                        help='DR name (edr, dr1...)')
    parser.add_argument("--redux", type=str, required=True,
                        help='redux name: fuji, guadalupe...')
    parser.add_argument("--new_spec_root", "-dir", type=Path, required=True,
                        help='Path to directory holding the new h5 spec data')
    parser.add_argument("--h5filename", "-fname", type=str,
                        help='file name for the new HDF5 file holding spectra data')
    parser.add_argument("--num_workers", "-n", type=int, default=10,
                        help='number of workers to make the dataset')
    parser.add_argument("--chkpt_dir", "-chk", type=Path, default=None,
                        help='directory to store checkpoints when processing files. If not provided, ignore checkpointing')
    parser.add_argument("--num_component", "-n_comp", type=int, default=1,
                        help='Number of separate component files to write into')
    return parser
    
    
def handle_sigterm(signum, frame):
    print("Caught SIGTERM, exiting with status 0...")
    sys.exit(0)

    
if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    signal.signal(signal.SIGTERM, handle_sigterm)
    logging.basicConfig(level=logging.INFO)
    if args.zcat is None:
        create_spec_h5_by_fits(args.dr, args.redux, args.new_spec_root, args.num_workers,
                   args.chkpt_dir, args.num_component)
    else:
        create_spec_h5_by_zcat(args.zcat, args.redux, args.new_spec_root, args.h5filename, args.num_workers,
                       args.chkpt_dir, args.num_component)
    make_h5_index(args.redux, args.new_spec_root, args.new_spec_root.parent)

