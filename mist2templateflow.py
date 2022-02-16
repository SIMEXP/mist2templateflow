
"""
Downloading NeuroImaging datasets: atlas datasets
"""
from pathlib import Path
import shutil
import os
import numpy as np
import pandas as pd

from sklearn.utils import Bunch

from nilearn.datasets.utils import _get_dataset_dir, _fetch_files
from nilearn.image import load_img


# download the release to data/raw
DOWNLOAD_URL = "https://figshare.com/ndownloader/files/9811081"
TEMPLATE = "MNI152NLin2009bSym"
ATLAS = "MIST"
DESCRIPTIONS = [7, 12, 20, 36, 64, 122, 197, 325, 444, "ROI", "ATOM", "Hierarchy"]

INPUT_DIR = "./data/raw"
OUTPUT_DIR = "./data/processed"


def fetch_atlas_mist(dimension, data_dir=None, url=None, resume=True, verbose=1):
    """Downloads MIST from https://figshare.com/ndownloader/files/9811081"""
    if dimension not in DESCRIPTIONS:
        raise ValueError(f"{dimension} doesn't exist.")
    if url is None:
        url = DOWNLOAD_URL

    opts = {'uncompress': True}

    dataset_name = "MIST2019"
    data_dir = _get_dataset_dir(dataset_name, data_dir=data_dir,
                                verbose=verbose)
    if dimension == "Hierarchy":
        filenames = [(os.path.join("Release", "Hierarchy", "MIST_PARCEL_ORDER_ROI.csv"), url, opts),
                     (os.path.join("Release", "Hierarchy", "MIST_PARCEL_ORDER.csv"), url, opts),
                     ]
        keys = ["Hierarchy_ROI", "Hierarchy"]

    elif dimension == "ATOM":
        filenames = [(os.path.join("Release", "Parcellations", "MIST_ATOM.nii.gz"), url, opts),]
        keys = ["maps"]
    else:
        filenames = [(os.path.join("Release", "Parcellations", f"MIST_{dimension}.nii.gz"), url, opts),
                    (os.path.join("Release", "Parcel_Information", f"MIST_{dimension}.csv"), url, opts)
                    ]
        keys = ["maps", "labels"]

    files_ = _fetch_files(data_dir, filenames, resume=resume, verbose=verbose)
    params = dict(zip(keys, files_))
    if dimension == "ATOM":
        atom_img = load_img(files_[0])
        n_atoms = np.unique(atom_img.dataobj).shape[-1]
        params["labels"] = list(range(1, n_atoms))

    tpf = convert_templateflow(TEMPLATE, ATLAS, dimension)
    params.update(tpf)

    return Bunch(**params)


def convert_templateflow(template, atlas, desc):
    folder_name = f"tpl-{template}"
    if desc != "Hierarchy":
        basenames = f"tpl-{template}_res-03_atlas-{atlas}_desc-{desc}_dseg"
        keys = ["tpf_maps", "tpf_labels"]
        filenames = [os.path.join(folder_name, f'{basenames}.nii.gz'),
                    os.path.join(folder_name, f'{basenames}.tsv')]
    else:
        descs = ["ParcelHierarchyROI", "ParcelHierarchy"]
        filenames = [os.path.join(folder_name, f"tpl-{template}_res-03_atlas-{atlas}_desc-{desc}_dseg.tsv")
                     for desc in descs]
        keys = [f"tpf_{desc}" for desc in descs]
    return dict(zip(keys, filenames))


if __name__ == "__main__":
    for desc in DESCRIPTIONS:
        dataset = fetch_atlas_mist(desc, data_dir=INPUT_DIR)
        if desc != "Hierarchy":
            nii = Path(dataset['maps'])
            output_file = Path(OUTPUT_DIR) / dataset['tpf_maps']

            if not output_file.parent.is_dir():
                output_file.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(nii, output_file)

            if desc == "ATOM":
                labels = pd.DataFrame(dataset['labels'], columns=["roi"])
            else:
                labels = pd.read_csv(dataset['labels'], sep=';')
            labels.to_csv(os.path.join(OUTPUT_DIR, dataset['tpf_labels']), index=False, sep='\t')
        else:
            for label, tpf in zip(["Hierarchy_ROI", "Hierarchy"], ["ParcelHierarchyROI", "ParcelHierarchy"]):
                df = pd.read_csv(dataset[label])
                df.to_csv(os.path.join(OUTPUT_DIR, dataset[f"tpf_{tpf}"]), index=False, sep='\t')
