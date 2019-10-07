from ripser import ripser, lower_star_img
from persim import PersImage
import h5py
import cv2
import subprocess
import os, sys
import pandas as pd
import numpy as np
from pathlib import Path
import fnmatch
from tqdm import tqdm


class HiddenPrints:
    """For blocking the annoying print output of PersImage, taken from
       https://stackoverflow.com/a/45669280."""

    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, "w")

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout


def persistence_image(img_filepath, var, output_dim):
    """Calculate the persistence image of the 0-dimensional homology diagram.

    The 0-dimensional homology is calculated using Ripser and converted to the
    correct format for the persim module, which constructs the image with
    a Gaussian kernel with a certain variance (var).
    The persistence image resolution is controlled by the output_dim parameter.
    """
    img = cv2.imread(img_filepath, 0)
    dgm = lower_star_img(img)
    dgm[~np.isfinite(dgm)] = -1
    dgm = dgm.astype(int)
    with HiddenPrints():
        imgs_obj = PersImage(spread=var, pixels=[output_dim, output_dim])
    pers_img = imgs_obj.transform(dgm)
    return pers_img


def persistence_image_loop(img_path_string, data_filename, output_dim, spread):
    """Create a HDF5 file of persistence images from a folder of images.

    The path to the image folder can be given relative to the working directory
    of this script, or can be given as an absolute path.

    Images are progressively added to data_filename.h5, labeled by their filename.
    """
    img_path = Path(img_path_string)
    n_files = len(fnmatch.filter(os.listdir(img_path), "*.png"))
    with h5py.File(data_filename + ".h5", mode="w") as store:
        with os.scandir(img_path) as folder:
            for entry in tqdm(folder, total=n_files):
                if entry.name.endswith(".png") and entry.is_file():
                    filepath = (img_path / entry.name).as_posix()
                    store.create_dataset(
                        name=entry.name,
                        data=persistence_image(filepath, spread, output_dim),
                    )
    print("Images written to " + data_filename + ".h5")


def extract_images(data_filename):
    """Convert an HDF5 file of images into a numpy array.

    Returns a dictionary with the numpy array and the associated image IDs.
    """
    with h5py.File(data_filename + ".h5", mode="r") as data:
        names = list(data.keys())
        dim = np.array(data[names[0]]).shape[0]
        out = np.zeros((len(names), dim, dim))
        for i, name in enumerate(names, start=0):
            img = np.array(data[name])
            out[i] = img
    return {"image_names": names, "pers_images": out}
