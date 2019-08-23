import cv2
import subprocess
import os
import pandas as pd
import numpy as np
import lungs_finder as lf
import fnmatch
from pathlib import Path
from tqdm import tqdm


def persistence_stats(ints):
    """Compute summary statistics from lifetimes of persistence intervals."""
    ints.replace(-1, 255, inplace=True)
    life = ints["Death"] - ints["Birth"]
    n_life = life / sum(life)
    midlife = (ints["Birth"] + ints["Death"]) / 2
    data = pd.concat([n_life.rename("n_life"), midlife.rename("midlife")], axis=1)
    summary = data.describe().iloc[1:7, :]
    summary.loc["skewness"] = data.skew()
    summary.loc["kurtosis"] = data.kurtosis()
    stack = summary.stack().swaplevel()
    stack.loc["entropy"] = -sum(n_life * np.log(n_life))
    return stack


def perseus_summarise(img_filepath, img_filename):
    """Find lungs in x-ray and compute their persistence curves and statistics.

    You must have installed Perseus from
    http://people.maths.ox.ac.uk/nanda/perseus/index.html
    and added the application to your $PATH folder or equivalent.
    To check this, type "perseus" in your shell."""
    img = cv2.imread(img_filepath, 0)
    lungs = lf.get_lungs(img)
    # When get_lungs fails, just use the whole image
    if len(lungs.T) == 0:
        lungs = img
    img_dim = np.shape(lungs)
    perseus_list = np.concatenate([np.array([2]), img_dim, lungs.flatten()])
    with open("temp.txt", "w") as temp:
        temp.write("\n".join(str(n) for n in perseus_list))
        subprocess.check_output("perseus cubtop temp.txt out")
        ints_0 = pd.read_csv("out_0.txt", sep=" ", names=["Birth", "Death"])
        ints_1 = pd.read_csv("out_1.txt", sep=" ", names=["Birth", "Death"])
        betti = pd.read_csv(
            "out_betti.txt", sep=" ", usecols=[2, 3], names=["b0", "b1"]
        )
        # Betti numbers are not always listed for small t, so we pad with zeros
        zeros = pd.DataFrame(np.zeros((256 - len(betti), 2)), columns=["b0", "b1"])
        betti = pd.concat([zeros, betti])
        betti.index = range(0, 256)
        betti_dat = betti.stack().swaplevel()
        stats_0 = persistence_stats(ints_0)
        stats_1 = persistence_stats(ints_1)
        series_list = [betti_dat, stats_0, stats_1]
        df = pd.concat(series_list).to_frame()
        df.columns = [img_filename]
    return df.T


def perseus_loop(img_path_string):
    """Create dataframe of topological features from folder of CXR images.

    The path to the image folder can be given relative to the working directory
    of this script, or can be given as an absolute path.
    Uses tqdm to track progress."""
    img_path = Path(img_path_string)
    out = []
    n_files = len(fnmatch.filter(os.listdir(img_path), "*.png"))
    with os.scandir(img_path) as folder:
        for entry in tqdm(folder, total=n_files):
            if entry.name.endswith(".png") and entry.is_file():
                filepath = (img_path / entry.name).as_posix()
                try:
                    out.append(perseus_summarise(filepath, entry.name))
                except subprocess.CalledProcessError:
                    print("Perseus error on image " + entry.name)
                    pass
    df = pd.concat(out)
    df.columns = pd.MultiIndex.from_tuples(df.columns)
    return df
