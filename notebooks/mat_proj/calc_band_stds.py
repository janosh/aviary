"""
Calculates the standard deviation (and mean) of the first 32 bands in a material's
brand structure. Inspired by Yunwei Zhang's idea to use the std of around 5 bands
closest to the Fermi level to classify a material as superconductor.
"""


# %%
import gzip
import os
import pickle

import numpy as np
import pandas as pd
from tqdm.std import tqdm

from aviary.utils import ROOT


# %%
DIR = f"{ROOT}/data/datasets/bandstructs"
files = os.listdir(f"{DIR}/zip")


# %%
band_stds, band_means, band_idxs = [], [], []
mp_ids, formulas, e_fermis = [], [], []
n_bands = 32


# %%
def find_n_nearest_idx(array, value, n=1):
    idx = np.argsort(np.abs(array - value))
    return idx[:n]


# %%
for filename in tqdm(files):
    with gzip.open(f"{DIR}/zip/{filename}", "rb") as file:
        bs_dict = pickle.loads(file.read())

    mp_id, formula = filename.replace(".zip", "").split(":")

    bands = bs_dict["bands"]["1"][:n_bands]

    if len(bands) < n_bands:
        continue

    means = np.mean(bands, axis=1)
    efermi = bs_dict["efermi"]
    band_means.append(means)

    stds = np.std(bands, axis=1)
    band_stds.append(stds)

    near_fermi_band_idxs = find_n_nearest_idx(means, efermi, 5)
    band_idxs.append(near_fermi_band_idxs)

    # use indices of the five bands with mean nearest the fermi level shifted down by 1
    # i.e. bias towards occupied bands
    stds = np.std(bands, axis=1)[near_fermi_band_idxs - 1]
    band_stds.append(stds)

    mp_ids.append(mp_id)
    formulas.append(formula)
    e_fermis.append(efermi)


# %%
df = pd.DataFrame([mp_ids, formulas, e_fermis, *zip(*band_means), *zip(*band_stds)]).T
col_names = lambda postfix: [f"band_{idx + 1}_{postfix}" for idx in range(n_bands)]

# df.columns = ["id", "formula", "e_fermi", *col_names("mean"), *col_names("std")]
df.columns = ["id", "formula", "e_fermi", *[f"band_std_{idx + 1}" for idx in range(5)]]
df = df.set_index("id")


# %%
# df.to_csv(f"{DIR}/yunwei_band_stds.csv", float_format="%g")
df.to_csv(f"{DIR}/32_bands_std+mean.csv", float_format="%g")


# reference values from Yunwei for NbN (mp-2634):
# 0.8331   0.6247   1.0066   1.1908   0.9328
df.loc["mp-2634"]


# %%
df = pd.read_csv(f"{DIR}/yunwei_band_stds.csv", index_col="id")


# %%
df.value_counts("formula").sort_values()


# %%
# data contains duplicate formulas (polymorphs)
n_poly = sum(df.value_counts("formula").sort_values() > 1)
print(f"number of polymorphs: {n_poly}")
df.value_counts("formula").sort_values().tail(50)
