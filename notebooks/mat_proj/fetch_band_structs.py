# %%
import gzip
import os
import pickle
import sys

import pandas as pd
from pymatgen.ext.matproj import MPRester, MPRestError
from tqdm import tqdm

from roost.utils import API_KEY, ROOT, fetch_mp

# %%
mpr = MPRester(API_KEY)
mp_version = mpr.get_database_version()
print("MP version:", mp_version)

save_dir = f"{ROOT}/data/datasets/bandstructs"
os.makedirs(save_dir, exist_ok=True)


# %% [markdown]
# # Fetch all binary material IDs

# %%
# fetch all binary MP materials (only on first run, data is stored and loaded below)
mp_ids = fetch_mp({"nelements": 2}, ["pretty_formula"]).rename(
    columns={"pretty_formula": "formula"}
)


# %%
# mp_ids.to_csv(f"{save_dir}/mp_binary_ids_db@{mp_version}.csv")
mp_ids = pd.read_csv(
    f"{save_dir}/mp_binary_ids_db@{mp_version}.csv", index_col="material_id"
)


# %% [markdown]
# # Fetch all IDs and spacegroup data for materials with 3 or less elements


# %%
# fetch spacegroup data for all materials with cubic lattice
# (eq. to space groups 195 to 230) and 5 or less elements
spacegroups = fetch_mp(
    {"nelements": {"$lte": 5}, "spacegroup.crystal_system": "cubic"},
    ["pretty_formula", "spacegroup"],
).rename(columns={"pretty_formula": "formula"})


# %%
# extract relevant data from spacegroup dict into individual columns
for col in ["number", "crystal_system", "symbol"]:
    spacegroups[col] = spacegroups.spacegroup.apply(lambda x: x[col])
spacegroups = spacegroups.drop(columns=["spacegroup"]).rename(
    columns={"number": "spacegroup_number"}
)

# %%
# save all materials with 5 or less elements along with
# their crystal system and and spacegroup number to csv
# spacegroups.to_csv(f"{save_dir}/mp_spacegroups_nelements<6_db@{mp_version}.csv")
spacegroups = pd.read_csv(
    f"{save_dir}/mp_spacegroups_nelements<6_db@2020_09_08.csv",
    index_col="material_id",
)


# %%
# count materials by crystal_system
mp_ids = spacegroups
spacegroups.value_counts("spacegroup_number")

# %% [markdown]
# # Testing


# %% get single band structure for testing
bs = mpr.get_bandstructure_by_material_id("mp-985582")
bs_dict = bs.as_dict()

# the largest storage requirement comes from bs.projections
# which we don't need so we delete it
for key in bs_dict.keys():
    bs_dict_val = pickle.dumps(bs_dict[key])
    print(key, sys.getsizeof(bs_dict_val))


# %%
# fetch and store the band structures (takes forever)
mp_ids["comment"] = ""  # create column to record errors for fetch attempts below
subset = mp_ids.formula.iloc[1000:2000]
n_ids = len(subset)
percent = -1

for idx, (id, formula) in enumerate(subset.iteritems()):
    file_path = f"{save_dir}/{id}:{formula}.zip"
    if idx * 100 // n_ids > percent:
        percent = idx * 100 // n_ids
        print(f"{percent}%")
    prog = f"{idx:>4}/{n_ids}"

    if os.path.isfile(file_path):
        print(f"{prog}: '{id}:{formula}.zip' already exists, continuing")
        continue

    if "Error" in mp_ids.comment[id]:
        print(mp_ids.comment[id])
        print(f"{prog}: {id} ({formula}) failed before, continuing")
        continue

    try:
        bs = mpr.get_bandstructure_by_material_id(id)
    except MPRestError:
        print(f"{prog}: query for {id} ({formula}) threw MPRestError")
        # https://matsci.org/t/15592
        mp_ids.comment[id] = "MPRestError: too many indices for array"
        continue

    try:
        bs_dict = bs.as_dict()
    except AttributeError:
        print(f"{prog}: query for '{id}' ({formula}) returned None")
        mp_ids.comment[id] = "AttributeError: bandstructure was None"
        continue

    del bs_dict["projections"]

    # %% save bandstructure dictionaries as pickled ZIP files, seems to be
    # best trade-off between speed and storage size
    with gzip.open(file_path, "wb") as file:
        pickle.dump(bs_dict, file, protocol=-1)

    print(f"{prog}: '{id}:{formula}.zip' saved")


# %%
# save comments column, allows skipping throwing or null entries next time
spacegroups.to_csv(f"{save_dir}/mp_spacegroups_nelements<6_db@{mp_version}.csv")


# %%
# how to load the stored band structs from disk
mp_ids["is_metal"] = None

for id, formula in tqdm(mp_ids.formula.head(100).iteritems()):
    file_path = f"{save_dir}/{id}:{formula}.zip"

    if os.path.isfile(file_path):

        with gzip.open(f"{save_dir}/{id}:{formula}.zip", "rb") as file:
            bs = pickle.loads(file.read())

        mp_ids.is_metal.loc[id] = bs["is_metal"]
