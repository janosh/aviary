# %%
import gzip
import os
import pickle

import matplotlib.pyplot as plt
from pymatgen.electronic_structure.bandstructure import BandStructureSymmLine
from pymatgen.electronic_structure.plotter import BSPlotter
from pymatgen.ext.matproj import MPRester

from aviary.utils import API_KEY, ROOT, fetch_mp


# %%
DIR = f"{ROOT}/data/datasets/bandstructs"
files = os.listdir(f"{DIR}/zip")

mp_ids = [f.split(":")[0] for f in files]


# %%
mpr = MPRester(API_KEY)
mp_version = mpr.get_database_version()


# %%
spacegroups = fetch_mp(
    {"material_id": {"$in": mp_ids}},
    [
        "pretty_formula",
        "spacegroup.number",
        "spacegroup.crystal_system",
        "nelements",
        "spacegroup.symbol",
    ],
).rename(
    columns={
        "pretty_formula": "formula",
        "spacegroup.number": "spacegroup",
        "spacegroup.symbol": "symbol",
        "spacegroup.crystal_system": "crystal_system",
    }
)

# spacegroups.to_csv(f"{DIR}/spacegroups_of_downloaded_band_structs.csv")


# %%
spacegroups["bravais"] = spacegroups.apply(
    lambda row: row.crystal_system + row.symbol[0], axis=1
)


# %%
spacegroups.value_counts("crystal_system")


# %%
spacegroups.value_counts("nelements")


# %%
idx = []
for key, val in spacegroups.value_counts("bravais").items():
    if val > 3:
        idx += spacegroups[spacegroups.bravais == key].head(3).index.tolist()

df = spacegroups.loc[idx]


# %% Plot k-points only
fig = plt.figure(figsize=[15, 65])
fig.patch.set_facecolor("white")

for idx, (id, row) in enumerate(df.iterrows(), 1):
    with gzip.open(f"{DIR}/zip/{id}:{row.formula}.zip", "rb") as file:
        bs_dict = pickle.loads(file.read())

    ax = fig.add_subplot(13, 3, idx, projection="3d")

    x, y, z = zip(*bs_dict["kpoints"])
    if (idx + 1) % 3 == 0:
        ax.set_title(f"{idx // 3 + 1} {row.bravais}\n\n{row.formula} ({id})")
    else:
        ax.set_title(f"{row.formula} ({id})")

    for key, pos in bs_dict["labels_dict"].items():
        ax.text(*pos, f"${key}$", size=20, zorder=10, color="black")

    ax.scatter(x, y, z)
    ax.set_xlabel("$k_x$")
    ax.set_ylabel("$k_y$")
    ax.set_zlabel("$k_z$")

fig.tight_layout(h_pad=5)

fig.savefig("3_kpoints_per_bravais.png", bbox_inches="tight", dpi=200)


# %% Plot k-points embedded in their respective Brillouin zones
fig = plt.figure(figsize=[15, 65])
fig.patch.set_facecolor("white")

for idx, (id, row) in enumerate(df.iterrows(), 1):
    with gzip.open(f"{DIR}/zip/{id}:{row.formula}.zip", "rb") as file:
        bs_dict = pickle.loads(file.read())

    ax = fig.add_subplot(13, 3, idx, projection="3d")

    if (idx + 1) % 3 == 0:
        ax.set_title(f"{idx // 3 + 1} {row.bravais}\n\n{row.formula} ({id})")
    else:
        ax.set_title(f"{row.formula} ({id})")
    BSPlotter(BandStructureSymmLine.from_dict(bs_dict)).plot_brillouin(ax=ax)

fig.tight_layout(h_pad=5)

fig.savefig("3_brillouin_per_bravais.png", bbox_inches="tight", dpi=200)
