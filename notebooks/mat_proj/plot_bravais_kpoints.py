# %%
import gzip
import os
import pickle

import matplotlib.pyplot as plt
from pymatgen import MPRester

from roost.utils import API_KEY, ROOT, fetch_mp

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
    ["pretty_formula", "spacegroup.number", "spacegroup.crystal_system"],
).rename(
    columns={
        "pretty_formula": "formula",
        "spacegroup.number": "spacegroup",
        "spacegroup.crystal_system": "crystal_system",
    }
)

spacegroups.to_csv(f"{DIR}/spacegroups_of_downloaded_band_structs.csv")


# %%
spacegroups.value_counts("crystal_system")


# %%
idx = []
for key in spacegroups.value_counts("crystal_system").keys():
    idx += spacegroups[spacegroups.crystal_system == key].head(3).index.tolist()

df = spacegroups.loc[idx]


# %%
fig = plt.figure(figsize=[15, 35])

for idx, (id, row) in enumerate(df.iterrows(), 1):
    with gzip.open(f"{DIR}/zip/{id}:{row.formula}.zip", "rb") as file:
        bs_dict = pickle.loads(file.read())

    ax = fig.add_subplot(7, 3, idx, projection="3d")

    x, y, z = zip(*bs_dict["kpoints"])
    if (idx + 1) % 3 == 0:
        ax.set_title(f"{row.crystal_system}\n\n{row.formula} ({id})")
    else:
        ax.set_title(f"{row.formula} ({id})")

    ax.scatter(x, y, z)
    ax.set_xlabel("$k_x$")
    ax.set_ylabel("$k_y$")
    ax.set_zlabel("$k_z$")

fig.tight_layout(h_pad=5)

fig.savefig("kpoints.png", bbox_inches="tight", dpi=300)
