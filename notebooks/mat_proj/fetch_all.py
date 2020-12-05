# %%
import pandas as pd
from pymatgen import MPRester

from roost.utils import ROOT, fetch_mp

# %%
id = ["material_id", "pretty_formula", "unit_cell_formula", "icsd_ids"]
energy = ["energy", "formation_energy_per_atom", "e_above_hull"]
electronic = ["dos", "bandstructure_uniform", "band_gap"]
structure = ["spacegroup", "nsites", "final_structure", "volume", "density"]
other = ["total_magnetization", "task_ids", "is_hubbard"]

all = id + energy + electronic + structure + other


# %%
# use empty criteria dict to fetch all of MP
df_mp = fetch_mp({}, all).rename(columns={"unit_cell_formula": "composition"})


# %%
def struct_to_sites(struct):

    eles = [atom.specie.symbol for atom in struct]
    coords = struct.frac_coords
    sites = [" @ ".join((el, " ".join(map(str, x)))) for el, x in zip(eles, coords)]
    return sites


df_mp.spacegroup = df_mp.spacegroup.apply(lambda x: x["number"])
df_mp.lattice = df_mp.final_structure.apply(
    lambda struct: struct.lattice.matrix.tolist()
)
df_mp.sites = df_mp.final_structure.apply(struct_to_sites)


# %%
df_mp.drop(columns=["final_structure"], inplace=True)


# %%
mp_version = MPRester().get_database_version()
csv_path = f"{ROOT}/data/datasets/large/mp_{mp_version}.csv"
df_mp.to_csv(csv_path)
df_mp = pd.read_csv(csv_path)


# %% add 'is_metal' column (assumes df.index is a list of MP IDs)
# we could also query for the band structure with its is_metal attribute.
# Would be slower since band structures contain lots of other data.

df_mp["band_gap"] = fetch_mp(
    {"material_id": {"$in": df_mp.index.to_list()}},
    ["band_gap", "material_id"],
)
df_mp["is_metal"] = df_mp.band_gap.apply(lambda bg: 1 if bg > 0 else 0)
