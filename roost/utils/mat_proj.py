from ast import literal_eval

import pandas as pd
from pymatgen import MPRester

from roost.utils.io import ROOT

# Materials Project API keys available at https://materialsproject.org/dashboard.
API_KEY = "X2UaF2zkPMcFhpnMN"


def fetch_mp(criteria={}, properties=[], save_to=None):
    """
    Fetch data from the Materials Project (MP). Docs at https://docs.materialsproject.org.
    Pymatgen MP source at https://pymatgen.org/_modules/pymatgen/ext/matproj.

    Note: Unlike ICSD - a database of materials that actually exist - MP has
    all structures where DFT+U converges. Those can be thermodynamically
    unstable if they lie above the convex hull. Set criteria = {"e_above_hull": 0}
    to get stable materials only.

    Args:
        criteria (dict, optional): filter criteria which returned items must
            satisfy, e.g. criteria = {"material_id": {"$in": ["mp-7988", "mp-69"]}}.
            Supports all features of the Mongo query syntax.
        properties (list, optional): quantities of interest, can be selected from
            https://materialsproject.org/docs/api#resources_1 or MPRester().supported_properties.
        save_to (str, optional): Pass a file path to save the data returned by MP
            API as CSV. Defaults to None.

    Returns:
        df: pandas DataFrame with a column for each requested property
    """

    properties = list({*properties, "material_id"})  # use set to remove dupes

    # MPRester connects to the Materials Project REST API
    with MPRester(API_KEY) as mp:
        # mp.query performs the actual API call
        data = mp.query(criteria, properties)

    if data:
        df = pd.DataFrame(data)[properties]  # ensure same column order as in properties

        df = df.set_index("material_id")

        if save_to:
            data.to_csv(ROOT + save_to, float_format="%g")

        return df
    else:
        raise ValueError("query returned no data")


def volume_from_lattice(lat):
    """Compute unit cell volume from list of lattice vectors via the
    scalar triple product of the unit vectors (14 operations / det).

    Args:
        lat ([Seq[Seq[float]]]): the lattice vectors

    Returns:
        [float]: the unit cell volume
    """
    u, v, w = lat
    d = u[0] * (v[1] * w[2] - v[2] * w[1])
    d += u[1] * (v[2] * w[0] - v[0] * w[2])
    d += u[2] * (v[0] * w[1] - v[1] * w[0])
    return d


def df_vol_from_lat(df):
    """Add volume column to a dataframe expected to have a lattice column
    specifying a list of lattice vectors.
    """
    # literal_eval used to convert strings to lists: '[1, 2, 3]' -> [1, 2, 3]
    df.lattice = df.lattice.apply(literal_eval)
    df.volume = df.lattice.apply(volume_from_lattice)
    return df
