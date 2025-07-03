from typing import Literal

import dask
import dask.bag as db
import pandas as pd

from .features import FeatureMap
from .spectrums import SpectrumMap


def link_ms2_to_feature(feature_hulls: pd.DataFrame,spectrum_map: SpectrumMap) -> list[str]:

    spectrum_id_list = []
    for mz_start,rt_start,mz_end,rt_end in zip(
        feature_hulls['MZstart'],
        feature_hulls['RTstart'],
        feature_hulls['MZend'],
        feature_hulls['RTend'],
    ):
        spectrum_id_list += spectrum_map.search_ms2_by_range(
            (mz_start,rt_start,mz_end,rt_end)
        )
    return spectrum_id_list

def link_ms2_and_feature_map(
    feature_map: FeatureMap,
    spectrum_map: SpectrumMap,
    key_id: Literal["feature","spectrum"] = "feature",
) -> pd.Series:

    feature_id_bag = db.from_sequence(
        zip(feature_map.feature_info.index, feature_map.feature_info['hull_num'])
    )
    feature_hulls_id_bag = feature_id_bag.map(
        lambda x: [x[0]+f"::{i}" for i in range(x[1])]
    )
    feature_hulls_bag = feature_hulls_id_bag.map(
        lambda x: feature_map.hulls.loc[x]
    )
    spectrum_id_bag = feature_hulls_bag.map(
        lambda x: link_ms2_to_feature(x,spectrum_map)
    )
    spectrum_id_list = dask.compute(
        spectrum_id_bag, scheduler="threads"
    )[0]
    if key_id == "feature":
        mapping_series = pd.Series(spectrum_id_list, index=feature_map.feature_info.index)
        mapping_series.index.name = "feature_id"
        mapping_series.name = "spectrum_id"
        return mapping_series
    else:
        mapping_series = pd.Series()
        mapping_series.index.name = "spectrum_id"
        mapping_series.name = "feature_id"
        for spectrum_ids, feature_id in zip(spectrum_id_list, feature_map.feature_info.index):
            for spectrum_id in spectrum_ids:
                mapping_series[spectrum_id] = feature_id
        return mapping_series
