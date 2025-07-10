from __future__ import annotations

from typing import Literal

import dask
import dask.bag as db
import numpy as np
import pandas as pd
import pyopenms as oms
from pydantic import BaseModel, ConfigDict


class FeatureMap(BaseModel):

    model_config = ConfigDict({"arbitrary_types_allowed": True})

    feature_info: pd.DataFrame | None = None
    hulls: pd.DataFrame | None = None

    @staticmethod
    def get_feature_metadata(feature: oms.Feature) -> dict[
        Literal[
            'hull_num',"hull_mz","hull_rt","hull_intensity",
            "isotope_pattern",
            "adduct_type","adduct_mass",
        ],
        str | float | int | np.ndarray
    ]:
        all_keys = []
        feature.getKeys(all_keys)
        all_keys = set(all_keys)
        metadata = {
            "hull_num": feature.getMetaValue("num_of_masstraces"),
            "hull_mz": np.array(feature.getMetaValue("masstrace_centroid_mz")),
            "hull_rt": np.array(feature.getMetaValue("masstrace_centroid_rt")),
            "hull_intensity": np.array(feature.getMetaValue("masstrace_intensity")),
            "isotope_pattern": np.cumsum(feature.getMetaValue("isotope_distances")),
        }
        if "dc_charge_adducts" in all_keys:
            metadata["adduct_type"] = feature.getMetaValue("dc_charge_adducts")
            metadata["adduct_mass"] = feature.getMetaValue("dc_charge_adduct_mass")
        return metadata

    @staticmethod
    def get_feature_info(feature_map: oms.FeatureMap) -> pd.DataFrame:
        feature_info = feature_map.get_df()[
            ["RT","mz","intensity","MZstart","RTstart","MZend","RTend"]
        ]
        feature_bag = db.from_sequence(feature_map)
        feature_metadata_bag = feature_bag.map(FeatureMap.get_feature_metadata)
        feature_metadata_list = dask.compute(
            feature_metadata_bag, scheduler="threads"
        )[0]
        feature_metadata_df = pd.DataFrame(
            feature_metadata_list,
            index=feature_info.index
        )
        feature_info = pd.concat([feature_info, feature_metadata_df], axis=1)
        return feature_info

    @staticmethod
    def get_hull_range(
        hull: oms.ConvexHull2D,
    ) -> dict[
        Literal["MZstart","RTstart","MZend","RTend"],
        float
    ]:
        rt_points = hull.getHullPoints()[:,0]
        mz_points = hull.getHullPoints()[:,1]
        return {
            "MZstart": np.min(mz_points),
            "RTstart": np.min(rt_points),
            "MZend": np.max(mz_points),
            "RTend": np.max(rt_points),
        }

    @staticmethod
    def get_hulls(
        feature_map: oms.FeatureMap,
        feature_xic: dict[dict[oms.MSChromatogram]],
    ) -> pd.DataFrame:
        rt_hulls = {}
        for feature_rt_hulls in feature_xic:
            for rt_hull in feature_rt_hulls:
                rt_hulls[rt_hull.getNativeID().replace("_","::")] = rt_hull
        mz_hulls = {}
        for feature in feature_map:
            for i,mz_hull in enumerate(feature.getConvexHulls()):
                mz_hulls[f"{feature.getUniqueId()}::{i}"] = mz_hull
        hulls = []
        hulls_id = list(mz_hulls.keys())
        for hull_id in hulls_id:
            hull = {}
            hull.update(FeatureMap.get_hull_range(mz_hulls[hull_id]))
            rt_points, intens_points = rt_hulls[hull_id].get_peaks()
            mz_points = mz_hulls[hull_id].getHullPoints()[:,1][:len(rt_points)]
            hull['rt_points'] = rt_points
            hull['mz_points'] = mz_points
            hull['intens_points'] = intens_points
            hulls.append(hull)
        hulls = pd.DataFrame(hulls, index=hulls_id)
        return hulls

    @classmethod
    def from_oms(
        cls,
        feature_map: oms.FeatureMap,
        feature_xic: list[list[oms.MSChromatogram]],
        exp_name: str,
    ) -> FeatureMap:
        feature_info = cls.get_feature_info(feature_map)
        hulls = cls.get_hulls(feature_map, feature_xic)
        feature_info.index = feature_info.index.map(lambda x: f"{exp_name}::{x}")
        feature_info.index.name = "feature_id"
        hulls.index = hulls.index.map(lambda x: f"{exp_name}::{x}")
        hulls.index.name = "hull_id"
        return cls(feature_info=feature_info, hulls=hulls)

    def get_oms_feature_map(self) -> oms.FeatureMap:
        feature_map = oms.FeatureMap()
        for feature_id, feature_row in self.feature_info.iterrows():
            feature = oms.Feature()
            feature.setUniqueId(int(feature_id.split("::")[1]))
            feature.setMZ(feature_row["mz"])
            feature.setRT(feature_row["RT"])
            feature.setIntensity(feature_row["intensity"])
            feature_map.push_back(feature)
        return feature_map
