from __future__ import annotations

import copy
import json
from typing import Literal

import dask
import dask.bag as db
import numpy as np
import pandas as pd
import polars as pl
import pyopenms as oms
import rtree
from pydantic import Field

from .ABCs import BaseMap


class FeatureMap(BaseMap):

    feature_index: pd.Index | None = Field(
        default=None,
        data_type="index",
        save_mode="csv",
        description="Feature信息表的索引"
    )
    feature_info: pl.DataFrame | None = Field(
        default=None,
        data_type="data",
        save_mode="sqlite",
        description="Feature信息表"
    )
    feature_rtree_index: rtree.index.Index | None = Field(
        default=None,
        data_type="index",
        save_mode="rtree",
        build_func="build_feature_rtree_index",
        init_func="init_feature_rtree_index",
        description="Feature的R-tree索引"
    )
    hull_index: pd.Index | None = Field(
        default=None,
        data_type="index",
        save_mode="csv",
        description="Hull信息表的索引"
    )
    hull_info: pl.DataFrame | None = Field(
        default=None,
        data_type="data",
        save_mode="sqlite",
        description="Hull信息表"
    )
    hull_rtree_index: rtree.index.Index | None = Field(
        default=None,
        data_type="index",
        save_mode="rtree",
        build_func="build_hull_rtree_index",
        init_func="init_hull_rtree_index",
        description="Hull的R-tree索引"
    )

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
            "hull_mz": np.array(feature.getMetaValue("masstrace_centroid_mz"),dtype=np.float32),
            "hull_rt": np.array(feature.getMetaValue("masstrace_centroid_rt"),dtype=np.float32),
            "hull_intensity": np.array(feature.getMetaValue("masstrace_intensity"),dtype=np.float32),
            "isotope_pattern": np.cumsum(feature.getMetaValue("isotope_distances"),dtype=np.float32),
        }
        if "dc_charge_adducts" in all_keys:
            metadata["adduct_type"] = feature.getMetaValue("dc_charge_adducts")
            metadata["adduct_mass"] = feature.getMetaValue("dc_charge_adduct_mass")
        return metadata

    @staticmethod
    def get_feature_info(
        feature_map: oms.FeatureMap,
        worker_type: Literal["threads", "processes", "synchronous"] = "threads",
        num_workers: int | None = None,
    ) -> pl.DataFrame:
        feature_info = feature_map.get_df()[
            ["RT","mz","intensity","MZstart","RTstart","MZend","RTend"]
        ]
        feature_info.index.name = "feature_id"
        feature_info = pl.from_pandas(feature_info,include_index=True)
        feature_info = feature_info.with_columns(
            pl.col("RT").cast(pl.Float32),
            pl.col("mz").cast(pl.Float32),
            pl.col("intensity").cast(pl.Float32),
            pl.col("MZstart").cast(pl.Float32),
            pl.col("RTstart").cast(pl.Float32),
            pl.col("MZend").cast(pl.Float32),
            pl.col("RTend").cast(pl.Float32),
        )
        feature_bag = db.from_sequence(feature_map, npartitions=num_workers)
        feature_metadata_bag = feature_bag.map(FeatureMap.get_feature_metadata)
        feature_metadata_list = dask.compute(
            feature_metadata_bag, scheduler=worker_type, num_workers=num_workers
        )[0]
        feature_metadata_df = pl.DataFrame(
            feature_metadata_list,
        )
        feature_metadata_df = feature_metadata_df.with_columns(
            pl.col("hull_num").cast(pl.Int32),
        )
        feature_info = pl.concat([feature_info, feature_metadata_df], how="horizontal")
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
    ) -> pl.DataFrame:
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
            hull['hull_id'] = hull_id
            hull.update(FeatureMap.get_hull_range(mz_hulls[hull_id]))
            rt_points, intens_points = rt_hulls[hull_id].get_peaks()
            mz_points = mz_hulls[hull_id].getHullPoints()[:,1][:len(rt_points)]
            hull['rt_points'] = rt_points.astype(np.float32)
            hull['mz_points'] = mz_points.astype(np.float32)
            hull['intens_points'] = intens_points.astype(np.float32)
            hulls.append(hull)
        hulls = pl.DataFrame(hulls)
        return hulls

    @classmethod
    def from_oms(
        cls,
        feature_map: oms.FeatureMap,
        feature_xic: list[list[oms.MSChromatogram]],
        exp_name: str,
        worker_type: Literal["threads", "processes", "synchronous"] = "threads",
        num_workers: int | None = None,
    ) -> FeatureMap:
        feature_info = cls.get_feature_info(feature_map, worker_type, num_workers)
        hull_info = cls.get_hulls(feature_map, feature_xic)
        feature_info = feature_info.with_columns(
            (f"{exp_name}::" + pl.col("feature_id").cast(str)).alias("feature_id"),
        )
        feature_index = pd.Index(feature_info["feature_id"].to_list())
        hull_info = hull_info.with_columns(
            (f"{exp_name}::" + pl.col("hull_id").cast(str)).alias("hull_id"),
        )
        hull_index = pd.Index(hull_info["hull_id"].to_list())
        return cls(
            exp_name=exp_name,
            feature_info=feature_info,
            hull_info=hull_info,
            feature_index=feature_index,
            hull_index=hull_index,
        )

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

    def build_feature_rtree_index(self, path : str | None = None) -> rtree.index.Index:
        if self.feature_info is None:
            raise ValueError(
                "Feature info dataframe must be loaded before initializing R-tree index"
            )
        feature_rtree_index = rtree.index.Index(path)
        for i,(mz_start,rt_start,mz_end,rt_end) in enumerate(
            zip(
                self.feature_info['MZstart'],
                self.feature_info['RTstart'],
                self.feature_info['MZend'],
                self.feature_info['RTend'],
            )
        ):
            feature_rtree_index.insert(
                id=i,
                coordinates=(mz_start,rt_start,mz_end,rt_end),
                obj=i
            )
        return feature_rtree_index

    def init_feature_rtree_index(self, path : str | None = None) -> None:
        self.feature_rtree_index = self.build_feature_rtree_index(path)

    def search_feature_by_range(
        self,
        coordinates: tuple[
            float, # min_mz
            float, # min_rt
            float, # max_mz
            float, # max_rt
        ],
        return_type: Literal["id", "indices", "df"] = "id",
    ) -> list[int] | list[str] | pl.DataFrame:
        if self.feature_rtree_index is None:
            self.init_feature_rtree_index()
        index = list(self.feature_rtree_index.intersection(coordinates))
        if return_type == "id":
            return self.feature_index[index].tolist()
        elif return_type == "df":
            return self.feature_info[index]
        else:
            return index

    def build_hull_rtree_index(self, path : str | None = None) -> rtree.index.Index:
        if self.hull_info is None:
            raise ValueError(
                "Hulls dataframe must be loaded before initializing R-tree index"
            )
        hull_rtree_index = rtree.index.Index(path)
        for i,(mz_start,rt_start,mz_end,rt_end) in enumerate(
            zip(
                self.hull_info['MZstart'],
                self.hull_info['RTstart'],
                self.hull_info['MZend'],
                self.hull_info['RTend'],
            )
        ):
            hull_rtree_index.insert(
                id=i,
                coordinates=(mz_start,rt_start,mz_end,rt_end),
                obj=i
            )
        return hull_rtree_index

    def init_hull_rtree_index(self, path : str | None = None) -> None:
        self.hull_rtree_index = self.build_hull_rtree_index(path)

    def search_hull_by_range(
        self,
        coordinates: tuple[
            float, # min_mz
            float, # min_rt
            float, # max_mz
            float, # max_rt
        ],
        return_type: Literal["id", "indices", "df"] = "id",
    ) -> list[int] | list[str] | pl.DataFrame:
        if self.hull_rtree_index is None:
            self.init_hull_rtree_index()
        index = list(self.hull_rtree_index.intersection(coordinates))
        if return_type == "id":
            return self.hull_index[index].tolist()
        elif return_type == "df":
            return self.hull_info[index]
        else:
            return index

    def save(self, save_dir_path: str):

        self_to_save = copy.copy(self)

        self_to_save.feature_info = self.feature_info.with_columns(
            pl.col("hull_mz").map_elements(
                lambda x: json.dumps(x.tolist()),
                return_dtype=pl.String,
            ),
            pl.col("hull_rt").map_elements(
                lambda x: json.dumps(x.tolist()),
                return_dtype=pl.String,
            ),
            pl.col("hull_intensity").map_elements(
                lambda x: json.dumps(x.tolist()),
                return_dtype=pl.String,
            ),
            pl.col("isotope_pattern").map_elements(
                lambda x: json.dumps(x.tolist()),
                return_dtype=pl.String,
            ),
        )

        self_to_save.hull_info = self.hull_info.with_columns(
            pl.col("rt_points").map_elements(
                lambda x: json.dumps(x.tolist()),
                return_dtype=pl.String,
            ),
            pl.col("mz_points").map_elements(
                lambda x: json.dumps(x.tolist()),
                return_dtype=pl.String,
            ),
            pl.col("intens_points").map_elements(
                lambda x: json.dumps(x.tolist()),
                return_dtype=pl.String,
            ),
        )

        super(FeatureMap, self_to_save).save(save_dir_path)

    @classmethod
    def load(cls, save_dir_path: str):

        data_dict = cls._base_load(save_dir_path)

        feature_info: pl.DataFrame | None = data_dict.pop("feature_info")

        if feature_info is not None:
            data_dict['feature_info'] = feature_info.with_columns(
                pl.col("hull_mz").map_elements(
                    lambda x: np.array(json.loads(x)),
                    return_dtype=pl.Object,
                ),
                pl.col("hull_rt").map_elements(
                    lambda x: np.array(json.loads(x)),
                    return_dtype=pl.Object,
                ),
                pl.col("hull_intensity").map_elements(
                    lambda x: np.array(json.loads(x)),
                    return_dtype=pl.Object,
                ),
                pl.col("isotope_pattern").map_elements(
                    lambda x: np.array(json.loads(x)),
                    return_dtype=pl.Object,
                ),
            )
        hull_info: pl.DataFrame | None = data_dict.pop("hull_info")
        if hull_info is not None:
            data_dict['hull_info'] = hull_info.with_columns(
                pl.col("rt_points").map_elements(
                    lambda x: np.array(json.loads(x)),
                    return_dtype=pl.Object,
                ),
                pl.col("mz_points").map_elements(
                    lambda x: np.array(json.loads(x)),
                    return_dtype=pl.Object,
                ),
                pl.col("intens_points").map_elements(
                    lambda x: np.array(json.loads(x)),
                    return_dtype=pl.Object,
                ),
            )

        return cls(**data_dict)
