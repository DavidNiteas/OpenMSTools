from __future__ import annotations

import copy
from typing import ClassVar, Literal

import dask
import dask.bag as db
import geopandas as gpd
import polars as pl
import pyopenms as oms
from pydantic import Field
from shapely.geometry import box

from ..module_abc.map_ABCs import BaseMap


class FeatureMap(BaseMap):

    table_schema: ClassVar[dict[str, dict]] = {
        "feature_info": {
                "RT": pl.Float32,
                "mz": pl.Float32,
                "intensity": pl.Float32,
                "MZstart": pl.Float32,
                "RTstart": pl.Float32,
                "MZend": pl.Float32,
                "RTend": pl.Float32,
                "hull_num": pl.Int8,
        },
        "hull_info": {
            "RTstart": pl.Float32,
            "RTend": pl.Float32,
            "MZstart": pl.Float32,
            "MZend": pl.Float32,
        }
    }

    feature_index: gpd.GeoDataFrame | None = Field(
        default=None,
        data_type="index",
        save_mode="parquet",
        description="Feature的空间索引表，基于geopandas"
    )
    feature_info: pl.DataFrame | None = Field(
        default=None,
        data_type="data",
        save_mode="sqlite",
        description="Feature信息表，基于polars"
    )
    hull_index: gpd.GeoDataFrame | None = Field(
        default=None,
        data_type="index",
        save_mode="parquet",
        description="Hull的空间索引表，基于geopandas"
    )
    hull_info: pl.DataFrame | None = Field(
        default=None,
        data_type="data",
        save_mode="sqlite",
        description="Hull信息表，基于polars"
    )

    @staticmethod
    def get_feature_metadata(feature: oms.Feature) -> dict[
        Literal[
            'hull_num',"hull_mz","hull_rt","hull_intensity",
            "isotope_pattern",
            "adduct_type","adduct_mass",
        ],
        str | float | int | list[float]
    ]:
        all_keys = []
        feature.getKeys(all_keys)
        all_keys = set(all_keys)
        metadata = {
            "hull_num": feature.getMetaValue("num_of_masstraces"),
            "hull_mz": feature.getMetaValue("masstrace_centroid_mz"),
            "hull_rt": feature.getMetaValue("masstrace_centroid_rt"),
            "hull_intensity": feature.getMetaValue("masstrace_intensity"),
            "isotope_pattern": feature.getMetaValue("isotope_distances"),
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
        feature_info = pl.from_pandas(
            feature_info,
            schema_overrides = {
                "RT": pl.Float32,
                "mz": pl.Float32,
                "intensity": pl.Float32,
                "MZstart": pl.Float32,
                "RTstart": pl.Float32,
                "MZend": pl.Float32,
                "RTend": pl.Float32,
            },
            include_index=True
        )
        feature_info = feature_info.with_columns(
            (pl.col("RT") / 60).alias("RT"),
            (pl.col("RTstart") / 60).alias("RTstart"),
            (pl.col("RTend") / 60).alias("RTend"),
        )
        feature_bag = db.from_sequence(feature_map, npartitions=num_workers)
        feature_metadata_bag = feature_bag.map(FeatureMap.get_feature_metadata)
        feature_metadata_list = dask.compute(
            feature_metadata_bag, scheduler=worker_type, num_workers=num_workers
        )[0]
        feature_metadata_df = pl.DataFrame(
            feature_metadata_list,
            schema_overrides={
                "hull_num": pl.Int8,
                "hull_mz": pl.List(pl.Float32),
                "hull_rt": pl.List(pl.Float32),
                "hull_intensity": pl.List(pl.Float32),
                "isotope_pattern": pl.List(pl.Float32),
            }
        )
        feature_metadata_df = feature_metadata_df.with_columns(
            pl.col("isotope_pattern").list.eval(pl.element().cum_sum()),
            pl.col("hull_rt").list.eval(pl.element() / 60),
        )
        if "adduct_mass" in feature_metadata_df.columns:
            feature_metadata_df = feature_metadata_df.with_columns(
                pl.col("adduct_mass").cast(pl.Float32),
            )
        feature_info = pl.concat([feature_info, feature_metadata_df], how="horizontal")
        return feature_info

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
            rt_points, intens_points = rt_hulls[hull_id].get_peaks()
            mz_points = mz_hulls[hull_id].getHullPoints()[:,1][:len(rt_points)]
            hull['rt_points'] = rt_points.tolist()
            hull['mz_points'] = mz_points.tolist()
            hull['intens_points'] = intens_points.tolist()
            hulls.append(hull)
        hull_info = pl.DataFrame(
            hulls,
            schema_overrides={
                'rt_points': pl.List(pl.Float32),
                'mz_points': pl.List(pl.Float32),
                'intens_points': pl.List(pl.Float32),
            }
        )
        hull_info = hull_info.with_columns(
            pl.col("rt_points").list.eval(pl.element() / 60),
        )
        hull_info = hull_info.with_columns(
            pl.col("rt_points").list.min().alias("RTstart"),
            pl.col("rt_points").list.max().alias("RTend"),
            pl.col("mz_points").list.min().alias("MZstart"),
            pl.col("mz_points").list.max().alias("MZend"),
        )
        hull_info = hull_info.select(
            "hull_id",
            "RTstart",
            "RTend",
            "MZstart",
            "MZend",
            "rt_points",
            "mz_points",
            "intens_points",
        )
        return hull_info

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
        feature_index = gpd.GeoDataFrame(
            {"iloc": range(len(feature_info))},
            index=feature_info["feature_id"].to_list(),
            geometry=[
                box(rt_start, mz_start, rt_end, mz_end) \
                    for rt_start, mz_start, rt_end, mz_end in zip(
                        feature_info["RTstart"],
                        feature_info["MZstart"],
                        feature_info["RTend"],
                        feature_info["MZend"],
                    )
            ],
        )
        hull_info = hull_info.with_columns(
            (f"{exp_name}::" + pl.col("hull_id").cast(str)).alias("hull_id"),
        )
        hull_index = gpd.GeoDataFrame(
            {"iloc": range(len(hull_info))},
            index=hull_info["hull_id"].to_list(),
            geometry=[
                box(rt_start, mz_start, rt_end, mz_end) \
                    for rt_start, mz_start, rt_end, mz_end in zip(
                        hull_info["RTstart"],
                        hull_info["MZstart"],
                        hull_info["RTend"],
                        hull_info["MZend"],
                    )
            ]
        )
        return cls(
            exp_name=exp_name,
            feature_info=feature_info,
            hull_info=hull_info,
            feature_index=feature_index,
            hull_index=hull_index,
        )

    def get_oms_feature_map(self) -> oms.FeatureMap:
        feature_map = oms.FeatureMap()
        feature_info = self.feature_info.select(
            "feature_id","mz", "RT", "intensity",
        ).with_columns(
            pl.col("feature_id").str.split("::").list.get(1).cast(pl.Int128).alias("feature_id"),
        )
        for i in range(len(feature_info)):
            feature = oms.Feature()
            feature.setUniqueId(feature_info[i, "feature_id"])
            feature.setMZ(feature_info[i, "mz"])
            feature.setRT(feature_info[i, "RT"])
            feature.setIntensity(feature_info[i, "intensity"])
            feature_map.push_back(feature)
        return feature_map

    def search_feature_by_range(
        self,
        coordinates: tuple[
            float, # min_rt
            float, # min_mz
            float, # max_rt
            float, # max_mz
        ],
        return_type: Literal["id", "indices", "df"] = "id",
    ) -> list[int] | list[str] | pl.DataFrame:
        iloc = list(self.feature_index.sindex.intersection(coordinates))
        if return_type == "id":
            return self.feature_index[iloc].index.tolist()
        elif return_type == "df":
            return self.feature_info[iloc]
        else:
            return iloc

    def search_hull_by_range(
        self,
        coordinates: tuple[
            float, # min_rt
            float, # min_mz
            float, # max_rt
            float, # max_mz
        ],
        return_type: Literal["id", "indices", "df"] = "id",
    ) -> list[int] | list[str] | pl.DataFrame:
        iloc = list(self.hull_index.sindex.intersection(coordinates))
        if return_type == "id":
            return self.hull_index[iloc].index.tolist()
        elif return_type == "df":
            return self.hull_info[iloc]
        else:
            return iloc

    def save(self, save_dir_path: str):

        self_to_save = copy.copy(self)

        self_to_save.feature_info = self.feature_info.with_columns(
            ("[" + pl.col("hull_mz").cast(pl.List(pl.String)).list.join(",") + "]")
            .alias("hull_mz"),
            ("[" + pl.col("hull_rt").cast(pl.List(pl.String)).list.join(",") + "]")
            .alias("hull_rt"),
            ("[" + pl.col("hull_intensity").cast(pl.List(pl.String)).list.join(",") + "]")
            .alias("hull_intensity"),
            ("[" + pl.col("isotope_pattern").cast(pl.List(pl.String)).list.join(",") + "]")
            .alias("isotope_pattern"),
        )

        self_to_save.hull_info = self.hull_info.with_columns(
            ("[" + pl.col("rt_points").cast(pl.List(pl.String)).list.join(",") + "]")
            .alias("rt_points"),
            ("[" + pl.col("mz_points").cast(pl.List(pl.String)).list.join(",") + "]")
            .alias("mz_points"),
            ("[" + pl.col("intens_points").cast(pl.List(pl.String)).list.join(",") + "]")
            .alias("intens_points"),
        )

        super(FeatureMap, self_to_save).save(save_dir_path)

    @classmethod
    def load(cls, save_dir_path: str):

        data_dict = cls._base_load(save_dir_path)

        feature_info: pl.DataFrame | None = data_dict.pop("feature_info")

        if feature_info is not None:
            data_dict['feature_info'] = feature_info.with_columns(
                pl.col("hull_mz")
                    .str.strip_chars_start("[")
                    .str.strip_chars_end("]")
                    .str.split(",")
                    .cast(pl.List(pl.Float32)),
                pl.col("hull_rt")
                    .str.strip_chars_start("[")
                    .str.strip_chars_end("]")
                    .str.split(",")
                    .cast(pl.List(pl.Float32)),
                pl.col("hull_intensity")
                    .str.strip_chars_start("[")
                    .str.strip_chars_end("]")
                    .str.split(",")
                    .cast(pl.List(pl.Float32)),
                pl.col("isotope_pattern")
                    .str.strip_chars_start("[")
                    .str.strip_chars_end("]")
                    .str.split(",")
                    .cast(pl.List(pl.Float32)),
            )

        hull_info: pl.DataFrame | None = data_dict.pop("hull_info")
        if hull_info is not None:
            data_dict['hull_info'] = hull_info.with_columns(
                pl.col("rt_points")
                    .str.strip_chars_start("[")
                    .str.strip_chars_end("]")
                    .str.split(",")
                    .cast(pl.List(pl.Float32)),
                pl.col("mz_points")
                    .str.strip_chars_start("[")
                    .str.strip_chars_end("]")
                    .str.split(",")
                    .cast(pl.List(pl.Float32)),
                pl.col("intens_points")
                    .str.strip_chars_start("[")
                    .str.strip_chars_end("]")
                    .str.split(",")
                    .cast(pl.List(pl.Float32)),
            )

        return cls(**data_dict)
